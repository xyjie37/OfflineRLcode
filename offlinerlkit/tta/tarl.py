import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from collections import deque

from offlinerlkit.policy.base_policy import BasePolicy


class TARLManager:
    """
    Test-Time Adaptation with Reinforcement Learning (TARL)
    
    TARL implements unsupervised test-time adaptation for continuous control tasks
    by minimizing action uncertainty (entropy) on low-entropy samples while 
    constraining policy drift via KL divergence regularization.
    
    Key Features:
    - Gaussian policy output for continuous action spaces
    - Entropy-based uncertainty quantification
    - Low-entropy sample filtering mechanism
    - LayerNorm-only parameter updates for stable adaptation
    - KL divergence regularization with frozen offline policy
    
    Reference: Test-Time Adaptation with Reinforcement Learning (TARL)
    """

    def __init__(
        self,
        policy: BasePolicy,
        env,
        config: Optional[Dict[str, Any]] = None
    ):
        self.policy = policy
        self.env = env
        self.config = config or {}

        self.device = self._get_policy_device()

        self.learning_rate = self.config.get('learning_rate', 1e-6)
        self.cache_capacity = self.config.get('cache_capacity', 1000)
        self.k_low_entropy = self.config.get('k_low_entropy', 10)
        self.kl_weight = self.config.get('kl_weight', 1.0)
        self.gradient_clip = self.config.get('gradient_clip', 0.5)

        self.state_cache = deque(maxlen=self.cache_capacity)
        self.entropy_cache = deque(maxlen=self.cache_capacity)

        self.offline_policy_state = self._save_policy_state()
        self.trainable_params = self._get_layernorm_params()
        self.param_names = [name for name, _ in self.trainable_params]
        
        if len(self.trainable_params) == 0:
            print("Warning: No LayerNorm parameters found in policy. "
                  "Falling back to all actor parameters with reduced learning rate.")
            self.trainable_params = self._get_actor_params()
            self.param_names = [name for name, _ in self.trainable_params]
        
        params_only = [param for _, param in self.trainable_params]
        self.params_only = params_only
        
        self.optimizer = torch.optim.Adam(
            params_only,
            lr=self.learning_rate
        )

        self.adaptation_step = 0

    def _get_policy_device(self) -> torch.device:
        """Get the device of the policy"""
        if hasattr(self.policy, 'actor') and hasattr(self.policy.actor, 'device'):
            return self.policy.actor.device
        elif hasattr(self.policy, 'critic1') and hasattr(self.policy.critit1, 'device'):
            return self.policy.critic1.device
        elif len(list(self.policy.parameters())) > 0:
            return next(self.policy.parameters()).device
        else:
            return torch.device('cpu')

    def _save_policy_state(self) -> Dict[str, torch.Tensor]:
        """Save current policy state for KL divergence computation"""
        state = {}
        for name, param in self.policy.named_parameters():
            state[name] = param.data.clone().detach()
        return state

    def _get_layernorm_params(self) -> List[Tuple[str, nn.Parameter]]:
        """
        Extract LayerNorm parameters for trainable adaptation
        
        Only LayerNorm parameters are updated during TTA to prevent
        policy collapse while allowing distribution shift adaptation.
        """
        trainable_params = []
        for name, module in self.policy.named_modules():
            if isinstance(module, nn.LayerNorm):
                for param_name, param in module.named_parameters():
                    full_name = f"{name}.{param_name}" if name else param_name
                    if param.requires_grad:
                        trainable_params.append((full_name, param))
        return trainable_params
    
    def _get_actor_params(self) -> List[Tuple[str, nn.Parameter]]:
        """
        Fallback: Get all parameters from actor module
        
        Used when no LayerNorm parameters are found in the policy.
        This provides a more aggressive adaptation strategy.
        """
        if not hasattr(self.policy, 'actor'):
            return []
        
        trainable_params = []
        for name, param in self.policy.actor.named_parameters():
            if param.requires_grad:
                trainable_params.append((name, param))
        return trainable_params

    def _compute_action_distribution(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute action distribution parameters for Gaussian policy
        
        Args:
            obs: Observation tensor
            
        Returns:
            mu: Mean vector (with gradients for backprop)
            sigma: Standard deviation (positive, with gradients for backprop)
        """
        if not hasattr(self.policy, 'actor'):
            raise ValueError("Policy must have an actor module for TARL")

        self.policy.eval()
        with torch.no_grad():
            action_dist = self.policy.actor(obs)

            if hasattr(action_dist, 'mean'):
                mu = action_dist.mean
            elif hasattr(action_dist, 'mode'):
                mu = action_dist.mode()[0]
            else:
                mu = action_dist

            if hasattr(action_dist, 'stddev'):
                sigma = action_dist.stddev
            elif hasattr(action_dist, 'scale'):
                sigma = action_dist.scale
            else:
                if hasattr(action_dist, 'log_std'):
                    log_std = action_dist.log_std
                    if isinstance(log_std, tuple):
                        log_std = torch.cat([ls.unsqueeze(0) for ls in log_std], dim=-1)
                    sigma = F.softplus(log_std) + 1e-6
                else:
                    sigma = torch.ones_like(mu)

        mu = mu.clone().requires_grad_(True)
        if sigma.requires_grad:
            sigma = sigma.clone()
        else:
            sigma = sigma.clone().requires_grad_(True)

        return mu, sigma

    def _compute_entropy(self, sigma: torch.Tensor) -> torch.Tensor:
        """
        Compute differential entropy of Gaussian policy
        
        H = 1/2 * (ln(2πσ²) + 1)
        
        Higher entropy indicates greater uncertainty about action selection.
        """
        log_var = torch.log(sigma + 1e-8)
        entropy = 0.5 * (torch.log(2 * np.pi * sigma + 1e-8) + 1)
        return entropy.mean()

    def _compute_kl_divergence(
        self,
        mu_tta: torch.Tensor,
        sigma_tta: torch.Tensor,
        mu_off: torch.Tensor,
        sigma_off: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute KL divergence between offline and adapted policies
        
        KL(N_off || N_tta) = 0.5 * [σ_tta²/σ_off² + (μ_off - μ_tta)²/σ_off² - 1 + ln(σ_off²/σ_tta²)]
        """
        var_off = sigma_off ** 2 + 1e-8
        var_tta = sigma_tta ** 2 + 1e-8

        kl = 0.5 * (
            var_tta / var_off +
            (mu_off - mu_tta) ** 2 / var_off -
            1 +
            torch.log(var_off / var_tta)
        )

        return kl.mean()

    def _get_offline_distribution(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get action distribution from frozen offline policy
        
        Uses the saved offline policy state to compute KL divergence
        without modifying the current policy.
        """
        self.policy.eval()
        
        if not hasattr(self.policy, 'actor'):
            return torch.zeros_like(obs[..., :1]).requires_grad_(False), torch.ones_like(obs[..., :1]).requires_grad_(False)
        
        try:
            actor = self.policy.actor
            
            class OfflineActorWrapper(nn.Module):
                def __init__(self, actor_module):
                    super().__init__()
                    self.backbone = actor_module.backbone
                    self.dist_net = actor_module.dist_net
                    self.device = actor_module.device
                
                def forward(self, obs):
                    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
                    logits = self.backbone(obs_tensor)
                    dist = self.dist_net(logits)
                    return dist
            
            offline_actor = OfflineActorWrapper(actor).to(self.device)
            offline_actor.load_state_dict({
                k.replace('actor.', '', 1): v 
                for k, v in self.offline_policy_state.items() 
                if k.startswith('actor.')
            })
            
            with torch.no_grad():
                action_dist = offline_actor(obs)
                
                if hasattr(action_dist, 'mean'):
                    mu_off = action_dist.mean.detach().clone().requires_grad_(False)
                elif hasattr(action_dist, 'mode'):
                    mu_off = action_dist.mode()[0].detach().clone().requires_grad_(False)
                else:
                    mu_off = action_dist.detach().clone().requires_grad_(False)
                
                if hasattr(action_dist, 'stddev'):
                    sigma_off = action_dist.stddev.detach().clone().requires_grad_(False)
                elif hasattr(action_dist, 'scale'):
                    sigma_off = action_dist.scale.detach().clone().requires_grad_(False)
                else:
                    sigma_off = torch.ones_like(mu_off)
            
            del offline_actor
            
            if mu_off.device != self.device:
                mu_off = mu_off.to(self.device)
                sigma_off = sigma_off.to(self.device)
            
            return mu_off, sigma_off
            
        except Exception as e:
            print(f"Warning: Failed to load offline actor state: {e}")
            return torch.zeros_like(obs[..., :1]).to(self.device).requires_grad_(False), torch.ones_like(obs[..., :1]).to(self.device).requires_grad_(False)

    def _compute_threshold(self) -> float:
        """
        Compute entropy threshold E_0 as the k-th smallest entropy value
        
        Only samples with entropy below this threshold participate in
        gradient updates.
        """
        if len(self.entropy_cache) < self.k_low_entropy:
            return float('inf')

        entropies = list(self.entropy_cache)
        sorted_entropies = sorted(entropies)
        threshold = sorted_entropies[self.k_low_entropy - 1]
        return threshold

    def _update(self, obs_batch: torch.Tensor):
        """
        Perform single TTA update step
        
        Loss = L_ent^con + λ * L_kl^t
        where:
        - L_ent^con: Entropy loss on low-entropy samples
        - L_kl^t: KL divergence between offline and adapted policies
        """
        self.policy.train()

        mu_tta, sigma_tta = self._compute_action_distribution(obs_batch)

        entropy = self._compute_entropy(sigma_tta)

        mu_off, sigma_off = self._get_offline_distribution(obs_batch)
        kl_div = self._compute_kl_divergence(mu_tta, sigma_tta, mu_off, sigma_off)

        total_loss = entropy + self.kl_weight * kl_div

        self.optimizer.zero_grad()
        total_loss.backward()

        if self.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.params_only, self.gradient_clip)

        self.optimizer.step()

        return {
            'entropy': entropy.item(),
            'kl_div': kl_div.item(),
            'total_loss': total_loss.item()
        }

    def _filter_low_entropy_samples(self, obs_batch: torch.Tensor) -> torch.Tensor:
        """
        Filter samples based on entropy threshold
        
        Returns observations with entropy below E_0
        """
        _, sigma = self._compute_action_distribution(obs_batch)
        entropies = 0.5 * (torch.log(2 * np.pi * sigma + 1e-8) + 1)
        entropies = entropies.mean(dim=-1)

        threshold = self._compute_threshold()
        mask = entropies <= threshold

        if mask.sum() == 0:
            mask[0] = True

        return obs_batch[mask]

    def _run_single_episode(self) -> Dict[str, Any]:
        """Run a single episode and collect state data"""
        reset_result = self.env.reset()
        if isinstance(reset_result, tuple):
            obs, _ = reset_result
        else:
            obs = reset_result

        episode_reward = 0
        episode_length = 0
        episode_transitions = []

        done = False
        while not done:
            with torch.no_grad():
                action = self.policy.select_action(obs)

            step_result = self.env.step(action)
            if len(step_result) == 5:
                next_obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                next_obs, reward, done, info = step_result

            transition = {
                'obs': obs.copy(),
                'action': action.copy(),
                'reward': reward,
                'next_obs': next_obs.copy(),
                'done': done
            }
            episode_transitions.append(transition)

            self.state_cache.append(obs.copy())

            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                _, sigma = self._compute_action_distribution(obs_tensor)
                entropy = self._compute_entropy(sigma)
                self.entropy_cache.append(entropy.item())

            obs = next_obs
            episode_reward += reward
            episode_length += 1

            if episode_length >= 1000:
                break

        return {
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            'transitions': episode_transitions,
            'adaptation_step': self.adaptation_step
        }

    def run_adaptation(self, num_episodes: int = 10) -> Tuple[List[Dict], Dict[str, Any]]:
        """
        Run TARL adaptation for specified number of episodes
        
        Args:
            num_episodes: Number of adaptation episodes
            
        Returns:
            adaptation_data: List of episode data
            summary: Summary statistics
        """
        adaptation_data = []
        loss_history = []
        entropy_history = []
        kl_history = []

        for episode in range(num_episodes):
            episode_data = self._run_single_episode()
            adaptation_data.append(episode_data)

            if len(self.state_cache) >= self.k_low_entropy:
                obs_batch = torch.FloatTensor(
                    np.array(list(self.state_cache))
                ).to(self.device)

                filtered_obs = self._filter_low_entropy_samples(obs_batch)

                if len(filtered_obs) > 0:
                    loss_info = self._update(filtered_obs)
                    loss_history.append(loss_info['total_loss'])
                    entropy_history.append(loss_info['entropy'])
                    kl_history.append(loss_info['kl_div'])

                    episode_data['adaptation_loss'] = loss_info['total_loss']
                    episode_data['entropy'] = loss_info['entropy']
                    episode_data['kl_div'] = loss_info['kl_div']

            self.adaptation_step += 1

            tta_status = "TARL"
            print(f"Episode {episode + 1}/{num_episodes} ({tta_status}): "
                  f"Reward: {episode_data['episode_reward']:.2f}, "
                  f"Length: {episode_data['episode_length']}, "
                  f"Loss: {episode_data.get('adaptation_loss', 0):.6f}, "
                  f"Entropy: {episode_data.get('entropy', 0):.4f}, "
                  f"KL: {episode_data.get('kl_div', 0):.4f}")

        summary = {
            'mean_reward': np.mean([d['episode_reward'] for d in adaptation_data]),
            'std_reward': np.std([d['episode_reward'] for d in adaptation_data]),
            'mean_length': np.mean([d['episode_length'] for d in adaptation_data]),
            'mean_loss': np.mean(loss_history) if loss_history else 0,
            'mean_entropy': np.mean(entropy_history) if entropy_history else 0,
            'mean_kl': np.mean(kl_history) if kl_history else 0,
            'cache_size': len(self.state_cache),
            'loss_history': loss_history,
            'entropy_history': entropy_history,
            'kl_history': kl_history,
            'adaptation_steps': self.adaptation_step
        }

        return adaptation_data, summary

    def evaluate_performance(self, num_episodes: int = 5) -> Dict[str, float]:
        """Evaluate current policy performance"""
        self.policy.eval()

        rewards = []
        lengths = []

        for _ in range(num_episodes):
            reset_result = self.env.reset()
            if isinstance(reset_result, tuple):
                obs, _ = reset_result
            else:
                obs = reset_result

            episode_reward = 0
            episode_length = 0
            done = False

            while not done:
                with torch.no_grad():
                    action = self.policy.select_action(obs, deterministic=True)

                step_result = self.env.step(action)
                if len(step_result) == 5:
                    obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:
                    obs, reward, done, info = step_result

                episode_reward += reward
                episode_length += 1

                if episode_length >= 1000:
                    break

            rewards.append(episode_reward)
            lengths.append(episode_length)

        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'mean_length': np.mean(lengths)
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get TARL adaptation statistics"""
        return {
            'adaptation_steps': self.adaptation_step,
            'cache_size': len(self.state_cache),
            'entropy_cache_size': len(self.entropy_cache),
            'trainable_params': len(self.trainable_params),
            'learning_rate': self.learning_rate,
            'kl_weight': self.kl_weight,
            'k_low_entropy': self.k_low_entropy
        }

    def save_checkpoint(self, save_path: str):
        """Save TARL checkpoint"""
        checkpoint = {
            'policy_state_dict': self.policy.state_dict(),
            'offline_policy_state': self.offline_policy_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'adaptation_step': self.adaptation_step,
            'config': self.config,
            'statistics': self.get_statistics()
        }
        torch.save(checkpoint, save_path)

    def load_checkpoint(self, checkpoint_path: str):
        """Load TARL checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.offline_policy_state = checkpoint['offline_policy_state']
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.adaptation_step = checkpoint['adaptation_step']
        self.config = checkpoint.get('config', {})
