import torch
import copy
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
        self.last_n_layers = self.config.get('last_n_layers', 2)
        self.adaptation_mode = self.config.get('adaptation_mode', 'layernorm')

        self.state_cache = deque(maxlen=self.cache_capacity)
        self.entropy_cache = deque(maxlen=self.cache_capacity)

        self._offline_policy_saved = True
        self.offline_policy_state = self._save_policy_state()
        self._offline_actor = self._create_frozen_actor()
        
        if self.adaptation_mode == 'layernorm':
            self.trainable_params = self._get_layernorm_params()
            self.param_names = [name for name, _ in self.trainable_params]
            
            if len(self.trainable_params) == 0:
                print("警告: 未找到LayerNorm参数，自动切换到last_n_layers模式")
                self.adaptation_mode = 'last_n_layers'
        
        if self.adaptation_mode == 'last_n_layers':
            self.trainable_params = self._get_last_n_layers(self.last_n_layers)
            self.param_names = [name for name, _ in self.trainable_params]
        
        assert len(self.trainable_params) > 0, "无法找到可训练参数，请检查adaptation_mode配置"
        
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

    def _create_frozen_actor(self):
        """Create a frozen copy of the actor for offline policy"""
        if not hasattr(self.policy, 'actor'):
            return None
        
        offline_actor = copy.deepcopy(self.policy.actor)
        
        offline_state = {
            k: v 
            for k, v in self.offline_policy_state.items() 
            if k.startswith('actor.')
        }
        
        mapped_state = {}
        for key, value in offline_state.items():
            new_key = key.replace('actor.', '', 1)
            mapped_state[new_key] = value
        
        offline_actor.load_state_dict(mapped_state, strict=True)
        offline_actor = offline_actor.to(self.device)
        offline_actor.eval()
        
        for param in offline_actor.parameters():
            param.requires_grad = False
        
        return offline_actor

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
        
        if not hasattr(self.policy, 'actor'):
            return trainable_params
        
        for name, module in self.policy.actor.named_modules():
            if isinstance(module, nn.LayerNorm):
                for param_name, param in module.named_parameters():
                    full_name = f"actor.{name}.{param_name}" if name else f"actor.{param_name}"
                    if param.requires_grad:
                        trainable_params.append((full_name, param))
        return trainable_params
    
    def _get_last_n_layers(self, n: int = 2) -> List[Tuple[str, nn.Parameter]]:
        """
        Get parameters from the last N layers of the actor network
        
        This provides an alternative adaptation strategy when LayerNorm is not available.
        Updates are restricted to the final layers to prevent catastrophic forgetting.
        
        Args:
            n: Number of last layers to include
            
        Returns:
            List of (name, parameter) tuples for trainable parameters
        """
        if not hasattr(self.policy, 'actor'):
            return []
        
        trainable_params = []
        
        if hasattr(self.policy.actor, 'backbone'):
            backbone_params = list(self.policy.actor.backbone.named_parameters())
            if len(backbone_params) >= n:
                selected_params = backbone_params[-n:]
            else:
                selected_params = backbone_params
            
            for name, param in selected_params:
                if param.requires_grad:
                    trainable_params.append((f"backbone.{name}", param))
        
        if hasattr(self.policy.actor, 'last'):
            for name, param in self.policy.actor.last.named_parameters():
                if param.requires_grad:
                    trainable_params.append((f"last.{name}", param))
        
        if hasattr(self.policy.actor, 'dist_net'):
            for name, param in self.policy.actor.dist_net.named_parameters():
                if param.requires_grad:
                    trainable_params.append((f"dist_net.{name}", param))
        
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

        self.policy.train()
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

        return mu, sigma

    def _compute_entropy(self, sigma: torch.Tensor) -> torch.Tensor:
        """
        Compute differential entropy of Gaussian policy
        
        H = 1/2 * (ln(2πσ²) + 1)
        
        Higher entropy indicates greater uncertainty about action selection.
        """
        entropy = 0.5 * (torch.log(2 * np.pi * sigma**2 + 1e-8) + 1)
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
        
        Uses a cached offline actor to compute KL divergence safely.
        """
        if not hasattr(self.policy, 'actor') or self._offline_actor is None:
            return torch.zeros_like(obs[..., :1]).requires_grad_(False), torch.ones_like(obs[..., :1]).requires_grad_(False)
        
        try:
            with torch.no_grad():
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
                action_dist = self._offline_actor(obs_tensor)
                
                if hasattr(action_dist, 'mean'):
                    mu_off = action_dist.mean.detach().clone()
                elif hasattr(action_dist, 'mode'):
                    mu_off = action_dist.mode()[0].detach().clone()
                else:
                    mu_off = action_dist.detach().clone()
                
                if hasattr(action_dist, 'stddev'):
                    sigma_off = action_dist.stddev.detach().clone()
                elif hasattr(action_dist, 'scale'):
                    sigma_off = action_dist.scale.detach().clone()
                else:
                    sigma_off = torch.ones_like(mu_off)
            
            mu_off = mu_off.requires_grad_(False)
            sigma_off = sigma_off.requires_grad_(False)
            
            if mu_off.device != self.device:
                mu_off = mu_off.to(self.device)
                sigma_off = sigma_off.to(self.device)
            
            return mu_off, sigma_off
            
        except Exception as e:
            return torch.zeros_like(obs[..., :1]).to(self.device).requires_grad_(False), torch.ones_like(obs[..., :1]).to(self.device).requires_grad_(False)

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
        
        assert mu_tta.requires_grad, "mu_tta must require grad!"
        assert sigma_tta.requires_grad, "sigma_tta must require grad!"
        
        entropy = self._compute_entropy(sigma_tta)

        mu_off, sigma_off = self._get_offline_distribution(obs_batch)
        kl_div = self._compute_kl_divergence(mu_tta, sigma_tta, mu_off, sigma_off)
        
        mu_diff = (mu_off - mu_tta).abs().mean().item()
        sigma_diff = (sigma_off - sigma_tta).abs().mean().item()
        
        total_loss = entropy + self.kl_weight * kl_div

        self.optimizer.zero_grad()
        total_loss.backward()
        
        has_nonzero_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in self.params_only
        )
        if not has_nonzero_grad:
            print(f"  WARNING: ZERO GRADIENT! KL={kl_div.item():.6f}, Entropy={entropy.item():.6f}")

        self.optimizer.step()

        return {
            'entropy': entropy.item(),
            'kl_div': kl_div.item(),
            'total_loss': total_loss.item()
        }

    def _filter_low_entropy_samples(self, obs_batch: torch.Tensor) -> torch.Tensor:
        """
        Filter samples based on entropy threshold within current batch
        
        Returns observations with entropy below E_0
        """
        _, sigma = self._compute_action_distribution(obs_batch)
        entropies = 0.5 * (torch.log(2 * np.pi * sigma**2 + 1e-8) + 1)
        entropies = entropies.mean(dim=-1)

        k = min(self.k_low_entropy, len(obs_batch))
        threshold = torch.kthvalue(entropies, k).values
        
        mask = entropies <= threshold

        if mask.sum() == 0:
            mask[0] = True

        return obs_batch[mask]

    def _run_single_episode(self) -> Dict[str, Any]:
        """Run a single episode and collect state data and perform updates"""
        reset_result = self.env.reset()
        if isinstance(reset_result, tuple):
            obs, _ = reset_result
        else:
            obs = reset_result

        episode_reward = 0
        episode_length = 0
        episode_transitions = []
        episode_loss = []
        episode_entropy = []
        episode_kl = []

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

            if len(self.state_cache) >= self.k_low_entropy and self.adaptation_step % 10 == 0:
                obs_batch = torch.FloatTensor(
                    np.array(list(self.state_cache))
                ).to(self.device)

                filtered_obs = self._filter_low_entropy_samples(obs_batch)

                if len(filtered_obs) > 0:
                    loss_info = self._update(filtered_obs)
                    episode_loss.append(loss_info['total_loss'])
                    episode_entropy.append(loss_info['entropy'])
                    episode_kl.append(loss_info['kl_div'])
            
            self.adaptation_step += 1

            if episode_length >= 1000:
                break

        if len(episode_loss) > 0:
            avg_loss = np.mean(episode_loss)
            avg_entropy = np.mean(episode_entropy)
            avg_kl = np.mean(episode_kl)
        else:
            avg_loss = 0.0
            avg_entropy = 0.0
            avg_kl = 0.0

        return {
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            'transitions': episode_transitions,
            'adaptation_step': self.adaptation_step,
            'adaptation_loss': avg_loss,
            'entropy': avg_entropy,
            'kl_div': avg_kl,
            'episode_loss': episode_loss,
            'episode_entropy': episode_entropy,
            'episode_kl': episode_kl
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

            if 'episode_loss' in episode_data and len(episode_data['episode_loss']) > 0:
                loss_history.extend(episode_data['episode_loss'])
                entropy_history.extend(episode_data['episode_entropy'])
                kl_history.extend(episode_data['episode_kl'])

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
            'k_low_entropy': self.k_low_entropy,
            'adaptation_mode': self.adaptation_mode,
            'last_n_layers': self.last_n_layers,
            'param_names': self.param_names[:5]
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
