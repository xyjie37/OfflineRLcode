import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from collections import deque

from offlinerlkit.policy.base_policy import BasePolicy


class STINTManager:
    """
    STINT: Stability-Triggered Intervention for Test-Time RL Adaptation
    
    Key Features:
    - Component A: Entropy surge trigger (unstable detection)
    - Component B: Reference policy trust region (KL constraint)
    - Component C: Parameter throttling (only update policy head or LayerNorm)
    
    Algorithm:
    1. Compute policy entropy H_t and moving average \bar{H}_t
    2. Trigger adaptation when H_t - \bar{H}_t > delta
    3. Apply stabilization loss with KL regularization
    4. Only update policy head or LayerNorm parameters
    5. Limit to K update steps per trigger
    
    Reference: Stability-Triggered Intervention for Test-Time RL Adaptation
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

        self.delta = self.config.get('delta', 0.5)
        self.lambda_kl = self.config.get('lambda_kl', 1.0)
        self.K = self.config.get('K', 3)
        self.beta = self.config.get('beta', 0.1)
        self.learning_rate = self.config.get('learning_rate', 1e-4)
        self.adaptation_mode = self.config.get('adaptation_mode', 'layernorm')

        self.entropy_moving_avg = 0.0

        self.offline_policy_state = self._save_policy_state()
        self._create_offline_actor()

        if self.adaptation_mode == 'layernorm':
            self.trainable_params = self._get_layernorm_params()
            if len(self.trainable_params) == 0:
                print("警告: 未找到LayerNorm参数，自动切换到policy_head模式")
                self.adaptation_mode = 'policy_head'
        
        if self.adaptation_mode == 'policy_head':
            self.trainable_params = self._get_policy_head_params()
        
        assert len(self.trainable_params) > 0, "无法找到可训练参数，请检查adaptation_mode配置"
        
        params_only = [param for _, param in self.trainable_params]
        self.params_only = params_only
        
        self.optimizer = torch.optim.Adam(
            params_only,
            lr=self.learning_rate
        )

        self.adaptation_step = 0
        self.trigger_count = 0

        self.entropy_history = []
        self.trigger_history = []

    def _get_policy_device(self) -> torch.device:
        """Get the device of the policy"""
        if hasattr(self.policy, 'actor') and hasattr(self.policy.actor, 'device'):
            return self.policy.actor.device
        elif hasattr(self.policy, 'critic1') and hasattr(self.policy.critic1, 'device'):
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

    def _create_offline_actor(self):
        """Create a frozen copy of the actor for offline policy"""
        if not hasattr(self.policy, 'actor'):
            self._offline_actor = None
            return
        
        self._offline_actor = copy.deepcopy(self.policy.actor)
        
        offline_state = {
            k: v 
            for k, v in self.offline_policy_state.items() 
            if k.startswith('actor.')
        }
        
        mapped_state = {}
        for key, value in offline_state.items():
            new_key = key.replace('actor.', '', 1)
            mapped_state[new_key] = value
        
        self._offline_actor.load_state_dict(mapped_state, strict=True)
        self._offline_actor = self._offline_actor.to(self.device)
        self._offline_actor.eval()
        
        for param in self._offline_actor.parameters():
            param.requires_grad = False

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

    def _get_policy_head_params(self) -> List[Tuple[str, nn.Parameter]]:
        """
        Extract policy head parameters for trainable adaptation
        
        Updates are restricted to the final layers (dist_net) to prevent
        catastrophic forgetting while allowing adaptation.
        """
        trainable_params = []
        
        if not hasattr(self.policy, 'actor'):
            return trainable_params
        
        if hasattr(self.policy.actor, 'dist_net'):
            for name, param in self.policy.actor.dist_net.named_parameters():
                if param.requires_grad:
                    trainable_params.append((f"dist_net.{name}", param))
        
        if hasattr(self.policy.actor, 'last'):
            for name, param in self.policy.actor.last.named_parameters():
                if param.requires_grad:
                    trainable_params.append((f"last.{name}", param))
        
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
            raise ValueError("Policy must have an actor module for STINT")

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

    def _check_trigger(self, obs: torch.Tensor) -> bool:
        """
        Component A: Entropy surge trigger
        
        Check if H_t - \bar{H}_t > delta
        
        Args:
            obs: Current observation
            
        Returns:
            triggered: Whether to trigger adaptation
        """
        _, sigma = self._compute_action_distribution(obs)
        H_t = self._compute_entropy(sigma).item()
        
        self.entropy_moving_avg = (1 - self.beta) * self.entropy_moving_avg + self.beta * H_t
        
        self.entropy_history.append(H_t)
        
        triggered = (H_t - self.entropy_moving_avg) > self.delta
        
        if triggered:
            self.trigger_count += 1
            self.trigger_history.append(self.adaptation_step)
        
        return triggered

    def _stabilization_update(self, obs: torch.Tensor) -> Dict[str, float]:
        """
        Perform stabilization intervention with K steps
        
        Loss = -H(π_θ(·|s_t)) + λ * D_KL(π_θ(·|s_t) || π_0(·|s_t))
        
        Args:
            obs: Observation tensor
            
        Returns:
            Dictionary with loss components
        """
        self.policy.train()
        
        mu_tta, sigma_tta = self._compute_action_distribution(obs)
        
        assert mu_tta.requires_grad, "mu_tta must require grad!"
        assert sigma_tta.requires_grad, "sigma_tta must require grad!"
        
        entropy = self._compute_entropy(sigma_tta)

        mu_off, sigma_off = self._get_offline_distribution(obs)
        kl_div = self._compute_kl_divergence(mu_tta, sigma_tta, mu_off, sigma_off)
        
        total_loss = -entropy + self.lambda_kl * kl_div

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {
            'entropy': entropy.item(),
            'kl_div': kl_div.item(),
            'total_loss': total_loss.item()
        }

    def _run_single_episode(self) -> Dict[str, Any]:
        """Run a single episode and perform STINT adaptation"""
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
        episode_triggers = []

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

            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            
            triggered = self._check_trigger(obs_tensor)
            episode_triggers.append(triggered)

            if triggered:
                for k in range(self.K):
                    loss_info = self._stabilization_update(obs_tensor)
                    episode_loss.append(loss_info['total_loss'])
                    episode_entropy.append(loss_info['entropy'])
                    episode_kl.append(loss_info['kl_div'])

            obs = next_obs
            episode_reward += reward
            episode_length += 1
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
            'episode_kl': episode_kl,
            'num_triggers': sum(episode_triggers)
        }

    def run_adaptation(self, num_episodes: int = 10) -> Tuple[List[Dict], Dict[str, Any]]:
        """
        Run STINT adaptation for specified number of episodes
        
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
        trigger_counts = []

        for episode in range(num_episodes):
            episode_data = self._run_single_episode()
            adaptation_data.append(episode_data)

            if 'episode_loss' in episode_data and len(episode_data['episode_loss']) > 0:
                loss_history.extend(episode_data['episode_loss'])
                entropy_history.extend(episode_data['episode_entropy'])
                kl_history.extend(episode_data['episode_kl'])
            
            trigger_counts.append(episode_data.get('num_triggers', 0))

            tta_status = "STINT"
            print(f"Episode {episode + 1}/{num_episodes} ({tta_status}): "
                  f"Reward: {episode_data['episode_reward']:.2f}, "
                  f"Length: {episode_data['episode_length']}, "
                  f"Loss: {episode_data.get('adaptation_loss', 0):.6f}, "
                  f"Entropy: {episode_data.get('entropy', 0):.4f}, "
                  f"KL: {episode_data.get('kl_div', 0):.4f}, "
                  f"Triggers: {episode_data.get('num_triggers', 0)}")

        summary = {
            'mean_reward': np.mean([d['episode_reward'] for d in adaptation_data]),
            'std_reward': np.std([d['episode_reward'] for d in adaptation_data]),
            'mean_length': np.mean([d['episode_length'] for d in adaptation_data]),
            'mean_loss': np.mean(loss_history) if loss_history else 0,
            'mean_entropy': np.mean(entropy_history) if entropy_history else 0,
            'mean_kl': np.mean(kl_history) if kl_history else 0,
            'total_triggers': sum(trigger_counts),
            'mean_triggers_per_episode': np.mean(trigger_counts),
            'loss_history': loss_history,
            'entropy_history': entropy_history,
            'kl_history': kl_history,
            'adaptation_steps': self.adaptation_step,
            'final_entropy_moving_avg': self.entropy_moving_avg
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
        """Get STINT adaptation statistics"""
        return {
            'adaptation_steps': self.adaptation_step,
            'trigger_count': self.trigger_count,
            'entropy_moving_avg': self.entropy_moving_avg,
            'trainable_params': len(self.trainable_params),
            'learning_rate': self.learning_rate,
            'delta': self.delta,
            'lambda_kl': self.lambda_kl,
            'K': self.K,
            'beta': self.beta
        }