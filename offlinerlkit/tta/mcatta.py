import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from collections import deque

from offlinerlkit.policy.base_policy import BasePolicy


class PerformancePredictor(nn.Module):
    """
    Performance Predictor f_φ
    
    Predicts performance trend based on policy smoothness and exploration breadth
    without using explicit rewards.
    """

    def __init__(self, input_dim: int = 2, hidden_dim: int = 64, output_dim: int = 1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, smoothness: torch.Tensor, breadth: torch.Tensor) -> torch.Tensor:
        """
        Args:
            smoothness: Policy smoothness metric (gradient norm)
            breadth: Exploration breadth metric (state feature variance)
        
        Returns:
            Predicted performance change
        """
        x = torch.cat([smoothness, breadth], dim=-1)
        return self.network(x)


class PolicyCache:
    """
    Policy cache for storing recent policies and their associated metrics
    """

    def __init__(self, capacity: int = 10):
        self.capacity = capacity
        self.policies = deque(maxlen=capacity)
        self.metrics = deque(maxlen=capacity)
        self.performance_trends = deque(maxlen=capacity)

    def add(self, policy_state: Dict[str, torch.Tensor], 
            metrics: Dict[str, float], 
            performance_trend: float):
        """Add a policy to the cache"""
        self.policies.append(policy_state)
        self.metrics.append(metrics)
        self.performance_trends.append(performance_trend)

    def get_recent(self, k: int = 5) -> List[Tuple[Dict, Dict, float]]:
        """Get k most recent policies"""
        recent = list(zip(self.policies, self.metrics, self.performance_trends))
        return recent[-k:]

    def get_performance_history(self) -> List[float]:
        """Get history of performance trends"""
        return list(self.performance_trends)

    def __len__(self):
        return len(self.policies)


class MCATTAManager:
    """
    Meta-Conservative Adaptive Test-Time Adaptation (MCATTA)
    
    Implements risk-driven conservative adaptation where the conservatism parameter
    λ is dynamically adjusted based on the reflexive impact of policy updates.
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

        self.lambda_min = self.config.get('lambda_min', 0.1)
        self.lambda_max = self.config.get('lambda_max', 10.0)
        self.lambda_init = self.config.get('lambda_init', 1.0)
        self.lambda_lr = self.config.get('lambda_lr', 0.01)
        self.policy_lr = self.config.get('policy_lr', 1e-4)
        self.batch_size = self.config.get('batch_size', 32)
        self.cache_capacity = self.config.get('cache_capacity', 10)

        self.lambda_t = self.lambda_init
        self.previous_delta = 0.0

        self.policy_cache = PolicyCache(capacity=self.cache_capacity)

        self.performance_predictor = PerformancePredictor().to(self.device)
        self.predictor_optimizer = torch.optim.Adam(
            self.performance_predictor.parameters(),
            lr=self.config.get('predictor_lr', 1e-3)
        )

        self.experience_buffer = deque(maxlen=10000)

        self.initial_policy_state = self._save_policy_state()

        self.adaptation_step = 0

        if hasattr(self.policy, 'actor'):
            self.policy_optimizer = torch.optim.Adam(
                self.policy.actor.parameters(),
                lr=self.policy_lr
            )

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
        """Save current policy state"""
        state = {}
        for name, param in self.policy.named_parameters():
            state[name] = param.data.clone().detach()
        return state

    def _compute_policy_entropy(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute policy entropy"""
        if hasattr(self.policy, 'actor'):
            action_dist = self.policy.actor(obs)
            if hasattr(action_dist, 'entropy'):
                return action_dist.entropy().mean()
        return torch.tensor(0.0, device=self.device)

    def _compute_kl_divergence(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence between current policy and initial policy"""
        if not hasattr(self.policy, 'actor'):
            return torch.tensor(0.0, device=self.device)

        current_dist = self.policy.actor(obs)
        
        with torch.no_grad():
            initial_actor = self._create_initial_actor()
            initial_dist = initial_actor(obs)
        
        kl = torch.distributions.kl.kl_divergence(current_dist, initial_dist)
        return kl.mean()

    def _create_initial_actor(self):
        """Create a copy of the initial actor"""
        # 创建actor的深拷贝，避免使用类类型创建实例
        import copy
        initial_actor = copy.deepcopy(self.policy.actor)
        return initial_actor

    def _compute_policy_smoothness(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Compute policy smoothness: gradient norm of policy w.r.t. actions
        
        Lower gradient norm -> more stable policy
        """
        if not hasattr(self.policy, 'actor'):
            return torch.tensor(0.0, device=self.device)

        obs = obs.requires_grad_(True)
        action_dist = self.policy.actor(obs)
        
        if hasattr(action_dist, 'mean'):
            actions = action_dist.mean
        else:
            actions = action_dist.mode()[0]
        
        grad = torch.autograd.grad(
            actions.sum(), obs, 
            create_graph=True, retain_graph=True
        )[0]
        
        smoothness = torch.norm(grad, dim=-1).mean()
        return smoothness

    def _compute_exploration_breadth(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Compute exploration breadth: variance of state-action features
        
        Higher variance -> more exploration
        """
        if not hasattr(self.policy, 'actor'):
            return torch.tensor(0.0, device=self.device)

        with torch.no_grad():
            action_dist = self.policy.actor(obs)
            if hasattr(action_dist, 'mean'):
                actions = action_dist.mean
            else:
                actions = action_dist.mode()[0]
            
            state_action_features = torch.cat([obs, actions], dim=-1)
            breadth = torch.var(state_action_features, dim=0).mean()
        
        return breadth

    def _predict_performance_change(
        self, 
        smoothness: torch.Tensor, 
        breadth: torch.Tensor
    ) -> torch.Tensor:
        """Predict performance change using the performance predictor"""
        return self.performance_predictor(smoothness, breadth)

    def _update_performance_predictor(
        self, 
        smoothness: torch.Tensor, 
        breadth: torch.Tensor, 
        target_delta: float
    ):
        """Update the performance predictor using unsupervised signals"""
        self.predictor_optimizer.zero_grad()
        
        predicted_delta = self._predict_performance_change(smoothness, breadth)
        
        target = torch.tensor([[target_delta]], device=self.device)
        loss = F.mse_loss(predicted_delta, target)
        
        loss.backward()
        self.predictor_optimizer.step()
        
        return loss.item()

    def _update_lambda(self, predicted_delta: float):
        """
        Update the conservatism parameter λ based on predicted performance change
        
        If predicted improvement, decrease λ (more exploration)
        If predicted degradation, increase λ (more conservative)
        """
        delta_change = predicted_delta - self.previous_delta
        self.previous_delta = predicted_delta
        
        lambda_update = -self.lambda_lr * np.sign(delta_change)
        self.lambda_t += lambda_update
        self.lambda_t = np.clip(self.lambda_t, self.lambda_min, self.lambda_max)

    def _inner_loop_update(self, obs_batch: torch.Tensor) -> torch.Tensor:
        """
        Inner loop: Policy update with adaptive conservatism
        
        Loss = Entropy + λ * KL_Divergence
        """
        entropy = self._compute_policy_entropy(obs_batch)
        kl_div = self._compute_kl_divergence(obs_batch)
        
        lambda_tensor = torch.tensor(self.lambda_t, device=self.device)
        loss = -entropy + lambda_tensor * kl_div
        
        return loss

    def _outer_loop_update(self, obs_batch: torch.Tensor):
        """
        Outer loop: Conservatism parameter update
        
        Uses performance prediction to adjust λ
        """
        smoothness = self._compute_policy_smoothness(obs_batch)
        breadth = self._compute_exploration_breadth(obs_batch)
        
        smoothness_scalar = smoothness.detach().cpu().item()
        breadth_scalar = breadth.detach().cpu().item()
        
        predicted_delta = self._predict_performance_change(
            smoothness.unsqueeze(0),
            breadth.unsqueeze(0)
        ).item()
        
        self._update_lambda(predicted_delta)
        
        return {
            'smoothness': smoothness_scalar,
            'breadth': breadth_scalar,
            'predicted_delta': predicted_delta,
            'lambda': self.lambda_t
        }

    def _sample_batch(self, batch_size: Optional[int] = None) -> torch.Tensor:
        """Sample a batch from experience buffer"""
        batch_size = batch_size or self.batch_size
        
        if len(self.experience_buffer) < batch_size:
            return None
        
        indices = np.random.choice(len(self.experience_buffer), batch_size, replace=False)
        batch = [self.experience_buffer[i] for i in indices]
        
        obs_data = np.array([t['obs'] for t in batch])
        obs_batch = torch.FloatTensor(obs_data).to(self.device)
        
        return obs_batch

    def run_adaptation(self, num_episodes: int = 10) -> Tuple[List[Dict], Dict[str, Any]]:
        """
        Run MCATTA adaptation
        
        Args:
            num_episodes: Number of adaptation episodes
            
        Returns:
            adaptation_data: List of episode data
            summary: Summary statistics
        """
        adaptation_data = []
        lambda_history = []
        performance_history = []
        
        for episode in range(num_episodes):
            episode_data = self._run_single_episode()
            adaptation_data.append(episode_data)
            
            obs_batch = self._sample_batch()
            if obs_batch is not None:
                outer_metrics = self._outer_loop_update(obs_batch)
                
                self.policy_optimizer.zero_grad()
                loss = self._inner_loop_update(obs_batch)
                loss.backward()
                self.policy_optimizer.step()
                
                episode_data.update({
                    'lambda': outer_metrics['lambda'],
                    'smoothness': outer_metrics['smoothness'],
                    'breadth': outer_metrics['breadth'],
                    'predicted_delta': outer_metrics['predicted_delta'],
                    'adaptation_loss': loss.item()
                })
                
                lambda_history.append(outer_metrics['lambda'])
                performance_history.append(outer_metrics['predicted_delta'])
                
                policy_state = self._save_policy_state()
                self.policy_cache.add(
                    policy_state,
                    {
                        'smoothness': outer_metrics['smoothness'],
                        'breadth': outer_metrics['breadth']
                    },
                    outer_metrics['predicted_delta']
                )
            
            self.adaptation_step += 1
            
            print(f"Episode {episode + 1}/{num_episodes}: "
                  f"Reward: {episode_data['episode_reward']:.2f}, "
                  f"Lambda: {self.lambda_t:.3f}, "
                  f"Predicted Delta: {episode_data.get('predicted_delta', 0):.4f}")
        
        summary = {
            'final_lambda': self.lambda_t,
            'lambda_history': lambda_history,
            'performance_history': performance_history,
            'mean_reward': np.mean([d['episode_reward'] for d in adaptation_data]),
            'std_reward': np.std([d['episode_reward'] for d in adaptation_data]),
            'cache_size': len(self.policy_cache)
        }
        
        return adaptation_data, summary

    def _run_single_episode(self) -> Dict[str, Any]:
        """Run a single episode and collect data"""
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
            
            self.experience_buffer.append(transition)
            
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

    def evaluate_performance(self, num_episodes: int = 5) -> Dict[str, float]:
        """Evaluate current policy performance"""
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

    def save_checkpoint(self, save_path: str):
        """Save MCATTA checkpoint"""
        checkpoint = {
            'policy_state_dict': self.policy.state_dict(),
            'lambda_t': self.lambda_t,
            'adaptation_step': self.adaptation_step,
            'performance_predictor_state_dict': self.performance_predictor.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'predictor_optimizer_state_dict': self.predictor_optimizer.state_dict(),
            'config': self.config
        }
        torch.save(checkpoint, save_path)

    def load_checkpoint(self, checkpoint_path: str):
        """Load MCATTA checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.lambda_t = checkpoint['lambda_t']
        self.adaptation_step = checkpoint['adaptation_step']
        self.performance_predictor.load_state_dict(checkpoint['performance_predictor_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.predictor_optimizer.load_state_dict(checkpoint['predictor_optimizer_state_dict'])
        self.config = checkpoint.get('config', {})
