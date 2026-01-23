import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from collections import deque

from offlinerlkit.policy.base_policy import BasePolicy


class EntropyDynamicsPredictor(nn.Module):
    """
    Entropy Dynamics Predictor f_φ
    
    Predicts entropy evolution as a conditional Gaussian distribution.
    f_φ(S_t, H_t) = (μ_t, logσ_t²)
    
    Uses prediction error as uncertainty signal for OOD detection.
    """

    def __init__(self, input_dim: int = 2, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, smoothness: torch.Tensor, entropy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            smoothness: Policy smoothness metric S_t = ||∇_s π_θ_t||
            entropy: Current policy entropy H_t
        
        Returns:
            mu: Predicted mean of entropy change (Ḣ_t)
            log_var: Log of predicted variance (for numerical stability)
        """
        if smoothness.dim() == 0:
            smoothness = smoothness.unsqueeze(0)
        if entropy.dim() == 0:
            entropy = entropy.unsqueeze(0)
        
        x = torch.cat([smoothness, entropy], dim=-1)
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        output = self.network(x)
        
        if output.dim() == 2:
            mu = output[:, 0]
            log_var = output[:, 1]
        else:
            mu = output[0]
            log_var = output[1]
            
        log_var = torch.clamp(log_var, min=-5, max=2)
        return mu.squeeze(-1), log_var.squeeze(-1)


class EntropyDynamicsCache:
    """
    Entropy Dynamics Cache D_H for storing (S_t, H_t, Ḣ_t) triplets.
    
    Used for:
    - Computing entropy acceleration ẍ_t
    - Training the entropy dynamics predictor
    - OOD detection via prediction error
    """

    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self.smoothness_history = deque(maxlen=capacity)
        self.entropy_history = deque(maxlen=capacity)
        self.entropy_rate_history = deque(maxlen=capacity)
        self.uncertainty_history = deque(maxlen=capacity)

    def add(self, smoothness: float, entropy: float, entropy_rate: float, uncertainty: float = 0.0):
        """Add entropy dynamics triplet to cache"""
        self.smoothness_history.append(smoothness)
        self.entropy_history.append(entropy)
        self.entropy_rate_history.append(entropy_rate)
        self.uncertainty_history.append(uncertainty)

    def get_recent(self, k: int = 5) -> List[Tuple[float, float, float, float]]:
        """Get k most recent entropy dynamics"""
        n = len(self)
        if n == 0:
            return []
        recent_k = min(k, n)
        return [
            (self.smoothness_history[-(i+1)], 
             self.entropy_history[-(i+1)], 
             self.entropy_rate_history[-(i+1)],
             self.uncertainty_history[-(i+1)])
            for i in range(recent_k)
        ][::-1]

    def compute_entropy_acceleration(self) -> float:
        """
        Compute entropy acceleration (curvature of entropy change)
        
        ẍ_t = ((H_t - H_{t-1}) - (H_{t-1} - H_{t-2})) / Δt²
        
        Δt is assumed to be 1 episode for simplicity.
        """
        if len(self.entropy_rate_history) < 2:
            return 0.0
        
        H_t = self.entropy_history[-1] if len(self.entropy_history) > 0 else 0.0
        H_t_1 = self.entropy_history[-2] if len(self.entropy_history) > 1 else 0.0
        H_t_2 = self.entropy_history[-3] if len(self.entropy_history) > 2 else 0.0
        
        H_dot_t = H_t - H_t_1
        H_dot_t_1 = H_t_1 - H_t_2
        
        entropy_acc = H_dot_t - H_dot_t_1
        return entropy_acc

    def get_uncertainty_statistics(self) -> Tuple[float, float]:
        """Get mean and std of uncertainty history"""
        if len(self.uncertainty_history) == 0:
            return 0.0, 0.0
        uncertainty_arr = np.array(self.uncertainty_history)
        return float(np.mean(uncertainty_arr)), float(np.std(uncertainty_arr))

    def __len__(self):
        return min(len(self.smoothness_history), self.capacity)

    def clear(self):
        """Clear the cache"""
        self.smoothness_history.clear()
        self.entropy_history.clear()
        self.entropy_rate_history.clear()
        self.uncertainty_history.clear()


class EDMSAManager:
    """
    Entropy-based Drift-adaptive Meta-Learning for Soft Actor-Critic (EDMSA)
    
    Algorithm for reward-free online adaptation using entropy as the only 
    observable performance proxy.
    
    Key innovations:
    1. Entropy acceleration ẍ_t for OOD detection
    2. Uncertainty-driven λ evolution via prediction error
    3. Adaptive learning rate for catastrophic drift prevention
    4. LayerNorm-only parameter updates for stable adaptation
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
        self.lambda_equilibrium = self.config.get('lambda_equilibrium', self.lambda_min)
        
        self.policy_lr = self.config.get('policy_lr', 1e-4)
        self.batch_size = self.config.get('batch_size', 32)
        self.cache_capacity = self.config.get('cache_capacity', 100)
        
        self.momentum = self.config.get('momentum', 0.9)
        self.damping_coef = self.config.get('damping_coef', 1.0)
        self.eta_lambda = self.config.get('eta_lambda', 0.01)
        self.gamma_dissipation = self.config.get('gamma_dissipation', 0.1)
        
        self.ood_threshold_std = self.config.get('ood_threshold_std', 2.0)
        self.lr_decay_factor = self.config.get('lr_decay_factor', 0.5)
        self.lr_recovery_steps = self.config.get('lr_recovery_steps', 10)
        
        self.adaptation_mode = self.config.get('adaptation_mode', 'layernorm')
        self.last_n_layers = self.config.get('last_n_layers', 2)
        
        self.lambda_t = self.lambda_init
        self.lambda_momentum = 0.0

        self.entropy_cache = EntropyDynamicsCache(capacity=self.cache_capacity)

        self.entropy_predictor = EntropyDynamicsPredictor().to(self.device)
        self.predictor_optimizer = torch.optim.Adam(
            self.entropy_predictor.parameters(),
            lr=self.config.get('predictor_lr', 1e-3)
        )

        self.experience_buffer = deque(maxlen=10000)

        self.initial_policy_state = self._save_policy_state()
        self._create_offline_actor()

        self.adaptation_step = 0
        self.current_policy_lr = self.policy_lr
        self.lr_recovery_counter = 0
        self.is_adapting_lr = False

        self.entropy_history = []
        self.uncertainty_history = []

        if hasattr(self.policy, 'actor'):
            self._setup_trainable_params()
            self.policy_optimizer = torch.optim.Adam(
                self.trainable_params,
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

    def _create_offline_actor(self):
        """Create a frozen copy of the initial actor for KL divergence computation"""
        import copy
        self._offline_actor = copy.deepcopy(self.policy.actor)
        for param in self._offline_actor.parameters():
            param.requires_grad = False

    def _setup_trainable_params(self):
        """
        Setup trainable parameters based on adaptation mode
        
        Modes:
        - 'layernorm': Only update LayerNorm parameters (most stable)
        - 'last_n_layers': Update last N layers (balanced)
        - 'all': Update all parameters (most adaptive, least stable)
        """
        if self.adaptation_mode == 'all':
            self.trainable_params = list(self.policy.actor.parameters())
            return
        
        trainable_params = []
        
        if self.adaptation_mode == 'layernorm':
            trainable_params = self._get_layernorm_params()
            if len(trainable_params) == 0:
                print("警告: 未找到LayerNorm参数，自动切换到last_n_layers模式")
                self.adaptation_mode = 'last_n_layers'
        
        if self.adaptation_mode == 'last_n_layers':
            trainable_params = self._get_last_n_layers_params()
        
        self.trainable_params = [param for _, param in trainable_params]

    def _get_layernorm_params(self) -> List[Tuple[str, nn.Parameter]]:
        """Extract LayerNorm parameters for trainable adaptation"""
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

    def _get_last_n_layers_params(self, n: int = None) -> List[Tuple[str, nn.Parameter]]:
        """Get parameters from the last N layers of the actor network"""
        if n is None:
            n = self.last_n_layers
        
        trainable_params = []
        
        if not hasattr(self.policy, 'actor'):
            return trainable_params
        
        actor_modules = list(self.policy.actor.modules())
        num_modules = len(actor_modules)
        
        last_n_start = max(0, num_modules - n)
        
        for i, (name, module) in enumerate(self.policy.actor.named_modules()):
            if i >= last_n_start:
                for param_name, param in module.named_parameters():
                    if param.requires_grad:
                        full_name = f"actor.{name}.{param_name}" if name else f"actor.{param_name}"
                        trainable_params.append((full_name, param))
        
        return trainable_params

    def _compute_policy_entropy(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute policy entropy"""
        if hasattr(self.policy, 'actor'):
            action_dist = self.policy.actor(obs)
            if hasattr(action_dist, 'entropy'):
                return action_dist.entropy().mean()
        return torch.tensor(0.0, device=self.device)

    def _compute_kl_divergence(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence between current policy and frozen offline policy"""
        if not hasattr(self.policy, 'actor') or not hasattr(self, '_offline_actor'):
            return torch.tensor(0.0, device=self.device)

        current_dist = self.policy.actor(obs)
        
        with torch.no_grad():
            initial_dist = self._offline_actor(obs)
        
        kl = torch.distributions.kl.kl_divergence(current_dist, initial_dist)
        return kl.mean()

    def _compute_policy_smoothness(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Compute policy smoothness: gradient norm of policy w.r.t. states
        
        S_t = E[||∇_s π_θ(a|s)||_2]
        """
        if not hasattr(self.policy, 'actor'):
            return torch.tensor(0.0, device=self.device)
        
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

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
        
        if grad is None:
            return torch.tensor(0.0, device=self.device)
        
        smoothness = torch.norm(grad, dim=-1).mean()
        return smoothness

    def _compute_episode_entropy(self, obs_batch: torch.Tensor) -> float:
        """
        Compute episode-averaged policy entropy
        
        H_t = E_{s~τ_t}[H(π_θt(·|s))]
        """
        entropy = self._compute_policy_entropy(obs_batch)
        return entropy.detach().cpu().item()

    def _predict_entropy_dynamics(
        self, 
        smoothness: torch.Tensor, 
        entropy: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict entropy change using the entropy dynamics predictor
        
        f_φ(S_t, H_t) = (μ_t, logσ_t²)
        
        Returns:
            mu: Predicted mean of entropy change (Ḣ_t)
            log_var: Log variance for uncertainty estimation
        """
        mu, log_var = self.entropy_predictor(smoothness, entropy)
        return mu, log_var

    def _compute_prediction_uncertainty(
        self, 
        predicted_mu: float, 
        predicted_log_var: float, 
        actual_entropy_rate: float
    ) -> float:
        """
        Compute prediction uncertainty as negative log-likelihood
        
        U_t = (Ḣ_t - μ_t)² / (2σ_t²) + 0.5·log(2πσ_t²)
        
        This is the NLL of the true entropy dynamics under the predicted distribution.
        """
        var = np.exp(np.clip(predicted_log_var, -5, 2))
        nll = (actual_entropy_rate - predicted_mu) ** 2 / (2 * var) + 0.5 * np.log(2 * np.pi * var)
        return float(nll)

    def _train_predictor_maximum_likelihood(
        self, 
        smoothness: torch.Tensor, 
        entropy: torch.Tensor, 
        entropy_rate: torch.Tensor
    ) -> float:
        """
        Train the entropy dynamics predictor with maximum likelihood
        
        L_φ^t = -log p_φ(Ḣ_t | S_t, H_t) = U_t
        
        The loss is exactly the uncertainty (NLL), so we minimize uncertainty.
        """
        self.predictor_optimizer.zero_grad()
        
        mu, log_var = self.entropy_predictor(smoothness, entropy)
        
        if mu.dim() == 0:
            mu = mu.unsqueeze(0)
            log_var = log_var.unsqueeze(0)
            entropy_rate = entropy_rate.unsqueeze(0) if entropy_rate.dim() == 0 else entropy_rate
        
        var = torch.exp(torch.clamp(log_var, min=-5, max=2))
        nll = 0.5 * torch.log(2 * np.pi * var) + 0.5 * (entropy_rate - mu) ** 2 / var
        loss = nll.mean()
        
        loss.backward()
        self.predictor_optimizer.step()
        
        return loss.item()

    def _outer_loop_lambda_evolution(self, uncertainty: float) -> Dict[str, float]:
        """
        Outer loop: λ evolution via gradient flow dynamics
        
        dλ/dt = η_λ·U_t - γ·(λ_t - λ_min)
        
        Discrete: λ_{t+1} = λ_t + Δt·[η_λ·U_t - γ·(λ_t - λ_min)]
        
        Returns:
            Dict containing evolution metrics
        """
        delta_lambda = self.eta_lambda * uncertainty - self.gamma_dissipation * (self.lambda_t - self.lambda_equilibrium)
        
        self.lambda_momentum = self.momentum * self.lambda_momentum + delta_lambda
        self.lambda_t = self.lambda_t + self.lambda_momentum
        self.lambda_t = np.clip(self.lambda_t, self.lambda_min, self.lambda_max)
        
        return {
            'delta_lambda': delta_lambda,
            'lambda_momentum': self.lambda_momentum
        }

    def _check_ood_and_adapt_lr(self, uncertainty: float) -> bool:
        """
        Check for OOD and adapt learning rate accordingly
        
        If U_t > E[U] + 2σ_U, trigger OOD and reduce learning rate.
        
        Returns:
            True if OOD detected and lr adapted
        """
        mean_u, std_u = self.entropy_cache.get_uncertainty_statistics()
        
        if mean_u > 0 and std_u > 0:
            threshold = mean_u + self.ood_threshold_std * std_u
            if uncertainty > threshold:
                self.current_policy_lr = self.policy_lr * self.lr_decay_factor
                self.lr_recovery_counter = self.lr_recovery_steps
                self.is_adapting_lr = True
                return True
        
        if self.is_adapting_lr:
            self.lr_recovery_counter -= 1
            if self.lr_recovery_counter <= 0:
                self.current_policy_lr = self.policy_lr
                self.is_adapting_lr = False
        
        return False

    def _load_policy_state(self, policy_state: Dict[str, torch.Tensor]):
        """Load policy state from saved state dict"""
        with torch.no_grad():
            for name, param in policy_state.items():
                if hasattr(self.policy, name):
                    target = getattr(self.policy, name)
                    if isinstance(target, nn.Parameter):
                        target.data.copy_(param)
                    elif hasattr(target, 'data'):
                        target.data.copy_(param)

    def _inner_loop_update(self, obs_batch: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Inner loop: Policy update with entropy-regularized KL control
        
        L_π^t(θ) = -H_t + (λ_t + β·|ẍ_t|)·D_KL^t
        
        where:
        - H_t = E[H(π_θt(·|s))] is the policy entropy
        - D_KL^t = E[D_KL(π_θt || π_θ0)] is the KL divergence from anchor
        - λ_t is the conservatism coefficient
        - β is the damping coefficient
        - ẍ_t is the entropy acceleration
        
        Returns:
            loss: The computed loss tensor
            entropy_acc: The entropy acceleration used for damping
        """
        entropy = self._compute_policy_entropy(obs_batch)
        kl_div = self._compute_kl_divergence(obs_batch)
        
        entropy_acc = self.entropy_cache.compute_entropy_acceleration()
        damping = self.damping_coef * abs(entropy_acc)
        
        lambda_tensor = torch.tensor(self.lambda_t, device=self.device)
        loss = -entropy + (lambda_tensor + damping) * kl_div
        
        return loss, entropy_acc

    def _outer_loop_update(self) -> Dict[str, Any]:
        """
        Outer loop: Entropy dynamics prediction and λ evolution
        
        1. Compute meta-features (S_t, H_t)
        2. Predict entropy change Ḣ_t using f_φ
        3. Compute prediction uncertainty U_t (NLL)
        4. Update λ via gradient flow dynamics
        5. Train predictor with maximum likelihood
        
        Returns:
            Dict containing all metrics for this update
        """
        if len(self.experience_buffer) < self.batch_size:
            return {
                'smoothness': 0.0,
                'entropy': 0.0,
                'entropy_rate': 0.0,
                'predicted_mu': 0.0,
                'predicted_log_var': 0.0,
                'uncertainty': 0.0,
                'lambda': self.lambda_t,
                'entropy_acceleration': 0.0,
                'predictor_loss': 0.0
            }
        
        obs_batch = self._sample_batch()
        if obs_batch is None:
            return {
                'smoothness': 0.0,
                'entropy': 0.0,
                'entropy_rate': 0.0,
                'predicted_mu': 0.0,
                'predicted_log_var': 0.0,
                'uncertainty': 0.0,
                'lambda': self.lambda_t,
                'entropy_acceleration': 0.0,
                'predictor_loss': 0.0
            }
        
        smoothness = self._compute_policy_smoothness(obs_batch)
        entropy = self._compute_policy_entropy(obs_batch)
        
        smoothness_scalar = smoothness.detach().cpu().item()
        entropy_scalar = entropy.detach().cpu().item()
        
        if len(self.entropy_history) > 0:
            prev_entropy = self.entropy_history[-1]
            entropy_rate = entropy_scalar - prev_entropy
        else:
            entropy_rate = 0.0
        
        predicted_mu, predicted_log_var = self._predict_entropy_dynamics(
            smoothness.unsqueeze(0),
            entropy.unsqueeze(0)
        )
        
        predicted_mu_scalar = predicted_mu.item()
        predicted_log_var_scalar = predicted_log_var.item()
        
        uncertainty = self._compute_prediction_uncertainty(
            predicted_mu_scalar,
            predicted_log_var_scalar,
            entropy_rate
        )
        
        self._outer_loop_lambda_evolution(uncertainty)
        
        self.entropy_cache.add(smoothness_scalar, entropy_scalar, entropy_rate, uncertainty)
        
        predictor_loss = self._train_predictor_maximum_likelihood(
            smoothness.unsqueeze(0),
            entropy.unsqueeze(0),
            torch.tensor(entropy_rate, device=self.device)
        )
        
        self.entropy_history.append(entropy_scalar)
        self.uncertainty_history.append(uncertainty)
        
        entropy_acc = self.entropy_cache.compute_entropy_acceleration()
        
        return {
            'smoothness': smoothness_scalar,
            'entropy': entropy_scalar,
            'entropy_rate': entropy_rate,
            'predicted_mu': predicted_mu_scalar,
            'predicted_log_var': predicted_log_var_scalar,
            'uncertainty': uncertainty,
            'lambda': self.lambda_t,
            'entropy_acceleration': entropy_acc,
            'predictor_loss': predictor_loss
        }

    def _sample_batch(self, batch_size: Optional[int] = None) -> Optional[torch.Tensor]:
        """Sample a batch from experience buffer"""
        batch_size = batch_size or self.batch_size
        
        if len(self.experience_buffer) < batch_size:
            return None
        
        sample_size = min(len(self.experience_buffer), batch_size)
        if sample_size == 0:
            return None
        
        indices = np.random.choice(len(self.experience_buffer), sample_size, replace=False)
        batch = [self.experience_buffer[i] for i in indices]
        
        try:
            obs_data = np.array([t['obs'] for t in batch])
            obs_batch = torch.FloatTensor(obs_data).to(self.device)
            return obs_batch
        except Exception as e:
            return None

    def run_adaptation(self, num_episodes: int = 10) -> Tuple[List[Dict], Dict[str, Any]]:
        """
        Run EDMSA adaptation (Entropy-based Drift-adaptive Meta-Learning)
        
        Algorithm:
        1. Online interaction and data collection
        2. Inner loop: Entropy-regularized policy improvement with acceleration damping
        3. Outer loop: Uncertainty-driven λ evolution
        4. Adaptive learning rate for OOD detection
        
        Args:
            num_episodes: Number of adaptation episodes
            
        Returns:
            adaptation_data: List of episode data
            summary: Summary statistics
        """
        adaptation_data = []
        lambda_history = []
        uncertainty_history = []
        entropy_history = []
        entropy_acc_history = []
        ood_detected_count = 0
        
        for episode in range(num_episodes):
            episode_data = self._run_single_episode()
            adaptation_data.append(episode_data)
            
            outer_metrics = self._outer_loop_update()
            
            if hasattr(self.policy, 'actor') and hasattr(self, 'policy_optimizer'):
                obs_batch = self._sample_batch()
                if obs_batch is not None:
                    self.policy_optimizer.zero_grad()
                    loss, entropy_acc = self._inner_loop_update(obs_batch)
                    loss.backward()
                    
                    for param_group in self.policy_optimizer.param_groups:
                        param_group['lr'] = self.current_policy_lr
                    
                    self.policy_optimizer.step()
                    
                    episode_data['adaptation_loss'] = loss.item()
                    episode_data['entropy_acceleration'] = entropy_acc
            
            ood_detected = self._check_ood_and_adapt_lr(outer_metrics['uncertainty'])
            if ood_detected:
                ood_detected_count += 1
                episode_data['ood_detected'] = True
            
            episode_data.update({
                'lambda': outer_metrics['lambda'],
                'smoothness': outer_metrics['smoothness'],
                'entropy': outer_metrics['entropy'],
                'entropy_rate': outer_metrics['entropy_rate'],
                'predicted_mu': outer_metrics['predicted_mu'],
                'predicted_log_var': outer_metrics['predicted_log_var'],
                'uncertainty': outer_metrics['uncertainty'],
                'predictor_loss': outer_metrics['predictor_loss'],
                'current_lr': self.current_policy_lr,
                'lr_adapting': self.is_adapting_lr
            })
            
            lambda_history.append(outer_metrics['lambda'])
            uncertainty_history.append(outer_metrics['uncertainty'])
            entropy_history.append(outer_metrics['entropy'])
            entropy_acc_history.append(outer_metrics['entropy_acceleration'])
            
            self.adaptation_step += 1
            
            lr_status = f", LR: {self.current_policy_lr:.2e}" if self.is_adapting_lr else ""
            print(f"Episode {episode + 1}/{num_episodes}: "
                  f"Reward: {episode_data['episode_reward']:.2f}, "
                  f"Entropy: {outer_metrics['entropy']:.3f}, "
                  f"Lambda: {self.lambda_t:.3f}, "
                  f"Uncertainty: {outer_metrics['uncertainty']:.4f}, "
                  f"OOD: {'Yes' if ood_detected else 'No'}{lr_status}")
        
        summary = {
            'final_lambda': self.lambda_t,
            'lambda_history': lambda_history,
            'uncertainty_history': uncertainty_history,
            'entropy_history': entropy_history,
            'entropy_acceleration_history': entropy_acc_history,
            'mean_reward': np.mean([d['episode_reward'] for d in adaptation_data]),
            'std_reward': np.std([d['episode_reward'] for d in adaptation_data]),
            'final_uncertainty': uncertainty_history[-1] if uncertainty_history else 0.0,
            'final_entropy': entropy_history[-1] if entropy_history else 0.0,
            'ood_detected_count': ood_detected_count,
            'final_lr': self.current_policy_lr
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
        """Save EDMSA checkpoint"""
        checkpoint = {
            'policy_state_dict': self.policy.state_dict(),
            'lambda_t': self.lambda_t,
            'lambda_momentum': self.lambda_momentum,
            'adaptation_step': self.adaptation_step,
            'entropy_predictor_state_dict': self.entropy_predictor.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'predictor_optimizer_state_dict': self.predictor_optimizer.state_dict(),
            'entropy_history': self.entropy_history,
            'uncertainty_history': self.uncertainty_history,
            'config': self.config
        }
        torch.save(checkpoint, save_path)

    def load_checkpoint(self, checkpoint_path: str):
        """Load EDMSA checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.lambda_t = checkpoint['lambda_t']
        self.lambda_momentum = checkpoint.get('lambda_momentum', 0.0)
        self.adaptation_step = checkpoint['adaptation_step']
        self.entropy_predictor.load_state_dict(checkpoint['entropy_predictor_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.predictor_optimizer.load_state_dict(checkpoint['predictor_optimizer_state_dict'])
        self.entropy_history = checkpoint.get('entropy_history', [])
        self.uncertainty_history = checkpoint.get('uncertainty_history', [])
        self.config = checkpoint.get('config', {})
