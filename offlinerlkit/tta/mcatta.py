import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from collections import deque

from offlinerlkit.policy.base_policy import BasePolicy


class ContrastiveCache:
    """
    Contrastive Cache for CCEA algorithm
    
    Maintains positive and negative caches with entropy-priority replacement:
    - C_pos: High-quality episodes (y_t = +1)
    - C_neg: Low-quality episodes (y_t = -1)
    
    Each entry stores (x_t, H_t) where x_t = [H_t, ΔH_t, S_t, V_t]^⊤
    """

    def __init__(self, pos_capacity: int = 100, neg_capacity: int = 100):
        self.pos_capacity = pos_capacity
        self.neg_capacity = neg_capacity
        self.pos_cache = deque(maxlen=pos_capacity)
        self.neg_cache = deque(maxlen=neg_capacity)

    def add_to_pos(self, meta_features: List[float], entropy: float):
        """Add to positive cache with entropy-priority replacement"""
        self.pos_cache.append((meta_features, entropy))

    def add_to_neg(self, meta_features: List[float], entropy: float):
        """Add to negative cache with entropy-priority replacement"""
        self.neg_cache.append((meta_features, entropy))

    def get_pos_entries(self) -> List[Tuple[List[float], float]]:
        """Get all positive cache entries"""
        return list(self.pos_cache)

    def get_neg_entries(self) -> List[Tuple[List[float], float]]:
        """Get all negative cache entries"""
        return list(self.neg_cache)

    def compute_contrastive_distance(self, query_features: List[float]) -> Tuple[float, float]:
        """
        Compute contrastive distances to positive and negative caches
        
        Returns:
            d_pos: Minimum distance to positive cache
            d_neg: Minimum distance to negative cache
        """
        d_pos = float('inf')
        d_neg = float('inf')
        
        if len(self.pos_cache) > 0:
            pos_features = np.array([entry[0] for entry in self.pos_cache])
            query_arr = np.array(query_features)
            distances = np.linalg.norm(pos_features - query_arr, axis=1)
            d_pos = float(np.min(distances))
        
        if len(self.neg_cache) > 0:
            neg_features = np.array([entry[0] for entry in self.neg_cache])
            query_arr = np.array(query_features)
            distances = np.linalg.norm(neg_features - query_arr, axis=1)
            d_neg = float(np.min(distances))
        
        return d_pos, d_neg

    def __len__(self):
        return len(self.pos_cache) + len(self.neg_cache)

    def clear(self):
        """Clear both caches"""
        self.pos_cache.clear()
        self.neg_cache.clear()


class CCEAManager:
    """
    Contrastive Cache-based Entropic Adaptation (CCEA) Algorithm
    
    Algorithm for stable online adaptation using contrastive caching and entropy-driven adaptation.
    
    Key innovations:
    1. 4D meta-feature extraction: x_t = [H_t, ΔH_t, S_t, V_t]^⊤
    2. Episode quality labeling: y_t ∈ {+1, -1, 0}
    3. Contrastive uncertainty: U_t^cont = σ((d_pos - d_neg) / τ)
    4. Dual-cache system with entropy-priority replacement
    5. Lyapunov-stable λ evolution via contrastive uncertainty
    6. LayerNorm-only parameter updates for stability
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

        self.lambda_min = self.config.get('lambda_min', 0.01)
        self.lambda_max = self.config.get('lambda_max', 2.0)
        self.lambda_init = self.config.get('lambda_init', 1.0)
        
        self.policy_lr = self.config.get('policy_lr', 1e-4)
        self.batch_size = self.config.get('batch_size', 32)
        self.pos_cache_capacity = self.config.get('pos_cache_capacity', 20)
        self.neg_cache_capacity = self.config.get('neg_cache_capacity', 20)
        
        self.gamma = self.config.get('gamma', 0.5)  # Episode-level entropy suppression coefficient
        self.tau = self.config.get('tau', 1.0)     # Temperature parameter for sigmoid
        
        self.entropy_low = self.config.get('entropy_low', 0.5)
        self.entropy_high = self.config.get('entropy_high', 2.0)
        self.delta_stable = self.config.get('delta_stable', 0.1)
        self.v_min = self.config.get('v_min', 0.1)
        
        self.warmup_episodes = self.config.get('warmup_episodes', 5)
        self.step_update_interval = self.config.get('step_update_interval', 50)
        self.step_lambda_momentum = self.config.get('step_lambda_momentum', 0.1)
        self.step_novelty_set_size = self.config.get('step_novelty_set_size', 50)
        self.inner_loop_step_interval = self.config.get('inner_loop_step_interval', 30)
        
        self.adaptation_mode = self.config.get('adaptation_mode', 'layernorm')
        
        self.lambda_t = self.lambda_init
        self.step_lambda_ema = self.lambda_init

        self.contrastive_cache = ContrastiveCache(
            pos_capacity=self.pos_cache_capacity,
            neg_capacity=self.neg_cache_capacity
        )

        self.experience_buffer = deque(maxlen=10000)

        self.initial_policy_state = self._save_policy_state()
        self._create_offline_actor()

        self.adaptation_step = 0

        self.entropy_history = []
        self.entropy_velocity_history = []
        self.smoothness_history = []
        self.novelty_history = []
        self.contrastive_uncertainty_history = []

        self.warmup_completed = False
        self.step_buffer = deque(maxlen=50)
        self.step_entropy_window = deque(maxlen=20)
        self.step_novelty_set = set()
        self.current_episode_transitions = []

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

    def _compute_state_novelty(self, episode_transitions: List[Dict]) -> float:
        """
        Compute state novelty ratio V_t
        
        V_t = UniqueStates(τ_t) / |τ_t|
        
        Args:
            episode_transitions: List of transitions from the episode
            
        Returns:
            novelty_ratio: Ratio of unique states to total states
        """
        if len(episode_transitions) == 0:
            return 0.0
        
        unique_states = set()
        for transition in episode_transitions:
            obs = transition['obs']
            obs_tuple = tuple(obs) if isinstance(obs, np.ndarray) else obs
            unique_states.add(obs_tuple)
        
        novelty_ratio = len(unique_states) / len(episode_transitions)
        return novelty_ratio

    def _compute_episode_quality_label(self, H_t: float, ΔH_t: float, V_t: float) -> int:
        """
        Compute episode quality label y_t
        
        y_t = +1 if H_t ∈ [H_low, H_high] and |ΔH_t| < δ_stable (high quality)
        y_t = -1 if H_t > H_high or V_t < v_min (low quality)
        y_t = 0 otherwise (neutral)
        
        Args:
            H_t: Current episode entropy
            ΔH_t: Entropy velocity
            V_t: State novelty ratio
            
        Returns:
            y_t: Episode quality label ∈ {+1, -1, 0}
        """
        # Adaptive novelty threshold: use max(0.5, mean of recent 10 novelty values)
        adaptive_v_min = max(0.5, np.mean(self.novelty_history[-10:]) if len(self.novelty_history) >= 10 else 0.5)
        
        if (self.entropy_low <= H_t <= self.entropy_high and 
            abs(ΔH_t) < self.delta_stable):
            return 1
        elif H_t > self.entropy_high or V_t < adaptive_v_min:
            return -1
        else:
            return 0

    def _compute_contrastive_uncertainty(self, meta_features: List[float]) -> float:
        """
        Compute contrastive uncertainty U_t^cont
        
        U_t^cont = σ((d_pos - d_neg) / τ)
        
        where:
        - d_pos = min distance to positive cache
        - d_neg = min distance to negative cache
        - σ is sigmoid function
        - τ is temperature parameter
        
        Args:
            meta_features: 4D meta-feature vector [H_t, ΔH_t, S_t, V_t]
            
        Returns:
            U_t^cont: Contrastive uncertainty ∈ (0, 1)
        """
        d_pos, d_neg = self.contrastive_cache.compute_contrastive_distance(meta_features)
        
        if d_pos == float('inf') and d_neg == float('inf'):
            return 0.5
        
        if d_pos == float('inf'):
            return 0.9
        
        if d_neg == float('inf'):
            return 0.1
        
        U_cont = 1.0 / (1.0 + np.exp(-(d_pos - d_neg) / self.tau))
        return U_cont

    def _warmup_thresholds(self):
        """
        Run warm-up episodes to dynamically initialize entropy thresholds
        
        Uses offline policy to collect entropy statistics for 5 episodes
        Implements per-episode adaptive threshold updates:
        - After every 3 episodes, update thresholds using recent 5 episodes
        - entropy_high = 75th percentile of recent entropies
        - entropy_low = 25th percentile of recent entropies
        - Initial entropy_high = max(warmup_mean + std, 0.15)
        """
        print("Running warm-up to initialize dynamic thresholds...")
        
        warmup_entropies = []
        episode_count = 0
        
        for _ in range(self.warmup_episodes):
            reset_result = self.env.reset()
            if isinstance(reset_result, tuple):
                obs, _ = reset_result
            else:
                obs = reset_result
            
            episode_entropy = 0.0
            episode_count = 0
            
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
                
                obs_batch = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                entropy = self._compute_policy_entropy(obs_batch)
                episode_entropy += entropy.detach().cpu().item()
                episode_count += 1
                
                obs = next_obs
                
                if episode_count >= 1000:
                    break
            
            if episode_count > 0:
                avg_entropy = episode_entropy / episode_count
                warmup_entropies.append(avg_entropy)
                episode_count += 1
                
                # Per-episode adaptive threshold update: every 3 episodes
                if episode_count % 3 == 0 and len(warmup_entropies) >= 5:
                    recent_entropies = warmup_entropies[-5:]
                    self.entropy_low = np.percentile(recent_entropies, 25)
                    self.entropy_high = np.percentile(recent_entropies, 75)
                    print(f"  Episode {episode_count}: Updated thresholds using recent 5 episodes")
                    print(f"    Recent entropies: {[f'{e:.4f}' for e in recent_entropies]}")
                    print(f"    Updated: low={self.entropy_low:.4f}, high={self.entropy_high:.4f}")
        
        if len(warmup_entropies) > 0:
            entropy_mean = np.mean(warmup_entropies)
            entropy_std = np.std(warmup_entropies)
            
            # Set initial entropy_high to max(warmup_mean + std, 0.15)
            self.entropy_high = max(entropy_mean + entropy_std, 0.15)
            self.entropy_low = max(0.05, entropy_mean - 3.0 * entropy_std)
            self.delta_stable = max(0.05, entropy_std)
            
            print(f"Warm-up completed:")
            print(f"  Entropy mean: {entropy_mean:.4f}")
            print(f"  Entropy std: {entropy_std:.4f}")
            print(f"  Final thresholds: low={self.entropy_low:.4f}, high={self.entropy_high:.4f}, stable={self.delta_stable:.4f}")
        
        self.warmup_completed = True

    def _aggregate_step_metrics(self) -> Dict[str, float]:
        """
        Aggregate step-level metrics for fast path uncertainty computation
        
        Returns:
            Dict containing H_t, ΔH_t, S_t, V_t
        """
        if len(self.step_buffer) == 0:
            return {'H_t': 0.0, 'ΔH_t': 0.0, 'S_t': 0.0, 'V_t': 0.0}
        
        # H_t: Exponential moving average of recent 5 steps
        recent_entropies = [t['entropy'] for t in list(self.step_buffer)[-5:]]
        H_t = np.mean(recent_entropies) if recent_entropies else 0.0
        
        # ΔH_t: Linear regression slope from step_entropy_window (20 steps)
        if len(self.step_entropy_window) >= 5:
            x = np.arange(len(self.step_entropy_window))
            y = np.array(list(self.step_entropy_window))
            if np.std(x) > 1e-6:
                slope = np.polyfit(x, y, 1)[0]
                ΔH_t = slope * len(self.step_entropy_window)
            else:
                ΔH_t = 0.0
        else:
            ΔH_t = 0.0
        
        # V_t: Novelty ratio from step_novelty_set
        V_t = len(self.step_novelty_set) / self.step_novelty_set_size
        
        # S_t: Mean gradient norm from step_buffer
        smoothness_values = [t.get('smoothness', 0.0) for t in self.step_buffer]
        S_t = np.mean(smoothness_values) if smoothness_values else 0.0
        
        return {'H_t': H_t, 'ΔH_t': ΔH_t, 'S_t': S_t, 'V_t': V_t}

    def _compute_contrastive_uncertainty_fast(self, meta_features: List[float]) -> float:
        """
        Compute contrastive uncertainty with cold-start fallback
        
        Cold-start mode: When cache entries < pos_capacity//2, use z-score based on warmup stats
        Normal mode: Use contrastive distance with adaptive tau
        
        Args:
            meta_features: 4D meta-feature vector [H_t, ΔH_t, S_t, V_t]
            
        Returns:
            U_t^cont: Contrastive uncertainty ∈ (0, 1)
        """
        H_t = meta_features[0]
        
        # Cold-start mode: use z-score
        if len(self.contrastive_cache.pos_cache) < self.pos_cache_capacity // 2:
            if not self.warmup_completed:
                return 0.5
            
            z_score = (H_t - (self.entropy_low + self.entropy_high) / 2) / ((self.entropy_high - self.entropy_low) / 2 + 1e-6)
            U_cont = 1.0 / (1.0 + np.exp(-z_score))
            return U_cont
        
        # Normal mode: use contrastive distance with adaptive tau
        if len(self.step_entropy_window) >= 5:
            adaptive_tau = max(0.1, np.std(list(self.step_entropy_window)))
        else:
            adaptive_tau = self.tau
        
        d_pos, d_neg = self.contrastive_cache.compute_contrastive_distance(meta_features)
        
        if d_pos == float('inf') and d_neg == float('inf'):
            return 0.5
        
        if d_pos == float('inf'):
            return 0.9
        
        if d_neg == float('inf'):
            return 0.1
        
        U_cont = 1.0 / (1.0 + np.exp(-(d_pos - d_neg) / adaptive_tau))
        return U_cont

    def _update_lambda_step(self, U_cont: float, H_t: float) -> float:
        """
        Update lambda via step-level EMA dynamics (Fast Path)
        
        λ_new = (1 - momentum) * λ_old + momentum * [λ_min + (λ_max - λ_min) * U_cont]
        
        No entropy suppression term in step-level updates to avoid high-frequency oscillation
        Removed max_drift logic to allow proper lambda adaptation
        
        Args:
            U_cont: Step-level contrastive uncertainty
            H_t: Current entropy
            
        Returns:
            Updated lambda value
        """
        lambda_target = self.lambda_min + (self.lambda_max - self.lambda_min) * U_cont
        lambda_new = (1.0 - self.step_lambda_momentum) * self.step_lambda_ema + self.step_lambda_momentum * lambda_target
        
        # Clip to hard bounds only
        lambda_new = np.clip(lambda_new, self.lambda_min, self.lambda_max)
        
        self.step_lambda_ema = lambda_new
        return lambda_new

    def _update_cache_episode(self, meta_features: List[float], H_t: float, y_t: int):
        """
        Update contrastive cache with episode-level sparse writing
        
        Sparse writing: Only update cache once per episode
        Entropy-priority replacement:
        - y_t=+1: Replace highest entropy entry in pos cache
        - y_t=-1: Replace lowest entropy entry in neg cache
        
        Args:
            meta_features: 4D meta-feature vector [H_t, ΔH_t, S_t, V_t]
            H_t: Current episode entropy
            y_t: Episode quality label
        """
        if y_t == 1:
            if len(self.contrastive_cache.pos_cache) >= self.pos_cache_capacity:
                # Replace highest entropy entry
                max_entropy_idx = np.argmax([entry[1] for entry in self.contrastive_cache.pos_cache])
                del self.contrastive_cache.pos_cache[max_entropy_idx]
            self.contrastive_cache.pos_cache.append((meta_features, H_t))
        elif y_t == -1:
            if len(self.contrastive_cache.neg_cache) >= self.neg_cache_capacity:
                # Replace lowest entropy entry
                min_entropy_idx = np.argmin([entry[1] for entry in self.contrastive_cache.neg_cache])
                del self.contrastive_cache.neg_cache[min_entropy_idx]
            self.contrastive_cache.neg_cache.append((meta_features, H_t))

    def _compute_episode_quality_label(self, H_t: float, ΔH_t: float, V_t: float) -> int:
        """
        Compute episode quality label y_t (internal helper for cache update)
        
        y_t = +1 if H_t ∈ [H_low, H_high] and |ΔH_t| < δ_stable and V_t > 0.8
        y_t = -1 if H_t < 0.5*H_low or H_t > 1.2*H_high or V_t < 0.5
        y_t = 0 otherwise
        
        Args:
            H_t: Current episode entropy
            ΔH_t: Entropy velocity
            V_t: State novelty ratio
            
        Returns:
            y_t: Episode quality label ∈ {+1, -1, 0}
        """
        if (self.entropy_low <= H_t <= self.entropy_high and 
            abs(ΔH_t) < self.delta_stable and V_t > 0.8):
            return 1
        elif H_t < 0.5 * self.entropy_low or H_t > 1.2 * self.entropy_high or V_t < 0.5:
            return -1
        else:
            return 0

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

    def _compute_episode_entropy(self) -> float:
        """
        Compute episode-averaged policy entropy
        
        H_t = E_{s~τ_t}[H(π_θt(·|s))]
        
        Uses the experience buffer to compute average entropy over the episode
        """
        if len(self.experience_buffer) == 0:
            return 0.0
        
        # Sample a batch of observations from the experience buffer
        obs_batch = self._sample_batch(batch_size=min(32, len(self.experience_buffer)))
        if obs_batch is None:
            return 0.0
        
        # Compute entropy for the batch
        entropy = self._compute_policy_entropy(obs_batch)
        return entropy.detach().cpu().item()



    def _outer_loop_update_episode(self, episode_entropy: float, episode_transitions: List[Dict]) -> Dict[str, Any]:
        """
        Outer loop (Slow Path): Episode-level cache update and lambda synchronization
        
        1. Compute meta-features x_t = [H_t, ΔH_t, S_t, V_t]^⊤
        2. Compute episode quality label y_t
        3. Update contrastive cache with sparse writing
        4. Synchronize lambda from step-level EMA
        5. Update history
        
        Args:
            episode_entropy: Current episode entropy H_t
            episode_transitions: List of transitions from the episode
            
        Returns:
            Dict containing all metrics for this update
        """
        # Compute meta-features x_t = [H_t, ΔH_t, S_t, V_t]^⊤
        H_t = episode_entropy
        H_t_1 = self.entropy_history[-1] if len(self.entropy_history) > 0 else H_t
        ΔH_t = H_t - H_t_1
        
        # Compute policy smoothness S_t
        obs_batch = self._sample_batch(batch_size=min(32, len(episode_transitions)))
        if obs_batch is not None:
            S_t = self._compute_policy_smoothness(obs_batch).detach().cpu().item()
        else:
            S_t = 0.0
        
        # Compute state novelty V_t
        V_t = self._compute_state_novelty(episode_transitions)
        
        meta_features = [H_t, ΔH_t, S_t, V_t]
        
        # Compute episode quality label y_t
        y_t = self._compute_episode_quality_label(H_t, ΔH_t, V_t)
        
        # Update contrastive cache with sparse writing (once per episode)
        self._update_cache_episode(meta_features, H_t, y_t)
        
        # Synchronize lambda from step-level EMA
        self.lambda_t = self.step_lambda_ema
        
        # Episode-level lambda recalibration (compensation mechanism)
        # Use episode-level U_cont to recalibrate lambda for better adaptation
        episode_U_cont = self._compute_contrastive_uncertainty_fast(meta_features)
        self.lambda_t = (1.0 - 0.3) * self.lambda_t + 0.3 * (self.lambda_min + (self.lambda_max - self.lambda_min) * episode_U_cont)
        self.lambda_t = np.clip(self.lambda_t, self.lambda_min, self.lambda_max)
        self.step_lambda_ema = self.lambda_t  # Sync step_ema to prevent drift
        
        # Update history
        self.entropy_history.append(H_t)
        self.entropy_velocity_history.append(ΔH_t)
        self.smoothness_history.append(S_t)
        self.novelty_history.append(V_t)
        
        # Compute contrastive uncertainty
        U_cont = self._compute_contrastive_uncertainty_fast(meta_features)
        self.contrastive_uncertainty_history.append(U_cont)
        
        return {
            'entropy': H_t,
            'entropy_velocity': ΔH_t,
            'smoothness': S_t,
            'novelty': V_t,
            'quality_label': y_t,
            'lambda': self.lambda_t,
            'contrastive_uncertainty': U_cont
        }

    def _inner_loop_update(self, obs_batch: torch.Tensor) -> torch.Tensor:
        """
        Inner loop: Policy update with entropy-regularized KL control
        
        L_π(θ) = -H(π_θ) + λ_t · D_KL(π_θ || π_θ0)
        
        where:
        - H(π_θ) is the policy entropy (maximization)
        - D_KL(π_θ || π_θ0) is the KL divergence from anchor policy (constraint)
        - λ_t is the conservatism coefficient (evolved via outer loop)
        
        Returns:
            loss: The computed loss tensor
        """
        entropy = self._compute_policy_entropy(obs_batch)
        kl_div = self._compute_kl_divergence(obs_batch)
        
        lambda_tensor = torch.tensor(self.lambda_t, device=self.device)
        loss = -entropy + lambda_tensor * kl_div
        
        return loss

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
        Run CCEA adaptation with Fast-Slow dual-scale architecture
        
        Fast Path (Step-level, every 10 steps):
        - Aggregate step metrics
        - Compute contrastive uncertainty (cold-start supported)
        - Update lambda via EMA smoothing
        - Inner loop policy update (every 30 steps)
        
        Slow Path (Episode-level, once per episode):
        - Outer loop cache update (sparse writing)
        - Synchronize lambda from step-level EMA
        - Update history records
        
        Args:
            num_episodes: Number of adaptation episodes
            
        Returns:
            adaptation_data: List of episode data
            summary: Summary statistics
        """
        adaptation_data = []
        lambda_history = []
        contrastive_uncertainty_history = []
        entropy_history = []
        entropy_velocity_history = []
        smoothness_history = []
        novelty_history = []
        quality_label_history = []
        
        # Run warm-up to initialize dynamic thresholds
        if not self.warmup_completed:
            self._warmup_thresholds()
        
        for episode in range(num_episodes):
            # Step 1: Run episode with Fast-Slow dual-scale updates
            episode_data = self._run_single_episode_with_updates()
            adaptation_data.append(episode_data)
            
            # Step 2: Compute episode entropy
            episode_entropy = self._compute_episode_entropy()
            
            # Step 3: Outer loop (Slow Path) - Episode-level cache update and lambda synchronization
            outer_metrics = self._outer_loop_update_episode(episode_entropy, episode_data['transitions'])
            
            # Update episode data with metrics
            episode_data.update({
                'lambda': outer_metrics['lambda'],
                'entropy': outer_metrics['entropy'],
                'entropy_velocity': outer_metrics['entropy_velocity'],
                'smoothness': outer_metrics['smoothness'],
                'novelty': outer_metrics['novelty'],
                'quality_label': outer_metrics['quality_label'],
                'contrastive_uncertainty': outer_metrics['contrastive_uncertainty']
            })
            
            # Track history
            lambda_history.append(outer_metrics['lambda'])
            contrastive_uncertainty_history.append(outer_metrics['contrastive_uncertainty'])
            entropy_history.append(outer_metrics['entropy'])
            entropy_velocity_history.append(outer_metrics['entropy_velocity'])
            smoothness_history.append(outer_metrics['smoothness'])
            novelty_history.append(outer_metrics['novelty'])
            quality_label_history.append(outer_metrics['quality_label'])
            
            self.adaptation_step += 1
            
            # Print progress
            print(f"Episode {episode + 1}/{num_episodes}: "
                  f"Reward: {episode_data['episode_reward']:.2f}, "
                  f"Entropy: {outer_metrics['entropy']:.3f}, "
                  f"ΔEntropy: {outer_metrics['entropy_velocity']:.3f}, "
                  f"Novelty: {outer_metrics['novelty']:.3f}, "
                  f"Quality: {outer_metrics['quality_label']:+d}, "
                  f"U_cont: {outer_metrics['contrastive_uncertainty']:.3f}, "
                  f"Lambda: {self.lambda_t:.3f}")
        
        summary = {
            'final_lambda': self.lambda_t,
            'lambda_history': lambda_history,
            'contrastive_uncertainty_history': contrastive_uncertainty_history,
            'entropy_history': entropy_history,
            'entropy_velocity_history': entropy_velocity_history,
            'smoothness_history': smoothness_history,
            'novelty_history': novelty_history,
            'quality_label_history': quality_label_history,
            'mean_reward': np.mean([d['episode_reward'] for d in adaptation_data]),
            'std_reward': np.std([d['episode_reward'] for d in adaptation_data]),
            'final_contrastive_uncertainty': contrastive_uncertainty_history[-1] if contrastive_uncertainty_history else 0.0,
            'final_entropy': entropy_history[-1] if entropy_history else 0.0,
            'final_entropy_velocity': entropy_velocity_history[-1] if entropy_velocity_history else 0.0
        }
        
        return adaptation_data, summary

    def _run_single_episode_with_updates(self) -> Dict[str, Any]:
        """
        Run a single episode with episode-level updates only
        
        Step-level Fast Path has been removed to avoid dynamics conflict.
        Lambda is now updated only at episode level via _outer_loop_update_episode.
        
        Returns:
            episode_data: Dictionary containing episode information
        """
        reset_result = self.env.reset()
        if isinstance(reset_result, tuple):
            obs, _ = reset_result
        else:
            obs = reset_result
        
        episode_reward = 0
        episode_length = 0
        episode_transitions = []
        self.current_episode_transitions = []
        
        done = False
        step_count = 0
        
        while not done:
            with torch.no_grad():
                action = self.policy.select_action(obs)
            
            step_result = self.env.step(action)
            if len(step_result) == 5:
                next_obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                next_obs, reward, done, info = step_result
            
            episode_reward += reward
            episode_length += 1
            step_count += 1
            
            # Store transition
            transition = {
                'obs': obs,
                'action': action,
                'reward': reward,
                'next_obs': next_obs,
                'done': done
            }
            episode_transitions.append(transition)
            self.experience_buffer.append(transition)
            self.current_episode_transitions.append(transition)
            
            # Compute step-level metrics
            obs_batch = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            entropy = self._compute_policy_entropy(obs_batch).detach().cpu().item()
            smoothness = self._compute_policy_smoothness(obs_batch).detach().cpu().item()
            
            # Update step buffers
            self.step_entropy_window.append(entropy)
            self.step_buffer.append({
                'entropy': entropy,
                'smoothness': smoothness,
                'obs': obs
            })
            
            # Update novelty set
            obs_key = tuple(np.round(obs, 2))
            self.step_novelty_set.add(obs_key)
            if len(self.step_novelty_set) > self.step_novelty_set_size:
                self.step_novelty_set.pop()
            
            # Fast Path removed to avoid dynamics conflict between step-level and episode-level lambda updates
            # Lambda is now updated only at episode level for stability
            
            obs = next_obs
            
            if episode_length >= 1000:
                break
        
        # Clear step buffers at end of episode
        self.step_buffer.clear()
        self.step_entropy_window.clear()
        self.step_novelty_set.clear()
        
        return {
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            'transitions': episode_transitions
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
        """Save CCEA checkpoint"""
        checkpoint = {
            'policy_state_dict': self.policy.state_dict(),
            'lambda_t': self.lambda_t,
            'adaptation_step': self.adaptation_step,
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'entropy_history': self.entropy_history,
            'entropy_velocity_history': self.entropy_velocity_history,
            'smoothness_history': self.smoothness_history,
            'novelty_history': self.novelty_history,
            'contrastive_uncertainty_history': self.contrastive_uncertainty_history,
            'pos_cache': list(self.contrastive_cache.pos_cache),
            'neg_cache': list(self.contrastive_cache.neg_cache),
            'config': self.config
        }
        torch.save(checkpoint, save_path)

    def load_checkpoint(self, checkpoint_path: str):
        """Load CCEA checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.lambda_t = checkpoint['lambda_t']
        self.adaptation_step = checkpoint['adaptation_step']
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.entropy_history = checkpoint.get('entropy_history', [])
        self.entropy_velocity_history = checkpoint.get('entropy_velocity_history', [])
        self.smoothness_history = checkpoint.get('smoothness_history', [])
        self.novelty_history = checkpoint.get('novelty_history', [])
        self.contrastive_uncertainty_history = checkpoint.get('contrastive_uncertainty_history', [])
        
        # Restore cache
        self.contrastive_cache.pos_cache.clear()
        self.contrastive_cache.neg_cache.clear()
        for entry in checkpoint.get('pos_cache', []):
            self.contrastive_cache.pos_cache.append(entry)
        for entry in checkpoint.get('neg_cache', []):
            self.contrastive_cache.neg_cache.append(entry)
        
        self.config = checkpoint.get('config', {})