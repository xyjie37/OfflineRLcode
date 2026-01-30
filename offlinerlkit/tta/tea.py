import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from collections import deque

from offlinerlkit.policy.base_policy import BasePolicy


class TEAManager:
    """
    Test-time Energy Adaptation for Offline RL (TEA-RL)
    
    TEA-RL通过对比散度（Contrastive Divergence）优化能量函数，
    在测试时降低测试状态能量同时提升负样本能量，防止策略崩溃。
    
    核心组件：
    - 能量函数建模（连续/离散动作空间）
    - SGLD负样本采样（Langevin动力学）
    - 对比散度损失（Contrastive Divergence）
    - LayerNorm参数限制更新
    - KL散度正则化（防止过拟合）
    
    Reference: Energy-Based Models for Test-Time Adaptation in Reinforcement Learning
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

        # 超参数配置
        self.learning_rate = self.config.get('learning_rate', 1e-6)  # 连续任务建议1e-7，离散任务建议1e-9
        self.sgld_step_size = self.config.get('sgld_step_size', 0.1)  # SGLD步长α
        self.sgld_steps = self.config.get('sgld_steps', 10)  # SGLD迭代步数T
        self.num_neg_samples = self.config.get('num_neg_samples', 10)  # 负样本数N
        self.kl_weight = self.config.get('kl_weight', 1.0)  # KL正则权重λ（离散1.5，连续1.0）
        self.action_space_type = self.config.get('action_space', 'continuous')  # 'continuous' or 'discrete'
        self.cache_capacity = self.config.get('cache_capacity', 1000)
        self.update_freq = self.config.get('update_freq', 10)  # 每多少步更新一次
        
        # 状态缓存
        self.state_cache = deque(maxlen=self.cache_capacity)
        
        # 保存离线策略用于KL约束
        self._offline_policy_saved = False
        self.offline_policy_state = self._save_policy_state()
        self._offline_actor = self._create_frozen_actor()
        self._offline_critic = self._create_frozen_critic() if self.action_space_type == 'discrete' else None
        
        # 配置可训练参数（仅LayerNorm）
        self.adaptation_mode = self.config.get('adaptation_mode', 'layernorm')
        if self.adaptation_mode == 'layernorm':
            self.trainable_params = self._get_layernorm_params()
            if len(self.trainable_params) == 0:
                print("Warning: No LayerNorm params found, switching to last_n_layers")
                self.adaptation_mode = 'last_n_layers'
        
        if self.adaptation_mode == 'last_n_layers':
            n_layers = self.config.get('last_n_layers', 2)
            self.trainable_params = self._get_last_n_layers(n_layers)
        
        assert len(self.trainable_params) > 0, "No trainable params found"
        
        self.param_names = [name for name, _ in self.trainable_params]
        params_only = [param for _, param in self.trainable_params]
        self.optimizer = torch.optim.Adam(params_only, lr=self.learning_rate)
        
        self.adaptation_step = 0

    def _get_policy_device(self) -> torch.device:
        """获取策略所在的设备"""
        if hasattr(self.policy, 'actor') and hasattr(self.policy.actor, 'device'):
            return self.policy.actor.device
        elif hasattr(self.policy, 'critic1') and hasattr(self.policy.critic1, 'device'):
            return self.policy.critic1.device
        elif len(list(self.policy.parameters())) > 0:
            return next(self.policy.parameters()).device
        else:
            return torch.device('cpu')

    def _save_policy_state(self) -> Dict[str, torch.Tensor]:
        """保存离线策略状态用于KL计算"""
        state = {}
        for name, param in self.policy.named_parameters():
            state[name] = param.data.clone().detach()
        self._offline_policy_saved = True
        return state

    def _create_frozen_actor(self):
        """创建冻结的离线策略副本用于KL计算"""
        if not hasattr(self.policy, 'actor'):
            return None
        
        offline_actor = copy.deepcopy(self.policy.actor)
        offline_actor = offline_actor.to(self.device)
        offline_actor.eval()
        
        for param in offline_actor.parameters():
            param.requires_grad = False
        return offline_actor

    def _create_frozen_critic(self):
        """创建冻结的Q函数副本（离散动作空间需要）"""
        if not hasattr(self.policy, 'critic') and not hasattr(self.policy, 'critic1'):
            return None
        
        critic = getattr(self.policy, 'critic', getattr(self.policy, 'critic1', None))
        if critic is None:
            return None
            
        offline_critic = copy.deepcopy(critic)
        offline_critic = offline_critic.to(self.device)
        offline_critic.eval()
        
        for param in offline_critic.parameters():
            param.requires_grad = False
        return offline_critic

    def _get_layernorm_params(self) -> List[Tuple[str, nn.Parameter]]:
        """获取LayerNorm层参数进行微调"""
        trainable_params = []
        
        if hasattr(self.policy, 'actor'):
            for name, module in self.policy.actor.named_modules():
                if isinstance(module, nn.LayerNorm):
                    for param_name, param in module.named_parameters():
                        full_name = f"actor.{name}.{param_name}" if name else f"actor.{param_name}"
                        if param.requires_grad:
                            trainable_params.append((full_name, param))
        return trainable_params

    def _get_last_n_layers(self, n: int = 2) -> List[Tuple[str, nn.Parameter]]:
        """获取最后N层的参数作为备选方案"""
        trainable_params = []
        
        if hasattr(self.policy, 'actor'):
            if hasattr(self.policy.actor, 'backbone'):
                backbone_params = list(self.policy.actor.backbone.named_parameters())
                selected = backbone_params[-n:] if len(backbone_params) >= n else backbone_params
                for name, param in selected:
                    if param.requires_grad:
                        trainable_params.append((f"backbone.{name}", param))
            
            # 添加输出层
            for layer_name in ['last', 'dist_net', 'mean_layer', 'log_std_layer']:
                if hasattr(self.policy.actor, layer_name):
                    layer = getattr(self.policy.actor, layer_name)
                    for name, param in layer.named_parameters():
                        if param.requires_grad:
                            trainable_params.append((f"{layer_name}.{name}", param))
        
        return trainable_params

    def _compute_energy_continuous(self, obs: torch.Tensor, action: Optional[torch.Tensor] = None
                                   ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        连续动作空间的能量计算
        E(s,a) = -log π(a|s) = 0.5*log(2πeσ²) + ||a-μ||²/(2σ²)
        
        返回:
            energy: 状态-动作联合能量
            base_energy: 仅基于状态的能量（对数配分函数的估计）
        """
        if not hasattr(self.policy, 'actor'):
            raise ValueError("Policy must have actor for continuous action space")
        
        self.policy.train()
        dist = self.policy.actor(obs)
        
        # 提取高斯参数
        if hasattr(dist, 'mean'):
            mu = dist.mean
            sigma = dist.stddev if hasattr(dist, 'stddev') else torch.ones_like(mu)
        else:
            # 处理直接输出mu/log_std的情况
            mu = dist[0] if isinstance(dist, tuple) else dist
            sigma = dist[1] if isinstance(dist, tuple) else torch.ones_like(mu)
        
        if action is None:
            # 如果没有提供动作，从策略采样计算期望能量（或对数配分函数）
            action = mu  # 使用均值作为代表
            
        # 计算负对数似然（能量）
        # E(s,a) = -log π(a|s)
        log_prob = -((action - mu) ** 2) / (2 * sigma ** 2 + 1e-8) - 0.5 * torch.log(2 * np.pi * sigma ** 2 + 1e-8)
        energy = -log_prob.sum(dim=-1, keepdim=True)  # [batch_size, 1]
        
        # 基础能量（仅状态相关部分，用于对比）
        base_energy = 0.5 * torch.log(2 * np.pi * (sigma ** 2) + 1e-8).sum(dim=-1, keepdim=True) + 0.5
        
        return energy, base_energy

    def _compute_energy_discrete(self, obs: torch.Tensor) -> torch.Tensor:
        """
        离散动作空间的能量计算
        E(s) = -log Σ_a exp(Q(s,a))
        
        这对应于softmax的log-sum-exp，即自由能
        """
        if not hasattr(self.policy, 'critic') and not hasattr(self.policy, 'critic1'):
            raise ValueError("Policy must have critic for discrete action space")
        
        critic = getattr(self.policy, 'critic', getattr(self.policy, 'critic1', None))
        self.policy.train()
        
        with torch.enable_grad():
            q_values = critic(obs)
            # LogSumExp trick for numerical stability
            max_q = q_values.max(dim=-1, keepdim=True)[0]
            exp_q = torch.exp(q_values - max_q)
            sum_exp_q = exp_q.sum(dim=-1, keepdim=True)
            free_energy = -torch.log(sum_exp_q + 1e-8) - max_q
            
        return free_energy  # [batch_size, 1]

    def _compute_energy_gradient(self, states: torch.Tensor, 
                                  actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算能量对输入的梯度 ∇_s E(s) 或 ∇_(s,a) E(s,a)
        用于SGLD采样
        """
        states = states.clone().requires_grad_(True)
        if actions is not None:
            actions = actions.clone().requires_grad_(True)
        
        if self.action_space_type == 'continuous':
            energy, _ = self._compute_energy_continuous(states, actions)
            # 对状态求梯度（假设能量依赖于状态通过策略网络）
            grad = torch.autograd.grad(energy.sum(), states, create_graph=False)[0]
        else:
            energy = self._compute_energy_discrete(states)
            grad = torch.autograd.grad(energy.sum(), states, create_graph=False)[0]
        
        return grad.detach()

    def _sgld_sample(self, initial_states: Optional[torch.Tensor] = None,
                     initial_actions: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        SGLD采样生成负样本
        s_{t+1} = s_t - (α/2) * ∇E(s_t) + √α * ε
        
        Args:
            initial_states: 初始状态，None时随机初始化
            initial_actions: 初始动作（连续空间）
        Returns:
            neg_states: 采样后的负样本状态
            neg_actions: 采样后的负样本动作（连续空间）
        """
        batch_size = initial_states.shape[0] if initial_states is not None else self.num_neg_samples
        
        # 初始化：低维用Uniform，高维从历史采样+扰动
        state_dim = self.env.observation_space.shape[0]
        neg_actions = None  # 初始化，避免未定义错误
        
        if initial_states is None:
            if state_dim <= 20:
                # 低维：Uniform采样，但需要检查边界是否有限
                low = self.env.observation_space.low
                high = self.env.observation_space.high
                
                # 检查并修复无限边界
                if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)):
                    # 如果有无限边界，使用历史缓存或标准正态初始化
                    if len(self.state_cache) > 0:
                        idx = np.random.choice(len(self.state_cache), min(batch_size, len(self.state_cache)))
                        neg_states = torch.FloatTensor(np.array([self.state_cache[i] for i in idx])).to(self.device)
                        noise = torch.randn_like(neg_states) * 0.1
                        neg_states = neg_states + noise
                    else:
                        neg_states = torch.randn(batch_size, state_dim).to(self.device) * 0.5
                else:
                    # 确保low和high是标量或数组，且low < high
                    low = np.clip(low, -10, 10)
                    high = np.clip(high, -10, 10)
                    # 确保low < high
                    low, high = np.minimum(low, high - 0.1), np.maximum(high, low + 0.1)
                    neg_states = np.random.uniform(low, high, (self.num_neg_samples, state_dim))
                    neg_states = torch.FloatTensor(neg_states).to(self.device)
            else:
                # 高维：从历史缓存采样+扰动
                if len(self.state_cache) > 0:
                    idx = np.random.choice(len(self.state_cache), min(batch_size, len(self.state_cache)))
                    neg_states = torch.FloatTensor(np.array([self.state_cache[i] for i in idx])).to(self.device)
                    noise = torch.randn_like(neg_states) * 0.1
                    neg_states = neg_states + noise
                else:
                    neg_states = torch.randn(batch_size, state_dim).to(self.device) * 0.5
        else:
            neg_states = initial_states.clone()
        
        # SGLD迭代
        for _ in range(self.sgld_steps):
            noise = torch.randn_like(neg_states) * np.sqrt(self.sgld_step_size)
            
            if self.action_space_type == 'continuous':
                # 连续空间：同时采样状态和动作
                if initial_actions is None:
                    # 从当前策略采样动作
                    with torch.no_grad():
                        dist = self.policy.actor(neg_states)
                        if hasattr(dist, 'sample'):
                            neg_actions = dist.sample()
                        else:
                            neg_actions = dist[0] if isinstance(dist, tuple) else dist
                else:
                    neg_actions = initial_actions.clone()
                
                grad_s = self._compute_energy_gradient(neg_states, neg_actions)
                neg_states = neg_states - (self.sgld_step_size / 2) * grad_s + noise
            else:
                # 离散空间：仅采样状态
                grad = self._compute_energy_gradient(neg_states)
                neg_states = neg_states - (self.sgld_step_size / 2) * grad + noise
        
        if self.action_space_type == 'continuous':
            return neg_states.detach(), neg_actions.detach() if neg_actions is not None else None
        else:
            return neg_states.detach(), None

    def _compute_kl_divergence(self, obs: torch.Tensor) -> torch.Tensor:
        """
        计算KL(π_off || π_current)用于防止过拟合
        """
        if self._offline_actor is None:
            return torch.tensor(0.0, device=self.device)
        
        with torch.no_grad():
            # 离线分布
            old_dist = self._offline_actor(obs)
            if hasattr(old_dist, 'mean'):
                old_mu, old_sigma = old_dist.mean, old_dist.stddev
            else:
                old_mu = old_dist[0] if isinstance(old_dist, tuple) else old_dist
                old_sigma = torch.ones_like(old_mu)
        
        # 当前分布
        current_dist = self.policy.actor(obs)
        if hasattr(current_dist, 'mean'):
            mu, sigma = current_dist.mean, current_dist.stddev
        else:
            mu = current_dist[0] if isinstance(current_dist, tuple) else current_dist
            sigma = torch.ones_like(mu)
        
        # KL(N_off || N_current) = 0.5 * (σ²/σ_off² + (μ_off-μ)²/σ_off² - 1 + log(σ_off²/σ²))
        var_old = old_sigma ** 2 + 1e-8
        var_current = sigma ** 2 + 1e-8
        
        kl = 0.5 * (
            var_current / var_old +
            (old_mu - mu) ** 2 / var_old -
            1 +
            torch.log(var_old / var_current)
        )
        return kl.mean()

    def _update(self, obs_batch: torch.Tensor) -> Dict[str, float]:
        """
        执行一次TEA-RL更新
        Loss = [E(s_test) - E(s_neg)] + λ * KL(π_off || π)
        """
        self.policy.train()
        
        # 正样本：当前测试状态
        if self.action_space_type == 'continuous':
            pos_energy, _ = self._compute_energy_continuous(obs_batch)
        else:
            pos_energy = self._compute_energy_discrete(obs_batch)
        
        # 生成负样本（SGLD）
        neg_states, neg_actions = self._sgld_sample(
            initial_states=None,  # 重新初始化
            initial_actions=None
        )
        
        # 负样本能量
        if self.action_space_type == 'continuous' and neg_actions is not None:
            neg_energy, _ = self._compute_energy_continuous(neg_states, neg_actions)
        else:
            neg_energy = self._compute_energy_discrete(neg_states)
        
        # 对比散度损失：降低正样本能量，提升负样本能量
        cd_loss = pos_energy.mean() - neg_energy.mean()
        
        # KL正则
        kl_loss = self._compute_kl_divergence(obs_batch)
        
        # 总损失
        total_loss = cd_loss + self.kl_weight * kl_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # 检查梯度
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 
                      for p in [param for _, param in self.trainable_params])
        if not has_grad:
            print(f"Warning: Zero gradient detected! CD={cd_loss.item():.4f}, KL={kl_loss.item():.4f}")
        
        self.optimizer.step()
        
        return {
            'cd_loss': cd_loss.item(),
            'kl_loss': kl_loss.item(),
            'pos_energy': pos_energy.mean().item(),
            'neg_energy': neg_energy.mean().item(),
            'total_loss': total_loss.item()
        }

    def _run_single_episode(self) -> Dict[str, Any]:
        """运行单个episode并实时适应"""
        reset_result = self.env.reset()
        if isinstance(reset_result, tuple):
            obs, _ = reset_result
        else:
            obs = reset_result

        episode_reward = 0
        episode_length = 0
        episode_transitions = []
        episode_stats = []

        done = False
        while not done:
            # 缓存状态用于负样本初始化（高维情况）
            self.state_cache.append(obs.copy())
            
            # 选择动作
            with torch.no_grad():
                action = self.policy.select_action(obs)
            
            # 执行动作
            step_result = self.env.step(action)
            if len(step_result) == 5:
                next_obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                next_obs, reward, done, info = step_result

            episode_transitions.append({
                'obs': obs.copy(),
                'action': action.copy() if isinstance(action, np.ndarray) else action,
                'reward': reward,
                'next_obs': next_obs.copy(),
                'done': done
            })

            # 测试时适应（定期更新）
            if len(self.state_cache) >= self.num_neg_samples and self.adaptation_step % self.update_freq == 0:
                obs_batch = torch.FloatTensor(
                    np.array(list(self.state_cache)[-self.num_neg_samples:])
                ).to(self.device)
                
                stats = self._update(obs_batch)
                episode_stats.append(stats)

            obs = next_obs
            episode_reward += reward
            episode_length += 1
            self.adaptation_step += 1

            if episode_length >= 1000:
                break

        # 计算平均统计量
        avg_stats = {
            'cd_loss': np.mean([s['cd_loss'] for s in episode_stats]) if episode_stats else 0,
            'kl_loss': np.mean([s['kl_loss'] for s in episode_stats]) if episode_stats else 0,
            'pos_energy': np.mean([s['pos_energy'] for s in episode_stats]) if episode_stats else 0,
            'neg_energy': np.mean([s['neg_energy'] for s in episode_stats]) if episode_stats else 0,
        }

        return {
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            'transitions': episode_transitions,
            'adaptation_step': self.adaptation_step,
            **avg_stats
        }

    def run_adaptation(self, num_episodes: int = 10) -> Tuple[List[Dict], Dict[str, Any]]:
        """
        运行完整的TEA-RL适应过程
        
        Args:
            num_episodes: 适应回合数
            
        Returns:
            adaptation_data: 每个episode的数据列表
            summary: 汇总统计
        """
        adaptation_data = []
        all_cd_losses = []
        all_kl_losses = []

        for episode in range(num_episodes):
            episode_data = self._run_single_episode()
            adaptation_data.append(episode_data)
            
            if episode_data.get('cd_loss', 0) != 0:
                all_cd_losses.append(episode_data['cd_loss'])
                all_kl_losses.append(episode_data['kl_loss'])

            print(f"Episode {episode + 1}/{num_episodes} (TEA-RL): "
                  f"Reward: {episode_data['episode_reward']:.2f}, "
                  f"Length: {episode_data['episode_length']}, "
                  f"CD Loss: {episode_data.get('cd_loss', 0):.4f}, "
                  f"KL: {episode_data.get('kl_loss', 0):.4f}, "
                  f"E_pos: {episode_data.get('pos_energy', 0):.4f}, "
                  f"E_neg: {episode_data.get('neg_energy', 0):.4f}")

        summary = {
            'mean_reward': np.mean([d['episode_reward'] for d in adaptation_data]),
            'std_reward': np.std([d['episode_reward'] for d in adaptation_data]),
            'mean_length': np.mean([d['episode_length'] for d in adaptation_data]),
            'mean_cd_loss': np.mean(all_cd_losses) if all_cd_losses else 0,
            'mean_kl_loss': np.mean(all_kl_losses) if all_kl_losses else 0,
            'cache_size': len(self.state_cache),
            'adaptation_steps': self.adaptation_step
        }

        return adaptation_data, summary

    def evaluate_performance(self, num_episodes: int = 5) -> Dict[str, float]:
        """评估当前策略性能（冻结状态）"""
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

        self.policy.train()  # 恢复训练模式
        
        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'mean_length': np.mean(lengths)
        }

    def get_statistics(self) -> Dict[str, Any]:
        """获取TEA-RL统计信息"""
        return {
            'adaptation_steps': self.adaptation_step,
            'cache_size': len(self.state_cache),
            'trainable_params': len(self.trainable_params),
            'learning_rate': self.learning_rate,
            'sgld_step_size': self.sgld_step_size,
            'sgld_steps': self.sgld_steps,
            'num_neg_samples': self.num_neg_samples,
            'kl_weight': self.kl_weight,
            'adaptation_mode': self.adaptation_mode,
            'action_space_type': self.action_space_type,
            'param_names': self.param_names[:5]  # 仅显示前5个参数名
        }

    def save_checkpoint(self, save_path: str):
        """保存TEA-RL检查点"""
        checkpoint = {
            'policy_state_dict': self.policy.state_dict(),
            'offline_policy_state': self.offline_policy_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'adaptation_step': self.adaptation_step,
            'config': self.config,
            'statistics': self.get_statistics()
        }
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved to {save_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """加载TEA-RL检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.offline_policy_state = checkpoint['offline_policy_state']
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.adaptation_step = checkpoint['adaptation_step']
        self.config = checkpoint.get('config', {})
        
        print(f"Checkpoint loaded from {checkpoint_path}")