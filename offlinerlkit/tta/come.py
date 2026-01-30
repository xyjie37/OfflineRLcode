import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from collections import deque

from offlinerlkit.policy.base_policy import BasePolicy


class COMEManager:
    """
    COME: Conservative Offline Model adaptation for test-time Evaluation
    
    COME算法通过Dirichlet证据理论量化不确定性，在测试时自适应阶段：
    1. 将策略输出转换为Dirichlet证据和主观意见（belief & uncertainty）
    2. 实施保守性范数约束防止策略漂移
    3. 筛选高置信度样本来计算COME损失（置信度熵 + 不确定性惩罚）
    4. 通过KL散度正则化保持与预训练策略的一致性
    5. 仅更新LayerNorm参数实现高效适应
    
    关键特性：
    - 基于Dirichlet证据的不确定性量化：u = K / S
    - 主观意见建模：b_k = (alpha_k - 1) / S
    - 保守性约束：||f(s)||_p 控制logits幅度
    - COME损失：-sum(b_k log b_k) - lambda_u * u log u
    - LayerNorm-only参数更新
    
    超参数配置：
    - 离散任务：lr=1e-9, lambda_KL=1.5
    - 连续任务：lr=1e-6, lambda_KL=1.0
    - 通用：p=2, tau=1.0, lambda_u=10, delta=0.1
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

        # 超参数配置（根据算法建议设置默认值）
        self.learning_rate = self.config.get('learning_rate', 1e-6)  # 连续任务默认1e-6，离散任务建议1e-9
        self.kl_weight = self.config.get('kl_weight', 1.0)  # 离散任务建议1.5，连续任务建议1.0
        self.lambda_u = self.config.get('lambda_u', 10.0)  # 不确定性项权重
        self.uncertainty_delta = self.config.get('uncertainty_delta', 0.1)  # 不确定性容忍阈值delta
        self.tau = self.config.get('tau', 1.0)  # logits幅度恢复系数
        self.p_norm = self.config.get('p_norm', 2)  # 范数约束参数p
        self.num_bins = self.config.get('num_bins', 10)  # 连续动作离散化bins数K
        self.cache_capacity = self.config.get('cache_capacity', 1000)
        self.update_freq = self.config.get('update_freq', 10)  # 更新频率
        
        # 动作空间类型检测
        self.is_discrete = hasattr(env.action_space, 'n')
        if self.is_discrete:
            self.num_actions = env.action_space.n
        else:
            self.num_actions = self.num_bins  # 连续任务使用bins数作为K
        
        # 状态缓存
        self.state_cache = deque(maxlen=self.cache_capacity)
        self.uncertainty_cache = deque(maxlen=self.cache_capacity)
        
        # 保存离线策略状态和创建冻结副本（用于KL散度计算）
        self.offline_policy_state = self._save_policy_state()
        self._offline_actor = self._create_frozen_actor()
        
        # 计算预训练策略的平均不确定性u_0
        self.u_0 = self._compute_pretrained_uncertainty()
        
        # 配置可训练参数（仅LayerNorm）
        self.adaptation_mode = self.config.get('adaptation_mode', 'layernorm')
        self.trainable_params = self._get_layernorm_params()
        self.param_names = [name for name, _ in self.trainable_params]
        
        if len(self.trainable_params) == 0:
            print("警告: 未找到LayerNorm参数，自动切换到last_n_layers模式")
            self.adaptation_mode = 'last_n_layers'
            self.trainable_params = self._get_last_n_layers(
                self.config.get('last_n_layers', 2)
            )
            self.param_names = [name for name, _ in self.trainable_params]
        
        assert len(self.trainable_params) > 0, "无法找到可训练参数，请检查网络结构"
        
        params_only = [param for _, param in self.trainable_params]
        self.params_only = params_only
        
        # 创建优化器
        self.optimizer = torch.optim.Adam(
            params_only,
            lr=self.learning_rate
        )

        self.adaptation_step = 0
        self.episode_count = 0

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
        """保存当前策略状态用于KL散度计算"""
        state = {}
        for name, param in self.policy.named_parameters():
            state[name] = param.data.clone().detach()
        return state

    def _create_frozen_actor(self):
        """创建冻结的离线策略actor副本用于KL计算"""
        if not hasattr(self.policy, 'actor'):
            return None
        
        offline_actor = copy.deepcopy(self.policy.actor)
        offline_actor = offline_actor.to(self.device)
        offline_actor.eval()
        
        for param in offline_actor.parameters():
            param.requires_grad = False
        
        return offline_actor

    def _compute_pretrained_uncertainty(self) -> float:
        """
        计算预训练策略的平均不确定性u_0
        
        在初始化阶段采样若干状态计算平均不确定性作为基准
        """
        if not hasattr(self.policy, 'actor'):
            return 0.5  # 默认值
        
        uncertainties = []
        num_samples = 100  # 采样状态数
        
        try:
            # 从环境中采样一些初始状态（这里使用随机采样作为近似）
            for _ in range(num_samples):
                reset_result = self.env.reset()
                if isinstance(reset_result, tuple):
                    obs, _ = reset_result
                else:
                    obs = reset_result
                
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    if self.is_discrete:
                        # 离散动作：获取logits
                        logits = self.policy.actor(obs_tensor)
                        # 应用保守约束（虽然这时是eval模式，但保持计算一致性）
                        logits_constrained = self._apply_conservative_constraint(logits)
                        # 转换为Dirichlet证据
                        alpha = torch.exp(F.relu(logits_constrained))
                        S = alpha.sum(dim=-1)
                        K = self.num_actions
                        u = K / S
                        uncertainties.append(u.item())
                    else:
                        # 连续动作：离散化后计算
                        mu, sigma = self._get_continuous_distribution(obs_tensor)
                        probs = self._discretize_gaussian(mu, sigma)
                        # 将概率转换为Dirichlet证据（使用逆变换）
                        # alpha_k = probs * S, 假设S=K（均匀不确定性情况）
                        # 更合理的做法是将概率视为belief，反推alpha
                        eps = 1e-8
                        probs = torch.clamp(probs, eps, 1.0 - eps)
                        # 从belief反推alpha: b_k = (alpha_k - 1) / S, u = K / S
                        # 假设u=0.5（中等不确定性），则S = K / 0.5 = 2K
                        # alpha_k = b_k * S + 1 = probs * 2K + 1
                        S_assumed = 2 * self.num_bins
                        alpha = probs * S_assumed + 1
                        S = alpha.sum(dim=-1)
                        u = self.num_bins / S
                        uncertainties.append(u.item())
        except Exception as e:
            print(f"计算预训练不确定性时出错: {e}，使用默认值0.5")
            return 0.5
        
        return np.mean(uncertainties) if uncertainties else 0.5

    def _get_layernorm_params(self) -> List[Tuple[str, nn.Parameter]]:
        """获取LayerNorm参数用于训练"""
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

    def _get_last_n_layers(self, n: int) -> List[Tuple[str, nn.Parameter]]:
        """获取最后N层参数作为备选"""
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

    def _apply_conservative_constraint(self, logits: torch.Tensor) -> torch.Tensor:
        """
        实施保守性范数约束
        
        f'(s_t) = f(s_t) / ||f(s_t)||_p * ||f(s_t)||_p^{no_grad} * tau
        
        Args:
            logits: 原始策略输出 (batch_size, num_actions) 或 (batch_size, action_dim)
        
        Returns:
            constrained_logits: 约束后的输出
        """
        # 计算p范数
        norm = torch.norm(logits, p=self.p_norm, dim=-1, keepdim=True)
        
        # 梯度截断的范数（作为常数）
        norm_no_grad = norm.detach()
        
        # 避免除零
        norm = torch.clamp(norm, min=1e-8)
        
        # 应用约束：归一化 * 原始范数（常数）* tau
        constrained_logits = (logits / norm) * norm_no_grad * self.tau
        
        return constrained_logits

    def _compute_dirichlet_params(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算Dirichlet证据和主观意见
        
        Args:
            logits: 经过保守约束后的网络输出 (batch_size, K)
        
        Returns:
            alpha: Dirichlet证据参数 (batch_size, K)
            belief: 主观意见b_k (batch_size, K)
            uncertainty: 整体不确定性u (batch_size,)
        """
        # alpha = exp(ReLU(logits))
        alpha = torch.exp(F.relu(logits))
        
        # Dirichlet强度 S = sum(alpha_k)
        S = alpha.sum(dim=-1, keepdim=True)  # (batch_size, 1)
        
        # 主观意见 b_k = (alpha_k - 1) / S
        belief = (alpha - 1) / S
        
        # 不确定性 u = K / S
        K = logits.shape[-1]
        uncertainty = K / S.squeeze(-1)  # (batch_size,)
        
        return alpha, belief, uncertainty

    def _discretize_gaussian(self, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        将连续高斯分布映射到离散bins上的概率分布
        
        假设动作空间范围通常为[-1, 1]或从环境获取，划分为K个等宽bins
        
        Args:
            mu: 均值 (batch_size, action_dim)
            sigma: 标准差 (batch_size, action_dim)
        
        Returns:
            probs: 在bins上的概率分布 (batch_size, num_bins)
        """
        batch_size = mu.shape[0]
        action_dim = mu.shape[-1] if len(mu.shape) > 1 else 1
        
        # 假设动作范围[-1, 1]，可根据实际环境调整
        action_min, action_max = -1.0, 1.0
        bin_edges = torch.linspace(action_min, action_max, self.num_bins + 1, device=self.device)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        if action_dim == 1:
            # 单维动作：直接计算每个bin的概率（使用正态分布CDF差分）
            mu = mu.squeeze(-1) if len(mu.shape) > 1 else mu  # (batch_size,)
            sigma = sigma.squeeze(-1) if len(sigma.shape) > 1 else sigma
            
            # 扩展维度以便广播计算 (batch_size, num_bins)
            mu_expanded = mu.unsqueeze(-1)  # (batch_size, 1)
            sigma_expanded = sigma.unsqueeze(-1)  # (batch_size, 1)
            centers_expanded = bin_centers.unsqueeze(0)  # (1, num_bins)
            
            # 计算每个bin中心的概率密度（近似）
            # 使用softmax转换以避免数值问题，保持相对概率
            log_probs = -0.5 * ((centers_expanded - mu_expanded) / (sigma_expanded + 1e-8)) ** 2
            probs = F.softmax(log_probs, dim=-1)  # 归一化为概率分布
        else:
            # 多维动作：取第一个维度作为代表（简化处理）
            # 或者可以将多维联合分布展平，但这里简化为对每个维度分别处理然后平均
            mu = mu[:, 0]  # 取第一个维度
            sigma = sigma[:, 0]
            
            mu_expanded = mu.unsqueeze(-1)
            sigma_expanded = sigma.unsqueeze(-1)
            centers_expanded = bin_centers.unsqueeze(0)
            
            log_probs = -0.5 * ((centers_expanded - mu_expanded) / (sigma_expanded + 1e-8)) ** 2
            probs = F.softmax(log_probs, dim=-1)
        
        return probs

    def _get_continuous_distribution(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取连续动作空间的高斯分布参数
        
        Returns:
            mu: 均值
            sigma: 标准差
        """
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
                sigma = torch.ones_like(mu) * 0.1
        
        return mu, sigma

    def _compute_come_loss(
        self, 
        belief: torch.Tensor, 
        uncertainty: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        计算COME损失
        
        L_COME = f_select * [ -sum(b_k log b_k) - lambda_u * u log u ]
        
        Args:
            belief: 主观意见 (batch_size, K)
            uncertainty: 不确定性 (batch_size,)
            mask: 样本筛选掩码 (batch_size,)
        
        Returns:
            loss: COME损失标量
        """
        eps = 1e-8
        
        # 置信度分布的熵：-sum(b_k log b_k)
        # 添加数值稳定性处理
        belief_clamped = torch.clamp(belief, eps, 1.0)
        belief_entropy = -torch.sum(belief_clamped * torch.log(belief_clamped), dim=-1)  # (batch_size,)
        
        # 不确定性惩罚项：-lambda_u * u log u
        # 注意：u在(0,1]区间，log u为负，所以-u log u为正
        u_clamped = torch.clamp(uncertainty, eps, 1.0)
        uncertainty_term = -self.lambda_u * u_clamped * torch.log(u_clamped)
        
        # COME损失（对每个样本）
        come_loss_per_sample = belief_entropy + uncertainty_term
        
        # 应用样本筛选掩码（只使用高置信度样本）
        if mask.sum() > 0:
            come_loss = (come_loss_per_sample * mask).sum() / (mask.sum() + eps)
        else:
            come_loss = torch.tensor(0.0, device=self.device)
        
        return come_loss

    def _compute_kl_divergence(self, obs: torch.Tensor) -> torch.Tensor:
        """
        计算KL散度正则项 KL(pi_off || pi_tta)
        
        对于离散动作：策略分布间的KL
        对于连续动作：高斯分布间的KL
        """
        if not hasattr(self.policy, 'actor') or self._offline_actor is None:
            return torch.tensor(0.0, device=self.device)
        
        kl_div = torch.tensor(0.0, device=self.device)
        
        if self.is_discrete:
            # 离散动作：计算策略分布的KL
            with torch.no_grad():
                offline_logits = self._offline_actor(obs)
                offline_probs = F.softmax(offline_logits, dim=-1)
            
            current_logits = self.policy.actor(obs)
            # 对当前策略也应用保守约束，保持与不确定性计算一致
            current_logits_constrained = self._apply_conservative_constraint(current_logits)
            current_log_probs = F.log_softmax(current_logits_constrained, dim=-1)
            
            # KL(off || current) = sum(off * log(off / current))
            kl_div = F.kl_div(current_log_probs, offline_probs, reduction='batchmean', log_target=False)
        else:
            # 连续动作：计算高斯分布的KL
            with torch.no_grad():
                offline_dist = self._offline_actor(obs)
                if hasattr(offline_dist, 'mean'):
                    mu_off = offline_dist.mean
                else:
                    mu_off = offline_dist
                
                if hasattr(offline_dist, 'stddev'):
                    sigma_off = offline_dist.stddev
                else:
                    sigma_off = torch.ones_like(mu_off) * 0.1
            
            mu_tta, sigma_tta = self._get_continuous_distribution(obs)
            
            # 高斯KL: 0.5 * (sigma_tta^2/sigma_off^2 + (mu_off-mu_tta)^2/sigma_off^2 - 1 + log(sigma_off^2/sigma_tta^2))
            var_off = sigma_off ** 2 + 1e-8
            var_tta = sigma_tta ** 2 + 1e-8
            
            kl = 0.5 * (
                var_tta / var_off +
                (mu_off - mu_tta) ** 2 / var_off -
                1 +
                torch.log(var_off / var_tta)
            )
            kl_div = kl.mean()
        
        return kl_div

    def _filter_high_confidence_samples(
        self, 
        states: torch.Tensor, 
        uncertainties: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        步骤2.3：样本筛选（保留高置信度样本）
        
        f_select(s_t) = I{u(s_t) <= u_0 + delta}
        
        Args:
            states: 状态批次 (batch_size, state_dim)
            uncertainties: 不确定性 (batch_size,)
        
        Returns:
            filtered_states: 筛选后的状态
            mask: 二进制掩码 (batch_size,)
        """
        threshold = self.u_0 + self.uncertainty_delta
        
        # 创建掩码：u <= u_0 + delta
        mask = (uncertainties <= threshold).float()
        
        if mask.sum() == 0:
            # 如果没有样本满足条件，至少保留一个不确定性最小的样本
            min_idx = torch.argmin(uncertainties)
            mask[min_idx] = 1.0
        
        return states, mask

    def _update(self, obs_batch: torch.Tensor) -> Dict[str, Any]:
        """
        执行单步COME更新
        
        总优化目标：L_total = L_COME + lambda_KL * L_KL
        仅更新LayerNorm参数
        """
        self.policy.train()
        
        batch_size = obs_batch.shape[0]
        
        if self.is_discrete:
            # 离散动作空间流程
            # 1. 获取原始logits
            logits = self.policy.actor(obs_batch)
            
            # 2. 应用保守性约束
            logits_constrained = self._apply_conservative_constraint(logits)
            
            # 3. 计算Dirichlet参数和不确定性
            alpha, belief, uncertainty = self._compute_dirichlet_params(logits_constrained)
            
            # 4. 样本筛选
            _, mask = self._filter_high_confidence_samples(obs_batch, uncertainty)
            
            # 5. 计算COME损失
            come_loss = self._compute_come_loss(belief, uncertainty, mask)
            
        else:
            # 连续动作空间流程
            # 1. 获取高斯分布参数
            mu, sigma = self._get_continuous_distribution(obs_batch)
            
            # 2. 离散化为bins上的概率分布
            probs = self._discretize_gaussian(mu, sigma)  # (batch_size, num_bins)
            
            # 3. 将概率转换为Dirichlet证据（逆变换）
            # 假设当前概率代表belief，反推alpha
            # b_k = probs (因为sum(probs)=1)
            # u = K / S, 假设我们希望保持当前的u与预训练时类似
            # 这里我们直接将probs视为belief，计算熵
            # 对于连续任务，我们简化为直接在离散化后的分布上计算COME
            
            eps = 1e-8
            probs_clamped = torch.clamp(probs, eps, 1.0)
            
            # 计算该分布的"不确定性"（使用熵的近似）
            # 或者直接使用高斯熵计算不确定性，但为了统一框架，我们在离散化后计算
            belief_entropy = -torch.sum(probs_clamped * torch.log(probs_clamped), dim=-1)
            
            # 对于连续任务，我们使用高斯分布的熵作为不确定性度量
            # H = 0.5 * log(2*pi*e*sigma^2)
            gaussian_entropy = 0.5 * torch.log(2 * np.pi * np.e * sigma ** 2).mean(dim=-1)
            
            # 归一化到[0,1]区间（近似）
            u_approx = torch.sigmoid(gaussian_entropy)
            
            # 筛选样本（基于近似不确定性）
            _, mask = self._filter_high_confidence_samples(obs_batch, u_approx)
            
            # COME损失（简化版：最小化belief熵和不确定性）
            come_loss = (belief_entropy * mask).sum() / (mask.sum() + eps)
            come_loss = come_loss + self.lambda_u * (u_approx * mask).mean()
        
        # 6. 计算KL散度
        kl_loss = self._compute_kl_divergence(obs_batch)
        
        # 7. 总损失
        total_loss = come_loss + self.kl_weight * kl_loss
        
        # 8. 反向传播和参数更新（仅LayerNorm）
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # 检查梯度
        has_nonzero_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in self.params_only
        )
        if not has_nonzero_grad and self.adaptation_step > 0:
            print(f"  WARNING: ZERO GRADIENT in COME! KL={kl_loss.item():.6f}, COME={come_loss.item():.6f}")
        
        self.optimizer.step()
        
        return {
            'come_loss': come_loss.item(),
            'kl_loss': kl_loss.item(),
            'total_loss': total_loss.item(),
            'mean_uncertainty': uncertainty.mean().item() if self.is_discrete else u_approx.mean().item(),
            'selected_samples': mask.sum().item(),
            'u_0': self.u_0
        }

    def _run_single_episode(self) -> Dict[str, Any]:
        """运行单回合并收集状态数据，执行COME适应"""
        reset_result = self.env.reset()
        if isinstance(reset_result, tuple):
            obs, _ = reset_result
        else:
            obs = reset_result

        episode_reward = 0
        episode_length = 0
        episode_transitions = []
        episode_losses = []
        episode_come_losses = []
        episode_kl_losses = []
        episode_uncertainties = []

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
                'action': action.copy() if isinstance(action, np.ndarray) else action,
                'reward': reward,
                'next_obs': next_obs.copy(),
                'done': done
            }
            episode_transitions.append(transition)

            # 缓存状态用于适应
            self.state_cache.append(obs.copy())

            # 计算当前状态的不确定性（用于日志记录）
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                if self.is_discrete:
                    logits = self.policy.actor(obs_tensor)
                    logits_constrained = self._apply_conservative_constraint(logits)
                    _, _, u = self._compute_dirichlet_params(logits_constrained)
                    current_u = u.item()
                else:
                    mu, sigma = self._get_continuous_distribution(obs_tensor)
                    gaussian_entropy = 0.5 * torch.log(2 * np.pi * np.e * sigma ** 2).mean()
                    current_u = torch.sigmoid(gaussian_entropy).item()
                
                self.uncertainty_cache.append(current_u)
                episode_uncertainties.append(current_u)

            obs = next_obs
            episode_reward += reward
            episode_length += 1

            # 触发更新：缓存足够且满足更新频率
            if len(self.state_cache) >= self.update_freq and self.adaptation_step % self.update_freq == 0:
                obs_batch = torch.FloatTensor(
                    np.array(list(self.state_cache)[-self.update_freq:])
                ).to(self.device)

                update_info = self._update(obs_batch)
                episode_losses.append(update_info['total_loss'])
                episode_come_losses.append(update_info['come_loss'])
                episode_kl_losses.append(update_info['kl_loss'])
            
            self.adaptation_step += 1

            if episode_length >= 1000:
                break

        self.episode_count += 1

        # 计算回合统计
        avg_loss = np.mean(episode_losses) if episode_losses else 0.0
        avg_come_loss = np.mean(episode_come_losses) if episode_come_losses else 0.0
        avg_kl_loss = np.mean(episode_kl_losses) if episode_kl_losses else 0.0
        avg_uncertainty = np.mean(episode_uncertainties) if episode_uncertainties else 0.0

        return {
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            'transitions': episode_transitions,
            'adaptation_step': self.adaptation_step,
            'total_loss': avg_loss,
            'come_loss': avg_come_loss,
            'kl_loss': avg_kl_loss,
            'mean_uncertainty': avg_uncertainty,
            'u_0': self.u_0,
            'loss_history': episode_losses,
            'uncertainty_history': episode_uncertainties
        }

    def run_adaptation(self, num_episodes: int = 10) -> Tuple[List[Dict], Dict[str, Any]]:
        """
        运行COME适应指定回合数
        
        Args:
            num_episodes: 适应回合数
        
        Returns:
            adaptation_data: 每回合数据列表
            summary: 汇总统计
        """
        adaptation_data = []
        
        for episode in range(num_episodes):
            episode_data = self._run_single_episode()
            adaptation_data.append(episode_data)
            
            print(f"Episode {episode + 1}/{num_episodes} (COME): "
                  f"Reward: {episode_data['episode_reward']:.2f}, "
                  f"Length: {episode_data['episode_length']}, "
                  f"Loss: {episode_data['total_loss']:.6f}, "
                  f"Unc: {episode_data['mean_uncertainty']:.4f} (u_0={self.u_0:.4f})")

        # 汇总统计
        summary = {
            'mean_reward': np.mean([d['episode_reward'] for d in adaptation_data]),
            'std_reward': np.std([d['episode_reward'] for d in adaptation_data]),
            'mean_length': np.mean([d['episode_length'] for d in adaptation_data]),
            'mean_come_loss': np.mean([d['come_loss'] for d in adaptation_data]),
            'mean_kl_loss': np.mean([d['kl_loss'] for d in adaptation_data]),
            'mean_uncertainty': np.mean([d['mean_uncertainty'] for d in adaptation_data]),
            'u_0': self.u_0,
            'cache_size': len(self.state_cache),
            'adaptation_steps': self.adaptation_step
        }

        return adaptation_data, summary

    def evaluate_performance(self, num_episodes: int = 5) -> Dict[str, float]:
        """评估当前策略性能"""
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
        """获取COME适应统计信息"""
        return {
            'algorithm': 'COME',
            'adaptation_steps': self.adaptation_step,
            'cache_size': len(self.state_cache),
            'uncertainty_cache_size': len(self.uncertainty_cache),
            'trainable_params': len(self.trainable_params),
            'learning_rate': self.learning_rate,
            'kl_weight': self.kl_weight,
            'lambda_u': self.lambda_u,
            'uncertainty_delta': self.uncertainty_delta,
            'u_0': self.u_0,
            'adaptation_mode': self.adaptation_mode,
            'is_discrete': self.is_discrete,
            'num_actions': self.num_actions,
            'param_names': self.param_names[:5]  # 仅显示前5个参数名
        }

    def save_checkpoint(self, save_path: str):
        """保存COME检查点"""
        checkpoint = {
            'policy_state_dict': self.policy.state_dict(),
            'offline_policy_state': self.offline_policy_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'adaptation_step': self.adaptation_step,
            'episode_count': self.episode_count,
            'u_0': self.u_0,
            'config': self.config,
            'statistics': self.get_statistics()
        }
        torch.save(checkpoint, save_path)

    def load_checkpoint(self, checkpoint_path: str):
        """加载COME检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.offline_policy_state = checkpoint['offline_policy_state']
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.adaptation_step = checkpoint['adaptation_step']
        self.episode_count = checkpoint.get('episode_count', 0)
        self.u_0 = checkpoint.get('u_0', 0.5)
        self.config = checkpoint.get('config', {})