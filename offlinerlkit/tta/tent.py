import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from collections import deque

from offlinerlkit.policy.base_policy import BasePolicy


class TentManager:
    """
    TENT-RL: Test-Time Adaptation for Continuous Control via Feature Variance Minimization
    
    核心机制（严格遵循TENT原始设计）：
    1. 仅更新LayerNorm的gamma/beta参数（<1%总参数）
    2. 最小化最后隐藏层特征的方差（而非动作熵）-> 对应分类任务中的预测熵
    3. Batch-wise更新（滑动窗口），单步更新会导致坍塌
    4. 无KL约束，依靠极小学习率(1e-6~1e-7)和梯度裁剪保证稳定性
    
    理论对应关系：
    - TENT分类: 熵 H(p) = -sum(p log p)  ->  预测不确定性
    - TENT-RL:   方差 log(Var[z] + eps)  ->  特征表征一致性（决策确定性）
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

        # ===== 关键超参数（TENT-RL推荐配置） =====
        # 学习率必须极小，防止LN参数震荡导致策略突变
        self.learning_rate = self.config.get('learning_rate', 1e-6)  # 默认1e-6，可调至1e-7
        self.batch_size = self.config.get('batch_size', 32)        # Batch更新是TENT的核心
        self.max_grad_norm = self.config.get('max_grad_norm', 0.1) # 梯度裁剪阈值
        self.adapt_interval = self.config.get('adapt_interval', 1) # 每N步适应一次
        self.feature_layer_name = self.config.get('feature_layer_name', None)  # 指定特征层名，默认自动识别最后一层LN前
        
        # 获取可训练的LayerNorm参数（gamma, beta）
        self.trainable_params = self._get_layernorm_params()
        assert len(self.trainable_params) > 0, "No LayerNorm params found in policy"
        
        self.param_names = [name for name, _ in self.trainable_params]
        params_only = [param for _, param in self.trainable_params]
        
        print(f"[TENT-RL] Found {len(params_only)} LayerNorm parameters to adapt ({len(params_only)/sum(p.numel() for p in policy.parameters())*100:.2f}% of total)")
        print(f"[TENT-RL] LR={self.learning_rate}, BatchSize={self.batch_size}, Objective=Feature Variance")
        
        # 使用SGD（TENT原始设计），但Adam也可行，只要学习率足够小
        optimizer_type = self.config.get('optimizer', 'sgd')  # 默认SGD遵循TENT原文
        if optimizer_type.lower() == 'adam':
            self.optimizer = torch.optim.Adam(params_only, lr=self.learning_rate, betas=(0.9, 0.999))
        else:
            self.optimizer = torch.optim.SGD(params_only, lr=self.learning_rate, momentum=0.9)
        
        # 状态缓存（滑动窗口，用于构建batch）
        self.state_cache = deque(maxlen=self.batch_size)
        self.step_count = 0
        
        # 注册hook以捕获中间层特征（如果网络不直接暴露）
        self.feature_hook = None
        self.current_features = None
        self._register_feature_hook()
        
        # 统计信息
        self.adaptation_step = 0
        self.uncertainty_history = []  # 记录特征方差

    def _get_policy_device(self) -> torch.device:
        """获取策略所在设备"""
        if hasattr(self.policy, 'actor') and hasattr(self.policy.actor, 'device'):
            return self.policy.actor.device
        elif len(list(self.policy.parameters())) > 0:
            return next(self.policy.parameters()).device
        return torch.device('cpu')

    def _get_layernorm_params(self) -> List[Tuple[str, nn.Parameter]]:
        """
        仅获取LayerNorm层的可训练参数（weight->gamma, bias->beta）
        严格遵循TENT：只更新归一化层的affine参数
        """
        params = []
        for name, module in self.policy.named_modules():
            if isinstance(module, nn.LayerNorm):
                # LayerNorm的weight就是gamma，bias就是beta
                for param_name, param in module.named_parameters():
                    if param.requires_grad:
                        params.append((f"{name}.{param_name}", param))
        return params

    def _register_feature_hook(self):
        """
        注册forward hook以捕获最后隐藏层特征（LN的输入）
        这是实现特征方差最小化的关键
        """
        # 如果指定了层名，直接找该层；否则找最后一个LayerNorm的输入
        if self.feature_layer_name:
            target_module = dict(self.policy.named_modules()).get(self.feature_layer_name)
            if target_module is None:
                print(f"[TENT-RL Warning] Specified layer {self.feature_layer_name} not found, using auto-detect")
                target_module = self._find_last_layernorm_input()
        else:
            target_module = self._find_last_layernorm_input()
            
        if target_module is not None:
            self.feature_hook = target_module.register_forward_hook(self._extract_feature_hook)
            print(f"[TENT-RL] Registered feature hook at {target_module}")

    def _find_last_layernorm_input(self):
        """
        自动寻找最后一个LayerNorm层，返回其前驱模块（即特征提取层）
        这样hook可以捕获进入LN前的特征z^(L)
        """
        layernorms = [m for m in self.policy.modules() if isinstance(m, nn.LayerNorm)]
        if not layernorms:
            return None
        
        # 获取最后一个LN层
        last_ln = layernorms[-1]
        
        # 寻找该LN层的"父模块"，即直接输出给LN的线性层或激活层
        # 这是一个启发式搜索，假设结构是 ... -> Linear/ReLU -> LayerNorm -> ...
        for name, module in self.policy.named_modules():
            for child_name, child_module in module.named_children():
                if child_module is last_ln:
                    # 找到包含该LN的父模块，尝试找前一个兄弟节点
                    children = list(module.named_children())
                    for i, (cname, cmod) in enumerate(children):
                        if cmod is last_ln and i > 0:
                            # 返回LN之前的那一层
                            return children[i-1][1]
        # 如果没找到，就hook LN本身（会拿到归一化后的特征，也可以接受）
        return last_ln

    def _extract_feature_hook(self, module, input, output):
        """
        Hook函数：捕获前向传播中的中间特征
        存储输入给LN的特征（未归一化的z）
        """
        # input是tuple，取第一个tensor
        if isinstance(input, tuple) and len(input) > 0:
            self.current_features = input[0]
        else:
            self.current_features = output  #  fallback

    def compute_feature_uncertainty(self, features: torch.Tensor) -> torch.Tensor:
        """
        计算特征不确定性（对应TENT中的预测熵）
        
        公式: U = (1/d) * sum_i log(Var[z_i] + epsilon)
        
        其中:
        - z是最后隐藏层特征 [batch_size, d]
        - Var是沿batch维度的方差
        - log保证梯度平滑，对应熵的log形式
        - epsilon防止方差为0
        """
        if features is None:
            return torch.tensor(0.0, device=self.device)
        
        # 确保是2D tensor [batch, features]
        if features.dim() == 1:
            features = features.unsqueeze(0)
        
        # 沿batch维度计算每个特征维度的方差
        # Var[z_i] = mean((z_i - mean(z_i))^2)
        mean_z = features.mean(dim=0, keepdim=True)  # [1, d]
        var_z = ((features - mean_z) ** 2).mean(dim=0)  # [d]
        
        # 不确定性度量: 平均对数方差
        # 这是与TENT熵公式 H = -sum p log p 对应的连续形式
        epsilon = 1e-5  # 防止log(0)
        uncertainty = torch.log(var_z + epsilon).mean()
        
        return uncertainty

    def adapt(self, obs: np.ndarray, force_update: bool = False) -> Dict[str, Any]:
        """
        执行TENT-RL适应步骤（Batch-wise更新）
        
        核心逻辑：
        1. 积累状态到滑动窗口（batch）
        2. 当batch满时，前向传播获取特征
        3. 计算特征方差损失（公式1）
        4. 仅更新LayerNorm参数（公式4）
        
        Args:
            obs: 当前观测 (numpy array)
            force_update: 是否强制更新（即使buffer未满，回合结束时使用）
            
        Returns:
            adaptation_info: 包含uncertainty（原entropy字段，保持接口兼容）、updated等
        """
        self.step_count += 1
        self.state_cache.append(obs)
        
        adapt_info = {'entropy': 0.0, 'uncertainty': 0.0, 'updated': False}
        
        # 检查是否满足更新条件：buffer满 或 强制更新
        buffer_ready = len(self.state_cache) >= self.batch_size
        should_update = (buffer_ready and self.step_count % self.adapt_interval == 0) or force_update
        
        if not should_update:
            return adapt_info
        
        # 确保LN层在训练模式（允许更新参数）
        self._enable_ln_training()
        
        # 准备batch数据
        obs_batch = np.array(list(self.state_cache))
        obs_tensor = torch.FloatTensor(obs_batch).to(self.device)
        
        try:
            self.optimizer.zero_grad()
            
            # 前向传播（会触发hook捕获特征）
            self.current_features = None
            
            with torch.enable_grad():
                # 前向传播以获取特征（不计算动作梯度，只保留特征图）
                if hasattr(self.policy, 'actor'):
                    # 标准离线RL策略结构
                    if hasattr(self.policy.actor, 'forward'):
                        _ = self.policy.actor(obs_tensor)
                    else:
                        _ = self.policy.actor.get_action(obs_tensor)
                else:
                    _ = self.policy(obs_tensor)
            
            # 获取捕获的特征（最后隐藏层，LN输入）
            features = self.current_features
            
            if features is None:
                print("[TENT-RL Warning] No features captured, check hook registration")
                return adapt_info
            
            # 计算不确定性损失：特征方差（公式1）
            loss = self.compute_feature_uncertainty(features)
            
            # 反向传播（仅计算LN参数的梯度）
            loss.backward()
            
            # 梯度裁剪（关键稳定性措施）
            torch.nn.utils.clip_grad_norm_(
                [p for _, p in self.trainable_params], 
                self.max_grad_norm
            )
            
            # 参数更新（公式4）
            self.optimizer.step()
            
            self.adaptation_step += 1
            uncertainty_val = loss.item()
            self.uncertainty_history.append(uncertainty_val)
            
            # 保持接口兼容：entropy字段实际存储的是特征不确定性
            adapt_info = {
                'entropy': uncertainty_val,      # 兼容旧接口
                'uncertainty': uncertainty_val,  # 新字段，更准确
                'updated': True,
                'adaptation_step': self.adaptation_step,
                'feature_mean_var': torch.exp(loss).item()  # 还原方差量级供参考
            }
            
        except Exception as e:
            print(f"[TENT-RL Warning] Adaptation failed: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            # 恢复eval模式用于动作选择（但LN参数已更新）
            self.policy.eval()
            
        return adapt_info

    def _enable_ln_training(self):
        """确保LayerNorm层处于可训练状态（同时保持其他层eval）"""
        for module in self.policy.modules():
            if isinstance(module, nn.LayerNorm):
                module.train()  # 允许更新LN参数
            else:
                module.eval()   # 其他层保持eval（关闭dropout等）

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        选择动作（纯推理模式，不更新参数）
        保持与原接口完全一致
        """
        self.policy.eval()
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if hasattr(self.policy, 'select_action'):
                action = self.policy.select_action(obs_tensor, deterministic)
            elif hasattr(self.policy, 'actor'):
                action_dist = self.policy.actor(obs_tensor)
                if deterministic:
                    action = action_dist.mean
                else:
                    action = action_dist.sample()
            else:
                action = self.policy(obs_tensor)
                
            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy()
                
        return action.flatten()

    def reset_episode(self):
        """
        每回合结束后调用（关键！）
        清空状态缓存并重置优化器动量，防止跨回合信息泄漏
        这是防止长期漂移的关键机制
        """
        self.state_cache.clear()
        self.step_count = 0
        self.optimizer.zero_grad()
        
        # 重置优化器状态（清除momentum）
        # 这对防止回合间梯度信息累积至关重要
        if isinstance(self.optimizer, torch.optim.Adam):
            for param_group in self.optimizer.param_groups:
                for p in param_group['params']:
                    state = self.optimizer.state[p]
                    if len(state) > 0:
                        state['exp_avg'].zero_()
                        state['exp_avg_sq'].zero_()
        elif isinstance(self.optimizer, torch.optim.SGD):
            for param_group in self.optimizer.param_groups:
                for p in param_group['params']:
                    state = self.optimizer.state[p]
                    if 'momentum_buffer' in state:
                        state['momentum_buffer'].zero_()

    def run_adaptation(self, num_episodes: int = 20) -> Tuple[List[Dict], Dict[str, Any]]:
        """
        运行完整的测试时适应流程
        接口与原代码完全一致
        """
        adaptation_data = []
        
        for episode in range(num_episodes):
            # 每回合开始重置状态（TENT设计要求）
            self.reset_episode()
            
            reset_result = self.env.reset()
            obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
            
            episode_reward = 0
            episode_length = 0
            episode_uncertainties = []
            
            done = False
            while not done:
                # 步骤1：批量适应（收集batch后更新LN参数）
                adapt_info = self.adapt(obs, force_update=False)
                if adapt_info['updated']:
                    episode_uncertainties.append(adapt_info['uncertainty'])
                
                # 步骤2：使用适应后的策略选择动作
                action = self.select_action(obs, deterministic=False)
                
                # 步骤3：执行动作
                step_result = self.env.step(action)
                if len(step_result) == 5:
                    obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:
                    obs, reward, done, info = step_result
                
                episode_reward += reward
                episode_length += 1
                
                # 安全保护：防止无限循环
                if episode_length >= 1000:
                    break
            
            # 回合结束：强制更新剩余buffer中的状态
            if len(self.state_cache) > 0:
                final_adapt = self.adapt(obs, force_update=True)
                if final_adapt['updated']:
                    episode_uncertainties.append(final_adapt['uncertainty'])
            
            # 记录回合统计
            episode_data = {
                'episode_reward': episode_reward,
                'episode_length': episode_length,
                'mean_entropy': np.mean(episode_uncertainties) if episode_uncertainties else 0.0,  # 兼容字段
                'mean_uncertainty': np.mean(episode_uncertainties) if episode_uncertainties else 0.0,
                'final_uncertainty': episode_uncertainties[-1] if episode_uncertainties else 0.0,
                'variance_reduction': (episode_uncertainties[0] - episode_uncertainties[-1]) if len(episode_uncertainties) > 1 else 0.0
            }
            adaptation_data.append(episode_data)
            
            print(f"Episode {episode + 1}/{num_episodes} (TENT-RL): "
                  f"Reward: {episode_reward:.2f}, "
                  f"Length: {episode_length}, "
                  f"Mean Uncertainty: {episode_data['mean_uncertainty']:.4f}, "
                  f"Variance Red: {episode_data['variance_reduction']:.4f}")
        
        # 汇总统计
        summary = {
            'mean_reward': np.mean([d['episode_reward'] for d in adaptation_data]),
            'std_reward': np.std([d['episode_reward'] for d in adaptation_data]),
            'mean_uncertainty': np.mean([d.get('mean_uncertainty', 0) for d in adaptation_data]),
            'total_adaptation_steps': self.adaptation_step
        }
        
        return adaptation_data, summary
