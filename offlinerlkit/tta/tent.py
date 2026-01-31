import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from collections import deque

from offlinerlkit.policy.base_policy import BasePolicy


class TentManager:
    """
    Tent: Fully Test-Time Adaptation
    
    Tent通过最小化预测熵来适应测试时分布变化，
    仅更新批归一化层的统计量和仿射参数。
    
    核心思想：
    - 在测试时启用批归一化层的梯度更新
    - 通过最小化预测熵来适应新分布
    - 保持网络权重不变，只更新BN统计量
    
    Reference: Fully Test-Time Adaptation by Entropy Minimization (ICLR 2021)
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
        self.learning_rate = self.config.get('learning_rate', 1e-3)
        self.momentum = self.config.get('momentum', 0.9)
        self.damping = self.config.get('damping', 0.0)
        
        # 配置可训练参数（仅BN层）
        self.trainable_params = self._get_bn_params()
        
        if len(self.trainable_params) == 0:
            print("Warning: No BatchNorm params found, using LayerNorm instead")
            self.trainable_params = self._get_layernorm_params()
        
        assert len(self.trainable_params) > 0, "No trainable params found"
        
        self.param_names = [name for name, _ in self.trainable_params]
        params_only = [param for _, param in self.trainable_params]
        
        # 使用SGD优化器（Tent原文使用）
        self.optimizer = torch.optim.SGD(
            params_only, 
            lr=self.learning_rate, 
            momentum=self.momentum,
            dampening=self.damping
        )
        
        # 状态缓存
        self.cache_capacity = self.config.get('cache_capacity', 100)
        self.state_cache = deque(maxlen=self.cache_capacity)
        
        # 统计信息
        self.adaptation_step = 0
        self.entropy_history = []

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

    def _get_bn_params(self) -> List[Tuple[str, nn.Parameter]]:
        """获取批归一化层的参数"""
        params = []
        for name, module in self.policy.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                # 启用梯度计算
                module.track_running_stats = True
                module.requires_grad_(True)
                # 收集可训练参数
                for param_name, param in module.named_parameters():
                    params.append((f"{name}.{param_name}", param))
        return params

    def _get_layernorm_params(self) -> List[Tuple[str, nn.Parameter]]:
        """获取层归一化层的参数"""
        params = []
        for name, module in self.policy.named_modules():
            if isinstance(module, nn.LayerNorm):
                module.requires_grad_(True)
                for param_name, param in module.named_parameters():
                    params.append((f"{name}.{param_name}", param))
        return params

    def _enable_bn_training(self):
        """启用BN层的训练模式"""
        for module in self.policy.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.train()

    def _disable_bn_training(self):
        """禁用BN层的训练模式"""
        for module in self.policy.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.eval()

    def compute_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """
        计算预测分布的熵
        
        Args:
            logits: 策略输出的logits
            
        Returns:
            熵值
        """
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
        return entropy.mean()

    def adapt(self, obs: np.ndarray) -> Dict[str, Any]:
        """
        执行一次Tent适应步骤
        
        Args:
            obs: 当前观测状态
            
        Returns:
            适应统计信息
        """
        self._enable_bn_training()
        
        # 转换观测为tensor
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        # 前向传播
        with torch.enable_grad():
            if hasattr(self.policy, 'actor'):
                action_dist = self.policy.actor(obs_tensor)
                if hasattr(action_dist, 'logits'):
                    logits = action_dist.logits
                else:
                    # 对于连续动作空间，使用动作分布的均值
                    logits = action_dist.mean
            else:
                logits = self.policy(obs_tensor)
            
            # 计算熵损失
            loss = self.compute_entropy(logits)
        
        # 反向传播和参数更新
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.adaptation_step += 1
        self.entropy_history.append(loss.item())
        
        # 缓存状态
        self.state_cache.append(obs)
        
        self._disable_bn_training()
        
        return {
            'entropy': loss.item(),
            'adaptation_step': self.adaptation_step
        }

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        选择动作
        
        Args:
            obs: 观测状态
            deterministic: 是否确定性选择
            
        Returns:
            动作
        """
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

    def reset(self):
        """重置适应状态"""
        self.adaptation_step = 0
        self.entropy_history.clear()
        self.state_cache.clear()
        self.optimizer.zero_grad()

    def get_stats(self) -> Dict[str, Any]:
        """获取适应统计信息"""
        stats = {
            'adaptation_step': self.adaptation_step,
            'entropy_history': self.entropy_history.copy(),
            'mean_entropy': np.mean(self.entropy_history) if self.entropy_history else 0.0,
            'cache_size': len(self.state_cache)
        }
        return stats
