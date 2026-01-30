import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Type
import os
import gym

from offlinerlkit.policy.base_policy import BasePolicy
from offlinerlkit.nets import MLP
from offlinerlkit.modules import ActorProb, Critic, TanhDiagGaussian


class ModelLoader:
    """
    统一的模型加载和适配接口
    
    支持功能：
    - 加载预训练模型
    - 模型权重验证
    - 策略适配到新环境
    - 渐进式适应支持
    """
    
    def __init__(self, policy_class: Type[BasePolicy], device: str = 'cuda'):
        self.policy_class = policy_class
        self.device = device
        
    def load_pretrained_model(self, checkpoint_path: str, env_config: Dict[str, Any]) -> BasePolicy:
        """
        加载预训练模型并进行环境适配
        
        Args:
            checkpoint_path: 模型检查点路径
            env_config: 环境配置信息
            
        Returns:
            加载并适配后的策略实例
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
            
        # 加载模型权重
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # 创建策略实例
        policy = self._create_policy(env_config)
        
        # 加载状态字典
        if 'policy' in checkpoint:
            policy.load_state_dict(checkpoint['policy'])
        elif 'model_state_dict' in checkpoint:
            policy.load_state_dict(checkpoint['model_state_dict'])
        else:
            # 尝试直接加载
            policy.load_state_dict(checkpoint)
            
        policy.eval()
        return policy
    
    def _create_policy(self, env_config: Dict[str, Any]) -> BasePolicy:
        """根据环境配置创建策略实例"""
        obs_dim = env_config.get('obs_dim', env_config.get('observation_space').shape[0])
        action_dim = env_config.get('action_dim', env_config.get('action_space').shape[0])
        action_space = env_config.get('action_space')
        
        if self.policy_class.__name__ == 'CQLPolicy':
            return self._create_cql_policy(obs_dim, action_dim, action_space)
        else:
            return self._create_generic_policy(obs_dim, action_dim, action_space)
    
    def _create_cql_policy(self, obs_dim: int, action_dim: int, action_space: gym.spaces.Space,
                          hidden_dims: list = None, use_layernorm_actor: bool = True) -> BasePolicy:
        """创建CQLPolicy实例"""
        if hidden_dims is None:
            hidden_dims = [256, 256, 256]
        
        actor_lr = 1e-4
        critic_lr = 3e-4
        gamma = 0.99
        tau = 0.005
        alpha = 0.2
        cql_weight = 5.0
        temperature = 1.0
        max_q_backup = False
        deterministic_backup = True
        with_lagrange = False
        lagrange_threshold = 10.0
        cql_alpha_lr = 3e-4
        num_repeat_actions = 10
        
        max_action = action_space.high[0]
        
        actor_backbone = MLP(input_dim=obs_dim, hidden_dims=hidden_dims, use_layernorm=use_layernorm_actor)
        critic1_backbone = MLP(input_dim=obs_dim + action_dim, hidden_dims=hidden_dims)
        critic2_backbone = MLP(input_dim=obs_dim + action_dim, hidden_dims=hidden_dims)
        
        dist = TanhDiagGaussian(
            latent_dim=getattr(actor_backbone, "output_dim"),
            output_dim=action_dim,
            unbounded=True,
            conditioned_sigma=True,
            max_mu=max_action
        )
        
        actor = ActorProb(actor_backbone, dist, self.device)
        critic1 = Critic(critic1_backbone, self.device)
        critic2 = Critic(critic2_backbone, self.device)
        
        actor_optim = torch.optim.Adam(actor.parameters(), lr=actor_lr)
        critic1_optim = torch.optim.Adam(critic1.parameters(), lr=critic_lr)
        critic2_optim = torch.optim.Adam(critic2.parameters(), lr=critic_lr)
        
        policy = self.policy_class(
            actor,
            critic1,
            critic2,
            actor_optim,
            critic1_optim,
            critic2_optim,
            action_space=action_space,
            tau=tau,
            gamma=gamma,
            alpha=alpha,
            cql_weight=cql_weight,
            temperature=temperature,
            max_q_backup=max_q_backup,
            deterministic_backup=deterministic_backup,
            with_lagrange=with_lagrange,
            lagrange_threshold=lagrange_threshold,
            cql_alpha_lr=cql_alpha_lr,
            num_repeart_actions=num_repeat_actions
        )
        
        return policy
    
    def _create_generic_policy(self, obs_dim: int, action_dim: int, action_space: gym.spaces.Space) -> BasePolicy:
        """创建通用策略实例（用于其他策略类）"""
        policy = self.policy_class(
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=self.device
        )
        
        return policy
    
    def adapt_model_to_shift(self, policy: BasePolicy, shift_config: Dict[str, Any], 
                           adaptation_strategy: str = 'freeze_actor') -> BasePolicy:
        """
        将模型适配到新的shift环境
        
        Args:
            policy: 原始策略
            shift_config: shift配置
            adaptation_strategy: 适应策略
                - 'freeze_actor': 冻结actor，只适应critic
                - 'fine_tune_all': 微调所有参数
                - 'progressive': 渐进式适应
                
        Returns:
            适配后的策略
        """
        adapted_policy = policy
        
        # 设置参数更新策略
        if adaptation_strategy == 'freeze_actor':
            self._freeze_actor_parameters(adapted_policy)
        elif adaptation_strategy == 'fine_tune_all':
            self._unfreeze_all_parameters(adapted_policy)
        elif adaptation_strategy == 'progressive':
            self._setup_progressive_adaptation(adapted_policy, shift_config)
            
        # 设置优化器
        self._setup_adaptation_optimizer(adapted_policy, shift_config)
        
        return adapted_policy
    
    def _freeze_actor_parameters(self, policy: BasePolicy):
        """冻结actor参数"""
        if hasattr(policy, 'actor'):
            for param in policy.actor.parameters():
                param.requires_grad = False
                
    def _unfreeze_all_parameters(self, policy: BasePolicy):
        """解冻所有参数"""
        for param in policy.parameters():
            param.requires_grad = True
            
    def _setup_progressive_adaptation(self, policy: BasePolicy, shift_config: Dict[str, Any]):
        """设置渐进式适应"""
        # 根据shift强度决定解冻程度
        shift_strength = self._calculate_shift_strength(shift_config)
        
        if shift_strength < 0.1:
            # 轻微shift，只解冻最后几层
            self._unfreeze_last_layers(policy, num_layers=1)
        elif shift_strength < 0.3:
            # 中等shift，解冻更多层
            self._unfreeze_last_layers(policy, num_layers=2)
        else:
            # 强烈shift，解冻所有参数
            self._unfreeze_all_parameters(policy)
    
    def _calculate_shift_strength(self, shift_config: Dict[str, Any]) -> float:
        """计算shift强度"""
        strength = 0.0
        
        # 动态参数shift强度
        if 'dynamic_shifts' in shift_config:
            dyn_shifts = shift_config['dynamic_shifts']
            if 'mass_scale' in dyn_shifts:
                strength += abs(1.0 - dyn_shifts['mass_scale']) * 0.3
            if 'friction_scale' in dyn_shifts:
                strength += abs(1.0 - dyn_shifts['friction_scale']) * 0.3
                
        # 观测shift强度
        if 'observation_shifts' in shift_config:
            obs_shifts = shift_config['observation_shifts']
            if 'noise_std' in obs_shifts:
                strength += obs_shifts['noise_std'] * 0.2
            if 'scale' in obs_shifts:
                strength += abs(1.0 - obs_shifts['scale']) * 0.2
                
        return min(strength, 1.0)
    
    def _unfreeze_last_layers(self, policy: BasePolicy, num_layers: int):
        """解冻最后几层参数"""
        # 这里需要根据具体的网络结构实现
        # 示例实现
        if hasattr(policy, 'actor') and hasattr(policy.actor, 'backbone'):
            layers = list(policy.actor.backbone.children())
            for layer in layers[-num_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
    
    def _setup_adaptation_optimizer(self, policy: BasePolicy, shift_config: Dict[str, Any]):
        """设置适应优化器"""
        # 获取需要训练的参数
        trainable_params = [p for p in policy.parameters() if p.requires_grad]
        
        if len(trainable_params) > 0:
            # 设置较小的学习率进行适应
            lr = shift_config.get('adaptation_lr', 1e-5)
            policy.adaptation_optimizer = torch.optim.Adam(trainable_params, lr=lr)
    
    def validate_model_compatibility(self, policy: BasePolicy, env_config: Dict[str, Any]) -> bool:
        """验证模型与环境的兼容性"""
        try:
            # 检查观测维度
            obs_dim = env_config.get('obs_dim', env_config.get('observation_space').shape[0])
            action_dim = env_config.get('action_dim', env_config.get('action_space').shape[0])
            
            # 创建测试输入
            test_obs = torch.randn(1, obs_dim).to(self.device)
            
            # 测试前向传播 - 使用 select_action 方法
            with torch.no_grad():
                action = policy.select_action(test_obs.cpu().numpy())
                
            # 检查输出维度
            if hasattr(action, 'shape'):
                return action.shape[-1] == action_dim if len(action.shape) > 0 else action.shape[0] == action_dim
            elif isinstance(action, np.ndarray):
                return action.shape[-1] == action_dim if len(action.shape) > 0 else len(action) == action_dim
            
            return True
            
        except Exception as e:
            print(f"Model compatibility validation failed: {e}")
            return False
    
    def save_adapted_model(self, policy: BasePolicy, save_path: str):
        """保存适配后的模型"""
        torch.save({
            'policy_state_dict': policy.state_dict(),
            'adaptation_config': getattr(policy, 'adaptation_config', {}),
            'shift_info': getattr(policy, 'shift_info', {})
        }, save_path)