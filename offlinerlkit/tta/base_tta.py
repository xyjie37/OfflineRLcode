"""
Base TTA Algorithm - 通用TTA算法基类

提供所有TTA算法的通用功能：
- 设备管理
- 策略状态保存
- LayerNorm参数提取
- Episode运行
- 性能评估
- 结果汇总
"""

import torch
import torch.nn as nn
import copy
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from collections import deque

from offlinerlkit.policy.base_policy import BasePolicy


class BaseTTAAlgorithm:
    """
    TTA算法基类
    
    所有TTA算法应该继承此类，实现以下方法：
    - _compute_adaptation_loss(): 计算适应损失
    - _should_adapt(): 判断是否应该触发适应（可选）
    - _get_algorithm_name(): 返回算法名称
    - _get_additional_summary(): 返回额外的摘要信息（可选）
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
        self.learning_rate = self.config.get('learning_rate', 1e-5)
        self.adaptation_mode = self.config.get('adaptation_mode', 'layernorm')
        
        # 保存离线策略状态
        self.offline_policy_state = self._save_policy_state()
        
        # 获取可训练参数（默认为LayerNorm）
        self.trainable_params = self._get_trainable_params()
        self.param_names = [name for name, _ in self.trainable_params]
        
        # 创建优化器
        params_only = [param for _, param in self.trainable_params]
        self.optimizer = torch.optim.Adam(params_only, lr=self.learning_rate)
        
        # 适应状态
        self.adaptation_step = 0
        
        # 缓存（可选）
        self.state_cache = deque(maxlen=self.config.get('cache_capacity', 1000))
        
        print(f"Initialized {self._get_algorithm_name()} with {len(self.trainable_params)} trainable parameters")
        print(f"  Adaptation mode: {self.adaptation_mode}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Device: {self.device}")
    
    def _get_policy_device(self) -> torch.device:
        """获取策略所在的设备"""
        if hasattr(self.policy, 'actor') and hasattr(self.policy.actor, 'device'):
            return self.policy.actor.device
        elif hasattr(self.policy, 'critic1') and hasattr(self.policy.critic1, 'device'):
            return self.policy.critic1.device
        elif hasattr(self.policy, 'critic') and hasattr(self.policy.critic, 'device'):
            return self.policy.critic.device
        elif len(list(self.policy.parameters())) > 0:
            return next(self.policy.parameters()).device
        else:
            return torch.device('cpu')
    
    def _save_policy_state(self) -> Dict[str, torch.Tensor]:
        """保存当前策略状态"""
        state = {}
        for name, param in self.policy.named_parameters():
            state[name] = param.data.clone().detach()
        return state
    
    def _get_trainable_params(self) -> List[Tuple[str, nn.Parameter]]:
        """
        获取可训练参数
        
        根据adaptation_mode返回不同的参数：
        - 'layernorm': 仅LayerNorm层参数
        - 'last_n_layers': 最后N层的参数
        - 'policy_head': 策略头部参数
        - 'all': 所有参数
        """
        trainable_params = []
        
        if self.adaptation_mode == 'layernorm':
            trainable_params = self._get_layernorm_params()
            if len(trainable_params) == 0:
                print("警告: 未找到LayerNorm参数，自动切换到last_n_layers模式")
                self.adaptation_mode = 'last_n_layers'
                n_layers = self.config.get('last_n_layers', 2)
                trainable_params = self._get_last_n_layers(n_layers)
        
        elif self.adaptation_mode == 'last_n_layers':
            n_layers = self.config.get('last_n_layers', 2)
            trainable_params = self._get_last_n_layers(n_layers)
        
        elif self.adaptation_mode == 'policy_head':
            trainable_params = self._get_policy_head_params()
        
        elif self.adaptation_mode == 'all':
            for name, param in self.policy.named_parameters():
                if param.requires_grad:
                    trainable_params.append((name, param))
        
        assert len(trainable_params) > 0, f"无法找到可训练参数，adaptation_mode={self.adaptation_mode}"
        
        return trainable_params
    
    def _get_layernorm_params(self) -> List[Tuple[str, nn.Parameter]]:
        """获取LayerNorm层参数"""
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
        """获取最后N层的参数"""
        trainable_params = []
        
        if hasattr(self.policy, 'actor'):
            if hasattr(self.policy.actor, 'backbone'):
                backbone_params = list(self.policy.actor.backbone.named_parameters())
                selected = backbone_params[-n:] if len(backbone_params) >= n else backbone_params
                for name, param in selected:
                    if param.requires_grad:
                        trainable_params.append((f"backbone.{name}", param))
            
            for layer_name in ['last', 'dist_net', 'mean_layer', 'log_std_layer']:
                if hasattr(self.policy.actor, layer_name):
                    layer = getattr(self.policy.actor, layer_name)
                    for name, param in layer.named_parameters():
                        if param.requires_grad:
                            trainable_params.append((f"{layer_name}.{name}", param))
        
        return trainable_params
    
    def _get_policy_head_params(self) -> List[Tuple[str, nn.Parameter]]:
        """获取策略头部参数"""
        trainable_params = []
        
        if hasattr(self.policy, 'actor'):
            for layer_name in ['last', 'dist_net', 'mean_layer', 'log_std_layer']:
                if hasattr(self.policy.actor, layer_name):
                    layer = getattr(self.policy.actor, layer_name)
                    for name, param in layer.named_parameters():
                        if param.requires_grad:
                            trainable_params.append((f"{layer_name}.{name}", param))
        
        return trainable_params
    
    def _run_single_episode(self) -> Dict[str, Any]:
        """运行单个episode"""
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
            self.state_cache.append(obs.copy())
            
            with torch.no_grad():
                action = self.policy.select_action(obs)
            
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

        self.policy.train()
        
        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'mean_length': np.mean(lengths)
        }
    
    def run_adaptation(self, num_episodes: int = 10) -> Tuple[List[Dict], Dict[str, Any]]:
        """
        运行完整的适应过程
        
        Args:
            num_episodes: 适应回合数
            
        Returns:
            adaptation_data: 每个episode的数据列表
            summary: 汇总统计
        """
        adaptation_data = []
        
        for episode in range(num_episodes):
            episode_data = self._run_single_episode()
            
            # 执行适应（由子类实现）
            if self._should_adapt(episode_data):
                self._perform_adaptation(episode_data)
            
            adaptation_data.append(episode_data)
            self.adaptation_step += 1
            
            # 打印进度
            self._print_episode_progress(episode + 1, num_episodes, episode_data)
        
        summary = self._compute_summary(adaptation_data)
        
        return adaptation_data, summary
    
    def _should_adapt(self, episode_data: Dict[str, Any]) -> bool:
        """
        判断是否应该触发适应
        
        子类可以重写此方法实现自定义的触发逻辑
        默认总是返回True
        """
        return True
    
    def _perform_adaptation(self, episode_data: Dict[str, Any]):
        """
        执行适应更新
        
        子类必须实现此方法
        """
        raise NotImplementedError("子类必须实现 _perform_adaptation 方法")
    
    def _print_episode_progress(self, episode: int, total_episodes: int, episode_data: Dict[str, Any]):
        """打印episode进度"""
        algo_name = self._get_algorithm_name()
        print(f"Episode {episode}/{total_episodes} ({algo_name}): "
              f"Reward: {episode_data['episode_reward']:.2f}, "
              f"Length: {episode_data['episode_length']}")
    
    def _compute_summary(self, adaptation_data: List[Dict]) -> Dict[str, Any]:
        """计算汇总统计"""
        summary = {
            'mean_reward': np.mean([d['episode_reward'] for d in adaptation_data]),
            'std_reward': np.std([d['episode_reward'] for d in adaptation_data]),
            'max_reward': max([d['episode_reward'] for d in adaptation_data]),
            'min_reward': min([d['episode_reward'] for d in adaptation_data]),
            'mean_length': np.mean([d['episode_length'] for d in adaptation_data]),
            'adaptation_steps': self.adaptation_step,
            'cache_size': len(self.state_cache)
        }
        
        # 添加子类的额外摘要
        additional_summary = self._get_additional_summary(adaptation_data)
        summary.update(additional_summary)
        
        return summary
    
    def _get_additional_summary(self, adaptation_data: List[Dict]) -> Dict[str, Any]:
        """
        获取额外的摘要信息
        
        子类可以重写此方法添加特定的统计信息
        """
        return {}
    
    def _get_algorithm_name(self) -> str:
        """返回算法名称"""
        return "BaseTTAAlgorithm"
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取算法统计信息"""
        return {
            'algorithm_name': self._get_algorithm_name(),
            'adaptation_steps': self.adaptation_step,
            'cache_size': len(self.state_cache),
            'trainable_params': len(self.trainable_params),
            'param_names': self.param_names[:5],
            'learning_rate': self.learning_rate,
            'adaptation_mode': self.adaptation_mode
        }
    
    def save_checkpoint(self, save_path: str):
        """保存检查点"""
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
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.offline_policy_state = checkpoint['offline_policy_state']
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.adaptation_step = checkpoint['adaptation_step']
        self.config = checkpoint.get('config', {})
        
        print(f"Checkpoint loaded from {checkpoint_path}")
