import d4rl

import gym
import numpy as np
from typing import Dict, Any, Optional, List


class ShiftedMujocoEnvWrapper(gym.Wrapper):
    """
    MUJOCO环境包装器，支持人工shift设置
    
    支持多种shift类型：
    - 动态参数shift（质量、摩擦系数等）
    - 观测空间shift
    - 奖励函数shift
    - 状态转移shift
    """
    
    def __init__(self, env, shift_config: Optional[Dict[str, Any]] = None):
        super().__init__(env)
        self.shift_config = shift_config or {}
        self._setup_shift_parameters()
        self._original_dynamics = self._capture_original_dynamics()
        
    def _setup_shift_parameters(self):
        """设置shift参数"""
        # 动态参数shift配置
        self.dynamic_shifts = self.shift_config.get('dynamic_shifts', {})
        # 观测空间shift配置
        self.observation_shifts = self.shift_config.get('observation_shifts', {})
        # 奖励函数shift配置
        self.reward_shifts = self.shift_config.get('reward_shifts', {})
        # 状态转移shift配置
        self.transition_shifts = self.shift_config.get('transition_shifts', {})
        
    def _capture_original_dynamics(self):
        """捕获原始动态参数"""
        if hasattr(self.env, 'model'):
            original_dynamics = {}
            # 保存原始质量
            original_dynamics['body_masses'] = self.env.model.body_mass.copy()
            # 保存原始摩擦系数
            original_dynamics['geom_frictions'] = self.env.model.geom_friction.copy()
            return original_dynamics
        return {}
    
    def _scale_body_masses(self, scale_factor: float):
        """缩放身体质量"""
        if hasattr(self.env, 'model') and self._original_dynamics:
            original_masses = self._original_dynamics['body_masses']
            self.env.model.body_mass[:] = original_masses * scale_factor
    
    def _scale_frictions(self, scale_factor: float):
        """缩放摩擦系数"""
        if hasattr(self.env, 'model') and self._original_dynamics:
            original_frictions = self._original_dynamics['geom_frictions']
            self.env.model.geom_friction[:] = original_frictions * scale_factor
    
    def _apply_observation_shift(self, observation: np.ndarray) -> np.ndarray:
        """应用观测空间shift"""
        if 'noise_std' in self.observation_shifts:
            noise_std = self.observation_shifts['noise_std']
            observation = observation + np.random.normal(0, noise_std, observation.shape)
            
        if 'bias' in self.observation_shifts:
            bias = self.observation_shifts['bias']
            observation = observation + bias
            
        if 'scale' in self.observation_shifts:
            scale = self.observation_shifts['scale']
            observation = observation * scale
            
        return observation
    
    def _apply_reward_shift(self, reward: float, observation: np.ndarray, action: np.ndarray) -> float:
        """应用奖励函数shift"""
        shifted_reward = reward
        
        if 'scale' in self.reward_shifts:
            shifted_reward = shifted_reward * self.reward_shifts['scale']
            
        if 'bias' in self.reward_shifts:
            shifted_reward = shifted_reward + self.reward_shifts['bias']
            
        if 'noise_std' in self.reward_shifts:
            noise_std = self.reward_shifts['noise_std']
            shifted_reward = shifted_reward + np.random.normal(0, noise_std)
            
        return shifted_reward
    
    def apply_dynamic_shift(self):
        """应用动态参数shift"""
        if 'mass_scale' in self.dynamic_shifts:
            self._scale_body_masses(self.dynamic_shifts['mass_scale'])
            
        if 'friction_scale' in self.dynamic_shifts:
            self._scale_frictions(self.dynamic_shifts['friction_scale'])
            
    def reset(self, **kwargs):
        """重置环境并应用shift"""
        obs = self.env.reset(**kwargs)
        self.apply_dynamic_shift()
        return self._apply_observation_shift(obs)
    
    def step(self, action):
        """执行一步并应用所有shift"""
        obs, reward, done, info = self.env.step(action)
        
        # 应用观测shift
        shifted_obs = self._apply_observation_shift(obs)
        
        # 应用奖励shift
        shifted_reward = self._apply_reward_shift(reward, obs, action)
        
        # 记录shift信息
        info['original_observation'] = obs
        info['original_reward'] = reward
        info['shift_config'] = self.shift_config
        
        return shifted_obs, shifted_reward, done, info
    
    def get_shift_info(self) -> Dict[str, Any]:
        """获取当前shift配置信息"""
        return {
            'dynamic_shifts': self.dynamic_shifts,
            'observation_shifts': self.observation_shifts,
            'reward_shifts': self.reward_shifts,
            'transition_shifts': self.transition_shifts
        }
    
    def set_shift_config(self, new_config: Dict[str, Any]):
        """动态更新shift配置"""
        self.shift_config.update(new_config)
        self._setup_shift_parameters()


def create_shifted_env(env_name: str, shift_config: Dict[str, Any]) -> ShiftedMujocoEnvWrapper:
    """
    创建shifted环境的便捷函数
    
    Args:
        env_name: 基础环境名称（D4RL格式，如 "hopper-medium-v2"）
        shift_config: shift配置字典
        
    Returns:
        ShiftedMujocoEnvWrapper实例
    """
    base_env = gym.make(env_name)
    return ShiftedMujocoEnvWrapper(base_env, shift_config)


def create_shift_intensity_configs(base_shift_type: str = 'mass', 
                                   num_levels: int = 4) -> Dict[str, Dict[str, Any]]:
    """
    创建不同强度的shift配置（4档强度）
    
    Args:
        base_shift_type: shift类型 ('mass', 'friction', 'observation', 'reward')
        num_levels: 强度档位数量
        
    Returns:
        包含不同强度配置的字典
    """
    shift_configs = {}
    
    if base_shift_type == 'mass':
        # 质量shift的4档强度
        scales = [0.5, 0.75, 1.25, 1.5]  # 从轻到重
        for i, scale in enumerate(scales, 1):
            shift_configs[f'mass_level_{i}'] = {
                'dynamic_shifts': {'mass_scale': scale},
                'level': i,
                'scale': scale
            }
    
    elif base_shift_type == 'friction':
        # 摩擦系数shift的4档强度
        scales = [0.3, 0.6, 1.4, 1.7]  # 从低到高
        for i, scale in enumerate(scales, 1):
            shift_configs[f'friction_level_{i}'] = {
                'dynamic_shifts': {'friction_scale': scale},
                'level': i,
                'scale': scale
            }
    
    elif base_shift_type == 'observation':
        # 观测噪声shift的4档强度
        noise_stds = [0.05, 0.1, 0.2, 0.3]  # 从低到高
        for i, noise_std in enumerate(noise_stds, 1):
            shift_configs[f'observation_noise_level_{i}'] = {
                'observation_shifts': {'noise_std': noise_std},
                'level': i,
                'noise_std': noise_std
            }
    
    elif base_shift_type == 'reward':
        # 奖励scale shift的4档强度
        scales = [0.5, 0.75, 1.25, 1.5]  # 从低到高
        for i, scale in enumerate(scales, 1):
            shift_configs[f'reward_scale_level_{i}'] = {
                'reward_shifts': {'scale': scale},
                'level': i,
                'scale': scale
            }
    
    elif base_shift_type == 'combined':
        # 组合shift的4档强度
        mass_scales = [0.6, 0.8, 1.2, 1.4]
        friction_scales = [0.5, 0.7, 1.3, 1.5]
        for i in range(1, num_levels + 1):
            shift_configs[f'combined_level_{i}'] = {
                'dynamic_shifts': {
                    'mass_scale': mass_scales[i-1],
                    'friction_scale': friction_scales[i-1]
                },
                'level': i,
                'mass_scale': mass_scales[i-1],
                'friction_scale': friction_scales[i-1]
            }
    
    return shift_configs


def get_all_shift_types() -> List[str]:
    """获取所有支持的shift类型"""
    return ['mass', 'friction', 'observation', 'reward', 'combined']


def create_custom_mass_shift_configs(mass_scales: List[float]) -> Dict[str, Dict[str, Any]]:
    """
    创建自定义质量shift配置
    
    Args:
        mass_scales: 质量缩放因子列表
        
    Returns:
        包含不同质量shift配置的字典
    """
    shift_configs = {}
    
    for i, scale in enumerate(mass_scales):
        if scale == 1.0:
            shift_name = 'no_shift'
            shift_label = 'no shift'
        elif scale < 1.0:
            shift_name = f'light_mass_{scale:.2f}'
            if scale >= 0.9:
                shift_label = 'mild light'
            elif scale >= 0.8:
                shift_label = 'moderate light'
            elif scale >= 0.7:
                shift_label = 'strong light'
            else:
                shift_label = 'extreme light'
        else:
            shift_name = f'heavy_mass_{scale:.2f}'
            if scale <= 1.1:
                shift_label = 'mild heavy'
            elif scale <= 1.25:
                shift_label = 'moderate heavy'
            elif scale <= 1.5:
                shift_label = 'strong heavy'
            else:
                shift_label = 'extreme heavy'
        
        shift_configs[shift_name] = {
            'dynamic_shifts': {'mass_scale': scale},
            'shift_type': 'mass',
            'mass_scale': scale,
            'shift_label': shift_label
        }
    
    return shift_configs