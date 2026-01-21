import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from collections import deque

from offlinerlkit.policy.base_policy import BasePolicy


class MetricsTracker:
    """性能指标跟踪器"""
    
    def __init__(self, window_size: int = 100, collapse_threshold: float = -100.0, 
                 collapse_window: int = 5):
        self.window_size = window_size
        self.collapse_threshold = collapse_threshold
        self.collapse_window = collapse_window
        
        self.episode_rewards = deque(maxlen=window_size)
        self.episode_lengths = deque(maxlen=window_size)
        self.adaptation_losses = deque(maxlen=window_size)
        self.policy_divergences = deque(maxlen=window_size)
        
        # Collapse tracking
        self.low_reward_count = 0
        self.collapse_detected = False
        self.collapse_episode = -1
        
    def update(self, episode_data: Dict[str, Any]):
        """更新指标"""
        if 'episode_reward' in episode_data:
            self.episode_rewards.append(episode_data['episode_reward'])
            
            # Check for collapse
            if episode_data['episode_reward'] < self.collapse_threshold:
                self.low_reward_count += 1
                if self.low_reward_count >= self.collapse_window and not self.collapse_detected:
                    self.collapse_detected = True
                    self.collapse_episode = len(self.episode_rewards)
            else:
                self.low_reward_count = 0
                
        if 'episode_length' in episode_data:
            self.episode_lengths.append(episode_data['episode_length'])
        if 'adaptation_loss' in episode_data:
            self.adaptation_losses.append(episode_data['adaptation_loss'])
        if 'policy_divergence' in episode_data:
            self.policy_divergences.append(episode_data['policy_divergence'])
            
    def get_summary(self) -> Dict[str, float]:
        """获取指标摘要"""
        summary = {}
        
        if self.episode_rewards:
            rewards = list(self.episode_rewards)
            summary['mean_reward'] = np.mean(rewards)
            summary['std_reward'] = np.std(rewards)
            summary['max_reward'] = max(rewards)
            summary['min_reward'] = min(rewards)
            summary['worst_case_return'] = min(rewards)  # Worst case return
            summary['median_reward'] = np.median(rewards)
            
        if self.episode_lengths:
            summary['mean_length'] = np.mean(list(self.episode_lengths))
            
        if self.adaptation_losses:
            summary['mean_loss'] = np.mean(list(self.adaptation_losses))
            
        if self.policy_divergences:
            divergences = list(self.policy_divergences)
            summary['mean_policy_divergence'] = np.mean([d.get('mean_norm', 0) for d in divergences])
            summary['max_policy_divergence'] = np.max([d.get('mean_norm', 0) for d in divergences])
            
        # Collapse metrics
        summary['collapse_detected'] = self.collapse_detected
        summary['collapse_rate'] = 1.0 if self.collapse_detected else 0.0
        summary['collapse_episode'] = self.collapse_episode if self.collapse_detected else -1
            
        return summary


class TTAManager:
    """
    Test-Time Adaptation管理器
    
    支持多种适应策略：
    - 在线微调 (online_finetune)
    - 元学习适应 (meta_learning)
    - 基于经验的适应 (experience_based)
    - 熵最小化 (entropy_minimization)
    - 不确定性最小化 (uncertainty_minimization)
    """
    
    def __init__(self, policy: BasePolicy, env, adaptation_config: Optional[Dict[str, Any]] = None):
        self.policy = policy
        self.env = env
        self.adaptation_config = adaptation_config or {}
        
        # 获取策略的设备
        self.device = self._get_policy_device()
        
        # 适应策略配置
        self.enable_tta = self.adaptation_config.get('enable_tta', True)
        self.adaptation_strategy = self.adaptation_config.get('strategy', 'online_finetune')
        self.adaptation_steps = self.adaptation_config.get('steps', 1000)
        self.batch_size = self.adaptation_config.get('batch_size', 32)
        self.learning_rate = self.adaptation_config.get('learning_rate', 1e-5)
        
        # 保存初始策略参数用于计算divergence
        self.initial_policy_state = self._save_policy_state()
        
        # 经验回放缓冲区
        self.experience_buffer = deque(maxlen=10000)
        
        # 指标跟踪器（支持collapse检测）
        collapse_threshold = self.adaptation_config.get('collapse_threshold', -100.0)
        collapse_window = self.adaptation_config.get('collapse_window', 5)
        self.metrics_tracker = MetricsTracker(collapse_threshold=collapse_threshold, 
                                               collapse_window=collapse_window)
        
        # 适应状态
        self.adaptation_step = 0
        self.best_performance = -float('inf')
    
    def _get_policy_device(self) -> torch.device:
        """获取策略的设备"""
        # 尝试从 actor 获取设备
        if hasattr(self.policy, 'actor') and hasattr(self.policy.actor, 'device'):
            return self.policy.actor.device
        # 尝试从 critic1 获取设备
        elif hasattr(self.policy, 'critic1') and hasattr(self.policy.critic1, 'device'):
            return self.policy.critic1.device
        # 尝试从第一个参数获取设备
        elif len(list(self.policy.parameters())) > 0:
            return next(self.policy.parameters()).device
        else:
            # 默认返回 CPU
            return torch.device('cpu')
    
    def _save_policy_state(self) -> Dict[str, torch.Tensor]:
        """保存当前策略状态"""
        state = {}
        for name, param in self.policy.named_parameters():
            state[name] = param.data.clone().detach()
        return state
    
    def _compute_policy_divergence(self) -> Dict[str, float]:
        """计算策略divergence（参数漂移范数）"""
        divergence = {}
        total_diff = 0.0
        param_count = 0
        
        for name, param in self.policy.named_parameters():
            if name in self.initial_policy_state:
                diff = torch.norm(param.data - self.initial_policy_state[name]).item()
                divergence[name] = diff
                total_diff += diff
                param_count += 1
        
        divergence['total_norm'] = total_diff
        divergence['mean_norm'] = total_diff / max(param_count, 1)
        return divergence
        
    def run_adaptation(self, num_episodes: int = 10) -> Tuple[List[Dict], Dict[str, float]]:
        """
        执行Test-Time Adaptation
        
        Args:
            num_episodes: 适应回合数
            
        Returns:
            adaptation_data: 适应过程数据
            metrics_summary: 性能指标摘要
        """
        adaptation_data = []
        
        for episode in range(num_episodes):
            episode_data = self._run_single_episode()
            adaptation_data.append(episode_data)
            
            # 在线适应（仅在TTA启用时）
            if self.enable_tta:
                if self.adaptation_strategy == 'online_finetune':
                    self._online_finetune(episode_data)
                elif self.adaptation_strategy == 'meta_learning':
                    self._meta_adapt(episode_data)
                elif self.adaptation_strategy == 'experience_based':
                    self._experience_based_adapt()
                elif self.adaptation_strategy == 'entropy_minimization':
                    self._entropy_minimization(episode_data)
                elif self.adaptation_strategy == 'uncertainty_minimization':
                    self._uncertainty_minimization(episode_data)
                
            # 更新指标
            self.metrics_tracker.update(episode_data)
            
            # 计算策略divergence
            episode_data['policy_divergence'] = self._compute_policy_divergence()
            
            # 记录适应进度
            self.adaptation_step += 1
            
            tta_status = "ON" if self.enable_tta else "OFF"
            print(f"Episode {episode + 1}/{num_episodes} (TTA: {tta_status}): "
                  f"Reward: {episode_data['episode_reward']:.2f}, "
                  f"Length: {episode_data['episode_length']}, "
                  f"Policy Divergence: {episode_data['policy_divergence']['mean_norm']:.6f}")
                
        return adaptation_data, self.metrics_tracker.get_summary()
    
    def _run_single_episode(self) -> Dict[str, Any]:
        """运行单个回合并收集数据"""
        # 处理新版本的 gym reset 返回值
        reset_result = self.env.reset()
        if isinstance(reset_result, tuple):
            obs, _ = reset_result  # 新版本的 gym 返回 (obs, info)
        else:
            obs = reset_result  # 旧版本的 gym 只返回 obs
            
        episode_reward = 0
        episode_length = 0
        episode_transitions = []
        
        done = False
        while not done:
            # 选择动作 - 使用 select_action 方法
            with torch.no_grad():
                action = self.policy.select_action(obs)
            
            # 执行动作 - 处理新版本的 gym step 返回值
            step_result = self.env.step(action)
            if len(step_result) == 5:  # 新版本: obs, reward, terminated, truncated, info
                next_obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:  # 旧版本: obs, reward, done, info
                next_obs, reward, done, info = step_result
            
            # 存储转移
            transition = {
                'obs': obs.copy(),
                'action': action.copy(),
                'reward': reward,
                'next_obs': next_obs.copy(),
                'done': done
            }
            episode_transitions.append(transition)
            
            # 更新状态
            obs = next_obs
            episode_reward += reward
            episode_length += 1
            
            # 添加到经验缓冲区
            self.experience_buffer.append(transition)
            
            if episode_length >= 1000:  # 防止无限循环
                break
                
        return {
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            'transitions': episode_transitions,
            'adaptation_step': self.adaptation_step
        }
    
    def _online_finetune(self, episode_data: Dict[str, Any]):
        """在线微调策略"""
        if len(self.experience_buffer) < self.batch_size:
            return
            
        # 采样小批量数据
        batch_indices = np.random.choice(len(self.experience_buffer), 
                                        size=min(self.batch_size, len(self.experience_buffer)), 
                                        replace=False)
        batch = [self.experience_buffer[i] for i in batch_indices]
        
        # 准备训练数据 - 优化张量创建
        obs_data = np.array([t['obs'] for t in batch])
        action_data = np.array([t['action'] for t in batch])
        reward_data = np.array([t['reward'] for t in batch])
        next_obs_data = np.array([t['next_obs'] for t in batch])
        done_data = np.array([t['done'] for t in batch])
        
        obs_batch = torch.FloatTensor(obs_data).to(self.device)
        action_batch = torch.FloatTensor(action_data).to(self.device)
        reward_batch = torch.FloatTensor(reward_data).to(self.device)
        next_obs_batch = torch.FloatTensor(next_obs_data).to(self.device)
        done_batch = torch.BoolTensor(done_data).to(self.device)
        
        # 计算损失（这里需要根据具体策略类调整）
        if hasattr(self.policy, 'adaptation_optimizer'):
            self.policy.adaptation_optimizer.zero_grad()
            
            # 计算策略损失（示例实现）
            loss = self._compute_adaptation_loss(obs_batch, action_batch, reward_batch, 
                                               next_obs_batch, done_batch)
            
            loss.backward()
            self.policy.adaptation_optimizer.step()
            
            episode_data['adaptation_loss'] = loss.item()
    
    def _compute_adaptation_loss(self, obs, action, reward, next_obs, done):
        """计算适应损失（需要根据具体策略实现）"""
        # 这里是一个示例实现，需要根据具体策略类调整
        
        # 如果是actor-critic策略，可以计算TD误差
        if hasattr(self.policy, 'critic1') and hasattr(self.policy, 'critic2'):
            with torch.no_grad():
                # 计算目标Q值 - 修复：从分布中获取动作张量
                next_action_dist = self.policy.actor(next_obs)
                next_actions = next_action_dist.mode()[0]  # 获取动作张量，而不是分布对象
                next_q1 = self.policy.critic1(next_obs, next_actions)
                next_q2 = self.policy.critic2(next_obs, next_actions)
                next_q = torch.min(next_q1, next_q2)
                target_q = reward + (1 - done.float()) * 0.99 * next_q
            
            # 计算当前Q值
            current_q1 = self.policy.critic1(obs, action)
            current_q2 = self.policy.critic2(obs, action)
            
            # TD误差损失
            td_loss = torch.nn.functional.mse_loss(current_q1, target_q) + \
                     torch.nn.functional.mse_loss(current_q2, target_q)
            
            return td_loss
        
        # 默认返回零损失
        return torch.tensor(0.0, requires_grad=True)
    
    def _meta_adapt(self, episode_data: Dict[str, Any]):
        """元学习适应策略"""
        # 这里可以实现基于元学习的适应策略
        # 例如：MAML、Reptile等
        pass
    
    def _experience_based_adapt(self):
        """基于经验的适应"""
        # 定期使用经验缓冲区进行批量训练
        if len(self.experience_buffer) >= 1000 and self.adaptation_step % 10 == 0:
            self._batch_training()
    
    def _batch_training(self):
        """批量训练"""
        if len(self.experience_buffer) < self.batch_size:
            return
            
        # 采样批量数据
        batch_size = min(len(self.experience_buffer), self.batch_size * 10)
        batch_indices = np.random.choice(len(self.experience_buffer), batch_size, replace=False)
        batch = [self.experience_buffer[i] for i in batch_indices]
        
        # 准备训练数据 - 优化张量创建
        obs_data = np.array([t['obs'] for t in batch])
        action_data = np.array([t['action'] for t in batch])
        reward_data = np.array([t['reward'] for t in batch])
        next_obs_data = np.array([t['next_obs'] for t in batch])
        done_data = np.array([t['done'] for t in batch])
        
        obs_batch = torch.FloatTensor(obs_data).to(self.device)
        action_batch = torch.FloatTensor(action_data).to(self.device)
        reward_batch = torch.FloatTensor(reward_data).to(self.device)
        next_obs_batch = torch.FloatTensor(next_obs_data).to(self.device)
        done_batch = torch.BoolTensor(done_data).to(self.device)
        
        # 多步训练
        num_steps = 10
        for step in range(num_steps):
            if hasattr(self.policy, 'adaptation_optimizer'):
                self.policy.adaptation_optimizer.zero_grad()
                loss = self._compute_adaptation_loss(obs_batch, action_batch, reward_batch, 
                                                   next_obs_batch, done_batch)
                loss.backward()
                self.policy.adaptation_optimizer.step()
    
    def _entropy_minimization(self, episode_data: Dict[str, Any]):
        """熵最小化策略 - 最简单的proxy更新"""
        if len(self.experience_buffer) < self.batch_size:
            return
            
        # 采样小批量数据
        batch_indices = np.random.choice(len(self.experience_buffer), 
                                        size=min(self.batch_size, len(self.experience_buffer)), 
                                        replace=False)
        batch = [self.experience_buffer[i] for i in batch_indices]
        
        # 准备训练数据
        obs_data = np.array([t['obs'] for t in batch])
        obs_batch = torch.FloatTensor(obs_data).to(self.device)
        
        if hasattr(self.policy, 'adaptation_optimizer'):
            self.policy.adaptation_optimizer.zero_grad()
            
            # 计算策略熵并最小化
            if hasattr(self.policy, 'actor'):
                action_dist = self.policy.actor(obs_batch)
                
                # 计算熵（负熵作为损失）
                if hasattr(action_dist, 'entropy'):
                    entropy = action_dist.entropy().mean()
                    loss = -entropy  # 最小化熵 = 最大化负熵
                else:
                    # 如果没有熵方法，使用动作方差作为代理
                    actions = self.policy.select_action(obs_batch)
                    loss = torch.var(actions)
                
                loss.backward()
                self.policy.adaptation_optimizer.step()
                
                episode_data['adaptation_loss'] = loss.item()
    
    def _uncertainty_minimization(self, episode_data: Dict[str, Any]):
        """不确定性最小化策略 - 最简单的proxy更新"""
        if len(self.experience_buffer) < self.batch_size:
            return
            
        # 采样小批量数据
        batch_indices = np.random.choice(len(self.experience_buffer), 
                                        size=min(self.batch_size, len(self.experience_buffer)), 
                                        replace=False)
        batch = [self.experience_buffer[i] for i in batch_indices]
        
        # 准备训练数据
        obs_data = np.array([t['obs'] for t in batch])
        action_data = np.array([t['action'] for t in batch])
        obs_batch = torch.FloatTensor(obs_data).to(self.device)
        action_batch = torch.FloatTensor(action_data).to(self.device)
        
        if hasattr(self.policy, 'adaptation_optimizer'):
            self.policy.adaptation_optimizer.zero_grad()
            
            # 计算Q值的不确定性（两个critic的差异）
            if hasattr(self.policy, 'critic1') and hasattr(self.policy, 'critic2'):
                with torch.no_grad():
                    actions = self.policy.select_action(obs_batch)
                
                q1_values = self.policy.critic1(obs_batch, actions)
                q2_values = self.policy.critic2(obs_batch, actions)
                
                # 使用两个critic的绝对差异作为不确定性
                uncertainty = torch.abs(q1_values - q2_values).mean()
                loss = uncertainty  # 最小化不确定性
                
                loss.backward()
                self.policy.adaptation_optimizer.step()
                
                episode_data['adaptation_loss'] = loss.item()
    
    def evaluate_performance(self, num_episodes: int = 5) -> Dict[str, float]:
        """评估当前策略性能"""
        rewards = []
        lengths = []
        
        for _ in range(num_episodes):
            # 处理新版本的 gym reset 返回值
            reset_result = self.env.reset()
            if isinstance(reset_result, tuple):
                obs, _ = reset_result  # 新版本的 gym 返回 (obs, info)
            else:
                obs = reset_result  # 旧版本的 gym 只返回 obs
                
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done:
                with torch.no_grad():
                    # 使用 select_action 方法
                    action = self.policy.select_action(obs, deterministic=True)
                
                # 处理新版本的 gym step 返回值
                step_result = self.env.step(action)
                if len(step_result) == 5:  # 新版本: obs, reward, terminated, truncated, info
                    obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:  # 旧版本: obs, reward, done, info
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
    
    def save_adaptation_checkpoint(self, save_path: str):
        """保存适应检查点"""
        checkpoint = {
            'policy_state_dict': self.policy.state_dict(),
            'adaptation_step': self.adaptation_step,
            'metrics': self.metrics_tracker.get_summary(),
            'adaptation_config': self.adaptation_config
        }
        
        if hasattr(self.policy, 'adaptation_optimizer'):
            checkpoint['optimizer_state_dict'] = self.policy.adaptation_optimizer.state_dict()
            
        torch.save(checkpoint, save_path)
    
    def load_adaptation_checkpoint(self, checkpoint_path: str):
        """加载适应检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.adaptation_step = checkpoint['adaptation_step']
        
        if hasattr(self.policy, 'adaptation_optimizer') and 'optimizer_state_dict' in checkpoint:
            self.policy.adaptation_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])