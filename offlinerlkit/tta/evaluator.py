import gym
import numpy as np
import torch
from typing import Dict, Any, List, Optional
import json
import os
from datetime import datetime

from offlinerlkit.tta.shifted_env import ShiftedMujocoEnvWrapper, create_shifted_env
from offlinerlkit.tta.model_loader import ModelLoader
from offlinerlkit.tta.tta_manager import TTAManager


class ShiftedPolicyEvaluator:
    """
    支持shift的完整评估框架
    
    功能：
    - 多环境shift配置管理
    - 多种适应策略评估
    - 性能指标记录和可视化
    - 结果保存和报告生成
    """
    
    def __init__(self, base_env_name: str, shift_configs: Dict[str, Dict], 
                 adaptation_configs: Dict[str, Dict], results_dir: str = './tta_results'):
        self.base_env_name = base_env_name
        self.shift_configs = shift_configs
        self.adaptation_configs = adaptation_configs
        self.results_dir = results_dir
        
        # 创建结果目录
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 评估结果存储
        self.evaluation_results = {}
        
    def evaluate_policy(self, policy_checkpoint: str, policy_class, 
                       evaluation_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        完整的策略评估流程
        
        Args:
            policy_checkpoint: 策略检查点路径
            policy_class: 策略类
            evaluation_config: 评估配置
            
        Returns:
            完整的评估结果
        """
        evaluation_config = evaluation_config or {}
        results = {}
        
        # 基础环境评估（无shift）
        print("Evaluating on base environment...")
        base_results = self._evaluate_single_config(
            policy_checkpoint, policy_class, {}, {}, 'base', evaluation_config
        )
        results['base'] = base_results
        
        # 各种shift配置评估
        for shift_name, shift_config in self.shift_configs.items():
            print(f"Evaluating on shift: {shift_name}")
            
            for adaptation_name, adaptation_config in self.adaptation_configs.items():
                print(f"  With adaptation: {adaptation_name}")
                
                config_key = f"{shift_name}_{adaptation_name}"
                
                shift_results = self._evaluate_single_config(
                    policy_checkpoint, policy_class, shift_config, adaptation_config, 
                    config_key, evaluation_config
                )
                
                results[config_key] = shift_results
                
        # 保存完整结果
        self._save_evaluation_results(results, evaluation_config)
        
        return results
    
    def _evaluate_single_config(self, policy_checkpoint: str, policy_class, 
                              shift_config: Dict[str, Any], adaptation_config: Dict[str, Any],
                              config_key: str, evaluation_config: Dict[str, Any]) -> Dict[str, Any]:
        """评估单个配置"""
        
        # 创建shift环境
        env = create_shifted_env(self.base_env_name, shift_config)
        
        # 获取环境配置
        env_config = {
            'observation_space': env.observation_space,
            'action_space': env.action_space,
            'obs_dim': env.observation_space.shape[0],
            'action_dim': env.action_space.shape[0]
        }
        
        # 加载模型
        model_loader = ModelLoader(policy_class, device=evaluation_config.get('device', 'cuda'))
        policy = model_loader.load_pretrained_model(policy_checkpoint, env_config)
        
        # 验证模型兼容性
        if not model_loader.validate_model_compatibility(policy, env_config):
            print(f"Warning: Model compatibility issue for {config_key}")
            
        # 适配模型到shift环境
        adapted_policy = model_loader.adapt_model_to_shift(
            policy, shift_config, 
            adaptation_config.get('strategy', 'freeze_actor')
        )
        
        # 初始性能评估（适应前）
        initial_performance = self._evaluate_initial_performance(adapted_policy, env, evaluation_config)
        
        # 执行Test-Time Adaptation
        tta_manager = TTAManager(adapted_policy, env, adaptation_config)
        adaptation_data, adaptation_metrics = tta_manager.run_adaptation(
            num_episodes=evaluation_config.get('adaptation_episodes', 10)
        )
        
        # 最终性能评估（适应后）
        final_performance = tta_manager.evaluate_performance(
            num_episodes=evaluation_config.get('evaluation_episodes', 5)
        )
        
        # 收集结果
        results = {
            'config_key': config_key,
            'shift_config': shift_config,
            'adaptation_config': adaptation_config,
            'initial_performance': initial_performance,
            'final_performance': final_performance,
            'adaptation_metrics': adaptation_metrics,
            'adaptation_data': adaptation_data,
            'performance_improvement': {
                'reward_improvement': final_performance['mean_reward'] - initial_performance['mean_reward'],
                'improvement_ratio': final_performance['mean_reward'] / max(initial_performance['mean_reward'], 1e-6)
            }
        }
        
        return results
    
    def _evaluate_initial_performance(self, policy, env, evaluation_config: Dict[str, Any]) -> Dict[str, float]:
        """评估初始性能（适应前）"""
        num_episodes = evaluation_config.get('initial_evaluation_episodes', 3)
        rewards = []
        lengths = []
        
        for _ in range(num_episodes):
            # 处理新版本的 gym reset 返回值
            reset_result = env.reset()
            if isinstance(reset_result, tuple):
                obs, _ = reset_result  # 新版本的 gym 返回 (obs, info)
            else:
                obs = reset_result  # 旧版本的 gym 只返回 obs
                
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done:
                with torch.no_grad():
                    # 使用 select_action 方法而不是直接调用 policy
                    action = policy.select_action(obs, deterministic=True)
                
                # 处理新版本的 gym step 返回值
                step_result = env.step(action)
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
    
    def _save_evaluation_results(self, results: Dict[str, Any], evaluation_config: Dict[str, Any]):
        """保存评估结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存详细结果
        results_filename = f"tta_results_{timestamp}.json"
        results_path = os.path.join(self.results_dir, results_filename)
        
        # 转换numpy类型为Python原生类型
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_numpy_types(results)
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        # 生成摘要报告
        summary = self._generate_summary_report(results, evaluation_config)
        summary_filename = f"tta_summary_{timestamp}.json"
        summary_path = os.path.join(self.results_dir, summary_filename)
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Results saved to: {results_path}")
        print(f"Summary saved to: {summary_path}")
        
        # 保存配置信息
        config_info = {
            'base_env_name': self.base_env_name,
            'shift_configs': self.shift_configs,
            'adaptation_configs': self.adaptation_configs,
            'evaluation_config': evaluation_config,
            'timestamp': timestamp
        }
        
        config_filename = f"tta_config_{timestamp}.json"
        config_path = os.path.join(self.results_dir, config_filename)
        
        with open(config_path, 'w') as f:
            json.dump(config_info, f, indent=2)
    
    def _generate_summary_report(self, results: Dict[str, Any], evaluation_config: Dict[str, Any]) -> Dict[str, Any]:
        """生成摘要报告"""
        summary = {
            'base_performance': results.get('base', {}).get('final_performance', {}),
            'shift_comparisons': {},
            'adaptation_effectiveness': {},
            'overall_statistics': {}
        }
        
        base_reward = summary['base_performance'].get('mean_reward', 0)
        
        # 比较不同shift配置
        for config_key, config_results in results.items():
            if config_key == 'base':
                continue
                
            final_reward = config_results['final_performance']['mean_reward']
            improvement = config_results['performance_improvement']['reward_improvement']
            
            summary['shift_comparisons'][config_key] = {
                'final_reward': final_reward,
                'improvement_over_initial': improvement,
                'relative_to_base': final_reward / max(base_reward, 1e-6)
            }
            
        # 计算适应效果统计
        adaptation_effects = []
        for config_key, config_results in results.items():
            if config_key != 'base':
                effect = config_results['performance_improvement']['reward_improvement']
                adaptation_effects.append(effect)
                
        if adaptation_effects:
            summary['overall_statistics'] = {
                'mean_improvement': np.mean(adaptation_effects),
                'std_improvement': np.std(adaptation_effects),
                'max_improvement': max(adaptation_effects),
                'min_improvement': min(adaptation_effects)
            }
            
        return summary
    
    def compare_strategies(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """比较不同适应策略的效果"""
        strategy_comparison = {}
        
        for config_key, config_results in results.items():
            if config_key == 'base':
                continue
                
            # 提取策略名称
            parts = config_key.split('_')
            if len(parts) >= 2:
                shift_name = parts[0]
                adaptation_name = '_'.join(parts[1:])
                
                if adaptation_name not in strategy_comparison:
                    strategy_comparison[adaptation_name] = []
                    
                strategy_comparison[adaptation_name].append({
                    'shift': shift_name,
                    'improvement': config_results['performance_improvement']['reward_improvement'],
                    'final_reward': config_results['final_performance']['mean_reward']
                })
                
        # 计算每个策略的平均效果
        strategy_stats = {}
        for strategy, data in strategy_comparison.items():
            improvements = [item['improvement'] for item in data]
            final_rewards = [item['final_reward'] for item in data]
            
            strategy_stats[strategy] = {
                'mean_improvement': np.mean(improvements),
                'std_improvement': np.std(improvements),
                'mean_final_reward': np.mean(final_rewards),
                'num_configs': len(data)
            }
            
        return strategy_stats
    
    def plot_results(self, results: Dict[str, Any], save_plots: bool = True):
        """绘制结果图表（需要matplotlib）"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # 设置样式
            sns.set_style("whitegrid")
            plt.figure(figsize=(12, 8))
            
            # 准备数据
            configs = []
            initial_rewards = []
            final_rewards = []
            
            for config_key, config_results in results.items():
                if 'initial_performance' in config_results and 'final_performance' in config_results:
                    configs.append(config_key)
                    initial_rewards.append(config_results['initial_performance']['mean_reward'])
                    final_rewards.append(config_results['final_performance']['mean_reward'])
                    
            # 创建条形图
            x = np.arange(len(configs))
            width = 0.35
            
            plt.bar(x - width/2, initial_rewards, width, label='Initial', alpha=0.7)
            plt.bar(x + width/2, final_rewards, width, label='After TTA', alpha=0.7)
            
            plt.xlabel('Configuration')
            plt.ylabel('Mean Reward')
            plt.title('Test-Time Adaptation Performance Comparison')
            plt.xticks(x, configs, rotation=45, ha='right')
            plt.legend()
            plt.tight_layout()
            
            if save_plots:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                plot_path = os.path.join(self.results_dir, f"tta_comparison_{timestamp}.png")
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                print(f"Plot saved to: {plot_path}")
                
            plt.show()
            
        except ImportError:
            print("Matplotlib not available for plotting")


def create_default_shift_configs() -> Dict[str, Dict[str, Any]]:
    """创建默认的shift配置"""
    return {
        'light_mass_shift': {
            'dynamic_shifts': {'mass_scale': 0.8}
        },
        'heavy_mass_shift': {
            'dynamic_shifts': {'mass_scale': 1.5}
        },
        'low_friction_shift': {
            'dynamic_shifts': {'friction_scale': 0.5}
        },
        'observation_noise_shift': {
            'observation_shifts': {'noise_std': 0.1}
        },
        'reward_scale_shift': {
            'reward_shifts': {'scale': 0.5}
        }
    }


def create_default_adaptation_configs() -> Dict[str, Dict[str, Any]]:
    """创建默认的适应配置"""
    return {
        'freeze_actor': {
            'strategy': 'online_finetune',
            'learning_rate': 1e-5,
            'batch_size': 32
        },
        'fine_tune_all': {
            'strategy': 'online_finetune', 
            'learning_rate': 1e-4,
            'batch_size': 64
        },
        'experience_based': {
            'strategy': 'experience_based',
            'learning_rate': 1e-5,
            'batch_size': 32
        }
    }


# 便捷函数
def run_tta_evaluation(env_name: str, policy_checkpoint: str, policy_class, 
                     shift_configs: Optional[Dict[str, Dict]] = None,
                     adaptation_configs: Optional[Dict[str, Dict]] = None,
                     results_dir: str = './tta_results') -> Dict[str, Any]:
    """
    运行完整的TTA评估流程
    
    Args:
        env_name: 环境名称
        policy_checkpoint: 策略检查点路径
        policy_class: 策略类
        shift_configs: shift配置
        adaptation_configs: 适应配置
        results_dir: 结果目录
        
    Returns:
        评估结果
    """
    if shift_configs is None:
        shift_configs = create_default_shift_configs()
    if adaptation_configs is None:
        adaptation_configs = create_default_adaptation_configs()
        
    evaluator = ShiftedPolicyEvaluator(env_name, shift_configs, adaptation_configs, results_dir)
    
    evaluation_config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'adaptation_episodes': 10,
        'evaluation_episodes': 5,
        'initial_evaluation_episodes': 3
    }
    
    results = evaluator.evaluate_policy(policy_checkpoint, policy_class, evaluation_config)
    
    # 生成策略比较
    strategy_stats = evaluator.compare_strategies(results)
    print("\nStrategy Comparison:")
    for strategy, stats in strategy_stats.items():
        print(f"{strategy}: Mean Improvement = {stats['mean_improvement']:.2f}")
        
    # 绘制结果
    evaluator.plot_results(results)
    
    return results