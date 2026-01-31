"""
通用TTA运行器 - Universal TTA Runner

提供统一的接口运行所有TTA算法：
- STINT
- TARL
- TEA
- CCEA
- 以及其他基于BaseTTAAlgorithm的算法
"""

import torch
import numpy as np
import csv
import os
from typing import Dict, Any, List, Optional

from offlinerlkit.tta.base_tta import BaseTTAAlgorithm
from offlinerlkit.tta.shifted_env import create_shifted_env
from offlinerlkit.tta.model_loader import ModelLoader
from offlinerlkit.policy import CQLPolicy


def run_tta_algorithm(
    algorithm_name: str,
    env_name: str,
    policy_checkpoint: str,
    shift_configs: Dict[str, Any],
    num_episodes: int = 20,
    results_csv_path: str = "./tta_results.csv",
    algorithm_config: Optional[Dict[str, Any]] = None,
    num_seeds: int = 3
) -> List[Dict]:
    """
    通用TTA算法运行器
    
    Args:
        algorithm_name: 算法名称 ('stint', 'tarl', 'tea', 'ccea')
        env_name: 环境名称
        policy_checkpoint: 策略检查点路径
        shift_configs: shift配置字典
        num_episodes: 每个seed的评估回合数
        results_csv_path: CSV结果文件路径
        algorithm_config: 算法特定配置（可选）
        num_seeds: 随机种子数量
    
    Returns:
        all_results: 所有实验结果列表
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Running {algorithm_name.upper()} algorithm...")
    
    # 获取算法配置
    algorithm_config = algorithm_config or _get_default_config(algorithm_name)
    
    all_results = []
    
    for shift_name, shift_config in shift_configs.items():
        print(f"\n{'='*60}")
        print(f"Testing shift: {shift_name}")
        print(f"Shift config: {shift_config}")
        print(f"{'='*60}")
        
        for seed in range(num_seeds):
            np.random.seed(seed)
            torch.manual_seed(seed)
            
            # 创建环境
            env = create_shifted_env(env_name, shift_config)
            
            # 加载策略
            model_loader = ModelLoader(CQLPolicy, device=device)
            env_config = {
                'observation_space': env.observation_space,
                'action_space': env.action_space,
                'obs_dim': env.observation_space.shape[0],
                'action_dim': env.action_space.shape[0]
            }
            policy = model_loader.load_pretrained_model(policy_checkpoint, env_config)
            
            # 创建算法管理器
            tta_manager = _create_algorithm_manager(
                algorithm_name, 
                policy, 
                env, 
                algorithm_config
            )
            
            # 运行适应
            adaptation_data, summary = tta_manager.run_adaptation(num_episodes=num_episodes)
            
            # 收集结果
            result_row = _create_result_row(
                shift_name, 
                shift_config, 
                seed, 
                algorithm_name, 
                adaptation_data, 
                summary
            )
            all_results.append(result_row)
            
            # 打印结果
            _print_seed_result(result_row)
    
    # 保存结果到CSV
    _save_results_to_csv(all_results, results_csv_path, algorithm_name)
    
    return all_results


def _get_default_config(algorithm_name: str) -> Dict[str, Any]:
    """
    获取算法的默认配置
    
    所有算法默认使用LayerNorm参数更新
    """
    base_config = {
        'adaptation_mode': 'layernorm',
        'learning_rate': 1e-5,
        'cache_capacity': 1000
    }
    
    if algorithm_name == 'stint':
        base_config.update({
            'delta': 0.5,
            'lambda_kl': 1.0,
            'K': 3,
            'beta': 0.1,
            'learning_rate': 1e-4
        })
    elif algorithm_name == 'tarl':
        base_config.update({
            'k_low_entropy': 10,
            'kl_weight': 1.0,
            'last_n_layers': 2,
            'learning_rate': 1e-6
        })
    elif algorithm_name == 'tea':
        base_config.update({
            'sgld_step_size': 0.1,
            'sgld_steps': 10,
            'num_neg_samples': 10,
            'kl_weight': 1.0,
            'action_space': 'continuous',
            'update_freq': 10,
            'learning_rate': 1e-6
        })
    elif algorithm_name == 'ccea':
        base_config.update({
            'lambda_min': 0.1,
            'lambda_max': 10.0,
            'lambda_init': 1.0,
            'pos_cache_capacity': 100,
            'neg_cache_capacity': 100,
            'gamma': 0.1,
            'tau': 1.0,
            'entropy_low': 0.5,
            'entropy_high': 2.0,
            'delta_stable': 0.1,
            'v_min': 0.1,
            'learning_rate': 1e-4
        })
    elif algorithm_name == 'come':
        base_config.update({
            'num_ensemble': 3,
            'uncertainty_threshold': 0.1,
            'kl_weight': 1.0,
            'conservative_factor': 0.9,
            'update_freq': 10,
            'learning_rate': 1e-6
        })
    elif algorithm_name == 'tent':
        base_config.update({
            'learning_rate': 1e-3,
            'momentum': 0.9,
            'damping': 0.0,
            'cache_capacity': 100
        })
    
    return base_config


def _create_algorithm_manager(
    algorithm_name: str,
    policy,
    env,
    config: Dict[str, Any]
) -> BaseTTAAlgorithm:
    """
    创建算法管理器
    
    Args:
        algorithm_name: 算法名称
        policy: 策略对象
        env: 环境
        config: 算法配置
    
    Returns:
        算法管理器实例
    """
    if algorithm_name == 'stint':
        from offlinerlkit.tta.stint import STINTManager
        return STINTManager(policy, env, config)
    elif algorithm_name == 'tarl':
        from offlinerlkit.tta.tarl import TARLManager
        return TARLManager(policy, env, config)
    elif algorithm_name == 'tea':
        from offlinerlkit.tta.tea import TEAManager
        return TEAManager(policy, env, config)
    elif algorithm_name == 'ccea':
        from offlinerlkit.tta.mcatta import CCEAManager
        return CCEAManager(policy, env, config)
    elif algorithm_name == 'come':
        from offlinerlkit.tta.come import COMEManager
        return COMEManager(policy, env, config)
    elif algorithm_name == 'tent':
        from offlinerlkit.tta.tent import TentManager
        return TentManager(policy, env, config)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")


def _create_result_row(
    shift_name: str,
    shift_config: Dict[str, Any],
    seed: int,
    algorithm_name: str,
    adaptation_data: List[Dict],
    summary: Dict[str, Any]
) -> Dict[str, Any]:
    """
    创建结果行
    
    根据算法类型提取不同的指标
    """
    result_row = {
        'shift_name': shift_name,
        'shift_label': shift_config.get('shift_label', 'unknown'),
        'mass_scale': shift_config.get('mass_scale', 1.0),
        'seed': seed,
        'algorithm': algorithm_name.upper(),
        'mean_reward': summary.get('mean_reward', 0),
        'std_reward': summary.get('std_reward', 0),
        'max_reward': summary.get('max_reward', 0),
        'min_reward': summary.get('min_reward', 0),
        'mean_length': summary.get('mean_length', 0),
        'adaptation_steps': summary.get('adaptation_steps', 0),
        'cache_size': summary.get('cache_size', 0)
    }
    
    # 添加算法特定的指标
    if algorithm_name == 'stint':
        result_row.update({
            'mean_loss': summary.get('mean_loss', 0),
            'mean_entropy': summary.get('mean_entropy', 0),
            'mean_kl': summary.get('mean_kl', 0),
            'total_triggers': summary.get('total_triggers', 0),
            'mean_triggers_per_episode': summary.get('mean_triggers_per_episode', 0),
            'final_entropy_moving_avg': summary.get('final_entropy_moving_avg', 0)
        })
    elif algorithm_name == 'tarl':
        result_row.update({
            'mean_entropy': summary.get('mean_entropy', 0),
            'mean_kl': summary.get('mean_kl', 0)
        })
    elif algorithm_name == 'tea':
        result_row.update({
            'mean_cd_loss': summary.get('mean_cd_loss', 0),
            'mean_kl_loss': summary.get('mean_kl_loss', 0),
            'mean_pos_energy': summary.get('mean_pos_energy', 0),
            'mean_neg_energy': summary.get('mean_neg_energy', 0)
        })
    elif algorithm_name == 'ccea':
        result_row.update({
            'final_lambda': summary.get('final_lambda', 0),
            'final_entropy': summary.get('final_entropy', 0),
            'final_entropy_velocity': summary.get('final_entropy_velocity', 0),
            'final_contrastive_uncertainty': summary.get('final_contrastive_uncertainty', 0),
            'lambda_history_mean': summary.get('lambda_history_mean', 0),
            'lambda_history_std': summary.get('lambda_history_std', 0),
            'contrastive_uncertainty_history_mean': summary.get('contrastive_uncertainty_history_mean', 0),
            'contrastive_uncertainty_history_std': summary.get('contrastive_uncertainty_history_std', 0)
        })
    elif algorithm_name == 'come':
        result_row.update({
            'mean_uncertainty': summary.get('mean_uncertainty', 0),
            'max_uncertainty': summary.get('max_uncertainty', 0),
            'min_uncertainty': summary.get('min_uncertainty', 0),
            'ensemble_disagreement': summary.get('ensemble_disagreement', 0)
        })
    elif algorithm_name == 'tent':
        result_row.update({
            'mean_entropy': summary.get('mean_entropy', 0),
            'entropy_history_mean': np.mean(summary.get('entropy_history', [])) if summary.get('entropy_history') else 0,
            'entropy_history_std': np.std(summary.get('entropy_history', [])) if summary.get('entropy_history') else 0
        })
    
    return result_row


def _print_seed_result(result_row: Dict[str, Any]):
    """打印单个seed的结果"""
    algo = result_row['algorithm']
    print(f"  Seed {result_row['seed']}: "
          f"Reward = {result_row['mean_reward']:.2f} ± {result_row['std_reward']:.2f}")
    
    # 打印算法特定的指标
    if algo == 'STINT':
        print(f"    Loss = {result_row['mean_loss']:.6f}, "
              f"Entropy = {result_row['mean_entropy']:.4f}, "
              f"KL = {result_row['mean_kl']:.4f}, "
              f"Triggers = {result_row['total_triggers']}")
    elif algo == 'TARL':
        print(f"    Entropy = {result_row['mean_entropy']:.4f}, "
              f"KL = {result_row['mean_kl']:.4f}")
    elif algo == 'TEA':
        print(f"    CD Loss = {result_row['mean_cd_loss']:.6f}, "
              f"KL Loss = {result_row['mean_kl_loss']:.6f}, "
              f"Pos Energy = {result_row['mean_pos_energy']:.4f}, "
              f"Neg Energy = {result_row['mean_neg_energy']:.4f}")
    elif algo == 'CCEA':
        print(f"    Final λ = {result_row['final_lambda']:.3f}, "
              f"Final Entropy = {result_row['final_entropy']:.3f}, "
              f"Final ΔEntropy = {result_row['final_entropy_velocity']:.3f}, "
              f"Final U_cont = {result_row['final_contrastive_uncertainty']:.3f}")
    elif algo == 'COME':
        print(f"    Mean Uncertainty = {result_row['mean_uncertainty']:.4f}, "
              f"Max Uncertainty = {result_row['max_uncertainty']:.4f}, "
              f"Ensemble Size = {result_row['ensemble_size']}, "
              f"Conservative Factor = {result_row['conservative_factor']:.2f}")


def _save_results_to_csv(results: List[Dict], csv_path: str, algorithm_name: str):
    """保存结果到CSV文件"""
    os.makedirs(os.path.dirname(csv_path) if os.path.dirname(csv_path) else '.', exist_ok=True)
    
    # 确定字段名
    fieldnames = [
        'shift_name', 'shift_label', 'mass_scale', 'seed', 'algorithm',
        'mean_reward', 'std_reward', 'max_reward', 'min_reward',
        'mean_length', 'adaptation_steps', 'cache_size'
    ]
    
    # 添加算法特定的字段
    algorithm_specific_fields = {
        'STINT': ['mean_loss', 'mean_entropy', 'mean_kl', 
                  'total_triggers', 'mean_triggers_per_episode', 'final_entropy_moving_avg'],
        'TARL': ['mean_entropy', 'mean_kl'],
        'TEA': ['mean_cd_loss', 'mean_kl_loss', 'mean_pos_energy', 'mean_neg_energy'],
        'CCEA': ['final_lambda', 'final_entropy', 'final_entropy_velocity', 
                  'final_contrastive_uncertainty', 'lambda_history_mean', 'lambda_history_std',
                  'contrastive_uncertainty_history_mean', 'contrastive_uncertainty_history_std']
    }
    
    if algorithm_name.upper() in algorithm_specific_fields:
        fieldnames.extend(algorithm_specific_fields[algorithm_name.upper()])
    
    # 格式化结果
    formatted_results = []
    for row in results:
        formatted_row = {}
        for key, value in row.items():
            if isinstance(value, float):
                formatted_row[key] = round(value, 6)
            else:
                formatted_row[key] = value
        formatted_results.append(formatted_row)
    
    # 写入CSV
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(formatted_results)
    
    print(f"\n{'='*60}")
    print(f"{algorithm_name.upper()} results saved to: {csv_path}")
    print(f"{'='*60}")


def compare_algorithms(
    algorithm_names: List[str],
    env_name: str,
    policy_checkpoint: str,
    shift_configs: Dict[str, Any],
    num_episodes: int = 20,
    results_dir: str = "./tta_comparison_results",
    num_seeds: int = 3
) -> Dict[str, Any]:
    """
    对比多个TTA算法的性能
    
    Args:
        algorithm_names: 算法名称列表
        env_name: 环境名称
        policy_checkpoint: 策略检查点路径
        shift_configs: shift配置字典
        num_episodes: 每个seed的评估回合数
        results_dir: 结果保存目录
        num_seeds: 随机种子数量
    
    Returns:
        comparison_summary: 对比摘要
    """
    os.makedirs(results_dir, exist_ok=True)
    
    all_results = []
    
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE TTA ALGORITHM COMPARISON")
    print(f"{'='*80}")
    print(f"Algorithms to compare: {', '.join([algo.upper() for algo in algorithm_names])}")
    
    for algo_name in algorithm_names:
        print(f"\n{'='*60}")
        print(f"Running {algo_name.upper()}...")
        print(f"{'='*60}")
        
        csv_path = os.path.join(results_dir, f"{algo_name}_results.csv")
        
        results = run_tta_algorithm(
            algorithm_name=algo_name,
            env_name=env_name,
            policy_checkpoint=policy_checkpoint,
            shift_configs=shift_configs,
            num_episodes=num_episodes,
            results_csv_path=csv_path,
            num_seeds=num_seeds
        )
        
        all_results.extend(results)
    
    # 生成对比摘要
    comparison_summary = _generate_comparison_summary(all_results, algorithm_names)
    
    # 保存对比摘要
    summary_csv_path = os.path.join(results_dir, "comparison_summary.csv")
    _save_comparison_summary_to_csv(comparison_summary, summary_csv_path)
    
    # 打印对比表格
    _print_comparison_table(comparison_summary)
    
    return comparison_summary


def _generate_comparison_summary(results: List[Dict], algorithm_names: List[str]) -> Dict[str, Any]:
    """生成对比摘要"""
    summary = {}
    
    for shift_name in set(r['shift_name'] for r in results):
        summary[shift_name] = {}
        
        for algo in algorithm_names:
            subset = [r for r in results if r['shift_name'] == shift_name and r['algorithm'].lower() == algo]
            
            if subset:
                summary[shift_name][algo.upper()] = {
                    'mean_reward': np.mean([r['mean_reward'] for r in subset]),
                    'std_reward': np.mean([r['std_reward'] for r in subset]),
                    'num_seeds': len(subset)
                }
    
    return summary


def _save_comparison_summary_to_csv(summary: Dict[str, Any], csv_path: str):
    """保存对比摘要到CSV"""
    fieldnames = ['shift_name', 'algorithm', 'mean_reward', 'std_reward', 'num_seeds']
    
    summary_data = []
    for shift_name, algorithms in summary.items():
        for algo, metrics in algorithms.items():
            summary_data.append({
                'shift_name': shift_name,
                'algorithm': algo,
                'mean_reward': round(metrics['mean_reward'], 6),
                'std_reward': round(metrics['std_reward'], 6),
                'num_seeds': metrics['num_seeds']
            })
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_data)
    
    print(f"\nComparison summary saved to: {csv_path}")


def _print_comparison_table(summary: Dict[str, Any]):
    """打印对比表格"""
    print(f"\n{'='*100}")
    print("COMPARISON SUMMARY")
    print(f"{'='*100}")
    print(f"{'Shift':<20} {'Algorithm':<15} {'Mean Reward':<15} {'Std Reward':<15}")
    print(f"{'-'*100}")
    
    for shift_name, algorithms in summary.items():
        for algo, metrics in algorithms.items():
            print(f"{shift_name:<20} {algo:<15} {metrics['mean_reward']:<15.2f} {metrics['std_reward']:<15.2f}")
    
    print(f"{'='*100}")
