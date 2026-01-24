"""
Test-Time Adaptation (TTA) 示例代码

展示如何使用OfflineRL-Kit的TTA框架在MUJOCO环境中进行模型评估和适应
支持可选的TTA开关，以及熵最小化和不确定性最小化策略
"""

import gym
import torch
import numpy as np
import csv
import os
from typing import Dict, Any, List

from offlinerlkit.tta import ShiftedPolicyEvaluator, run_tta_evaluation, CCEAManager, TARLManager
from offlinerlkit.policy import CQLPolicy


def basic_tta_example(enable_tta: bool = True, tta_strategy: str = 'entropy_minimization'):
    """基础TTA示例"""
    
    env_name = "hopper-medium-v2"
    policy_checkpoint = "./checkpoints/cql_hopper_medium.pt"
    results_dir = "./tta_results"
    
    shift_configs = {
        'light_mass': {
            'dynamic_shifts': {'mass_scale': 0.8}
        },
        'heavy_mass': {
            'dynamic_shifts': {'mass_scale': 1.5}
        },
        'low_friction': {
            'dynamic_shifts': {'friction_scale': 0.5}
        },
        'observation_noise': {
            'observation_shifts': {'noise_std': 0.1}
        }
    }
    
    adaptation_configs = {
        'tta_adapt': {
            'strategy': tta_strategy,
            'learning_rate': 1e-5,
            'batch_size': 32
        }
    }
    
    print(f"Starting TTA evaluation (TTA: {'ON' if enable_tta else 'OFF'}, Strategy: {tta_strategy})...")
    
    results = run_tta_evaluation(
        env_name=env_name,
        policy_checkpoint=policy_checkpoint,
        policy_class=CQLPolicy,
        shift_configs=shift_configs,
        adaptation_configs=adaptation_configs,
        results_dir=results_dir,
        enable_tta=enable_tta
    )
    
    print("TTA evaluation completed!")
    return results


def advanced_tta_example(enable_tta: bool = True, tta_strategy: str = 'uncertainty_minimization'):
    """高级TTA示例 - 自定义配置"""
    
    evaluator = ShiftedPolicyEvaluator(
        base_env_name="walker2d-medium-v2",
        shift_configs={
            'complex_shift': {
                'dynamic_shifts': {
                    'mass_scale': 1.2,
                    'friction_scale': 0.8
                },
                'observation_shifts': {
                    'noise_std': 0.05,
                    'bias': np.array([0.1] * 17)
                },
                'reward_shifts': {
                    'scale': 0.8,
                    'bias': -0.1
                }
            }
        },
        adaptation_configs={
            'progressive_adapt': {
                'strategy': tta_strategy,
                'learning_rate': 1e-5,
                'batch_size': 16,
                'adaptation_steps': 2000
            }
        },
        results_dir="./advanced_tta_results"
    )
    
    evaluation_config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'adaptation_episodes': 15,
        'evaluation_episodes': 8,
        'initial_evaluation_episodes': 5,
        'enable_tta': enable_tta
    }
    
    results = evaluator.evaluate_policy(
        policy_checkpoint="./checkpoints/cql_walker2d_medium.pt",
        policy_class=CQLPolicy,
        evaluation_config=evaluation_config
    )
    
    strategy_stats = evaluator.compare_strategies(results)
    print("\nAdvanced TTA Results:")
    for strategy, stats in strategy_stats.items():
        print(f"{strategy}:")
        print(f"  Mean Improvement: {stats['mean_improvement']:.2f}")
        print(f"  Mean Final Reward: {stats['mean_final_reward']:.2f}")
    
    return results


def run_tta_experiment(env_name: str, policy_checkpoint: str, 
                       shift_configs: Dict[str, Any],
                       enable_tta: bool = True,
                       tta_strategy: str = 'entropy_minimization',
                       num_episodes: int = 20,
                       results_csv_path: str = "./tta_results.csv",
                       learning_rate: float = 1e-5):
    """
    运行TTA实验并保存结果到CSV
    
    Args:
        env_name: 环境名称
        policy_checkpoint: 策略检查点路径
        shift_configs: shift配置字典
        enable_tta: 是否启用TTA
        tta_strategy: TTA策略 ('entropy_minimization' 或 'uncertainty_minimization')
        num_episodes: 评估回合数
        results_csv_path: CSV结果文件路径
        learning_rate: TTA学习率
    """
    from offlinerlkit.tta.shifted_env import create_shifted_env
    from offlinerlkit.tta.tta_manager import TTAManager
    from offlinerlkit.tta.model_loader import ModelLoader
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"TTA: {'ENABLED' if enable_tta else 'DISABLED'}")
    print(f"Strategy: {tta_strategy}")
    
    all_results = []
    
    for shift_name, shift_config in shift_configs.items():
        print(f"\nTesting shift: {shift_name}")
        print(f"Shift config: {shift_config}")
        
        for seed in range(3):
            np.random.seed(seed)
            torch.manual_seed(seed)
            
            env = create_shifted_env(env_name, shift_config)
            
            model_loader = ModelLoader(CQLPolicy, device=device)
            env_config = {
                'observation_space': env.observation_space,
                'action_space': env.action_space,
                'obs_dim': env.observation_space.shape[0],
                'action_dim': env.action_space.shape[0]
            }
            policy = model_loader.load_pretrained_model(policy_checkpoint, env_config)
            
            # 解冻参数以支持TTA训练
            if enable_tta and tta_strategy != 'none':
                lr = learning_rate
                
                if tta_strategy == 'entropy_minimization':
                    # 熵最小化：优化 actor 网络
                    for param in policy.actor.parameters():
                        param.requires_grad = True
                    policy.adaptation_optimizer = torch.optim.Adam(policy.actor.parameters(), lr=lr)
                    
                elif tta_strategy == 'uncertainty_minimization':
                    # 不确定性最小化：优化 critic 网络
                    for param in policy.critic1.parameters():
                        param.requires_grad = True
                    for param in policy.critic2.parameters():
                        param.requires_grad = True
                    policy.adaptation_optimizer = torch.optim.Adam(
                        list(policy.critic1.parameters()) + list(policy.critic2.parameters()), 
                        lr=lr
                    )
            
            adaptation_config = {
                'enable_tta': enable_tta,
                'strategy': tta_strategy,
                'learning_rate': learning_rate,
                'batch_size': 32,
                'collapse_threshold': -100.0,
                'collapse_window': 5
            }
            
            tta_manager = TTAManager(policy, env, adaptation_config)
            adaptation_data, metrics_summary = tta_manager.run_adaptation(num_episodes=num_episodes)
            
            result_row = {
                'shift_name': shift_name,
                'seed': seed,
                'enable_tta': enable_tta,
                'tta_strategy': tta_strategy,
                'mean_reward': metrics_summary.get('mean_reward', 0),
                'std_reward': metrics_summary.get('std_reward', 0),
                'max_reward': metrics_summary.get('max_reward', 0),
                'min_reward': metrics_summary.get('min_reward', 0),
                'worst_case_return': metrics_summary.get('worst_case_return', 0),
                'mean_length': metrics_summary.get('mean_length', 0),
                'mean_policy_divergence': metrics_summary.get('mean_policy_divergence', 0),
                'collapse_detected': metrics_summary.get('collapse_detected', False),
                'collapse_rate': metrics_summary.get('collapse_rate', 0)
            }
            all_results.append(result_row)
            
            print(f"  Seed {seed}: Reward = {result_row['mean_reward']:.2f} ± {result_row['std_reward']:.2f}")
    
    save_results_to_csv(all_results, results_csv_path)
    
    return all_results


def save_results_to_csv(results: List[Dict], csv_path: str):
    """保存结果到CSV文件"""
    os.makedirs(os.path.dirname(csv_path) if os.path.dirname(csv_path) else '.', exist_ok=True)
    
    fieldnames = ['shift_name', 'seed', 'enable_tta', 'tta_strategy', 
                  'mean_reward', 'std_reward', 'max_reward', 'min_reward',
                  'worst_case_return', 'mean_length', 'mean_policy_divergence',
                  'collapse_detected', 'collapse_rate']
    
    formatted_results = []
    for row in results:
        formatted_row = {}
        for key, value in row.items():
            if isinstance(value, float):
                formatted_row[key] = round(value, 6)
            else:
                formatted_row[key] = value
        formatted_results.append(formatted_row)
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(formatted_results)
    
    print(f"\nResults saved to: {csv_path}")


def run_comparison_experiment(env_name: str, policy_checkpoint: str,
                              shift_configs: Dict[str, Any],
                              num_episodes: int = 20,
                              results_dir: str = "./tta_comparison_results"):
    """
    运行对比实验：比较有TTA和无TTA的效果
    
    Args:
        env_name: 环境名称
        policy_checkpoint: 策略检查点路径
        shift_configs: shift配置字典
        num_episodes: 评估回合数
        results_dir: 结果保存目录
    """
    os.makedirs(results_dir, exist_ok=True)
    
    strategies = ['entropy_minimization', 'uncertainty_minimization']
    all_results = []
    
    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"Testing strategy: {strategy}")
        print(f"{'='*60}")
        
        for enable_tta in [True, False]:
            tta_status = "WITH_TTA" if enable_tta else "WITHOUT_TTA"
            csv_path = os.path.join(results_dir, f"{strategy}_{'tta' if enable_tta else 'no_tta'}.csv")
            
            results = run_tta_experiment(
                env_name=env_name,
                policy_checkpoint=policy_checkpoint,
                shift_configs=shift_configs,
                enable_tta=enable_tta,
                tta_strategy=strategy,
                num_episodes=num_episodes,
                results_csv_path=csv_path
            )
            
            all_results.extend(results)
    
    summary_csv = os.path.join(results_dir, "summary.csv")
    save_summary_to_csv(all_results, summary_csv)
    
    print_comparison_summary(all_results)
    
    return all_results


def save_summary_to_csv(results: List[Dict], csv_path: str):
    """保存汇总结果到CSV"""
    fieldnames = ['shift_name', 'enable_tta', 'tta_strategy', 
                  'mean_reward', 'std_reward', 'worst_case_return', 
                  'mean_policy_divergence', 'collapse_rate']
    
    summary_data = []
    for shift_name in set(r['shift_name'] for r in results):
        for enable_tta in [True, False]:
            for strategy in set(r['tta_strategy'] for r in results):
                subset = [r for r in results if r['shift_name'] == shift_name 
                         and r['enable_tta'] == enable_tta 
                         and r['tta_strategy'] == strategy]
                if subset:
                    summary_data.append({
                        'shift_name': shift_name,
                        'enable_tta': enable_tta,
                        'tta_strategy': strategy,
                        'mean_reward': np.mean([r['mean_reward'] for r in subset]),
                        'std_reward': np.mean([r['std_reward'] for r in subset]),
                        'worst_case_return': np.mean([r['worst_case_return'] for r in subset]),
                        'mean_policy_divergence': np.mean([r['mean_policy_divergence'] for r in subset]),
                        'collapse_rate': np.mean([r['collapse_rate'] for r in subset])
                    })
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_data)
    
    print(f"Summary saved to: {csv_path}")


def print_comparison_summary(results: List[Dict]):
    """打印对比摘要"""
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    for strategy in set(r['tta_strategy'] for r in results):
        print(f"\nStrategy: {strategy}")
        for shift_name in set(r['shift_name'] for r in results):
            tta_result = [r for r in results if r['shift_name'] == shift_name 
                         and r['enable_tta'] and r['tta_strategy'] == strategy]
            no_tta_result = [r for r in results if not r['enable_tta'] 
                            and r['tta_strategy'] == strategy]
            
            if tta_result and no_tta_result:
                tta_mean = np.mean([r['mean_reward'] for r in tta_result])
                no_tta_mean = np.mean([r['mean_reward'] for r in no_tta_result])
                improvement = tta_mean - no_tta_mean
                improvement_pct = (improvement / max(no_tta_mean, 1e-6)) * 100
                
                print(f"  {shift_name}:")
                print(f"    Without TTA: {no_tta_mean:.2f}")
                print(f"    With TTA: {tta_mean:.2f}")
                print(f"    Improvement: {improvement:.2f} ({improvement_pct:.1f}%)")


def run_ccea_experiment(env_name: str, policy_checkpoint: str,
                          shift_configs: Dict[str, Any],
                          num_episodes: int = 20,
                          results_csv_path: str = "./ccea_results.csv"):
    """
    运行CCEA实验并保存结果到CSV
    
    CCEA (Contrastive Cache-based Entropic Adaptation) 特点：
    - 4D元特征提取: x_t = [H_t, ΔH_t, S_t, V_t]^⊤
    - Episode质量标签: y_t ∈ {+1, -1, 0}
    - 对比不确定性: U_t^cont = σ((d_pos - d_neg) / τ)
    - 双缓存系统(C_pos, C_neg)与熵优先级替换
    - 李雅普诺夫稳定的λ演化机制
    - LayerNorm-only参数更新策略
    
    Args:
        env_name: 环境名称
        policy_checkpoint: 策略检查点路径
        shift_configs: shift配置字典
        num_episodes: 评估回合数
        results_csv_path: CSV结果文件路径
    """
    from offlinerlkit.tta.shifted_env import create_shifted_env
    from offlinerlkit.tta.model_loader import ModelLoader
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print("Running CCEA (Contrastive Cache-based Entropic Adaptation)...")
    
    all_results = []
    
    for shift_name, shift_config in shift_configs.items():
        print(f"\nTesting shift: {shift_name}")
        print(f"Shift config: {shift_config}")
        
        for seed in range(3):
            np.random.seed(seed)
            torch.manual_seed(seed)
            
            env = create_shifted_env(env_name, shift_config)
            
            model_loader = ModelLoader(CQLPolicy, device=device)
            env_config = {
                'observation_space': env.observation_space,
                'action_space': env.action_space,
                'obs_dim': env.observation_space.shape[0],
                'action_dim': env.action_space.shape[0]
            }
            policy = model_loader.load_pretrained_model(policy_checkpoint, env_config)
            
            # CCEA算法配置
            ccea_config = {
                'lambda_min': 0.1,
                'lambda_max': 10.0,
                'lambda_init': 1.0,
                'policy_lr': 1e-4,
                'batch_size': 32,
                'pos_cache_capacity': 100,
                'neg_cache_capacity': 100,
                'gamma': 0.1,      # 熵抑制系数
                'tau': 1.0,         # 温度参数
                'entropy_low': 0.5,   # 熵下限
                'entropy_high': 2.0,  # 熵上限
                'delta_stable': 0.1,  # 稳定阈值
                'v_min': 0.1,        # 最小新颖性比例
                'adaptation_mode': 'layernorm',
            }
            
            ccea_manager = CCEAManager(policy, env, ccea_config)
            adaptation_data, summary = ccea_manager.run_adaptation(num_episodes=num_episodes)
            
            result_row = {
                'shift_name': shift_name,
                'seed': seed,
                'algorithm': 'CCEA',
                'mean_reward': summary.get('mean_reward', 0),
                'std_reward': summary.get('std_reward', 0),
                'final_lambda': summary.get('final_lambda', 0),
                'final_entropy': summary.get('final_entropy', 0),
                'final_entropy_velocity': summary.get('final_entropy_velocity', 0),
                'final_contrastive_uncertainty': summary.get('final_contrastive_uncertainty', 0),
                'lambda_history_mean': np.mean(summary.get('lambda_history', [0])),
                'lambda_history_std': np.std(summary.get('lambda_history', [0])),
                'contrastive_uncertainty_history_mean': np.mean(summary.get('contrastive_uncertainty_history', [0])),
                'contrastive_uncertainty_history_std': np.std(summary.get('contrastive_uncertainty_history', [0]))
            }
            all_results.append(result_row)
            
            print(f"  Seed {seed}: Reward = {result_row['mean_reward']:.2f} ± {result_row['std_reward']:.2f}, "
                  f"Final λ = {result_row['final_lambda']:.3f}, "
                  f"Final Entropy = {result_row['final_entropy']:.3f}, "
                  f"Final ΔEntropy = {result_row['final_entropy_velocity']:.3f}, "
                  f"Final U_cont = {result_row['final_contrastive_uncertainty']:.3f}")
    
    save_ccea_results_to_csv(all_results, results_csv_path)
    
    return all_results


def run_tarl_experiment(env_name: str, policy_checkpoint: str,
                        shift_configs: Dict[str, Any],
                        num_episodes: int = 20,
                        results_csv_path: str = "./tarl_results.csv"):
    """
    运行TARL实验并保存结果到CSV
    
    TARL (Test-Time Adaptation with Reinforcement Learning) 特点：
    - 基于动作不确定性的熵最小化
    - 低熵样本筛选机制
    - 仅更新LayerNorm参数保证稳定性
    - KL散度正则化防止策略漂移
    
    Args:
        env_name: 环境名称
        policy_checkpoint: 策略检查点路径
        shift_configs: shift配置字典
        num_episodes: 评估回合数
        results_csv_path: CSV结果文件路径
    """
    from offlinerlkit.tta.shifted_env import create_shifted_env
    from offlinerlkit.tta.model_loader import ModelLoader
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print("Running TARL (Test-Time Adaptation with Reinforcement Learning)...")
    
    all_results = []
    
    for shift_name, shift_config in shift_configs.items():
        print(f"\nTesting shift: {shift_name}")
        print(f"Shift config: {shift_config}")
        
        for seed in range(3):
            np.random.seed(seed)
            torch.manual_seed(seed)
            
            env = create_shifted_env(env_name, shift_config)
            
            model_loader = ModelLoader(CQLPolicy, device=device)
            env_config = {
                'observation_space': env.observation_space,
                'action_space': env.action_space,
                'obs_dim': env.observation_space.shape[0],
                'action_dim': env.action_space.shape[0]
            }
            policy = model_loader.load_pretrained_model(policy_checkpoint, env_config)
            
            tarl_config = {
                'learning_rate': 1e-6,
                'cache_capacity': 1000,
                'k_low_entropy': 10,
                'kl_weight': 1.0,
                'gradient_clip': 0.5
            }
            
            tarl_manager = TARLManager(policy, env, tarl_config)
            adaptation_data, summary = tarl_manager.run_adaptation(num_episodes=num_episodes)
            
            result_row = {
                'shift_name': shift_name,
                'seed': seed,
                'algorithm': 'TARL',
                'mean_reward': summary.get('mean_reward', 0),
                'std_reward': summary.get('std_reward', 0),
                'mean_length': summary.get('mean_length', 0),
                'mean_loss': summary.get('mean_loss', 0),
                'mean_entropy': summary.get('mean_entropy', 0),
                'mean_kl': summary.get('mean_kl', 0),
                'cache_size': summary.get('cache_size', 0),
                'adaptation_steps': summary.get('adaptation_steps', 0)
            }
            all_results.append(result_row)
            
            print(f"  Seed {seed}: Reward = {result_row['mean_reward']:.2f} ± {result_row['std_reward']:.2f}, "
                  f"Loss = {result_row['mean_loss']:.6f}, "
                  f"Entropy = {result_row['mean_entropy']:.4f}, "
                  f"KL = {result_row['mean_kl']:.4f}")
    
    save_tarl_results_to_csv(all_results, results_csv_path)
    
    return all_results


def save_tarl_results_to_csv(results: List[Dict], csv_path: str):
    """保存TARL结果到CSV文件"""
    os.makedirs(os.path.dirname(csv_path) if os.path.dirname(csv_path) else '.', exist_ok=True)
    
    fieldnames = ['shift_name', 'seed', 'algorithm', 'mean_reward', 'std_reward',
                  'mean_length', 'mean_loss', 'mean_entropy', 'mean_kl',
                  'cache_size', 'adaptation_steps']
    
    formatted_results = []
    for row in results:
        formatted_row = {}
        for key, value in row.items():
            if isinstance(value, float):
                formatted_row[key] = round(value, 6)
            else:
                formatted_row[key] = value
        formatted_results.append(formatted_row)
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(formatted_results)
    
    print(f"\nTARL results saved to: {csv_path}")


def save_ccea_results_to_csv(results: List[Dict], csv_path: str):
    """保存CCEA结果到CSV文件"""
    os.makedirs(os.path.dirname(csv_path) if os.path.dirname(csv_path) else '.', exist_ok=True)
    
    fieldnames = ['shift_name', 'seed', 'algorithm', 
                  'mean_reward', 'std_reward', 'final_lambda',
                  'final_entropy', 'final_entropy_velocity', 'final_contrastive_uncertainty',
                  'lambda_history_mean', 'lambda_history_std',
                  'contrastive_uncertainty_history_mean', 'contrastive_uncertainty_history_std']
    
    formatted_results = []
    for row in results:
        formatted_row = {}
        for key, value in row.items():
            if isinstance(value, float):
                formatted_row[key] = round(value, 6)
            else:
                formatted_row[key] = value
        formatted_results.append(formatted_row)
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(formatted_results)
    
    print(f"\nCCEA results saved to: {csv_path}")


def compare_tta_strategies(env_name: str, policy_checkpoint: str,
                          shift_configs: Dict[str, Any],
                          num_episodes: int = 20,
                          results_dir: str = "./tta_comparison_results"):
    """
    对比不同TTA策略的性能
    
    包括：
    - 无TTA (No TTA)
    - 熵最小化 (Entropy Minimization)
    - 不确定性最小化 (Uncertainty Minimization)
    - CCEA (Contrastive Cache-based Entropic Adaptation)
    
    Args:
        env_name: 环境名称
        policy_checkpoint: 策略检查点路径
        shift_configs: shift配置字典
        num_episodes: 评估回合数
        results_dir: 结果保存目录
    """
    os.makedirs(results_dir, exist_ok=True)
    
    all_results = []
    
    print(f"\n{'='*80}")
    print("COMPREHENSIVE TTA STRATEGY COMPARISON")
    print(f"{'='*80}")
    
    strategies = [
        ('No TTA', 'none', False),
        ('Entropy Minimization', 'entropy_minimization', True),
        ('Uncertainty Minimization', 'uncertainty_minimization', True)
    ]
    
    for strategy_name, strategy_type, enable_tta in strategies:
        print(f"\n{'='*60}")
        print(f"Testing: {strategy_name}")
        print(f"{'='*60}")
        
        csv_path = os.path.join(results_dir, f"{strategy_name.replace(' ', '_')}.csv")
        
        if strategy_name == 'No TTA':
            results = run_tta_experiment(
                env_name=env_name,
                policy_checkpoint=policy_checkpoint,
                shift_configs=shift_configs,
                enable_tta=False,
                tta_strategy='none',
                num_episodes=num_episodes,
                results_csv_path=csv_path
            )
        else:
            results = run_tta_experiment(
                env_name=env_name,
                policy_checkpoint=policy_checkpoint,
                shift_configs=shift_configs,
                enable_tta=True,
                tta_strategy=strategy_type,
                num_episodes=num_episodes,
                results_csv_path=csv_path
            )
        
        all_results.extend(results)
    
    print(f"\n{'='*60}")
    print("Testing: CCEA (Contrastive Cache-based Entropic Adaptation)")
    ccea_csv_path = os.path.join(results_dir, "CCEA.csv")
    ccea_results = run_ccea_experiment(
        env_name=env_name,
        policy_checkpoint=policy_checkpoint,
        shift_configs=shift_configs,
        num_episodes=num_episodes,
        results_csv_path=ccea_csv_path
    )
    all_results.extend(ccea_results)
    
    print(f"\n{'='*60}")
    print("Testing: TARL")
    print(f"{'='*60}")
    
    tarl_csv_path = os.path.join(results_dir, "TARL.csv")
    tarl_results = run_tarl_experiment(
        env_name=env_name,
        policy_checkpoint=policy_checkpoint,
        shift_configs=shift_configs,
        num_episodes=num_episodes,
        results_csv_path=tarl_csv_path
    )
    
    all_results.extend(tarl_results)
    
    comparison_summary = generate_comparison_summary(all_results)
    summary_csv = os.path.join(results_dir, "comparison_summary.csv")
    save_comparison_summary_to_csv(comparison_summary, summary_csv)
    
    print_comparison_summary_table(comparison_summary)
    
    return all_results


def generate_comparison_summary(results: List[Dict]) -> Dict[str, Any]:
    """生成对比摘要"""
    summary = {}
    
    for shift_name in set(r['shift_name'] for r in results):
        summary[shift_name] = {}
        
        algorithms = set(r.get('algorithm', r.get('tta_strategy', 'none')) for r in results 
                        if r['shift_name'] == shift_name)
        
        for algo in algorithms:
            if algo == 'CCEA':
                subset = [r for r in results if r['shift_name'] == shift_name and r.get('algorithm') == 'CCEA']
            elif algo == 'TARL':
                subset = [r for r in results if r['shift_name'] == shift_name and r.get('algorithm') == 'TARL']
            else:
                subset = [r for r in results if r['shift_name'] == shift_name and r.get('tta_strategy') == algo]
            
            if subset:
                summary[shift_name][algo] = {
                    'mean_reward': np.mean([r['mean_reward'] for r in subset]),
                    'std_reward': np.mean([r['std_reward'] for r in subset]),
                    'num_seeds': len(subset)
                }
    
    return summary


def save_comparison_summary_to_csv(summary: Dict[str, Any], csv_path: str):
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
    
    print(f"Comparison summary saved to: {csv_path}")


def print_comparison_summary_table(summary: Dict[str, Any]):
    """打印对比摘要表格"""
    print(f"\n{'='*100}")
    print("COMPREHENSIVE COMPARISON SUMMARY")
    print(f"{'='*100}")
    print(f"{'Shift':<20} {'Algorithm':<25} {'Mean Reward':<15} {'Std Reward':<15}")
    print(f"{'-'*100}")
    
    for shift_name, algorithms in summary.items():
        for algo, metrics in algorithms.items():
            print(f"{shift_name:<20} {algo:<25} {metrics['mean_reward']:<15.2f} {metrics['std_reward']:<15.2f}")
    
    print(f"{'='*100}")


def ccea_demo():
    """CCEA演示函数"""
    print("=" * 80)
    print("CCEA (Contrastive Cache-based Entropic Adaptation) Demo")
    print("=" * 80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    shift_configs = {
        'normal': {},
        'light_mass': {'dynamic_shifts': {'mass_scale': 0.8}},
        'heavy_mass': {'dynamic_shifts': {'mass_scale': 1.5}},
        'low_friction': {'dynamic_shifts': {'friction_scale': 0.5}},
        'observation_noise': {'observation_shifts': {'noise_std': 0.1}}
    }
    
    try:
        print("\nRunning CCEA Experiment...")
        results = run_ccea_experiment(
            env_name="hopper-medium-v2",
            policy_checkpoint="./checkpoints/cql_hopper_medium.pt",
            shift_configs=shift_configs,
            num_episodes=20,
            results_csv_path="./ccea_results.csv"
        )
        
        print("\n" + "=" * 80)
        print("CCEA experiment completed successfully!")
        print("Results saved to: ./ccea_results.csv")
        print("=" * 80)
        
    except Exception as e:
        print(f"Error running CCEA experiment: {e}")
        print("Make sure you have the required checkpoints and dependencies.")


def main():
    """主函数 - 运行 TTA 实验"""
    
    print("=" * 80)
    print("OfflineRL-Kit TTA Framework Demo")
    print("=" * 80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    shift_configs = {
        'normal': {},
        'light_mass': {'dynamic_shifts': {'mass_scale': 0.8}},
        'heavy_mass': {'dynamic_shifts': {'mass_scale': 1.5}},
        'low_friction': {'dynamic_shifts': {'friction_scale': 0.5}},
        'observation_noise': {'observation_shifts': {'noise_std': 0.1}}
    }
    
    try:
        print("\n1. Running TTA Experiment with Entropy Minimization...")
        results_entropy = run_tta_experiment(
            env_name="hopper-medium-v2",
            policy_checkpoint="./checkpoints/cql_hopper_medium.pt",
            shift_configs=shift_configs,
            enable_tta=True,
            tta_strategy='entropy_minimization',
            num_episodes=20,
            results_csv_path="./tta_results_entropy.csv"
        )
        
        print("\n2. Running TTA Experiment with Uncertainty Minimization...")
        results_uncertainty = run_tta_experiment(
            env_name="hopper-medium-v2",
            policy_checkpoint="./checkpoints/cql_hopper_medium.pt",
            shift_configs=shift_configs,
            enable_tta=True,
            tta_strategy='uncertainty_minimization',
            num_episodes=20,
            results_csv_path="./tta_results_uncertainty.csv"
        )
        
        print("\n3. Running Experiment WITHOUT TTA...")
        results_no_tta = run_tta_experiment(
            env_name="hopper-medium-v2",
            policy_checkpoint="./checkpoints/cql_hopper_medium.pt",
            shift_configs=shift_configs,
            enable_tta=False,
            tta_strategy='none',
            num_episodes=20,
            results_csv_path="./tta_results_no_tta.csv"
        )
        
        print("\n4. Running CCEA Experiment...")
        results_ccea = run_ccea_experiment(
            env_name="hopper-medium-v2",
            policy_checkpoint="./checkpoints/cql_hopper_medium.pt",
            shift_configs=shift_configs,
            num_episodes=20,
            results_csv_path="./ccea_results.csv"
        )
        
        print("\n5. Running TARL Experiment...")
        results_tarl = run_tarl_experiment(
            env_name="hopper-medium-v2",
            policy_checkpoint="./checkpoints/cql_hopper_medium.pt",
            shift_configs=shift_configs,
            num_episodes=20,
            results_csv_path="./tarl_results.csv"
        )
        
        print("\n" + "=" * 80)
        print("All experiments completed successfully!")
        print("Results saved to CSV files:")
        print("  - ./tta_results_entropy.csv")
        print("  - ./tta_results_uncertainty.csv")
        print("  - ./tta_results_no_tta.csv")
        print("  - ./ccea_results.csv")
        print("  - ./tarl_results.csv")
        print("=" * 80)
        
    except Exception as e:
        print(f"Error running experiments: {e}")
        print("Make sure you have the required checkpoints and dependencies.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OfflineRL-Kit TTA Framework CLI")
    parser.add_argument("--enable_tta", action="store_true", help="Enable Test-Time Adaptation (TTA)")
    parser.add_argument("--tta_strategy", type=str, choices=["entropy_minimization", "uncertainty_minimization", "ccea", "tarl", "none"], default="entropy_minimization", help="TTA strategy to use")
    parser.add_argument("--env", type=str, default="hopper-medium-v2", help="Environment name (e.g., hopper-medium-v2)")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/cql_hopper_medium.pt", help="Path to policy checkpoint file")
    parser.add_argument("--episodes", type=int, default=20, help="Number of episodes per experiment")
    parser.add_argument("--output", type=str, default="./tta_results.csv", help="Output CSV file path for results")
    parser.add_argument("--shifts", type=str, nargs="+", default=["light_mass", "heavy_mass", "low_friction", "observation_noise"], help="List of shift configurations to test")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for TTA adaptation")

    args = parser.parse_args()

    print("=" * 80)
    print("OfflineRL-Kit TTA Framework CLI")
    print("=" * 80)
    print(f"Environment: {args.env}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Episodes: {args.episodes}")
    print(f"TTA Enabled: {args.enable_tta}")
    print(f"TTA Strategy: {args.tta_strategy}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Output: {args.output}")
    print(f"Shifts: {args.shifts}")

    shift_configs = {}
    if "normal" in args.shifts:
        shift_configs["normal"] = {}
    if "light_mass" in args.shifts:
        shift_configs["light_mass"] = {"dynamic_shifts": {"mass_scale": 0.8}}
    if "heavy_mass" in args.shifts:
        shift_configs["heavy_mass"] = {"dynamic_shifts": {"mass_scale": 1.5}}
    if "low_friction" in args.shifts:
        shift_configs["low_friction"] = {"dynamic_shifts": {"friction_scale": 0.5}}
    if "observation_noise" in args.shifts:
        shift_configs["observation_noise"] = {"observation_shifts": {"noise_std": 0.1}}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    try:
        if args.tta_strategy == "ccea":
            print(f"\nRunning CCEA Experiment...")
            results = run_ccea_experiment(
                env_name=args.env,
                policy_checkpoint=args.checkpoint,
                shift_configs=shift_configs,
                num_episodes=args.episodes,
                results_csv_path=args.output
            )
        elif args.tta_strategy == "tarl":
            print(f"\nRunning TARL Experiment...")
            results = run_tarl_experiment(
                env_name=args.env,
                policy_checkpoint=args.checkpoint,
                shift_configs=shift_configs,
                num_episodes=args.episodes,
                results_csv_path=args.output
            )
        else:
            print(f"\nRunning TTA Experiment with strategy: {args.tta_strategy}")
            results = run_tta_experiment(
                env_name=args.env,
                policy_checkpoint=args.checkpoint,
                shift_configs=shift_configs,
                enable_tta=args.enable_tta,
                tta_strategy=args.tta_strategy,
                num_episodes=args.episodes,
                results_csv_path=args.output,
                learning_rate=args.learning_rate
            )

        print("\n" + "=" * 80)
        print("Experiment completed successfully!")
        print(f"Results saved to: {args.output}")
        print("=" * 80)

    except Exception as e:
        print(f"Error running experiment: {e}")
        print("Make sure you have the required checkpoints and dependencies.")
