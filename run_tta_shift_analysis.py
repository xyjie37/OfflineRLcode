#!/usr/bin/env python3
"""
TTA Shift Analysis Script
测试各种偏移程度下有TTA和无TTA的效果
支持TARL, CCEA, 熵最小化, 不确定性最小化等策略
"""

import argparse
import os
import sys
from typing import Dict, Any, List

import numpy as np
import torch

from offlinerlkit.tta.shifted_env import create_custom_mass_shift_configs
from offlinerlkit.policy import CQLPolicy
from offlinerlkit.tta.stint import STINTManager


def parse_args():
    parser = argparse.ArgumentParser(description='TTA Shift Analysis Script')
    
    parser.add_argument('--env', type=str, default='hopper-medium-v2',
                        help='Environment name')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Policy checkpoint path')
    parser.add_argument('--episodes', type=int, default=20,
                        help='Number of episodes per experiment')
    parser.add_argument('--results_dir', type=str, default='./analysis',
                        help='Results directory')
    
    parser.add_argument('--mass_scales', type=float, nargs='+',
                        default=[1.0, 0.9, 0.8, 0.7, 0.6, 1.1, 1.25, 1.5, 1.75],
                        help='Mass scale factors (default: 9 levels from 0.6 to 1.75)')
    
    parser.add_argument('--strategies', type=str, nargs='+',
                        default=['tarl', 'ccea', 'stint', 'entropy_minimization', 'uncertainty_minimization'],
                        choices=['tarl', 'ccea', 'stint', 'entropy_minimization', 'uncertainty_minimization'],
                        help='TTA strategies to test')
    
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2],
                        help='Random seeds for experiments')
    
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                        help='Learning rate for TTA')
    
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu)')
    
    return parser.parse_args()


def run_single_strategy(args, strategy: str, shift_configs: Dict[str, Dict[str, Any]],
                        enable_tta: bool) -> List[Dict]:
    """运行单个策略的实验"""
    from offlinerlkit.tta.shifted_env import create_shifted_env
    from offlinerlkit.tta.model_loader import ModelLoader
    from offlinerlkit.tta.tta_manager import TTAManager
    from offlinerlkit.tta.tarl import TARLManager
    from offlinerlkit.tta.mcatta import CCEAManager
    
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*80}")
    print(f"Running: {strategy.upper()} {'WITH' if enable_tta else 'WITHOUT'} TTA")
    print(f"{'='*80}")
    
    all_results = []
    
    for shift_name, shift_config in shift_configs.items():
        print(f"\nShift: {shift_name} (label: {shift_config.get('shift_label', 'unknown')}, scale: {shift_config.get('mass_scale', 1.0)})")
        
        for seed in args.seeds:
            np.random.seed(seed)
            torch.manual_seed(seed)
            
            env = create_shifted_env(args.env, shift_config)
            
            model_loader = ModelLoader(CQLPolicy, device=device)
            env_config = {
                'observation_space': env.observation_space,
                'action_space': env.action_space,
                'obs_dim': env.observation_space.shape[0],
                'action_dim': env.action_space.shape[0]
            }
            policy = model_loader.load_pretrained_model(args.checkpoint, env_config)
            
            result_row = {
                'shift_name': shift_name,
                'shift_label': shift_config.get('shift_label', 'unknown'),
                'mass_scale': shift_config.get('mass_scale', 1.0),
                'seed': seed,
                'strategy': strategy,
                'enable_tta': enable_tta
            }
            
            if not enable_tta:
                adaptation_config = {
                    'enable_tta': False,
                    'strategy': strategy,
                    'learning_rate': args.learning_rate,
                    'batch_size': 32,
                    'collapse_threshold': -100.0,
                    'collapse_window': 5
                }
                tta_manager = TTAManager(policy, env, adaptation_config)
                adaptation_data, metrics_summary = tta_manager.run_adaptation(num_episodes=args.episodes)
                
                result_row.update({
                    'mean_reward': metrics_summary.get('mean_reward', 0),
                    'std_reward': metrics_summary.get('std_reward', 0),
                    'max_reward': metrics_summary.get('max_reward', 0),
                    'min_reward': metrics_summary.get('min_reward', 0),
                    'worst_case_return': metrics_summary.get('worst_case_return', 0),
                    'mean_length': metrics_summary.get('mean_length', 0),
                    'mean_policy_divergence': metrics_summary.get('mean_policy_divergence', 0),
                    'collapse_detected': metrics_summary.get('collapse_detected', False),
                    'collapse_rate': metrics_summary.get('collapse_rate', 0)
                })
                
            elif strategy == 'tarl':
                tarl_config = {
                    'learning_rate': args.learning_rate,
                    'cache_capacity': 1000,
                    'k_low_entropy': 10,
                    'kl_weight': 1.0,
                    'gradient_clip': 0.5
                }
                tarl_manager = TARLManager(policy, env, tarl_config)
                adaptation_data, summary = tarl_manager.run_adaptation(num_episodes=args.episodes)
                
                result_row.update({
                    'mean_reward': summary.get('mean_reward', 0),
                    'std_reward': summary.get('std_reward', 0),
                    'mean_length': summary.get('mean_length', 0),
                    'mean_loss': summary.get('mean_loss', 0),
                    'mean_entropy': summary.get('mean_entropy', 0),
                    'mean_kl': summary.get('mean_kl', 0),
                    'cache_size': summary.get('cache_size', 0),
                    'adaptation_steps': summary.get('adaptation_steps', 0),
                    'worst_case_return': summary.get('worst_case_return', 0),
                    'collapse_detected': summary.get('collapse_detected', False),
                    'collapse_rate': summary.get('collapse_rate', 0)
                })
                
            elif strategy == 'ccea':
                ccea_config = {
                    'lambda_min': 0.1,
                    'lambda_max': 10.0,
                    'lambda_init': 1.0,
                    'policy_lr': 1e-4,
                    'batch_size': 32,
                    'pos_cache_capacity': 100,
                    'neg_cache_capacity': 100,
                    'gamma': 0.1,
                    'tau': 1.0,
                    'entropy_low': 0.5,
                    'entropy_high': 2.0,
                    'delta_stable': 0.1,
                    'v_min': 0.1,
                    'adaptation_mode': 'layernorm',
                }
                ccea_manager = CCEAManager(policy, env, ccea_config)
                adaptation_data, summary = ccea_manager.run_adaptation(num_episodes=args.episodes)
                
                result_row.update({
                    'mean_reward': summary.get('mean_reward', 0),
                    'std_reward': summary.get('std_reward', 0),
                    'final_lambda': summary.get('final_lambda', 0),
                    'final_entropy': summary.get('final_entropy', 0),
                    'final_entropy_velocity': summary.get('final_entropy_velocity', 0),
                    'final_contrastive_uncertainty': summary.get('final_contrastive_uncertainty', 0),
                    'lambda_history_mean': np.mean(summary.get('lambda_history', [0])),
                    'lambda_history_std': np.std(summary.get('lambda_history', [0])),
                    'contrastive_uncertainty_history_mean': np.mean(summary.get('contrastive_uncertainty_history', [0])),
                    'contrastive_uncertainty_history_std': np.std(summary.get('contrastive_uncertainty_history', [0])),
                    'worst_case_return': summary.get('worst_case_return', 0),
                    'collapse_detected': summary.get('collapse_detected', False),
                    'collapse_rate': summary.get('collapse_rate', 0)
                })
                
            elif strategy == 'stint':
                stint_config = {
                    'delta': 0.5,
                    'lambda_kl': 1.0,
                    'K': 3,
                    'beta': 0.1,
                    'learning_rate': args.learning_rate,
                    'adaptation_mode': 'layernorm'
                }
                stint_manager = STINTManager(policy, env, stint_config)
                adaptation_data, summary = stint_manager.run_adaptation(num_episodes=args.episodes)
                
                result_row.update({
                    'mean_reward': summary.get('mean_reward', 0),
                    'std_reward': summary.get('std_reward', 0),
                    'mean_length': summary.get('mean_length', 0),
                    'mean_loss': summary.get('mean_loss', 0),
                    'mean_entropy': summary.get('mean_entropy', 0),
                    'mean_kl': summary.get('mean_kl', 0),
                    'total_triggers': summary.get('total_triggers', 0),
                    'mean_triggers_per_episode': summary.get('mean_triggers_per_episode', 0),
                    'adaptation_steps': summary.get('adaptation_steps', 0),
                    'final_entropy_moving_avg': summary.get('final_entropy_moving_avg', 0),
                    'worst_case_return': summary.get('worst_case_return', 0),
                    'collapse_detected': summary.get('collapse_detected', False),
                    'collapse_rate': summary.get('collapse_rate', 0)
                })
                
            elif strategy == 'entropy_minimization':
                lr = args.learning_rate
                for param in policy.actor.parameters():
                    param.requires_grad = True
                policy.adaptation_optimizer = torch.optim.Adam(policy.actor.parameters(), lr=lr)
                
                adaptation_config = {
                    'enable_tta': True,
                    'strategy': 'entropy_minimization',
                    'learning_rate': args.learning_rate,
                    'batch_size': 32,
                    'collapse_threshold': -100.0,
                    'collapse_window': 5
                }
                tta_manager = TTAManager(policy, env, adaptation_config)
                adaptation_data, metrics_summary = tta_manager.run_adaptation(num_episodes=args.episodes)
                
                result_row.update({
                    'mean_reward': metrics_summary.get('mean_reward', 0),
                    'std_reward': metrics_summary.get('std_reward', 0),
                    'max_reward': metrics_summary.get('max_reward', 0),
                    'min_reward': metrics_summary.get('min_reward', 0),
                    'worst_case_return': metrics_summary.get('worst_case_return', 0),
                    'mean_length': metrics_summary.get('mean_length', 0),
                    'mean_policy_divergence': metrics_summary.get('mean_policy_divergence', 0),
                    'collapse_detected': metrics_summary.get('collapse_detected', False),
                    'collapse_rate': metrics_summary.get('collapse_rate', 0)
                })
                
            elif strategy == 'uncertainty_minimization':
                lr = args.learning_rate
                for param in policy.critic1.parameters():
                    param.requires_grad = True
                for param in policy.critic2.parameters():
                    param.requires_grad = True
                policy.adaptation_optimizer = torch.optim.Adam(
                    list(policy.critic1.parameters()) + list(policy.critic2.parameters()), 
                    lr=lr
                )
                
                adaptation_config = {
                    'enable_tta': True,
                    'strategy': 'uncertainty_minimization',
                    'learning_rate': args.learning_rate,
                    'batch_size': 32,
                    'collapse_threshold': -100.0,
                    'collapse_window': 5
                }
                tta_manager = TTAManager(policy, env, adaptation_config)
                adaptation_data, metrics_summary = tta_manager.run_adaptation(num_episodes=args.episodes)
                
                result_row.update({
                    'mean_reward': metrics_summary.get('mean_reward', 0),
                    'std_reward': metrics_summary.get('std_reward', 0),
                    'max_reward': metrics_summary.get('max_reward', 0),
                    'min_reward': metrics_summary.get('min_reward', 0),
                    'worst_case_return': metrics_summary.get('worst_case_return', 0),
                    'mean_length': metrics_summary.get('mean_length', 0),
                    'mean_policy_divergence': metrics_summary.get('mean_policy_divergence', 0),
                    'collapse_detected': metrics_summary.get('collapse_detected', False),
                    'collapse_rate': metrics_summary.get('collapse_rate', 0)
                })
            
            all_results.append(result_row)
            print(f"  Seed {seed}: Reward = {result_row['mean_reward']:.2f} ± {result_row['std_reward']:.2f}")
    
    return all_results


def save_results_to_csv(results: List[Dict], csv_path: str):
    """保存结果到CSV文件"""
    import csv
    
    os.makedirs(os.path.dirname(csv_path) if os.path.dirname(csv_path) else '.', exist_ok=True)
    
    fieldnames = ['shift_name', 'shift_label', 'mass_scale', 'seed', 'strategy', 'enable_tta',
                  'mean_reward', 'std_reward', 'max_reward', 'min_reward',
                  'worst_case_return', 'mean_length', 'mean_policy_divergence',
                  'collapse_detected', 'collapse_rate', 'mean_loss', 'mean_entropy', 'mean_kl',
                  'cache_size', 'adaptation_steps', 'final_lambda', 'final_entropy_velocity',
                  'final_contrastive_uncertainty', 'lambda_history_mean', 'lambda_history_std',
                  'contrastive_uncertainty_history_mean', 'contrastive_uncertainty_history_std',
                  'total_triggers', 'mean_triggers_per_episode', 'final_entropy_moving_avg']
    
    formatted_results = []
    for row in results:
        formatted_row = {}
        for key in fieldnames:
            value = row.get(key, '')
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


def main():
    args = parse_args()
    
    print("=" * 80)
    print("TTA Shift Analysis Script")
    print("=" * 80)
    print(f"Environment: {args.env}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Episodes: {args.episodes}")
    print(f"Mass scales: {args.mass_scales}")
    print(f"Strategies: {args.strategies}")
    print(f"Seeds: {args.seeds}")
    print(f"Results directory: {args.results_dir}")
    print("=" * 80)
    
    os.makedirs(args.results_dir, exist_ok=True)
    
    shift_configs = create_custom_mass_shift_configs(args.mass_scales)
    
    print(f"\nCreated {len(shift_configs)} shift configurations:")
    for shift_name, shift_config in shift_configs.items():
        print(f"  - {shift_name}: mass_scale={shift_config['mass_scale']:.2f}, label={shift_config['shift_label']}")
    
    all_results = []
    
    print(f"\n{'='*80}")
    print(f"Running TTA strategies: {args.strategies}")
    print(f"{'='*80}")
    
    for strategy in args.strategies:
        results = run_single_strategy(args, strategy, shift_configs, enable_tta=True)
        all_results.extend(results)
    
    csv_path = os.path.join(args.results_dir, 'tta_shift_analysis.csv')
    save_results_to_csv(all_results, csv_path)
    
    print("\n" + "=" * 80)
    print("Analysis completed!")
    print("=" * 80)
    print(f"Total experiments run: {len(all_results)}")
    print(f"Results saved to: {csv_path}")


if __name__ == '__main__':
    main()