#!/usr/bin/env python3
"""
完整的TTA实验脚本
实现：
1. 可开关的TTA策略
2. 简单的proxy更新（熵最小化或不确定性最小化）
3. 环境shift强度扫描（4档强度）
4. 指标收集：Average return, Worst case return, Collapse rate, Policy divergence
5. 有TTA和无TTA的指标对比
"""

import torch
import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, Any, List
import argparse

from offlinerlkit.tta.shifted_env import create_shifted_env, create_shift_intensity_configs, get_all_shift_types
from offlinerlkit.tta.tta_manager import TTAManager
from offlinerlkit.tta.model_loader import ModelLoader


class TTAExperiment:
    """TTA实验管理器"""
    
    def __init__(self, env_name: str, policy_checkpoint: str, policy_class,
                 results_dir: str = './tta_experiment_results'):
        self.env_name = env_name
        self.policy_checkpoint = policy_checkpoint
        self.policy_class = policy_class
        self.results_dir = results_dir
        
        # 创建结果目录
        os.makedirs(results_dir, exist_ok=True)
        
        # 实验结果存储
        self.experiment_results = {}
        
    def run_full_experiment(self, shift_types: List[str] = None, 
                           num_seeds: int = 3,
                           num_episodes_per_seed: int = 20,
                           adaptation_strategies: List[str] = None) -> Dict[str, Any]:
        """
        运行完整实验
        
        Args:
            shift_types: 要测试的shift类型列表
            num_seeds: 随机种子数量
            num_episodes_per_seed: 每个种子的episode数量
            adaptation_strategies: 适应策略列表
            
        Returns:
            完整的实验结果
        """
        if shift_types is None:
            shift_types = ['mass']  # 默认测试质量shift
        if adaptation_strategies is None:
            adaptation_strategies = ['entropy_minimization', 'uncertainty_minimization']
        
        print("=" * 80)
        print("TTA Experiment Configuration")
        print("=" * 80)
        print(f"Environment: {self.env_name}")
        print(f"Policy checkpoint: {self.policy_checkpoint}")
        print(f"Shift types: {shift_types}")
        print(f"Adaptation strategies: {adaptation_strategies}")
        print(f"Number of seeds: {num_seeds}")
        print(f"Episodes per seed: {num_episodes_per_seed}")
        print("=" * 80)
        
        # 对每个shift类型进行实验
        for shift_type in shift_types:
            print(f"\n{'='*80}")
            print(f"Testing shift type: {shift_type}")
            print(f"{'='*80}")
            
            shift_type_results = self._run_shift_type_experiment(
                shift_type, num_seeds, num_episodes_per_seed, adaptation_strategies
            )
            self.experiment_results[shift_type] = shift_type_results
        
        # 保存完整结果
        self._save_experiment_results()
        
        # 生成对比报告
        comparison_report = self._generate_comparison_report()
        
        return comparison_report
    
    def _run_shift_type_experiment(self, shift_type: str, num_seeds: int,
                                   num_episodes_per_seed: int,
                                   adaptation_strategies: List[str]) -> Dict[str, Any]:
        """运行单个shift类型的实验"""
        shift_type_results = {}
        
        # 获取4档强度的shift配置
        shift_configs = create_shift_intensity_configs(shift_type, num_levels=4)
        
        # 对每个强度档位进行实验
        for shift_name, shift_config in shift_configs.items():
            print(f"\n{'-'*80}")
            print(f"Testing shift: {shift_name}")
            print(f"Shift config: {shift_config}")
            print(f"{'-'*80}")
            
            shift_level_results = self._run_shift_level_experiment(
                shift_name, shift_config, num_seeds, num_episodes_per_seed, adaptation_strategies
            )
            shift_type_results[shift_name] = shift_level_results
        
        return shift_type_results
    
    def _run_shift_level_experiment(self, shift_name: str, shift_config: Dict[str, Any],
                                   num_seeds: int, num_episodes_per_seed: int,
                                   adaptation_strategies: List[str]) -> Dict[str, Any]:
        """运行单个强度档位的实验"""
        shift_level_results = {}
        
        # 对每个适应策略进行实验
        for strategy in adaptation_strategies:
            print(f"\nTesting adaptation strategy: {strategy}")
            
            # 运行有TTA的实验
            tta_results = self._run_single_experiment(
                shift_config, strategy, num_seeds, num_episodes_per_seed, enable_tta=True
            )
            
            # 运行无TTA的实验（baseline）
            baseline_results = self._run_single_experiment(
                shift_config, strategy, num_seeds, num_episodes_per_seed, enable_tta=False
            )
            
            # 计算改进
            improvement = self._compute_improvement(tta_results, baseline_results)
            
            shift_level_results[strategy] = {
                'with_tta': tta_results,
                'without_tta': baseline_results,
                'improvement': improvement
            }
            
            # 打印摘要
            self._print_experiment_summary(shift_name, strategy, tta_results, baseline_results, improvement)
        
        return shift_level_results
    
    def _run_single_experiment(self, shift_config: Dict[str, Any], strategy: str,
                               num_seeds: int, num_episodes_per_seed: int,
                               enable_tta: bool) -> Dict[str, Any]:
        """运行单个实验配置"""
        all_seed_results = []
        
        for seed in range(num_seeds):
            # 设置随机种子
            np.random.seed(seed)
            torch.manual_seed(seed)
            
            # 创建shift环境
            env = create_shifted_env(self.env_name, shift_config)
            
            # 加载策略
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model_loader = ModelLoader(self.policy_class, device=device)
            env_config = {
                'observation_space': env.observation_space,
                'action_space': env.action_space,
                'obs_dim': env.observation_space.shape[0],
                'action_dim': env.action_space.shape[0]
            }
            policy = model_loader.load_pretrained_model(self.policy_checkpoint, env_config)
            
            # 创建适应配置
            adaptation_config = {
                'enable_tta': enable_tta,
                'strategy': strategy,
                'learning_rate': 1e-5,
                'batch_size': 32,
                'collapse_threshold': -100.0,
                'collapse_window': 5
            }
            
            # 创建TTA管理器
            tta_manager = TTAManager(policy, env, adaptation_config)
            
            # 运行适应
            adaptation_data, metrics_summary = tta_manager.run_adaptation(
                num_episodes=num_episodes_per_seed
            )
            
            # 收集结果
            seed_result = {
                'seed': seed,
                'adaptation_data': adaptation_data,
                'metrics_summary': metrics_summary,
                'enable_tta': enable_tta
            }
            all_seed_results.append(seed_result)
        
        # 聚合所有种子的结果
        aggregated_results = self._aggregate_seed_results(all_seed_results)
        
        return aggregated_results
    
    def _aggregate_seed_results(self, seed_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """聚合多个种子的结果"""
        aggregated = {
            'num_seeds': len(seed_results),
            'mean_reward_across_seeds': [],
            'worst_case_return_across_seeds': [],
            'collapse_rate_across_seeds': [],
            'mean_policy_divergence_across_seeds': [],
            'std_reward_across_seeds': []
        }
        
        for seed_result in seed_results:
            metrics = seed_result['metrics_summary']
            aggregated['mean_reward_across_seeds'].append(metrics.get('mean_reward', 0))
            aggregated['worst_case_return_across_seeds'].append(metrics.get('worst_case_return', 0))
            aggregated['collapse_rate_across_seeds'].append(metrics.get('collapse_rate', 0))
            aggregated['mean_policy_divergence_across_seeds'].append(metrics.get('mean_policy_divergence', 0))
            aggregated['std_reward_across_seeds'].append(metrics.get('std_reward', 0))
        
        # 计算统计量
        aggregated['average_return'] = np.mean(aggregated['mean_reward_across_seeds'])
        aggregated['std_return'] = np.std(aggregated['mean_reward_across_seeds'])
        aggregated['worst_case_return'] = np.min(aggregated['worst_case_return_across_seeds'])
        aggregated['collapse_rate'] = np.mean(aggregated['collapse_rate_across_seeds'])
        aggregated['mean_policy_divergence'] = np.mean(aggregated['mean_policy_divergence_across_seeds'])
        
        return aggregated
    
    def _compute_improvement(self, tta_results: Dict[str, Any], 
                            baseline_results: Dict[str, Any]) -> Dict[str, float]:
        """计算TTA相对于baseline的改进"""
        improvement = {
            'average_return_improvement': tta_results['average_return'] - baseline_results['average_return'],
            'average_return_improvement_ratio': (tta_results['average_return'] / 
                                                max(baseline_results['average_return'], 1e-6)),
            'worst_case_return_improvement': tta_results['worst_case_return'] - baseline_results['worst_case_return'],
            'collapse_rate_reduction': baseline_results['collapse_rate'] - tta_results['collapse_rate'],
            'policy_divergence': tta_results['mean_policy_divergence']
        }
        return improvement
    
    def _print_experiment_summary(self, shift_name: str, strategy: str,
                                 tta_results: Dict[str, Any], baseline_results: Dict[str, Any],
                                 improvement: Dict[str, float]):
        """打印实验摘要"""
        print(f"\n{'='*80}")
        print(f"Summary for {shift_name} with {strategy}")
        print(f"{'='*80}")
        print(f"Without TTA:")
        print(f"  Average Return: {baseline_results['average_return']:.2f} ± {baseline_results['std_return']:.2f}")
        print(f"  Worst Case Return: {baseline_results['worst_case_return']:.2f}")
        print(f"  Collapse Rate: {baseline_results['collapse_rate']:.2%}")
        print(f"\nWith TTA:")
        print(f"  Average Return: {tta_results['average_return']:.2f} ± {tta_results['std_return']:.2f}")
        print(f"  Worst Case Return: {tta_results['worst_case_return']:.2f}")
        print(f"  Collapse Rate: {tta_results['collapse_rate']:.2%}")
        print(f"  Policy Divergence: {improvement['policy_divergence']:.6f}")
        print(f"\nImprovement:")
        print(f"  Average Return: {improvement['average_return_improvement']:.2f} "
              f"({improvement['average_return_improvement_ratio']:.2%})")
        print(f"  Worst Case Return: {improvement['worst_case_return_improvement']:.2f}")
        print(f"  Collapse Rate Reduction: {improvement['collapse_rate_reduction']:.2%}")
        print(f"{'='*80}\n")
    
    def _save_experiment_results(self):
        """保存实验结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存完整结果
        results_filename = f"tta_experiment_results_{timestamp}.json"
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
        
        serializable_results = convert_numpy_types(self.experiment_results)
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\nExperiment results saved to: {results_path}")
    
    def _generate_comparison_report(self) -> Dict[str, Any]:
        """生成对比报告"""
        report = {
            'overall_summary': {},
            'shift_type_comparisons': {},
            'strategy_comparisons': {},
            'best_configurations': {}
        }
        
        # 总体摘要
        all_improvements = []
        all_collapse_reductions = []
        
        for shift_type, shift_results in self.experiment_results.items():
            for shift_name, shift_level_results in shift_results.items():
                for strategy, results in shift_level_results.items():
                    improvement = results['improvement']
                    all_improvements.append(improvement['average_return_improvement'])
                    all_collapse_reductions.append(improvement['collapse_rate_reduction'])
        
        if all_improvements:
            report['overall_summary'] = {
                'mean_improvement': np.mean(all_improvements),
                'std_improvement': np.std(all_improvements),
                'max_improvement': np.max(all_improvements),
                'min_improvement': np.min(all_improvements),
                'mean_collapse_reduction': np.mean(all_collapse_reductions)
            }
        
        # Shift类型对比
        for shift_type, shift_results in self.experiment_results.items():
            shift_improvements = []
            for shift_name, shift_level_results in shift_results.items():
                for strategy, results in shift_level_results.items():
                    shift_improvements.append(results['improvement']['average_return_improvement'])
            
            report['shift_type_comparisons'][shift_type] = {
                'mean_improvement': np.mean(shift_improvements),
                'std_improvement': np.std(shift_improvements)
            }
        
        # 策略对比
        strategy_data = {}
        for shift_type, shift_results in self.experiment_results.items():
            for shift_name, shift_level_results in shift_results.items():
                for strategy, results in shift_level_results.items():
                    if strategy not in strategy_data:
                        strategy_data[strategy] = []
                    strategy_data[strategy].append(results['improvement']['average_return_improvement'])
        
        for strategy, improvements in strategy_data.items():
            report['strategy_comparisons'][strategy] = {
                'mean_improvement': np.mean(improvements),
                'std_improvement': np.std(improvements)
            }
        
        # 保存对比报告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"tta_comparison_report_{timestamp}.json"
        report_path = os.path.join(self.results_dir, report_filename)
        
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
        
        serializable_report = convert_numpy_types(report)
        
        with open(report_path, 'w') as f:
            json.dump(serializable_report, f, indent=2)
        
        print(f"Comparison report saved to: {report_path}")
        
        return report


def main():
    parser = argparse.ArgumentParser(description='TTA Experiment')
    parser.add_argument('--env_name', type=str, default='hopper-medium-v2',
                       help='Environment name')
    parser.add_argument('--policy_checkpoint', type=str, required=True,
                       help='Path to policy checkpoint')
    parser.add_argument('--policy_class', type=str, required=True,
                       help='Policy class name (e.g., CQLPolicy)')
    parser.add_argument('--shift_types', type=str, nargs='+', default=['mass'],
                       help='Shift types to test')
    parser.add_argument('--num_seeds', type=int, default=3,
                       help='Number of random seeds')
    parser.add_argument('--num_episodes', type=int, default=20,
                       help='Number of episodes per seed')
    parser.add_argument('--results_dir', type=str, default='./tta_experiment_results',
                       help='Results directory')
    
    args = parser.parse_args()
    
    # 动态导入策略类
    from offlinerlkit.policy import model_free
    policy_class = getattr(model_free, args.policy_class)
    
    # 创建实验
    experiment = TTAExperiment(
        env_name=args.env_name,
        policy_checkpoint=args.policy_checkpoint,
        policy_class=policy_class,
        results_dir=args.results_dir
    )
    
    # 运行实验
    adaptation_strategies = ['entropy_minimization', 'uncertainty_minimization']
    comparison_report = experiment.run_full_experiment(
        shift_types=args.shift_types,
        num_seeds=args.num_seeds,
        num_episodes_per_seed=args.num_episodes,
        adaptation_strategies=adaptation_strategies
    )
    
    # 打印最终摘要
    print("\n" + "="*80)
    print("FINAL EXPERIMENT SUMMARY")
    print("="*80)
    print(f"Mean Improvement: {comparison_report['overall_summary']['mean_improvement']:.2f} ± "
          f"{comparison_report['overall_summary']['std_improvement']:.2f}")
    print(f"Max Improvement: {comparison_report['overall_summary']['max_improvement']:.2f}")
    print(f"Mean Collapse Reduction: {comparison_report['overall_summary']['mean_collapse_reduction']:.2%}")
    print("="*80)


if __name__ == "__main__":
    main()
