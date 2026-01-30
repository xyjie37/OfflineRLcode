"""
通用TTA运行器使用示例

展示如何使用通用运行器运行多个TTA算法
"""

from offlinerlkit.tta import run_tta_algorithm, compare_algorithms


def example_single_algorithm():
    """示例：运行单个算法"""
    shift_configs = {
        'normal': {},
        'light_mass': {'dynamic_shifts': {'mass_scale': 0.8}},
        'heavy_mass': {'dynamic_shifts': {'mass_scale': 1.5}},
        'low_friction': {'dynamic_shifts': {'friction_scale': 0.5}},
        'observation_noise': {'observation_shifts': {'noise_std': 0.1}}
    }
    
    # 运行 TEA 算法
    results = run_tta_algorithm(
        algorithm_name='tea',
        env_name='hopper-medium-v2',
        policy_checkpoint='./checkpoints/cql_hopper_medium.pt',
        shift_configs=shift_configs,
        num_episodes=20,
        results_csv_path='./tea_results.csv',
        algorithm_config={
            'learning_rate': 1e-6,
            'sgld_step_size': 0.1,
            'sgld_steps': 10,
            'num_neg_samples': 10,
            'kl_weight': 1.0,
            'action_space': 'continuous',
            'cache_capacity': 1000,
            'update_freq': 10,
            'adaptation_mode': 'layernorm'
        },
        num_seeds=3
    )
    
    print(f"\nTEA experiment completed! Total results: {len(results)}")


def example_compare_algorithms():
    """示例：对比多个算法"""
    shift_configs = {
        'normal': {},
        'light_mass': {'dynamic_shifts': {'mass_scale': 0.8}},
        'heavy_mass': {'dynamic_shifts': {'mass_scale': 1.5}}
    }
    
    # 对比多个算法
    comparison_summary = compare_algorithms(
        algorithm_names=['stint', 'tarl', 'tea'],
        env_name='hopper-medium-v2',
        policy_checkpoint='./checkpoints/cql_hopper_medium.pt',
        shift_configs=shift_configs,
        num_episodes=20,
        results_dir='./tta_comparison_results',
        num_seeds=3
    )
    
    print(f"\nAlgorithm comparison completed!")


def example_custom_algorithm():
    """示例：使用自定义配置运行算法"""
    shift_configs = {
        'normal': {}
    }
    
    # 使用自定义配置运行 STINT
    results = run_tta_algorithm(
        algorithm_name='stint',
        env_name='hopper-medium-v2',
        policy_checkpoint='./checkpoints/cql_hopper_medium.pt',
        shift_configs=shift_configs,
        num_episodes=20,
        results_csv_path='./stint_custom_results.csv',
        algorithm_config={
            'delta': 0.8,  # 自定义触发阈值
            'lambda_kl': 2.0,  # 更强的KL约束
            'K': 5,  # 每次触发更多更新步数
            'beta': 0.05,  # 更小的熵平滑系数
            'learning_rate': 5e-5,  # 更小的学习率
            'adaptation_mode': 'layernorm'  # 仅更新LayerNorm
        },
        num_seeds=5
    )
    
    print(f"\nCustom STINT experiment completed!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="通用TTA运行器示例")
    parser.add_argument("--mode", type=str, choices=['single', 'compare', 'custom'], 
                      default='single', help="运行模式")
    parser.add_argument("--algorithm", type=str, choices=['stint', 'tarl', 'tea', 'ccea', 'come'], 
                      default='tea', help="算法名称（single模式）")
    parser.add_argument("--env", type=str, default='hopper-medium-v2', help="环境名称")
    parser.add_argument("--checkpoint", type=str, default='./checkpoints/cql_hopper_medium.pt', 
                      help="策略检查点路径")
    parser.add_argument("--episodes", type=int, default=20, help="每个seed的episode数")
    parser.add_argument("--seeds", type=int, default=3, help="随机种子数")
    parser.add_argument("--output", type=str, default='./tta_results.csv', help="输出CSV路径")
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        print(f"Running single algorithm: {args.algorithm}")
        shift_configs = {
            'normal': {},
            'light_mass': {'dynamic_shifts': {'mass_scale': 0.8}},
            'heavy_mass': {'dynamic_shifts': {'mass_scale': 1.5}}
        }
        
        results = run_tta_algorithm(
            algorithm_name=args.algorithm,
            env_name=args.env,
            policy_checkpoint=args.checkpoint,
            shift_configs=shift_configs,
            num_episodes=args.episodes,
            results_csv_path=args.output,
            num_seeds=args.seeds
        )
        
    elif args.mode == 'compare':
        print("Running algorithm comparison")
        shift_configs = {
            'normal': {},
            'light_mass': {'dynamic_shifts': {'mass_scale': 0.8}},
            'heavy_mass': {'dynamic_shifts': {'mass_scale': 1.5}}
        }
        
        comparison_summary = compare_algorithms(
            algorithm_names=['stint', 'tarl', 'tea'],
            env_name=args.env,
            policy_checkpoint=args.checkpoint,
            shift_configs=shift_configs,
            num_episodes=args.episodes,
            results_dir='./tta_comparison_results',
            num_seeds=args.seeds
        )
        
    elif args.mode == 'custom':
        print(f"Running custom {args.algorithm}")
        shift_configs = {
            'normal': {}
        }
        
        results = run_tta_algorithm(
            algorithm_name=args.algorithm,
            env_name=args.env,
            policy_checkpoint=args.checkpoint,
            shift_configs=shift_configs,
            num_episodes=args.episodes,
            results_csv_path=args.output,
            algorithm_config={
                'learning_rate': 1e-5,
                'adaptation_mode': 'layernorm'
            },
            num_seeds=args.seeds
        )
