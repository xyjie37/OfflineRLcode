# TTA实验框架使用指南

本指南介绍了如何使用TTA（Test-Time Adaptation）实验框架进行完整的实验，包括可开关的TTA策略、简单的proxy更新、环境shift强度扫描以及多种指标收集。

## 功能概述

### 1. 可开关的TTA策略
- 通过 `enable_tta` 参数控制是否启用TTA
- 支持多种适应策略：在线微调、元学习、基于经验的适应、熵最小化、不确定性最小化

### 2. 简单的Proxy更新
- **熵最小化**：最小化策略熵，使策略更加确定性
- **不确定性最小化**：最小化Q值的不确定性（使用双critic的差异）

### 3. 环境Shift强度扫描
- 支持4档强度的shift配置
- 支持多种shift类型：质量、摩擦系数、观测噪声、奖励scale、组合shift

### 4. 指标收集
- **Average Return**：平均回报
- **Worst Case Return**：最坏情况回报（跨seed和shift）
- **Collapse Rate**：崩溃率（回报连续低于阈值N个episode）
- **Policy Divergence**：策略漂移范数（参数变化）

### 5. 有TTA和无TTA的指标对比
- 自动计算TTA相对于baseline的改进
- 生成详细的对比报告

## 快速开始

### 基本使用示例

```python
from offlinerlkit.tta.shifted_env import create_shifted_env, create_shift_intensity_configs
from offlinerlkit.tta.tta_manager import TTAManager

# 创建shift环境
shift_configs = create_shift_intensity_configs('mass', num_levels=4)
shift_config = shift_configs['mass_level_1']

env = create_shifted_env('hopper-medium-v2', shift_config)

# 创建TTA管理器（启用TTA）
adaptation_config = {
    'enable_tta': True,
    'strategy': 'entropy_minimization',
    'learning_rate': 1e-5,
    'batch_size': 32,
    'collapse_threshold': -100.0,
    'collapse_window': 5
}

tta_manager = TTAManager(policy, env, adaptation_config)

# 运行适应
adaptation_data, metrics = tta_manager.run_adaptation(num_episodes=20)

# 查看结果
print(f"Average Return: {metrics['mean_reward']:.2f}")
print(f"Worst Case Return: {metrics['worst_case_return']:.2f}")
print(f"Collapse Rate: {metrics['collapse_rate']:.2%}")
print(f"Policy Divergence: {metrics['mean_policy_divergence']:.6f}")
```

### 运行完整实验

使用实验运行器进行完整的对比实验：

```bash
python -m offlinerlkit.tta.experiment_runner \
    --env_name hopper-medium-v2 \
    --policy_checkpoint ./checkpoints/cql_hopper.pt \
    --policy_class CQLPolicy \
    --shift_types mass friction \
    --num_seeds 3 \
    --num_episodes 20 \
    --results_dir ./tta_experiment_results
```

## 配置说明

### TTA配置参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable_tta` | bool | True | 是否启用TTA |
| `strategy` | str | 'online_finetune' | 适应策略 |
| `learning_rate` | float | 1e-5 | 学习率 |
| `batch_size` | int | 32 | 批次大小 |
| `collapse_threshold` | float | -100.0 | 崩溃阈值 |
| `collapse_window` | int | 5 | 崩溃检测窗口 |

### 适应策略选项

- `online_finetune`：在线微调
- `meta_learning`：元学习适应
- `experience_based`：基于经验的适应
- `entropy_minimization`：熵最小化
- `uncertainty_minimization`：不确定性最小化

### Shift类型选项

- `mass`：质量shift
- `friction`：摩擦系数shift
- `observation`：观测噪声shift
- `reward`：奖励scale shift
- `combined`：组合shift

## 实验流程

### 1. 单个Shift配置对比

```python
from offlinerlkit.tta.shifted_env import create_shift_intensity_configs

# 获取4档强度的shift配置
shift_configs = create_shift_intensity_configs('mass', num_levels=4)

for shift_name, shift_config in shift_configs.items():
    # 有TTA
    adaptation_config_with = {
        'enable_tta': True,
        'strategy': 'entropy_minimization',
        'learning_rate': 1e-5,
        'batch_size': 32
    }
    tta_manager_with = TTAManager(policy, env, adaptation_config_with)
    _, metrics_with = tta_manager_with.run_adaptation(num_episodes=20)
    
    # 无TTA
    adaptation_config_without = {
        'enable_tta': False,
        'strategy': 'entropy_minimization',
        'learning_rate': 1e-5,
        'batch_size': 32
    }
    tta_manager_without = TTAManager(policy, env, adaptation_config_without)
    _, metrics_without = tta_manager_without.run_adaptation(num_episodes=20)
    
    # 计算改进
    improvement = metrics_with['mean_reward'] - metrics_without['mean_reward']
    print(f"{shift_name}: Improvement = {improvement:.2f}")
```

### 2. 多种子实验

```python
num_seeds = 3
results_with_tta = []
results_without_tta = []

for seed in range(num_seeds):
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 创建新策略
    policy = create_policy()
    
    # 有TTA
    tta_manager_with = TTAManager(policy, env, adaptation_config_with)
    _, metrics_with = tta_manager_with.run_adaptation(num_episodes=20)
    results_with_tta.append(metrics_with)
    
    # 无TTA
    tta_manager_without = TTAManager(policy, env, adaptation_config_without)
    _, metrics_without = tta_manager_without.run_adaptation(num_episodes=20)
    results_without_tta.append(metrics_without)

# 聚合结果
avg_with_tta = np.mean([r['mean_reward'] for r in results_with_tta])
avg_without_tta = np.mean([r['mean_reward'] for r in results_without_tta])
worst_case_with = np.min([r['worst_case_return'] for r in results_with_tta])
worst_case_without = np.min([r['worst_case_return'] for r in results_without_tta])
collapse_rate_with = np.mean([r['collapse_rate'] for r in results_with_tta])
collapse_rate_without = np.mean([r['collapse_rate'] for r in results_without_tta])

print(f"Average Return: {avg_with_tta:.2f} vs {avg_without_tta:.2f}")
print(f"Worst Case Return: {worst_case_with:.2f} vs {worst_case_without:.2f}")
print(f"Collapse Rate: {collapse_rate_with:.2%} vs {collapse_rate_without:.2%}")
```

### 3. 完整实验流程

使用 `TTAExperiment` 类进行完整的实验：

```python
from offlinerlkit.tta.experiment_runner import TTAExperiment

experiment = TTAExperiment(
    env_name='hopper-medium-v2',
    policy_checkpoint='./checkpoints/cql_hopper.pt',
    policy_class=CQLPolicy,
    results_dir='./tta_experiment_results'
)

comparison_report = experiment.run_full_experiment(
    shift_types=['mass', 'friction'],
    num_seeds=3,
    num_episodes_per_seed=20,
    adaptation_strategies=['entropy_minimization', 'uncertainty_minimization']
)

print(f"Mean Improvement: {comparison_report['overall_summary']['mean_improvement']:.2f}")
print(f"Max Improvement: {comparison_report['overall_summary']['max_improvement']:.2f}")
```

## 结果输出

### 指标说明

1. **Average Return**：所有episode的平均回报
2. **Worst Case Return**：所有episode中的最低回报
3. **Collapse Rate**：回报连续低于阈值的比例
4. **Policy Divergence**：策略参数相对于初始状态的漂移范数

### 输出文件

实验会生成以下文件：

- `tta_experiment_results_<timestamp>.json`：完整的实验结果
- `tta_comparison_report_<timestamp>.json`：对比报告摘要

### 示例输出

```
================================================================================
Summary for mass_level_1 with entropy_minimization
================================================================================
Without TTA:
  Average Return: 1234.56 ± 45.67
  Worst Case Return: 1100.23
  Collapse Rate: 0.00%

With TTA:
  Average Return: 1345.78 ± 38.91
  Worst Case Return: 1250.45
  Collapse Rate: 0.00%
  Policy Divergence: 0.001234

Improvement:
  Average Return: 111.22 (9.01%)
  Worst Case Return: 150.22
  Collapse Rate Reduction: 0.00%
================================================================================
```

## 示例脚本

### 演示脚本

运行演示脚本查看完整示例：

```bash
python examples/tta_experiment_demo.py
```

### 实验脚本

运行完整实验：

```bash
python -m offlinerlkit.tta.experiment_runner \
    --env_name hopper-medium-v2 \
    --policy_checkpoint ./checkpoints/cql_hopper.pt \
    --policy_class CQLPolicy \
    --shift_types mass \
    --num_seeds 3 \
    --num_episodes 20
```

## 注意事项

1. **Collapse检测**：默认阈值为-100.0，窗口为5个episode，可根据具体环境调整
2. **学习率**：TTA的学习率通常比训练时小，建议使用1e-5到1e-4
3. **Shift强度**：4档强度从轻到重，可根据具体环境调整
4. **随机种子**：建议使用多个随机种子（3-5个）以确保结果可靠性
5. **策略兼容性**：确保策略类实现了 `select_action` 方法和 `parameters` 方法

## 扩展功能

### 添加新的适应策略

在 `TTAManager` 类中添加新的方法：

```python
def _my_custom_strategy(self, episode_data: Dict[str, Any]):
    """自定义适应策略"""
    # 实现你的策略
    pass
```

然后在 `run_adaptation` 方法中添加对应的分支。

### 添加新的Shift类型

在 `create_shift_intensity_configs` 函数中添加新的shift类型：

```python
elif base_shift_type == 'my_custom_shift':
    # 实现4档强度的自定义shift
    pass
```

## 常见问题

### Q: 如何禁用TTA？
A: 在 `adaptation_config` 中设置 `enable_tta=False`。

### Q: 如何调整collapse检测阈值？
A: 在 `adaptation_config` 中设置 `collapse_threshold` 和 `collapse_window` 参数。

### Q: 如何使用真实的策略检查点？
A: 使用 `ModelLoader` 类加载真实的策略检查点：

```python
from offlinerlkit.tta.model_loader import ModelLoader

model_loader = ModelLoader(policy_class, device='cuda')
policy = model_loader.load_pretrained_model(checkpoint_path, env_config)
```

### Q: 如何可视化实验结果？
A: 可以使用matplotlib或其他可视化工具绘制结果图表，参考 `ShiftedPolicyEvaluator.plot_results` 方法。

## 参考资料

- [TTA管理器](file:///Users/jerryjie/Documents/NeurIPS/OfflineRL-Kit-main/offlinerlkit/tta/tta_manager.py)
- [Shift环境](file:///Users/jerryjie/Documents/NeurIPS/OfflineRL-Kit-main/offlinerlkit/tta/shifted_env.py)
- [实验运行器](file:///Users/jerryjie/Documents/NeurIPS/OfflineRL-Kit-main/offlinerlkit/tta/experiment_runner.py)
- [演示脚本](file:///Users/jerryjie/Documents/NeurIPS/OfflineRL-Kit-main/examples/tta_experiment_demo.py)
