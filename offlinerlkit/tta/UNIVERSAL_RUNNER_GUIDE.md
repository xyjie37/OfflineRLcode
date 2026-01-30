# 通用TTA运行器使用指南

## 概述

通用TTA运行器提供了一个统一的接口来运行和对比多个TTA算法，包括：
- **STINT**: Stability-Triggered Intervention
- **TARL**: Test-Time Adaptation with Reinforcement Learning
- **TEA**: Test-time Energy Adaptation
- **CCEA**: Contrastive Cache-based Entropic Adaptation

## 核心特性

### 1. 统一的参数更新策略
所有算法默认仅更新 **LayerNorm** 层参数，确保适应过程的稳定性：
```python
config = {
    'adaptation_mode': 'layernorm',  # 默认
    # 其他选项: 'last_n_layers', 'policy_head', 'all'
}
```

### 2. 通用基类 `BaseTTAAlgorithm`
所有TTA算法继承自 `BaseTTAAlgorithm`，提供：
- 设备管理
- 策略状态保存
- LayerNorm参数提取
- Episode运行
- 性能评估
- 结果汇总

### 3. 统一的运行接口
使用 `run_tta_algorithm()` 函数运行任何算法：
```python
from offlinerlkit.tta import run_tta_algorithm

results = run_tta_algorithm(
    algorithm_name='tea',  # stint, tarl, tea, ccea
    env_name='hopper-medium-v2',
    policy_checkpoint='./checkpoints/policy.pth',
    shift_configs=shift_configs,
    num_episodes=20,
    results_csv_path='./results.csv'
)
```

## 使用方法

### 方法1：运行单个算法

```bash
python examples/universal_tta_example.py \
    --mode single \
    --algorithm tea \
    --env hopper-medium-v2 \
    --checkpoint ./checkpoints/policy.pth \
    --episodes 20 \
    --seeds 3 \
    --output ./tea_results.csv
```

### 方法2：对比多个算法

```bash
python examples/universal_tta_example.py \
    --mode compare \
    --env hopper-medium-v2 \
    --checkpoint ./checkpoints/policy.pth \
    --episodes 20 \
    --seeds 3
```

这会自动运行 STINT、TARL、TEA 三个算法并生成对比结果。

### 方法3：使用Python API

```python
from offlinerlkit.tta import run_tta_algorithm, compare_algorithms

# 定义shift配置
shift_configs = {
    'normal': {},
    'light_mass': {'dynamic_shifts': {'mass_scale': 0.8}},
    'heavy_mass': {'dynamic_shifts': {'mass_scale': 1.5}},
    'low_friction': {'dynamic_shifts': {'friction_scale': 0.5}}
}

# 运行单个算法
results = run_tta_algorithm(
    algorithm_name='tea',
    env_name='hopper-medium-v2',
    policy_checkpoint='./checkpoints/policy.pth',
    shift_configs=shift_configs,
    num_episodes=20,
    results_csv_path='./tea_results.csv',
    num_seeds=3
)

# 对比多个算法
comparison = compare_algorithms(
    algorithm_names=['stint', 'tarl', 'tea'],
    env_name='hopper-medium-v2',
    policy_checkpoint='./checkpoints/policy.pth',
    shift_configs=shift_configs,
    num_episodes=20,
    results_dir='./comparison_results',
    num_seeds=3
)
```

## 算法配置

### 通用配置（所有算法）

```python
config = {
    'adaptation_mode': 'layernorm',  # 参数更新模式
    'learning_rate': 1e-5,           # 学习率
    'cache_capacity': 1000              # 缓存容量
}
```

### STINT 特定配置

```python
stint_config = {
    'delta': 0.5,              # 熵突增触发阈值
    'lambda_kl': 1.0,          # KL正则权重
    'K': 3,                     # 每次触发的更新步数
    'beta': 0.1,                # 熵平滑系数
    'learning_rate': 1e-4,       # 学习率
    'adaptation_mode': 'layernorm'
}
```

### TARL 特定配置

```python
tarl_config = {
    'k_low_entropy': 10,         # 低熵样本数量
    'kl_weight': 1.0,           # KL正则权重
    'last_n_layers': 2,          # 最后N层可训练
    'learning_rate': 1e-6,       # 学习率
    'adaptation_mode': 'layernorm'
}
```

### TEA 特定配置

```python
tea_config = {
    'sgld_step_size': 0.1,       # SGLD步长
    'sgld_steps': 10,            # SGLD迭代步数
    'num_neg_samples': 10,        # 负样本数
    'kl_weight': 1.0,            # KL正则权重
    'action_space': 'continuous',   # 动作空间类型
    'update_freq': 10,            # 更新频率
    'learning_rate': 1e-6,       # 学习率
    'adaptation_mode': 'layernorm'
}
```

### CCEA 特定配置

```python
ccea_config = {
    'lambda_min': 0.1,           # λ最小值
    'lambda_max': 10.0,          # λ最大值
    'lambda_init': 1.0,           # λ初始值
    'pos_cache_capacity': 100,     # 正样本缓存容量
    'neg_cache_capacity': 100,     # 负样本缓存容量
    'gamma': 0.1,                # 熵抑制系数
    'tau': 1.0,                 # 温度参数
    'entropy_low': 0.5,           # 熵下限
    'entropy_high': 2.0,          # 熵上限
    'delta_stable': 0.1,         # 稳定阈值
    'v_min': 0.1,                # 最小新颖性比例
    'learning_rate': 1e-4,        # 学习率
    'adaptation_mode': 'layernorm'
}
```

## 参数更新模式

### 1. layernorm（推荐）
仅更新 LayerNorm 层参数，最稳定：
```python
config = {'adaptation_mode': 'layernorm'}
```

### 2. last_n_layers
更新最后 N 层的参数：
```python
config = {
    'adaptation_mode': 'last_n_layers',
    'last_n_layers': 2
}
```

### 3. policy_head
仅更新策略头部参数：
```python
config = {'adaptation_mode': 'policy_head'}
```

### 4. all
更新所有参数（不推荐，容易过拟合）：
```python
config = {'adaptation_mode': 'all'}
```

## 输出结果

### 单算法结果CSV

包含以下列：
- `shift_name`: shift配置名称
- `shift_label`: shift标签
- `mass_scale`: mass缩放因子
- `seed`: 随机种子
- `algorithm`: 算法名称
- `mean_reward`: 平均奖励
- `std_reward`: 奖励标准差
- `max_reward`: 最大奖励
- `min_reward`: 最小奖励
- `mean_length`: 平均episode长度
- `adaptation_steps`: 适应步数
- `cache_size`: 缓存大小
- 算法特定指标（如CD Loss, KL Loss等）

### 对比结果CSV

包含以下列：
- `shift_name`: shift配置名称
- `algorithm`: 算法名称
- `mean_reward`: 平均奖励
- `std_reward`: 奖励标准差
- `num_seeds`: 种子数量

## 扩展新算法

### 步骤1：继承 BaseTTAAlgorithm

```python
from offlinerlkit.tta.base_tta import BaseTTAAlgorithm

class MyTTAAlgorithm(BaseTTAAlgorithm):
    def __init__(self, policy, env, config=None):
        super().__init__(policy, env, config)
        # 你的初始化代码
    
    def _perform_adaptation(self, episode_data):
        """实现你的适应逻辑"""
        # 计算损失
        loss = self._compute_my_loss(episode_data)
        
        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def _get_algorithm_name(self):
        return "MyAlgorithm"
    
    def _get_additional_summary(self, adaptation_data):
        """返回额外的摘要信息"""
        return {
            'my_metric': np.mean([d.get('my_metric', 0) for d in adaptation_data])
        }
```

### 步骤2：在 universal_runner.py 中注册

```python
def _create_algorithm_manager(algorithm_name, policy, env, config):
    if algorithm_name == 'my_algorithm':
        from offlinerlkit.tta.my_algorithm import MyTTAAlgorithm
        return MyTTAAlgorithm(policy, env, config)
    # ... 其他算法
```

### 步骤3：添加默认配置

```python
def _get_default_config(algorithm_name):
    if algorithm_name == 'my_algorithm':
        base_config.update({
            'my_param': 1.0,
            'learning_rate': 1e-5
        })
    # ... 其他算法
```

### 步骤4：更新结果收集

```python
def _create_result_row(...):
    if algorithm_name == 'my_algorithm':
        result_row.update({
            'my_metric': summary.get('my_metric', 0)
        })
    # ... 其他算法
```

## 优势

### 1. 代码复用
- 所有算法共享通用的设备管理、状态保存、参数提取等逻辑
- 减少重复代码，提高可维护性

### 2. 统一接口
- 所有算法使用相同的运行接口
- 便于对比和实验

### 3. 易于扩展
- 新算法只需继承 `BaseTTAAlgorithm`
- 实现核心适应逻辑即可

### 4. 灵活配置
- 支持多种参数更新模式
- 默认使用最稳定的 LayerNorm 模式

### 5. 完整的实验支持
- 多种子实验
- 多shift配置
- 自动结果汇总
- CSV导出

## 示例命令

### 运行TEA算法
```bash
python examples/universal_tta_example.py \
    --mode single \
    --algorithm tea \
    --env hopper-medium-v2 \
    --checkpoint ./checkpoints/policy.pth \
    --episodes 20 \
    --seeds 3 \
    --output ./tea_results.csv
```

### 对比所有算法
```bash
python examples/universal_tta_example.py \
    --mode compare \
    --env hopper-medium-v2 \
    --checkpoint ./checkpoints/policy.pth \
    --episodes 20 \
    --seeds 3
```

### 使用自定义配置
```bash
python examples/universal_tta_example.py \
    --mode custom \
    --algorithm stint \
    --env hopper-medium-v2 \
    --checkpoint ./checkpoints/policy.pth \
    --episodes 20 \
    --seeds 5 \
    --output ./stint_custom.csv
```

## 注意事项

1. **检查点文件**: 确保策略检查点文件存在且格式正确
2. **设备**: 自动检测CUDA，如果没有CUDA则使用CPU
3. **随机种子**: 每个seed会重新初始化环境
4. **参数更新**: 默认使用LayerNorm模式，确保稳定性
5. **结果目录**: 自动创建输出目录
