# TTA Experiment Framework User Guide

This guide explains how to use the TTA (Test-Time Adaptation) experiment framework to run complete experiments, including switchable TTA strategies, simple proxy updates, environment shift intensity sweeps, and multi-metric collection.

## Feature Overview

### 1. Switchable TTA Strategies
- Control whether TTA is enabled via the `enable_tta` parameter
- Supports multiple adaptation strategies: online fine-tuning, meta-learning, experience-based adaptation, entropy minimization, uncertainty minimization

### 2. Simple Proxy Updates
- **Entropy minimization**: minimize policy entropy to make the policy more deterministic
- **Uncertainty minimization**: minimize Q-value uncertainty (using the difference between two critics)

### 3. Environment Shift Intensity Sweep
- Supports 4 levels of shift intensity
- Supports multiple shift types: mass, friction coefficient, observation noise, reward scaling, combined shift

### 4. Metric Collection
- **Average Return**: mean return
- **Worst Case Return**: worst-case return (across seeds and shifts)
- **Collapse Rate**: collapse rate (return continuously below a threshold for N episodes)
- **Policy Divergence**: policy drift norm (parameter change)

### 5. Metric Comparison With and Without TTA
- Automatically computes TTA improvement relative to baseline
- Generates detailed comparison reports

## Quick Start

### Basic Usage Example

```python
from offlinerlkit.tta.shifted_env import create_shifted_env, create_shift_intensity_configs
from offlinerlkit.tta.tta_manager import TTAManager

# Create a shifted environment
shift_configs = create_shift_intensity_configs('mass', num_levels=4)
shift_config = shift_configs['mass_level_1']

env = create_shifted_env('hopper-medium-v2', shift_config)

# Create a TTA manager (enable TTA)
adaptation_config = {
    'enable_tta': True,
    'strategy': 'entropy_minimization',
    'learning_rate': 1e-5,
    'batch_size': 32,
    'collapse_threshold': -100.0,
    'collapse_window': 5
}

tta_manager = TTAManager(policy, env, adaptation_config)

# Run adaptation
adaptation_data, metrics = tta_manager.run_adaptation(num_episodes=20)

# View results
print(f"Average Return: {metrics['mean_reward']:.2f}")
print(f"Worst Case Return: {metrics['worst_case_return']:.2f}")
print(f"Collapse Rate: {metrics['collapse_rate']:.2%}")
print(f"Policy Divergence: {metrics['mean_policy_divergence']:.6f}")
