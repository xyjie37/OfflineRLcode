#!/bin/bash

# TTA Shift Analysis Bash Script - STINT Only
# 只测试STINT算法在各种偏移程度下的效果

ENV="hopper-medium-v2"
CHECKPOINT="./checkpoints/policy.pth"
EPISODES=20
RESULT_DIR="./analysis"

# 默认质量缩放因子（9档）
MASS_SCALES="1.0 0.9 0.8 0.7 0.6 1.1 1.25 1.5 1.75"

# 只运行STINT算法
STRATEGIES="stint"

# 随机种子
SEEDS="0 1 2"

# 学习率
LEARNING_RATE=1e-5

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --env)
            ENV="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --episodes)
            EPISODES="$2"
            shift 2
            ;;
        --results_dir)
            RESULT_DIR="$2"
            shift 2
            ;;
        --mass_scales)
            MASS_SCALES="$2"
            shift 2
            ;;
        --strategies)
            STRATEGIES="$2"
            shift 2
            ;;
        --seeds)
            SEEDS="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --env ENV                 Environment name (default: hopper-medium-v2)"
            echo "  --checkpoint PATH         Policy checkpoint path (default: ./checkpoints/policy.pth)"
            echo "  --episodes NUM            Number of episodes (default: 20)"
            echo "  --results_dir DIR         Results directory (default: ./analysis)"
            echo "  --mass_scales SCALES      Mass scale factors (default: '1.0 0.9 0.8 0.7 0.6 1.1 1.25 1.5 1.75')"
            echo "  --strategies STRATEGIES   TTA strategies (default: 'stint')"
            echo "  --seeds SEEDS             Random seeds (default: '0 1 2')"
            echo "  --learning_rate LR        Learning rate (default: 1e-5)"
            echo "  --help                    Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "TTA Shift Analysis Script - STINT Only"
echo "=========================================="
echo "Environment: $ENV"
echo "Checkpoint: $CHECKPOINT"
echo "Episodes: $EPISODES"
echo "Results directory: $RESULT_DIR"
echo "Mass scales: $MASS_SCALES"
echo "Strategies: $STRATEGIES"
echo "Seeds: $SEEDS"
echo "Learning rate: $LEARNING_RATE"
echo "=========================================="

python run_tta_shift_analysis.py \
    --env "$ENV" \
    --checkpoint "$CHECKPOINT" \
    --episodes "$EPISODES" \
    --results_dir "$RESULT_DIR" \
    --mass_scales $MASS_SCALES \
    --strategies $STRATEGIES \
    --seeds $SEEDS \
    --learning_rate "$LEARNING_RATE"

echo ""
echo "=========================================="
echo "STINT Analysis completed!"
echo "=========================================="
echo "Results saved to: $RESULT_DIR/tta_shift_analysis.csv"
echo "=========================================="
