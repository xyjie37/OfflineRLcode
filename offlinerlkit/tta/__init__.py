"""
Test-Time Adaptation (TTA) Module for OfflineRL-Kit

This module provides tools for evaluating pre-trained policies in shifted environments
with support for test-time adaptation.
"""

from offlinerlkit.tta.shifted_env import ShiftedMujocoEnvWrapper
from offlinerlkit.tta.model_loader import ModelLoader
from offlinerlkit.tta.tta_manager import TTAManager, run_tea_experiment
from offlinerlkit.tta.evaluator import ShiftedPolicyEvaluator, run_tta_evaluation
from offlinerlkit.tta.mcatta import CCEAManager, ContrastiveCache
from offlinerlkit.tta.tarl import TARLManager
from offlinerlkit.tta.stint import STINTManager
from offlinerlkit.tta.tea import TEAManager
from offlinerlkit.tta.come import COMEManager
from offlinerlkit.tta.base_tta import BaseTTAAlgorithm
from offlinerlkit.tta.universal_runner import run_tta_algorithm, compare_algorithms

__all__ = [
    "ShiftedMujocoEnvWrapper",
    "ModelLoader", 
    "TTAManager",
    "run_tea_experiment",
    "ShiftedPolicyEvaluator",
    "run_tta_evaluation",
    "CCEAManager",
    "ContrastiveCache",
    "TARLManager",
    "STINTManager",
    "TEAManager",
    "COMEManager",
    "BaseTTAAlgorithm",
    "run_tta_algorithm",
    "compare_algorithms"
]