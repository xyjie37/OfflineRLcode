"""
Test-Time Adaptation (TTA) Module for OfflineRL-Kit

This module provides tools for evaluating pre-trained policies in shifted environments
with support for test-time adaptation.
"""

from offlinerlkit.tta.shifted_env import ShiftedMujocoEnvWrapper
from offlinerlkit.tta.model_loader import ModelLoader
from offlinerlkit.tta.tta_manager import TTAManager
from offlinerlkit.tta.evaluator import ShiftedPolicyEvaluator, run_tta_evaluation
from offlinerlkit.tta.mcatta import CCEAManager, ContrastiveCache
from offlinerlkit.tta.tarl import TARLManager

__all__ = [
    "ShiftedMujocoEnvWrapper",
    "ModelLoader", 
    "TTAManager",
    "ShiftedPolicyEvaluator",
    "run_tta_evaluation",
    "CCEAManager",
    "ContrastiveCache",
    "TARLManager"
]
