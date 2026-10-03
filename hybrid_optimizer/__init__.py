"""
Closed-Loop Genetic-LLM Hybrid System for Autonomous Software Optimization
============================================================================
Reference implementation accompanying the paper of the same name.
"""

from .orchestrator import ClosedLoopOptimizer
from .genome import CodeGenome
from .fitness import FitnessReport

__all__ = ["ClosedLoopOptimizer", "CodeGenome", "FitnessReport"]
__version__ = "1.0.0"
