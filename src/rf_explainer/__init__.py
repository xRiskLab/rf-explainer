# -*- coding: utf-8 -*-
"""
__init__.py.

This module is the entry point for the package.
"""

from .analyzer import RandomForestAnalyzer
from .visualizer import RandomForestVisualizer

__all__ = [
    "RandomForestAnalyzer",
    "RandomForestVisualizer",
]

# Add dynamic version retrieval
try:
    from importlib.metadata import version
    __version__ = version("rf-explainer")
except ImportError:
    __version__ = "unknown"
