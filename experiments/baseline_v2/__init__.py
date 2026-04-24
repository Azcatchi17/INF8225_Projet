"""Isolated baseline v2 experiment.

Reproduces the collaborator's pancreas→tumor cascade (test_msd_no_resnet.py)
on top of the existing agentic/ stack, without mutating the main package.
"""
from . import dino_v2, eval_v2  # noqa: F401
