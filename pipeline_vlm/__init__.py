"""Agentic segmentation pipeline — VLM-only variant.

Replaces Grounding DINO with Gemini's native object detection. Everything
else (MedSAM, best-iter policy, trust region, mask analysis loop) is
inherited from the base `pipeline` package.
"""
