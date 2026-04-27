"""Smoke test: every iteration module is discoverable.

Heavy modules (those that import torch, mmdet, MedSAM) are only checked
for *discoverability* via importlib.find_spec; we do not actually
import them, because the test environment may not have CUDA-enabled
torch. Light modules (proposal_strategy, slice_stack) are fully
imported so we catch regressions in their internal logic.

Run from repo root:
    python -m unittest tests.test_imports
"""
from __future__ import annotations

import importlib
import importlib.util
import unittest


DISCOVERABLE_MODULES = [
    "experiments",
    "experiments.msd",
    "experiments.msd._shared.proposal_strategy",
    "experiments.msd.dino_medsam_cascade",
    "experiments.msd.dino_medsam_cascade.cascade_detector",
    "experiments.msd.dino_medsam_cascade.evaluation",
    "experiments.msd.dino_medsam_cascade.postprocess",
    "experiments.msd.resnet18_recall.calibrate_detector",
    "experiments.msd.resnet18_recall.calibrate_threshold",
    "experiments.msd.resnet18_recall.evaluate",
    "experiments.msd.resnet18_recall.extract_hard_negatives",
    "experiments.msd.resnet18_recall.train_classifier",
    "experiments.msd.three_slice_context.calibrate_threshold",
    "experiments.msd.three_slice_context.evaluate",
    "experiments.msd.three_slice_context.extract_hard_negatives",
    "experiments.msd.three_slice_context.score",
    "experiments.msd.three_slice_context.slice_stack",
    "experiments.msd.three_slice_context.train_classifier",
    "experiments.msd.resnet50_wide_crop.calibrate_threshold",
    "experiments.msd.resnet50_wide_crop.evaluate",
    "experiments.msd.resnet50_wide_crop.extract_hard_negatives",
    "experiments.msd.resnet50_wide_crop.score",
    "experiments.msd.resnet50_wide_crop.train_classifier",
]

# These can be fully imported (no torch / mmdet dependency)
LIGHT_MODULES = [
    "experiments.msd._shared.proposal_strategy",
    "experiments.msd.three_slice_context.slice_stack",
    "experiments.msd.dino_medsam_cascade.postprocess",
]


class TestModuleDiscoverability(unittest.TestCase):
    def test_all_modules_discoverable(self):
        missing = []
        for module_name in DISCOVERABLE_MODULES:
            spec = importlib.util.find_spec(module_name)
            if spec is None:
                missing.append(module_name)
        self.assertEqual(missing, [], f"Missing modules: {missing}")


class TestLightModuleImports(unittest.TestCase):
    def test_proposal_strategy_imports(self):
        from experiments.msd._shared.proposal_strategy import (
            ProposalConfig,
            clip_box,
            expand_box,
            extract_dino_candidates,
        )
        cfg = ProposalConfig()
        self.assertEqual(cfg.crop_margin, 8)
        # clip_box swaps inverted coords
        self.assertEqual(
            clip_box([10, 10, 5, 5], 100, 100), [5.0, 5.0, 10.0, 10.0]
        )

    def test_slice_stack_filename_parsing(self):
        from experiments.msd.three_slice_context.slice_stack import parse_slice

        self.assertEqual(parse_slice("pancreas_015_s32.png"), (15, 32))
        self.assertIsNone(parse_slice("foo/bar.png"))

    def test_postprocess_pad_box(self):
        from experiments.msd.dino_medsam_cascade.postprocess import pad_box

        # 10% pad on each side of a (10,10)->(50,90) box, image 100x100
        out = pad_box((10, 10, 50, 90), 0.1, 100, 100)
        self.assertAlmostEqual(out[0], 6.0)
        self.assertAlmostEqual(out[2], 54.0)


if __name__ == "__main__":
    unittest.main()
