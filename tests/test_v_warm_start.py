import json
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import torch

from rankers.dataset.build_candidate_metric_labels import METRIC_DEFINITION
from rankers.fusion.inference import load_model as load_public_v_model
from rankers.v_ranker import train as v_train
from rankers.v_ranker.video_ranker import CHECKPOINT_FORMAT, Config, VRanker


class _CPUVRanker(VRanker):
    def to(self, *args, **kwargs):
        return super().to("cpu")


def _write_checkpoint(path: Path, *, domain: str = "navigation", config: Config | None = None):
    config = config or Config(hidden_dim=8, semantic_hidden=8)
    model = VRanker(config)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "domain": domain,
        "model_config": asdict(config),
        "state_dict": {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        },
    }
    torch.save(payload, path)
    return model, payload


class VRankerWarmStartTest(unittest.TestCase):
    def test_strict_init_and_save_reload_preserve_all_tensors(self) -> None:
        config = Config(hidden_dim=8, semantic_hidden=8)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _ = _write_checkpoint(root / "canonical.pt", config=config)
            loaded = v_train.load_v_checkpoint(
                root / "canonical.pt", "navigation", config=config
            )
            target = VRanker(config)
            initialization = v_train.initialize_v_model(
                target, loaded, "navigation", config=config
            )
            for name, value in source.state_dict().items():
                torch.testing.assert_close(
                    target.state_dict()[name], value, rtol=0, atol=0
                )
            self.assertEqual(initialization["optimizer"], "fresh")
            self.assertEqual(initialization["metric_transform"], "none_unnormalized_labels")

            new_path = root / "rmse-model.pt"
            torch.save(
                {
                    "format": CHECKPOINT_FORMAT,
                    "domain": "navigation",
                    "model_config": asdict(config),
                    "state_dict": {
                        name: value.detach().cpu().clone()
                        for name, value in target.state_dict().items()
                    },
                    "metric_definition": dict(METRIC_DEFINITION),
                },
                new_path,
            )
            payload = torch.load(new_path, map_location="cpu", weights_only=True)
            self.assertEqual(payload["metric_definition"], METRIC_DEFINITION)
            self.assertNotIn("init_checkpoint", payload)
            reloaded = load_public_v_model(new_path, "navigation", "cpu")
            for name, value in target.state_dict().items():
                torch.testing.assert_close(
                    reloaded.state_dict()[name], value, rtol=0, atol=0
                )

    def test_bad_domain_and_config_are_rejected(self) -> None:
        config = Config(hidden_dim=8, semantic_hidden=8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.pt"
            _write_checkpoint(path, config=config)
            with self.assertRaises(ValueError):
                v_train.load_v_checkpoint(path, "driving", config=config)
            bad = torch.load(path, map_location="cpu", weights_only=True)
            bad["model_config"]["hidden_dim"] = 16
            torch.save(bad, path)
            with self.assertRaisesRegex(ValueError, "model_config"):
                v_train.load_v_checkpoint(path, "navigation", config=config)

    def test_load_labels_rejects_missing_or_wrong_metric_definition(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.json"
            base = {
                "domain": "navigation",
                "split": "train",
                "candidate_count_per_sample": 8,
                "records": [],
            }
            path.write_text(json.dumps(base), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "metric_definition"):
                v_train.load_labels(path, {}, "navigation", "train")
            base["metric_definition"] = {"ATE": "mean_l2", "RPE": "mean_l2"}
            path.write_text(json.dumps(base), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "metric_definition"):
                v_train.load_labels(path, {}, "navigation", "train")

    def test_epoch_zero_is_reference_only_and_best_is_trained(self) -> None:
        config = Config(hidden_dim=8, semantic_hidden=8)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _ = _write_checkpoint(root / "canonical.pt", config=config)
            output = root / "output"
            argv = [
                "train",
                "--train-evidence",
                "unused",
                "--train-labels",
                "unused",
                "--validation-evidence",
                "unused",
                "--validation-labels",
                "unused",
                "--init-checkpoint",
                str(root / "canonical.pt"),
                "--output-dir",
                str(output),
                "--epochs",
                "1",
            ]

            def fake_objective(model, *_args):
                return next(model.parameters()).sum()

            with patch.object(v_train, "Config", side_effect=lambda **_kwargs: config), patch.object(
                v_train, "VRanker", _CPUVRanker
            ), patch.object(
                v_train,
                "load_evidence",
                return_value=("navigation", {"sample": {}}),
            ), patch.object(
                v_train,
                "load_labels",
                return_value={"sample": [{}] * 8},
            ), patch.object(
                v_train,
                "label_definition",
                return_value=dict(METRIC_DEFINITION),
            ), patch.object(
                v_train, "objective", side_effect=fake_objective
            ), patch.object(v_train, "validation_loss", side_effect=[1.0, 2.0]), patch.object(
                sys, "argv", argv
            ):
                self.assertEqual(v_train.main(), 0)

            report = json.loads(
                (output / "training_report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(report["best_epoch"], 1)
            self.assertEqual(report["epoch0_validation_loss"], 1.0)
            self.assertEqual(report["init_checkpoint"], str(root / "canonical.pt"))
            self.assertEqual([entry["epoch"] for entry in report["history"]], [0, 1])
            self.assertTrue(report["history"][0]["reference_only"])
            self.assertGreaterEqual(report["history"][1]["elapsed_seconds"], 0.0)
            self.assertEqual(report["history"][1]["sample_count"], 1)

            payload = torch.load(
                output / "model.pt", map_location="cpu", weights_only=True
            )
            self.assertEqual(payload["metric_definition"], METRIC_DEFINITION)
            self.assertNotIn("init_checkpoint", payload)
            self.assertFalse(
                torch.equal(
                    payload["state_dict"]["semantic_position"],
                    source.state_dict()["semantic_position"],
                )
            )


if __name__ == "__main__":
    unittest.main()
