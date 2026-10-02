import tempfile
import unittest
from pathlib import Path

import torch

from rankers.tr_ranker import train as tr
from rankers.tr_ranker.inference import load_model as load_public_tr_model
from rankers.tr_ranker.trajectory_ranker import TrajectoryRanker


def _stats(values: dict[str, tuple[float, float]]) -> dict[str, dict[str, float]]:
    return {
        metric: {"mean": mean, "std": std}
        for metric, (mean, std) in values.items()
    }


def _checkpoint(model: TrajectoryRanker, statistics: dict[str, dict[str, float]]) -> dict:
    return {
        "format": tr.CHECKPOINT_FORMAT,
        "domain": model.domain,
        "model_state": {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        },
        "action_dim": model.action_dim,
        "action_mean": model.action_mean.reshape(-1).detach().cpu().clone(),
        "action_std": model.action_std.reshape(-1).detach().cpu().clone(),
        "horizon": model.horizon,
        "action_fields": list(tr.CANDIDATE_FIELDS[model.domain]),
        "hidden": model.hidden,
        "layers": model.layers,
        "dropout": model.dropout,
        "target_statistics": statistics,
        # A legacy source may omit this field; the new output must include it.
    }


class TRWarmStartTest(unittest.TestCase):
    def test_raw_outputs_are_preserved_by_affine_initialization(self) -> None:
        source_mean = torch.tensor([1.0, 2.0, 3.0])
        source_std = torch.tensor([2.0, 3.0, 4.0])
        old_stats = _stats(
            {"nDTW": (1.0, 2.0), "ATE": (2.0, 3.0), "RPE": (3.0, 4.0)}
        )
        new_stats = _stats(
            {"nDTW": (2.0, 4.0), "ATE": (4.0, 6.0), "RPE": (5.0, 8.0)}
        )
        source = TrajectoryRanker(
            3,
            source_mean,
            source_std,
            horizon=16,
            hidden=8,
            layers=1,
            domain="navigation",
        )
        payload = _checkpoint(source, old_stats)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy-model.pt"
            torch.save(payload, path)
            loaded = tr.load_tr_checkpoint(
                path, "navigation", action_dim=3, hidden=8, layers=1, dropout=0.1
            )

        target = TrajectoryRanker(
            3,
            torch.zeros(3),
            torch.ones(3),
            horizon=16,
            hidden=8,
            layers=1,
            domain="navigation",
        )
        report = tr.initialize_tr_model(target, loaded, new_stats)
        source.eval()
        target.eval()
        text = torch.randn(2, 4, 4096)
        text_mask = torch.ones(2, 4, dtype=torch.bool)
        actions = torch.randn(2, 128, 16, 3)
        action_mask = torch.ones(2, 128, 16, dtype=torch.bool)
        with torch.no_grad():
            source_prediction = source(text, text_mask, actions, action_mask)
            target_prediction = target(text, text_mask, actions, action_mask)

        self.assertTrue(
            torch.equal(target.action_mean.reshape(-1), source_mean)
        )
        self.assertTrue(torch.equal(target.action_std.reshape(-1), source_std))
        self.assertTrue(
            torch.equal(source_prediction["sr_logit"], target_prediction["sr_logit"])
        )
        for key, metric in (
            ("ndtw_z", "nDTW"),
            ("ate_z", "ATE"),
            ("rpe_z", "RPE"),
        ):
            old_raw = old_stats[metric]["mean"] + old_stats[metric]["std"] * source_prediction[key]
            new_raw = new_stats[metric]["mean"] + new_stats[metric]["std"] * target_prediction[key]
            torch.testing.assert_close(old_raw, new_raw, rtol=1e-5, atol=1e-5)
        self.assertEqual(report["metric_transform"]["sr_logit"], "unchanged")
        self.assertFalse(report["metric_transform_exact"])
        self.assertIn("not a true RMSE conversion", report["metric_definition_note"])

    def test_target_statistics_use_new_rmse_label_values(self) -> None:
        labels = {
            "a": {
                "SR": torch.tensor([0.0, 1.0]),
                "nDTW": torch.tensor([0.0, 1.0]),
                "ATE": torch.tensor([1.0, 3.0]),
                "RPE": torch.tensor([2.0, 6.0]),
            },
            "b": {
                "SR": torch.tensor([1.0, 0.0]),
                "nDTW": torch.tensor([2.0, 4.0]),
                "ATE": torch.tensor([5.0, 7.0]),
                "RPE": torch.tensor([8.0, 10.0]),
            },
        }
        statistics = tr.target_statistics(labels)
        self.assertAlmostEqual(statistics["ATE"]["mean"], 4.0)
        self.assertAlmostEqual(statistics["ATE"]["std"], 2.2360679775, places=6)
        self.assertAlmostEqual(statistics["RPE"]["mean"], 6.5)
        self.assertAlmostEqual(statistics["RPE"]["std"], 2.9580398915, places=6)

    def test_new_checkpoint_roundtrips_through_public_loader(self) -> None:
        source = TrajectoryRanker(
            3,
            torch.tensor([1.0, 2.0, 3.0]),
            torch.tensor([2.0, 3.0, 4.0]),
            horizon=16,
            hidden=8,
            layers=1,
            domain="navigation",
        )
        payload = _checkpoint(
            source,
            _stats(
                {
                    "SR": (0.5, 0.5),
                    "nDTW": (0.0, 1.0),
                    "ATE": (0.0, 1.0),
                    "RPE": (0.0, 1.0),
                }
            ),
        )
        payload["metric_definition"] = dict(tr.METRIC_DEFINITION)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save(payload, path)
            loaded, loaded_payload = load_public_tr_model(
                path, "navigation", torch.device("cpu")
            )
        for name, value in source.state_dict().items():
            torch.testing.assert_close(value, loaded.state_dict()[name], rtol=0, atol=0)
        self.assertEqual(loaded_payload["metric_definition"], tr.METRIC_DEFINITION)

    def test_loader_rejects_missing_or_mismatched_rmse_definition(self) -> None:
        row = {"sample_id": "sample"}
        candidates = [row]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.pt"
            for definition in (None, {"ATE": "mean_position_l2", "RPE": "mean_step_displacement_l2"}):
                payload = {
                    "domain": "navigation",
                    "split": "train",
                    "records": [
                        {
                            "sample_id": "sample",
                            "SR": torch.zeros(128),
                            "nDTW": torch.zeros(128),
                            "ATE": torch.zeros(128),
                            "RPE": torch.zeros(128),
                        }
                    ],
                }
                if definition is not None:
                    payload["metric_definition"] = definition
                torch.save(payload, path)
                with self.assertRaises(ValueError):
                    tr.load_labels(path, candidates, "navigation", "train")


if __name__ == "__main__":
    unittest.main()
