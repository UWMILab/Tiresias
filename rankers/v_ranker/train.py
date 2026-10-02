import argparse
import json
import random
import time
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from lcwm.datasets.protocol import PUBLIC_SPLITS, select_training_items
from rankers.dataset.build_candidate_metric_labels import METRIC_DEFINITION
from rankers.v_ranker.video_ranker import (
    CHECKPOINT_FORMAT,
    VRanker,
    Config,
    batch,
    load_evidence,
)


def _metric_definition(
    payload: Mapping[str, Any], path: Path
) -> dict[str, str] | None:
    value = payload.get("metric_definition")
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != set(METRIC_DEFINITION):
        raise ValueError(f"invalid metric_definition in {path}")
    result = {str(key): str(item) for key, item in value.items()}
    if set(result) != set(METRIC_DEFINITION):
        raise ValueError(f"invalid metric_definition in {path}")
    return result


def _require_rmse_labels(
    train: Mapping[str, str] | None,
    validation: Mapping[str, str] | None,
) -> dict[str, str]:
    if (
        dict(train or {}) != METRIC_DEFINITION
        or dict(validation or {}) != METRIC_DEFINITION
    ):
        raise ValueError(
            "training and validation labels must declare the RMSE metric_definition"
        )
    return dict(METRIC_DEFINITION)


def _v_config(payload: Mapping[str, Any], path: Path) -> Config:
    raw = payload.get("model_config")
    expected_keys = set(asdict(Config()))
    if not isinstance(raw, Mapping) or set(raw) != expected_keys:
        raise ValueError(f"{path} has an invalid V-Ranker model_config")
    try:
        config = Config(**dict(raw))
        valid = (
            type(config.hidden_dim) is int
            and config.hidden_dim >= 1
            and type(config.dinov3_dim) is int
            and config.dinov3_dim >= 1
            and type(config.semantic_hidden) is int
            and config.semantic_hidden >= 1
            and type(config.semantic_tokens) is int
            and config.semantic_tokens >= 1
            and np.isfinite(float(config.dropout))
            and 0 <= float(config.dropout) < 1
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{path} has an invalid V-Ranker model_config") from exc
    if not valid:
        raise ValueError(f"{path} has an invalid V-Ranker model_config")
    return config


def load_v_checkpoint(
    path: Path, domain: str, *, config: Config | None = None
) -> dict[str, Any]:
    """Load a compatible V-Ranker checkpoint on CPU for strict initialization."""
    path = Path(path)
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, Mapping) or value.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"invalid V-Ranker checkpoint: {path}")
    if value.get("domain") != domain:
        raise ValueError(f"{path} domain does not match {domain}")
    source_config = _v_config(value, path)
    if config is not None and asdict(source_config) != asdict(config):
        raise ValueError(f"{path} model_config does not match the current config")
    state = value.get("state_dict")
    if not isinstance(state, Mapping):
        raise ValueError(f"{path} is missing state_dict")
    result = dict(value)
    result["model_config"] = asdict(source_config)
    result["metric_definition"] = _metric_definition(result, path)
    result["_path"] = path.name
    return result


def initialize_v_model(
    model: VRanker,
    checkpoint: Mapping[str, Any],
    domain: str,
    *,
    config: Config | None = None,
) -> dict[str, Any]:
    """Strictly copy every source tensor into ``model`` without reinitializing heads."""
    path = Path(str(checkpoint.get("_path", "<checkpoint>")))
    if (
        checkpoint.get("format") != CHECKPOINT_FORMAT
        or checkpoint.get("domain") != domain
    ):
        raise ValueError("V-Ranker checkpoint domain/format differs")
    source_config = _v_config(checkpoint, path)
    if config is not None and asdict(source_config) != asdict(config):
        raise ValueError("V-Ranker checkpoint model_config differs")
    source_definition = _metric_definition(checkpoint, path)
    state = checkpoint.get("state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("V-Ranker checkpoint is missing state_dict")
    if any(
        not isinstance(key, str)
        or not isinstance(value, torch.Tensor)
        or not bool(torch.isfinite(value).all())
        for key, value in state.items()
    ):
        raise ValueError("V-Ranker checkpoint contains invalid state_dict")
    try:
        model.load_state_dict(
            {key: value.detach().clone() for key, value in state.items()}, strict=True
        )
    except (RuntimeError, TypeError) as exc:
        raise ValueError("V-Ranker checkpoint tensors are incompatible") from exc
    return {
        "mode": "warm_start",
        "source": "init_checkpoint",
        "source_checkpoint": path.name,
        "source_metric_definition": source_definition,
        "target_metric_definition": dict(METRIC_DEFINITION),
        "metric_transform": "none_unnormalized_labels",
        "metric_transform_exact": source_definition == METRIC_DEFINITION,
        "metric_definition_changed": source_definition != METRIC_DEFINITION,
        "metric_definition_note": (
            "V labels are unnormalized; tensors are copied verbatim without "
            "metric conversion"
        ),
        "optimizer": "fresh",
    }


def load_labels(
    path: Path,
    evidence: dict[str, dict[str, Any]],
    domain: str,
    split: str,
) -> dict[str, list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"invalid V-Ranker label file: {path}")
    if (
        payload.get("domain") != domain
        or payload.get("split") != split
        or int(payload.get("candidate_count_per_sample", -1)) != 8
    ):
        raise ValueError(f"label file does not match {domain}/{split}: {path}")
    definition = _metric_definition(payload, path)
    if definition != METRIC_DEFINITION:
        raise ValueError(
            f"label file does not declare the RMSE metric_definition: {path}"
        )
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in payload["records"]:
        grouped.setdefault(str(row["sample_key"]), []).append(row)
    if set(grouped) != set(evidence):
        raise ValueError("evidence and label identities differ")
    for key, values in grouped.items():
        values.sort(key=lambda row: int(row["top8_slot"]))
        if len(values) != 8:
            raise ValueError(f"{key}: expected eight labels")
    return grouped


def label_definition(path: Path) -> dict[str, str] | None:
    """Read a label file's declared metric definition for audit/reporting."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"invalid V-Ranker label file: {path}")
    return _metric_definition(payload, path)


def objective(
    model: VRanker,
    domain: str,
    group: dict[str, Any],
    labels: list[dict[str, Any]],
    device: str,
) -> torch.Tensor:
    wan, action, dino, text, _ = batch(group, device)
    prediction = model(domain, wan, action, dino, text)
    quality = torch.tensor([row["quality_label"] for row in labels], device=device)
    distribution = torch.softmax(quality / 0.35, dim=0)
    listwise = -(
        distribution * torch.log_softmax(prediction["score"] / 0.35, dim=0)
    ).sum()
    best = int(quality.argmax())
    worst = int(quality.argmin())
    margin = F.relu(
        torch.tensor(0.20, device=device)
        - (prediction["score"][best] - prediction["score"][worst])
    )
    auxiliary = torch.stack(
        (
            F.mse_loss(prediction["score"], quality),
            F.mse_loss(
                prediction["language"],
                torch.tensor([row["language_label"] for row in labels], device=device),
            ),
            F.mse_loss(
                prediction["semantic"],
                torch.tensor([row["semantic_label"] for row in labels], device=device),
            ),
            F.mse_loss(
                prediction["motion"],
                torch.tensor([row["motion_label"] for row in labels], device=device),
            ),
        )
    ).mean()
    return listwise + margin + auxiliary


@torch.inference_mode()
def validation_loss(
    model: VRanker,
    domain: str,
    evidence: dict[str, dict[str, Any]],
    labels: dict[str, list[dict[str, Any]]],
    device: str,
) -> float:
    model.eval()
    return float(
        np.mean(
            [
                float(objective(model, domain, evidence[key], labels[key], device))
                for key in sorted(evidence)
            ]
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-evidence", type=Path, nargs="+", required=True)
    parser.add_argument("--train-labels", type=Path, required=True)
    parser.add_argument("--validation-evidence", type=Path, nargs="+", required=True)
    parser.add_argument("--validation-labels", type=Path, required=True)
    parser.add_argument("--train-split", choices=PUBLIC_SPLITS, default="train")
    parser.add_argument(
        "--validation-split", choices=PUBLIC_SPLITS, default="val_seen"
    )
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--data_ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if not np.isfinite(args.data_ratio) or not 0 < args.data_ratio <= 1:
        parser.error("--data_ratio must be in (0, 1]")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(args.output_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_domain, all_train_evidence = load_evidence(
        args.train_evidence, args.train_split
    )
    validation_domain, validation_evidence = load_evidence(
        args.validation_evidence, args.validation_split
    )
    if train_domain != validation_domain:
        raise ValueError("training and validation domains differ")
    all_train_labels = load_labels(
        args.train_labels, all_train_evidence, train_domain, args.train_split
    )
    validation_labels = load_labels(
        args.validation_labels,
        validation_evidence,
        validation_domain,
        args.validation_split,
    )
    train_keys = select_training_items(
        list(all_train_evidence),
        args.data_ratio,
        key=lambda key: str(key),
    )
    train_evidence = {key: all_train_evidence[key] for key in train_keys}
    train_labels = {key: all_train_labels[key] for key in train_keys}

    device = "cuda:0"
    config = Config()
    metric_definition = _require_rmse_labels(
        label_definition(args.train_labels),
        label_definition(args.validation_labels),
    )
    init_payload = None
    if args.init_checkpoint is not None:
        init_payload = load_v_checkpoint(
            args.init_checkpoint,
            train_domain,
            config=config,
        )
    model = VRanker(config).to(device)
    if init_payload is None:
        initialization = {
            "mode": "random",
            "source": None,
            "target_metric_definition": metric_definition,
            "optimizer": "fresh",
        }
    else:
        initialization = initialize_v_model(
            model,
            init_payload,
            train_domain,
            config=config,
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    best_loss = float("inf")
    best_state = None
    best_epoch = None
    history = []
    generator = torch.Generator().manual_seed(args.seed)
    keys = sorted(train_evidence)
    epoch0_validation_loss = None
    if init_payload is not None:
        epoch0_started = time.perf_counter()
        epoch0_validation_loss = validation_loss(
            model, train_domain, validation_evidence, validation_labels, device
        )
        epoch0_elapsed = time.perf_counter() - epoch0_started
        history.append(
            {
                "epoch": 0,
                "loss": None,
                "validation_loss": epoch0_validation_loss,
                "sample_count": len(train_evidence),
                "elapsed_seconds": epoch0_elapsed,
                "elapsed_sec": epoch0_elapsed,
                "reference_only": True,
            }
        )
    for epoch in range(args.epochs):
        epoch_started = time.perf_counter()
        model.train()
        losses = []
        for index in torch.randperm(len(keys), generator=generator).tolist():
            key = keys[index]
            loss = objective(
                model, train_domain, train_evidence[key], train_labels[key], device
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        current = validation_loss(
            model, train_domain, validation_evidence, validation_labels, device
        )
        if not np.isfinite(current):
            raise ValueError(f"non-finite validation loss at epoch {epoch + 1}")
        elapsed_seconds = time.perf_counter() - epoch_started
        entry = {
            "epoch": epoch + 1,
            "loss": float(np.mean(losses)),
            "validation_loss": current,
            "sample_count": len(train_evidence),
            "elapsed_seconds": elapsed_seconds,
            "elapsed_sec": elapsed_seconds,
        }
        history.append(entry)
        print(json.dumps(entry), flush=True)
        # Only trained epochs are eligible for checkpoint selection.
        if best_state is None or current < best_loss:
            best_loss = current
            best_epoch = epoch + 1
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": CHECKPOINT_FORMAT,
            "domain": train_domain,
            "model_config": asdict(config),
            "state_dict": best_state,
            "metric_definition": metric_definition,
        },
        args.output_dir / "model.pt",
    )
    (args.output_dir / "training_report.json").write_text(
        json.dumps(
            {
                "history": history,
                "best_validation_loss": best_loss,
                "best_epoch": best_epoch,
                "selected_epoch": best_epoch,
                "sample_count": len(train_evidence),
                "train_sample_count": len(train_evidence),
                "validation_sample_count": len(validation_evidence),
                "metric_definition": metric_definition,
                "init_checkpoint": (
                    str(args.init_checkpoint)
                    if args.init_checkpoint is not None
                    else None
                ),
                "init_path": (
                    str(args.init_checkpoint)
                    if args.init_checkpoint is not None
                    else None
                ),
                "initialized_from": (
                    args.init_checkpoint.name
                    if args.init_checkpoint is not None
                    else None
                ),
                "epoch0_validation_loss": epoch0_validation_loss,
                "initialization": initialization,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
