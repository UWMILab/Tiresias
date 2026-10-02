import argparse
import json
import math
import random
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from lcwm.datasets.protocol import PUBLIC_SPLITS, select_training_items
from ldp.policy import CANDIDATE_FIELDS, HORIZON
from rankers.dataset.build_candidate_metric_labels import METRIC_DEFINITION

from rankers.tr_ranker.trajectory_ranker import (
    CHECKPOINT_FORMAT,
    DROPOUT,
    METRICS,
    MODEL_SIZE,
    TrajectoryRanker,
    apply_action_adapter,
    load_action_adapter,
    make_action_batch,
    make_text_batch,
    trajectory_score,
)


_TR_STANDARDIZED_ROWS = {"nDTW": 1, "ATE": 2, "RPE": 3}


def _metric_definition(
    payload: Mapping[str, Any], path: Path
) -> dict[str, str] | None:
    """Read metric metadata; legacy checkpoints may omit it."""

    value = payload.get("metric_definition")
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != set(METRIC_DEFINITION):
        raise ValueError(f"invalid metric_definition in {path}")
    return {str(key): str(item) for key, item in value.items()}


def _require_rmse_labels(
    train: dict[str, str] | None, validation: dict[str, str] | None
) -> dict[str, str]:
    if train != validation or train != METRIC_DEFINITION:
        raise ValueError(
            "training and validation labels must declare one RMSE metric_definition"
        )
    return dict(train)


def _require_label_definition(
    payload: Mapping[str, Any], path: Path
) -> dict[str, str]:
    definition = _metric_definition(payload, path)
    if definition != METRIC_DEFINITION:
        raise ValueError(
            f"label file must declare the RMSE metric_definition in {path}"
        )
    return dict(definition)


def _input_contract(domain: str, adapter: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "domain": domain,
        "action_fields": list(CANDIDATE_FIELDS[domain]),
        "adapter": adapter,
    }


def load_tr_checkpoint(
    path: Path,
    domain: str,
    *,
    action_dim: int | None = None,
    hidden: int | None = None,
    layers: int | None = None,
    dropout: float | None = None,
    adapter: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load a compatible TR checkpoint on CPU for strict initialization."""

    path = Path(path)
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, Mapping) or value.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"invalid trajectory-ranker checkpoint: {path}")
    if value.get("domain") != domain:
        raise ValueError(f"{path} domain does not match {domain}")

    expected_dim = (
        len(CANDIDATE_FIELDS[domain]) if action_dim is None else int(action_dim)
    )
    try:
        source_dim = int(value["action_dim"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{path} has an invalid action_dim") from exc
    if source_dim != expected_dim:
        raise ValueError(f"{path} action_dim does not match {expected_dim}")

    try:
        source_horizon = int(value.get("horizon", HORIZON[domain]))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} has an invalid horizon") from exc
    if source_horizon != HORIZON[domain]:
        raise ValueError(f"{path} horizon does not match the current config")

    try:
        source_hidden = int(value.get("hidden", MODEL_SIZE[domain][0]))
        source_layers = int(value.get("layers", MODEL_SIZE[domain][1]))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} has an invalid model config") from exc
    if source_hidden < 8 or source_hidden % 8 or source_layers < 1:
        raise ValueError(f"{path} has an invalid model config")
    if hidden is not None and source_hidden != int(hidden):
        raise ValueError(f"{path} hidden does not match the current config")
    if layers is not None and source_layers != int(layers):
        raise ValueError(f"{path} layers does not match the current config")

    try:
        source_dropout = float(value.get("dropout", DROPOUT))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} has an invalid dropout") from exc
    if not np.isfinite(source_dropout) or not 0 <= source_dropout < 1:
        raise ValueError(f"{path} has an invalid dropout")
    if dropout is not None and source_dropout != float(dropout):
        raise ValueError(f"{path} dropout does not match the current config")

    action_fields = value.get("action_fields", CANDIDATE_FIELDS[domain])
    if tuple(action_fields) != CANDIDATE_FIELDS[domain]:
        raise ValueError(f"{path} action_fields do not match {domain}")

    try:
        mean = torch.as_tensor(value["action_mean"], dtype=torch.float32)
        std = torch.as_tensor(value["action_std"], dtype=torch.float32)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{path} has invalid action normalization") from exc
    if (
        mean.shape != (expected_dim,)
        or std.shape != (expected_dim,)
        or not torch.isfinite(mean).all()
        or not torch.isfinite(std).all()
        or not (std > 0).all()
    ):
        raise ValueError(f"{path} has invalid action normalization")
    mean = mean.detach().clone()
    std = std.detach().clone()

    state = value.get("model_state")
    if not isinstance(state, Mapping):
        raise ValueError(f"{path} is missing model_state")
    for name, tensor in state.items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError(f"{path} contains an invalid model_state")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{path} contains non-finite model_state tensors")
    state_mean = state.get("action_mean")
    state_std = state.get("action_std")
    if (
        not isinstance(state_mean, torch.Tensor)
        or not isinstance(state_std, torch.Tensor)
        or state_mean.shape != (1, 1, 1, expected_dim)
        or state_std.shape != (1, 1, 1, expected_dim)
        or not torch.equal(state_mean.reshape(-1).float(), mean)
        or not torch.equal(state_std.reshape(-1).float(), std)
    ):
        raise ValueError(f"{path} action normalization is inconsistent")

    input_normalization = value.get("input_normalization")
    if input_normalization is not None and input_normalization != _input_contract(
        domain, adapter
    ):
        raise ValueError(f"{path} input normalization contract differs")

    result = dict(value)
    result.update(
        action_mean=mean,
        action_std=std,
        metric_definition=_metric_definition(result, path),
        _path=path.name,
    )
    return result


def _stats(
    stats: Mapping[str, Any] | None, metric: str
) -> tuple[float, float] | None:
    if stats is None or metric not in stats:
        return None
    value = stats[metric]
    if not isinstance(value, Mapping):
        raise ValueError(f"invalid target_statistics for {metric}")
    try:
        mean, std = float(value["mean"]), float(value["std"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid target_statistics for {metric}") from exc
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
        raise ValueError(f"invalid target_statistics for {metric}")
    return mean, std


def initialize_tr_model(
    model: TrajectoryRanker,
    checkpoint: Mapping[str, Any],
    target_statistics: Mapping[str, Any],
    *,
    target_metric_definition: Mapping[str, str] = METRIC_DEFINITION,
) -> dict[str, Any]:
    """Remap head standardization without changing raw predictions.

    This initializes the model; RMSE supervision still requires training.
    """

    if (
        checkpoint.get("format") != CHECKPOINT_FORMAT
        or checkpoint.get("domain") != model.domain
    ):
        raise ValueError("trajectory-ranker checkpoint domain/format differs")
    if int(checkpoint.get("action_dim", -1)) != model.action_dim:
        raise ValueError("trajectory-ranker checkpoint action_dim differs")

    source = checkpoint.get("model_state")
    if not isinstance(source, Mapping):
        raise ValueError("trajectory-ranker checkpoint is missing model_state")
    for name, value in source.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise ValueError("trajectory-ranker checkpoint contains invalid model_state")
        if not torch.isfinite(value).all():
            raise ValueError("trajectory-ranker checkpoint contains non-finite tensors")
    state = {key: value.detach().clone() for key, value in source.items()}

    weight, bias = state.get("heads.weight"), state.get("heads.bias")
    if (
        not isinstance(weight, torch.Tensor)
        or not isinstance(bias, torch.Tensor)
        or weight.shape != (4, model.hidden)
        or bias.shape != (4,)
    ):
        raise ValueError("trajectory-ranker checkpoint has an invalid metric head")

    transformed: list[str] = []
    skipped: list[str] = []
    for metric, row in _TR_STANDARDIZED_ROWS.items():
        old = _stats(checkpoint.get("target_statistics"), metric)
        new = _stats(target_statistics, metric)
        if old is None or new is None:
            skipped.append(metric)
            continue
        scale = old[1] / new[1]
        state["heads.weight"][row].mul_(scale)
        state["heads.bias"][row].mul_(scale).add_((old[0] - new[0]) / new[1])
        transformed.append(metric)

    try:
        model.load_state_dict(state, strict=True)
    except (RuntimeError, TypeError) as exc:
        raise ValueError("trajectory-ranker checkpoint tensors are incompatible") from exc

    source_definition = _metric_definition(
        checkpoint, Path(str(checkpoint.get("_path", "<checkpoint>")))
    )
    target_definition = {str(key): str(value) for key, value in target_metric_definition.items()}
    changed = source_definition != target_definition
    exact = not changed and not skipped
    return {
        "mode": "warm_start",
        "source": "init_checkpoint",
        "source_checkpoint": Path(
            str(checkpoint.get("_path", "<checkpoint>"))
        ).name,
        "source_metric_definition": source_definition,
        "target_metric_definition": target_definition,
        "metric_definition_changed": changed,
        "metric_transform": {
            "method": "affine_standardization",
            "transformed_metrics": transformed,
            "skipped_metrics": skipped,
            "sr_logit": "unchanged",
        },
        "metric_transform_exact": exact,
        "metric_definition_note": (
            "affine initialization only; old and new ATE/RPE definitions are "
            "not equivalent and this is not a true RMSE conversion"
            if changed
            else "standardization remap for an unchanged metric definition"
        ),
        "input_normalization": "source_checkpoint",
        "optimizer": "fresh",
    }


def _atomic_torch_save(value: Any, path: Path) -> None:
    """Create a checkpoint without replacing an existing output file."""

    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        torch.save(value, temporary)
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_write_json(value: Any, path: Path) -> None:
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def load_candidates(
    paths: list[Path], domain: str, split: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        value = torch.load(path, map_location="cpu", weights_only=True)
        if (
            value.get("domain") != domain
            or value.get("split") != split
            or int(value.get("candidate_count", -1)) != 128
            or int(value.get("horizon", -1)) != HORIZON[domain]
            or tuple(value.get("action_fields", ())) != CANDIDATE_FIELDS[domain]
        ):
            raise ValueError(f"invalid candidate file: {path}")
        rows.extend(value["records"])
    rows.sort(key=lambda row: (int(row["dataset_index"]), str(row["sample_id"])))
    if not rows or len({str(row["sample_id"]) for row in rows}) != len(rows):
        raise ValueError(f"empty or duplicate candidates: {path}")
    return rows


def load_labels(
    path: Path,
    rows: list[dict[str, Any]],
    domain: str,
    split: str,
) -> dict[str, dict[str, Any]]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, Mapping):
        raise ValueError(f"invalid label file: {path}")
    if value.get("domain") != domain or value.get("split") != split:
        raise ValueError(f"label file does not match {domain}/{split}: {path}")
    _require_label_definition(value, path)
    if not isinstance(value.get("records"), list):
        raise ValueError(f"invalid label records in {path}")
    labels = {str(row["sample_id"]): row for row in value["records"]}
    if set(labels) != {str(row["sample_id"]) for row in rows}:
        raise ValueError("candidate and label identities differ")
    for sample_id, row in labels.items():
        for metric in METRICS:
            tensor = torch.as_tensor(row[metric]).float()
            if tensor.shape != (128,) or not torch.isfinite(tensor).all():
                raise ValueError({"sample_id": sample_id, "metric": metric})
    return labels


def label_definition(path: Path) -> dict[str, str] | None:
    """Return the declared metric definition, including ``None`` for legacy files."""

    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, Mapping):
        raise ValueError(f"invalid label file: {path}")
    return _metric_definition(value, path)


def target_statistics(labels: dict[str, dict[str, Any]]) -> dict[str, dict[str, float]]:
    if not labels:
        raise ValueError("cannot compute target_statistics for empty labels")
    output: dict[str, dict[str, float]] = {}
    for metric in METRICS:
        values = torch.cat(
            [torch.as_tensor(row[metric]).float() for row in labels.values()]
        )
        output[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(unbiased=False).clamp_min(1e-6)),
        }
    return output


def action_statistics(
    rows: list[dict[str, Any]],
    domain: str,
    adapter: dict[str, Any] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    values = []
    for row in rows:
        action = torch.as_tensor(row["candidate_actions"]).float().unsqueeze(0)
        mask = torch.as_tensor(row["candidate_masks"]).bool().unsqueeze(0)
        action = apply_action_adapter(action, [row], domain, adapter)
        values.append(action[mask].reshape(-1, action.shape[-1]))
    merged = torch.cat(values)
    return merged.mean(0), merged.std(0, unbiased=False).clamp_min(1e-6)


def targets_for(
    rows: list[dict[str, Any]],
    labels: dict[str, dict[str, Any]],
    statistics: dict[str, dict[str, float]],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    output: dict[str, torch.Tensor] = {}
    for metric in METRICS:
        raw = torch.stack(
            [
                torch.as_tensor(labels[str(row["sample_id"])][metric]).float()
                for row in rows
            ]
        ).to(device)
        output[metric] = raw
        if metric != "SR":
            output[f"{metric}_z"] = (raw - statistics[metric]["mean"]) / statistics[
                metric
            ]["std"]
    return output


def objective(
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    positive_weight: torch.Tensor,
) -> torch.Tensor:
    return (
        F.binary_cross_entropy_with_logits(
            prediction["sr_logit"], target["SR"], pos_weight=positive_weight
        )
        + F.huber_loss(prediction["ndtw_z"], target["nDTW_z"])
        + F.huber_loss(prediction["ate_z"], target["ATE_z"])
        + F.huber_loss(prediction["rpe_z"], target["RPE_z"])
    )


@torch.inference_mode()
def evaluate(
    model: TrajectoryRanker,
    rows: list[dict[str, Any]],
    labels: dict[str, dict[str, Any]],
    cache: dict[str, Any],
    statistics: dict[str, dict[str, float]],
    adapter: dict[str, Any] | None,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    model.eval()
    selected = {metric: [] for metric in METRICS}
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        text, text_mask = make_text_batch(batch, cache, device)
        actions, action_mask = make_action_batch(batch, device, model.domain, adapter)
        prediction = model(text, text_mask, actions, action_mask)
        scores = trajectory_score(prediction, statistics).float().cpu()
        for row, score in zip(batch, scores):
            top8 = sorted(range(128), key=lambda index: (-float(score[index]), index))[
                :8
            ]
            label = labels[str(row["sample_id"])]
            best = max(
                top8,
                key=lambda index: (
                    float(label["SR"][index]),
                    float(label["nDTW"][index]),
                    -float(label["ATE"][index]),
                    -float(label["RPE"][index]),
                    -index,
                ),
            )
            for metric in METRICS:
                selected[metric].append(float(label[metric][best]))
    return {metric: float(np.mean(values)) for metric, values in selected.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=tuple(HORIZON), required=True)
    parser.add_argument("--train-candidates", type=Path, nargs="+", required=True)
    parser.add_argument("--train-labels", type=Path, required=True)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument(
        "--validation-candidates", type=Path, nargs="+", required=True
    )
    parser.add_argument("--validation-labels", type=Path, required=True)
    parser.add_argument("--validation-cache", type=Path, required=True)
    parser.add_argument("--train-split", choices=PUBLIC_SPLITS, default="train")
    parser.add_argument(
        "--validation-split", choices=PUBLIC_SPLITS, default="val_seen"
    )
    parser.add_argument("--action-config", type=Path)
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        help="Initialize from an existing TR checkpoint with a fresh optimizer.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--data_ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    started = time.monotonic()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if not np.isfinite(args.data_ratio) or not 0 < args.data_ratio <= 1:
        parser.error("--data_ratio must be in (0, 1]")
    if args.output_dir.exists():
        if not args.output_dir.is_dir() or any(args.output_dir.iterdir()):
            raise FileExistsError(args.output_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    all_train_rows = load_candidates(
        args.train_candidates, args.domain, args.train_split
    )
    validation_rows = load_candidates(
        args.validation_candidates, args.domain, args.validation_split
    )
    all_train_labels = load_labels(
        args.train_labels, all_train_rows, args.domain, args.train_split
    )
    validation_labels = load_labels(
        args.validation_labels,
        validation_rows,
        args.domain,
        args.validation_split,
    )
    train_definition = label_definition(args.train_labels)
    validation_definition = label_definition(args.validation_labels)
    metric_definition = _require_rmse_labels(
        train_definition, validation_definition
    )
    train_rows = select_training_items(
        all_train_rows,
        args.data_ratio,
        key=lambda row: str(row["sample_key"]),
    )
    train_labels = {
        str(row["sample_id"]): all_train_labels[str(row["sample_id"])]
        for row in train_rows
    }
    if not train_rows:
        raise ValueError("training data is empty after applying --data_ratio")
    train_cache = torch.load(args.train_cache, map_location="cpu", weights_only=True)
    validation_cache = torch.load(
        args.validation_cache, map_location="cpu", weights_only=True
    )
    adapter = load_action_adapter(args.action_config, args.domain)
    current_action_mean, current_action_std = action_statistics(
        train_rows, args.domain, adapter
    )
    statistics = target_statistics(train_labels)
    positives = sum(
        float(torch.as_tensor(train_labels[str(row["sample_id"])]["SR"]).sum())
        for row in train_rows
    )
    positive_weight = torch.tensor(
        (128 * len(train_rows) - positives) / max(positives, 1.0), device=device
    )

    action_dim = int(torch.as_tensor(train_rows[0]["candidate_actions"]).shape[-1])
    hidden, layers = MODEL_SIZE[args.domain]
    init_payload = None
    if args.init_checkpoint is not None:
        init_payload = load_tr_checkpoint(
            args.init_checkpoint,
            args.domain,
            action_dim=action_dim,
            hidden=hidden,
            layers=layers,
            dropout=DROPOUT,
            adapter=adapter,
        )
        # Preserve the source input normalization when warm-starting.
        action_mean = init_payload["action_mean"]
        action_std = init_payload["action_std"]
    else:
        action_mean = current_action_mean
        action_std = current_action_std
    model = TrajectoryRanker(
        action_dim,
        action_mean,
        action_std,
        horizon=HORIZON[args.domain],
        hidden=hidden,
        layers=layers,
        domain=args.domain,
    ).to(device)
    if init_payload is None:
        initialization = {
            "mode": "random",
            "source": None,
            "target_metric_definition": metric_definition,
            "input_normalization": "training_data",
            "optimizer": "fresh",
        }
    else:
        initialization = initialize_tr_model(
            model,
            init_payload,
            statistics,
            target_metric_definition=metric_definition,
        )
        initialization["checkpoint_path"] = str(args.init_checkpoint.resolve())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-3
    )
    generator = torch.Generator().manual_seed(args.seed)
    best_key = (-math.inf, -math.inf, -math.inf, -math.inf)
    best_state = None
    best_metrics = None
    history = []
    initial_baseline = None
    selected_epoch = None

    if init_payload is not None:
        epoch0_started = time.perf_counter()
        initial_baseline = evaluate(
            model,
            validation_rows,
            validation_labels,
            validation_cache,
            statistics,
            adapter,
            device,
            args.batch_size,
        )
        epoch0_elapsed = float(time.perf_counter() - epoch0_started)
        # Only trained epochs are eligible for the RMSE checkpoint.
        history.append(
            {
                "epoch": 0,
                "loss": None,
                "validation": initial_baseline,
                "status": "reference_only",
                "checkpoint_eligible": False,
                "reference_only": True,
                "sample_count": len(train_rows),
                "elapsed_seconds": epoch0_elapsed,
                "elapsed_sec": epoch0_elapsed,
            }
        )

    for epoch in range(args.epochs):
        epoch_started = time.perf_counter()
        model.train()
        losses = []
        order = torch.randperm(len(train_rows), generator=generator).tolist()
        for start in range(0, len(order), args.batch_size):
            batch = [
                train_rows[index] for index in order[start : start + args.batch_size]
            ]
            text, text_mask = make_text_batch(batch, train_cache, device)
            actions, action_mask = make_action_batch(
                batch, device, args.domain, adapter
            )
            target = targets_for(batch, train_labels, statistics, device)
            prediction = model(text, text_mask, actions, action_mask)
            loss = objective(prediction, target, positive_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        metrics = evaluate(
            model,
            validation_rows,
            validation_labels,
            validation_cache,
            statistics,
            adapter,
            device,
            args.batch_size,
        )
        if any(not np.isfinite(value) for value in metrics.values()):
            raise ValueError(f"non-finite validation metric at epoch {epoch + 1}")
        key = (metrics["SR"], metrics["nDTW"], -metrics["ATE"], -metrics["RPE"])
        elapsed_seconds = float(time.perf_counter() - epoch_started)
        entry = {
            "epoch": epoch + 1,
            "loss": float(np.mean(losses)),
            "validation": metrics,
            "status": "rmse_training_epoch",
            "checkpoint_eligible": True,
            "sample_count": len(train_rows),
            "elapsed_seconds": elapsed_seconds,
            "elapsed_sec": elapsed_seconds,
        }
        history.append(entry)
        print(json.dumps(entry), flush=True)
        if key > best_key:
            best_key = key
            best_metrics = metrics
            selected_epoch = epoch + 1
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if selected_epoch is None:
        raise RuntimeError("training did not produce an eligible RMSE checkpoint")
    elapsed_sec = float(time.monotonic() - started)
    action_mean_cpu = action_mean.detach().cpu().clone()
    action_std_cpu = action_std.detach().cpu().clone()
    # Keep local paths in the training report, not in the checkpoint.
    public_initialization = dict(initialization)
    public_initialization.pop("checkpoint_path", None)
    public_initialization.pop("source_checkpoint", None)
    checkpoint = {
        "format": CHECKPOINT_FORMAT,
        "domain": args.domain,
        "model_state": best_state,
        "action_dim": action_dim,
        "action_mean": action_mean_cpu,
        "action_std": action_std_cpu,
        "horizon": HORIZON[args.domain],
        "action_fields": list(CANDIDATE_FIELDS[args.domain]),
        "input_normalization": _input_contract(args.domain, adapter),
        "target_statistics": statistics,
        "metric_definition": dict(metric_definition),
        "initialization": public_initialization,
        "checkpoint_epoch": selected_epoch,
        "training_status": "rmse_trained",
        "hidden": hidden,
        "layers": layers,
        "dropout": DROPOUT,
        "score_formula": "z(SR)+z(nDTW)-z(ATE)-z(RPE)",
    }
    report = {
        "history": history,
        "best_validation": best_metrics,
        "best_epoch": selected_epoch,
        "selected_epoch": selected_epoch,
        "initialization": initialization,
        "initial_baseline": initial_baseline,
        "epoch0_validation": initial_baseline,
        "epoch0_evaluation": initial_baseline,
        "metric_definition": dict(metric_definition),
        "sample_count": len(train_rows),
        "train_sample_count": len(train_rows),
        "validation_sample_count": len(validation_rows),
        "training_sample_count": len(train_rows),
        "elapsed_sec": elapsed_sec,
        "elapsed_seconds": elapsed_sec,
        "init_checkpoint": (
            str(args.init_checkpoint) if args.init_checkpoint is not None else None
        ),
        "init_path": (
            str(args.init_checkpoint) if args.init_checkpoint is not None else None
        ),
        "initialized_from": (
            args.init_checkpoint.name if args.init_checkpoint is not None else None
        ),
        "training_status": "rmse_trained",
    }
    _atomic_torch_save(checkpoint, args.output_dir / "model.pt")
    _atomic_write_json(report, args.output_dir / "training_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
