import argparse
import json
from pathlib import Path
from typing import Any

from lcwm.datasets.protocol import (
    PLANNING_ANCHOR_FRAME_IDX,
    INFERENCE_SPLITS,
    read_records,
)
from ldp.action_utils import driving_source_dt

DOMAINS = ("navigation", "driving", "robot", "gaming")


def resolve_input_path(value: Any, dataset_root: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else dataset_root / path


def initial_image(row: dict[str, Any], dataset_root: Path) -> Path:
    value = row.get("first_rgb_path")
    if not value:
        raise KeyError("source row must provide first_rgb_path")
    path = resolve_input_path(value, dataset_root).resolve()
    try:
        frame_index = int(path.stem)
    except ValueError as exc:
        raise ValueError(
            f"planning first_rgb_path must use a numeric frame name: {path}"
        ) from exc
    if frame_index != PLANNING_ANCHOR_FRAME_IDX:
        raise ValueError(
            f"planning initial observation must be source frame "
            f"{PLANNING_ANCHOR_FRAME_IDX}, got {frame_index}: {path}"
        )
    header = path.read_bytes()[:12]
    if header.startswith((b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff")) or (
        header[:4] == b"RIFF" and header[8:12] == b"WEBP"
    ):
        return path
    raise ValueError(f"invalid initial image: {path}")


def identity_of(row: dict[str, Any]) -> str:
    return str(row["sample_key"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=DOMAINS, required=True)
    parser.add_argument("--split", choices=INFERENCE_SPLITS, default="val_seen")
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        help="Root used to resolve relative video/image paths (default: manifest directory).",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = read_records(args.source_manifest)
    dataset_root = (
        args.dataset_root.resolve()
        if args.dataset_root is not None
        else args.source_manifest.resolve().parent
    )
    output = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if str(row.get("domain")) != args.domain:
            raise ValueError(
                {
                    "row": index,
                    "expected_domain": args.domain,
                    "actual": row.get("domain"),
                }
            )
        if str(row.get("split")) != args.split:
            raise ValueError(
                f"row {index} is for split {row.get('split')!r}, not {args.split!r}"
            )
        key = identity_of(row)
        if not key or key in seen:
            raise ValueError(f"duplicate/empty identity: {key!r}")
        seen.add(key)
        instruction = str(row.get("instruction") or "").strip()
        if not instruction:
            raise ValueError(f"empty instruction: {key}")
        image = initial_image(row, dataset_root)
        record = {
            "sample_key": key,
            "dataset_index": int(
                row.get("dataset_index", row.get("sample_idx", index))
            ),
            "domain": args.domain,
            "instruction": instruction,
            "initial_observation_path": str(image),
            "source_name": str(row["source_name"]),
        }
        if args.domain == "driving":
            record["source_dt"] = driving_source_dt(
                row, context=f"{key} Driving metadata"
            )
        output.append(record)
    output.sort(key=lambda row: (int(row["dataset_index"]), str(row["sample_key"])))
    payload = {
        "domain": args.domain,
        "split": args.split,
        "record_count": len(output),
        "records": output,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "domain": args.domain,
                "split": args.split,
                "record_count": len(output),
                "output": str(args.output.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
