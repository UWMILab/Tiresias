import argparse
import json
import subprocess
import sys
from pathlib import Path

from lcwm.datasets.protocol import read_records
from lcwm.config import (
    DOMAINS,
    subprocess_environment,
    validate_checkpoint,
    validate_wan_runtime,
    world_model_checkpoint,
)

ROOT = Path(__file__).resolve().parents[1]


def validate_action_source(path: Path) -> None:
    records = read_records(path)
    stored_sources = {str(row.get("action_source", "")).strip() for row in records}
    if stored_sources != {"diffusion_policy_top1"}:
        raise ValueError(
            "manifest action_source must be 'diffusion_policy_top1', "
            f"got {sorted(stored_sources)}"
        )


def validate_runtime(checkpoint: Path) -> None:
    validate_wan_runtime(ROOT)
    validate_checkpoint(checkpoint)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=DOMAINS, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--wan-model-root",
        type=Path,
        required=True,
        help="Directory containing the official Wan2.2-TI2V-5B model files.",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=("float16", "bfloat16", "float32"),
    )
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--dinov3-repo", type=Path, required=True)
    parser.add_argument("--dinov3-checkpoint", type=Path, required=True)
    parser.add_argument("--save-mp4", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    checkpoint = args.checkpoint or world_model_checkpoint(ROOT, args.domain)
    validate_runtime(checkpoint.resolve())
    if not args.wan_model_root.is_dir():
        raise FileNotFoundError(args.wan_model_root)
    if not args.dinov3_repo.is_dir():
        raise FileNotFoundError(args.dinov3_repo)
    if not args.dinov3_checkpoint.is_file():
        raise FileNotFoundError(args.dinov3_checkpoint)
    validate_action_source(args.manifest.resolve())

    command = [
        sys.executable,
        "-m",
        "lcwm.inference",
        "--domain",
        args.domain,
        "--manifest",
        str(args.manifest.resolve()),
        "--wan-root",
        str(args.wan_model_root.resolve()),
        "--lora-path",
        str(checkpoint.resolve()),
        "--out-dir",
        str(args.out_dir.resolve()),
        "--torch-dtype",
        args.dtype,
        "--fps",
        str(args.fps),
        "--seed",
        str(args.seed),
        "--guidance-scale",
        str(args.guidance_scale),
        "--num-inference-steps",
        str(args.num_inference_steps),
        "--start-index",
        str(args.start_index),
        "--num-shards",
        str(args.num_shards),
        "--shard-index",
        str(args.shard_index),
        "--dinov3-repo",
        str(args.dinov3_repo.resolve()),
        "--semantic-weights",
        str(args.dinov3_checkpoint.resolve()),
    ]
    if args.limit is not None:
        command.extend(("--limit", str(args.limit)))
    if args.save_mp4:
        command.append("--save-mp4")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "domain": args.domain,
                    "command": command,
                },
                indent=2,
            )
        )
        return 0
    subprocess.run(
        command,
        check=True,
        cwd=ROOT,
        env=subprocess_environment(ROOT),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
