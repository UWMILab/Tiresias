import argparse
import json
from pathlib import Path

from lcwm.config import (
    ACTION_DIMS,
    validate_wan_runtime,
    world_model_checkpoint,
)

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=tuple(ACTION_DIMS), required=True)
    parser.add_argument("--wan-model-root", type=Path, required=True)
    parser.add_argument("--dinov3-repo", type=Path, required=True)
    parser.add_argument("--dinov3-checkpoint", type=Path, required=True)
    parser.add_argument("--wm-checkpoint", type=Path)
    parser.add_argument(
        "--torch-dtype",
        choices=("float16", "bfloat16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(args.output)
    if not args.wan_model_root.is_dir():
        raise FileNotFoundError(args.wan_model_root)
    if not args.dinov3_repo.is_dir():
        raise FileNotFoundError(args.dinov3_repo)
    if not args.dinov3_checkpoint.is_file():
        raise FileNotFoundError(args.dinov3_checkpoint)
    if args.num_inference_steps <= 0:
        raise ValueError("num-inference-steps must be positive")

    checkpoint = (
        args.wm_checkpoint.resolve()
        if args.wm_checkpoint is not None
        else world_model_checkpoint(ROOT, args.domain).resolve()
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    validate_wan_runtime(ROOT)
    payload = {
        "domain": args.domain,
        "wm_checkpoint_path": str(checkpoint),
        "wan_root": str(args.wan_model_root.resolve()),
        "dinov3_repo": str(args.dinov3_repo.resolve()),
        "dinov3_checkpoint_path": str(args.dinov3_checkpoint.resolve()),
        "device": "cuda:0",
        "torch_dtype": args.torch_dtype,
        "guidance_scale": float(args.guidance_scale),
        "num_inference_steps": int(args.num_inference_steps),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
