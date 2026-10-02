import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

from lcwm.config import (
    DOMAINS,
    subprocess_environment,
    validate_wan_runtime,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=DOMAINS, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--semantic-latent-root", type=Path, required=True)
    parser.add_argument(
        "--wan-model-root",
        type=Path,
        required=True,
        help="Directory containing the official Wan2.2-TI2V-5B files.",
    )
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--data_ratio", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-processes", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if not 0 < args.data_ratio <= 1:
        parser.error("--data_ratio must be in (0, 1]")
    if args.num_workers < 0:
        parser.error("--num-workers must be non-negative")
    if args.num_processes < 1:
        parser.error("--num-processes must be at least 1")
    action_config_path = None
    if args.domain in {"gaming", "robot"}:
        action_config_path = args.dataset_root / "action_config.json"
        if not action_config_path.is_file():
            parser.error(f"Action config not found: {action_config_path}")
        action_config = json.loads(action_config_path.read_text(encoding="utf-8"))
        if action_config.get("domain") != args.domain:
            parser.error(f"Invalid action config: {action_config_path}")

    validate_wan_runtime(ROOT)
    wan_root = args.wan_model_root.resolve()
    diffusion_shards = sorted(wan_root.glob("diffusion_pytorch_model-*.safetensors"))
    if not diffusion_shards:
        raise FileNotFoundError(f"Wan2.2 diffusion weights not found in {wan_root}")
    model_paths = json.dumps(
        [
            [str(path) for path in diffusion_shards],
            str(wan_root / "models_t5_umt5-xxl-enc-bf16.pth"),
            str(wan_root / "Wan2.2_VAE.pth"),
        ]
    )
    tokenizer_path = args.tokenizer_path or wan_root / "google" / "umt5-xxl"
    command = [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        "--num_machines",
        "1",
        "--num_processes",
        str(args.num_processes),
        "--mixed_precision",
        "bf16",
        "--dynamo_backend",
        "no",
        "--module",
        "lcwm.train",
        "--domain",
        args.domain,
        "--dataset-base-path",
        str(args.dataset_root.resolve()),
        "--dataset-metadata-path",
        str(args.train_manifest.resolve()),
        "--data_ratio",
        str(args.data_ratio),
        "--semantic-latent-root",
        str(args.semantic_latent_root.resolve()),
        "--model-paths",
        model_paths,
        "--tokenizer-path",
        str(tokenizer_path.resolve()),
        "--output-path",
        str(args.output_dir.resolve()),
        "--dataset-num-workers",
        str(args.num_workers),
        "--learning-rate",
        str(args.learning_rate),
        "--num-epochs",
        str(args.epochs),
    ]
    if args.num_processes > 1:
        command.insert(3, "--multi_gpu")
    if args.dry_run:
        print(json.dumps({"domain": args.domain, "command": command}, indent=2))
        return 0
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(args.output_dir)
    subprocess.run(
        command,
        check=True,
        cwd=ROOT,
        env=subprocess_environment(ROOT),
    )
    if action_config_path is not None:
        shutil.copyfile(action_config_path, args.output_dir / "action_config.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
