import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import torch
from PIL import Image

from lcwm.config import DOMAINS
from lcwm.datasets.protocol import read_records
from lcwm.semantic_backbone import image_tensor, load_dinov3

MODEL_NAME = "dinov3_vitl16"
IMAGE_SIZE = 256
EXPECTED_DIM = 1024
ROOT = Path(__file__).resolve().parents[2]


def first_image(path: Path) -> Image.Image:
    if path.is_dir():
        images = [
            item
            for item in path.iterdir()
            if item.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        ]
        if not images:
            raise FileNotFoundError(f"no RGB frame in {path}")

        def key(item: Path):
            try:
                return (0, int(item.stem))
            except ValueError:
                return (1, item.name)

        with Image.open(sorted(images, key=key)[0]) as source:
            return source.convert("RGB").copy()
    reader = imageio.get_reader(path)
    try:
        return Image.fromarray(reader.get_data(0)).convert("RGB")
    finally:
        reader.close()


def export_features(
    rows: list[dict],
    dataset_root: Path,
    output_root: Path,
    model,
    *,
    batch_size: int,
    shard_index: int,
    shard_count: int,
) -> None:
    assigned = [
        (index, row)
        for index, row in enumerate(rows)
        if index % shard_count == shard_index
    ]
    output = output_root / dataset_root.name
    output.mkdir(parents=True, exist_ok=True)
    device = "cuda:0"
    written = 0
    for start in range(0, len(assigned), batch_size):
        pending = []
        images = []
        for _, row in assigned[start : start + batch_size]:
            target = output / f"{row['trajectory_id']}_semantic.pt"
            if target.exists():
                raise FileExistsError(target)
            source = Path(row["video"])
            if not source.is_absolute():
                source = dataset_root / source
            pending.append(target)
            images.append(image_tensor(first_image(source), IMAGE_SIZE))
        if not images:
            continue
        batch = torch.stack(images).to(device)
        with (
            torch.inference_mode(),
            torch.autocast(device_type="cuda", dtype=torch.bfloat16),
        ):
            features = model.forward_features(batch)
        cls = features["x_norm_clstoken"].detach().cpu().to(torch.float16)
        patch = features["x_norm_patchtokens"].detach().cpu().to(torch.float16)
        if cls.shape[-1] != EXPECTED_DIM or patch.shape[-1] != EXPECTED_DIM:
            raise RuntimeError(
                f"unexpected DINOv3 dimensions: {cls.shape}/{patch.shape}"
            )
        for offset, target in enumerate(pending):
            payload = {
                "cls_tokens": cls[offset : offset + 1].clone(),
                "patch_tokens": patch[offset : offset + 1].clone(),
            }
            temporary = target.with_suffix(".tmp")
            torch.save(payload, temporary)
            temporary.replace(target)
            written += 1
        print(
            json.dumps({"shard": shard_index, "written": written}),
            flush=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=(*DOMAINS, "all"), required=True)
    parser.add_argument(
        "--data-root", type=Path, default=ROOT / "data" / "processed"
    )
    parser.add_argument(
        "--pretrained-root", type=Path, default=ROOT / "pretrained_models"
    )
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard")
    domains = DOMAINS if args.domain == "all" else (args.domain,)
    data_root = args.data_root.resolve()
    jobs = [
        (
            data_root / domain / "wm" / "train.jsonl",
            data_root / domain / "wm",
            data_root / domain / "features",
        )
        for domain in domains
    ]
    for manifest, _, _ in jobs:
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
    pretrained_root = args.pretrained_root.resolve()
    model = load_dinov3(
        pretrained_root / "dinov3",
        pretrained_root / "dinov3_weights" / "dinov3_vitl16_pretrain_lvd1689m.pth",
        MODEL_NAME,
        "cuda:0",
    )
    for manifest, dataset_root, output_root in jobs:
        export_features(
            read_records(manifest),
            dataset_root,
            output_root,
            model,
            batch_size=args.batch_size,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
