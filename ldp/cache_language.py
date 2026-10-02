import argparse
import json
from pathlib import Path

from lcwm.config import DOMAINS
from lcwm.datasets.protocol import PUBLIC_SPLITS, read_records

ROOT = Path(__file__).resolve().parents[1]


def instructions(path: Path) -> list[str]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            rows = payload.get("records", [])
        elif isinstance(payload, list):
            rows = payload
        else:
            raise ValueError(f"unsupported JSON manifest: {path}")
    else:
        rows = read_records(path)
    values = sorted({str(row["instruction"]).strip() for row in rows})
    if not values or not all(values):
        raise ValueError(f"empty instruction in {path}")
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=(*DOMAINS, "all"), required=True)
    parser.add_argument(
        "--data-root", type=Path, default=ROOT / "data" / "processed"
    )
    parser.add_argument(
        "--wan-model-root",
        type=Path,
        default=ROOT / "pretrained_models" / "Wan2.2-TI2V-5B",
    )
    parser.add_argument(
        "--input-manifest",
        type=Path,
        help="Inference-only manifest containing custom instructions.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output cache for --input-manifest.",
    )
    args = parser.parse_args()
    if (args.input_manifest is None) != (args.output is None):
        parser.error("--input-manifest and --output must be provided together")
    if args.input_manifest is not None:
        if args.domain == "all":
            parser.error("--input-manifest requires one domain")
        jobs = [(args.input_manifest.resolve(), args.output.resolve())]
    else:
        domains = DOMAINS if args.domain == "all" else (args.domain,)
        data_root = args.data_root.resolve()
        jobs = [
            (
                data_root / domain / "planning" / f"{split}.jsonl",
                data_root / domain / f"language_{split}.pt",
            )
            for domain in domains
            for split in PUBLIC_SPLITS
        ]
    for manifest, output in jobs:
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
        if output.exists():
            raise FileExistsError(output)

    import torch
    from lcwm.wan22.diffsynth.core.loader.model import load_model
    from lcwm.wan22.diffsynth.models.wan_video_text_encoder import (
        HuggingfaceTokenizer,
        WanTextEncoder,
    )

    root = args.wan_model_root.resolve()
    device = "cuda:0"
    tokenizer = HuggingfaceTokenizer(
        str(root / "google" / "umt5-xxl"), seq_len=96, clean="whitespace"
    )
    encoder = load_model(
        WanTextEncoder,
        str(root / "models_t5_umt5-xxl-enc-bf16.pth"),
        torch_dtype=torch.bfloat16,
        device=device,
    ).eval()
    for manifest, output in jobs:
        embeddings = {}
        masks = {}
        with torch.inference_mode():
            for text in instructions(manifest):
                ids, mask = tokenizer(text, return_mask=True, add_special_tokens=True)
                ids = ids.to(device)
                mask = mask.to(device)
                length = int(mask.sum())
                value = encoder(ids, mask)[0, :length].detach().cpu().to(torch.float16)
                embeddings[text] = value
                masks[text] = torch.ones(length, dtype=torch.bool)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"token_embeddings": embeddings, "token_masks": masks},
            output,
        )
        print(f"Cached {len(embeddings)} instructions in {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
