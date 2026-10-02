import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

from lcwm.config import DOMAINS, validate_wan_runtime
from lcwm.datasets.protocol import PUBLIC_SPLITS
from lcwm.datasets.raw_input import INPUT_SCOPE, prepare_input

ROOT = Path(__file__).resolve().parent


def pipeline_environment() -> dict[str, str]:
    environment = os.environ.copy()
    paths = [str(ROOT)]
    existing = environment.get("PYTHONPATH")
    if existing:
        paths.extend(entry for entry in existing.split(os.pathsep) if entry)
    environment["PYTHONPATH"] = os.pathsep.join(paths)
    return environment


def run_stage(
    name: str,
    command: list[str],
    *,
    output: Path | None,
    dry_run: bool,
) -> None:
    if output is not None and output.exists():
        raise FileExistsError(f"{name} output already exists: {output}")
    if dry_run:
        print(json.dumps({"stage": name, "command": command}, indent=2))
        return
    print(f"Running {name}", flush=True)
    subprocess.run(command, cwd=ROOT, env=pipeline_environment(), check=True)
    if output is not None and not output.exists():
        raise RuntimeError(f"{name} did not create {output}")


def require_file(path: Path, *, dry_run: bool) -> Path:
    path = path.expanduser().resolve()
    if not dry_run and not path.is_file():
        raise FileNotFoundError(path)
    return path


def require_dir(path: Path, *, dry_run: bool) -> Path:
    path = path.expanduser().resolve()
    if not dry_run and not path.is_dir():
        raise FileNotFoundError(path)
    return path


def planning_action_config(
    weights_dir: Path, domain: str, *, dry_run: bool
) -> Path | None:
    if domain not in {"gaming", "robot"}:
        return None
    path = weights_dir / "diffusion_policy" / domain / "proposal" / "action_config.json"
    return require_file(path, dry_run=dry_run)


def world_model_action_config(
    weights_dir: Path, domain: str, *, dry_run: bool
) -> Path | None:
    if domain not in {"gaming", "robot"}:
        return None
    path = weights_dir / "world_model" / domain / "action_config.json"
    return require_file(path, dry_run=dry_run)


def full_pipeline(args: argparse.Namespace) -> int:
    domain = args.domain
    validate_wan_runtime(ROOT)
    run_dir = args.run_dir.expanduser().resolve()
    weights_dir = (ROOT / "weights").resolve()
    if not args.dry_run:
        run_dir.mkdir(parents=True, exist_ok=True)
    proposal = weights_dir / "diffusion_policy" / domain / "proposal" / "LDP.pt"
    tr_checkpoint = weights_dir / "rankers" / domain / "128_to_8" / "model.pt"
    v_ranker_checkpoint = weights_dir / "rankers" / domain / "8_to_1" / "model.pt"
    wm_checkpoint = weights_dir / "world_model" / domain / "model.safetensors"
    proposal = require_file(proposal, dry_run=args.dry_run)
    tr_checkpoint = require_file(tr_checkpoint, dry_run=args.dry_run)
    v_ranker_checkpoint = require_file(v_ranker_checkpoint, dry_run=args.dry_run)
    wm_checkpoint = require_file(wm_checkpoint, dry_run=args.dry_run)

    queries = require_file(args.queries, dry_run=args.dry_run)
    language_cache = require_file(args.language_cache, dry_run=args.dry_run)
    source_manifest = require_file(args.source_manifest, dry_run=args.dry_run)
    wm_manifest = require_file(args.wm_manifest, dry_run=args.dry_run)
    plan_action_config = planning_action_config(
        weights_dir, domain, dry_run=args.dry_run
    )
    wm_action_config = world_model_action_config(
        weights_dir, domain, dry_run=args.dry_run
    )
    wan_model_root = require_dir(args.wan_model_root, dry_run=args.dry_run)
    dinov3_repo = require_dir(args.dinov3_repo, dry_run=args.dry_run)
    dinov3_checkpoint = require_file(args.dinov3_checkpoint, dry_run=args.dry_run)
    dataset_root = (
        args.dataset_root.expanduser().resolve()
        if args.dataset_root
        else source_manifest.parent
    )
    wm_root = (
        args.wm_root.expanduser().resolve() if args.wm_root else wm_manifest.parent
    )
    if not args.dry_run and not wm_root.is_dir():
        raise FileNotFoundError(wm_root)
    candidates = run_dir / "candidates_n128.pt"
    selection = run_dir / "tr_top8_selection.pt"
    top8 = run_dir / "top8.json"
    observations = run_dir / "initial_observations.json"
    wm_runtime = run_dir / "wm_config.json"
    evidence = run_dir / "wm_evidence.pt"
    top1 = run_dir / "fused_top1.json"
    selected_actions = run_dir / "selected_actions.jsonl"
    wm_input_root = run_dir / "wm_input"
    action_manifest = wm_input_root / f"{args.split}.jsonl"
    video_dir = run_dir / "videos"

    python = sys.executable
    candidate_command = [
        python,
        "-m",
        "ldp.generate",
        "--domain",
        domain,
        "--split",
        args.split,
        "--checkpoint",
        str(proposal),
        "--inputs",
        str(queries),
        "--language-cache",
        str(language_cache),
        "--output",
        str(candidates),
        "--batch-size",
        str(args.ldp_batch_size),
        "--ddim-steps",
        str(args.ddim_steps),
        "--guidance",
        str(args.ldp_guidance),
        "--seed",
        str(args.seed),
    ]
    if plan_action_config is not None:
        candidate_command.extend(("--action-config", str(plan_action_config)))
    run_stage(
        "candidates_128",
        candidate_command,
        output=candidates,
        dry_run=args.dry_run,
    )

    top8_command = [
        python,
        "-m",
        "rankers.tr_ranker.inference",
        "--domain",
        domain,
        "--split",
        args.split,
        "--candidate-shards",
        str(candidates),
        "--language-cache",
        str(language_cache),
        "--tr-checkpoint",
        str(tr_checkpoint),
        "--output",
        str(selection),
        "--batch-size",
        str(args.ranker_batch_size),
    ]
    if plan_action_config is not None:
        top8_command.extend(("--action-config", str(plan_action_config)))
    run_stage(
        "rank_128_to_8",
        top8_command,
        output=selection,
        dry_run=args.dry_run,
    )
    run_stage(
        "materialize_top8",
        [
            python,
            "-m",
            "rankers.fusion.shortlist",
            "--domain",
            domain,
            "--split",
            args.split,
            "--candidate-shards",
            str(candidates),
            "--selection",
            str(selection),
            "--output",
            str(top8),
        ],
        output=top8,
        dry_run=args.dry_run,
    )
    observation_command = [
        python,
        "-m",
        "rankers.dataset.build_stage2_initial_observations",
        "--domain",
        domain,
        "--split",
        args.split,
        "--source-manifest",
        str(source_manifest),
        "--dataset-root",
        str(dataset_root),
        "--output",
        str(observations),
    ]
    run_stage(
        "stage2_initial_observations",
        observation_command,
        output=observations,
        dry_run=args.dry_run,
    )
    wm_config_command = [
        python,
        "-m",
        "lcwm.datasets.build_config",
        "--domain",
        domain,
        "--wan-model-root",
        str(wan_model_root),
        "--dinov3-repo",
        str(dinov3_repo),
        "--dinov3-checkpoint",
        str(dinov3_checkpoint),
        "--wm-checkpoint",
        str(wm_checkpoint),
        "--torch-dtype",
        args.dtype,
        "--guidance-scale",
        str(args.guidance_scale),
        "--num-inference-steps",
        str(args.num_inference_steps),
        "--output",
        str(wm_runtime),
    ]
    run_stage(
        "world_model_config",
        wm_config_command,
        output=wm_runtime,
        dry_run=args.dry_run,
    )
    evidence_command = [
        python,
        "-m",
        "rankers.v_ranker.build_evidence",
        "--domain",
        domain,
        "--split",
        args.split,
        "--top8",
        str(top8),
        "--initial-observations",
        str(observations),
        "--wm-config",
        str(wm_runtime),
        "--output",
        str(evidence),
        "--base-seed",
        str(args.seed),
    ]
    if wm_action_config is not None:
        evidence_command.extend(("--action-config", str(wm_action_config)))
    run_stage(
        "world_model_evidence",
        evidence_command,
        output=evidence,
        dry_run=args.dry_run,
    )
    run_stage(
        "rank_8_to_1",
        [
            python,
            "-m",
            "rankers.fusion.inference",
            "--domain",
            domain,
            "--split",
            args.split,
            "--evidence",
            str(evidence),
            "--v-ranker-checkpoint",
            str(v_ranker_checkpoint),
            "--output",
            str(top1),
        ],
        output=top1,
        dry_run=args.dry_run,
    )
    run_stage(
        "materialize_top1_action",
        [
            python,
            "-m",
            "rankers.fusion.export_actions",
            "--domain",
            domain,
            "--split",
            args.split,
            "--top8",
            str(top8),
            "--top1",
            str(top1),
            "--source-manifest",
            str(source_manifest),
            "--output",
            str(selected_actions),
        ],
        output=selected_actions,
        dry_run=args.dry_run,
    )

    adapter_command = [
        python,
        "-m",
        "rankers.fusion.actions_to_wm",
        "--domain",
        domain,
        "--selection",
        str(selected_actions),
        "--wm-manifest",
        str(wm_manifest),
        "--wm-root",
        str(wm_root),
        "--output-root",
        str(wm_input_root),
        "--split",
        args.split,
    ]
    if wm_action_config is not None:
        adapter_command.extend(("--action-config", str(wm_action_config)))
    run_stage(
        "adapt_top1_action_to_world_model",
        adapter_command,
        output=action_manifest,
        dry_run=args.dry_run,
    )

    video_command = [
        python,
        "-m",
        "lcwm.pure_infer",
        "--domain",
        domain,
        "--manifest",
        str(action_manifest),
        "--wan-model-root",
        str(wan_model_root),
        "--checkpoint",
        str(wm_checkpoint),
        "--out-dir",
        str(video_dir),
        "--dtype",
        args.dtype,
        "--fps",
        "8",
        "--seed",
        str(args.seed),
        "--guidance-scale",
        str(args.guidance_scale),
        "--num-inference-steps",
        str(args.num_inference_steps),
        "--dinov3-repo",
        str(dinov3_repo),
        "--dinov3-checkpoint",
        str(dinov3_checkpoint),
        "--save-mp4",
    ]
    run_stage(
        "generate_final_video",
        video_command,
        output=None,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        print(f"Dry run complete for {domain}: {run_dir}")
    else:
        print(f"Finished {domain}: {run_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=(*DOMAINS, "all"), required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--split", choices=PUBLIC_SPLITS)
    mode.add_argument("--input", type=Path)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument(
        "--pretrained-root", type=Path, default=ROOT / "pretrained_models"
    )
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    parser.add_argument(
        "--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16"
    )
    parser.add_argument("--ldp-batch-size", type=int, default=2)
    parser.add_argument("--ranker-batch-size", type=int, default=8)
    parser.add_argument("--ddim-steps", type=int, default=12)
    parser.add_argument("--ldp-guidance", type=float, default=1.5)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.ldp_batch_size < 1:
        parser.error("--ldp-batch-size must be at least 1")
    if args.ranker_batch_size < 1:
        parser.error("--ranker-batch-size must be at least 1")
    if args.ddim_steps < 1:
        parser.error("--ddim-steps must be at least 1")
    if not math.isfinite(args.ldp_guidance) or args.ldp_guidance < 0:
        parser.error("--ldp-guidance must be non-negative")
    if not math.isfinite(args.guidance_scale) or args.guidance_scale < 0:
        parser.error("--guidance-scale must be non-negative")
    if args.num_inference_steps < 1:
        parser.error("--num-inference-steps must be at least 1")
    if args.input is not None and args.domain == "all":
        parser.error("--input requires one domain")
    domains = DOMAINS if args.domain == "all" else (args.domain,)
    data_root = args.data_root.expanduser().resolve()
    pretrained_root = args.pretrained_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    for domain in domains:
        processed = data_root / domain
        domain_args = argparse.Namespace(**vars(args))
        domain_args.domain = domain
        if args.input is None:
            split = args.split
            domain_args.queries = processed / "planning" / f"{split}.json"
            domain_args.source_manifest = processed / "planning" / f"{split}.jsonl"
            domain_args.wm_manifest = processed / "wm" / f"{split}.jsonl"
            domain_args.run_dir = output_root / domain / split
            domain_args.language_cache = processed / f"language_{split}.pt"
            domain_args.dataset_root = processed / "planning"
            domain_args.wm_root = processed / "wm"
        else:
            domain_args.run_dir = output_root / domain / "input"
            artifacts = prepare_input(
                domain,
                args.input,
                domain_args.run_dir,
                dry_run=args.dry_run,
            )
            split = INPUT_SCOPE
            domain_args.queries = artifacts.queries
            domain_args.source_manifest = artifacts.planning_manifest
            domain_args.wm_manifest = artifacts.wm_manifest
            domain_args.language_cache = domain_args.run_dir / "input_language.pt"
            domain_args.dataset_root = domain_args.run_dir
            domain_args.wm_root = domain_args.run_dir
        domain_args.split = split
        domain_args.wan_model_root = pretrained_root / "Wan2.2-TI2V-5B"
        domain_args.dinov3_repo = pretrained_root / "dinov3"
        domain_args.dinov3_checkpoint = (
            pretrained_root / "dinov3_weights" / "dinov3_vitl16_pretrain_lvd1689m.pth"
        )
        if args.input is not None:
            run_stage(
                "cache_input_language",
                [
                    sys.executable,
                    "-m",
                    "ldp.cache_language",
                    "--domain",
                    domain,
                    "--wan-model-root",
                    str(domain_args.wan_model_root),
                    "--input-manifest",
                    str(domain_args.queries),
                    "--output",
                    str(domain_args.language_cache),
                ],
                output=domain_args.language_cache,
                dry_run=args.dry_run,
            )
        full_pipeline(domain_args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
