import os
from pathlib import Path

DOMAINS = ("navigation", "driving", "gaming", "robot")
ACTION_DIMS = {"navigation": 3, "driving": 4, "gaming": 4, "robot": 7}
GAMING_METERS_PER_UNIT = {"CS": 0.0254, "MC": 1.0}

SYSTEM_PROMPTS = {
    "navigation": (
        "You are a language-conditioned navigation world model. One initial observation is provided "
        "together with the route instruction and the full action sequence. Generate the complete future "
        "agent-view video in one shot. Treat the instruction as the route semantics and the action sequence "
        "as transition control for timing, direction, and motion scale. Keep motion physically plausible, "
        "temporally continuous, and faithful to both instruction and action."
    ),
    "driving": (
        "You are a language-conditioned self-driving world model. One initial front-facing observation is "
        "provided together with the route instruction and the full ego-action sequence. Generate the complete "
        "future driving video in one shot. Treat the instruction as route semantics and the action sequence "
        "as transition control for speed, direction, and motion scale. Keep road geometry, ego-motion, traffic, "
        "and lighting physically plausible and temporally continuous. Be faithful to both instruction and action."
    ),
    "gaming": (
        "You are a language-conditioned first-person gaming world model. One initial observation is provided "
        "together with the gameplay instruction and the full ego-action sequence. Generate the complete future "
        "first-person gaming video in one shot. Treat the instruction as task semantics and the action sequence "
        "as transition control for forward motion, lateral strafing, yaw rotation (looking left/right), and "
        "vertical movement (jumping, falling, stairs). Keep scene geometry, player motion, lighting, and object "
        "appearance physically plausible and temporally continuous. Be faithful to both instruction and action."
    ),
    "robot": (
        "You are a language-conditioned robot-manipulation world model. One initial observation is provided "
        "together with the manipulation instruction and the full 7-DoF end-effector action sequence. Generate "
        "the complete future video in one shot. Preserve the viewpoint stated at the beginning of the instruction: "
        "a wrist-mounted first-person view must remain wrist-mounted first-person, and an external third-person "
        "view must remain external third-person. In an external third-person view, camera extrinsics are fixed: "
        "never pan, tilt, zoom, dolly, orbit, or track the robot. In a wrist-mounted first-person view, the camera "
        "moves only because it is rigidly attached to the robot wrist. The actor is a robotic arm with a mechanical "
        "gripper; never replace it with a human hand, human arm, skin, or fingers. Preserve whether the initial "
        "scene is real or simulated rather than changing visual domain. Treat the instruction as task and contact "
        "semantics, and the action sequence as transition control for end-effector translation, rotation, and "
        "gripper opening or closing. Maintain robot identity, geometry, kinematic continuity, object permanence, "
        "contact, grasp, lift, transport, placement, and release."
    ),
}

_GENERAL_NEGATIVE_PROMPT = (
    "oversaturated colors, overexposed, static, blurry details, subtitles, style, "
    "artwork, painting, still image, overall gray tone, worst quality, low quality, "
    "JPEG compression artifacts, ugly, malformed, extra fingers, poorly drawn hands, "
    "poorly drawn face, deformed, disfigured, malformed limbs, fused fingers, frozen "
    "frame, cluttered background, crowd, duplicate objects"
)

NEGATIVE_PROMPTS = {
    "navigation": _GENERAL_NEGATIVE_PROMPT,
    "driving": _GENERAL_NEGATIVE_PROMPT,
    "gaming": _GENERAL_NEGATIVE_PROMPT,
    "robot": (
        "oversaturated colors, overexposed, static, blurry details, subtitles, style, "
        "artwork, painting, still image, overall gray tone, worst quality, low quality, "
        "JPEG compression artifacts, ugly, malformed, human hand, human arm, human "
        "fingers, skin, flesh, person, biological hand, extra fingers, poorly drawn "
        "hands, deformed gripper, missing robot arm, embodiment change, viewpoint change, "
        "camera cut, real-to-sim style change, sim-to-real style change, frozen frame, "
        "cluttered background, crowd, duplicate objects"
    ),
}


def joint_text(
    domain: str, instruction: str, *, system_prompt: str | None = None
) -> str:
    if domain not in SYSTEM_PROMPTS:
        raise ValueError(f"unsupported domain: {domain}")
    expected = SYSTEM_PROMPTS[domain]
    prompt = expected if system_prompt is None else str(system_prompt).strip()
    if prompt != expected:
        raise ValueError(f"{domain}: unexpected system prompt")
    instruction = str(instruction).strip()
    if not instruction:
        raise ValueError(f"{domain}: empty instruction")
    return f"System: {prompt}\nInstruction: {instruction}"


def validate_wan_runtime(repo_root: Path) -> Path:
    runtime = (
        repo_root
        / "lcwm"
        / "wan22"
        / "diffsynth"
        / "pipelines"
        / "wan_video.py"
    )
    if not runtime.is_file():
        raise FileNotFoundError(runtime)
    return runtime


def world_model_checkpoint(repo_root: Path, domain: str) -> Path:
    if domain not in DOMAINS:
        raise ValueError(f"unsupported domain: {domain}")
    return repo_root / "weights" / "world_model" / domain / "model.safetensors"


def validate_checkpoint(checkpoint: Path) -> None:
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)


def subprocess_environment(repo_root: Path) -> dict[str, str]:
    environment = os.environ.copy()
    entries = [str(repo_root)]
    current = environment.get("PYTHONPATH")
    if current:
        entries.extend(item for item in current.split(os.pathsep) if item)
    environment["PYTHONPATH"] = os.pathsep.join(entries)
    return environment
