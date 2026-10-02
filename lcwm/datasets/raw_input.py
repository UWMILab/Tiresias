from __future__ import annotations

import json
import pickletools
import re
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lcwm.config import GAMING_METERS_PER_UNIT
from lcwm.datasets.protocol import PLANNING_ANCHOR_FRAME_IDX, write_records
from ldp.action_utils import DRIVING_DT
from ldp.policy import CANDIDATE_FIELDS, HORIZON


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
INPUT_SCOPE = "input"


@dataclass(frozen=True)
class InputArtifacts:
    queries: Path
    planning_manifest: Path
    wm_manifest: Path


@dataclass(frozen=True)
class RawTrajectory:
    trajectory_id: str
    instruction: str
    source_name: str
    frames: tuple[Path, ...]


@dataclass(frozen=True)
class _PickleGlobal:
    module: str
    name: str


@dataclass(frozen=True)
class _NumpyDType:
    code: str
    byteorder: str = "="


@dataclass(frozen=True)
class _NumpyArrayShell:
    pass


METADATA_KEYS = {
    "instruction",
    "language_instruction",
    "prompt",
    "text",
    "source_name",
    "game_domain",
}
PICKLE_STRING_OPS = {
    "UNICODE",
    "BINUNICODE",
    "BINUNICODE8",
    "SHORT_BINUNICODE",
    "STRING",
    "BINSTRING",
    "SHORT_BINSTRING",
}
PICKLE_NUMBER_OPS = {
    "INT",
    "BININT",
    "BININT1",
    "BININT2",
    "LONG",
    "LONG1",
    "LONG4",
    "FLOAT",
    "BINFLOAT",
}
PICKLE_MEMO_WRITES = {"PUT", "BINPUT", "LONG_BINPUT"}
PICKLE_MEMO_READS = {"GET", "BINGET", "LONG_BINGET"}
PICKLE_NOOPS = {"PROTO", "FRAME"}
PICKLE_OPAQUE_PUSHES = {
    "EMPTY_LIST",
    "EMPTY_SET",
    "EXT1",
    "EXT2",
    "EXT4",
    "NEXT_BUFFER",
    "PERSID",
}
PICKLE_BYTES_OPS = {"BINBYTES", "BINBYTES8", "SHORT_BINBYTES", "BYTEARRAY8"}
PICKLE_MARK_CONTAINERS = {"LIST", "FROZENSET", "OBJ", "INST"}
PICKLE_FIXED_TUPLES = {"TUPLE1": 1, "TUPLE2": 2, "TUPLE3": 3}
PICKLE_CONSTRUCTORS = {"NEWOBJ": 2, "NEWOBJ_EX": 3}
_PICKLE_MARK = object()
_PICKLE_OPAQUE = object()


def _normal(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower()).removesuffix("segment")


def _pop_mark(stack: list[Any], path: Path) -> list[Any]:
    for index in range(len(stack) - 1, -1, -1):
        if stack[index] is _PICKLE_MARK:
            values = stack[index + 1 :]
            del stack[index:]
            return values
    raise ValueError(f"invalid pickle mark stack: {path}")


def _set_items(target: Any, values: list[Any], path: Path) -> None:
    if len(values) % 2:
        raise ValueError(f"invalid pickle dictionary entries: {path}")
    if not isinstance(target, dict):
        return
    for index in range(0, len(values), 2):
        key, value = values[index : index + 2]
        if isinstance(key, (str, int, float, bool, bytes)):
            target[key] = value


def _pickle_global(argument: Any) -> _PickleGlobal:
    module, name = str(argument).split(" ", 1)
    return _PickleGlobal(module, name)


def _numpy_float(dtype: _NumpyDType, value: Any) -> float | None:
    raw = bytes(value) if isinstance(value, (bytes, bytearray)) else None
    formats = {"f4": "f", "float32": "f", "f8": "d", "float64": "d"}
    code = formats.get(dtype.code)
    if raw is None or code is None or len(raw) != struct.calcsize(code):
        return None
    prefix = dtype.byteorder if dtype.byteorder in {"<", ">", "="} else "="
    return float(struct.unpack(prefix + code, raw)[0])


def _safe_reduce(function: Any, arguments: Any) -> Any:
    if not isinstance(function, _PickleGlobal) or not isinstance(arguments, tuple):
        return _PICKLE_OPAQUE
    if function.module == "numpy" and function.name == "dtype":
        if arguments and isinstance(arguments[0], str):
            return _NumpyDType(arguments[0])
        return _PICKLE_OPAQUE
    if function.module.endswith(".multiarray") and function.name == "scalar":
        if len(arguments) == 2 and isinstance(arguments[0], _NumpyDType):
            value = _numpy_float(arguments[0], arguments[1])
            return value if value is not None else _PICKLE_OPAQUE
        return _PICKLE_OPAQUE
    if function.module.endswith(".numeric") and function.name == "_frombuffer":
        if (
            len(arguments) == 4
            and isinstance(arguments[1], _NumpyDType)
            and arguments[2] == ()
        ):
            value = _numpy_float(arguments[1], arguments[0])
            return value if value is not None else _PICKLE_OPAQUE
    if function.module.endswith(".multiarray") and function.name == "_reconstruct":
        return _NumpyArrayShell()
    return _PICKLE_OPAQUE


def _scalar_metadata(path: Path) -> dict[str, Any]:
    """Read top-level scalar metadata without constructing pickled arrays."""

    stack: list[Any] = []
    memo: dict[int, Any] = {}
    root: Any = _PICKLE_OPAQUE
    try:
        with path.open("rb") as stream:
            for opcode, argument, _ in pickletools.genops(stream):
                name = opcode.name
                if name in PICKLE_NOOPS:
                    continue
                if name in PICKLE_STRING_OPS:
                    if isinstance(argument, bytes):
                        argument = argument.decode("utf-8")
                    stack.append(argument)
                elif name in PICKLE_NUMBER_OPS:
                    stack.append(argument)
                elif name == "NONE":
                    stack.append(None)
                elif name == "NEWTRUE":
                    stack.append(True)
                elif name == "NEWFALSE":
                    stack.append(False)
                elif name == "EMPTY_DICT":
                    stack.append({})
                elif name == "EMPTY_TUPLE":
                    stack.append(())
                elif name == "MARK":
                    stack.append(_PICKLE_MARK)
                elif name == "MEMOIZE":
                    memo[len(memo)] = stack[-1]
                elif name in PICKLE_MEMO_WRITES:
                    memo[int(argument)] = stack[-1]
                elif name in PICKLE_MEMO_READS:
                    stack.append(memo[int(argument)])
                elif name in PICKLE_BYTES_OPS:
                    stack.append(bytes(argument))
                elif name in PICKLE_OPAQUE_PUSHES:
                    stack.append(_PICKLE_OPAQUE)
                elif name == "GLOBAL":
                    stack.append(_pickle_global(argument))
                elif name == "STACK_GLOBAL":
                    module, global_name = stack[-2:]
                    del stack[-2:]
                    stack.append(_PickleGlobal(str(module), str(global_name)))
                elif name == "TUPLE":
                    stack.append(tuple(_pop_mark(stack, path)))
                elif name in PICKLE_MARK_CONTAINERS:
                    _pop_mark(stack, path)
                    stack.append(_PICKLE_OPAQUE)
                elif name in PICKLE_FIXED_TUPLES:
                    count = PICKLE_FIXED_TUPLES[name]
                    values = tuple(stack[-count:])
                    del stack[-count:]
                    stack.append(values)
                elif name in PICKLE_CONSTRUCTORS:
                    count = PICKLE_CONSTRUCTORS[name]
                    del stack[-count:]
                    stack.append(_PICKLE_OPAQUE)
                elif name == "REDUCE":
                    function, arguments = stack[-2:]
                    del stack[-2:]
                    stack.append(_safe_reduce(function, arguments))
                elif name == "BUILD":
                    instance, state = stack[-2:]
                    del stack[-2:]
                    if (
                        isinstance(instance, _NumpyDType)
                        and isinstance(state, tuple)
                        and len(state) > 1
                        and state[1] in {"<", ">", "=", "|"}
                    ):
                        instance = _NumpyDType(instance.code, state[1])
                    elif (
                        isinstance(instance, _NumpyArrayShell)
                        and isinstance(state, tuple)
                        and len(state) == 5
                        and state[1] == ()
                        and isinstance(state[2], _NumpyDType)
                    ):
                        value = _numpy_float(state[2], state[4])
                        instance = value if value is not None else _PICKLE_OPAQUE
                    else:
                        instance = _PICKLE_OPAQUE
                    stack.append(instance)
                elif name == "DICT":
                    values = _pop_mark(stack, path)
                    target: dict[Any, Any] = {}
                    _set_items(target, values, path)
                    stack.append(target)
                elif name == "SETITEM":
                    value = stack.pop()
                    key = stack.pop()
                    _set_items(stack[-1], [key, value], path)
                elif name == "SETITEMS":
                    values = _pop_mark(stack, path)
                    _set_items(stack[-1], values, path)
                elif name in {"APPEND", "ADDITEMS"}:
                    if name == "APPEND":
                        stack.pop()
                    else:
                        _pop_mark(stack, path)
                elif name == "APPENDS":
                    _pop_mark(stack, path)
                elif name == "BINPERSID":
                    stack.pop()
                    stack.append(_PICKLE_OPAQUE)
                elif name == "READONLY_BUFFER":
                    continue
                elif name == "DUP":
                    stack.append(stack[-1])
                elif name == "POP":
                    stack.pop()
                elif name == "POP_MARK":
                    _pop_mark(stack, path)
                elif name == "STOP":
                    root = stack.pop()
                    break
                else:
                    raise ValueError(f"unsupported pickle opcode {name}: {path}")
    except (IndexError, KeyError, UnicodeDecodeError, ValueError) as error:
        raise ValueError(f"cannot read scalar metadata from {path}") from error
    if not isinstance(root, dict):
        raise TypeError(f"{path} must contain a dictionary")
    return {
        key: value
        for key, value in root.items()
        if key in METADATA_KEYS and isinstance(value, (str, int, float))
    }


def _numbered_frames(directory: Path) -> tuple[Path, ...]:
    def key(path: Path) -> tuple[int, int | str]:
        try:
            return 0, int(path.stem)
        except ValueError:
            return 1, path.name

    frames = tuple(
        sorted(
            (
                path
                for path in directory.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ),
            key=key,
        )
    )
    required = PLANNING_ANCHOR_FRAME_IDX + 1
    if len(frames) < required:
        raise ValueError(
            f"{directory} has {len(frames)} RGB frames; at least {required} are required"
        )
    return frames


def _instruction(payload: dict[str, Any], path: Path) -> str:
    for key in ("instruction", "language_instruction", "prompt", "text"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise KeyError(f"{path} has no instruction")


def _declared_source(payload: dict[str, Any]) -> str | None:
    for key in ("source_name", "game_domain"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _canonical_source(domain: str, directory: Path, payload: dict[str, Any]) -> str:
    aliases = {
        "navigation": {"lcvn": "LCVN", "tartandrive": "TartanDrive"},
        "driving": {
            "nuscenes": "nuScenes",
            "kitti360": "KITTI360",
            "waymo": "Waymo",
            "drivingdojo": "DrivingDojo",
        },
        "gaming": {
            "cs": "CS",
            "csgo": "CS",
            "mc": "MC",
            "minecraft": "MC",
        },
        "robot": {
            "droid": "DROID",
            "bridgedatav2": "BridgeDataV2",
            "bridgev2": "BridgeDataV2",
            "rt1": "RT-1",
            "liberowrist": "LIBERO_wrist",
            "liberoagent": "LIBERO_agent",
            "liberoagentview": "LIBERO_agent",
        },
    }[domain]
    declared = _declared_source(payload)
    if declared is not None:
        canonical = aliases.get(_normal(declared))
        if canonical is not None:
            return canonical
        if domain != "gaming":
            return declared

    for part in reversed(directory.parts):
        compact = re.sub(r"[^a-z0-9]+", "", part.lower())
        if domain == "robot" and compact.startswith("libero"):
            if compact.endswith("wrist"):
                return "LIBERO_wrist"
            if compact.endswith(("agent", "agentview")):
                return "LIBERO_agent"
        canonical = aliases.get(_normal(part))
        if canonical is not None:
            return canonical
        for alias, candidate in aliases.items():
            if compact.startswith(f"{alias}segment") or (
                domain == "robot" and compact.startswith(alias)
            ):
                return candidate

    name = directory.name
    lower = name.lower()
    if domain == "navigation":
        return "TartanDrive" if re.match(r"^\d{8}_", name) else "LCVN"
    if domain == "driving":
        if lower.startswith("scene"):
            return "nuScenes"
        if lower.startswith("2013_05_28_drive_"):
            return "KITTI360"
        first = name.split("_", 1)[0]
        return "Waymo" if first.isdigit() and len(first) > 10 else "DrivingDojo"
    if domain == "gaming":
        if lower.startswith(("aba_", "abca_")):
            return "MC"
        if lower.startswith(("dust2_", "inferno_", "mirage_")):
            return "CS"
        raise ValueError(
            f"cannot identify the Gaming source for {directory}; "
            "set source_name to CS or MC in traj_data.pkl"
        )
    if lower.startswith("traj_droid_"):
        return "DROID"
    if lower.startswith("bridgev2_"):
        return "BridgeDataV2"
    if lower.startswith("rt1_"):
        return "RT-1"
    if lower.startswith("libero_") and "_wrist_" in lower:
        return "LIBERO_wrist"
    if lower.startswith("libero_") and "_agentview_" in lower:
        return "LIBERO_agent"
    return "Robot"


def _discover(domain: str, input_path: Path) -> list[RawTrajectory]:
    root = input_path.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    direct = root / "traj_data.pkl"
    paths = [direct] if direct.is_file() else sorted(root.rglob("traj_data.pkl"))
    if not paths:
        raise FileNotFoundError(f"no traj_data.pkl found under {root}")

    rows = []
    identities: set[str] = set()
    for path in paths:
        directory = path.parent
        payload = _scalar_metadata(path)
        relative = Path(directory.name) if direct.is_file() else directory.relative_to(root)
        trajectory_id = "__".join(relative.parts)
        if not trajectory_id or trajectory_id in identities:
            raise ValueError(f"duplicate or empty trajectory identity: {trajectory_id!r}")
        identities.add(trajectory_id)
        rows.append(
            RawTrajectory(
                trajectory_id=trajectory_id,
                instruction=_instruction(payload, path),
                source_name=_canonical_source(domain, directory, payload),
                frames=_numbered_frames(directory),
            )
        )
    return rows


def prepare_input(
    domain: str,
    input_path: Path,
    output_dir: Path,
    *,
    dry_run: bool = False,
) -> InputArtifacts:
    if domain not in HORIZON:
        raise ValueError(f"unsupported domain: {domain}")
    trajectories = _discover(domain, input_path)
    output_dir = output_dir.expanduser().resolve()
    artifacts = InputArtifacts(
        queries=output_dir / "input_queries.json",
        planning_manifest=output_dir / "input_planning.jsonl",
        wm_manifest=output_dir / "input_wm.jsonl",
    )
    context_root = output_dir / "input_frames"
    occupied = [
        path
        for path in (
            artifacts.queries,
            artifacts.planning_manifest,
            artifacts.wm_manifest,
            context_root,
        )
        if path.exists()
    ]
    if occupied:
        raise FileExistsError(occupied[0])
    if dry_run:
        return artifacts

    query_rows = []
    planning_rows = []
    wm_rows = []
    for index, trajectory in enumerate(trajectories):
        context_dir = context_root / trajectory.trajectory_id
        context_dir.mkdir(parents=True)
        copied = []
        for frame_index, source in enumerate(
            trajectory.frames[: PLANNING_ANCHOR_FRAME_IDX + 1]
        ):
            destination = context_dir / f"{frame_index:06d}{source.suffix.lower()}"
            shutil.copy2(source, destination)
            copied.append(destination)
        anchor = copied[PLANNING_ANCHOR_FRAME_IDX]
        sample_key = f"{INPUT_SCOPE}:{trajectory.trajectory_id}"
        sample_id = (
            trajectory.trajectory_id if domain == "navigation" else sample_key
        )
        common: dict[str, Any] = {
            "domain": domain,
            "split": INPUT_SCOPE,
            "dataset_index": index,
            "sample_idx": index,
            "sample_id": sample_id,
            "sample_key": sample_key,
            "trajectory_id": trajectory.trajectory_id,
            "instruction": trajectory.instruction,
            "source_name": trajectory.source_name,
        }
        if domain == "driving":
            common["source_dt"] = DRIVING_DT
        query_rows.append(
            {
                key: common[key]
                for key in (
                    "domain",
                    "split",
                    "dataset_index",
                    "sample_id",
                    "sample_key",
                    "trajectory_id",
                    "instruction",
                    "source_name",
                )
            }
        )
        planning = {
            **common,
            "video": str(context_dir),
            "first_rgb_path": str(anchor),
            "action_fields": ";".join(CANDIDATE_FIELDS[domain]),
            "future_horizon": HORIZON[domain],
        }
        if domain == "gaming":
            planning["translation_scale_to_meter"] = GAMING_METERS_PER_UNIT[
                trajectory.source_name
            ]
        planning_rows.append(planning)
        wm_rows.append(
            {
                **common,
                "video": str(context_dir),
                "action_fields": ";".join(CANDIDATE_FIELDS[domain]),
                "action_dim": len(CANDIDATE_FIELDS[domain]),
                "context_frame_idx": PLANNING_ANCHOR_FRAME_IDX,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts.queries.write_text(
        json.dumps(
            {
                "domain": domain,
                "split": INPUT_SCOPE,
                "record_count": len(query_rows),
                "records": query_rows,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    write_records(artifacts.planning_manifest, planning_rows)
    write_records(artifacts.wm_manifest, wm_rows)
    return artifacts
