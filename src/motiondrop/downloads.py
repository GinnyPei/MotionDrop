"""Reproducible model and benchmark asset downloads.

The public benchmark archives are large enough that downloading and extracting
them as one opaque snapshot is both wasteful and difficult to resume.  This
module downloads pinned upstream revisions, extracts one archive at a time,
and records enough state to make interrupted runs safe to repeat.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import stat
import tarfile
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import BinaryIO
from zipfile import BadZipFile, ZipFile, ZipInfo

from motiondrop.paths import data_dir, models_dir

GIB = 1024**3
DOWNLOAD_STATE_VERSION = 1

OVOBENCH_REPO = "JoeLeelyf/OVO-Bench"
OVOBENCH_REVISION = "fec29e3385747b5642d995370143ba92d2819bd2"
OVOBENCH_CODE_REVISION = "c34093f9a072d93f9e1c6b923446dfba71f4528e"
OVOBENCH_ANNOTATION_URL = (
    "https://raw.githubusercontent.com/JoeLeelyf/OVO-Bench/"
    f"{OVOBENCH_CODE_REVISION}/data/ovo_bench_new.json"
)

STREAMINGBENCH_REPO = "mjuicem/StreamingBench"
STREAMINGBENCH_REVISION = "48872fa707124474ce4c5172ddc58efb8bc88058"


@dataclass(frozen=True)
class ModelDownloadSpec:
    alias: str
    repo_id: str
    revision: str
    bytes: int
    model_type: str


MODEL_DOWNLOAD_SPECS = {
    "llava-ov-7b": ModelDownloadSpec(
        "llava-ov-7b",
        "llava-hf/llava-onevision-qwen2-7b-ov-hf",
        "0d50680527681998e456c7b78950205bedd8a068",
        16_073_497_233,
        "llava_onevision",
    ),
    "qwen3-vl-8b": ModelDownloadSpec(
        "qwen3-vl-8b",
        "Qwen/Qwen3-VL-8B-Instruct",
        "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
        17_545_915_883,
        "qwen3_vl",
    ),
    "qwen3.5-9b": ModelDownloadSpec(
        "qwen3.5-9b",
        "Qwen/Qwen3.5-9B",
        "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        19_329_393_661,
        "qwen3_5",
    ),
}


@dataclass(frozen=True)
class ArchiveSpec:
    filename: str
    target: str
    compressed_bytes: int
    extracted_bytes: int
    proactive: bool = False


STREAMING_ARCHIVES = (
    ArchiveSpec(
        "Anomaly Context Understanding.zip", "videos/anomaly_context", 9_178_531_292, 9_222_226_473
    ),
    ArchiveSpec(
        "Emotion Recognition.zip", "videos/emotion_recognition", 5_678_597_769, 5_701_320_003
    ),
    ArchiveSpec(
        "Misleading Context Understanding.zip",
        "videos/misleading_context",
        5_905_047_800,
        5_929_470_798,
    ),
    ArchiveSpec(
        "Multimodal Alignment.zip", "videos/multimodal_alignment", 9_582_631_101, 9_598_165_166
    ),
    ArchiveSpec(
        "Real-Time Visual Understanding_1-50.zip",
        "videos/realtime_visual_understanding",
        9_080_551_796,
        9_096_107_208,
    ),
    ArchiveSpec(
        "Real-Time Visual Understanding_51-100.zip",
        "videos/realtime_visual_understanding",
        8_708_188_924,
        8_732_535_827,
    ),
    ArchiveSpec(
        "Real-Time Visual Understanding_101-150.zip",
        "videos/realtime_visual_understanding",
        8_218_047_488,
        8_259_830_225,
    ),
    ArchiveSpec(
        "Real-Time Visual Understanding_151-200.zip",
        "videos/realtime_visual_understanding",
        17_984_397_733,
        18_008_599_576,
    ),
    ArchiveSpec(
        "Real-Time Visual Understanding_201-250.zip",
        "videos/realtime_visual_understanding",
        2_230_277_873,
        2_255_246_687,
    ),
    ArchiveSpec(
        "Real-Time Visual Understanding_251-300.zip",
        "videos/realtime_visual_understanding",
        18_838_513_104,
        18_862_302_697,
    ),
    ArchiveSpec(
        "Real-Time Visual Understanding_301-350.zip",
        "videos/realtime_visual_understanding",
        16_909_427_777,
        17_103_706_576,
    ),
    ArchiveSpec(
        "Real-Time Visual Understanding_351-400.zip",
        "videos/realtime_visual_understanding",
        17_902_292_310,
        17_919_685_551,
    ),
    ArchiveSpec(
        "Real-Time Visual Understanding_401-450.zip",
        "videos/realtime_visual_understanding",
        18_224_908_176,
        18_241_226_723,
    ),
    ArchiveSpec(
        "Real-Time Visual Understanding_451-500.zip",
        "videos/realtime_visual_understanding",
        13_468_580_983,
        13_518_651_924,
    ),
    ArchiveSpec(
        "Scene Understanding_1-25.zip", "videos/scene_understanding", 5_662_435_592, 5_666_703_629
    ),
    ArchiveSpec(
        "Scene Understanding_26-50.zip", "videos/scene_understanding", 8_167_060_655, 8_178_364_025
    ),
    ArchiveSpec(
        "Sequential Question Answering_1-25.zip",
        "videos/sequential_question_answering",
        5_974_660_729,
        5_983_093_081,
    ),
    ArchiveSpec(
        "Sequential Question Answering_26-50.zip",
        "videos/sequential_question_answering",
        7_353_922_281,
        7_369_836_251,
    ),
    ArchiveSpec(
        "Source Discrimination.zip", "videos/source_discrimination", 4_690_676_888, 4_703_360_525
    ),
    ArchiveSpec(
        "Proactive Output_1-25.zip", "videos/proactive_output", 4_477_402_711, 4_491_461_172, True
    ),
    ArchiveSpec(
        "Proactive Output_26-50.zip", "videos/proactive_output", 4_311_616_739, 4_323_667_355, True
    ),
)

STREAMING_ANNOTATIONS = (
    "StreamingBench/Contextual_Understanding.csv",
    "StreamingBench/Omni_Source_Understanding.csv",
    "StreamingBench/Proactive_Output.csv",
    "StreamingBench/Proactive_Output_50.csv",
    "StreamingBench/Real_Time_Visual_Understanding.csv",
    "StreamingBench/Sequential_Question_Answering.csv",
)
STREAMING_ANNOTATION_BYTES = 1_857_660

OVOBENCH_PART_SIZES = {
    **{f"chunked_videos.tar.parta{suffix}": 10_737_418_240 for suffix in "abcdefghijklmn"},
    "chunked_videos.tar.partao": 2_953_297_920,
}
OVOBENCH_ARCHIVE_BYTES = sum(OVOBENCH_PART_SIZES.values())

MODEL_STATE_FILE = ".motiondrop-download.json"


class DownloadError(RuntimeError):
    """An actionable asset preparation failure."""


def _retry_hub(label: str, operation, *, attempts: int = 5):
    """Retry a Hub operation; repeated calls resume local partial downloads."""

    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except Exception as exc:
            if attempt == attempts:
                raise DownloadError(
                    f"Hugging Face operation failed after {attempts} attempts ({label}): {exc}"
                ) from exc
            delay = min(2 ** (attempt - 1), 8)
            print(
                f"[download] Hub retry {attempt}/{attempts - 1} for {label} in {delay}s: {exc}",
                flush=True,
            )
            time.sleep(delay)
    raise AssertionError("unreachable")


def format_bytes(value: int) -> str:
    """Format a byte count using binary units."""

    if value < 1024:
        return f"{value} B"
    number = float(value)
    for unit in ("KiB", "MiB", "GiB", "TiB"):
        number /= 1024.0
        if number < 1024.0 or unit == "TiB":
            return f"{number:.1f} {unit}"
    raise AssertionError("unreachable")


def normalize_selection(values: Iterable[str], choices: Iterable[str], label: str) -> list[str]:
    """Normalize comma/space separated ``all`` and ``none`` selections."""

    ordered_choices = list(choices)
    tokens = [token for value in values for token in value.split(",") if token]
    if not tokens:
        raise ValueError(f"{label} selection cannot be empty")
    if "all" in tokens:
        if len(tokens) != 1:
            raise ValueError(f"Use 'all' by itself for {label}")
        return ordered_choices
    if "none" in tokens:
        if len(tokens) != 1:
            raise ValueError(f"Use 'none' by itself for {label}")
        return []
    unknown = sorted(set(tokens) - set(ordered_choices))
    if unknown:
        raise ValueError(
            f"Unknown {label}: {', '.join(unknown)}; choices: {', '.join(ordered_choices)}"
        )
    selected = set(tokens)
    return [choice for choice in ordered_choices if choice in selected]


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_model_directory(path: Path, alias: str) -> dict:
    """Validate checkpoint metadata and every indexed safetensors shard."""

    spec = MODEL_DOWNLOAD_SPECS[alias]
    config_path = path / "config.json"
    if not config_path.is_file():
        raise DownloadError(f"Missing model config: {config_path}")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DownloadError(f"Invalid model config: {config_path}") from exc
    if config.get("model_type") != spec.model_type:
        raise DownloadError(
            f"Unexpected model_type in {config_path}: {config.get('model_type')!r}; "
            f"expected {spec.model_type!r}"
        )

    index_paths = sorted(path.glob("*.safetensors.index.json"))
    if not index_paths:
        single_files = sorted(path.glob("*.safetensors"))
        if not single_files:
            raise DownloadError(f"No safetensors weights found under {path}")
        shards = single_files
    else:
        shards_by_name: dict[str, Path] = {}
        for index_path in index_paths:
            try:
                index = json.loads(index_path.read_text(encoding="utf-8"))
                filenames = set(index["weight_map"].values())
            except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
                raise DownloadError(f"Invalid weight index: {index_path}") from exc
            for filename in filenames:
                shards_by_name[filename] = path / filename
        shards = [shards_by_name[name] for name in sorted(shards_by_name)]

    missing = [str(shard) for shard in shards if not shard.is_file() or shard.stat().st_size == 0]
    if missing:
        raise DownloadError(f"Missing model weight shards; first paths: {missing[:3]}")
    if not (path / "preprocessor_config.json").is_file():
        raise DownloadError(f"Missing processor config under {path}")
    return {
        "alias": alias,
        "model_type": spec.model_type,
        "weight_shards": len(shards),
        "weight_bytes": sum(shard.stat().st_size for shard in shards),
    }


def model_is_ready(path: Path, alias: str) -> bool:
    try:
        validate_model_directory(path, alias)
        return True
    except (DownloadError, OSError):
        return False


def downloaded_model_path(models_dir: str | Path, alias: str) -> Path | None:
    """Return a validated project-local checkpoint path when one exists."""

    path = Path(models_dir).expanduser() / alias
    state_path = path / MODEL_STATE_FILE
    if not state_path.is_file() or not model_is_ready(path, alias):
        return None
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    spec = MODEL_DOWNLOAD_SPECS[alias]
    if state.get("repo_id") != spec.repo_id:
        return None
    if state.get("revision") != spec.revision:
        return None
    return path.resolve()


def local_model_path(models_dir: str | Path, alias: str) -> Path | None:
    """Return a usable local checkpoint for inference.

    Prefers a pinned download; otherwise accepts a complete checkpoint of the
    right type at ``models_dir/alias`` or ``models_dir/<repo name>`` (the
    directory name a plain Hugging Face download uses).
    """

    pinned = downloaded_model_path(models_dir, alias)
    if pinned is not None:
        return pinned
    root = Path(models_dir).expanduser()
    for name in (alias, MODEL_DOWNLOAD_SPECS[alias].repo_id.rsplit("/", 1)[-1]):
        if model_is_ready(root / name, alias):
            return (root / name).resolve()
    return None


def _safe_relative_path(name: str) -> Path:
    if "\\" in name:
        raise DownloadError(f"Archive member uses a backslash path: {name!r}")
    pure = PurePosixPath(name)
    if pure.is_absolute() or any(part == ".." for part in pure.parts):
        raise DownloadError(f"Unsafe archive member path: {name!r}")
    parts = [part for part in pure.parts if part not in {"", "."}]
    if not parts:
        return Path()
    return Path(*parts)


def _safe_destination(root: Path, relative: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    resolved_root = root.resolve()
    resolved_parent = destination.parent.resolve()
    if not resolved_parent.is_relative_to(resolved_root):
        raise DownloadError(f"Archive path escapes extraction root: {relative}")
    if destination.is_symlink():
        raise DownloadError(f"Refusing to overwrite symlink: {destination}")
    return destination


def _copy_atomic(source: BinaryIO, destination: Path, expected_size: int) -> None:
    temporary = destination.with_name(f".{destination.name}.motiondrop-part")
    written = 0
    try:
        with temporary.open("wb") as output:
            while True:
                block = source.read(8 * 1024 * 1024)
                if not block:
                    break
                output.write(block)
                written += len(block)
        if written != expected_size:
            raise DownloadError(
                f"Short archive member {destination}: wrote {written}, expected {expected_size}"
            )
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


class _ConcatenatedReader(io.RawIOBase):
    """Expose split files as one sequential stream and optionally consume them."""

    def __init__(self, paths: list[Path], *, delete_consumed: bool):
        super().__init__()
        self.paths = paths
        self.delete_consumed = delete_consumed
        self.index = -1
        self.current: BinaryIO | None = None
        self.current_path: Path | None = None

    def readable(self) -> bool:
        return True

    def _next(self) -> bool:
        if self.current is not None:
            self.current.close()
            if self.delete_consumed and self.current_path is not None:
                self.current_path.unlink(missing_ok=True)
        self.index += 1
        if self.index >= len(self.paths):
            self.current = None
            self.current_path = None
            return False
        self.current_path = self.paths[self.index]
        self.current = self.current_path.open("rb")
        return True

    def readinto(self, buffer) -> int:
        view = memoryview(buffer)
        total = 0
        while total < len(view):
            if self.current is None and not self._next():
                break
            assert self.current is not None
            count = self.current.readinto(view[total:])
            if count:
                total += count
            else:
                self._next()
        return total

    def close(self) -> None:
        if self.current is not None:
            self.current.close()
            self.current = None
        super().close()


def extract_split_tar(
    part_paths: list[Path],
    destination: Path,
    *,
    delete_consumed: bool,
) -> dict:
    """Safely and resumably extract a tar archive split across ``part_paths``."""

    reader = _ConcatenatedReader(part_paths, delete_consumed=delete_consumed)
    extracted_files = skipped_files = extracted_bytes = 0
    try:
        with (
            io.BufferedReader(reader, buffer_size=8 * 1024 * 1024) as buffered,
            tarfile.open(fileobj=buffered, mode="r|") as archive,
        ):
            for member in archive:
                relative = _safe_relative_path(member.name)
                if not relative.parts:
                    continue
                if member.isdir():
                    _safe_destination(destination, relative).mkdir(parents=True, exist_ok=True)
                    continue
                if not member.isfile():
                    raise DownloadError(f"Unsupported non-regular tar member: {member.name!r}")
                target = _safe_destination(destination, relative)
                if target.is_file() and target.stat().st_size == member.size:
                    skipped_files += 1
                    continue
                source = archive.extractfile(member)
                if source is None:
                    raise DownloadError(f"Could not read tar member: {member.name}")
                with source:
                    _copy_atomic(source, target, member.size)
                extracted_files += 1
                extracted_bytes += member.size
    except tarfile.TarError as exc:
        raise DownloadError(f"Invalid split tar archive: {exc}") from exc
    if delete_consumed:
        for path in part_paths:
            path.unlink(missing_ok=True)
    return {
        "extracted_files": extracted_files,
        "skipped_files": skipped_files,
        "written_bytes": extracted_bytes,
    }


def _zip_info_is_symlink(info: ZipInfo) -> bool:
    mode = (info.external_attr >> 16) & 0o170000
    return mode == stat.S_IFLNK


def extract_zip(archive_path: Path, destination: Path) -> list[dict]:
    """Safely extract one StreamingBench zip and return a verification manifest."""

    records: list[dict] = []
    seen: set[Path] = set()
    try:
        with ZipFile(archive_path) as archive:
            for info in archive.infolist():
                relative = _safe_relative_path(info.filename)
                if not relative.parts or "__MACOSX" in relative.parts:
                    continue
                if relative.name == ".DS_Store":
                    continue
                if _zip_info_is_symlink(info):
                    raise DownloadError(f"Refusing zip symlink: {info.filename!r}")
                if info.is_dir():
                    _safe_destination(destination, relative).mkdir(parents=True, exist_ok=True)
                    continue
                if relative in seen:
                    raise DownloadError(f"Duplicate zip member: {info.filename!r}")
                seen.add(relative)
                target = _safe_destination(destination, relative)
                if not (target.is_file() and target.stat().st_size == info.file_size):
                    with archive.open(info) as source:
                        _copy_atomic(source, target, info.file_size)
                records.append({"path": relative.as_posix(), "bytes": info.file_size})
    except DownloadError:
        raise
    except (BadZipFile, RuntimeError) as exc:
        raise DownloadError(f"Invalid zip archive {archive_path}: {exc}") from exc
    return records


def validate_ovobench(root: Path) -> dict:
    """Validate every clip referenced by the pinned OVO-Bench annotations."""

    backward_tasks = ("EPM", "ASI", "HLD")
    realtime_tasks = ("STU", "OJR", "ATR", "ACR", "OCR", "FPD")
    forward_tasks = ("REC", "SSR", "CRR")

    annotation_path = root / "ovo_bench_new.json"
    if not annotation_path.is_file():
        raise DownloadError(f"Missing OVO-Bench annotations: {annotation_path}")
    try:
        annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DownloadError(f"Invalid OVO-Bench annotations: {annotation_path}") from exc
    if not isinstance(annotations, list):
        raise DownloadError("OVO-Bench annotations must contain a JSON list")

    videos = root / "chunked_videos"
    expected: list[Path] = []
    all_tasks = set(backward_tasks + realtime_tasks + forward_tasks)
    for annotation in annotations:
        task = annotation.get("task")
        if task not in all_tasks:
            raise DownloadError(f"Unknown OVO-Bench task in annotations: {task!r}")
        sample_id = int(annotation["id"])
        if task in backward_tasks + realtime_tasks:
            expected.append(videos / f"{sample_id}.mp4")
        else:
            expected.extend(
                videos / f"{sample_id}_{index}.mp4"
                for index in range(len(annotation.get("test_info", [])))
            )
    missing = [str(path) for path in expected if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise DownloadError(
            f"OVO-Bench is missing {len(missing)}/{len(expected)} clips; first paths: {missing[:5]}"
        )
    if len(annotations) != 1_640 or len(expected) != 3_035:
        raise DownloadError(
            f"Unexpected pinned OVO-Bench counts: {len(annotations)} annotations, "
            f"{len(expected)} inference clips"
        )
    return {
        "annotations": len(annotations),
        "inference_clips": len(expected),
        "tasks": sorted({annotation["task"] for annotation in annotations}),
    }


def ovobench_is_ready(root: Path) -> bool:
    try:
        validate_ovobench(root)
        return True
    except (DownloadError, ImportError, OSError, ValueError, KeyError, TypeError):
        return False


def validate_streamingbench(root: Path) -> dict:
    """Validate all four-choice StreamingBench rows against extracted videos."""

    required = (
        "StreamingBench/Contextual_Understanding.csv",
        "StreamingBench/Omni_Source_Understanding.csv",
        "StreamingBench/Real_Time_Visual_Understanding.csv",
        "StreamingBench/Sequential_Question_Answering.csv",
    )
    missing_files = [str(root / path) for path in required if not (root / path).is_file()]
    if missing_files:
        raise DownloadError(f"Missing StreamingBench annotations: {missing_files}")
    from motiondrop.benchmarks.streamingbench_data import DATASET_SPECS, load_choice_manifest

    try:
        annotations, report = load_choice_manifest(root, sorted(DATASET_SPECS))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise DownloadError(f"Invalid StreamingBench layout under {root}: {exc}") from exc
    missing = sum(item["missing_video_rows"] for item in report["datasets"].values())
    if missing:
        raise DownloadError(f"StreamingBench has {missing} annotation rows with missing videos")
    if report["total_rows"] != 4_250 or report["unique_videos"] != 850:
        raise DownloadError(
            "Unexpected pinned StreamingBench counts: "
            f"{report['total_rows']} questions, {report['unique_videos']} videos"
        )
    empty_videos = sorted(
        {item["video_path"] for item in annotations if Path(item["video_path"]).stat().st_size == 0}
    )
    if empty_videos:
        raise DownloadError(
            f"StreamingBench contains {len(empty_videos)} empty videos; "
            f"first paths: {empty_videos[:5]}"
        )
    return {
        "questions": report["total_rows"],
        "videos": report["unique_videos"],
        "datasets": sorted(DATASET_SPECS),
    }


def _archive_marker_path(root: Path, spec: ArchiveSpec) -> Path:
    return root / ".motiondrop" / "streamingbench" / f"{spec.filename}.json"


def _archive_marker_is_ready(root: Path, spec: ArchiveSpec) -> bool:
    marker_path = _archive_marker_path(root, spec)
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if marker.get("revision") != STREAMINGBENCH_REVISION:
        return False
    if marker.get("filename") != spec.filename or marker.get("target") != spec.target:
        return False
    target = root / spec.target
    records = marker.get("files")
    if not isinstance(records, list) or not records:
        return False
    try:
        normalized = [(_safe_relative_path(record["path"]), record["bytes"]) for record in records]
        return all(
            (target / relative).is_file() and (target / relative).stat().st_size == expected_bytes
            for relative, expected_bytes in normalized
        )
    except (KeyError, TypeError, OSError):
        return False


def streamingbench_is_ready(root: Path, *, include_proactive: bool) -> bool:
    try:
        validate_streamingbench(root)
    except (DownloadError, ImportError, OSError, ValueError, KeyError, TypeError):
        return False
    if include_proactive:
        return all(
            _archive_marker_is_ready(root, spec) for spec in STREAMING_ARCHIVES if spec.proactive
        )
    return True


def _download_url(url: str, destination: Path) -> None:
    import httpx

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.download")
    for attempt in range(1, 4):
        try:
            print(f"[download] {url}", flush=True)
            with httpx.stream("GET", url, follow_redirects=True, timeout=120.0) as response:
                response.raise_for_status()
                with temporary.open("wb") as handle:
                    for block in response.iter_bytes(chunk_size=1024 * 1024):
                        handle.write(block)
            temporary.replace(destination)
            return
        except (httpx.HTTPError, OSError) as exc:
            temporary.unlink(missing_ok=True)
            if attempt == 3:
                raise DownloadError(f"Failed to download {url}: {exc}") from exc
            print(f"[download] Retry {attempt}/2 after error: {exc}", flush=True)


def download_models(aliases: list[str], models_dir: Path, *, max_workers: int) -> None:
    from huggingface_hub import snapshot_download

    for alias in aliases:
        target = models_dir / alias
        spec = MODEL_DOWNLOAD_SPECS[alias]
        repo_id = spec.repo_id
        if downloaded_model_path(models_dir, alias) is not None:
            print(f"[model:{alias}] Ready: {target}", flush=True)
            continue
        print(
            f"[model:{alias}] Downloading {repo_id}@{spec.revision} ({format_bytes(spec.bytes)})",
            flush=True,
        )
        _retry_hub(
            repo_id,
            lambda repo_id=repo_id, revision=spec.revision, target=target: snapshot_download(
                repo_id=repo_id,
                revision=revision,
                local_dir=target,
                max_workers=max_workers,
            ),
        )
        summary = validate_model_directory(target, alias)
        _write_json(
            target / MODEL_STATE_FILE,
            {
                "schema": DOWNLOAD_STATE_VERSION,
                "repo_id": repo_id,
                "revision": spec.revision,
                "completed_at": _utc_now(),
                "validation": summary,
            },
        )
        print(f"[model:{alias}] Ready: {target}", flush=True)


def _download_ovobench(root: Path, *, keep_archives: bool, max_workers: int) -> None:
    from huggingface_hub import snapshot_download

    archive_dir = root.parent / ".downloads" / "ovobench"
    if ovobench_is_ready(root):
        if not keep_archives:
            for filename in OVOBENCH_PART_SIZES:
                (archive_dir / filename).unlink(missing_ok=True)
        print(f"[dataset:ovobench] Ready: {root}", flush=True)
        return
    print(
        "[dataset:ovobench] OVO-Bench data is governed by the upstream dataset license.",
        flush=True,
    )
    _download_url(
        OVOBENCH_ANNOTATION_URL,
        root / "ovo_bench_new.json",
    )
    print(
        f"[dataset:ovobench] Downloading 15 pre-chunked tar parts "
        f"({format_bytes(OVOBENCH_ARCHIVE_BYTES)})",
        flush=True,
    )
    _retry_hub(
        OVOBENCH_REPO,
        lambda: snapshot_download(
            repo_id=OVOBENCH_REPO,
            repo_type="dataset",
            revision=OVOBENCH_REVISION,
            allow_patterns=list(OVOBENCH_PART_SIZES),
            local_dir=archive_dir,
            max_workers=max_workers,
        ),
    )
    part_paths = [archive_dir / filename for filename in OVOBENCH_PART_SIZES]
    for path, expected_size in zip(part_paths, OVOBENCH_PART_SIZES.values(), strict=True):
        if not path.is_file() or path.stat().st_size != expected_size:
            actual = path.stat().st_size if path.exists() else None
            raise DownloadError(
                f"OVO-Bench part size mismatch for {path}: {actual}; expected {expected_size}"
            )
    print(f"[dataset:ovobench] Extracting into {root}", flush=True)
    extraction = extract_split_tar(
        part_paths,
        root,
        delete_consumed=not keep_archives,
    )
    summary = validate_ovobench(root)
    _write_json(
        root / ".motiondrop" / "ovobench.json",
        {
            "schema": DOWNLOAD_STATE_VERSION,
            "dataset_repo": OVOBENCH_REPO,
            "dataset_revision": OVOBENCH_REVISION,
            "annotation_revision": OVOBENCH_CODE_REVISION,
            "completed_at": _utc_now(),
            "extraction": extraction,
            "validation": summary,
        },
    )
    print(f"[dataset:ovobench] Ready: {root}", flush=True)


def _download_streaming_annotations(root: Path) -> None:
    from huggingface_hub import hf_hub_download

    for filename in STREAMING_ANNOTATIONS:
        _retry_hub(
            filename,
            lambda filename=filename: hf_hub_download(
                repo_id=STREAMINGBENCH_REPO,
                repo_type="dataset",
                revision=STREAMINGBENCH_REVISION,
                filename=filename,
                local_dir=root,
            ),
        )


def _download_streamingbench(
    root: Path,
    *,
    include_proactive: bool,
    keep_archives: bool,
) -> None:
    from huggingface_hub import hf_hub_download

    supported_ready = streamingbench_is_ready(root, include_proactive=False)
    proactive_ready = all(
        _archive_marker_is_ready(root, spec) for spec in STREAMING_ARCHIVES if spec.proactive
    )
    if supported_ready and (not include_proactive or proactive_ready):
        print(f"[dataset:streamingbench] Ready: {root}", flush=True)
        return
    print(
        "[dataset:streamingbench] StreamingBench data is governed by the upstream dataset terms.",
        flush=True,
    )
    _download_streaming_annotations(root)
    archive_dir = root.parent / ".downloads" / "streamingbench"
    selected = [spec for spec in STREAMING_ARCHIVES if include_proactive or not spec.proactive]
    if supported_ready:
        selected = [spec for spec in selected if spec.proactive]
    for index, spec in enumerate(selected, start=1):
        if _archive_marker_is_ready(root, spec):
            if not keep_archives:
                (archive_dir / spec.filename).unlink(missing_ok=True)
            print(
                f"[dataset:streamingbench] [{index}/{len(selected)}] Ready: {spec.filename}",
                flush=True,
            )
            continue
        print(
            f"[dataset:streamingbench] [{index}/{len(selected)}] Downloading "
            f"{spec.filename} ({format_bytes(spec.compressed_bytes)})",
            flush=True,
        )
        archive_path = Path(
            _retry_hub(
                spec.filename,
                lambda filename=spec.filename: hf_hub_download(
                    repo_id=STREAMINGBENCH_REPO,
                    repo_type="dataset",
                    revision=STREAMINGBENCH_REVISION,
                    filename=filename,
                    local_dir=archive_dir,
                ),
            )
        )
        if archive_path.stat().st_size != spec.compressed_bytes:
            raise DownloadError(
                f"StreamingBench archive size mismatch for {archive_path}: "
                f"{archive_path.stat().st_size}; expected {spec.compressed_bytes}"
            )
        print(
            f"[dataset:streamingbench] [{index}/{len(selected)}] Extracting to "
            f"{root / spec.target}",
            flush=True,
        )
        records = extract_zip(archive_path, root / spec.target)
        _write_json(
            _archive_marker_path(root, spec),
            {
                "schema": DOWNLOAD_STATE_VERSION,
                "dataset_repo": STREAMINGBENCH_REPO,
                "revision": STREAMINGBENCH_REVISION,
                "filename": spec.filename,
                "target": spec.target,
                "completed_at": _utc_now(),
                "files": records,
            },
        )
        if not keep_archives:
            archive_path.unlink(missing_ok=True)

    summary = validate_streamingbench(root)
    _write_json(
        root / ".motiondrop" / "streamingbench.json",
        {
            "schema": DOWNLOAD_STATE_VERSION,
            "dataset_repo": STREAMINGBENCH_REPO,
            "revision": STREAMINGBENCH_REVISION,
            "include_proactive": include_proactive,
            "completed_at": _utc_now(),
            "validation": summary,
        },
    )
    print(f"[dataset:streamingbench] Ready: {root}", flush=True)


def _nearest_existing(path: Path) -> Path:
    path = path.expanduser().resolve()
    while not path.exists():
        if path.parent == path:
            break
        path = path.parent
    return path


def _tree_bytes(path: Path) -> int:
    """Return logical file bytes tracked under one asset directory."""

    if path.is_file():
        return path.stat().st_size
    if not path.is_dir():
        return 0
    total = 0
    for directory, _, filenames in os.walk(path):
        directory_path = Path(directory)
        for filename in filenames:
            file_path = directory_path / filename
            try:
                total += file_path.stat(follow_symlinks=False).st_size
            except FileNotFoundError:
                # A concurrent/resumed downloader may atomically replace files.
                continue
    return total


def _required_space(
    models: list[str],
    datasets: list[str],
    models_dir: Path,
    data_dir: Path,
    *,
    include_proactive: bool,
    keep_archives: bool,
) -> list[tuple[Path, int]]:
    requirements: list[tuple[Path, int]] = []
    model_bytes = 0
    for alias in models:
        target = models_dir / alias
        if downloaded_model_path(models_dir, alias) is not None:
            continue
        expected = MODEL_DOWNLOAD_SPECS[alias].bytes
        model_bytes += max(expected - min(_tree_bytes(target), expected), 0)
    if model_bytes:
        requirements.append((models_dir, model_bytes))

    data_bytes = 0
    if "ovobench" in datasets and not ovobench_is_ready(data_dir / "ovobench"):
        expected = OVOBENCH_ARCHIVE_BYTES * (2 if keep_archives else 1)
        current = _tree_bytes(data_dir / "ovobench") + _tree_bytes(
            data_dir / ".downloads" / "ovobench"
        )
        data_bytes += max(expected - min(current, expected), 0)
    if "streamingbench" in datasets and not streamingbench_is_ready(
        data_dir / "streamingbench", include_proactive=include_proactive
    ):
        archives = [spec for spec in STREAMING_ARCHIVES if include_proactive or not spec.proactive]
        extracted = sum(spec.extracted_bytes for spec in archives) + STREAMING_ANNOTATION_BYTES
        compressed = sum(spec.compressed_bytes for spec in archives)
        expected = extracted + (
            compressed if keep_archives else max(spec.compressed_bytes for spec in archives)
        )
        current = _tree_bytes(data_dir / "streamingbench") + _tree_bytes(
            data_dir / ".downloads" / "streamingbench"
        )
        data_bytes += max(expected - min(current, expected), 0)
    if data_bytes:
        requirements.append((data_dir, data_bytes))
    return requirements


def check_disk_space(requirements: list[tuple[Path, int]], *, enforce: bool) -> None:
    """Check aggregate requirements independently for each destination filesystem."""

    grouped: dict[int, dict] = {}
    for destination, required in requirements:
        anchor = _nearest_existing(destination)
        device = os.stat(anchor).st_dev
        entry = grouped.setdefault(device, {"anchor": anchor, "required": 0})
        entry["required"] += required
    for entry in grouped.values():
        raw_required = entry["required"]
        margin = max(5 * GIB, int(raw_required * 0.03))
        required = raw_required + margin
        free = shutil.disk_usage(entry["anchor"]).free
        print(
            f"[disk] {entry['anchor']}: need about {format_bytes(required)} "
            f"including safety margin; {format_bytes(free)} available",
            flush=True,
        )
        if free < required:
            message = (
                f"Insufficient free space on the filesystem containing {entry['anchor']}: "
                f"need about {format_bytes(required)}, have {format_bytes(free)}. "
                "Choose other --models-dir/--data-dir locations, select fewer resources, "
                "or use --skip-disk-check after verifying capacity yourself."
            )
            if enforce:
                raise DownloadError(message)
            print(f"[disk] WARNING: {message}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="motiondrop download",
        description="Download pinned model checkpoints and ready-to-run benchmark datasets.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        metavar="MODEL",
        help="Model aliases, 'all', or 'none' (default: all).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        metavar="DATASET",
        help="ovobench, streamingbench, 'all', or 'none' (default: all).",
    )
    parser.add_argument("--models-dir", type=Path, default=models_dir())
    parser.add_argument("--data-dir", type=Path, default=data_dir())
    parser.add_argument(
        "--include-proactive",
        action="store_true",
        help="Also download StreamingBench Proactive Output (not evaluated by this runner).",
    )
    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help="Keep source archives after successful extraction (requires much more space).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Parallel Hugging Face snapshot workers (default: 4).",
    )
    parser.add_argument(
        "--skip-disk-check",
        action="store_true",
        help="Skip the conservative free-space preflight.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show selections and storage estimates without downloading.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.models = normalize_selection(args.models, MODEL_DOWNLOAD_SPECS, "models")
        args.datasets = normalize_selection(
            args.datasets, ("ovobench", "streamingbench"), "datasets"
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.max_workers < 1:
        parser.error("--max-workers must be positive")
    if not args.models and not args.datasets:
        parser.error("Select at least one model or dataset")
    return args


def run(args: argparse.Namespace) -> None:
    models_dir = args.models_dir.expanduser().resolve()
    data_dir = args.data_dir.expanduser().resolve()
    print(f"[plan] Models: {', '.join(args.models) or 'none'}", flush=True)
    print(f"[plan] Datasets: {', '.join(args.datasets) or 'none'}", flush=True)
    print(f"[plan] Model directory: {models_dir}", flush=True)
    print(f"[plan] Data directory: {data_dir}", flush=True)
    if "streamingbench" in args.datasets and not args.include_proactive:
        print(
            "[plan] StreamingBench: 8 supported four-choice tasks (Proactive Output omitted)",
            flush=True,
        )
    requirements = _required_space(
        args.models,
        args.datasets,
        models_dir,
        data_dir,
        include_proactive=args.include_proactive,
        keep_archives=args.keep_archives,
    )
    if not args.skip_disk_check:
        check_disk_space(requirements, enforce=not args.dry_run)
    if args.dry_run:
        print("[plan] Dry run complete; no files were downloaded.", flush=True)
        return

    if args.models:
        download_models(args.models, models_dir, max_workers=args.max_workers)
    if "ovobench" in args.datasets:
        _download_ovobench(
            data_dir / "ovobench",
            keep_archives=args.keep_archives,
            max_workers=args.max_workers,
        )
    if "streamingbench" in args.datasets:
        _download_streamingbench(
            data_dir / "streamingbench",
            include_proactive=args.include_proactive,
            keep_archives=args.keep_archives,
        )
    print("[download] All selected resources are ready.", flush=True)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        run(args)
    except (DownloadError, OSError) as exc:
        raise SystemExit(f"[download] ERROR: {exc}") from exc


__all__ = [
    "MODEL_DOWNLOAD_SPECS",
    "STREAMING_ARCHIVES",
    "DownloadError",
    "downloaded_model_path",
    "local_model_path",
    "extract_split_tar",
    "extract_zip",
    "format_bytes",
    "main",
    "normalize_selection",
    "validate_model_directory",
    "validate_ovobench",
    "validate_streamingbench",
]


if __name__ == "__main__":
    main()
