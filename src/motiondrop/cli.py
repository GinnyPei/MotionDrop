"""MotionDrop command-line interface."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys

import torch

from motiondrop import __version__
from motiondrop.models import format_model_table


def _doctor(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="motiondrop doctor")
    parser.add_argument(
        "--strict-cuda",
        action="store_true",
        help="Return a non-zero exit code when CUDA is unavailable.",
    )
    args = parser.parse_args(argv)
    packages = {}
    for name in (
        "motiondrop",
        "torch",
        "torchvision",
        "transformers",
        "accelerate",
        "decord2",
        "av",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    cuda_available = torch.cuda.is_available()
    report = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "cuda_available": cuda_available,
        "cuda_device_count": torch.cuda.device_count() if cuda_available else 0,
        "cuda_devices": (
            [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
            if cuda_available
            else []
        ),
    }
    print(json.dumps(report, indent=2))
    if args.strict_cuda and not cuda_available:
        raise SystemExit(1)


def _help() -> str:
    return """usage: motiondrop COMMAND [ARGS]

Commands:
  models          List supported model aliases and default checkpoints
  doctor          Check the Python, package, and CUDA environment
  download        Download models and ready-to-run benchmark datasets
  ovobench        Run OVO-Bench evaluation
  streamingbench  Run StreamingBench evaluation
  stream          Run the pipelined streaming-video runtime
  demo            Serve the interactive streaming-video web demo
  live-eval       Evaluate long-running + instant tasks on the live runtime
  fit-latency     Profile and fit the SLO latency model
  version         Print the installed MotionDrop version

Run `motiondrop COMMAND --help` for command-specific arguments.
"""


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help", "help"}:
        print(_help())
        return
    command, command_argv = argv[0], argv[1:]
    if command == "models":
        if command_argv:
            raise SystemExit("motiondrop models takes no arguments")
        print(format_model_table())
    elif command == "doctor":
        _doctor(command_argv)
    elif command == "download":
        from motiondrop.downloads import main as download_main

        download_main(command_argv)
    elif command == "ovobench":
        from motiondrop.benchmarks.ovobench import main as benchmark_main

        benchmark_main(command_argv)
    elif command == "streamingbench":
        from motiondrop.benchmarks.streamingbench import main as benchmark_main

        benchmark_main(command_argv)
    elif command == "stream":
        from motiondrop.streaming_runtime.run import main as stream_main

        stream_main(command_argv)
    elif command == "demo":
        from motiondrop.streaming_runtime.demo import main as demo_main

        demo_main(command_argv)
    elif command == "live-eval":
        from motiondrop.streaming_runtime.live_eval import main as live_eval_main

        live_eval_main(command_argv)
    elif command == "fit-latency":
        from motiondrop.streaming_runtime.profile import main as profile_main

        profile_main(command_argv)
    elif command in {"version", "--version", "-V"}:
        if command_argv:
            raise SystemExit("motiondrop version takes no arguments")
        print(__version__)
    else:
        print(f"Unknown command: {command}\n", file=sys.stderr)
        print(_help(), file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
