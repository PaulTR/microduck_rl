#!/usr/bin/env python3
"""Training and testing workflow script for the Microduck forward jump task.

Usage:
    # 1. Fast smoke test (catches 95% of config bugs in seconds):
    uv run python scripts/train_jump.py --smoke-test

    # 2. Full training on Linux GPU (4096 envs, 4000 iters):
    uv run python scripts/train_jump.py --train

    # 3. Training with custom parameters:
    uv run python scripts/train_jump.py --train --num-envs 2048 --max-iterations 3000

    # 4. Submit to Hugging Face Jobs:
    uv run python scripts/train_jump.py --train --hf-jobs

    # 5. Export trained checkpoint to ONNX (with baked normalizer):
    uv run python scripts/train_jump.py --export --checkpoint logs/microduck_jump/<run_id>/model_4000.pt

    # 6. Rehearse exported policy in CPU MuJoCo viewer (BAM actuators):
    uv run python scripts/train_jump.py --rehearse --onnx policy_jump.onnx
"""

import argparse
import subprocess
import sys

TASK_ID = "Mjlab-Jump-Flat-MicroDuck"


def run_cmd(cmd: list[str]) -> int:
    print(f"\n[train_jump] Running: {' '.join(cmd)}\n")
    return subprocess.run(cmd).returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Microduck two-footed forward jump training & execution runner."
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a 5-iteration smoke test at 64 envs (always run first!).",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Launch full training run.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=4096,
        help="Number of parallel environments (default: 4096 for GPU training).",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=4000,
        help="Maximum PPO iterations (default: 4000).",
    )
    parser.add_argument(
        "--hf-jobs",
        action="store_true",
        help="Submit training job to Hugging Face Jobs.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from previous checkpoint.",
    )
    parser.add_argument(
        "--load-checkpoint",
        type=str,
        default=None,
        help="Checkpoint path or model_XXXX.pt to load when resuming.",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export checkpoint to ONNX with baked normalizer.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path to export (required with --export).",
    )
    parser.add_argument(
        "--output-onnx",
        type=str,
        default="microduck_jump.onnx",
        help="Output ONNX filename (default: microduck_jump.onnx).",
    )
    parser.add_argument(
        "--rehearse",
        action="store_true",
        help="Rehearse ONNX policy in CPU viewer with BAM actuators.",
    )
    parser.add_argument(
        "--onnx",
        type=str,
        default="microduck_jump.onnx",
        help="ONNX path for rehearsal (default: microduck_jump.onnx).",
    )

    args = parser.parse_args()

    if not any([args.smoke_test, args.train, args.export, args.rehearse]):
        parser.print_help()
        print("\nNotice: Please specify --smoke-test, --train, --export, or --rehearse.")
        return 0

    if args.smoke_test:
        print("=== Running Smoke Test (64 envs, 5 iterations) ===")
        cmd = [
            "uv", "run", "train", TASK_ID,
            "--env.scene.num-envs", "64",
            "--agent.max_iterations", "5",
        ]
        rc = run_cmd(cmd)
        if rc != 0:
            print("[train_jump] Smoke test failed!", file=sys.stderr)
            return rc
        print("[train_jump] Smoke test succeeded!")

    if args.train:
        print(f"=== Launching Training ({args.num_envs} envs, {args.max_iterations} iters) ===")
        cmd = [
            "uv", "run", "train", TASK_ID,
            "--env.scene.num-envs", str(args.num_envs),
            "--agent.max_iterations", str(args.max_iterations),
        ]
        if args.hf_jobs:
            cmd.append("--hf-jobs")
        if args.resume:
            cmd.extend(["--agent.resume", "True"])
        if args.load_checkpoint:
            cmd.extend(["--agent.load_checkpoint", args.load_checkpoint])
        rc = run_cmd(cmd)
        if rc != 0:
            print("[train_jump] Training process exited with error code.", file=sys.stderr)
            return rc

    if args.export:
        if not args.checkpoint:
            print("Error: --export requires --checkpoint <path/to/model_XXXX.pt>", file=sys.stderr)
            return 1
        print(f"=== Exporting Checkpoint {args.checkpoint} to {args.output_onnx} ===")
        cmd = [
            "uv", "run", "python", "scripts/export.py", TASK_ID,
            "--checkpoint", args.checkpoint,
            "--output", args.output_onnx,
        ]
        rc = run_cmd(cmd)
        if rc != 0:
            print("[train_jump] Export failed.", file=sys.stderr)
            return rc

    if args.rehearse:
        print(f"=== Rehearsing Jump Policy in MuJoCo Viewer ({args.onnx}) ===")
        print("Press 'J' in the viewer to trigger the forward jump!")
        cmd = [
            "uv", "run", "python", "scripts/infer_policy.py",
            "--jump", args.onnx,
            "--new-cmd-obs",
        ]
        rc = run_cmd(cmd)
        if rc != 0:
            return rc

    return 0


if __name__ == "__main__":
    sys.exit(main())
