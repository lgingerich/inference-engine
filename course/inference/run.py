#!/usr/bin/env python3
"""
run.py — Run inference course modules.

Usage:
    uv run course/inference/run.py              # Validate all modules (pass/fail only)
    uv run course/inference/run.py 0            # Run module 0 — streams full output
    uv run course/inference/run.py 0-3          # Validate modules 0 through 3
    uv run course/inference/run.py list         # List all modules

Or run a module DIRECTLY to read it at your own pace:
    uv run python course/inference/00_why_inference_is_hard.py

When you run a single module, ALL output is streamed live — you'll see
every explanation, every example, every code output. When you run multiple
modules, output is suppressed and only pass/fail is shown (use direct run
or single-module mode to actually read the content).

Each module builds on the previous one and ends with a summary
of what you learned. The inference course assumes you've completed
the transformers course (modules 0-7 at minimum).
"""

import os
import subprocess
import sys
import time
from pathlib import Path

COURSE_DIR = Path(__file__).parent
PROJECT_ROOT = str(COURSE_DIR.parent.parent.resolve())
MODULES = [
    ("00_why_inference_is_hard",   "Why Inference Is Hard: Prefill vs Decode, GPU Memory Hierarchy"),
    ("01_autoregressive_loop",     "The Raw Autoregressive Loop & Why It's O(n²)"),
    ("02_kv_cache",                "KV Cache: The First and Most Important Optimization"),
    ("03_batching",                "Batching & Scheduling: Serving Multiple Requests"),
    ("04_quantization",            "Quantization: Making Models Fit in Memory"),
    ("05_attention_optimizations", "Attention Optimizations: FlashAttention & PagedAttention"),
    ("06_speculative_decoding",    "Speculative Decoding: Draft + Verify for Speed"),
    ("07_distributed",             "Distributed Inference: Tensor & Pipeline Parallelism"),
    ("08_serving",                 "The Serving Stack: HTTP API, Streaming, Metrics"),
    ("09_production",              "Production Engines: vLLM, TGI, SGLang & The State of the Art"),
]

BASE_ENV = {**os.environ, "PYTHONPATH": PROJECT_ROOT}


def run_module_streaming(module_name):
    """Run a single module, streaming ALL output to the terminal.

    This is what you want when you're READING a module. Every print()
    statement, every explanation, every code output appears inline.
    """
    module_path = COURSE_DIR / f"{module_name}.py"
    if not module_path.exists():
        print(f"  Module not found: {module_path}")
        return False

    start = time.time()
    result = subprocess.run(
        [sys.executable, str(module_path)],
        capture_output=False,  # STREAM — user sees everything
        timeout=60,
        env=BASE_ENV,
    )
    elapsed = time.time() - start

    if result.returncode == 0:
        print(f"\n  Completed in {elapsed:.1f}s")
        return True
    else:
        print(f"\n  FAILED (exit code {result.returncode})")
        return False


def run_module_quiet(module_name):
    """Run a module silently — only report pass/fail.

    Used when validating multiple modules at once (running all 10
    with full output would be overwhelming).
    """
    module_path = COURSE_DIR / f"{module_name}.py"
    if not module_path.exists():
        print(f"  Module not found: {module_path}")
        return False

    start = time.time()
    result = subprocess.run(
        [sys.executable, str(module_path)],
        capture_output=True,
        text=True,
        timeout=60,
        env=BASE_ENV,
    )
    elapsed = time.time() - start

    if result.returncode == 0:
        print(f"  OK  {module_name}")
        return True
    else:
        print(f"  FAIL  {module_name}  (exit {result.returncode})")
        stderr = result.stderr.strip()
        if stderr:
            for line in stderr.split("\n")[-3:]:
                print(f"       {line}")
        else:
            for line in result.stdout.strip().split("\n")[-3:]:
                print(f"       {line}")
        return False


def main():
    args = sys.argv[1:]

    if not args:
        modules_to_run = list(range(len(MODULES)))
    elif args[0] == "list":
        print("Available modules:")
        for i, (name, desc) in enumerate(MODULES):
            print(f"  {i}: {name}")
            print(f"     {desc}")
        print()
        print("To READ a module (streams full content):")
        print("  uv run course/inference/run.py 0")
        print()
        print("To run a module directly (same thing):")
        print("  uv run python course/inference/00_why_inference_is_hard.py")
        return
    elif "-" in args[0]:
        try:
            start, end = map(int, args[0].split("-"))
            modules_to_run = list(range(start, end + 1))
        except ValueError:
            print(f"Invalid range: {args[0]}")
            sys.exit(1)
    else:
        try:
            modules_to_run = [int(args[0])]
        except ValueError:
            print(f"Invalid module: {args[0]}")
            print("Usage: uv run course/inference/run.py [module|range|list]")
            sys.exit(1)

    for m in modules_to_run:
        if m < 0 or m >= len(MODULES):
            print(f"Invalid module: {m} (valid: 0-{len(MODULES)-1})")
            sys.exit(1)

    # Single module → stream everything so the user can read it
    if len(modules_to_run) == 1:
        name, desc = MODULES[modules_to_run[0]]
        print("=" * 70)
        print(f"  INFERENCE COURSE — Module {modules_to_run[0]}: {name}")
        print(f"  {desc}")
        print("=" * 70)
        print()
        success = run_module_streaming(name)
        sys.exit(0 if success else 1)

    # Multiple modules → quiet pass/fail validation
    print("=" * 50)
    print(f"  Validating {len(modules_to_run)} inference modules...")
    print(f"  (Run a single module for full content: run.py 0)")
    print("=" * 50)
    print()

    passed = 0
    failed = 0
    for m in modules_to_run:
        name, _ = MODULES[m]
        if run_module_quiet(name):
            passed += 1
        else:
            failed += 1

    print()
    print("=" * 50)
    print(f"  {passed} passed, {failed} failed out of {len(modules_to_run)}")
    print("=" * 50)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
