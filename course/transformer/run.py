#!/usr/bin/env python3
"""
run.py — Run transformer course modules.

Usage:
    uv run course/transformer/run.py              # Validate all modules (pass/fail only)
    uv run course/transformer/run.py 0            # Run module 0 — streams full output
    uv run course/transformer/run.py 0-3          # Validate modules 0 through 3
    uv run course/transformer/run.py list         # List all modules

Or run a module DIRECTLY to read it at your own pace:
    uv run python course/transformer/00_prerequisites.py

When you run a single module, ALL output is streamed live — you'll see
every explanation, every example, every calculation. When you run multiple
modules, output is suppressed and only pass/fail is shown (use direct run
or single-module mode to actually read the content).
"""

import os
import subprocess
import sys
import time
from pathlib import Path

COURSE_DIR = Path(__file__).parent
PROJECT_ROOT = str(COURSE_DIR.parent.parent.resolve())
MODULES = [
    ("00_prerequisites",      "Prerequisites: Tensors, Matrix Multiplication, Softmax"),
    ("01_tokenization",       "Tokenization: Text to Numbers (character, word, subword/BPE)"),
    ("02_embeddings",         "Embeddings: Token IDs to Meaningful Vectors"),
    ("03_simple_attention",   "Simple Attention: The Core Intuition (fuzzy lookup)"),
    ("04_self_attention",     "Self-Attention: Learned Q/K/V Projections"),
    ("05_multi_head_attention","Multi-Head Attention + Positional Encoding"),
    ("06_transformer_block",  "Full Transformer Block: FFN + LayerNorm + Residuals"),
    ("07_mini_gpt",           "Mini GPT: Complete Decoder-Only Transformer"),
    ("08_training",           "Training: Loss, Backpropagation, and Learning"),
    ("09_advanced_topics",    "Advanced Topics: RoPE, GQA, Flash Attention, SwiGLU, etc."),
    ("10_alignment",          "Alignment: SFT, PPO, DPO, GRPO, and RL for Reasoning"),
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

    Used when validating multiple modules at once (running all
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
        print("  uv run course/transformer/run.py 0")
        print()
        print("To run a module directly (same thing):")
        print("  uv run python course/transformer/00_prerequisites.py")
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
            print("Usage: uv run course/transformer/run.py [module|range|list]")
            sys.exit(1)

    for m in modules_to_run:
        if m < 0 or m >= len(MODULES):
            print(f"Invalid module: {m} (valid: 0-{len(MODULES)-1})")
            sys.exit(1)

    # Single module → stream everything so the user can read it
    if len(modules_to_run) == 1:
        name, desc = MODULES[modules_to_run[0]]
        print("=" * 70)
        print(f"  TRANSFORMER COURSE — Module {modules_to_run[0]}: {name}")
        print(f"  {desc}")
        print("=" * 70)
        print()
        success = run_module_streaming(name)
        sys.exit(0 if success else 1)

    # Multiple modules → quiet pass/fail validation
    print("=" * 50)
    print(f"  Validating {len(modules_to_run)} transformer modules...")
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
