#!/usr/bin/env python3
"""Record LLMPerf summaries and render simple performance-history charts."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BENCHMARK_DIR = Path(__file__).resolve().parent
DEFAULT_HISTORY = BENCHMARK_DIR / "history.csv"
DEFAULT_CHART = BENCHMARK_DIR / "charts" / "performance.png"


@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    unit: str
    higher_is_better: bool


CHART_METRICS = [
    Metric("output_tok_s", "TPS (output tokens per second)", "tok/s", True),
    Metric("ttft_p50_s", "TTFT (time to first token)", "s", False),
    Metric("tpot_p50_s", "TPOT (time per output token)", "s", False),
    Metric("e2e_p50_s", "E2E (end-to-end latency)", "s", False),
]

CSV_FIELDS = [
    "trial_id",
    "platform",
    "change",
    "run_label",
    "run_id",
    "date",
    "git_commit",
    "benchmark",
    "precision",
    "model",
    "mean_input_tokens",
    "mean_output_tokens",
    "num_concurrent_requests",
    "completed_requests",
    "errors",
    "error_rate",
    "output_tok_s",
    "request_tok_s_p50",
    "request_tok_s_p95",
    "ttft_p50_s",
    "ttft_p95_s",
    "ttft_p99_s",
    "tpot_p50_s",
    "tpot_p95_s",
    "tpot_p99_s",
    "e2e_p50_s",
    "e2e_p95_s",
    "e2e_p99_s",
    "output_tokens_mean",
    "output_tokens_p50",
    "output_tokens_p95",
]


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return data


def run_id_from_summary_path(path: Path) -> str:
    parent = path.parent.name
    return parent if parent else path.stem


def date_from_run_id(run_id: str) -> str:
    raw = run_id.split("_", 1)[0]
    try:
        return datetime.strptime(raw, "%Y%m%dT%H%M%SZ").date().isoformat()
    except ValueError:
        return datetime.now(timezone.utc).date().isoformat()


def extract_row(
    summary_path: Path,
    trial_id: str | None,
    platform: str | None,
    change: str | None,
    run_label: str | None,
    precision: str | None,
) -> dict[str, Any]:
    summary = load_json(summary_path)

    def get(name: str) -> Any:
        if name not in summary:
            raise KeyError(f"{summary_path} is missing {name}")
        return summary[name]

    benchmark = str(summary.get("benchmark", ""))
    trial = trial_id or str(summary.get("trial_id", ""))
    platform_key = platform or str(summary.get("platform", "")) or str(summary.get("platform_slug", ""))
    change_key = change if change is not None else str(summary.get("change", "")) or str(summary.get("change_slug", ""))
    label = run_label or str(summary.get("run_label", ""))
    if not label and trial and platform_key:
        label = f"{trial}-{platform_key}"
        if change_key:
            label = f"{label}-{change_key}"
    if not label:
        label = benchmark or run_id_from_summary_path(summary_path)

    return {
        "trial_id": trial,
        "platform": platform_key,
        "change": change_key,
        "run_label": label,
        "run_id": run_id_from_summary_path(summary_path),
        "date": date_from_run_id(run_id_from_summary_path(summary_path)),
        "git_commit": str(summary.get("git_commit", "")),
        "benchmark": benchmark,
        "precision": precision or str(summary.get("precision", "")),
        "model": str(get("model")),
        "mean_input_tokens": int(get("mean_input_tokens")),
        "mean_output_tokens": int(get("mean_output_tokens")),
        "num_concurrent_requests": int(get("num_concurrent_requests")),
        "completed_requests": int(get("results_num_completed_requests")),
        "errors": int(get("results_number_errors")),
        "error_rate": float(get("results_error_rate")),
        "output_tok_s": float(get("results_mean_output_throughput_token_per_s")),
        "request_tok_s_p50": float(get("results_request_output_throughput_token_per_s_quantiles_p50")),
        "request_tok_s_p95": float(get("results_request_output_throughput_token_per_s_quantiles_p95")),
        "ttft_p50_s": float(get("results_ttft_s_quantiles_p50")),
        "ttft_p95_s": float(get("results_ttft_s_quantiles_p95")),
        "ttft_p99_s": float(get("results_ttft_s_quantiles_p99")),
        "tpot_p50_s": float(get("results_inter_token_latency_s_quantiles_p50")),
        "tpot_p95_s": float(get("results_inter_token_latency_s_quantiles_p95")),
        "tpot_p99_s": float(get("results_inter_token_latency_s_quantiles_p99")),
        "e2e_p50_s": float(get("results_end_to_end_latency_s_quantiles_p50")),
        "e2e_p95_s": float(get("results_end_to_end_latency_s_quantiles_p95")),
        "e2e_p99_s": float(get("results_end_to_end_latency_s_quantiles_p99")),
        "output_tokens_mean": float(get("results_number_output_tokens_mean")),
        "output_tokens_p50": float(get("results_number_output_tokens_quantiles_p50")),
        "output_tokens_p95": float(get("results_number_output_tokens_quantiles_p95")),
    }


def quantile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("cannot compute quantile of empty values")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def load_engine_metrics(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if not isinstance(data, dict):
                raise ValueError(f"{path} contained a non-object JSONL row")
            rows.append(data)
    return rows


def apply_engine_metrics(row: dict[str, Any], metrics_path: Path | None) -> None:
    if metrics_path is None:
        return
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"engine metrics file not found: {metrics_path}. "
            "The server likely failed before completing generation; check server logs above."
        )

    metrics = load_engine_metrics(metrics_path)
    if not metrics:
        raise ValueError(
            f"engine metrics file is empty: {metrics_path}. "
            "The server likely failed before completing generation; check server logs above."
        )
    completed_requests = int(row["completed_requests"])
    if len(metrics) < completed_requests:
        raise ValueError(
            f"{metrics_path} has {len(metrics)} engine metric rows, "
            f"but {completed_requests} completed requests were reported"
        )

    # Warmup requests are logged before LLMPerf requests; keep the measured tail.
    measured = metrics[-completed_requests:]

    generated_tokens = [float(metric["generated_tokens"]) for metric in measured]
    decode_tokens = [float(metric["decode_tokens"]) for metric in measured]
    decode_seconds = [float(metric["decode_s"]) for metric in measured]
    decode_tok_s = [float(metric["decode_tok_s"]) for metric in measured]
    decode_tpot = [
        seconds / tokens
        for seconds, tokens in zip(decode_seconds, decode_tokens)
        if tokens > 0.0
    ]

    total_decode_tokens = sum(decode_tokens)
    total_decode_seconds = sum(decode_seconds)
    row["output_tok_s"] = (
        total_decode_tokens / total_decode_seconds
        if total_decode_seconds > 0.0
        else 0.0
    )
    row["request_tok_s_p50"] = quantile(decode_tok_s, 0.50)
    row["request_tok_s_p95"] = quantile(decode_tok_s, 0.95)
    row["tpot_p50_s"] = quantile(decode_tpot, 0.50)
    row["tpot_p95_s"] = quantile(decode_tpot, 0.95)
    row["tpot_p99_s"] = quantile(decode_tpot, 0.99)
    row["output_tokens_mean"] = sum(generated_tokens) / len(generated_tokens)
    row["output_tokens_p50"] = quantile(generated_tokens, 0.50)
    row["output_tokens_p95"] = quantile(generated_tokens, 0.95)

    print(
        "    engine metrics applied: "
        f"{completed_requests} measured requests, "
        f"{row['output_tok_s']:.2f} decode tok/s, "
        f"generated tokens p50={row['output_tokens_p50']:.0f}"
    )


def validate_output_tokens(row: dict[str, Any], min_output_tokens: int | None) -> None:
    if min_output_tokens is None:
        return

    output_p50 = float(row["output_tokens_p50"])
    output_mean = float(row["output_tokens_mean"])
    if output_p50 < min_output_tokens:
        raise SystemExit(
            "error: output-token validation failed: "
            f"p50={output_p50:.1f}, mean={output_mean:.1f}, required_p50>={min_output_tokens}. "
            "This run likely stopped early and is not comparable with the fixed benchmark series."
        )

    print(
        "    output-token validation passed: "
        f"p50={output_p50:.1f}, mean={output_mean:.1f}, required_p50>={min_output_tokens}"
    )


def read_history(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def check_trial_available(path: Path, trial_id: str) -> None:
    for row in read_history(path):
        if row.get("trial_id") == trial_id:
            label = row.get("run_label", "")
            raise SystemExit(f"error: trial_id {trial_id!r} already exists in {path} ({label})")
    print(f"    trial_id {trial_id} is available")


def write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda row: str(row["run_id"]))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def upsert_history(path: Path, row: dict[str, Any]) -> list[dict[str, Any]]:
    rows = read_history(path)
    rows = [existing for existing in rows if existing.get("run_id") != row["run_id"]]
    rows.append(row)
    write_history(path, rows)
    return sorted(rows, key=lambda existing: str(existing["run_id"]))


def scale(values: list[float], low: float, high: float, value: float) -> float:
    if high == low:
        return 0.5
    return (value - low) / (high - low)


def points_for_metric(rows: list[dict[str, Any]], metric: Metric, x0: float, y0: float, width: float, height: float) -> str:
    values = [float(row[metric.key]) for row in rows]
    low = min(values)
    high = max(values)
    if low == high:
        pad = abs(low) * 0.1 or 1.0
        low -= pad
        high += pad

    points = []
    count = len(rows)
    for index, row in enumerate(rows):
        x = x0 + (width * index / max(count - 1, 1))
        y = y0 + height - (scale(values, low, high, float(row[metric.key])) * height)
        points.append(f"{x:.1f},{y:.1f}")
    return " ".join(points)


def render_chart(path: Path, rows: list[dict[str, Any]]) -> None:
    matplotlib = importlib.import_module("matplotlib")
    matplotlib.use("Agg")
    plt = importlib.import_module("matplotlib.pyplot")

    path.parent.mkdir(parents=True, exist_ok=True)

    labels = [str(row.get("run_label") or row["run_id"]) for row in rows]
    fig, axes = plt.subplots(
        len(CHART_METRICS),
        1,
        figsize=(10, 9),
        sharex=True,
        constrained_layout=True,
    )
    fig.patch.set_facecolor("#0f172a")
    fig.suptitle("LLMPerf performance history", color="#e2e8f0", fontsize=16, fontweight="bold")

    for axis, metric in zip(axes, CHART_METRICS):
        values = [float(row[metric.key]) for row in rows]
        latest = values[-1]
        best = max(values) if metric.higher_is_better else min(values)
        color = "#38bdf8" if metric.higher_is_better else "#a78bfa"

        axis.set_facecolor("#111827")
        axis.plot(labels, values, marker="o", linewidth=2.5, color=color)
        axis.set_title(
            f"{metric.label}    latest {latest:.4g} {metric.unit} / best {best:.4g} {metric.unit}",
            color="#e2e8f0",
            loc="left",
            fontsize=11,
            fontweight="bold",
        )
        axis.tick_params(colors="#94a3b8", labelsize=9)
        axis.grid(True, color="#334155", linewidth=0.7, alpha=0.65)
        for spine in axis.spines.values():
            spine.set_color("#334155")

    axes[-1].set_xlabel("Run", color="#94a3b8")
    axes[-1].tick_params(axis="x", rotation=15)
    fig.text(
        0.01,
        0.01,
        "Generated from benchmarks/history.csv. Higher throughput is better; lower latency is better.",
        color="#94a3b8",
        fontsize=9,
    )
    fig.savefig(path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def print_summary(row: dict[str, Any]) -> None:
    print("    run label:       {}".format(row["run_label"]))
    print("    workload:        {}".format(row["benchmark"]))
    print("    platform:        {}".format(row["platform"]))
    print("    output tok/s:     {:.2f}".format(row["output_tok_s"]))
    print("    TTFT p50 / p95:   {:.3f}s / {:.3f}s".format(row["ttft_p50_s"], row["ttft_p95_s"]))
    print("    TPOT p50 / p95:   {:.4f}s / {:.4f}s".format(row["tpot_p50_s"], row["tpot_p95_s"]))
    print("    E2E p50 / p95:    {:.3f}s / {:.3f}s".format(row["e2e_p50_s"], row["e2e_p95_s"]))
    print("    request tok/s p50: {:.2f}".format(row["request_tok_s_p50"]))
    print("    requests/errors:  {} / {}".format(row["completed_requests"], row["errors"]))
    print()
    print("    Markdown row:")
    print(
        "    | {run_label} | {date} | {platform} | {precision} | {model} | CHANGE | {output_tok_s:.2f} | "
        "{ttft_p50_s:.3f} | {ttft_p95_s:.3f} | {tpot_p50_s:.4f} | {tpot_p95_s:.4f} | "
        "{e2e_p50_s:.3f} | {e2e_p95_s:.3f} | {completed_requests} | {errors} | NOTES |".format(**row)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Record LLMPerf metrics and render charts.")
    parser.add_argument("summary", type=Path, nargs="?", help="Path to an LLMPerf *_summary.json file.")
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--chart", type=Path, default=DEFAULT_CHART)
    parser.add_argument("--check-trial-id", default=None, help="Fail if this trial id is already in history.")
    parser.add_argument("--trial-id", default=None, help="Globally unique short trial id, e.g. t1.")
    parser.add_argument("--platform", default=None, help="Short platform key, e.g. m3_metal.")
    parser.add_argument("--change", default=None, help="Optional short change key, e.g. flash_attention.")
    parser.add_argument("--run-label", default=None, help="Short x-axis label, e.g. t1-m3_metal.")
    parser.add_argument("--precision", default=None, help="Precision label for the history row.")
    parser.add_argument(
        "--min-output-tokens",
        type=int,
        default=None,
        help="Fail if the run's p50 generated output-token count is below this floor.",
    )
    parser.add_argument(
        "--engine-metrics",
        type=Path,
        default=None,
        help="Optional JSONL file of server-side engine metrics to use for token throughput/counts.",
    )
    args = parser.parse_args()

    if args.check_trial_id:
        check_trial_available(args.history, args.check_trial_id)
        return

    if args.summary is None:
        parser.error("summary is required unless --check-trial-id is used")

    row = extract_row(
        args.summary,
        args.trial_id,
        args.platform,
        args.change,
        args.run_label,
        args.precision,
    )
    apply_engine_metrics(row, args.engine_metrics)
    validate_output_tokens(row, args.min_output_tokens)
    rows = upsert_history(args.history, row)
    render_chart(args.chart, rows)

    print_summary(row)
    print(f"    history:         {args.history}")
    print(f"    chart:           {args.chart}")


if __name__ == "__main__":
    main()
