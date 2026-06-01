"""Benchmark inference latency across the three ONNX model variants.

Measures wall-clock inference latency for FP32, optimized FP32, and INT8 models
using ``onnxruntime.InferenceSession`` under a controlled, representative load:
batch size 1, sequence length 128 (the deployment scenario for a single user
typing one sentence). We use a fixed tokenized input so the measurement isolates
model execution from tokenization noise.

Methodology (so the numbers are defensible in an interview):
  * 200 warm-up runs to stabilize CPU frequency / caches / ORT internal buffers.
  * 500 measured runs, from which we report p50 / p95 / p99 latency.
  * Percentiles, not just the mean, because tail latency is what users feel.

We also record model load (session-init) time and on-disk size per variant.
Results are written to ``benchmark_results.json`` so the README and walkthrough
notebook render real measured numbers rather than hardcoded estimates.

Note: latency is hardware-dependent. Numbers measured on the machine that runs
this script (CPU EP). The relative ordering FP32 > optimized > INT8 is the
portable takeaway.
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import onnxruntime as ort

MODELS_DIR: Path = Path(__file__).parent / "models"
RESULTS_PATH: Path = Path(__file__).parent / "benchmark_results.json"

SEQ_LEN: int = 128
BATCH_SIZE: int = 1
WARMUP_RUNS: int = 200
MEASURED_RUNS: int = 500

VARIANTS: Dict[str, Path] = {
    "FP32": MODELS_DIR / "distilbert-sst2.onnx",
    "Optimized FP32": MODELS_DIR / "distilbert-sst2-optimized.onnx",
    "INT8": MODELS_DIR / "distilbert-sst2-int8.onnx",
}


@dataclass
class VariantResult:
    """Measured benchmark numbers for a single model variant."""

    variant: str
    size_mb: float
    load_time_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float


def _make_inputs() -> Dict[str, np.ndarray]:
    """Build a fixed (batch=1, seq=128) integer input for repeatable timing."""
    rng = np.random.default_rng(seed=0)
    # Token ids in a realistic vocab range; attention_mask all ones (no padding
    # effect on compute for a single full-length sequence).
    input_ids = rng.integers(
        low=999, high=29000, size=(BATCH_SIZE, SEQ_LEN), dtype=np.int64
    )
    attention_mask = np.ones((BATCH_SIZE, SEQ_LEN), dtype=np.int64)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def benchmark_variant(name: str, path: Path) -> VariantResult:
    """Time load + inference for one variant and return its measurements."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run export.py and optimize.py before benchmarking."
        )

    size_mb = path.stat().st_size / (1024 * 1024)

    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1  # deterministic single-thread timing

    load_start = time.perf_counter()
    session = ort.InferenceSession(
        str(path),
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )
    load_time_ms = (time.perf_counter() - load_start) * 1000.0

    inputs = _make_inputs()
    output_names = ["logits"]

    for _ in range(WARMUP_RUNS):
        session.run(output_names, inputs)

    latencies_ms: List[float] = []
    for _ in range(MEASURED_RUNS):
        start = time.perf_counter()
        session.run(output_names, inputs)
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

    arr = np.array(latencies_ms)
    result = VariantResult(
        variant=name,
        size_mb=round(size_mb, 2),
        load_time_ms=round(load_time_ms, 2),
        p50_ms=round(float(np.percentile(arr, 50)), 2),
        p95_ms=round(float(np.percentile(arr, 95)), 2),
        p99_ms=round(float(np.percentile(arr, 99)), 2),
        mean_ms=round(float(arr.mean()), 2),
    )
    return result


def print_table(results: List[VariantResult]) -> None:
    """Render a clean fixed-width comparison table to stdout."""
    print("\n" + "=" * 78)
    print("INFERENCE LATENCY  (batch=1, seq=128, "
          f"{WARMUP_RUNS} warmup + {MEASURED_RUNS} measured runs)")
    print("=" * 78)
    header = (
        f"{'Variant':<16}{'Size(MB)':>10}{'Load(ms)':>10}"
        f"{'p50(ms)':>10}{'p95(ms)':>10}{'p99(ms)':>10}"
    )
    print(header)
    print("-" * 78)
    for r in results:
        print(
            f"{r.variant:<16}{r.size_mb:>10.2f}{r.load_time_ms:>10.2f}"
            f"{r.p50_ms:>10.2f}{r.p95_ms:>10.2f}{r.p99_ms:>10.2f}"
        )
    print("=" * 78)


def save_results(results: List[VariantResult]) -> None:
    """Persist results + environment metadata to benchmark_results.json."""
    payload = {
        "config": {
            "batch_size": BATCH_SIZE,
            "seq_len": SEQ_LEN,
            "warmup_runs": WARMUP_RUNS,
            "measured_runs": MEASURED_RUNS,
        },
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "onnxruntime": ort.__version__,
        },
        "results": [asdict(r) for r in results],
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2))
    print(f"\nResults written to {RESULTS_PATH}")


def main() -> None:
    results: List[VariantResult] = []
    for name, path in VARIANTS.items():
        print(f"Benchmarking {name}...")
        results.append(benchmark_variant(name, path))
    print_table(results)
    save_results(results)


if __name__ == "__main__":
    main()
