"""Graph-optimize and INT8-quantize the ONNX model, then ship it to the web app.

Two transformations, in order:

1. Graph optimization (``onnxruntime.transformers.optimizer``, ``model_type="bert"``).
   Fuses attention, layer-norm, and GELU subgraphs into single optimized kernels.
   Stays in FP32; this is purely a graph-rewrite for speed, no accuracy change.

2. INT8 dynamic quantization (``quantize_dynamic``, ``QuantType.QInt8``).
   Stores weights as INT8 and quantizes activations on-the-fly at inference. This
   roughly quarters the model size (FP32 -> INT8) and speeds up CPU/WASM inference,
   at the cost of a small, bounded accuracy drop. We assert INT8 stays within
   0.5% of FP32 accuracy on the SST-2 validation set.

Why dynamic (not static) quantization: dynamic needs no calibration dataset and
computes activation scales per-inference, which is ideal for a portfolio pipeline
and for variable-length text inputs. Static quantization can be marginally faster
but requires a representative calibration set and more tuning.

Finally, the INT8 model is copied automatically into ``web/public/model/`` so the
browser app always serves the latest quantized weights — no manual step.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np
import onnxruntime as ort
from datasets import load_dataset
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.transformers.optimizer import optimize_model
from transformers import AutoTokenizer, PreTrainedTokenizerBase

MODELS_DIR: Path = Path(__file__).parent / "models"
MODEL_DIR: Path = MODELS_DIR / "distilbert-sst2-finetuned"
FP32_PATH: Path = MODELS_DIR / "distilbert-sst2.onnx"
OPTIMIZED_PATH: Path = MODELS_DIR / "distilbert-sst2-optimized.onnx"
INT8_PATH: Path = MODELS_DIR / "distilbert-sst2-int8.onnx"
WEB_MODEL_PATH: Path = (
    Path(__file__).parent.parent / "web" / "public" / "model" / "distilbert-sst2-int8.onnx"
)

MAX_SEQ_LENGTH: int = 128
ACCURACY_TOLERANCE: float = 0.005  # INT8 must stay within 0.5% of FP32


def optimize_graph() -> None:
    """Step 1: fuse BERT subgraphs into optimized ONNX kernels (still FP32)."""
    if not FP32_PATH.exists():
        raise FileNotFoundError(
            f"{FP32_PATH} not found. Run export.py before optimize.py."
        )
    # num_heads / hidden_size for distilbert-base-uncased.
    optimized = optimize_model(
        str(FP32_PATH),
        model_type="bert",
        num_heads=12,
        hidden_size=768,
    )
    optimized.save_model_to_file(str(OPTIMIZED_PATH))
    print(f"Graph-optimized model saved -> {OPTIMIZED_PATH}")


def quantize_int8() -> None:
    """Step 2: INT8 dynamic quantization of the optimized graph."""
    quantize_dynamic(
        model_input=str(OPTIMIZED_PATH),
        model_output=str(INT8_PATH),
        weight_type=QuantType.QInt8,
    )
    print(f"INT8-quantized model saved -> {INT8_PATH}")


def _size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def print_size_table() -> None:
    """Print original -> optimized -> INT8 sizes with percentage reduction."""
    fp32 = _size_mb(FP32_PATH)
    opt = _size_mb(OPTIMIZED_PATH)
    int8 = _size_mb(INT8_PATH)

    def pct(after: float) -> str:
        return f"{(1 - after / fp32) * 100:5.1f}%"

    print("\n" + "=" * 60)
    print("MODEL SIZE")
    print("=" * 60)
    print(f"{'Variant':<22}{'Size (MB)':>12}{'Reduction':>14}")
    print("-" * 60)
    print(f"{'Original FP32':<22}{fp32:>12.2f}{'  baseline':>14}")
    print(f"{'Optimized FP32':<22}{opt:>12.2f}{pct(opt):>14}")
    print(f"{'INT8':<22}{int8:>12.2f}{pct(int8):>14}")
    print("=" * 60)


def _evaluate_accuracy(
    onnx_path: Path, tokenizer: PreTrainedTokenizerBase
) -> float:
    """Accuracy of an ONNX variant on the full SST-2 validation split."""
    dataset = load_dataset("glue", "sst2", split="validation")
    sentences: List[str] = list(dataset["sentence"])
    labels = np.array(dataset["label"], dtype=np.int64)

    session = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )

    preds: List[int] = []
    batch_size = 32
    for start in range(0, len(sentences), batch_size):
        batch = sentences[start : start + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
        )
        inputs: Dict[str, np.ndarray] = {
            "input_ids": enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
        }
        logits = session.run(["logits"], inputs)[0]
        preds.extend(np.argmax(logits, axis=-1).tolist())

    return float((np.array(preds) == labels).mean())


def check_accuracy() -> Dict[str, float]:
    """Evaluate all three variants and assert INT8 is within tolerance of FP32."""
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))

    print("\nEvaluating accuracy on SST-2 validation set...")
    fp32_acc = _evaluate_accuracy(FP32_PATH, tokenizer)
    opt_acc = _evaluate_accuracy(OPTIMIZED_PATH, tokenizer)
    int8_acc = _evaluate_accuracy(INT8_PATH, tokenizer)

    print("\n" + "=" * 60)
    print("ACCURACY")
    print("=" * 60)
    print(f"{'Original FP32':<22}{fp32_acc * 100:>8.2f}%")
    print(f"{'Optimized FP32':<22}{opt_acc * 100:>8.2f}%")
    print(f"{'INT8':<22}{int8_acc * 100:>8.2f}%")
    print("=" * 60)

    drop = fp32_acc - int8_acc
    print(f"INT8 accuracy drop vs FP32: {drop * 100:.3f}% "
          f"(tolerance {ACCURACY_TOLERANCE * 100:.1f}%)")
    assert drop <= ACCURACY_TOLERANCE, (
        f"INT8 accuracy dropped {drop * 100:.3f}%, exceeding the "
        f"{ACCURACY_TOLERANCE * 100:.1f}% tolerance."
    )
    return {"fp32": fp32_acc, "optimized": opt_acc, "int8": int8_acc}


def copy_to_web() -> None:
    """Copy the INT8 model into the web app's public/model directory."""
    WEB_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(INT8_PATH, WEB_MODEL_PATH)
    print("\n" + "=" * 60)
    print("MODEL COPIED TO WEB")
    print(f"  {INT8_PATH}")
    print(f"  -> {WEB_MODEL_PATH}")
    print("=" * 60)


def main() -> None:
    optimize_graph()
    quantize_int8()
    print_size_table()
    check_accuracy()
    copy_to_web()


if __name__ == "__main__":
    main()
