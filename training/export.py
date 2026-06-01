"""Export the fine-tuned DistilBERT model from PyTorch to ONNX.

ONNX (Open Neural Network Exchange) is the bridge between the PyTorch training
world and the deployment targets in this project: onnxruntime-web (browser
WebAssembly) and onnxruntime in a Cloudflare Worker. We export with
``opset_version=14`` (broad runtime support) and declare dynamic axes for the
batch and sequence dimensions so a single exported graph serves any input shape.

Critically, we do not trust the export blindly: we run 1000 validation examples
through both the original PyTorch model and the exported ONNX graph and assert
that the logits agree to within 1e-4. A correct export is the foundation for
every benchmark and deployment downstream, so we verify it loudly.

Usage:
    python export.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import onnx
import onnxruntime as ort
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

MODEL_DIR: Path = Path(__file__).parent / "models" / "distilbert-sst2-finetuned"
ONNX_PATH: Path = Path(__file__).parent / "models" / "distilbert-sst2.onnx"
EXPORT_RESULTS_PATH: Path = Path(__file__).parent / "export_results.json"
OPSET_VERSION: int = 14
MAX_SEQ_LENGTH: int = 128
NUM_VERIFY_EXAMPLES: int = 1000
TOLERANCE: float = 1e-4


def load_pytorch_model() -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load the fine-tuned model + tokenizer in eval mode on CPU."""
    if not (MODEL_DIR / "config.json").exists():
        raise FileNotFoundError(
            f"Fine-tuned model not found at {MODEL_DIR}. "
            "Run training on Colab and unzip the model into training/models/ first."
        )
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model.eval()
    return model, tokenizer


def export_to_onnx(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase
) -> None:
    """Trace the model with a dummy batch and export to ONNX with dynamic axes."""
    ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)

    # A representative dummy input. The concrete shape here does not constrain
    # runtime shapes because we declare batch/sequence as dynamic below.
    dummy = tokenizer(
        "the export tracer needs a representative sentence",
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
    )
    input_ids = dummy["input_ids"]
    attention_mask = dummy["attention_mask"]

    dynamic_axes: Dict[str, Dict[int, str]] = {
        "input_ids": {0: "batch", 1: "sequence"},
        "attention_mask": {0: "batch", 1: "sequence"},
        "logits": {0: "batch"},
    }

    with torch.no_grad():
        torch.onnx.export(
            model,
            (input_ids, attention_mask),
            str(ONNX_PATH),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes=dynamic_axes,
            opset_version=OPSET_VERSION,
            do_constant_folding=True,
        )

    # Structural validation of the produced graph.
    onnx.checker.check_model(onnx.load(str(ONNX_PATH)))


def _pytorch_logits(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    sentences: List[str],
) -> np.ndarray:
    """Run a batch of sentences through the PyTorch model, return logits."""
    enc = tokenizer(
        sentences,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
    )
    with torch.no_grad():
        out = model(**enc)
    return out.logits.numpy()


def _onnx_logits(
    session: ort.InferenceSession,
    tokenizer: PreTrainedTokenizerBase,
    sentences: List[str],
) -> np.ndarray:
    """Run a batch of sentences through the ONNX session, return logits."""
    enc = tokenizer(
        sentences,
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
    )
    inputs = {
        "input_ids": enc["input_ids"].astype(np.int64),
        "attention_mask": enc["attention_mask"].astype(np.int64),
    }
    return session.run(["logits"], inputs)[0]


def verify_parity(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase
) -> float:
    """Compare PyTorch vs ONNX logits over validation examples.

    Returns the maximum absolute logit difference observed. Raises
    ``AssertionError`` if it exceeds ``TOLERANCE``.
    """
    dataset = load_dataset("glue", "sst2", split="validation")
    sentences: List[str] = list(dataset["sentence"])[:NUM_VERIFY_EXAMPLES]

    session = ort.InferenceSession(
        str(ONNX_PATH), providers=["CPUExecutionProvider"]
    )

    max_diff = 0.0
    batch_size = 32
    for start in range(0, len(sentences), batch_size):
        batch = sentences[start : start + batch_size]
        pt = _pytorch_logits(model, tokenizer, batch)
        on = _onnx_logits(session, tokenizer, batch)
        max_diff = max(max_diff, float(np.max(np.abs(pt - on))))

    print(f"Verified {len(sentences)} examples.")
    print(f"Max absolute logit difference: {max_diff:.2e} (tolerance {TOLERANCE:.0e})")
    assert max_diff < TOLERANCE, (
        f"ONNX export diverges from PyTorch by {max_diff:.2e} "
        f"(> {TOLERANCE:.0e}). Export is NOT trustworthy."
    )
    return max_diff


def report_size() -> float:
    """Print and return the ONNX model size in MB."""
    size_mb = ONNX_PATH.stat().st_size / (1024 * 1024)
    print(f"ONNX model size: {size_mb:.2f} MB")
    return size_mb


def main() -> None:
    model, tokenizer = load_pytorch_model()
    print(f"Exporting to ONNX (opset {OPSET_VERSION}) -> {ONNX_PATH}")
    export_to_onnx(model, tokenizer)
    size_mb = report_size()
    max_diff = verify_parity(model, tokenizer)

    # Persist verification result so walkthrough.ipynb / README can cite the real
    # measured parity number instead of a hardcoded value.
    EXPORT_RESULTS_PATH.write_text(
        json.dumps(
            {
                "opset_version": OPSET_VERSION,
                "size_mb": round(size_mb, 2),
                "num_verify_examples": min(NUM_VERIFY_EXAMPLES, 872),
                "max_abs_logit_diff": max_diff,
                "tolerance": TOLERANCE,
                "verified": max_diff < TOLERANCE,
            },
            indent=2,
        )
    )

    print("\n" + "=" * 60)
    print("EXPORT VERIFIED")
    print("=" * 60)


if __name__ == "__main__":
    main()
