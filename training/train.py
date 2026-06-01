"""Fine-tune DistilBERT on the SST-2 sentiment classification task.

This is the first stage of the EdgeSentiment pipeline. We take the
``distilbert-base-uncased`` checkpoint and fine-tune a 2-class sequence
classifier on the Stanford Sentiment Treebank (SST-2) subset of GLUE.

Why DistilBERT: it retains ~97% of BERT-base's GLUE performance while being
~40% smaller and ~60% faster, which matters when the end goal is sub-100ms
inference inside a browser WebAssembly runtime and a Cloudflare Worker.

The trained model + tokenizer are saved to ``./models/distilbert-sst2-finetuned``
so the downstream ``export.py`` / ``optimize.py`` / ``benchmark.py`` stages can
load them without re-training.

Note: training is intended to run on a GPU (Google Colab T4). Use
``train_colab.ipynb`` for the Colab-ready version, which calls into this module.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import evaluate
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

MODEL_NAME: str = "distilbert-base-uncased"
OUTPUT_DIR: Path = Path(__file__).parent / "models" / "distilbert-sst2-finetuned"
MAX_SEQ_LENGTH: int = 128
TARGET_ACCURACY: float = 0.91


@dataclass(frozen=True)
class TrainConfig:
    """Hyperparameters for fine-tuning. Defaults match the project spec."""

    num_epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    seed: int = 42


# --------------------------------------------------------------------------- #
# Data + metrics
# --------------------------------------------------------------------------- #

_accuracy_metric = evaluate.load("accuracy")


def build_tokenize_fn(tokenizer: PreTrainedTokenizerBase):
    """Return a closure that tokenizes the SST-2 ``sentence`` field.

    Padding is left to the dynamic ``DataCollatorWithPadding`` so we pad to the
    longest example in each batch rather than to ``MAX_SEQ_LENGTH`` globally.
    """

    def tokenize(batch: Dict[str, Any]) -> Dict[str, Any]:
        return tokenizer(
            batch["sentence"],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
        )

    return tokenize


def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    """Accuracy on the validation set, used by the Trainer for evaluation."""
    logits = eval_pred.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    predictions = np.argmax(logits, axis=-1)
    result = _accuracy_metric.compute(
        predictions=predictions, references=eval_pred.label_ids
    )
    assert result is not None  # evaluate returns Optional; SST-2 always yields a value
    return {"accuracy": float(result["accuracy"])}


def load_splits(
    tokenizer: PreTrainedTokenizerBase,
) -> tuple[Dataset, Dataset]:
    """Load and tokenize the SST-2 train/validation splits from GLUE."""
    raw = load_dataset("glue", "sst2")
    tokenize = build_tokenize_fn(tokenizer)
    train_ds = raw["train"].map(tokenize, batched=True)
    eval_ds = raw["validation"].map(tokenize, batched=True)
    return train_ds, eval_ds


# --------------------------------------------------------------------------- #
# Training entry point
# --------------------------------------------------------------------------- #


def train(config: TrainConfig = TrainConfig(), output_dir: Path = OUTPUT_DIR) -> float:
    """Fine-tune DistilBERT on SST-2 and persist the model + tokenizer.

    Returns the final validation accuracy. Raises ``AssertionError`` if the
    accuracy does not clear ``TARGET_ACCURACY`` (>91%), surfacing regressions
    loudly rather than silently shipping a weak model.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label={0: "negative", 1: "positive"},
        label2id={"negative": 0, "positive": 1},
    )

    train_ds, eval_ds = load_splits(tokenizer)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=64,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        seed=config.seed,
        report_to=[],  # no W&B / TensorBoard by default
        fp16=_cuda_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    accuracy = float(metrics["eval_accuracy"])

    # Persist the best model + tokenizer to a clean, predictable path.
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Save the training log history so the walkthrough notebook can plot the
    # loss curve from real numbers instead of re-running training.
    _save_log_history(trainer, output_dir, accuracy)

    print("\n" + "=" * 60)
    print(f"Final validation accuracy: {accuracy * 100:.2f}%")
    print(f"Target accuracy:           {TARGET_ACCURACY * 100:.2f}%")
    print("=" * 60)

    assert accuracy >= TARGET_ACCURACY, (
        f"Validation accuracy {accuracy:.4f} is below target "
        f"{TARGET_ACCURACY:.4f}. Investigate before exporting."
    )
    print("TARGET ACCURACY MET\n")
    return accuracy


def _save_log_history(trainer: Trainer, output_dir: Path, accuracy: float) -> None:
    """Dump Trainer log history + final accuracy to ``training_log.json``."""
    payload = {
        "final_accuracy": accuracy,
        "log_history": trainer.state.log_history,
    }
    (output_dir / "training_log.json").write_text(json.dumps(payload, indent=2))


def _cuda_available() -> bool:
    """Return True if a CUDA GPU is available (enables fp16 on Colab T4)."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


if __name__ == "__main__":
    # Guardrail: this project trains on Colab T4, never locally. Set
    # EDGE_SENTIMENT_ALLOW_LOCAL_TRAIN=1 to override (not recommended).
    if not os.environ.get("EDGE_SENTIMENT_ALLOW_LOCAL_TRAIN"):
        if not _cuda_available():
            raise SystemExit(
                "No CUDA GPU detected. Training is intended for Google Colab T4 "
                "via train_colab.ipynb. Set EDGE_SENTIMENT_ALLOW_LOCAL_TRAIN=1 "
                "to force CPU training (slow, not recommended)."
            )
    train()
