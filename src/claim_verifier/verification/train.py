"""Fine-tune a sequence classifier on SciFact's train rationale labels."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .data import LABELS
from .verifier import DEFAULT_NLI_MODEL


def _encode_examples(tokenizer: Any, examples: list[dict[str, Any]], max_length: int):
    evidence = [" ".join(example["evidence"]) for example in examples]
    claims = [example["claim"] for example in examples]
    encodings = tokenizer(
        evidence,
        claims,
        truncation=True,
        max_length=max_length,
    )
    label_to_id = {label: index for index, label in enumerate(LABELS)}
    labels = [label_to_id[example["label"]] for example in examples]
    return encodings, labels


def _make_loader(
    encodings: Any,
    labels: list[int],
    tokenizer: Any,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    rows = [
        {key: encodings[key][index] for key in encodings}
        for index in range(len(labels))
    ]
    generator = torch.Generator()
    generator.manual_seed(seed)

    def collate(features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch_labels = torch.tensor([feature["labels"] for feature in features])
        model_features = [
            {key: value for key, value in feature.items() if key != "labels"}
            for feature in features
        ]
        batch = tokenizer.pad(model_features, padding=True, return_tensors="pt")
        batch["labels"] = batch_labels
        return batch

    for index, row in enumerate(rows):
        row["labels"] = labels[index]
    return DataLoader(
        rows,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        collate_fn=collate,
    )


def _evaluate(model: Any, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    gold: list[int] = []
    predicted: list[int] = []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels")
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch).logits
            predicted.extend(logits.argmax(dim=-1).cpu().tolist())
            gold.extend(labels.tolist())
    return {
        "accuracy": float(accuracy_score(gold, predicted)),
        "macro_f1": float(
            f1_score(gold, predicted, labels=[0, 1], average="macro", zero_division=0)
        ),
    }


def train_verifier(
    train_examples: list[dict[str, Any]],
    validation_examples: list[dict[str, Any]],
    output_dir: str | Path = "checkpoints/verifier",
    model_name: str = DEFAULT_NLI_MODEL,
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    max_length: int = 512,
    seed: int = 42,
) -> list[dict[str, float]]:
    """Train on train_examples and retain the epoch with best validation F1."""
    if not train_examples or not validation_examples:
        raise ValueError("Training and validation examples must both be non-empty")
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(LABELS),
        id2label={index: label for index, label in enumerate(LABELS)},
        label2id={label: index for index, label in enumerate(LABELS)},
        ignore_mismatched_sizes=True,
    ).to(device)

    train_encodings, train_labels = _encode_examples(tokenizer, train_examples, max_length)
    val_encodings, val_labels = _encode_examples(tokenizer, validation_examples, max_length)
    train_loader = _make_loader(
        train_encodings, train_labels, tokenizer, batch_size, shuffle=True, seed=seed
    )
    val_loader = _make_loader(
        val_encodings, val_labels, tokenizer, batch_size, shuffle=False, seed=seed
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps = epochs * len(train_loader)
    warmup_steps = int(total_steps * 0.1)

    def lr_scale(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)

    checkpoint_dir = Path(output_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_macro_f1 = -1.0
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            labels = batch.pop("labels").to(device)
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**batch, labels=labels)
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            running_loss += float(outputs.loss.detach().cpu())

        metrics = _evaluate(model, val_loader, device)
        epoch_result = {
            "epoch": float(epoch),
            "train_loss": running_loss / max(1, len(train_loader)),
            **metrics,
        }
        history.append(epoch_result)
        print(
            f"Epoch {epoch}/{epochs} | loss={epoch_result['train_loss']:.4f} "
            f"| val_accuracy={metrics['accuracy']:.4f} "
            f"| val_macro_f1={metrics['macro_f1']:.4f}"
        )

        if metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = metrics["macro_f1"]
            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            metadata = {
                "base_model": model_name,
                "best_epoch": epoch,
                "validation_accuracy": metrics["accuracy"],
                "validation_macro_f1": metrics["macro_f1"],
                "labels": LABELS,
                "seed": seed,
                "epochs_requested": epochs,
                "train_examples": len(train_examples),
                "validation_examples": len(validation_examples),
                "max_length": max_length,
            }
            (checkpoint_dir / "training_metadata.json").write_text(
                json.dumps(metadata, indent=2), encoding="utf-8"
            )

    return history
