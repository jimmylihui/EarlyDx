"""Completion-only full fine-tuning of the Qwen3.5 language backbone (Appendix K)."""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from earlydx import (
    answer_labels,
    ensure_complete,
    file_hash,
    read_rows,
    run_metadata,
    validate_rows,
    verdict_map,
)


def encode_example(tokenizer, row, max_length):
    text = tokenizer.apply_chat_template(
        row["messages"], tokenize=False, add_generation_prompt=False
    )
    marker = "<|im_start|>assistant\n"
    j = text.rfind(marker)
    if j < 0:
        raise ValueError("Tokenizer template has no Qwen assistant boundary")
    tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
    prompt = tokenizer(text[: j + len(marker)], add_special_tokens=False)["input_ids"]
    if tokens[: len(prompt)] != prompt:
        raise ValueError("Tokenizer changed tokens at the prompt/completion boundary")
    labels = [-100] * len(prompt) + tokens[len(prompt) :]
    ids, labels = tokens[:max_length], labels[:max_length]
    if all(x == -100 for x in labels):
        raise ValueError(
            f"stay {row['stay_id']}: prompt leaves no supervised tokens; revise serialization"
        )
    return {"input_ids": ids, "labels": labels}


def main(default_size="4B"):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", choices=("2B", "4B"), default=default_size)
    ap.add_argument("--format", choices=("cot", "direct"), default="cot")
    ap.add_argument(
        "--revision",
        required=True,
        help="Exact base model revision, recorded for reproducibility",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", help="A checkpoint directory")
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="Short single-GPU plumbing check, not a training run",
    )
    a = ap.parse_args()
    if not a.revision.strip() or "<" in a.revision:
        ap.error("revision must be an actual checkpoint revision")
    rows = sorted(validate_rows(read_rows(a.train)), key=lambda r: r["stay_id"])
    ensure_complete(a.train, rows)
    if not rows or any(r.get("split") != "train" for r in rows):
        raise ValueError("Training input must contain only the training split")
    for row in rows:
        if any(v["verdict"] == "unsupported" for v in verdict_map(row).values()):
            raise ValueError("Unsupported labels must not enter training")
        answer_labels(row["messages"][-1]["content"])
        if a.format == "cot" and "<think>" not in row["messages"][-1]["content"]:
            raise ValueError("CoT training requires generated rationales")
        if a.format == "direct" and "<think>" in row["messages"][-1]["content"]:
            raise ValueError("Direct training requires the answer-only training file")
    model_id = f"Qwen/Qwen3.5-{a.size}"
    world = int(os.environ.get("WORLD_SIZE", "1"))
    batch = 2 if a.size == "2B" else 1
    if 48 % (batch * world):
        raise ValueError("GPU count must allow effective batch size 48")
    recipe = {
        "stage": "sft",
        "seed": a.seed,
        "model": model_id,
        "revision": a.revision,
        "format": a.format,
        "train_sha256": file_hash(a.train),
        "max_length": 3072,
        "epochs": 1,
        "learning_rate": 1e-5,
        "warmup_ratio": 0.03,
        "scheduler": "cosine",
        "effective_batch_size": 48,
        "world_size": world,
        "smoke": a.smoke,
        "optimizer": "adamw_torch",
        "bf16": True,
        "gradient_checkpointing": True,
        "vision_backbone": "frozen",
        "deepspeed_zero_stage": 2 if a.size == "4B" and not a.smoke else None,
    }
    if int(os.environ.get("RANK", "0")) == 0:
        # The sentinel is a file inside the output directory, so other DDP ranks creating
        # the directory cannot race with the provenance check.
        run_metadata(Path(a.out) / "training-run.json", recipe)
    import torch
    from torch.utils.data import Dataset
    from transformers import (
        AutoTokenizer,
        AutoModelForImageTextToText,
        Trainer,
        TrainingArguments,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=a.revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    class Examples(Dataset):
        def __init__(self):
            self.data = [
                encode_example(tokenizer, r, 3072)
                for r in (rows[:64] if a.smoke else rows)
            ]

        def __len__(self):
            return len(self.data)

        def __getitem__(self, i):
            return self.data[i]

    def collate(items):
        mx = max(len(x["input_ids"]) for x in items)
        return {
            "input_ids": torch.tensor(
                [
                    x["input_ids"]
                    + [tokenizer.pad_token_id] * (mx - len(x["input_ids"]))
                    for x in items
                ]
            ),
            "labels": torch.tensor(
                [x["labels"] + [-100] * (mx - len(x["labels"])) for x in items]
            ),
            "attention_mask": torch.tensor(
                [
                    [1] * len(x["input_ids"]) + [0] * (mx - len(x["input_ids"]))
                    for x in items
                ]
            ),
        }

    model = AutoModelForImageTextToText.from_pretrained(
        model_id, revision=a.revision, dtype=torch.bfloat16
    )
    model.config.use_cache = False
    for name, param in model.named_parameters():
        if any(
            k in name.lower()
            for k in ("visual", "vision", "image", "patch_embed", "merger")
        ):
            param.requires_grad = False
    cfg = TrainingArguments(
        output_dir=a.out,
        seed=a.seed,
        data_seed=a.seed,
        per_device_train_batch_size=batch,
        gradient_accumulation_steps=48 // (batch * world),
        num_train_epochs=1,
        learning_rate=1e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        optim="adamw_torch",
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=5,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        report_to="none",
        max_steps=3 if a.smoke else -1,
        ddp_find_unused_parameters=False,
        deepspeed=str(Path(__file__).with_name("zero2_offload.json"))
        if a.size == "4B" and not a.smoke
        else None,
    )
    trainer = Trainer(
        model=model, args=cfg, train_dataset=Examples(), data_collator=collate
    )
    trainer.train(resume_from_checkpoint=a.resume)
    if not a.smoke:
        trainer.save_model(a.out)
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(a.out)
    print(f"Training completed: {a.out}")


if __name__ == "__main__":
    main()
