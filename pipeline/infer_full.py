"""Greedy prediction from a saved post-trained checkpoint; no gold enters the prompt."""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from earlydx import (
    answer_labels,
    file_hash,
    identity,
    input_of,
    read_rows,
    run_metadata,
    transform_input,
    validate_rows,
)


def select_shard(rows, shard, num_shards):
    if num_shards < 1 or not 0 <= shard < num_shards:
        raise ValueError("Require 0 <= shard < num_shards")
    return [(i, r) for i, r in enumerate(rows) if i % num_shards == shard]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test", required=True)
    ap.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Local trained checkpoint including tokenizer",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--window-hours", type=int, choices=(0, 6, 24), default=0)
    ap.add_argument(
        "--timestamp", choices=("charttime", "storetime"), default="charttime"
    )
    controls = ap.add_mutually_exclusive_group()
    controls.add_argument("--demo-cc-only", action="store_true")
    controls.add_argument(
        "--drop-modality",
        choices=(
            "radiology",
            "labs",
            "history",
            "ecg",
            "echo",
            "vitals",
            "medications",
        ),
    )
    a = ap.parse_args()
    if not a.checkpoint.is_dir() or not (a.checkpoint / "config.json").exists():
        ap.error("checkpoint must be an existing local model directory")
    if a.batch_size < 1:
        ap.error("batch-size must be positive")
    rows = validate_rows(
        read_rows(a.test), window=a.window_hours, timestamp=a.timestamp
    )
    if not rows or any(r.get("split") != "test" for r in rows):
        raise ValueError("Inference requires the held-out test split")
    selected = select_shard(rows, a.shard, a.num_shards)
    # Hash local weights too: replacing weights at the same path must invalidate resume.
    weights = sorted(a.checkpoint.glob("*.safetensors")) + sorted(
        a.checkpoint.glob("pytorch_model*.bin")
    )
    if not weights:
        raise ValueError("No saved model weights found")
    model_files = (
        weights
        + [a.checkpoint / "config.json"]
        + sorted(a.checkpoint.glob("tokenizer*"))
    )
    run_metadata(
        a.out,
        {
            "stage": "inference",
            "test_sha256": file_hash(a.test),
            "checkpoint": str(a.checkpoint.resolve()),
            "model_files": {f.name: file_hash(f) for f in model_files},
            "shard": a.shard,
            "num_shards": a.num_shards,
            "max_input_tokens": 3000,
            "max_new_tokens": 2048,
            "do_sample": False,
            "demo_cc_only": a.demo_cc_only,
            "drop_modality": a.drop_modality,
        },
    )
    old = read_rows(a.out) if Path(a.out).exists() else []
    validate_rows(old, window=a.window_hours, timestamp=a.timestamp)
    done = {r["stay_id"] for r in old}
    if done - {r["stay_id"] for _, r in selected}:
        raise ValueError("Saved predictions do not belong to this shard")
    todo = [(i, r) for i, r in selected if r["stay_id"] not in done]
    if not todo:
        print("Shard already complete")
        return
    import torch
    from transformers import AutoTokenizer, AutoModelForImageTextToText

    tok = AutoTokenizer.from_pretrained(a.checkpoint)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForImageTextToText.from_pretrained(
        a.checkpoint, dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    with open(a.out, "a") as out:
        for start in range(0, len(todo), a.batch_size):
            batch = todo[start : start + a.batch_size]
            inputs = [
                transform_input(
                    input_of(r), demo_cc=a.demo_cc_only, drop_modality=a.drop_modality
                )
                for _, r in batch
            ]
            prompts = [
                tok.apply_chat_template(
                    [
                        {
                            "role": "user",
                            "content": text
                            + "\n\nBased on this presentation, what are the patient's diagnoses?",
                        }
                    ],
                    tokenize=False,
                    add_generation_prompt=False,
                )
                + "<|im_start|>assistant\n"
                for text in inputs
            ]
            enc = tok(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=3000,
                add_special_tokens=False,
            ).to("cuda")
            with torch.no_grad():
                generated = model.generate(**enc, max_new_tokens=2048, do_sample=False)
            for (idx, row), text, tokens in zip(
                batch, inputs, generated[:, enc.input_ids.shape[1] :]
            ):
                reply = tok.decode(tokens, skip_special_tokens=True)
                try:
                    pred = answer_labels(reply)
                    fmt = True
                except ValueError:
                    pred, fmt = [], False
                think = re.search(r"<think>(.*?)</think>", reply, re.S)
                rec = {
                    **identity(row),
                    "idx": idx,
                    "input": text,
                    "pred": pred,
                    "think": think.group(1).strip() if think else "",
                    "gen": reply,
                    "fmt": fmt,
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            print(
                f"shard {a.shard}: {min(start + a.batch_size, len(todo))}/{len(todo)}",
                flush=True,
            )


if __name__ == "__main__":
    main()
