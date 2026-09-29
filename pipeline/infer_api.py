"""Generate held-out predictions using direct Azure OpenAI or Anthropic APIs."""

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import llm_backend as llm
from earlydx import (
    answer_labels,
    ensure_complete,
    file_hash,
    identity,
    input_of,
    object_hash,
    read_rows,
    run_metadata,
    validate_rows,
)
from prompts import INFERENCE


async def run(argv=None, client=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--role", required=True, choices=sorted(llm.HOSTED))
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conc", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument(
        "--limit",
        type=int,
        help="Optional prefix subsample; default is the entire test file",
    )
    ap.add_argument("--window-hours", type=int, choices=(0, 6, 24), default=0)
    ap.add_argument(
        "--timestamp", choices=("charttime", "storetime"), default="charttime"
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config/schema without sending requests or writing predictions",
    )
    a = ap.parse_args(argv)
    if min(a.conc, a.max_tokens) < 1 or (a.limit is not None and a.limit < 1):
        ap.error("Concurrency, token budget and optional limit must be positive")
    if not a.out.endswith(".jsonl") or Path(a.out).resolve() == Path(a.test).resolve():
        ap.error("Use a separate .jsonl output path")
    rows = validate_rows(
        read_rows(a.test), window=a.window_hours, timestamp=a.timestamp
    )
    ensure_complete(a.test, rows)
    if not rows or any(r.get("split") != "test" for r in rows):
        raise ValueError("API prediction requires the held-out test split")
    total = len(rows)
    rows = rows[: a.limit] if a.limit else rows
    backend = llm.provenance(a.role)
    metadata = {
        "stage": "api-inference",
        "backend": backend,
        "test_sha256": file_hash(a.test),
        "prompt_sha256": object_hash(INFERENCE),
        "max_tokens": a.max_tokens,
        "temperature": None,
        "limit": a.limit,
        "expected_records": len(rows),
        "total_test_records": total,
        "window_hours": a.window_hours,
        "timestamp": a.timestamp,
    }
    if a.dry_run:
        print(
            json.dumps(
                {
                    "role": a.role,
                    "selected": len(rows),
                    "total": total,
                    "backend": backend,
                },
                indent=2,
            )
        )
        return
    # Check credentials before creating output files; their values never enter metadata.
    cfg = llm.config(a.role)
    key = cfg.get(
        "api_key_env",
        "AZURE_OPENAI_API_KEY" if cfg["provider"] == "azure" else "ANTHROPIC_API_KEY",
    )
    import os

    if not os.environ.get(key):
        raise ValueError(f"Missing API credential in environment variable {key}")
    run_metadata(a.out, metadata)
    old = (
        validate_rows(read_rows(a.out), window=a.window_hours, timestamp=a.timestamp)
        if Path(a.out).exists()
        else []
    )
    by_stay = {r["stay_id"]: r for r in rows}
    for r in old:
        ref = by_stay.get(r["stay_id"])
        if (
            not ref
            or any(
                r.get(k) != ref.get(k) for k in ("subject_id", "hadm_id", "evidence")
            )
            or r.get("input") != input_of(ref)
        ):
            raise ValueError("Saved predictions do not match this test selection")
        if not isinstance(r.get("pred"), list):
            raise ValueError("Saved prediction is malformed")
    done = {r["stay_id"] for r in old}
    todo = [(i, r) for i, r in enumerate(rows) if r["stay_id"] not in done]
    if not todo:
        print(f"{a.role}: all selected encounters already complete")
        return
    lock, sem, failed = asyncio.Lock(), asyncio.Semaphore(a.conc), []
    owned = client is None
    if owned:
        client = httpx.AsyncClient(trust_env=False, follow_redirects=False)
    try:
        with open(a.out, "a") as out:

            async def one(idx, row):
                text = input_of(row)
                prompt = INFERENCE.replace("{INPUT}", text)
                async with sem:
                    try:
                        reply = await llm.complete(
                            a.role, prompt, max_tokens=a.max_tokens, client=client
                        )
                    except llm.LLMCallError:
                        failed.append(row["stay_id"])
                        return
                # A returned refusal/format failure is a model outcome, not a transport failure.
                try:
                    pred = answer_labels(reply["text"])
                    fmt = not reply["refused"]
                except ValueError:
                    pred, fmt = [], False
                if reply["refused"]:
                    pred = []
                think = re.search(r"<think>(.*?)</think>", reply["text"], re.S)
                rec = {
                    **identity(row),
                    "idx": idx,
                    "input": text,
                    "pred": pred,
                    "fmt": fmt,
                    "gen": reply["text"],
                    "think": think.group(1).strip() if think else "",
                    "api": {k: v for k, v in reply.items() if k != "text"},
                }
                async with lock:
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out.flush()
                    done.add(row["stay_id"])

            await asyncio.gather(*(one(i, r) for i, r in todo))
    finally:
        if owned:
            await client.aclose()
    print(
        f"{a.role}: {len(done)}/{len(rows)} selected encounters completed; {len(failed)} requests failed"
    )
    if failed:
        raise RuntimeError(
            "Incomplete API predictions; rerun the same command to retry missing encounters before evaluation"
        )


if __name__ == "__main__":
    asyncio.run(run())
