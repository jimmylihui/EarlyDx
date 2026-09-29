"""Validated, resumable verifier/teacher stages; output retains encounter provenance."""

import argparse
import asyncio
import copy
import json
import re
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import llm_backend as llm
from earlydx import (
    answer_labels,
    file_hash,
    gold_of,
    input_of,
    object_hash,
    read_rows,
    run_metadata,
    validate_rows,
    verdict_map,
)
from prompts import VERIFIER, TEACHER


def parse_verdicts(text, row):
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    parsed = json.loads(text)
    result = {**row, "label_verdicts": parsed["verdicts"]}
    verdict_map(result)
    return result


def parse_rationale(text, row):
    if not re.search(r"<think>\s*\S.*?</think>", text, re.S):
        raise ValueError("Missing rationale")
    if {d.casefold() for d in answer_labels(text)} != {
        d.casefold() for d in gold_of(row)
    }:
        raise ValueError("Teacher changed the target labels")
    out = copy.deepcopy(row)
    out["messages"][-1]["content"] = text.strip()
    return out


async def run(role, argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True, help="JSONL, separate from the input")
    ap.add_argument("--conc", type=int, default=16)
    a = ap.parse_args(argv)
    if Path(a.input).resolve() == Path(a.out).resolve() or not a.out.endswith(".jsonl"):
        ap.error("Use a separate .jsonl output")
    if a.conc < 1:
        ap.error("conc must be positive")
    rows = validate_rows(read_rows(a.input))
    prompt = VERIFIER if role == "verifier" else TEACHER
    if role == "teacher":
        for r in rows:
            if not gold_of(r) or any(
                v["verdict"] == "unsupported" for v in verdict_map(r).values()
            ):
                raise ValueError(
                    "Teacher input must contain only supported/partial labels; run prepare_dataset first"
                )
    run_metadata(
        a.out,
        {
            "stage": role,
            "input_sha256": file_hash(a.input),
            "expected_records": len(rows),
            "backend": llm.provenance(role),
            "prompt_sha256": object_hash(prompt),
            "temperature": 0 if role == "verifier" else 0.3,
            "max_tokens": 4000 if role == "verifier" else 2048,
        },
    )
    previous = validate_rows(read_rows(a.out)) if Path(a.out).exists() else []
    known = {r["stay_id"] for r in rows}
    done = {r["stay_id"] for r in previous}
    if done - known:
        raise ValueError("Output contains encounters absent from input")
    for r in previous:
        verdict_map(r) if role == "verifier" else parse_rationale(
            r["messages"][-1]["content"], r
        )
    sem, lock = asyncio.Semaphore(a.conc), asyncio.Lock()
    failed = []
    with open(a.out, "a") as fout:

        async def one(client, row):
            if row["stay_id"] in done:
                return
            labels = gold_of(row)
            p = (
                prompt.replace("{INPUT}", input_of(row))
                .replace("{DXS}", json.dumps(labels))
                .replace("{DX}", "; ".join(labels))
            )
            result = None
            async with sem:
                if role == "verifier" and not labels:
                    result = {**row, "label_verdicts": []}
                else:
                    for _ in range(3):
                        try:
                            text = await llm.chat(
                                role,
                                p,
                                max_tokens=4000 if role == "verifier" else 2048,
                                temperature=0 if role == "verifier" else 0.3,
                                client=client,
                            )
                            result = (
                                parse_verdicts(text, row)
                                if role == "verifier"
                                else parse_rationale(text, row)
                            )
                            break
                        except (llm.LLMCallError, ValueError, KeyError):
                            continue
            if result is None:
                failed.append(row["stay_id"])
                return
            async with lock:
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                fout.flush()
                done.add(row["stay_id"])
                if len(done) % 100 == 0:
                    print(f"{len(done)}/{len(rows)} completed", flush=True)

        async with httpx.AsyncClient() as client:
            await asyncio.gather(*(one(client, row) for row in rows))
    print(f"{role}: {len(done)}/{len(rows)} completed; {len(failed)} failed")
    if failed:
        raise RuntimeError(
            "Incomplete annotations. Rerun the same command to retry; do not continue downstream."
        )
