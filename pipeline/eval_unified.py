"""Section 3.5: one-to-one semantic matching, three evidence tracks, micro/example metrics.

Only predictions are read from each system; reference labels and evidence classes come from
independent test/verdict files. One judge and prompt are used for every system. The cache is
bound to the exact model revision and prompt. Transport failures stop evaluation.
"""

import argparse
import asyncio
import glob
import json
import sys
from pathlib import Path

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import llm_backend as llm
from earlydx import (
    file_hash,
    ensure_complete,
    gold_of,
    input_of,
    object_hash,
    read_rows,
    run_metadata,
    unique_labels,
    validate_rows,
    verdict_map,
    write_rows,
)
from prompts import JUDGE

BLOCKS = ("supported", "partial", "secondary")


def cache_key(gold, pred):
    return object_hash([sorted(gold), sorted(pred)])


def prf(tp, g, p):
    precision, recall = tp / p if p else 0.0, tp / g if g else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * tp / (g + p) if g + p else 0.0,
        "jaccard": tp / (g + p - tp) if g + p - tp else 0.0,
    }


def summarize(triples):
    values = list(triples)
    totals = [sum(v[i] for v in values) for i in range(3)]
    result = {
        "n_encounters": len(values),
        "matched": totals[0],
        "gold_labels": totals[1],
        "predictions": totals[2],
        **prf(*totals),
    }
    result["example"] = {
        k: float(np.mean([prf(*v)[k] for v in values])) if values else 0.0
        for k in ("precision", "recall", "f1", "jaccard")
    }
    result["complete_match_rate"] = (
        sum(tp == g == p for tp, g, p in values) / len(values) if values else 0.0
    )
    return result


def references(test, verdict_rows):
    by_stay = {r["stay_id"]: r for r in verdict_rows}
    if len(by_stay) != len(verdict_rows):
        raise ValueError("Duplicate encounters in verifier output")
    gold = {}
    for r in test:
        v = by_stay.get(r["stay_id"])
        if not v:
            raise ValueError(f"Missing verifier record for stay {r['stay_id']}")
        if any(v.get(k) != r.get(k) for k in ("subject_id", "hadm_id")) or v.get(
            "evidence"
        ) != r.get("reference_evidence", r["evidence"]):
            raise ValueError(
                "Verifier record and reference record have different provenance"
            )
        classes = verdict_map(v)
        labels = gold_of(r)
        if any(d.casefold() not in classes for d in labels):
            raise ValueError("Every test reference label needs an evidence class")
        s = [d for d in labels if classes[d.casefold()]["verdict"] == "supported"]
        p = [d for d in labels if classes[d.casefold()]["verdict"] == "partial"]
        gold[r["stay_id"]] = {"supported": s, "partial": p, "secondary": s + p}
    return gold


def load_system(system, base, test, window, timestamp, budget=None):
    files = []
    for pattern in system["files"]:
        path = str(Path(base) / pattern)
        hits = sorted(glob.glob(path))
        if not hits:
            raise FileNotFoundError(f"{system['name']}: no files match {path}")
        files.extend(hits)
    rows = []
    for f in files:
        data = read_rows(f)
        ensure_complete(f, data)
        rows.extend(data)
    validate_rows(rows, window=window, timestamp=timestamp)
    if not rows:
        raise ValueError("System has no prediction records; no score can be computed")
    by_stay = {r["stay_id"]: r for r in test}
    result = {}
    for r in rows:
        if r["stay_id"] not in by_stay:
            raise ValueError("Prediction does not belong to the held-out test file")
        reference = by_stay[r["stay_id"]]
        if any(r[k] != reference[k] for k in ("subject_id", "hadm_id", "evidence")):
            raise ValueError("Prediction and test record have different provenance")
        if system.get("input_variant", "full") == "full" and input_of(r) != input_of(
            reference
        ):
            raise ValueError(
                "Full-input prediction used a different prompt; label input variants explicitly"
            )
        if not isinstance(r.get("pred"), list):
            raise ValueError(
                "Prediction rows must include a list named pred (empty is allowed)"
            )
        pred = unique_labels(r["pred"])
        result[r["stay_id"]] = pred[:budget] if budget else pred
    return result, {
        "files": {str(f): file_hash(f) for f in files},
        "n_predictions": len(rows),
        "format_failures": sum(r.get("fmt") is False for r in rows),
    }


class Judge:
    def __init__(self, path, backend, cache_only=False, prompt=JUDGE):
        self.path, self.cache_only, self.prompt = str(path), cache_only, prompt
        meta = {
            "stage": "judge-cache",
            "backend": backend,
            "prompt_sha256": object_hash(prompt),
            "temperature": 0,
            "max_tokens": 120,
        }
        if cache_only and not Path(str(path) + ".meta.json").exists():
            raise ValueError("Cache-only requires a cache with model/prompt provenance")
        run_metadata(path, meta)
        self.cache = {}
        for r in read_rows(path) if Path(path).exists() else []:
            if set(r) != {"k", "m"} or type(r["m"]) is not int or r["m"] < 0:
                raise ValueError("Invalid judge cache entry")
            if r["k"] in self.cache and self.cache[r["k"]] != r["m"]:
                raise ValueError("Conflicting cached judgments")
            self.cache[r["k"]] = r["m"]

    def lookup(self, gold, pred):
        if not gold or not pred:
            return 0
        result = self.cache.get(cache_key(gold, pred))
        if result is not None and result > min(len(gold), len(pred)):
            raise ValueError("Cached matched count exceeds one-to-one matching bound")
        return result

    async def fill(self, pairs, concurrency):
        if self.cache_only and pairs:
            raise RuntimeError(
                f"{len(pairs)} judgments missing from cache; no partial score produced"
            )
        sem, lock = asyncio.Semaphore(concurrency), asyncio.Lock()
        with open(self.path, "a") as out:

            async def one(client, gold, pred):
                prompt = self.prompt.replace("{G}", json.dumps(gold)).replace(
                    "{P}", json.dumps(pred)
                )
                async with sem:
                    for _ in range(3):
                        reply = await llm.chat(
                            "judge",
                            prompt,
                            max_tokens=120,
                            temperature=0,
                            client=client,
                        )
                        try:
                            count = json.loads(reply)["matched_pairs"]
                            if type(count) is not int or not 0 <= count <= min(
                                len(gold), len(pred)
                            ):
                                continue
                        except (ValueError, KeyError, TypeError):
                            continue
                        async with lock:
                            key = cache_key(gold, pred)
                            self.cache[key] = count
                            out.write(json.dumps({"k": key, "m": count}) + "\n")
                            out.flush()
                        return
                raise llm.LLMCallError(
                    "Judge did not return a valid one-to-one matched count"
                )

            async with httpx.AsyncClient() as client:
                await asyncio.gather(*(one(client, g, p) for g, p in pairs))


def bootstrap(values, subjects, repeats, seed=2026, reference=None):
    """Resample patients, keeping every encounter of each sampled patient together."""
    if not values:
        return None
    keys = (
        sorted(set(values) & set(reference))
        if reference is not None
        else sorted(values)
    )
    if not keys:
        return None
    patients = sorted({subjects[k] for k in keys})
    index = {p: i for i, p in enumerate(patients)}
    a, b = np.zeros((len(patients), 3)), np.zeros((len(patients), 3))
    for k in keys:
        a[index[subjects[k]]] += values[k]
        if reference is not None:
            b[index[subjects[k]]] += reference[k]
    rng, scores = np.random.default_rng(seed), []
    for _ in range(repeats):
        sample = rng.integers(0, len(patients), size=len(patients))
        score = prf(*a[sample].sum(axis=0))["f1"]
        if reference is not None:
            score -= prf(*b[sample].sum(axis=0))["f1"]
        scores.append(score)
    result = {
        "n_patients": len(patients),
        "n_encounters": len(keys),
        "ci95": [float(x) for x in np.percentile(scores, [2.5, 97.5])],
    }
    if reference is not None:
        arr = np.array(scores)
        result["p_two_sided"] = min(
            1.0,
            2 * (1 + min(int((arr <= 0).sum()), int((arr >= 0).sum()))) / (repeats + 1),
        )
        result["min_detectable_p"] = 2 / (repeats + 1)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--verdicts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--window-hours", type=int, choices=(0, 6, 24), default=0)
    ap.add_argument(
        "--timestamp", choices=("charttime", "storetime"), default="charttime"
    )
    ap.add_argument("--cache-file")
    ap.add_argument("--cache-only", action="store_true")
    ap.add_argument(
        "--common",
        action="store_true",
        help="Restrict to encounters predicted by every system",
    )
    ap.add_argument(
        "--prediction-budget",
        type=int,
        help="Score the first k distinct predictions (Appendix A)",
    )
    ap.add_argument("--bootstrap", type=int, default=0)
    ap.add_argument(
        "--ref", help="System name for paired patient bootstrap comparisons"
    )
    ap.add_argument("--conc", type=int, default=16)
    a = ap.parse_args()
    if (
        a.conc < 1
        or a.bootstrap < 0
        or (a.prediction_budget is not None and a.prediction_budget < 1)
    ):
        ap.error("Invalid concurrency, bootstrap count or prediction budget")
    test = validate_rows(
        read_rows(a.test), window=a.window_hours, timestamp=a.timestamp
    )
    if not test or any(r.get("split") != "test" for r in test):
        raise ValueError("Evaluation requires a nonempty held-out test split")
    verdicts = read_rows(a.verdicts)
    gold = references(test, verdicts)
    systems = json.loads(Path(a.manifest).read_text())["systems"]
    names = [s["name"] for s in systems]
    if not names or len(set(names)) != len(names):
        raise ValueError("Manifest needs at least one system and unique system names")
    if a.ref and a.ref not in names:
        raise ValueError("Reference system is absent from manifest")
    preds, stats = {}, {}
    for s in systems:
        preds[s["name"]], stats[s["name"]] = load_system(
            s,
            Path(a.manifest).resolve().parent,
            test,
            a.window_hours,
            a.timestamp,
            a.prediction_budget,
        )
    scope = {n: set(preds[n]) for n in names}
    if a.common:
        shared = set.intersection(*(scope[n] for n in names))
        if not shared:
            raise ValueError("Systems have no shared test encounters")
        scope = {n: shared for n in names}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    backend = llm.provenance("judge")
    cache = a.cache_file or str(
        out / f"judge_cache.{object_hash([backend, JUDGE])[:16]}.jsonl"
    )
    judge = Judge(cache, backend, a.cache_only)
    needed = {}
    for name in names:
        for stay in scope[name]:
            for block in BLOCKS:
                g, p = gold[stay][block], preds[name][stay]
                if judge.lookup(g, p) is None:
                    needed[cache_key(g, p)] = (g, p)
    if needed:
        asyncio.run(judge.fill(list(needed.values()), a.conc))
    results, details, per_system = [], [], {}
    subjects = {r["stay_id"]: r["subject_id"] for r in test}
    for s in systems:
        name = s["name"]
        per_system[name] = {}
        result = {
            "name": name,
            "group": s.get("group", ""),
            "input_variant": s.get("input_variant", "full"),
            "n_encounters": len(scope[name]),
            "n_test": len(test),
            "subsample": len(scope[name]) < len(test),
            "empty_predictions": sum(not preds[name][h] for h in scope[name]),
            "average_diagnoses": float(
                np.mean([len(preds[name][h]) for h in scope[name]])
            )
            if scope[name]
            else 0,
            "blocks": {},
            **stats[name],
        }
        for block in BLOCKS:
            triples = {}
            for stay in sorted(scope[name]):
                g, p = gold[stay][block], preds[name][stay]
                if not g:
                    continue
                count = judge.lookup(g, p)
                triples[stay] = (count, len(g), len(p))
                details.append(
                    {
                        "system": name,
                        "stay_id": stay,
                        "subject_id": subjects[stay],
                        "block": block,
                        "matched": count,
                        "gold_labels": len(g),
                        "predictions": len(p),
                        **prf(count, len(g), len(p)),
                    }
                )
            result["blocks"][block] = summarize(triples.values())
            if block == "supported":
                per_system[name] = triples
        results.append(result)
    if a.bootstrap:
        for r in results:
            r["bootstrap_supported"] = bootstrap(
                per_system[r["name"]], subjects, a.bootstrap
            )
            if a.ref and a.ref != r["name"]:
                r["paired_vs_reference"] = {
                    "reference": a.ref,
                    "statistics": bootstrap(
                        per_system[r["name"]],
                        subjects,
                        a.bootstrap,
                        reference=per_system[a.ref],
                    ),
                }
    meta = {
        "test_sha256": file_hash(a.test),
        "verdicts_sha256": file_hash(a.verdicts),
        "manifest_sha256": file_hash(a.manifest),
        "judge": backend,
        "prompt_sha256": object_hash(JUDGE),
        "window_hours": a.window_hours,
        "timestamp": a.timestamp,
        "common": a.common,
        "prediction_budget": a.prediction_budget,
        "bootstrap": a.bootstrap,
    }
    run_metadata(
        out / "results.json", {"stage": "evaluation", **meta, "prediction_files": stats}
    )
    (out / "results.json").write_text(
        json.dumps({"meta": meta, "systems": results}, indent=2) + "\n"
    )
    write_rows(out / "per_encounter.jsonl", details)
    lines = [
        "| System | n | Track | Precision | Recall | F1 |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for r in results:
        for b in BLOCKS:
            v = r["blocks"][b]
            label = r["name"] + ("†" if r["subsample"] else "")
            lines.append(
                f"| {label} | {v['n_encounters']} | {b} | {v['precision']:.4f} | {v['recall']:.4f} | {v['f1']:.4f} |"
            )
    lines += [
        "",
        "† Predictions cover only a subset of test encounters. Counts and all metrics are computed from this run.",
    ]
    (out / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
