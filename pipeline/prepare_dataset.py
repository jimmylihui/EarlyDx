"""Keep supported/partial labels and apply a fixed, patient-disjoint split (§4.1)."""

import argparse
import copy
import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from earlydx import (
    check_disjoint,
    file_hash,
    ensure_complete,
    gold_of,
    read_rows,
    read_split,
    run_metadata,
    validate_rows,
    verdict_map,
    write_rows,
)


def prepare(rows, splits):
    train, test, clean = [], [], []
    for raw in sorted(rows, key=lambda r: r["stay_id"]):
        vmap = verdict_map(raw)
        labels = [
            d
            for d in gold_of(raw)
            if vmap[d.casefold()]["verdict"] in {"supported", "partial"}
        ]
        if not labels:
            continue
        r = copy.deepcopy(raw)
        if r["subject_id"] not in splits:
            raise ValueError(f"Patient {r['subject_id']} missing from fixed split")
        r["split"] = splits[r["subject_id"]]
        r["kept_verdicts"] = [vmap[d.casefold()] for d in labels]
        r.pop("label_verdicts", None)
        r["messages"][-1]["content"] = "<answer>" + "; ".join(labels) + "</answer>"
        clean.append(r)
        (train if r["split"] == "train" else test).append(r)
    check_disjoint(train, test)
    if not train or not test:
        raise ValueError("Both train and test must contain retained encounters")
    return clean, train, test


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-dir", default="data")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--split-file", help="Published credentialed CSV: subject_id,split"
    )
    group.add_argument(
        "--new-split",
        action="store_true",
        help="New experimental split; NOT the paper split",
    )
    ap.add_argument("--test-fraction", type=float, default=0.06)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    rows = validate_rows(read_rows(a.input))
    ensure_complete(a.input, rows)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if a.split_file:
        splits = read_split(a.split_file)
        split_info = {"split_sha256": file_hash(a.split_file), "split_kind": "provided"}
    else:
        if not 0 < a.test_fraction < 1:
            ap.error("test-fraction must lie strictly between 0 and 1")
        retained = [
            r
            for r in rows
            if any(
                v["verdict"] in {"supported", "partial"}
                for v in verdict_map(r).values()
            )
        ]
        subjects = sorted({r["subject_id"] for r in retained})
        random.Random(a.seed).shuffle(subjects)
        n = max(1, round(len(subjects) * a.test_fraction))
        splits = {sid: ("test" if i < n else "train") for i, sid in enumerate(subjects)}
        split_info = {
            "split_kind": "new-experiment",
            "seed": a.seed,
            "test_fraction": a.test_fraction,
        }
    clean, train, test = prepare(rows, splits)
    counts = dict(
        cohort=len(rows), retained=len(clean), train=len(train), test=len(test)
    )
    for name, data in [
        ("cohort_clean", clean),
        ("sft_train_direct", train),
        ("sft_test", test),
    ]:
        dst = out / f"{name}.jsonl"
        run_metadata(
            dst, {"stage": "prepare", "input_sha256": file_hash(a.input), **split_info}
        )
        write_rows(dst, data)
    if a.new_split:
        with (out / "generated_split.csv").open("w") as f:
            w = csv.writer(f)
            w.writerow(["subject_id", "split"])
            w.writerows(sorted(splits.items()))
    (out / "counts.json").write_text(
        json.dumps({**counts, **split_info}, indent=2) + "\n"
    )
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
