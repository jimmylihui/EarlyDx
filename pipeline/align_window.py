"""Join rebuilt sensitivity inputs to the original held-out labels without re-auditing."""

import argparse
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from earlydx import file_hash, read_rows, run_metadata, validate_rows, write_rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--inputs",
        required=True,
        help="Rebuilt cohort for the alternate window/timestamp",
    )
    ap.add_argument(
        "--reference-test", required=True, help="Original W=0 held-out test file"
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--window-hours", type=int, choices=(0, 6, 24), default=0)
    ap.add_argument(
        "--timestamp", choices=("charttime", "storetime"), default="charttime"
    )
    a = ap.parse_args()
    inputs = validate_rows(
        read_rows(a.inputs), window=a.window_hours, timestamp=a.timestamp
    )
    refs = validate_rows(read_rows(a.reference_test))
    by_stay = {r["stay_id"]: r for r in inputs}
    out = []
    for ref in refs:
        if ref.get("split") != "test":
            raise ValueError("Reference must be the held-out test split")
        new = by_stay.get(ref["stay_id"])
        if new is None or any(new[k] != ref[k] for k in ("subject_id", "hadm_id")):
            raise ValueError(
                "Rebuilt inputs must include every reference encounter with the same identity"
            )
        row = copy.deepcopy(ref)
        row["messages"][0] = copy.deepcopy(new["messages"][0])
        row["reference_evidence"] = copy.deepcopy(ref["evidence"])
        row["evidence"] = copy.deepcopy(new["evidence"])
        out.append(row)
    run_metadata(
        a.out,
        {
            "stage": "align-window",
            "inputs_sha256": file_hash(a.inputs),
            "reference_sha256": file_hash(a.reference_test),
        },
    )
    write_rows(a.out, out)


if __name__ == "__main__":
    main()
