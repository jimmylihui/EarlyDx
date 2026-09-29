"""Section 3.1: index every ED stay linked to a hospital admission."""

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from earlydx import file_hash, run_metadata


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out", default="data/cohort_index.json")
    ap.add_argument("--hosp-version", default="3.1")
    a = ap.parse_args()
    import pandas as pd

    adm_path = a.data_root / f"mimic-iv/{a.hosp_version}/hosp/admissions.csv.gz"
    matches = glob.glob(str(a.data_root / "mimic-iv-ed/*/ed/edstays.csv.gz"))
    if len(matches) != 1:
        raise ValueError("Keep exactly one MIMIC-IV-ED version under data-root")
    adm = pd.read_csv(adm_path, usecols=["hadm_id"])
    ed = pd.read_csv(matches[0], usecols=["subject_id", "hadm_id", "stay_id"])
    ed = ed[ed.hadm_id.notna() & ed.hadm_id.isin(adm.hadm_id)].copy()
    if ed.stay_id.duplicated().any():
        raise ValueError("ED source contains duplicate stay_id values")
    ed = ed.sort_values("stay_id")
    idx = [[int(r.subject_id), int(r.hadm_id), int(r.stay_id)] for r in ed.itertuples()]
    run_metadata(
        a.out,
        {
            "stage": "cohort",
            "admissions_sha256": file_hash(adm_path),
            "edstays_sha256": file_hash(matches[0]),
            "n": len(idx),
        },
    )
    Path(a.out).write_text(json.dumps(idx) + "\n")
    print(f"{len(idx)} admitted ED encounters -> {a.out}")


if __name__ == "__main__":
    main()
