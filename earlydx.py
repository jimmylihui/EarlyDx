"""Shared data contract and reproducibility helpers (no model dependencies)."""

import csv
import hashlib
import json
import re
from pathlib import Path

PAPER = "https://arxiv.org/abs/2607.28788v1"
VERDICTS = {"supported", "partial", "unsupported"}
QUESTION = "\n\nBased on this presentation, what are the patient's diagnoses?"
VAGUE = [
    r"\bcirculatory disease",
    r"disease of (the )?[\w ]+?(tract|system|organ)s?,? unspecified",
    r"\bunspecified disease\b",
    r"ill[- ]defined",
    r"other and unspecified disorders? of",
    r"disorder of [\w ]+?(system|tract), unspecified",
    r"\b(condition|dis|disorder|disease)s?,? (nec|nos)\b",
]


def keep_label(code, version, title):
    """Section 3.3: administrative/symptom code chapters and overly generic titles.

    Do not discard a disease just because its title contains 'pain', 'cough', etc.
    """
    code = str(code).strip().upper()
    if not code or not str(title).strip():
        return False
    if int(version) == 10:
        if code[0] in "RVWXYZ":
            return False
    elif int(version) == 9:
        if code.startswith(("E", "V")):
            return False
        try:
            if 780 <= int(code[:3]) <= 799:
                return False
        except ValueError:
            raise ValueError(f"Invalid ICD-9 code: {code}")
    else:
        raise ValueError(f"Unknown ICD version: {version}")
    return not any(re.search(p, str(title).lower()) for p in VAGUE)


def unique_labels(labels):
    out, seen = [], set()
    for label in labels:
        if not isinstance(label, str):
            raise ValueError("A diagnosis must be a string")
        label = label.strip()
        if label and label.casefold() not in seen:
            out.append(label)
            seen.add(label.casefold())
    return out


def answer_labels(text):
    match = re.search(r"<answer>(.*?)</answer>", text, re.S)
    if not match:
        raise ValueError("Missing <answer>...</answer> block")
    return unique_labels(match.group(1).split(";"))


def gold_of(row):
    return (
        unique_labels(row["gold"])
        if "gold" in row
        else answer_labels(row["messages"][-1]["content"])
    )


def input_of(row):
    text = row["messages"][0]["content"] if "messages" in row else row["input"]
    return text.split(QUESTION)[0]


def read_rows(path):
    path = Path(path)
    with path.open() as f:
        return (
            json.load(f)
            if path.suffix == ".json"
            else [json.loads(l) for l in f if l.strip()]
        )


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        if path.suffix == ".json":
            json.dump(rows, f, ensure_ascii=False, indent=2)
        else:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def object_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def run_metadata(output, metadata):
    """Refuse to append/resume using different input, prompt, model or settings."""
    path = Path(str(output) + ".meta.json")
    root = Path(__file__).resolve().parent
    code_files = [
        root / "earlydx.py",
        root / "prompts.py",
        root / "llm_backend.py",
    ] + sorted((root / "pipeline").glob("*.py"))
    code_files.append(root / "pipeline" / "zero2_offload.json")
    implementation = object_hash(
        {str(p.relative_to(root)): file_hash(p) for p in code_files}
    )
    metadata = {
        "schema": 1,
        "paper": PAPER,
        "implementation_sha256": implementation,
        **metadata,
    }
    if path.exists():
        if json.loads(path.read_text()) != metadata:
            raise ValueError(
                f"Run configuration changed: {path}. Use a new output path."
            )
    elif Path(output).exists():
        raise ValueError(
            f"Existing output has no provenance: {output}. Use a new output path."
        )
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def identity(row):
    return {
        k: row[k]
        for k in (
            "subject_id",
            "hadm_id",
            "stay_id",
            "evidence",
            "reference_evidence",
            "split",
        )
        if k in row
    }


def validate_rows(rows, *, window=0, timestamp="charttime", require_evidence=True):
    seen = set()
    for row in rows:
        if not all(k in row for k in ("subject_id", "hadm_id", "stay_id")):
            raise ValueError("Records must carry subject_id, hadm_id and stay_id")
        if row["stay_id"] in seen:
            raise ValueError(f"Duplicate stay_id: {row['stay_id']}")
        seen.add(row["stay_id"])
        ev = row.get("evidence")
        if require_evidence and (
            not ev
            or ev.get("window_hours") != window
            or ev.get("timestamp") != timestamp
        ):
            raise ValueError(
                f"stay {row['stay_id']}: evidence metadata does not match W={window}, {timestamp}"
            )
    return rows


def verdict_map(row):
    result = {}
    for v in row.get("label_verdicts", row.get("kept_verdicts", [])):
        label = v["dx"].strip().casefold()
        if label in result or v.get("verdict") not in VERDICTS:
            raise ValueError(f"Duplicate label or invalid evidence class: {label}")
        result[label] = v
    if set(result) != {d.casefold() for d in gold_of(row)}:
        raise ValueError(
            f"Evidence audit must classify each reference label exactly once: stay {row.get('stay_id')}"
        )
    return result


def read_split(path):
    """Credentialed CSV with subject_id,split; repeated subjects must agree."""
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            sid, split = int(r["subject_id"]), r["split"].strip().lower()
            if split not in {"train", "test"} or (sid in out and out[sid] != split):
                raise ValueError(f"Invalid/conflicting split for patient {sid}")
            out[sid] = split
    return out


def check_disjoint(train, test):
    overlap = {r["subject_id"] for r in train} & {r["subject_id"] for r in test}
    if overlap:
        raise ValueError(
            f"Patient leakage: {len(overlap)} patients occur in train and test"
        )


def transform_input(text, *, demo_cc=False, drop_modality=None):
    """Inference-only controls, keeping complete multi-line sections together."""
    prefixes = {
        "radiology": ("Radiology (",),
        "labs": ("Initial labs (", "Baseline labs ("),
        "history": ("Past medical history", "Past ED diagnoses:"),
        "ecg": ("ECG:",),
        "echo": ("Echocardiogram:",),
        "medications": ("Home meds:",),
        "vitals": ("Triage:", "ED serial vitals:"),
    }
    if drop_modality and drop_modality not in prefixes:
        raise ValueError(f"Unknown modality {drop_modality}")
    keep, output = True, []
    for line in text.split(QUESTION)[0].splitlines():
        if line and not line[0].isspace():
            keep = (
                line.startswith(("Demographics:", "Chief complaint:"))
                if demo_cc
                else not (drop_modality and line.startswith(prefixes[drop_modality]))
            )
        if keep:
            output.append(line)
    return "\n".join(output)


def ensure_complete(path, rows):
    meta = Path(str(path) + ".meta.json")
    if meta.exists():
        expected = json.loads(meta.read_text()).get("expected_records")
        if expected is not None and len(rows) != expected:
            raise ValueError(
                f"{path}: incomplete stage output ({len(rows)}/{expected}); resume that stage first"
            )
