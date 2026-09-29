"""Synthetic contract/regression tests. No MIMIC records or paper result fixtures."""

import asyncio
import copy
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import earlydx as dx
import llm_backend as llm
from pipeline.annotate import parse_rationale, parse_verdicts
from pipeline.eval_unified import (
    Judge,
    bootstrap,
    cache_key,
    load_system,
    references,
    summarize,
)
from pipeline.infer_full import select_shard
from pipeline.prepare_dataset import prepare
from pipeline.sft import encode_example


def record(stay=1, subject=1, labels=("Condition A", "Condition B")):
    return {
        "stay_id": stay,
        "hadm_id": 10 + subject,
        "subject_id": subject,
        "evidence": {
            "window_hours": 0,
            "timestamp": "charttime",
            "admittime": "2020-01-01 12:00:00",
        },
        "messages": [
            {
                "role": "user",
                "content": "Demographics: synthetic\nChief complaint: synthetic"
                + dx.QUESTION,
            },
            {
                "role": "assistant",
                "content": "<answer>" + "; ".join(labels) + "</answer>",
            },
        ],
        "label_verdicts": [
            {"dx": label, "verdict": "supported" if i == 0 else "partial"}
            for i, label in enumerate(labels)
        ],
    }


def test_labels_and_audit_validation():
    assert not dx.keep_label("R05", 10, "Cough")
    assert not dx.keep_label("78650", 9, "Chest pain")
    assert not dx.keep_label("Z00", 10, "Examination")
    assert dx.keep_label("I200", 10, "Unstable angina with chest pain")
    assert not dx.keep_label("I99", 10, "Unspecified disease")
    r = record()
    with pytest.raises(ValueError, match="each reference label"):
        parse_verdicts('{"verdicts":[]}', r)
    with pytest.raises(ValueError, match="changed the target"):
        parse_rationale(
            "<think>synthetic reason</think><answer>Other condition</answer>", r
        )
    assert dx.answer_labels("<answer>A; a; B</answer>") == ["A", "B"]


def test_filter_and_subject_split():
    rows = [record(1, 1), record(2, 1), record(3, 2)]
    rows[0]["label_verdicts"][1]["verdict"] = "unsupported"
    clean, train, test = prepare(rows, {1: "train", 2: "test"})
    assert {r["stay_id"] for r in train} == {1, 2}
    assert {r["stay_id"] for r in test} == {3}
    assert dx.gold_of(clean[0]) == ["Condition A"]
    dx.check_disjoint(train, test)
    with pytest.raises(ValueError, match="Patient leakage"):
        dx.check_disjoint(train, train)
    with pytest.raises(ValueError, match="missing from fixed split"):
        prepare(rows, {1: "train"})


def test_provenance_and_incomplete_output(tmp_path):
    out = tmp_path / "rows.jsonl"
    dx.run_metadata(out, {"expected_records": 2})
    dx.write_rows(out, [record()])
    with pytest.raises(ValueError, match="incomplete stage"):
        dx.ensure_complete(out, dx.read_rows(out))
    with pytest.raises(ValueError, match="configuration changed"):
        dx.run_metadata(out, {"expected_records": 3})
    r = record()
    r["evidence"]["window_hours"] = 6
    with pytest.raises(ValueError, match="evidence metadata"):
        dx.validate_rows([r])


def test_masks_and_global_shards():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return "prompt<|im_start|>assistant\n" + messages[-1]["content"]

        def __call__(self, text, **kwargs):
            return {"input_ids": [ord(c) for c in text]}

    encoded = encode_example(Tokenizer(), record(), 3072)
    n = len("prompt<|im_start|>assistant\n")
    assert encoded["labels"][:n] == [-100] * n
    assert encoded["labels"][n:] == encoded["input_ids"][n:]
    with pytest.raises(ValueError, match="no supervised tokens"):
        encode_example(Tokenizer(), record(), n)
    rows = [record(i, i) for i in range(7)]
    a, b = select_shard(rows, 0, 2), select_shard(rows, 1, 2)
    assert [i for i, _ in a] == [0, 2, 4, 6]
    assert sorted(i for i, _ in a + b) == list(range(7))


def test_ablation_removes_continuation_lines():
    text = "Demographics: A\nChief complaint: B\nRadiology (2): report\n  second report\nInitial labs (1): value"
    assert "second report" not in dx.transform_input(text, drop_modality="radiology")
    assert (
        dx.transform_input(text, demo_cc=True) == "Demographics: A\nChief complaint: B"
    )


def test_encounter_mapping_does_not_use_prompt_prefix(tmp_path):
    r1, r2 = record(1, 1), record(2, 1)
    # Both stays deliberately have the same admission and the same prompt.
    for r in (r1, r2):
        r["split"] = "test"
    predictions = [
        {**dx.identity(r), "input": dx.input_of(r), "pred": ["Condition A"]}
        for r in (r1, r2)
    ]
    dx.write_rows(tmp_path / "p.jsonl", predictions)
    pred, stats = load_system(
        {"name": "synthetic", "files": ["p.jsonl"]}, tmp_path, [r1, r2], 0, "charttime"
    )
    assert set(pred) == {1, 2}
    dx.write_rows(tmp_path / "p.jsonl", predictions + predictions[:1])
    with pytest.raises(ValueError, match="Duplicate stay"):
        load_system(
            {"name": "synthetic", "files": ["p.jsonl"]},
            tmp_path,
            [r1, r2],
            0,
            "charttime",
        )


def test_reference_classes_and_sensitivity():
    r = record()
    r["split"] = "test"
    gold = references([r], [r])
    assert gold[1]["supported"] == ["Condition A"]
    assert gold[1]["partial"] == ["Condition B"]
    assert len(gold[1]["secondary"]) == 2
    new = copy.deepcopy(r)
    new["reference_evidence"] = copy.deepcopy(r["evidence"])
    new["evidence"]["window_hours"] = 6
    assert references([new], [r]) == gold
    changed = copy.deepcopy(r)
    changed["subject_id"] = 99
    with pytest.raises(ValueError, match="different provenance"):
        references([changed], [r])


def test_metrics_and_patient_bootstrap():
    result = summarize([(1, 1, 2), (0, 1, 0)])
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5
    assert result["f1"] == 0.5
    assert result["example"]["precision"] == 0.25
    assert result["example"]["f1"] == pytest.approx(1 / 3)
    # All encounters belong to one patient; they must always resample together.
    ci = bootstrap({1: (1, 1, 2), 2: (0, 1, 0)}, {1: 7, 2: 7}, 50)
    assert ci["ci95"] == [0.5, 0.5]
    assert bootstrap({1: (1, 1, 1)}, {1: 1, 2: 2}, 20, reference={2: (1, 1, 1)}) is None


def test_judge_cache_identity_bounds_and_failure(tmp_path, monkeypatch):
    cache = tmp_path / "cache.jsonl"
    backend = {"model": "synthetic-test-judge", "revision": "test-only"}
    judge = Judge(cache, backend)

    async def ok(*args, **kwargs):
        return '{"matched_pairs":1}'

    monkeypatch.setattr(llm, "chat", ok)
    asyncio.run(judge.fill([(["A"], ["A"])], 1))
    assert Judge(cache, backend, cache_only=True).lookup(["A"], ["A"]) == 1
    with pytest.raises(ValueError, match="configuration changed"):
        Judge(cache, {**backend, "revision": "other"})
    with pytest.raises(RuntimeError, match="missing from cache"):
        asyncio.run(Judge(cache, backend, cache_only=True).fill([(["B"], ["B"])], 1))

    async def bad(*args, **kwargs):
        return '{"matched_pairs":9}'

    monkeypatch.setattr(llm, "chat", bad)
    with pytest.raises(llm.LLMCallError):
        asyncio.run(judge.fill([(["B"], ["B"])], 1))
    assert judge.lookup(["B"], ["B"]) is None


def test_backend_rejects_wrong_role_or_placeholder():
    config = {
        "provider": "local",
        "base_url": "http://127.0.0.1:8000/v1",
        "model": "Qwen3.5-27B",
        "checkpoint": "synthetic-test",
        "revision": "test-only",
    }
    llm.check("judge", config)
    with pytest.raises(ValueError, match="paper role"):
        llm.check("verifier", config)
    with pytest.raises(ValueError, match="actual revision"):
        llm.check("judge", {**config, "revision": "<fill>"})
    with pytest.raises(llm.ComplianceError):
        llm.check("judge", {**config, "base_url": "https://third-party.invalid/v1"})


def synthetic_modules(root):
    def table(path, rows, columns=None):
        dest = root / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows, columns=columns).to_csv(dest, index=False)

    hosp = "mimic-iv/3.1/hosp/"
    ed = "mimic-iv-ed/2.2/ed/"
    note = "mimic-iv-note/2.2/note/"
    table(
        hosp + "admissions.csv.gz",
        [
            dict(
                subject_id=1,
                hadm_id=101,
                admittime="2020-01-01 12:00:00",
                race="synthetic",
                admission_location="ED",
            )
        ],
    )
    table(
        hosp + "patients.csv.gz",
        [dict(subject_id=1, gender="F", anchor_age=40, anchor_year=2019)],
    )
    table(
        ed + "edstays.csv.gz",
        [
            dict(
                subject_id=1,
                hadm_id=101,
                stay_id=1001,
                intime="2020-01-01 08:00:00",
                arrival_transport="WALK IN",
            )
        ],
    )
    table(
        ed + "diagnosis.csv.gz",
        [
            dict(
                subject_id=1,
                stay_id=1001,
                icd_code="I200",
                icd_version=10,
                icd_title="Synthetic disease",
                seq_num=1,
            )
        ],
    )
    table(
        ed + "triage.csv.gz",
        [
            dict(
                stay_id=1001,
                chiefcomplaint="synthetic",
                temperature=98,
                heartrate=70,
                resprate=16,
                o2sat=99,
                sbp=120,
                dbp=80,
                acuity=2,
            )
        ],
    )
    table(
        ed + "vitalsign.csv.gz",
        [
            dict(
                stay_id=1001,
                charttime=f"2020-01-01 {hour}:00:00",
                heartrate=hr,
                o2sat=99,
            )
            for hour, hr in [("10", 70), ("13", 999)]
        ],
    )
    table(ed + "medrecon.csv.gz", [], ["stay_id", "name"])
    table(
        hosp + "labevents.csv.gz",
        [
            dict(
                subject_id=1,
                hadm_id=101,
                itemid=item,
                charttime=f"2020-01-01 {hour}:00:00",
                storetime=f"2020-01-01 {store}:00:00",
                value=value,
                valuenum=1,
                flag="abnormal",
            )
            for item, hour, store, value in [
                (1, "10", "10", "AVAILABLE"),
                (2, "13", "13", "FUTURE_LAB"),
                (50912, "11", "13", "DELAYED_LAB"),
            ]
        ],
    )
    table(
        hosp + "d_labitems.csv.gz",
        [dict(itemid=i, label=f"Lab{i}") for i in (1, 2, 50912)],
    )
    table(
        hosp + "diagnoses_icd.csv.gz",
        [],
        ["subject_id", "hadm_id", "icd_code", "icd_version"],
    )
    table(
        hosp + "d_icd_diagnoses.csv.gz", [], ["icd_code", "icd_version", "long_title"]
    )
    table(
        hosp + "omr.csv.gz",
        [],
        ["subject_id", "chartdate", "result_name", "result_value"],
    )
    table(
        "mimic-iv-ecg/1.0/machine_measurements.csv",
        [
            dict(
                subject_id=1,
                ecg_time=f"2020-01-01 {hour}:00:00",
                rr_interval=1000,
                qrs_end=100,
                qrs_onset=0,
                t_end=300,
                p_onset=-50,
                **{f"report_{i}": (label if i == 0 else "") for i in range(18)},
            )
            for hour, label in [("10", "ECG_NOW"), ("13", "ECG_FUTURE")]
        ],
    )
    table(
        "mimic-iv-echo/structured-measurement.csv.gz",
        [
            dict(
                subject_id=1,
                measurement_datetime=f"2020-01-01 {hour}:00:00",
                measurement=label,
                result=1,
                unit="x",
            )
            for hour, label in [("10", "ECHO_NOW"), ("13", "ECHO_FUTURE")]
        ],
    )
    table(
        note + "radiology.csv.gz",
        [
            dict(
                subject_id=1,
                hadm_id=101,
                charttime=f"2020-01-01 {hour}:00:00",
                storetime="2020-01-01 13:00:00",
                text="FINDINGS: " + label,
            )
            for hour, label in [("11", "RAD_DELAYED"), ("13", "RAD_FUTURE")]
        ],
    )


def command(*args):
    p = subprocess.run(
        [sys.executable, *map(str, args)], cwd=ROOT, text=True, capture_output=True
    )
    assert p.returncode == 0, p.stdout + "\n" + p.stderr
    return p


def test_builder_temporal_cutoffs_end_to_end(tmp_path):
    synthetic_modules(tmp_path / "modules")
    index = tmp_path / "index.json"
    command(
        "pipeline/make_cohort_index.py",
        "--data-root",
        tmp_path / "modules",
        "--out",
        index,
    )
    assert json.loads(index.read_text()) == [[1, 101, 1001]]
    paths = {}
    for name, flags in [
        ("event", []),
        ("available", ["--timestamp", "storetime"]),
        ("wide", ["--window-hours", "6"]),
    ]:
        paths[name] = tmp_path / f"{name}.jsonl"
        command(
            "pipeline/build_planA.py",
            "--data-root",
            tmp_path / "modules",
            "--cohort-index",
            index,
            "--out",
            paths[name],
            *flags,
        )
    event, available, wide = [
        dx.input_of(dx.read_rows(paths[n])[0]) for n in ("event", "available", "wide")
    ]
    assert "41yo" in event
    assert "FUTURE" not in event and "HR999" not in event
    assert "RAD_DELAYED" in event and "DELAYED_LAB" in event
    assert "RAD_DELAYED" not in available and "DELAYED_LAB" not in available
    assert "creatinine None" in available
    assert "AVAILABLE" in available
    for text in ("RAD_FUTURE", "FUTURE_LAB", "ECG_FUTURE", "ECHO_FUTURE", "HR999"):
        assert text in wide


def test_evaluation_cli_uses_test_gold_and_all_predictions(tmp_path):
    r = record()
    r["split"] = "test"
    dx.write_rows(tmp_path / "test.jsonl", [r])
    dx.write_rows(tmp_path / "verdicts.jsonl", [r])
    pred = {
        **dx.identity(r),
        "input": dx.input_of(r),
        "gold": ["FORGED GOLD"],
        "pred": ["Condition A", "Extra"],
    }
    dx.write_rows(tmp_path / "pred.jsonl", [pred])
    (tmp_path / "systems.json").write_text(
        json.dumps({"systems": [{"name": "synthetic", "files": ["pred.jsonl"]}]})
    )
    backend = {
        "provider": "local",
        "base_url": "http://127.0.0.1:8002/v1",
        "model": "Qwen3.5-27B",
        "checkpoint": "synthetic-test",
        "revision": "test-only",
    }
    (tmp_path / "backends.json").write_text(json.dumps({"judge": backend}))
    cache = tmp_path / "cache.jsonl"
    Judge(cache, backend)
    dx.write_rows(
        cache,
        [
            {"k": cache_key(g, pred["pred"]), "m": m}
            for g, m in [
                (["Condition A"], 1),
                (["Condition B"], 0),
                (["Condition A", "Condition B"], 1),
            ]
        ],
    )
    import os

    result = subprocess.run(
        [
            sys.executable,
            "pipeline/eval_unified.py",
            "--manifest",
            str(tmp_path / "systems.json"),
            "--test",
            str(tmp_path / "test.jsonl"),
            "--verdicts",
            str(tmp_path / "verdicts.jsonl"),
            "--out",
            str(tmp_path / "out"),
            "--cache-file",
            str(cache),
            "--cache-only",
        ],
        cwd=ROOT,
        env={**os.environ, "EARLYDX_BACKENDS": str(tmp_path / "backends.json")},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    scores = json.loads((tmp_path / "out/results.json").read_text())["systems"][0]
    assert scores["blocks"]["supported"]["recall"] == 1
    assert scores["blocks"]["supported"]["precision"] == 0.5
    assert scores["blocks"]["secondary"]["gold_labels"] == 2


def test_annotation_cleaning_and_cot_chain(tmp_path, monkeypatch):
    from pipeline.annotate import run

    raw = [record(1, 1), record(2, 1), record(3, 2)]
    for r in raw:
        r.pop("label_verdicts")
    dx.write_rows(tmp_path / "raw.jsonl", raw)
    backends = {
        role: {
            "provider": "local",
            "base_url": f"http://127.0.0.1:{port}/v1",
            "model": model,
            "checkpoint": "synthetic-test",
            "revision": "test-only",
        }
        for role, port, model in [
            ("verifier", 8000, "MiniMax-M3"),
            ("teacher", 8001, "MiMo-V2.5"),
        ]
    }
    cfg = tmp_path / "backends.json"
    cfg.write_text(json.dumps(backends))
    monkeypatch.setattr(llm, "CONFIG_PATH", str(cfg))
    calls = []

    async def fake_chat(role, prompt, **kwargs):
        calls.append(role)
        if role == "verifier":
            return json.dumps(
                {
                    "verdicts": [
                        {"dx": "Condition A", "verdict": "supported"},
                        {"dx": "Condition B", "verdict": "unsupported"},
                    ]
                }
            )
        return (
            "<think>Reason from synthetic evidence.</think><answer>Condition A</answer>"
        )

    monkeypatch.setattr(llm, "chat", fake_chat)
    args = [
        "--input",
        str(tmp_path / "raw.jsonl"),
        "--out",
        str(tmp_path / "verified.jsonl"),
    ]
    asyncio.run(run("verifier", args))
    assert len(calls) == 3
    asyncio.run(run("verifier", args))
    assert len(calls) == 3  # Complete records are not queried again.
    (tmp_path / "split.csv").write_text("subject_id,split\n1,train\n2,test\n")
    command(
        "pipeline/prepare_dataset.py",
        "--input",
        tmp_path / "verified.jsonl",
        "--split-file",
        tmp_path / "split.csv",
        "--out-dir",
        tmp_path / "data",
    )
    train, test = (
        dx.read_rows(tmp_path / "data/sft_train_direct.jsonl"),
        dx.read_rows(tmp_path / "data/sft_test.jsonl"),
    )
    assert len(train) == 2 and len(test) == 1
    assert all(dx.gold_of(r) == ["Condition A"] for r in train + test)
    asyncio.run(
        run(
            "teacher",
            [
                "--input",
                str(tmp_path / "data/sft_train_direct.jsonl"),
                "--out",
                str(tmp_path / "cot.jsonl"),
            ],
        )
    )
    cot = dx.read_rows(tmp_path / "cot.jsonl")
    assert {r["stay_id"] for r in cot} == {1, 2}
    assert all("<think>" in r["messages"][-1]["content"] for r in cot)
    # Truncating a completed output must not silently turn it into a smaller dataset.
    dx.write_rows(
        tmp_path / "verified.jsonl", dx.read_rows(tmp_path / "verified.jsonl")[:1]
    )
    with pytest.raises(ValueError, match="incomplete"):
        dx.ensure_complete(
            tmp_path / "verified.jsonl", dx.read_rows(tmp_path / "verified.jsonl")
        )


def test_window_alignment_preserves_gold(tmp_path):
    original = record()
    original["split"] = "test"
    rebuilt = copy.deepcopy(original)
    rebuilt["evidence"]["window_hours"] = 6
    rebuilt["messages"][0]["content"] = "New evidence" + dx.QUESTION
    rebuilt["messages"][1]["content"] = "<answer>Different labels</answer>"
    dx.write_rows(tmp_path / "original.jsonl", [original])
    dx.write_rows(tmp_path / "rebuilt.jsonl", [rebuilt])
    command(
        "pipeline/align_window.py",
        "--inputs",
        tmp_path / "rebuilt.jsonl",
        "--reference-test",
        tmp_path / "original.jsonl",
        "--out",
        tmp_path / "aligned.jsonl",
        "--window-hours",
        "6",
    )
    aligned = dx.read_rows(tmp_path / "aligned.jsonl")[0]
    assert dx.gold_of(aligned) == dx.gold_of(original)
    assert dx.input_of(aligned) == "New evidence"
    assert aligned["reference_evidence"]["window_hours"] == 0
    assert references([aligned], [original])[1]["supported"] == ["Condition A"]


def test_annotation_failure_stops_instead_of_scoring(tmp_path, monkeypatch):
    from pipeline.annotate import run

    dx.write_rows(tmp_path / "input.jsonl", [record()])
    monkeypatch.setattr(llm, "provenance", lambda role: {"model": "synthetic-test"})

    async def malformed(*args, **kwargs):
        return '{"verdicts":[]}'

    monkeypatch.setattr(llm, "chat", malformed)
    with pytest.raises(RuntimeError, match="Incomplete annotations"):
        asyncio.run(
            run(
                "verifier",
                [
                    "--input",
                    str(tmp_path / "input.jsonl"),
                    "--out",
                    str(tmp_path / "output.jsonl"),
                ],
            )
        )
    assert dx.read_rows(tmp_path / "output.jsonl") == []
