# EarlyDx: An Admission-Anchored Benchmark for Open-Ended Generation of Evidence-Supported ED-Encounter Diagnoses

## Setup

Use a Python environment with the dependencies in `requirements.txt`. GPU training requires
CUDA and a model-compatible Transformers installation; see [MODELS.md](MODELS.md).
For data/evaluation checks without GPU dependencies:

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

Access MIMIC through its credentialed distribution and comply with its data use agreement.
The builder expects a local root containing the hospital, ED, note, ECG and echo modules:

```text
MIMIC_ROOT/
  mimic-iv/3.1/hosp/
  mimic-iv-ed/<one-version>/ed/
  mimic-iv-note/2.2/note/
  mimic-iv-ecg/1.0/machine_measurements.csv
  mimic-iv-echo/structured-measurement.csv.gz
```

Hospital, note and ECG versions are configurable through the builder's CLI. Exactly one ED
version must be present. All module files remain local.

For the verifier, rationale teacher and judge, copy `backends.example.json` to `backends.json`.
Fill in the actual checkpoints/revisions and start local OpenAI-compatible servers, for example:

```bash
vllm serve /path/to/checkpoint --served-model-name Qwen3.5-27B --host 127.0.0.1 --port 8002
```

Use the paper's MiniMax-M3 verifier, MiMo-V2.5 teacher and Qwen3.5-27B judge. Configuration
placeholders are rejected. Institutional hosts can be listed in `EARLYDX_LOCAL_HOSTS`.
Credentials, if needed, are read from the environment variable named by `api_key_env` in the
local backend configuration. The corpus-processing methods do not call public model routers.

## Data construction

Set `MIMIC_ROOT` to the downloaded modules. Commands below are run from the repository root.

```bash
python pipeline/make_cohort_index.py --data-root "$MIMIC_ROOT" --out data/cohort_index.json
python pipeline/build_planA.py --data-root "$MIMIC_ROOT" \
  --cohort-index data/cohort_index.json --out data/cohort.jsonl
python pipeline/judge_labels.py --input data/cohort.jsonl --out data/verified.jsonl
python pipeline/prepare_dataset.py --input data/verified.jsonl \
  --split-file /path/to/subject-info.csv --out-dir data
```

- **Cohort:** every ED stay linked to a hospital admission, identified by `stay_id`.
- **Time anchor:** hospital `admittime`, with `--window-hours 0` by default. The builder filters
  timed evidence before serialization. `--window-hours 6` and `24` are sensitivity settings.
- **Timestamp convention:** event time by default. `--timestamp storetime` uses availability
  time for labs and radiology, including prior baseline labs (Appendix D).
- **Inputs:** demographics, chief complaint, triage/serial vitals, home medications, outpatient
  baseline measurements, labs, ECG interpretations, echo measurements, radiology findings,
  and earlier-admission/earlier-ED history. Missing modalities are represented as `None`.
- **Labels:** ED diagnosis titles as free text. Remove administrative/symptom code chapters and
  overly generic NOS/NEC patterns; then classify every label as supported, partial or unsupported.
- **Training/evaluation pool:** retain supported and partial labels; exclude encounters with
  neither. Preserve the original verifier output separately.
- **Split:** the CSV must contain `subject_id,split`, where split is `train` or `test`. Every
  encounter of a patient stays in the same partition. Obtain the actual fixed split separately;
  this repository does not include it. `--new-split` explicitly creates a new experimental
  split and cannot reproduce the paper's particular split.

The serializer preserves the existing text budgets (report snippets, modality limits and lab
text truncation). Event-time filtering can include results finalized later; it is not a claim
that every included result was readable at admission. History uses the earlier-admission rule
in §3.2. Identifiers, evidence-window metadata and input hashes are retained through the pipeline.

For additional windows, rebuild inputs with the desired window and reuse the same patient
partition. To hold gold labels/verdicts fixed in a sensitivity experiment, use `pipeline/align_window.py --inputs NEW_INPUTS --reference-test data/sft_test.jsonl
--out NEW_TEST --window-hours W --timestamp CONVENTION`. This joins by `stay_id` and records
the original reference window separately. Evaluate the new test inputs against the original
verifier file using matching `--window-hours`/`--timestamp` settings; do not re-audit labels.

## Post-training

The preparation step writes `data/sft_train_direct.jsonl` and `data/sft_test.jsonl`.
Generate CoT targets only for the training examples, then train:

```bash
python pipeline/gen_cot.py --input data/sft_train_direct.jsonl --out data/sft_train_cot.jsonl

torchrun --nproc_per_node=3 pipeline/sft_qwen_4b.py \
  --train data/sft_train_cot.jsonl --out models/qwen4b-cot \
  --format cot --revision "$QWEN_REVISION"
```

Set `QWEN_REVISION` to the exact base-model revision. Use `--format direct` with the answer-only
training file for direct supervision. `pipeline/sft_qwen.py` runs the corresponding 2B recipe.
Targets and evidence classes are checked before training. See [MODELS.md](MODELS.md) for the
training settings and [prompts.py](prompts.py) for the paper's pipeline prompts.

Generate predictions from a saved post-trained checkpoint:

```bash
python pipeline/infer_full.py --test data/sft_test.jsonl \
  --checkpoint models/qwen4b-cot --out results/predictions/shard0.jsonl
```

For multiple GPUs, run separate processes with `--shard N --num-shards K` and a distinct output
path per shard. Global indices and ED stay IDs remain stable. `--demo-cc-only` and
`--drop-modality` implement inference-time evidence ablations; label such systems with an
explicit `input_variant` in the evaluation manifest.

## Azure OpenAI and Anthropic prediction APIs

Configure `gpt-5.5` and `claude-opus-4.8` in `backends.json` using the additional entries in
`backends.example.json`. Supply an existing Azure deployment and exact provider model revisions.
Set `AZURE_OPENAI_API_KEY` and `ANTHROPIC_API_KEY` through your local environment or secret manager;
never put keys in this repository. An optional Anthropic `workspace_id` supports keys requiring
an explicit workspace.

For MIMIC-derived input, only after the relevant account arrangements are effective, set:

- `EARLYDX_AZURE_REVIEW_OPTOUT=1`: confirm the Azure deployment's approved human-review opt-out.
- `EARLYDX_ANTHROPIC_ZDR=1`: confirm ZDR applies to the specific Anthropic organization, model
  and endpoint being used, including any applicable exceptions.

These are operator attestations, not automatic evidence of compliance. See the
[PhysioNet guidance](https://physionet.org/news/post/gpt-responsible-use/),
[Azure data handling documentation](https://learn.microsoft.com/en-us/azure/foundry/responsible-ai/openai/data-privacy),
and [Anthropic ZDR scope](https://privacy.claude.com/en/articles/8956058-i-have-a-zero-data-retention-agreement-with-anthropic-what-products-does-it-apply-to).

```bash
python pipeline/infer_api.py --role gpt-5.5 --test data/sft_test.jsonl \
  --out results/predictions/azure-gpt55.jsonl
python pipeline/infer_api.py --role claude-opus-4.8 --test data/sft_test.jsonl \
  --out results/predictions/anthropic-opus48.jsonl
```

Add `--dry-run` to validate the selected records and backend configuration without requests
or output files. By default, every test encounter is processed. `--limit` explicitly selects a
prefix subsample, and `--conc` / `--max-tokens` control concurrency and generation budget.
Rerunning the same command resumes by `stay_id`; changed inputs, models or settings require a
new output path. Failed requests stop the run with incomplete coverage rather than becoming
empty diagnoses. Returned refusals or malformed model answers are recorded as empty predictions
with the corresponding format/refusal status. API response metadata is saved locally.

Azure uses [Chat Completions v1](https://learn.microsoft.com/en-us/rest/api/microsoft-foundry/azureopenai/chat)
with `store=false`; Anthropic uses the [Messages API](https://platform.claude.com/docs/en/build-with-claude/working-with-messages)
at its fixed official endpoint. Neither adapter uses OpenRouter. Verification, rationale
generation and semantic judging still use the local backends.

Copy `systems.api.example.json` to `systems.json` to evaluate these two prediction files with
the same local judge. You can add the post-trained model's files to that manifest as another
system. Completed subsamples are marked as partial coverage; interrupted generated files must
be resumed to completion before evaluation.

## Evaluation

Use `systems.example.json` for post-trained predictions or `systems.api.example.json` for
hosted predictions as the starting point for `systems.json`. List files relative to the
manifest's directory. Each row must contain `subject_id`, `hadm_id`, `stay_id`, `evidence`,
`input` and a `pred` list. These fields are emitted by `infer_full.py`; external model predictions
can use the same schema. Empty `pred` lists are scored; missing rows are reported as partial coverage.

```bash
python pipeline/eval_unified.py --manifest systems.json \
  --test data/sft_test.jsonl --verdicts data/verified.jsonl --out results/evaluation
```

The evaluator uses reference labels from the test file and evidence classes from the verifier
file, never gold copied into prediction files. One local judge performs one-to-one semantic
matching, using the Appendix J prompt. Synonyms, abbreviations and specificity differences are
accepted as described in that prompt.

Three tracks are scored separately: **supported** (primary), **partial**, and **secondary**
(supported + partial). A stay enters a track only if it has reference labels in that track.
Unsupported labels never enter the gold set. All distinct predictions count in the precision
denominator, including predictions of partial labels in the primary track.

For matched count `m`, gold count `g` and predicted count `p`, precision is `m/p`, recall is
`m/g`, F1 is `2m/(g+p)`, and Jaccard is `m/(g+p-m)`. Micro metrics pool counts; example metrics
average per-stay scores. Complete match requires `m == g == p`. Zero denominators yield zero.

Optional controls:

- `--prediction-budget K`: truncate each prediction list to its first K diagnoses, including
  supported recall at a matched budget.
- `--common`: evaluate all systems on their shared prediction coverage.
- `--bootstrap B --ref "System name"`: patient-level bootstrap intervals and paired comparisons.
- `--cache-only`: require every needed judgment to be present; do not contact the model server.

Caches are tied to model checkpoint/revision and prompt. Unparseable judgments and exhausted
requests stop the run; they are not converted to missed diagnoses. Results are computed into
`results.json`, `per_encounter.jsonl` and `table.md`, with actual coverage and run provenance.
No result values are bundled in the repository.
