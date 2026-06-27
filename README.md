# EarlyDx: A Benchmark for Open-Vocabulary Admission Diagnosis with LLMs

Code to reproduce **EarlyDx**. The benchmark is derived from **MIMIC-IV** (ED + hosp + note
modules); per the PhysioNet Credentialed Health Data License, **we do not redistribute any
MIMIC-derived data or models here**. Credentialed users can regenerate the full dataset from
the scripts below.

## ⚠️ Data access (required)
1. Obtain **credentialed access** to MIMIC-IV, MIMIC-IV-ED, and MIMIC-IV-Note on
   [PhysioNet](https://physionet.org/) (CITI training + signed DUA).
2. Download the modules locally. Set the MIMIC root paths inside `pipeline/build_planA.py`
   (`mimic-iv/`, `mimic-iv-ed/`).
3. The generated dataset and fine-tuned models are released to credentialed users via
   PhysioNet (not on public hubs).

## Setup
```bash
conda create -n earlydx python=3.10 && conda activate earlydx
pip install torch transformers==5.13.0.dev0 deepspeed accelerate httpx pandas
# LLM-judge / CoT generation use OpenRouter; put your keys in or_keys.json (list of strings)
echo '["YOUR_OPENROUTER_KEY"]' > or_keys.json
```

## Reproduction pipeline
```bash
# 1. Build the ED→admission cohort index
python pipeline/make_cohort_index.py

# 2. Build admission-time samples (t0-clipped, multi-source input + ED diagnosis labels)
python pipeline/build_planA.py

# 3. Evidence verifier: classify each label supported / partial / unsupported
python pipeline/judge_labels.py            # writes label_verdicts

# 4. Generate chain-of-thought supervision (gold-conditioned, supported+partial only)
python pipeline/gen_cot.py                 # -> cohort_cot_final.jsonl

# 5. Train (subject-disjoint split). 2B = DDP; 4B = DeepSpeed ZeRO-2 (zero2_offload.json)
torchrun --nproc_per_node=3 pipeline/sft_qwen.py
torchrun --nproc_per_node=3 pipeline/sft_qwen_4b.py

# 6. Inference on the test set (3-GPU sharded)
for s in 0 1 2; do CUDA_VISIBLE_DEVICES=$s python pipeline/infer_full.py $s 3 & done; wait

# 7. Evaluate with the LLM-judge (semantic matching -> micro/example P/R/F1, Jaccard)
python pipeline/eval_full.py qwen
python pipeline/eval_allbyverdict.py       # supported / partial / all breakdown
```

## Baselines (`baselines/`)
Zero-shot prompting of general and medical LLMs. API models read keys from `or_keys.json`
or environment variables (`OPENROUTER_KEY`, `HF_TOKEN`) — **no keys are committed**.
- `gpt_run.py`, `opus_run.py`, `nemotron_full.py` — API zero-shot (OpenRouter).
- `infer_zeroshot.py`, `infer_openbio.py`, `infer_huatuo.py`, `infer_medgemma.py` — local
  open / medical LLMs.

## Analysis & figures (`analysis/`)
- `modality_attr.py` + `plot_modality.py` / `plot_disease_modality.py` — evidence-modality
  attribution from CoT.
- `plot_avgdx.py`, `plot_complete.py` — average #diagnoses and complete-match rate.
- `cot_review.py`, `cot_agree.py`, `verifier_agree.py` — reliability / consistency checks.

## Metrics
Diagnoses are open-vocabulary free text; we evaluate with an **LLM-as-judge** that computes a
one-to-one semantic matching, yielding micro- and example-averaged Precision/Recall/F1 and the
Jaccard index. The judge model is fixed and its decisions are cached (`judge_cache.jsonl`) for
reproducibility.

## License / Citation
Code: MIT. Data & models: credentialed access via PhysioNet (MIMIC DUA).
Please cite the EarlyDx paper and MIMIC-IV.
