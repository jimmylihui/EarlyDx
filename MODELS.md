# Model roles and training recipe

This file describes the method in [EarlyDx](https://arxiv.org/abs/2607.28788v1).
It is a configuration guide, not a record of completed experiments or model access dates.
No model weights or evaluation outputs are distributed here.

| Stage | Paper model | Configuration |
|---|---|---|
| Evidence verification (§3.3) | MiniMax M3 | `verifier` in `backends.json`; local; temperature 0 |
| Gold-conditioned rationale generation (Appendix G) | MiMo-V2.5 | `teacher`; local; temperature 0.3 |
| Semantic evaluation (§3.5) | Qwen3.5-27B | `judge`; local; temperature 0 |
| Main post-training (§4.2) | Qwen3.5-4B | `pipeline/sft_qwen_4b.py` |
| Capacity control (Appendix K/L) | Qwen3.5-2B | `pipeline/sft_qwen.py` |
| Hosted zero-shot prediction (§4.2/R) | GPT-5.5 | `gpt-5.5`; direct Azure OpenAI deployment |
| Hosted zero-shot prediction (§4.2/R) | Claude Opus 4.8 | `claude-opus-4.8`; direct Anthropic Messages API |

The names in `backends.example.json` are serving aliases, not verified download locations.
Set the actual checkpoint and immutable revision and serve that checkpoint under the matching
alias. Placeholders are rejected. Keep the server launch configuration and logs with each run.
The client records the configured identity; it cannot independently attest the server's weights.

## Post-training settings (Appendix K)

- Full language-backbone fine-tuning; the unused vision backbone is frozen. No LoRA.
- Completion-only loss: mask prompt tokens and padding with `-100`.
- AdamW, one epoch, learning rate `1e-5`, cosine decay, warmup ratio `0.03`.
- Effective batch size `48`, maximum sequence length `3072`, bfloat16, gradient checkpointing.
- 2B: distributed data parallelism. 4B: DeepSpeed ZeRO-2 with optimizer CPU offload.
- CoT targets contain `<think>...</think><answer>...</answer>`; direct targets contain only the answer.
- Greedy post-trained inference, at most `2048` new tokens; no gold labels are passed to inference.

Training truncates the serialized sequence at the maximum length and fails if no supervised
completion token remains. Inference retains the existing input budget of `3000` tokens. Those
serialization/truncation choices should be recorded when changing input formatting.

`requirements.txt` gives installable version ranges. Model architecture support depends on the
Transformers release; choose a version that supports the actual checkpoint. Save `pip freeze`
and the training metadata with each run. No tested GPU environment or completed GPU run is
claimed by this repository's method examples.

## Hosted prediction adapters

`pipeline/infer_api.py` uses the zero-shot prompt in `prompts.py` and never sends reference
labels. It records the configured revision, returned model name, request/response IDs, finish
reason and usage in local prediction files. Service access and the deployment's actual model
version must be checked in the provider account; example names are not proof of availability.

Azure requests use `max_completion_tokens`, with `store=false`. Anthropic requests use
`max_tokens`. Sampling parameters are omitted for these hosted roles; in particular, Opus 4.8
must not be sent `temperature`. The local verifier/teacher/judge retain their original sampling
settings. Token budget and provider-native generation behavior are recorded separately from
the greedy post-trained inference recipe.

Before sending MIMIC-derived text, the operator must confirm the required Azure human-review
opt-out or applicable Anthropic zero-data-retention arrangement. The configuration flags only
record that confirmation; they do not apply for approval or verify an agreement.
