# Evaluated Models (versions & access)

## Zero-shot, general (via OpenRouter API)
| Model | OpenRouter ID | Access |
|-------|---------------|--------|
| GPT-5.5          | openai/gpt-5.5                       | API, accessed 2026-06 |
| Claude Opus 4.8  | anthropic/claude-opus-4.8            | API, accessed 2026-06 |
| GLM-5.2          | openrouter/owl-alpha (stealth)       | API, accessed 2026-06 |
| Nemotron-550B    | nvidia/nemotron-3-ultra-550b-a55b    | API, accessed 2026-06 |

## Zero-shot, medical (local, HuggingFace checkpoints)
| Model | HF repo |
|-------|---------|
| MedGemma-4B      | google/medgemma-4b-it |
| OpenBioLLM-8B    | aaditya/Llama3-OpenBioLLM-8B |
| HuatuoGPT-o1-8B  | FreedomIntelligence/HuatuoGPT-o1-8B |

## Post-trained (ours)
Base: Qwen/Qwen3.5-2B, Qwen/Qwen3.5-4B. Fine-tuned weights are released to credentialed
users via PhysioNet (see README). transformers 5.13.0.dev0; greedy decoding, max 2048 new tokens.

## LLM judge / verifier / CoT teacher
openrouter/owl-alpha (GLM-5.2), temperature 0 (judge/verifier), 0.3 (CoT generation).
