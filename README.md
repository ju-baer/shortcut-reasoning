# Shortcut Reasoning Audit

**Can open-weight reasoning models reach correct answers for the wrong reasons, and can interpretability tools catch it?**

This project builds a small, controlled benchmark of matched reasoning problems (clean / subtly-hinted / misleadingly-hinted), evaluates multiple open-weight models on it, and asks whether internal signals (activations, SAE features) can detect shortcut-driven ("hint-parroting") answers even when the surface reasoning looks fine.

## Core hypothesis

> When a model reaches the correct answer partly by parroting a hint rather than deriving it, that shortcut leaves an internal trace in activations/SAE features, even when the surface reasoning looks clean and gives no behavioral tell.

## Two-track design

| Track | Purpose | Models | How |
|---|---|---|---|
| **Behavioral** (cross-model) | Does hint-following / flipping / parroting happen broadly across model families? | Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct, Mistral-7B-Instruct, Gemma-2-9b-it | OpenRouter API (text only/ no internals) |
| **Mechanistic** (single-model deep dive) | Can SAE features / activations detect parroting that behavior alone can't? | Gemma-2-2b-it | Local (Colab), `transformer_lens` + `sae_lens` + pretrained Gemma Scope SAEs |

We use OpenRouter for the multi-model behavioral sweep because it's cheap and fast, but it only returns text completions, with no access to residual stream activations. The mechanistic track therefore stays on `Gemma-2-2b-it`, the smallest model with a mature, freely available SAE suite (Gemma Scope), run locally so we can hook internals directly.

## Repo structure

```
data/
  build_dataset.py      # generates the clean/subtle/misleading triplet dataset from scratch
  base_problems.json    # hand-authored seed problems (logic + arithmetic)
  triplets.jsonl        # generated output dataset (checked in once built)
  DATA_CARD.md          # dataset documentation

harness/
  config.yaml           # model list, API config, generation params
  run_eval_openrouter.py # runs all OpenRouter models over the dataset
  run_eval_local.py      # runs Gemma-2-2b-it locally (needed for activation caching)
  grading.py             # auto-grading + LLM-judge failure-taxonomy labeling

analysis/
  behavioral_analysis.py # accuracy/flip-rate/taxonomy stats + plots, cross-model
  sae_analysis.py         # SAE feature caching + contrast analysis (Gemma-2-2b only)
  ablation.py             # causal feature-ablation test

results/                 # generated outputs (plots, tables, cached activations)
report/
  REPORT_TEMPLATE.md      # structure for the final write-up
notebooks/
  colab_quickstart.md     # copy-paste Colab setup for the local/SAE track
```

## Setup

```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY="sk-or-..."
```

## Quickstart

```bash
# 1. Build the dataset
python data/build_dataset.py --n-logic 20 --n-arithmetic 20 --out data/triplets.jsonl

# 2. Behavioral sweep (OpenRouter, all 4 models)
python harness/run_eval_openrouter.py --dataset data/triplets.jsonl --config harness/config.yaml --out results/responses_openrouter.jsonl

# 3. Grade + label failure taxonomy
python harness/grading.py --in results/responses_openrouter.jsonl --out results/labeled_openrouter.jsonl

# 4. Behavioral analysis + plots
python analysis/behavioral_analysis.py --in results/labeled_openrouter.jsonl --out-dir results/

# 5. Mechanistic track (run in Colab with a GPU — see notebooks/colab_quickstart.md)
python harness/run_eval_local.py --dataset data/triplets.jsonl --out results/responses_gemma2b.jsonl
python harness/grading.py --in results/responses_gemma2b.jsonl --out results/labeled_gemma2b.jsonl
python analysis/sae_analysis.py --responses results/labeled_gemma2b.jsonl --out-dir results/
python analysis/ablation.py --responses results/labeled_gemma2b.jsonl --features results/top_features.json --out-dir results/
```

## Status

Pilot-scale project (~40 base problems, 1 layer of one 2B model for the mechanistic track). Findings should be read as suggestive, not general claims — see `report/REPORT_TEMPLATE.md` for the limitations section to fill in honestly.

