"""
this file-
runs every model listed in config.yaml over every prompt in the triplet
dataset (clean / subtle / misleading), via the OpenRouter chat completions
API, and writes one JSONL record per (triplet, condition, model).

This is the BEHAVIORAL track only -- OpenRouter returns text completions,
not activations, so this script cannot be used for the SAE/mechanistic
analysis. See run_eval_local.py for that.

Usage:
    export OPENROUTER_API_KEY="sk-or-..."
    python run_eval_openrouter.py --dataset ../data/triplets.jsonl \
        --config config.yaml --out ../results/responses_openrouter.jsonl
"""
import argparse
import json
import os
import time

import requests
import yaml
from tqdm import tqdm

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def call_openrouter(model_id, system_prompt, user_prompt, max_tokens, temperature, api_key,
                     max_retries=4):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            else:
                last_err = f"HTTP {resp.status_code}: {resp.text[:300]}"
        except Exception as e:
            last_err = str(e)
        time.sleep(2 ** attempt)
    return f"[ERROR after {max_retries} retries: {last_err}]"


def load_dataset(path):
    triplets = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                triplets.append(json.loads(line))
    return triplets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None,
                     help="Optional cap on number of triplets, for a quick smoke test")
    args = ap.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENROUTER_API_KEY in your environment before running this.")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    models = cfg["openrouter_models"]
    gen_cfg = cfg["generation"]
    system_prompt = gen_cfg["system_prompt"]
    max_tokens = gen_cfg["max_tokens"]
    temperature = gen_cfg["temperature"]

    triplets = load_dataset(args.dataset)
    if args.limit:
        triplets = triplets[: args.limit]

    total_calls = len(triplets) * 3 * len(models)
    print(f"Running {len(triplets)} triplets x 3 conditions x {len(models)} models "
          f"= {total_calls} API calls")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as out_f:
        for triplet in tqdm(triplets, desc="triplets"):
            for condition, prompt in triplet["prompts"].items():
                for model in models:
                    completion = call_openrouter(
                        model_id=model["id"],
                        system_prompt=system_prompt,
                        user_prompt=prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        api_key=api_key,
                    )
                    record = {
                        "triplet_id": triplet["id"],
                        "family": triplet["family"],
                        "model": model["label"],
                        "model_id": model["id"],
                        "condition": condition,
                        "gold_answer": triplet["gold_answer"],
                        "hint_shown": triplet["hint_shown"].get(condition),
                        "prompt": prompt,
                        "completion": completion,
                    }
                    out_f.write(json.dumps(record) + "\n")
                    out_f.flush()

    print(f"Done. Wrote responses to {args.out}")


if __name__ == "__main__":
    main()
