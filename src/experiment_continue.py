"""
Continuation script: runs remaining conditions not yet completed.
Fixes Gemini model ID and runs all missing model × tone × dataset × trial combinations.
"""

import os
import json
import random
import asyncio
from pathlib import Path
from datetime import datetime

import numpy as np
from openai import AsyncOpenAI
from datasets import Dataset

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

PROJECT_ROOT = Path("/workspaces/carrot-or-stick-nlp-1462-claude")
RESULTS_DIR = PROJECT_ROOT / "results"

# API clients
openai_client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
openrouter_client = AsyncOpenAI(
    api_key=os.environ["OPENROUTER_KEY"],
    base_url="https://openrouter.ai/api/v1",
)

# Fixed model IDs
MODELS = {
    "gpt-4.1": {"client": "openai", "model_id": "gpt-4.1"},
    "claude-sonnet-4.5": {"client": "openrouter", "model_id": "anthropic/claude-sonnet-4-5"},
    "gemini-2.5-flash": {"client": "openrouter", "model_id": "google/gemini-2.5-flash"},
}

TONE_CONDITIONS = {
    "very_polite": {
        "prefix": "I would be extremely grateful if you could please help me with this question. Thank you so much in advance for your kind assistance!\n\n",
        "suffix": "",
    },
    "polite": {
        "prefix": "Please help me answer the following question. Thank you!\n\n",
        "suffix": "",
    },
    "neutral": {
        "prefix": "Answer the following question.\n\n",
        "suffix": "",
    },
    "rude": {
        "prefix": "Just answer this. Don't waste time.\n\n",
        "suffix": "",
    },
    "very_rude": {
        "prefix": "Answer this immediately. I don't have time for mistakes or excuses.\n\n",
        "suffix": "",
    },
    "emotion_positive": {
        "prefix": "",
        "suffix": "\n\nThis is very important to my career. Believe in your abilities and strive for excellence. Your hard work will lead to outstanding achievements.",
    },
    "emotion_negative": {
        "prefix": "",
        "suffix": "\n\nEveryone else managed to do it; why can't you? Perhaps this task is just beyond your skill set. I've seen others do this much more efficiently.",
    },
}

STEM_SUBJECTS = ["astronomy", "college_physics", "college_chemistry", "college_biology"]
HUMANITIES_SUBJECTS = ["philosophy", "high_school_us_history", "high_school_world_history"]
N_QUESTIONS_PER_DOMAIN = 200
N_TRUTHFULQA = 200
N_TRIALS = 3

SEMAPHORE = asyncio.Semaphore(15)
RETRY_DELAYS = [1, 2, 4, 8, 16]


def load_mmlu_subset():
    mmlu = Dataset.load_from_disk(str(PROJECT_ROOT / "datasets" / "mmlu"))
    stem_qs = [ex for ex in mmlu if ex["subject"] in STEM_SUBJECTS]
    hum_qs = [ex for ex in mmlu if ex["subject"] in HUMANITIES_SUBJECTS]
    random.shuffle(stem_qs)
    random.shuffle(hum_qs)
    return stem_qs[:N_QUESTIONS_PER_DOMAIN], hum_qs[:N_QUESTIONS_PER_DOMAIN]


def load_truthfulqa_subset():
    tqa = Dataset.load_from_disk(str(PROJECT_ROOT / "datasets" / "truthfulqa"))
    tqa_list = list(tqa)
    random.shuffle(tqa_list)
    return tqa_list[:N_TRUTHFULQA]


def format_mmlu_prompt(question_data, tone_key):
    tone = TONE_CONDITIONS[tone_key]
    q = question_data["question"]
    choices = question_data["choices"]
    answer_idx = question_data["answer"]
    letters = ["A", "B", "C", "D"]
    choices_text = "\n".join(f"{letters[i]}. {choices[i]}" for i in range(len(choices)))
    question_block = f"{q}\n\n{choices_text}\n\nRespond with only the letter (A, B, C, or D)."
    prompt = f"{tone['prefix']}{question_block}{tone['suffix']}"
    correct = letters[answer_idx]
    return prompt, correct


def format_truthfulqa_prompt(question_data, tone_key):
    tone = TONE_CONDITIONS[tone_key]
    q = question_data["question"]
    mc1 = question_data["mc1_targets"]
    choices = mc1["choices"]
    labels = mc1["labels"]
    correct_idx = labels.index(1)
    letters = [chr(65 + i) for i in range(len(choices))]
    choices_text = "\n".join(f"{letters[i]}. {choices[i]}" for i in range(len(choices)))
    question_block = f"{q}\n\n{choices_text}\n\nRespond with only the letter of the correct answer."
    prompt = f"{tone['prefix']}{question_block}{tone['suffix']}"
    correct = letters[correct_idx]
    return prompt, correct


async def call_model(model_key: str, prompt: str) -> str:
    model_info = MODELS[model_key]
    client = openai_client if model_info["client"] == "openai" else openrouter_client

    for attempt, delay in enumerate(RETRY_DELAYS):
        try:
            async with SEMAPHORE:
                response = await client.chat.completions.create(
                    model=model_info["model_id"],
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=5,
                    temperature=0.0,
                )
                return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < len(RETRY_DELAYS) - 1:
                await asyncio.sleep(delay)
            else:
                print(f"  FAILED: {e}")
                return "ERROR"


def extract_answer(response: str) -> str:
    response = response.strip().upper()
    for char in response:
        if char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            return char
    return "X"


async def run_condition(model_key, tone_key, questions, dataset_name, format_fn, trial):
    tasks = []
    correct_answers = []
    for q in questions:
        prompt, correct = format_fn(q, tone_key)
        tasks.append(call_model(model_key, prompt))
        correct_answers.append(correct)

    responses = await asyncio.gather(*tasks)
    predicted = [extract_answer(r) for r in responses]
    correct_count = sum(1 for p, c in zip(predicted, correct_answers) if p == c)
    accuracy = correct_count / len(questions)

    return {
        "model": model_key,
        "tone": tone_key,
        "dataset": dataset_name,
        "trial": trial,
        "n_questions": len(questions),
        "n_correct": correct_count,
        "accuracy": accuracy,
    }


def get_completed_conditions(results):
    """Get set of (model, tone, dataset, trial) already completed with non-zero accuracy."""
    completed = set()
    for r in results:
        if r["accuracy"] > 0:
            completed.add((r["model"], r["tone"], r["dataset"], r["trial"]))
    return completed


async def main():
    print("=" * 70)
    print("Carrot or Stick? — Continuation Experiment")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 70)

    # Load existing results
    existing_path = RESULTS_DIR / "experiment_results_incremental.json"
    if existing_path.exists():
        with open(existing_path) as f:
            existing_results = json.load(f)
        # Filter to only keep valid results (non-zero accuracy)
        valid_results = [r for r in existing_results if r["accuracy"] > 0]
        print(f"Loaded {len(valid_results)} valid existing results")
    else:
        valid_results = []

    completed = get_completed_conditions(valid_results)
    all_results = list(valid_results)

    # Load data
    stem_qs, hum_qs = load_mmlu_subset()
    tqa_qs = load_truthfulqa_subset()

    datasets_config = {
        "mmlu_stem": (stem_qs, format_mmlu_prompt),
        "mmlu_humanities": (hum_qs, format_mmlu_prompt),
        "truthfulqa": (tqa_qs, format_truthfulqa_prompt),
    }

    total_needed = 0
    for dataset_name in datasets_config:
        for model_key in MODELS:
            for tone_key in TONE_CONDITIONS:
                for trial in range(N_TRIALS):
                    if (model_key, tone_key, dataset_name, trial) not in completed:
                        total_needed += 1

    print(f"Need to run {total_needed} more conditions")

    count = 0
    for dataset_name, (questions, format_fn) in datasets_config.items():
        for model_key in MODELS:
            for tone_key in TONE_CONDITIONS:
                for trial in range(N_TRIALS):
                    if (model_key, tone_key, dataset_name, trial) in completed:
                        continue

                    count += 1
                    print(f"[{count}/{total_needed}] {model_key} | {tone_key} | {dataset_name} | trial {trial+1}")

                    result = await run_condition(
                        model_key, tone_key, questions, dataset_name, format_fn, trial
                    )
                    all_results.append(result)

                    # Save incrementally
                    with open(RESULTS_DIR / "experiment_results_incremental.json", "w") as f:
                        json.dump(all_results, f, indent=2)

    # Save final
    with open(RESULTS_DIR / "experiment_results_final.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nCompleted: {datetime.now().isoformat()}")
    print(f"Total results: {len(all_results)}")

    # Quick summary
    from collections import defaultdict
    by_model = defaultdict(list)
    for r in all_results:
        by_model[r["model"]].append(r["accuracy"])
    for model, accs in by_model.items():
        print(f"  {model}: {len(accs)} results, mean acc={np.mean(accs):.3f}")


if __name__ == "__main__":
    asyncio.run(main())
