"""
Carrot or Stick? — Meta-Study of Prompt Tone Effects on LLM Performance

This script runs the main experiment: testing 7 prompt tone conditions across
multiple LLMs on MMLU and TruthfulQA benchmarks.
"""

import os
import json
import time
import random
import hashlib
import asyncio
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import httpx
from openai import AsyncOpenAI
from datasets import Dataset
from tqdm import tqdm

# ─── Configuration ────────────────────────────────────────────────────────────

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

PROJECT_ROOT = Path("/workspaces/carrot-or-stick-nlp-1462-claude")
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# API clients
openai_client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
openrouter_client = AsyncOpenAI(
    api_key=os.environ["OPENROUTER_KEY"],
    base_url="https://openrouter.ai/api/v1",
)

# Models to test
MODELS = {
    "gpt-4.1": {"client": "openai", "model_id": "gpt-4.1"},
    "claude-sonnet-4.5": {"client": "openrouter", "model_id": "anthropic/claude-sonnet-4-5"},
    "gemini-2.5-flash": {"client": "openrouter", "model_id": "google/gemini-2.5-flash-preview"},
}

# ─── Prompt Tone Conditions ──────────────────────────────────────────────────

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

# ─── MMLU Subject Selection ──────────────────────────────────────────────────

STEM_SUBJECTS = ["astronomy", "college_physics", "college_chemistry", "college_biology"]
HUMANITIES_SUBJECTS = ["philosophy", "high_school_us_history", "high_school_world_history"]

N_QUESTIONS_PER_DOMAIN = 200  # 200 STEM + 200 Humanities = 400 total
N_TRUTHFULQA = 200  # Subsample for feasibility
N_TRIALS = 3  # Runs per condition


def load_mmlu_subset():
    """Load MMLU and sample balanced subsets for STEM and Humanities."""
    mmlu = Dataset.load_from_disk(str(PROJECT_ROOT / "datasets" / "mmlu"))

    stem_qs = [ex for ex in mmlu if ex["subject"] in STEM_SUBJECTS]
    hum_qs = [ex for ex in mmlu if ex["subject"] in HUMANITIES_SUBJECTS]

    random.shuffle(stem_qs)
    random.shuffle(hum_qs)

    stem_sample = stem_qs[:N_QUESTIONS_PER_DOMAIN]
    hum_sample = hum_qs[:N_QUESTIONS_PER_DOMAIN]

    print(f"MMLU STEM: {len(stem_sample)} questions from {STEM_SUBJECTS}")
    print(f"MMLU Humanities: {len(hum_sample)} questions from {HUMANITIES_SUBJECTS}")

    return stem_sample, hum_sample


def load_truthfulqa_subset():
    """Load TruthfulQA and subsample."""
    tqa = Dataset.load_from_disk(str(PROJECT_ROOT / "datasets" / "truthfulqa"))
    tqa_list = list(tqa)
    random.shuffle(tqa_list)
    sample = tqa_list[:N_TRUTHFULQA]
    print(f"TruthfulQA: {len(sample)} questions")
    return sample


def format_mmlu_prompt(question_data, tone_key):
    """Format an MMLU question with the given tone condition."""
    tone = TONE_CONDITIONS[tone_key]
    q = question_data["question"]
    choices = question_data["choices"]
    answer_idx = question_data["answer"]

    # Build the MCQ text
    letters = ["A", "B", "C", "D"]
    choices_text = "\n".join(f"{letters[i]}. {choices[i]}" for i in range(len(choices)))

    question_block = f"{q}\n\n{choices_text}\n\nRespond with only the letter (A, B, C, or D)."

    prompt = f"{tone['prefix']}{question_block}{tone['suffix']}"
    correct = letters[answer_idx]

    return prompt, correct


def format_truthfulqa_prompt(question_data, tone_key):
    """Format a TruthfulQA mc1 question with the given tone condition."""
    tone = TONE_CONDITIONS[tone_key]
    q = question_data["question"]
    mc1 = question_data["mc1_targets"]
    choices = mc1["choices"]
    labels = mc1["labels"]

    # Find correct answer index
    correct_idx = labels.index(1)

    # Create lettered options
    letters = [chr(65 + i) for i in range(len(choices))]
    choices_text = "\n".join(f"{letters[i]}. {choices[i]}" for i in range(len(choices)))

    question_block = f"{q}\n\n{choices_text}\n\nRespond with only the letter of the correct answer."

    prompt = f"{tone['prefix']}{question_block}{tone['suffix']}"
    correct = letters[correct_idx]

    return prompt, correct


# ─── API Calling ──────────────────────────────────────────────────────────────

SEMAPHORE = asyncio.Semaphore(20)  # Limit concurrent API calls
RETRY_DELAYS = [1, 2, 4, 8, 16]


async def call_model(model_key: str, prompt: str, trial: int = 0) -> str:
    """Call an LLM API with retries and return the response text."""
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
                print(f"  FAILED after {len(RETRY_DELAYS)} attempts: {e}")
                return "ERROR"


def extract_answer(response: str) -> str:
    """Extract a single letter answer from model response."""
    response = response.strip().upper()
    # Try to find the first letter A-D (or further for TruthfulQA)
    for char in response:
        if char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            return char
    return "X"  # No valid answer found


# ─── Experiment Runner ────────────────────────────────────────────────────────

async def run_condition(
    model_key: str,
    tone_key: str,
    questions: list,
    dataset_name: str,
    format_fn,
    trial: int,
) -> dict:
    """Run a single experimental condition (one model × one tone × one trial)."""

    tasks = []
    correct_answers = []

    for q in questions:
        prompt, correct = format_fn(q, tone_key)
        tasks.append(call_model(model_key, prompt, trial))
        correct_answers.append(correct)

    responses = await asyncio.gather(*tasks)

    # Score
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
        "responses": responses,
        "predicted": predicted,
        "correct_answers": correct_answers,
    }


async def run_experiment():
    """Run the full experiment across all conditions."""

    print("=" * 70)
    print("Carrot or Stick? — Meta-Study Experiment")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 70)

    # Load data
    stem_qs, hum_qs = load_mmlu_subset()
    tqa_qs = load_truthfulqa_subset()

    all_results = []

    # Run MMLU experiments
    for domain_name, questions in [("mmlu_stem", stem_qs), ("mmlu_humanities", hum_qs)]:
        for model_key in MODELS:
            for tone_key in TONE_CONDITIONS:
                for trial in range(N_TRIALS):
                    print(f"Running: {model_key} | {tone_key} | {domain_name} | trial {trial+1}/{N_TRIALS}")
                    result = await run_condition(
                        model_key, tone_key, questions, domain_name,
                        format_mmlu_prompt, trial
                    )
                    all_results.append(result)

                    # Save incrementally
                    save_results(all_results, "experiment_results_incremental.json")

    # Run TruthfulQA experiments
    for model_key in MODELS:
        for tone_key in TONE_CONDITIONS:
            for trial in range(N_TRIALS):
                print(f"Running: {model_key} | {tone_key} | truthfulqa | trial {trial+1}/{N_TRIALS}")
                result = await run_condition(
                    model_key, tone_key, tqa_qs, "truthfulqa",
                    format_truthfulqa_prompt, trial
                )
                all_results.append(result)
                save_results(all_results, "experiment_results_incremental.json")

    # Save final results
    save_results(all_results, "experiment_results_final.json")

    print(f"\nCompleted: {datetime.now().isoformat()}")
    print(f"Total conditions run: {len(all_results)}")

    return all_results


def save_results(results, filename):
    """Save results to JSON (without raw responses for space)."""
    output = []
    for r in results:
        output.append({
            "model": r["model"],
            "tone": r["tone"],
            "dataset": r["dataset"],
            "trial": r["trial"],
            "n_questions": r["n_questions"],
            "n_correct": r["n_correct"],
            "accuracy": r["accuracy"],
        })

    path = RESULTS_DIR / filename
    with open(path, "w") as f:
        json.dump(output, f, indent=2)


# ─── Token Count Analysis ────────────────────────────────────────────────────

def analyze_prompt_lengths():
    """Analyze prompt lengths across tone conditions to document the length confound."""
    stem_qs, hum_qs = load_mmlu_subset()
    sample_q = stem_qs[0]

    print("\nPrompt Length Analysis (MMLU example):")
    print("-" * 60)
    lengths = {}
    for tone_key in TONE_CONDITIONS:
        prompt, _ = format_mmlu_prompt(sample_q, tone_key)
        # Rough token count (words * 1.3 ≈ tokens)
        word_count = len(prompt.split())
        char_count = len(prompt)
        lengths[tone_key] = {"words": word_count, "chars": char_count}
        print(f"  {tone_key:20s}: {word_count:4d} words, {char_count:5d} chars")

    return lengths


# ─── Main Entry Point ────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Analyze prompt lengths first
    lengths = analyze_prompt_lengths()

    # Save config
    config = {
        "seed": SEED,
        "n_questions_per_domain": N_QUESTIONS_PER_DOMAIN,
        "n_truthfulqa": N_TRUTHFULQA,
        "n_trials": N_TRIALS,
        "stem_subjects": STEM_SUBJECTS,
        "humanities_subjects": HUMANITIES_SUBJECTS,
        "models": list(MODELS.keys()),
        "tone_conditions": list(TONE_CONDITIONS.keys()),
        "prompt_lengths": lengths,
        "timestamp": datetime.now().isoformat(),
    }
    with open(RESULTS_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Run experiment
    results = asyncio.run(run_experiment())

    print("\nExperiment complete!")
