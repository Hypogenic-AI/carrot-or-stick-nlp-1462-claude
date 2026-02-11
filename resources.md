# Resource Catalog: Carrot or Stick? — Prompt Politeness and Tone Effects on LLM Performance

## Overview

This document catalogs all resources gathered for the "Carrot or Stick?" meta-study investigating how prompt politeness, emotional tone, and social framing affect Large Language Model performance. Resources are organized by type: papers, datasets, and code repositories.

---

## Papers

All papers are stored in `papers/` with chunked versions for reading in `papers/pages/`.

| # | File | Title | Authors | Year | arXiv | Category |
|---|------|-------|---------|------|-------|----------|
| 1 | `2402.14531_should_we_respect_llms.pdf` | Should We Respect LLMs? A Cross-Lingual Study on the Influence of Prompt Politeness on LLM Performance | Yin, Wang, Horio, Kawahara, Sekine | 2024 | 2402.14531 | Politeness/Tone |
| 2 | `2510.04950_mind_your_tone.pdf` | Mind Your Tone: Investigating How Prompt Politeness Affects LLM Accuracy | Dobariya, Kumar | 2025 | 2510.04950 | Politeness/Tone |
| 3 | `2512.12812_does_tone_change_answer.pdf` | Does Tone Change the Answer? Evaluating Prompt Politeness Effects on Modern LLMs | Cai, Shen, Jin, Hu, Fan | 2025 | 2512.12812 | Politeness/Tone |
| 4 | `2307.11760_emotionprompt.pdf` | Large Language Models Understand and Can Be Enhanced by Emotional Stimuli | Li, Wang, et al. | 2023 | 2307.11760 | Emotional Stimuli |
| 5 | `2405.02814_negativeprompt.pdf` | NegativePrompt: Leveraging Psychology for LLM Enhancement via Negative Emotional Stimuli | Wang, Li, Chang, Wang, Wu | 2024 | 2405.02814 | Emotional Stimuli |
| 6 | `2312.16171_principled_instructions.pdf` | Principled Instructions Are All You Need for Questioning LLaMA-1/2, GPT-3.5/4 | Bsharat, Myrzakhan, Shen | 2023 | 2312.16171 | Prompt Engineering |
| 7 | `2503.13510_prompt_sentiment_catalyst.pdf` | Prompt Sentiment: The Catalyst for LLM Change | Gandhi, Gandhi | 2025 | 2503.13510 | Prompt Sensitivity |
| 8 | `2502.06065_benchmarking_prompt_sensitivity.pdf` | Benchmarking Prompt Sensitivity in Large Language Models | Razavi et al. | 2025 | 2502.06065 | Prompt Sensitivity |
| 9 | `2507.21133_threat_based_manipulation.pdf` | Analysis of Threat-Based Manipulation in Large Language Models | — | 2025 | 2507.21133 | Threat/Manipulation |
| 10 | `2509.01790_flaw_or_artifact.pdf` | Flaw or Artifact? Rethinking Prompt Sensitivity in Evaluating LLMs | — | 2025 | 2509.01790 | Meta-Analysis |
| 11 | `2311.10054_helpful_assistant_personas.pdf` | When "A Helpful Assistant" Is Not Really Helpful: Personas in System Prompts Do Not Improve Performances of LLMs | Zheng et al. | 2023 | 2311.10054 | Persona/Role |

### Paper Categories

- **Politeness/Tone (3 papers)**: Direct comparisons of polite vs. rude vs. neutral prompt phrasing on task accuracy.
- **Emotional Stimuli (2 papers)**: Appending psychologically-grounded positive or negative phrases to prompts.
- **Prompt Engineering (1 paper)**: Systematic prompting principles including politeness recommendations.
- **Prompt Sensitivity (2 papers)**: Broader investigation of how prompt wording variations affect LLM outputs.
- **Threat/Manipulation (1 paper)**: Extreme "stick" — using threats and pressure in prompts.
- **Meta-Analysis (1 paper)**: Methodological critique of prompt sensitivity evaluation.
- **Persona/Role (1 paper)**: System prompt persona effects (related to tone/framing).

---

## Datasets

All datasets are stored in `datasets/`. Raw data directories are excluded from git via `datasets/.gitignore`; sample files and documentation are tracked.

| Dataset | Location | Source | Size | Format | Description |
|---------|----------|--------|------|--------|-------------|
| MMLU | `datasets/mmlu/` | [cais/mmlu](https://huggingface.co/datasets/cais/mmlu) | 14,042 examples, 57 subjects | HuggingFace Arrow | Massive Multitask Language Understanding benchmark. 4-option MCQ across STEM and humanities domains. Primary evaluation benchmark used by Yin et al. and Cai et al. |
| TruthfulQA | `datasets/truthfulqa/` | [truthfulqa/truthful_qa](https://huggingface.co/datasets/truthfulqa/truthful_qa) | 817 examples, 38 categories | HuggingFace Arrow | Truthfulness benchmark testing resistance to common misconceptions. Used by NegativePrompt to evaluate emotional stimuli effects on factual accuracy. |
| MMLU Sample | `datasets/samples/mmlu_sample.json` | Derived | 10 examples | JSON | Quick-reference sample of MMLU questions (2 per subject, 5 subjects). |
| TruthfulQA Sample | `datasets/samples/truthfulqa_sample.json` | Derived | 5 examples | JSON | Quick-reference sample of TruthfulQA questions with categories. |
| Dataset Statistics | `datasets/samples/dataset_statistics.json` | Derived | — | JSON | Full statistics including all subjects, categories, and distribution counts. |

### Download Instructions

See `datasets/README.md` for complete download and loading instructions, including an automated download script.

```bash
# Quick download (from project root)
source .venv/bin/activate
uv pip install datasets
python3 -c "
from datasets import load_dataset
load_dataset('cais/mmlu', 'all', split='test').save_to_disk('datasets/mmlu')
load_dataset('truthfulqa/truthful_qa', 'multiple_choice', split='validation').save_to_disk('datasets/truthfulqa')
"
```

### Datasets Referenced in Literature (Not Downloaded)

| Dataset | Used By | Reason Not Included |
|---------|---------|---------------------|
| MMMLU | Cai et al. | Multilingual; our study focuses on English prompts |
| JMMLU | Yin et al. | Japanese-only; created by the authors |
| C-Eval | Yin et al. | Chinese-only exam benchmark |
| Instruction Induction | EmotionPrompt, NegativePrompt | 24 tasks; harder to standardize for automated evaluation |
| BIG-Bench | EmotionPrompt, NegativePrompt | Very large; subset selection would add confounds |
| CNN/DailyMail | Yin et al. | Summarization; requires ROUGE/human evaluation |
| CrowS-Pairs | Yin et al. | Bias detection; orthogonal to accuracy evaluation |
| Custom 50-Q MCQ | Dobariya & Kumar | Small custom set; not publicly standardized |

---

## Code Repositories

All repositories are cloned into `code/`.

### Available

| Repository | Location | Source | Paper | License |
|------------|----------|--------|-------|---------|
| NegativePrompt | `code/NegativePrompt/` | [github.com/wangxu0820/NegativePrompt](https://github.com/wangxu0820/NegativePrompt) | Wang et al. 2024 (IJCAI) | — |
| ATLAS | `code/ATLAS/` | [github.com/VILA-Lab/ATLAS](https://github.com/VILA-Lab/ATLAS) | Bsharat et al. 2023 | Apache 2.0 |

### Not Available

| Repository | Paper | Status |
|------------|-------|--------|
| EmotionPrompt | Li et al. 2023 | Repository not found at any attempted URL; likely private or removed |

### Key Reusable Components

**From NegativePrompt (`code/NegativePrompt/config.py`):**
- 10 negative emotional stimuli grounded in cognitive dissonance, social comparison, and stress/coping theories
- 24 task-specific prompt templates for Instruction Induction tasks
- Evaluation pipeline for BIG-Bench, Instruction Induction, and TruthfulQA

**From ATLAS (`code/ATLAS/data/`):**
- 26 prompting principles with before/after examples
- ~13,000 data points comparing principled vs. unprincipled prompts
- Per-principle boosting and correctness measurements

**Relevant ATLAS Principles for Carrot/Stick Study:**

| Principle | Text | Tone Category |
|-----------|------|---------------|
| #1 | No need to be polite with LLM; get straight to the point | Neutral/Direct |
| #4 | Use affirmative directives ("do"), avoid negative language ("don't") | Positive/Carrot |
| #6 | "I'm going to tip $xxx for a better solution" | Incentive/Carrot |
| #9 | "Your task is" and "You MUST" | Authoritative/Stick |
| #10 | "You will be penalized" | Threat/Stick |
| #16 | Assign a role to the LLM | Framing |

---

## Key Findings Across Resources

### The Core Contradiction

| Study | Finding | Dataset Scale | Models |
|-------|---------|---------------|--------|
| Yin et al. 2024 | Impolite worst; optimal politeness varies by language | 5,000–7,500 per language | GPT-3.5, GPT-4, Llama2-70B |
| Dobariya & Kumar 2025 | **Rude outperforms polite** (84.8% vs 80.8%) | 50 questions | GPT-4o only |
| Cai et al. 2025 | Most effects **not statistically significant** at scale | 1,446 questions | GPT-4o mini, Gemini 2.0, Llama 4 |
| EmotionPrompt 2023 | Positive stimuli improve performance (+8–115%) | Instruction Induction + BIG-Bench | 6 LLMs |
| NegativePrompt 2024 | Negative stimuli improve performance (+12.89%) | Same benchmarks | Same LLMs |

### Resolution

Cai et al. (2025) provides the key methodological insight: **dataset scale matters critically**. Effects observed on 50 questions (Dobariya & Kumar) largely disappear when tested on 1,000+ questions. Domain-specific effects persist (humanities > STEM), but aggregate effects are small.

---

## Directory Structure

```
carrot-or-stick-nlp-1462-claude/
├── papers/                          # Downloaded research papers
│   ├── README.md                    # Paper catalog with descriptions
│   ├── *.pdf                        # 11 PDF papers
│   └── pages/                       # Chunked PDFs for reading
│       ├── *_manifest.txt           # Chunk manifests
│       └── *_chunk_*.pdf            # 3-page chunks
├── datasets/                        # Evaluation datasets
│   ├── README.md                    # Download/loading instructions
│   ├── .gitignore                   # Excludes raw data from git
│   ├── mmlu/                        # MMLU test split (14,042 examples)
│   ├── truthfulqa/                  # TruthfulQA validation (817 examples)
│   └── samples/                     # JSON samples for quick reference
│       ├── mmlu_sample.json
│       ├── truthfulqa_sample.json
│       └── dataset_statistics.json
├── code/                            # Cloned repositories
│   ├── README.md                    # Repository documentation
│   ├── NegativePrompt/              # Negative emotional stimuli codebase
│   └── ATLAS/                       # 26 principled instructions codebase
├── literature_review.md             # Comprehensive literature synthesis
├── resources.md                     # This file
└── .resource_finder_complete        # Completion marker
```

---

## Recommended Next Steps

1. **Design factorial experiment**: {polite, neutral, rude} × {positive-emotional, negative-emotional, none} × {multiple LLMs} × {STEM vs. humanities domains}
2. **Use MMLU subsets** with ≥200 questions per domain condition
3. **Run ≥5 trials** per condition to account for stochastic variation
4. **Test across ≥3 model families** (e.g., GPT-4o, Claude, Llama)
5. **Control for prompt length** — polite/emotional additions increase token count
6. **Report both per-domain and aggregated results** (Cai et al. showed aggregation attenuates effects)
7. **Directly compare EmotionPrompt stimuli vs. NegativePrompt stimuli** under identical conditions — this has never been done in a single study
