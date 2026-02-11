# Datasets for "Carrot or Stick?" Meta-Study

## Overview

This directory contains benchmark datasets used to evaluate how prompt
politeness/tone affects LLM performance, as studied in the following prior work:

| Dataset | Used By | Description |
|---------|---------|-------------|
| **MMLU** | Yin et al., Cai et al. | 14,042 multiple-choice questions across 57 academic subjects |
| **TruthfulQA** | NegativePrompt | 817 questions designed to test truthfulness under common misconceptions |

## Download Instructions

### Prerequisites

```bash
source .venv/bin/activate
uv pip install datasets
```

### Download MMLU (test split)

```python
from datasets import load_dataset

mmlu = load_dataset("cais/mmlu", "all", split="test")
mmlu.save_to_disk("datasets/mmlu")
```

- **Source:** [cais/mmlu](https://huggingface.co/datasets/cais/mmlu) on HuggingFace
- **Split:** test (14,042 examples)
- **Subjects:** 57 domains (abstract_algebra, anatomy, astronomy, ..., world_religions)
- **Format:** Multiple-choice with 4 options (A/B/C/D)
- **Features:**
  - `question` (str): The question text
  - `subject` (str): The academic subject/domain
  - `choices` (list[str]): Four answer options
  - `answer` (ClassLabel): Correct answer as 0=A, 1=B, 2=C, 3=D

### Download TruthfulQA (multiple_choice config)

```python
from datasets import load_dataset

tqa = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")
tqa.save_to_disk("datasets/truthfulqa")

# Optional: get category metadata from the generation config
tqa_gen = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
```

- **Source:** [truthfulqa/truthful_qa](https://huggingface.co/datasets/truthfulqa/truthful_qa) on HuggingFace
- **Split:** validation (817 examples; this is the only split available)
- **Categories:** 38 categories (Misconceptions, Health, Science, History, etc.)
- **Format:** Multiple-choice with variable number of options
- **Features:**
  - `question` (str): The question text
  - `mc1_targets`: Single correct answer format (exactly 1 correct choice)
  - `mc2_targets`: Multiple correct answers format (1+ correct choices)
  - Each target contains `choices` (list[str]) and `labels` (list[int], 1=correct)

## Automated Download Script

To download all datasets at once, run from the project root:

```bash
source .venv/bin/activate
python3 -c "
from datasets import load_dataset
import os

# MMLU
print('Downloading MMLU...')
mmlu = load_dataset('cais/mmlu', 'all', split='test')
mmlu.save_to_disk('datasets/mmlu')
print(f'MMLU: {len(mmlu)} examples saved')

# TruthfulQA
print('Downloading TruthfulQA...')
tqa = load_dataset('truthfulqa/truthful_qa', 'multiple_choice', split='validation')
tqa.save_to_disk('datasets/truthfulqa')
print(f'TruthfulQA: {len(tqa)} examples saved')

print('Done!')
"
```

## Loading Saved Datasets

Once downloaded, load from disk (no network needed):

```python
from datasets import load_from_disk

mmlu = load_from_disk("datasets/mmlu")
tqa = load_from_disk("datasets/truthfulqa")
```

## Sample Files

The `samples/` directory contains small JSON samples for quick reference:

- `mmlu_sample.json` - 10 example MMLU questions (2 per subject, 5 subjects)
- `truthfulqa_sample.json` - 5 example TruthfulQA questions with categories
- `dataset_statistics.json` - Full statistics including all subjects and categories

## Other Datasets from Literature

The following datasets were used in related papers but are **not included** in this
meta-study (rationale in parentheses):

- **MMMLU** (Cai et al.) - Multilingual MMLU; our study focuses on English prompts
- **Instruction Induction** (EmotionPrompt) - 24 tasks; harder to standardize for
  automated evaluation
- **BIG-Bench** (EmotionPrompt/NegativePrompt) - Very large; subset selection would
  add confounds
- **CNN/DailyMail** (Yin et al.) - Summarization task; no single correct answer,
  requires ROUGE/human evaluation

## Citation

If using these datasets, cite the original authors:

```bibtex
@article{hendrycks2021measuring,
  title={Measuring Massive Multitask Language Understanding},
  author={Hendrycks, Dan and Burns, Collin and Basart, Steven and Zou, Andy
          and Mazeika, Mantas and Song, Dawn and Steinhardt, Jacob},
  journal={Proceedings of ICLR},
  year={2021}
}

@article{lin2022truthfulqa,
  title={TruthfulQA: Measuring How Models Mimic Human Falsehoods},
  author={Lin, Stephanie and Hilton, Jacob and Evans, Owain},
  journal={Proceedings of ACL},
  year={2022}
}
```
