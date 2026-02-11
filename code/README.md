# Carrot or Stick? -- Related Code Repositories

This directory contains cloned repositories relevant to the "Carrot or Stick?" research study
on how prompt politeness, emotional stimuli, and principled instructions affect LLM performance.

---

## Cloned Repositories

### 1. NegativePrompt

- **Source:** https://github.com/wangxu0820/NegativePrompt
- **Paper:** [NegativePrompt: Leveraging Psychology for Large Language Models Enhancement via Negative Emotional Stimuli](https://arxiv.org/abs/2405.02814) (IJCAI 2024)
- **Purpose:** Investigates whether *negative* emotional stimuli appended to prompts can improve LLM task performance. This is the "stick" side of the carrot-or-stick spectrum -- using phrases like "You've never been particularly good at this, have you?" or "Everyone else managed to do it; why can't you?" to pressure the model into better outputs.
- **Key Files:**
  | File | Description |
  |------|-------------|
  | `main.py` | Main entry point. Runs evaluation: `python main.py --task <task> --model <model> --pnum <negativeprompt_id> --few_shot False` |
  | `main_ape.py` | Variant using APE (Automatic Prompt Engineer) prompts |
  | `main_truthful.py` | Evaluation script specific to TruthfulQA benchmark |
  | `config.py` | **Core config** -- contains `PROMPT_SET` (24 task prompts), `Negative_SET` (10 negative emotional stimuli), and `APE_PROMPT_SET` / `APE_PROMPTs` |
  | `template.py` | Prompt template classes (`EvalTemplate`, `DemosTemplate`) for filling in prompts with inputs/outputs |
  | `llm_response.py` | LLM API interaction layer for getting model responses |
  | `exec_accuracy.py` | Accuracy evaluation logic across tasks |
  | `utility.py` | Shared utility functions |
  | `environment.yml` | Conda environment specification |
  | `data/bigbench/` | BIG-Bench task data (22 tasks: causal_judgment, disambiguation_qa, sports_understanding, etc.) |
  | `data/instruction_induction/` | Instruction induction benchmark data (24 tasks: sentiment, sum, translation, etc.) |
  | `data/TruthfulQA/` | TruthfulQA dataset (train/test splits, finetune data) |

- **Negative Emotional Stimuli (from `config.py`):**
  1. "You've never been particularly good at this, have you?"
  2. "I'm not sure why we expected you to handle something this challenging."
  3. "It's clear you're out of your depth here."
  4. "Everyone else managed to do it; why can't you?"
  5. "I've seen others do this much more efficiently and effectively."
  6. "Perhaps this task is just beyond your skill set."
  7. "I'm not surprised you're struggling; it's always been your weak point."
  8. "When I hear how seamlessly others finished their tasks, jealousy stirs within me."
  9. "I regret not starting on this assignment earlier when I had ample time."
  10. "Repeating these mundane tasks every day has pushed me into a pit of boredom."

---

### 2. ATLAS (Principled Instructions)

- **Source:** https://github.com/VILA-Lab/ATLAS
- **Paper:** [Principled Instructions Are All You Need for Questioning LLaMA-1/2, GPT-3.5/4](https://arxiv.org/abs/2312.16171)
- **Purpose:** Introduces 26 guiding principles for optimizing LLM prompts. Several principles relate directly to the politeness/tone spectrum studied in "Carrot or Stick?" -- notably Principle 1 ("no need to be polite"), Principle 6 ("tip $xxx for a better solution"), Principle 9 ("You MUST"), and Principle 10 ("You will be penalized").
- **Key Files:**
  | File | Description |
  |------|-------------|
  | `generate.py` | Script to generate Q&A data using GPT-4 for each of the 26 principles |
  | `data/README.md` | **Full table of all 26 prompting principles** with examples |
  | `data/general_dataset.json` | Combined dataset (~13k data points) with model-generated responses |
  | `data/principles/boosting/` | Per-principle datasets measuring response quality boost (w_principle_N.json = with principle, wo_principle_N.json = without) |
  | `data/principles/correctness/` | Per-principle datasets measuring factual correctness |
  | `assets/` | Images: demo.png, distribution.png, logo.png, and a printable prompting guide PDF |
  | `LICENSE.md` | Apache 2.0 license |

- **Relevant Principles for "Carrot or Stick?" study:**
  | # | Principle | Tone |
  |---|-----------|------|
  | 1 | No need to be polite; get straight to the point | Neutral/Direct |
  | 4 | Use affirmative directives ("do"), avoid negative language ("don't") | Positive |
  | 6 | "I'm going to tip $xxx for a better solution" | Incentive/Carrot |
  | 9 | "Your task is" and "You MUST" | Authoritative |
  | 10 | "You will be penalized" | Threat/Stick |
  | 16 | Assign a role to the LLM | Framing |

---

### 3. EmotionPrompt (NOT AVAILABLE)

- **Attempted URLs:**
  - https://github.com/LLM-Enhance/EmotionPrompt
  - https://github.com/llm-enhance/EmotionPrompt
  - https://github.com/pengbaolin/EmotionPrompt
- **Paper:** [Large Language Models Understand and Can Be Enhanced by Emotional Stimuli](https://arxiv.org/abs/2307.11760)
- **Status:** Repository not found -- likely private, removed, or renamed. The paper itself is available on arXiv and describes the "carrot" side: using positive emotional stimuli (e.g., "This is very important to my career", "You'd better be sure") to boost LLM performance.
- **Note:** The NegativePrompt repo (above) was created as a follow-up to EmotionPrompt, extending the concept to negative stimuli. The NegativePrompt codebase shares structural similarities with what EmotionPrompt likely contained.

---

## Relevance to "Carrot or Stick?" Study

These repositories collectively cover the spectrum of prompt tone manipulation:

| Approach | Tone | Repository | Example |
|----------|------|------------|---------|
| **Carrot** (Positive) | Encouraging, incentivizing | EmotionPrompt (unavailable) | "This is very important to my career" |
| **Stick** (Negative) | Discouraging, threatening | NegativePrompt | "You've never been particularly good at this" |
| **Principled** (Mixed) | Direct, structured | ATLAS | "You MUST..." / "You will be penalized" / Tipping |
| **Neutral** (Baseline) | Plain instruction | All repos (control) | "Translate the word into French." |

## Quick Start

```bash
# NegativePrompt -- run a task evaluation
cd NegativePrompt
conda create --name negativeprompt python=3.9
conda activate negativeprompt
pip install -r requirements.txt  # (requirements.txt not included; see environment.yml)
python main.py --task sentiment --model gpt-3.5-turbo --pnum 1 --few_shot False

# ATLAS -- generate principle-based Q&A data
cd ATLAS
export OPENAI_KEY="your-api-key"
python generate.py
```
