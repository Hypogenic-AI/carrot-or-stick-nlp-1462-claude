# Carrot or Stick? A Meta-Study of Prompt Tone Effects on LLM Performance

Does being polite to LLMs improve their accuracy, or is it better to be strict and commanding? This project runs a controlled meta-experiment to answer that question definitively.

## Key Findings

- **Tone effects are real but strongly model-dependent**: GPT-4.1 and Gemini 2.5 Flash are robust to tone (±1-3%), while Claude Sonnet 4.5 shows up to ±12% accuracy swings
- **Neither "carrot" nor "stick" is universally better**: Rude prompts slightly outperform polite ones on average (+1.3% vs. −1.2%), but the effect is small and inconsistent
- **The literature disagrees because different studies test different models**: Model heterogeneity, not methodological error, is the primary cause of contradictory findings
- **Emotional prompt suffixes can disrupt instruction following**: EmotionPrompt-style suffixes caused Claude to begin explaining instead of answering, creating misleading accuracy drops
- **Practical recommendation**: Write clear, direct prompts. Don't waste tokens on excessive politeness or rudeness

## Experiment Design

- **7 tone conditions**: Very Polite, Polite, Neutral, Rude, Very Rude, EmotionPrompt (positive), NegativePrompt (negative)
- **3 models**: GPT-4.1, Claude Sonnet 4.5, Gemini 2.5 Flash
- **3 benchmarks**: MMLU STEM (200Q), MMLU Humanities (200Q), TruthfulQA (200Q)
- **189 experimental runs** (7 × 3 × 3 × 3 trials), ~37,800 API calls total

## Project Structure

```
├── REPORT.md                    # Full research report with results
├── planning.md                  # Research plan and methodology
├── literature_review.md         # Comprehensive literature review (11 papers)
├── resources.md                 # Resource catalog
├── src/
│   ├── experiment.py            # Main experiment script
│   ├── experiment_continue.py   # Continuation script (fixed model IDs)
│   └── analysis.py              # Statistical analysis and visualization
├── results/
│   ├── experiment_results_final.json    # All 189 result records
│   ├── analysis_summary.json            # Statistical summary
│   ├── config.json                      # Experiment configuration
│   └── plots/                           # All visualizations
│       ├── accuracy_by_tone.png
│       ├── effect_sizes_forest.png
│       ├── heatmap_mmlu_stem.png
│       ├── heatmap_mmlu_humanities.png
│       ├── heatmap_truthfulqa.png
│       ├── domain_comparison.png
│       └── meta_forest_plot.png
├── datasets/                    # MMLU and TruthfulQA (HuggingFace Arrow format)
├── papers/                      # 11 research papers (PDFs)
└── code/                        # Cloned repos (NegativePrompt, ATLAS)
```

## How to Reproduce

```bash
# 1. Create and activate environment
uv venv && source .venv/bin/activate

# 2. Install dependencies
uv add openai httpx numpy scipy pandas matplotlib seaborn datasets tqdm

# 3. Set API keys
export OPENAI_API_KEY="your-key"
export OPENROUTER_KEY="your-key"

# 4. Download datasets (if not present)
python -c "
from datasets import load_dataset
load_dataset('cais/mmlu', 'all', split='test').save_to_disk('datasets/mmlu')
load_dataset('truthfulqa/truthful_qa', 'multiple_choice', split='validation').save_to_disk('datasets/truthfulqa')
"

# 5. Run experiment
python src/experiment_continue.py

# 6. Run analysis
python src/analysis.py
```

## See Also

- [REPORT.md](REPORT.md) for the full research report
- [planning.md](planning.md) for experimental design rationale
- [literature_review.md](literature_review.md) for background on all 11 papers reviewed
