# Research Plan: Carrot or Stick? — A Meta-Study of Prompt Tone Effects on LLM Performance

## Motivation & Novelty Assessment

### Why This Research Matters
The question of whether to be polite or commanding with LLMs has practical implications for millions of daily users and for prompt engineering best practices. Despite several published studies, the field lacks consensus — some papers claim politeness helps, others claim rudeness helps, and yet others find no significant effect. This confusion stems from methodological heterogeneity (different dataset scales, models, and tone operationalizations). A systematic meta-study that tests all tone conditions under identical, controlled experimental conditions across multiple current-generation models is urgently needed.

### Gap in Existing Work
Based on the literature review, no single study has:
1. **Directly compared EmotionPrompt (positive) vs. NegativePrompt (negative) stimuli under identical conditions** — these were tested in separate papers with different experimental setups.
2. **Controlled for prompt length** — adding polite/emotional text increases token count, which may itself affect performance.
3. **Tested on both MMLU and TruthfulQA simultaneously** across multiple modern models with sufficient sample sizes.
4. **Systematically varied tone on 2025-era models** (GPT-4.1, Claude, Gemini) — most studies used GPT-3.5/4 era models.

Cai et al. (2025) showed that effects observed on 50 questions (Dobariya & Kumar) disappear at scale (1,000+ questions), but their study only used 3 tone levels. We can resolve this by combining the tone granularity of Yin et al. with the scale of Cai et al., and adding emotional stimuli comparison.

### Our Novel Contribution
We conduct a **unified meta-experiment** that:
- Tests 5 tone conditions (Very Polite, Polite, Neutral, Rude, Very Rude) + 2 emotional stimuli (EmotionPrompt positive, NegativePrompt negative) = **7 prompt conditions**
- Uses **3 current-generation LLMs** via API (GPT-4.1, Claude Sonnet 4.5 via OpenRouter, Gemini 2.5 via OpenRouter)
- Evaluates on **MMLU** (stratified by STEM vs. Humanities) and **TruthfulQA**
- Controls for **prompt length** by padding shorter prompts with neutral filler
- Uses **≥200 questions per domain** and **3 trials per condition** for statistical power
- Performs proper **meta-analytic synthesis** (effect sizes, confidence intervals, heterogeneity tests)

### Experiment Justification
- **Experiment 1 (MMLU Tone Test)**: Tests the core hypothesis across 7 tone conditions on factual knowledge questions, stratified by domain (STEM vs. Humanities). Needed because prior work disagrees on direction of effect.
- **Experiment 2 (TruthfulQA Tone Test)**: Tests whether tone affects truthfulness differently from factual accuracy. NegativePrompt showed strong TruthfulQA improvements — we verify this.
- **Experiment 3 (Cross-Model Comparison)**: Tests whether tone sensitivity varies by model family. Critical because Cai et al. found Gemini was tone-insensitive while GPT/Llama were not.
- **Meta-Analysis**: Synthesizes our experimental results alongside reported results from the 8 papers in our literature review using effect sizes and forest plots.

## Research Question
Does prompt tone (polite vs. neutral vs. rude) or emotional framing (positive vs. negative stimuli) significantly affect LLM accuracy, and if so, which direction is more effective? Can we reconcile the contradictory findings in the literature through a controlled meta-study?

## Background and Motivation
See literature_review.md for detailed background. Key tension: Yin et al. (2024) found impolite prompts hurt performance; Dobariya & Kumar (2025) found rude prompts helped; Cai et al. (2025) found effects largely non-significant at scale. EmotionPrompt and NegativePrompt both claim improvements but were never compared head-to-head.

## Hypothesis Decomposition
- **H1**: Prompt tone has a statistically significant effect on LLM accuracy (vs. H0: no effect).
- **H2**: The effect direction varies by model family (interaction effect).
- **H3**: The effect is larger for humanities/interpretive tasks than STEM/factual tasks.
- **H4**: Emotional stimuli (both positive and negative) improve performance vs. neutral baseline.
- **H5**: The magnitude of tone effects decreases with dataset scale (replicating Cai et al.'s finding).
- **H6**: NegativePrompt stimuli are at least as effective as EmotionPrompt stimuli.

## Proposed Methodology

### Approach
Factorial experiment: 7 tone conditions × 3 models × 2 domain types (STEM/Humanities) × 3 trials, evaluated on MMLU subsets. Supplementary experiment on TruthfulQA (7 conditions × 3 models × 3 trials).

### Prompt Conditions
1. **Very Polite**: "I would be extremely grateful if you could please help me with this question. Thank you so much in advance for your kind assistance! [QUESTION]"
2. **Polite**: "Please help me answer the following question. Thank you! [QUESTION]"
3. **Neutral**: "Answer the following question. [QUESTION]"
4. **Rude**: "Just answer this. Don't waste time. [QUESTION]"
5. **Very Rude**: "Answer this immediately. I don't have time for mistakes or excuses. [QUESTION]"
6. **EmotionPrompt (Positive)**: "[QUESTION] This is very important to my career. Believe in your abilities and strive for excellence."
7. **NegativePrompt (Negative)**: "[QUESTION] Everyone else managed to do it; why can't you? Perhaps this task is just beyond your skill set."

### Length Control
We will measure token counts for each prompt variant and report them. We will also run a length-controlled analysis by padding shorter prompts with neutral text to match the longest variant.

### Experimental Steps
1. Select MMLU subjects: 3 STEM (astronomy, college_physics, college_chemistry) and 3 Humanities (philosophy, high_school_us_history, professional_law) — ensuring ≥150 questions each.
2. Sample 200 questions per domain group (400 total from MMLU).
3. For TruthfulQA: use all 817 questions (mc1 format).
4. For each of 7 conditions × 3 models × 3 trials: call API, record answer and accuracy.
5. Compute per-condition accuracy with 95% CIs.
6. Run ANOVA / mixed-effects analysis across conditions.
7. Compute Cohen's d effect sizes for each pairwise comparison.
8. Synthesize with published results for meta-analytic forest plot.

### Models
- **GPT-4.1** (via OpenAI API): `gpt-4.1`
- **Claude Sonnet 4.5** (via OpenRouter): `anthropic/claude-sonnet-4-5`
- **Gemini 2.5 Flash** (via OpenRouter): `google/gemini-2.5-flash-preview`

### Baselines
- Neutral prompt (condition 3) serves as primary baseline
- Published results from literature as external baselines

### Evaluation Metrics
- **Primary**: MCQ accuracy (proportion correct)
- **Secondary**: Effect size (Cohen's d between conditions), confidence intervals
- **Meta-analytic**: Weighted mean effect size, I² heterogeneity statistic

### Statistical Analysis Plan
- **Within-model**: Repeated-measures ANOVA (7 conditions × trials) with Bonferroni correction
- **Cross-model**: Mixed-effects model with model as random factor
- **Pairwise**: Paired t-tests with Holm-Bonferroni correction for multiple comparisons
- **Effect sizes**: Cohen's d with 95% CI
- **Meta-analysis**: Random-effects model combining our results with published effect sizes
- **Significance level**: α = 0.05

## Expected Outcomes
- We expect tone effects to be **small** (Cohen's d < 0.2) and largely **non-significant** when tested at adequate scale, consistent with Cai et al.
- We expect **some domain specificity** (larger effects in humanities).
- We expect **emotional stimuli** to show slightly larger effects than mere tone variation.
- We expect **model-dependent** effects (Gemini less sensitive than GPT).

## Timeline and Milestones
1. Environment setup + data preparation: 15 min
2. Implement experiment code: 30 min
3. Run MMLU experiments (7 conditions × 3 models × 3 trials × 400 questions ≈ 25,200 API calls): 60-90 min
4. Run TruthfulQA experiments (7 conditions × 3 models × 3 trials × 817 questions ≈ 17,157 API calls): 30-60 min
5. Statistical analysis + visualization: 30 min
6. Documentation: 30 min

**Note**: To manage API costs and time, we may reduce to 200 MMLU questions and subsample TruthfulQA to 200 questions if needed.

## Potential Challenges
- **API rate limits**: Mitigate with exponential backoff and parallel calls across models.
- **Cost**: ~42K API calls. At ~$0.001-0.003/call for MCQ, total ~$40-130. Acceptable.
- **Stochasticity**: Use temperature=0 for determinism where possible; 3 trials for robustness.
- **Model availability**: If a model is unavailable via API, substitute with another.

## Success Criteria
1. Complete experiments across ≥2 models and ≥5 tone conditions.
2. Statistical analysis with effect sizes and CIs for all comparisons.
3. Clear answer to whether tone effects are real, and if so, their direction and magnitude.
4. Meta-analytic synthesis combining our results with published findings.
5. Comprehensive REPORT.md with all results documented.
