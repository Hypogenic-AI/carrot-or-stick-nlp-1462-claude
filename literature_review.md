# Literature Review: Carrot or Stick? — How Prompt Politeness and Tone Affect LLM Performance

## Research Area Overview

A growing body of research investigates how the pragmatic aspects of prompts — particularly politeness, emotional tone, and social framing — influence Large Language Model (LLM) performance. This "Carrot or Stick?" question asks whether polite/encouraging prompts ("carrots") or strict/demanding prompts ("sticks") yield better results from LLMs. The existing empirical evidence is notably **mixed and contradictory**, making this an ideal candidate for a meta-study.

The research spans three main threads:
1. **Politeness/tone studies** — directly testing polite vs. rude prompt phrasing
2. **Emotional stimuli studies** — appending positive or negative psychological phrases to prompts
3. **Prompt sensitivity studies** — broader investigation of how prompt wording affects LLM behavior

---

## Key Papers

### Paper 1: Should We Respect LLMs? (Yin et al., 2024)
- **Authors**: Ziqi Yin, Hao Wang, Kaito Horio, Daisuke Kawahara, Satoshi Sekine
- **Venue**: SICon 2024 (ACL Workshop), arXiv:2402.14531
- **Key Contribution**: First cross-lingual study of prompt politeness effects on LLMs across English, Chinese, and Japanese.
- **Methodology**: Designed 8 prompts per language at politeness levels from 1 (impolite) to 8 (very polite). Validated with native speaker questionnaires. Tested on GPT-3.5, GPT-4, Llama2-70B, ChatGLM3, Swallow-70B.
- **Tasks**: Summarization (CNN/DailyMail, XL-Sum), language understanding (MMLU, C-Eval, JMMLU), stereotypical bias detection (CrowS-Pairs, CHBias).
- **Datasets Used**: MMLU (5,700 questions), C-Eval (5,200), JMMLU (7,536 — constructed by the authors), CNN/DailyMail, XL-Sum, CrowS-Pairs, CHBias.
- **Results**:
  - Impolite prompts often result in poor performance (especially level 1 — most rude)
  - Overly polite language does NOT guarantee better outcomes
  - Optimal politeness level varies by language: English GPT-3.5 peaked at level 8; Japanese GPT-4 peaked at level 4; Chinese GPT-4 peaked at levels 4-6
  - Llama2-70B showed scores nearly proportional to politeness levels
  - RLHF/SFT amplifies sensitivity to politeness (base model less sensitive)
  - Bias increases at extreme politeness levels (both very polite and very rude)
- **Code Available**: No public repository found
- **Relevance**: Foundational paper; most comprehensive study. Created JMMLU benchmark. Shows language/culture dependence.

### Paper 2: Mind Your Tone (Dobariya & Kumar, 2025)
- **Authors**: Om Dobariya, Akhil Kumar (Penn State)
- **Venue**: arXiv:2510.04950
- **Key Contribution**: Found that **rude prompts outperform polite ones** on ChatGPT-4o — contradicting Yin et al.
- **Methodology**: 50 base MCQ questions (math, science, history) × 5 tone variants (Very Polite → Very Rude) = 250 prompts. 10 runs per tone. Paired sample t-tests.
- **Datasets Used**: Custom dataset of 50 MCQ questions generated via ChatGPT Deep Research.
- **Results**:
  - Very Polite: 80.8%, Polite: 81.4%, Neutral: 82.2%, Rude: 82.8%, **Very Rude: 84.8%**
  - Most pairwise differences statistically significant (p < 0.05)
  - Very Rude significantly outperformed all other tones
- **Limitations**: Small dataset (50 questions), single model (GPT-4o), English only
- **Code Available**: Anonymous GitHub (dataset and code)
- **Relevance**: Key contradictory finding; suggests newer LLMs may respond differently to tone.

### Paper 3: Does Tone Change the Answer? (Cai et al., 2025)
- **Authors**: Hanyu Cai, Binqi Shen, Lier Jin, Lan Hu, Xiaojing Fan
- **Venue**: arXiv:2512.12812
- **Key Contribution**: Largest cross-model study: GPT-4o mini, Gemini 2.0 Flash, Llama 4 Scout on MMMLU.
- **Methodology**: 3 tone variants (Very Friendly, Neutral, Very Rude) on 6 MMMLU tasks (3 STEM, 3 Humanities). 10 trials per question per tone. Mean differences + 95% CI + pairwise t-tests.
- **Datasets Used**: MMMLU benchmark (Anatomy: 135, Astronomy: 152, College Biology: 144, US History: 204, Philosophy: 311, Professional Law: 500 questions).
- **Results**:
  - 27/36 comparisons favor Neutral/Friendly over Rude (directionally)
  - Only 4 comparisons reached statistical significance — all in Humanities
  - GPT and Llama show some tone sensitivity; **Gemini is tone-insensitive**
  - Effects **diminish substantially** when aggregated across domains
  - Very Friendly does NOT always outperform Neutral
- **Code Available**: No
- **Relevance**: Resolves contradictions — dataset scale matters. Small datasets (50 questions) amplify tone effects that disappear at scale.

### Paper 4: EmotionPrompt (Li et al., 2023)
- **Authors**: Cheng Li, Jindong Wang, et al. (Microsoft, CAS, BNU)
- **Venue**: arXiv:2307.11760 (NeurIPS 2023-adjacent)
- **Key Contribution**: Proposed appending emotional stimuli to prompts, showing LLMs can be enhanced by psychological phrases.
- **Methodology**: 11 emotional stimuli designed from 3 psychological theories (self-monitoring, social cognitive theory, cognitive emotion regulation). Tested on 6 LLMs × 45 tasks. Human study with 106 participants.
- **Emotional Stimuli Examples**:
  - EP01: "Write your answer and give me a confidence score between 0-1"
  - EP02: "This is very important to my career"
  - EP03: "You'd better be sure"
  - EP07: "Believe in your abilities and strive for excellence"
  - EP08: "Embrace challenges as opportunities for growth"
- **Datasets Used**: Instruction Induction (24 tasks), BIG-Bench (21 tasks), TruthfulQA
- **Results**:
  - 8.00% relative improvement on Instruction Induction
  - 115% relative improvement on BIG-Bench
  - 10.9% improvement on generative tasks (human evaluation)
  - EP02 ("important to my career") most effective on Instruction Induction; EP06 (compound) best on BIG-Bench
  - Works across 6 LLMs (T5, Vicuna, Llama 2, BLOOM, ChatGPT, GPT-4)
- **Code Available**: https://github.com/LLM-Enhance/EmotionPrompt
- **Relevance**: Demonstrates "carrot" approach works. Provides reusable emotional stimuli templates.

### Paper 5: NegativePrompt (Wang et al., 2024)
- **Authors**: Xu Wang, Cheng Li, Yi Chang, Jindong Wang, Yuan Wu
- **Venue**: IJCAI 2024, arXiv:2405.02814
- **Key Contribution**: Shows **negative** emotional stimuli ("stick") also improve LLM performance — sometimes more than positive ones.
- **Methodology**: 10 negative stimuli from 3 theories (cognitive dissonance, social comparison, stress/coping). Same experimental setup as EmotionPrompt.
- **Negative Stimuli Examples**:
  - NP01: "You've never been particularly good at this, have you?"
  - NP04: "Perhaps this task is just beyond your skill set"
  - NP06: "Everyone else managed to do it; why can't you?"
  - NP09: "I regret not starting on this assignment earlier"
- **Datasets Used**: Instruction Induction (24 tasks), BIG-Bench (21 tasks), TruthfulQA
- **Results**:
  - 12.89% relative improvement on Instruction Induction (higher than EmotionPrompt's 8%)
  - 46.25% improvement on BIG-Bench
  - Significantly enhances truthfulness on TruthfulQA
  - On some individual tasks, NegativePrompt outperforms EmotionPrompt
- **Code Available**: https://github.com/wangxu0820/NegativePrompt
- **Relevance**: Direct evidence that "stick" can work as well or better than "carrot."

### Paper 6: Principled Instructions (Bsharat et al., 2023)
- **Authors**: Sondos Mahmoud Bsharat, Aidar Myrzakhan, Zhiqiang Shen (MBZUAI)
- **Venue**: arXiv:2312.16171
- **Key Contribution**: 26 guiding principles for prompting LLMs. Principle 1: "No need to be polite with LLM."
- **Methodology**: 13K data points, human evaluation comparing principled vs. unprincipled prompts across LLaMA-1/2, GPT-3.5/4.
- **Results**: Removing politeness ("please", "thank you") improved responses by ~5%.
- **Code Available**: https://github.com/VILA-Lab/ATLAS
- **Relevance**: Supports the "neutral is optimal" hypothesis; provides a principled framework.

### Paper 7: Prompt Sentiment: The Catalyst for LLM Change (Gandhi & Gandhi, 2025)
- **Authors**: Vishal Gandhi, Sagar Gandhi
- **Venue**: arXiv:2503.13510
- **Key Contribution**: Systematic examination of sentiment effects across 5 LLMs and 6 application domains.
- **Methodology**: Transformed prompts into multiple sentiment variants; evaluated coherence, factuality, and bias.
- **Models**: Claude, DeepSeek, GPT-4, Gemini, LLaMA
- **Results**:
  - Negative prompts reduce factual accuracy (~8.4% decrease)
  - Positive prompts increase verbosity and sentiment propagation
  - Effects strongest in creative writing, weakest in legal/technical domains
- **Relevance**: Shows sentiment effects are domain-dependent.

### Paper 8: Threat-Based Manipulation in LLMs (2025)
- **Venue**: arXiv:2507.21133
- **Key Contribution**: Examines "stick" via threat-based manipulation of Claude, GPT-4, Gemini.
- **Results**: Threats can paradoxically enhance performance (up to +1336% effect size in some cases) while also revealing vulnerabilities. Systematic certainty manipulation observed.
- **Relevance**: Extreme "stick" approach; shows dual nature of pressure-based prompting.

---

## Common Methodologies

1. **Tone variant generation**: Creating matched prompt variants at different politeness levels — used by Yin et al. (8 levels), Dobariya & Kumar (5 levels), Cai et al. (3 levels)
2. **Emotional stimulus appending**: Adding psychological phrases after task prompts — used by EmotionPrompt (11 stimuli), NegativePrompt (10 stimuli)
3. **MCQ accuracy evaluation**: Measuring correctness on multiple-choice benchmarks — used by all papers
4. **Statistical testing**: Paired t-tests (most common), confidence intervals, effect size analysis
5. **Cross-model comparison**: Testing across multiple LLM families to assess generalizability

## Standard Baselines

- **Vanilla/neutral prompt**: No politeness modifier (baseline in all studies)
- **Zero-shot-CoT**: "Let's think step by step" (baseline in EmotionPrompt)
- **APE**: Automatic Prompt Engineering (baseline in EmotionPrompt)

## Evaluation Metrics

- **Accuracy**: Proportion correct on MCQ tasks (primary metric in all studies)
- **BERTScore / ROUGE-L**: For summarization tasks (Yin et al.)
- **Bias Index**: Custom metric for stereotypical bias detection (Yin et al.)
- **TruthfulQA metrics**: Truthfulness and informativeness (NegativePrompt)
- **Human evaluation**: Performance, truthfulness, responsibility (EmotionPrompt)

## Datasets in the Literature

| Dataset | Used By | Task | Scale |
|---------|---------|------|-------|
| MMLU | Yin et al., Cai et al. | Language understanding MCQ | 17,844 questions |
| MMMLU | Cai et al. | Multilingual MMLU | 57 domains |
| JMMLU | Yin et al. | Japanese MMLU | 7,536 questions |
| C-Eval | Yin et al. | Chinese exam MCQ | 5,200 questions |
| Instruction Induction | EmotionPrompt, NegativePrompt | Task inference | 24 tasks |
| BIG-Bench | EmotionPrompt, NegativePrompt | Challenging tasks | 21 curated tasks |
| TruthfulQA | NegativePrompt | Truthfulness eval | 817 questions |
| CNN/DailyMail | Yin et al. | Summarization | 500 test samples |
| CrowS-Pairs | Yin et al. | Bias detection | Multiple bias categories |
| Custom 50-Q MCQ | Dobariya & Kumar | Mixed domain MCQ | 250 (50 × 5 tones) |

## Gaps and Opportunities

1. **Contradictory findings**: Yin et al. find polite > rude; Dobariya & Kumar find rude > polite; Cai et al. find effects are mostly non-significant at scale. A meta-study can systematically reconcile these.

2. **Dataset scale effects**: Cai et al. demonstrated that tone effects observed on 50 questions disappear on larger benchmarks. This is a critical methodological insight for designing our experiments.

3. **Model generation effects**: Results differ across model generations (GPT-3.5 vs GPT-4 vs GPT-4o). Newer models may be more robust to tone variation.

4. **Carrot vs. stick comparison**: EmotionPrompt (positive) and NegativePrompt (negative) have never been directly compared under identical conditions in a single study. NegativePrompt claims higher improvements on some benchmarks.

5. **Confound: prompt length**: Polite/emotional additions increase prompt length, which itself may affect performance. No study adequately controls for this.

6. **Missing: interaction with task difficulty**: Some evidence that tone effects are stronger on humanities/interpretive tasks vs. STEM/factual tasks, but not systematically studied.

---

## Recommendations for Our Experiment

### Recommended Datasets
1. **MMLU** (primary) — Large, multi-domain, MCQ format, used by multiple papers. Enables comparison.
2. **TruthfulQA** — Tests a different dimension (truthfulness vs. accuracy). Used by NegativePrompt.
3. Optionally: **BIG-Bench** subset — For harder tasks where emotional effects are larger.

### Recommended Approach
- Design a factorial experiment: {polite, neutral, rude} × {positive-emotional, negative-emotional, none} × {multiple LLMs} × {multiple domains}
- Use MMLU subsets from different domains (STEM vs. humanities) to test domain interaction
- Include multiple runs per condition for statistical robustness
- Control for prompt length as a confound

### Recommended Baselines
1. Neutral/vanilla prompt (no tone modifier)
2. Polite prompt variants (from Yin et al. and Dobariya & Kumar)
3. Rude prompt variants (from same sources)
4. EmotionPrompt stimuli (EP02, EP06)
5. NegativePrompt stimuli (NP04, NP06)

### Recommended Metrics
1. **Accuracy** on MCQ tasks (primary)
2. **Effect size** (Cohen's d) for comparing conditions
3. **Statistical significance** via paired t-tests or bootstrapped confidence intervals
4. **Domain-stratified analysis** (STEM vs. Humanities)

### Methodological Considerations
- Use ≥200 questions per condition (Cai et al. showed 50 is insufficient)
- Run each condition multiple times (≥5 trials) to account for stochastic variation
- Test across ≥3 model families (GPT, open-source like Llama, and one more)
- Report both per-task and aggregated results (Cai et al. showed aggregation attenuates effects)
