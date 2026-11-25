# Comprehensive Summary and Comparison of Adaptive Test-Time Compute (TTC) & Query-Aware Reasoning in LLMs and VLMs

This document provides a rigorous, in-depth comparative analysis of all **11 research papers** located across the root workspace and the `VLM/` folder. The surveyed works represent the frontier in **Query-Aware Reasoning**, **Difficulty-Adaptive Computation**, **Token Budgeting**, and **Multimodal Test-Time Scaling**.

---

## Table of Contents
1. [Executive Overview & Taxonomy](#1-executive-overview--taxonomy)
2. [Master Comparison Table](#2-master-comparison-table)
3. [Deep Dive: Text-LLM Adaptive Reasoning Approaches](#3-deep-dive-text-llm-adaptive-reasoning-approaches)
   - [3.1 TALE: Token-Budget-Aware LLM Reasoning (2412.18547v6)](#31-tale-token-budget-aware-llm-reasoning-241218547v6)
   - [3.2 DAST: Difficulty-Adaptive Slow Thinking for LRMs (2503.04472v3)](#32-dast-difficulty-adaptive-slow-thinking-for-lrms-250304472v3)
   - [3.3 SelfBudgeter: Adaptive Token Allocation for LLM Reasoning (2505.11274v6 - Root)](#33-selfbudgeter-adaptive-token-allocation-for-llm-reasoning-250511274v6---root)
   - [3.4 Predictive Scheduling for Inference-Time Reasoning (2602.01237v1)](#34-predictive-scheduling-for-inference-time-reasoning-260201237v1)
   - [3.5 Adaptive TTC via Constrained Policy Optimization / SOLVE-THEN-LEARN (2604.14853v1)](#35-adaptive-ttc-via-constrained-policy-optimization--solve-then-learn-260414853v1)
   - [3.6 Learning When to Think: Self-Routing GRPO (2608.20256v2)](#36-learning-when-to-think-self-routing-grpo-260820256v2)
   - [3.7 Reasoning on a Budget: Survey of Controllable & Adaptive TTC (2507.02076v1)](#37-reasoning-on-a-budget-survey-of-controllable--adaptive-ttc-250702076v1)
4. [Deep Dive: Multimodal & Vision-Language Model (VLM) Approaches](#4-deep-dive-multimodal--vision-language-model-vlm-approaches)
   - [4.1 Ada-DeVL: Adaptive Decoupling of Visual & Logical Pathways (VLM/2026.emnlp-main.417)](#41-ada-devl-adaptive-decoupling-of-visual--logical-pathways-vlm2026emnlp-main417)
   - [4.2 Does Reasoning Improve Seeing? Probing & Intervening VLM Circuits (VLM/2505.11274v6)](#42-does-reasoning-improve-seeing-probing--intervening-vlm-circuits-vlm250511274v6)
   - [4.3 AVIS: Adaptive Test-Time Scaling for Vision-Language Models (VLM/2606.11576v1)](#43-avis-adaptive-test-time-scaling-for-vision-language-models-vlm260611576v1)
   - [4.4 SPARC: Separating Perception and Reasoning Circuits (VLM/avogaro26a)](#44-sparc-separating-perception-and-reasoning-circuits-vlmavogaro26a)
5. [Cross-Cutting Synthesis Across the Four Core Dimensions](#5-cross-cutting-synthesis-across-the-four-core-dimensions)
   - [5.1 How Hardness / Complexity is Characterized](#51-how-hardness--complexity-is-characterized)
   - [5.2 Decision Mechanisms: When & How to Trigger Additional Compute](#52-decision-mechanisms-when--how-to-trigger-additional-compute)
   - [5.3 Training vs. Inference Overhead Trade-offs](#53-training-vs-inference-overhead-trade-offs)
   - [5.4 Empirical Frontiers: Model Sizes & Benchmark Distributions](#54-empirical-frontiers-model-sizes--benchmark-distributions)
6. [Architectural Insights for Query-Aware VLM System Design](#6-architectural-insights-for-query-aware-vlm-system-design)

---

## 1. Executive Overview & Taxonomy

Large Reasoning Models (LRMs) such as DeepSeek-R1, OpenAI o1/o3, and recent Vision-Language Models (VLMs) have demonstrated that scaling computation at inference time via Chain-of-Thought (CoT), verification, or parallel rollouts dramatically improves accuracy on complex tasks. However, **uniform test-time compute allocation incurs severe inefficiencies**:
1. **Overthinking on simple queries**: Models output thousands of verbose tokens solving basic arithmetic or perceptual identification, inflating latency, monetary cost, and KV-cache footprint.
2. **Underthinking on hard queries**: Models hit arbitrary context/token caps before exploring sufficient solution paths on competition-level or fine-grained queries.
3. **Multimodal Mirage Reasoning & Visual Drift**: In VLMs, spending extended CoT tokens on visual tasks often leads to hallucinated visual attributes ("mirage reasoning") where reasoning diverges from grounding rather than improving accuracy.

The 11 papers under study propose fundamentally different mechanisms to detect prompt complexity and dynamically scale test-time compute:
- **Pre-Generation External Predictors**: Lightweight MLPs, layer probes, or tabular Gradient Boosted Machines (GBMs) inspect query embeddings/features before generation to select token caps or parallel sample counts.
- **Intrinsic / Self-Budgeting Policies**: The reasoning model is trained via Reinforcement Learning (GRPO, SimPO) to output a `<budget>` tag, select a routing mode (`NoThink`, `Short`, `Long`), or internalize length preferences.
- **Decoupled Perception vs. Reasoning Pathways (VLMs)**: Visual context scaling (pruning, image cropping, visual search) is split from logical reasoning (deliberative CoT, rollout search), avoiding monolithic scaling bottlenecks.

```
                                      Test-Time Compute (TTC) Paradigm
                                                     │
                     ┌───────────────────────────────┴───────────────────────────────┐
                     ▼                                                               ▼
           L1: Controllability                                             L2: Adaptiveness
     (Adhering to fixed/user budget)                                (Dynamic scaling per sample)
                     │                                                               │
        ┌────────────┴────────────┐                        ┌─────────────────────────┼─────────────────────────┐
        ▼                         ▼                        ▼                         ▼                         ▼
 Prompt Constrained         Post-Trained             Pre-Generation            Intrinsic Policy           Decoupled Visuo-
 (e.g. TALE-EP)             (e.g. TALE-PT)           Router / Probing           Self-Routing              Reasoning (VLMs)
                                                     - SOLVE-THEN-LEARN        - DAST (SimPO)             - Ada-DeVL (2x2)
                                                     - Predictive Sched.       - SelfBudgeter (GRPO)      - AVIS (VCS x VRS)
                                                     - Does Reasoning See?     - Learning When to Think   - SPARC (IRD search)
```

---

## 2. Master Comparison Table

The table below provides a comprehensive side-by-side comparison answering the four specific focus areas requested:

| Paper & Reference | 1. Selection / Definition of Hard vs. Simple Prompts | 2. Decision Mechanism: When & How More Compute is Allocated | 3. Overhead & Training Required | 4. Models & Benchmarks Evaluated | Primary Empirical Gains |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **TALE**<br>`2412.18547v6` | Minimal token budget $B^*$ required for correctness, found via offline **binary search** over token lengths $[0, \text{vanilla\_len}]$ using an "implicit monotonicity" assumption. | **TALE-EP**: Zero-shot prompt asks estimator to classify difficulty into budget tiers ($B$), prepending `[Token Budget: B]`.<br>**TALE-PT**: Model internalizes budget awareness into its weights. | **TALE-EP**: 0 training; 1 extra LLM prompt (~50-100 tokens).<br>**TALE-PT**: SFT on binary-searched budgets; **0 inference overhead**. | **Models**: LLaMA-2/3/3.1-8B, Qwen2.5-7B, Mistral-7B, GPT-3.5.<br>**Data**: GSM8K, MATH, SVAMP, ASDiv. | **~67% token cut** (TALE-EP) with $<3\%$ accuracy drop; **~50% token cut** (TALE-PT). |
| **DAST**<br>`2503.04472v3` | **Token Length Budget (TLB)**: $L_{\text{budget}} = p \cdot L_r + (1-p) \cdot L_m$. Combines pass rate $p$, average correct length $L_r$, and penalty ceiling $L_m$. High $p$ / low $L_r$ = simple; low $p$ / high $L_r$ = hard. | Model intrinsically controls CoT length per query during token-by-token generation; trained via budget-aware reward shaping that penalizes overthinking simple queries and rewards deep thinking on complex ones. | **Training**: Offline sampling from MATH, TLB calculation, pairwise preference generation, fine-tuning via **SimPO**.<br>**Inference**: **Zero overhead** (no router, no extra tokens). | **Models**: DeepSeek-R1-Distill-Qwen-7B, DeepSeek-R1-Distill-Qwen-32B, QwQ-32B-Preview.<br>**Data**: MATH-500, AIME 2024, OlympiadBench. | **$>30\%$ to $46\%$ token reduction** while preserving or improving accuracy across all MATH difficulty levels. |
| **SelfBudgeter**<br>`2505.11274v6` (Root) | Difficulty defined by necessary reasoning depth; empirical responses on GSM8K, MATH, and s1k-1.1 establish reference budget bounds. | Model explicitly generates `<budget> B </budget>` token estimate as its first output, then generates the solution constrained by $B$. Precision Budget reward ($PreB$) enforces strict adherence. | **Training**: Stage 1 Cold-Start SFT (4.5k samples); Stage 2 **Budget-Guided GRPO** with format, $PreB$, and accuracy rewards.<br>**Inference**: $\sim 5-10$ tokens to output `<budget>` tag. | **Models**: DeepSeek-R1-Distill-Qwen-1.5B/7B, Eurus-2-7B-PRIME, Qwen-2.5-7B-Simple-RL.<br>**Data**: GSM8K, MATH500, AIME2025. | **61% average token compression** on math benchmarks with zero accuracy degradation (matches base R1 distill). |
| **Predictive Scheduling**<br>`2602.01237v1` | Stratified into discrete difficulty tiers (Easy/Medium/Hard) and minimal oracle token length targets (grids from 16 to 256 tokens). | Pre-run lightweight predictors: (a) Linear/MLP probe on intermediate hidden states (layers 12–17), or (b) LoRA text classifier. A **Greedy Batch Allocator** dynamically distributes a global token pool to maximize expected batch accuracy. | **Training**: Offline training of MLP probes on hidden states or LoRA on text.<br>**Inference**: Forward pass up to layer 12–17 + greedy knapsack solver ($<5$ ms CPU). | **Models**: DeepSeek-R1-Distill-Qwen-1.5B, Llama-3-8B.<br>**Data**: GSM8K, MATH subsets. | **Up to +7.9% absolute accuracy** over uniform budgeting at identical aggregate token cost; closes $>50\%$ of oracle gap. |
| **Constrained Policy Opt (SOLVE-THEN-LEARN)**<br>`2604.14853v1` | Marginal accuracy gain per compute unit. Modeled via **Lagrangian Duality**: queries where $\mathbb{E}[R_i(b)] - \lambda C(b)$ jumps with higher compute are assigned more rollouts. | **Amortized Policy Learning**: A lightweight **Gradient Boosted Machine (GBM)** predicts the optimal compute tier $b \in \{1, 4, 8, 16, 32\}$ from cheap input surface/perplexity features before generation. | **Training**: One-time oracle calibration + GBM training ($<1$ s CPU).<br>**Inference**: **$<1$ ms CPU inference**; 0 GPU FLOPs; no extra LLM tokens. | **Models**: DeepSeek-V3 (`deepseek-chat`), GPT-4o-mini, Qwen2.5-7B.<br>**Data**: MATH (levels 2–4), GSM8K. | Significantly outperforms heuristic thresholds (entropy, self-consistency) under strict average compute budgets. |
| **Learning When to Think**<br>`2608.20256v2` | Problem complexity categorized into three discrete regimes: `NoThink` (trivial), `Short` (intermediate), and `Long` (extended multi-step search). | **First-Token Self-Routing**: Model outputs `<NoThink>`, `<Short>`, or `<Long>` as its very first token. Hard per-mode token caps (1,024, 3,000, uncapped) strictly constrain execution. GRPO balance term prevents collapse. | **Training**: End-to-end **GRPO** (90 steps on 4x H100s) on MATH-lighteval with shaped rewards and MoE-style advantage balancing.<br>**Inference**: Exactly **1 single token** of overhead. | **Models**: DeepSeek-R1-Distill-Qwen-1.5B.<br>**Data**: MATH-lighteval (train), MATH-500, GSM8K (OOD easy), AIME 2024 (OOD hard). | **41% token reduction on MATH-500**; **76% token reduction on GSM8K**; Pareto-dominates fixed-mode baselines without retraining. |
| **Reasoning on a Budget (Survey)**<br>`2507.02076v1` | Surveys the entire landscape: defines two-tiered taxonomy: **L1 Controllability** (fixed user budget) vs. **L2 Adaptiveness** (dynamic difficulty/confidence scaling). | Synthesizes 3 main paradigms: (1) Pre-generation routing (classifiers/probes), (2) In-flight dynamic stopping (entropy, verifier confidence), (3) Self-regulated RL (reward shaping). | Comprehensive survey of overheads: Prompting (0 train, high variance) vs. SFT/RL (heavy train, 0 infer overhead) vs. External Routers (cheap train, tiny infer overhead). | **Models**: Claude 3.7 Sonnet, OpenAI o1/o3-mini, DeepSeek-R1, QwQ-32B, Gemini 2.0 Flash Thinking.<br>**Data**: AIME, MATH-500, GSM8K, GPQA, LiveCodeBench. | Benchmarks commercial LLM thinking budget adherence; establishes standard taxonomy and evaluation criteria. |
| **Ada-DeVL**<br>`VLM/2026.emnlp-main.417` | Decoupled **2x2 visuo-reasoning taxonomy**: Low/High Visual Perception Depth $\times$ Low/High Logical Reasoning Intensity (DIR, VGI, PDL, CVRD). | Model outputs pathway indicator tokens at start of generation to trigger 1 of 4 pathways, allocating visual perception depth (visual tokens) and logical CoT depth accordingly. | **Training**: Stage 1 **Self-Distillation SFT** (synthesizes pathway data without cross-model bias); Stage 2 **Structure-Aware GRPO**.<br>**Inference**: Zero external router overhead. | **Models**: Qwen2.5-VL-7B, Qwen3-VL-8B, Vision-SR1, Qwen3-VL-8B-Thinking.<br>**Data**: MMMU, MMMU-Pro, MathVista, ChartQA, DocVQA, HallusionBench. | **Qwen2.5-VL-7B**: +2.1% acc on MMMU with **39.6% token savings**; **Qwen3-VL-8B**: 64.0% acc on MMMU with **44.6% token savings**. |
| **Does Reasoning Improve Seeing?**<br>`VLM/2505.11274v6` | Evaluates two oracle properties: (1) **Reasoning Helpfulness** (binary: does thinking improve acc over instruct?), and (2) **Desired Generation Length** (minimal token budget). | Linear probes on final-layer last-token embeddings classify helpfulness and predict length. Head attribution reveals lower-layer **length heads** and higher-layer **reasoning heads** for causal intervention. | **Training**: Training-free for base model; trains lightweight linear probe on frozen embeddings.<br>**Inference**: Linear probe dot-product ($<0.1$ ms). Enables routing easy/perceptual queries to direct instruct mode. | **Models**: InternVL-2.5 (7B, 8B, 78B), Qwen2.5-VL-7B, Qwen3-VL-8B (Instruct vs. Thinking).<br>**Data**: MathVista, MMMU, ChartQA, DocVQA, InfoVQA, RealWorldQA, POPE. | Prevents "mirage reasoning" and visual drift; saves thousands of tokens by avoiding useless CoT on purely perceptual tasks. |
| **AVIS**<br>`VLM/2606.11576v1` | Decomposes test-time scaling into two orthogonal axes: **Visual Context Scaling (VCS)** (visual token density) and **Visual Reasoning Scaling (VRS)** (reasoning rollout count $K$). | **Joint Adaptive Policy**: Evaluates query difficulty to select optimal $(r_{\text{vcs}}, K_{\text{vrs}})$ pair. VCS uses **Key Diversity Visual (KDV) Pruning**; VRS scales rollouts via early confidence agreements. | **Training**: KDV pruning is completely **training-free** ($O(N)$ key diversity); lightweight policy network trained on calibration set.<br>**Inference**: Saves 50–70% visual KV cache + reduces rollouts. | **Models**: Qwen2-VL-7B, InternVL-2.5-8B, LLaVA-OneVision-7B.<br>**Data**: MMMU, MathVista, MMStar, ChartQA, DocVQA, HallusionBench, CountBench. | Outperforms fixed-budget baselines; dramatically lowers total FLOPs and KV-cache footprint while improving accuracy. |
| **SPARC**<br>`VLM/avogaro26a` | Identifies tasks where failure is caused by visual search/localization bottleneck (small objects, complex scenes) rather than multi-step deduction. | **Two-Stage Modular Architecture**: Decouples Perception Circuit (Implicit Relevance Detection - IRD finds crop boxes) from Reasoning Circuit. **Asymmetric Compute Scaling**: Applies self-consistency rollouts **only** to the cheap perception stage! | **Training**: Zero-shot prompting or lightweight perception fine-tuning (only predicts coordinates; no long CoT rollouts needed during RL).<br>**Inference**: Bounding box search uses $\sim 20$ tokens vs. 3,000-token CoT rollouts. | **Models**: Qwen2-VL-7B, Qwen3-VL-4B, Qwen3-VL-8B.<br>**Data**: $V^*$ VQA benchmark, MMMU, RealWorldQA. | **+6.7 points on $V^*$ VQA** over Qwen3-VL-4B; outperforms "thinking with images" with **200x lower token budget**. |

---

## 3. Deep Dive: Text-LLM Adaptive Reasoning Approaches

### 3.1 TALE: Token-Budget-Aware LLM Reasoning (2412.18547v6)
*Tingxu Han, Zhenting Wang, Chunrong Fang, Shiyu Zhao, Shiqing Ma, Zhenyu Chen*

#### 1. Selection & Definition of Hard vs. Simple Prompts
- **Definition of Difficulty**: A problem's difficulty is defined operationally as the **minimal token budget** $\beta^*$ under which the model can still produce a correct final answer.
- **Offline Binary Search Algorithm**: Under an "implicit monotonicity assumption" (if a model succeeds at budget $\beta$, it generally succeeds at $\beta' > \beta$; if it fails at $\beta$, it fails at $\beta'' < \beta$), the authors run a binary search over token budgets between $0$ and the Vanilla CoT length $L_{\text{vanilla}}$. The smallest budget yielding a correct answer is designated as the target budget for that query.

#### 2. Decision Mechanism for Hard Samples & Compute Allocation
- The paper introduces two distinct deployment mechanisms:
  - **TALE-EP (Estimation & Prompting)**: A lightweight estimator (zero-shot prompt) evaluates the question and classifies it into a recommended token budget range. The system then dynamically injects an explicit budget prompt: `[Token Budget: B tokens]` into the inference instruction.
  - **TALE-PT (Post-Training Internalization)**: The model is fine-tuned on the optimal searched token budgets so that it intrinsically generates concise or extended chains based on difficulty without needing an explicit budget constraint in the prompt.

#### 3. Overhead and Training Required
- **TALE-EP**: Requires **zero training**. Inference overhead consists of one additional lightweight LLM call to predict the budget (~50–100 tokens), which is amortized by the subsequent 67% reduction in generation length.
- **TALE-PT**: Requires offline data preparation via binary search and standard Supervised Fine-Tuning (SFT). Once trained, it incurs **zero inference overhead** (single standard forward pass).

#### 4. Models and Datasets / Benchmarks Evaluated
- **Models**: LLaMA-2-7B, LLaMA-3-8B, LLaMA-3.1-8B-Instruct, Qwen-2.5-7B-Instruct, Mistral-7B-Instruct, GPT-3.5-Turbo.
- **Datasets**: GSM8K, MATH, SVAMP, ASDiv.
- **Key Results**: TALE-EP achieves an average **67% reduction in token consumption** with less than 3% accuracy degradation; TALE-PT achieves ~50% token reduction.

---

### 3.2 DAST: Difficulty-Adaptive Slow Thinking for LRMs (2503.04472v3)
*Yi Shen, Jian Zhang, Jieyun Huang, Shuming Shi, Wenjing Zhang, Jiangze Yan, Ning Wang, Kai Wang, Zhaoxiang Liu, Shiguo Lian*

#### 1. Selection & Definition of Hard vs. Simple Prompts
- **Token Length Budget (TLB) Metric**: Rather than relying purely on prompt heuristics or empirical sampling pass rates, DAST formalizes difficulty via:
  $$L_{\text{budget}} = p \cdot L_r + (1 - p) \cdot L_m$$
  where $p \in [0, 1]$ is the empirical pass rate across sampled candidate responses, $L_r$ is the average length of correct responses, and $L_m$ is the maximum response ceiling (length penalty).
- For simple questions, $p \to 1$, so $L_{\text{budget}} \approx L_r$ (short). For complex questions, $p \to 0$, so $L_{\text{budget}} \to L_m$ (permitting extended thinking).

#### 2. Decision Mechanism for Hard Samples & Compute Allocation
- **Budget-Aware Reward Shaping**: During training, responses are evaluated against $L_{\text{budget}}$. Correct responses that are overly long receive penalty signals; correct responses that solve the problem near or below $L_{\text{budget}}$ receive maximum reward. Incorrect responses below $L_{\text{budget}}$ are penalized for underthinking.
- **End-to-End Autonomous Scaling**: The model adjusts its own thinking length token-by-token during autoregressive generation. There is no external router, classifier, or pre-generation decision step.

#### 3. Overhead and Training Required
- **Training Pipeline**:
  1. Offline sampling of responses from MATH to calculate TLB.
  2. Reward score calibration and pairwise preference dataset construction $\{x, y_w, y_l\}$.
  3. Fine-tuning using **SimPO (Simple Preference Optimization)**, which eliminates the reference model memory footprint required by standard DPO.
- **Inference Overhead**: **Zero overhead** in latency, memory, or parameters.

#### 4. Models and Datasets / Benchmarks Evaluated
- **Models**: DeepSeek-R1-Distill-Qwen-7B, DeepSeek-R1-Distill-Qwen-32B, QwQ-32B-Preview.
- **Datasets**: MATH-500 (stratified levels 1–5), AIME 2024, OlympiadBench.
- **Key Results**: Reduces token usage by **35% to 46%** on MATH-500 while improving accuracy from 74.8% to 76.7% (7B model), and effectively maintains high performance on hard competition problems (AIME).

---

### 3.3 SelfBudgeter: Adaptive Token Allocation for LLM Reasoning (2505.11274v6 - Root)
*Zheng Li, Qingxiu Dong, Jingyuan Ma, Di Zhang, Kai Jia, Zhifang Sui*

#### 1. Selection & Definition of Hard vs. Simple Prompts
- Difficulty is defined by the minimal token budget required to reach a correct solution. Pre-training analysis partitions training datasets (GSM8K, MATH, and s1k-1.1) into empirical difficulty distributions based on correct reasoning trace lengths.

#### 2. Decision Mechanism for Hard Samples & Compute Allocation
- **Two-Stage Generation Protocol**: When presented with a query, the model explicitly self-estimates and emits the token budget first:
  `<budget> B </budget> \n <solution> ... </solution>`
- Alternatively, if an external user or system specifies a budget, the model conditions its reasoning directly on the provided value.
- **Precision Budget Reward ($PreB$)**: A cosine-shaped reward penalizes length deviation $| \ell - B | / B > \alpha$. For correct answers, the reward peaks at $(1-\alpha)B$ to encourage conciseness; for incorrect answers, it demands full exploration up to $B$.

#### 3. Overhead and Training Required
- **Training Framework**:
  1. **Cold-Start SFT**: 4,500 curated samples (1.5k GSM8K, 1.5k MATH, 1.5k s1k-1.1) pre-populated with `<budget>` tags to establish the syntax.
  2. **Budget-Guided GRPO**: Reinforcement learning optimizing a composite objective: format reward $r_f$, accuracy reward $r_{\text{acc}}$, budget adherence reward $PB$, and precision reward $PreB$.
- **Inference Overhead**: Minimal token overhead (~5–10 tokens) to emit the `<budget>` tag before beginning the solution.

#### 4. Models and Datasets / Benchmarks Evaluated
- **Models**: DeepSeek-R1-Distill-Qwen-1.5B, DeepSeek-R1-Distill-Qwen-7B, Eurus-2-7B-PRIME, Qwen-2.5-7B-Simple-RL.
- **Datasets**: GSM8K, MATH500, AIME2025.
- **Key Results**: Delivers an average **61% token compression** across mathematical tasks. On MATH500, SelfBudgeter-7B matches the base model accuracy (86.87% vs. 86.73%) while cutting tokens from 5,387 to 2,666.

---

### 3.4 Predictive Scheduling for Inference-Time Reasoning (2602.01237v1)
*Katrina Brown, Aneesh Muppidi, Rana Shahout*

#### 1. Selection & Definition of Hard vs. Simple Prompts
- Queries are categorized into discrete difficulty tiers (Easy, Medium, Hard) and continuous required reasoning lengths ($16$ to $256$ tokens in steps of 16), derived from oracle execution runs on GSM8K and MATH.

#### 2. Decision Mechanism for Hard Samples & Compute Allocation
- **Pre-Run Lightweight Predictors**:
  - *Hidden-State Probing MLP*: Evaluates intermediate transformer hidden states after a shallow forward pass. Systematic layer attribution proves that **middle layers (12–17)** contain the peak predictive signal for reasoning length.
  - *LoRA Text Classifier*: Fine-tuned directly on raw question text to classify difficulty before decoding.
- **Greedy Batch Allocator**: Operates globally across a batch of incoming queries under a fixed aggregate token budget $B_{\text{total}}$. It formulates allocation as a knapsack optimization problem, assigning higher token allowances to queries where marginal accuracy gain is highest.

#### 3. Overhead and Training Required
- **Training**: Offline training of a shallow 2-layer MLP on frozen transformer representations or lightweight LoRA fine-tuning on question text.
- **Inference Overhead**: Truncated forward pass (only layers 1–17) or single lightweight LoRA classification, followed by a greedy sorting algorithm ($O(N \log N)$), completing in under 5 ms on CPU.

#### 4. Models and Datasets / Benchmarks Evaluated
- **Models**: DeepSeek-R1-Distill-Qwen-1.5B, Llama-3-8B.
- **Datasets**: GSM8K, MATH subsets.
- **Key Results**: Delivers up to **+7.9% absolute accuracy improvement** over uniform token budgeting at identical aggregate compute, recovering over 50% of the gap to an oracle with perfect foresight.

---

### 3.5 Adaptive TTC via Constrained Policy Optimization / SOLVE-THEN-LEARN (2604.14853v1)
*Zhiyuan Zhai*

#### 1. Selection & Definition of Hard vs. Simple Prompts
- **Marginal Return per Compute Unit**: Formulates compute allocation as:
  $$\max_{\{b_i\}} \frac{1}{N} \sum_{i=1}^N \mathbb{E}[R_i(b_i)] \quad \text{s.t.} \quad \frac{1}{N} \sum_{i=1}^N C(b_i) \le B$$
- Simple prompts saturate accuracy at low compute ($b_0 = 1$ sample). Hard prompts show steep accuracy improvements when scaled to higher rollouts ($b \in \{4, 8, 16, 32\}$). Intractable prompts fail at all compute levels and are starved of wasted compute.

#### 2. Decision Mechanism for Hard Samples & Compute Allocation
- **SOLVE-THEN-LEARN Pipeline**:
  - **Lagrangian Oracle (Solve Stage)**: By dual decomposition, the global constraint reduces to uncoupled per-query decisions:
    $$b_i^*(\lambda) = \arg\max_{b \in \mathcal{B}} \left( \mathbb{E}[R_i(b)] - \lambda C(b) \right)$$
    Sweeping $\lambda$ generates ground-truth compute labels for a calibration dataset.
  - **Amortized Policy Learning (Learn Stage)**: A lightweight **Gradient Boosted Machine (GBM)** or small neural net is trained on cheap text surface features (length, syntactic depth, character distributions, question keywords) to predict $b_i^*(\lambda)$ directly.

#### 3. Overhead and Training Required
- **Training**: One-time offline generation of oracle calibration curves. Training the GBM classifier takes **less than 1 second** on CPU.
- **Inference Overhead**: **$<1$ ms CPU latency**, zero GPU FLOPs, and zero extra tokens. Completely non-invasive to the underlying LLM.

#### 4. Models and Datasets / Benchmarks Evaluated
- **Models**: DeepSeek-V3 (`deepseek-chat`), GPT-4o-mini, Qwen2.5-7B.
- **Datasets**: MATH (stratified across levels 2–4), GSM8K.
- **Key Results**: Consistently Pareto-dominates uniform allocation, self-consistency confidence stopping, and entropy heuristics across the entire compute curve.

---

### 3.6 Learning When to Think: Self-Routing GRPO (2608.20256v2)
*Gijs Kassenaar, Zhao Yang, Vincent François-Lavet*

#### 1. Selection & Definition of Hard vs. Simple Prompts
- Queries are categorized into three discrete operational regimes:
  - `NoThink`: Trivial queries solvable by direct answer (e.g., standard arithmetic).
  - `Short`: Moderate complexity requiring concise reasoning steps.
  - `Long`: High complexity requiring extensive exploration, reflection, and backtracking.

#### 2. Decision Mechanism for Hard Samples & Compute Allocation
- **Single First-Token Routing inside Policy**: The model itself outputs a routing token (`<NoThink>`, `<Short>`, or `<Long>`) as the very first token of its autoregressive response.
- **Hard Per-Mode Token Caps**:
  - `NoThink`: Capped at 1,024 tokens.
  - `Short`: Capped at 3,000 tokens.
  - `Long`: Uncapped (up to full 16,384 context).
- **GRPO Reward Shaping & Load Balancing**:
  - Each mode has a tailored reward: `NoThink` receives high base reward with a steep length penalty; `Short` receives moderate base reward with mild penalty; `Long` receives standard accuracy reward with no penalty.
  - An auxiliary load-balancing term is added directly to the advantage function (analogous to MoE routing) to prevent mode collapse.

#### 3. Overhead and Training Required
- **Training**: End-to-end **GRPO** (group size $G=8$, 90 steps on 4x H100 GPUs) on the MATH-lighteval training split. Requires no separate router network and no SFT warm-up.
- **Inference Overhead**: Exactly **1 single token** of overhead.

#### 4. Models and Datasets / Benchmarks Evaluated
- **Models**: DeepSeek-R1-Distill-Qwen-1.5B.
- **Datasets**: MATH-lighteval (train), MATH-500 (test), GSM8K (OOD easy), AIME 2024 (OOD hard).
- **Key Results**: Cuts mean response length on MATH-500 from 4,796 to 2,811 tokens (**41% reduction**) while matching base accuracy (0.782 vs. 0.796). On GSM8K, it reduces tokens by **76%** without retraining.

---

### 3.7 Reasoning on a Budget: Survey of Controllable & Adaptive TTC (2507.02076v1)
*Mohammad Ali Alomrani, Yingxue Zhang, et al. (Huawei Noah's Ark Lab, McGill, Mila)*

#### 1. Selection & Definition of Hard vs. Simple Prompts
- Establishes a formal two-tiered taxonomy:
  - **L1 Controllability**: Techniques operating under an explicit external compute budget (e.g., maximum token ceiling, specified thinking budget).
  - **L2 Adaptiveness**: Techniques where compute scales dynamically based on input difficulty or intrinsic model uncertainty.

#### 2. Decision Mechanism for Hard Samples & Compute Allocation
- Catalogs test-time compute decision mechanisms across three levels:
  1. *Prompt-Level Routing*: Dispatching queries to different model sizes or prompting strategies based on pre-generation classifiers.
  2. *In-Flight Dynamic Stopping*: Halting decoding or tree-search when verifier confidence crosses a threshold or token entropy drops.
  3. *Reinforcement Learning with Length Penalties*: Optimizing policy models to modulate reasoning length directly (e.g., CPO, IBPO, DAST).

#### 3. Overhead and Training Required
- Evaluates the trade-offs across paradigms:
  - Prompt-based budget control incurs high adherence variance and token drift.
  - Classifier-based routing incurs small inference latency but requires separate training data.
  - RLVR fine-tuning provides near-perfect budget adherence and zero inference overhead, but requires substantial GPU resources for rollouts.

#### 4. Models and Datasets / Benchmarks Evaluated
- **Evaluated Frontier Models**: Claude 3.7 Sonnet (Thinking mode), OpenAI o1/o3-mini, DeepSeek-R1, QwQ-32B, Gemini 2.0 Flash Thinking.
- **Benchmarks**: AIME 2024, MATH-500, GSM8K, GPQA, LiveCodeBench, ARC-AGI.

---

## 4. Deep Dive: Multimodal & Vision-Language Model (VLM) Approaches

### 4.1 Ada-DeVL: Adaptive Decoupling of Visual & Logical Pathways (VLM/2026.emnlp-main.417)
*Haoyuan Fan, Jie Zhao, Yufeng Shi, Wengang Zhou, Houqiang Li (USTC, HFUT, Huawei)*

#### 1. Selection & Definition of Hard vs. Simple Prompts
- Error analysis on MMMU and MMMU-Pro reveals that VLM failures stem from two distinct dimensions: **Visual Perception Errors** (failure to ground fine details) and **Logical Reasoning Errors** (failure in multi-step deduction).
- Ada-DeVL conceptualizes a **2x2 Visuo-Reasoning Taxonomy**:
  1. **Direct Intuitive Reasoning (DIR)**: Low Visual, Low Logic (simple recognition, OCR, direct captioning).
  2. **Visually-Grounded Intuition (VGI)**: High Visual, Low Logic (dense visual perception, high resolution, spatial relationships, simple deduction).
  3. **Pure Deliberative Logic (PDL)**: Low Visual, High Logic (text-heavy logic, math word problems with static diagrams).
  4. **Comprehensive Visuo-Reasoning Deliberation (CVRD)**: High Visual, High Logic (complex STEM diagrams requiring both multi-step visual verification and deep logical deduction).

#### 2. Decision Mechanism for Hard Samples & Compute Allocation
- The model outputs special pathway indicator tokens at the start of decoding to activate one of the four pathways:
  - Visual perception depth is modulated via visual inspection tokens and attention masks.
  - Logical reasoning depth is modulated via CoT chain length.
- The selection is learned intrinsically through reinforcement learning.

#### 3. Overhead and Training Required
- **Two-Stage Training Pipeline**:
  1. **Self-Distillation SFT**: The model generates its own candidate solutions across pathways; only accurate and length-efficient traces are kept, eliminating the exposure bias and hallucinations of cross-model distillation.
  2. **Structure-Aware GRPO**: A composite reward balancing task correctness, pathway format adherence, diversity constraints (to prevent collapse into a single pathway), and length efficiency.
- **Inference Overhead**: Zero external router overhead; exactly one pathway selector token.

#### 4. Models and Datasets / Benchmarks Evaluated
- **Models**: Qwen2.5-VL-7B, Qwen3-VL-8B, Vision-SR1, Qwen3-VL-8B-Thinking.
- **Datasets**: MMMU, MMMU-Pro, MathVista, ChartQA, DocVQA, HallusionBench.
- **Key Results**:
  - Qwen2.5-VL-7B: Achieves **54.0% on MMMU** (surpassing Vision-SR1) while saving **39.6% of tokens**.
  - Qwen3-VL-8B: Achieves **64.0% on MMMU** with a **44.6% token reduction**, matching Qwen3-VL-8B-Thinking while consuming 1,560 vs. 4,889 tokens per query.

---

### 4.2 Does Reasoning Improve Seeing? Probing & Intervening VLM Circuits (VLM/2505.11274v6)
*Jing Bi, Luchuan Song, Dingxin Zhang, Pinxin Liu, Guangyu Sun, Lianggong Bruce Wen, Weidong Cai, Chen Chen, Chenliang Xu*

#### 1. Selection & Definition of Hard vs. Simple Prompts
- The authors uncover a critical counter-intuitive finding: **explicit reasoning frequently degrades visual performance**. Long CoT chains cause models to lose visual grounding and enter "mirage reasoning."
- They define two oracle properties per multimodal query:
  1. **Reasoning Helpfulness**: A binary oracle label indicating whether Thinking mode improves accuracy over direct Instruct mode ($\Delta \text{Acc} > 0$).
  2. **Desired Generation Length**: The minimal token budget required for a correct answer.

#### 2. Decision Mechanism for Hard Samples & Compute Allocation
- **Linear Probing of Internal Representations**: Training simple linear probes on final-layer last-token embeddings reveals that **Thinking models linearly encode both reasoning helpfulness and desired length**, whereas Instruct models do not.
- **Dual Circuit Discovery & Head Attribution**:
  - *Length-Control Attention Heads*: Located in **lower transformer layers**, governing output verbosity.
  - *Reasoning/Difficulty Heads*: Located in **higher transformer layers**, encoding problem difficulty.
- **Adaptive Execution**: Probes dynamically route queries: purely perceptual or simple tasks execute in fast Instruct mode (saving thousands of tokens), while complex reasoning queries activate Thinking mode.

#### 3. Overhead and Training Required
- **Training**: Completely **training-free for the base VLM**. Only requires training lightweight linear probes or ridge regressors on pre-extracted embeddings.
- **Inference Overhead**: A single vector dot product ($<0.1$ ms on CPU/GPU), adding zero noticeable latency.

#### 4. Models and Datasets / Benchmarks Evaluated
- **Models**: InternVL-2.5 (7B, 8B, 78B), Qwen2.5-VL-7B, Qwen3-VL-8B (Instruct vs. Thinking checkpoints).
- **Datasets**: MathVista, MMMU, ChartQA, DocVQA, InfoVQA, RealWorldQA, POPE.
- **Key Results**: Causal scaling of length heads controls output length with minimal accuracy degradation; scaling reasoning heads on helpful queries boosts accuracy while filtering out harmful reasoning on perception tasks.

---

### 4.3 AVIS: Adaptive Test-Time Scaling for Vision-Language Models (VLM/2606.11576v1)
*Ahmadreza Jeddi, Minh N. Le, Amirhossein Kazerouni, Hakki Karaimer, Hue Nguyen, Iqbal Mohomed, Michael Brudno, Konstantinos G. Derpanis, Alex Levinshtein, Babak Taati, Radek Grzeszczuk (Samsung AI Center, U of Toronto, Vector Inst)*

#### 1. Selection & Definition of Hard vs. Simple Prompts
- Test-time compute is decomposed into two distinct, orthogonal axes:
  1. **Visual Context Scaling (VCS)**: Modulating the number of visual tokens passed to the LLM (visual evidence density).
  2. **Visual Reasoning Scaling (VRS)**: Modulating reasoning compute via candidate rollout search ($K$ paths).
- A sample is categorized based on whether it is visually redundant (admitting high visual token pruning) and whether its reasoning requires multi-path exploration ($K > 1$).

#### 2. Decision Mechanism for Hard Samples & Compute Allocation
- **Key Diversity Visual (KDV) Pruning**: A training-free $O(N)$ rule based on key-state cosine diversity in self-attention that prunes redundant background visual tokens while preserving informative foreground patches.
- **Joint Adaptive Policy**: Evaluates query difficulty to select the optimal pair $(r_{\text{vcs}}, K_{\text{vrs}})$. If early rollout agreements indicate high confidence, search terminates; if ambiguity persists, additional rollouts and higher visual resolution are allocated.

#### 3. Overhead and Training Required
- **Training**: KDV pruning is completely **training-free**. The joint allocation policy is calibrated offline using lightweight regression/classification on validation rollouts.
- **Inference Overhead**: Net **reduction in computational overhead**: KDV pruning slashes visual KV cache size by 50–70%, accelerating prefill and decoding while suppressing redundant CoT rollouts.

#### 4. Models and Datasets / Benchmarks Evaluated
- **Models**: Qwen2-VL-7B, InternVL-2.5-8B, LLaVA-OneVision-7B.
- **Datasets**: MMMU, MathVista, MMStar, ChartQA, DocVQA, HallusionBench, CountBench.
- **Key Results**: Achieves Pareto-dominance over both fixed visual pruning (KDV-only) and fixed reasoning search (VRS-only), cutting overall FLOPs significantly while matching or exceeding full-compute accuracy.

---

### 4.4 SPARC: Separating Perception and Reasoning Circuits (VLM/avogaro26a)
*Niccolo Avogaro, Nayanika Debnath, Li Mi, Thomas Frick, Junling Wang, Zexue He, Hang Hua, Konrad Schindler, Mattia Rigotti (ETH Zurich, IBM Research, MIT-IBM Watson AI Lab)*

#### 1. Selection & Definition of Hard vs. Simple Prompts
- Targets tasks where errors stem from **perceptual resolution bottlenecks** (locating small targets, reading fine-print labels, crowded scenes) rather than multi-step logical deduction.
- Monolithic VLMs fail here because small perceptual mistakes at the start of a long CoT trajectory cascade into catastrophic errors ("mirage reasoning").

#### 2. Decision Mechanism for Hard Samples & Compute Allocation
- **Two-Stage Modular Architecture**:
  - *Stage 1 (Perception Circuit - Implicit Relevance Detection / IRD)*: Explicit visual search module that outputs bounding box coordinates of question-relevant image crops.
  - *Stage 2 (Reasoning Circuit)*: Crops are magnified and re-fed alongside the global context to the reasoning engine.
- **Asymmetric Test-Time Scaling**: Instead of scaling compute by generating $K$ full 3,000-token CoT rollouts, SPARC applies self-consistency / beam search **selectively to the perception stage**, generating multiple candidate crop coordinates (~20 tokens each).

#### 3. Overhead and Training Required
- **Training**: Zero-shot prompting or highly efficient perception fine-tuning. Because IRD outputs only bounding box coordinates, RL or SFT requires rolling out only ~20 tokens per iteration rather than thousands of reasoning tokens.
- **Inference Overhead**: Massive net compute reduction: generating multiple coordinate proposals costs orders of magnitude fewer tokens than full multi-modal CoT rollouts.

#### 4. Models and Datasets / Benchmarks Evaluated
- **Models**: Qwen2-VL-7B, Qwen3-VL-4B, Qwen3-VL-8B.
- **Datasets**: $V^*$ VQA benchmark (high-resolution small-object visual search), MMMU, RealWorldQA.
- **Key Results**: Improves Qwen3-VL-4B on the $V^*$ benchmark by **+6.7 points**, outperforming monolithic "thinking with images" baselines with a **200x lower token budget**.

---

## 5. Cross-Cutting Synthesis Across the Four Core Dimensions

### 5.1 How Hardness / Complexity is Characterized

| Characterization Approach | Representative Papers | Methodological Principle | Strengths & Limitations |
| :--- | :--- | :--- | :--- |
| **Minimal Oracle Budget Search** | TALE (`2412.18547v6`), SelfBudgeter (`2505.11274v6`) | Offline binary search or empirical trace filtering to find smallest budget $\beta^*$ where $LLM(x, \beta^*) = y$. | **Strength**: Direct measurement of model-specific capacity.<br>**Limitation**: Expensive offline precomputation. |
| **Statistical Token Length Budget (TLB)** | DAST (`2503.04472v3`) | $L_{\text{budget}} = p \cdot L_r + (1-p) L_m$, combining sample pass rate $p$ with average correct length. | **Strength**: Smoothly handles ambiguity without rigid thresholds.<br>**Limitation**: Requires multi-candidate sampling per query. |
| **Constrained Marginal Return** | SOLVE-THEN-LEARN (`2604.14853v1`) | Lagrangian multiplier $\lambda$ captures marginal accuracy gain per compute unit $\Delta \mathbb{E}[R] / \Delta C$. | **Strength**: Mathematically optimal under global batch constraints.<br>**Limitation**: Saturated or zero-gain queries must be handled carefully. |
| **Discrete Operating Regimes** | Learning When to Think (`2608.20256v2`), Ada-DeVL (`2026.emnlp-main.417`) | 3 modes (`NoThink`/`Short`/`Long`) or 2x2 grid (Perception Depth $\times$ Reasoning Intensity). | **Strength**: Clean operational boundaries; easily enforced via token caps.<br>**Limitation**: Coarse-grained allocation within a bucket. |
| **Reasoning Helpfulness Probing** | Does Reasoning Improve Seeing? (`2505.11274v6`) | Binary oracle label: does explicit CoT improve accuracy over direct instruct mode ($\Delta \text{Acc} > 0$)? | **Strength**: Crucial for multimodal tasks where reasoning can hurt performance.<br>**Limitation**: Requires both instruct and thinking models. |

---

### 5.2 Decision Mechanisms: When & How to Trigger Additional Compute

```
1. Pre-Generation Routing (Before Decoding Starts)
   ├── Surface Feature Classifier (GBM in SOLVE-THEN-LEARN) ──> <1 ms CPU
   ├── Intermediate Transformer Probe (Layers 12-17 in Pred. Sched.) ──> Partial forward pass
   └── Linear Probe on Last Token (Does Reasoning Improve Seeing?) ──> O(d) dot product

2. First-Token Self-Routing (Inside Policy)
   ├── Mode Indicator Token (<NoThink> / <Short> / <Long> in Learning When to Think)
   ├── Pathway Indicator Token (DIR / VGI / PDL / CVRD in Ada-DeVL)
   └── Explicit Budget Tag (<budget> B </budget> in SelfBudgeter)

3. Two-Stage Decoupled Search (Multimodal Execution)
   ├── Asymmetric Perception Scaling (SPARC: Self-consistency on IRD crop coordinates)
   └── Joint Context & Reasoning Scaling (AVIS: KDV visual pruning + early rollout agreement)
```

1. **Pre-Generation External Classifiers (SOLVE-THEN-LEARN, Predictive Scheduling)**:
   - Make the decision *before* running full generation.
   - Ideal for batch scheduling in cloud APIs where token allocations across multi-tenant workloads must be bounded.
2. **First-Token Self-Routing (Learning When to Think, Ada-DeVL, SelfBudgeter)**:
   - The LLM acts as its own router by emitting a special token at $t=1$.
   - Eliminates auxiliary models and feature extraction pipelines. The decision is learned end-to-end via RL (GRPO).
3. **Decoupled Perception-Reasoning Allocation (Ada-DeVL, SPARC, AVIS)**:
   - Prevents the monolithic failure mode where visual perception and logic are scaled with the same costly CoT mechanism.
   - Allows allocating compute where it matters: dense visual tokens / crops for perception, CoT rollouts for logic.

---

### 5.3 Training vs. Inference Overhead Trade-offs

```
                       Inference Latency Overhead
                                   ▲
                                   │
      High                         │      • TALE-EP (1 extra LLM prompt)
                                   │
                                   │      • Predictive Sched. (Partial forward pass)
                                   │
      Low                          │      • SOLVE-THEN-LEARN (<1 ms GBM)
                                   │      • Does Reasoning See? (<0.1 ms Probe)
                                   │      • SelfBudgeter (~5 budget tokens)
                                   │      • Learning When to Think (1 token)
      Zero                         │      • DAST / TALE-PT / Ada-DeVL (Intrinsic)
                                   └───────────────────────────────────────────────►
                                   Zero         Low           Medium        High
                                                Offline Training Compute
```

- **Training-Free / Heuristic Approaches**:
  - *TALE-EP*: Zero training, but adds a second LLM query.
  - *KDV Pruning (AVIS)*: Zero training; calculates cosine diversity in $O(N)$ key operations during attention.
- **Lightweight Probe / Classifier Training**:
  - *SOLVE-THEN-LEARN*: Offline calibration + GBM training in $<1$ second on CPU.
  - *Does Reasoning Improve Seeing?*: Linear ridge regression on frozen last-token representations.
- **Post-Training / Reinforcement Learning**:
  - *DAST*: SimPO preference optimization. Zero inference overhead.
  - *SelfBudgeter*: Cold-start SFT + Budget-guided GRPO.
  - *Learning When to Think*: GRPO (90 steps on 4x H100s). Zero auxiliary models.
  - *Ada-DeVL*: Self-distillation SFT + structure-aware GRPO.

---

### 5.4 Empirical Frontiers: Model Sizes & Benchmark Distributions

The reviewed papers span open-weights and proprietary models from **1.5B to 78B+ parameters**:
- **Small-Scale Models (1.5B - 8B)**: DeepSeek-R1-Distill-Qwen-1.5B, Qwen2.5-VL-7B, Qwen3-VL-4B/8B, LLaMA-3-8B.
  - *Observation*: Small models suffer the most from "verbosity compensation" and overthinking; adaptive compute yields the largest percentage gains on smaller architectures (up to 76% token savings on easy tasks).
- **Mid-to-Large Scale Models (32B - 78B+)**: DeepSeek-R1-Distill-Qwen-32B, QwQ-32B, InternVL-2.5-78B, DeepSeek-V3.
  - *Observation*: Larger models exhibit clearer linear separability in their internal representations for difficulty probing, making linear routing exceptionally accurate.
- **Benchmark Saturation vs. Competition Hardness**:
  - **Easy / Saturated**: GSM8K, SVAMP (base models achieve $>90-95\%$; adaptive routing slashes tokens by 60–76% with zero accuracy loss).
  - **Intermediate**: MATH-500, MathVista, ChartQA, DocVQA (requires balanced reasoning; adaptive methods preserve or slightly boost accuracy).
  - **Hard / Competition**: AIME 2024/2025, OlympiadBench, MMMU, MMMU-Pro, $V^*$ VQA (adaptive methods correctly avoid truncation, allocating maximum compute or high-resolution search).

---

## 6. Architectural Insights for Query-Aware VLM System Design

For designing a state-of-the-art **Query-Aware Multimodal Vision-Language Model**, the collective findings from these 11 papers provide five foundational architectural principles:

```
                            Proposed Query-Aware VLM Architecture
                                              │
                                              ▼
                             Multimodal Query (Image + Text)
                                              │
                      ┌───────────────────────┴───────────────────────┐
                      ▼                                               ▼
             Visual Complexity Check                        Question Complexity Check
             (KDV Key Diversity / IRD)                       (Trained Linear Probe / Token)
                      │                                               │
                      ▼                                               ▼
           Visual Context Scaling (VCS)                    Visual Reasoning Scaling (VRS)
        ┌─────────────┴─────────────┐                   ┌─────────────┴─────────────┐
        ▼                           ▼                   ▼                           ▼
   Low Visual                  High Visual           Direct / Short              Deep CoT /
   (Pruned KDV)             (Salient IRD Crops)      (Fast Instruct)          Rollout Search (K)
        └─────────────┬─────────────┘                   └─────────────┬─────────────┘
                      │                                               │
                      └───────────────────────┬───────────────────────┘
                                              ▼
                                 Decoupled Visuo-Reasoning
                                  Execution Engine (Ada-DeVL)
```

1. **Never Apply Monolithic Test-Time Scaling to Multimodal Queries**:
   - Uniformly extending CoT length on vision-language queries causes **mirage reasoning** and visual hallucination (*Does Reasoning Improve Seeing?*). Perception depth and logical reasoning depth must be scaled independently (*Ada-DeVL*).
2. **Decouple Visual Search from Deliberative Reasoning**:
   - For perception-heavy queries, scaling compute via bounding box search (*SPARC*) is **200x more token-efficient** than generating multi-turn CoT rollouts.
3. **Prune Visual Tokens Before Decoding**:
   - Visual tokens dominate the KV-cache. Applying training-free Key Diversity Visual pruning (*AVIS*) removes 50–70% of visual context without degrading reasoning accuracy.
4. **Use First-Token Self-Routing via GRPO**:
   - To avoid external router latency and deployment complexity, train the VLM end-to-end with GRPO (*Learning When to Think*, *Ada-DeVL*) to emit its pathway or budget token at $t=1$. Hard token caps prevent mode drift.
5. **Calibrate Rewards with Difficulty-Aware Budgets**:
   - Directly penalizing response length degrades hard competition queries. Adopting DAST's **Token Length Budget (TLB)** or SelfBudgeter's **$PreB$ reward** ensures that simple queries are compressed while challenging problems receive sufficient compute.

---
*Summary compiled from analysis of papers: 2412.18547v6, 2503.04472v3, 2505.11274v6 (Root), 2507.02076v1, 2602.01237v1, 2604.14853v1, 2608.20256v2, VLM/2026.emnlp-main.417, VLM/2505.11274v6, VLM/2606.11576v1, VLM/avogaro26a.*
