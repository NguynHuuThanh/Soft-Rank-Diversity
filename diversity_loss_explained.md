# Soft Top-K Diversity Loss for Knowledge-Enhanced Conversational Recommendation

> A differentiable diversity regularization approach that encourages knowledge-graph-aware recommender systems to produce diverse recommendation lists, moving beyond accuracy-only optimization.

---

## Table of Contents

1. [Motivation](#1-motivation)
2. [Background & Notation](#2-background--notation)
3. [The Diversity Loss Function](#3-the-diversity-loss-function)
   - 3.1 [Intuitive Explanation](#31-intuitive-explanation)
   - 3.2 [Step-by-Step Formulation](#32-step-by-step-formulation)
   - 3.3 [Matrix Form — The Complete Formula](#33-matrix-form--the-complete-formula)
4. [Why Not Hard Top-K?](#4-why-not-hard-top-k)
5. [Comparison with DPP-Based Approaches](#5-comparison-with-dpp-based-approaches)
6. [Proof of Differentiability](#6-proof-of-differentiability)
   - 6.1 [Forward Pass Composition](#61-forward-pass-composition)
   - 6.2 [Gradient Derivation](#62-gradient-derivation)
   - 6.3 [Smoothness Class](#63-smoothness-class)
7. [Integration into the KECRS System](#7-integration-into-the-kecrs-system)
   - 7.1 [Combined Training Objective](#71-combined-training-objective)
   - 7.2 [System Architecture Overview](#72-system-architecture-overview)
   - 7.3 [Training Pipeline](#73-training-pipeline)
8. [Per-Turn Diversity Evaluation Metrics](#8-per-turn-diversity-evaluation-metrics)
   - 8.1 [Intra-List Distance (ILD@k)](#81-intra-list-distance-ildk)
   - 8.2 [KG-Entity Coverage@k](#82-kg-entity-coveragek)
   - 8.3 [Category Coverage@k](#83-category-coveragek)
9. [Hyperparameters](#9-hyperparameters)
10. [Implementation Reference](#10-implementation-reference)
11. [Summary](#11-summary)

---

## 1. Motivation

Traditional conversational recommender systems (CRS) are trained with **accuracy-only objectives** — typically cross-entropy loss that maximizes the probability of the ground-truth item. While this successfully teaches the model to predict relevant items, it often leads to a well-known problem: **homogeneous recommendation lists**.

Consider a user who mentions liking *The Dark Knight*. An accuracy-optimized system might recommend *Batman Begins*, *The Dark Knight Rises*, *Batman v Superman*, *Justice League*, and *Joker* — all highly relevant, but all extremely similar (same franchise, same genre, same actors). The user gets no opportunity to discover that they might also enjoy *Inception* (same director, different genre) or *V for Vendetta* (similar theme, different universe).

**The diversity-accuracy trade-off** is a fundamental challenge in recommender systems:

- **High accuracy, low diversity**: The system always recommends the "safest" items — those most similar to what the user already likes.
- **High diversity, low accuracy**: Randomly sampling from the catalog gives maximum diversity but terrible relevance.
- **The sweet spot**: Recommend items that are both **relevant** AND **diverse** — covering different genres, directors, themes, and styles.

Our approach introduces a **differentiable diversity loss** that is added to the standard recommendation objective during training. This allows the neural network to learn representations that **naturally spread probability mass** across dissimilar items, rather than concentrating it on a cluster of near-identical candidates.

### Why Knowledge Graph Diversity?

In knowledge-graph-enhanced recommender systems like KECRS, each item (movie) is connected to rich structural information — genres, directors, actors, themes, awards, etc. — through a knowledge graph. Two movies are "diverse" not just if their learned embeddings differ, but if they **cover different regions of the knowledge graph**. Our loss operates on the RGCN-produced embeddings that already encode this KG structure, meaning minimizing pairwise similarity in embedding space implicitly encourages coverage of diverse KG neighborhoods.

---

## 2. Background & Notation

| Symbol | Description | Shape / Type |
|--------|-------------|--------------|
| $M$ | Total number of candidate movies | Scalar (e.g., 6730) |
| $d$ | Embedding dimension | Scalar (e.g., 128) |
| $B$ | Batch size (number of users per batch) | Scalar |
| $\mathbf{s}_i \in \mathbb{R}^M$ | Raw recommendation scores for user $i$ | Vector |
| $\mathbf{E} \in \mathbb{R}^{M \times d}$ | Item embedding matrix (output of RGCN) | Matrix |
| $\mathbf{e}_j \in \mathbb{R}^d$ | Embedding vector of item $j$ (row $j$ of $\mathbf{E}$) | Vector |
| $\tau > 0$ | Temperature parameter for softmax | Scalar (e.g., 0.1) |
| $\lambda \geq 0$ | Diversity loss weight | Scalar |
| $k$ | Number of top items for evaluation metrics | Scalar (e.g., 10) |
| $\mathcal{G} = (\mathcal{V}, \mathcal{E}, \mathcal{R})$ | Knowledge graph with entities, edges, relation types | Graph |

### The KECRS Recommendation Pipeline (Simplified)

```
User conversation history
        │
        ▼
  Mentioned entities (seed set)
        │
        ▼
  RGCN encodes KG → node embeddings E ∈ ℝ^{M×d}
        │
        ▼
  Self-attention over seed set → user embedding u_i ∈ ℝ^d
        │
        ▼
  Scores: s_i = E · u_i + bias ∈ ℝ^M
        │
        ▼
  Training loss = CrossEntropy(s_i, target) + λ · DiversityLoss(s_i, E)
```

---

## 3. The Diversity Loss Function

### 3.1 Intuitive Explanation

Imagine you are distributing "attention tokens" across all $M$ movies. The model's prediction scores determine how many tokens each movie gets — higher-scored movies get more tokens. We want to **penalize the model when it gives many tokens to movies that are very similar to each other**.

Here is the core idea in plain language:

1. **Convert scores to weights**: Use a softmax with low temperature ($\tau$) to turn raw scores into a probability-like distribution. Low temperature makes this distribution "peaky" — almost all weight concentrates on the top-scored items. This approximates selecting the top-k items, but in a smooth, differentiable way.

2. **Measure pairwise similarity**: Compute cosine similarity between every pair of movie embeddings. This gives us an $M \times M$ similarity matrix $\mathbf{S}$ where $S_{jl} = 1$ means items $j$ and $l$ are identical, and $S_{jl} = 0$ means they are completely different.

3. **Compute weighted similarity**: For each user, compute the "expected pairwise similarity" of the selected items: $\mathbf{w}^\top \mathbf{S} \mathbf{w}$. If the high-weight items are all similar (high $S_{jl}$ values among them), this quantity is large. If the high-weight items are diverse (low $S_{jl}$ values), this quantity is small.

4. **Average over the batch**: Take the mean over all users in the batch.

**Minimizing this loss encourages the model to push high scores toward items whose embeddings are far apart in the KG-informed embedding space.**

### 3.2 Step-by-Step Formulation

#### Step 1: Soft Item Selection via Temperature-Scaled Softmax

Given the raw recommendation scores $\mathbf{s}_i = (s_{i,1}, s_{i,2}, \ldots, s_{i,M})$ for user $i$, we compute soft selection weights:

$$
w_{i,j} = \frac{\exp(s_{i,j} / \tau)}{\sum_{l=1}^{M} \exp(s_{i,l} / \tau)}, \quad j = 1, \ldots, M
$$

where $\tau > 0$ is the **temperature** hyperparameter.

**Effect of temperature:**

| Temperature $\tau$ | Behavior | Analogy |
|---------------------|----------|---------|
| $\tau \to 0^+$ | $\mathbf{w}_i$ approaches a one-hot vector on the highest-scored item | Hard argmax |
| $\tau$ small (e.g., 0.1) | Weight concentrates sharply on top few items | Approximate top-k |
| $\tau = 1$ | Standard softmax | Balanced selection |
| $\tau \to \infty$ | Uniform distribution | Random selection |

We use $\tau = 0.1$ (default) so that the soft selection weights closely approximate the hard top-k selection, while remaining fully differentiable.

#### Step 2: Cosine Similarity Matrix

We $\ell_2$-normalize all item embeddings and compute their pairwise cosine similarities:

$$
\hat{\mathbf{e}}_j = \frac{\mathbf{e}_j}{\|\mathbf{e}_j\|_2}, \quad \forall j \in \{1, \ldots, M\}
$$

$$
S_{jl} = \hat{\mathbf{e}}_j^\top \hat{\mathbf{e}}_l = \cos(\mathbf{e}_j, \mathbf{e}_l)
$$

In matrix form:

$$
\hat{\mathbf{E}} = \text{normalize}(\mathbf{E}, p=2, \text{dim}=-1), \quad \mathbf{S} = \hat{\mathbf{E}} \hat{\mathbf{E}}^\top \in \mathbb{R}^{M \times M}
$$

Note that $S_{jj} = 1$ (every item is perfectly similar to itself) and $-1 \leq S_{jl} \leq 1$ for all $j, l$.

#### Step 3: Weighted Pairwise Similarity (Per-User)

For each user $i$, the diversity loss measures the expected pairwise similarity under the soft selection weights:

$$
\ell_i^{\text{div}} = \sum_{j=1}^{M} \sum_{l=1}^{M} w_{i,j} \cdot S_{jl} \cdot w_{i,l}
$$

This is a quadratic form:

$$
\ell_i^{\text{div}} = \mathbf{w}_i^\top \mathbf{S} \mathbf{w}_i
$$

**Interpretation**: This quantity is high when the model assigns large weights to items that are highly similar to each other (high $S_{jl}$ among items with high $w_{i,j}$ and $w_{i,l}$). It is low when the highly-weighted items are dissimilar.

Expanding the quadratic form to understand its components:

$$
\mathbf{w}_i^\top \mathbf{S} \mathbf{w}_i = \underbrace{\sum_{j=1}^{M} w_{i,j}^2 \cdot S_{jj}}_{\text{self-similarity (always } = \|\mathbf{w}_i\|_2^2 \text{)}} + \underbrace{\sum_{j \neq l} w_{i,j} \cdot w_{i,l} \cdot S_{jl}}_{\text{cross-similarity (what we want to minimize)}}
$$

The self-similarity term $\|\mathbf{w}_i\|_2^2$ cannot be reduced below $1/M$ (uniform distribution), and reducing it means spreading weight more evenly — which itself promotes diversity. The cross-similarity term is directly the weighted average of pairwise similarities among different items.

#### Step 4: Batch Average

The final diversity loss over a batch of $B$ users:

$$
\mathcal{L}_{\text{div}} = \frac{1}{B} \sum_{i=1}^{B} \mathbf{w}_i^\top \mathbf{S} \mathbf{w}_i
$$

### 3.3 Matrix Form — The Complete Formula

Combining all steps, the **complete diversity loss** in one expression:

$$
\boxed{
\mathcal{L}_{\text{div}} = \frac{1}{B} \sum_{i=1}^{B} \left[ \text{softmax}\!\left(\frac{\mathbf{s}_i}{\tau}\right) \right]^\top \!\!\!\left(\hat{\mathbf{E}} \hat{\mathbf{E}}^\top\right) \left[ \text{softmax}\!\left(\frac{\mathbf{s}_i}{\tau}\right) \right]
}
$$

where:
- $\mathbf{s}_i \in \mathbb{R}^M$ are the raw recommendation scores for user $i$
- $\hat{\mathbf{E}} \in \mathbb{R}^{M \times d}$ is the row-wise $\ell_2$-normalized item embedding matrix
- $\tau > 0$ is temperature
- $B$ is the batch size

Or equivalently in compact notation with $\mathbf{W} = \text{softmax}(\mathbf{S}_{\text{score}} / \tau) \in \mathbb{R}^{B \times M}$:

$$
\mathcal{L}_{\text{div}} = \frac{1}{B} \text{tr}\!\left(\mathbf{W} \mathbf{S} \mathbf{W}^\top\right)
$$

where $\text{tr}(\cdot)$ denotes the matrix trace.

---

## 4. Why Not Hard Top-K?

A natural question: why not simply take the top-k items by score, compute their pairwise similarity, and use that as the loss?

**The `argmax`/`argsort` operation is not differentiable.** When you select the top-k items by index, you create a discrete, combinatorial operation. The gradient of "which items are in the top-k" with respect to the scores is zero almost everywhere (the selection doesn't change for small perturbations) and undefined at the boundary (when two items have exactly the same score).

| Approach | Differentiable? | Approximation Quality | Gradient Signal |
|----------|----------------|----------------------|-----------------|
| Hard top-k via `argsort` | ❌ No | Exact | None (zero gradient) |
| Gumbel-Softmax | ✅ Yes | Moderate | Noisy, high variance |
| Our soft top-k (temp. softmax) | ✅ Yes | Excellent at low $\tau$ | Clean, low variance |
| Straight-through estimator | ⚠️ Biased | Exact in forward | Biased gradient |

Our temperature-scaled softmax approach provides an excellent approximation to hard top-k selection while maintaining clean, well-defined gradients throughout the computation graph.

**Example**: With $\tau = 0.1$ and scores $[10.0, 9.5, 9.0, 5.0, 2.0]$:

$$
\mathbf{w} = \text{softmax}([100, 95, 90, 50, 20]) \approx [0.924, 0.061, 0.004, \sim 0, \sim 0]
$$

The vast majority of weight is on the top 2-3 items, effectively making this a "soft top-3". The model receives gradient signal to push these specific items apart in embedding space.

---

## 5. Comparison with DPP-Based Approaches

**Determinantal Point Processes (DPP)** are a popular framework for diverse subset selection. It is important to clarify how our approach differs:

| Aspect | DPP | Our Soft Top-K Loss |
|--------|-----|---------------------|
| **Objective** | Maximize $\det(\mathbf{L}_S)$ for selected subset $S$ | Minimize $\mathbf{w}^\top \mathbf{S} \mathbf{w}$ |
| **Selection mechanism** | Sampling/MAP inference (combinatorial) | Differentiable softmax weighting |
| **Training** | Typically used at inference time | End-to-end differentiable training |
| **Complexity** | $O(M^3)$ for exact MAP, or $O(k^2 M)$ for greedy | $O(M^2 d)$ for similarity matrix |
| **Gradient** | Requires specialized gradient estimators | Standard backpropagation |
| **Relevance-diversity** | Encoded in kernel $L = \text{diag}(q) \cdot S \cdot \text{diag}(q)$ | Balanced via $\lambda$ weight |

**Key difference**: DPP maximizes the determinant of a kernel matrix for a discrete subset, which captures **repulsion** between selected items. Our loss directly penalizes weighted pairwise similarity in a fully differentiable manner, allowing standard gradient descent training.

Both approaches aim to increase diversity, but our method is:
- **Simpler** to implement (no sampling, no specialized inference)
- **Fully differentiable** (no gradient estimation needed)
- **End-to-end trainable** (modifies the learned representations, not just the post-hoc re-ranking)

---

## 6. Proof of Differentiability

We now rigorously prove that $\mathcal{L}_{\text{div}}$ is differentiable (in fact, infinitely differentiable — $C^\infty$) with respect to the model parameters.

### 6.1 Forward Pass Composition

The diversity loss is a composition of the following operations. We trace the computation graph from model parameters $\theta$ to the scalar loss:

$$
\theta \xrightarrow{f_1} \mathbf{E} \xrightarrow{f_2} \hat{\mathbf{E}} \xrightarrow{f_3} \mathbf{S} \xrightarrow{f_5(\cdot, f_4(\theta))} \ell
$$

And separately:

$$
\theta \xrightarrow{g_1} \mathbf{s}_i \xrightarrow{g_2} \mathbf{w}_i \xrightarrow{f_5(\cdot, \mathbf{S})} \ell
$$

where:

| Step | Operation | Function Class |
|------|-----------|---------------|
| $f_1$: RGCN forward | $\theta \mapsto \mathbf{E}$ | Composition of linear transforms + ReLU → $C^\infty$ in practice |
| $f_2$: $\ell_2$-normalization | $\mathbf{e}_j \mapsto \mathbf{e}_j / \|\mathbf{e}_j\|_2$ | $C^\infty$ for $\|\mathbf{e}_j\| \neq 0$ |
| $f_3$: Matrix multiplication | $\hat{\mathbf{E}} \mapsto \hat{\mathbf{E}}\hat{\mathbf{E}}^\top$ | $C^\infty$ (polynomial) |
| $g_1$: Score computation | $\theta \mapsto \mathbf{s}_i$ | $C^\infty$ (linear) |
| $g_2$: Temperature softmax | $\mathbf{s}_i \mapsto \text{softmax}(\mathbf{s}_i / \tau)$ | $C^\infty$ (exp and division by positive quantity) |
| $f_5$: Quadratic form | $(\mathbf{w}_i, \mathbf{S}) \mapsto \mathbf{w}_i^\top \mathbf{S} \mathbf{w}_i$ | $C^\infty$ (polynomial) |
| Batch mean | $\frac{1}{B}\sum_i$ | $C^\infty$ (linear) |

**By the chain rule**, a composition of $C^\infty$ functions is $C^\infty$. Since every step in the computation is $C^\infty$ (given the mild assumption that $\|\mathbf{e}_j\| \neq 0$, which holds in practice since RGCN weights are randomly initialized and continuously updated), the entire loss $\mathcal{L}_{\text{div}}$ is **infinitely differentiable** with respect to all model parameters $\theta$.

### 6.2 Gradient Derivation

We derive the gradient explicitly to show it has a clean, closed-form expression. Focus on one user $i$ (batch averaging is trivial).

**Goal**: Compute $\frac{\partial \ell_i^{\text{div}}}{\partial \mathbf{s}_i}$ where $\ell_i^{\text{div}} = \mathbf{w}_i^\top \mathbf{S} \mathbf{w}_i$.

#### Step A: Gradient of quadratic form w.r.t. $\mathbf{w}_i$

Since $\mathbf{S}$ is symmetric ($S_{jl} = S_{lj}$):

$$
\frac{\partial \ell_i^{\text{div}}}{\partial \mathbf{w}_i} = 2 \mathbf{S} \mathbf{w}_i \in \mathbb{R}^M
$$

> **Proof**: $\ell = \mathbf{w}^\top \mathbf{S} \mathbf{w} = \sum_{j,l} w_j S_{jl} w_l$. Taking $\frac{\partial \ell}{\partial w_k} = \sum_l S_{kl} w_l + \sum_j w_j S_{jk} = 2 \sum_l S_{kl} w_l = 2(\mathbf{S}\mathbf{w})_k$ since $S_{jk} = S_{kj}$.

#### Step B: Jacobian of softmax

The Jacobian of $\mathbf{w} = \text{softmax}(\mathbf{z})$ where $\mathbf{z} = \mathbf{s} / \tau$ is:

$$
\frac{\partial w_j}{\partial z_k} = w_j (\delta_{jk} - w_k)
$$

In matrix form:

$$
\mathbf{J}_{\text{softmax}} = \text{diag}(\mathbf{w}) - \mathbf{w}\mathbf{w}^\top \in \mathbb{R}^{M \times M}
$$

And since $\mathbf{z} = \mathbf{s} / \tau$:

$$
\frac{\partial \mathbf{w}}{\partial \mathbf{s}} = \frac{1}{\tau} \mathbf{J}_{\text{softmax}}
$$

#### Step C: Chain rule

Combining Steps A and B:

$$
\frac{\partial \ell_i^{\text{div}}}{\partial \mathbf{s}_i} = \frac{\partial \mathbf{w}_i}{\partial \mathbf{s}_i}^\top \cdot \frac{\partial \ell_i^{\text{div}}}{\partial \mathbf{w}_i} = \frac{1}{\tau} \mathbf{J}_{\text{softmax}}^\top \cdot 2\mathbf{S}\mathbf{w}_i
$$

Since $\mathbf{J}_{\text{softmax}}$ is symmetric:

$$
\boxed{
\frac{\partial \ell_i^{\text{div}}}{\partial \mathbf{s}_i} = \frac{2}{\tau} \left[\text{diag}(\mathbf{w}_i) - \mathbf{w}_i \mathbf{w}_i^\top\right] \mathbf{S} \mathbf{w}_i
}
$$

Expanding element-wise, the gradient for item $k$'s score is:

$$
\frac{\partial \ell_i^{\text{div}}}{\partial s_{i,k}} = \frac{2}{\tau} w_{i,k} \left[(\mathbf{S}\mathbf{w}_i)_k - \mathbf{w}_i^\top \mathbf{S} \mathbf{w}_i\right]
$$

**Interpretation**: The gradient for item $k$ is proportional to:
- $w_{i,k}$: Items already receiving high weight get larger gradients (the model focuses on adjusting the scores of items it's already likely to recommend)
- $(\mathbf{S}\mathbf{w}_i)_k - \mathbf{w}_i^\top \mathbf{S} \mathbf{w}_i$: The difference between item $k$'s weighted similarity to the soft top-k and the overall expected similarity. If item $k$ is **more similar** to the current soft selection than average, the gradient is **positive**, pushing its score **down** (since we minimize the loss). If item $k$ is **less similar**, the gradient is negative, allowing its score to increase.

This is exactly the behavior we want: the gradient nudges the model to reduce scores of items that are redundant (similar to what's already recommended) and increase scores of items that are different.

### 6.3 Smoothness Class

**Theorem**: $\mathcal{L}_{\text{div}}(\theta)$ is $C^\infty$ (infinitely differentiable) with respect to $\theta$, provided all item embeddings have non-zero norm.

**Proof sketch**:

1. The softmax function $\sigma: \mathbb{R}^M \to (0,1)^M$ is $C^\infty$ because:
   - $\exp(\cdot)$ is $C^\infty$
   - Division by a positive quantity ($\sum_j \exp(z_j) > 0$ always) preserves smoothness
   - The output is always strictly positive: $w_j > 0, \forall j$

2. $\ell_2$-normalization $\mathbf{e} \mapsto \mathbf{e}/\|\mathbf{e}\|_2$ is $C^\infty$ on $\mathbb{R}^d \setminus \{\mathbf{0}\}$ because:
   - $\|\mathbf{e}\|_2 = \sqrt{\sum_k e_k^2}$ is $C^\infty$ for $\mathbf{e} \neq \mathbf{0}$
   - Division by a nonzero quantity preserves smoothness

3. The quadratic form $\mathbf{w}^\top \mathbf{S} \mathbf{w}$ is a polynomial in $\mathbf{w}$ and $\mathbf{S}$, hence $C^\infty$.

4. The RGCN forward pass is a composition of:
   - Linear transformations: $C^\infty$
   - ReLU activations: $C^0$ but $C^\infty$ almost everywhere (non-differentiable only at exactly zero, a set of measure zero). In practice, this does not affect gradient-based optimization.

5. By the **chain rule for smooth maps**: $f \circ g \in C^\infty$ when both $f, g \in C^\infty$. $\square$

**Practical note**: In PyTorch, all these operations have registered backward functions (autograd), so the gradient is computed automatically. The above proof confirms that autograd will produce correct, well-defined gradients throughout training.

---

## 7. Integration into the KECRS System

### 7.1 Combined Training Objective

The total training loss for KECRS with diversity regularization is:

$$
\boxed{
\mathcal{L}_{\text{total}} = \underbrace{\mathcal{L}_{\text{CE}}(\mathbf{s}_i, y_i)}_{\text{accuracy term}} + \underbrace{\lambda \cdot \mathcal{L}_{\text{div}}(\mathbf{s}_i, \hat{\mathbf{E}})}_{\text{diversity regularizer}}
}
$$

where:

- $\mathcal{L}_{\text{CE}}$ is the standard **cross-entropy loss** for recommendation:

$$
\mathcal{L}_{\text{CE}} = -\frac{1}{B}\sum_{i=1}^{B} \log \frac{\exp(s_{i, y_i})}{\sum_{j=1}^{M} \exp(s_{i,j})}
$$

  Here $y_i$ is the index of the ground-truth item for user $i$.

- $\mathcal{L}_{\text{div}}$ is the diversity loss defined in Section 3.

- $\lambda \geq 0$ is the **diversity weight** hyperparameter that controls the strength of the diversity regularization.

**Expanded form of the complete loss**:

$$
\mathcal{L}_{\text{total}} = -\frac{1}{B}\sum_{i=1}^{B} \log \frac{\exp(s_{i, y_i})}{\sum_{j=1}^{M} \exp(s_{i,j})} + \frac{\lambda}{B} \sum_{i=1}^{B} \left[\text{softmax}\!\left(\frac{\mathbf{s}_i}{\tau}\right)\right]^\top \left(\hat{\mathbf{E}}\hat{\mathbf{E}}^\top\right) \left[\text{softmax}\!\left(\frac{\mathbf{s}_i}{\tau}\right)\right]
$$

### 7.2 System Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    KECRS System                         │
│                                                         │
│  ┌─────────────┐   ┌───────────────────────────────┐   │
│  │  Knowledge   │   │     RGCN Encoder               │   │
│  │   Graph      │──▶│  Produces E ∈ ℝ^{M×d}         │   │
│  │ (entities,   │   │  (all movie embeddings)        │   │
│  │  relations)  │   └──────────┬────────────────────┘   │
│  └─────────────┘              │                         │
│                               ▼                         │
│  ┌─────────────┐   ┌───────────────────────────────┐   │
│  │  User's      │   │  Self-Attention Aggregation    │   │
│  │  Seed Set    │──▶│  u_i = Attn(E[seed_set])     │   │
│  │ (mentioned   │   └──────────┬────────────────────┘   │
│  │  entities)   │              │                         │
│  └─────────────┘              ▼                         │
│                     ┌───────────────────────────────┐   │
│                     │  Score Computation             │   │
│                     │  s_i = E · u_i + bias          │   │
│                     └──────────┬────────────────────┘   │
│                                │                         │
│                  ┌─────────────┼─────────────┐          │
│                  ▼                           ▼          │
│     ┌─────────────────────┐   ┌────────────────────┐   │
│     │ Cross-Entropy Loss  │   │  Diversity Loss    │   │
│     │ ℒ_CE(s_i, y_i)     │   │  ℒ_div(s_i, Ê)    │   │
│     └──────────┬──────────┘   └────────┬───────────┘   │
│                │                        │               │
│                ▼                        ▼               │
│     ┌──────────────────────────────────────────┐       │
│     │       ℒ_total = ℒ_CE + λ · ℒ_div        │       │
│     └──────────────────────────────────────────┘       │
│                         │                               │
│                         ▼                               │
│                  Backpropagation                         │
│          (updates RGCN + attention weights)              │
└─────────────────────────────────────────────────────────┘
```

### 7.3 Training Pipeline

The training process with diversity loss follows these steps for each batch:

1. **Forward pass through RGCN**: Compute node embeddings $\mathbf{E}$ from the knowledge graph.
2. **User representation**: For each user's seed set (mentioned entities), apply self-attention over their embeddings to produce user embedding $\mathbf{u}_i$.
3. **Score computation**: $\mathbf{s}_i = \mathbf{E} \cdot \mathbf{u}_i + \text{bias}$.
4. **Cross-entropy loss**: $\mathcal{L}_{\text{CE}} = \text{CrossEntropy}(\mathbf{s}_i, y_i)$.
5. **Diversity loss** (if $\lambda > 0$):
   - Compute soft weights: $\mathbf{w}_i = \text{softmax}(\mathbf{s}_i / \tau)$
   - Normalize embeddings: $\hat{\mathbf{E}} = \text{normalize}(\mathbf{E})$
   - Compute similarity matrix: $\mathbf{S} = \hat{\mathbf{E}}\hat{\mathbf{E}}^\top$
   - Compute weighted similarity: $\ell_i^{\text{div}} = \mathbf{w}_i^\top \mathbf{S} \mathbf{w}_i$
6. **Combined loss**: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \lambda \cdot \mathcal{L}_{\text{div}}$
7. **Backward pass**: Compute gradients of $\mathcal{L}_{\text{total}}$ w.r.t. all parameters.
8. **Optimizer step**: Update parameters via Adam optimizer.

**Key insight about gradient flow**: The diversity loss $\mathcal{L}_{\text{div}}$ gradient flows back through:
- The **scores** $\mathbf{s}_i$ → adjusting how user embeddings map to item scores
- The **item embeddings** $\mathbf{E}$ → adjusting the RGCN to produce more diverse representations
- The **attention layer** → adjusting how user preferences are aggregated

This means the model learns **holistically** to produce diverse recommendations, rather than just re-ranking post-hoc.

---

## 8. Per-Turn Diversity Evaluation Metrics

Beyond the training loss, we evaluate diversity using three complementary per-turn metrics computed on the hard top-k items during validation/testing.

### 8.1 Intra-List Distance (ILD@k)

**Definition**: Average pairwise cosine distance among the top-k recommended items.

$$
\text{ILD@}k = \frac{2}{k(k-1)} \sum_{j=1}^{k} \sum_{l=j+1}^{k} \left(1 - \cos(\mathbf{e}_{r_j}, \mathbf{e}_{r_l})\right)
$$

where $r_1, r_2, \ldots, r_k$ are the top-k recommended item indices, and $\cos(\cdot, \cdot)$ denotes cosine similarity.

| ILD Value | Interpretation |
|-----------|---------------|
| 0 | All top-k items have identical embeddings |
| 0.5 | Moderate diversity |
| 1.0 | All top-k items are maximally dissimilar (orthogonal embeddings) |
| > 1.0 | Items are anti-correlated (possible but rare with KG embeddings) |

**ILD directly measures the geometric spread of recommended items in embedding space.** It is the evaluation-time analog of the diversity loss.

### 8.2 KG-Entity Coverage@k

**Definition**: Fraction of all reachable KG entities that are covered by the 1-hop neighborhoods of the top-k recommended items.

$$
\text{KG\_Coverage@}k = \frac{\left|\bigcup_{j=1}^{k} \mathcal{N}(r_j)\right|}{|\mathcal{V}_{\text{reachable}}|}
$$

where:
- $\mathcal{N}(r_j) = \{v : (r_j, \text{rel}, v) \in \mathcal{E}\}$ is the set of 1-hop neighbors of item $r_j$ in the KG.
- $\mathcal{V}_{\text{reachable}} = \bigcup_{m \in \text{all movies}} \mathcal{N}(m)$ is the set of all entities reachable from any movie.

**Interpretation**: If the top-k recommendations collectively "touch" many different entities in the KG (different actors, directors, genres, studios, etc.), coverage is high. If they all share the same neighbors (same franchise, same actors), coverage is low.

### 8.3 Category Coverage@k

**Definition**: Fraction of distinct KG relation types represented in the 1-hop neighborhoods of the top-k items.

$$
\text{Cat\_Coverage@}k = \frac{\left|\bigcup_{j=1}^{k} \left\{ \text{rel} : (r_j, \text{rel}, v) \in \mathcal{E} \right\}\right|}{|\mathcal{R}|}
$$

where $|\mathcal{R}|$ is the total number of distinct relation types in the KG.

**Interpretation**: This measures whether the recommended items span different **types** of relationships. For example, if all top-k items only have "genre" and "actor" edges but no "director", "writer", or "award" edges, category coverage is low. Diverse recommendations should cover many relation types.

### Metric Relationships

| Metric | What it Measures | Granularity | Sensitive to |
|--------|-----------------|------------|--------------|
| ILD@k | Embedding-space diversity | Fine-grained | Learned representation quality |
| KG_Coverage@k | Entity-level KG diversity | Entity-level | KG connectivity, item distinctness |
| Cat_Coverage@k | Relation-type diversity | Category-level | KG schema breadth coverage |

These metrics are **complementary**: ILD captures smooth embedding distances, KG Coverage captures discrete entity spread, and Category Coverage captures structural variety.

---

## 9. Hyperparameters

| Parameter | Flag | Default | Range | Effect |
|-----------|------|---------|-------|--------|
| Diversity weight $\lambda$ | `--diversity-weight` | 0.0 | $[0, 1]$ | Strength of diversity regularization. 0 = disabled (baseline). Higher values push stronger diversity at potential accuracy cost. |
| Temperature $\tau$ | `--diversity-temperature` | 0.1 | $(0, \infty)$ | Controls softmax sharpness. Lower = closer to hard top-k. Higher = smoother, more uniform weights. |
| Top-k $k$ | `--diversity-topk` | 10 | $\{1, 2, \ldots\}$ | Number of top items for evaluation metrics (ILD, coverage). Does NOT affect the loss (loss uses all items weighted by softmax). |

### Tuning Guidelines

**Diversity weight $\lambda$**:
- Start with $\lambda = 0$ (baseline) to establish accuracy benchmarks.
- Gradually increase: try $\lambda \in \{0.001, 0.01, 0.05, 0.1, 0.5\}$.
- Monitor recall@50 (accuracy) and ILD@10 (diversity) — look for the "knee" of the Pareto curve where diversity improves significantly without large accuracy drops.
- Values above 1.0 are typically too aggressive and will hurt accuracy.

**Temperature $\tau$**:
- $\tau = 0.1$ works well in most settings.
- If the model has very high scores (e.g., logits > 50), you may need higher $\tau$ to prevent numerical overflow in $\exp(s/\tau)$.
- Lower $\tau$ focuses the diversity penalty on fewer top items; higher $\tau$ spreads it across more items.

**Top-k $k$**:
- $k = 10$ is standard for diversity evaluation in recommendation systems.
- Matches typical "recommendation page" lengths in real applications.
- Can also evaluate at $k = 5, 20, 50$ for different granularities.

---

## 10. Implementation Reference

The diversity loss is implemented across three files in the KECRS codebase:

### Core Loss Function (`parlai/agents/kecrs/modules.py`)

```python
def compute_diversity_loss(self, scores, nodes_features):
    """
    Compute a differentiable diversity loss using soft top-k selection.
    Uses softmax-weighted pairwise cosine similarity over item embeddings.
    Minimizing this encourages the model to spread scores across dissimilar items.
    """
    # Soft item selection weights via temperature-scaled softmax
    # scores: [batch, M], nodes_features: [M, dim]
    item_weights = F.softmax(scores / self.diversity_temperature, dim=-1)  # [batch, M]

    # Normalize item embeddings for cosine similarity
    normed_features = F.normalize(nodes_features, p=2, dim=-1)  # [M, dim]

    # Compute cosine similarity matrix: [M, M]
    sim_matrix = torch.mm(normed_features, normed_features.t())  # [M, M]

    # Weighted pairwise similarity per user:
    # w^T S w gives expected pairwise similarity of the soft selection
    Sw = torch.mm(item_weights, sim_matrix)  # [batch, M]
    weighted_sim = (item_weights * Sw).sum(dim=-1)  # [batch]

    diversity_loss = weighted_sim.mean()
    return diversity_loss
```

### Forward Pass Integration (`parlai/agents/kecrs/modules.py`)

```python
def forward(self, seed_sets, labels):
    u_emb, nodes_features = self.kg_movie_score(seed_sets)
    scores = F.linear(u_emb, nodes_features, self.output.bias)
    base_loss = self.criterion(scores, labels)

    if self.diversity_weight > 0:
        diversity_loss = self.compute_diversity_loss(scores, nodes_features)
        loss = base_loss + self.diversity_weight * diversity_loss
    else:
        diversity_loss = torch.tensor(0.0, device=scores.device)
        loss = base_loss

    return dict(
        scores=scores.detach(),
        base_loss=base_loss,
        diversity_loss=diversity_loss,
        loss=loss,
        nodes_features=nodes_features.detach()
    )
```

### CLI Arguments (`parlai/agents/kecrs/kecrs.py`)

```python
agent.add_argument("-divw", "--diversity-weight", type=float, default=0.0,
                   help="Weight for the KG-based diversity loss. 0 disables it.")
agent.add_argument("-divt", "--diversity-temperature", type=float, default=0.1,
                   help="Temperature for softmax in diversity loss soft top-k.")
agent.add_argument("-divk", "--diversity-topk", type=int, default=10,
                   help="Top-k for per-turn diversity evaluation metrics.")
```

### Per-Turn Metrics (`parlai/agents/kecrs/kecrs.py`, in `eval_step`)

```python
# ILD@k: Intra-List Distance
topk_embs = node_feats[topk_items]
normed_embs = F.normalize(topk_embs, p=2, dim=-1)
cos_sim_matrix = torch.mm(normed_embs, normed_embs.t())
mask = torch.triu(torch.ones(div_k, div_k), diagonal=1).bool()
pairwise_distances = 1.0 - cos_sim_matrix[mask]
ild_value = pairwise_distances.mean().item()

# KG-Entity Coverage@k
covered_entities = set()
for item_id in topk_items:
    covered_entities.update(self.model.movie_kg_neighbors.get(item_id, set()))
kg_cov = len(covered_entities) / self.model.total_reachable_entities

# Category Coverage@k
covered_relations = set()
for item_id in topk_items:
    if item_id in self.model.kg:
        for rel, tail in self.model.kg[item_id]:
            covered_relations.add(rel)
cat_cov = len(covered_relations) / self.model.total_relation_types
```

---

## 11. Summary

| Aspect | Details |
|--------|---------|
| **What** | A differentiable diversity loss based on soft top-k item selection via temperature-scaled softmax |
| **Formula** | $\mathcal{L}\_{\text{div}} = \frac{1}{B} \sum\_{i=1}^{B} \mathbf{w}\_i^\top \mathbf{S} \mathbf{w}\_i$ where $\mathbf{w}\_i = \text{softmax}(\mathbf{s}\_i / \tau)$ and $\mathbf{S} = \hat{\mathbf{E}}\hat{\mathbf{E}}^\top$ |
| **Combined loss** | $\mathcal{L}\_{\text{total}} = \mathcal{L}\_{\text{CE}} + \lambda \cdot \mathcal{L}\_{\text{div}}$ |
| **Why it works** | Penalizes high pairwise similarity among top-scored items, pushing the model to recommend diverse items |
| **Differentiability** | $C^\infty$ — proven by composition of smooth functions (softmax, normalization, quadratic form) |
| **Gradient** | $\frac{\partial \ell}{\partial s\_k} = \frac{2}{\tau} w\_k [(\mathbf{S}\mathbf{w})\_k - \mathbf{w}^\top\mathbf{S}\mathbf{w}]$ — reduces scores of redundant items, increases scores of diverse items |
| **Evaluation** | ILD@k (embedding distance), KG Coverage@k (entity spread), Category Coverage@k (relation-type breadth) |
| **Key hyperparameters** | $\lambda$ (loss weight), $\tau$ (temperature), $k$ (evaluation top-k) |

---

*This document describes the diversity loss function implemented in the KECRS system on the `kecrs-loss-diverse` branch.*
