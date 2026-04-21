# Speaking guide — Slide 7, Model Architectures

Target: ~15 seconds per model on the slide itself (they'll read the spec
table while you narrate). Below is a confident, plain-English version of
each architecture, in the order they appear on the slide.

---

## 1 · Ridge Regression — "the linear baseline"

**What to say (spoken, ~15 sec):**

> Ridge is the interpretable baseline. It learns one coefficient per
> feature — so a prediction is just the intercept plus a weighted sum of
> the 41 inputs. The α=10 L² penalty stops the model from assigning
> wildly large coefficients to correlated features. Because it's just 41
> numbers plus an intercept, I ship the whole model to the browser —
> that's what powers the live predictor.

**If asked to go deeper:**
- Closed-form solution: `β̂ = (XᵀX + αI)⁻¹ Xᵀy`. No iterative training.
- α=10 was cross-validated on the training set across α ∈ {0.1, 1, 10, 100}.
- The intercept is +1.191 VORP — that's the average outcome for a
  training-set-mean player (makes sense because standardized features
  center the intercept on the mean of y).
- Coefficients are published in `model_meta.json` — the biggest positive
  weight is `ws_40_vs_pos`, the biggest negative is `ws_40` itself
  (they're correlated, and ridge balances them).

---

## 2 · XGBoost Regressor — "200 shallow trees stacked"

**What to say (spoken, ~20 sec):**

> XGBoost is an ensemble of 200 small decision trees, built one at a
> time. Each new tree is trained on the residuals — the mistakes — of
> all the trees before it. Each tree is shallow: max depth 4 means at
> most sixteen leaves, so no single tree can overfit. A learning rate of
> 0.01 means each tree contributes tiny adjustments to the running
> prediction. Row and column subsampling at 70% decorrelates the trees —
> each tree sees a different random slice of the data. Together, 200 of
> these stacked up carve the non-linear feature space.

**If asked to go deeper:**
- Objective is `reg:squarederror` — minimize squared loss, same as OLS,
  but optimized greedily one tree at a time via second-order gradient
  information (Newton-style).
- `min_child_weight=5` means a split can't create a leaf with fewer
  than 5 "samples" (really, summed Hessian). Prevents memorization.
- The greedy split search is what lets trees beat deep learning on
  small-n tabular data: at each node, XGBoost tries every feature and
  every threshold, picks the split that maximizes the gain.
- SHAP `TreeExplainer` gives exact per-prediction attributions in
  polynomial time — no approximation needed, because tree ensembles
  have an exact decomposition into feature contributions.

---

## 3 · Multi-Layer Perceptron — "a small neural net, mostly for comparison"

**What to say (spoken, ~15 sec):**

> The MLP is a two-hidden-layer neural network: 41 inputs, then 128
> neurons, then 64 neurons, then one output. Each layer is a
> fully-connected weighted sum followed by a ReLU activation — which is
> what lets the network learn non-linear relationships. Trained with
> Adam and L² weight decay. Early stopping on a 15% validation slice;
> it converged in 25 epochs. About 13,000 trainable parameters. It's
> mostly here as a sanity check — deep learning shines at ten-thousand-
> plus samples, so at 851 it predictably underperforms the boosted
> trees.

**If asked to go deeper:**
- ReLU activation `f(x) = max(0, x)`: "negative → zero, positive stays
  positive". Without a non-linearity like this, stacking linear layers
  collapses to a single linear model — same as Ridge.
- Adam = stochastic gradient descent with per-parameter adaptive
  learning rates. Converges faster than vanilla SGD.
- Weight count is exactly `(41×128 + 128) + (128×64 + 64) + (64×1 + 1)
  = 5,248 + 8,256 + 65 = 13,569` trainable.
- Early stopping means: every epoch, evaluate on a held-out 15% of
  training; if val loss hasn't improved in 10 epochs, stop. Converged
  in 25 epochs here, well short of the 1000-epoch ceiling.

---

## 4 · Per-Position K-Means — "unsupervised archetypes"

**What to say (spoken, ~20 sec):**

> K-means is unsupervised — it groups players using only the features,
> with no reference to NBA outcomes. For each position group, I pick k
> centroids randomly, assign each player to the nearest one, move the
> centroid to the mean of its group, and repeat until nothing changes.
> I fit three separate K-means — k=3 for guards, k=3 for forwards, k=2
> for centers. A single global K-means just kept re-discovering
> "interior vs perimeter"; per-position gives richer archetypes within
> each role. The clustering itself happens in the full 41-dimensional
> scaled feature space; the PCA projection is only for the 2D scatter.

**If asked to go deeper:**
- Objective is `argmin_C Σ ‖xᵢ − μ_C(i)‖²` — sum of squared distances
  from each point to its assigned centroid. That sum is called
  "inertia"; the three I fit have inertias 12632, 13515, and 3898.
- `k-means++` initialization: the first centroid is random, subsequent
  ones are chosen with probability proportional to their squared
  distance from existing centroids. Avoids unlucky starts that local-
  minimum on a bad clustering.
- `n_init=10` means run the whole algorithm 10 times with different
  random seeds and keep the lowest-inertia result.
- PCA is a separate step: project 41-dim → 2-dim by finding the two
  orthogonal axes with the most variance. First two components explain
  36.6% of total variance. Useful for plotting, not for clustering —
  clustering happens in the full 41-dim space.

---

## One thing to remember

The room will respect you more for saying "I don't know — that's outside
what I measured" than for bluffing. If someone asks "why not a
transformer" or "why not catboost", the honest answer is: tabular data at
n=851 is the regime where shallow tree ensembles dominate, and the
project scope didn't justify scanning more model families.
