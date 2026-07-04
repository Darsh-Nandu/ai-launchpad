/* Module 6 - Machine Learning Algorithms (part 5) */

CONTENT["ml/cross-validation"] = {
  html: String.raw`
<h1>Cross-Validation</h1>
<p class="lead">One train/test split gives you one noisy estimate of performance -
and a temptation to tune until you've quietly overfit the test set.
Cross-validation gives you many estimates, an uncertainty band, and a protocol
that keeps you honest.</p>

<h2>The problem with a single split</h2>
<p>Split 1,000 rows 80/20 and your score rests on 200 test examples. Re-split with
a different seed and the score can easily move by several points - you're
measuring the split as much as the model. Worse: every time you peek at the test
score and adjust something, you leak a little of the test set into your decisions.
After fifty peeks, the test score is a rigged election.</p>

<h2>K-fold cross-validation</h2>
<p>Cut the training data into \(k\) equal folds (5 or 10 is standard). Train on
\(k-1\) folds, validate on the held-out one; rotate so every fold gets a turn as
validator; average the \(k\) scores.</p>
<div class="diagram">
<svg viewBox="0 0 660 250" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Five rows showing 5-fold cross validation: in each row a different fifth of the data is the validation fold">
  <text x="20" y="38" class="d-text-sm">round 1</text>
  <rect x="90" y="22" width="100" height="26" rx="4" class="d-box-soft"/>
  <rect x="194" y="22" width="100" height="26" rx="4" class="d-box-muted"/>
  <rect x="298" y="22" width="100" height="26" rx="4" class="d-box-muted"/>
  <rect x="402" y="22" width="100" height="26" rx="4" class="d-box-muted"/>
  <rect x="506" y="22" width="100" height="26" rx="4" class="d-box-muted"/>
  <text x="20" y="80" class="d-text-sm">round 2</text>
  <rect x="90" y="64" width="100" height="26" rx="4" class="d-box-muted"/>
  <rect x="194" y="64" width="100" height="26" rx="4" class="d-box-soft"/>
  <rect x="298" y="64" width="100" height="26" rx="4" class="d-box-muted"/>
  <rect x="402" y="64" width="100" height="26" rx="4" class="d-box-muted"/>
  <rect x="506" y="64" width="100" height="26" rx="4" class="d-box-muted"/>
  <text x="20" y="122" class="d-text-sm">round 3</text>
  <rect x="90" y="106" width="100" height="26" rx="4" class="d-box-muted"/>
  <rect x="194" y="106" width="100" height="26" rx="4" class="d-box-muted"/>
  <rect x="298" y="106" width="100" height="26" rx="4" class="d-box-soft"/>
  <rect x="402" y="106" width="100" height="26" rx="4" class="d-box-muted"/>
  <rect x="506" y="106" width="100" height="26" rx="4" class="d-box-muted"/>
  <text x="20" y="164" class="d-text-sm">round 4</text>
  <rect x="90" y="148" width="100" height="26" rx="4" class="d-box-muted"/>
  <rect x="194" y="148" width="100" height="26" rx="4" class="d-box-muted"/>
  <rect x="298" y="148" width="100" height="26" rx="4" class="d-box-muted"/>
  <rect x="402" y="148" width="100" height="26" rx="4" class="d-box-soft"/>
  <rect x="506" y="148" width="100" height="26" rx="4" class="d-box-muted"/>
  <text x="20" y="206" class="d-text-sm">round 5</text>
  <rect x="90" y="190" width="100" height="26" rx="4" class="d-box-muted"/>
  <rect x="194" y="190" width="100" height="26" rx="4" class="d-box-muted"/>
  <rect x="298" y="190" width="100" height="26" rx="4" class="d-box-muted"/>
  <rect x="402" y="190" width="100" height="26" rx="4" class="d-box-muted"/>
  <rect x="506" y="190" width="100" height="26" rx="4" class="d-box-soft"/>
  <text x="330" y="240" text-anchor="middle" class="d-text-sm">green = validation fold · gray = training folds · final score = mean ± std of the 5</text>
</svg>
<div class="caption">Every row is one train/validate round; every example validates exactly once.</div>
</div>
<p>You get \(k\) scores → report <strong>mean ± std</strong>. The std matters: "0.86 ±
0.01" and "0.86 ± 0.08" are very different claims
(<a href="#/math/confidence-intervals">Module 1</a> again). Variants you'll need:</p>
<ul>
  <li><strong>Stratified k-fold</strong> - each fold preserves class proportions.
      Default for classification; essential when imbalanced.</li>
  <li><strong>TimeSeriesSplit</strong> - folds respect chronology (train on past,
      validate on future). Shuffling time-series data lets the model "remember the
      future" - one of the most common real-world evaluation sins.</li>
  <li><strong>GroupKFold</strong> - all rows from one patient/user/session stay in the
      same fold. Without it, near-duplicate rows straddle the boundary and scores
      inflate.</li>
</ul>

<h2>In code - including the part everyone gets wrong</h2>
<pre><code class="language-python">from sklearn.model_selection import (cross_val_score, StratifiedKFold,
                                     train_test_split)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True)

# hold out a FINAL test set first - CV happens inside the training portion
X_tr, X_test, y_tr, y_test = train_test_split(X, y, stratify=y,
                                              test_size=0.2, random_state=0)

# the pipeline is the leak-proofing: scaler re-fits inside each fold
pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
scores = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="f1")
print(f"CV F1: {scores.mean():.3f} ± {scores.std():.3f}")

# ...tune hyperparameters using CV scores only...

# the test set is touched ONCE, at the very end:
pipe.fit(X_tr, y_tr)
print("final test F1:", pipe.score(X_test, y_test).round(3))
</code></pre>
<div class="callout warn">
  <span class="co-title">The golden protocol</span>
  <strong>Test set:</strong> locked in a vault, used once, at the end.
  <strong>CV on the training set:</strong> for all model choices and tuning.
  <strong>Every fitted preprocessing step</strong> (scaling, imputation, encoding,
  feature selection, SMOTE) lives <em>inside</em> the pipeline so it re-fits per fold.
  Preprocess-then-CV on pre-transformed data is the classic silent leak - the
  validation folds contributed to the statistics that transformed them.
</div>

<h2>How many folds?</h2>
<table>
  <tr><th>Choice</th><th>Trade-off</th></tr>
  <tr><td>k = 5</td><td>the pragmatic default: decent estimate, 5× cost</td></tr>
  <tr><td>k = 10</td><td>slightly less pessimistic bias, 2× the cost of k=5; standard in papers</td></tr>
  <tr><td>Leave-one-out (k = m)</td><td>nearly unbiased but expensive and high-variance; only for tiny datasets</td></tr>
  <tr><td>Repeated k-fold</td><td>repeat 5-fold with different shuffles for a tighter mean - cheap insurance for small data</td></tr>
</table>

<h2>Common interview questions</h2>
<div class="qa">
  <div class="qa-q">Q1. Why cross-validate instead of a single validation split?</div>
  <div class="qa-a"><p>A single split's score has high variance (it depends on which rows landed where)
  and wastes data (the validation slice never trains). CV uses every row for both roles
  across rounds and yields mean ± std - an estimate <em>with</em> an error bar.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q2. When is standard k-fold actively wrong?</div>
  <div class="qa-a"><p>Time series (shuffling leaks the future - use TimeSeriesSplit), grouped data
  (a user's rows split across folds are near-duplicates - use GroupKFold), and any
  pipeline whose preprocessing was fit before splitting (leakage regardless of splitter).</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q3. Your CV score is 0.91 but the final test score is 0.82. Name likely causes.</div>
  <div class="qa-a"><p>Tuning overfit to the CV folds (too many decisions read off the same folds),
  leakage in preprocessing, group/time structure ignored by the splitter, or distribution
  shift between the data and the final test period. The gap itself is the diagnostic.</p></div>
</div>

<h2>Quick quiz</h2>
<div class="quiz">
  <div class="quiz-q" data-answer="1">
    <p class="quiz-question">1. In 5-fold CV, each example is used for validation…</p>
    <button class="quiz-opt">5 times</button>
    <button class="quiz-opt">Exactly once</button>
    <button class="quiz-opt">A random number of times</button>
    <div class="quiz-explain hidden">Each fold serves as validator in exactly one round (and as training data in the other four).</div>
  </div>
  <div class="quiz-q" data-answer="2">
    <p class="quiz-question">2. Scaling the full dataset, then running CV on the scaled data, is wrong because…</p>
    <button class="quiz-opt">Scaling changes the labels</button>
    <button class="quiz-opt">CV requires raw features</button>
    <button class="quiz-opt">Each validation fold influenced the μ/σ used to transform its own training data - leakage</button>
    <div class="quiz-explain hidden">The transform was fit on data that includes the "unseen" folds. Put the scaler in the pipeline so it re-fits on each round's training folds only.</div>
  </div>
  <div class="quiz-q" data-answer="0">
    <p class="quiz-question">3. For a dataset of daily sales through time, the right CV scheme is…</p>
    <button class="quiz-opt">TimeSeriesSplit - always train on the past, validate on the future</button>
    <button class="quiz-opt">Shuffled 10-fold</button>
    <button class="quiz-opt">Leave-one-out</button>
    <div class="quiz-explain hidden">Shuffled folds let the model train on tomorrow to predict yesterday - a fantasy that collapses in production. Chronological splits simulate real deployment.</div>
  </div>
</div>
`};

CONTENT["ml/bias-variance"] = {
  html: String.raw`
<h1>Bias-Variance Tradeoff</h1>
<p class="lead">Why do models fail? In exactly two ways: by being too simple to
capture the truth (bias) or too flexible to ignore the noise (variance).
Every hyperparameter you will ever tune is secretly a dial between these two.</p>

<h2>The two failure modes, felt first</h2>
<p>Fit a straight line to clearly curved data: it misses systematically, in the
same way, on every resample - <strong>underfitting / high bias</strong>. The model's
assumptions are too rigid to represent reality.</p>
<p>Fit a degree-15 polynomial to 20 noisy points: it threads every point, wiggling
wildly - and a fresh sample of 20 points from the same process would produce a
<em>completely different</em> wiggle - <strong>overfitting / high variance</strong>.
The model is reading meaning into noise.</p>
<div class="diagram">
<svg viewBox="0 0 660 240" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Three fits to the same curved data: a straight line underfits, a moderate curve fits well, a wild polynomial overfits">
  <!-- underfit -->
  <rect x="15" y="25" width="195" height="170" rx="8" class="d-box-muted"/>
  <text x="112" y="47" text-anchor="middle" class="d-text">high bias</text>
  <circle cx="45" cy="150" r="4" class="d-dot"/><circle cx="75" cy="110" r="4" class="d-dot"/>
  <circle cx="105" cy="95" r="4" class="d-dot"/><circle cx="135" cy="100" r="4" class="d-dot"/>
  <circle cx="165" cy="130" r="4" class="d-dot"/><circle cx="190" cy="165" r="4" class="d-dot"/>
  <line x1="35" y1="130" x2="200" y2="120" class="d-line-accent"/>
  <text x="112" y="185" text-anchor="middle" class="d-text-sm">too rigid: misses the curve</text>
  <!-- good -->
  <rect x="232" y="25" width="195" height="170" rx="8" class="d-box-muted"/>
  <text x="329" y="47" text-anchor="middle" class="d-text">good fit</text>
  <circle cx="262" cy="150" r="4" class="d-dot"/><circle cx="292" cy="110" r="4" class="d-dot"/>
  <circle cx="322" cy="95" r="4" class="d-dot"/><circle cx="352" cy="100" r="4" class="d-dot"/>
  <circle cx="382" cy="130" r="4" class="d-dot"/><circle cx="407" cy="165" r="4" class="d-dot"/>
  <path d="M 252 165 Q 322 60 417 172" class="d-curve"/>
  <text x="329" y="185" text-anchor="middle" class="d-text-sm">captures signal, ignores noise</text>
  <!-- overfit -->
  <rect x="449" y="25" width="195" height="170" rx="8" class="d-box-muted"/>
  <text x="546" y="47" text-anchor="middle" class="d-text">high variance</text>
  <circle cx="479" cy="150" r="4" class="d-dot"/><circle cx="509" cy="110" r="4" class="d-dot"/>
  <circle cx="539" cy="95" r="4" class="d-dot"/><circle cx="569" cy="100" r="4" class="d-dot"/>
  <circle cx="599" cy="130" r="4" class="d-dot"/><circle cx="624" cy="165" r="4" class="d-dot"/>
  <path d="M 469 170 C 480 90, 495 190, 509 110 C 520 60, 530 140, 539 95 C 548 60, 560 150, 569 100 C 578 65, 590 175, 599 130 C 607 100, 618 190, 632 150"
        class="d-line-accent" fill="none"/>
  <text x="546" y="185" text-anchor="middle" class="d-text-sm">threads every noisy point</text>
</svg>
<div class="caption">Same data, three model complexities. Left fails on everything;
right fails only on new data - which is worse, because it looks like success.</div>
</div>

<h2>The decomposition (where the names come from)</h2>
<p>For squared error, the expected test error at a point provably splits into
three terms - this is a theorem, not a metaphor:</p>
<div class="math-box">
$$\mathbb{E}\big[(y - \hat{f}(x))^2\big] =
\underbrace{\big(\mathbb{E}[\hat{f}(x)] - f(x)\big)^2}_{\text{bias}^2}
+ \underbrace{\mathbb{E}\big[(\hat{f}(x) - \mathbb{E}[\hat{f}(x)])^2\big]}_{\text{variance}}
+ \underbrace{\sigma^2}_{\text{irreducible noise}}$$
</div>
<p>Imagine retraining your model on many parallel-universe datasets drawn from the
same source. <strong>Bias</strong>: how far the <em>average</em> of all those models sits
from the truth (a systematic miss). <strong>Variance</strong>: how much the models
<em>disagree with each other</em> (sensitivity to which sample you happened to get).
<strong>Noise</strong>: the part of \(y\) no model can predict - the error floor.
Complexity moves the two terms in opposite directions; total error is U-shaped,
and the tuning game is finding the bottom of the U.</p>

<h2>Diagnosis: read the train/validation gap</h2>
<table>
  <tr><th>Symptom</th><th>Diagnosis</th><th>Treatment</th></tr>
  <tr><td>Train score low, val score low (both bad, small gap)</td><td><strong>High bias</strong></td><td>bigger/more flexible model, more/better features, less regularization, train longer</td></tr>
  <tr><td>Train score great, val score much worse (big gap)</td><td><strong>High variance</strong></td><td>more data, regularization, simpler model, feature selection, ensembling, early stopping</td></tr>
  <tr><td>Both great</td><td>done (verify no leakage!)</td><td>ship it</td></tr>
</table>
<p>Note the asymmetry of remedies: <strong>more data cures variance but never
bias</strong> - a straight line stays straight no matter how many curved points you
show it. Learning curves (score vs training-set size) make the call visually:
converged-but-low curves = bias; a persistent gap that narrows with data =
variance, keep collecting.</p>
<pre><code class="language-python">from sklearn.model_selection import learning_curve
import numpy as np, matplotlib.pyplot as plt

sizes, tr, val = learning_curve(model, X, y, cv=5,
                                train_sizes=np.linspace(0.1, 1.0, 8))
plt.plot(sizes, tr.mean(1), "o-", label="train")
plt.plot(sizes, val.mean(1), "o-", label="validation")
plt.fill_between(sizes, val.mean(1)-val.std(1), val.mean(1)+val.std(1), alpha=0.2)
plt.xlabel("training examples"); plt.legend()
# curves converged and low  -> bias problem: more data won't help
# wide persistent gap       -> variance problem: more data WILL help
</code></pre>

<h2>The dial, model by model</h2>
<p>You've already been turning this dial all module - here's the unified view:</p>
<table>
  <tr><th>Model</th><th>Toward low bias (flexible)</th><th>Toward low variance (rigid)</th></tr>
  <tr><td><a href="#/ml/knn">KNN</a></td><td>small k</td><td>large k</td></tr>
  <tr><td><a href="#/ml/decision-trees">Tree</a></td><td>deep, small leaves</td><td>shallow, min_samples_leaf ↑</td></tr>
  <tr><td><a href="#/ml/svm">SVM</a></td><td>large C, large γ</td><td>small C, small γ</td></tr>
  <tr><td><a href="#/ml/regularization">Linear + reg.</a></td><td>small λ</td><td>large λ</td></tr>
  <tr><td><a href="#/ml/gradient-boosting">Boosting</a></td><td>more trees, deeper</td><td>fewer trees, shallow, low rate</td></tr>
  <tr><td>Neural nets (Module 7)</td><td>more layers/units</td><td>dropout, early stopping, weight decay</td></tr>
</table>
<p>And the two great variance-reducers that don't increase bias much:
<strong>more data</strong> and <strong>ensembling</strong>
(<a href="#/ml/random-forest">forests</a> are the tradeoff hack: deep trees keep
bias low, averaging kills the variance).</p>

<h2>Common interview questions</h2>
<div class="qa">
  <div class="qa-q">Q1. Define bias and variance in terms of retraining on fresh samples.</div>
  <div class="qa-a"><p>Bias: the systematic gap between the average prediction (across models trained on
  different samples) and the truth. Variance: the spread of those predictions around their
  own average. Total expected squared error = bias² + variance + irreducible noise.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q2. Training accuracy 97%, validation 74%. What is it, and name three fixes.</div>
  <div class="qa-a"><p>High variance (overfitting). Fixes: regularization (L2/dropout), reduce model
  complexity, get more data, ensemble, early stopping, or prune features. (Making the model
  bigger would worsen it - that's the high-bias remedy.)</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q3. Why doesn't more data fix underfitting?</div>
  <div class="qa-a"><p>Bias comes from the model family's rigidity, not sample scarcity: the best straight
  line through a parabola is wrong at infinite data too. You must change the model
  (flexibility, features), not feed it more of what it can't express.</p></div>
</div>

<h2>Quick quiz</h2>
<div class="quiz">
  <div class="quiz-q" data-answer="1">
    <p class="quiz-question">1. Which error component can no model ever remove?</p>
    <button class="quiz-opt">Bias</button>
    <button class="quiz-opt">Irreducible noise σ²</button>
    <button class="quiz-opt">Variance</button>
    <div class="quiz-explain hidden">Noise in the labels themselves (measurement error, inherent randomness) is the floor under every model's error.</div>
  </div>
  <div class="quiz-q" data-answer="0">
    <p class="quiz-question">2. Train MSE = 4.1, validation MSE = 4.3, but both are far above the target. You should…</p>
    <button class="quiz-opt">Increase model capacity or add features - this is bias, not variance</button>
    <button class="quiz-opt">Add regularization</button>
    <button class="quiz-opt">Collect more data</button>
    <div class="quiz-explain hidden">Small gap + poor performance = underfitting. Regularization and more data attack variance, which isn't the problem here.</div>
  </div>
  <div class="quiz-q" data-answer="2">
    <p class="quiz-question">3. Random forests manage the tradeoff by…</p>
    <button class="quiz-opt">Using shallow trees to reduce variance</button>
    <button class="quiz-opt">Adding an L2 penalty to each tree</button>
    <button class="quiz-opt">Keeping deep low-bias trees and averaging away their variance</button>
    <div class="quiz-explain hidden">Each deep tree is low-bias/high-variance; decorrelated averaging cancels the variance while preserving the low bias - the cleverest free lunch in classical ML.</div>
  </div>
</div>
`};

CONTENT["ml/regularization"] = {
  html: String.raw`
<h1>Regularization (L1/L2, Ridge, Lasso, Elastic Net)</h1>
<p class="lead">Overfitting shows up in linear models as huge, frantic coefficients -
the model contorting to thread every training point. Regularization adds one term
to the loss that says: <em>fit the data, but keep the weights calm.</em></p>

<h2>The idea: penalize complexity in the loss itself</h2>
<div class="math-box">
$$J(\mathbf{w}) = \underbrace{\frac{1}{2m}\sum_{i}\big(\hat{y}^{(i)} - y^{(i)}\big)^2}_{\text{fit the data}}
\;+\; \underbrace{\lambda \cdot \text{Penalty}(\mathbf{w})}_{\text{stay simple}}$$
</div>
<p>λ is the exchange rate between the two goals: λ = 0 recovers plain regression;
λ → ∞ crushes all weights to zero (predict the mean). In between lies the sweet
spot - found, as always, by <a href="#/ml/cross-validation">cross-validation</a>.
The two classic penalties differ in one exponent, and that tiny difference changes
everything:</p>
<div class="math-box">
$$\text{Ridge (L2): } \lambda \sum_j w_j^2
\qquad\qquad
\text{Lasso (L1): } \lambda \sum_j |w_j|$$
</div>

<h2>Ridge: shrink everything, keep everything</h2>
<p>The squared penalty punishes large weights ferociously (doubling a weight
quadruples its cost) but barely notices small ones. Result: all coefficients
shrink smoothly toward zero, none reach it. Ridge excels when
<strong>many features each carry a little signal</strong>, and it's the standard cure
for <a href="#/eda/correlation-analysis">multicollinearity</a> - where plain
regression assigns wild offsetting coefficients (+503 and −497) to correlated
twins, ridge calmly gives each ≈ +3.</p>
<h2>Lasso: shrink some to exactly zero</h2>
<p>The absolute-value penalty charges the same marginal rate (λ) no matter how
small a weight gets - so weights whose predictive contribution can't cover that
constant rent get evicted: set to <em>exactly</em> zero. Lasso does
<a href="#/features/feature-selection">feature selection</a> while it trains.
Use it when you suspect <strong>only a few features matter</strong> or need an
interpretable short list.</p>
<div class="diagram">
<svg viewBox="0 0 640 300" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Geometric view: elliptical loss contours meeting an L1 diamond at a corner on the axis, versus an L2 circle at a point off-axis">
  <!-- L1 panel -->
  <text x="160" y="35" text-anchor="middle" class="d-text">Lasso: diamond has corners</text>
  <line x1="40" y1="160" x2="280" y2="160" class="d-grid"/>
  <line x1="160" y1="45" x2="160" y2="285" class="d-grid"/>
  <polygon points="160,100 220,160 160,220 100,160" class="d-box-soft"/>
  <ellipse cx="265" cy="90" rx="75" ry="38" transform="rotate(-20 265 90)" class="d-line" fill="none"/>
  <ellipse cx="265" cy="90" rx="115" ry="62" transform="rotate(-20 265 90)" class="d-line" fill="none"/>
  <circle cx="160" cy="103" r="6" class="d-dot"/>
  <text x="172" y="95" class="d-text-accent">solution ON the axis:</text>
  <text x="172" y="111" class="d-text-accent">w₁ = 0 exactly</text>
  <!-- L2 panel -->
  <text x="480" y="35" text-anchor="middle" class="d-text">Ridge: circle has no corners</text>
  <line x1="360" y1="160" x2="600" y2="160" class="d-grid"/>
  <line x1="480" y1="45" x2="480" y2="285" class="d-grid"/>
  <circle cx="480" cy="160" r="60" class="d-box-soft"/>
  <ellipse cx="585" cy="90" rx="75" ry="38" transform="rotate(-20 585 90)" class="d-line" fill="none"/>
  <ellipse cx="585" cy="90" rx="115" ry="62" transform="rotate(-20 585 90)" class="d-line" fill="none"/>
  <circle cx="521" cy="117" r="6" class="d-dot"/>
  <text x="500" y="100" class="d-text-accent">solution off-axis:</text>
  <text x="500" y="116" class="d-text-accent">small but nonzero</text>
</svg>
<div class="caption">The famous picture: minimizing loss subject to a weight budget.
The loss contours hit the L1 diamond at a corner (an axis → a zeroed weight);
the L2 circle has no corners to catch on.</div>
</div>
<h2>Elastic Net: both at once</h2>
<p>A weighted mix of L1 and L2. The practical reason it exists: with correlated
features, pure Lasso arbitrarily keeps one twin and drops the other (unstable
selections); the L2 component makes it share credit across the group while the
L1 component still prunes. Default choice when you want selection <em>and</em> have
correlated features - i.e., often.</p>

<h2>In code</h2>
<pre><code class="language-python">import numpy as np
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# data where only 5 of 60 features matter, several correlated
rng = np.random.default_rng(0)
X = rng.normal(size=(200, 60))
X[:, 5] = X[:, 0] + rng.normal(0, 0.05, 200)          # a correlated twin
y = 3*X[:, 0] - 2*X[:, 1] + X[:, 2] + 0.5*X[:, 3] - X[:, 4] + rng.normal(0, 1, 200)

for name, reg in [("plain", LinearRegression()),
                  ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 25))),
                  ("lasso", LassoCV(cv=5)),
                  ("enet",  ElasticNetCV(cv=5, l1_ratio=0.5))]:
    model = make_pipeline(StandardScaler(), reg).fit(X, y)   # SCALE FIRST!
    w = model[-1].coef_
    print(f"{name:6s}  max|w|={np.abs(w).max():6.2f}   zeros={np.sum(w == 0):2d}/60")
# plain: large messy weights, 0 zeros - fits noise
# ridge: everything shrunk,   0 zeros
# lasso: ~55 exact zeros - found the sparse truth
</code></pre>
<div class="callout warn">
  <span class="co-title">Two mechanics you must not skip</span>
  <strong>Scale first:</strong> the penalty charges per unit of weight, and weights
  depend on feature units - unscaled, the feature measured in millimeters gets
  unfairly punished versus the one in kilometers.
  <strong>Don't penalize the bias:</strong> b just centers the predictions; libraries
  exclude it automatically. Also note sklearn's aliases: λ is <code>alpha</code> in
  Ridge/Lasso but <code>C = 1/λ</code> in LogisticRegression/SVC - bigger C means
  <em>less</em> regularization.
</div>

<h2>Common interview questions</h2>
<div class="qa">
  <div class="qa-q">Q1. Why does L1 produce exact zeros and L2 doesn't?</div>
  <div class="qa-a"><p>Two views. Gradient: L1's pull is a constant λ regardless of weight size, so weak
  weights get dragged all the way to 0; L2's pull (2λw) fades as w shrinks - asymptotic,
  never arriving. Geometric: the L1 constraint region is a diamond whose corners sit on the
  axes; loss contours typically touch a corner, and corners mean zeroed coordinates.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q2. How do you choose λ?</div>
  <div class="qa-a"><p>Cross-validation over a logarithmic grid (e.g. 10⁻³ … 10³) - sklearn's RidgeCV/
  LassoCV automate it. λ is a bias-variance dial: too small → overfit; too large → underfit;
  the CV curve's minimum is the answer.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q3. When would you pick Elastic Net over Lasso?</div>
  <div class="qa-a"><p>Correlated features (Lasso's picks become arbitrary and unstable), or when the
  number of relevant features may exceed the sample count (Lasso caps selections at m).
  The L2 component stabilizes selection and shares weight across correlated groups.</p></div>
</div>

<h2>Quick quiz</h2>
<div class="quiz">
  <div class="quiz-q" data-answer="1">
    <p class="quiz-question">1. Increasing λ moves the model toward…</p>
    <button class="quiz-opt">Lower bias, higher variance</button>
    <button class="quiz-opt">Higher bias, lower variance</button>
    <button class="quiz-opt">Both lower</button>
    <div class="quiz-explain hidden">A stronger penalty rigidifies the model (calmer weights): less able to chase noise (variance ↓) and less able to fit real structure (bias ↑). λ slides you along the tradeoff.</div>
  </div>
  <div class="quiz-q" data-answer="0">
    <p class="quiz-question">2. You need a model that automatically discards most of 5,000 features. Reach for…</p>
    <button class="quiz-opt">Lasso (L1)</button>
    <button class="quiz-opt">Ridge (L2)</button>
    <button class="quiz-opt">Unregularized regression</button>
    <div class="quiz-explain hidden">Only the L1 penalty zeroes coefficients exactly - training and feature selection in one convex problem.</div>
  </div>
  <div class="quiz-q" data-answer="2">
    <p class="quiz-question">3. In sklearn, LogisticRegression(C=0.01) versus C=100: which is more regularized?</p>
    <button class="quiz-opt">C=100</button>
    <button class="quiz-opt">They're equal</button>
    <button class="quiz-opt">C=0.01 - C is the inverse of λ</button>
    <div class="quiz-explain hidden">C = 1/λ, so small C = strong penalty. A perennial source of tuned-it-backwards bugs.</div>
  </div>
</div>
`};

CONTENT["ml/hyperparameter-tuning"] = {
  html: String.raw`
<h1>Hyperparameter Tuning</h1>
<p class="lead">Parameters are learned from data; hyperparameters - k, depth, λ, C,
learning rate - are chosen by <em>you</em>, before training. Tuning is the art of
making those choices systematically instead of superstitiously.</p>

<h2>The setup: an outer optimization loop</h2>
<p>Training optimizes weights <em>given</em> hyperparameters; tuning optimizes
hyperparameters by repeatedly training and measuring - always with
<a href="#/ml/cross-validation">cross-validation</a>, never the test set.
Three search strategies, in ascending intelligence:</p>

<h2>Grid search: exhaustive and exhausting</h2>
<pre><code class="language-python">from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

pipe = make_pipeline(StandardScaler(), SVC())
grid = GridSearchCV(pipe, {
    "svc__C":     [0.1, 1, 10, 100],
    "svc__gamma": [0.001, 0.01, 0.1, 1],
}, cv=5, n_jobs=-1).fit(X_train, y_train)
print(grid.best_params_, grid.best_score_.round(3))
</code></pre>
<p>Tries every combination: 4 × 4 × 5 folds = 80 fits. Fine for ≤ 2–3
hyperparameters on coarse <em>logarithmic</em> grids (0.001, 0.01, 0.1 - never
1, 2, 3 for scale-type knobs). But it curses exponentially: 6 knobs × 5 values =
15,625 combos, and most of the budget is wasted re-testing values of knobs that
turn out not to matter.</p>

<h2>Random search: smarter than it sounds</h2>
<pre><code class="language-python">from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform, randint

rand = RandomizedSearchCV(pipe, {
    "svc__C":     loguniform(1e-2, 1e3),     # continuous, log-scaled
    "svc__gamma": loguniform(1e-4, 1e1),
}, n_iter=50, cv=5, n_jobs=-1, random_state=0).fit(X_train, y_train)
</code></pre>
<p>Sample random combinations from distributions. The classic Bergstra &amp; Bengio
result explains why it wins: when only some hyperparameters matter (the usual
case), a 25-point grid tests just 5 <em>distinct</em> values of each knob, but 25
random points test 25 distinct values of every knob - far better coverage of the
dimensions that actually matter. Plus: any budget works (n_iter=50), and
continuous ranges beat arbitrary grid points. <strong>Default choice for 3+
hyperparameters.</strong></p>

<h2>Bayesian optimization: search that learns</h2>
<p>Both methods above are amnesiacs - trial #40 ignores everything learned in
trials 1–39. Bayesian optimization builds a probabilistic model of
"score as a function of hyperparameters", and picks each next trial to balance
<em>exploitation</em> (near the best so far) with <em>exploration</em> (high
uncertainty regions). Typically finds better configs in fewer trials - which
matters when one training run costs hours.</p>
<pre><code class="language-python"># pip install optuna
import optuna
from sklearn.model_selection import cross_val_score
import xgboost as xgb

def objective(trial):
    model = xgb.XGBClassifier(
        n_estimators=trial.suggest_int("n_estimators", 100, 1500),
        learning_rate=trial.suggest_float("lr", 1e-3, 0.3, log=True),
        max_depth=trial.suggest_int("max_depth", 3, 9),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
    )
    return cross_val_score(model, X_train, y_train, cv=5, scoring="f1").mean()

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=60, show_progress_bar=True)
print(study.best_params, round(study.best_value, 3))
# bonus: optuna prunes bad trials early and plots param importances
</code></pre>

<h2>Craft knowledge that saves days</h2>
<ul>
  <li><strong>Tune what matters.</strong> Rough priority - boosting: learning_rate,
      n_estimators (via early stopping), max_depth. SVM: C, gamma. Forest: mostly
      max_features. Neural nets: learning rate above all.</li>
  <li><strong>Log scale for multiplicative knobs</strong> (rates, λ, C, γ): the
      interesting difference is 0.001 vs 0.01, not 0.41 vs 0.42.</li>
  <li><strong>Coarse → fine:</strong> wide random sweep first, then zoom into the
      promising region.</li>
  <li><strong>Beware tuning-overfit:</strong> after 500 trials, the best CV score is
      optimistically biased (you selected the luckiest fold-noise). Confirm the
      final config once on the untouched test set - and expect a small drop.</li>
  <li><strong>Diminishing returns are real:</strong> tuning typically buys 1–3 points;
      better features and cleaner data buy more. Don't polish a model built on a
      leaky dataset.</li>
</ul>

<h2>The complete, honest workflow</h2>
<div class="diagram">
<svg viewBox="0 0 660 120" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Pipeline: split data, cross-validated search on training portion, final fit, single test evaluation">
  <defs>
    <marker id="htarr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M0,0 L10,5 L0,10 z" style="fill: var(--text-muted)"/>
    </marker>
  </defs>
  <rect x="15" y="35" width="115" height="50" rx="8" class="d-box-soft"/>
  <text x="72" y="56" text-anchor="middle" class="d-text-sm">split once:</text>
  <text x="72" y="72" text-anchor="middle" class="d-text-sm">train ∣ test 🔒</text>
  <line x1="130" y1="60" x2="158" y2="60" class="d-line" marker-end="url(#htarr)"/>
  <rect x="161" y="35" width="150" height="50" rx="8" class="d-box-soft"/>
  <text x="236" y="56" text-anchor="middle" class="d-text-sm">search (random/BO)</text>
  <text x="236" y="72" text-anchor="middle" class="d-text-sm">scored by CV on train</text>
  <line x1="311" y1="60" x2="339" y2="60" class="d-line" marker-end="url(#htarr)"/>
  <rect x="342" y="35" width="140" height="50" rx="8" class="d-box-soft"/>
  <text x="412" y="56" text-anchor="middle" class="d-text-sm">refit best config</text>
  <text x="412" y="72" text-anchor="middle" class="d-text-sm">on ALL of train</text>
  <line x1="482" y1="60" x2="510" y2="60" class="d-line" marker-end="url(#htarr)"/>
  <rect x="513" y="35" width="130" height="50" rx="8" class="d-box"/>
  <text x="578" y="56" text-anchor="middle" class="d-text-sm">test set: touched</text>
  <text x="578" y="72" text-anchor="middle" class="d-text-sm">once, reported once</text>
</svg>
<div class="caption">Search inside CV; the test set stays locked until the single final measurement.</div>
</div>

<h2>Common interview questions</h2>
<div class="qa">
  <div class="qa-q">Q1. Parameters vs hyperparameters?</div>
  <div class="qa-a"><p>Parameters are learned by the training algorithm from data (weights, split
  thresholds, centroids). Hyperparameters configure the learning itself (k, depth, λ,
  learning rate) and must be set before training - hence the outer search loop.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q2. Why does random search often beat grid search at equal budget?</div>
  <div class="qa-a"><p>Performance usually depends strongly on a few hyperparameters. A grid re-tests the
  same few values of each knob many times; random sampling gives every trial a fresh value
  of every knob, covering the important dimensions much more densely.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q3. After 300 Optuna trials your best CV F1 is 0.912; the test set gives 0.885. Explain.</div>
  <div class="qa-a"><p>Selection bias: with enough trials, the maximum CV score partially reflects lucky
  fold noise, not just true quality - a mild overfit to the validation protocol. The test
  number is the honest one; report it, and consider nested CV when the gap matters.</p></div>
</div>

<h2>Quick quiz</h2>
<div class="quiz">
  <div class="quiz-q" data-answer="2">
    <p class="quiz-question">1. Which of these is a hyperparameter?</p>
    <button class="quiz-opt">A linear model's learned weight w₃</button>
    <button class="quiz-opt">A decision tree's chosen split threshold</button>
    <button class="quiz-opt">A random forest's max_features setting</button>
    <div class="quiz-explain hidden">Weights and split points are learned from data. max_features is set before training and controls how learning behaves.</div>
  </div>
  <div class="quiz-q" data-answer="0">
    <p class="quiz-question">2. Budget: 30 trials, 5 hyperparameters, one training run is expensive. Best strategy?</p>
    <button class="quiz-opt">Bayesian optimization (or random search) over log-scaled ranges</button>
    <button class="quiz-opt">A full 5-dimensional grid</button>
    <button class="quiz-opt">Tune on the test set to save time</button>
    <div class="quiz-explain hidden">A meaningful grid over 5 dims needs thousands of runs. BO extracts the most from 30 expensive trials; the test set is never a tuning instrument.</div>
  </div>
  <div class="quiz-q" data-answer="1">
    <p class="quiz-question">3. Sensible search values for a learning rate are…</p>
    <button class="quiz-opt">0.1, 0.2, 0.3, 0.4, 0.5</button>
    <button class="quiz-opt">0.0001, 0.001, 0.01, 0.1</button>
    <button class="quiz-opt">1, 2, 3, 4, 5</button>
    <div class="quiz-explain hidden">Rates act multiplicatively - behavior changes across orders of magnitude, so search on a log scale.</div>
  </div>
</div>
`};
