/* Module 6 - Machine Learning Algorithms (part 3) */

CONTENT["ml/random-forest"] = {
  html: String.raw`
<h1>Random Forest</h1>
<p class="lead">One decision tree is unstable and overfits. A random forest trains
hundreds of <em>deliberately different</em> trees and lets them vote. The noise
cancels, the signal survives - the wisdom-of-crowds effect, engineered.</p>

<h2>Why averaging works: the variance-cancellation trick</h2>
<p>Recall the tree's disease from the <a href="#/ml/decision-trees">last chapter</a>:
high variance - tiny data changes rebuild the whole tree. But high-variance,
low-bias models are exactly the ones averaging helps. If you average \(T\)
predictors whose errors were <em>independent</em>, the error variance would shrink by
a factor of \(T\). Errors of trees trained on the same data are far from
independent, though - so the forest's entire design is about <strong>decorrelating
the trees</strong>. Two randomness injections do it:</p>
<h3>1 · Bagging (bootstrap aggregating)</h3>
<p>Each tree trains on a <strong>bootstrap sample</strong>: draw \(m\) rows from the
training set <em>with replacement</em>. Each tree therefore sees a different ~63% of
unique rows (the rest are duplicates), so each learns slightly different quirks.</p>
<h3>2 · Feature randomness</h3>
<p>At <em>every split</em>, the tree may only consider a random subset of features
(√n of them, typically, for classification). Without this, every tree would open
with the same dominant feature and the forest would be one tree in a trench coat.
Forcing variety makes trees explore different structure - and lets weaker-but-real
signals get used.</p>

<div class="diagram">
<svg viewBox="0 0 660 280" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Training data bootstrapped into three samples, each growing a different tree, predictions combined by majority vote">
  <defs>
    <marker id="rfarr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M0,0 L10,5 L0,10 z" style="fill: var(--text-muted)"/>
    </marker>
  </defs>
  <rect x="20" y="105" width="120" height="50" rx="8" class="d-box"/>
  <text x="80" y="126" text-anchor="middle" class="d-text">training data</text>
  <text x="80" y="143" text-anchor="middle" class="d-text-sm">m rows</text>
  <line x1="140" y1="115" x2="205" y2="60" class="d-line" marker-end="url(#rfarr)"/>
  <line x1="140" y1="130" x2="205" y2="130" class="d-line" marker-end="url(#rfarr)"/>
  <line x1="140" y1="145" x2="205" y2="200" class="d-line" marker-end="url(#rfarr)"/>
  <text x="145" y="80" class="d-text-sm">bootstrap</text>
  <rect x="210" y="35" width="150" height="50" rx="8" class="d-box-soft"/>
  <text x="285" y="56" text-anchor="middle" class="d-text">sample 1 → tree 1</text>
  <text x="285" y="73" text-anchor="middle" class="d-text-sm">random √n features/split</text>
  <rect x="210" y="105" width="150" height="50" rx="8" class="d-box-soft"/>
  <text x="285" y="126" text-anchor="middle" class="d-text">sample 2 → tree 2</text>
  <text x="285" y="143" text-anchor="middle" class="d-text-sm">different rows, features</text>
  <rect x="210" y="175" width="150" height="50" rx="8" class="d-box-soft"/>
  <text x="285" y="196" text-anchor="middle" class="d-text">sample T → tree T</text>
  <text x="285" y="213" text-anchor="middle" class="d-text-sm">… hundreds of them</text>
  <line x1="360" y1="60" x2="430" y2="118" class="d-line" marker-end="url(#rfarr)"/>
  <line x1="360" y1="130" x2="430" y2="130" class="d-line" marker-end="url(#rfarr)"/>
  <line x1="360" y1="200" x2="430" y2="142" class="d-line" marker-end="url(#rfarr)"/>
  <rect x="435" y="105" width="120" height="50" rx="8" class="d-box"/>
  <text x="495" y="126" text-anchor="middle" class="d-text">majority vote</text>
  <text x="495" y="143" text-anchor="middle" class="d-text-sm">(mean, for regression)</text>
  <line x1="555" y1="130" x2="610" y2="130" class="d-line" marker-end="url(#rfarr)"/>
  <text x="622" y="135" class="d-text-accent">ŷ</text>
  <text x="330" y="262" text-anchor="middle" class="d-text-sm">individually overfit trees → collectively stable prediction</text>
</svg>
<div class="caption">Bagging + per-split feature sampling = decorrelated trees whose
errors cancel in the vote.</div>
</div>

<h2>The free lunch: out-of-bag evaluation</h2>
<p>Each tree never saw ~37% of the rows (the ones its bootstrap missed). Those rows
are a ready-made validation set <em>for that tree</em>. Aggregate over the forest and
you get the <strong>OOB score</strong> - an honest performance estimate with no
held-out split and no extra compute. It typically tracks cross-validation closely.</p>

<h2>In code</h2>
<pre><code class="language-python">from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import pandas as pd

data = load_breast_cancer()
Xtr, Xte, ytr, yte = train_test_split(data.data, data.target,
                                      stratify=data.target, random_state=0)

rf = RandomForestClassifier(
    n_estimators=400,        # more trees never hurts accuracy, only time
    max_features="sqrt",     # the decorrelation knob
    min_samples_leaf=2,      # mild per-tree smoothing
    oob_score=True,          # free validation
    n_jobs=-1, random_state=0,
).fit(Xtr, ytr)

print("OOB estimate :", round(rf.oob_score_, 3))
print("test accuracy:", round(rf.score(Xte, yte), 3))   # they should agree

# feature importance (prefer permutation importance for serious use)
imp = pd.Series(rf.feature_importances_, index=data.feature_names)
print(imp.sort_values(ascending=False).head(5).round(3))
</code></pre>
<p>Tuning notes: <code>n_estimators</code> - more is monotonically better (returns
diminish; 200–500 is typical). The knob that actually matters is
<code>max_features</code>: lower = more decorrelation = less variance but more bias.
Unlike single trees, forests tolerate full-depth trees because averaging absorbs
the overfitting; <code>min_samples_leaf</code> of 1–5 is common.</p>

<h2>Strengths, limits, and when boosting beats it</h2>
<table>
  <tr><th>Strengths</th><th>Limits</th></tr>
  <tr><td>Excellent accuracy out of the box, little tuning</td><td>Slower/heavier at prediction time than a single model (hundreds of trees)</td></tr>
  <tr><td>Robust to outliers, junk features, unscaled data</td><td>Loses the single tree's readable-flowchart interpretability</td></tr>
  <tr><td>OOB score + feature importances built in</td><td>Can't extrapolate beyond the training range (all trees predict constants per region)</td></tr>
  <tr><td>Hard to badly overfit; parallelizes perfectly</td><td>Usually edged out by tuned <a href="#/ml/gradient-boosting">gradient boosting</a> on tabular benchmarks</td></tr>
</table>
<p>Rule of thumb: random forest is the <em>reliable default</em> - the model you fit
first to see what's achievable. Boosting is the <em>tuned finisher</em>.</p>

<h2>Common interview questions</h2>
<div class="qa">
  <div class="qa-q">Q1. Why does a random forest overfit less than its individual trees?</div>
  <div class="qa-a"><p>Each tree overfits differently (different bootstrap rows, different feature
  subsets), so their errors are partially independent and average out in the vote,
  slashing variance while keeping the trees' low bias. The model's variance falls roughly
  with the number of <em>decorrelated</em> trees.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q2. What would happen if you bagged trees but removed per-split feature sampling?</div>
  <div class="qa-a"><p>You'd have plain bagging: trees stay highly correlated because the same dominant
  features win the top splits in every tree - and correlated errors don't cancel. Feature
  randomness is what makes the "random" forest work.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q3. Explain the out-of-bag score.</div>
  <div class="qa-a"><p>For each row, average the predictions of only the trees whose bootstrap sample
  excluded it (~37% of trees), then score those predictions. It's cross-validation-quality
  evaluation obtained for free during training.</p></div>
</div>

<h2>Quick quiz</h2>
<div class="quiz">
  <div class="quiz-q" data-answer="1">
    <p class="quiz-question">1. The two sources of randomness in a random forest are…</p>
    <button class="quiz-opt">Random labels and random learning rates</button>
    <button class="quiz-opt">Bootstrap row sampling and per-split feature subsets</button>
    <button class="quiz-opt">Random tree depths and random seeds</button>
    <div class="quiz-explain hidden">Rows (bagging) and columns (max_features). Both exist to decorrelate the trees so their errors cancel.</div>
  </div>
  <div class="quiz-q" data-answer="2">
    <p class="quiz-question">2. Increasing n_estimators from 300 to 3000 will most likely…</p>
    <button class="quiz-opt">Cause overfitting</button>
    <button class="quiz-opt">Dramatically improve accuracy</button>
    <button class="quiz-opt">Barely change accuracy, but cost 10× the compute</button>
    <div class="quiz-explain hidden">More trees only stabilize the average - performance plateaus. Forests don't overfit with more trees; they just get slower.</div>
  </div>
  <div class="quiz-q" data-answer="0">
    <p class="quiz-question">3. A house-price forest trained on homes up to ₹2 crore is asked about a ₹10-crore mansion. It will predict…</p>
    <button class="quiz-opt">Something near the top of its training range - trees can't extrapolate</button>
    <button class="quiz-opt">An accurately extrapolated high price</button>
    <button class="quiz-opt">An error</button>
    <div class="quiz-explain hidden">Every leaf predicts a constant learned from training rows; inputs beyond the seen range fall into edge leaves. Linear models extrapolate (dangerously); trees refuse to (also dangerously).</div>
  </div>
</div>
`};

CONTENT["ml/gradient-boosting"] = {
  html: String.raw`
<h1>Gradient Boosting (XGBoost, LightGBM, CatBoost)</h1>
<p class="lead">Random forests average independent trees. Boosting does the opposite:
trains trees <em>sequentially</em>, each one focused purely on the mistakes left by
the ones before. The result has dominated tabular-data competitions for a decade.</p>

<h2>The idea: a committee of error-correctors</h2>
<p>Start with a terrible model - say, "predict the mean price for every house."
Compute its errors (residuals). Now train a small tree <em>to predict those
errors</em>. Add its output to the model. Compute the new, smaller residuals.
Train another tree on those. Repeat hundreds of times:</p>
<div class="math-box">
$$F_0(\mathbf{x}) = \bar{y}
\qquad
F_t(\mathbf{x}) = F_{t-1}(\mathbf{x}) + \eta \cdot h_t(\mathbf{x})$$
</div>
<p>where \(h_t\) is a small tree fit to the current residuals and
\(\eta\) (the <strong>learning rate</strong>, ~0.01–0.3) shrinks each tree's
contribution. Worked micro-example, one house with true price 100:</p>
<ul>
  <li>\(F_0\) predicts 70 → residual +30.</li>
  <li>Tree 1 learns ≈ +30 for such houses; with η = 0.3 we add 9 → \(F_1\) = 79 → residual +21.</li>
  <li>Tree 2 corrects again → \(F_2\) ≈ 85.3 → residual +14.7 … each step eats ~30% of the remaining error.</li>
</ul>
<p>Small deliberate steps, hundreds of correctors - boosting reduces <em>bias</em>
gradually (vs. forests, which reduce <em>variance</em> in one parallel shot).</p>

<h2>Why "gradient" boosting</h2>
<p>The residual trick generalizes beautifully. For squared error, the negative
gradient of the loss with respect to the current prediction <em>is</em> the residual:</p>
<div class="math-box">
$$-\frac{\partial}{\partial F(\mathbf{x}^{(i)})} \tfrac{1}{2}\big(y^{(i)} - F(\mathbf{x}^{(i)})\big)^2 = y^{(i)} - F(\mathbf{x}^{(i)})$$
</div>
<p>So "fit a tree to the residuals" is really "fit a tree to the negative gradient
of the loss" - <strong>gradient descent, but in function space</strong>: each tree is one
descent step. Swap in any differentiable loss (log-loss for classification, Huber
for robustness, quantile for prediction intervals) and the same machine works.
That's the deep link to <a href="#/math/calculus">Module 1's gradient descent</a> -
same idea, the "parameter" being updated is the prediction function itself.</p>

<h2>The three modern implementations</h2>
<table>
  <tr><th></th><th>XGBoost</th><th>LightGBM</th><th>CatBoost</th></tr>
  <tr><td>Claim to fame</td><td>the original Kaggle king; regularized objective, battle-tested</td><td>fastest on large data (histogram splits, leaf-wise growth)</td><td>best native categorical handling; great defaults</td></tr>
  <tr><td>Tree growth</td><td>level-wise (balanced)</td><td>leaf-wise (deeper, more expressive, easier to overfit)</td><td>symmetric (oblivious) trees</td></tr>
  <tr><td>Categoricals</td><td>encode yourself</td><td>native (integer-coded)</td><td>native, with target-statistics done safely</td></tr>
  <tr><td>Pick it when…</td><td>you want the ecosystem/default</td><td>data is big and speed matters</td><td>many categorical features, minimal tuning time</td></tr>
</table>
<p>All three: handle missing values natively, provide feature importance and early
stopping, and need <em>no feature scaling</em> (they're trees).</p>

<h2>In code, with the two knobs that matter</h2>
<pre><code class="language-python"># pip install xgboost
import xgboost as xgb
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

X, y = fetch_california_housing(return_X_y=True)
Xtr, Xval, ytr, yval = train_test_split(X, y, random_state=0)

model = xgb.XGBRegressor(
    n_estimators=2000,        # set high; early stopping picks the real number
    learning_rate=0.05,       # smaller = better + slower (needs more trees)
    max_depth=5,              # boosted trees stay SHALLOW (3-8)
    subsample=0.8,            # row sampling per tree  ) stochastic
    colsample_bytree=0.8,     # column sampling        ) regularization
    early_stopping_rounds=50, # stop when val error stalls
    eval_metric="mae",
)
model.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)

print("best iteration:", model.best_iteration)
print("val MAE:", round(mean_absolute_error(yval, model.predict(Xval)), 3))
</code></pre>
<p>The golden pairing: <strong>learning_rate × n_estimators</strong>. Halve the rate,
double the trees, usually gain a little accuracy. Always use early stopping on a
validation set - it converts "how many trees?" from a hyperparameter into a
measurement. Note the contrast with forests: boosted trees are <em>shallow</em>
(each corrects a little; depth 5 ≈ enough for 5-way interactions), and boosting
<em>can</em> overfit with too many trees - sequential error-chasing eventually chases
noise, which is exactly what the validation curve guards against.</p>

<h2>Forest vs boosting: the mental model</h2>
<table>
  <tr><th></th><th>Random Forest</th><th>Gradient Boosting</th></tr>
  <tr><td>Trees are…</td><td>independent, deep, parallel</td><td>sequential, shallow, cooperative</td></tr>
  <tr><td>Attacks…</td><td>variance (by averaging)</td><td>bias (by correcting)</td></tr>
  <tr><td>Overfits when…</td><td>rarely, gracefully</td><td>too many trees / rate too high - needs early stopping</td></tr>
  <tr><td>Tuning effort</td><td>minimal</td><td>moderate, rewarded</td></tr>
  <tr><td>Typical result</td><td>very good</td><td>slightly better, tuned</td></tr>
</table>

<h2>Common interview questions</h2>
<div class="qa">
  <div class="qa-q">Q1. Bagging vs boosting in one breath.</div>
  <div class="qa-a"><p>Bagging trains many independent models on bootstrap samples in parallel and
  averages them (variance reduction). Boosting trains models sequentially, each fitting the
  previous ensemble's errors, and sums them with a shrinkage factor (bias reduction).</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q2. Why is it called *gradient* boosting?</div>
  <div class="qa-a"><p>Each new tree is fit to the negative gradient of the loss with respect to current
  predictions - a gradient-descent step taken in function space. Residuals are just the
  special case of squared-error loss.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q3. Your boosted model's validation error decreases, then creeps up after tree 400. What do you do?</div>
  <div class="qa-a"><p>That's the overfitting turn: later trees are fitting noise. Use early stopping
  (keep ~tree 400), and/or lower the learning rate, increase subsample/colsample
  randomness, or reduce max_depth.</p></div>
</div>

<h2>Quick quiz</h2>
<div class="quiz">
  <div class="quiz-q" data-answer="2">
    <p class="quiz-question">1. In boosting, each new tree is trained on…</p>
    <button class="quiz-opt">A fresh bootstrap sample, independently</button>
    <button class="quiz-opt">The same targets as every other tree</button>
    <button class="quiz-opt">The errors (negative loss gradients) of the ensemble so far</button>
    <div class="quiz-explain hidden">Sequential error-correction is the defining mechanic - each tree sees only what remains unexplained.</div>
  </div>
  <div class="quiz-q" data-answer="0">
    <p class="quiz-question">2. You halve learning_rate from 0.1 to 0.05. To compensate you should roughly…</p>
    <button class="quiz-opt">Double n_estimators (and rely on early stopping)</button>
    <button class="quiz-opt">Halve max_depth</button>
    <button class="quiz-opt">Double the training data</button>
    <div class="quiz-explain hidden">Smaller steps need more of them. The rate×trees product is the primary budget; smaller rates usually land at slightly better minima.</div>
  </div>
  <div class="quiz-q" data-answer="1">
    <p class="quiz-question">3. Unlike random forests, gradient boosting can overfit as trees are added because…</p>
    <button class="quiz-opt">It uses deeper trees</button>
    <button class="quiz-opt">Later trees sequentially chase residual noise once real signal is exhausted</button>
    <button class="quiz-opt">It ignores the loss function</button>
    <div class="quiz-explain hidden">Averaging independent trees only stabilizes; summing error-correctors keeps "improving" on training noise. Early stopping is the standard cure.</div>
  </div>
</div>
`};

CONTENT["ml/svm"] = {
  html: String.raw`
<h1>Support Vector Machines</h1>
<p class="lead">Many lines can separate two classes. SVM picks the one with the most
breathing room - the maximum margin - and, via the kernel trick, does it even
when the classes aren't linearly separable at all.</p>

<h2>The maximum-margin idea</h2>
<p>A separating line that skims past training points is fragile: a slightly shifted
test point crosses it. SVM instead maximizes the <strong>margin</strong> - the distance
from the boundary to the nearest points of either class. Those nearest points are
the <strong>support vectors</strong>: they alone determine the boundary; every other
point could be deleted without changing anything.</p>
<div class="diagram">
<svg viewBox="0 0 640 300" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Two point classes separated by a line with dashed margin lines; circled points on the margins are the support vectors">
  <rect x="20" y="20" width="600" height="250" rx="8" class="d-box-muted"/>
  <!-- boundary -->
  <line x1="230" y1="270" x2="430" y2="20" class="d-line-accent"/>
  <!-- margins -->
  <line x1="160" y1="270" x2="360" y2="20" class="d-line" stroke-dasharray="6 4"/>
  <line x1="300" y1="270" x2="500" y2="20" class="d-line" stroke-dasharray="6 4"/>
  <!-- class A -->
  <circle cx="105" cy="105" r="6" class="d-dot"/>
  <circle cx="140" cy="170" r="6" class="d-dot"/>
  <circle cx="80" cy="210" r="6" class="d-dot"/>
  <circle cx="185" cy="115" r="6" class="d-dot"/>
  <circle cx="230" cy="112" r="6" class="d-dot"/>
  <circle cx="212" cy="205" r="6" class="d-dot"/>
  <!-- class B -->
  <circle cx="520" cy="100" r="6" class="d-dot-muted"/>
  <circle cx="560" cy="170" r="6" class="d-dot-muted"/>
  <circle cx="495" cy="215" r="6" class="d-dot-muted"/>
  <circle cx="445" cy="120" r="6" class="d-dot-muted"/>
  <circle cx="425" cy="205" r="6" class="d-dot-muted"/>
  <!-- support vectors circled -->
  <circle cx="230" cy="112" r="11" fill="none" class="d-line-accent"/>
  <circle cx="212" cy="205" r="11" fill="none" class="d-line-accent"/>
  <circle cx="445" cy="120" r="11" fill="none" class="d-line-accent"/>
  <circle cx="425" cy="205" r="11" fill="none" class="d-line-accent"/>
  <text x="330" y="255" class="d-text-sm">margin</text>
  <line x1="322" y1="248" x2="288" y2="222" class="d-line" marker-end="none"/>
  <text x="40" y="45" class="d-text-accent">circled = support vectors (the only points that matter)</text>
</svg>
<div class="caption">The widest street between the classes. The boundary is defined
entirely by the few points touching the curbs.</div>
</div>
<p>Formally, with labels \(y \in \{-1, +1\}\), the margin width is
\(2/\lVert\mathbf{w}\rVert\), so SVM solves:</p>
<div class="math-box">
$$\min_{\mathbf{w}, b} \; \tfrac{1}{2}\lVert \mathbf{w} \rVert^2
\quad \text{subject to} \quad
y^{(i)}(\mathbf{w}^\top \mathbf{x}^{(i)} + b) \ge 1 \;\; \forall i$$
</div>
<p>Minimizing \(\lVert\mathbf{w}\rVert\) = maximizing the street width, while every
point stays on its correct side of the curb.</p>

<h2>Real data overlaps: the soft margin and C</h2>
<p>Perfect separation is rare and chasing it means overfitting to noise. The
<em>soft-margin</em> SVM allows violations, charging a penalty per intruding point,
with a knob <strong>C</strong> setting the exchange rate:</p>
<ul>
  <li><strong>Large C:</strong> violations are expensive → narrower margin, contorts to
      classify every training point → low bias, high variance.</li>
  <li><strong>Small C:</strong> violations are cheap → wide, calm margin that tolerates
      stragglers → high bias, low variance.</li>
</ul>
<p>C is <em>the</em> regularization dial of SVMs - tune it by cross-validation, always.</p>

<h2>The kernel trick: curves without leaving flatland</h2>
<p>Data that's inseparable in its original space often becomes separable after
adding derived dimensions (e.g. two concentric rings separate cleanly once you add
\(z = x_1^2 + x_2^2\) - the inner ring sits low, the outer high, and a flat plane
splits them). Computing such feature maps explicitly can be astronomically
expensive. The trick: the SVM's math only ever uses <strong>dot products</strong>
between points, and a <strong>kernel function</strong> computes the dot product
<em>as if</em> in the expanded space without ever going there:</p>
<div class="math-box">
$$K(\mathbf{a}, \mathbf{b}) = \phi(\mathbf{a})^\top \phi(\mathbf{b})
\qquad\text{computed directly from } \mathbf{a}, \mathbf{b}$$
$$\text{RBF kernel:}\;\; K(\mathbf{a}, \mathbf{b}) = \exp\!\big(-\gamma \lVert \mathbf{a} - \mathbf{b} \rVert^2\big)
\;\;\text{(an infinite-dimensional } \phi\text{!)}$$
</div>
<p>The RBF (Gaussian) kernel is the everyday choice: a similarity that decays with
distance, whose reach is set by <strong>γ</strong>. Small γ → far-reaching influence →
smooth boundaries; large γ → each point only influences its close vicinity →
wiggly boundaries that can overfit. C and γ together define the model's
flexibility; grid-search them jointly.</p>

<h2>In code</h2>
<pre><code class="language-python">from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=400, noise=0.25, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=0)

# scaling is MANDATORY: SVM lives on distances/dot products
pipe = make_pipeline(StandardScaler(), SVC(kernel="rbf"))

grid = GridSearchCV(pipe, {
    "svc__C":     [0.1, 1, 10, 100],
    "svc__gamma": [0.01, 0.1, 1, 10],
}, cv=5).fit(Xtr, ytr)

print("best:", grid.best_params_)
print("test accuracy:", round(grid.score(Xte, yte), 3))
print("support vectors:", grid.best_estimator_[-1].n_support_)
# only these points define the boundary - often a small fraction of the data
</code></pre>

<h2>Where SVMs shine - and where they've faded</h2>
<table>
  <tr><th>Strengths</th><th>Limits</th></tr>
  <tr><td>Excellent in high dimensions, even n_features &gt; m (text!)</td><td>Training scales ~O(m²)–O(m³): painful past ~50–100k rows</td></tr>
  <tr><td>Maximum margin = principled built-in generalization</td><td>Two interacting hyperparameters (C, γ) that must be tuned</td></tr>
  <tr><td>Kernels: non-linear power with convex training (one global optimum)</td><td>No native probabilities (calibration is a bolt-on); memory grows with support vectors</td></tr>
  <tr><td>Memory-efficient predictions (only SVs kept)</td><td>On big tabular data, boosting usually wins; on huge perceptual data, deep nets</td></tr>
</table>
<p>Modern niche: small-to-medium datasets, high-dimensional features, when you want
a strong convex-optimization baseline. Historically, SVMs ruled the 1995–2012 era -
and the margin/kernel concepts still appear everywhere (hinge loss, similarity
functions, max-margin ideas in modern theory).</p>

<h2>Common interview questions</h2>
<div class="qa">
  <div class="qa-q">Q1. What is a support vector?</div>
  <div class="qa-a"><p>A training point lying on or inside the margin (or misclassified) - the only points
  with nonzero weight in the solution. The boundary is a function of them alone; the rest of
  the data could vanish without changing the model.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q2. Explain the kernel trick to a non-mathematician.</div>
  <div class="qa-a"><p>To separate tangled classes you'd like extra derived features (like "distance from
  center"), but building them explicitly can be infinitely expensive. A kernel is a shortcut
  formula that gives the geometry (all the dot products) of that richer space directly, so
  the SVM behaves as if the features existed without ever computing them.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q3. Your RBF-SVM has 100% train accuracy, 71% test. Which knobs, which direction?</div>
  <div class="qa-a"><p>Classic overfit: decrease γ (widen each point's influence → smoother boundary)
  and/or decrease C (allow margin violations). Then re-grid-search - the two interact.</p></div>
</div>

<h2>Quick quiz</h2>
<div class="quiz">
  <div class="quiz-q" data-answer="1">
    <p class="quiz-question">1. Deleting a well-classified point far from the margin does what to the SVM boundary?</p>
    <button class="quiz-opt">Shifts it toward the deleted point</button>
    <button class="quiz-opt">Nothing - only support vectors define the boundary</button>
    <button class="quiz-opt">Makes the margin wider</button>
    <div class="quiz-explain hidden">Non-support-vectors have zero coefficient in the solution. This sparsity is a defining property of SVMs.</div>
  </div>
  <div class="quiz-q" data-answer="0">
    <p class="quiz-question">2. Very large C means…</p>
    <button class="quiz-opt">Misclassifying training points is heavily punished → risk of overfitting</button>
    <button class="quiz-opt">A very wide margin that ignores outliers</button>
    <button class="quiz-opt">The kernel becomes linear</button>
    <div class="quiz-explain hidden">C prices margin violations. Expensive violations force the boundary to contort around every training point - low bias, high variance.</div>
  </div>
  <div class="quiz-q" data-answer="2">
    <p class="quiz-question">3. Concentric-ring data becomes linearly separable when you add the feature x₁² + x₂². The kernel trick's role is…</p>
    <button class="quiz-opt">To add that column to your DataFrame</button>
    <button class="quiz-opt">To remove the need for a boundary entirely</button>
    <button class="quiz-opt">To get the effect of such expansions without computing them explicitly</button>
    <div class="quiz-explain hidden">Kernels supply dot products of the expanded representation directly - even for expansions with infinitely many dimensions (RBF).</div>
  </div>
</div>
`};

CONTENT["ml/naive-bayes"] = {
  html: String.raw`
<h1>Naive Bayes</h1>
<p class="lead">Take Bayes' theorem from Module 1, add one heroic simplification -
"assume every feature is independent" - and you get a classifier that trains in
one pass, needs almost no data, and filtered the world's spam for a decade.</p>

<h2>From Bayes' theorem to a classifier</h2>
<p>We want the most probable class given the features. Bayes' theorem
(<a href="#/math/probability">derived here</a>) turns that into quantities we can
count from training data:</p>
<div class="math-box">
$$P(c \mid \mathbf{x}) \;\propto\; P(c)\,\cdot\, P(\mathbf{x} \mid c)$$
</div>
<p>\(P(c)\) - the prior - is just class frequency (20% of email is spam). The
problem is the likelihood \(P(\mathbf{x} \mid c)\): the probability of this exact
<em>combination</em> of thousands of features requires astronomical data to estimate.
Enter the naive assumption - features are conditionally independent given the
class - which factorizes the monster into countable pieces:</p>
<div class="math-box">
$$P(\mathbf{x} \mid c) \approx \prod_{j=1}^{n} P(x_j \mid c)
\qquad\Longrightarrow\qquad
\hat{y} = \arg\max_c \; P(c) \prod_j P(x_j \mid c)$$
</div>
<p>Each factor is a one-word count: "how often does 'free' appear in spam?" -
easy to estimate even from small data. The assumption is false ("free" and "offer"
obviously co-occur), so the estimated probabilities are distorted… but the
<em>argmax over classes</em> often survives the distortion. Bad probability
estimates, decent decisions - the key to why the method works at all.</p>

<h2>Worked spam example (with the email table from Module 1)</h2>
<p>Priors: \(P(\text{spam}) = 0.2\), \(P(\text{ham}) = 0.8\). Word likelihoods
learned from counts: \(P(\text{"free"}\mid\text{spam}) = 0.6\),
\(P(\text{"free"}\mid\text{ham}) = 0.05\); \(P(\text{"meeting"}\mid\text{spam}) = 0.05\),
\(P(\text{"meeting"}\mid\text{ham}) = 0.4\). New email: "free meeting".</p>
<div class="math-box">
$$\text{spam: } 0.2 \times 0.6 \times 0.05 = 0.006
\qquad
\text{ham: } 0.8 \times 0.05 \times 0.4 = 0.016$$
</div>
<p>Ham wins, 0.016 vs 0.006 (posterior ≈ 73% ham after normalizing). Two practical
wrinkles the real implementations handle:</p>
<ul>
  <li><strong>Zero counts:</strong> a word never seen in spam gives
      \(P(x_j\mid\text{spam}) = 0\), annihilating the whole product. Fix:
      <strong>Laplace smoothing</strong> - add a pseudo-count \(\alpha\) (usually 1) to
      every count: \(P(x_j \mid c) = \frac{n_{jc} + \alpha}{n_c + \alpha\, |V|}\).</li>
  <li><strong>Underflow:</strong> multiplying 10,000 small probabilities hits zero in
      floating point. Fix: sum <strong>log</strong>-probabilities instead - same argmax,
      stable arithmetic.</li>
</ul>

<h2>Three variants for three data types</h2>
<table>
  <tr><th>Variant</th><th>Feature type</th><th>Typical use</th></tr>
  <tr><td><strong>Multinomial</strong></td><td>counts</td><td>text: word/token counts or TF-IDF - the classic</td></tr>
  <tr><td><strong>Bernoulli</strong></td><td>binary present/absent</td><td>short text, feature flags</td></tr>
  <tr><td><strong>Gaussian</strong></td><td>continuous</td><td>numeric features, modeled as a normal per class</td></tr>
</table>

<h2>In code: a spam filter in ten lines</h2>
<pre><code class="language-python">from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

texts = ["free money offer now", "meeting at noon tomorrow",
         "free free lottery winner", "project meeting notes attached",
         "claim your free prize", "lunch meeting rescheduled"]
labels = [1, 0, 1, 0, 1, 0]                      # 1 = spam

clf = make_pipeline(CountVectorizer(),           # text -> word-count matrix
                    MultinomialNB(alpha=1.0))    # alpha = Laplace smoothing
clf.fit(texts, labels)

tests = ["free meeting today", "winner claim prize now"]
print(clf.predict(tests))                        # [0 1]
print(clf.predict_proba(tests).round(3))
# treat these probabilities as rankings, not calibrated truths
</code></pre>

<h2>Strengths &amp; weaknesses</h2>
<table>
  <tr><th>Strengths</th><th>Weaknesses</th></tr>
  <tr><td>Trains in a single pass - fastest learner in this module</td><td>The independence lie distorts probabilities (over-confident)</td></tr>
  <tr><td>Works with tiny training sets; handles 100k+ features happily</td><td>Correlated features get double-counted ("free" + "offer" ≈ one signal counted twice)</td></tr>
  <tr><td>Naturally incremental (update counts as data streams in)</td><td>Gaussian variant's normality assumption often poor</td></tr>
  <tr><td>Great baseline for any text classification task</td><td>Rarely the accuracy winner; linear SVM / logistic regression usually edge it on text</td></tr>
</table>
<p>Its modern role: the <strong>first model you fit on any text problem</strong> -
trains in seconds, sets an honest baseline, and occasionally refuses to be beaten
by anything cheap.</p>

<h2>Common interview questions</h2>
<div class="qa">
  <div class="qa-q">Q1. What exactly is "naive" about Naive Bayes?</div>
  <div class="qa-a"><p>The assumption that features are conditionally independent given the class, so the
  joint likelihood factorizes into a product of per-feature probabilities. It's false in
  practice, but it makes estimation tractable and the resulting <em>classifications</em>
  (not the probabilities) are often surprisingly good.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q2. Why is Laplace smoothing necessary?</div>
  <div class="qa-a"><p>Any feature value unseen in a class during training would get probability zero,
  and one zero factor vetoes the entire product regardless of all other evidence. Adding
  pseudo-counts keeps every probability positive and shrinks estimates sensibly for rare
  events.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q3. Naive Bayes says 99.9% spam; a calibrated model says 85%. Why the gap?</div>
  <div class="qa-a"><p>Correlated words are treated as independent witnesses, so evidence gets counted
  multiple times and posteriors get pushed toward extremes. The ranking is fine;
  the confidence is inflated. Calibrate (Platt/isotonic) if you need real probabilities.</p></div>
</div>

<h2>Quick quiz</h2>
<div class="quiz">
  <div class="quiz-q" data-answer="1">
    <p class="quiz-question">1. The naive independence assumption buys…</p>
    <button class="quiz-opt">Better probability estimates</button>
    <button class="quiz-opt">Tractable estimation - per-feature counts instead of joint combinations</button>
    <button class="quiz-opt">A non-linear decision boundary</button>
    <div class="quiz-explain hidden">Estimating P(x|c) jointly needs data for every feature combination; factorized, each feature only needs its own counts. That's the entire trade.</div>
  </div>
  <div class="quiz-q" data-answer="0">
    <p class="quiz-question">2. The word "quantum" never appeared in training spam. Without smoothing, an email containing it…</p>
    <button class="quiz-opt">Can never be classified spam - the zero multiplies everything away</button>
    <button class="quiz-opt">Is automatically spam</button>
    <button class="quiz-opt">Is handled fine</button>
    <div class="quiz-explain hidden">One zero factor makes P(spam|x) = 0 regardless of "free lottery winner" filling the rest of the email. Laplace smoothing exists precisely for this.</div>
  </div>
  <div class="quiz-q" data-answer="2">
    <p class="quiz-question">3. Implementations sum log-probabilities rather than multiplying probabilities because…</p>
    <button class="quiz-opt">Logs make the model more accurate</button>
    <button class="quiz-opt">Multiplication is slow</button>
    <button class="quiz-opt">Products of thousands of small numbers underflow to zero; log-sums are stable and preserve the argmax</button>
    <div class="quiz-explain hidden">log is monotonic, so argmax is unchanged - and 10,000 additions of moderate negatives beats a product that floating point rounds to 0.</div>
  </div>
</div>
`};
