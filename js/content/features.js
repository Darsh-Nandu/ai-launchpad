/* Module 5 - Feature Engineering (part 1) */

CONTENT["features/encoding"] = {
  html: String.raw`
<h1>Encoding Categorical Variables</h1>
<p class="lead">Models compute with numbers, but data arrives as words: city names,
product types, education levels. Encoding is the translation - and a bad
translation invents mathematical claims your data never made.</p>

<h2>The core danger: fake order</h2>
<p>The naive fix - <code>red=0, green=1, blue=2</code> - silently tells the model that
blue &gt; green &gt; red, that blue is "twice" green, and that the average of red and
blue is green. For colors, that's nonsense the model will dutifully learn.
Every encoding choice below is about when numeric structure is <em>true</em>
versus <em>invented</em>.</p>
<p>First, know your two categorical types:</p>
<ul>
  <li><strong>Nominal</strong> - no order: city, color, product category.</li>
  <li><strong>Ordinal</strong> - genuine order: education (HS &lt; BSc &lt; MSc &lt; PhD),
      satisfaction (low/medium/high), t-shirt sizes.</li>
</ul>

<h2>One-hot encoding - the safe default for nominal</h2>
<p>One new binary column per category; exactly one is "hot" per row.
No fake order, no fake distances:</p>
<pre><code class="language-python">import pandas as pd

df = pd.DataFrame({"city": ["Delhi", "Mumbai", "Delhi", "Chennai"],
                   "rent": [30, 55, 32, 25]})

pd.get_dummies(df, columns=["city"])          # quick pandas way
#    rent  city_Chennai  city_Delhi  city_Mumbai
#      30         False        True        False ...

# production way - remembers categories, handles unseen ones:
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
X = enc.fit_transform(df[["city"]])
</code></pre>
<p>Two practical notes. For linear models, drop one column
(<code>drop="first"</code>) - the columns otherwise sum to 1, a perfect collinearity
called the <em>dummy variable trap</em>. And prefer the sklearn encoder in
pipelines: <code>get_dummies</code> on train and test separately can produce
<em>different columns</em> if a category is absent from one split - a classic
silent bug.</p>
<p>Weakness: <strong>cardinality</strong>. One-hot on a 10,000-value column (user IDs,
ZIP codes) explodes into 10,000 sparse columns. That's when you reach for target
encoding below.</p>

<h2>Ordinal encoding - when order is real</h2>
<pre><code class="language-python">from sklearn.preprocessing import OrdinalEncoder

edu_order = [["high school", "bachelor", "master", "phd"]]
enc = OrdinalEncoder(categories=edu_order)
df2 = pd.DataFrame({"edu": ["master", "high school", "phd", "bachelor"]})
print(enc.fit_transform(df2[["edu"]]).ravel())    # [2. 0. 3. 1.]
</code></pre>
<p>Always pass the explicit order - the default is alphabetical, which would rank
"bachelor" &lt; "high school". Caveat: the encoding also asserts <em>equal
spacing</em> (master − bachelor = phd − master), which may or may not be true;
tree models don't care (they only use order), linear models do.</p>

<h2>Target encoding - for high-cardinality columns</h2>
<p>Replace each category with the <em>mean of the target</em> for that category:
city → average churn rate of customers in that city. One column, any cardinality,
and the encoding is directly informative.</p>
<div class="callout warn">
  <span class="co-title">This is the leakage minefield</span>
  A naive target encoding computed on all data injects the answer into the features:
  a category seen once encodes its own row's target exactly. Symptoms: brilliant
  validation scores, production collapse. The fixes are mandatory, not optional -
  out-of-fold encoding (each row's value computed from folds that exclude it) and
  smoothing toward the global mean for rare categories.
</div>
<pre><code class="language-python"># the honest way: out-of-fold target encoding with smoothing
import numpy as np
from sklearn.model_selection import KFold

def target_encode(train, col, target, n_splits=5, smooth=10):
    global_mean = train[target].mean()
    encoded = pd.Series(np.nan, index=train.index)
    for tr_idx, val_idx in KFold(n_splits, shuffle=True,
                                 random_state=0).split(train):
        fold = train.iloc[tr_idx]
        stats = fold.groupby(col)[target].agg(["mean", "count"])
        # shrink rare categories toward the global mean
        smoothed = ((stats["mean"] * stats["count"] + global_mean * smooth)
                    / (stats["count"] + smooth))
        encoded.iloc[val_idx] = train[col].iloc[val_idx].map(smoothed)
    return encoded.fillna(global_mean)

# (or use category_encoders.TargetEncoder / sklearn's TargetEncoder ≥1.3,
#  which implement exactly this)
</code></pre>

<h2>Choosing an encoding</h2>
<table>
  <tr><th>Situation</th><th>Encoding</th><th>Watch out for</th></tr>
  <tr><td>Nominal, few categories (&lt; ~15)</td><td>One-hot</td><td>dummy trap for linear models</td></tr>
  <tr><td>Ordinal</td><td>Ordinal with explicit order</td><td>equal-spacing assumption</td></tr>
  <tr><td>Nominal, huge cardinality</td><td>Target encoding</td><td>leakage - out-of-fold only</td></tr>
  <tr><td>Any categorical, tree/boosting model</td><td>Ordinal often works fine; LightGBM/CatBoost handle categories natively</td><td>arbitrary order is OK for trees, meaningless for linear</td></tr>
  <tr><td>Text-like, open vocabulary</td><td>Hashing trick or learned embeddings (Module 7+)</td><td>hash collisions; needs data volume</td></tr>
  <tr><td>Rare categories</td><td>Group into "other" first</td><td>categories seen once are pure noise</td></tr>
</table>

<h2>Common interview questions</h2>
<div class="qa">
  <div class="qa-q">Q1. Why is label-encoding city names dangerous for linear regression but tolerable for random forests?</div>
  <div class="qa-a"><p>Linear models use the numeric value arithmetically - a coefficient multiplies
  "Delhi = 2", asserting fake magnitude and order. Trees only ask ordering questions
  (city ≤ 1.5?), so an arbitrary order just becomes arbitrary - but findable - split
  points; wasteful but not misleading.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q2. What's the dummy variable trap?</div>
  <div class="qa-a"><p>Full one-hot columns always sum to 1, duplicating the intercept - perfect
  multicollinearity that makes linear coefficients unidentifiable. Drop one category
  (the "reference level"); its effect is absorbed into the intercept.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q3. How does target encoding leak, exactly?</div>
  <div class="qa-a"><p>If a row's own target contributes to its category's mean, the feature partially
  <em>contains the label</em> - worst for rare categories, where the "mean" is nearly the
  row's own answer. Out-of-fold computation breaks the loop; smoothing handles the rare
  categories.</p></div>
</div>

<h2>Quick quiz</h2>
<div class="quiz">
  <div class="quiz-q" data-answer="1">
    <p class="quiz-question">1. Encoding satisfaction = {low, medium, high} for a linear model. Best choice?</p>
    <button class="quiz-opt">One-hot - always safest</button>
    <button class="quiz-opt">Ordinal 0/1/2 - the order is real and worth exposing</button>
    <button class="quiz-opt">Target encoding</button>
    <div class="quiz-explain hidden">Genuine order is information; ordinal encoding hands it to the model directly. One-hot would work but forces the model to rediscover the ordering from data.</div>
  </div>
  <div class="quiz-q" data-answer="2">
    <p class="quiz-question">2. A 5,000-value merchant_id column for a churn model. Best approach?</p>
    <button class="quiz-opt">One-hot encoding</button>
    <button class="quiz-opt">Alphabetical ordinal encoding</button>
    <button class="quiz-opt">Out-of-fold target encoding with smoothing (or native categorical support)</button>
    <div class="quiz-explain hidden">One-hot creates 5,000 columns; alphabetical order is meaningless. Target encoding compresses to one informative column - done out-of-fold to avoid leaking labels.</div>
  </div>
  <div class="quiz-q" data-answer="0">
    <p class="quiz-question">3. Your one-hot test set is missing a column the training set had. Root cause?</p>
    <button class="quiz-opt">get_dummies was run separately per split; a category was absent from test</button>
    <button class="quiz-opt">The model has too many parameters</button>
    <button class="quiz-opt">The CSV was corrupted</button>
    <div class="quiz-explain hidden">Encoders must be FIT on train and APPLIED to test (OneHotEncoder with handle_unknown="ignore") so the column space stays identical.</div>
  </div>
</div>
`};

CONTENT["features/scaling"] = {
  html: String.raw`
<h1>Scaling &amp; Normalization</h1>
<p class="lead">Age spans 0–100; income spans 0–10,000,000. Feed both to a
distance-based or gradient-based model and income drowns out age - not because
it matters more, but because its <em>numbers are bigger</em>. Scaling removes that
accident of units.</p>

<h2>Who needs scaling, and why</h2>
<table>
  <tr><th>Model family</th><th>Needs scaling?</th><th>Reason</th></tr>
  <tr><td>KNN, K-Means, SVM</td><td><strong>Yes - critical</strong></td><td>distances: the biggest-unit feature dominates the metric</td></tr>
  <tr><td>Linear/Logistic + regularization</td><td><strong>Yes</strong></td><td>L1/L2 penalize coefficients; unscaled features get unfairly penalized</td></tr>
  <tr><td>Neural networks</td><td><strong>Yes</strong></td><td>gradient descent converges far faster on comparable scales</td></tr>
  <tr><td>PCA</td><td><strong>Yes</strong></td><td>maximizes variance - raw variance is unit-dependent</td></tr>
  <tr><td>Trees, Random Forest, Boosting</td><td><strong>No</strong></td><td>splits use rank order only; monotonic rescaling changes nothing</td></tr>
</table>
<p>Intuition for the distance case: customers A and B differ by 40 years and ₹1,000
income. Squared distance = 40² + 1000² = 1,600 + 1,000,000 - age contributes 0.2%.
Rescale both to [0, 1] and age gets its rightful vote.</p>

<h2>The three standard scalers</h2>
<h3>Standardization (z-score) - the default</h3>
<div class="math-box">
$$x' = \frac{x - \mu}{\sigma} \qquad \text{result: mean 0, std 1}$$
</div>
<p>Doesn't bound the range, handles new extreme values gracefully, plays perfectly
with models that assume roughly centered data. When unsure, use this.</p>
<h3>Min-max normalization - when you need a fixed range</h3>
<div class="math-box">
$$x' = \frac{x - x_{min}}{x_{max} - x_{min}} \qquad \text{result: } [0, 1]$$
</div>
<p>Natural for pixel intensities and anything feeding into bounded activations.
Fragile to outliers: one extreme point compresses everyone else into a sliver
(a single billionaire squeezes all normal incomes into [0, 0.001]).</p>
<h3>Robust scaling - when outliers are a fact of life</h3>
<div class="math-box">
$$x' = \frac{x - \text{median}}{\mathrm{IQR}}$$
</div>
<p>Median and IQR ignore extremes (<a href="#/math/descriptive-statistics">Module 1</a>),
so genuine outliers don't distort the scaling of the sane majority.</p>
<pre><code class="language-python">from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import numpy as np

X = np.array([[25, 30_000], [35, 45_000], [45, 60_000],
              [30, 38_000], [52, 2_000_000]])     # note the outlier income

print(StandardScaler().fit_transform(X).round(2))
print(MinMaxScaler().fit_transform(X).round(3))   # outlier crushes the rest
print(RobustScaler().fit_transform(X).round(2))   # majority keeps its spread
</code></pre>
<p>Related but different: log transforms (<code>np.log1p</code>) fix <em>skew</em>
(shape), scalers fix <em>units</em> (magnitude). Skewed feature? Log first,
then scale. And "normalization" sometimes means scaling each <em>row</em> to unit
length (<code>Normalizer</code>) - used for text vectors, not tabular columns;
don't confuse the two.</p>

<h2>The rule that separates professionals from tutorials</h2>
<div class="callout warn">
  <span class="co-title">Fit on train. Transform train and test. Never refit.</span>
  The scaler's μ and σ are <em>learned parameters</em>. Computing them on all data
  before splitting leaks test-set statistics into training - your model gets a
  subtle preview of data it's supposed to have never seen. Scale inside a Pipeline
  and the rule enforces itself.
</div>
<pre><code class="language-python">from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score

X_train, X_test, y_train, y_test = train_test_split(X_all, y, random_state=0)

pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
pipe.fit(X_train, y_train)          # scaler fits ONLY on X_train
pipe.score(X_test, y_test)          # test data scaled with train's μ, σ

# bonus: cross_val_score on a pipeline re-fits the scaler per fold - correctly
print(cross_val_score(pipe, X_train, y_train, cv=5).mean())
</code></pre>

<h2>Common interview questions</h2>
<div class="qa">
  <div class="qa-q">Q1. Why doesn't a random forest need scaled features?</div>
  <div class="qa-a"><p>A split like "income ≤ 47,000" depends only on the ordering of values, and any
  monotonic scaling preserves order - the tree finds the equivalent threshold and produces
  identical predictions. Distance- and gradient-based models have no such invariance.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q2. Standardization vs min-max - how do you choose?</div>
  <div class="qa-a"><p>Standardize by default (robust-ish, unbounded, suits centered-data assumptions).
  Min-max when a bounded range is genuinely required (pixels, some neural inputs) and
  outliers are controlled. Robust scaler when heavy outliers would corrupt μ/σ.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q3. What goes wrong if you scale before the train/test split?</div>
  <div class="qa-a"><p>Test rows influence the μ/σ (or min/max) applied to training data - information
  leakage. Scores get optimistically biased; the effect is small for scaling but the habit
  generalizes to encoders and imputers where leaks are large. Pipelines make it a non-issue.</p></div>
</div>

<h2>Quick quiz</h2>
<div class="quiz">
  <div class="quiz-q" data-answer="2">
    <p class="quiz-question">1. KNN on unscaled data with age (0–100) and salary (20k–200k) will effectively…</p>
    <button class="quiz-opt">Weight both features equally</button>
    <button class="quiz-opt">Ignore salary</button>
    <button class="quiz-opt">Ignore age - salary's magnitude dominates every distance</button>
    <div class="quiz-explain hidden">Distances are sums of squared differences; salary differences are ~1000× larger, contributing ~1,000,000× more after squaring.</div>
  </div>
  <div class="quiz-q" data-answer="1">
    <p class="quiz-question">2. A feature has severe outliers you've decided to keep. Which scaler?</p>
    <button class="quiz-opt">MinMaxScaler</button>
    <button class="quiz-opt">RobustScaler</button>
    <button class="quiz-opt">No scaler exists for this case</button>
    <div class="quiz-explain hidden">Median/IQR are outlier-immune, so the bulk of the data keeps a sensible spread. Min-max is the worst choice - the outlier defines the range.</div>
  </div>
  <div class="quiz-q" data-answer="0">
    <p class="quiz-question">3. Where should StandardScaler learn its μ and σ?</p>
    <button class="quiz-opt">From the training split only, then applied unchanged to test</button>
    <button class="quiz-opt">From the whole dataset for stability</button>
    <button class="quiz-opt">Separately on train and on test</button>
    <div class="quiz-explain hidden">Fitting on everything leaks test statistics; refitting on test scales it differently than training (also wrong). One fit, on train - pipelines enforce it.</div>
  </div>
</div>
`};

CONTENT["features/imbalanced-data"] = {
  html: String.raw`
<h1>Handling Imbalanced Data</h1>
<p class="lead">Fraud: 0.2% of transactions. Machine failures: 1 in 10,000.
On such data a model that predicts "all normal" scores 99.8% accuracy while
detecting nothing. Imbalance isn't an edge case - it's the natural state of
most valuable problems.</p>

<h2>Step zero: fix your metric, not your data</h2>
<p>The first casualty of imbalance is accuracy itself. With a 99/1 split, the
do-nothing classifier gets 99% - accuracy has become a broken instrument.
Before touching the data, switch to metrics that see the minority class:</p>
<ul>
  <li><strong>Precision</strong> - of the flagged, how many were real?</li>
  <li><strong>Recall</strong> - of the real ones, how many did we catch?</li>
  <li><strong>F1</strong> - their harmonic mean; <strong>PR-AUC</strong> - the
      threshold-free summary that (unlike ROC-AUC) stays honest under heavy
      imbalance.</li>
</ul>
<p>Full treatment in <a href="#/ml/evaluation-metrics">Model Evaluation Metrics</a>;
here it's enough that <em>every</em> later decision is judged by these, never by
accuracy. And always establish the trivial baselines first - "flag nothing" and
"flag everything" - your model must beat both meaningfully.</p>

<h2>Lever 1: class weights (try this first)</h2>
<p>Most algorithms accept a weighting that makes minority mistakes cost more -
no data manipulation, one argument:</p>
<pre><code class="language-python">from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# "balanced" sets weight inversely proportional to class frequency
clf = LogisticRegression(class_weight="balanced", max_iter=1000)
rf  = RandomForestClassifier(class_weight="balanced")

# neural nets: the same idea via pos_weight in the loss
# torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([99.0]))
</code></pre>
<p>Effectively the model now minimizes a re-weighted loss where one fraud error
costs as much as 99 normal-class errors. Simple, leak-proof, and often all you need.</p>

<h2>Lever 2: resampling</h2>
<h3>Random undersampling - throw away majority rows</h3>
<p>Fast, and surprisingly effective when the majority class is huge and redundant.
Cost: discarded information. Good first move when you have millions of majority
examples anyway.</p>
<h3>Random oversampling - duplicate minority rows</h3>
<p>Keeps all data, but exact duplicates invite overfitting: the model memorizes
the few minority points it keeps re-seeing.</p>
<h3>SMOTE - synthesize new minority points</h3>
<p>Instead of copying, SMOTE interpolates: pick a minority point, pick one of its
minority neighbors, and create a new sample along the line between them.</p>
<div class="diagram">
<svg viewBox="0 0 640 240" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="SMOTE: synthetic minority points created along lines between existing minority neighbors">
  <rect x="20" y="20" width="600" height="190" rx="8" class="d-box-muted"/>
  <!-- majority cloud -->
  <circle cx="120" cy="70" r="5" class="d-dot-muted"/><circle cx="160" cy="95" r="5" class="d-dot-muted"/>
  <circle cx="95" cy="120" r="5" class="d-dot-muted"/><circle cx="185" cy="60" r="5" class="d-dot-muted"/>
  <circle cx="140" cy="150" r="5" class="d-dot-muted"/><circle cx="210" cy="120" r="5" class="d-dot-muted"/>
  <circle cx="80" cy="80" r="5" class="d-dot-muted"/><circle cx="230" cy="85" r="5" class="d-dot-muted"/>
  <text x="150" y="190" text-anchor="middle" class="d-text-sm">majority class (many)</text>
  <!-- minority points + synthetic -->
  <circle cx="430" cy="80" r="6" class="d-dot"/>
  <circle cx="530" cy="120" r="6" class="d-dot"/>
  <circle cx="470" cy="170" r="6" class="d-dot"/>
  <line x1="430" y1="80" x2="530" y2="120" class="d-line-accent" stroke-dasharray="5 4"/>
  <line x1="430" y1="80" x2="470" y2="170" class="d-line-accent" stroke-dasharray="5 4"/>
  <line x1="530" y1="120" x2="470" y2="170" class="d-line-accent" stroke-dasharray="5 4"/>
  <circle cx="480" cy="100" r="5" class="d-dot" opacity="0.45"/>
  <circle cx="450" cy="125" r="5" class="d-dot" opacity="0.45"/>
  <circle cx="500" cy="145" r="5" class="d-dot" opacity="0.45"/>
  <text x="480" y="205" text-anchor="middle" class="d-text-sm">minority: real (solid) + SMOTE synthetics (faded) on connecting lines</text>
</svg>
<div class="caption">SMOTE fills in the minority region with plausible neighbors instead
of exact copies.</div>
</div>
<pre><code class="language-python"># pip install imbalanced-learn
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import cross_val_score

pipe = ImbPipeline([
    ("smote", SMOTE(random_state=0)),
    ("clf", LogisticRegression(max_iter=1000)),
])
# inside CV, SMOTE runs on each training fold only - exactly right
print(cross_val_score(pipe, X, y, cv=5, scoring="f1").mean())
</code></pre>
<div class="callout warn">
  <span class="co-title">The two resampling sins</span>
  1) <strong>Resampling before the split:</strong> synthetic copies of a minority point
  land in both train and test - the model is tested on near-duplicates of its
  training data. Resample <em>inside</em> the pipeline, training folds only.
  2) <strong>Resampling the test set at all:</strong> the test set must keep the real-world
  class ratio, or every metric you report describes a fantasy world.
</div>

<h2>Lever 3: move the threshold</h2>
<p>Classifiers output probabilities; 0.5 is just a default cutoff. Often the
cheapest "fix" for imbalance is choosing the threshold that matches the actual
costs of false alarms vs misses:</p>
<pre><code class="language-python">import numpy as np
from sklearn.metrics import precision_recall_curve

proba = clf.fit(X_train, y_train).predict_proba(X_test)[:, 1]
prec, rec, thr = precision_recall_curve(y_test, proba)

f1 = 2 * prec * rec / (prec + rec + 1e-12)
best = np.argmax(f1)
print(f"best threshold ≈ {thr[best]:.2f}  (precision {prec[best]:.2f}, recall {rec[best]:.2f})")

y_pred = (proba &gt;= thr[best]).astype(int)     # instead of the default 0.5
</code></pre>
<p>In fraud detection you might run at threshold 0.15 - many false alarms, few
missed frauds - because a missed fraud costs 100× a wasted review. The threshold
is a <em>business decision</em> wearing a mathematical costume.</p>

<h2>A sane playbook</h2>
<ol>
  <li>Switch metrics (F1 / PR-AUC) + establish trivial baselines.</li>
  <li>Try <code>class_weight="balanced"</code> - one line, often sufficient.</li>
  <li>Tune the decision threshold to the real cost ratio.</li>
  <li>If minority is truly tiny (&lt; a few hundred rows), add SMOTE or
      undersampling - inside the pipeline.</li>
  <li>Strong general choice under imbalance: gradient boosting
      (<code>scale_pos_weight</code> in XGBoost) - trees + weighting cope well.</li>
  <li>Extreme rarity (&lt; ~0.1%)? Consider reframing as
      <a href="#/advanced/autoencoders">anomaly detection</a>.</li>
</ol>

<h2>Common interview questions</h2>
<div class="qa">
  <div class="qa-q">Q1. Why is accuracy misleading at 99:1 imbalance, and what do you use instead?</div>
  <div class="qa-a"><p>The majority-class-always classifier scores 99% while having zero skill on the class
  that matters. Use precision/recall/F1 on the minority class, PR-AUC across thresholds,
  and compare against the trivial baselines explicitly.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q2. How does SMOTE differ from random oversampling, and what are its limits?</div>
  <div class="qa-a"><p>It interpolates new points between minority neighbors rather than duplicating,
  reducing memorization. Limits: it can synthesize nonsense in overlap regions or across
  sub-clusters, struggles with categorical features (use SMOTENC), and adds little when the
  minority class is well-represented already.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q3. Your colleague applied SMOTE to the whole dataset, then split. Validation F1 is 0.93. Trustworthy?</div>
  <div class="qa-a"><p>No. Synthetic points derived from a training row can appear in validation - the model
  is graded on near-copies of what it studied. Re-run with resampling inside CV folds;
  expect a substantially lower, honest score.</p></div>
</div>

<h2>Quick quiz</h2>
<div class="quiz">
  <div class="quiz-q" data-answer="1">
    <p class="quiz-question">1. Disease at 0.5% prevalence; your model predicts "healthy" for everyone. Its accuracy?</p>
    <button class="quiz-opt">0.5%</button>
    <button class="quiz-opt">99.5% - while detecting zero patients</button>
    <button class="quiz-opt">50%</button>
    <div class="quiz-explain hidden">Accuracy rewards agreeing with the base rate. Recall on the sick class is 0 - the number that actually matters here.</div>
  </div>
  <div class="quiz-q" data-answer="0">
    <p class="quiz-question">2. Which is a valid place for SMOTE?</p>
    <button class="quiz-opt">Inside the pipeline, applied to each training fold only</button>
    <button class="quiz-opt">On the full dataset before splitting</button>
    <button class="quiz-opt">On the test set, to balance the evaluation</button>
    <div class="quiz-explain hidden">Resampling belongs to training only. Test data must keep the real class ratio, and no synthetic point may straddle the train/test boundary.</div>
  </div>
  <div class="quiz-q" data-answer="2">
    <p class="quiz-question">3. Missing a fraud costs ~100× a false alarm. You should…</p>
    <button class="quiz-opt">Keep the 0.5 threshold - it's standard</button>
    <button class="quiz-opt">Raise the threshold to 0.9 for precision</button>
    <button class="quiz-opt">Lower the threshold well below 0.5, trading false alarms for recall</button>
    <div class="quiz-explain hidden">Asymmetric costs move the optimal operating point. A low threshold catches more fraud at the price of reviews - exactly the trade the cost ratio demands.</div>
  </div>
</div>
`};
