/* Module 3 - Data Visualization Lab (part 2) */

CONTENT["dataviz/chart-picker"] = {
  html: String.raw`
<h1>Chart-Picker Guide</h1>
<p class="lead">Chart choice isn't taste - it follows from two questions:
<em>what do I want to show?</em> and <em>what kinds of variables do I have?</em>
This page is the decision table you'll come back to for years.</p>

<h2>Start from the question, not the chart</h2>
<table>
  <tr><th>You want to show…</th><th>Variables</th><th>First choice</th><th>Alternatives / notes</th></tr>
  <tr><td>Distribution of one variable</td><td>1 numeric</td><td><strong>Histogram</strong></td><td>KDE/density (smooth), ECDF (no binning bias), box plot (compact)</td></tr>
  <tr><td>Distribution across groups</td><td>1 numeric × 1 categorical</td><td><strong>Box plot</strong></td><td>Violin (shows shape/bimodality), strip/swarm for small n</td></tr>
  <tr><td>Relationship between two numerics</td><td>2 numeric</td><td><strong>Scatter</strong></td><td>+ trend line (regplot); density heatmap when points overplot</td></tr>
  <tr><td>Relationship, many variables</td><td>3+ numeric</td><td><strong>Pair plot</strong></td><td>Correlation heatmap for the compact summary</td></tr>
  <tr><td>Quantity per category</td><td>1 categorical × 1 numeric</td><td><strong>Bar chart</strong></td><td>Sort bars by value; horizontal bars for long labels; y starts at 0</td></tr>
  <tr><td>Trend over time / order</td><td>ordered x × numeric</td><td><strong>Line chart</strong></td><td>Multiple lines for groups; area chart only for stacked totals</td></tr>
  <tr><td>Parts of a whole</td><td>proportions</td><td><strong>Stacked bar</strong></td><td>Pie only with ≤ 4 slices and big differences; angles read poorly</td></tr>
  <tr><td>Matrix of values / intensities</td><td>2 categorical × numeric</td><td><strong>Heatmap</strong></td><td>Confusion matrices, correlations, activity-by-hour-and-day</td></tr>
  <tr><td>Comparison of two conditions per item</td><td>paired values</td><td><strong>Slope / dumbbell chart</strong></td><td>Beats side-by-side bars for before/after stories</td></tr>
  <tr><td>Flow between stages</td><td>sequential counts</td><td><strong>Funnel / Sankey</strong></td><td>Conversion pipelines, user journeys</td></tr>
</table>

<h2>The reasoning behind the table</h2>
<p>Human vision decodes some encodings far more accurately than others. The ranking,
established experimentally by Cleveland &amp; McGill in 1984, is roughly:</p>
<div class="math-box" style="font-family: var(--font-ui); font-size: 15px;">
position on a common scale &nbsp;≻&nbsp; length &nbsp;≻&nbsp; angle / slope
&nbsp;≻&nbsp; area &nbsp;≻&nbsp; color intensity &nbsp;≻&nbsp; volume
</div>
<p>Every rule of thumb falls out of this: scatter and line charts (position) beat
everything for precision; bar charts (length) are the reliable workhorse - but only
if the baseline is zero, since truncation destroys the length encoding; pie charts
(angle) rank low, which is why they only work when differences are huge; and 3-D
charts (volume) sit at the bottom, which is why they're almost always a mistake.</p>
<p><strong>Color deserves special care:</strong> use <em>sequential</em> palettes for
ordered values (light→dark), <em>diverging</em> palettes for values around a midpoint
(correlations: blue–white–red), and <em>qualitative</em> palettes only for unordered
categories. Never encode ordered data with rainbow categorical colors - and check
your palette is colorblind-safe (about 8% of men are red-green colorblind;
seaborn's <code>"colorblind"</code> palette exists for this).</p>

<h2>Common pairings in one snippet</h2>
<pre><code class="language-python">import seaborn as sns
import matplotlib.pyplot as plt

df = sns.load_dataset("penguins").dropna()

fig, axes = plt.subplots(2, 2, figsize=(11, 8))

# distribution → histogram
sns.histplot(df, x="body_mass_g", kde=True, ax=axes[0, 0])
axes[0, 0].set_title("distribution: histogram")

# distribution per group → box/violin
sns.violinplot(df, x="species", y="body_mass_g", ax=axes[0, 1])
axes[0, 1].set_title("groups: violin")

# relationship → scatter (+hue)
sns.scatterplot(df, x="flipper_length_mm", y="body_mass_g",
                hue="species", ax=axes[1, 0])
axes[1, 0].set_title("relationship: scatter")

# category quantity → sorted horizontal bar
counts = df["species"].value_counts().sort_values()
axes[1, 1].barh(counts.index, counts.values)
axes[1, 1].set_title("counts: sorted bars")

plt.tight_layout(); plt.show()
</code></pre>

<h2>Anti-patterns to un-learn</h2>
<ul>
  <li><strong>Pie chart with 12 slices</strong> → sorted horizontal bar chart, every time.</li>
  <li><strong>Line chart for unordered categories</strong> - a line implies continuity;
      connecting "Delhi–Mumbai–Chennai" draws a meaningless trend.</li>
  <li><strong>Dual y-axes</strong> - by rescaling either axis you can force any two curves
      to "correlate". Use two stacked panels.</li>
  <li><strong>Averages without spread</strong> - a bar of means hides everything Anscombe
      taught. Add error bars, or show the raw points when n is small.</li>
  <li><strong>Alphabetical bar order</strong> - sort by value; the ranking is usually
      the message.</li>
</ul>

<h2>Quick quiz</h2>
<div class="quiz">
  <div class="quiz-q" data-answer="1">
    <p class="quiz-question">1. Comparing revenue across 15 product categories with long names. Best chart?</p>
    <button class="quiz-opt">Pie chart</button>
    <button class="quiz-opt">Sorted horizontal bar chart</button>
    <button class="quiz-opt">Line chart</button>
    <div class="quiz-explain hidden">Length on a common baseline is precisely decodable, horizontal orientation fits long labels, and sorting turns the chart into a ranking. 15 pie slices are unreadable; a line implies order that doesn't exist.</div>
  </div>
  <div class="quiz-q" data-answer="2">
    <p class="quiz-question">2. Which encoding do humans decode most accurately?</p>
    <button class="quiz-opt">Color intensity</button>
    <button class="quiz-opt">Angle (pie slices)</button>
    <button class="quiz-opt">Position on a common scale</button>
    <div class="quiz-explain hidden">Cleveland &amp; McGill's ranking: position ≻ length ≻ angle ≻ area ≻ color. It's why scatter/line/bar charts dominate serious data work.</div>
  </div>
  <div class="quiz-q" data-answer="0">
    <p class="quiz-question">3. A correlation heatmap should use which palette type?</p>
    <button class="quiz-opt">Diverging (e.g. blue–white–red centered at 0)</button>
    <button class="quiz-opt">Qualitative (distinct hues)</button>
    <button class="quiz-opt">Rainbow</button>
    <div class="quiz-explain hidden">Correlations live on [−1, +1] around a meaningful zero: a diverging palette makes sign visible (color) and magnitude visible (intensity), with white marking "no correlation".</div>
  </div>
</div>
`};

CONTENT["dataviz/ml-visualizations"] = {
  html: String.raw`
<h1>Visualizations for ML</h1>
<p class="lead">Once you start training models, plots stop being decoration and
become <em>diagnostic instruments</em>. These five - confusion matrix, ROC curve,
loss curves, decision boundaries, feature importance - tell you not just how good
a model is, but what <em>kind</em> of wrong it is.</p>

<h2>Confusion matrix: what kind of mistakes?</h2>
<p>Accuracy compresses a classifier into one number; the confusion matrix keeps the
full story - a table of <em>actual class × predicted class</em>:</p>
<pre><code class="language-python">from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=0)
clf = LogisticRegression(max_iter=5000).fit(Xtr, ytr)

ConfusionMatrixDisplay.from_estimator(clf, Xte, yte,
                                      display_labels=["malignant", "benign"],
                                      cmap="Blues")
</code></pre>
<p>Read the <em>off-diagonal</em> cells: they are the two different failure modes.
A cancer screen that misses tumors (false negatives) and one that scares healthy
patients (false positives) can have identical accuracy and wildly different
consequences. For multi-class problems, the matrix also reveals <em>which pairs</em>
of classes get confused (3 vs 8 in digit recognition) - pointing exactly at what
to fix. Full metric definitions live in
<a href="#/ml/evaluation-metrics">Model Evaluation Metrics</a>.</p>

<h2>ROC curve: every threshold at once</h2>
<p>Classifiers output scores; <em>you</em> choose the cutoff. The ROC curve plots
true-positive rate vs false-positive rate for <strong>every possible threshold</strong>,
so you see the whole menu of trade-offs before committing:</p>
<div class="diagram">
<svg viewBox="0 0 640 320" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="ROC plot with diagonal chance line, a good model curve bowing to the top left, and a perfect corner point">
  <line x1="80" y1="270" x2="590" y2="270" class="d-line"/>
  <line x1="80" y1="270" x2="80" y2="40" class="d-line"/>
  <text x="300" y="300" class="d-text-sm">False Positive Rate →</text>
  <text x="60" y="160" class="d-text-sm" transform="rotate(-90 60 160)">True Positive Rate →</text>
  <!-- chance diagonal -->
  <line x1="80" y1="270" x2="560" y2="60" class="d-line" stroke-dasharray="6 4"/>
  <text x="380" y="185" class="d-text-sm">random guessing (AUC = 0.5)</text>
  <!-- good model -->
  <path d="M 80 270 C 110 110, 200 70, 560 60" class="d-curve"/>
  <text x="170" y="80" class="d-text-accent">good model (AUC ≈ 0.93)</text>
  <!-- perfect corner -->
  <circle cx="80" cy="60" r="5" class="d-dot"/>
  <text x="92" y="55" class="d-text-sm">perfect (AUC = 1.0)</text>
  <!-- threshold dot -->
  <circle cx="150" cy="97" r="5" class="d-dot"/>
  <text x="160" y="112" class="d-text-sm">one specific threshold</text>
</svg>
<div class="caption">Each point on the curve is one threshold. AUC - the area under
the curve - summarizes ranking quality across all of them.</div>
</div>
<pre><code class="language-python">from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
RocCurveDisplay.from_estimator(clf, Xte, yte)
# for heavily imbalanced data, prefer the precision-recall curve:
PrecisionRecallDisplay.from_estimator(clf, Xte, yte)
</code></pre>

<h2>Loss curves: the ECG of training</h2>
<p>Plot training and validation loss per epoch, and the curve <em>shape</em> diagnoses
the problem:</p>
<div class="diagram">
<svg viewBox="0 0 660 260" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Three loss curve patterns: healthy convergence, overfitting where validation loss rises, and underfitting where both stay high">
  <!-- healthy -->
  <rect x="15" y="25" width="195" height="185" rx="6" class="d-box-muted"/>
  <text x="112" y="48" text-anchor="middle" class="d-text">healthy</text>
  <path d="M 30 70 C 70 190, 130 195, 195 197" class="d-curve"/>
  <path d="M 30 60 C 75 175, 135 185, 195 188" class="d-line" fill="none" stroke-dasharray="5 3"/>
  <text x="30" y="230" class="d-text-sm">both fall, small gap → keep training</text>
  <!-- overfitting -->
  <rect x="235" y="25" width="195" height="185" rx="6" class="d-box-muted"/>
  <text x="332" y="48" text-anchor="middle" class="d-text">overfitting</text>
  <path d="M 250 70 C 290 190, 350 197, 415 199" class="d-curve"/>
  <path d="M 250 60 C 290 150, 320 140, 415 85" class="d-line" fill="none" stroke-dasharray="5 3"/>
  <circle cx="322" cy="139" r="4" class="d-dot"/>
  <text x="250" y="230" class="d-text-sm">val loss turns up → stop/regularize</text>
  <!-- underfitting -->
  <rect x="455" y="25" width="195" height="185" rx="6" class="d-box-muted"/>
  <text x="552" y="48" text-anchor="middle" class="d-text">underfitting</text>
  <path d="M 470 90 C 520 115, 570 120, 635 122" class="d-curve"/>
  <path d="M 470 80 C 520 108, 570 112, 635 114" class="d-line" fill="none" stroke-dasharray="5 3"/>
  <text x="470" y="230" class="d-text-sm">both plateau high → bigger model</text>
  <text x="20" y="252" class="d-text-sm">solid = training loss · dashed = validation loss</text>
</svg>
<div class="caption">Three shapes, three prescriptions. The gap between the curves is
the generalization gap - the visual form of the bias-variance tradeoff.</div>
</div>
<p>The point where validation loss turns upward while training loss keeps falling
is overfitting beginning - the basis of <em>early stopping</em>
(<a href="#/dl/dl-regularization">Regularization in DL</a>). Plot losses on a log
y-axis when they span orders of magnitude.</p>

<h2>Decision boundaries: what the model actually learned</h2>
<p>For 2-D data (or 2 chosen features), color the plane by the model's prediction.
Nothing exposes model character faster: logistic regression draws a straight line,
k-NN draws islands around points, an overfit tree draws jagged slivers that trace
individual noise points:</p>
<pre><code class="language-python">from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

X2, y2 = make_moons(n_samples=300, noise=0.25, random_state=0)
for k, ax in zip([1, 25], plt.subplots(1, 2, figsize=(10, 4))[1]):
    knn = KNeighborsClassifier(k).fit(X2, y2)
    DecisionBoundaryDisplay.from_estimator(knn, X2, ax=ax, alpha=0.4)
    ax.scatter(X2[:, 0], X2[:, 1], c=y2, s=15, edgecolor="k")
    ax.set_title(f"k = {k}: {'jagged = overfit' if k == 1 else 'smooth'}")
plt.show()
</code></pre>

<h2>Feature importance: which inputs matter?</h2>
<pre><code class="language-python">from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import pandas as pd

data = load_breast_cancer()
rf = RandomForestClassifier(random_state=0).fit(Xtr, ytr)

# permutation importance: shuffle a feature, measure the damage (model-agnostic)
imp = permutation_importance(rf, Xte, yte, n_repeats=10, random_state=0)
top = (pd.Series(imp.importances_mean, index=data.feature_names)
         .sort_values().tail(10))
top.plot.barh(xerr=imp.importances_std[top.index.argsort()][-10:])
</code></pre>
<p>Sorted horizontal bars with error bars - always. Two cautions: importance is
about <em>this model's</em> reliance, not causation; and correlated features split
credit between them, so near-zero importance doesn't prove a feature is useless.</p>

<h2>Quick quiz</h2>
<div class="quiz">
  <div class="quiz-q" data-answer="1">
    <p class="quiz-question">1. Training loss ↓ steadily; validation loss fell, then has risen for 10 epochs. Diagnosis?</p>
    <button class="quiz-opt">Underfitting - train longer</button>
    <button class="quiz-opt">Overfitting - stop early or regularize</button>
    <button class="quiz-opt">The learning rate is too low</button>
    <div class="quiz-explain hidden">The model keeps memorizing training data while getting worse on unseen data - the definitive overfitting signature. Roll back to the epoch where validation loss bottomed out.</div>
  </div>
  <div class="quiz-q" data-answer="2">
    <p class="quiz-question">2. Where does a perfect classifier sit on the ROC plot?</p>
    <button class="quiz-opt">Bottom-right corner</button>
    <button class="quiz-opt">On the diagonal</button>
    <button class="quiz-opt">Top-left corner</button>
    <div class="quiz-explain hidden">Top-left = 100% true-positive rate with 0% false-positive rate. The diagonal is coin-flipping; below it is worse than chance (flip the predictions!).</div>
  </div>
  <div class="quiz-q" data-answer="0">
    <p class="quiz-question">3. A digit classifier's confusion matrix shows a hot cell at (actual 3, predicted 8). Value of this insight?</p>
    <button class="quiz-opt">It localizes the failure - collect/augment 3s and 8s, or add features that separate them</button>
    <button class="quiz-opt">None - accuracy already told us everything</button>
    <button class="quiz-opt">It proves the labels are wrong</button>
    <div class="quiz-explain hidden">The matrix turns "92% accurate" into "specifically confuses 3 with 8" - an actionable engineering target instead of a grade.</div>
  </div>
</div>
`};

CONTENT["dataviz/paper-visualizations"] = {
  html: String.raw`
<h1>Visualizations in Research Papers</h1>
<p class="lead">Open any ML paper and you'll meet the same six figure types.
Learning to read them fluently - and to make them honestly - is a real research
skill, and it transfers directly to writing strong project reports.</p>

<h2>1 · Architecture diagrams</h2>
<p>The boxes-and-arrows figure (usually Figure 1) showing data flowing through the
model. Conventions to know: data flows left→right or bottom→top; boxes are layers
or modules with output shapes often annotated on the arrows; repeated structure is
drawn once and marked "× N" (the Transformer paper's encoder stack is the famous
example). When <em>you</em> draw one: consistent shapes per operation type, label
every arrow with its tensor shape, and show where the loss attaches. If a reader
can't reconstruct the model from your diagram, it's decoration, not documentation.</p>

<h2>2 · Training curves with uncertainty bands</h2>
<p>Metric vs training step, with a <strong>shaded band</strong> around each line -
mean ± standard deviation (or a 95% CI) across several random seeds. The band is
the integrity part: single-seed curves can differ dramatically, and a paper
comparing single runs may be comparing luck. When bands overlap heavily, the
methods are not meaningfully different - an application of
<a href="#/math/confidence-intervals">Confidence Intervals</a>.</p>
<pre><code class="language-python">import numpy as np
import matplotlib.pyplot as plt

steps = np.arange(100)
runs = [72 * (1 - np.exp(-steps / 30)) + np.random.default_rng(s).normal(0, 1.5, 100)
        for s in range(5)]                       # 5 seeds
runs = np.array(runs)
mean, std = runs.mean(0), runs.std(0)

plt.plot(steps, mean, label="ours (5 seeds)")
plt.fill_between(steps, mean - std, mean + std, alpha=0.25)   # the band
plt.xlabel("training step"); plt.ylabel("accuracy (%)"); plt.legend()
</code></pre>

<h2>3 · Attention heatmaps</h2>
<p>For attention models (Module 10): a grid where cell \((i, j)\) shows how much
output position \(i\) attends to input position \(j\), dark = strong. In
translation, correctly aligned word pairs light up; in vision, they show
<em>where the model looked</em>. Caveat carried from the interpretability
literature: attention weights are suggestive, not proof of what information the
model used - treat them as hypotheses, not explanations.</p>

<h2>4 · Embedding plots (t-SNE / UMAP)</h2>
<p>High-dimensional representations (Module 5) squeezed to 2-D so clusters become
visible - the classic being MNIST digits separating into ten islands.
Reading rules that save you from over-interpretation:</p>
<ul>
  <li>Cluster <em>membership</em> is meaningful; inter-cluster <em>distances</em> and
      cluster <em>sizes</em> largely are not (t-SNE especially distorts them).</li>
  <li>Axes have no units or meaning - they're arbitrary projection coordinates.</li>
  <li>Different perplexity/seed values change the picture; trust structure that
      survives across settings.</li>
</ul>

<h2>5 · Ablation studies</h2>
<p>The "which parts actually matter?" figure: start from the full system, remove one
component at a time, and chart the performance drop. Usually a small table or bar
chart ("full: 84.2 · − pretraining: 79.1 · − augmentation: 82.6"). This is the
most trustworthy figure type in most papers, because it isolates each design
choice's contribution. When reading one, check the components were removed
<em>one at a time from the full system</em>, not accumulated in a convenient order.</p>

<h2>6 · Comparison tables &amp; bar charts</h2>
<p>Main-results tables live by conventions: <strong>bold</strong> = best per column,
± values = std across seeds, ↑/↓ arrows mark whether higher or lower is better.
Healthy skepticism checklist: Are baselines tuned as carefully as the proposed
method? Are the ± bands disjoint or overlapping? Is the improvement large relative
to seed-to-seed variance?</p>

<h2>Rules for making your own</h2>
<table>
  <tr><th>Rule</th><th>Why</th></tr>
  <tr><td>One message per figure; state it in the caption</td><td>Captions are read before (often instead of) body text</td></tr>
  <tr><td>Self-contained captions</td><td>"Figure 3: Validation accuracy vs epochs, 5 seeds, shaded = ±1 std."</td></tr>
  <tr><td>Vector export (PDF/SVG) for line art</td><td>Crisp at any zoom; PNG only for image grids/heatmaps</td></tr>
  <tr><td>Fonts ≥ 8pt <em>after</em> shrinking to column width</td><td>The #1 reviewer complaint about figures</td></tr>
  <tr><td>Same colors/markers for the same method across all figures</td><td>Readers build a visual vocabulary for your paper</td></tr>
  <tr><td>Colorblind-safe palettes + distinct line styles</td><td>Also survives grayscale printing</td></tr>
</table>

<h2>Quick quiz</h2>
<div class="quiz">
  <div class="quiz-q" data-answer="1">
    <p class="quiz-question">1. The shaded band around a training curve typically represents…</p>
    <button class="quiz-opt">The range of the y-axis</button>
    <button class="quiz-opt">Variability (±std or CI) across runs with different random seeds</button>
    <button class="quiz-opt">The learning-rate schedule</button>
    <div class="quiz-explain hidden">Training is stochastic; the band shows how much runs vary. Two methods whose bands overlap heavily haven't been shown to differ.</div>
  </div>
  <div class="quiz-q" data-answer="2">
    <p class="quiz-question">2. In a t-SNE plot, which conclusion is safe?</p>
    <button class="quiz-opt">"Cluster A is twice as spread out as cluster B"</button>
    <button class="quiz-opt">"Clusters A and B are far apart, so the classes are very different"</button>
    <button class="quiz-opt">"Points of class A group together, so the representation separates class A"</button>
    <div class="quiz-explain hidden">t-SNE preserves local neighborhoods, not global distances or densities. Membership and separation of clusters are readable; sizes and gaps between them are artifacts.</div>
  </div>
  <div class="quiz-q" data-answer="0">
    <p class="quiz-question">3. An ablation table's purpose is to show…</p>
    <button class="quiz-opt">How much each component contributes, by removing them one at a time</button>
    <button class="quiz-opt">The model's performance on many datasets</button>
    <button class="quiz-opt">Training speed on different hardware</button>
    <div class="quiz-explain hidden">Ablations isolate cause and effect within the system design - the closest thing papers have to a controlled experiment on their own architecture.</div>
  </div>
</div>
`};
