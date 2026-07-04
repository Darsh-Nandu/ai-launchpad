/* Module 5 - Feature Engineering (part 2) */

CONTENT["features/feature-selection"] = {
  html: String.raw`
<h1>Feature Selection Methods</h1>
<p class="lead">More features is not better. Irrelevant columns add noise, redundant
ones add instability, and every extra dimension makes overfitting easier and
models slower and harder to explain. Selection is subtraction as a strategy.</p>

<h2>Why fewer features can mean better models</h2>
<ul>
  <li><strong>Overfitting:</strong> with enough random columns, <em>some</em> will correlate
      with the target by luck; the model learns ghosts.</li>
  <li><strong>Curse of dimensionality:</strong> data gets exponentially sparser per added
      dimension - distance-based methods degrade fast.</li>
  <li><strong>Operations:</strong> every production feature is a pipeline you must
      compute, monitor, and debug at 3 a.m. forever.</li>
  <li><strong>Interpretability:</strong> a 12-feature model can be explained to a
      regulator; a 400-feature one cannot.</li>
</ul>
<p>Three families of methods, in increasing order of cost and quality:</p>

<h2>Family 1: filter methods - score columns, keep the best</h2>
<p>Evaluate each feature <em>independently of any model</em>, using statistics from
Module 1: variance, correlation with the target, mutual information (which,
unlike correlation, catches non-linear dependence), chi-squared for categoricals.</p>
<pre><code class="language-python">from sklearn.feature_selection import (VarianceThreshold,
                                       SelectKBest, mutual_info_classif)

# step 0: kill near-constant columns (no information by definition)
X_var = VarianceThreshold(threshold=0.01).fit_transform(X)

# keep the k best by mutual information with the target
selector = SelectKBest(mutual_info_classif, k=20).fit(X_train, y_train)
X_sel = selector.transform(X_train)
scores = pd.Series(selector.scores_, index=feature_names).sort_values()
</code></pre>
<p><strong>Pros:</strong> fast, scalable, model-agnostic - the right first pass on very
wide data. <strong>Cons:</strong> blind to interactions (an XOR pair scores zero
individually) and to redundancy (keeps five copies of the same signal if all five
score well).</p>

<h2>Family 2: wrapper methods - let a model audition subsets</h2>
<p>Train models on candidate feature subsets and keep what helps. The classic is
<strong>Recursive Feature Elimination</strong> (RFE): fit, drop the weakest feature,
repeat.</p>
<pre><code class="language-python">from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression

rfe = RFECV(LogisticRegression(max_iter=2000),
            step=1, cv=5, scoring="f1")
rfe.fit(X_train, y_train)
print("optimal #features:", rfe.n_features_)
print("kept:", feature_names[rfe.support_])
</code></pre>
<p><strong>Pros:</strong> sees features in combination, tuned to your actual model.
<strong>Cons:</strong> expensive (many model fits), and can overfit the validation
signal if used carelessly - cross-validate the whole selection, as
<code>RFECV</code> does.</p>

<h2>Family 3: embedded methods - selection built into training</h2>
<p><strong>Lasso (L1)</strong> is the star: its penalty drives weak coefficients to
<em>exactly zero</em>, so training and selection happen simultaneously
(the geometry of why is in <a href="#/ml/regularization">Regularization</a>):</p>
<pre><code class="language-python">from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

pipe = make_pipeline(StandardScaler(),        # L1 requires scaled features!
                     LassoCV(cv=5)).fit(X_train, y_train)
coefs = pd.Series(pipe[-1].coef_, index=feature_names)
print("dropped:", (coefs == 0).sum(), "features")
print(coefs[coefs != 0].sort_values())
</code></pre>
<p>Tree-based importance + <code>SelectFromModel</code> is the other everyday embedded
route. Prefer <strong>permutation importance</strong> over impurity-based importance
for the ranking - impurity importance inflates high-cardinality features. One trick
worth stealing: add a column of pure random noise; any feature ranking <em>below
the noise column</em> has earned its deletion.</p>

<h2>Choosing a strategy</h2>
<table>
  <tr><th>Situation</th><th>Sensible approach</th></tr>
  <tr><td>Thousands of columns (genomics, text)</td><td>Filter first (variance + MI) to a few hundred, then embedded</td></tr>
  <tr><td>Tabular, &lt; ~100 columns, need interpretability</td><td>Lasso or RFECV</td></tr>
  <tr><td>Boosting model, moderate width</td><td>Often none needed - but permutation importance + noise column to prune</td></tr>
  <tr><td>Correlated feature clusters</td><td>Handle redundancy first (<a href="#/eda/correlation-analysis">VIF</a>, clustering) - importance scores split across duplicates</td></tr>
</table>
<div class="callout warn">
  <span class="co-title">The selection leak</span>
  Selecting features using <em>all</em> the data, then cross-validating the model on
  the same data, is a famous silent error: the selection already peeked at every
  fold's answers. With enough random features, this alone manufactures a
  great-looking model from pure noise. Selection must live <em>inside</em> the CV
  pipeline, like every other fitted step.
</div>

<h2>Common interview questions</h2>
<div class="qa">
  <div class="qa-q">Q1. Filter vs wrapper vs embedded - one line each.</div>
  <div class="qa-a"><p>Filter: score features independently with statistics (fast, ignores interactions).
  Wrapper: search feature subsets by training models (thorough, expensive). Embedded:
  the training objective itself prunes features (Lasso's zeroed coefficients,
  tree importances) - a practical middle ground.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q2. Why does L1 zero out coefficients while L2 only shrinks them?</div>
  <div class="qa-a"><p>The L1 penalty's constant gradient (±λ) can outweigh a weak feature's loss gradient
  all the way to zero, and its diamond-shaped constraint has corners <em>on the axes</em> -
  solutions land there. L2's penalty gradient vanishes near zero, so weights approach but
  never reach it.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q3. You selected the top-20 features on the full dataset, then got 5-fold CV accuracy of 0.91. What's wrong?</div>
  <div class="qa-a"><p>Selection saw the validation folds' labels before CV began - the 0.91 is
  contaminated. Redo with selection inside each fold (Pipeline + cross_val_score); the
  honest number will be lower, sometimes dramatically so on wide data.</p></div>
</div>

<h2>Quick quiz</h2>
<div class="quiz">
  <div class="quiz-q" data-answer="2">
    <p class="quiz-question">1. Two features are individually useless but jointly predictive (XOR-like). Which method family is guaranteed to miss this?</p>
    <button class="quiz-opt">Wrapper methods</button>
    <button class="quiz-opt">Embedded tree methods</button>
    <button class="quiz-opt">Univariate filter methods</button>
    <div class="quiz-explain hidden">Filters score one column at a time - combination effects are invisible by construction. Wrappers and trees evaluate features in context.</div>
  </div>
  <div class="quiz-q" data-answer="0">
    <p class="quiz-question">2. A feature ranks below an injected random-noise column in permutation importance. Conclusion?</p>
    <button class="quiz-opt">It's contributing less than pure noise - a strong candidate for removal</button>
    <button class="quiz-opt">The noise column is a great feature</button>
    <button class="quiz-opt">Permutation importance is broken</button>
    <div class="quiz-explain hidden">The noise column sets an empirical floor for "no real signal". Anything under it is fitting randomness.</div>
  </div>
  <div class="quiz-q" data-answer="1">
    <p class="quiz-question">3. Lasso selected different features on two nearby data samples. Most likely reason?</p>
    <button class="quiz-opt">Lasso is nondeterministic</button>
    <button class="quiz-opt">Correlated features - Lasso arbitrarily picks one representative per group</button>
    <button class="quiz-opt">The learning rate was too high</button>
    <div class="quiz-explain hidden">Among near-duplicates, L1 keeps whichever fits the noise slightly better this time. Cluster or combine correlated features first (or use Elastic Net) for stable selections.</div>
  </div>
</div>
`};

CONTENT["features/dimensionality-reduction"] = {
  html: String.raw`
<h1>Dimensionality Reduction (PCA, t-SNE, UMAP)</h1>
<p class="lead">Instead of choosing which columns to keep, dimensionality reduction
<em>builds new axes</em> - fewer dimensions that preserve most of what mattered.
PCA does it with linear algebra you already know; t-SNE and UMAP do it for
your eyes.</p>

<h2>PCA: find the directions where the data actually varies</h2>
<p>Picture a flat elliptical cloud of points in 3-D - like a pancake floating in
space. It technically has 3 coordinates, but almost all the variation lives in the
pancake's 2-D plane. PCA finds that plane automatically: it rotates the axes so
that <strong>PC1</strong> points along the direction of maximum variance,
<strong>PC2</strong> along the maximum remaining variance (perpendicular to PC1),
and so on.</p>
<div class="diagram">
<svg viewBox="0 0 640 300" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Elliptical scatter cloud with two principal component arrows: PC1 along the long axis, PC2 along the short axis">
  <defs>
    <marker id="pcaarr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto">
      <path d="M0,0 L10,5 L0,10 z" style="fill: var(--accent)"/>
    </marker>
  </defs>
  <line x1="50" y1="260" x2="600" y2="260" class="d-line"/>
  <line x1="70" y1="280" x2="70" y2="30" class="d-line"/>
  <!-- tilted elliptical cloud -->
  <g>
    <circle cx="160" cy="220" r="4" class="d-dot-muted"/><circle cx="200" cy="205" r="4" class="d-dot-muted"/>
    <circle cx="235" cy="185" r="4" class="d-dot-muted"/><circle cx="270" cy="175" r="4" class="d-dot-muted"/>
    <circle cx="305" cy="150" r="4" class="d-dot-muted"/><circle cx="340" cy="140" r="4" class="d-dot-muted"/>
    <circle cx="375" cy="115" r="4" class="d-dot-muted"/><circle cx="410" cy="105" r="4" class="d-dot-muted"/>
    <circle cx="445" cy="85" r="4" class="d-dot-muted"/><circle cx="480" cy="70" r="4" class="d-dot-muted"/>
    <circle cx="225" cy="215" r="4" class="d-dot-muted"/><circle cx="290" cy="185" r="4" class="d-dot-muted"/>
    <circle cx="355" cy="155" r="4" class="d-dot-muted"/><circle cx="420" cy="130" r="4" class="d-dot-muted"/>
    <circle cx="255" cy="150" r="4" class="d-dot-muted"/><circle cx="320" cy="120" r="4" class="d-dot-muted"/>
    <circle cx="385" cy="90" r="4" class="d-dot-muted"/><circle cx="195" cy="180" r="4" class="d-dot-muted"/>
  </g>
  <!-- PC arrows -->
  <line x1="320" y1="150" x2="490" y2="62" class="d-line-accent" marker-end="url(#pcaarr)"/>
  <text x="500" y="55" class="d-text-accent">PC1 - max variance</text>
  <line x1="320" y1="150" x2="272" y2="62" class="d-line-accent" marker-end="url(#pcaarr)"/>
  <text x="150" y="52" class="d-text-accent">PC2 ⟂ PC1</text>
  <text x="90" y="290" class="d-text-sm">Projecting onto PC1 alone keeps most of the cloud's structure - 2-D becomes 1-D cheaply.</text>
</svg>
<div class="caption">PCA rotates to the data's natural axes. Keeping the first few
components = keeping the directions where things actually happen.</div>
</div>
<p>The math connects three chapters of Module 1: standardize the data
(<a href="#/features/scaling">scaling matters</a> - PCA chases variance, and raw
variance is unit-dependent), compute the covariance matrix, and take its
<a href="#/math/linear-algebra">eigenvectors</a>:</p>
<div class="math-box">
$$\mathbf{C} = \frac{1}{m-1}\mathbf{X}^\top\mathbf{X}
\qquad
\mathbf{C}\mathbf{v}_k = \lambda_k \mathbf{v}_k$$
</div>
<p>Eigenvectors \(\mathbf{v}_k\) are the principal components (new axes);
eigenvalues \(\lambda_k\) are the variance each captures. The
<strong>explained variance ratio</strong> \(\lambda_k / \sum_j \lambda_j\) tells you
how many components you need:</p>
<pre><code class="language-python">from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np

pca_pipe = make_pipeline(StandardScaler(), PCA())
pca_pipe.fit(X)
pca = pca_pipe[-1]

cum = np.cumsum(pca.explained_variance_ratio_)
print(np.round(cum[:10], 3))
# e.g. [0.42 0.61 0.72 0.79 0.84 ...]  -> 5 components ≈ 84% of variance

# keep enough for 95%: PCA(n_components=0.95) does it automatically
X_reduced = make_pipeline(StandardScaler(),
                          PCA(n_components=0.95)).fit_transform(X)
print(X.shape, "->", X_reduced.shape)
</code></pre>
<p><strong>Uses:</strong> compressing correlated features before a linear model or KNN,
de-noising, decorrelating inputs, speeding up training, and 2-D plotting.
<strong>Costs:</strong> components are blends of all original features ("0.4·income +
0.3·age − …"), so interpretability suffers; and PCA is unsupervised - the
direction of maximum variance is not always the direction that predicts your
target.</p>

<h2>t-SNE and UMAP: maps for your eyes</h2>
<p>PCA is linear - it can only rotate and project. When the data lives on a curved
manifold (the classic "Swiss roll"), linear projection squashes distinct regions
on top of each other. t-SNE and UMAP instead try to preserve <em>neighborhoods</em>:
points that were close in high-D should stay close in the 2-D map.</p>
<ul>
  <li><strong>t-SNE</strong> - converts pairwise distances to "who is whose neighbor"
      probabilities and matches them in 2-D. Superb cluster pictures; slow on big
      data; the <code>perplexity</code> knob (≈ effective neighborhood size, try 5–50)
      changes the picture noticeably.</li>
  <li><strong>UMAP</strong> - same spirit, better engineering: much faster, preserves a
      bit more global structure, has <code>transform()</code> for new points. The
      modern default for embedding visualization.</li>
</ul>
<pre><code class="language-python">from sklearn.manifold import TSNE
# import umap  (pip install umap-learn)
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

digits = load_digits()                      # 1797 images, 64 dims
emb = TSNE(perplexity=30, random_state=0).fit_transform(digits.data)

plt.figure(figsize=(7, 6))
sc = plt.scatter(emb[:, 0], emb[:, 1], c=digits.target,
                 cmap="tab10", s=8)
plt.legend(*sc.legend_elements(), title="digit")
plt.title("64-dimensional digits, mapped to 2-D: ten islands appear")
</code></pre>
<div class="callout warn">
  <span class="co-title">Reading rules (same as in the papers chapter)</span>
  In t-SNE/UMAP plots, trust cluster <em>membership</em>; distrust cluster
  <em>sizes</em> and <em>distances between clusters</em> - both are distorted by
  construction. And don't feed t-SNE output into a classifier: these are
  visualization tools, not preprocessing (use PCA or UMAP-with-care for that).
</div>

<h2>Which tool when</h2>
<table>
  <tr><th>Goal</th><th>Tool</th><th>Why</th></tr>
  <tr><td>Compress features for modeling</td><td>PCA</td><td>fast, deterministic, invertible-ish, transform() for new data</td></tr>
  <tr><td>Visualize cluster structure</td><td>UMAP (or t-SNE)</td><td>preserves neighborhoods → visible islands</td></tr>
  <tr><td>De-correlate features</td><td>PCA</td><td>components are orthogonal by construction</td></tr>
  <tr><td>Interpretability of features</td><td><a href="#/features/feature-selection">Selection</a>, not reduction</td><td>original columns survive</td></tr>
  <tr><td>Compress + non-linear structure + new data</td><td>UMAP, or autoencoders (<a href="#/advanced/autoencoders">Module 11</a>)</td><td>learned non-linear embeddings</td></tr>
</table>

<h2>Common interview questions</h2>
<div class="qa">
  <div class="qa-q">Q1. Walk through what PCA computes and how you choose the number of components.</div>
  <div class="qa-a"><p>Standardize; take the covariance matrix; its eigenvectors (sorted by eigenvalue)
  are orthogonal directions of decreasing variance; project onto the top k. Choose k via the
  cumulative explained-variance curve (elbow or a 90–95% target) - or by downstream
  cross-validated performance.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q2. Why must you scale before PCA?</div>
  <div class="qa-a"><p>PCA maximizes variance, and variance depends on units: an income column in rupees
  has a million times the variance of age in years, so PC1 would simply <em>be</em> income.
  Standardizing puts features on equal footing so the components reflect structure, not units.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q3. Is PCA feature selection?</div>
  <div class="qa-a"><p>No - selection keeps a subset of original columns; PCA creates new synthetic axes
  that mix all of them. Reduction can achieve better compression, but every original feature
  must still be collected in production, and interpretability of individual inputs is lost.</p></div>
</div>

<h2>Quick quiz</h2>
<div class="quiz">
  <div class="quiz-q" data-answer="1">
    <p class="quiz-question">1. PC1 explains 45%, PC2 25%, PC3 10% of variance. Keeping PC1+PC2 preserves…</p>
    <button class="quiz-opt">45% of the variance</button>
    <button class="quiz-opt">70% of the variance</button>
    <button class="quiz-opt">100% of the information</button>
    <div class="quiz-explain hidden">Explained variance is additive across orthogonal components: 45 + 25 = 70%. (Variance ≠ all information - a low-variance direction can still be predictive.)</div>
  </div>
  <div class="quiz-q" data-answer="2">
    <p class="quiz-question">2. In your t-SNE plot, cluster A sits 3× farther from B than from C. Safe conclusion?</p>
    <button class="quiz-opt">A is 3× more different from B than from C</button>
    <button class="quiz-opt">B has more members than C</button>
    <button class="quiz-opt">None - inter-cluster distances in t-SNE are not meaningful</button>
    <div class="quiz-explain hidden">t-SNE optimizes local neighborhoods; global geometry is a side effect. Only "these points group together" survives scrutiny.</div>
  </div>
  <div class="quiz-q" data-answer="0">
    <p class="quiz-question">3. PCA on unscaled data where one feature is in millions will produce…</p>
    <button class="quiz-opt">A first component that is essentially that one feature</button>
    <button class="quiz-opt">Balanced components</button>
    <button class="quiz-opt">An error</button>
    <div class="quiz-explain hidden">Maximum-variance direction ≈ the biggest-unit column. The "reduction" would just relabel that feature as PC1. Standardize first, always.</div>
  </div>
</div>
`};
