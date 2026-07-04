/* Module 6 - Machine Learning Algorithms (part 2) */

CONTENT["ml/learning-paradigms"] = {
  html: String.raw`
<h1>Supervised vs Unsupervised vs Reinforcement Learning</h1>
<p class="lead">All of machine learning sorts into three paradigms, distinguished by
one question: <em>what kind of feedback does the algorithm get?</em> Answers, structure,
or rewards.</p>

<h2>Supervised learning: learning from answer keys</h2>
<p>You provide examples <em>with the correct answers attached</em> - labeled data
\((\mathbf{x}, y)\) - and the algorithm learns the mapping \(\hat{y} = f(\mathbf{x})\).
Like studying with solved past papers. Two sub-flavors by target type:</p>
<ul>
  <li><strong>Regression</strong> - \(y\) is a continuous number: price, temperature,
      demand. (<a href="#/ml/linear-regression">Linear Regression</a> and friends.)</li>
  <li><strong>Classification</strong> - \(y\) is a category: spam/not, disease A/B/C,
      digit 0–9. (<a href="#/ml/logistic-regression">Logistic Regression</a> onward.)</li>
</ul>
<p>Supervised learning is ~80% of industrial ML, because businesses have historical
records with outcomes attached: past loans → repaid or not; past emails → marked
spam or not. Its bottleneck is exactly that requirement: <strong>labels are
expensive</strong> - someone or something must have produced the answer key.</p>

<h2>Unsupervised learning: finding structure without answers</h2>
<p>Only \(\mathbf{x}\), no \(y\). The algorithm hunts for structure that was always
there: groups, patterns, compressible regularities. Like sorting a box of mixed
LEGO by similarity without being told the categories. Main tasks:</p>
<ul>
  <li><strong>Clustering</strong> - discover groups:
      <a href="#/ml/kmeans">K-Means</a>, <a href="#/ml/hierarchical-clustering">hierarchical</a>,
      <a href="#/ml/dbscan">DBSCAN</a>. (Customer segments, topic groups.)</li>
  <li><strong>Dimensionality reduction</strong> - compress while preserving structure:
      <a href="#/features/dimensionality-reduction">PCA, t-SNE, UMAP</a>.</li>
  <li><strong>Anomaly detection</strong> - model "normal", flag what doesn't fit
      (fraud, failures).</li>
</ul>
<p>Evaluation is the hard part: with no ground truth, "good clusters" is partly a
judgment call - a recurring theme when we reach clustering.</p>

<h2>Reinforcement learning: learning from consequences</h2>
<p>No dataset at all. An <strong>agent</strong> acts in an <strong>environment</strong>,
receives <strong>rewards</strong>, and learns a <strong>policy</strong> (state → action) that
maximizes long-run reward. Like training a dog: no manual, just treats. Two
signature difficulties: rewards are <em>delayed</em> (the winning move happened forty
moves ago - the credit assignment problem) and the agent must balance
<em>exploration vs exploitation</em> (try new things vs repeat what works).
AlphaGo, robotics, and RLHF for language models are the famous applications.
This course focuses on the first two paradigms; RL deserves its own course.</p>

<div class="diagram">
<svg viewBox="0 0 660 240" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Three panels: supervised with labeled points and a boundary, unsupervised with unlabeled clusters, reinforcement with an agent-environment loop">
  <!-- supervised -->
  <rect x="15" y="25" width="200" height="170" rx="8" class="d-box-muted"/>
  <text x="115" y="48" text-anchor="middle" class="d-text">Supervised</text>
  <circle cx="55" cy="90" r="6" class="d-dot"/><circle cx="80" cy="115" r="6" class="d-dot"/>
  <circle cx="60" cy="140" r="6" class="d-dot"/>
  <circle cx="160" cy="90" r="6" class="d-dot-muted"/><circle cx="175" cy="120" r="6" class="d-dot-muted"/>
  <circle cx="150" cy="145" r="6" class="d-dot-muted"/>
  <line x1="115" y1="70" x2="115" y2="165" class="d-line-accent" stroke-dasharray="6 4"/>
  <text x="115" y="185" text-anchor="middle" class="d-text-sm">labels → learn the boundary</text>
  <!-- unsupervised -->
  <rect x="230" y="25" width="200" height="170" rx="8" class="d-box-muted"/>
  <text x="330" y="48" text-anchor="middle" class="d-text">Unsupervised</text>
  <circle cx="275" cy="90" r="6" class="d-dot-muted"/><circle cx="295" cy="105" r="6" class="d-dot-muted"/>
  <circle cx="270" cy="115" r="6" class="d-dot-muted"/>
  <circle cx="375" cy="130" r="6" class="d-dot-muted"/><circle cx="390" cy="112" r="6" class="d-dot-muted"/>
  <circle cx="380" cy="150" r="6" class="d-dot-muted"/>
  <ellipse cx="282" cy="103" rx="34" ry="30" class="d-line-accent" fill="none" stroke-dasharray="5 4"/>
  <ellipse cx="382" cy="130" rx="34" ry="32" class="d-line-accent" fill="none" stroke-dasharray="5 4"/>
  <text x="330" y="185" text-anchor="middle" class="d-text-sm">no labels → find the groups</text>
  <!-- RL -->
  <rect x="445" y="25" width="200" height="170" rx="8" class="d-box-muted"/>
  <text x="545" y="48" text-anchor="middle" class="d-text">Reinforcement</text>
  <rect x="470" y="70" width="70" height="34" rx="6" class="d-box-soft"/>
  <text x="505" y="92" text-anchor="middle" class="d-text-sm">agent</text>
  <rect x="550" y="130" width="80" height="34" rx="6" class="d-box-soft"/>
  <text x="590" y="152" text-anchor="middle" class="d-text-sm">environment</text>
  <path d="M 540 84 C 590 84, 605 100, 600 128" class="d-line" fill="none"/>
  <text x="600" y="105" class="d-text-sm">action</text>
  <path d="M 550 150 C 490 155, 470 135, 495 106" class="d-line" fill="none"/>
  <text x="452" y="140" class="d-text-sm">reward,</text>
  <text x="452" y="155" class="d-text-sm">state</text>
  <text x="545" y="185" text-anchor="middle" class="d-text-sm">act → observe → improve</text>
</svg>
<div class="caption">The three paradigms, by feedback type: answers, structure, rewards.</div>
</div>

<h2>Choosing the paradigm is choosing the question</h2>
<table>
  <tr><th>Business question</th><th>Paradigm</th><th>Why</th></tr>
  <tr><td>"Which customers will churn next month?"</td><td>Supervised (classification)</td><td>history contains labeled outcomes</td></tr>
  <tr><td>"What will this house sell for?"</td><td>Supervised (regression)</td><td>continuous labeled target</td></tr>
  <tr><td>"What kinds of customers do we even have?"</td><td>Unsupervised (clustering)</td><td>no predefined answer exists</td></tr>
  <tr><td>"Is this transaction weird?"</td><td>Unsupervised (anomaly)</td><td>fraud too rare/novel for labels</td></tr>
  <tr><td>"What ad-bidding strategy maximizes revenue?"</td><td>Reinforcement</td><td>sequential decisions, delayed reward</td></tr>
</table>
<p>Between the main paradigms live useful hybrids you'll meet later:
<strong>semi-supervised</strong> (few labels + many unlabeled examples) and
<strong>self-supervised</strong> (invent labels from the data itself - mask a word,
predict it - the trick that trains GPT and BERT, Module 10).</p>

<h2>Quick quiz</h2>
<div class="quiz">
  <div class="quiz-q" data-answer="1">
    <p class="quiz-question">1. Predicting delivery time from distance, traffic, and order size is…</p>
    <button class="quiz-opt">Unsupervised learning</button>
    <button class="quiz-opt">Supervised regression</button>
    <button class="quiz-opt">Reinforcement learning</button>
    <div class="quiz-explain hidden">Historical deliveries have known durations (labels), and the target is continuous → supervised regression.</div>
  </div>
  <div class="quiz-q" data-answer="2">
    <p class="quiz-question">2. Grouping news articles by topic without any predefined topic list is…</p>
    <button class="quiz-opt">Classification</button>
    <button class="quiz-opt">Regression</button>
    <button class="quiz-opt">Clustering (unsupervised)</button>
    <div class="quiz-explain hidden">No labels exist - the topics themselves must be discovered from structure. With a fixed topic list and labeled examples, it would become classification.</div>
  </div>
  <div class="quiz-q" data-answer="0">
    <p class="quiz-question">3. The defining challenge pair of reinforcement learning is…</p>
    <button class="quiz-opt">Delayed rewards (credit assignment) and exploration vs exploitation</button>
    <button class="quiz-opt">Missing values and outliers</button>
    <button class="quiz-opt">Overfitting and underfitting</button>
    <div class="quiz-explain hidden">Which of my many past actions earned this reward? And should I try something new or repeat what worked? These two define RL's difficulty.</div>
  </div>
</div>
`};

CONTENT["ml/logistic-regression"] = {
  html: String.raw`
<h1>Logistic Regression</h1>
<p class="lead">Despite the name, it's a <em>classifier</em> - and the most used one in
industry. It takes linear regression's machinery and bends the output into a
probability. Master it and you've already learned half of neural networks.</p>

<h2>Why not just use linear regression for yes/no?</h2>
<p>Try predicting "will churn (1) / won't (0)" with a straight line and two things
break: predictions escape [0, 1] (what's a probability of 1.7?), and extreme
x-values drag the line, moving the decision point absurdly. We need a function
that takes the familiar linear score \(z = \mathbf{w}^\top\mathbf{x} + b\) - which
ranges over all of \((-\infty, \infty)\) - and squashes it into a probability.</p>

<h2>The sigmoid: squashing scores into probabilities</h2>
<div class="math-box">
$$\sigma(z) = \frac{1}{1 + e^{-z}}
\qquad\qquad
\hat{y} = \sigma(\mathbf{w}^\top \mathbf{x} + b) = P(y = 1 \mid \mathbf{x})$$
</div>
<div class="diagram">
<svg viewBox="0 0 640 280" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Sigmoid curve from 0 to 1 with midpoint 0.5 at z equals 0, and the decision threshold marked">
  <line x1="40" y1="230" x2="610" y2="230" class="d-line"/>
  <line x1="320" y1="255" x2="320" y2="30" class="d-line" stroke-dasharray="3 3"/>
  <line x1="40" y1="40" x2="610" y2="40" class="d-grid" stroke-dasharray="3 3"/>
  <path d="M 50 228 C 200 226, 250 215, 320 135 C 390 55, 440 44, 600 42" class="d-curve"/>
  <circle cx="320" cy="135" r="5" class="d-dot"/>
  <text x="330" y="128" class="d-text-accent">σ(0) = 0.5 - the decision boundary</text>
  <text x="615" y="234" class="d-text-sm">z</text>
  <text x="45" y="35" class="d-text-sm">1</text>
  <text x="45" y="225" class="d-text-sm">0</text>
  <text x="80" y="205" class="d-text-sm">z ≪ 0 → P ≈ 0 → predict class 0</text>
  <text x="360" y="80" class="d-text-sm">z ≫ 0 → P ≈ 1 → predict class 1</text>
</svg>
<div class="caption">The sigmoid maps any real score to (0, 1), steepest at the middle
where the model is least sure.</div>
</div>
<p>Predicting class 1 when \(\hat{y} \ge 0.5\) is the same as \(z \ge 0\), so the
<strong>decision boundary is still the straight line/hyperplane</strong>
\(\mathbf{w}^\top\mathbf{x} + b = 0\) - logistic regression is a linear classifier
that happens to also give you calibrated confidence on each side of the line.</p>
<p>Interpretability bonus: the raw score \(z\) is the <em>log-odds</em>,
\(z = \ln\frac{P}{1-P}\), so each weight \(w_j\) says: "+1 unit of feature \(j\)
multiplies the odds by \(e^{w_j}\)". This is why medicine, credit scoring, and
insurance - fields that must justify decisions - still run on logistic regression.</p>

<h2>The loss: why not MSE, and what instead</h2>
<p>Plugging the sigmoid into squared error creates a lumpy, non-convex loss with
flat regions where the gradient dies. The right loss comes from maximum likelihood:
maximize the probability the model assigns to the true labels, i.e. minimize
<strong>binary cross-entropy (log loss)</strong>:</p>
<div class="math-box">
$$J(\mathbf{w}, b) = -\frac{1}{m}\sum_{i=1}^{m}\Big[\, y^{(i)} \ln \hat{y}^{(i)} + (1 - y^{(i)}) \ln (1 - \hat{y}^{(i)}) \,\Big]$$
</div>
<p>Read it per example: if \(y = 1\), the loss is \(-\ln\hat{y}\) - tiny when the model
says 0.99, <em>infinite</em> as it approaches 0. Confident wrong answers are punished
brutally; that's the behavior you want from a probability machine. And this loss is
convex: one global minimum, no traps.</p>
<details class="disclosure">
<summary>Show the beautiful part: the gradient comes out identical to linear regression's</summary>
<div class="disclosure-body">
<p>Two chain-rule facts make everything collapse. First, the sigmoid's derivative:
\(\sigma'(z) = \sigma(z)(1 - \sigma(z))\). Second, differentiating the loss for one
example with respect to \(z\):</p>
$$\frac{\partial J}{\partial \hat{y}} = -\frac{y}{\hat{y}} + \frac{1-y}{1-\hat{y}}
\qquad
\frac{\partial \hat{y}}{\partial z} = \hat{y}(1-\hat{y})$$
$$\frac{\partial J}{\partial z}
= \left(-\frac{y}{\hat{y}} + \frac{1-y}{1-\hat{y}}\right)\hat{y}(1-\hat{y})
= \hat{y} - y$$
<p>The messy terms cancel to the plain error. So the weight gradient is:</p>
$$\boxed{\;\frac{\partial J}{\partial \mathbf{w}} = \frac{1}{m}\mathbf{X}^\top(\hat{\mathbf{y}} - \mathbf{y})\;}$$
<p>- the <em>same formula</em> as linear regression, with sigmoid outputs. This pairing
(sigmoid + cross-entropy → clean gradient) is engineered, not coincidental, and it
returns as the output layer of every binary-classification neural network.</p>
</div>
</details>

<h2>From scratch, then with sklearn</h2>
<pre><code class="language-python">import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# toy data: hours studied + previous score -> pass/fail
rng = np.random.default_rng(0)
m = 200
X = np.column_stack([rng.uniform(0, 10, m), rng.uniform(30, 100, m)])
true_z = 0.9 * X[:, 0] + 0.06 * X[:, 1] - 8
y = (rng.random(m) &lt; sigmoid(true_z)).astype(float)

# standardize (Module 5!), add gradient descent
Xs = (X - X.mean(0)) / X.std(0)
w, b, alpha = np.zeros(2), 0.0, 0.5

for epoch in range(300):
    y_hat = sigmoid(Xs @ w + b)          # forward: probabilities
    error = y_hat - y                     # the clean gradient signal
    w -= alpha * Xs.T @ error / m
    b -= alpha * error.mean()
    if epoch % 100 == 0:
        loss = -np.mean(y*np.log(y_hat+1e-12) + (1-y)*np.log(1-y_hat+1e-12))
        print(f"epoch {epoch:3d}  loss {loss:.4f}")

acc = ((sigmoid(Xs @ w + b) &gt;= 0.5) == y).mean()
print("train accuracy:", round(acc, 3))

# --- sklearn, production style -------------------------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

clf = make_pipeline(StandardScaler(),
                    LogisticRegression())   # L2 regularization by default!
clf.fit(X, y)
print(clf.predict_proba(X[:3]))             # calibrated probabilities
</code></pre>
<p>Note sklearn's default L2 penalty (<code>C</code> = inverse strength) - in practice
logistic regression is almost always regularized
(<a href="#/ml/regularization">why, later</a>). For more than two classes, the
sigmoid generalizes to <strong>softmax</strong> and the same story continues
(multinomial logistic regression).</p>

<h2>Strengths, limits, when to reach for it</h2>
<table>
  <tr><th>Strengths</th><th>Limits</th></tr>
  <tr><td>Outputs real probabilities (usually well-calibrated)</td><td>Linear decision boundary only - needs engineered features for curves</td></tr>
  <tr><td>Coefficients have exact odds-ratio interpretations</td><td>Struggles when classes are perfectly separable (weights blow up - regularize)</td></tr>
  <tr><td>Fast, stable, scales to millions of rows, convex training</td><td>Sensitive to strongly correlated features (unstable coefficients)</td></tr>
  <tr><td>The default baseline for every classification problem</td><td>Usually beaten on raw accuracy by boosting on messy tabular data</td></tr>
</table>

<h2>Common interview questions</h2>
<div class="qa">
  <div class="qa-q">Q1. Why cross-entropy instead of MSE for classification?</div>
  <div class="qa-a"><p>With a sigmoid, MSE gives a non-convex loss whose gradient contains a
  \(\hat{y}(1-\hat{y})\) factor that vanishes exactly when the model is confidently wrong -
  learning stalls. Cross-entropy is convex, matches maximum likelihood, and its gradient
  \((\hat{y} - y)\) stays strong until errors are fixed.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q2. Is logistic regression's decision boundary linear? What if the data isn't?</div>
  <div class="qa-a"><p>Yes - \(\mathbf{w}^\top\mathbf{x} + b = 0\). For curved boundaries, add engineered
  features (x², interactions), use kernels (<a href="#/ml/svm">SVM</a>), trees, or neural
  nets. The sigmoid curves the <em>probability</em>, never the boundary.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q3. A coefficient for "num_support_tickets" is 0.69. Interpret it.</div>
  <div class="qa-a"><p>Each additional ticket multiplies the odds of churn by \(e^{0.69} \approx 2\) -
  doubling the odds - holding other features fixed. This direct interpretability is the
  model's superpower.</p></div>
</div>

<h2>Quick quiz</h2>
<div class="quiz">
  <div class="quiz-q" data-answer="1">
    <p class="quiz-question">1. σ(z) = 0.5 exactly when…</p>
    <button class="quiz-opt">z = 0.5</button>
    <button class="quiz-opt">z = 0</button>
    <button class="quiz-opt">z = 1</button>
    <div class="quiz-explain hidden">σ(0) = 1/(1+e⁰) = 1/2. That's why "predict 1 if probability ≥ 0.5" equals "predict 1 if the linear score is ≥ 0".</div>
  </div>
  <div class="quiz-q" data-answer="0">
    <p class="quiz-question">2. The model predicts P = 0.99 for a customer who doesn't churn. Cross-entropy for this example is…</p>
    <button class="quiz-opt">−ln(0.01) ≈ 4.6 - a huge penalty for confident wrongness</button>
    <button class="quiz-opt">0.01 - a tiny penalty</button>
    <button class="quiz-opt">0.99</button>
    <div class="quiz-explain hidden">With y = 0 the loss is −ln(1−ŷ) = −ln(0.01). Confidence is expensive when wrong - exactly the incentive a probability model needs.</div>
  </div>
  <div class="quiz-q" data-answer="2">
    <p class="quiz-question">3. Compared to linear regression, logistic regression's gradient formula is…</p>
    <button class="quiz-opt">Completely different</button>
    <button class="quiz-opt">Undefined - the loss isn't differentiable</button>
    <button class="quiz-opt">Identical in form: Xᵀ(ŷ − y)/m, with ŷ now a sigmoid output</button>
    <div class="quiz-explain hidden">The sigmoid+cross-entropy pairing makes the messy derivative terms cancel to (ŷ − y). One update rule to rule them all - and it carries straight into neural nets.</div>
  </div>
</div>
`};

CONTENT["ml/knn"] = {
  html: String.raw`
<h1>K-Nearest Neighbors</h1>
<p class="lead">KNN has no training phase, no weights, no loss function.
Its entire theory: <em>you are probably similar to your neighbors.</em>
To classify a point, look at the k closest known examples and take a vote.</p>

<h2>The algorithm in full</h2>
<ol>
  <li>Store the training data. (That's the whole "training".)</li>
  <li>For a new point, compute its distance to every stored example.</li>
  <li>Take the \(k\) closest.</li>
  <li>Classification: majority vote (weighted by closeness, optionally).
      Regression: average their targets.</li>
</ol>
<p>KNN is a <strong>lazy</strong> (instance-based) learner: zero cost up front, all cost
at prediction time - the mirror image of everything else in this module. It is also
<strong>non-parametric</strong>: no fixed equation form; the "model" is the data itself,
so it can trace arbitrarily curvy boundaries given enough examples.</p>
<div class="math-box">
$$d_{euclid}(\mathbf{a}, \mathbf{b}) = \sqrt{\textstyle\sum_j (a_j - b_j)^2}
\qquad
d_{manhattan}(\mathbf{a}, \mathbf{b}) = \textstyle\sum_j |a_j - b_j|$$
</div>
<p>Distance is doing all the work - which triggers the iron rule from Module 5:
<strong><a href="#/features/scaling">scale your features</a></strong>, or the
biggest-unit column becomes the only voter.</p>

<h2>Choosing k: the bias-variance dial in its purest form</h2>
<div class="diagram">
<svg viewBox="0 0 640 250" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Two decision boundaries: k=1 jagged islands around noise points, k=15 smooth boundary">
  <rect x="20" y="30" width="290" height="180" rx="8" class="d-box-muted"/>
  <text x="165" y="52" text-anchor="middle" class="d-text">k = 1 - memorizes noise</text>
  <path d="M 40 160 C 70 120, 90 190, 120 140 C 140 105, 150 180, 180 120 C 205 80, 220 160, 250 100 C 270 70, 285 110, 295 90"
        class="d-line-accent" fill="none"/>
  <circle cx="105" cy="95" r="4" class="d-dot"/><circle cx="140" cy="75" r="4" class="d-dot"/>
  <circle cx="230" cy="70" r="4" class="d-dot"/><circle cx="75" cy="180" r="4" class="d-dot-muted"/>
  <circle cx="160" cy="185" r="4" class="d-dot-muted"/><circle cx="245" cy="170" r="4" class="d-dot-muted"/>
  <circle cx="150" cy="140" r="4" class="d-dot-muted"/>
  <text x="165" y="200" text-anchor="middle" class="d-text-sm">boundary snakes around single points</text>
  <!-- right -->
  <rect x="330" y="30" width="290" height="180" rx="8" class="d-box-muted"/>
  <text x="475" y="52" text-anchor="middle" class="d-text">k = 15 - smooth consensus</text>
  <path d="M 350 150 C 420 130, 520 115, 610 95" class="d-line-accent" fill="none"/>
  <circle cx="415" cy="95" r="4" class="d-dot"/><circle cx="450" cy="80" r="4" class="d-dot"/>
  <circle cx="540" cy="70" r="4" class="d-dot"/><circle cx="385" cy="180" r="4" class="d-dot-muted"/>
  <circle cx="470" cy="185" r="4" class="d-dot-muted"/><circle cx="555" cy="160" r="4" class="d-dot-muted"/>
  <circle cx="460" cy="120" r="4" class="d-dot"/>
  <text x="475" y="200" text-anchor="middle" class="d-text-sm">one noisy point can't bend it</text>
</svg>
<div class="caption">Small k trusts individuals (low bias, high variance); large k trusts
the crowd (higher bias, low variance). Same tradeoff as everywhere in ML - visible here
with the naked eye.</div>
</div>
<ul>
  <li><strong>k = 1:</strong> perfect training accuracy, jagged boundary, every noise
      point owns an island - overfitting incarnate.</li>
  <li><strong>k = m:</strong> everyone gets the majority class - underfitting incarnate.</li>
  <li>In between: choose by <a href="#/ml/cross-validation">cross-validation</a>;
      odd k avoids ties; \(\sqrt{m}\) is a folk starting point, not a law.</li>
</ul>

<h2>The curse of dimensionality: where KNN goes to die</h2>
<p>KNN's premise - nearby points are similar - quietly dies as dimensions grow.
In high-dimensional space, volume concentrates in corners, and <em>everything
becomes almost equally far from everything</em>: the ratio between the nearest and
farthest neighbor's distance approaches 1. "Nearest" stops meaning anything.
Concretely: with 20+ informative dimensions, expect KNN to disappoint;
with 100+, don't bother without
<a href="#/features/dimensionality-reduction">dimensionality reduction</a> first.
Irrelevant features hurt doubly: they add noise to every distance while adding
no signal - KNN, unlike trees or Lasso, has no mechanism to ignore them.</p>

<h2>In code</h2>
<pre><code class="language-python">from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, stratify=y, random_state=0)

pipe = make_pipeline(StandardScaler(), KNeighborsClassifier())

# choose k honestly, by cross-validation
grid = GridSearchCV(pipe,
                    {"kneighborsclassifier__n_neighbors": range(1, 32, 2),
                     "kneighborsclassifier__weights": ["uniform", "distance"]},
                    cv=5)
grid.fit(Xtr, ytr)
print("best params:", grid.best_params_)
print("test accuracy:", round(grid.score(Xte, yte), 3))
</code></pre>
<p><code>weights="distance"</code> lets closer neighbors vote louder - usually a small
free win. For speed at scale, sklearn builds KD-trees/ball-trees automatically;
at web scale everyone switches to <em>approximate</em> nearest neighbors (FAISS,
Annoy, HNSW) - the same algorithm powering vector databases and embedding search
today, which is why KNN never really went out of style.</p>

<h2>Strengths &amp; weaknesses</h2>
<table>
  <tr><th>Strengths</th><th>Weaknesses</th></tr>
  <tr><td>Zero training time; trivially simple to explain</td><td>Slow predictions on big data (distance to everyone)</td></tr>
  <tr><td>Naturally non-linear, multi-class, and incremental (just add rows)</td><td>Curse of dimensionality; helpless against irrelevant features</td></tr>
  <tr><td>A strong "is there local structure?" diagnostic baseline</td><td>Needs scaling; memory-hungry (stores everything)</td></tr>
  <tr><td>Distance-weighted votes give soft confidence</td><td>No model to inspect - zero global interpretability</td></tr>
</table>

<h2>Common interview questions</h2>
<div class="qa">
  <div class="qa-q">Q1. Why is KNN called a lazy learner, and what does that trade?</div>
  <div class="qa-a"><p>It builds nothing at training time - just stores data. Cost moves to inference:
  O(m·n) distance computations per query (before tree/ANN indexing). Eager learners pay
  once at training and answer fast forever; KNN is the reverse.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q2. How does k relate to bias and variance?</div>
  <div class="qa-a"><p>Small k → low bias, high variance (flexible, noise-sensitive: k=1 has 0 training
  error by definition). Large k → high bias, low variance (smooth, may miss structure).
  Cross-validate to find the sweet spot - KNN is the cleanest illustration of the
  <a href="#/ml/bias-variance">tradeoff</a> in all of ML.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q3. Your KNN performs great on 5 features, terribly after adding 200 more. Why?</div>
  <div class="qa-a"><p>Curse of dimensionality plus irrelevant-feature noise: distances become uniform and
  dominated by the 200 uninformative columns. Fix: feature selection, PCA/UMAP first, or
  switch to a model that self-selects features (trees, Lasso).</p></div>
</div>

<h2>Quick quiz</h2>
<div class="quiz">
  <div class="quiz-q" data-answer="0">
    <p class="quiz-question">1. With k = 1, training-set accuracy is always…</p>
    <button class="quiz-opt">100% - every point's nearest neighbor is itself</button>
    <button class="quiz-opt">Around 50%</button>
    <button class="quiz-opt">Unpredictable</button>
    <div class="quiz-explain hidden">Which is exactly why training accuracy tells you nothing about a 1-NN model - evaluation must use held-out data.</div>
  </div>
  <div class="quiz-q" data-answer="1">
    <p class="quiz-question">2. Forgetting to scale features before KNN means…</p>
    <button class="quiz-opt">Slightly slower training</button>
    <button class="quiz-opt">The largest-unit feature silently dominates every distance</button>
    <button class="quiz-opt">Nothing - KNN is scale-invariant</button>
    <div class="quiz-explain hidden">Distance sums squared differences: a salary column in thousands out-shouts an age column completely. Scaling is not optional for distance-based models.</div>
  </div>
  <div class="quiz-q" data-answer="2">
    <p class="quiz-question">3. "In high dimensions, the nearest neighbor is barely nearer than the farthest point." This is…</p>
    <button class="quiz-opt">A bug in sklearn</button>
    <button class="quiz-opt">Overfitting</button>
    <button class="quiz-opt">The curse of dimensionality - distance contrast collapses</button>
    <div class="quiz-explain hidden">Volume spreads exponentially with dimensions; all pairwise distances concentrate around the same value, and "nearest" loses its meaning.</div>
  </div>
</div>
`};

CONTENT["ml/decision-trees"] = {
  html: String.raw`
<h1>Decision Trees</h1>
<p class="lead">A decision tree is a flowchart of yes/no questions learned from data -
the only major model your grandmother could audit by reading it. The catch:
left unchecked, it will happily memorize your dataset, noise and all.</p>

<h2>The model: twenty questions, learned optimally</h2>
<div class="diagram">
<svg viewBox="0 0 660 300" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="A small decision tree for loan approval: income question at root, then credit score and employment years, leaves show approve or deny">
  <rect x="245" y="20" width="170" height="44" rx="8" class="d-box"/>
  <text x="330" y="40" text-anchor="middle" class="d-text">income &gt; 50k?</text>
  <text x="330" y="56" text-anchor="middle" class="d-text-sm">root (all 1000 applicants)</text>
  <line x1="285" y1="64" x2="185" y2="105" class="d-line"/>
  <line x1="375" y1="64" x2="475" y2="105" class="d-line"/>
  <text x="205" y="88" class="d-text-sm">no (620)</text>
  <text x="420" y="88" class="d-text-sm">yes (380)</text>
  <rect x="95" y="108" width="180" height="44" rx="8" class="d-box"/>
  <text x="185" y="128" text-anchor="middle" class="d-text">credit score &gt; 700?</text>
  <text x="185" y="144" text-anchor="middle" class="d-text-sm">internal node</text>
  <rect x="390" y="108" width="170" height="44" rx="8" class="d-box"/>
  <text x="475" y="128" text-anchor="middle" class="d-text">employed &gt; 2 yrs?</text>
  <line x1="140" y1="152" x2="95" y2="195" class="d-line"/>
  <line x1="230" y1="152" x2="275" y2="195" class="d-line"/>
  <line x1="430" y1="152" x2="395" y2="195" class="d-line"/>
  <line x1="520" y1="152" x2="560" y2="195" class="d-line"/>
  <rect x="35" y="198" width="120" height="42" rx="8" class="d-box-muted"/>
  <text x="95" y="217" text-anchor="middle" class="d-text">deny</text>
  <text x="95" y="233" text-anchor="middle" class="d-text-sm">92% repaid: no</text>
  <rect x="215" y="198" width="120" height="42" rx="8" class="d-box-soft"/>
  <text x="275" y="217" text-anchor="middle" class="d-text">approve</text>
  <text x="275" y="233" text-anchor="middle" class="d-text-sm">85% repaid: yes</text>
  <rect x="335" y="198" width="120" height="42" rx="8" class="d-box-muted"/>
  <text x="395" y="217" text-anchor="middle" class="d-text">deny</text>
  <rect x="500" y="198" width="120" height="42" rx="8" class="d-box-soft"/>
  <text x="560" y="217" text-anchor="middle" class="d-text">approve</text>
  <text x="560" y="233" text-anchor="middle" class="d-text-sm">96% repaid: yes</text>
  <text x="330" y="280" text-anchor="middle" class="d-text-sm">leaves predict the majority class (or mean, for regression) of their training members</text>
</svg>
<div class="caption">A learned flowchart. Every prediction is a walk from root to leaf -
and the path IS the explanation.</div>
</div>
<p>Note what trees get for free: non-linear boundaries (axis-aligned rectangles),
feature interactions (the meaning of "credit &gt; 700" depends on the income branch
you took), zero need for scaling (<a href="#/features/scaling">splits only use
order</a>), native multi-class, and human-readable logic.</p>

<h2>How splits are chosen: impurity</h2>
<p>Training asks, at every node: <em>which question best un-mixes the classes?</em>
"Mixed-ness" is measured by an impurity function over the class proportions
\(p_k\) in a node:</p>
<div class="math-box">
$$\text{Gini}(t) = 1 - \sum_k p_k^2
\qquad\qquad
\text{Entropy}(t) = -\sum_k p_k \log_2 p_k$$
</div>
<p>Both are 0 for a pure node (all one class) and maximal for a 50/50 mix; Gini is
the slightly cheaper default, and they almost always agree in practice. The
algorithm (CART) tries <em>every feature and every threshold</em>, computing each
candidate's <strong>impurity reduction</strong>:</p>
<div class="math-box">
$$\Delta = \text{Impurity(parent)} \;-\; \frac{m_L}{m}\,\text{Impurity(left)} \;-\; \frac{m_R}{m}\,\text{Impurity(right)}$$
</div>
<p>and takes the greediest split. Then it recurses on each child until a stopping
rule fires. Worked example: a node holds 40 repaid / 40 defaulted
(Gini \(= 1 - 0.5^2 - 0.5^2 = 0.5\)). Splitting on "income &gt; 50k" gives
left = 30/10 (Gini 0.375) and right = 10/30 (Gini 0.375):
\(\Delta = 0.5 - 0.5(0.375) - 0.5(0.375) = 0.125\). Any split with a bigger Δ wins.
(For regression trees, replace impurity with variance of the target - leaves
predict their mean.)</p>

<h2>The disease: unlimited depth = memorization</h2>
<p>Let the recursion run forever and every leaf ends up with one training point:
100% training accuracy, garbage generalization. A fully-grown tree is a lookup
table wearing a flowchart costume. The controls (<em>pre-pruning</em>):</p>
<pre><code class="language-python">from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt

X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, stratify=y, random_state=0)

# unconstrained: memorizes
deep = DecisionTreeClassifier(random_state=0).fit(Xtr, ytr)
print(f"deep:   train {deep.score(Xtr, ytr):.3f} | test {deep.score(Xte, yte):.3f}")
# e.g.  train 1.000 | test 0.91   <- the gap is the overfit

# constrained: generalizes
tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=10,
                              random_state=0).fit(Xtr, ytr)
print(f"pruned: train {tree.score(Xtr, ytr):.3f} | test {tree.score(Xte, yte):.3f}")

# the payoff: read the actual model
plt.figure(figsize=(16, 7))
plot_tree(tree, feature_names=load_breast_cancer().feature_names,
          class_names=["malignant", "benign"], filled=True, fontsize=8)
</code></pre>
<p>Key knobs: <code>max_depth</code> (hard complexity cap),
<code>min_samples_leaf</code> (no leaf may hold fewer than n examples - my favorite:
it directly forbids one-point memorization), and <code>ccp_alpha</code>
(post-pruning: grow fully, then cut branches that don't pay their complexity rent -
tune it by cross-validation).</p>

<h2>Two more quirks worth knowing</h2>
<ul>
  <li><strong>Instability:</strong> tiny data changes can flip an early split and rebuild
      the whole tree differently - high variance. This weakness is exactly what
      <a href="#/ml/random-forest">Random Forests</a> exploit: average many unstable
      trees and the noise cancels.</li>
  <li><strong>Axis-aligned bias:</strong> trees cut parallel to axes, so a diagonal
      boundary gets approximated by a staircase of many splits. Rotating features
      (PCA) sometimes helps.</li>
</ul>

<h2>Common interview questions</h2>
<div class="qa">
  <div class="qa-q">Q1. How does a tree decide its splits?</div>
  <div class="qa-a"><p>Greedy exhaustive search: for every feature and threshold, compute the weighted
  impurity (Gini/entropy) reduction of the resulting children; take the best; recurse.
  Greedy means locally optimal - the globally optimal tree is NP-hard, so no lookahead.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q2. Why do trees overfit so readily, and name three defenses.</div>
  <div class="qa-a"><p>Unbounded recursive splitting can carve a region around every noise point
  (leaves of size 1). Defenses: cap depth, require min_samples_leaf, cost-complexity
  post-pruning (ccp_alpha) - or ensemble them away (bagging/boosting).</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q3. Why don't decision trees need feature scaling?</div>
  <div class="qa-a"><p>A split "x &gt; t" depends only on the ordering of values; any monotonic transform
  maps to an equivalent threshold with identical partitions. Distances and gradients -
  the scale-sensitive machinery - never appear.</p></div>
</div>

<h2>Quick quiz</h2>
<div class="quiz">
  <div class="quiz-q" data-answer="1">
    <p class="quiz-question">1. A node holds 50/50 of two classes. Its Gini impurity is…</p>
    <button class="quiz-opt">0 - perfectly pure</button>
    <button class="quiz-opt">0.5 - maximally impure for two classes</button>
    <button class="quiz-opt">1.0</button>
    <div class="quiz-explain hidden">Gini = 1 − (0.5² + 0.5²) = 0.5, the worst case for binary. A pure node scores 0 - the target every split chases.</div>
  </div>
  <div class="quiz-q" data-answer="0">
    <p class="quiz-question">2. min_samples_leaf = 20 prevents overfitting by…</p>
    <button class="quiz-opt">Forbidding leaves that represent fewer than 20 examples - no memorizing individuals</button>
    <button class="quiz-opt">Limiting the tree to 20 total leaves</button>
    <button class="quiz-opt">Dropping features with under 20 unique values</button>
    <div class="quiz-explain hidden">Every prediction region must be backed by at least 20 training points, so single noise points can't own territory.</div>
  </div>
  <div class="quiz-q" data-answer="2">
    <p class="quiz-question">3. Train accuracy 1.00, test accuracy 0.78 on your unconstrained tree. The diagnosis and cure?</p>
    <button class="quiz-opt">Underfitting - grow deeper</button>
    <button class="quiz-opt">Perfect model - ship it</button>
    <button class="quiz-opt">Overfitting - prune (max_depth / min_samples_leaf / ccp_alpha) or ensemble</button>
    <div class="quiz-explain hidden">The 22-point generalization gap is textbook variance. Constrain the tree, or better: hand the problem to a Random Forest - next chapter.</div>
  </div>
</div>
`};
