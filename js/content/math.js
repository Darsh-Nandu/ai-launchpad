/* Module 1 - Math & Stats Foundations */

CONTENT["math/linear-algebra"] = {
  html: String.raw`
<h1>Linear Algebra for ML</h1>
<p class="lead">Vectors, matrices, the dot product, and eigenvalues - the four ideas
that almost every ML algorithm is built out of.</p>

<h2>Data is just vectors and matrices</h2>
<p>Open any spreadsheet of data. One <strong>row</strong> - say, a house with
3 bedrooms, 120 m², built in 1995 - is just a list of numbers:
<code>[3, 120, 1995]</code>. Linear algebra is simply the math of such lists
(<em>vectors</em>) and tables of them (<em>matrices</em>).</p>
<p>That's the whole secret: <strong>a data point is a vector, a dataset is a matrix</strong>.
When an ML model "learns", it is doing arithmetic on these vectors and matrices -
mostly multiplications and additions, billions of times. Learn the small set of
rules for that arithmetic and every formula later in this course becomes readable.</p>
<p>A useful mental picture: a vector is an <em>arrow from the origin</em> in space.
A house described by 3 numbers is an arrow in 3-dimensional space; a customer
described by 40 features is an arrow in 40-dimensional space. You can't picture
40 dimensions - nobody can - but the arithmetic works identically, so intuition
built in 2D carries over.</p>

<h2>Vectors and matrices, formally</h2>
<h3>Vectors</h3>
<p>A vector \(\mathbf{x} \in \mathbb{R}^n\) is an ordered list of \(n\) real numbers.
In ML, a vector usually represents one of two things:</p>
<ul>
  <li><strong>A data point:</strong> one row of your dataset - its features.</li>
  <li><strong>Model parameters:</strong> the weights \(\mathbf{w}\) the model learns.</li>
</ul>
<h3>Matrices</h3>
<p>A matrix \(\mathbf{X} \in \mathbb{R}^{m \times n}\) is a grid with \(m\) rows and
\(n\) columns. In ML the convention is: <strong>rows = examples, columns = features</strong>.
A dataset of 1,000 houses with 3 features each is a \(1000 \times 3\) matrix.</p>
<h3>Why multiplication matters</h3>
<p>The single most executed operation in machine learning is the matrix-vector
product \(\mathbf{X}\mathbf{w}\): "apply the same weighted sum to every row at once."
A neural network layer, a linear regression prediction, an attention score -
all are matrix products. GPUs exist in ML because they do exactly this fast.</p>

<div class="callout">
  <span class="co-title">Notation reminder</span>
  Lowercase bold \(\mathbf{x}, \mathbf{w}\) = vectors. Uppercase bold \(\mathbf{X}, \mathbf{W}\) = matrices.
  \(x_i\) = the \(i\)-th element of \(\mathbf{x}\). \(\mathbf{X}^\top\) = transpose (flip rows and columns).
  This convention is used on every page of this site.
</div>

<h2>The four operations you must know</h2>
<h3>Dot product</h3>
<p>For two vectors of the same length, multiply element-wise and add up:</p>
<div class="math-box">
$$\mathbf{a} \cdot \mathbf{b} \;=\; \mathbf{a}^\top \mathbf{b} \;=\; \sum_{i=1}^{n} a_i\, b_i$$
</div>
<p>Geometrically, the dot product measures <strong>alignment</strong>:</p>
<div class="math-box">
$$\mathbf{a} \cdot \mathbf{b} = \lVert \mathbf{a} \rVert \, \lVert \mathbf{b} \rVert \cos\theta$$
</div>
<p>where \(\theta\) is the angle between the arrows. Same direction → large positive.
Perpendicular → zero. Opposite → negative. This is why "similarity" in ML
(recommendations, embeddings, attention) is usually a dot product or its scaled
cousin, <em>cosine similarity</em>.</p>

<h3>Norm (length of a vector)</h3>
<div class="math-box">
$$\lVert \mathbf{x} \rVert_2 = \sqrt{\textstyle\sum_i x_i^2}
\qquad\qquad
\lVert \mathbf{x} \rVert_1 = \textstyle\sum_i |x_i|$$
</div>
<p>The L2 norm is ordinary Euclidean length; the L1 norm is the "taxicab" distance.
These two reappear in Module 6 as <strong>Ridge (L2)</strong> and <strong>Lasso (L1)</strong>
regularization.</p>

<h3>Matrix multiplication</h3>
<p>\(\mathbf{C} = \mathbf{A}\mathbf{B}\) is defined when the inner dimensions agree:
\((m \times n)(n \times p) \rightarrow (m \times p)\). Each entry is a dot product
of a row of \(\mathbf{A}\) with a column of \(\mathbf{B}\):</p>
<div class="math-box">
$$C_{ij} = \sum_{k=1}^{n} A_{ik} B_{kj}$$
</div>
<p>Note that \(\mathbf{A}\mathbf{B} \neq \mathbf{B}\mathbf{A}\) in general - order matters.</p>

<h3>Eigenvalues and eigenvectors</h3>
<p>For a square matrix \(\mathbf{A}\), an eigenvector \(\mathbf{v}\) is a special
direction that the matrix does <em>not rotate</em> - it only stretches it, by a
factor \(\lambda\) (the eigenvalue):</p>
<div class="math-box">
$$\mathbf{A}\mathbf{v} = \lambda \mathbf{v}, \qquad \mathbf{v} \neq \mathbf{0}$$
</div>
<p>Intuition: a matrix transforms space (stretching, squashing, rotating).
Eigenvectors are the axes along which that transformation is a pure stretch.
In ML you meet them in <strong>PCA</strong> (Module 5), where the eigenvectors of the
data's covariance matrix point along the directions of greatest variance -
the "natural axes" of your data.</p>

<details class="disclosure">
<summary>Show: how eigenvalues are actually found (characteristic equation)</summary>
<div class="disclosure-body">
<p>Rearranging \(\mathbf{A}\mathbf{v} = \lambda\mathbf{v}\) gives
\((\mathbf{A} - \lambda \mathbf{I})\mathbf{v} = \mathbf{0}\). A non-zero solution
\(\mathbf{v}\) exists only if the matrix \(\mathbf{A} - \lambda\mathbf{I}\) is singular,
i.e. its determinant is zero:</p>
$$\det(\mathbf{A} - \lambda \mathbf{I}) = 0$$
<p>For a \(2\times 2\) matrix
\(\mathbf{A} = \begin{pmatrix} a &amp; b \\ c &amp; d \end{pmatrix}\) this expands to the
quadratic \(\lambda^2 - (a+d)\lambda + (ad - bc) = 0\). Solve it, get the eigenvalues,
substitute each back to find its eigenvector. In practice you never do this by hand
for real data - <code>np.linalg.eig</code> does it - but knowing where the numbers
come from removes the mystery.</p>
</div>
</details>


<div class="diagram">
<svg viewBox="0 0 640 300" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Left: two vectors and the angle between them. Right: dataset as a matrix.">
  <!-- left: vectors and angle -->
  <line x1="40" y1="260" x2="300" y2="260" class="d-line"/>
  <line x1="60" y1="280" x2="60" y2="40" class="d-line"/>
  <text x="295" y="278" class="d-text-sm">x₁</text>
  <text x="42" y="46" class="d-text-sm">x₂</text>
  <!-- vector a -->
  <line x1="60" y1="260" x2="250" y2="120" class="d-line-accent"/>
  <circle cx="250" cy="120" r="4" class="d-dot"/>
  <text x="258" y="116" class="d-text-accent">a = (4, 3)</text>
  <!-- vector b -->
  <line x1="60" y1="260" x2="160" y2="70" class="d-line"/>
  <circle cx="160" cy="70" r="4" class="d-dot-muted"/>
  <text x="150" y="56" class="d-text-sm">b = (2, 4)</text>
  <!-- angle arc -->
  <path d="M 120 216 A 74 74 0 0 1 130 200" class="d-line"/>
  <text x="132" y="212" class="d-text-sm">θ</text>
  <text x="60" y="296" class="d-text-sm">a · b = ‖a‖‖b‖cos θ  - small angle → large dot product → “similar”</text>

  <!-- right: dataset as matrix -->
  <text x="380" y="60" class="d-text-accent">Dataset = matrix X (m × n)</text>
  <rect x="380" y="75" width="220" height="150" rx="6" class="d-box"/>
  <line x1="380" y1="112" x2="600" y2="112" class="d-grid"/>
  <line x1="380" y1="150" x2="600" y2="150" class="d-grid"/>
  <line x1="380" y1="187" x2="600" y2="187" class="d-grid"/>
  <line x1="453" y1="75" x2="453" y2="225" class="d-grid"/>
  <line x1="526" y1="75" x2="526" y2="225" class="d-grid"/>
  <text x="398" y="98" class="d-text-sm">3</text>
  <text x="470" y="98" class="d-text-sm">120</text>
  <text x="545" y="98" class="d-text-sm">1995</text>
  <text x="398" y="135" class="d-text-sm">2</text>
  <text x="470" y="135" class="d-text-sm">85</text>
  <text x="545" y="135" class="d-text-sm">2003</text>
  <text x="398" y="172" class="d-text-sm">4</text>
  <text x="470" y="172" class="d-text-sm">150</text>
  <text x="545" y="172" class="d-text-sm">1988</text>
  <text x="398" y="210" class="d-text-sm">⋮</text>
  <text x="470" y="210" class="d-text-sm">⋮</text>
  <text x="545" y="210" class="d-text-sm">⋮</text>
  <text x="380" y="248" class="d-text-sm">each row = one example (a vector)</text>
  <text x="380" y="266" class="d-text-sm">each column = one feature</text>
</svg>
<div class="caption">Left: the dot product measures how aligned two vectors are.
Right: a dataset is just a matrix - one row per example.</div>
</div>

<h2>All of it in NumPy</h2>
<p>Everything above, in NumPy. Run it and check the numbers against the formulas:</p>
<pre><code class="language-python">import numpy as np

# --- vectors -----------------------------------------------------------
a = np.array([4.0, 3.0])
b = np.array([2.0, 4.0])

dot = a @ b                        # 4*2 + 3*4 = 20.0  (same as np.dot(a, b))
norm_a = np.linalg.norm(a)         # sqrt(16 + 9) = 5.0
norm_b = np.linalg.norm(b)

# angle between the vectors, from  a·b = ‖a‖‖b‖cosθ
cos_theta = dot / (norm_a * norm_b)
print("dot product :", dot)
print("cosine sim  :", round(cos_theta, 3))   # 1.0 = same direction

# --- matrices ----------------------------------------------------------
# 3 houses, 3 features (bedrooms, area, year) -> a 3x3 matrix
X = np.array([[3, 120, 1995],
              [2,  85, 2003],
              [4, 150, 1988]], dtype=float)

w = np.array([10.0, 0.5, 0.01])    # a weight vector (one weight per feature)

# One matrix-vector product = a weighted sum applied to EVERY row at once.
# This exact operation is the heart of linear regression and neural nets.
predictions = X @ w
print("predictions :", predictions)

# --- eigenvalues -------------------------------------------------------
A = np.array([[2.0, 1.0],
              [1.0, 2.0]])
eigenvalues, eigenvectors = np.linalg.eig(A)
print("eigenvalues :", eigenvalues)            # [3. 1.]

# verify the definition  A v = λ v  for the first pair
v, lam = eigenvectors[:, 0], eigenvalues[0]
print("A @ v       :", A @ v)
print("λ * v       :", lam * v)                # identical -> definition holds
</code></pre>

<h2>Where you'll use this</h2>
<table>
  <tr><th>Concept</th><th>Shows up in</th></tr>
  <tr><td>Dot product</td><td>Linear regression predictions, neural network layers, attention scores, cosine similarity in recommendations &amp; embeddings</td></tr>
  <tr><td>Matrix multiplication</td><td>Every neural network forward pass; batch processing of data</td></tr>
  <tr><td>L1 / L2 norms</td><td>Lasso &amp; Ridge regularization, distance metrics in KNN and K-Means</td></tr>
  <tr><td>Eigenvalues / eigenvectors</td><td>PCA (dimensionality reduction), understanding covariance, spectral clustering</td></tr>
  <tr><td>Transpose</td><td>The normal equation in linear regression, gradient formulas in backprop</td></tr>
</table>

<h2>Common interview questions</h2>
<div class="qa">
  <div class="qa-q">Q1. What does the dot product of two vectors tell you?</div>
  <div class="qa-a"><p>How aligned they are. It equals \(\lVert\mathbf{a}\rVert\lVert\mathbf{b}\rVert\cos\theta\):
  positive when they point the same way, zero when perpendicular (orthogonal), negative
  when opposite. Normalized by the lengths, it becomes cosine similarity - the standard
  similarity measure for embeddings.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q2. What is an eigenvector, in one sentence?</div>
  <div class="qa-a"><p>A direction that a matrix transformation only stretches (by the eigenvalue)
  without rotating - in PCA, the covariance matrix's eigenvectors are the directions of
  maximum variance in the data.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q3. Why must the inner dimensions match in matrix multiplication?</div>
  <div class="qa-a"><p>Because each output entry is a dot product of a row of the first matrix with a
  column of the second - and a dot product needs both vectors to have the same length.
  \((m\times n)(n\times p)\rightarrow(m\times p)\).</p></div>
</div>

<h2>Quick quiz</h2>
<div class="quiz">
  <div class="quiz-q" data-answer="2">
    <p class="quiz-question">1. Two vectors have a dot product of 0. What does that mean geometrically?</p>
    <button class="quiz-opt">They point in the same direction</button>
    <button class="quiz-opt">They have zero length</button>
    <button class="quiz-opt">They are perpendicular (orthogonal)</button>
    <div class="quiz-explain hidden">Since \(\mathbf{a}\cdot\mathbf{b} = \lVert\mathbf{a}\rVert\lVert\mathbf{b}\rVert\cos\theta\), a zero dot product (with non-zero vectors) forces \(\cos\theta = 0\), i.e. a 90° angle.</div>
  </div>
  <div class="quiz-q" data-answer="1">
    <p class="quiz-question">2. A dataset has 500 examples and 12 features. Using this site's convention, its matrix shape is…</p>
    <button class="quiz-opt">12 × 500</button>
    <button class="quiz-opt">500 × 12</button>
    <button class="quiz-opt">500 × 500</button>
    <div class="quiz-explain hidden">Rows = examples, columns = features, so \(\mathbf{X} \in \mathbb{R}^{500 \times 12}\).</div>
  </div>
  <div class="quiz-q" data-answer="0">
    <p class="quiz-question">3. In \(\mathbf{A}\mathbf{v} = \lambda\mathbf{v}\), the eigenvalue λ describes…</p>
    <button class="quiz-opt">How much the eigenvector is stretched</button>
    <button class="quiz-opt">The angle the eigenvector is rotated by</button>
    <button class="quiz-opt">The number of dimensions of the matrix</button>
    <div class="quiz-explain hidden">The matrix leaves the eigenvector's direction unchanged and scales it by λ - a pure stretch (or flip, if λ is negative).</div>
  </div>
</div>
`};
