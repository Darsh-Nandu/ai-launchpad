/* Module 7 - Deep Learning Foundations */

CONTENT["dl/perceptron"] = {
  html: String.raw`
<h1>Biological Neuron → Perceptron</h1>
<p class="lead">Deep learning starts with a single question: can we make a tiny
mathematical unit that behaves like one brain cell? The 1958 answer - the
perceptron - is where every neural network begins.</p>

<h2>From brain cell to arithmetic</h2>
<p>A biological neuron is a tiny voting machine. It receives electrical signals from
thousands of other neurons through its <em>dendrites</em>. Some connections are strong
(their votes count a lot), some are weak. The cell body adds up all the incoming votes,
and if the total crosses a threshold, the neuron <strong>fires</strong> a signal down its
<em>axon</em> to the next neurons. Below the threshold: silence.</p>
<p>The perceptron copies this with arithmetic:</p>
<ul>
  <li>incoming signals → <strong>inputs</strong> \(x_1, x_2, \dots, x_n\)</li>
  <li>connection strengths → <strong>weights</strong> \(w_1, w_2, \dots, w_n\)</li>
  <li>the cell body's summing → a <strong>weighted sum</strong> \(\mathbf{w}^\top\mathbf{x} + b\)</li>
  <li>fire / don't fire → a <strong>step function</strong> that outputs 1 or 0</li>
</ul>
<p>Everyday analogy: deciding whether to go out. Weather matters a lot to you
(big weight), distance a little (small weight), and you have a general laziness
threshold (the bias). You weigh the factors; if the total enthusiasm crosses your
threshold, you go. That's a perceptron making a decision.</p>

<h2>The model: a weighted vote with a threshold</h2>
<p>The perceptron is a <strong>binary classifier</strong>: it takes a feature vector and
outputs 0 or 1. It computes a weighted sum, then applies a hard threshold:</p>
<div class="math-box">
$$z = \mathbf{w}^\top \mathbf{x} + b
\qquad\qquad
\hat{y} = \text{step}(z) = \begin{cases} 1 &amp; \text{if } z \ge 0 \\ 0 &amp; \text{otherwise} \end{cases}$$
</div>
<p>Geometrically, \(\mathbf{w}^\top\mathbf{x} + b = 0\) defines a line (in 2D) or a
hyperplane (in general) - the <strong>decision boundary</strong>. Everything on one side is
classified 1, everything on the other side 0. Training means rotating and shifting
this boundary until the two classes sit on the correct sides.</p>
<p>Crucially, the perceptron doesn't get its weights from a programmer - it
<strong>learns them from labeled examples</strong>, using an update rule so simple you can
do it by hand.</p>

<h2>The learning rule</h2>
<h3>The perceptron learning rule</h3>
<p>Show the perceptron one example \((\mathbf{x}, y)\) at a time. If it predicts
correctly, do nothing. If it's wrong, nudge the weights <em>toward</em> the correct answer:</p>
<div class="math-box">
$$\mathbf{w} \leftarrow \mathbf{w} + \alpha\,\left(y - \hat{y}\right)\mathbf{x}
\qquad\qquad
b \leftarrow b + \alpha\,\left(y - \hat{y}\right)$$
</div>
<p>Read it case by case - the error \((y - \hat{y})\) can only be −1, 0 or +1:</p>
<ul>
  <li><strong>Correct</strong> (\(y = \hat{y}\)): the error term is 0 → no change.</li>
  <li><strong>Predicted 0, should be 1</strong> (error = +1): add \(\alpha\mathbf{x}\) to the
      weights → the sum \(z\) increases for this input → next time it's closer to firing.</li>
  <li><strong>Predicted 1, should be 0</strong> (error = −1): subtract \(\alpha\mathbf{x}\)
      → \(z\) decreases → closer to staying silent.</li>
</ul>
<p><strong>Convergence theorem (Rosenblatt, 1962):</strong> if the two classes can be
separated by some line at all ("linearly separable"), this loop is guaranteed to find
one in a finite number of updates. If they can't be separated, it loops forever -
which is exactly the famous XOR problem below.</p>

<details class="disclosure">
<summary>Show: why XOR breaks a single perceptron</summary>
<div class="disclosure-body">
<p>XOR outputs 1 for inputs (0,1) and (1,0), and 0 for (0,0) and (1,1). Plot those four
points: the two 1s sit on one diagonal, the two 0s on the other. No single straight
line can put both 1s on one side and both 0s on the other - try it.</p>
<p>Formally: we'd need \(b \lt 0\) (for point 0,0), \(w_1 + b \ge 0\) and
\(w_2 + b \ge 0\) (for the two 1-points), but adding those two gives
\(w_1 + w_2 + 2b \ge 0\), while point (1,1) requires \(w_1 + w_2 + b \lt 0\).
The constraints contradict each other, so no solution exists.</p>
<p>This limitation, published by Minsky &amp; Papert in 1969, froze neural-network
research for years. The fix - stacking perceptrons into layers so the network can
bend the boundary - is the <a href="#/dl/mlp">Multi-Layer Perceptron</a>, the very
next topic.</p>
</div>
</details>


<div class="diagram">
<svg viewBox="0 0 660 300" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Perceptron: inputs times weights flow into a summation unit, then a step function produces the output">
  <defs>
    <marker id="parr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto">
      <path d="M0,0 L10,5 L0,10 z" style="fill: var(--text-muted)"/>
    </marker>
  </defs>
  <!-- inputs -->
  <circle cx="70" cy="70" r="24" class="d-box-soft"/>
  <text x="70" y="75" text-anchor="middle" class="d-text">x₁</text>
  <circle cx="70" cy="150" r="24" class="d-box-soft"/>
  <text x="70" y="155" text-anchor="middle" class="d-text">x₂</text>
  <circle cx="70" cy="230" r="24" class="d-box-soft"/>
  <text x="70" y="235" text-anchor="middle" class="d-text">x₃</text>
  <text x="70" y="30" text-anchor="middle" class="d-text-sm">inputs (dendrites)</text>
  <!-- weighted connections -->
  <line x1="94" y1="70" x2="268" y2="138" class="d-line" marker-end="url(#parr)"/>
  <line x1="94" y1="150" x2="268" y2="150" class="d-line" marker-end="url(#parr)"/>
  <line x1="94" y1="230" x2="268" y2="162" class="d-line" marker-end="url(#parr)"/>
  <text x="165" y="95" class="d-text-accent">w₁</text>
  <text x="165" y="143" class="d-text-accent">w₂</text>
  <text x="165" y="215" class="d-text-accent">w₃</text>
  <!-- bias -->
  <text x="300" y="55" text-anchor="middle" class="d-text-sm">bias b</text>
  <line x1="300" y1="62" x2="300" y2="112" class="d-line" marker-end="url(#parr)"/>
  <!-- summation -->
  <circle cx="300" cy="150" r="30" class="d-box"/>
  <text x="300" y="157" text-anchor="middle" class="d-text" style="font-size:17px">Σ</text>
  <text x="300" y="200" text-anchor="middle" class="d-text-sm">z = w·x + b</text>
  <text x="300" y="218" text-anchor="middle" class="d-text-sm">(cell body sums)</text>
  <line x1="330" y1="150" x2="398" y2="150" class="d-line" marker-end="url(#parr)"/>
  <!-- step activation -->
  <rect x="402" y="118" width="104" height="64" rx="8" class="d-box"/>
  <polyline points="418,168 452,168 452,134 490,134" class="d-curve"/>
  <text x="454" y="200" text-anchor="middle" class="d-text-sm">step(z)</text>
  <text x="454" y="218" text-anchor="middle" class="d-text-sm">(fire / don't fire)</text>
  <line x1="506" y1="150" x2="580" y2="150" class="d-line" marker-end="url(#parr)"/>
  <!-- output -->
  <text x="600" y="156" class="d-text-accent" style="font-size:15px">ŷ ∈ {0, 1}</text>
  <text x="600" y="178" class="d-text-sm">(axon)</text>
</svg>
<div class="caption">The perceptron. Signals arrive, get scaled by weights, are summed with a bias,
and pass through a hard threshold - fire (1) or stay silent (0).</div>
</div>

<h2>A perceptron from scratch</h2>
<p>A complete perceptron from scratch - no frameworks, just the two formulas above.
It learns the AND gate in a handful of passes:</p>
<pre><code class="language-python">import numpy as np

class Perceptron:
    """Rosenblatt's perceptron: step activation + error-driven updates."""

    def __init__(self, n_features, lr=0.1):
        self.w = np.zeros(n_features)   # start with all-zero weights
        self.b = 0.0
        self.lr = lr                    # learning rate (α)

    def predict(self, x):
        z = self.w @ x + self.b         # weighted sum      z = w·x + b
        return 1 if z >= 0 else 0       # step activation   ŷ = step(z)

    def fit(self, X, y, epochs=10):
        for epoch in range(epochs):
            mistakes = 0
            for xi, yi in zip(X, y):
                error = yi - self.predict(xi)    # -1, 0, or +1
                if error != 0:
                    self.w += self.lr * error * xi   # nudge toward correct
                    self.b += self.lr * error
                    mistakes += 1
            print(f"epoch {epoch + 1}: {mistakes} mistakes, "
                  f"w = {self.w}, b = {self.b:.2f}")
            if mistakes == 0:           # converged: everything classified
                break

# --- teach it the AND gate ----------------------------------------------
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([0, 0, 0, 1])              # AND: fires only when both inputs are 1

p = Perceptron(n_features=2)
p.fit(X, y)

print("\npredictions:")
for xi in X:
    print(f"  {xi} -> {p.predict(xi)}")

# The learned boundary is the line  w₁x₁ + w₂x₂ + b = 0.
# Try replacing y with XOR labels [0, 1, 1, 0] and watch it never converge -
# that's the limitation the Multi-Layer Perceptron (next topic) solves.
</code></pre>

<h2>Strengths &amp; limits</h2>
<table>
  <tr><th>Strengths</th><th>Limits</th></tr>
  <tr>
    <td>Simplest possible trainable model - perfect for building intuition</td>
    <td>Only solves linearly separable problems (famously fails on XOR)</td>
  </tr>
  <tr>
    <td>Guaranteed to converge when a separating line exists</td>
    <td>Hard step output: no probabilities, no confidence, not differentiable</td>
  </tr>
  <tr>
    <td>Updates are cheap and online (one example at a time)</td>
    <td>Finds <em>a</em> separating line, not the best one (unlike SVMs)</td>
  </tr>
  <tr>
    <td>Its structure (weights → sum → activation) is the exact building block of deep networks</td>
    <td>Never used alone in practice - it's the atom, not the molecule</td>
  </tr>
</table>
<p><strong>Why it still matters:</strong> replace the step with a smooth activation
(sigmoid, ReLU) and stack layers of these units - that's a neural network, and the
rest of Module 7 is exactly that story.</p>

<h2>Common interview questions</h2>
<div class="qa">
  <div class="qa-q">Q1. What is the difference between a perceptron and logistic regression?</div>
  <div class="qa-a"><p>Both compute \(\mathbf{w}^\top\mathbf{x} + b\), but the perceptron applies a hard
  step (output 0/1, updates only on mistakes), while logistic regression applies a sigmoid
  (output = probability, trained by gradient descent on cross-entropy loss). The sigmoid's
  smoothness is what makes logistic regression differentiable - and differentiability is what
  deep learning is built on.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q2. Why can't a single perceptron learn XOR?</div>
  <div class="qa-a"><p>Its decision boundary is a single straight line, and XOR's positive points
  sit on opposite diagonal corners - no line can separate them. It takes a hidden layer
  (an MLP) to compose two lines into a bent region that isolates the diagonal.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q3. What role does the bias b play?</div>
  <div class="qa-a"><p>It shifts the decision boundary away from the origin. Without a bias, the line
  \(\mathbf{w}^\top\mathbf{x} = 0\) is forced through the origin, which most real
  boundaries don't pass through. Equivalently, the bias is the neuron's firing threshold
  moved to the other side of the equation.</p></div>
</div>

<h2>Quick quiz</h2>
<div class="quiz">
  <div class="quiz-q" data-answer="1">
    <p class="quiz-question">1. A perceptron predicts ŷ = 0 but the true label is y = 1. What happens to the weights?</p>
    <button class="quiz-opt">Nothing - updates only happen on correct predictions</button>
    <button class="quiz-opt">They move by +α·x, making the neuron more likely to fire on this input</button>
    <button class="quiz-opt">They are reset to zero</button>
    <div class="quiz-explain hidden">The error is y − ŷ = +1, so the update is w ← w + α·x. The weighted sum for this input increases, pushing it toward firing next time.</div>
  </div>
  <div class="quiz-q" data-answer="2">
    <p class="quiz-question">2. The perceptron convergence guarantee holds only when…</p>
    <button class="quiz-opt">The learning rate is exactly 0.1</button>
    <button class="quiz-opt">There are fewer than 100 training examples</button>
    <button class="quiz-opt">The classes are linearly separable</button>
    <div class="quiz-explain hidden">Rosenblatt's theorem: if any separating hyperplane exists, the rule finds one in finitely many updates. If none exists (like XOR), training cycles forever.</div>
  </div>
  <div class="quiz-q" data-answer="0">
    <p class="quiz-question">3. In the biological analogy, the weights w correspond to…</p>
    <button class="quiz-opt">The strength of synaptic connections between neurons</button>
    <button class="quiz-opt">The speed of the electrical signal</button>
    <button class="quiz-opt">The number of neurons in the brain</button>
    <div class="quiz-explain hidden">Strong synapses = large weights (that input's vote counts more). Learning, in both brains and perceptrons, is adjusting connection strengths.</div>
  </div>
</div>
`};
