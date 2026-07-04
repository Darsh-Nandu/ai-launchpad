/* Module 7 - Deep Learning Foundations (part 3) */

CONTENT["dl/backpropagation"] = {
  html: String.raw`
<h1>Backpropagation</h1>
<p class="lead">The algorithm that trains every neural network on Earth is the
<a href="#/math/calculus">chain rule</a>, applied systematically, backwards,
with aggressive reuse of intermediate results. This chapter derives it fully -
slowly, then in matrix form.</p>

<h2>The question backprop answers</h2>
<p>After a forward pass we have a loss - one number. The network has millions of
weights. For gradient descent we need
\(\partial J / \partial w\) for <em>every single one</em>: "if I nudge this weight,
how does the loss move?" Computing each derivative separately would cost one
forward pass per weight - millions of passes. Backprop gets <strong>all of them in
one backward sweep</strong>, at roughly the cost of two forward passes. That
efficiency is the entire reason deep learning is computationally possible.</p>

<h2>Step 1: one neuron, one chain</h2>
<p>Take the last layer of a binary classifier:
\(z = \mathbf{w}^\top \mathbf{a} + b\), then \(\hat{y} = \sigma(z)\), then
\(J = -[y \ln \hat{y} + (1-y)\ln(1-\hat{y})]\). The loss depends on \(w_1\) only
through a chain of intermediate quantities:</p>
<div class="math-box">
$$w_1 \;\rightarrow\; z \;\rightarrow\; \hat{y} \;\rightarrow\; J
\qquad\Longrightarrow\qquad
\frac{\partial J}{\partial w_1} =
\frac{\partial J}{\partial \hat{y}} \cdot
\frac{\partial \hat{y}}{\partial z} \cdot
\frac{\partial z}{\partial w_1}$$
</div>
<p>Each local factor is easy (derived in
<a href="#/ml/logistic-regression">Logistic Regression</a>):
\(\partial J/\partial\hat{y} \cdot \partial\hat{y}/\partial z = \hat{y} - y\), and
\(\partial z/\partial w_1 = a_1\). So
\(\partial J/\partial w_1 = (\hat{y} - y)\,a_1\). Notice the structure:
<strong>(error signal arriving at the neuron) × (input that came through this
weight)</strong>. All of backprop is this pattern, repeated.</p>

<h2>Step 2: the key trick - reuse the error signal</h2>
<p>Define the error signal at layer \(l\) as
\(\boldsymbol{\delta}^{[l]} \equiv \partial J / \partial \mathbf{z}^{[l]}\) -
"how much does the loss care about this layer's pre-activation?"
Why this quantity? Because <em>every</em> gradient in layer \(l\) factors through it,
and because it can be computed <em>recursively from the layer above</em>. Two
observations do all the work:</p>
<p><strong>(a) Weight gradients come free once you have δ.</strong> Since
\(z^{[l]}_i = \sum_j W^{[l]}_{ij} a^{[l-1]}_j + b^{[l]}_i\):</p>
<div class="math-box">
$$\frac{\partial J}{\partial W^{[l]}_{ij}} = \delta^{[l]}_i \, a^{[l-1]}_j
\qquad
\frac{\partial J}{\partial b^{[l]}_i} = \delta^{[l]}_i$$
</div>
<p><strong>(b) δ flows backwards through the weights.</strong> A unit in layer \(l\)
influences the loss only via all the units of layer \(l+1\) it feeds. Chain rule,
summing over those paths:</p>
<div class="math-box">
$$\delta^{[l]}_j = \Big( \sum_i W^{[l+1]}_{ij}\, \delta^{[l+1]}_i \Big) \cdot g'\big(z^{[l]}_j\big)$$
</div>
<p>In words: take the errors of the next layer, send them backwards through the
same weights that sent activations forward (that's the transpose), and gate by
the local activation slope. If a ReLU unit was off (\(g' = 0\)), no blame flows
through it - the backward pass respects the forward routing.</p>

<h2>Step 3: the full algorithm in four matrix equations</h2>
<div class="math-box">
$$\text{(output layer)}\quad
\boldsymbol{\delta}^{[L]} = \hat{\mathbf{y}} - \mathbf{y}
\qquad\text{(softmax or sigmoid + cross-entropy)}$$
$$\text{(recurse down)}\quad
\boldsymbol{\delta}^{[l]} = \big(\mathbf{W}^{[l+1]\top} \boldsymbol{\delta}^{[l+1]}\big) \odot g'\big(\mathbf{z}^{[l]}\big)$$
$$\text{(gradients)}\quad
\frac{\partial J}{\partial \mathbf{W}^{[l]}} = \boldsymbol{\delta}^{[l]} \, \mathbf{a}^{[l-1]\top}
\qquad
\frac{\partial J}{\partial \mathbf{b}^{[l]}} = \boldsymbol{\delta}^{[l]}$$
</div>
<p>(⊙ = element-wise product.) Now the caching from
<a href="#/dl/forward-propagation">Forward Propagation</a> pays off: the formulas
consume exactly the stored \(\mathbf{z}^{[l]}\) and \(\mathbf{a}^{[l]}\). One forward
sweep computing and caching, one backward sweep consuming - then a gradient
descent step, and repeat.</p>
<div class="diagram">
<svg viewBox="0 0 660 230" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Forward pass arrows left to right through three layers; backward pass arrows right to left carrying delta errors from the loss">
  <defs>
    <marker id="bpf" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M0,0 L10,5 L0,10 z" style="fill: var(--text-muted)"/>
    </marker>
    <marker id="bpb" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M0,0 L10,5 L0,10 z" style="fill: var(--accent)"/>
    </marker>
  </defs>
  <rect x="30" y="80" width="100" height="50" rx="8" class="d-box"/>
  <text x="80" y="110" text-anchor="middle" class="d-text">layer 1</text>
  <rect x="205" y="80" width="100" height="50" rx="8" class="d-box"/>
  <text x="255" y="110" text-anchor="middle" class="d-text">layer 2</text>
  <rect x="380" y="80" width="100" height="50" rx="8" class="d-box"/>
  <text x="430" y="110" text-anchor="middle" class="d-text">output</text>
  <rect x="545" y="80" width="80" height="50" rx="8" class="d-box-soft"/>
  <text x="585" y="110" text-anchor="middle" class="d-text">loss J</text>
  <!-- forward arrows (top) -->
  <line x1="130" y1="90" x2="203" y2="90" class="d-line" marker-end="url(#bpf)"/>
  <line x1="305" y1="90" x2="378" y2="90" class="d-line" marker-end="url(#bpf)"/>
  <line x1="480" y1="90" x2="543" y2="90" class="d-line" marker-end="url(#bpf)"/>
  <text x="300" y="60" text-anchor="middle" class="d-text-sm">forward: compute a¹, a², ŷ - cache everything</text>
  <!-- backward arrows (bottom) -->
  <line x1="543" y1="122" x2="482" y2="122" class="d-line-accent" marker-end="url(#bpb)"/>
  <line x1="378" y1="122" x2="307" y2="122" class="d-line-accent" marker-end="url(#bpb)"/>
  <line x1="203" y1="122" x2="132" y2="122" class="d-line-accent" marker-end="url(#bpb)"/>
  <text x="300" y="165" text-anchor="middle" class="d-text-accent">backward: δ³ = ŷ − y → δ² = W³ᵀδ³ ⊙ g′(z²) → δ¹ …</text>
  <text x="330" y="200" text-anchor="middle" class="d-text-sm">at each layer, gradients = δ (from the right) × cached activations (from the left)</text>
</svg>
<div class="caption">One pass forward, one pass backward. Every weight's gradient is
"error arriving from above × input that came from below."</div>
</div>

<h2>Step 4: the same numbers, by hand</h2>
<p>Finish the worked example from Forward Propagation
(x=(1,2), a¹=(0, 1.4), ŷ=0.257, y=1, ReLU hidden, sigmoid output):</p>
<div class="math-box">
$$\delta^{[2]} = \hat{y} - y = -0.743$$
$$\frac{\partial J}{\partial \mathbf{W}^{[2]}} = \delta^{[2]}\mathbf{a}^{[1]\top} = (-0.743)(0,\; 1.4) = (0,\; -1.04)$$
$$\boldsymbol{\delta}^{[1]} = \mathbf{W}^{[2]\top}\delta^{[2]} \odot g'(\mathbf{z}^{[1]})
= \begin{pmatrix} 0.6 \\ -0.9 \end{pmatrix}(-0.743) \odot \begin{pmatrix} 0 \\ 1 \end{pmatrix}
= \begin{pmatrix} 0 \\ 0.669 \end{pmatrix}$$
$$\frac{\partial J}{\partial \mathbf{W}^{[1]}} = \boldsymbol{\delta}^{[1]} \mathbf{x}^\top
= \begin{pmatrix} 0 &amp; 0 \\ 0.669 &amp; 1.337 \end{pmatrix}$$
</div>
<p>Read the zeros: hidden unit 1 was inactive (ReLU gave 0), so it receives no
blame and its weights don't move - sensible, since it contributed nothing.
Every nonzero gradient points the right way: e.g. \(W^{[2]}_2\) gets gradient
−1.04, so the update <em>increases</em> that (negative) weight toward making ŷ
larger - toward the true label 1. The arithmetic is mechanical; the sense it
makes is the beauty.</p>

<h2>In code (and how to trust your math)</h2>
<pre><code class="language-python">import numpy as np

def sigmoid(z): return 1 / (1 + np.exp(-z))

# tiny network, same as the worked example
W1 = np.array([[0.5, -0.3], [0.2, 0.8]]); b1 = np.array([0.1, -0.4])
W2 = np.array([[0.6, -0.9]]);             b2 = np.array([0.2])
x  = np.array([1.0, 2.0]); y = 1.0

# forward (with cache)
z1 = W1 @ x + b1;  a1 = np.maximum(0, z1)
z2 = W2 @ a1 + b2; y_hat = sigmoid(z2)

# backward - the four equations, verbatim
d2  = y_hat - y                       # δ² : (1,)
dW2 = np.outer(d2, a1)                # δ² a¹ᵀ
db2 = d2
d1  = (W2.T @ d2) * (z1 &gt; 0)          # W²ᵀδ² ⊙ ReLU′(z¹)
dW1 = np.outer(d1, x)
db1 = d1

print("dW2:", dW2.round(3))           # [[ 0.    -1.04 ]]
print("dW1:\n", dW1.round(3))         # [[0. 0.], [0.669 1.337]]

# --- gradient checking: numerically verify one entry --------------------
def loss_fn(W1_):
    a1_ = np.maximum(0, W1_ @ x + b1)
    p   = sigmoid(W2 @ a1_ + b2)
    return float(-(y*np.log(p) + (1-y)*np.log(1-p)))

eps = 1e-6
W1p = W1.copy(); W1p[1, 0] += eps
W1m = W1.copy(); W1m[1, 0] -= eps
numeric = (loss_fn(W1p) - loss_fn(W1m)) / (2 * eps)
print("numeric:", round(numeric, 4), "| analytic:", round(dW1[1, 0], 4))  # match!
</code></pre>
<p>That last block - <strong>gradient checking</strong> with the central difference
from <a href="#/math/calculus">Calculus</a> - is how you debug any hand-written
backward pass. Frameworks (PyTorch autograd) automate the entire backward sweep by
recording the forward graph and applying these same local rules; you'll see it in
<a href="#/dl/nn-in-pytorch">The Same Network in PyTorch</a>.</p>

<h2>Common interview questions</h2>
<div class="qa">
  <div class="qa-q">Q1. Why is backprop efficient compared to computing each weight's gradient independently?</div>
  <div class="qa-a"><p>Naively, each of N weights needs its own perturbed forward pass: O(N) passes.
  Backprop shares the common sub-chains: each layer's δ is computed once and reused for all
  that layer's weight gradients, giving all N derivatives in one backward sweep - O(1)
  passes. It's dynamic programming over the chain rule.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q2. What role does g′(z) play in the backward pass?</div>
  <div class="qa-a"><p>It gates the error flow: δ is multiplied element-wise by the local activation slope.
  Saturated sigmoid units (slope ≈ 0) and inactive ReLUs (slope 0) block blame from passing -
  which is precisely the mechanism behind <a href="#/dl/vanishing-gradients">vanishing
  gradients</a> and dying ReLUs.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q3. Why does the backward pass use Wᵀ?</div>
  <div class="qa-a"><p>Forward, W maps layer l's activations to layer l+1's units; backward, blame must
  travel the same wiring in reverse - each unit collects error from every unit it fed,
  weighted by the same connection strengths. Mathematically it's the Jacobian of the linear
  map, which is Wᵀ.</p></div>
</div>

<h2>Quick quiz</h2>
<div class="quiz">
  <div class="quiz-q" data-answer="1">
    <p class="quiz-question">1. δ² arrives at a hidden layer whose ReLU unit was inactive (z &lt; 0). The error flowing through that unit is…</p>
    <button class="quiz-opt">Unchanged</button>
    <button class="quiz-opt">Zero - g′(z) = 0 gates it off</button>
    <button class="quiz-opt">Negated</button>
    <div class="quiz-explain hidden">The ⊙ g′(z) factor zeroes blame for units that didn't participate in the forward computation. No contribution, no correction.</div>
  </div>
  <div class="quiz-q" data-answer="0">
    <p class="quiz-question">2. The gradient of a weight W_ij equals…</p>
    <button class="quiz-opt">(error signal of the unit it feeds) × (activation of the unit it comes from)</button>
    <button class="quiz-opt">The weight's own value</button>
    <button class="quiz-opt">The loss divided by the number of weights</button>
    <div class="quiz-explain hidden">∂J/∂W_ij = δ_i · a_j - blame from above times input from below. Memorize this shape; it recurs in every architecture.</div>
  </div>
  <div class="quiz-q" data-answer="2">
    <p class="quiz-question">3. Your hand-coded backward pass disagrees with the numerical gradient. Which is wrong?</p>
    <button class="quiz-opt">The numerical one - floating point noise</button>
    <button class="quiz-opt">Neither; disagreement is normal</button>
    <button class="quiz-opt">Almost certainly your analytic code - central differences are the arbiter</button>
    <div class="quiz-explain hidden">Central-difference checks are accurate to ~1e-8 relative error. A real mismatch (>1e-4) means a bug in the backward math - usually a missing transpose or activation gate.</div>
  </div>
</div>
`};

CONTENT["dl/gradient-descent-variants"] = {
  html: String.raw`
<h1>Gradient Descent Variants</h1>
<p class="lead">One decision hides inside "compute the gradient": on <em>how much
data</em>? All of it, one example, or a small batch? The answer changes speed,
memory, noise - and even the quality of the minimum you find.</p>

<h2>The three variants</h2>
<h3>Batch gradient descent: the whole dataset per step</h3>
<div class="math-box">
$$\mathbf{w} \leftarrow \mathbf{w} - \alpha \,\nabla J(\mathbf{w}; \text{all } m \text{ examples})$$
</div>
<p>The gradient is exact, the descent path is smooth, convergence proofs apply.
But one step costs a full pass over the data - with 10M examples you take one
step per 10M-example computation, and the whole dataset must fit in memory.
Fine for the small classical problems of Module 6; a non-starter for deep learning.</p>
<h3>Stochastic gradient descent (SGD): one example per step</h3>
<p>The opposite extreme: estimate the gradient from a single random example.
Steps are nearly free but wildly noisy - each example pulls toward what's best
<em>for it</em>. The path staggers drunkenly toward the minimum and then rattles
around it, never settling, unless the learning rate is decayed.</p>
<h3>Mini-batch gradient descent: the universal compromise</h3>
<div class="math-box">
$$\mathbf{w} \leftarrow \mathbf{w} - \alpha \,\nabla J(\mathbf{w}; \text{next } B \text{ examples}),
\qquad B \in [16, 512] \text{ typically}$$
</div>
<p>Average the gradient over a small batch: noise shrinks like
\(1/\sqrt{B}\) (<a href="#/math/confidence-intervals">standard-error logic</a>),
while the matrix math stays big enough to saturate a GPU
(<a href="#/dl/forward-propagation">the batching payoff</a>). This is what
everyone means by "SGD" in deep learning practice.</p>
<div class="diagram">
<svg viewBox="0 0 660 250" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Three descent paths on contour plots: batch smooth and direct, mini-batch slightly wobbly, stochastic very erratic">
  <!-- batch -->
  <rect x="15" y="25" width="200" height="185" rx="8" class="d-box-muted"/>
  <text x="115" y="47" text-anchor="middle" class="d-text">batch</text>
  <ellipse cx="115" cy="130" rx="75" ry="48" class="d-grid" fill="none"/>
  <ellipse cx="115" cy="130" rx="42" ry="26" class="d-grid" fill="none"/>
  <circle cx="115" cy="130" r="4" class="d-dot"/>
  <polyline points="35,195 70,168 95,150 108,138 115,130" class="d-line-accent" fill="none"/>
  <text x="115" y="200" text-anchor="middle" class="d-text-sm">smooth, expensive steps</text>
  <!-- minibatch -->
  <rect x="230" y="25" width="200" height="185" rx="8" class="d-box-muted"/>
  <text x="330" y="47" text-anchor="middle" class="d-text">mini-batch</text>
  <ellipse cx="330" cy="130" rx="75" ry="48" class="d-grid" fill="none"/>
  <ellipse cx="330" cy="130" rx="42" ry="26" class="d-grid" fill="none"/>
  <circle cx="330" cy="130" r="4" class="d-dot"/>
  <polyline points="250,195 285,160 305,168 318,140 325,145 330,132" class="d-line-accent" fill="none"/>
  <text x="330" y="200" text-anchor="middle" class="d-text-sm">slightly noisy, cheap, GPU-friendly</text>
  <!-- sgd -->
  <rect x="445" y="25" width="200" height="185" rx="8" class="d-box-muted"/>
  <text x="545" y="47" text-anchor="middle" class="d-text">stochastic (B = 1)</text>
  <ellipse cx="545" cy="130" rx="75" ry="48" class="d-grid" fill="none"/>
  <ellipse cx="545" cy="130" rx="42" ry="26" class="d-grid" fill="none"/>
  <circle cx="545" cy="130" r="4" class="d-dot"/>
  <polyline points="465,195 500,140 480,120 520,155 505,100 545,145 530,118 552,138 542,125"
            class="d-line-accent" fill="none"/>
  <text x="545" y="200" text-anchor="middle" class="d-text-sm">staggers; rattles around the minimum</text>
</svg>
<div class="caption">Same bowl, three data-per-step choices. Mini-batch keeps most of
the speed and most of the stability.</div>
</div>

<h2>Vocabulary that interviews love</h2>
<ul>
  <li><strong>Epoch:</strong> one full pass through the training set.</li>
  <li><strong>Iteration/step:</strong> one weight update. With m = 50,000 and B = 128,
      one epoch = 391 iterations.</li>
  <li><strong>Shuffling:</strong> reshuffle example order every epoch - otherwise the
      model sees the same batch sequence (and any ordering bias, e.g. sorted labels)
      repeatedly.</li>
  <li><strong>Learning-rate schedules:</strong> noise near the minimum shrinks if steps
      do. Common: step decay, cosine decay, and warmup (start small to survive the
      chaotic first steps, then rise). Any serious training run uses one.</li>
</ul>

<h2>The surprise: noise is a feature</h2>
<p>Mini-batch noise isn't just tolerated - it helps. Deep loss surfaces are full of
saddle points and narrow crevices
(<a href="#/math/calculus">where gradients vanish</a>); noisy gradients jiggle the
weights off saddles and out of sharp, brittle minima toward wide, flat ones -
which empirically generalize better. Pure batch descent, with its perfect
gradients, can park itself in exactly the sharp minima you don't want.
Batch size thus tunes an implicit regularizer: very large batches often need
learning-rate and schedule tricks to match small-batch generalization.</p>

<h2>In code</h2>
<pre><code class="language-python">import numpy as np

rng = np.random.default_rng(0)
m, n = 10_000, 20
X = rng.normal(size=(m, n))
true_w = rng.normal(size=n)
y = X @ true_w + rng.normal(0, 0.5, m)

def mse_grad(w, Xb, yb):
    return 2 * Xb.T @ (Xb @ w - yb) / len(yb)

def train(batch_size, alpha=0.05, epochs=5):
    w = np.zeros(n)
    for epoch in range(epochs):
        idx = rng.permutation(m)                 # shuffle each epoch!
        for start in range(0, m, batch_size):
            batch = idx[start:start + batch_size]
            w -= alpha * mse_grad(w, X[batch], y[batch])
    return np.linalg.norm(w - true_w)

for B in [m, 256, 1]:                            # batch, mini-batch, SGD
    err = train(B)
    steps = 5 * (m // B if B &gt; 1 else m)
    name = {m: "batch", 256: "mini-batch", 1: "SGD"}[B]
    print(f"{name:10s}  B={B:6d}  steps={steps:6d}  |w - w*| = {err:.4f}")
# mini-batch reaches comparable error with far better hardware efficiency
</code></pre>

<h2>Choosing the batch size</h2>
<table>
  <tr><th>Batch size</th><th>Gradient quality</th><th>Hardware</th><th>Generalization</th></tr>
  <tr><td>Full (batch GD)</td><td>exact</td><td>memory-bound, slow steps</td><td>can settle in sharp minima</td></tr>
  <tr><td>256–4096 (large)</td><td>very clean</td><td>excellent GPU utilization</td><td>may need LR scaling + warmup</td></tr>
  <tr><td>32–128 (standard)</td><td>usefully noisy</td><td>good</td><td>the safe default</td></tr>
  <tr><td>1 (pure SGD)</td><td>chaotic</td><td>terrible utilization</td><td>works, but obsolete in practice</td></tr>
</table>
<p>Practical default: the largest power of two that fits your GPU memory, capped
around a few hundred - then tune the learning rate to match. Which learning-rate
machinery to bolt on top (momentum, Adam) is the
<a href="#/dl/optimizers">next chapter</a>.</p>

<h2>Common interview questions</h2>
<div class="qa">
  <div class="qa-q">Q1. Batch vs stochastic vs mini-batch - trade-offs?</div>
  <div class="qa-a"><p>Batch: exact gradient, smooth path, one slow step per full pass, heavy memory.
  SGD (B=1): cheapest steps, extreme noise, poor hardware use. Mini-batch: noise averaged
  down by √B, steps map to efficient matrix ops on GPUs - the universal practice. Noise
  also aids escaping saddles/sharp minima.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q2. Define epoch vs iteration, and compute: m=60,000, B=32 - iterations per epoch?</div>
  <div class="qa-a"><p>Epoch = full pass over data; iteration = one parameter update from one batch.
  60,000/32 = 1,875 iterations per epoch.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q3. Why decay the learning rate over training?</div>
  <div class="qa-a"><p>The gradient noise floor keeps mini-batch training rattling around the minimum with
  amplitude ∝ α. Early on, big steps speed progress; late, shrinking α lets the weights
  settle. Schedules (step, cosine, warmup+decay) institutionalize this.</p></div>
</div>

<h2>Quick quiz</h2>
<div class="quiz">
  <div class="quiz-q" data-answer="1">
    <p class="quiz-question">1. m = 12,800 examples, batch size 128, 10 epochs. Total weight updates?</p>
    <button class="quiz-opt">128</button>
    <button class="quiz-opt">1,000</button>
    <button class="quiz-opt">12,800</button>
    <div class="quiz-explain hidden">100 iterations per epoch × 10 epochs = 1,000 updates.</div>
  </div>
  <div class="quiz-q" data-answer="2">
    <p class="quiz-question">2. Quadrupling the batch size reduces gradient noise by a factor of about…</p>
    <button class="quiz-opt">4</button>
    <button class="quiz-opt">16</button>
    <button class="quiz-opt">2 - noise scales as 1/√B</button>
    <div class="quiz-explain hidden">Averaging B samples shrinks the standard error by √B - the same √n law as confidence intervals. Diminishing returns are built in.</div>
  </div>
  <div class="quiz-q" data-answer="0">
    <p class="quiz-question">3. A benefit of gradient noise in deep networks is…</p>
    <button class="quiz-opt">Escaping saddle points and sharp minima, often improving generalization</button>
    <button class="quiz-opt">Lower memory usage per step</button>
    <button class="quiz-opt">Exact convergence guarantees</button>
    <div class="quiz-explain hidden">The jiggle is an implicit regularizer - one reason "worse" gradient estimates can produce better final models than perfect ones.</div>
  </div>
</div>
`};

CONTENT["dl/optimizers"] = {
  html: String.raw`
<h1>Optimizers (SGD, Momentum, RMSProp, Adam, AdamW)</h1>
<p class="lead">Plain gradient descent treats every step as its first - no memory,
no adaptation. Modern optimizers add two upgrades: <em>remember where you've been
heading</em> (momentum) and <em>scale each weight's step to its own terrain</em>
(adaptivity). Adam is simply both at once.</p>

<h2>The problem with plain SGD</h2>
<p>Loss surfaces are ravines: steep across, shallow along. Plain SGD bounces
between the steep walls while crawling along the valley floor - the zigzag you
saw in the <a href="#/math/calculus">contour diagram</a>. And with noisy mini-batch
gradients (<a href="#/dl/gradient-descent-variants">last chapter</a>), each step
also jitters randomly. Two ideas fix the two problems.</p>

<h2>Momentum: a heavy ball, not a cautious walker</h2>
<div class="math-box">
$$\mathbf{v} \leftarrow \beta\, \mathbf{v} + \nabla J(\mathbf{w})
\qquad
\mathbf{w} \leftarrow \mathbf{w} - \alpha\, \mathbf{v}
\qquad (\beta \approx 0.9)$$
</div>
<p>Keep an exponentially-decaying running sum of past gradients and step along
<em>that</em>. Two effects: components that keep pointing the same way (along the
ravine) accumulate - up to a \(1/(1-\beta) = 10\times\) amplification - while
components that alternate signs (across the ravine) cancel. The ball barrels down
the valley, plows through small bumps and saddle points, and smooths out
mini-batch noise for free. (Nesterov momentum is a refinement that looks ahead
before accumulating; same idea sharpened.)</p>

<h2>RMSProp: a personal learning rate for every weight</h2>
<div class="math-box">
$$\mathbf{s} \leftarrow \rho\, \mathbf{s} + (1-\rho)\, (\nabla J)^2
\qquad
\mathbf{w} \leftarrow \mathbf{w} - \frac{\alpha}{\sqrt{\mathbf{s}} + \epsilon} \,\nabla J$$
</div>
<p>Track each weight's <em>recent squared gradient</em> \(\mathbf{s}\) (element-wise),
and divide its step by \(\sqrt{s}\). Weights with chronically huge gradients get
throttled; weights with tiny gradients (rare features, flat directions) get
boosted. Every weight now moves at a sensible pace regardless of how badly scaled
its input is - the optimizer version of
<a href="#/features/scaling">feature scaling</a>.</p>

<h2>Adam: momentum + RMSProp, with a correction</h2>
<div class="math-box">
$$\mathbf{m} \leftarrow \beta_1 \mathbf{m} + (1-\beta_1)\nabla J
\qquad
\mathbf{v} \leftarrow \beta_2 \mathbf{v} + (1-\beta_2)(\nabla J)^2$$
$$\hat{\mathbf{m}} = \frac{\mathbf{m}}{1-\beta_1^t}, \quad
\hat{\mathbf{v}} = \frac{\mathbf{v}}{1-\beta_2^t}
\qquad
\mathbf{w} \leftarrow \mathbf{w} - \alpha \frac{\hat{\mathbf{m}}}{\sqrt{\hat{\mathbf{v}}} + \epsilon}$$
</div>
<p>First moment \(\mathbf{m}\) = momentum (direction memory, β₁ = 0.9). Second moment
\(\mathbf{v}\) = RMSProp (per-weight scaling, β₂ = 0.999). The hats are
<strong>bias correction</strong>: both accumulators start at zero and would
underestimate early in training; dividing by \(1-\beta^t\) fixes the cold start
(at t = 1 with β₂ = 0.999, the raw v is 1000× too small - without correction the
first steps would be huge). Defaults (α = 0.001, 0.9, 0.999) work shockingly often;
that robustness is why Adam is the de-facto default.</p>
<h3>AdamW: fixing weight decay</h3>
<p>Classic "L2 regularization" added to the gradient gets divided by
\(\sqrt{\hat{v}}\) inside Adam - so heavily-updated weights receive <em>weaker</em>
regularization, muddling the two mechanisms. AdamW <strong>decouples</strong> decay
from the gradient path: \(\mathbf{w} \leftarrow \mathbf{w} - \alpha(\ldots) -
\alpha\lambda\mathbf{w}\), applied directly. Cleaner, and the standard for training
Transformers (<a href="#/transformers/architecture">Module 10</a>).</p>

<h2>Watching them race</h2>
<pre><code class="language-python">import numpy as np

# a ravine: steep in w2, shallow in w1 (condition number 50)
grad = lambda w: np.array([1.0 * w[0], 50.0 * w[1]])
start = np.array([-8.0, 1.5])

def run(update, steps=120, **state):
    w, path = start.copy(), []
    for t in range(1, steps + 1):
        w = update(w, grad(w), t, state)
        path.append(np.linalg.norm(w))          # distance to minimum (0,0)
    return path[-1]

def sgd(w, g, t, s):        return w - 0.018 * g
def momentum(w, g, t, s):
    s["v"] = 0.9 * s.get("v", 0) + g
    return w - 0.018 * s["v"]
def adam(w, g, t, s):
    s["m"] = 0.9  * s.get("m", 0) + 0.1   * g
    s["v"] = 0.999* s.get("v", 0) + 0.001 * g**2
    mh = s["m"] / (1 - 0.9**t); vh = s["v"] / (1 - 0.999**t)
    return w - 0.3 * mh / (np.sqrt(vh) + 1e-8)

for name, fn in [("SGD", sgd), ("momentum", momentum), ("Adam", adam)]:
    print(f"{name:10s} final distance to minimum: {run(fn):.5f}")
# SGD zigzags (limited by the steep direction); momentum accelerates;
# Adam handles the 50x scale mismatch effortlessly.
</code></pre>

<h2>Which one, when</h2>
<table>
  <tr><th>Optimizer</th><th>Memory</th><th>Character</th><th>Reach for it when</th></tr>
  <tr><td>SGD</td><td>-</td><td>simple, needs careful α + schedule</td><td>baselines, theory</td></tr>
  <tr><td>SGD + momentum</td><td>1× params</td><td>fast, great generalization when tuned</td><td>CNNs/vision (with a good LR schedule, often beats Adam's final accuracy)</td></tr>
  <tr><td>RMSProp</td><td>1×</td><td>adaptive scaling</td><td>legacy RNN work; mostly superseded</td></tr>
  <tr><td><strong>Adam</strong></td><td>2×</td><td>robust defaults, fast progress</td><td><strong>the default first choice</strong></td></tr>
  <tr><td><strong>AdamW</strong></td><td>2×</td><td>Adam + honest weight decay</td><td>Transformers, anything regularized</td></tr>
</table>
<p>The one hyperparameter that still towers over the choice of optimizer:
<strong>the learning rate</strong>. An Adam at lr = 0.1 diverges; an SGD at the right
lr with cosine decay wins benchmarks. Tune α first (log scale -
<a href="#/ml/hyperparameter-tuning">Module 6</a>), optimizer second.</p>

<h2>Common interview questions</h2>
<div class="qa">
  <div class="qa-q">Q1. What two problems does momentum solve?</div>
  <div class="qa-a"><p>Ravine zigzag: oscillating gradient components cancel in the running average while
  consistent ones accumulate - faster progress along the valley. Noise: the exponential
  average smooths mini-batch jitter. It also carries the iterate through saddles and small
  bumps like a heavy ball.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q2. Explain Adam's two moments and why bias correction exists.</div>
  <div class="qa-a"><p>m = exponential average of gradients (direction memory); v = exponential average of
  squared gradients (per-weight step scaling). Both start at 0, so early averages are
  shrunken toward zero; dividing by (1 − βᵗ) rescales them to unbiased estimates -
  without it, v's underestimate would make the first effective steps explode.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q3. Adam vs AdamW?</div>
  <div class="qa-a"><p>Adam applies L2 penalty through the gradient, where the adaptive √v̂ denominator
  distorts it per-weight. AdamW applies weight decay directly to the weights, decoupled from
  the adaptive machinery - mathematically cleaner regularization and consistently better in
  practice; it's the Transformer-era standard.</p></div>
</div>

<h2>Quick quiz</h2>
<div class="quiz">
  <div class="quiz-q" data-answer="2">
    <p class="quiz-question">1. In a narrow ravine, momentum speeds convergence because…</p>
    <button class="quiz-opt">It increases the learning rate everywhere</button>
    <button class="quiz-opt">It computes exact gradients</button>
    <button class="quiz-opt">Oscillating (cross-ravine) components cancel while consistent (along-ravine) ones accumulate</button>
    <div class="quiz-explain hidden">The exponential average is a sign-sensitive filter: alternating +/− kills itself; steady direction compounds up to 1/(1−β)×.</div>
  </div>
  <div class="quiz-q" data-answer="1">
    <p class="quiz-question">2. Adam's per-weight step size is proportional to…</p>
    <button class="quiz-opt">The weight's magnitude</button>
    <button class="quiz-opt">m̂ / (√v̂ + ε) - direction memory scaled by recent gradient size</button>
    <button class="quiz-opt">The batch size</button>
    <div class="quiz-explain hidden">Momentum in the numerator, RMS scaling in the denominator: each weight gets a step suited to its own gradient history.</div>
  </div>
  <div class="quiz-q" data-answer="0">
    <p class="quiz-question">3. Training diverges immediately with Adam at lr = 0.1. First move?</p>
    <button class="quiz-opt">Drop the learning rate ~100× toward the 1e-3 default</button>
    <button class="quiz-opt">Switch to plain SGD at the same rate</button>
    <button class="quiz-opt">Increase β₂</button>
    <div class="quiz-explain hidden">No optimizer survives a wildly wrong α. Adam's default 1e-3 exists for a reason; tune it on a log scale before touching anything else.</div>
  </div>
</div>
`};

CONTENT["dl/weight-initialization"] = {
  html: String.raw`
<h1>Weight Initialization</h1>
<p class="lead">Where you <em>start</em> the weights decides whether signals and
gradients survive the trip through a deep network - or quietly die before
training even begins. Two named recipes (Xavier, He) solve it with one variance
calculation.</p>

<h2>Two ways to start wrong</h2>
<h3>All zeros: the symmetry trap</h3>
<p>Initialize every weight to 0 and all neurons in a layer compute the identical
output, receive the identical gradient, and take the identical update - forever.
A 512-unit layer becomes one neuron photocopied 512 times. Randomness isn't a
nicety; it's what <strong>breaks the symmetry</strong> so different units can learn
different features. (Biases can be zero - the weights already differ.)</p>
<h3>Wrong scale: exponential death or explosion</h3>
<p>Random, but how big? Each layer roughly multiplies its input's variance by
\(n \cdot \mathrm{Var}(w)\), where \(n\) is the fan-in. If that factor is 0.5 per
layer, activations shrink by \(0.5^{20} \approx 10^{-6}\) after 20 layers - signals
vanish. If it's 2, they blow up \(10^6\times\). The same amplification hits
gradients on the way back
(<a href="#/dl/vanishing-gradients">next chapter's disease</a>). The cure is to
aim the factor at exactly 1.</p>

<h2>The fix: set the variance so signals keep their size</h2>
<p>For a linear layer, \(\mathrm{Var}(z) = n_{in} \mathrm{Var}(w)\mathrm{Var}(x)\)
(independent, zero-mean). Demanding \(\mathrm{Var}(z) = \mathrm{Var}(x)\) gives
\(\mathrm{Var}(w) = 1/n_{in}\) - then adjust for the activation's effect:</p>
<div class="math-box">
$$\textbf{Xavier/Glorot} \; (\tanh, \sigma):\quad
\mathrm{Var}(w) = \frac{2}{n_{in} + n_{out}}$$
$$\textbf{He/Kaiming} \; (\text{ReLU family}):\quad
\mathrm{Var}(w) = \frac{2}{n_{in}}$$
</div>
<p>The He factor of 2 has a clean reason: ReLU zeroes the negative half of every
distribution, discarding half the variance - so you start with twice as much.
The rule of thumb is exactly that simple: <strong>tanh/sigmoid → Xavier,
ReLU → He</strong>. Frameworks default sensibly (PyTorch uses Kaiming-style init
for Linear/Conv layers), but you should know what's underneath - especially when
a custom layer trains mysteriously badly.</p>

<h2>Watch the difference</h2>
<pre><code class="language-python">import numpy as np

rng = np.random.default_rng(0)
n, depth = 512, 30
x = rng.normal(size=(1000, n))

for name, std in [("too small (0.01)", 0.01),
                  ("He  (√(2/n))    ", np.sqrt(2 / n)),
                  ("too big (0.3)   ", 0.3)]:
    a = x.copy()
    for _ in range(depth):                       # 30 ReLU layers
        W = rng.normal(0, std, (n, n))
        a = np.maximum(0, a @ W)
    print(f"{name}  activation std after {depth} layers: {a.std():.3e}")

# too small: ~1e-40  - signal annihilated (and so will gradients be)
# He:        ~1e+00  - alive and well at depth 30
# too big:   ~1e+20  - numeric explosion
</code></pre>
<p>One number (the init std) separates a trainable 30-layer network from a dead or
exploding one. This experiment is worth actually running once - it makes the
next chapter visceral.</p>

<h2>Beyond the basic recipes</h2>
<ul>
  <li><strong>Uniform vs normal:</strong> either works if the variance matches the
      formula (the uniform bound is \(\pm\sqrt{3\,\mathrm{Var}}\)).</li>
  <li><strong>Output-layer biases:</strong> initialize to match the base rate (e.g.
      log-odds of the positive class) so the first predictions aren't absurd.</li>
  <li><strong>Modern architectures cheat:</strong>
      <a href="#/dl/dl-regularization">Batch normalization</a> re-standardizes
      between layers, and residual connections (Module 8's ResNet) give signals a
      bypass highway - both reduce (not eliminate) initialization sensitivity.</li>
  <li><strong>Transfer learning is initialization:</strong> starting from pretrained
      weights (<a href="#/transformers/transfer-learning">Module 10</a>) is the
      ultimate init - you begin at a good minimum's doorstep instead of a random
      point.</li>
</ul>

<h2>Common interview questions</h2>
<div class="qa">
  <div class="qa-q">Q1. Why can't we initialize all weights to the same value?</div>
  <div class="qa-a"><p>Identical weights → identical activations → identical gradients → identical updates:
  all neurons in a layer remain clones for the entire training run (the symmetry problem).
  Random initialization gives each unit a distinct starting point so gradients can
  differentiate them.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q2. Derive why He init uses 2/n_in.</div>
  <div class="qa-a"><p>For z = Σwᵢxᵢ with independent zero-mean terms, Var(z) = n·Var(w)·Var(x). ReLU
  halves the variance of its input (negative half → 0). Requiring the post-ReLU variance to
  stay constant: n·Var(w)·½ = 1 → Var(w) = 2/n_in.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q3. A deep custom network trains fine at 5 layers but not at 40. Initialization-related causes?</div>
  <div class="qa-a"><p>Per-layer variance amplification ≠ 1 compounds exponentially with depth: activations
  or gradients vanish/explode by layer 40 while 5 layers tolerated the mismatch. Check
  activation stds layer-by-layer; fix with He/Xavier matching the activation, batch norm,
  or residual connections.</p></div>
</div>

<h2>Quick quiz</h2>
<div class="quiz">
  <div class="quiz-q" data-answer="1">
    <p class="quiz-question">1. A layer initialized with all weights = 0.5 (identical) will…</p>
    <button class="quiz-opt">Train slower but fine</button>
    <button class="quiz-opt">Keep every neuron identical forever - the layer acts as one unit</button>
    <button class="quiz-opt">Explode</button>
    <div class="quiz-explain hidden">Symmetry is never broken: same inputs, same outputs, same gradients, same updates. Scale wasn't the issue - sameness was.</div>
  </div>
  <div class="quiz-q" data-answer="0">
    <p class="quiz-question">2. For a ReLU layer with 800 inputs, He initialization draws weights with std…</p>
    <button class="quiz-opt">√(2/800) = 0.05</button>
    <button class="quiz-opt">2/800 = 0.0025</button>
    <button class="quiz-opt">0.5</button>
    <div class="quiz-explain hidden">Var(w) = 2/n_in = 0.0025, so std = √0.0025 = 0.05. The factor 2 compensates ReLU's discarded negative half.</div>
  </div>
  <div class="quiz-q" data-answer="2">
    <p class="quiz-question">3. Activations shrink 10× per layer through your 25-layer net at init. By the last layer they're roughly…</p>
    <button class="quiz-opt">250× smaller</button>
    <button class="quiz-opt">25× smaller</button>
    <button class="quiz-opt">10²⁵× smaller - effectively zero</button>
    <div class="quiz-explain hidden">Per-layer factors multiply, so mismatches compound exponentially with depth. This is why initialization is a make-or-break detail for deep nets.</div>
  </div>
</div>
`};
