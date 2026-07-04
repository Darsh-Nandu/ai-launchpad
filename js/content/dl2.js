/* Module 7 - Deep Learning Foundations (part 2) */

CONTENT["dl/mlp"] = {
  html: String.raw`
<h1>Multi-Layer Perceptron / ANN</h1>
<p class="lead">One perceptron draws one straight line - and XOR broke it.
The fix: stack perceptrons in layers, let early layers invent new features,
and let later layers combine them. That stack is the Multi-Layer Perceptron,
and it can approximate <em>any</em> function.</p>

<h2>Solving XOR by hand: the proof of concept</h2>
<p>Recall from the <a href="#/dl/perceptron">Perceptron chapter</a>: no single line
separates XOR's diagonal classes. But watch what two lines can do. Build two hidden
neurons: \(h_1\) fires for "x₁ OR x₂" and \(h_2\) fires for "x₁ AND x₂". In the new
coordinate system \((h_1, h_2)\), the four XOR points become: (0,0)→(0,0),
(0,1)→(1,0), (1,0)→(1,0), (1,1)→(1,1). Now XOR is just
"\(h_1\) AND NOT \(h_2\)" - <strong>linearly separable</strong>. An output neuron
finishes the job.</p>
<p>That's the whole philosophy of deep learning in one example: <strong>hidden layers
re-represent the input until the problem becomes easy.</strong> You hand-crafted
features in Module 5; a neural network learns them.</p>

<h2>The architecture</h2>
<div class="diagram">
<svg viewBox="0 0 660 320" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="MLP with 3 input nodes, two hidden layers of 4 nodes, and 2 output nodes, fully connected">
  <!-- input layer -->
  <circle cx="80" cy="100" r="20" class="d-box-soft"/><text x="80" y="105" text-anchor="middle" class="d-text-sm">x₁</text>
  <circle cx="80" cy="160" r="20" class="d-box-soft"/><text x="80" y="165" text-anchor="middle" class="d-text-sm">x₂</text>
  <circle cx="80" cy="220" r="20" class="d-box-soft"/><text x="80" y="225" text-anchor="middle" class="d-text-sm">x₃</text>
  <!-- hidden 1 -->
  <circle cx="260" cy="70" r="20" class="d-box"/><circle cx="260" cy="130" r="20" class="d-box"/>
  <circle cx="260" cy="190" r="20" class="d-box"/><circle cx="260" cy="250" r="20" class="d-box"/>
  <!-- hidden 2 -->
  <circle cx="440" cy="70" r="20" class="d-box"/><circle cx="440" cy="130" r="20" class="d-box"/>
  <circle cx="440" cy="190" r="20" class="d-box"/><circle cx="440" cy="250" r="20" class="d-box"/>
  <!-- output -->
  <circle cx="600" cy="130" r="20" class="d-box-soft"/><text x="600" y="135" text-anchor="middle" class="d-text-sm">ŷ₁</text>
  <circle cx="600" cy="190" r="20" class="d-box-soft"/><text x="600" y="195" text-anchor="middle" class="d-text-sm">ŷ₂</text>
  <!-- connections input->h1 -->
  <g class="d-grid">
    <line x1="100" y1="100" x2="240" y2="70"/><line x1="100" y1="100" x2="240" y2="130"/>
    <line x1="100" y1="100" x2="240" y2="190"/><line x1="100" y1="100" x2="240" y2="250"/>
    <line x1="100" y1="160" x2="240" y2="70"/><line x1="100" y1="160" x2="240" y2="130"/>
    <line x1="100" y1="160" x2="240" y2="190"/><line x1="100" y1="160" x2="240" y2="250"/>
    <line x1="100" y1="220" x2="240" y2="70"/><line x1="100" y1="220" x2="240" y2="130"/>
    <line x1="100" y1="220" x2="240" y2="190"/><line x1="100" y1="220" x2="240" y2="250"/>
    <line x1="280" y1="70" x2="420" y2="70"/><line x1="280" y1="70" x2="420" y2="130"/>
    <line x1="280" y1="70" x2="420" y2="190"/><line x1="280" y1="70" x2="420" y2="250"/>
    <line x1="280" y1="130" x2="420" y2="70"/><line x1="280" y1="130" x2="420" y2="130"/>
    <line x1="280" y1="130" x2="420" y2="190"/><line x1="280" y1="130" x2="420" y2="250"/>
    <line x1="280" y1="190" x2="420" y2="70"/><line x1="280" y1="190" x2="420" y2="130"/>
    <line x1="280" y1="190" x2="420" y2="190"/><line x1="280" y1="190" x2="420" y2="250"/>
    <line x1="280" y1="250" x2="420" y2="70"/><line x1="280" y1="250" x2="420" y2="130"/>
    <line x1="280" y1="250" x2="420" y2="190"/><line x1="280" y1="250" x2="420" y2="250"/>
    <line x1="460" y1="70" x2="580" y2="130"/><line x1="460" y1="70" x2="580" y2="190"/>
    <line x1="460" y1="130" x2="580" y2="130"/><line x1="460" y1="130" x2="580" y2="190"/>
    <line x1="460" y1="190" x2="580" y2="130"/><line x1="460" y1="190" x2="580" y2="190"/>
    <line x1="460" y1="250" x2="580" y2="130"/><line x1="460" y1="250" x2="580" y2="190"/>
  </g>
  <text x="80" y="290" text-anchor="middle" class="d-text-sm">input layer</text>
  <text x="260" y="290" text-anchor="middle" class="d-text-sm">hidden layer 1</text>
  <text x="440" y="290" text-anchor="middle" class="d-text-sm">hidden layer 2</text>
  <text x="600" y="290" text-anchor="middle" class="d-text-sm">output layer</text>
  <text x="330" y="35" text-anchor="middle" class="d-text-accent">each arrow = one learnable weight</text>
</svg>
<div class="caption">A fully-connected MLP. "Deep" learning = more than one hidden layer.</div>
</div>
<p>Each layer computes the same two-step move you know from the perceptron -
weighted sum, then activation - but for a whole vector of neurons at once
(the matrix form from <a href="#/math/linear-algebra">Module 1</a>):</p>
<div class="math-box">
$$\mathbf{a}^{[l]} = g\big(\mathbf{W}^{[l]} \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}\big)$$
</div>
<p>where \(\mathbf{a}^{[0]} = \mathbf{x}\), \(\mathbf{W}^{[l]}\) is layer \(l\)'s weight
matrix, and \(g\) is a non-linear activation. Parameter count adds up fast: a
3→4→4→2 network has 3×4+4 + 4×4+4 + 4×2+2 = 46 learnable numbers; GPT-class models
have billions - same equation.</p>

<h2>Why the non-linearity is non-negotiable</h2>
<p>Remove \(g\) and stack as many linear layers as you like:
\(\mathbf{W}^{[2]}(\mathbf{W}^{[1]}\mathbf{x}) = (\mathbf{W}^{[2]}\mathbf{W}^{[1]})\mathbf{x}\) -
still one matrix, still one straight boundary. A 100-layer <em>linear</em> network
is exactly a 1-layer one. The activation between layers is what lets composition
create genuinely new shapes: bends compose into curves, curves into regions.
The menu of activations gets its <a href="#/dl/activation-functions">own chapter
next</a>.</p>
<p>The payoff is the <strong>Universal Approximation Theorem</strong>: an MLP with one
hidden layer (enough neurons, non-linear activation) can approximate any
continuous function to any precision. So why go deep instead of wide? Efficiency:
deep networks reuse intermediate features hierarchically (edges → shapes →
objects), often needing exponentially fewer neurons than a shallow net for the
same function - and the theorem says nothing about whether training will
<em>find</em> the right weights. Architecture is about learnability, not just
expressibility.</p>

<h2>In code: watch XOR fall</h2>
<pre><code class="language-python">import numpy as np
from sklearn.neural_network import MLPClassifier

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([0, 1, 1, 0])                     # XOR - the perceptron-killer

# one hidden layer of 4 ReLU neurons is plenty
mlp = MLPClassifier(hidden_layer_sizes=(4,), activation="relu",
                    solver="adam", max_iter=5000, random_state=2)
mlp.fit(X, y)
print("predictions:", mlp.predict(X))          # [0 1 1 0]  ✓

# peek at the learned re-representation (hidden activations):
W1, b1 = mlp.coefs_[0], mlp.intercepts_[0]
hidden = np.maximum(0, X @ W1 + b1)            # ReLU by hand
print("hidden coordinates:\n", hidden.round(2))
# in this learned space, the classes are linearly separable -
# exactly the hand-crafted h1/h2 trick, discovered automatically
</code></pre>
<p>On real data you'll use PyTorch (<a href="#/dl/nn-in-pytorch">end of this
module</a>); sklearn's MLP is fine for illustration but has no GPU support and
limited knobs.</p>

<h2>Reading an architecture like a practitioner</h2>
<table>
  <tr><th>Design choice</th><th>Convention</th></tr>
  <tr><td>Output layer, regression</td><td>1 linear neuron (no activation)</td></tr>
  <tr><td>Output layer, binary classification</td><td>1 sigmoid neuron + binary cross-entropy</td></tr>
  <tr><td>Output layer, k classes</td><td>k neurons + softmax + cross-entropy</td></tr>
  <tr><td>Hidden activations</td><td>ReLU family (next chapter)</td></tr>
  <tr><td>Width/depth for tabular data</td><td>1–3 hidden layers, 16–512 units - then compare honestly against <a href="#/ml/gradient-boosting">boosting</a>, which often wins on tables</td></tr>
</table>

<h2>Common interview questions</h2>
<div class="qa">
  <div class="qa-q">Q1. Why can an MLP solve XOR when a perceptron can't?</div>
  <div class="qa-a"><p>Hidden neurons each draw a line; the non-linear activation turns their outputs into
  new coordinates in which the classes become linearly separable; the output neuron then
  needs only one line. Depth = learned change of representation.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q2. What happens if all activations are removed from a 10-layer network?</div>
  <div class="qa-a"><p>The product of ten weight matrices collapses into one matrix - an expensive linear
  model. Non-linearity between layers is what makes depth meaningful.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q3. The Universal Approximation Theorem says one hidden layer suffices. Why build deep networks?</div>
  <div class="qa-a"><p>The theorem guarantees existence, not efficiency or trainability: the required width
  can be astronomically large, and gradient descent may never find it. Depth builds features
  hierarchically and reuses them, achieving the same functions with far fewer parameters -
  and empirically, trains better on perceptual data.</p></div>
</div>

<h2>Quick quiz</h2>
<div class="quiz">
  <div class="quiz-q" data-answer="1">
    <p class="quiz-question">1. A network 4 → 8 → 3 (fully connected, with biases) has how many parameters?</p>
    <button class="quiz-opt">15</button>
    <button class="quiz-opt">67</button>
    <button class="quiz-opt">96</button>
    <div class="quiz-explain hidden">Layer 1: 4×8 weights + 8 biases = 40. Layer 2: 8×3 + 3 = 27. Total 67.</div>
  </div>
  <div class="quiz-q" data-answer="0">
    <p class="quiz-question">2. The hidden layers' real job is to…</p>
    <button class="quiz-opt">Re-represent the input so the final layer's job becomes linearly easy</button>
    <button class="quiz-opt">Slow down training for stability</button>
    <button class="quiz-opt">Store the training data</button>
    <div class="quiz-explain hidden">Feature learning: what you did by hand in Module 5, the network does automatically, layer by layer.</div>
  </div>
  <div class="quiz-q" data-answer="2">
    <p class="quiz-question">3. For 10-class digit classification, the output layer should be…</p>
    <button class="quiz-opt">1 sigmoid neuron</button>
    <button class="quiz-opt">10 ReLU neurons</button>
    <button class="quiz-opt">10 neurons with softmax</button>
    <div class="quiz-explain hidden">Softmax turns 10 scores into a probability distribution over classes, paired with cross-entropy loss. Sigmoid handles binary; ReLU is for hidden layers.</div>
  </div>
</div>
`};

CONTENT["dl/activation-functions"] = {
  html: String.raw`
<h1>Activation Functions</h1>
<p class="lead">The activation is the bend between layers - remove it and deep
networks collapse into linear models. Each candidate bend has a shape, a
derivative, and a failure mode, and the history of deep learning is partly the
history of picking better ones.</p>

<h2>The gallery</h2>
<div class="diagram">
<svg viewBox="0 0 660 420" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Six activation function curves: sigmoid, tanh, ReLU, Leaky ReLU, GELU, and softmax as a bar chart">
  <!-- sigmoid -->
  <rect x="15" y="20" width="200" height="180" rx="8" class="d-box-muted"/>
  <text x="115" y="42" text-anchor="middle" class="d-text">sigmoid</text>
  <line x1="30" y1="150" x2="200" y2="150" class="d-grid"/>
  <line x1="115" y1="55" x2="115" y2="190" class="d-grid"/>
  <path d="M 30 148 C 75 146, 95 135, 115 110 C 135 85, 155 74, 200 72" class="d-curve"/>
  <text x="115" y="185" text-anchor="middle" class="d-text-sm">(0, 1) · saturates both ends</text>
  <!-- tanh -->
  <rect x="230" y="20" width="200" height="180" rx="8" class="d-box-muted"/>
  <text x="330" y="42" text-anchor="middle" class="d-text">tanh</text>
  <line x1="245" y1="110" x2="415" y2="110" class="d-grid"/>
  <line x1="330" y1="55" x2="330" y2="190" class="d-grid"/>
  <path d="M 245 168 C 290 166, 310 140, 330 110 C 350 80, 370 54, 415 52" class="d-curve"/>
  <text x="330" y="185" text-anchor="middle" class="d-text-sm">(−1, 1) · zero-centered</text>
  <!-- relu -->
  <rect x="445" y="20" width="200" height="180" rx="8" class="d-box-muted"/>
  <text x="545" y="42" text-anchor="middle" class="d-text">ReLU</text>
  <line x1="460" y1="150" x2="630" y2="150" class="d-grid"/>
  <line x1="545" y1="55" x2="545" y2="190" class="d-grid"/>
  <polyline points="460,150 545,150 630,60" class="d-curve" fill="none"/>
  <text x="545" y="185" text-anchor="middle" class="d-text-sm">max(0, z) · dead below 0</text>
  <!-- leaky relu -->
  <rect x="15" y="215" width="200" height="180" rx="8" class="d-box-muted"/>
  <text x="115" y="237" text-anchor="middle" class="d-text">Leaky ReLU</text>
  <line x1="30" y1="345" x2="200" y2="345" class="d-grid"/>
  <line x1="115" y1="250" x2="115" y2="385" class="d-grid"/>
  <polyline points="30,360 115,345 200,255" class="d-curve" fill="none"/>
  <text x="115" y="380" text-anchor="middle" class="d-text-sm">small slope survives below 0</text>
  <!-- gelu -->
  <rect x="230" y="215" width="200" height="180" rx="8" class="d-box-muted"/>
  <text x="330" y="237" text-anchor="middle" class="d-text">GELU</text>
  <line x1="245" y1="345" x2="415" y2="345" class="d-grid"/>
  <line x1="330" y1="250" x2="330" y2="385" class="d-grid"/>
  <path d="M 245 347 C 290 349, 310 352, 330 345 C 345 338, 365 300, 415 255" class="d-curve"/>
  <text x="330" y="380" text-anchor="middle" class="d-text-sm">smooth ReLU · transformers' pick</text>
  <!-- softmax -->
  <rect x="445" y="215" width="200" height="180" rx="8" class="d-box-muted"/>
  <text x="545" y="237" text-anchor="middle" class="d-text">softmax (output only)</text>
  <rect x="470" y="330" width="28" height="35" class="d-box-soft"/>
  <rect x="510" y="270" width="28" height="95" class="d-box-soft"/>
  <rect x="550" y="345" width="28" height="20" class="d-box-soft"/>
  <rect x="590" y="320" width="28" height="45" class="d-box-soft"/>
  <text x="545" y="382" text-anchor="middle" class="d-text-sm">scores → probabilities (sum = 1)</text>
</svg>
<div class="caption">The six you must know. Hidden layers: ReLU family. Outputs:
sigmoid (binary), softmax (multi-class), none (regression).</div>
</div>

<h2>Sigmoid and tanh: the historical pair</h2>
<div class="math-box">
$$\sigma(z) = \frac{1}{1+e^{-z}}, \;\; \sigma' = \sigma(1-\sigma) \le 0.25
\qquad
\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}, \;\; \tanh' = 1 - \tanh^2 \le 1$$
</div>
<p>Both squash smoothly - and both <strong>saturate</strong>: for |z| ≳ 4 the curve is
flat and the derivative ≈ 0. During backpropagation, gradients multiply through
every layer; sigmoid's max derivative of 0.25 means ten layers shrink the signal
by up to \(0.25^{10} \approx 10^{-6}\) - the
<a href="#/dl/vanishing-gradients">vanishing gradient problem</a> that stalled deep
learning for years. tanh is the better sibling (zero-centered, steeper), still used
inside <a href="#/sequence/lstm">LSTM gates</a>. Sigmoid survives where its (0,1)
range is the point: binary output layers and gates.</p>

<h2>ReLU: embarrassingly simple, era-defining</h2>
<div class="math-box">
$$\mathrm{ReLU}(z) = \max(0, z)
\qquad
\mathrm{ReLU}'(z) = \begin{cases} 1 &amp; z \gt 0 \\ 0 &amp; z \lt 0 \end{cases}$$
</div>
<p>For positive inputs the derivative is exactly 1 - gradients pass through
<em>undiminished</em>, no matter how deep the stack. That single property (plus
being nearly free to compute) made networks of 100+ layers trainable and powered
the 2012 deep-learning explosion. Bonus: negative inputs output exact 0, giving
sparse activations.</p>
<p>Its failure mode is the <strong>dying ReLU</strong>: a neuron pushed into
always-negative territory outputs 0 forever, gets gradient 0 forever, and never
recovers - a permanently dead unit. Large learning rates and bad initialization
cause it; you can watch it happen as a fraction of hidden units flatlining at
zero across the whole dataset.</p>

<h2>The fixes: Leaky ReLU, ELU, GELU</h2>
<div class="math-box">
$$\mathrm{LeakyReLU}(z) = \max(0.01 z,\; z)
\qquad
\mathrm{GELU}(z) = z \cdot \Phi(z)$$
</div>
<p><strong>Leaky ReLU</strong> gives the negative side a small slope (0.01) - dead
neurons keep a faint pulse and can recover. <strong>ELU</strong> uses a smooth
exponential below zero. <strong>GELU</strong> ("gate by the probability a standard
normal is below z") is a smooth, slightly curved ReLU - the default in
Transformers, BERT and GPT (<a href="#/transformers/architecture">Module 10</a>).
In practice the accuracy differences among the ReLU family are small; the
important divide is <em>saturating (sigmoid/tanh) vs non-saturating (ReLU family)
for hidden layers</em>.</p>

<h2>Softmax: the output-layer special</h2>
<div class="math-box">
$$\mathrm{softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$$
</div>
<p>Turns a vector of k scores ("logits") into a probability distribution:
positive, summing to 1, biggest score → biggest share, exponentially so.
It's not for hidden layers - it's the standard multi-class output, always paired
with cross-entropy (<a href="#/dl/loss-functions">next chapter</a>). Numerical
tip baked into every framework: subtract \(\max(\mathbf{z})\) before exponentiating
to avoid overflow - same softmax, stable floats.</p>

<h2>See them and pick them</h2>
<pre><code class="language-python">import numpy as np
import matplotlib.pyplot as plt

z = np.linspace(-5, 5, 400)
funcs = {
    "sigmoid":    1 / (1 + np.exp(-z)),
    "tanh":       np.tanh(z),
    "ReLU":       np.maximum(0, z),
    "LeakyReLU":  np.where(z &gt; 0, z, 0.01 * z),
}
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for name, f in funcs.items():
    axes[0].plot(z, f, label=name)
    axes[1].plot(z[1:], np.diff(f) / np.diff(z), label=name + "′")
axes[0].set_title("activations"); axes[1].set_title("their derivatives")
for ax in axes: ax.legend(); ax.grid(alpha=0.3)
plt.show()
# the right panel is the one that matters: flat derivative = dead learning
</code></pre>
<table>
  <tr><th>Where</th><th>Use</th><th>Because</th></tr>
  <tr><td>Hidden layers (default)</td><td>ReLU</td><td>fast, non-saturating, battle-tested</td></tr>
  <tr><td>Hidden layers (dead neurons observed)</td><td>Leaky ReLU / ELU</td><td>negative side keeps gradient alive</td></tr>
  <tr><td>Transformers</td><td>GELU</td><td>smooth, empirically better there</td></tr>
  <tr><td>Binary output</td><td>sigmoid</td><td>one probability</td></tr>
  <tr><td>Multi-class output</td><td>softmax</td><td>a distribution over classes</td></tr>
  <tr><td>Regression output</td><td>none (linear)</td><td>unbounded target</td></tr>
  <tr><td>RNN/LSTM internals</td><td>tanh + sigmoid</td><td>bounded state, gate semantics</td></tr>
</table>

<h2>Common interview questions</h2>
<div class="qa">
  <div class="qa-q">Q1. Why did ReLU largely replace sigmoid in hidden layers?</div>
  <div class="qa-a"><p>Sigmoid saturates with derivative ≤ 0.25, so products of many layer-derivatives
  vanish - deep nets couldn't train. ReLU's derivative is exactly 1 for active units,
  preserving gradient magnitude through depth, and it costs one comparison to compute.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q2. What is a dying ReLU and how do you prevent it?</div>
  <div class="qa-a"><p>A unit whose input is negative for every example: output 0, gradient 0, learning 0 -
  permanently. Prevent with lower learning rates, sensible
  <a href="#/dl/weight-initialization">initialization</a>, or a leaky/ELU/GELU variant that
  keeps a nonzero negative-side slope.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q3. Why does softmax exponentiate rather than just normalizing the scores?</div>
  <div class="qa-a"><p>exp makes every output positive regardless of sign, makes the mapping differentiable
  and monotonic, sharpens contrasts (score gaps become probability ratios), and combines
  with cross-entropy to give the beautifully simple gradient ŷ − y.</p></div>
</div>

<h2>Quick quiz</h2>
<div class="quiz">
  <div class="quiz-q" data-answer="2">
    <p class="quiz-question">1. A 20-layer network with sigmoid activations trains at a crawl; early layers barely change. The mechanism?</p>
    <button class="quiz-opt">Dying ReLU</button>
    <button class="quiz-opt">Exploding gradients</button>
    <button class="quiz-opt">Vanishing gradients - twenty factors of ≤ 0.25 multiply to nearly nothing</button>
    <div class="quiz-explain hidden">Backprop multiplies layer derivatives; saturating activations shrink the product exponentially with depth. Switch hidden layers to the ReLU family.</div>
  </div>
  <div class="quiz-q" data-answer="1">
    <p class="quiz-question">2. ReLU's derivative for z &gt; 0 is…</p>
    <button class="quiz-opt">0.25 at most</button>
    <button class="quiz-opt">Exactly 1</button>
    <button class="quiz-opt">z</button>
    <div class="quiz-explain hidden">Slope 1 above zero - the gradient superhighway that makes very deep networks trainable.</div>
  </div>
  <div class="quiz-q" data-answer="0">
    <p class="quiz-question">3. Softmax outputs for logits (2.0, 1.0, 0.1) will…</p>
    <button class="quiz-opt">Be positive, sum to 1, and rank in the same order as the logits</button>
    <button class="quiz-opt">Contain a negative value</button>
    <button class="quiz-opt">All equal 1/3</button>
    <div class="quiz-explain hidden">Softmax preserves ranking, exponentially sharpens differences, and produces a valid probability distribution (~0.66, 0.24, 0.10 here).</div>
  </div>
</div>
`};

CONTENT["dl/loss-functions"] = {
  html: String.raw`
<h1>Loss Functions</h1>
<p class="lead">The loss is the network's only definition of "wrong". Gradient
descent will minimize whatever you write here with total literal-mindedness -
so the loss function is where you encode what you actually want.</p>

<h2>What makes a good loss</h2>
<p>A loss \(J(\hat{y}, y)\) must be: <strong>minimized exactly at the right
answer</strong>, <strong>differentiable</strong> (gradients must flow), and shaped so
its gradients <em>point the right way with useful magnitude</em> (this last one is
where MSE-for-classification fails). Losses also encode priorities: squaring
prioritizes big errors; absolute values treat all errors evenly. Choosing a loss
is a modeling decision, not a formality.</p>

<h2>Regression losses</h2>
<div class="math-box">
$$\mathrm{MSE} = \frac{1}{m}\sum_i (\hat{y}^{(i)} - y^{(i)})^2
\qquad
\mathrm{MAE} = \frac{1}{m}\sum_i |\hat{y}^{(i)} - y^{(i)}|$$
</div>
<ul>
  <li><strong>MSE</strong> - the default. Smooth gradient proportional to the error
      (big miss → big correction); corresponds to Gaussian noise assumptions;
      punishes outliers quadratically - which is either a feature or a disaster
      depending on your data.</li>
  <li><strong>MAE</strong> - constant-magnitude gradient; robust to outliers; optimal
      prediction is the <em>median</em> (vs MSE's mean). Non-differentiable at 0 (a
      kink frameworks handle) and its constant gradient can bounce around minima.</li>
  <li><strong>Huber</strong> - quadratic for small errors, linear for large: MSE's
      smoothness near the answer, MAE's calm about outliers. One knob δ sets the
      crossover. The pragmatic choice for noisy regression.</li>
</ul>

<h2>Classification: cross-entropy rules</h2>
<p>You met binary cross-entropy in
<a href="#/ml/logistic-regression">Logistic Regression</a>; the multi-class version
pairs with softmax. With one-hot true labels, only the true class's predicted
probability matters:</p>
<div class="math-box">
$$\mathrm{CE} = -\sum_{k} y_k \ln \hat{y}_k \;=\; -\ln \hat{y}_{\text{true class}}$$
</div>
<p>Read it as <em>surprise</em>: predicted 0.9 for the truth → loss 0.105 (mild);
predicted 0.01 → loss 4.6 (severe); predicted → 0 → loss → ∞. Cross-entropy is
maximum likelihood in disguise, and it demolishes confident wrongness - exactly
the pressure that makes probability outputs honest.</p>
<div class="diagram">
<svg viewBox="0 0 640 260" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Loss curves versus predicted probability of the true class: cross-entropy explodes toward zero probability while squared error stays bounded">
  <line x1="70" y1="220" x2="600" y2="220" class="d-line"/>
  <line x1="70" y1="220" x2="70" y2="30" class="d-line"/>
  <text x="330" y="248" text-anchor="middle" class="d-text-sm">predicted probability assigned to the TRUE class →</text>
  <text x="75" y="245" class="d-text-sm">0</text>
  <text x="590" y="245" class="d-text-sm">1</text>
  <text x="40" y="130" class="d-text-sm" transform="rotate(-90 40 130)">loss</text>
  <!-- cross-entropy: -ln(p) -->
  <path d="M 92 38 C 110 95, 150 150, 230 180 C 330 208, 480 216, 595 219" class="d-curve"/>
  <text x="150" y="60" class="d-text-accent">cross-entropy: −ln(p)</text>
  <!-- squared error (1-p)^2 scaled -->
  <path d="M 70 130 C 200 145, 400 195, 595 219" class="d-line" fill="none" stroke-dasharray="6 4"/>
  <text x="105" y="118" class="d-text-sm">squared error (bounded)</text>
</svg>
<div class="caption">Cross-entropy's explosion near p = 0 is the point: confident wrong
answers must hurt enough to generate strong corrective gradients.</div>
</div>
<p>The engineering reason it beats MSE for classification: with a
sigmoid/softmax output, cross-entropy's gradient with respect to the logits is
simply \(\hat{y} - y\), while MSE's picks up an extra \(\sigma'(z)\) factor that
vanishes precisely when the model is confidently wrong - the worst possible
moment to stop learning.</p>

<h2>Hinge loss: the SVM's voice</h2>
<div class="math-box">
$$L_{hinge} = \max\big(0,\; 1 - y\,z\big), \qquad y \in \{-1, +1\}$$
</div>
<p>Zero loss once an example is correct <em>with margin</em> (\(y\,z \ge 1\)), linear
penalty otherwise. This is exactly <a href="#/ml/svm">max-margin</a> thinking as a
loss function: it doesn't chase probabilities, it chases a safety buffer, and it
ignores well-classified points entirely (they contribute no gradient - the
sparsity that gives SVMs their support vectors).</p>

<h2>The decision table</h2>
<table>
  <tr><th>Task</th><th>Output layer</th><th>Loss</th><th>PyTorch</th></tr>
  <tr><td>Regression, clean data</td><td>linear</td><td>MSE</td><td><code>nn.MSELoss</code></td></tr>
  <tr><td>Regression, outliers</td><td>linear</td><td>Huber / MAE</td><td><code>nn.HuberLoss</code> / <code>nn.L1Loss</code></td></tr>
  <tr><td>Binary classification</td><td>1 logit</td><td>binary cross-entropy</td><td><code>nn.BCEWithLogitsLoss</code></td></tr>
  <tr><td>Multi-class</td><td>k logits</td><td>cross-entropy</td><td><code>nn.CrossEntropyLoss</code></td></tr>
  <tr><td>Max-margin classification</td><td>score</td><td>hinge</td><td><code>nn.HingeEmbeddingLoss</code></td></tr>
  <tr><td>Imbalanced classification</td><td>logits</td><td>weighted CE / focal loss</td><td><code>pos_weight=</code> arg</td></tr>
</table>
<div class="callout warn">
  <span class="co-title">The #1 practical bug in this area</span>
  PyTorch's <code>CrossEntropyLoss</code> and <code>BCEWithLogitsLoss</code> apply
  softmax/sigmoid <em>internally</em> - feed them <strong>raw logits</strong>, not
  probabilities. Adding your own softmax before them double-squashes, trains
  poorly, and fails silently. (Reason they're fused: the combined
  log-softmax-plus-CE is numerically stabler.)
</div>

<h2>Quick comparison in code</h2>
<pre><code class="language-python">import numpy as np

y_true = 1
for p in [0.99, 0.9, 0.5, 0.1, 0.01]:
    ce  = -np.log(p)                 # cross-entropy
    mse = (1 - p) ** 2               # squared error on the probability
    print(f"p(true)={p:4.2f}   CE={ce:6.3f}   MSE={mse:6.3f}")
# p=0.01: CE = 4.6 vs MSE = 0.98 - CE screams, MSE mumbles.

# regression: MSE vs Huber under one outlier
errors = np.array([0.5, -0.3, 0.8, -0.2, 25.0])       # one wild miss
mse_grad_share = (errors**2 / (errors**2).sum())[-1]
print(f"\nthe outlier contributes {mse_grad_share:.0%} of total MSE")
# ≈ 99.8% - the whole model bends to serve one point. Huber caps that.
</code></pre>

<h2>Common interview questions</h2>
<div class="qa">
  <div class="qa-q">Q1. MSE's optimum is the mean; MAE's is the median. Why does that matter?</div>
  <div class="qa-a"><p>It defines what "best single prediction" means under each loss. With skewed targets
  (delivery times), MSE-trained models predict the outlier-inflated mean; MAE-trained ones
  the typical median. Pick the loss that matches the cost of being wrong in your domain.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q2. Why is cross-entropy preferred to MSE for classification, mechanically?</div>
  <div class="qa-a"><p>Through sigmoid/softmax, CE's logit-gradient is (ŷ − y): strong when wrong,
  regardless of saturation. MSE's gradient carries a σ′(z) factor that ≈ 0 for saturated
  (confidently wrong) units - learning stalls exactly when correction is most needed. CE is
  also convex in the logits; MSE-after-sigmoid is not.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q3. When would you reach for focal loss?</div>
  <div class="qa-a"><p>Extreme class imbalance (object detection's 1000:1 background). It down-weights
  easy, already-correct examples by (1−p)^γ so training focuses on the hard minority -
  standard CE would drown the rare class in easy-negative gradient.</p></div>
</div>

<h2>Quick quiz</h2>
<div class="quiz">
  <div class="quiz-q" data-answer="1">
    <p class="quiz-question">1. Your regression data contains a few enormous label errors. Which loss limits their influence?</p>
    <button class="quiz-opt">MSE</button>
    <button class="quiz-opt">Huber (or MAE)</button>
    <button class="quiz-opt">Cross-entropy</button>
    <div class="quiz-explain hidden">MSE lets one 25-unit error contribute 99%+ of the total loss. Huber grows linearly past δ, capping each outlier's pull.</div>
  </div>
  <div class="quiz-q" data-answer="0">
    <p class="quiz-question">2. The model assigns probability 0.001 to the true class. Cross-entropy for that example ≈…</p>
    <button class="quiz-opt">−ln(0.001) ≈ 6.9</button>
    <button class="quiz-opt">0.001</button>
    <button class="quiz-opt">0.999</button>
    <div class="quiz-explain hidden">Loss = −ln(p_true). Near-zero probability on the truth = near-infinite loss = a violent corrective gradient. That's by design.</div>
  </div>
  <div class="quiz-q" data-answer="2">
    <p class="quiz-question">3. Your PyTorch model applies softmax, then feeds the result to nn.CrossEntropyLoss. The consequence?</p>
    <button class="quiz-opt">Correct and standard</button>
    <button class="quiz-opt">A runtime error</button>
    <button class="quiz-opt">Silent double-softmax: flattened gradients, degraded training</button>
    <div class="quiz-explain hidden">CrossEntropyLoss expects raw logits and applies log-softmax internally. It won't crash - it will just quietly train worse. Remove your softmax.</div>
  </div>
</div>
`};

CONTENT["dl/forward-propagation"] = {
  html: String.raw`
<h1>Forward Propagation</h1>
<p class="lead">The forward pass is just "compute the prediction": data flows in,
matrices multiply, activations bend, a number comes out. Master the shapes and
the caching, and backpropagation (next chapter) becomes an unwinding rather than
a mystery.</p>

<h2>The recipe, layer by layer</h2>
<p>For every layer \(l = 1 \dots L\), two lines - the linear step and the bend:</p>
<div class="math-box">
$$\mathbf{z}^{[l]} = \mathbf{W}^{[l]} \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}
\qquad\qquad
\mathbf{a}^{[l]} = g^{[l]}\big(\mathbf{z}^{[l]}\big)$$
</div>
<p>with \(\mathbf{a}^{[0]} = \mathbf{x}\) and the final \(\mathbf{a}^{[L]} = \hat{y}\)
feeding the <a href="#/dl/loss-functions">loss</a>. Everything else on this page
is bookkeeping for these two lines.</p>

<h2>Worked example, every number shown</h2>
<p>Network 2 → 2 → 1 (ReLU hidden, sigmoid output). Input \(\mathbf{x} = (1, 2)\).</p>
<div class="math-box">
$$\mathbf{W}^{[1]} = \begin{pmatrix} 0.5 &amp; -0.3 \\ 0.2 &amp; 0.8 \end{pmatrix},\;
\mathbf{b}^{[1]} = \begin{pmatrix} 0.1 \\ -0.4 \end{pmatrix}
\qquad
\mathbf{W}^{[2]} = (0.6\;\; -0.9),\; b^{[2]} = 0.2$$
$$\mathbf{z}^{[1]} = \begin{pmatrix} 0.5(1) - 0.3(2) + 0.1 \\ 0.2(1) + 0.8(2) - 0.4 \end{pmatrix}
= \begin{pmatrix} 0 \\ 1.4 \end{pmatrix}
\;\xrightarrow{\text{ReLU}}\;
\mathbf{a}^{[1]} = \begin{pmatrix} 0 \\ 1.4 \end{pmatrix}$$
$$z^{[2]} = 0.6(0) - 0.9(1.4) + 0.2 = -1.06
\;\xrightarrow{\sigma}\;
\hat{y} = \sigma(-1.06) \approx 0.257$$
</div>
<p>Prediction: 25.7% probability of class 1. If the true label is \(y = 1\), the
loss is \(-\ln(0.257) \approx 1.36\) - and reducing that number is
backpropagation's job.</p>

<h2>Batching: many examples in one matrix multiply</h2>
<p>Networks never process one example at a time. Stack a <strong>batch</strong> of
\(B\) examples as rows of \(\mathbf{X}\) (our Module 1 convention) and the same two
lines process all of them simultaneously:</p>
<div class="math-box">
$$\mathbf{Z}^{[l]} = \mathbf{A}^{[l-1]} \mathbf{W}^{[l]\top} + \mathbf{b}^{[l]}
\qquad
(B, n_{l-1}) \times (n_{l-1}, n_l) \rightarrow (B, n_l)$$
</div>
<p>The bias - shape \((n_l,)\) - is added to every row by
<a href="#/python/numpy">broadcasting</a>. This is why GPUs matter: one big matrix
multiply saturates thousands of cores, where a Python loop over examples would
crawl. <strong>Shape discipline is the debugging skill:</strong> narrate every tensor -
"batch 64 in, (64, 784) times (784, 128) is (64, 128), plus bias (128,), ReLU,
next". Shape mismatches are ~90% of first deep-learning bugs, and the error
messages name exactly these dimensions.</p>

<h2>The cache: forward's gift to backward</h2>
<div class="diagram">
<svg viewBox="0 0 660 190" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Forward pass through three layers with cached z and a values stored below each layer, feeding into the loss">
  <defs>
    <marker id="fparr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M0,0 L10,5 L0,10 z" style="fill: var(--text-muted)"/>
    </marker>
  </defs>
  <rect x="15" y="45" width="70" height="44" rx="7" class="d-box-soft"/>
  <text x="50" y="72" text-anchor="middle" class="d-text">X</text>
  <line x1="85" y1="67" x2="115" y2="67" class="d-line" marker-end="url(#fparr)"/>
  <rect x="118" y="45" width="120" height="44" rx="7" class="d-box"/>
  <text x="178" y="65" text-anchor="middle" class="d-text-sm">layer 1</text>
  <text x="178" y="80" text-anchor="middle" class="d-text-sm">z¹ = XW¹+b¹, a¹=g(z¹)</text>
  <line x1="238" y1="67" x2="268" y2="67" class="d-line" marker-end="url(#fparr)"/>
  <rect x="271" y="45" width="120" height="44" rx="7" class="d-box"/>
  <text x="331" y="65" text-anchor="middle" class="d-text-sm">layer 2</text>
  <text x="331" y="80" text-anchor="middle" class="d-text-sm">z², a²</text>
  <line x1="391" y1="67" x2="421" y2="67" class="d-line" marker-end="url(#fparr)"/>
  <rect x="424" y="45" width="120" height="44" rx="7" class="d-box"/>
  <text x="484" y="65" text-anchor="middle" class="d-text-sm">output layer</text>
  <text x="484" y="80" text-anchor="middle" class="d-text-sm">ŷ = a³</text>
  <line x1="544" y1="67" x2="574" y2="67" class="d-line" marker-end="url(#fparr)"/>
  <rect x="577" y="45" width="65" height="44" rx="7" class="d-box-soft"/>
  <text x="609" y="72" text-anchor="middle" class="d-text">loss</text>
  <!-- cache -->
  <rect x="118" y="120" width="426" height="40" rx="7" class="d-box-muted" stroke-dasharray="6 4"/>
  <text x="331" y="145" text-anchor="middle" class="d-text-sm">cache: every zˡ and aˡ saved during the forward pass</text>
  <line x1="178" y1="89" x2="178" y2="118" class="d-grid"/>
  <line x1="331" y1="89" x2="331" y2="118" class="d-grid"/>
  <line x1="484" y1="89" x2="484" y2="118" class="d-grid"/>
  <text x="331" y="180" text-anchor="middle" class="d-text-sm">backprop will need these exact values - this cache is why training uses more memory than inference</text>
</svg>
<div class="caption">Forward computes and remembers. The stored activations are the
raw material for the backward pass.</div>
</div>
<p>Backpropagation's formulas (next chapter) contain terms like
\(g'(\mathbf{z}^{[l]})\) and \(\mathbf{a}^{[l-1]}\) - values produced mid-forward.
Frameworks silently record them (PyTorch's autograd graph); it's why training
needs several times the memory of inference, why <code>with torch.no_grad():</code>
saves memory at eval time, and why huge models hit memory limits from
<em>activations</em>, not weights.</p>

<h2>In code: a complete forward pass</h2>
<pre><code class="language-python">import numpy as np

def relu(z):    return np.maximum(0, z)
def sigmoid(z): return 1 / (1 + np.exp(-z))

rng = np.random.default_rng(0)

# architecture: 4 features -> 8 hidden -> 3 hidden -> 1 output
sizes = [4, 8, 3, 1]
params = {}
for l in range(1, len(sizes)):
    params[f"W{l}"] = rng.normal(0, 0.5, (sizes[l-1], sizes[l]))
    params[f"b{l}"] = np.zeros(sizes[l])

def forward(X, params):
    cache = {"a0": X}
    a = X
    L = len(sizes) - 1
    for l in range(1, L + 1):
        z = a @ params[f"W{l}"] + params[f"b{l}"]        # linear
        a = sigmoid(z) if l == L else relu(z)            # bend
        cache[f"z{l}"], cache[f"a{l}"] = z, a            # remember!
        print(f"layer {l}: {cache[f'a{l-1}'].shape} @ "
              f"{params[f'W{l}'].shape} -> {a.shape}")
    return a, cache

X_batch = rng.normal(size=(32, 4))          # batch of 32 examples
y_hat, cache = forward(X_batch, params)
# layer 1: (32, 4) @ (4, 8) -> (32, 8)
# layer 2: (32, 8) @ (8, 3) -> (32, 3)
# layer 3: (32, 3) @ (3, 1) -> (32, 1)
print("predictions:", y_hat[:3].ravel().round(3))
</code></pre>
<p>Every deep learning framework is an industrialized version of this function -
plus automatic differentiation of it, which is where we go next.</p>

<h2>Common interview questions</h2>
<div class="qa">
  <div class="qa-q">Q1. Walk through one forward pass of a 2-layer network, naming shapes.</div>
  <div class="qa-a"><p>Batch (B, n₀) times W₁ᵀ (n₀, n₁) plus broadcast bias → z₁ (B, n₁); activation
  elementwise → a₁; times W₂ᵀ (n₁, n₂) plus b₂ → z₂; output activation → ŷ (B, n₂);
  loss reduces to a scalar. Cache z's and a's for backprop.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q2. Why does training use so much more memory than inference?</div>
  <div class="qa-a"><p>Backprop needs the intermediate activations of every layer for every example in
  the batch, so the forward pass stores them all. Inference discards each layer's output as
  soon as the next is computed. Hence torch.no_grad()/eval-mode memory savings, activation
  checkpointing, and batch-size limits.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q3. Why are batches processed as one matrix multiplication instead of a loop?</div>
  <div class="qa-a"><p>Hardware: one (B, n)×(n, k) GEMM uses optimized BLAS/GPU kernels with perfect
  parallelism and memory locality - orders of magnitude faster than B separate
  vector products. Batching is the entire reason GPUs accelerate deep learning.</p></div>
</div>

<h2>Quick quiz</h2>
<div class="quiz">
  <div class="quiz-q" data-answer="1">
    <p class="quiz-question">1. Batch (64, 20) enters a layer with 50 units. The layer's output shape is…</p>
    <button class="quiz-opt">(20, 50)</button>
    <button class="quiz-opt">(64, 50)</button>
    <button class="quiz-opt">(64, 20)</button>
    <div class="quiz-explain hidden">(64, 20) @ (20, 50) → (64, 50): batch size flows through unchanged; the feature dimension becomes the layer width.</div>
  </div>
  <div class="quiz-q" data-answer="0">
    <p class="quiz-question">2. In the worked example, hidden neuron 1 produced a¹₁ = 0. Its contribution to the output was…</p>
    <button class="quiz-opt">Nothing - 0.6 × 0 = 0; ReLU switched it off for this input</button>
    <button class="quiz-opt">0.6</button>
    <button class="quiz-opt">Negative</button>
    <div class="quiz-explain hidden">z was exactly 0 → ReLU output 0 → the neuron is inactive for this example. Different inputs activate different subsets - that's ReLU's sparse coding.</div>
  </div>
  <div class="quiz-q" data-answer="2">
    <p class="quiz-question">3. The forward pass caches zˡ and aˡ because…</p>
    <button class="quiz-opt">They're needed to print progress bars</button>
    <button class="quiz-opt">They're the model's parameters</button>
    <button class="quiz-opt">Backpropagation's gradient formulas reuse exactly these values</button>
    <div class="quiz-explain hidden">Terms like g′(zˡ) and aˡ⁻¹ appear verbatim in the backward equations. Recomputing them would double the work - so we remember them.</div>
  </div>
</div>
`};
