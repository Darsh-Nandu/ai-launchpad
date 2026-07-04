/* Module 7 - Deep Learning Foundations (part 4) */

CONTENT["dl/vanishing-gradients"] = {
  html: String.raw`
<h1>Vanishing / Exploding Gradients</h1>
<p class="lead">Backpropagation multiplies numbers layer after layer - and long
products of numbers do only two things: rush toward zero or blow up toward
infinity. This single arithmetic fact shaped a decade of deep learning
architecture design.</p>

<h2>The mechanism: products compound</h2>
<p>From <a href="#/dl/backpropagation">Backpropagation</a>, the error signal
reaching layer \(l\) is a product across all layers above it:</p>
<div class="math-box">
$$\boldsymbol{\delta}^{[l]} = \Big(\prod_{k=l+1}^{L} \mathbf{W}^{[k]\top} \odot g'\big(\mathbf{z}^{[k]}\big)\Big)\, \boldsymbol{\delta}^{[L]}$$
</div>
<p>Each factor scales the signal. If typical factors are ~0.25 (sigmoid's max
slope!), thirty layers give \(0.25^{30} \approx 10^{-18}\): the early layers
receive essentially zero gradient and <strong>stop learning entirely</strong> -
they stay stuck at their random initialization while the top layers train.
If typical factors are ~1.5, thirty layers give \(1.5^{30} \approx 200{,}000\):
updates become violent, weights fly to infinity, and the loss goes to
<code>NaN</code>. The window of survival is razor-thin and shrinks with depth.</p>

<h2>Recognizing each in the wild</h2>
<table>
  <tr><th>Symptom</th><th>Diagnosis</th></tr>
  <tr><td>Loss plateaus early; deep layers' weights change, early layers' barely move; deeper models do <em>worse</em> than shallow ones</td><td><strong>Vanishing</strong></td></tr>
  <tr><td>Loss spikes, oscillates wildly, or hits NaN within a few steps; weight norms racing upward</td><td><strong>Exploding</strong></td></tr>
</table>
<p>The direct instrument: log per-layer gradient norms. Healthy training shows
norms of the same order of magnitude across layers; vanishing shows a smooth
exponential decay from output to input; exploding shows growth in the same
direction.</p>
<pre><code class="language-python"># PyTorch: watch the gradient flow after loss.backward()
for name, p in model.named_parameters():
    if p.grad is not None:
        print(f"{name:30s} grad norm: {p.grad.norm().item():.2e}")
# healthy: e-02 .. e-01 everywhere
# vanishing: e-02 at the top decaying to e-09 at the bottom
</code></pre>

<h2>The toolbox that made deep networks possible</h2>
<p>Every fix attacks one factor in the product:</p>
<table>
  <tr><th>Fix</th><th>Attacks</th><th>How</th></tr>
  <tr><td><a href="#/dl/activation-functions">ReLU-family activations</a></td><td>the \(g'\) factors</td><td>derivative is exactly 1 for active units - multiplying by 1 is harmless at any depth</td></tr>
  <tr><td><a href="#/dl/weight-initialization">He/Xavier init</a></td><td>the \(\mathbf{W}\) factors</td><td>start with per-layer amplification ≈ 1</td></tr>
  <tr><td><a href="#/dl/dl-regularization">Batch normalization</a></td><td>drifting activation scales</td><td>re-standardize between layers throughout training, not just at init</td></tr>
  <tr><td><strong>Residual connections</strong> (ResNet, Module 8)</td><td>the product structure itself</td><td>skip paths give gradients an addition-based highway: \(\frac{\partial}{\partial x}(x + F(x)) = 1 + F'\) - the "1" always survives</td></tr>
  <tr><td><strong>Gradient clipping</strong></td><td>explosions</td><td>cap the gradient norm before each update - the standard seatbelt for RNNs</td></tr>
  <tr><td>LSTM gates / attention</td><td>recurrence depth</td><td>Module 9's whole storyline - the same disease across <em>time</em> instead of layers</td></tr>
</table>
<pre><code class="language-python"># gradient clipping in PyTorch - two lines, standard for RNNs/Transformers
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
</code></pre>
<div class="callout">
  <span class="co-title">Why this chapter matters historically</span>
  Pre-2010, deep networks simply didn't train - sigmoid activations plus naive
  initialization guaranteed vanishing gradients past a few layers. The modern
  stack (ReLU 2010–12, He init 2015, batch norm 2015, ResNet 2015) is, in large
  part, a sequence of answers to this one problem - each unlocking another order
  of magnitude of depth, from 5 layers to 1,000+.
</div>

<h2>Common interview questions</h2>
<div class="qa">
  <div class="qa-q">Q1. Why do gradients vanish specifically in <em>early</em> layers?</div>
  <div class="qa-a"><p>The error signal reaching layer l is a product over all layers above it - early
  layers sit at the end of the longest product. With per-factor magnitude &lt; 1, the signal
  decays exponentially in the number of layers it traverses.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q2. How do residual connections help, mathematically?</div>
  <div class="qa-a"><p>A block computes x + F(x), so its local Jacobian is I + F′. Backward, the identity
  term passes gradients through unattenuated regardless of F′ - turning a multiplicative
  chain into (mostly) additive flow. Depth stops being fatal.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q3. Your training loss becomes NaN at step 300. List your first three suspects.</div>
  <div class="qa-a"><p>Exploding gradients (add clipping, check per-layer norms), learning rate too high
  (drop 10×), and numerical hazards in the loss (log(0) - use fused logits losses).
  Check weight/grad norms right before the blow-up to confirm which.</p></div>
</div>

<h2>Quick quiz</h2>
<div class="quiz">
  <div class="quiz-q" data-answer="0">
    <p class="quiz-question">1. Sigmoid's derivative never exceeds 0.25. In a 20-layer sigmoid network, gradients at layer 1 are scaled by at most…</p>
    <button class="quiz-opt">0.25²⁰ ≈ 10⁻¹² - effectively zero</button>
    <button class="quiz-opt">0.25 × 20 = 5</button>
    <button class="quiz-opt">0.25</button>
    <div class="quiz-explain hidden">Factors multiply, not add. Exponential decay with depth is the whole disease - and why saturating activations were abandoned for hidden layers.</div>
  </div>
  <div class="quiz-q" data-answer="2">
    <p class="quiz-question">2. Gradient clipping is primarily a remedy for…</p>
    <button class="quiz-opt">Vanishing gradients</button>
    <button class="quiz-opt">Overfitting</button>
    <button class="quiz-opt">Exploding gradients</button>
    <div class="quiz-explain hidden">It caps the update magnitude when the gradient norm spikes. It cannot resurrect a vanished gradient - there's nothing left to clip.</div>
  </div>
  <div class="quiz-q" data-answer="1">
    <p class="quiz-question">3. A 56-layer plain CNN underperforms its 20-layer version on TRAINING data. This famous observation motivated…</p>
    <button class="quiz-opt">Dropout</button>
    <button class="quiz-opt">Residual connections (ResNet)</button>
    <button class="quiz-opt">Data augmentation</button>
    <div class="quiz-explain hidden">Worse training (not just test) error means an optimization failure, not overfitting. Skip connections let depth help rather than hurt - Module 8 tells the story.</div>
  </div>
</div>
`};

CONTENT["dl/dl-regularization"] = {
  html: String.raw`
<h1>Regularization in Deep Learning</h1>
<p class="lead">A network with a million parameters can memorize any training set.
Three tools keep it honest: dropout (forced redundancy), batch normalization
(stabilized training with a noise bonus), and early stopping (quit while you're
ahead).</p>

<h2>Dropout: training an ensemble by sabotage</h2>
<p>Each training step, every hidden unit is switched off with probability \(p\)
(commonly 0.1–0.5). The network can't rely on any single neuron or fragile
co-adaptation of neurons - features must be redundant and independently useful,
like a team where anyone might be sick tomorrow.</p>
<div class="diagram">
<svg viewBox="0 0 640 250" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Two copies of a small network: full network at inference, and a training step where random crossed-out neurons are dropped">
  <text x="160" y="35" text-anchor="middle" class="d-text">training step (p = 0.5)</text>
  <g>
    <circle cx="70" cy="80" r="16" class="d-box-soft"/><circle cx="70" cy="130" r="16" class="d-box-soft"/>
    <circle cx="70" cy="180" r="16" class="d-box-soft"/>
    <circle cx="160" cy="70" r="16" class="d-box"/>
    <circle cx="160" cy="120" r="16" class="d-box-muted"/>
    <line x1="149" y1="109" x2="171" y2="131" class="d-line-accent"/><line x1="171" y1="109" x2="149" y2="131" class="d-line-accent"/>
    <circle cx="160" cy="170" r="16" class="d-box"/>
    <circle cx="160" cy="220" r="16" class="d-box-muted"/>
    <line x1="149" y1="209" x2="171" y2="231" class="d-line-accent"/><line x1="171" y1="209" x2="149" y2="231" class="d-line-accent"/>
    <circle cx="250" cy="125" r="16" class="d-box-soft"/>
    <line x1="86" y1="80" x2="144" y2="70" class="d-grid"/><line x1="86" y1="130" x2="144" y2="70" class="d-grid"/>
    <line x1="86" y1="180" x2="144" y2="70" class="d-grid"/><line x1="86" y1="80" x2="144" y2="170" class="d-grid"/>
    <line x1="86" y1="130" x2="144" y2="170" class="d-grid"/><line x1="86" y1="180" x2="144" y2="170" class="d-grid"/>
    <line x1="176" y1="70" x2="234" y2="125" class="d-grid"/><line x1="176" y1="170" x2="234" y2="125" class="d-grid"/>
  </g>
  <text x="160" y="245" text-anchor="middle" class="d-text-sm">crossed units: silenced this step only</text>
  <text x="480" y="35" text-anchor="middle" class="d-text">next step: a different subnetwork</text>
  <g>
    <circle cx="390" cy="80" r="16" class="d-box-soft"/><circle cx="390" cy="130" r="16" class="d-box-soft"/>
    <circle cx="390" cy="180" r="16" class="d-box-soft"/>
    <circle cx="480" cy="70" r="16" class="d-box-muted"/>
    <line x1="469" y1="59" x2="491" y2="81" class="d-line-accent"/><line x1="491" y1="59" x2="469" y2="81" class="d-line-accent"/>
    <circle cx="480" cy="120" r="16" class="d-box"/>
    <circle cx="480" cy="170" r="16" class="d-box"/>
    <circle cx="480" cy="220" r="16" class="d-box"/>
    <circle cx="570" cy="125" r="16" class="d-box-soft"/>
    <line x1="406" y1="80" x2="464" y2="120" class="d-grid"/><line x1="406" y1="130" x2="464" y2="120" class="d-grid"/>
    <line x1="406" y1="180" x2="464" y2="120" class="d-grid"/><line x1="406" y1="80" x2="464" y2="170" class="d-grid"/>
    <line x1="406" y1="130" x2="464" y2="170" class="d-grid"/><line x1="406" y1="180" x2="464" y2="220" class="d-grid"/>
    <line x1="496" y1="120" x2="554" y2="125" class="d-grid"/><line x1="496" y1="170" x2="554" y2="125" class="d-grid"/>
    <line x1="496" y1="220" x2="554" y2="125" class="d-grid"/>
  </g>
  <text x="480" y="245" text-anchor="middle" class="d-text-sm">≈ training 2ⁿ thinned networks that share weights</text>
</svg>
<div class="caption">Every step trains a random subnetwork. At test time all units are
on - an implicit average of the ensemble.</div>
</div>
<p>At inference, dropout is off and all units fire; to keep the expected activation
scale identical, frameworks use <em>inverted dropout</em> - surviving units are
scaled by \(1/(1-p)\) during training. The ensemble view is the right intuition:
training samples from \(2^n\) weight-sharing subnetworks, and test time
approximates their average - averaging being the great variance-killer from
<a href="#/ml/random-forest">Random Forest</a>.</p>

<h2>Batch normalization: standardize between layers, forever</h2>
<p>You standardized inputs in <a href="#/features/scaling">Module 5</a> - but hidden
layers' inputs are other layers' outputs, whose distributions <em>drift during
training</em> as the weights below them change. BatchNorm re-standardizes at every
layer, every step, using the current mini-batch:</p>
<div class="math-box">
$$\hat{z} = \frac{z - \mu_{batch}}{\sqrt{\sigma^2_{batch} + \epsilon}}
\qquad\qquad
z_{out} = \gamma\, \hat{z} + \beta$$
</div>
<p>The learnable \(\gamma, \beta\) matter: they let the network <em>undo</em> the
normalization where it helps, so BatchNorm constrains nothing - it just gives
optimization a well-conditioned playing field. Effects: tolerates ~10× larger
learning rates, reduces initialization sensitivity, fights
<a href="#/dl/vanishing-gradients">vanishing gradients</a>, and adds a mild
regularizing noise (each example's normalization depends on its random
batch-mates).</p>
<p>The operational quirk that causes real bugs: at inference there is no batch -
BatchNorm uses <strong>running averages</strong> of μ and σ collected during
training. Forgetting <code>model.eval()</code> in PyTorch (which switches to those
averages, and also disables dropout) is a classic source of "why are my test
predictions weird?". Its cousin <strong>LayerNorm</strong> - normalizing across
features instead of the batch - has no batch dependence at all, which is why
Transformers use it (<a href="#/transformers/architecture">Module 10</a>).</p>

<h2>Early stopping: the free regularizer</h2>
<p>Track validation loss during training; when it stops improving for \(k\)
evaluations ("patience"), stop and restore the best checkpoint. You saw the shape
in <a href="#/dataviz/ml-visualizations">loss curves</a> - training loss falls
forever, validation loss turns upward where memorization begins; early stopping
simply parks at the turn. It costs nothing, combines with everything, and
effectively tunes "training duration" as a complexity knob.</p>

<h2>All three in PyTorch</h2>
<pre><code class="language-python">import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(64, 128),
    nn.BatchNorm1d(128),      # normalize -> then activate (common order)
    nn.ReLU(),
    nn.Dropout(0.3),          # after activation
    nn.Linear(128, 128),
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 1),
)

opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

best_val, patience, bad_epochs = float("inf"), 7, 0
for epoch in range(200):
    model.train()                     # dropout ON, batchnorm uses batch stats
    for xb, yb in train_loader:
        opt.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(model(xb), yb)
        loss.backward()
        opt.step()

    model.eval()                      # dropout OFF, batchnorm uses running stats
    with torch.no_grad():
        val = sum(nn.functional.binary_cross_entropy_with_logits(model(xb), yb)
                  for xb, yb in val_loader)

    if val &lt; best_val - 1e-4:
        best_val, bad_epochs = val, 0
        torch.save(model.state_dict(), "best.pt")     # checkpoint the best
    else:
        bad_epochs += 1
        if bad_epochs &gt;= patience:                    # early stopping
            print(f"stopping at epoch {epoch}")
            break

model.load_state_dict(torch.load("best.pt"))
</code></pre>
<p>Note <code>weight_decay</code> in AdamW - L2-style shrinkage from
<a href="#/ml/regularization">Module 6</a> lives on in deep learning, joined by
<strong>data augmentation</strong> (Module 8: flips, crops, noise - regularization
via the data instead of the weights).</p>

<h2>Choosing among them</h2>
<table>
  <tr><th>Tool</th><th>Default dose</th><th>Notes</th></tr>
  <tr><td>Early stopping</td><td>always, patience 5–10</td><td>free; keep the best checkpoint</td></tr>
  <tr><td>Weight decay (AdamW)</td><td>1e-2 … 1e-4</td><td>the modern L2; tune on log scale</td></tr>
  <tr><td>Dropout</td><td>0.1–0.5 on fully-connected layers</td><td>rare in conv layers; check train-mode is off at eval</td></tr>
  <tr><td>BatchNorm</td><td>standard in CNNs</td><td>unstable with tiny batches (&lt; ~8); LayerNorm for Transformers/RNNs</td></tr>
  <tr><td>Data augmentation</td><td>task-dependent</td><td>often the single biggest win in vision</td></tr>
</table>

<h2>Common interview questions</h2>
<div class="qa">
  <div class="qa-q">Q1. Why does dropout behave differently at train vs test time?</div>
  <div class="qa-a"><p>Training randomly silences units (sampling subnetworks); testing uses the full
  network as an implicit ensemble average, with inverted-dropout scaling keeping expected
  activations consistent. Frameworks switch via model.train()/model.eval() - forgetting the
  switch is a canonical bug.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q2. What do BatchNorm's γ and β do, and why include them?</div>
  <div class="qa-a"><p>They rescale and reshift the normalized activations - learnable freedom to restore
  any distribution the network prefers, including undoing the normalization. BN thus aids
  optimization without limiting expressiveness.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q3. Validation loss bottomed at epoch 23 and has climbed since; training loss still falls. What do you ship?</div>
  <div class="qa-a"><p>The epoch-23 checkpoint. Post-23 improvements are memorization. This is early
  stopping in action - which is why you checkpoint on best-validation, not last-epoch.</p></div>
</div>

<h2>Quick quiz</h2>
<div class="quiz">
  <div class="quiz-q" data-answer="1">
    <p class="quiz-question">1. Dropout fights overfitting by…</p>
    <button class="quiz-opt">Reducing the learning rate</button>
    <button class="quiz-opt">Preventing co-adaptation - features must work without relying on specific other neurons</button>
    <button class="quiz-opt">Deleting weights permanently</button>
    <div class="quiz-explain hidden">Random silencing forces redundancy and independence; the test-time full network then averages the trained subnetworks.</div>
  </div>
  <div class="quiz-q" data-answer="2">
    <p class="quiz-question">2. Your model performs great in training mode but badly with model.eval(). A prime suspect is…</p>
    <button class="quiz-opt">The optimizer</button>
    <button class="quiz-opt">The loss function</button>
    <button class="quiz-opt">BatchNorm's running statistics (or dropout) misbehaving between modes</button>
    <div class="quiz-explain hidden">eval() swaps batch statistics for running averages and disables dropout. Stale/mismatched running stats (e.g., tiny batches, distribution shift) are the classic culprit.</div>
  </div>
  <div class="quiz-q" data-answer="0">
    <p class="quiz-question">3. Early stopping controls model complexity through…</p>
    <button class="quiz-opt">Training duration - halting before the network has time to memorize noise</button>
    <button class="quiz-opt">The number of layers</button>
    <button class="quiz-opt">The batch size</button>
    <div class="quiz-explain hidden">Effective capacity grows with optimization time; stopping at the validation minimum is a complexity cap that costs nothing extra to compute.</div>
  </div>
</div>
`};

CONTENT["dl/nn-from-scratch"] = {
  html: String.raw`
<h1>Building a Neural Network from Scratch (NumPy only)</h1>
<p class="lead">Everything from this module - forward pass, backprop, mini-batches,
He init, cross-entropy - assembled into ~90 lines of NumPy that learn a
non-linear decision boundary. Type it, run it, break it, fix it: this exercise
is where understanding becomes permanent.</p>

<h2>The task</h2>
<p>Two interleaved crescent moons - the dataset that
<a href="#/dl/perceptron">kills any linear model</a>. Architecture:
2 → 32 → 16 → 1 (ReLU, ReLU, sigmoid), binary cross-entropy loss, mini-batch
gradient descent with momentum. Every formula below has appeared in a previous
chapter; this page just makes them shake hands.</p>

<h2>The complete implementation</h2>
<pre><code class="language-python">import numpy as np

rng = np.random.default_rng(0)

# ---------- data: two moons -------------------------------------------------
def make_moons(n=1000, noise=0.15):
    t = rng.uniform(0, np.pi, n // 2)
    a = np.column_stack([np.cos(t), np.sin(t)])                  # upper moon
    b = np.column_stack([1 - np.cos(t), 0.5 - np.sin(t)])        # lower moon
    X = np.vstack([a, b]) + rng.normal(0, noise, (n, 2))
    y = np.r_[np.zeros(n // 2), np.ones(n // 2)]
    idx = rng.permutation(n)
    return X[idx], y[idx].reshape(-1, 1)

X, y = make_moons()
X = (X - X.mean(0)) / X.std(0)                # standardize (Module 5!)
split = 800
Xtr, ytr, Xval, yval = X[:split], y[:split], X[split:], y[split:]

# ---------- model -----------------------------------------------------------
sizes = [2, 32, 16, 1]

def init_params():
    p = {}
    for l in range(1, len(sizes)):
        # He initialization: std = sqrt(2 / fan_in)
        p[f"W{l}"] = rng.normal(0, np.sqrt(2 / sizes[l-1]),
                                (sizes[l-1], sizes[l]))
        p[f"b{l}"] = np.zeros(sizes[l])
    return p

relu    = lambda z: np.maximum(0, z)
sigmoid = lambda z: 1 / (1 + np.exp(-z))

def forward(Xb, p):
    """Two lines per layer: linear, bend. Cache everything for backward."""
    c = {"a0": Xb}
    c["z1"] = Xb @ p["W1"] + p["b1"];      c["a1"] = relu(c["z1"])
    c["z2"] = c["a1"] @ p["W2"] + p["b2"]; c["a2"] = relu(c["z2"])
    c["z3"] = c["a2"] @ p["W3"] + p["b3"]; c["a3"] = sigmoid(c["z3"])
    return c

def backward(yb, p, c):
    """The four backprop equations, batch form. B = batch size."""
    B = len(yb)
    g = {}
    d3 = (c["a3"] - yb) / B                          # δ³ = (ŷ - y)/B
    g["W3"] = c["a2"].T @ d3;  g["b3"] = d3.sum(0)
    d2 = (d3 @ p["W3"].T) * (c["z2"] &gt; 0)            # δ² = δ³W³ᵀ ⊙ ReLU′
    g["W2"] = c["a1"].T @ d2;  g["b2"] = d2.sum(0)
    d1 = (d2 @ p["W2"].T) * (c["z1"] &gt; 0)
    g["W1"] = c["a0"].T @ d1;  g["b1"] = d1.sum(0)
    return g

def bce(y_true, y_hat, eps=1e-9):
    return -np.mean(y_true*np.log(y_hat+eps) + (1-y_true)*np.log(1-y_hat+eps))

# ---------- training loop: mini-batches + momentum ---------------------------
params  = init_params()
vel     = {k: np.zeros_like(v) for k, v in params.items()}
alpha, beta, batch = 0.1, 0.9, 64

for epoch in range(1, 151):
    order = rng.permutation(len(Xtr))                # shuffle each epoch
    for s in range(0, len(Xtr), batch):
        idx = order[s:s+batch]
        cache = forward(Xtr[idx], params)
        grads = backward(ytr[idx], params, cache)
        for k in params:                             # momentum update
            vel[k] = beta * vel[k] + grads[k]
            params[k] -= alpha * vel[k]

    if epoch % 30 == 0:
        tr  = bce(ytr,  forward(Xtr,  params)["a3"])
        va  = bce(yval, forward(Xval, params)["a3"])
        acc = ((forward(Xval, params)["a3"] &gt; 0.5) == yval).mean()
        print(f"epoch {epoch:3d}   train {tr:.3f}   val {va:.3f}   val acc {acc:.3f}")

# epoch  30   train 0.204   val 0.221   val acc 0.930
# epoch 150   train 0.058   val 0.083   val acc 0.975   <- curved boundary learned
</code></pre>

<h2>Reading the code like a checklist</h2>
<ul>
  <li><strong>He init</strong> - the <code>sqrt(2/fan_in)</code>
      (<a href="#/dl/weight-initialization">why</a>). Replace with std 0.01 and
      watch training stall: the vanishing-signal experiment, live.</li>
  <li><strong>Forward caching</strong> - every z and a saved
      (<a href="#/dl/forward-propagation">why</a>).</li>
  <li><strong>δ chain</strong> - <code>(d @ W.T) * (z &gt; 0)</code> is
      \(\delta \mathbf{W}^\top \odot g'\) verbatim
      (<a href="#/dl/backpropagation">derivation</a>). Note the batch form:
      weight gradients become <code>aᵀ @ δ</code>, summing over the batch.</li>
  <li><strong>/B in δ³</strong> - averages the loss over the batch so the learning
      rate doesn't depend on batch size.</li>
  <li><strong>Shuffle + mini-batches + momentum</strong> - Modules
      <a href="#/dl/gradient-descent-variants">GD variants</a> and
      <a href="#/dl/optimizers">optimizers</a> in four lines.</li>
</ul>

<h2>Experiments that teach (really do these)</h2>
<ol>
  <li><strong>Remove the hidden layers</strong> (sizes = [2, 1]): accuracy drops to
      ~85% - a straight line through curved data. The XOR lesson, quantified.</li>
  <li><strong>Break the init</strong> (std = 0.01 or 10): watch loss freeze or NaN -
      <a href="#/dl/vanishing-gradients">vanishing/exploding</a> firsthand.</li>
  <li><strong>Gradient-check backward()</strong> with the central-difference snippet
      from <a href="#/dl/backpropagation">Backpropagation</a> - the professional
      habit for any hand-written gradient.</li>
  <li><strong>Overfit on purpose:</strong> shrink the data to 40 points, train 2,000
      epochs, watch train→0 while val climbs - then add early stopping.</li>
</ol>

<h2>Quick quiz</h2>
<div class="quiz">
  <div class="quiz-q" data-answer="1">
    <p class="quiz-question">1. In the batch backward pass, the weight gradient is a.T @ δ rather than outer(δ, a) because…</p>
    <button class="quiz-opt">They're unrelated quantities</button>
    <button class="quiz-opt">The matrix product sums each example's outer product over the whole batch in one operation</button>
    <button class="quiz-opt">NumPy has no outer product</button>
    <div class="quiz-explain hidden">(n_prev, B) @ (B, n) accumulates Σᵢ aᵢ δᵢᵀ across the batch - the batched version of the single-example rule, computed as one GEMM.</div>
  </div>
  <div class="quiz-q" data-answer="0">
    <p class="quiz-question">2. Replacing He init with tiny weights (std 0.01) in this 3-layer ReLU net causes…</p>
    <button class="quiz-opt">Activations and gradients to shrink layer by layer - training crawls or stalls</button>
    <button class="quiz-opt">Immediate NaN</button>
    <button class="quiz-opt">No difference</button>
    <div class="quiz-explain hidden">Per-layer variance falls below 1 and compounds. Even at 3 layers you'll see visibly slower convergence; at 30 layers, death.</div>
  </div>
  <div class="quiz-q" data-answer="2">
    <p class="quiz-question">3. The (z &gt; 0) factor in backward() implements…</p>
    <button class="quiz-opt">Gradient clipping</button>
    <button class="quiz-opt">Dropout</button>
    <button class="quiz-opt">ReLU's derivative gating the error flow</button>
    <div class="quiz-explain hidden">g′(z) for ReLU is 1 where z &gt; 0, else 0 - inactive units pass no blame, exactly as the backprop equations require.</div>
  </div>
</div>
`};

CONTENT["dl/nn-in-pytorch"] = {
  html: String.raw`
<h1>The Same Network in PyTorch</h1>
<p class="lead">Now rebuild the from-scratch network in PyTorch and map every line
to its NumPy twin. The framework isn't magic - it's your backward() function,
generalized, GPU-accelerated, and guaranteed correct.</p>

<h2>The dictionary: NumPy → PyTorch</h2>
<table>
  <tr><th>You wrote (NumPy)</th><th>PyTorch equivalent</th></tr>
  <tr><td>He-initialized W, b dictionaries</td><td><code>nn.Linear(n_in, n_out)</code> (sensible init included)</td></tr>
  <tr><td><code>forward()</code> with explicit cache</td><td><code>model(x)</code> - autograd records the graph invisibly</td></tr>
  <tr><td>hand-derived <code>backward()</code></td><td><code>loss.backward()</code> - reverse-mode autodiff, exact</td></tr>
  <tr><td>momentum update loop</td><td><code>torch.optim.SGD(momentum=0.9)</code> / <code>AdamW</code></td></tr>
  <tr><td><code>bce()</code> with eps hacks</td><td><code>nn.BCEWithLogitsLoss()</code> - fused, numerically stable</td></tr>
  <tr><td>manual batching + shuffling</td><td><code>DataLoader(shuffle=True)</code></td></tr>
</table>

<h2>The full program</h2>
<pre><code class="language-python">import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import make_moons

# ---------- data (same two moons) -----------------------------------------
X, y = make_moons(n_samples=1000, noise=0.15, random_state=0)
X = (X - X.mean(0)) / X.std(0)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)     # shape (N, 1)

train_ds = TensorDataset(X[:800], y[:800])
val_X, val_y = X[800:], y[800:]
loader = DataLoader(train_ds, batch_size=64, shuffle=True)  # shuffle = our rng.permutation

# ---------- model: 2 -> 32 -> 16 -> 1 ---------------------------------------
model = nn.Sequential(
    nn.Linear(2, 32), nn.ReLU(),
    nn.Linear(32, 16), nn.ReLU(),
    nn.Linear(16, 1),            # NO sigmoid here: we output raw logits
)

loss_fn = nn.BCEWithLogitsLoss()             # sigmoid + BCE, fused &amp; stable
opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device); val_X, val_y = val_X.to(device), val_y.to(device)

# ---------- the canonical training loop -------------------------------------
for epoch in range(1, 151):
    model.train()
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()                  # 1. clear old gradients (they accumulate!)
        logits = model(xb)               # 2. forward (graph recorded)
        loss = loss_fn(logits, yb)       # 3. scalar loss
        loss.backward()                  # 4. YOUR backward(), automated
        opt.step()                       # 5. momentum update

    if epoch % 30 == 0:
        model.eval()
        with torch.no_grad():                        # no graph = less memory
            val_logits = model(val_X)
            val_loss = loss_fn(val_logits, val_y)
            acc = ((torch.sigmoid(val_logits) &gt; 0.5) == val_y.bool()).float().mean()
        print(f"epoch {epoch:3d}   val loss {val_loss:.3f}   val acc {acc:.3f}")

# epoch 150   val loss ≈ 0.08   val acc ≈ 0.975 - matching the NumPy build
</code></pre>

<h2>The five-line liturgy (memorize it)</h2>
<p><code>zero_grad → forward → loss → backward → step</code>. Every PyTorch project
on Earth beats this drum. Three habits around it prevent 90% of beginner bugs:</p>
<ul>
  <li><strong>zero_grad first:</strong> PyTorch <em>accumulates</em> gradients by design
      (useful for gradient accumulation tricks); forget it and every step uses the
      sum of all past gradients - training mysteriously diverges.</li>
  <li><strong>Logits out, fused loss in:</strong> no sigmoid in the model;
      <code>BCEWithLogitsLoss</code>/<code>CrossEntropyLoss</code> handle it
      (<a href="#/dl/loss-functions">the double-softmax trap</a>).</li>
  <li><strong>train()/eval() + no_grad():</strong> mode switches for
      dropout/batchnorm, and graph-free inference for memory
      (<a href="#/dl/forward-propagation">why the graph costs memory</a>).</li>
</ul>

<h2>What autograd actually does</h2>
<p>Every tensor operation you run in forward is logged into a computation graph
with its local derivative rule - exactly the (value, local-gradient) pairs your
NumPy cache held. <code>loss.backward()</code> walks the graph in reverse
topological order applying the chain rule - your δ-recursion, generalized to
arbitrary graphs (branches, skips, attention…). You can peek:</p>
<pre><code class="language-python">w = torch.tensor([2.0], requires_grad=True)
x = torch.tensor([3.0])
loss = (w * x - 5) ** 2          # J = (wx - 5)²
loss.backward()
print(w.grad)                    # dJ/dw = 2(wx-5)·x = 2(1)(3) = 6.0  ✓
</code></pre>
<p>Same chain rule you proved by hand in <a href="#/math/calculus">Calculus</a> -
now with bookkeeping done by machine, at GPU speed, for any architecture you can
compose. That composability is the real gift: the CNNs (Module 8), LSTMs
(Module 9) and Transformers (Module 10) ahead are all "just" new forward
functions; autograd derives their training for free.</p>

<h2>Common interview questions</h2>
<div class="qa">
  <div class="qa-q">Q1. Why must you call optimizer.zero_grad() each iteration?</div>
  <div class="qa-a"><p>backward() adds into .grad rather than overwriting (supporting accumulation across
  micro-batches). Without zeroing, updates use the running sum of all past gradients -
  effectively an unbounded momentum that wrecks training.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q2. What does loss.backward() do, in terms of this module's math?</div>
  <div class="qa-a"><p>Reverse-mode automatic differentiation: traverse the recorded computation graph
  from the loss backwards, applying each operation's local derivative and the chain rule -
  the δ-recursion of backpropagation, generalized to arbitrary DAGs, yielding ∂loss/∂p in
  every parameter's .grad.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q3. Why keep sigmoid out of the model and use BCEWithLogitsLoss?</div>
  <div class="qa-a"><p>The fused log-sum-exp implementation avoids the catastrophic cancellation of
  log(sigmoid(z)) at extreme z, and prevents the double-squash bug. Rule: models emit
  logits; losses own the final non-linearity.</p></div>
</div>

<h2>Quick quiz</h2>
<div class="quiz">
  <div class="quiz-q" data-answer="2">
    <p class="quiz-question">1. The canonical PyTorch step order is…</p>
    <button class="quiz-opt">forward → step → backward → zero_grad</button>
    <button class="quiz-opt">backward → forward → zero_grad → step</button>
    <button class="quiz-opt">zero_grad → forward → loss → backward → step</button>
    <div class="quiz-explain hidden">Clear old gradients, compute predictions and loss, differentiate, update. Any other order silently corrupts the gradients.</div>
  </div>
  <div class="quiz-q" data-answer="0">
    <p class="quiz-question">2. Forgetting zero_grad() causes…</p>
    <button class="quiz-opt">Gradients to accumulate across steps - effectively adding runaway momentum</button>
    <button class="quiz-opt">A compile error</button>
    <button class="quiz-opt">Slower but correct training</button>
    <div class="quiz-explain hidden">.grad sums by design. The bug produces no error message - just training that inexplicably diverges. A rite of passage.</div>
  </div>
  <div class="quiz-q" data-answer="1">
    <p class="quiz-question">3. with torch.no_grad(): during validation exists to…</p>
    <button class="quiz-opt">Freeze the weights permanently</button>
    <button class="quiz-opt">Skip recording the computation graph - saving memory and time when gradients aren't needed</button>
    <button class="quiz-opt">Disable the GPU</button>
    <div class="quiz-explain hidden">No graph = no activation caching = inference at a fraction of training's memory. Weights are untouched either way (that's optimizer.step()'s job).</div>
  </div>
</div>
`};
