/* Module 1 - Math & Stats Foundations (part 2):
   Calculus, Probability, Descriptive Statistics */

CONTENT["math/calculus"] = {
  html: String.raw`
<h1>Calculus for ML</h1>
<p class="lead">Machine learning is the art of turning knobs (weights) to reduce an
error. Calculus answers the only question that matters while turning them:
<em>"if I nudge this knob a little, does the error go up or down - and how fast?"</em></p>

<h2>Why ML needs calculus at all</h2>
<p>Every model on this site is trained the same way: define a number that measures how
wrong the model is (the <strong>loss</strong>), then adjust the weights to make that number
smaller. A modern network has millions of weights. Trying values at random would take
longer than the age of the universe - but calculus gives us, for every single weight
simultaneously, the exact direction to nudge it. That is the entire trick.</p>
<p>You need three tools from calculus, and only three: the <strong>derivative</strong>,
the <strong>gradient</strong> (derivative for many variables at once), and the
<strong>chain rule</strong> (derivative of nested functions). Everything else is detail.</p>

<h2>The derivative: sensitivity of output to input</h2>
<p>Take a function \(f(x)\). Its derivative at a point measures how much the output
changes per tiny change of input - the <strong>slope</strong> of the curve there:</p>
<div class="math-box">
$$f'(x) \;=\; \frac{df}{dx} \;=\; \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$
</div>
<p>Read it as a recipe: nudge the input by a tiny \(h\), see how much the output moved,
divide, and let the nudge shrink to zero. The result is a number with a sign, and
the sign is what ML uses:</p>
<ul>
  <li>\(f'(x) \gt 0\) - function is rising here. To decrease \(f\), move \(x\) <em>left</em>.</li>
  <li>\(f'(x) \lt 0\) - function is falling here. To decrease \(f\), move \(x\) <em>right</em>.</li>
  <li>\(f'(x) = 0\) - flat: a minimum, a maximum, or a saddle. Training stops moving.</li>
</ul>
<p>Notice the pattern in both cases: <strong>to reduce \(f\), step against the sign of the
derivative</strong>. That single sentence <em>is</em> gradient descent.</p>

<div class="diagram">
<svg viewBox="0 0 640 300" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="A parabola with tangent lines: negative slope on the left, zero at the bottom, positive on the right">
  <line x1="40" y1="260" x2="610" y2="260" class="d-line"/>
  <line x1="60" y1="280" x2="60" y2="30" class="d-line"/>
  <!-- parabola -->
  <path d="M 90 60 Q 330 460 570 60" class="d-curve"/>
  <!-- tangent left (negative slope) -->
  <line x1="105" y1="55" x2="235" y2="185" class="d-line" stroke-dasharray="5 4"/>
  <circle cx="170" cy="120" r="5" class="d-dot"/>
  <text x="90" y="40" class="d-text-sm">slope &lt; 0 → step right</text>
  <!-- tangent bottom (zero slope) -->
  <line x1="270" y1="238" x2="390" y2="238" class="d-line" stroke-dasharray="5 4"/>
  <circle cx="330" cy="238" r="5" class="d-dot"/>
  <text x="285" y="285" class="d-text-sm">slope = 0 → minimum</text>
  <!-- tangent right (positive slope) -->
  <line x1="425" y1="185" x2="555" y2="55" class="d-line" stroke-dasharray="5 4"/>
  <circle cx="490" cy="120" r="5" class="d-dot"/>
  <text x="455" y="40" class="d-text-sm">slope &gt; 0 → step left</text>
  <text x="580" y="280" class="d-text-sm">w</text>
  <text x="35" y="25" class="d-text-sm">loss</text>
</svg>
<div class="caption">A loss curve over one weight. Wherever you stand, stepping against
the slope's sign walks you toward the bottom.</div>
</div>

<h2>The rules you'll actually use</h2>
<p>You will rarely compute a limit by hand. In practice, derivatives come from a small
table of rules applied mechanically:</p>
<table>
  <tr><th>Rule</th><th>Formula</th><th>Example</th></tr>
  <tr><td>Constant</td><td>\(\frac{d}{dx}c = 0\)</td><td>\(\frac{d}{dx}7 = 0\)</td></tr>
  <tr><td>Power</td><td>\(\frac{d}{dx}x^n = n\,x^{n-1}\)</td><td>\(\frac{d}{dx}x^3 = 3x^2\)</td></tr>
  <tr><td>Constant multiple</td><td>\(\frac{d}{dx}\,c\,f = c\,f'\)</td><td>\(\frac{d}{dx}5x^2 = 10x\)</td></tr>
  <tr><td>Sum</td><td>\((f+g)' = f' + g'\)</td><td>\(\frac{d}{dx}(x^2+x) = 2x+1\)</td></tr>
  <tr><td>Product</td><td>\((fg)' = f'g + fg'\)</td><td>\(\frac{d}{dx}\,x e^x = e^x + x e^x\)</td></tr>
  <tr><td>Exponential</td><td>\(\frac{d}{dx}e^x = e^x\)</td><td>the function that is its own slope</td></tr>
  <tr><td>Logarithm</td><td>\(\frac{d}{dx}\ln x = \frac{1}{x}\)</td><td>why log-loss gradients are so clean</td></tr>
</table>
<p>Sanity-check on the squared error from Linear Regression: if \(f(w) = (w-3)^2\),
the chain rule gives \(f'(w) = 2(w-3)\). At \(w=5\) the slope is \(+4\)
(go left, toward 3); at \(w=0\) it's \(-6\) (go right, toward 3).
The derivative always points the way home.</p>

<h2>Many knobs at once: partial derivatives and the gradient</h2>
<p>A loss depends on <em>all</em> the weights: \(J(w_1, w_2, \dots, w_n)\).
A <strong>partial derivative</strong> \(\frac{\partial J}{\partial w_i}\) asks the same
sensitivity question for one weight <em>while pretending all the others are frozen
constants</em>. Compute it with the exact same rules - just treat every other symbol
like a number.</p>
<p>Stack all the partials into one vector and you get the <strong>gradient</strong>:</p>
<div class="math-box">
$$\nabla J \;=\; \left( \frac{\partial J}{\partial w_1},\; \frac{\partial J}{\partial w_2},\; \dots,\; \frac{\partial J}{\partial w_n} \right)$$
</div>
<p>The gradient has one famous geometric property, and it's the reason it runs the world:
<strong>\(\nabla J\) points in the direction of steepest <em>increase</em> of \(J\)</strong>,
and its length says how steep that climb is. So the direction of steepest
<em>decrease</em> is exactly \(-\nabla J\). Hence the universal training update:</p>
<div class="math-box">
$$\mathbf{w} \;\leftarrow\; \mathbf{w} - \alpha \, \nabla J(\mathbf{w})$$
</div>
<p>where \(\alpha\) is the learning rate. Quick worked example: for
\(J(w_1, w_2) = w_1^2 + 3w_2^2\),</p>
<div class="math-box">
$$\frac{\partial J}{\partial w_1} = 2w_1, \qquad
\frac{\partial J}{\partial w_2} = 6w_2, \qquad
\nabla J = (2w_1,\; 6w_2)$$
</div>
<p>At the point \((1, 1)\) the gradient is \((2, 6)\): the loss climbs three times
faster along \(w_2\) than along \(w_1\), so the descent step moves mostly in the
\(w_2\) direction. Gradient descent automatically works hardest on the weights
that matter most right now.</p>

<div class="diagram">
<svg viewBox="0 0 640 300" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Elliptical contour lines of a loss bowl with a zigzag gradient descent path approaching the center">
  <defs>
    <marker id="garr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto">
      <path d="M0,0 L10,5 L0,10 z" style="fill: var(--accent)"/>
    </marker>
  </defs>
  <ellipse cx="320" cy="150" rx="270" ry="120" class="d-line" fill="none"/>
  <ellipse cx="320" cy="150" rx="200" ry="88" class="d-line" fill="none"/>
  <ellipse cx="320" cy="150" rx="130" ry="57" class="d-line" fill="none"/>
  <ellipse cx="320" cy="150" rx="62" ry="27" class="d-line" fill="none"/>
  <circle cx="320" cy="150" r="5" class="d-dot"/>
  <text x="332" y="146" class="d-text-sm">minimum</text>
  <!-- descent path -->
  <polyline points="80,240 190,100 250,195 295,122 313,168 320,150" class="d-line-accent" marker-end="url(#garr)" fill="none"/>
  <circle cx="80" cy="240" r="5" class="d-dot"/>
  <text x="52" y="264" class="d-text-sm">start</text>
  <text x="90" y="60" class="d-text-sm">each step follows −∇J: locally straight downhill</text>
</svg>
<div class="caption">Contour view of a loss bowl. Gradient descent repeatedly steps in
the locally steepest downhill direction −∇J until it settles at the minimum.</div>
</div>

<h2>The chain rule: the engine of deep learning</h2>
<p>Real models are <strong>compositions</strong> - functions inside functions. A neural
network is literally: multiply, add, activate, multiply, add, activate… loss.
The chain rule says how sensitivity flows through a composition:</p>
<div class="math-box">
$$y = f(g(x)) \;\;\Longrightarrow\;\;
\frac{dy}{dx} = \frac{dy}{dg} \cdot \frac{dg}{dx}$$
</div>
<p>In words: <em>a small change in \(x\) changes \(g\), and that change in \(g\) changes
\(y\) - multiply the two sensitivities.</em> Like gears: if the first gear turns the
second at ratio 3, and the second turns the third at ratio 2, the first turns the
third at ratio 6.</p>
<p>Worked example, done exactly the way backpropagation will do it:</p>
<div class="math-box">
$$y = (3x + 1)^2. \quad \text{Let } u = 3x+1 \text{ (inner)}, \;\; y = u^2 \text{ (outer)}.$$
$$\frac{dy}{du} = 2u, \qquad \frac{du}{dx} = 3
\qquad\Longrightarrow\qquad
\frac{dy}{dx} = 2u \cdot 3 = 6(3x+1)$$
</div>
<p>Check at \(x = 1\): the formula says slope \(= 6 \cdot 4 = 24\). Numerically,
\(y(1.001) = (4.003)^2 \approx 16.024\), and \((16.024 - 16)/0.001 \approx 24\). ✓</p>
<p>When we reach <a href="#/dl/backpropagation">Backpropagation</a>, you'll see it is
nothing but this rule applied link by link along the network, from the loss backwards
to every weight - multiplying local sensitivities as it goes. If you're comfortable
with the gear analogy, you already understand backprop's core.</p>

<h2>Seeing it run: gradient descent in ten lines</h2>
<p>The code below minimizes \(f(w) = (w-3)^2 + 2\) - a bowl whose bottom is at
\(w = 3\), \(f = 2\). Watch the derivative shrink as we approach the minimum:</p>
<pre><code class="language-python">import numpy as np

f      = lambda w: (w - 3) ** 2 + 2     # the loss curve
grad_f = lambda w: 2 * (w - 3)          # its derivative (power + chain rule)

w = -4.0          # start far from the minimum
alpha = 0.1       # learning rate

for step in range(30):
    g = grad_f(w)
    w = w - alpha * g                   # THE update: step against the slope
    if step % 5 == 0:
        print(f"step {step:2d}   w = {w:7.4f}   f(w) = {f(w):8.5f}   slope = {g:8.4f}")

# The slope starts at -14 (steep, far from minimum) and decays toward 0.
# Steps automatically get smaller near the bottom - no schedule needed.

# --- numerical derivative: how frameworks *verify* gradients -------------
h = 1e-6
w0 = 1.7
numeric  = (f(w0 + h) - f(w0 - h)) / (2 * h)   # central difference
analytic = grad_f(w0)
print("numeric :", numeric)                     # ≈ -2.6
print("analytic:", analytic)                    # exactly -2.6
</code></pre>
<p>The last four lines show the <em>central difference</em> trick: you can approximate
any derivative by evaluating the function twice. It's too slow to train with
(two evaluations per weight, per step), but it's the standard way to unit-test a
hand-written gradient - you will use it again when we build a network from scratch.</p>

<h2>Common interview questions</h2>
<div class="qa">
  <div class="qa-q">Q1. Why does the gradient point in the direction of steepest ascent?</div>
  <div class="qa-a"><p>The change of \(J\) for a small unit step \(\mathbf{u}\) is the
  directional derivative \(\nabla J \cdot \mathbf{u} = \lVert\nabla J\rVert\cos\theta\).
  This is maximized when \(\theta = 0\), i.e. when the step is parallel to \(\nabla J\).
  Any other direction climbs slower - and the opposite direction descends fastest.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q2. What's the difference between a derivative, a partial derivative, and a gradient?</div>
  <div class="qa-a"><p>Derivative: sensitivity of a single-input function. Partial derivative: sensitivity
  with respect to one input of a many-input function, others held fixed. Gradient: all partial
  derivatives collected into a vector - the complete local slope information.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q3. Gradient descent stopped changing but the loss isn't great. What might have happened?</div>
  <div class="qa-a"><p>The gradient is ≈ 0, which happens at local minima, saddle points, or plateaus -
  not only at the global minimum. In deep networks saddle points and flat regions are far more
  common than bad local minima; remedies include momentum-based optimizers
  (<a href="#/dl/optimizers">Optimizers</a>) and better initialization.</p></div>
</div>

<h2>Quick quiz</h2>
<div class="quiz">
  <div class="quiz-q" data-answer="1">
    <p class="quiz-question">1. The loss J(w) has derivative J′(w) = −8 at the current w. To reduce the loss you should…</p>
    <button class="quiz-opt">Decrease w (move left)</button>
    <button class="quiz-opt">Increase w (move right)</button>
    <button class="quiz-opt">Stop - you're at a minimum</button>
    <div class="quiz-explain hidden">Negative slope means the function is falling as w increases, so moving right reduces J. The update w ← w − α·(−8) = w + 8α does exactly that.</div>
  </div>
  <div class="quiz-q" data-answer="2">
    <p class="quiz-question">2. For y = (5x − 2)³, dy/dx is…</p>
    <button class="quiz-opt">3(5x − 2)²</button>
    <button class="quiz-opt">15x²</button>
    <button class="quiz-opt">15(5x − 2)²</button>
    <div class="quiz-explain hidden">Chain rule: outer derivative 3(5x−2)² times inner derivative 5.</div>
  </div>
  <div class="quiz-q" data-answer="0">
    <p class="quiz-question">3. At point (2, 0), the loss J(w₁,w₂) = w₁² + 3w₂² has gradient (4, 0). This tells us…</p>
    <button class="quiz-opt">Only w₁ needs adjusting right now; J is locally flat along w₂</button>
    <button class="quiz-opt">w₂ is more important than w₁</button>
    <button class="quiz-opt">The minimum is at (4, 0)</button>
    <div class="quiz-explain hidden">A zero partial derivative means small changes in that weight don't change the loss at this point - all the local improvement comes from moving w₁ toward 0.</div>
  </div>
</div>
`};

CONTENT["math/probability"] = {
  html: String.raw`
<h1>Probability Basics</h1>
<p class="lead">Data is noisy, models are uncertain, and predictions are bets.
Probability is the grammar for talking about all three precisely - and Bayes'
theorem, which you'll derive here in two lines, powers everything from spam
filters to medical AI.</p>

<h2>What a probability is</h2>
<p>A probability is a number between 0 and 1 measuring how likely an event is.
Two readings coexist, and both matter in ML:</p>
<ul>
  <li><strong>Frequentist:</strong> the long-run fraction. "P(heads) = 0.5" means: flip
      forever, half come up heads.</li>
  <li><strong>Bayesian:</strong> a degree of belief that updates with evidence.
      "P(spam) = 0.92" for one specific email is this kind - that email won't be
      repeated forever; the number expresses confidence.</li>
</ul>
<p>The rules below hold under either reading. Three axioms generate everything:
\(P(A) \ge 0\); \(P(\text{something happens}) = 1\); and for events that can't
co-occur, probabilities add: \(P(A \text{ or } B) = P(A) + P(B)\).</p>

<h2>Joint, marginal, conditional - one table to rule them all</h2>
<p>Almost every probability idea can be read off a simple counting table.
Suppose we inspect <strong>1,000 emails</strong>, noting whether each is spam and
whether it contains the word "free":</p>
<table>
  <tr><th></th><th>contains "free"</th><th>no "free"</th><th>total</th></tr>
  <tr><td><strong>spam</strong></td><td>120</td><td>80</td><td>200</td></tr>
  <tr><td><strong>not spam</strong></td><td>40</td><td>760</td><td>800</td></tr>
  <tr><td><strong>total</strong></td><td>160</td><td>840</td><td>1000</td></tr>
</table>
<ul>
  <li><strong>Joint</strong> - both at once: \(P(\text{spam and "free"}) = 120/1000 = 0.12\).</li>
  <li><strong>Marginal</strong> - ignore the other variable (read the table margins):
      \(P(\text{spam}) = 200/1000 = 0.2\).</li>
  <li><strong>Conditional</strong> - restrict attention to a sub-population:
      among the 160 "free" emails, 120 are spam, so
      \(P(\text{spam} \mid \text{"free"}) = 120/160 = 0.75\).</li>
</ul>
<p>The conditional-probability formula just formalizes that restriction:</p>
<div class="math-box">
$$P(A \mid B) = \frac{P(A \cap B)}{P(B)}$$
</div>
<p>Note the asymmetry - this is the single most common probability mistake:
\(P(\text{spam}\mid\text{"free"}) = 0.75\) but
\(P(\text{"free"}\mid\text{spam}) = 120/200 = 0.6\). "Probability of A given B"
and "probability of B given A" are different questions with different answers.</p>

<h3>Independence</h3>
<p>Two events are <strong>independent</strong> when knowing one tells you nothing about the
other: \(P(A \mid B) = P(A)\), or equivalently \(P(A \cap B) = P(A)\,P(B)\).
In our table, "free" and spam are clearly <em>not</em> independent
(0.75 ≠ 0.2) - which is exactly why the word is useful for filtering.
<a href="#/ml/naive-bayes">Naive Bayes</a> is called "naive" because it <em>assumes</em>
all words are independent given the class - false, but useful.</p>

<h2>Bayes' theorem: reversing the conditional</h2>
<p>Often you know \(P(\text{evidence}\mid\text{cause})\) but need
\(P(\text{cause}\mid\text{evidence})\). Write \(P(A \cap B)\) two ways using the
conditional formula - \(P(A\mid B)P(B) = P(B\mid A)P(A)\) - and divide:</p>
<div class="math-box">
$$P(A \mid B) = \frac{P(B \mid A)\; P(A)}{P(B)}$$
</div>
<p>That's the whole derivation. The four pieces have names worth knowing:
\(P(A)\) is the <em>prior</em> (belief before evidence), \(P(B\mid A)\) the
<em>likelihood</em>, \(P(B)\) the <em>evidence</em>, and \(P(A\mid B)\) the
<em>posterior</em> (belief after evidence).</p>

<h3>The worked example everyone gets wrong</h3>
<p>A disease affects <strong>1% of the population</strong>. A test catches
<strong>99%</strong> of true cases, but also gives false positives for
<strong>5%</strong> of healthy people. You test positive. How worried should you be?</p>
<p>Most people say ~99%. Run the numbers on 10,000 people:</p>
<div class="diagram">
<svg viewBox="0 0 640 260" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Tree: 10000 people split into 100 sick and 9900 healthy; the sick give 99 positives, the healthy give 495 false positives">
  <rect x="20" y="105" width="150" height="44" rx="7" class="d-box"/>
  <text x="95" y="124" text-anchor="middle" class="d-text">10,000 people</text>
  <text x="95" y="141" text-anchor="middle" class="d-text-sm">tested</text>
  <line x1="170" y1="115" x2="255" y2="62" class="d-line"/>
  <line x1="170" y1="139" x2="255" y2="192" class="d-line"/>
  <text x="184" y="72" class="d-text-sm">1% sick</text>
  <text x="180" y="192" class="d-text-sm">99% healthy</text>
  <rect x="258" y="40" width="130" height="42" rx="7" class="d-box-soft"/>
  <text x="323" y="66" text-anchor="middle" class="d-text">100 sick</text>
  <rect x="258" y="172" width="130" height="42" rx="7" class="d-box-muted"/>
  <text x="323" y="198" text-anchor="middle" class="d-text">9,900 healthy</text>
  <line x1="388" y1="61" x2="470" y2="61" class="d-line"/>
  <line x1="388" y1="193" x2="470" y2="193" class="d-line"/>
  <text x="396" y="52" class="d-text-sm">99% test +</text>
  <text x="396" y="184" class="d-text-sm">5% test +</text>
  <rect x="473" y="40" width="146" height="42" rx="7" class="d-box-soft"/>
  <text x="546" y="60" text-anchor="middle" class="d-text">99 true</text>
  <text x="546" y="75" text-anchor="middle" class="d-text">positives</text>
  <rect x="473" y="172" width="146" height="42" rx="7" class="d-box-muted"/>
  <text x="546" y="192" text-anchor="middle" class="d-text">495 false</text>
  <text x="546" y="207" text-anchor="middle" class="d-text">positives</text>
  <text x="320" y="245" text-anchor="middle" class="d-text-accent">P(sick | positive) = 99 / (99 + 495) ≈ 17%</text>
</svg>
<div class="caption">Because healthy people vastly outnumber sick ones, even a small
false-positive rate produces most of the positives.</div>
</div>
<p>Formally:</p>
<div class="math-box">
$$P(\text{sick} \mid +) =
\frac{0.99 \times 0.01}{0.99 \times 0.01 + 0.05 \times 0.99} \approx 0.167$$
</div>
<p>Only ~17%, because the prior (1%) is so small. Lesson for ML: <strong>base rates
dominate when classes are imbalanced</strong> - the same phenomenon will bite us again
with accuracy on imbalanced datasets in
<a href="#/ml/evaluation-metrics">Model Evaluation Metrics</a>.</p>

<h2>Random variables, expectation, variance</h2>
<p>A <strong>random variable</strong> \(X\) is a number produced by a random process -
a die roll, a user's session length, a pixel's noise. Two summaries appear constantly:</p>
<div class="math-box">
$$\mathbb{E}[X] = \sum_x x\, P(X = x)
\qquad\qquad
\mathrm{Var}(X) = \mathbb{E}\big[(X - \mathbb{E}[X])^2\big]$$
</div>
<p>Expectation is the probability-weighted average - the long-run mean.
Variance is the expected squared distance from that mean - the spread.
For one fair die: \(\mathbb{E}[X] = (1+2+\dots+6)/6 = 3.5\) and
\(\mathrm{Var}(X) = \frac{(1-3.5)^2 + \dots + (6-3.5)^2}{6} \approx 2.92\).</p>
<p>Why ML cares: a loss function is an expectation over data
(\(J = \mathbb{E}[\text{error}]\)), mini-batch gradients are noisy <em>estimates</em>
of an expectation, and the bias-variance tradeoff (Module 6) is literally a statement
about \(\mathbb{E}\) and \(\mathrm{Var}\) of model predictions.</p>

<h2>Check everything by simulation</h2>
<p>The great habit probability rewards: when unsure, simulate. Ten lines of NumPy
settle any argument:</p>
<pre><code class="language-python">import numpy as np
rng = np.random.default_rng(0)

# --- law of large numbers: the sample mean converges to E[X] -------------
rolls = rng.integers(1, 7, size=1_000_000)
print("mean of die rolls:", rolls.mean())        # -> 3.5 (E[X])
print("variance         :", rolls.var())         # -> 2.92

# --- verify the medical-test result by brute force ------------------------
N = 1_000_000
sick     = rng.random(N) &lt; 0.01                  # 1% prior
positive = np.where(sick,
                    rng.random(N) &lt; 0.99,        # sensitivity 99%
                    rng.random(N) &lt; 0.05)        # false-positive rate 5%

p_sick_given_pos = sick[positive].mean()
print("P(sick | positive) ≈", round(p_sick_given_pos, 3))   # ≈ 0.167 ✓
</code></pre>

<h2>Common interview questions</h2>
<div class="qa">
  <div class="qa-q">Q1. State Bayes' theorem and name its parts.</div>
  <div class="qa-a"><p>\(P(A\mid B) = P(B\mid A)P(A)/P(B)\): posterior = likelihood × prior / evidence.
  It converts "probability of evidence given cause" into "probability of cause given evidence."</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q2. A test is 99% accurate; you test positive for a rare condition. Why isn't the probability you have it 99%?</div>
  <div class="qa-a"><p>Because the posterior also depends on the prior. When the condition is rare, false
  positives from the huge healthy population outnumber true positives from the tiny sick one.
  With 1% prevalence and a 5% false-positive rate, the posterior is only ~17%.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q3. What does independence mean, and why does Naive Bayes assume it?</div>
  <div class="qa-a"><p>\(P(A\cap B) = P(A)P(B)\) - knowing one event doesn't change the other's probability.
  Naive Bayes assumes features are independent <em>given the class</em> so the joint likelihood
  factorizes into a simple product; it's wrong in reality but keeps the model fast, and it
  often ranks classes correctly anyway.</p></div>
</div>

<h2>Quick quiz</h2>
<div class="quiz">
  <div class="quiz-q" data-answer="1">
    <p class="quiz-question">1. In the email table, P("free" | not spam) is…</p>
    <button class="quiz-opt">40/160 = 0.25</button>
    <button class="quiz-opt">40/800 = 0.05</button>
    <button class="quiz-opt">160/1000 = 0.16</button>
    <div class="quiz-explain hidden">Condition on "not spam": of the 800 non-spam emails, 40 contain "free" → 0.05. (40/160 answers the reversed question P(not spam | "free").)</div>
  </div>
  <div class="quiz-q" data-answer="0">
    <p class="quiz-question">2. In Bayes' theorem, the "prior" is…</p>
    <button class="quiz-opt">Your belief in the hypothesis before seeing the evidence</button>
    <button class="quiz-opt">The probability of the evidence under the hypothesis</button>
    <button class="quiz-opt">The final updated probability</button>
    <div class="quiz-explain hidden">Prior → (× likelihood, ÷ evidence) → posterior. In the medical example the prior was the 1% base rate - and it dominated the answer.</div>
  </div>
  <div class="quiz-q" data-answer="2">
    <p class="quiz-question">3. X is a fair coin flip paying 1 for heads, 0 for tails. E[X] is…</p>
    <button class="quiz-opt">1</button>
    <button class="quiz-opt">0</button>
    <button class="quiz-opt">0.5</button>
    <div class="quiz-explain hidden">E[X] = 1·0.5 + 0·0.5 = 0.5. An expectation needn't be a value X can actually take.</div>
  </div>
</div>
`};

CONTENT["math/descriptive-statistics"] = {
  html: String.raw`
<h1>Descriptive Statistics</h1>
<p class="lead">Before any model touches your data, you must be able to summarize a
column of numbers in one breath: where is its center, how spread out is it, and
what shape does it have? Center, spread, shape - that's this whole page.</p>

<h2>Center: mean, median, mode</h2>
<p>Say a startup has nine employees and a founder. Monthly salaries (in thousands):</p>
<p style="text-align:center"><code>30, 32, 35, 35, 38, 40, 42, 45, 48, 400</code></p>
<div class="math-box">
$$\text{mean } \bar{x} = \frac{1}{m}\sum_{i=1}^{m} x_i = 74.5
\qquad
\text{median} = 39
\qquad
\text{mode} = 35$$
</div>
<ul>
  <li><strong>Mean</strong> - the balance point. Every value participates, so a single
      extreme value (the founder's 400) drags it far above what anyone typical earns.</li>
  <li><strong>Median</strong> - the middle value when sorted (here, average of the 5th and
      6th). The founder could earn 400 or 4,000,000 - the median stays 39. It is
      <strong>robust to outliers</strong>.</li>
  <li><strong>Mode</strong> - the most frequent value. The only option for categorical data
      ("most common browser").</li>
</ul>
<p>Rule of thumb you'll use weekly in EDA: <strong>if mean ≫ median, suspect
right-skew or outliers</strong> (income, house prices, session durations all behave
this way). Report the median for skewed data; the mean alone would mislead.</p>

<h2>Spread: variance, standard deviation, IQR</h2>
<p>Two datasets can share a mean and be wildly different: {49, 50, 51} and
{0, 50, 100}. Spread measures capture the difference.</p>
<div class="math-box">
$$s^2 = \frac{1}{m-1} \sum_{i=1}^{m} (x_i - \bar{x})^2
\qquad\qquad
s = \sqrt{s^2}$$
</div>
<p>Variance is the average squared distance from the mean. Its units are squared
(dollars² - meaningless to humans), so we usually quote its square root, the
<strong>standard deviation</strong> \(s\), which lives back in the original units.
For {49, 50, 51}: \(s = 1\). For {0, 50, 100}: \(s = 50\). Same mean, very
different story.</p>
<details class="disclosure">
<summary>Why divide by m − 1 instead of m? (Bessel's correction)</summary>
<div class="disclosure-body">
<p>Because \(\bar{x}\) was computed <em>from the same sample</em>, the data points are on
average slightly closer to \(\bar{x}\) than to the true population mean - the sample
mean "chases" its own data. Dividing by \(m\) therefore systematically underestimates
the true variance. Dividing by the smaller \(m-1\) inflates the result just enough to
make it unbiased on average. Technically: only \(m-1\) residuals are free to vary,
since they must sum to zero - one <em>degree of freedom</em> was spent estimating the
mean. Watch out in code: NumPy defaults to \(1/m\) (<code>ddof=0</code>), Pandas
defaults to \(1/(m-1)\) (<code>ddof=1</code>) - a classic source of
"why don't my numbers match?"</p>
</div>
</details>
<p>The robust alternative: the <strong>interquartile range</strong>
\(\mathrm{IQR} = Q_3 - Q_1\), the width of the middle 50% of the data
(\(Q_3\) = 75th percentile, \(Q_1\) = 25th). Like the median, it ignores extremes -
and it powers the standard outlier rule
(<a href="#/eda/outliers">Outlier Detection</a>): flag anything outside
\([Q_1 - 1.5\,\mathrm{IQR},\; Q_3 + 1.5\,\mathrm{IQR}]\).</p>

<h2>Shape: skewness and kurtosis</h2>
<p>Center and spread don't distinguish a symmetric bell from a lopsided one.
Two more "moments" of the data do:</p>
<div class="math-box">
$$\text{skewness} = \frac{1}{m} \sum_{i=1}^{m} \left( \frac{x_i - \bar{x}}{s} \right)^{3}
\qquad\qquad
\text{kurtosis} = \frac{1}{m} \sum_{i=1}^{m} \left( \frac{x_i - \bar{x}}{s} \right)^{4} - \;3$$
</div>
<ul>
  <li><strong>Skewness</strong> - asymmetry. Positive (right-skew): long tail to the right,
      mean pulled above median - incomes, prices, wait times. Negative: mirror image -
      exam scores bunched near a ceiling. Near 0: symmetric.</li>
  <li><strong>Kurtosis</strong> (excess, hence the −3) - tail heaviness. Positive: more
      extreme outliers than a normal distribution (stock returns!). Negative: flatter,
      thin-tailed. The normal distribution scores exactly 0.</li>
</ul>

<div class="diagram">
<svg viewBox="0 0 640 250" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Three distribution shapes: right-skewed with mean right of median, symmetric with mean equal to median, left-skewed with mean left of median">
  <!-- right skew -->
  <path d="M 30 200 C 55 40, 95 40, 125 130 C 150 190, 180 198, 205 200" class="d-curve"/>
  <line x1="76" y1="205" x2="76" y2="70" class="d-line" stroke-dasharray="4 3"/>
  <line x1="103" y1="205" x2="103" y2="105" class="d-line-accent" stroke-dasharray="4 3"/>
  <text x="30" y="222" class="d-text-sm">median</text>
  <text x="95" y="238" class="d-text-accent">mean →</text>
  <text x="55" y="30" class="d-text">right-skew (+)</text>
  <!-- symmetric -->
  <path d="M 240 200 C 275 195, 290 40, 320 40 C 350 40, 365 195, 400 200" class="d-curve"/>
  <line x1="320" y1="205" x2="320" y2="45" class="d-line-accent" stroke-dasharray="4 3"/>
  <text x="272" y="228" class="d-text-sm">mean = median</text>
  <text x="270" y="30" class="d-text">symmetric (0)</text>
  <!-- left skew -->
  <path d="M 435 200 C 460 198, 490 190, 515 130 C 545 40, 585 40, 610 200" class="d-curve"/>
  <line x1="564" y1="205" x2="564" y2="70" class="d-line" stroke-dasharray="4 3"/>
  <line x1="537" y1="205" x2="537" y2="105" class="d-line-accent" stroke-dasharray="4 3"/>
  <text x="570" y="222" class="d-text-sm">median</text>
  <text x="468" y="238" class="d-text-accent">← mean</text>
  <text x="495" y="30" class="d-text">left-skew (−)</text>
</svg>
<div class="caption">Skew drags the mean toward the long tail, away from the median.</div>
</div>

<h2>Why ML cares about all this</h2>
<ul>
  <li><strong>Feature scaling</strong> (Module 5) standardizes each feature to mean 0,
      std 1 - you need to know what those are and when outliers distort them.</li>
  <li><strong>Skewed targets</strong> (prices, counts) usually train better after a log
      transform - detected by comparing mean vs median, or by the skewness number.</li>
  <li><strong>High kurtosis</strong> warns you that squared-error losses will be dominated
      by a few extreme points; consider robust losses or clipping.</li>
  <li><strong>Every EDA</strong> (Module 4) starts with exactly these summaries via
      <code>df.describe()</code>.</li>
</ul>

<h2>All of it in code</h2>
<pre><code class="language-python">import numpy as np
import pandas as pd
from scipy import stats

salaries = np.array([30, 32, 35, 35, 38, 40, 42, 45, 48, 400])
s = pd.Series(salaries)

print("mean   :", s.mean())          # 74.5  - dragged up by the founder
print("median :", s.median())        # 39.0  - robust
print("mode   :", s.mode()[0])       # 35

print("std (sample, ddof=1):", s.std())          # pandas default: m-1
print("std (population)    :", s.std(ddof=0))    # numpy default:  m

q1, q3 = s.quantile(0.25), s.quantile(0.75)
iqr = q3 - q1
lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
print("IQR:", iqr)
print("outliers:", s[(s &lt; lo) | (s &gt; hi)].tolist())   # [400]

print("skewness:", stats.skew(salaries))       # strongly positive (right tail)
print("kurtosis:", stats.kurtosis(salaries))   # large: one extreme dominates

# the one-liner you will run on every dataset you ever meet:
print(s.describe())
</code></pre>

<h2>Common interview questions</h2>
<div class="qa">
  <div class="qa-q">Q1. When would you report the median instead of the mean?</div>
  <div class="qa-a"><p>Whenever the data is skewed or has outliers - incomes, prices, latencies.
  The mean is distorted by extremes; the median describes the typical case. If the two
  differ a lot, that difference is itself worth reporting: it signals skew.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q2. Why is standard deviation preferred over variance in reports?</div>
  <div class="qa-a"><p>They contain the same information, but standard deviation is in the data's
  original units (dollars, not dollars²), so statements like "±1 std ≈ ±15k" are
  interpretable. Variance is preferred inside the math because it adds cleanly for
  independent variables.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q3. Your feature has skewness +4. What do you do before modeling?</div>
  <div class="qa-a"><p>Consider a log (or Box-Cox / Yeo-Johnson) transform to compress the right tail,
  check for outliers driving the tail, and prefer robust statistics (median/IQR) when
  summarizing it. Linear models and neural nets both train more happily on roughly
  symmetric features.</p></div>
</div>

<h2>Quick quiz</h2>
<div class="quiz">
  <div class="quiz-q" data-answer="2">
    <p class="quiz-question">1. A dataset of house prices has mean 480k and median 310k. The distribution is…</p>
    <button class="quiz-opt">Left-skewed</button>
    <button class="quiz-opt">Symmetric</button>
    <button class="quiz-opt">Right-skewed</button>
    <div class="quiz-explain hidden">Mean well above median = a long right tail of expensive houses pulling the mean up. Classic for prices.</div>
  </div>
  <div class="quiz-q" data-answer="1">
    <p class="quiz-question">2. Which measure of spread is most robust to a single extreme outlier?</p>
    <button class="quiz-opt">Standard deviation</button>
    <button class="quiz-opt">IQR</button>
    <button class="quiz-opt">Variance</button>
    <div class="quiz-explain hidden">IQR only looks at the 25th–75th percentile range, so extremes can't touch it. Variance and std square the outlier's distance - they're the <em>most</em> affected.</div>
  </div>
  <div class="quiz-q" data-answer="0">
    <p class="quiz-question">3. Excess kurtosis of +6 on a returns dataset means…</p>
    <button class="quiz-opt">Far more extreme values than a normal distribution would produce</button>
    <button class="quiz-opt">The data is perfectly normal</button>
    <button class="quiz-opt">The data has no outliers</button>
    <div class="quiz-explain hidden">High positive kurtosis = heavy tails. Models (and risk estimates) that assume normality will badly underestimate how often extreme events occur.</div>
  </div>
</div>
`};
