/* Module 10 - Transformers & Modern Architectures (part 1) */

CONTENT["transformers/self-attention"] = {
  html: String.raw`
<h1>Self-Attention &amp; Multi-Head Attention</h1>
<p class="lead">Take the attention mechanism from Module 9 and point a sequence at
<em>itself</em>: every word asks every other word "are you relevant to me?" and
updates its meaning accordingly. Computed with three matrices and one softmax,
this is the engine of every modern language model.</p>

<h2>Why a word needs to look at its neighbors</h2>
<p>In "The animal didn't cross the street because <em>it</em> was too tired",
what does "it" mean? Every reader silently links "it" → "animal" (swap "tired"
for "wide" and the link flips to "street"). A word's meaning depends on context -
so a good representation of "it" must be built by <em>consulting the other
words</em>. Self-attention is that consultation, run for all words at once,
in both directions, any distance.</p>

<h2>Queries, keys, values: the library analogy</h2>
<p>Each word's embedding \(\mathbf{x}_i\) is projected three ways with learned
matrices:</p>
<div class="math-box">
$$\mathbf{q}_i = \mathbf{W}_Q \mathbf{x}_i \qquad
\mathbf{k}_i = \mathbf{W}_K \mathbf{x}_i \qquad
\mathbf{v}_i = \mathbf{W}_V \mathbf{x}_i$$
</div>
<ul>
  <li><strong>Query:</strong> what I'm looking for ("I'm a pronoun - who's my referent?")</li>
  <li><strong>Key:</strong> what I advertise ("I'm an animate noun!")</li>
  <li><strong>Value:</strong> what I hand over if selected (my actual content)</li>
</ul>
<p>Word \(i\) scores its query against every word's key (dot product), softmaxes
the scores into weights, and collects the weighted sum of values - exactly
Module 9's <a href="#/sequence/attention">attention</a>, with all three roles
coming from the same sequence. In matrix form, all words at once:</p>
<div class="math-box">
$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) =
\text{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}$$
</div>
<p>The \(\sqrt{d_k}\) is a practical guard: dot products of \(d_k\)-dimensional
vectors have variance ~\(d_k\), and large scores push softmax into its
<a href="#/dl/activation-functions">saturated zone</a> where gradients die.
Dividing by \(\sqrt{d_k}\) keeps scores in the healthy range. (You can now read
the famous equation aloud: "similarity scores, scaled, softmaxed, used to blend
values." Nothing you haven't built already.)</p>

<h2>Worked micro-example</h2>
<p>Sentence: "cat sat here", with toy 1-D queries/keys. Say
q(sat)·k(cat)=4, q(sat)·k(sat)=2, q(sat)·k(here)=0 (after scaling).
Softmax → weights (0.84, 0.11, 0.05). The new representation of "sat" becomes
0.84·v(cat) + 0.11·v(sat) + 0.05·v(here) - "sat" is now <em>mostly informed by
its subject</em>. Every word gets this treatment simultaneously, and stacking
layers lets these enriched meanings consult each other again.</p>

<h2>Multi-head: several conversations at once</h2>
<p>One attention pattern must compromise: should "it" attend to its referent, or
to the verb governing it, or to nearby adjectives? <strong>Multi-head attention</strong>
runs \(h\) attention operations in parallel (each with its own learned
\(\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V\) in a smaller subspace \(d_{model}/h\)),
then concatenates and mixes the results:</p>
<div class="math-box">
$$\text{head}_j = \text{Attention}(\mathbf{X}\mathbf{W}_Q^{(j)}, \mathbf{X}\mathbf{W}_K^{(j)}, \mathbf{X}\mathbf{W}_V^{(j)})
\qquad
\text{MHA} = [\text{head}_1; \dots; \text{head}_h]\,\mathbf{W}_O$$
</div>
<div class="diagram">
<svg viewBox="0 0 660 260" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Input embeddings split into three parallel attention heads, each producing its own pattern, concatenated and projected to the output">
  <defs>
    <marker id="mharr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M0,0 L10,5 L0,10 z" style="fill: var(--text-muted)"/>
    </marker>
  </defs>
  <rect x="25" y="100" width="110" height="50" rx="8" class="d-box-soft"/>
  <text x="80" y="122" text-anchor="middle" class="d-text-sm">embeddings X</text>
  <text x="80" y="138" text-anchor="middle" class="d-text-sm">(T × d_model)</text>
  <line x1="135" y1="110" x2="195" y2="60" class="d-line" marker-end="url(#mharr)"/>
  <line x1="135" y1="125" x2="195" y2="125" class="d-line" marker-end="url(#mharr)"/>
  <line x1="135" y1="140" x2="195" y2="190" class="d-line" marker-end="url(#mharr)"/>
  <rect x="198" y="35" width="180" height="50" rx="8" class="d-box"/>
  <text x="288" y="57" text-anchor="middle" class="d-text-sm">head 1: QK<tspan baseline-shift="super" font-size="9">T</tspan>/√d → softmax → V</text>
  <text x="288" y="74" text-anchor="middle" class="d-text-sm">learns: pronoun ↔ referent</text>
  <rect x="198" y="100" width="180" height="50" rx="8" class="d-box"/>
  <text x="288" y="122" text-anchor="middle" class="d-text-sm">head 2 (own W_Q, W_K, W_V)</text>
  <text x="288" y="139" text-anchor="middle" class="d-text-sm">learns: verb ↔ subject</text>
  <rect x="198" y="165" width="180" height="50" rx="8" class="d-box"/>
  <text x="288" y="187" text-anchor="middle" class="d-text-sm">head 3</text>
  <text x="288" y="204" text-anchor="middle" class="d-text-sm">learns: adjacent-word syntax</text>
  <line x1="378" y1="60" x2="440" y2="115" class="d-line" marker-end="url(#mharr)"/>
  <line x1="378" y1="125" x2="440" y2="125" class="d-line" marker-end="url(#mharr)"/>
  <line x1="378" y1="190" x2="440" y2="135" class="d-line" marker-end="url(#mharr)"/>
  <rect x="443" y="100" width="105" height="50" rx="8" class="d-box-soft"/>
  <text x="495" y="122" text-anchor="middle" class="d-text-sm">concat heads</text>
  <text x="495" y="138" text-anchor="middle" class="d-text-sm">· project W_O</text>
  <line x1="548" y1="125" x2="600" y2="125" class="d-line" marker-end="url(#mharr)"/>
  <text x="615" y="130" class="d-text">out</text>
  <text x="330" y="245" text-anchor="middle" class="d-text-sm">each head runs in d_model/h dimensions - total compute ≈ one full-size attention</text>
</svg>
<div class="caption">Heads specialize. Probing trained models finds heads tracking
syntax, coreference, positional neighbors - none of it programmed.</div>
</div>

<h2>The trade that defines the architecture</h2>
<table>
  <tr><th></th><th>RNN/LSTM</th><th>Self-attention</th></tr>
  <tr><td>Path between distant words</td><td>O(distance) steps</td><td><strong>O(1)</strong> - direct</td></tr>
  <tr><td>Parallelism over sequence</td><td>none (inherently serial)</td><td><strong>total</strong> - one matrix multiply</td></tr>
  <tr><td>Cost in sequence length T</td><td>O(T)</td><td><strong>O(T²)</strong> - every pair scored</td></tr>
  <tr><td>Word order</td><td>built in</td><td><strong>absent</strong> - needs positional encoding (next chapter)</td></tr>
</table>
<p>The O(T²) cost is the price of the all-pairs shortcut - the reason "context
length" is the headline spec of every LLM, and a whole research industry
(sparse/linear/flash attention) exists to soften it. The order-blindness is the
other bill to pay: shuffle the input words and self-attention's outputs shuffle
identically. Fixing that is the <a href="#/transformers/positional-encoding">next
chapter</a>.</p>

<h2>In code: scaled dot-product attention in eight lines</h2>
<pre><code class="language-python">import torch
import torch.nn.functional as F

T, d_model, h = 6, 64, 4               # 6 tokens, width 64, 4 heads
x = torch.randn(1, T, d_model)

Wq = torch.nn.Linear(d_model, d_model, bias=False)
Wk = torch.nn.Linear(d_model, d_model, bias=False)
Wv = torch.nn.Linear(d_model, d_model, bias=False)

def heads(t):  # (B, T, d) -> (B, h, T, d/h)
    return t.view(1, T, h, d_model // h).transpose(1, 2)

Q, K, V = heads(Wq(x)), heads(Wk(x)), heads(Wv(x))

scores = Q @ K.transpose(-2, -1) / (d_model // h) ** 0.5   # (B, h, T, T)
A = F.softmax(scores, dim=-1)          # each row: where this token looks
out = (A @ V).transpose(1, 2).reshape(1, T, d_model)

print(A[0, 0].round(decimals=2))       # head 0's T×T attention map
print(A[0, 0].sum(-1))                 # each row sums to 1.0
# torch equivalent: torch.nn.MultiheadAttention(d_model, h)
</code></pre>

<h2>Common interview questions</h2>
<div class="qa">
  <div class="qa-q">Q1. Explain Q, K, V in plain language.</div>
  <div class="qa-a"><p>Three learned projections of the same token: the query is what it's searching for,
  the key is what it advertises to searchers, the value is the content it contributes when
  matched. Attention = match queries to keys, use the match strengths to blend values.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q2. Why divide by √d_k?</div>
  <div class="qa-a"><p>Dot products of d_k-dim random vectors grow with variance ∝ d_k; unscaled scores
  saturate the softmax (one weight ≈ 1, rest ≈ 0), whose gradients then vanish. Scaling
  by √d_k normalizes score variance to ~1, keeping the softmax trainable.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q3. Why multiple heads instead of one big attention?</div>
  <div class="qa-a"><p>One softmax produces one focus pattern per token - a forced compromise between
  different relationship types. Independent heads in subspaces let the model attend to
  referents, syntax, and neighbors simultaneously, at essentially the same total cost.</p></div>
</div>

<h2>Quick quiz</h2>
<div class="quiz">
  <div class="quiz-q" data-answer="1">
    <p class="quiz-question">1. In self-attention, the attention matrix for a T-token sequence has shape…</p>
    <button class="quiz-opt">T × d_model</button>
    <button class="quiz-opt">T × T - every token scores every token</button>
    <button class="quiz-opt">d_model × d_model</button>
    <div class="quiz-explain hidden">All-pairs scoring is both the power (O(1) paths) and the O(T²) cost that limits context length.</div>
  </div>
  <div class="quiz-q" data-answer="0">
    <p class="quiz-question">2. Feeding self-attention the same words in shuffled order produces…</p>
    <button class="quiz-opt">Identically shuffled outputs - the operation itself ignores position</button>
    <button class="quiz-opt">An error</button>
    <button class="quiz-opt">Completely different representations</button>
    <div class="quiz-explain hidden">Q·K scoring is order-blind (a set operation). Positional encodings must inject order - the next chapter's whole job.</div>
  </div>
  <div class="quiz-q" data-answer="2">
    <p class="quiz-question">3. The key advantage of self-attention over an LSTM for connecting words 200 positions apart is…</p>
    <button class="quiz-opt">Fewer parameters</button>
    <button class="quiz-opt">Lower total FLOPs</button>
    <button class="quiz-opt">A one-step direct path - no gradient survives 200 recurrent multiplications as well as 1 dot product</button>
    <div class="quiz-explain hidden">Distance-independent interaction (plus full parallelism during training). The costs - O(T²) and order-blindness - were judged worth it, and the field agreed.</div>
  </div>
</div>
`};

CONTENT["transformers/positional-encoding"] = {
  html: String.raw`
<h1>Positional Encoding</h1>
<p class="lead">Self-attention treats a sentence as a <em>bag</em> of tokens -
"dog bites man" and "man bites dog" come out identical. Positional encoding is
the fix: stamp each token's embedding with a vector that says <em>where</em> it is.</p>

<h2>The requirements for a good position stamp</h2>
<p>Naive ideas fail informatively. Appending the raw index (1, 2, 3, …) explodes
in magnitude for long texts and generalizes badly past training lengths.
Normalizing to [0, 1] makes "position 0.5" mean different absolute places in
different-length sentences. What we want:</p>
<ul>
  <li>bounded values, unique pattern per position;</li>
  <li>a consistent notion of <em>relative distance</em> (position 7 relates to 5 the
      way 107 relates to 105);</li>
  <li>graceful extension to sequences longer than any seen in training.</li>
</ul>

<h2>The sinusoidal solution</h2>
<p>The original Transformer assigns position \(pos\) a vector whose dimension
pairs \((2i, 2i+1)\) hold a sine and cosine at a frequency that decreases
geometrically with \(i\):</p>
<div class="math-box">
$$PE_{(pos, 2i)} = \sin\!\left(\frac{pos}{10000^{2i/d_{model}}}\right)
\qquad
PE_{(pos, 2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$
</div>
<p>The right mental model is an <strong>odometer in continuous form</strong> (or a
binary counter): fast-spinning dials in the low dimensions distinguish adjacent
positions; slow dials in the high dimensions distinguish distant regions.
Together the dial readings uniquely fingerprint every position, with all values
politely bounded in [−1, 1].</p>
<div class="diagram">
<svg viewBox="0 0 660 240" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Sinusoids of decreasing frequency across positions: a fast wave, a medium wave and a slow wave stacked; each position reads a unique combination">
  <line x1="40" y1="60" x2="620" y2="60" class="d-grid"/>
  <path d="M 40 60 q 12 -22 24 0 t 24 0 t 24 0 t 24 0 t 24 0 t 24 0 t 24 0 t 24 0 t 24 0 t 24 0 t 24 0 t 24 0 t 24 0 t 24 0 t 24 0 t 24 0 t 24 0 t 24 0 t 24 0 t 24 0 t 24 0 t 24 0 t 24 0 t 24 0" class="d-curve" fill="none"/>
  <text x="628" y="64" class="d-text-sm">dim 0 (fast)</text>
  <line x1="40" y1="125" x2="620" y2="125" class="d-grid"/>
  <path d="M 40 125 q 40 -38 80 0 t 80 0 t 80 0 t 80 0 t 80 0 t 80 0 t 80 0" class="d-curve" fill="none"/>
  <text x="628" y="129" class="d-text-sm">dim 8</text>
  <line x1="40" y1="190" x2="620" y2="190" class="d-grid"/>
  <path d="M 40 190 C 180 140, 340 140, 480 190 C 550 213, 590 215, 620 210" class="d-curve" fill="none"/>
  <text x="628" y="194" class="d-text-sm">dim 16 (slow)</text>
  <line x1="230" y1="40" x2="230" y2="215" class="d-line-accent" stroke-dasharray="5 4"/>
  <text x="238" y="228" class="d-text-accent">position p reads one value off every dial → a unique fingerprint</text>
</svg>
<div class="caption">Geometrically spaced frequencies: low dims resolve neighbors,
high dims resolve regions - a continuous odometer.</div>
</div>
<p>The elegant property that motivated sinusoids: for any offset \(k\),
\(PE_{pos+k}\) is a <em>linear function</em> of \(PE_{pos}\) (sum-angle identities -
rotation by a fixed angle). So "attend to the token 3 steps back" is expressible
as a learned linear map, independent of absolute position - relative reasoning
for free.</p>

<h2>How it's applied: addition, then attention does the rest</h2>
<div class="math-box">
$$\mathbf{x}_i^{input} = \mathbf{embedding}(w_i) + \mathbf{PE}_i$$
</div>
<p>Just added - same dimension, no extra parameters. Each token enters the network
carrying <em>what</em> it is plus <em>where</em> it is, and the Q/K projections can
then score positional agreement alongside semantic agreement. "Dog bites man"
and "man bites dog" now produce different attention patterns - order restored.</p>

<h2>The modern menu</h2>
<table>
  <tr><th>Scheme</th><th>Idea</th><th>Used by</th></tr>
  <tr><td>Sinusoidal (fixed)</td><td>the formula above; no parameters, extends to any length</td><td>original Transformer</td></tr>
  <tr><td>Learned absolute</td><td>one trainable vector per position index</td><td>BERT, GPT-2 (simple, capped at trained length)</td></tr>
  <tr><td>Relative (T5-style)</td><td>learn biases on attention scores by distance i−j</td><td>T5, many successors</td></tr>
  <tr><td><strong>RoPE</strong> (rotary)</td><td>rotate Q/K by position-dependent angles so dot products depend only on relative offset</td><td>Llama, most modern LLMs</td></tr>
  <tr><td>ALiBi</td><td>penalize attention linearly with distance</td><td>length-extrapolation specialists</td></tr>
</table>
<p>The trend line: from absolute stamps toward <em>relative</em> schemes baked into
the attention computation itself - because "how far apart" matters more than
"which absolute slot", especially when extrapolating past training lengths.</p>

<h2>In code</h2>
<pre><code class="language-python">import torch, math

def sinusoidal_pe(max_len, d_model):
    pe = torch.zeros(max_len, d_model)
    pos = torch.arange(max_len).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d_model, 2).float()
                    * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)      # even dims
    pe[:, 1::2] = torch.cos(pos * div)      # odd dims
    return pe

pe = sinusoidal_pe(max_len=100, d_model=64)
print(pe.shape)                              # (100, 64)
print(pe.min().item(), pe.max().item())      # bounded: -1.0 .. 1.0

# uniqueness check: nearest neighbor of each position's code is itself
sims = pe @ pe.T
print((sims.argmax(1) == torch.arange(100)).all().item())   # True

# usage: x = token_embedding(tokens) + pe[:tokens.size(1)]
</code></pre>

<h2>Common interview questions</h2>
<div class="qa">
  <div class="qa-q">Q1. Why do Transformers need positional encoding when RNNs don't?</div>
  <div class="qa-a"><p>RNNs consume tokens sequentially - order is implicit in the processing itself.
  Self-attention is a set operation over all tokens at once (permutation-equivariant), so
  order must be injected explicitly into the representations.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q2. Why sines and cosines at multiple frequencies rather than the raw index?</div>
  <div class="qa-a"><p>Bounded values at any length; a unique multi-scale fingerprint per position; and the
  rotation property - PE(pos+k) is a fixed linear transform of PE(pos) - which makes relative
  offsets learnable as linear maps and supports extrapolation beyond training lengths.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q3. What's the practical weakness of learned absolute positions, and what replaced them?</div>
  <div class="qa-a"><p>They exist only up to the trained maximum length (nothing to look up beyond it) and
  encode absolute rather than relative structure. Modern LLMs use relative schemes - RoPE
  rotates Q and K so attention scores depend on token distance directly.</p></div>
</div>

<h2>Quick quiz</h2>
<div class="quiz">
  <div class="quiz-q" data-answer="2">
    <p class="quiz-question">1. Positional encodings are combined with token embeddings by…</p>
    <button class="quiz-opt">Concatenation, doubling the width</button>
    <button class="quiz-opt">Multiplication</button>
    <button class="quiz-opt">Element-wise addition, same dimension</button>
    <div class="quiz-explain hidden">Simple addition - the network learns to keep the "what" and "where" components usable within one vector.</div>
  </div>
  <div class="quiz-q" data-answer="0">
    <p class="quiz-question">2. Low (fast) dimensions of the sinusoidal code distinguish…</p>
    <button class="quiz-opt">Adjacent positions; slow dimensions distinguish far-apart regions</button>
    <button class="quiz-opt">Different words</button>
    <button class="quiz-opt">Different attention heads</button>
    <div class="quiz-explain hidden">Multi-scale frequencies act like odometer dials: fast wheels for local resolution, slow wheels for global position.</div>
  </div>
  <div class="quiz-q" data-answer="1">
    <p class="quiz-question">3. Without any positional encoding, a Transformer language model would treat its input as…</p>
    <button class="quiz-opt">A reversed sequence</button>
    <button class="quiz-opt">An unordered bag of tokens</button>
    <button class="quiz-opt">A single long word</button>
    <div class="quiz-explain hidden">Attention scores are order-invariant; all syntactic information carried by word order would simply not exist for the model.</div>
  </div>
</div>
`};

CONTENT["transformers/architecture"] = {
  html: String.raw`
<h1>The Transformer Architecture</h1>
<p class="lead">Self-attention, positional encoding, residual connections, layer
norm, and a small MLP - the Transformer is five familiar parts arranged in a
repeatable block, stacked N times. This page assembles the machine that runs
the modern AI era.</p>

<h2>The encoder block: the repeating unit</h2>
<div class="diagram">
<svg viewBox="0 0 660 330" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Transformer encoder block: input plus positional encoding flows through multi-head attention with a residual add and layer norm, then a feed-forward network with another add and norm, repeated N times">
  <defs>
    <marker id="txarr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M0,0 L10,5 L0,10 z" style="fill: var(--text-muted)"/>
    </marker>
  </defs>
  <rect x="230" y="280" width="200" height="36" rx="7" class="d-box-soft"/>
  <text x="330" y="303" text-anchor="middle" class="d-text-sm">embeddings + positional encoding</text>
  <line x1="330" y1="280" x2="330" y2="252" class="d-line" marker-end="url(#txarr)"/>
  <rect x="230" y="212" width="200" height="38" rx="7" class="d-box"/>
  <text x="330" y="236" text-anchor="middle" class="d-text-sm">multi-head self-attention</text>
  <line x1="330" y1="212" x2="330" y2="184" class="d-line" marker-end="url(#txarr)"/>
  <rect x="230" y="148" width="200" height="34" rx="7" class="d-box-muted"/>
  <text x="330" y="170" text-anchor="middle" class="d-text-sm">Add &amp; LayerNorm</text>
  <path d="M 445 298 C 520 298, 520 165, 432 165" class="d-line-accent" fill="none" marker-end="url(#txarr)"/>
  <text x="530" y="240" class="d-text-accent">residual skip</text>
  <line x1="330" y1="148" x2="330" y2="120" class="d-line" marker-end="url(#txarr)"/>
  <rect x="230" y="82" width="200" height="36" rx="7" class="d-box"/>
  <text x="330" y="100" text-anchor="middle" class="d-text-sm">feed-forward MLP</text>
  <text x="330" y="113" text-anchor="middle" class="d-text-sm">(d → 4d → d, GELU)</text>
  <line x1="330" y1="82" x2="330" y2="56" class="d-line" marker-end="url(#txarr)"/>
  <rect x="230" y="20" width="200" height="34" rx="7" class="d-box-muted"/>
  <text x="330" y="42" text-anchor="middle" class="d-text-sm">Add &amp; LayerNorm</text>
  <path d="M 218 165 C 130 165, 130 37, 228 37" class="d-line-accent" fill="none" marker-end="url(#txarr)"/>
  <text x="80" y="105" class="d-text-accent">residual skip</text>
  <text x="480" y="45" class="d-text">× N blocks</text>
  <text x="480" y="65" class="d-text-sm">(6 originally; 96 in GPT-3)</text>
</svg>
<div class="caption">One encoder block: attend (tokens talk to each other), then
feed-forward (each token thinks alone) - both wrapped in skip connections and
normalization.</div>
</div>
<p>Read the division of labor: <strong>attention mixes information <em>across</em>
tokens</strong>; the <strong>feed-forward network processes each token
independently</strong> (the same 2-layer MLP applied at every position - where much
of a model's factual "knowledge" is believed to live; it's also ~⅔ of the
parameters). The wrapping you can explain from earlier chapters:
<a href="#/cnn/classic-architectures">residual connections</a> keep gradients
alive through dozens of blocks, and <strong>LayerNorm</strong> - normalizing across
each token's features rather than across the batch - is the batch-independent
cousin of <a href="#/dl/dl-regularization">BatchNorm</a>, chosen precisely
because sequence models hate batch coupling.</p>

<h2>The decoder block: two changes</h2>
<ul>
  <li><strong>Masked self-attention:</strong> when generating, a token may not peek at
      the future. The mask sets scores for positions \(j \gt i\) to \(-\infty\)
      before the softmax → future weights become exactly 0. During training this
      allows the beautiful trick: predict <em>every</em> next-token in a sentence
      simultaneously, in one parallel pass, each position honestly blind to its
      future.</li>
  <li><strong>Cross-attention:</strong> a second attention layer where queries come
      from the decoder but keys/values come from the encoder - Module 9's
      <a href="#/sequence/attention">translation attention</a>, ported verbatim.</li>
</ul>

<h2>The full original machine (and its descendants' surgery)</h2>
<p>The 2017 "Attention Is All You Need" model: encoder stack (6 blocks) reads the
source; decoder stack (6 blocks) generates the target using masked self-attention
+ cross-attention; a final linear + softmax over the vocabulary picks each token.
Trained on translation, it beat the best RNN systems while training in a
fraction of the time - because <em>every position processes in parallel</em>;
no more waiting for step t−1.</p>
<p>The famous split of the family tree
(<a href="#/transformers/bert-gpt">next chapter</a>): encoder-only (BERT -
understanding), decoder-only (GPT - generation), encoder-decoder (T5 -
sequence-to-sequence). All are rearrangements of the one block above.</p>

<h2>Sizing intuition: where the parameters live</h2>
<div class="math-box">
$$\text{per block: } \underbrace{4\,d^2}_{\text{attention (Q,K,V,O)}} + \underbrace{8\,d^2}_{\text{FFN } (d \to 4d \to d)} = 12\,d^2
\qquad
\text{model} \approx 12\, N\, d^2 + \text{embeddings}$$
</div>
<p>Sanity check on GPT-3: N = 96, d = 12288 → 12 × 96 × 12288² ≈ 174B ≈ its
publicized 175B. The scaling recipe of the LLM era is visible in the formula:
grow N and d, feed with data, and - the surprising empirical law - quality keeps
improving predictably.</p>

<h2>In code: a working block, then the real thing</h2>
<pre><code class="language-python">import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, d=256, heads=8, ff=1024, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout,
                                          batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(ff, d))
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)

    def forward(self, x, mask=None):
        a, _ = self.attn(x, x, x, attn_mask=mask)   # self-attention: Q=K=V=x
        x = self.ln1(x + a)                          # residual + norm
        x = self.ln2(x + self.ff(x))                 # residual + norm
        return x

x = torch.randn(4, 50, 256)                          # 4 sequences, 50 tokens
block = TransformerBlock()
print(block(x).shape)                                # (4, 50, 256)

# causal mask for a decoder: upper triangle = -inf (no peeking)
T = 50
causal = torch.triu(torch.full((T, T), float("-inf")), diagonal=1)
print(block(x, mask=causal).shape)

# production: nn.TransformerEncoderLayer / nn.TransformerEncoder stack these
</code></pre>

<h2>Common interview questions</h2>
<div class="qa">
  <div class="qa-q">Q1. Walk through one encoder block.</div>
  <div class="qa-a"><p>Input (embeddings + positions) → multi-head self-attention (tokens exchange
  information) → add the input back (residual) and LayerNorm → position-wise feed-forward
  MLP (each token processed independently, d→4d→d with GELU) → residual + LayerNorm.
  Stack N of these.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q2. What does the causal mask do, and why does it enable parallel training?</div>
  <div class="qa-a"><p>It zeroes (via −∞ pre-softmax) all attention to future positions, so position i's
  representation depends only on tokens ≤ i. Every position can then be trained to predict
  its next token simultaneously in one pass - teacher forcing without sequential
  processing.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q3. Why LayerNorm instead of BatchNorm in Transformers?</div>
  <div class="qa-a"><p>BatchNorm couples examples through batch statistics - awkward with variable-length
  sequences, small batches, and autoregressive inference (batch of 1). LayerNorm normalizes
  each token's own feature vector, identical in training and inference, batch-size
  independent.</p></div>
</div>

<h2>Quick quiz</h2>
<div class="quiz">
  <div class="quiz-q" data-answer="1">
    <p class="quiz-question">1. In a Transformer block, information moves BETWEEN tokens in…</p>
    <button class="quiz-opt">The feed-forward layer</button>
    <button class="quiz-opt">The self-attention layer only</button>
    <button class="quiz-opt">The LayerNorm</button>
    <div class="quiz-explain hidden">FFN and LayerNorm act on each position independently. All cross-token communication happens in attention - which is why masking it controls information flow completely.</div>
  </div>
  <div class="quiz-q" data-answer="2">
    <p class="quiz-question">2. Masked self-attention sets future positions' scores to −∞ so that…</p>
    <button class="quiz-opt">Training is faster</button>
    <button class="quiz-opt">The model uses less memory</button>
    <button class="quiz-opt">After softmax their weights are exactly zero - no information leaks backward from the future</button>
    <div class="quiz-explain hidden">e^(−∞) = 0. Generation must be causal, and training must match that constraint while still parallelizing across positions.</div>
  </div>
  <div class="quiz-q" data-answer="0">
    <p class="quiz-question">3. The Transformer's decisive speed advantage over RNN seq2seq during training is…</p>
    <button class="quiz-opt">All sequence positions are processed in parallel - no step-by-step dependency</button>
    <button class="quiz-opt">Fewer parameters</button>
    <button class="quiz-opt">It skips backpropagation</button>
    <div class="quiz-explain hidden">RNNs must wait for h_{t−1}; attention computes all positions as one matrix product. That parallelism is what made trillion-token training feasible.</div>
  </div>
</div>
`};
