/* Module 9 - Sequence Models (part 2) */

CONTENT["sequence/gru"] = {
  html: String.raw`
<h1>GRU - Gated Recurrent Unit</h1>
<p class="lead">The GRU (2014) asked a pragmatic question: how much of the LSTM's
machinery is actually necessary? Its answer - two gates instead of three, one
state instead of two - performs about as well, trains faster, and clarifies
what gating is really doing.</p>

<h2>The simplifications</h2>
<p>Two structural changes relative to the <a href="#/sequence/lstm">LSTM</a>:</p>
<ul>
  <li><strong>Merge the states:</strong> no separate cell state - the hidden state
      \(\mathbf{h}_t\) is both memory and output.</li>
  <li><strong>Couple erase and write:</strong> the LSTM's independent forget and input
      gates become one <strong>update gate</strong> \(\mathbf{z}_t\): whatever fraction
      of the old you keep, the complement is what the new fills. Keep 80% old ⇒
      accept 20% new, per dimension.</li>
</ul>
<div class="math-box">
$$\mathbf{z}_t = \sigma(\mathbf{W}_z[\mathbf{h}_{t-1}, \mathbf{x}_t])
\qquad \text{update gate: how much of the old to keep}$$
$$\mathbf{r}_t = \sigma(\mathbf{W}_r[\mathbf{h}_{t-1}, \mathbf{x}_t])
\qquad \text{reset gate: how much history the candidate may read}$$
$$\tilde{\mathbf{h}}_t = \tanh(\mathbf{W}_h[\mathbf{r}_t \odot \mathbf{h}_{t-1},\; \mathbf{x}_t])$$
$$\mathbf{h}_t = \mathbf{z}_t \odot \mathbf{h}_{t-1} \;+\; (1 - \mathbf{z}_t) \odot \tilde{\mathbf{h}}_t$$
</div>
<p>The last line is the whole story: <strong>new state = weighted blend of old state
and fresh candidate</strong>, with learned per-dimension blending weights. Set
\(z \approx 1\) and information coasts across timesteps nearly untouched - the
same additive gradient highway as the LSTM belt, with one fewer moving part.
The reset gate handles a different job: when starting a new clause or sentence,
it lets the candidate temporarily ignore stale history without erasing it.</p>

<h2>GRU vs LSTM: the practical comparison</h2>
<table>
  <tr><th></th><th>LSTM</th><th>GRU</th></tr>
  <tr><td>Gates</td><td>3 (forget, input, output)</td><td>2 (update, reset)</td></tr>
  <tr><td>States</td><td>2 (cell + hidden)</td><td>1 (hidden)</td></tr>
  <tr><td>Parameters per unit</td><td>4 weight blocks</td><td>3 weight blocks (~25% fewer)</td></tr>
  <tr><td>Speed / memory</td><td>baseline</td><td>~25–30% cheaper</td></tr>
  <tr><td>Accuracy</td><td colspan="2">task-dependent, usually within noise of each other</td></tr>
  <tr><td>Choose it when</td><td>very long dependencies, large data, when in doubt "the classic"</td><td>smaller datasets, tight compute/latency, quicker experiments</td></tr>
</table>
<p>A decade of head-to-heads produced no consistent winner - the honest summary is
<em>"try the GRU first for speed; switch to LSTM if the task seems to need finer
memory control."</em> The deeper lesson: what matters is <strong>gated additive state
updates</strong>, not the exact gate arrangement. Both are instances of one design
pattern - and both were ultimately outflanked by attention (two chapters ahead),
which sidesteps the sequential-memory bottleneck entirely.</p>

<h2>In code: a drop-in swap</h2>
<pre><code class="language-python">import torch
import torch.nn as nn

x = torch.randn(32, 100, 128)          # batch of 32 sequences, 100 steps

lstm = nn.LSTM(128, 256, batch_first=True)
gru  = nn.GRU(128, 256, batch_first=True)

out_l, (h_l, c_l) = lstm(x)            # LSTM returns hidden AND cell state
out_g, h_g        = gru(x)             # GRU returns just the hidden state

print(out_l.shape, out_g.shape)        # both (32, 100, 256) - interchangeable

for name, m in [("LSTM", lstm), ("GRU", gru)]:
    print(f"{name}: {sum(p.numel() for p in m.parameters()):,} params")
# LSTM: 395,264   GRU: 296,448  - the 3/4 ratio, as predicted
</code></pre>

<h2>Common interview questions</h2>
<div class="qa">
  <div class="qa-q">Q1. What did the GRU remove from the LSTM, and what did it keep?</div>
  <div class="qa-a"><p>Removed: the separate cell state and the independent output gate; merged
  forget/input into one update gate (z and 1−z). Kept: the essential mechanism - sigmoid
  gates controlling an additive, element-wise state update that lets gradients survive
  long spans.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q2. Update gate vs reset gate - how do their roles differ?</div>
  <div class="qa-a"><p>Update gate z acts at the blend step: how much old state survives into hₜ
  (long-term retention). Reset gate r acts earlier, inside the candidate: how much history
  the new proposal is allowed to condition on (short-term context switching). Retention vs
  readability - different levers.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q3. When might you actually prefer a GRU over an LSTM?</div>
  <div class="qa-a"><p>Limited data (fewer parameters = less overfitting), tight latency or memory budgets
  (edge devices), or rapid experimentation. With abundant data and very long dependencies,
  LSTMs sometimes edge ahead - but expect differences within noise more often than not.</p></div>
</div>

<h2>Quick quiz</h2>
<div class="quiz">
  <div class="quiz-q" data-answer="2">
    <p class="quiz-question">1. In hₜ = z⊙h_{t−1} + (1−z)⊙h̃ₜ, a dimension with z = 0.95 means…</p>
    <button class="quiz-opt">95% of the new candidate is accepted</button>
    <button class="quiz-opt">The dimension is reset</button>
    <button class="quiz-opt">95% old memory kept, 5% new information blended in</button>
    <div class="quiz-explain hidden">z gates the OLD state; its complement gates the new. Coupling them is the GRU's signature economy over the LSTM's independent gates.</div>
  </div>
  <div class="quiz-q" data-answer="1">
    <p class="quiz-question">2. Relative to an LSTM of equal hidden size, a GRU has…</p>
    <button class="quiz-opt">More parameters</button>
    <button class="quiz-opt">About 25% fewer parameters (3 weight blocks vs 4)</button>
    <button class="quiz-opt">Half the hidden dimensions</button>
    <div class="quiz-explain hidden">Update, reset, candidate = 3 blocks against the LSTM's forget, input, candidate, output = 4.</div>
  </div>
  <div class="quiz-q" data-answer="0">
    <p class="quiz-question">3. The shared principle that lets both LSTM and GRU learn long dependencies is…</p>
    <button class="quiz-opt">Gated additive state updates that give gradients a multiplicative-free path</button>
    <button class="quiz-opt">Having exactly three gates</button>
    <button class="quiz-opt">Using ReLU activations</button>
    <div class="quiz-explain hidden">The specific gate count is negotiable; the additive highway with learned gates is the invariant - the same idea as ResNet skips, applied to time.</div>
  </div>
</div>
`};

CONTENT["sequence/seq2seq"] = {
  html: String.raw`
<h1>Sequence-to-Sequence Models</h1>
<p class="lead">Translation maps one sequence to a <em>different-length</em> other:
5 English words in, 7 German words out. Seq2seq solves it with two RNNs -
one reads, one writes - passing a single summary vector between them.
Its elegant flaw sets up the most important idea of modern AI.</p>

<h2>Encoder–decoder: read everything, then write</h2>
<div class="diagram">
<svg viewBox="0 0 660 260" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Encoder RNN reads the source sentence into a context vector; decoder RNN generates the target sentence token by token from that vector">
  <defs>
    <marker id="s2arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M0,0 L10,5 L0,10 z" style="fill: var(--text-muted)"/>
    </marker>
  </defs>
  <!-- encoder -->
  <text x="150" y="35" text-anchor="middle" class="d-text">encoder (reads)</text>
  <rect x="35" y="60" width="64" height="46" rx="8" class="d-box"/>
  <rect x="115" y="60" width="64" height="46" rx="8" class="d-box"/>
  <rect x="195" y="60" width="64" height="46" rx="8" class="d-box"/>
  <line x1="99" y1="83" x2="112" y2="83" class="d-line" marker-end="url(#s2arr)"/>
  <line x1="179" y1="83" x2="192" y2="83" class="d-line" marker-end="url(#s2arr)"/>
  <text x="67" y="135" text-anchor="middle" class="d-text-sm">"I"</text>
  <text x="147" y="135" text-anchor="middle" class="d-text-sm">"love"</text>
  <text x="227" y="135" text-anchor="middle" class="d-text-sm">"you"</text>
  <!-- context -->
  <line x1="259" y1="83" x2="300" y2="83" class="d-line" marker-end="url(#s2arr)"/>
  <rect x="303" y="58" width="76" height="50" rx="10" class="d-box-soft" style="stroke-width:2"/>
  <text x="341" y="80" text-anchor="middle" class="d-text-accent">context</text>
  <text x="341" y="97" text-anchor="middle" class="d-text-accent">vector c</text>
  <line x1="379" y1="83" x2="420" y2="83" class="d-line" marker-end="url(#s2arr)"/>
  <!-- decoder -->
  <text x="520" y="35" text-anchor="middle" class="d-text">decoder (writes)</text>
  <rect x="423" y="60" width="64" height="46" rx="8" class="d-box"/>
  <rect x="503" y="60" width="64" height="46" rx="8" class="d-box"/>
  <rect x="583" y="60" width="64" height="46" rx="8" class="d-box"/>
  <line x1="487" y1="83" x2="500" y2="83" class="d-line" marker-end="url(#s2arr)"/>
  <line x1="567" y1="83" x2="580" y2="83" class="d-line" marker-end="url(#s2arr)"/>
  <text x="455" y="135" text-anchor="middle" class="d-text-sm">"Ich"</text>
  <text x="535" y="135" text-anchor="middle" class="d-text-sm">"liebe"</text>
  <text x="615" y="135" text-anchor="middle" class="d-text-sm">"dich"</text>
  <!-- autoregressive loop -->
  <path d="M 455 106 C 455 165, 535 165, 535 110" class="d-grid" fill="none" marker-end="url(#s2arr)"/>
  <path d="M 535 106 C 535 165, 615 165, 615 110" class="d-grid" fill="none" marker-end="url(#s2arr)"/>
  <text x="535" y="185" text-anchor="middle" class="d-text-sm">each output word is fed back as the next input (autoregression)</text>
  <text x="330" y="230" text-anchor="middle" class="d-text-sm">the ONLY channel between the two networks is the fixed-size vector c</text>
</svg>
<div class="caption">Two RNNs (LSTMs in practice) with a single vector handshake between
them. Input and output lengths are fully decoupled.</div>
</div>
<p>The <strong>encoder</strong> consumes the source sequence and its final hidden
state becomes the <strong>context vector</strong> - the entire sentence compressed
into, say, 512 numbers. The <strong>decoder</strong> is a conditional language model:
initialized with that context, it generates target tokens one at a time, each
prediction fed back as the next input, until it emits a special
<code>&lt;EOS&gt;</code> (end-of-sequence) token. That stopping mechanism is how a
5-word input can yield a 7-word output.</p>

<h2>Two training tricks worth knowing by name</h2>
<ul>
  <li><strong>Teacher forcing:</strong> during training, feed the decoder the <em>true</em>
      previous token, not its own (possibly wrong) guess - otherwise one early
      mistake derails the whole sequence and learning crawls. The cost: at test
      time the model must ride its own predictions (a train/test mismatch called
      exposure bias).</li>
  <li><strong>Beam search:</strong> at generation time, greedy word-by-word choices can
      paint you into a corner ("greedy" picks the best next word, not the best
      sentence). Beam search keeps the k best partial sentences alive at each step
      - better outputs for a small compute multiple.</li>
</ul>

<h2>The bottleneck: one vector can't hold a paragraph</h2>
<p>Everything the decoder will ever know about the source must squeeze through
that single fixed-size vector. For an 8-word sentence, fine. For a 60-word
sentence, the early words get progressively overwritten by later processing -
translation quality measurably <strong>collapses with source length</strong>
(the famous 2014 curves). It's a suitcase problem: same suitcase, more luggage.</p>
<p>The diagnosis is sharper than "vector too small": <em>why must the decoder view
the whole source through one static summary, when at each output word it really
needs to look at different input words?</em> Translating "dich" the decoder needs
"you", not a blend of everything. Let the decoder <strong>look back at all encoder
states, choosing where to focus per output step</strong> - that's
<a href="#/sequence/attention">attention</a>, the next chapter, and the hinge
of the entire modern era.</p>

<h2>In code: the skeleton</h2>
<pre><code class="language-python">import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, vocab, emb=256, hidden=512):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb)
        self.rnn = nn.LSTM(emb, hidden, batch_first=True)
    def forward(self, src):
        _, (h, c) = self.rnn(self.emb(src))
        return h, c                          # the context (final states)

class Decoder(nn.Module):
    def __init__(self, vocab, emb=256, hidden=512):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb)
        self.rnn = nn.LSTM(emb, hidden, batch_first=True)
        self.out = nn.Linear(hidden, vocab)
    def forward(self, tgt_in, state):
        out, state = self.rnn(self.emb(tgt_in), state)
        return self.out(out), state          # logits per position

enc, dec = Encoder(8000), Decoder(6000)
src    = torch.randint(1, 8000, (16, 12))    # 16 source sentences, 12 tokens
tgt_in = torch.randint(1, 6000, (16, 15))    # shifted target (teacher forcing)

state = enc(src)                             # read → context
logits, _ = dec(tgt_in, state)               # write, conditioned on context
print(logits.shape)                          # (16, 15, 6000)
# loss: CrossEntropyLoss(logits.view(-1, 6000), tgt_out.view(-1))
</code></pre>
<p>Note the moving part count: the entire source-target interface is
<code>state</code> - two tensors of shape (1, 16, 512). That line is the bottleneck
you just read about, in executable form.</p>

<h2>Common interview questions</h2>
<div class="qa">
  <div class="qa-q">Q1. How does seq2seq handle different input and output lengths?</div>
  <div class="qa-a"><p>By decoupling reading from writing: the encoder consumes any-length input into
  fixed-size states; the decoder autoregressively emits tokens until it produces EOS.
  Lengths never need to match because generation length is decided by the decoder's own
  stopping token.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q2. What is teacher forcing and what problem does it create?</div>
  <div class="qa-a"><p>Training the decoder on ground-truth previous tokens instead of its own outputs -
  stabilizes and accelerates learning. It creates exposure bias: at inference the model
  conditions on its own (imperfect) outputs, a distribution it never saw during training,
  so errors can compound.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q3. Why does vanilla seq2seq degrade on long inputs?</div>
  <div class="qa-a"><p>All source information must pass through one fixed-size context vector; capacity
  is constant while content grows, and early tokens are overwritten during encoding.
  Attention removes the bottleneck by letting the decoder access all encoder states
  directly.</p></div>
</div>

<h2>Quick quiz</h2>
<div class="quiz">
  <div class="quiz-q" data-answer="1">
    <p class="quiz-question">1. The decoder knows when to stop generating because…</p>
    <button class="quiz-opt">It counts to the input length</button>
    <button class="quiz-opt">It learns to emit a special end-of-sequence token</button>
    <button class="quiz-opt">The encoder sends a stop signal</button>
    <div class="quiz-explain hidden">EOS is part of the vocabulary and the training targets - generation halts when the model predicts it. That's what frees output length from input length.</div>
  </div>
  <div class="quiz-q" data-answer="0">
    <p class="quiz-question">2. During training with teacher forcing, the decoder's input at step t is…</p>
    <button class="quiz-opt">The ground-truth token t−1</button>
    <button class="quiz-opt">Its own prediction from step t−1</button>
    <button class="quiz-opt">The encoder's final state again</button>
    <div class="quiz-explain hidden">Truth in training, own-predictions at inference - the deliberate mismatch (exposure bias) that keeps training stable.</div>
  </div>
  <div class="quiz-q" data-answer="2">
    <p class="quiz-question">3. Translation quality of bottleneck seq2seq falls with sentence length primarily because…</p>
    <button class="quiz-opt">Longer sentences have rarer words</button>
    <button class="quiz-opt">Beam search fails</button>
    <button class="quiz-opt">A fixed-size vector must represent an arbitrarily long source</button>
    <div class="quiz-explain hidden">Constant capacity, growing content. The 2014 attention paper's headline plot showed exactly this curve - flat with attention, collapsing without.</div>
  </div>
</div>
`};

CONTENT["sequence/attention"] = {
  html: String.raw`
<h1>The Attention Mechanism</h1>
<p class="lead">Instead of forcing the whole source through one vector, let the
decoder <em>look back at everything</em> and learn where to focus for each output
word. This one idea fixed translation, then quietly took over all of deep
learning.</p>

<h2>Intuition first: translation with your finger on the page</h2>
<p>A human translator producing the 5th German word doesn't recall the English
sentence from memory - she <em>glances back</em>, finger on the relevant English
words. Attention gives the decoder that finger: at every output step it assigns
a <strong>weight to each input position</strong> (how relevant is that word right
now?) and reads back a weighted blend. The weights change every step - "Ich"
looks at "I", "liebe" looks at "love".</p>

<h2>The math: soft dictionary lookup</h2>
<p>At decoder step \(t\), with decoder state \(\mathbf{s}_t\) and encoder states
\(\mathbf{h}_1 \dots \mathbf{h}_T\):</p>
<div class="math-box">
$$e_{t,i} = \text{score}(\mathbf{s}_t, \mathbf{h}_i)
\qquad \text{(how relevant is input } i \text{ right now? e.g. } \mathbf{s}_t^\top \mathbf{h}_i)$$
$$\alpha_{t,i} = \frac{\exp(e_{t,i})}{\sum_j \exp(e_{t,j})}
\qquad \text{(softmax} \rightarrow \text{attention weights, summing to 1)}$$
$$\mathbf{c}_t = \sum_i \alpha_{t,i}\, \mathbf{h}_i
\qquad \text{(the context: a weighted average of ALL encoder states)}$$
</div>
<p>Three moves you already own: a <a href="#/math/linear-algebra">dot product</a>
as a similarity score, a <a href="#/dl/activation-functions">softmax</a> to turn
scores into a distribution, and a weighted average. The decoder then predicts
from \([\mathbf{s}_t; \mathbf{c}_t]\) - its own state plus a custom-built summary
for <em>this specific word</em>. Every operation is differentiable, so where to
attend is <strong>learned by backprop</strong> like everything else - no supervision
about alignments is ever given, yet correct word-alignments emerge.</p>
<div class="diagram">
<svg viewBox="0 0 660 290" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Decoder state queries four encoder states; attention weights 0.05, 0.8, 0.1, 0.05 highlight the second input word, producing a weighted context vector">
  <defs>
    <marker id="atarr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M0,0 L10,5 L0,10 z" style="fill: var(--accent)"/>
    </marker>
  </defs>
  <!-- encoder states -->
  <rect x="40" y="40" width="90" height="44" rx="8" class="d-box"/>
  <text x="85" y="67" text-anchor="middle" class="d-text-sm">h₁ "the"</text>
  <rect x="195" y="40" width="90" height="44" rx="8" class="d-box" style="stroke-width:2.5"/>
  <text x="240" y="67" text-anchor="middle" class="d-text-sm">h₂ "cat"</text>
  <rect x="350" y="40" width="90" height="44" rx="8" class="d-box"/>
  <text x="395" y="67" text-anchor="middle" class="d-text-sm">h₃ "sat"</text>
  <rect x="505" y="40" width="90" height="44" rx="8" class="d-box"/>
  <text x="550" y="67" text-anchor="middle" class="d-text-sm">h₄ "down"</text>
  <!-- weights -->
  <text x="85" y="115" text-anchor="middle" class="d-text-sm">α = 0.05</text>
  <text x="240" y="115" text-anchor="middle" class="d-text-accent">α = 0.80</text>
  <text x="395" y="115" text-anchor="middle" class="d-text-sm">α = 0.10</text>
  <text x="550" y="115" text-anchor="middle" class="d-text-sm">α = 0.05</text>
  <!-- arrows to context -->
  <line x1="85" y1="122" x2="295" y2="185" class="d-grid" marker-end="none"/>
  <line x1="240" y1="122" x2="315" y2="182" class="d-line-accent" style="stroke-width:3" marker-end="url(#atarr)"/>
  <line x1="395" y1="122" x2="345" y2="182" class="d-grid"/>
  <line x1="550" y1="122" x2="370" y2="185" class="d-grid"/>
  <rect x="280" y="188" width="110" height="44" rx="9" class="d-box-soft" style="stroke-width:2"/>
  <text x="335" y="215" text-anchor="middle" class="d-text-accent">context cₜ</text>
  <!-- query -->
  <rect x="490" y="188" width="140" height="44" rx="9" class="d-box"/>
  <text x="560" y="209" text-anchor="middle" class="d-text-sm">decoder state sₜ</text>
  <text x="560" y="225" text-anchor="middle" class="d-text-sm">(generating "Katze")</text>
  <line x1="488" y1="210" x2="393" y2="210" class="d-line" marker-end="none" stroke-dasharray="5 4"/>
  <text x="440" y="200" class="d-text-sm">queries</text>
  <text x="330" y="266" text-anchor="middle" class="d-text-sm">cₜ ≈ mostly h₂: the decoder built itself a "cat"-focused summary for this one step</text>
</svg>
<div class="caption">One decoding step. Scores → softmax → weighted average. Next step,
new weights, new focus.</div>
</div>

<h2>Why this was transformative</h2>
<ul>
  <li><strong>The bottleneck dies:</strong> capacity now scales with input length -
      every encoder state remains accessible. Long-sentence translation quality
      stopped collapsing (the flat curve in Bahdanau et al., 2014).</li>
  <li><strong>Gradients take shortcuts:</strong> the path from output word to the
      relevant input word is now <em>one step</em> - not a march through fifty
      recurrent multiplications. Attention is a learned skip connection across
      time (<a href="#/dl/vanishing-gradients">the highway theme</a>, third
      appearance).</li>
  <li><strong>Free interpretability:</strong> plot the α matrix and you get the
      <a href="#/dataviz/paper-visualizations">attention heatmaps</a> -
      word alignments the model learned without ever being told about alignment.</li>
</ul>

<h2>In code: attention in ten lines</h2>
<pre><code class="language-python">import torch
import torch.nn.functional as F

B, T, d = 1, 4, 8                       # batch, source length, state dim
h = torch.randn(B, T, d)                # encoder states h₁..h₄
s = torch.randn(B, d)                   # decoder state at this step

scores  = torch.bmm(h, s.unsqueeze(2)).squeeze(2)   # dot products: (B, T)
alpha   = F.softmax(scores, dim=1)                   # attention weights
context = torch.bmm(alpha.unsqueeze(1), h).squeeze(1)  # Σ αᵢ hᵢ: (B, d)

print("weights:", alpha.squeeze().round(decimals=3))   # sums to 1
print("context:", context.shape)
# the decoder now predicts from [s ; context] - a per-step custom summary
</code></pre>

<h2>The cliffhanger</h2>
<p>By 2017, attention was patching RNNs everywhere. Then a paper asked the
impertinent question in its title - <em>"Attention Is All You Need"</em>:
if attention can connect any position to any other in one step, why keep the
recurrence at all? Delete the RNN, keep only attention (applied by the sequence
<em>to itself</em>), process every position in parallel. That architecture is the
<strong>Transformer</strong> - <a href="#/transformers/self-attention">Module 10</a> -
and the mechanism you just learned is its entire engine.</p>

<h2>Common interview questions</h2>
<div class="qa">
  <div class="qa-q">Q1. Walk through computing one attention context vector.</div>
  <div class="qa-a"><p>Score the decoder state against every encoder state (dot product or a small MLP);
  softmax the scores into weights αᵢ summing to 1; return Σ αᵢhᵢ. Concatenate with the
  decoder state for the output prediction. All differentiable - trained end-to-end with no
  alignment labels.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q2. How does attention fix the seq2seq bottleneck AND the long-range gradient problem?</div>
  <div class="qa-a"><p>Bottleneck: the decoder reads all T encoder states each step, so representational
  capacity grows with input length. Gradients: the weighted-sum path connects any output
  directly to any input in O(1) steps, bypassing the chain of recurrent multiplications
  that vanishes signals.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q3. Are attention weights explanations?</div>
  <div class="qa-a"><p>They show where the model <em>read</em>, which is genuinely informative
  (translation alignments emerge beautifully) - but "attended to" isn't proof of "decided
  because of"; information also flows through the value vectors and other paths. Treat
  heatmaps as evidence, not verdicts.</p></div>
</div>

<h2>Quick quiz</h2>
<div class="quiz">
  <div class="quiz-q" data-answer="1">
    <p class="quiz-question">1. The softmax in attention exists to…</p>
    <button class="quiz-opt">Speed up the computation</button>
    <button class="quiz-opt">Turn arbitrary relevance scores into positive weights that sum to 1</button>
    <button class="quiz-opt">Prevent overfitting</button>
    <div class="quiz-explain hidden">The context must be a convex blend of encoder states - a weighted average needs a proper distribution over positions.</div>
  </div>
  <div class="quiz-q" data-answer="0">
    <p class="quiz-question">2. Unlike the fixed context vector, attention's context cₜ…</p>
    <button class="quiz-opt">Is rebuilt at every output step with fresh focus weights</button>
    <button class="quiz-opt">Is computed once and reused</button>
    <button class="quiz-opt">Has size proportional to input length</button>
    <div class="quiz-explain hidden">Per-step recomputation is the point: each output word gets a summary tailored to what IT needs from the source.</div>
  </div>
  <div class="quiz-q" data-answer="2">
    <p class="quiz-question">3. Attention weights are learned via…</p>
    <button class="quiz-opt">Human-annotated word alignments</button>
    <button class="quiz-opt">A separate alignment model trained first</button>
    <button class="quiz-opt">Ordinary backpropagation - every operation in the mechanism is differentiable</button>
    <div class="quiz-explain hidden">Score → softmax → weighted sum is smooth end-to-end; the translation loss alone teaches the model where to look. The emergent alignments were a famous surprise.</div>
  </div>
</div>
`};
