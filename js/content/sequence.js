/* Module 9 - Sequence Models (part 1) */

CONTENT["sequence/why-sequence-models"] = {
  html: String.raw`
<h1>Why Sequence Models</h1>
<p class="lead">"Dog bites man" and "man bites dog" contain identical words.
Order <em>is</em> the meaning - and the architectures we've built so far are
structurally incapable of respecting it. This chapter is about exactly what's
missing and what any fix must provide.</p>

<h2>What makes sequences different</h2>
<p>Text, audio, sensor streams, stock prices, DNA - sequential data has three
properties tabular rows and images don't share:</p>
<ul>
  <li><strong>Order carries meaning:</strong> shuffle a table's rows and nothing
      changes; shuffle a sentence's words and meaning is destroyed.</li>
  <li><strong>Variable length:</strong> sentences have 3 words or 300. MLPs and CNNs
      demand fixed-size inputs.</li>
  <li><strong>Long-range dependencies:</strong> "The <em>keys</em> to the cabinet …
      <em>are</em> on the table" - the verb agrees with a noun 10 words back.
      In "I grew up in France … I speak fluent ___", the answer sits an entire
      paragraph earlier.</li>
</ul>

<h2>Why the MLP fails</h2>
<p>Force a sentence into an MLP and you must fix a length (pad/truncate) and
flatten. Then each position gets <strong>its own private weights</strong> - the
network that learned "not" matters at position 3 knows nothing about "not" at
position 7. It's the same disease CNNs cured for images (position-tied weights,
<a href="#/cnn/why-cnns">no sharing</a>), but along time instead of space.
And a bag-of-words representation (ignore order entirely) literally cannot
distinguish "dog bites man" from "man bites dog".</p>

<h2>Why 1-D CNNs only half-succeed</h2>
<p>Convolutions do share weights across positions - a 1-D CNN sliding over text is
a legitimate sequence tool (fast, parallel, good for local patterns like phrases
and n-grams). But a filter sees a fixed window; to connect "France" with a word
40 positions later you must stack many layers until receptive fields reach that
far. Local: excellent. Arbitrary-distance dependencies: structurally awkward.</p>

<h2>What we actually need: memory</h2>
<p>The natural spec for a sequence processor reads like a description of
<em>reading</em>: process tokens one at a time, in order, while maintaining a
running summary of everything so far; share the same processing weights at every
step; let the summary at step \(t\) influence the processing at step \(t+1\).
That running summary is called a <strong>hidden state</strong> - and adding it to a
neural network gives the <strong>Recurrent Neural Network</strong>, next chapter.</p>
<div class="diagram">
<svg viewBox="0 0 660 250" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Three tasks: many-to-one sentiment, one-to-many captioning, many-to-many translation, shown as sequences of boxes">
  <text x="110" y="35" text-anchor="middle" class="d-text">many → one</text>
  <rect x="30" y="55" width="40" height="34" rx="6" class="d-box-soft"/>
  <rect x="80" y="55" width="40" height="34" rx="6" class="d-box-soft"/>
  <rect x="130" y="55" width="40" height="34" rx="6" class="d-box-soft"/>
  <line x1="100" y1="89" x2="100" y2="115" class="d-line"/>
  <rect x="75" y="118" width="50" height="34" rx="6" class="d-box"/>
  <text x="100" y="140" text-anchor="middle" class="d-text-sm">😊/☹</text>
  <text x="110" y="185" text-anchor="middle" class="d-text-sm">sentiment analysis</text>
  <text x="330" y="35" text-anchor="middle" class="d-text">one → many</text>
  <rect x="290" y="55" width="50" height="34" rx="6" class="d-box"/>
  <text x="315" y="77" text-anchor="middle" class="d-text-sm">🖼</text>
  <line x1="315" y1="89" x2="315" y2="115" class="d-line"/>
  <rect x="250" y="118" width="40" height="34" rx="6" class="d-box-soft"/>
  <rect x="300" y="118" width="40" height="34" rx="6" class="d-box-soft"/>
  <rect x="350" y="118" width="40" height="34" rx="6" class="d-box-soft"/>
  <text x="330" y="185" text-anchor="middle" class="d-text-sm">image captioning</text>
  <text x="545" y="35" text-anchor="middle" class="d-text">many → many</text>
  <rect x="465" y="55" width="40" height="34" rx="6" class="d-box-soft"/>
  <rect x="515" y="55" width="40" height="34" rx="6" class="d-box-soft"/>
  <rect x="565" y="55" width="40" height="34" rx="6" class="d-box-soft"/>
  <line x1="535" y1="89" x2="535" y2="115" class="d-line"/>
  <rect x="465" y="118" width="40" height="34" rx="6" class="d-box"/>
  <rect x="515" y="118" width="40" height="34" rx="6" class="d-box"/>
  <rect x="565" y="118" width="40" height="34" rx="6" class="d-box"/>
  <text x="545" y="185" text-anchor="middle" class="d-text-sm">translation, speech-to-text</text>
  <text x="330" y="225" text-anchor="middle" class="d-text-sm">sequence models must flex across all these input/output shapes - fixed-size architectures can't</text>
</svg>
<div class="caption">The task zoo sequence models must serve. Note how many shapes
"variable length" actually takes.</div>
</div>

<h2>One more must-have: representing words as vectors</h2>
<p>Neural networks eat numbers, not strings. One-hot encoding a 50,000-word
vocabulary gives sparse orthogonal vectors where "cat" and "kitten" are as
unrelated as "cat" and "carburetor". The fix - used by every model from here to
GPT - is <strong>embeddings</strong>: each word gets a learned dense vector
(~100–1000 dims) where similar words end up nearby, trained like any other
weights. The <a href="#/features/dimensionality-reduction">t-SNE plots</a> of
these spaces famously show king − man + woman ≈ queen. In PyTorch it's one line:
<code>nn.Embedding(vocab_size, 300)</code> - remember it; every model in Modules
9–10 starts with it.</p>

<h2>Common interview questions</h2>
<div class="qa">
  <div class="qa-q">Q1. Why can't a standard MLP handle variable-length sequences?</div>
  <div class="qa-a"><p>Its first weight matrix has a fixed input dimension, forcing pad/truncate; worse,
  flattening ties each timestep to private weights, so patterns learned at one position
  don't transfer to another, and parameter count grows with sequence length.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q2. What do 1-D CNNs do well and badly on text?</div>
  <div class="qa-a"><p>Well: local patterns (n-grams), with weight sharing and total parallelism - still
  used for fast text classification. Badly: long-range dependencies, which require stacking
  many layers to grow the receptive field far enough.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q3. Why are learned embeddings better than one-hot vectors for words?</div>
  <div class="qa-a"><p>One-hot vectors are huge, sparse, and all mutually orthogonal - no notion of
  similarity, so nothing learned about "good" transfers to "great". Dense learned embeddings
  place related words near each other, letting the model generalize across vocabulary.</p></div>
</div>

<h2>Quick quiz</h2>
<div class="quiz">
  <div class="quiz-q" data-answer="2">
    <p class="quiz-question">1. "Dog bites man" vs "man bites dog" breaks bag-of-words models because…</p>
    <button class="quiz-opt">The words are too rare</button>
    <button class="quiz-opt">The sentences have different lengths</button>
    <button class="quiz-opt">They contain identical words - only the order differs, which the representation discards</button>
    <div class="quiz-explain hidden">Identical multisets of tokens → identical bag-of-words vectors → identical predictions. Order-awareness is the entire game.</div>
  </div>
  <div class="quiz-q" data-answer="0">
    <p class="quiz-question">2. Sentiment classification of a review is which sequence shape?</p>
    <button class="quiz-opt">Many-to-one</button>
    <button class="quiz-opt">One-to-many</button>
    <button class="quiz-opt">Many-to-many</button>
    <div class="quiz-explain hidden">A whole sequence in, one label out. Translation is many-to-many; captioning is one-to-many.</div>
  </div>
  <div class="quiz-q" data-answer="1">
    <p class="quiz-question">3. The core capability a hidden state adds is…</p>
    <button class="quiz-opt">Faster matrix multiplication</button>
    <button class="quiz-opt">A running memory of everything processed so far, carried between timesteps</button>
    <button class="quiz-opt">Larger vocabulary support</button>
    <div class="quiz-explain hidden">The hidden state is the model's summary-so-far, updated at each step with shared weights - the mechanical definition of "reading in order".</div>
  </div>
</div>
`};

CONTENT["sequence/rnn"] = {
  html: String.raw`
<h1>Recurrent Neural Networks (RNN)</h1>
<p class="lead">An RNN is one small network applied over and over: at each
timestep it combines the new input with its memory of everything before, and
passes the updated memory forward. Elegant, powerful - and haunted by a
gradient problem you can now predict before reading it.</p>

<h2>The recurrence</h2>
<div class="math-box">
$$\mathbf{h}_t = \tanh\big(\mathbf{W}_{xh}\, \mathbf{x}_t + \mathbf{W}_{hh}\, \mathbf{h}_{t-1} + \mathbf{b}\big)
\qquad\qquad
\mathbf{y}_t = \mathbf{W}_{hy}\, \mathbf{h}_t$$
</div>
<p>Read it as a sentence: <em>new memory = a blend of (what I just saw) and (what I
remembered), squashed by tanh.</em> Three weight matrices total - input→hidden,
hidden→hidden, hidden→output - <strong>reused at every timestep</strong>. A
20-word sentence and a 200-word essay use exactly the same parameters; only the
number of steps differs. Weight sharing across <em>time</em>, just as CNNs share
across space.</p>

<h2>Unrolling: the RNN is a deep network in disguise</h2>
<div class="diagram">
<svg viewBox="0 0 660 250" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Left: compact RNN cell with a loop. Right: the same cell unrolled across four timesteps with hidden state flowing left to right">
  <defs>
    <marker id="rnarr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M0,0 L10,5 L0,10 z" style="fill: var(--text-muted)"/>
    </marker>
  </defs>
  <!-- compact -->
  <rect x="40" y="95" width="80" height="56" rx="9" class="d-box"/>
  <text x="80" y="128" text-anchor="middle" class="d-text">RNN</text>
  <path d="M 105 95 C 130 55, 55 55, 68 92" class="d-line-accent" fill="none" marker-end="url(#rnarr)"/>
  <text x="88" y="48" class="d-text-accent">h</text>
  <line x1="80" y1="200" x2="80" y2="153" class="d-line" marker-end="url(#rnarr)"/>
  <text x="80" y="218" text-anchor="middle" class="d-text-sm">xₜ</text>
  <text x="80" y="240" text-anchor="middle" class="d-text-sm">compact view</text>
  <text x="160" y="123" class="d-text" style="font-size:18px">=</text>
  <!-- unrolled -->
  <g>
    <rect x="200" y="95" width="70" height="56" rx="9" class="d-box"/>
    <rect x="320" y="95" width="70" height="56" rx="9" class="d-box"/>
    <rect x="440" y="95" width="70" height="56" rx="9" class="d-box"/>
    <rect x="560" y="95" width="70" height="56" rx="9" class="d-box"/>
  </g>
  <text x="235" y="128" text-anchor="middle" class="d-text-sm">t=1</text>
  <text x="355" y="128" text-anchor="middle" class="d-text-sm">t=2</text>
  <text x="475" y="128" text-anchor="middle" class="d-text-sm">t=3</text>
  <text x="595" y="128" text-anchor="middle" class="d-text-sm">t=4</text>
  <line x1="270" y1="123" x2="317" y2="123" class="d-line-accent" marker-end="url(#rnarr)"/>
  <line x1="390" y1="123" x2="437" y2="123" class="d-line-accent" marker-end="url(#rnarr)"/>
  <line x1="510" y1="123" x2="557" y2="123" class="d-line-accent" marker-end="url(#rnarr)"/>
  <text x="293" y="113" class="d-text-accent">h₁</text>
  <text x="413" y="113" class="d-text-accent">h₂</text>
  <text x="533" y="113" class="d-text-accent">h₃</text>
  <line x1="235" y1="195" x2="235" y2="153" class="d-line" marker-end="url(#rnarr)"/>
  <line x1="355" y1="195" x2="355" y2="153" class="d-line" marker-end="url(#rnarr)"/>
  <line x1="475" y1="195" x2="475" y2="153" class="d-line" marker-end="url(#rnarr)"/>
  <line x1="595" y1="195" x2="595" y2="153" class="d-line" marker-end="url(#rnarr)"/>
  <text x="235" y="215" text-anchor="middle" class="d-text-sm">"I"</text>
  <text x="355" y="215" text-anchor="middle" class="d-text-sm">"love"</text>
  <text x="475" y="215" text-anchor="middle" class="d-text-sm">"this"</text>
  <text x="595" y="215" text-anchor="middle" class="d-text-sm">"movie"</text>
  <text x="420" y="70" text-anchor="middle" class="d-text-sm">SAME weights in every copy - unrolled, it's a 4-layer-deep network through time</text>
</svg>
<div class="caption">One cell, looped - equivalently, a chain as deep as the sequence
is long. That equivalence is both the power and the curse.</div>
</div>
<p>Training runs backprop on the unrolled graph -
<strong>backpropagation through time</strong> (BPTT). Nothing new conceptually:
unroll, apply the <a href="#/dl/backpropagation">chain rule</a>, sum each shared
weight's gradients across all the timesteps where it was used.</p>

<h2>The curse you can now predict</h2>
<p>A 100-word sequence unrolls into a 100-layer-deep network - through the
<em>same</em> matrix every step. The gradient flowing from step 100 back to step 1
gets multiplied ~100 times by \(\mathbf{W}_{hh}^\top \odot \tanh'\). You know
this movie (<a href="#/dl/vanishing-gradients">Vanishing/Exploding
Gradients</a>): factors &lt; 1 → the signal from distant past
<strong>vanishes</strong>; factors &gt; 1 → it <strong>explodes</strong>. It's worse than
in deep MLPs because the same matrix repeats: the behavior is governed by powers
of one matrix, so decay/growth is relentlessly geometric.</p>
<p>Consequences, concretely: a vanilla RNN learns dependencies over ~5–10 steps
well, and essentially cannot connect "France" to "French" forty words later -
the learning signal for that connection arrives ~\(10^{-8}\) times weaker than
for adjacent words. Standard mitigations: <strong>gradient clipping</strong> for the
explosions (the two-line fix), and for the vanishing - a redesigned cell with
protected memory: the <a href="#/sequence/lstm">LSTM, next chapter</a>.</p>

<h2>In code: character-level RNN</h2>
<pre><code class="language-python">import torch
import torch.nn as nn

# tiny character model: predict the next character of a string
text = "hello world, hello deep learning, hello sequences! "
chars = sorted(set(text)); V = len(chars)
c2i = {c: i for i, c in enumerate(chars)}
data = torch.tensor([c2i[c] for c in text])

class CharRNN(nn.Module):
    def __init__(self, vocab, hidden=64):
        super().__init__()
        self.emb = nn.Embedding(vocab, 32)       # chars -> dense vectors
        self.rnn = nn.RNN(32, hidden, batch_first=True)
        self.out = nn.Linear(hidden, vocab)

    def forward(self, x, h=None):
        z = self.emb(x)                          # (B, T) -> (B, T, 32)
        out, h = self.rnn(z, h)                  # out: (B, T, hidden)
        return self.out(out), h                  # logits for every step

model = CharRNN(V)
opt = torch.optim.Adam(model.parameters(), lr=3e-3)
loss_fn = nn.CrossEntropyLoss()

xb = data[:-1].unsqueeze(0)      # input:  all chars but last
yb = data[1:].unsqueeze(0)       # target: all chars shifted by one

for step in range(400):
    opt.zero_grad()
    logits, _ = model(xb)
    loss = loss_fn(logits.view(-1, V), yb.view(-1))
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # ALWAYS clip RNNs
    opt.step()
    if step % 100 == 0:
        print(f"step {step:3d}  loss {loss.item():.3f}")

# generate: feed a seed, sample, feed the sample back in - autoregression
idx = torch.tensor([[c2i["h"]]]); h = None; out = "h"
for _ in range(40):
    logits, h = model(idx, h)
    idx = torch.multinomial(logits[0, -1].softmax(-1), 1).unsqueeze(0)
    out += chars[idx.item()]
print(out)      # after training: "hello world, hello deep..." - it learned!
</code></pre>
<p>That generation loop - predict, sample, feed back - is
<strong>autoregression</strong>, the exact procedure GPT uses (with a vastly better
architecture) to write essays. You've now seen it at its smallest.</p>

<h2>Common interview questions</h2>
<div class="qa">
  <div class="qa-q">Q1. What is backpropagation through time?</div>
  <div class="qa-a"><p>Ordinary backprop applied to the RNN unrolled across timesteps: gradients flow
  backwards through the chain of hidden states, and each shared weight's gradient is the
  sum of its per-timestep contributions. Truncated BPTT caps the unroll length for long
  sequences.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q2. Why do vanilla RNNs vanish worse than equally deep MLPs?</div>
  <div class="qa-a"><p>Each backward step multiplies by the SAME W_hh (times tanh′ ≤ 1). The product across
  T steps behaves like the T-th power of one matrix: whether it dies or explodes is set by
  W_hh's largest singular value, and the effect is geometric in sequence length - no
  layer-by-layer variety to average things out.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q3. Why do RNN embeddings and weights not grow with sequence length?</div>
  <div class="qa-a"><p>The same three matrices are applied at every step - parameters depend only on
  input/hidden sizes. Length changes compute (more steps), not model size - the property
  that makes variable-length input natural.</p></div>
</div>

<h2>Quick quiz</h2>
<div class="quiz">
  <div class="quiz-q" data-answer="1">
    <p class="quiz-question">1. The hidden state hₜ is best described as…</p>
    <button class="quiz-opt">The t-th word's embedding</button>
    <button class="quiz-opt">A learned running summary of the sequence up to step t</button>
    <button class="quiz-opt">The network's output probabilities</button>
    <div class="quiz-explain hidden">It blends the new input with the previous summary each step - the model's working memory, whose contents training shapes to serve the task.</div>
  </div>
  <div class="quiz-q" data-answer="2">
    <p class="quiz-question">2. Processing a 100-token sequence, an RNN uses how many distinct weight matrices?</p>
    <button class="quiz-opt">100 sets, one per step</button>
    <button class="quiz-opt">10 sets</button>
    <button class="quiz-opt">One set, reused at all 100 steps</button>
    <div class="quiz-explain hidden">Weight sharing across time - the same W_xh, W_hh, W_hy everywhere. That's why the unrolled network is deep but not parameter-heavy.</div>
  </div>
  <div class="quiz-q" data-answer="0">
    <p class="quiz-question">3. An RNN language model connects nearby words well but ignores context 50 words back. The mechanism?</p>
    <button class="quiz-opt">Vanishing gradients through 50 repeated multiplications by W_hh</button>
    <button class="quiz-opt">The vocabulary is too small</button>
    <button class="quiz-opt">The learning rate is too high</button>
    <div class="quiz-explain hidden">The learning signal linking distant tokens decays geometrically with distance. Fixing exactly this is the LSTM's reason to exist.</div>
  </div>
</div>
`};

CONTENT["sequence/lstm"] = {
  html: String.raw`
<h1>LSTM - Long Short-Term Memory</h1>
<p class="lead">The LSTM's fix for vanishing gradients is architectural, not
mathematical: give the network a protected memory lane where information flows
by <em>addition</em> instead of repeated multiplication - and install three
learnable gates to control what enters, leaves, and gets erased.</p>

<h2>The core idea: a conveyor belt with gates</h2>
<p>An LSTM cell carries <strong>two</strong> states: the working memory \(\mathbf{h}_t\)
(like the RNN's) and a new <strong>cell state</strong> \(\mathbf{c}_t\) - a conveyor
belt running straight through time. Each step, the belt is updated by
<em>element-wise scaling and adding</em>, never by a full matrix multiply. Three
gates - small sigmoid networks outputting values in (0, 1), "how much to let
through" - regulate the belt:</p>
<div class="math-box">
$$\mathbf{f}_t = \sigma(\mathbf{W}_f [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f)
\qquad \text{forget gate: what to erase from the belt}$$
$$\mathbf{i}_t = \sigma(\mathbf{W}_i [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i)
\qquad
\tilde{\mathbf{c}}_t = \tanh(\mathbf{W}_c [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_c)$$
$$\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} \;+\; \mathbf{i}_t \odot \tilde{\mathbf{c}}_t
\qquad \text{the belt update: erase, then add}$$
$$\mathbf{o}_t = \sigma(\mathbf{W}_o [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o)
\qquad
\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t)$$
</div>
<div class="diagram">
<svg viewBox="0 0 660 300" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="LSTM cell: the cell state runs as a straight horizontal belt across the top, modified by a forget gate multiplication and an input gate addition; the output gate reads from it to produce the hidden state">
  <defs>
    <marker id="lsarr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M0,0 L10,5 L0,10 z" style="fill: var(--text-muted)"/>
    </marker>
  </defs>
  <rect x="20" y="20" width="620" height="260" rx="10" class="d-box-muted"/>
  <!-- belt -->
  <line x1="35" y1="70" x2="625" y2="70" class="d-line-accent" marker-end="url(#lsarr)"/>
  <text x="48" y="58" class="d-text-accent">c_{t−1}</text>
  <text x="585" y="58" class="d-text-accent">c_t</text>
  <text x="330" y="42" text-anchor="middle" class="d-text-sm">the cell-state conveyor belt: only ⊙ and + touch it - no matrix multiplies</text>
  <!-- forget x -->
  <circle cx="200" cy="70" r="16" class="d-box"/>
  <text x="200" y="76" text-anchor="middle" class="d-text">⊙</text>
  <line x1="200" y1="160" x2="200" y2="88" class="d-line" marker-end="url(#lsarr)"/>
  <rect x="160" y="163" width="80" height="36" rx="7" class="d-box-soft"/>
  <text x="200" y="186" text-anchor="middle" class="d-text-sm">forget fₜ</text>
  <!-- add -->
  <circle cx="390" cy="70" r="16" class="d-box"/>
  <text x="390" y="76" text-anchor="middle" class="d-text">+</text>
  <line x1="390" y1="160" x2="390" y2="88" class="d-line" marker-end="url(#lsarr)"/>
  <rect x="320" y="163" width="140" height="36" rx="7" class="d-box-soft"/>
  <text x="390" y="186" text-anchor="middle" class="d-text-sm">input iₜ ⊙ candidate c̃ₜ</text>
  <!-- output -->
  <rect x="500" y="163" width="90" height="36" rx="7" class="d-box-soft"/>
  <text x="545" y="186" text-anchor="middle" class="d-text-sm">output oₜ</text>
  <line x1="545" y1="163" x2="545" y2="118" class="d-line" marker-end="url(#lsarr)"/>
  <line x1="560" y1="70" x2="560" y2="100" class="d-line"/>
  <rect x="520" y="100" width="80" height="0" fill="none"/>
  <text x="590" y="115" class="d-text-sm">tanh(cₜ)</text>
  <text x="620" y="150" class="d-text">hₜ</text>
  <line x1="580" y1="128" x2="612" y2="142" class="d-line" marker-end="url(#lsarr)"/>
  <!-- inputs -->
  <line x1="90" y1="250" x2="90" y2="205" class="d-line" marker-end="url(#lsarr)"/>
  <text x="90" y="268" text-anchor="middle" class="d-text-sm">xₜ and h_{t−1} feed all three gates</text>
  <line x1="120" y1="228" x2="180" y2="200" class="d-grid"/>
  <line x1="150" y1="235" x2="360" y2="200" class="d-grid"/>
  <line x1="180" y1="242" x2="520" y2="200" class="d-grid"/>
</svg>
<div class="caption">The belt (top) is the long-term memory highway. Gates decide what
to erase (⊙ fₜ), what to write (+ iₜ⊙c̃ₜ), and what to reveal (oₜ).</div>
</div>

<h2>Why the gradients survive</h2>
<p>Backpropagate along the belt: \(\partial \mathbf{c}_t / \partial \mathbf{c}_{t-1}
= \mathbf{f}_t\) - element-wise, no \(\mathbf{W}_{hh}\), no tanh′ chain. Where the
network sets a forget gate near 1, the gradient passes essentially intact
across that step; keep it near 1 for 100 steps and signal from step 100 reaches
step 1 alive. It's the same trick as
<a href="#/cnn/classic-architectures">ResNet's skip connections</a> - turn
multiplicative paths into (mostly) additive ones - discovered for time in 1997,
for depth in 2015. (Practical footnote: initialize the forget-gate bias positive
so the network starts life remembering rather than forgetting.)</p>

<h2>Reading the gates as behavior</h2>
<p>On "The keys to the cabinet <em>are</em>…": when the subject "keys" arrives, the
input gate writes plural=true onto the belt; through the prepositional phrase
the forget gates keep it (f ≈ 1) while irrelevant details get erased; at the
verb, the output gate reads the plural feature out to choose "are" over "is".
Nobody programs this - gate weights are ordinary parameters shaped by
backprop - but trained LSTMs demonstrably learn exactly such bookkeeping
(counting, bracket balancing, subject-verb agreement across clauses).</p>

<h2>In code</h2>
<pre><code class="language-python">import torch
import torch.nn as nn

class SentimentLSTM(nn.Module):
    def __init__(self, vocab=20_000, emb=128, hidden=256):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=0)
        self.lstm = nn.LSTM(emb, hidden, num_layers=2,
                            batch_first=True, bidirectional=True,
                            dropout=0.3)
        self.head = nn.Linear(2 * hidden, 1)      # 2x: forward + backward

    def forward(self, x):
        z = self.emb(x)                            # (B, T, emb)
        out, (h, c) = self.lstm(z)                 # h: (layers*2, B, hidden)
        last = torch.cat([h[-2], h[-1]], dim=1)    # final fwd + bwd states
        return self.head(last)                     # logits

model = SentimentLSTM()
x = torch.randint(1, 20_000, (32, 200))            # 32 reviews, 200 tokens
print(model(x).shape)                              # (32, 1)

# parameter check: 4 gate blocks per layer/direction - the LSTM tax
lstm_params = sum(p.numel() for p in model.lstm.parameters())
print(f"LSTM params: {lstm_params/1e6:.1f}M")      # 4x an equivalent RNN
</code></pre>
<p>Two practical notes from the code: <strong>bidirectional</strong> LSTMs run a second
pass right-to-left and concatenate - for tasks where the whole sequence is
available up front (classification, tagging), seeing the future roughly for free;
and the 4× parameter count (four W-blocks: f, i, c̃, o) is the price of the
gates.</p>

<h2>Common interview questions</h2>
<div class="qa">
  <div class="qa-q">Q1. Name the three gates and their jobs.</div>
  <div class="qa-a"><p>Forget gate: scales the previous cell state (what to erase). Input gate: scales the
  candidate update (what new information to write). Output gate: scales tanh(cₜ) (how much
  of the memory to expose as the working hidden state). All are sigmoids conditioned on
  [h_{t−1}, x_t].</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q2. Mechanically, why don't LSTM gradients vanish like RNN gradients?</div>
  <div class="qa-a"><p>The cell-state path's Jacobian is diag(fₜ) - element-wise gates, no repeated
  W_hh multiplication or tanh′ chain. With gates near 1, gradient magnitude is preserved
  across many steps; the network <em>learns</em> where to keep the highway open. (Exploding
  is still possible; clipping stays standard.)</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q3. Why two states, h and c?</div>
  <div class="qa-a"><p>Separation of concerns: c is the protected long-term storage (updated only by
  gated erase/write), h is the task-facing working output (a gated, squashed read of c).
  This lets the model carry information silently for many steps without it distorting
  every intermediate output.</p></div>
</div>

<h2>Quick quiz</h2>
<div class="quiz">
  <div class="quiz-q" data-answer="1">
    <p class="quiz-question">1. A forget gate value of 0.99 for some cell dimension means…</p>
    <button class="quiz-opt">That memory is almost fully erased this step</button>
    <button class="quiz-opt">That memory is almost fully retained this step</button>
    <button class="quiz-opt">The output is 0.99</button>
    <div class="quiz-explain hidden">cₜ = f ⊙ c_{t−1} + …: f multiplies the old memory, so 0.99 = keep. (The name is arguably backwards - it's really a "remember gate".)</div>
  </div>
  <div class="quiz-q" data-answer="0">
    <p class="quiz-question">2. The cell state avoids vanishing gradients because it is updated by…</p>
    <button class="quiz-opt">Element-wise scaling and addition, not repeated matrix multiplication</button>
    <button class="quiz-opt">A larger learning rate</button>
    <button class="quiz-opt">Skipping backpropagation entirely</button>
    <div class="quiz-explain hidden">The belt's step-to-step Jacobian is just the forget gate values - the same "make the path additive" trick as ResNet skips.</div>
  </div>
  <div class="quiz-q" data-answer="2">
    <p class="quiz-question">3. Compared to a vanilla RNN with the same hidden size, an LSTM has roughly…</p>
    <button class="quiz-opt">The same parameters</button>
    <button class="quiz-opt">2× parameters</button>
    <button class="quiz-opt">4× parameters</button>
    <div class="quiz-explain hidden">Four weight blocks (forget, input, candidate, output) where the RNN has one. The gates are worth it - but they're not free.</div>
  </div>
</div>
`};
