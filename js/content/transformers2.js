/* Module 10 - Transformers & Modern Architectures (part 2) */

CONTENT["transformers/bert-gpt"] = {
  html: String.raw`
<h1>BERT / GPT Overview</h1>
<p class="lead">Take the Transformer, keep only half of it, and train it on raw
internet text with a fill-in-the-blank game - that's BERT. Keep the other half
and play guess-the-next-word - that's GPT. Two halves, two philosophies, one
revolution: pretraining.</p>

<h2>The shared insight: self-supervised pretraining</h2>
<p>Labeled data is scarce; text is infinite. Both models manufacture labels
<em>from the text itself</em> (<a href="#/ml/learning-paradigms">self-supervised
learning</a>): hide part of the input and predict it. To do this well at scale,
a model is forced to absorb grammar, facts, style, and reasoning patterns -
because they all help predict hidden words. Pretrain once on billions of words;
then adapt cheaply to any specific task
(<a href="#/transformers/transfer-learning">next chapter</a>).</p>

<h2>BERT: the reader (encoder-only, 2018)</h2>
<p>A stack of Transformer <em>encoder</em> blocks - bidirectional attention, every
token sees the whole sentence. Its pretraining game is
<strong>masked language modeling</strong>: hide 15% of tokens, predict them from
both sides.</p>
<div class="math-box" style="font-family: var(--font-ui); font-size: 15px;">
"The [MASK] sat on the mat and purred." → predict "cat" - using the left
context AND the giveaway "purred" on the right.
</div>
<p>Because it reads bidirectionally, BERT builds superb <em>representations</em> -
but it can't naturally generate text (it fills blanks; it doesn't continue).
Its habitat is <strong>understanding tasks</strong>: classification, named-entity
recognition, extractive question answering, and - in its descendants - the
embedding models behind semantic search and RAG retrieval.</p>

<h2>GPT: the writer (decoder-only, 2018–)</h2>
<p>A stack of Transformer <em>decoder</em> blocks with
<a href="#/transformers/architecture">causal masking</a> - each token sees only
its past. The pretraining game is <strong>next-token prediction</strong>:</p>
<div class="math-box">
$$\max \;\sum_t \log P(w_t \mid w_1, \dots, w_{t-1})$$
</div>
<p>Generation is then <a href="#/sequence/rnn">autoregression</a>, exactly like
your character RNN: predict, sample, append, repeat. The astonishing empirical
discovery of the GPT line: <strong>scale changes kind, not just degree</strong>.
GPT-2 (1.5B) wrote coherent paragraphs; GPT-3 (175B) exhibited
<em>in-context learning</em> - show it a few examples <em>in the prompt</em> and it
performs the task with <strong>no weight updates at all</strong>. Prompting began
to replace fine-tuning as the adaptation mechanism.</p>
<p>One more stage separates a raw language model from a usable assistant:
<strong>alignment</strong> - instruction tuning plus reinforcement learning from
human feedback (RLHF), which shapes "predict plausible next words" into
"be helpful, follow instructions." That's the recipe behind ChatGPT-class
systems, modern LLMs, and assistants like the one you may be reading this
course with.</p>

<h2>Side by side</h2>
<table>
  <tr><th></th><th>BERT</th><th>GPT</th></tr>
  <tr><td>Transformer half</td><td>encoder stack</td><td>decoder stack (causal mask)</td></tr>
  <tr><td>Attention direction</td><td>bidirectional</td><td>left-to-right only</td></tr>
  <tr><td>Pretraining game</td><td>masked-token fill-in</td><td>next-token prediction</td></tr>
  <tr><td>Superpower</td><td>rich representations for understanding</td><td>open-ended generation; in-context learning</td></tr>
  <tr><td>Can't naturally…</td><td>generate fluent continuations</td><td>use right-context when encoding</td></tr>
  <tr><td>Typical use today</td><td>embeddings, search, classifiers</td><td>chat, code, agents - the LLM era</td></tr>
  <tr><td>Family</td><td>RoBERTa, DistilBERT, sentence-transformers</td><td>GPT-2/3/4, Llama, Claude, Gemini…</td></tr>
</table>
<p>(The third branch - encoder-decoder, e.g. T5 - keeps both halves and frames
every task as text-to-text; conceptually it's Module 9's
<a href="#/sequence/seq2seq">seq2seq</a> rebuilt from Transformer blocks.)</p>

<h2>Feel both in ten lines</h2>
<pre><code class="language-python"># pip install transformers
from transformers import pipeline

# BERT-style: fill in the blank (bidirectional context)
unmask = pipeline("fill-mask", model="distilbert-base-uncased")
for r in unmask("The [MASK] sat on the mat and purred.")[:3]:
    print(f"{r['token_str']:>8s}  {r['score']:.3f}")
#     cat  0.6+   - the right-side clue "purred" did its job

# GPT-style: continue the text (causal generation)
gen = pipeline("text-generation", model="gpt2")
print(gen("Machine learning is", max_length=30,
          num_return_sequences=1)[0]["generated_text"])

# BERT-style embeddings: the vectors behind semantic search
from transformers import AutoTokenizer, AutoModel
import torch
tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
enc = AutoModel.from_pretrained("distilbert-base-uncased")
with torch.no_grad():
    vec = enc(**tok("a sentence to embed", return_tensors="pt")
              ).last_hidden_state.mean(1)
print(vec.shape)        # (1, 768) - cosine-compare these for similarity
</code></pre>

<h2>Common interview questions</h2>
<div class="qa">
  <div class="qa-q">Q1. Why can't BERT generate text and why can't GPT look ahead?</div>
  <div class="qa-a"><p>BERT's bidirectional training means each position already "knows" its right
  context - there's no coherent way to sample forward. GPT's causal mask structurally
  forbids attending to future positions, which is exactly what makes autoregressive
  sampling valid. Each design trades one capability for the other.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q2. What is in-context learning?</div>
  <div class="qa-a"><p>Adapting behavior from examples placed in the prompt, with zero gradient updates -
  e.g., three translation pairs followed by a fourth source sentence, and the model
  continues the pattern. An emergent capability of large autoregressive models that turned
  prompting into a programming interface.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q3. Why is next-token prediction such a powerful training signal?</div>
  <div class="qa-a"><p>Predicting the next word well requires whatever helps: syntax, world knowledge,
  arithmetic, narrative logic. The objective is trivially self-supervised (all text is
  labeled by itself), infinitely scalable, and every capability that reduces prediction
  loss gets learned as a side effect.</p></div>
</div>

<h2>Quick quiz</h2>
<div class="quiz">
  <div class="quiz-q" data-answer="1">
    <p class="quiz-question">1. For classifying support tickets by topic with modest data, the natural starting point is…</p>
    <button class="quiz-opt">A GPT model prompted from scratch each time</button>
    <button class="quiz-opt">A BERT-family encoder fine-tuned as a classifier</button>
    <button class="quiz-opt">A vanilla RNN trained from zero</button>
    <div class="quiz-explain hidden">Understanding task + labeled examples = encoder territory: bidirectional context, cheap fine-tuning, small deployable model. (A prompted LLM also works - at higher per-call cost.)</div>
  </div>
  <div class="quiz-q" data-answer="0">
    <p class="quiz-question">2. BERT's masked-LM pretraining differs from GPT's objective in that…</p>
    <button class="quiz-opt">BERT predicts hidden tokens using both directions; GPT predicts the next token using only the left context</button>
    <button class="quiz-opt">BERT uses labeled data; GPT doesn't</button>
    <button class="quiz-opt">GPT is not a Transformer</button>
    <div class="quiz-explain hidden">Fill-in-the-blank vs continue-the-story - both self-supervised, but they produce a reader and a writer respectively.</div>
  </div>
  <div class="quiz-q" data-answer="2">
    <p class="quiz-question">3. "Translate: sea → mer, dog → chien, house → ?" answered correctly by a frozen GPT-3 demonstrates…</p>
    <button class="quiz-opt">Fine-tuning</button>
    <button class="quiz-opt">Masked language modeling</button>
    <button class="quiz-opt">In-context learning - pattern completion from the prompt alone</button>
    <div class="quiz-explain hidden">No weights changed; the "learning" happened inside one forward pass over the prompt. Scale bought a qualitatively new interface.</div>
  </div>
</div>
`};

CONTENT["transformers/transfer-learning"] = {
  html: String.raw`
<h1>Transfer Learning &amp; Fine-tuning</h1>
<p class="lead">Nobody trains language models from scratch anymore - and nobody
should. The modern workflow: take a model pretrained on billions of words,
and adapt it to your task with a few thousand examples. This chapter is the
how, the why, and the menu of options.</p>

<h2>Why transfer works</h2>
<p>Pretraining forces a model to learn reusable substrate: word meanings, syntax,
discourse structure, facts. Your sentiment classifier needs all of that -
and the only task-specific part is a thin mapping from "understood text" to
"positive/negative". Training from scratch re-learns the substrate from your
5,000 examples (impossible); transfer learning inherits it and spends your
5,000 examples on just the thin part. It's
<a href="#/dl/weight-initialization">initialization</a>, perfected: instead of
random weights, you start from knowledge.</p>

<h2>The adaptation menu, cheapest first</h2>
<table>
  <tr><th>Method</th><th>What trains</th><th>Data needed</th><th>When</th></tr>
  <tr><td><strong>Zero/few-shot prompting</strong></td><td>nothing</td><td>0–10 examples in the prompt</td><td>exploration; low volume; strongest available LLM</td></tr>
  <tr><td><strong>Feature extraction</strong></td><td>a small head on frozen embeddings</td><td>hundreds+</td><td>quick classical pipeline: embed → logistic regression</td></tr>
  <tr><td><strong>Head-only fine-tuning</strong></td><td>new final layer(s), body frozen</td><td>hundreds–thousands</td><td>small data; guaranteed not to damage the body</td></tr>
  <tr><td><strong>Full fine-tuning</strong></td><td>every weight, small LR</td><td>thousands+</td><td>best quality when data and compute allow</td></tr>
  <tr><td><strong>PEFT / LoRA</strong></td><td>tiny adapter matrices (~0.1–1% of weights)</td><td>thousands</td><td>full-FT quality at a fraction of memory; the modern default for LLMs</td></tr>
</table>
<p>LoRA deserves its one-paragraph explanation: instead of updating a weight
matrix \(\mathbf{W}\), freeze it and learn a low-rank correction
\(\Delta \mathbf{W} = \mathbf{B}\mathbf{A}\) where \(\mathbf{A}, \mathbf{B}\) are thin
(rank 4–64). The update lives in a tiny parameter budget, trains fast, and can
be merged into \(\mathbf{W}\) afterward - or swapped like a cartridge per task.
(Rank matters because fine-tuning changes are empirically low-rank - the model
mostly re-weights what it already knows.)</p>

<h2>Fine-tuning a BERT classifier, end to end</h2>
<pre><code class="language-python"># pip install transformers datasets
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer)
from datasets import load_dataset
import numpy as np

ds = load_dataset("imdb")                       # 25k labeled movie reviews
tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tok(batch["text"], truncation=True, max_length=256)
ds = ds.map(tokenize, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2)
# ^ pretrained body + a NEW randomly-initialized classification head

args = TrainingArguments(
    output_dir="out",
    learning_rate=2e-5,              # note: ~100x smaller than from-scratch
    num_train_epochs=2,              # 2-4 epochs is typical - it's a NUDGE
    per_device_train_batch_size=16,
    eval_strategy="epoch",
    load_best_model_at_end=True,     # early-stopping flavor
)

def metrics(p):
    return {"acc": (np.argmax(p.predictions, 1) == p.label_ids).mean()}

Trainer(model=model, args=args,
        train_dataset=ds["train"].shuffle(seed=0).select(range(5000)),
        eval_dataset=ds["test"].select(range(2000)),
        compute_metrics=metrics).train()
# ~93% accuracy from 5k examples in minutes -
# from-scratch models need 100x the data to approach this
</code></pre>
<p>The two numbers to respect in that code: <strong>learning rate 2e-5</strong> and
<strong>2 epochs</strong>. Fine-tuning is a nudge, not a re-education - large rates
or long training cause <strong>catastrophic forgetting</strong>: the model overwrites
its pretrained knowledge chasing your small dataset, and generalization
collapses. (Discriminative learning rates - smaller for early layers, larger
for the head - are the refined version of the same caution.)</p>

<h2>Practical wisdom</h2>
<ul>
  <li><strong>Match domains when possible:</strong> legal text → a legal-pretrained
      model; code → a code model. The closer the pretraining distribution, the
      less your data must teach. If no match exists, <em>continued pretraining</em>
      on unlabeled domain text before fine-tuning helps.</li>
  <li><strong>Freeze more when data is small:</strong> the less data, the fewer
      parameters you should unfreeze (head-only → LoRA → full, as data grows).</li>
  <li><strong>Evaluate for forgetting:</strong> test on a general benchmark before and
      after - a fine-tuned model that aced your task but lost basic competence is
      a common, quiet failure.</li>
  <li><strong>The same playbook runs vision:</strong> pretrained ResNet/ViT bodies,
      new heads, small LRs - worked example in
      <a href="#/advanced/cnn-transfer-learning">Transfer Learning in CNNs</a>.</li>
</ul>

<h2>Common interview questions</h2>
<div class="qa">
  <div class="qa-q">Q1. Why is the fine-tuning learning rate ~100× smaller than training from scratch?</div>
  <div class="qa-a"><p>The weights already encode valuable knowledge; large steps would overwrite it
  (catastrophic forgetting). A small LR makes adaptation a local adjustment around a good
  starting point rather than a re-randomization.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q2. Explain LoRA and why it's popular.</div>
  <div class="qa-a"><p>Freeze W; learn ΔW = BA with A, B of tiny rank r, training ~0.1–1% of parameters.
  Memory/computation drop enormously (fine-tune a 7B model on one consumer GPU), quality
  approaches full fine-tuning, and adapters are swappable per task or mergeable into the
  base weights.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q3. When would you choose prompting over fine-tuning?</div>
  <div class="qa-a"><p>Little/no labeled data, fast iteration, low request volume, or task definitions
  that change often. Fine-tuning wins when you have thousands of examples, need a small
  cheap deployable model, require consistent structured behavior, or must run offline.</p></div>
</div>

<h2>Quick quiz</h2>
<div class="quiz">
  <div class="quiz-q" data-answer="2">
    <p class="quiz-question">1. You have 800 labeled examples and a pretrained encoder. Safest effective strategy?</p>
    <button class="quiz-opt">Full fine-tuning at lr = 1e-2</button>
    <button class="quiz-opt">Train a fresh Transformer from scratch</button>
    <button class="quiz-opt">Freeze the body; train only a small classification head (or LoRA)</button>
    <div class="quiz-explain hidden">Tiny data can't safely update millions of weights. Frozen-body approaches spend your 800 examples on the only part that's actually new.</div>
  </div>
  <div class="quiz-q" data-answer="0">
    <p class="quiz-question">2. After aggressive fine-tuning, your model aces the new task but fails simple general questions it used to handle. This is…</p>
    <button class="quiz-opt">Catastrophic forgetting</button>
    <button class="quiz-opt">Vanishing gradients</button>
    <button class="quiz-opt">Data leakage</button>
    <div class="quiz-explain hidden">The small-dataset gradients overwrote pretrained knowledge. Cures: lower LR, fewer epochs, LoRA/frozen layers, mixing in general data.</div>
  </div>
  <div class="quiz-q" data-answer="1">
    <p class="quiz-question">3. LoRA trains ~0.5% of a model's parameters yet nearly matches full fine-tuning because…</p>
    <button class="quiz-opt">The other 99.5% of parameters were unnecessary</button>
    <button class="quiz-opt">Task adaptation is empirically low-rank - mostly re-weighting existing capabilities</button>
    <button class="quiz-opt">It uses a bigger learning rate</button>
    <div class="quiz-explain hidden">Fine-tuning deltas have low intrinsic rank: the model already knows how; adaptation is a small rotation of that knowledge, which thin BA matrices capture.</div>
  </div>
</div>
`};
