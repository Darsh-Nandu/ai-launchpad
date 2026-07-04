/* Module 11 - Advanced / Extra Topics */

CONTENT["advanced/autoencoders"] = {
  html: String.raw`
<h1>Autoencoders</h1>
<p class="lead">Train a network to output <em>exactly what it was given</em> -
through a bottleneck too narrow to just copy. To succeed it must discover
compact structure in the data. That accidental-looking trick yields compression,
denoising, and one of the best anomaly detectors in industry.</p>

<h2>The architecture: squeeze, then reconstruct</h2>
<div class="diagram">
<svg viewBox="0 0 660 240" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Hourglass-shaped autoencoder: wide input narrows through the encoder to a small latent code, then widens through the decoder to the reconstruction">
  <polygon points="60,40 200,90 200,150 60,200" class="d-box-soft"/>
  <text x="125" y="125" text-anchor="middle" class="d-text">encoder</text>
  <rect x="285" y="90" width="90" height="60" rx="9" class="d-box" style="stroke-width:2.5"/>
  <text x="330" y="115" text-anchor="middle" class="d-text-accent">latent z</text>
  <text x="330" y="133" text-anchor="middle" class="d-text-sm">(the bottleneck)</text>
  <polygon points="460,90 600,40 600,200 460,150" class="d-box-soft"/>
  <text x="535" y="125" text-anchor="middle" class="d-text">decoder</text>
  <line x1="200" y1="120" x2="283" y2="120" class="d-line"/>
  <line x1="375" y1="120" x2="458" y2="120" class="d-line"/>
  <text x="35" y="120" text-anchor="middle" class="d-text-sm">x</text>
  <text x="628" y="120" text-anchor="middle" class="d-text-sm">x̂</text>
  <text x="330" y="225" text-anchor="middle" class="d-text-sm">loss = ‖x − x̂‖² - reconstruct the input through the squeeze</text>
</svg>
<div class="caption">The hourglass. Input 784 dims → latent 32 dims → output 784 dims:
copying is impossible, so understanding becomes necessary.</div>
</div>
<div class="math-box">
$$\mathbf{z} = f_{enc}(\mathbf{x}) \qquad
\hat{\mathbf{x}} = f_{dec}(\mathbf{z}) \qquad
J = \lVert \mathbf{x} - \hat{\mathbf{x}} \rVert^2$$
</div>
<p>It's <a href="#/ml/learning-paradigms">self-supervised</a>: the data is its own
label. And the concept rhymes with <a href="#/features/dimensionality-reduction">PCA</a>
deliberately - a linear autoencoder with MSE loss learns exactly PCA's subspace;
non-linear layers make it a <em>curved</em> PCA, able to flatten manifolds PCA
can't.</p>

<h2>What the bottleneck buys</h2>
<ul>
  <li><strong>Learned compression / embeddings:</strong> the 32-dim \(\mathbf{z}\) is a
      dense representation of a 784-dim image - usable as features, for
      visualization, or for similarity search.</li>
  <li><strong>Denoising:</strong> train with corrupted inputs but clean targets
      (\(\hat{x} = f(x + \text{noise})\), loss against \(x\)) - the network learns
      to project noisy points back onto the data manifold. This variant also
      learns better features than the vanilla version.</li>
  <li><strong>Anomaly detection</strong> - the industrial killer app, next.</li>
</ul>

<h2>Anomaly detection: the reconstruction-error trick</h2>
<p>Train an autoencoder on <strong>normal data only</strong> (normal transactions,
healthy machine vibrations). It becomes fluent at compressing normality.
Feed it something abnormal - a fraud pattern, a bearing fault - and it
reconstructs poorly, because it never learned a shortcut for that structure:</p>
<div class="math-box">
$$\text{anomaly score}(\mathbf{x}) = \lVert \mathbf{x} - f_{dec}(f_{enc}(\mathbf{x})) \rVert^2
\qquad \text{flag if above a threshold set on normal validation data}$$
</div>
<p>Why this matters strategically: it needs <strong>no anomaly labels at all</strong> -
the answer to <a href="#/features/imbalanced-data">extreme class imbalance</a>
when the rare class is too rare or too novel to label. Fraud, intrusion,
manufacturing defects, medical screening - same recipe.</p>

<h2>In code: anomaly detection on digits</h2>
<pre><code class="language-python">import torch
import torch.nn as nn
from sklearn.datasets import load_digits

X = torch.tensor(load_digits().data, dtype=torch.float32) / 16.0  # (1797, 64)
y = load_digits().target

# train ONLY on "normal" digits (0-8); digit 9 will play the anomaly
normal = X[y != 9]

ae = nn.Sequential(                       # 64 -> 8 -> 64 hourglass
    nn.Linear(64, 32), nn.ReLU(),
    nn.Linear(32, 8),                     # the bottleneck
    nn.ReLU(),
    nn.Linear(8, 32), nn.ReLU(),
    nn.Linear(32, 64), nn.Sigmoid(),
)
opt = torch.optim.Adam(ae.parameters(), lr=1e-3)

for epoch in range(300):
    opt.zero_grad()
    loss = nn.functional.mse_loss(ae(normal), normal)   # reconstruct thyself
    loss.backward(); opt.step()

with torch.no_grad():
    err = ((ae(X) - X) ** 2).mean(dim=1)

print(f"reconstruction error, digits 0-8: {err[y != 9].mean():.4f}")
print(f"reconstruction error, digit 9   : {err[y == 9].mean():.4f}")
# the unseen class reconstructs ~2x worse - threshold on err = detector
</code></pre>

<h2>The generative cousin: VAE (one paragraph you should know)</h2>
<p>A vanilla autoencoder's latent space has holes - decode a random \(\mathbf{z}\)
and you usually get garbage, because training never organized the space between
encodings. The <strong>Variational Autoencoder</strong> fixes this by encoding each
input as a <em>distribution</em> (mean + variance) and penalizing deviation from a
standard normal (a KL term in the loss). Result: a smooth, complete latent space
where sampling and interpolation produce valid data - a true generative model,
and the conceptual ancestor of today's diffusion models. Contrast with
<a href="#/advanced/gans">GANs</a>, next: VAEs learn an explicit compressed
representation (but blurrier samples); GANs learn to fool a critic (sharper
samples, no encoder).</p>

<h2>Common interview questions</h2>
<div class="qa">
  <div class="qa-q">Q1. Why does the bottleneck have to be narrower than the input?</div>
  <div class="qa-a"><p>With latent dimension ≥ input dimension (and capacity to spare), the identity
  function is a perfect, useless solution. The squeeze forces lossy compression, and
  minimizing reconstruction loss under that constraint extracts the data's regularities.
  (Alternatives to narrowness exist - sparsity penalties, denoising - same principle:
  make copying impossible.)</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q2. How do autoencoders detect anomalies without any anomaly labels?</div>
  <div class="qa-a"><p>Trained on normal data only, the model compresses normal structure well; inputs
  that deviate from that structure reconstruct with high error. Score = reconstruction
  error, threshold calibrated on normal validation data. It's one-class learning -
  ideal when anomalies are rare, expensive to label, or novel.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q3. Autoencoder vs PCA?</div>
  <div class="qa-a"><p>A linear AE with MSE recovers the PCA subspace. Non-linear AEs generalize it:
  curved manifolds, arbitrary decoders, task-shaped latents. Costs: no closed form, no
  orthogonality/variance-ranking of components, needs training. PCA for speed and
  interpretability; AEs for expressiveness.</p></div>
</div>

<h2>Quick quiz</h2>
<div class="quiz">
  <div class="quiz-q" data-answer="1">
    <p class="quiz-question">1. An autoencoder's training labels are…</p>
    <button class="quiz-opt">Class labels from annotators</button>
    <button class="quiz-opt">The inputs themselves</button>
    <button class="quiz-opt">Random noise</button>
    <div class="quiz-explain hidden">Reconstruction is self-supervised: x in, x expected out. That's why it runs on unlimited unlabeled data.</div>
  </div>
  <div class="quiz-q" data-answer="0">
    <p class="quiz-question">2. For fraud detection with almost no labeled fraud, the autoencoder approach trains on…</p>
    <button class="quiz-opt">Normal transactions only, flagging high reconstruction error later</button>
    <button class="quiz-opt">Fraud cases only</button>
    <button class="quiz-opt">A balanced 50/50 sample</button>
    <div class="quiz-explain hidden">Model normality; abnormality reveals itself as reconstruction failure. No fraud labels needed - the answer to extreme imbalance.</div>
  </div>
  <div class="quiz-q" data-answer="2">
    <p class="quiz-question">3. Decoding a random latent vector from a vanilla autoencoder usually yields garbage because…</p>
    <button class="quiz-opt">The decoder is too small</button>
    <button class="quiz-opt">Random numbers are illegal inputs</button>
    <button class="quiz-opt">Training only shaped the space AT encodings of real data - the gaps between them are unorganized</button>
    <div class="quiz-explain hidden">Nothing constrained the latent space's geometry between training points. The VAE's KL term fixes exactly this, making the space smooth and sampleable.</div>
  </div>
</div>
`};

CONTENT["advanced/gans"] = {
  html: String.raw`
<h1>GANs - Generative Adversarial Networks</h1>
<p class="lead">Two networks, locked in a game: a Forger creates fake data, a
Detective learns to spot fakes, and each one's improvement forces the other to
improve. At equilibrium, the fakes are indistinguishable from reality.
"The most interesting idea in ML in ten years" - Yann LeCun, 2016.</p>

<h2>The game</h2>
<ul>
  <li><strong>Generator G:</strong> takes random noise \(\mathbf{z}\), outputs a fake
      sample \(G(\mathbf{z})\). Never sees real data directly - its only teacher is
      the detective's verdict.</li>
  <li><strong>Discriminator D:</strong> a plain binary classifier
      (<a href="#/ml/logistic-regression">Module 6 machinery</a>): real or fake?</li>
</ul>
<div class="math-box">
$$\min_G \max_D \;\; \mathbb{E}_{x}\big[\log D(x)\big] + \mathbb{E}_{z}\big[\log(1 - D(G(z)))\big]$$
</div>
<p>Read the minimax aloud: D maximizes its classification success; G minimizes it
(G wants \(D(G(z))\) → 1, "fooled"). Training alternates: a D step (better
detection) then a G step (better forgery), gradients for G flowing
<em>through D</em> - the detective's brain becomes the forger's teacher.
At the theoretical optimum, the generator's distribution equals the data
distribution and D is reduced to coin-flipping (D = 0.5 everywhere).</p>
<div class="diagram">
<svg viewBox="0 0 660 250" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Noise enters the generator producing fakes; fakes and real samples enter the discriminator, whose verdict trains both networks">
  <defs>
    <marker id="gnarr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M0,0 L10,5 L0,10 z" style="fill: var(--text-muted)"/>
    </marker>
  </defs>
  <rect x="25" y="55" width="90" height="46" rx="8" class="d-box-muted"/>
  <text x="70" y="83" text-anchor="middle" class="d-text-sm">noise z</text>
  <line x1="115" y1="78" x2="160" y2="78" class="d-line" marker-end="url(#gnarr)"/>
  <rect x="163" y="55" width="120" height="46" rx="8" class="d-box"/>
  <text x="223" y="83" text-anchor="middle" class="d-text">Generator</text>
  <line x1="283" y1="78" x2="345" y2="105" class="d-line" marker-end="url(#gnarr)"/>
  <text x="300" y="72" class="d-text-sm">fakes</text>
  <rect x="25" y="150" width="120" height="46" rx="8" class="d-box-soft"/>
  <text x="85" y="178" text-anchor="middle" class="d-text-sm">real dataset</text>
  <line x1="145" y1="173" x2="345" y2="135" class="d-line" marker-end="url(#gnarr)"/>
  <text x="230" y="175" class="d-text-sm">reals</text>
  <rect x="348" y="95" width="130" height="52" rx="8" class="d-box"/>
  <text x="413" y="118" text-anchor="middle" class="d-text">Discriminator</text>
  <text x="413" y="136" text-anchor="middle" class="d-text-sm">real or fake?</text>
  <line x1="478" y1="121" x2="540" y2="121" class="d-line" marker-end="url(#gnarr)"/>
  <rect x="543" y="98" width="95" height="46" rx="8" class="d-box-muted"/>
  <text x="590" y="126" text-anchor="middle" class="d-text-sm">verdict</text>
  <path d="M 590 145 C 590 225, 223 225, 223 104" class="d-line-accent" fill="none" marker-end="url(#gnarr)"/>
  <text x="400" y="222" text-anchor="middle" class="d-text-accent">gradients flow back through D to teach G what fooled it</text>
</svg>
<div class="caption">The adversarial loop. G never sees real data - it only learns
from what the improving detective still catches.</div>
</div>

<h2>Why training GANs is famously hard</h2>
<p>You're not descending one loss - you're seeking the equilibrium of a
<em>game</em>, and games can oscillate, collapse, or stall:</p>
<ul>
  <li><strong>Mode collapse:</strong> G finds one output that reliably fools D and
      produces only that (one face, over and over). Diverse data, collapsed
      generator.</li>
  <li><strong>Vanishing generator gradients:</strong> if D becomes too good early,
      \(D(G(z)) \approx 0\) with saturated confidence - G's gradient dies
      (<a href="#/dl/vanishing-gradients">the usual suspect</a>). The standard
      "non-saturating" fix: G maximizes \(\log D(G(z))\) instead of minimizing
      \(\log(1 - D(G(z)))\).</li>
  <li><strong>No honest progress meter:</strong> both losses can hover while quality
      improves - or degrade while losses look stable. Practitioners watch samples
      and metrics like FID, not loss curves.</li>
</ul>
<p>The stabilization literature is vast (DCGAN's architectural rules, Wasserstein
GAN's alternative distance, spectral norm, two-timescale learning rates…);
the summary worth remembering: <strong>balance the two players and give G
gradients it can use.</strong></p>

<h2>A minimal GAN: learning a 1-D distribution</h2>
<pre><code class="language-python">import torch
import torch.nn as nn

real_data = lambda n: torch.randn(n, 1) * 1.5 + 4.0    # target: N(4, 1.5)

G = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 1))
D = nn.Sequential(nn.Linear(1, 32), nn.LeakyReLU(0.2), nn.Linear(32, 1))

opt_g = torch.optim.Adam(G.parameters(), lr=1e-3)
opt_d = torch.optim.Adam(D.parameters(), lr=1e-3)
bce = nn.BCEWithLogitsLoss()

for step in range(3000):
    # --- discriminator step: real->1, fake->0 ---------------------------
    real, noise = real_data(64), torch.randn(64, 8)
    fake = G(noise).detach()               # detach: don't update G here
    d_loss = bce(D(real), torch.ones(64, 1)) + \
             bce(D(fake), torch.zeros(64, 1))
    opt_d.zero_grad(); d_loss.backward(); opt_d.step()

    # --- generator step: make D say "real" ------------------------------
    fake = G(torch.randn(64, 8))           # no detach: gradients flow thru D
    g_loss = bce(D(fake), torch.ones(64, 1))    # non-saturating objective
    opt_g.zero_grad(); g_loss.backward(); opt_g.step()

    if step % 1000 == 0:
        s = G(torch.randn(2000, 8))
        print(f"step {step}: fake mean {s.mean():.2f} std {s.std():.2f} "
              f"(target 4.00 / 1.50)")
# the generator's distribution walks to N(4, 1.5) - no sample ever labeled
</code></pre>
<p>Note the two craft details: <code>detach()</code> in the D step (freeze G's
contribution) and its absence in the G step (that's the whole teaching channel).</p>

<h2>Legacy and today</h2>
<p>GANs powered the 2014–2020 generative image era: StyleGAN's
this-person-does-not-exist faces, pix2pix/CycleGAN translation (sketch→photo,
horse→zebra), super-resolution. Today <strong>diffusion models</strong> (iterative
denoising) have taken the image-generation crown - stabler training, better
diversity - but GANs remain relevant for speed (one forward pass vs many
denoising steps), and adversarial training survives everywhere: as perceptual
losses, in super-resolution, and conceptually in adversarial robustness.
The two-network idea outlived its original throne.</p>

<h2>Common interview questions</h2>
<div class="qa">
  <div class="qa-q">Q1. Explain the GAN objective in plain terms.</div>
  <div class="qa-a"><p>A minimax game: the discriminator maximizes its ability to separate real from
  generated; the generator minimizes that same quantity by making fakes the discriminator
  scores as real. At equilibrium the generated distribution matches the data and D is at
  chance.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q2. What is mode collapse and how is it addressed?</div>
  <div class="qa-a"><p>The generator concentrates on a few outputs that fool the current D, abandoning the
  data's diversity. Mitigations: minibatch discrimination (D sees sample diversity),
  Wasserstein/other losses, unrolled D steps, and diversity-encouraging regularizers.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q3. Why did diffusion models largely displace GANs for image generation?</div>
  <div class="qa-a"><p>A single stable likelihood-style objective (no game to balance), no mode collapse,
  better sample diversity, and easy conditioning. GANs keep the edge in inference speed -
  one pass versus tens of denoising steps - so they persist where latency rules.</p></div>
</div>

<h2>Quick quiz</h2>
<div class="quiz">
  <div class="quiz-q" data-answer="1">
    <p class="quiz-question">1. The generator learns from…</p>
    <button class="quiz-opt">Direct comparison of its output to real images</button>
    <button class="quiz-opt">Gradients flowing back through the discriminator's verdict</button>
    <button class="quiz-opt">A labeled dataset of fakes</button>
    <div class="quiz-explain hidden">G never touches real data. Its loss is D's opinion, differentiated through D's weights - the detective is the curriculum.</div>
  </div>
  <div class="quiz-q" data-answer="0">
    <p class="quiz-question">2. Your face-GAN produces the same three faces regardless of input noise. This is…</p>
    <button class="quiz-opt">Mode collapse</button>
    <button class="quiz-opt">Overfitting the discriminator</button>
    <button class="quiz-opt">Vanishing gradients</button>
    <div class="quiz-explain hidden">The generator found a few reliable fool-the-detective outputs and abandoned diversity - the signature GAN pathology.</div>
  </div>
  <div class="quiz-q" data-answer="2">
    <p class="quiz-question">3. At the theoretical optimum of GAN training, the discriminator outputs…</p>
    <button class="quiz-opt">1 for everything</button>
    <button class="quiz-opt">0 for everything</button>
    <button class="quiz-opt">0.5 for everything - real and fake are indistinguishable</button>
    <div class="quiz-explain hidden">When the generated distribution equals the data distribution, no classifier can do better than chance. The forger has won by making the game unwinnable.</div>
  </div>
</div>
`};

CONTENT["advanced/cnn-transfer-learning"] = {
  html: String.raw`
<h1>Transfer Learning in CNNs</h1>
<p class="lead">A CNN trained on ImageNet's 1.2M photos has already learned edges,
textures, shapes, and object parts. Your 2,000 X-rays or product photos don't
need to reteach any of that - borrow the vision, replace the verdict.</p>

<h2>Why vision transfers so well</h2>
<p>Recall the <a href="#/cnn/why-cnns">hierarchy</a>: early CNN layers learn edges
and color blobs, middle layers textures and patterns, late layers object parts
and task-specific concepts. The early layers are nearly <strong>universal</strong> -
every visual task on Earth needs edge detectors - and universality decays with
depth. That gradient dictates the strategy: <em>keep the general early layers,
adapt the specific late ones.</em></p>

<h2>The two standard recipes</h2>
<h3>Recipe 1 - Feature extraction (small data: ~100s of images)</h3>
<p>Freeze the entire pretrained body; replace the classifier head; train only
that. The CNN becomes a fixed feature machine, your training touches a few
thousand parameters, and overfitting has almost nothing to grab.</p>
<h3>Recipe 2 - Fine-tuning (more data: ~1,000s+)</h3>
<p>Start from recipe 1, then unfreeze some (or all) of the body and continue with
a <strong>much smaller learning rate</strong> (10–100× lower) - the same nudge-don't-
re-educate logic as <a href="#/transformers/transfer-learning">LLM fine-tuning</a>.
Common refinement: leave the earliest blocks frozen (universal features),
tune the later ones (task-specific features).</p>

<h2>In code: ResNet-18 onto a custom task</h2>
<pre><code class="language-python">import torch
import torch.nn as nn
import torchvision.models as models

# ---------- recipe 1: feature extraction ----------------------------------
model = models.resnet18(weights="IMAGENET1K_V1")     # pretrained on 1.2M images

for p in model.parameters():
    p.requires_grad = False                          # freeze the body

model.fc = nn.Linear(model.fc.in_features, 5)        # new 5-class head
# only the head is trainable:
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"training {trainable:,} of {sum(p.numel() for p in model.parameters()):,} params")
# training 2,565 of 11,179,077 - 0.02%!

opt = torch.optim.AdamW(model.fc.parameters(), lr=1e-3)
# ...train a few epochs: usually strong accuracy already...

# ---------- recipe 2: fine-tune the deeper half ----------------------------
for p in model.layer4.parameters():      # unfreeze the last residual stage
    p.requires_grad = True

opt = torch.optim.AdamW([
    {"params": model.layer4.parameters(), "lr": 1e-5},   # gentle on the body
    {"params": model.fc.parameters(),     "lr": 1e-4},   # bolder on the head
])
# ...continue training: typically another 1-3% accuracy...
</code></pre>
<p>Two details that quietly decide success: use the <strong>same preprocessing as
pretraining</strong> (ImageNet models expect 224×224 inputs normalized with
ImageNet's channel means/stds - mismatch silently costs accuracy), and keep
<a href="#/dl/dl-regularization">BatchNorm</a> layers in eval mode when frozen
(their running statistics belong to ImageNet; letting tiny batches of X-rays
overwrite them destabilizes everything).</p>

<h2>How much it helps (typical CIFAR-10-scale numbers)</h2>
<table>
  <tr><th>Approach</th><th>Data needed</th><th>Typical accuracy</th><th>Training time</th></tr>
  <tr><td>Small CNN from scratch (<a href="#/cnn/cnn-project">Module 8's project</a>)</td><td>50k images</td><td>~91%</td><td>30+ epochs</td></tr>
  <tr><td>Frozen ResNet + new head</td><td>works with 1–5k</td><td>~93%</td><td>2–3 epochs</td></tr>
  <tr><td>Fine-tuned ResNet</td><td>10k+</td><td>~95–97%</td><td>5–10 epochs</td></tr>
</table>
<p>The smaller your dataset, the bigger transfer's advantage - at a few hundred
images per class, from-scratch training isn't merely worse, it's hopeless, while
a frozen pretrained body remains perfectly workable. This is why transfer is the
<em>default</em> in applied vision: medical imaging, defect inspection, agriculture,
retail - everyone starts from ImageNet (or increasingly, from self-supervised
and CLIP-style backbones - same recipes, stronger starting points).</p>

<h2>When transfer struggles</h2>
<ul>
  <li><strong>Extreme domain gaps:</strong> spectrograms, radar, histopathology at odd
      magnifications - early layers still help, late layers may mislead; unfreeze
      more, or find a domain-specific pretrained model.</li>
  <li><strong>Tiny images:</strong> ImageNet backbones expect 224×224; on 32×32 inputs
      their aggressive early downsampling destroys detail (adapt the stem, or
      upsample inputs).</li>
  <li><strong>Very different output structures:</strong> detection/segmentation need
      architectural additions (feature pyramids, decoders), not just a new head -
      though the pretrained backbone still does the heavy lifting.</li>
</ul>

<h2>Common interview questions</h2>
<div class="qa">
  <div class="qa-q">Q1. Feature extraction vs fine-tuning - when each?</div>
  <div class="qa-a"><p>Feature extraction (frozen body, new head): very small datasets, fast iteration,
  weak hardware - minimal overfitting risk. Fine-tuning (unfreeze some/all at low LR):
  more data available or domain differs from ImageNet - higher ceiling, higher risk.
  A common path is: extract first, then progressively unfreeze while validation improves.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q2. Why unfreeze the LAST layers rather than the first?</div>
  <div class="qa-a"><p>Depth correlates with specificity: early layers hold universal edge/texture
  detectors useful for any task; late layers hold ImageNet-specific concepts most in need
  of adaptation. Tuning the end adapts the specific while preserving the general.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q3. A frozen pretrained model performs far below expectations. Name two classic causes.</div>
  <div class="qa-a"><p>Preprocessing mismatch (wrong input size or normalization statistics - the model
  sees distributions it never trained on) and BatchNorm mishandling (frozen weights but
  updating running stats, or vice versa). Both are silent - no errors, just degraded
  accuracy.</p></div>
</div>

<h2>Quick quiz</h2>
<div class="quiz">
  <div class="quiz-q" data-answer="1">
    <p class="quiz-question">1. With 300 labeled images per class, the right starting point is…</p>
    <button class="quiz-opt">A fresh ResNet trained from scratch</button>
    <button class="quiz-opt">A frozen pretrained backbone with a new small head</button>
    <button class="quiz-opt">Full fine-tuning at lr = 0.1</button>
    <div class="quiz-explain hidden">Hundreds of images can train thousands of parameters (a head), not millions (a body). Freeze what you can't afford to train.</div>
  </div>
  <div class="quiz-q" data-answer="0">
    <p class="quiz-question">2. When fine-tuning is enabled, the body's learning rate should be…</p>
    <button class="quiz-opt">10–100× smaller than a from-scratch rate - adapt, don't overwrite</button>
    <button class="quiz-opt">10× larger to speed adaptation</button>
    <button class="quiz-opt">Exactly zero</button>
    <div class="quiz-explain hidden">Pretrained weights are the asset; catastrophic forgetting is the risk. Small steps refine the features without erasing them.</div>
  </div>
  <div class="quiz-q" data-answer="2">
    <p class="quiz-question">3. Early CNN layers transfer across tasks better than late layers because…</p>
    <button class="quiz-opt">They have fewer parameters</button>
    <button class="quiz-opt">They train faster</button>
    <button class="quiz-opt">They encode universal primitives (edges, textures) while late layers encode task-specific concepts</button>
    <div class="quiz-explain hidden">Every visual task needs edge detectors; few need ImageNet's "golden retriever" unit. Generality decays with depth - the fact the whole freezing strategy is built on.</div>
  </div>
</div>
`};

CONTENT["advanced/deployment"] = {
  html: String.raw`
<h1>Model Deployment Basics</h1>
<p class="lead">A model in a notebook helps nobody. Deployment is the unglamorous
last mile: save the model <em>and its preprocessing</em>, wrap it in an API,
containerize it, and - the part everyone skips - watch it degrade in production.</p>

<h2>Step 1: save the whole pipeline, not just the model</h2>
<p>The #1 rookie disaster: saving the model but rebuilding preprocessing "the same
way" in the serving code. Any drift between training-time and serving-time
transformations (<em>training-serving skew</em>) silently corrupts predictions.
The fix is structural - make preprocessing part of the artifact:</p>
<pre><code class="language-python">import joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier

# ONE object: scaling + model together (Module 5's pipeline discipline pays off)
pipe = make_pipeline(StandardScaler(), GradientBoostingClassifier())
pipe.fit(X_train, y_train)

joblib.dump(pipe, "model_v1.joblib")             # sklearn: joblib
loaded = joblib.load("model_v1.joblib")
assert (loaded.predict(X_test) == pipe.predict(X_test)).all()

# PyTorch: save weights (state_dict), never the pickled object
# torch.save(model.state_dict(), "model_v1.pt")
# model.load_state_dict(torch.load("model_v1.pt")); model.eval()

# cross-framework / production runtimes: export to ONNX
# torch.onnx.export(model, sample_input, "model.onnx")
</code></pre>
<p>Version everything together: model file, preprocessing, feature list,
training-data snapshot hash, and metrics. "Which model is running and what was
it trained on?" must have a one-line answer.</p>

<h2>Step 2: serve it as an API</h2>
<p>The standard pattern - a small FastAPI service exposing
<code>/predict</code>:</p>
<pre><code class="language-python"># serve.py  ·  run with: uvicorn serve:app --port 8000
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()
pipe = joblib.load("model_v1.joblib")            # load ONCE at startup

class Features(BaseModel):                        # schema = input validation
    age: float
    income: float
    tenure_months: float

@app.post("/predict")
def predict(f: Features):
    X = [[f.age, f.income, f.tenure_months]]
    proba = float(pipe.predict_proba(X)[0, 1])
    return {"churn_probability": round(proba, 4),
            "model_version": "v1"}

@app.get("/health")
def health():
    return {"status": "ok"}                       # for load balancers
</code></pre>
<pre><code class="language-python"># client side - any language, any app:
import requests
r = requests.post("http://localhost:8000/predict",
                  json={"age": 34, "income": 72000, "tenure_months": 18})
print(r.json())        # {"churn_probability": 0.2731, "model_version": "v1"}
</code></pre>
<p>The Pydantic schema is doing real work: malformed requests are rejected with
clear errors instead of becoming NaN predictions. Batch alternative: for
non-interactive use cases (nightly scoring of all customers), skip the API -
a scheduled job writing predictions to a table is simpler and cheaper.</p>

<h2>Step 3: containerize</h2>
<p>"Works on my machine" dies in a container - the same environment everywhere:</p>
<pre><code class="language-python"># Dockerfile
# FROM python:3.11-slim
# WORKDIR /app
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt
# COPY serve.py model_v1.joblib ./
# CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]

# build & run:
#   docker build -t churn-model:v1 .
#   docker run -p 8000:8000 churn-model:v1
</code></pre>
<p>Pin exact library versions in requirements.txt - an unpinned sklearn upgrade
can change prediction behavior or break unpickling entirely.</p>

<h2>Step 4: the part that separates professionals - monitoring</h2>
<p>A deployed model starts dying the day it ships, because <strong>the world
drifts</strong> away from the training data:</p>
<ul>
  <li><strong>Data drift:</strong> input distributions shift (new marketing channel →
      younger users than training ever saw). Detect by comparing live feature
      distributions against training statistics (PSI, KS tests -
      <a href="#/math/hypothesis-testing">Module 1</a>, employed!).</li>
  <li><strong>Concept drift:</strong> the input→output relationship itself changes
      (pandemic rewrites purchasing behavior; fraudsters adapt <em>to your
      model</em>). Detect by tracking live performance once true labels arrive.</li>
</ul>
<p>Minimum viable monitoring: log every request's features and prediction;
dashboard the prediction distribution (a sudden shift in average churn score is
an alarm even before labels arrive); evaluate against ground truth as it becomes
available; and set a retraining trigger (scheduled, or drift-threshold-based).
Plan the retraining pipeline <em>before</em> launch - "how will we update this?"
asked after degradation is firefighting, not engineering.</p>
<div class="callout">
  <span class="co-title">Deployment checklist</span>
  ☐ Preprocessing inside the saved artifact &nbsp;
  ☐ Versioned model + data + metrics &nbsp;
  ☐ Input validation at the API boundary &nbsp;
  ☐ Health endpoint &nbsp;
  ☐ Pinned dependencies, containerized &nbsp;
  ☐ Request/prediction logging &nbsp;
  ☐ Drift dashboards + alert thresholds &nbsp;
  ☐ Retraining plan with owner
</div>

<h2>Common interview questions</h2>
<div class="qa">
  <div class="qa-q">Q1. What is training-serving skew and how do you prevent it?</div>
  <div class="qa-a"><p>Any difference between how features were computed at training time and at serving
  time - reimplemented scalers, different date handling, changed category mappings. Prevent
  it structurally: serialize preprocessing with the model (pipelines), share feature code
  between training and serving, and add prediction-parity tests to CI.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q2. Data drift vs concept drift?</div>
  <div class="qa-a"><p>Data drift: P(X) changes - inputs look different, detectable immediately without
  labels. Concept drift: P(y|X) changes - the world's rules changed, detectable only as
  labels arrive. Both degrade performance; both are hypothesis-testing problems on streams.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q3. Why monitor the prediction distribution when you don't yet have true labels?</div>
  <div class="qa-a"><p>It's the earliest smoke alarm: if the average predicted probability jumps from 0.2
  to 0.5 overnight, either the inputs drifted or a pipeline broke - actionable now, weeks
  before label-based metrics could confirm it.</p></div>
</div>

<h2>Quick quiz</h2>
<div class="quiz">
  <div class="quiz-q" data-answer="1">
    <p class="quiz-question">1. Your API rebuilds StandardScaler statistics from each incoming batch. The bug class is…</p>
    <button class="quiz-opt">Concept drift</button>
    <button class="quiz-opt">Training-serving skew - serving transforms differ from training's fitted transforms</button>
    <button class="quiz-opt">Mode collapse</button>
    <div class="quiz-explain hidden">The scaler's μ/σ were fitted on training data and must be reused verbatim. Recomputing them per batch feeds the model differently-scaled features than it learned on.</div>
  </div>
  <div class="quiz-q" data-answer="0">
    <p class="quiz-question">2. Fraudsters changing tactics specifically to evade your deployed model is…</p>
    <button class="quiz-opt">Concept drift (adversarial flavor) - P(y|X) changed</button>
    <button class="quiz-opt">Data drift only</button>
    <button class="quiz-opt">A reason accuracy can be checked less often</button>
    <div class="quiz-explain hidden">The relationship between features and fraud changed - the harshest form, since the world is optimizing against you. Adversarial domains need the fastest retraining loops.</div>
  </div>
  <div class="quiz-q" data-answer="2">
    <p class="quiz-question">3. For scoring all 2M customers every night (no real-time need), the right serving shape is…</p>
    <button class="quiz-opt">A GPU-backed REST API</button>
    <button class="quiz-opt">A websocket stream</button>
    <button class="quiz-opt">A scheduled batch job writing scores to a table</button>
    <div class="quiz-explain hidden">Match the serving pattern to the consumption pattern. Batch is simpler, cheaper, and trivially retryable; APIs earn their complexity only when requests are interactive.</div>
  </div>
</div>
`};
