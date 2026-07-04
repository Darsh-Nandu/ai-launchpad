/* Module 8 - Convolutional Neural Networks (part 2) */

CONTENT["cnn/pooling"] = {
  html: String.raw`
<h1>Pooling Layers</h1>
<p class="lead">After convolution finds patterns, pooling summarizes them: shrink
each small neighborhood to one number and keep only the strongest evidence.
Less resolution, more robustness, zero parameters.</p>

<h2>Max pooling: keep the loudest signal</h2>
<p>Slide a window (typically 2×2, stride 2) over each feature map and keep the
<strong>maximum</strong> in each window - the map halves in width and height:</p>
<div class="diagram">
<svg viewBox="0 0 640 240" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="A 4 by 4 grid divided into four 2 by 2 blocks, each reduced to its maximum in a 2 by 2 output">
  <text x="160" y="35" text-anchor="middle" class="d-text">4×4 feature map</text>
  <g>
    <rect x="60" y="50" width="200" height="160" class="d-box"/>
    <line x1="160" y1="50" x2="160" y2="210" class="d-line"/>
    <line x1="60" y1="130" x2="260" y2="130" class="d-line"/>
    <line x1="110" y1="50" x2="110" y2="210" class="d-grid"/><line x1="210" y1="50" x2="210" y2="210" class="d-grid"/>
    <line x1="60" y1="90" x2="260" y2="90" class="d-grid"/><line x1="60" y1="170" x2="260" y2="170" class="d-grid"/>
  </g>
  <text x="80" y="76" class="d-text-sm">1</text><text x="130" y="76" class="d-text-sm">3</text>
  <text x="180" y="76" class="d-text-sm">2</text><text x="230" y="76" class="d-text-sm">0</text>
  <text x="80" y="116" class="d-text-sm">5</text><text x="130" y="116" class="d-text-sm">2</text>
  <text x="180" y="116" class="d-text-sm">1</text><text x="230" y="116" class="d-text-sm">7</text>
  <text x="80" y="156" class="d-text-sm">0</text><text x="130" y="156" class="d-text-sm">1</text>
  <text x="180" y="156" class="d-text-sm">4</text><text x="230" y="156" class="d-text-sm">2</text>
  <text x="80" y="196" class="d-text-sm">3</text><text x="130" y="196" class="d-text-sm">2</text>
  <text x="180" y="196" class="d-text-sm">1</text><text x="230" y="196" class="d-text-sm">1</text>
  <text x="305" y="130" class="d-text" style="font-size:20px">→</text>
  <text x="450" y="35" text-anchor="middle" class="d-text">2×2 output (max of each block)</text>
  <g>
    <rect x="370" y="70" width="160" height="120" class="d-box-soft"/>
    <line x1="450" y1="70" x2="450" y2="190" class="d-line"/>
    <line x1="370" y1="130" x2="530" y2="130" class="d-line"/>
  </g>
  <text x="405" y="105" text-anchor="middle" class="d-text">5</text>
  <text x="490" y="105" text-anchor="middle" class="d-text">7</text>
  <text x="405" y="165" text-anchor="middle" class="d-text">3</text>
  <text x="490" y="165" text-anchor="middle" class="d-text">4</text>
  <text x="450" y="215" text-anchor="middle" class="d-text-sm">"the pattern fired somewhere in this block - that's all we keep"</text>
</svg>
<div class="caption">2×2 max pooling, stride 2: 75% of the values are discarded;
the strongest detection per region survives.</div>
</div>
<p>Why maximum? A feature map value means "my pattern is here this strongly"
(<a href="#/cnn/convolution">the similarity reading</a>). For downstream layers,
the question is usually <em>"did the pattern occur in this region?"</em> - and the
max answers exactly that, discarding where precisely. That deliberate blurring
buys <strong>local translation invariance</strong>: shift the input a pixel or two
and the max in each window usually doesn't change, so the network's judgment
doesn't either.</p>

<h2>What pooling buys, and what it costs</h2>
<ul>
  <li><strong>Compute &amp; memory:</strong> a 2×2/stride-2 pool cuts activations 4× -
      every later layer works on a quarter of the data.</li>
  <li><strong>Growing receptive fields:</strong> after pooling, the next 3×3 conv
      spans twice the original pixels - hierarchy accelerates.</li>
  <li><strong>Zero parameters:</strong> nothing to learn, nothing to overfit.</li>
  <li><strong>Cost - spatial precision:</strong> exact locations are gone. For
      classification ("is there a cat?") that's fine; for segmentation/detection
      ("which pixels are cat?") architectures must work around it (skip
      connections, dilated convs).</li>
</ul>
<p>Variants: <strong>average pooling</strong> (smooth summary rather than peak -
common historically, and as the final layer); <strong>global average pooling</strong>
(average each feature map to a single number - turns H×W×512 into 512 values,
replacing giant flatten+dense layers and their millions of parameters; standard
in ResNet and after). During backprop, max pooling routes the gradient only to
the winning cell - the perfect analogue of ReLU's gating.</p>

<h2>In code</h2>
<pre><code class="language-python">import torch
import torch.nn as nn

x = torch.tensor([[[[1., 3., 2., 0.],
                    [5., 2., 1., 7.],
                    [0., 1., 4., 2.],
                    [3., 2., 1., 1.]]]])          # (1, 1, 4, 4)

print(nn.MaxPool2d(2)(x).squeeze())
# tensor([[5., 7.],
#         [3., 4.]])   - matches the diagram

print(nn.AvgPool2d(2)(x).squeeze())
# tensor([[2.75, 2.50],
#         [1.50, 2.00]])

# global average pooling: the modern classifier head
feat = torch.randn(8, 512, 7, 7)                  # batch of feature stacks
gap = nn.AdaptiveAvgPool2d(1)(feat).flatten(1)    # -> (8, 512)
print(gap.shape)                                   # feeds one small Linear
</code></pre>

<h2>Pooling vs strided convolution</h2>
<p>Both halve the map. Max pooling is fixed and parameter-free; a stride-2
convolution <em>learns</em> its downsampling. Modern practice mixes freely:
VGG-era nets pool after conv blocks; ResNet mostly strides; either way the
architecture rhythm survives - <em>convolve, activate, downsample, double the
channels, repeat</em>.</p>

<h2>Common interview questions</h2>
<div class="qa">
  <div class="qa-q">Q1. Why max rather than average in hidden layers?</div>
  <div class="qa-a"><p>Feature maps are detection scores; the max preserves "the pattern occurred here"
  even if it fired at one pixel, while averaging dilutes a strong single detection with
  surrounding zeros. Average pooling suits the final global summary, where overall presence
  across the map is the question.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q2. How does gradient flow through max pooling?</div>
  <div class="qa-a"><p>The gradient passes untouched to the input cell that won the max; the other cells
  in the window get zero (they didn't influence the output). Same winner-take-all gating
  logic as ReLU's derivative.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q3. What problem does global average pooling solve?</div>
  <div class="qa-a"><p>The old flatten-to-dense head (e.g. 7×7×512 = 25,088 → 4096) held most of a
  network's parameters and overfit badly. GAP reduces each map to one number: 512 values,
  zero parameters, and input-size flexibility as a bonus.</p></div>
</div>

<h2>Quick quiz</h2>
<div class="quiz">
  <div class="quiz-q" data-answer="1">
    <p class="quiz-question">1. A 2×2 max pool (stride 2) on a 16×16×64 stack outputs…</p>
    <button class="quiz-opt">8×8×32</button>
    <button class="quiz-opt">8×8×64</button>
    <button class="quiz-opt">16×16×32</button>
    <div class="quiz-explain hidden">Pooling shrinks spatial axes only - each of the 64 maps is pooled independently. Channels never change.</div>
  </div>
  <div class="quiz-q" data-answer="0">
    <p class="quiz-question">2. Pooling contributes translation invariance because…</p>
    <button class="quiz-opt">Small shifts rarely change which value is the max within a window</button>
    <button class="quiz-opt">It adds more parameters to absorb shifts</button>
    <button class="quiz-opt">It enlarges the image</button>
    <div class="quiz-explain hidden">The exact position is deliberately forgotten; only regional presence survives, so ±1-pixel jitter leaves the output (mostly) unchanged.</div>
  </div>
  <div class="quiz-q" data-answer="2">
    <p class="quiz-question">3. How many learnable parameters does a MaxPool2d(2) layer add?</p>
    <button class="quiz-opt">4 per channel</button>
    <button class="quiz-opt">Depends on input size</button>
    <button class="quiz-opt">Zero</button>
    <div class="quiz-explain hidden">Pooling is a fixed function - nothing is learned, nothing can overfit. Its "cost" is only the discarded spatial precision.</div>
  </div>
</div>
`};

CONTENT["cnn/classic-architectures"] = {
  html: String.raw`
<h1>Classic Architectures (LeNet → ResNet)</h1>
<p class="lead">Five networks tell the whole story of deep learning for vision -
each one a response to the previous one's limits, and together the reason
"deep" stopped being a dream and became infrastructure.</p>

<h2>LeNet-5 (1998): the proof of concept</h2>
<p>Yann LeCun's digit reader for bank checks: two conv+pool stages, then small
dense layers - ~60k parameters. It established the template every successor
follows (<em>convolve, pool, repeat, classify</em>) and worked brilliantly on
28×28 digits. It couldn't scale further: no GPUs, no big datasets, and
sigmoid/tanh activations that <a href="#/dl/vanishing-gradients">strangled
gradients</a> in deeper stacks.</p>

<h2>AlexNet (2012): the big bang</h2>
<p>Won ImageNet 2012 by a margin so large (top-5 error 16% vs 26% for the
runner-up) that computer vision changed direction within a year. Architecturally
LeNet scaled up - 8 layers, 60M parameters - but three enablers mattered:
<strong>ReLU</strong> (trained 6× faster than tanh, gradients survived),
<strong>GPUs</strong> (two GTX 580s made a two-week training run feasible), and
<strong>dropout</strong> (60M parameters without total overfitting). Every deep
learning technique in Module 7 has a "popularized by AlexNet" footnote.</p>

<h2>VGG-16 (2014): simplicity as a principle</h2>
<p>One idea executed with discipline: <strong>only 3×3 convolutions</strong>
(<a href="#/cnn/filters-padding-stride">two of them see 5×5, cheaper and
bendier</a>), stacked 16 layers deep in a clean rhythm - 64, 128, 256, 512
channels, halving space as channels double. Beautifully uniform, still a common
teaching architecture and feature extractor; hopelessly heavy by modern
standards (138M parameters, ~90% of them in the old-style dense head that
<a href="#/cnn/pooling">global average pooling</a> later abolished).</p>

<h2>Inception/GoogLeNet (2014): why choose a filter size?</h2>
<p>At each layer, run 1×1, 3×3, 5×5 convs <em>and</em> pooling in parallel and
concatenate the results - let the network pick its scale per feature.
The trick making it affordable: <strong>1×1 bottlenecks</strong> compress channels
before the expensive 3×3/5×5 ops. 22 layers deep with 12× <em>fewer</em>
parameters than AlexNet - proof that architecture beats brute size.</p>

<h2>ResNet (2015): the skip connection that unlocked depth</h2>
<p>The paradox that motivated it: a 56-layer plain CNN had <em>worse training
error</em> than its 20-layer version - not overfitting, an optimization failure
(<a href="#/dl/vanishing-gradients">the product-of-factors disease</a>).
The fix is two lines of algebra:</p>
<div class="math-box">
$$\text{plain block: } \mathbf{y} = F(\mathbf{x})
\qquad\qquad
\text{residual block: } \mathbf{y} = F(\mathbf{x}) + \mathbf{x}$$
</div>
<div class="diagram">
<svg viewBox="0 0 640 230" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="A residual block: input x flows through two conv layers producing F of x, while a skip connection carries x around them; the two are added">
  <defs>
    <marker id="rsarr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M0,0 L10,5 L0,10 z" style="fill: var(--text-muted)"/>
    </marker>
  </defs>
  <text x="60" y="120" class="d-text">x</text>
  <line x1="80" y1="115" x2="135" y2="115" class="d-line" marker-end="url(#rsarr)"/>
  <rect x="138" y="90" width="130" height="50" rx="8" class="d-box"/>
  <text x="203" y="112" text-anchor="middle" class="d-text-sm">conv 3×3 · BN · ReLU</text>
  <text x="203" y="128" text-anchor="middle" class="d-text-sm">conv 3×3 · BN</text>
  <line x1="268" y1="115" x2="330" y2="115" class="d-line" marker-end="url(#rsarr)"/>
  <text x="295" y="105" class="d-text-sm">F(x)</text>
  <circle cx="352" cy="115" r="18" class="d-box-soft"/>
  <text x="352" y="121" text-anchor="middle" class="d-text" style="font-size:17px">+</text>
  <!-- skip path -->
  <path d="M 92 115 C 92 40, 352 40, 352 96" class="d-line-accent" fill="none" marker-end="url(#rsarr)"/>
  <text x="220" y="35" text-anchor="middle" class="d-text-accent">identity skip: x rides around the block</text>
  <line x1="370" y1="115" x2="430" y2="115" class="d-line" marker-end="url(#rsarr)"/>
  <text x="440" y="120" class="d-text">ReLU(F(x) + x)</text>
  <text x="320" y="200" text-anchor="middle" class="d-text-sm">the block learns the RESIDUAL - the correction to apply on top of x</text>
</svg>
<div class="caption">The residual block. If the best thing to do is nothing, learning
F = 0 is easy - and gradients always have the identity path home.</div>
</div>
<p>Two readings of why it works. <strong>Learning:</strong> layers now learn a
<em>correction</em> to their input rather than a full transformation - and "no
correction" (F = 0) is trivially learnable, so extra depth can never hurt
training. <strong>Gradients:</strong> backward through \(x + F(x)\) includes the
derivative of the identity - a clean 1 - so error signals reach layer 1 of a
152-layer network undamaged. ResNet-152 won ImageNet 2015 (3.6% top-5, past
human-level), and skip connections became universal - Transformers
(<a href="#/transformers/architecture">Module 10</a>) are full of them.</p>

<h2>The scoreboard</h2>
<table>
  <tr><th>Network</th><th>Year</th><th>Depth</th><th>Params</th><th>Contribution</th></tr>
  <tr><td>LeNet-5</td><td>1998</td><td>5</td><td>60K</td><td>the conv-pool-dense template</td></tr>
  <tr><td>AlexNet</td><td>2012</td><td>8</td><td>60M</td><td>ReLU + GPU + dropout; the breakthrough</td></tr>
  <tr><td>VGG-16</td><td>2014</td><td>16</td><td>138M</td><td>all-3×3 discipline; depth via small filters</td></tr>
  <tr><td>Inception</td><td>2014</td><td>22</td><td>5M</td><td>multi-scale branches; 1×1 bottlenecks</td></tr>
  <tr><td>ResNet-152</td><td>2015</td><td>152</td><td>60M</td><td>skip connections; depth without limit</td></tr>
</table>
<p>Since then: EfficientNet (principled width/depth/resolution scaling),
MobileNet (depthwise convs for phones), ConvNeXt (CNNs modernized with
Transformer-era tricks), and Vision Transformers (Module 10's ideas applied to
pixels). But the five above contain every load-bearing concept.</p>
<pre><code class="language-python"># all of them, one import away - and pretrained:
import torchvision.models as models

resnet = models.resnet18(weights="IMAGENET1K_V1")
vgg    = models.vgg16(weights="IMAGENET1K_V1")

n = sum(p.numel() for p in resnet.parameters())
print(f"resnet18: {n/1e6:.1f}M params")
print(resnet.layer1[0])        # inspect a real residual block
# using these as feature extractors = transfer learning (Module 11)
</code></pre>

<h2>Common interview questions</h2>
<div class="qa">
  <div class="qa-q">Q1. What exactly did skip connections fix?</div>
  <div class="qa-a"><p>The degradation problem: deeper plain nets trained <em>worse</em> (higher training
  error - optimization, not overfitting). Skips give gradients an identity path (derivative
  1) through any depth, and reframe each block as learning an easy-to-zero residual, so
  added layers can't hurt.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q2. Why did AlexNet succeed in 2012 when similar ideas existed in 1998?</div>
  <div class="qa-a"><p>Three missing ingredients arrived: data (ImageNet's 1.2M labeled images), compute
  (consumer GPUs ≈ 100× speedup), and training fixes (ReLU against vanishing gradients,
  dropout against overfitting 60M parameters). The idea was old; the ecosystem was new.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q3. Where do VGG-16's 138M parameters mostly live, and what replaced that design?</div>
  <div class="qa-a"><p>~120M sit in the dense head (flattened 7×7×512 → 4096 → 4096). Global average
  pooling (Inception onward) reduced each map to one value, shrinking heads to near zero and
  cutting overfitting - the dense-heavy design died with it.</p></div>
</div>

<h2>Quick quiz</h2>
<div class="quiz">
  <div class="quiz-q" data-answer="2">
    <p class="quiz-question">1. A ResNet block computes y = F(x) + x. If the optimal behavior is "change nothing", the block must learn…</p>
    <button class="quiz-opt">F(x) = x</button>
    <button class="quiz-opt">F(x) = 2x</button>
    <button class="quiz-opt">F(x) = 0 - push all weights toward zero, which is easy</button>
    <div class="quiz-explain hidden">Identity comes free via the skip; the layers only model the correction. A plain block would have to learn the full identity transform through non-linearities - surprisingly hard.</div>
  </div>
  <div class="quiz-q" data-answer="0">
    <p class="quiz-question">2. The chronological order of breakthroughs is…</p>
    <button class="quiz-opt">LeNet → AlexNet → VGG/Inception → ResNet</button>
    <button class="quiz-opt">AlexNet → LeNet → ResNet → VGG</button>
    <button class="quiz-opt">ResNet → VGG → AlexNet → LeNet</button>
    <div class="quiz-explain hidden">1998 → 2012 → 2014 → 2015. Fourteen quiet years, then three explosive ones - the gap was data and GPUs, not ideas.</div>
  </div>
  <div class="quiz-q" data-answer="1">
    <p class="quiz-question">3. Inception's 1×1 convolutions before the 3×3/5×5 branches exist to…</p>
    <button class="quiz-opt">Add spatial context</button>
    <button class="quiz-opt">Compress channels first, making the expensive convolutions affordable</button>
    <button class="quiz-opt">Replace pooling</button>
    <div class="quiz-explain hidden">A 5×5 conv on 256 channels is brutal; on 64 bottlenecked channels it's cheap. Same trick powers ResNet-50's bottleneck blocks.</div>
  </div>
</div>
`};

CONTENT["cnn/cnn-project"] = {
  html: String.raw`
<h1>CNN Code Walkthrough: Image Classification</h1>
<p class="lead">Everything in this module, assembled: a compact VGG-style CNN
trained on CIFAR-10 (60,000 tiny photos, 10 classes) with augmentation,
batch norm, early stopping, and a confusion-matrix debrief. This is the
complete recipe you'll adapt for real projects.</p>

<h2>1 · Data, with augmentation</h2>
<pre><code class="language-python">import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T

# augmentation = free regularization (Module 7): each epoch sees new variants
train_tf = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize((0.49, 0.48, 0.45), (0.25, 0.24, 0.26)),  # channel μ, σ
])
test_tf = T.Compose([T.ToTensor(),
                     T.Normalize((0.49, 0.48, 0.45), (0.25, 0.24, 0.26))])

train_ds = torchvision.datasets.CIFAR10("data", train=True,  download=True, transform=train_tf)
test_ds  = torchvision.datasets.CIFAR10("data", train=False, download=True, transform=test_tf)
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2)
test_dl  = torch.utils.data.DataLoader(test_ds,  batch_size=256)
classes = train_ds.classes   # plane, car, bird, cat, deer, dog, frog, horse, ship, truck
</code></pre>
<p>Note: augmentation on <em>training only</em> (the test set must stay fixed), and
normalization statistics computed from training data - the
<a href="#/features/scaling">fit-on-train rule</a>, in vision costume.</p>

<h2>2 · The model: three conv blocks, GAP head</h2>
<pre><code class="language-python">def block(c_in, c_out):
    """conv-BN-ReLU ×2, then downsample: the standard rhythm."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(),
        nn.Conv2d(c_out, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(),
        nn.MaxPool2d(2),
    )

model = nn.Sequential(
    block(3, 64),        # 32x32x3   -> 16x16x64
    block(64, 128),      # 16x16x64  -> 8x8x128
    block(128, 256),     # 8x8x128   -> 4x4x256    (space halves, channels double)
    nn.AdaptiveAvgPool2d(1),          # global average pool -> 1x1x256
    nn.Flatten(),                     # -> 256
    nn.Dropout(0.3),
    nn.Linear(256, 10),               # logits - no softmax (fused in the loss)
)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(sum(p.numel() for p in model.parameters()) / 1e6, "M params")   # ~1.1M
</code></pre>
<p>Read the shape narration in the comments - that habit from
<a href="#/dl/forward-propagation">Forward Propagation</a> is how you design
CNNs without a calculator. 1.1M parameters vs VGG-16's 138M, courtesy of the
GAP head.</p>

<h2>3 · Training loop with the full toolkit</h2>
<pre><code class="language-python">loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=5e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(          # warmup + cosine decay
    opt, max_lr=3e-3, epochs=30, steps_per_epoch=len(train_dl))

def evaluate():
    model.eval(); correct = total = 0
    with torch.no_grad():
        for xb, yb in test_dl:
            xb, yb = xb.to(device), yb.to(device)
            correct += (model(xb).argmax(1) == yb).sum().item()
            total += len(yb)
    return correct / total

best_acc, patience, bad = 0.0, 5, 0
for epoch in range(1, 31):
    model.train()
    for xb, yb in train_dl:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        loss_fn(model(xb), yb).backward()
        opt.step(); sched.step()

    acc = evaluate()
    print(f"epoch {epoch:2d}   test acc {acc:.3f}")
    if acc &gt; best_acc:
        best_acc, bad = acc, 0
        torch.save(model.state_dict(), "cifar_best.pt")
    else:
        bad += 1
        if bad &gt;= patience: break                    # early stopping

model.load_state_dict(torch.load("cifar_best.pt"))
print(f"best: {best_acc:.3f}")     # ~0.90-0.92 after 30 epochs on one GPU
</code></pre>
<p>Context for that number: random guessing = 10%, classical ML on raw pixels
≈ 40–50%, this small CNN ≈ 91%, state of the art ≈ 99%+. The leap from 50 to 91
is the whole argument for learned convolutional features.</p>

<h2>4 · The debrief: where does it fail?</h2>
<pre><code class="language-python">from sklearn.metrics import confusion_matrix, classification_report
import numpy as np, seaborn as sns, matplotlib.pyplot as plt

model.eval(); preds, labels = [], []
with torch.no_grad():
    for xb, yb in test_dl:
        preds += model(xb.to(device)).argmax(1).cpu().tolist()
        labels += yb.tolist()

cm = confusion_matrix(labels, preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=classes, yticklabels=classes)
plt.xlabel("predicted"); plt.ylabel("actual")

print(classification_report(labels, preds, target_names=classes, digits=3))
# the classic finding: cat ↔ dog is the hottest off-diagonal pair,
# animals confuse with animals, vehicles with vehicles - the network's
# errors are SEMANTIC, which tells you it learned real visual structure.
</code></pre>
<p>This last step - <a href="#/dataviz/ml-visualizations">confusion-matrix
reading</a> - turns "91%" into engineering direction: more cat/dog data,
targeted augmentation, or a finer-grained model for the animal classes.</p>

<h2>Where to go from here</h2>
<ul>
  <li><strong>Do better with less:</strong> swap the model for a pretrained ResNet-18
      and fine-tune - <a href="#/advanced/cnn-transfer-learning">Transfer Learning
      in CNNs</a> shows this reaching ~95% in a fraction of the epochs.</li>
  <li><strong>Ablate:</strong> remove augmentation (watch overfitting bloom), remove
      BatchNorm (watch training slow), remove the scheduler - one change at a
      time, like a <a href="#/dataviz/paper-visualizations">paper ablation</a>.</li>
  <li><strong>Inspect filters:</strong> plot <code>model[0][0].weight</code> - layer-1
      filters visibly become edge and color detectors. Seeing it once is worth
      ten diagrams.</li>
</ul>

<h2>Quick quiz</h2>
<div class="quiz">
  <div class="quiz-q" data-answer="1">
    <p class="quiz-question">1. Why is augmentation applied to training data but never to test data?</p>
    <button class="quiz-opt">Augmenting test data is too slow</button>
    <button class="quiz-opt">Training-time variety regularizes; the test set must stay fixed to measure honestly</button>
    <button class="quiz-opt">PyTorch forbids it</button>
    <div class="quiz-explain hidden">Random flips/crops effectively enlarge the training set. Randomizing the test set would make the metric itself noisy and non-comparable. (Test-time augmentation exists, but as a deliberate averaging technique.)</div>
  </div>
  <div class="quiz-q" data-answer="0">
    <p class="quiz-question">2. After block(128, 256), a 8×8×128 tensor becomes…</p>
    <button class="quiz-opt">4×4×256</button>
    <button class="quiz-opt">8×8×256</button>
    <button class="quiz-opt">16×16×64</button>
    <div class="quiz-explain hidden">Two same-padded convs keep 8×8 while lifting channels to 256; the MaxPool2d(2) then halves space to 4×4. Halve space, double channels - the rhythm.</div>
  </div>
  <div class="quiz-q" data-answer="2">
    <p class="quiz-question">3. Cat↔dog dominates the confusion matrix. The most targeted next step is…</p>
    <button class="quiz-opt">Lower the learning rate</button>
    <button class="quiz-opt">Remove the dropout</button>
    <button class="quiz-opt">More/better data and capacity focused on the confused pair</button>
    <div class="quiz-explain hidden">The matrix localizes the failure to one semantic boundary. Generic knob-turning wastes effort; the fix should attack the specific confusion.</div>
  </div>
</div>
`};
