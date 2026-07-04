/* Module 8 - Convolutional Neural Networks (part 1) */

CONTENT["cnn/why-cnns"] = {
  html: String.raw`
<h1>Why CNNs for Images</h1>
<p class="lead">Feed a modest 224×224 photo to a fully-connected network and the
first layer alone needs ~77 million weights. CNNs cut that thousands-fold with
two ideas stolen from the structure of images themselves: look locally, and
reuse what you learn everywhere.</p>

<h2>The problem with flattening</h2>
<p>An MLP takes a vector, so a 224×224×3 image must be flattened into 150,528
numbers. Two disasters follow:</p>
<ul>
  <li><strong>Parameter explosion:</strong> a first hidden layer of just 512 units
      needs 150,528 × 512 ≈ 77M weights - begging to overfit, brutal to train.</li>
  <li><strong>Structure destroyed:</strong> flattening throws away the 2-D
      neighborhood. Pixels (50, 51) and (51, 51) - physically adjacent - land ~224
      positions apart in the vector, and the MLP has no idea they were neighbors.
      Worse, an MLP that learns "cat at top-left" has learned <em>nothing</em> about
      cats at bottom-right: each position has its own private weights.</li>
</ul>

<h2>Two properties of images, two design principles</h2>
<p><strong>Locality:</strong> meaningful visual patterns - an edge, a corner, an eye -
live in small neighborhoods. You don't need the whole image to detect an edge at
(50, 51); a few surrounding pixels suffice. So: connect each unit only to a small
<em>local patch</em> (its receptive field), not to every pixel.</p>
<p><strong>Translation invariance:</strong> an edge is an edge wherever it appears.
The detector that finds it at the top-left works identically at the bottom-right.
So: learn <em>one</em> small detector and <strong>slide it across the whole
image</strong> - weight sharing.</p>
<div class="diagram">
<svg viewBox="0 0 660 270" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Left: fully connected layer connecting every pixel to every unit. Right: a small shared filter sliding locally across the image.">
  <!-- FC side -->
  <text x="160" y="35" text-anchor="middle" class="d-text">fully connected: everyone talks to everyone</text>
  <rect x="40" y="60" width="110" height="110" rx="6" class="d-box-muted"/>
  <text x="95" y="190" text-anchor="middle" class="d-text-sm">image (all pixels)</text>
  <circle cx="255" cy="80" r="13" class="d-box"/>
  <circle cx="255" cy="120" r="13" class="d-box"/>
  <circle cx="255" cy="160" r="13" class="d-box"/>
  <g class="d-grid">
    <line x1="150" y1="70" x2="242" y2="80"/><line x1="150" y1="100" x2="242" y2="80"/>
    <line x1="150" y1="130" x2="242" y2="80"/><line x1="150" y1="160" x2="242" y2="80"/>
    <line x1="150" y1="70" x2="242" y2="120"/><line x1="150" y1="100" x2="242" y2="120"/>
    <line x1="150" y1="130" x2="242" y2="120"/><line x1="150" y1="160" x2="242" y2="120"/>
    <line x1="150" y1="70" x2="242" y2="160"/><line x1="150" y1="100" x2="242" y2="160"/>
    <line x1="150" y1="130" x2="242" y2="160"/><line x1="150" y1="160" x2="242" y2="160"/>
  </g>
  <text x="160" y="225" text-anchor="middle" class="d-text-sm">150k pixels × 512 units ≈ 77M weights</text>
  <!-- CNN side -->
  <text x="500" y="35" text-anchor="middle" class="d-text">convolution: one small filter, slid everywhere</text>
  <rect x="380" y="60" width="110" height="110" rx="6" class="d-box-muted"/>
  <rect x="390" y="70" width="30" height="30" class="d-box-soft" style="stroke: var(--accent); stroke-width: 2"/>
  <rect x="430" y="95" width="30" height="30" fill="none" style="stroke: var(--accent); stroke-width: 1.2" stroke-dasharray="4 3"/>
  <rect x="450" y="130" width="30" height="30" fill="none" style="stroke: var(--accent); stroke-width: 1.2" stroke-dasharray="4 3"/>
  <text x="435" y="190" text-anchor="middle" class="d-text-sm">same 3×3 weights at every position</text>
  <rect x="560" y="85" width="60" height="60" rx="6" class="d-box"/>
  <text x="590" y="120" text-anchor="middle" class="d-text-sm">feature</text>
  <text x="590" y="134" text-anchor="middle" class="d-text-sm">map</text>
  <line x1="490" y1="115" x2="558" y2="115" class="d-line"/>
  <text x="500" y="225" text-anchor="middle" class="d-text-sm">3×3×3 filter = 27 weights, reused everywhere</text>
</svg>
<div class="caption">Same job, six orders of magnitude fewer parameters - and the
notion of "nearby pixels" is preserved instead of destroyed.</div>
</div>
<p>The arithmetic is staggering: 77,000,000 weights become 27 (a 3×3×3 filter) -
and because the filter slides, detecting a pattern <em>anywhere</em> requires
learning it <em>once</em>. Fewer parameters also means less overfitting: the
architecture itself encodes the prior "images are made of local, repeatable
patterns," which is regularization by design.</p>

<h2>The third idea: hierarchy</h2>
<p>Stack convolutional layers and receptive fields compound: layer 1 sees 3×3
pixels (enough for edges), layer 2 sees 3×3 <em>of layer 1's outputs</em> ≈ 5×5
pixels (corners, textures), and by layer 10 a unit effectively sees most of the
image (faces, wheels, dogs). This edges → parts → objects hierarchy isn't
programmed - it <em>emerges</em> from training, and it echoes the structure
neuroscientists found in the visual cortex (simple cells → complex cells, the
1962 Hubel &amp; Wiesel work that inspired CNNs in the first place).</p>

<h2>Seeing the parameter math</h2>
<pre><code class="language-python"># parameters for the first layer processing a 224x224x3 image

# fully connected -> 512 units
fc = 224 * 224 * 3 * 512 + 512
print(f"fully connected: {fc:,}")            # 77,070,848

# convolution -> 64 filters of 3x3 (each sees all 3 color channels)
conv = (3 * 3 * 3) * 64 + 64
print(f"convolution    : {conv:,}")          # 1,792

print(f"ratio          : {fc / conv:,.0f}x") # ~43,000x fewer
</code></pre>
<p>What those 64 filters compute, precisely - the sliding dot product - is the
<a href="#/cnn/convolution">next chapter</a>.</p>

<h2>Common interview questions</h2>
<div class="qa">
  <div class="qa-q">Q1. Why do CNNs beat MLPs on images?</div>
  <div class="qa-a"><p>They exploit image structure: local connectivity matches the locality of visual
  patterns; weight sharing matches translation invariance ("an edge is an edge anywhere");
  the resulting parameter count is orders of magnitude smaller, which improves sample
  efficiency and generalization; and stacked layers build a feature hierarchy.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q2. What is a receptive field?</div>
  <div class="qa-a"><p>The region of the input image that influences a given unit's output. It grows with
  depth as layers stack - early units see pixels, deep units effectively see the whole
  image, which is how local operations end up recognizing global objects.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q3. An MLP trained on centered digits fails on shifted ones; a CNN doesn't. Why?</div>
  <div class="qa-a"><p>The MLP's weights are tied to absolute pixel positions - a shift presents a
  never-seen input pattern. The CNN's shared filters produce the same response wherever the
  digit appears (equivariance), so shifted digits activate the same features, just at a
  different location.</p></div>
</div>

<h2>Quick quiz</h2>
<div class="quiz">
  <div class="quiz-q" data-answer="1">
    <p class="quiz-question">1. Weight sharing in CNNs means…</p>
    <button class="quiz-opt">All layers use the same weights</button>
    <button class="quiz-opt">One filter's weights are reused at every spatial position of the image</button>
    <button class="quiz-opt">Two networks share a GPU</button>
    <div class="quiz-explain hidden">The same small detector slides across the image - pattern learned once, detected everywhere, parameters divided by the number of positions.</div>
  </div>
  <div class="quiz-q" data-answer="2">
    <p class="quiz-question">2. A 5×5 filter over RGB images (16 filters total) has how many weights (no biases)?</p>
    <button class="quiz-opt">25</button>
    <button class="quiz-opt">400</button>
    <button class="quiz-opt">1,200</button>
    <div class="quiz-explain hidden">Each filter spans all input channels: 5×5×3 = 75 weights; ×16 filters = 1,200. Filter depth always equals input channel count.</div>
  </div>
  <div class="quiz-q" data-answer="0">
    <p class="quiz-question">3. Deep CNN layers detect objects while early ones detect edges because…</p>
    <button class="quiz-opt">Receptive fields compound with depth - each layer combines the previous layer's local patterns into larger ones</button>
    <button class="quiz-opt">Deep layers have bigger filters by rule</button>
    <button class="quiz-opt">Early layers are frozen during training</button>
    <div class="quiz-explain hidden">3×3 on top of 3×3 sees 5×5; keep stacking and effective coverage grows to the whole image. The hierarchy emerges from training, not programming.</div>
  </div>
</div>
`};

CONTENT["cnn/convolution"] = {
  html: String.raw`
<h1>The Convolution Operation</h1>
<p class="lead">One operation to learn: place a small grid of weights on the image,
multiply element-wise, sum to a single number, slide, repeat. That sliding dot
product is convolution - and everything a CNN "sees", it sees through it.</p>

<h2>The mechanics, by hand</h2>
<p>A 3×3 filter on a 5×5 image. At each position, overlay the filter, multiply
matching cells, add up the 9 products - one output pixel:</p>
<div class="diagram">
<svg viewBox="0 0 660 280" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="A 3 by 3 filter overlaid on a 5 by 5 input grid produces one cell of the 3 by 3 output feature map via element-wise multiply and sum">
  <!-- input grid -->
  <text x="105" y="35" text-anchor="middle" class="d-text">input 5×5</text>
  <g>
    <rect x="30" y="50" width="150" height="150" class="d-box"/>
    <line x1="60" y1="50" x2="60" y2="200" class="d-grid"/><line x1="90" y1="50" x2="90" y2="200" class="d-grid"/>
    <line x1="120" y1="50" x2="120" y2="200" class="d-grid"/><line x1="150" y1="50" x2="150" y2="200" class="d-grid"/>
    <line x1="30" y1="80" x2="180" y2="80" class="d-grid"/><line x1="30" y1="110" x2="180" y2="110" class="d-grid"/>
    <line x1="30" y1="140" x2="180" y2="140" class="d-grid"/><line x1="30" y1="170" x2="180" y2="170" class="d-grid"/>
    <rect x="30" y="50" width="90" height="90" fill="none" style="stroke: var(--accent); stroke-width: 2.5"/>
  </g>
  <text x="45" y="70" class="d-text-sm">1</text><text x="75" y="70" class="d-text-sm">0</text><text x="105" y="70" class="d-text-sm">1</text>
  <text x="45" y="100" class="d-text-sm">0</text><text x="75" y="100" class="d-text-sm">1</text><text x="105" y="100" class="d-text-sm">0</text>
  <text x="45" y="130" class="d-text-sm">1</text><text x="75" y="130" class="d-text-sm">1</text><text x="105" y="130" class="d-text-sm">1</text>
  <text x="105" y="222" text-anchor="middle" class="d-text-sm">highlighted patch (current position)</text>
  <!-- filter -->
  <text x="290" y="35" text-anchor="middle" class="d-text">filter 3×3</text>
  <g>
    <rect x="245" y="75" width="90" height="90" class="d-box-soft"/>
    <line x1="275" y1="75" x2="275" y2="165" class="d-grid"/><line x1="305" y1="75" x2="305" y2="165" class="d-grid"/>
    <line x1="245" y1="105" x2="335" y2="105" class="d-grid"/><line x1="245" y1="135" x2="335" y2="135" class="d-grid"/>
  </g>
  <text x="258" y="95" class="d-text-sm">1</text><text x="288" y="95" class="d-text-sm">0</text><text x="318" y="95" class="d-text-sm">−1</text>
  <text x="258" y="125" class="d-text-sm">1</text><text x="288" y="125" class="d-text-sm">0</text><text x="318" y="125" class="d-text-sm">−1</text>
  <text x="258" y="155" class="d-text-sm">1</text><text x="288" y="155" class="d-text-sm">0</text><text x="318" y="155" class="d-text-sm">−1</text>
  <text x="290" y="190" text-anchor="middle" class="d-text-sm">(a vertical-edge detector)</text>
  <text x="212" y="125" class="d-text" style="font-size:19px">⊙</text>
  <text x="360" y="125" class="d-text" style="font-size:19px">→ Σ</text>
  <!-- output -->
  <text x="530" y="35" text-anchor="middle" class="d-text">feature map 3×3</text>
  <g>
    <rect x="485" y="75" width="90" height="90" class="d-box"/>
    <line x1="515" y1="75" x2="515" y2="165" class="d-grid"/><line x1="545" y1="75" x2="545" y2="165" class="d-grid"/>
    <line x1="485" y1="105" x2="575" y2="105" class="d-grid"/><line x1="485" y1="135" x2="575" y2="135" class="d-grid"/>
    <rect x="485" y="75" width="30" height="30" class="d-box-soft" style="stroke: var(--accent); stroke-width: 2.5"/>
  </g>
  <text x="495" y="95" class="d-text-sm">0</text>
  <text x="530" y="222" text-anchor="middle" class="d-text-sm">1·1+0·0+1·(−1) + 0+0−0 + 1+0−1 = 0</text>
  <text x="330" y="262" text-anchor="middle" class="d-text-sm">slide the filter one step right → compute the next output cell → repeat over the whole image</text>
</svg>
<div class="caption">One filter position = one dot product = one output pixel.
The full sweep produces a "feature map": a picture of where the pattern occurs.</div>
</div>
<div class="math-box">
$$S(i, j) = \sum_{u}\sum_{v} I(i+u,\, j+v)\; K(u, v) \; + \; b$$
</div>
<p>It's the <a href="#/math/linear-algebra">dot product</a> again - and dot products
measure <em>similarity</em>. So each output pixel answers: <strong>"how much does this
patch look like my filter's pattern?"</strong> The feature map is literally a heat map
of pattern matches. The example filter (+1s left, −1s right) fires strongly where
bright-left-dark-right transitions occur: a vertical edge detector.</p>

<h2>The part that makes it deep learning</h2>
<p>Classic image processing spent decades hand-designing such filters (Sobel,
Gabor…). A CNN treats the 9 filter weights as <strong>learnable parameters</strong> -
initialized randomly, shaped by <a href="#/dl/backpropagation">backpropagation</a>
into whatever detectors minimize the loss. Train on faces and layer-1 filters
become edge/color detectors, mid layers become eye/nose detectors - nobody
designed them. The convolution layer is "just" a linear layer with locality and
weight-sharing constraints; everything from Module 7 (activations after it,
gradients through it) applies unchanged.</p>

<h2>Channels: the third dimension</h2>
<p>Real inputs have depth - RGB images have 3 channels; hidden layers have 64, 256…
The rules:</p>
<ul>
  <li>A filter always spans <strong>all input channels</strong>: on RGB, a "3×3" filter
      is really 3×3×3 = 27 weights, summing evidence across colors into one number
      per position → one 2-D feature map.</li>
  <li>A layer learns <strong>many filters</strong> (say 64), each hunting a different
      pattern → the layer's output is a stack of 64 feature maps: a new "image"
      with 64 channels, ready for the next layer.</li>
</ul>
<div class="math-box">
$$\text{input } H \times W \times C_{in}
\;\;\xrightarrow{\;N \text{ filters of } k \times k \times C_{in}\;}\;\;
\text{output } H' \times W' \times N$$
$$\text{parameters} = N \times (k \times k \times C_{in} + 1)$$
</div>

<h2>In code: convolution from scratch, then the real thing</h2>
<pre><code class="language-python">import numpy as np

def conv2d(image, kernel):
    """Naive 2-D convolution (no padding, stride 1) - for understanding."""
    H, W = image.shape
    k = kernel.shape[0]
    out = np.zeros((H - k + 1, W - k + 1))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            patch = image[i:i+k, j:j+k]
            out[i, j] = np.sum(patch * kernel)      # the sliding dot product
    return out

# an image with a vertical edge: dark left half, bright right half
img = np.zeros((6, 6)); img[:, 3:] = 1.0

vertical_edge = np.array([[ 1, 0, -1],
                          [ 1, 0, -1],
                          [ 1, 0, -1]], float)

print(conv2d(img, vertical_edge))
# zeros everywhere EXCEPT a column of -3s exactly where the edge is:
# the feature map has located the pattern.

# --- the real thing: a PyTorch conv layer --------------------------------
import torch.nn as nn
layer = nn.Conv2d(in_channels=3, out_channels=64,
                  kernel_size=3, padding=1)
n_params = sum(p.numel() for p in layer.parameters())
print(n_params)         # 64 * (3*3*3) + 64 = 1,792 - matching our formula
</code></pre>
<p>(Pedantic footnote with interview value: frameworks actually compute
<em>cross-correlation</em> - convolution without flipping the kernel. Since the
kernel is learned, the flip is irrelevant; the name stuck.)</p>

<h2>Common interview questions</h2>
<div class="qa">
  <div class="qa-q">Q1. What does one value in a feature map represent?</div>
  <div class="qa-a"><p>The dot-product similarity between the filter's pattern and the image patch at
  that location (plus bias, usually through a ReLU). High value = "my pattern is here."
  The whole map is a spatial detection heat map for that one pattern.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q2. How many parameters does Conv2d(128, 256, kernel_size=3) have?</div>
  <div class="qa-a"><p>256 filters × (3×3×128 weights) + 256 biases = 295,168. Note what's absent: the
  image size - convolution parameters are independent of input resolution, unlike fully
  connected layers.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q3. Why does a filter need depth equal to the input's channel count?</div>
  <div class="qa-a"><p>At each position it must combine evidence from every input channel (all colors,
  or all previous feature maps) into one response - so it holds k×k weights per channel and
  sums across them. Cross-channel mixing is where combinations like "red AND round" form.</p></div>
</div>

<h2>Quick quiz</h2>
<div class="quiz">
  <div class="quiz-q" data-answer="1">
    <p class="quiz-question">1. A 3×3 filter slides over an 8×8 image (stride 1, no padding). Output size?</p>
    <button class="quiz-opt">8×8</button>
    <button class="quiz-opt">6×6</button>
    <button class="quiz-opt">5×5</button>
    <div class="quiz-explain hidden">The filter fits in 8−3+1 = 6 positions per axis. The general formula gets refined with padding and stride in the next chapter.</div>
  </div>
  <div class="quiz-q" data-answer="0">
    <p class="quiz-question">2. In a trained CNN, the filter weights come from…</p>
    <button class="quiz-opt">Backpropagation - they're learned like any other parameters</button>
    <button class="quiz-opt">A library of standard filters (Sobel etc.)</button>
    <button class="quiz-opt">The image itself</button>
    <div class="quiz-explain hidden">That's the historical break: hand-engineered filters out, gradient-learned detectors in. The network invents whatever filters serve the task.</div>
  </div>
  <div class="quiz-q" data-answer="2">
    <p class="quiz-question">3. A conv layer outputs 32 feature maps. This means the layer has…</p>
    <button class="quiz-opt">32 output pixels</button>
    <button class="quiz-opt">A stride of 32</button>
    <button class="quiz-opt">32 filters, each detecting its own pattern across the image</button>
    <div class="quiz-explain hidden">One filter → one map. The output stack (H×W×32) becomes the 32-channel "image" the next layer convolves over.</div>
  </div>
</div>
`};

CONTENT["cnn/filters-padding-stride"] = {
  html: String.raw`
<h1>Filters, Padding, Stride</h1>
<p class="lead">Three knobs control every convolutional layer: how big the filter,
whether to pad the borders, and how far to jump between positions. Master one
formula and you can size any CNN in your head.</p>

<h2>The one formula</h2>
<div class="math-box">
$$O = \left\lfloor \frac{N + 2P - F}{S} \right\rfloor + 1$$
</div>
<p>\(N\) = input size, \(F\) = filter size, \(P\) = padding, \(S\) = stride,
\(O\) = output size (per spatial axis). Everything below is this formula's
consequences.</p>

<h2>Padding: saving the borders</h2>
<p>Without padding ("valid"), two problems: the map shrinks by \(F-1\) every layer
(a 32×32 image survives only ~15 layers of 3×3 before vanishing), and border
pixels are seen by fewer filter positions than central ones - corners get one
vote, centers get nine. Padding adds a ring of zeros so the filter can center on
every real pixel:</p>
<ul>
  <li><strong>"valid"</strong> (P = 0): no padding, output shrinks. 8×8 → 6×6 with a
      3×3 filter.</li>
  <li><strong>"same"</strong> (P = (F−1)/2): output size = input size. For 3×3: P=1;
      for 5×5: P=2. The default in modern architectures - size stays constant
      through conv layers, and <em>you</em> decide where to downsample.</li>
</ul>

<h2>Stride: jumping to downsample</h2>
<p>Stride is the step size of the slide. S = 1 checks every position; S = 2 checks
every other one, halving the output's width and height (quartering the pixels) -
a built-in downsampler. Modern networks often use stride-2 convolutions instead
of <a href="#/cnn/pooling">pooling</a> to shrink maps: same effect, but the
downsampling is <em>learned</em>.</p>
<div class="diagram">
<svg viewBox="0 0 660 240" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Left: padded input with a ring of zeros keeps output size same. Right: stride 2 makes the filter jump two cells, halving the output.">
  <!-- padding -->
  <text x="165" y="30" text-anchor="middle" class="d-text">padding P=1 ("same")</text>
  <rect x="60" y="45" width="180" height="150" class="d-box-muted" stroke-dasharray="5 4"/>
  <rect x="90" y="70" width="120" height="100" class="d-box"/>
  <text x="150" y="125" text-anchor="middle" class="d-text-sm">real image</text>
  <text x="75" y="62" class="d-text-sm">0</text><text x="130" y="62" class="d-text-sm">0</text>
  <text x="185" y="62" class="d-text-sm">0</text><text x="228" y="62" class="d-text-sm">0</text>
  <text x="75" y="188" class="d-text-sm">0</text><text x="228" y="188" class="d-text-sm">0</text>
  <text x="150" y="220" text-anchor="middle" class="d-text-sm">zero ring lets the filter center on border pixels</text>
  <!-- stride -->
  <text x="490" y="30" text-anchor="middle" class="d-text">stride S=2</text>
  <rect x="390" y="45" width="200" height="150" class="d-box"/>
  <g class="d-grid">
    <line x1="440" y1="45" x2="440" y2="195"/><line x1="490" y1="45" x2="490" y2="195"/><line x1="540" y1="45" x2="540" y2="195"/>
    <line x1="390" y1="95" x2="590" y2="95"/><line x1="390" y1="145" x2="590" y2="145"/>
  </g>
  <rect x="390" y="45" width="70" height="70" fill="none" style="stroke: var(--accent); stroke-width: 2.5"/>
  <rect x="490" y="45" width="70" height="70" fill="none" style="stroke: var(--accent); stroke-width: 1.4" stroke-dasharray="5 3"/>
  <defs>
    <marker id="starr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M0,0 L10,5 L0,10 z" style="fill: var(--accent)"/>
    </marker>
  </defs>
  <line x1="462" y1="80" x2="487" y2="80" class="d-line-accent" marker-end="url(#starr)"/>
  <text x="490" y="220" text-anchor="middle" class="d-text-sm">filter jumps 2 cells → output is half the size</text>
</svg>
<div class="caption">Padding preserves size and border information; stride trades
resolution for speed and larger effective receptive fields.</div>
</div>

<h2>Filter size: why 3×3 won</h2>
<p>Early CNNs used 7×7 and 11×11 filters. VGG's 2014 insight retired them:
<strong>two stacked 3×3 layers see a 5×5 area</strong> (three see 7×7) with fewer
parameters and more non-linearity:</p>
<div class="math-box">
$$\text{two } 3{\times}3 \text{: } 2 \times 9C^2 = 18C^2 \text{ weights, 2 ReLUs}
\qquad \text{vs} \qquad
\text{one } 5{\times}5 \text{: } 25C^2 \text{ weights, 1 ReLU}$$
</div>
<p>Same coverage, 28% cheaper, twice the bends. Hence the modern default:
<strong>3×3, "same" padding, stride 1</strong> - downsampling handled separately.
The special case <strong>1×1 convolution</strong> looks pointless but isn't: it mixes
<em>channels</em> at each position (a per-pixel fully-connected layer), used
everywhere to compress 256 channels to 64 cheaply (bottlenecks in ResNet/Inception,
<a href="#/cnn/classic-architectures">next chapters</a>).</p>

<h2>Worked sizing: a real network in your head</h2>
<pre><code class="language-python">import torch
import torch.nn as nn

x = torch.zeros(1, 3, 32, 32)              # CIFAR-sized input

net = nn.Sequential(
    nn.Conv2d(3,   32, 3, padding=1),       # 32x32 -> 32x32   ("same")
    nn.ReLU(),
    nn.Conv2d(32,  64, 3, padding=1, stride=2),  # -> 16x16     (stride!)
    nn.ReLU(),
    nn.Conv2d(64, 128, 3, padding=1, stride=2),  # -> 8x8
    nn.ReLU(),
    nn.Conv2d(128, 128, 3),                 # no padding: 8-3+1 -> 6x6
)

for layer in net:
    x = layer(x)
    if isinstance(layer, nn.Conv2d):
        print(f"{str(layer.kernel_size):8s} stride {layer.stride[0]} "
              f"pad {layer.padding[0]} -> {tuple(x.shape)}")
# (3, 3)  stride 1 pad 1 -> (1, 32, 32, 32)
# (3, 3)  stride 2 pad 1 -> (1, 64, 16, 16)
# (3, 3)  stride 2 pad 1 -> (1, 128, 8, 8)
# (3, 3)  stride 1 pad 0 -> (1, 128, 6, 6)
</code></pre>
<p>Verify each against the formula - e.g. layer 2: (32 + 2·1 − 3)/2 + 1 = 16.5 →
floor → 16. ✓ Notice the standard CNN rhythm: <strong>spatial size halves while
channels double</strong> - trading where-resolution for what-richness as information
flows deeper.</p>

<h2>Common interview questions</h2>
<div class="qa">
  <div class="qa-q">Q1. Compute: 224×224 input, 7×7 filter, stride 2, padding 3. Output?</div>
  <div class="qa-a"><p>(224 + 6 − 7)/2 + 1 = 112.5 → floor 112 (+1 already applied: (223)/2=111.5→111, +1 = 112).
  Output 112×112 - this is literally ResNet's first layer.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q2. Why prefer two 3×3 convs over one 5×5?</div>
  <div class="qa-a"><p>Identical 5×5 receptive field with 18C² vs 25C² parameters, plus an extra ReLU for
  more non-linear expressiveness. The VGG principle - depth of small filters beats breadth
  of big ones - still shapes architectures today.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q3. What is a 1×1 convolution good for?</div>
  <div class="qa-a"><p>Per-position channel mixing: it can't see spatial patterns, but it linearly
  recombines the C input channels at each pixel - the cheap way to compress or expand
  channel depth (bottlenecks), add non-linearity, and glue architecture blocks together.</p></div>
</div>

<h2>Quick quiz</h2>
<div class="quiz">
  <div class="quiz-q" data-answer="1">
    <p class="quiz-question">1. For a 3×3 filter, "same" output size requires padding…</p>
    <button class="quiz-opt">P = 0</button>
    <button class="quiz-opt">P = 1</button>
    <button class="quiz-opt">P = 3</button>
    <div class="quiz-explain hidden">P = (F−1)/2 = 1: one ring of zeros replaces the (F−1) = 2 pixels of shrinkage.</div>
  </div>
  <div class="quiz-q" data-answer="2">
    <p class="quiz-question">2. 64×64 input, 3×3 filter, stride 2, padding 1. Output size?</p>
    <button class="quiz-opt">64×64</button>
    <button class="quiz-opt">31×31</button>
    <button class="quiz-opt">32×32</button>
    <div class="quiz-explain hidden">(64 + 2 − 3)/2 + 1 = 31.5 + 1 → floor(31.5) = 31, +1 = 32. Stride 2 halves the map.</div>
  </div>
  <div class="quiz-q" data-answer="0">
    <p class="quiz-question">3. The typical deep-CNN pattern as you go deeper is…</p>
    <button class="quiz-opt">Spatial size shrinks while channel count grows</button>
    <button class="quiz-opt">Both grow</button>
    <button class="quiz-opt">Channels shrink while spatial size grows</button>
    <div class="quiz-explain hidden">32×32×32 → 16×16×64 → 8×8×128: less "where", more "what" - the information gets re-encoded from positions into features.</div>
  </div>
</div>
`};
