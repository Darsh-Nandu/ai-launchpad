/* Module 6 - Machine Learning Algorithms (part 4) */

CONTENT["ml/kmeans"] = {
  html: String.raw`
<h1>K-Means Clustering</h1>
<p class="lead">No labels, no answers - just points and a hunch that groups exist.
K-Means is the simplest machine for finding them: guess k centers, assign, move,
repeat until nothing changes.</p>

<h2>The algorithm: two steps, alternated forever</h2>
<ol>
  <li><strong>Initialize:</strong> place \(k\) centroids (cluster centers).</li>
  <li><strong>Assign:</strong> each point joins its nearest centroid.</li>
  <li><strong>Update:</strong> each centroid moves to the mean of its members.</li>
  <li>Repeat 2–3 until assignments stop changing.</li>
</ol>
<div class="diagram">
<svg viewBox="0 0 660 240" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Three panels: random initial centroids, points assigned to nearest centroid, centroids moved to cluster means">
  <!-- panel 1 -->
  <rect x="15" y="25" width="195" height="175" rx="8" class="d-box-muted"/>
  <text x="112" y="47" text-anchor="middle" class="d-text">1 · initialize</text>
  <circle cx="55" cy="90" r="4" class="d-dot-muted"/><circle cx="75" cy="110" r="4" class="d-dot-muted"/>
  <circle cx="60" cy="130" r="4" class="d-dot-muted"/><circle cx="150" cy="150" r="4" class="d-dot-muted"/>
  <circle cx="170" cy="130" r="4" class="d-dot-muted"/><circle cx="160" cy="170" r="4" class="d-dot-muted"/>
  <path d="M 100 165 l 6 6 m 0 -6 l -6 6" class="d-line-accent"/>
  <path d="M 130 70 l 6 6 m 0 -6 l -6 6" class="d-line-accent"/>
  <text x="112" y="192" text-anchor="middle" class="d-text-sm">✕ = random centroids</text>
  <!-- panel 2 -->
  <rect x="232" y="25" width="195" height="175" rx="8" class="d-box-muted"/>
  <text x="329" y="47" text-anchor="middle" class="d-text">2 · assign to nearest</text>
  <circle cx="272" cy="90" r="4" class="d-dot"/><circle cx="292" cy="110" r="4" class="d-dot"/>
  <circle cx="277" cy="130" r="4" class="d-dot"/>
  <circle cx="367" cy="150" r="4" class="d-dot-muted"/><circle cx="387" cy="130" r="4" class="d-dot-muted"/>
  <circle cx="377" cy="170" r="4" class="d-dot-muted"/>
  <path d="M 317 165 l 6 6 m 0 -6 l -6 6" class="d-line-accent"/>
  <path d="M 347 70 l 6 6 m 0 -6 l -6 6" class="d-line-accent"/>
  <line x1="272" y1="90" x2="349" y2="74" class="d-grid"/>
  <line x1="367" y1="150" x2="321" y2="167" class="d-grid"/>
  <text x="329" y="192" text-anchor="middle" class="d-text-sm">colors = current membership</text>
  <!-- panel 3 -->
  <rect x="449" y="25" width="195" height="175" rx="8" class="d-box-muted"/>
  <text x="546" y="47" text-anchor="middle" class="d-text">3 · move to the mean</text>
  <circle cx="489" cy="90" r="4" class="d-dot"/><circle cx="509" cy="110" r="4" class="d-dot"/>
  <circle cx="494" cy="130" r="4" class="d-dot"/>
  <circle cx="584" cy="150" r="4" class="d-dot-muted"/><circle cx="604" cy="130" r="4" class="d-dot-muted"/>
  <circle cx="594" cy="170" r="4" class="d-dot-muted"/>
  <path d="M 494 107 l 6 6 m 0 -6 l -6 6" class="d-line-accent"/>
  <path d="M 591 147 l 6 6 m 0 -6 l -6 6" class="d-line-accent"/>
  <text x="546" y="192" text-anchor="middle" class="d-text-sm">repeat 2–3 until stable</text>
</svg>
<div class="caption">Assign, average, repeat. Each iteration can only lower the total
within-cluster distance, so the loop always converges.</div>
</div>
<p>Formally, K-Means minimizes <strong>inertia</strong> (within-cluster sum of squares):</p>
<div class="math-box">
$$J = \sum_{i=1}^{m} \lVert \mathbf{x}^{(i)} - \boldsymbol{\mu}_{c^{(i)}} \rVert^2$$
</div>
<p>Both steps provably decrease \(J\), so it converges - but only to a
<em>local</em> minimum that depends on the starting centroids. Two standard defenses:
<strong>k-means++</strong> initialization (spread the initial centroids apart -
sklearn's default) and multiple restarts (<code>n_init=10</code>, keep the best run).</p>

<h2>Choosing k: elbow and silhouette</h2>
<p>K-Means can't tell you k - inertia always falls as k grows (k = m gives zero).
Two heuristics triangulate a sensible choice:</p>
<ul>
  <li><strong>Elbow:</strong> plot inertia vs k; pick the bend where extra clusters stop
      paying rent.</li>
  <li><strong>Silhouette score</strong> ∈ [−1, 1]: for each point, compare its average
      distance to its own cluster (a) vs the nearest other cluster (b):
      \(s = (b - a)/\max(a, b)\). Averages near ~0.5+ suggest real structure;
      near 0 suggests you're slicing a continuum.</li>
</ul>
<pre><code class="language-python">import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=600, centers=4, cluster_std=1.1, random_state=7)
X = StandardScaler().fit_transform(X)        # scaling: mandatory (distances!)

ks = range(2, 10)
inertias, sils = [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
    inertias.append(km.inertia_)
    sils.append(silhouette_score(X, km.labels_))

fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.5))
a1.plot(ks, inertias, "o-"); a1.set_title("elbow: look for the bend")
a2.plot(ks, sils, "o-");     a2.set_title("silhouette: look for the peak")
plt.show()                                    # both point to k = 4 here

km = KMeans(n_clusters=4, n_init=10, random_state=0).fit(X)
print("cluster sizes:", np.bincount(km.labels_))
print("centroids (interpret them!):", km.cluster_centers_.round(2))
</code></pre>
<p>The last line is the deliverable in practice: <strong>centroids are cluster
personas</strong>. In customer segmentation, "centroid 2 = high spend, low frequency,
recent" becomes "big-ticket occasional buyers" - the labels the business never had.</p>

<h2>What K-Means assumes (and how it fails)</h2>
<ul>
  <li><strong>Spherical, similar-size clusters:</strong> distance-to-mean geometry carves
      space into convex cells. Elongated, nested, or crescent-shaped clusters get
      butchered (<a href="#/ml/dbscan">DBSCAN</a> handles those).</li>
  <li><strong>Every point belongs somewhere:</strong> no concept of noise - outliers join
      (and drag) the nearest centroid.</li>
  <li><strong>Means must make sense:</strong> continuous features only; scaling required;
      for categorical data use k-modes instead.</li>
</ul>

<h2>Common interview questions</h2>
<div class="qa">
  <div class="qa-q">Q1. Why does K-Means always converge, and why isn't that the global optimum?</div>
  <div class="qa-a"><p>Assignment and update each monotonically decrease the finite objective J, and
  there are finitely many partitions - so it must stop. But it stops at whatever local
  minimum the initialization led to; k-means++ and multiple restarts mitigate.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q2. How do you evaluate clustering with no ground truth?</div>
  <div class="qa-a"><p>Internal metrics (silhouette, Davies-Bouldin), stability across
  subsamples/initializations, and - decisively - interpretability and usefulness of the
  clusters to the domain. Unsupervised evaluation is always part judgment.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q3. One feature is income in rupees, another is age. What happens without scaling?</div>
  <div class="qa-a"><p>Income's magnitude dominates every distance, so clusters form along income alone -
  age is ignored. Standardize first; K-Means is as scale-sensitive as KNN.</p></div>
</div>

<h2>Quick quiz</h2>
<div class="quiz">
  <div class="quiz-q" data-answer="1">
    <p class="quiz-question">1. K-Means' two alternating steps are…</p>
    <button class="quiz-opt">Split and prune</button>
    <button class="quiz-opt">Assign points to nearest centroid; move centroids to member means</button>
    <button class="quiz-opt">Sample and vote</button>
    <div class="quiz-explain hidden">Both steps lower the within-cluster sum of squares, guaranteeing convergence (to a local optimum).</div>
  </div>
  <div class="quiz-q" data-answer="2">
    <p class="quiz-question">2. Inertia at k=8 is lower than at k=4. Therefore…</p>
    <button class="quiz-opt">k=8 is the better clustering</button>
    <button class="quiz-opt">The algorithm failed at k=4</button>
    <button class="quiz-opt">Nothing - inertia always decreases with k; use the elbow/silhouette instead</button>
    <div class="quiz-explain hidden">More centroids always sit closer to points (k=m gives inertia 0). Raw inertia comparisons across k are meaningless.</div>
  </div>
  <div class="quiz-q" data-answer="0">
    <p class="quiz-question">3. Two crescent-moon clusters interlocking. K-Means will…</p>
    <button class="quiz-opt">Cut each crescent in half - it can only draw convex, blob-shaped regions</button>
    <button class="quiz-opt">Find them perfectly</button>
    <button class="quiz-opt">Refuse to run</button>
    <div class="quiz-explain hidden">Distance-to-centroid partitions are convex cells; crescents aren't. Density-based clustering (DBSCAN) is built for exactly this.</div>
  </div>
</div>
`};

CONTENT["ml/hierarchical-clustering"] = {
  html: String.raw`
<h1>Hierarchical Clustering</h1>
<p class="lead">Instead of forcing one flat answer with a preset k, build the
<em>entire family tree</em> of clusters - from every-point-alone to
everything-in-one - and read off any level you like from the dendrogram.</p>

<h2>The agglomerative algorithm</h2>
<ol>
  <li>Start: every point is its own cluster (m clusters).</li>
  <li>Find the two <em>closest</em> clusters. Merge them.</li>
  <li>Repeat until one cluster remains, recording every merge and its distance.</li>
</ol>
<p>The recorded merge history <em>is</em> the model - drawn as a
<strong>dendrogram</strong>:</p>
<div class="diagram">
<svg viewBox="0 0 640 300" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="A dendrogram of six points merging progressively, with a horizontal cut line producing two clusters">
  <!-- leaves -->
  <text x="80" y="285" text-anchor="middle" class="d-text-sm">A</text>
  <text x="150" y="285" text-anchor="middle" class="d-text-sm">B</text>
  <text x="220" y="285" text-anchor="middle" class="d-text-sm">C</text>
  <text x="380" y="285" text-anchor="middle" class="d-text-sm">D</text>
  <text x="450" y="285" text-anchor="middle" class="d-text-sm">E</text>
  <text x="540" y="285" text-anchor="middle" class="d-text-sm">F</text>
  <!-- merge AB at h=60 -->
  <line x1="80" y1="270" x2="80" y2="210" class="d-line"/>
  <line x1="150" y1="270" x2="150" y2="210" class="d-line"/>
  <line x1="80" y1="210" x2="150" y2="210" class="d-line"/>
  <!-- merge (AB)C at h=150 -->
  <line x1="115" y1="210" x2="115" y2="150" class="d-line"/>
  <line x1="220" y1="270" x2="220" y2="150" class="d-line"/>
  <line x1="115" y1="150" x2="220" y2="150" class="d-line"/>
  <!-- merge DE at h=80 -->
  <line x1="380" y1="270" x2="380" y2="190" class="d-line"/>
  <line x1="450" y1="270" x2="450" y2="190" class="d-line"/>
  <line x1="380" y1="190" x2="450" y2="190" class="d-line"/>
  <!-- merge (DE)F at h=130 -->
  <line x1="415" y1="190" x2="415" y2="140" class="d-line"/>
  <line x1="540" y1="270" x2="540" y2="140" class="d-line"/>
  <line x1="415" y1="140" x2="540" y2="140" class="d-line"/>
  <!-- final merge at h=40 -->
  <line x1="167" y1="150" x2="167" y2="55" class="d-line"/>
  <line x1="477" y1="140" x2="477" y2="55" class="d-line"/>
  <line x1="167" y1="55" x2="477" y2="55" class="d-line"/>
  <!-- cut line -->
  <line x1="40" y1="105" x2="600" y2="105" class="d-line-accent" stroke-dasharray="7 5"/>
  <text x="560" y="97" class="d-text-accent">cut here → 2 clusters</text>
  <text x="30" y="60" class="d-text-sm">merge</text>
  <text x="30" y="74" class="d-text-sm">distance</text>
  <text x="30" y="88" class="d-text-sm">↑</text>
</svg>
<div class="caption">Height = distance at which clusters merged. Cutting the tree at any
height yields a flat clustering; tall vertical stretches mark natural gaps.</div>
</div>
<p>Two readings make dendrograms valuable: <strong>cut anywhere</strong> to get any
number of clusters without re-running, and <strong>long vertical gaps</strong> -
where clusters resisted merging across a wide distance range - signal the
"natural" cluster count.</p>

<h2>Linkage: what "distance between clusters" means</h2>
<p>Points have distances; clusters need a definition. The choice changes everything:</p>
<table>
  <tr><th>Linkage</th><th>Cluster distance =</th><th>Character</th></tr>
  <tr><td><strong>Single</strong></td><td>closest pair across clusters</td><td>finds elongated/chained shapes; suffers "chaining" - one noisy bridge merges everything</td></tr>
  <tr><td><strong>Complete</strong></td><td>farthest pair</td><td>compact, similar-diameter clusters; outlier-sensitive</td></tr>
  <tr><td><strong>Average</strong></td><td>mean of all cross-pairs</td><td>the reasonable compromise</td></tr>
  <tr><td><strong>Ward</strong> (default)</td><td>merge that least increases within-cluster variance</td><td>K-Means-like balanced blobs; the usual best start</td></tr>
</table>

<h2>In code</h2>
<pre><code class="language-python">import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

X, _ = make_blobs(n_samples=60, centers=3, cluster_std=1.0, random_state=2)
X = StandardScaler().fit_transform(X)

Z = linkage(X, method="ward")          # the full merge history

plt.figure(figsize=(10, 4))
dendrogram(Z)
plt.axhline(y=9, linestyle="--")       # eyeball the big gap, cut there
plt.title("dendrogram - cut at the tall gap")
plt.show()

labels = fcluster(Z, t=3, criterion="maxclust")   # or cut by distance: t=9
print("cluster sizes:", np.bincount(labels)[1:])

# sklearn equivalent (no dendrogram, but pipeline-friendly):
from sklearn.cluster import AgglomerativeClustering
agg = AgglomerativeClustering(n_clusters=3, linkage="ward").fit(X)
</code></pre>

<h2>Hierarchical vs K-Means</h2>
<table>
  <tr><th></th><th>Hierarchical</th><th>K-Means</th></tr>
  <tr><td>k needed upfront?</td><td>No - cut later, anywhere</td><td>Yes</td></tr>
  <tr><td>Output</td><td>full tree of nested structure</td><td>one flat partition</td></tr>
  <tr><td>Deterministic?</td><td>Yes (no random init)</td><td>No (restarts needed)</td></tr>
  <tr><td>Cost</td><td>O(m²) memory/time - caps at ~10-50k points</td><td>fast, scales to millions</td></tr>
  <tr><td>Natural fit</td><td>taxonomies: species, documents, market segments with sub-segments</td><td>large flat segmentation</td></tr>
</table>
<p>The nested structure is the real differentiator: "electronics → phones → budget
phones" is information a flat clustering simply cannot express. Biology's
phylogenetic trees are dendrograms of genetic distance - the method predates ML.</p>

<h2>Common interview questions</h2>
<div class="qa">
  <div class="qa-q">Q1. How do you choose the number of clusters from a dendrogram?</div>
  <div class="qa-a"><p>Cut across the tallest vertical span - the distance range where no merges wanted
  to happen. The number of lines the cut crosses is your cluster count. (Validate with
  silhouette scores across candidate cuts.)</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q2. What is the chaining problem?</div>
  <div class="qa-a"><p>Single linkage merges clusters if <em>any</em> two points are close, so a thin
  bridge of noise points can daisy-chain distinct clusters into one long snake. Complete,
  average, or Ward linkage resist it.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q3. Why not use hierarchical clustering on 5 million rows?</div>
  <div class="qa-a"><p>The pairwise-distance matrix alone is ~O(m²) - 25×10¹² entries. Standard practice:
  K-Means down to a few hundred micro-clusters, then hierarchical on the centroids to get
  the interpretable tree.</p></div>
</div>

<h2>Quick quiz</h2>
<div class="quiz">
  <div class="quiz-q" data-answer="0">
    <p class="quiz-question">1. The height of a merge in a dendrogram represents…</p>
    <button class="quiz-opt">The distance between the two clusters when they merged</button>
    <button class="quiz-opt">The number of points involved</button>
    <button class="quiz-opt">The order of merging only</button>
    <div class="quiz-explain hidden">Height encodes dissimilarity - which is why a tall stretch with no merges marks a natural boundary between cluster counts.</div>
  </div>
  <div class="quiz-q" data-answer="2">
    <p class="quiz-question">2. Which linkage would best recover a long, snake-shaped cluster?</p>
    <button class="quiz-opt">Ward</button>
    <button class="quiz-opt">Complete</button>
    <button class="quiz-opt">Single</button>
    <div class="quiz-explain hidden">Single linkage only needs consecutive points to be close - perfect for elongated shapes (and the same property causes chaining on noisy data).</div>
  </div>
  <div class="quiz-q" data-answer="1">
    <p class="quiz-question">3. Unlike K-Means, agglomerative clustering run twice on the same data gives…</p>
    <button class="quiz-opt">Different results each time</button>
    <button class="quiz-opt">Identical results - there is no random initialization</button>
    <button class="quiz-opt">Results depending on the random seed</button>
    <div class="quiz-explain hidden">The merge sequence is fully determined by the distances and linkage rule. Determinism is an underrated practical virtue.</div>
  </div>
</div>
`};

CONTENT["ml/dbscan"] = {
  html: String.raw`
<h1>DBSCAN</h1>
<p class="lead">Density-Based Spatial Clustering of Applications with Noise. The name
is a mouthful but the idea is elegant: clusters are <em>dense regions</em> separated
by sparse ones. No k to choose, any cluster shape, and - uniquely - an explicit
concept of noise.</p>

<h2>Two parameters, three kinds of points</h2>
<p>DBSCAN sees the world through two knobs: <strong>eps</strong> (ε - the radius of each
point's neighborhood) and <strong>min_samples</strong> (how many neighbors make a
neighborhood "dense"). Every point then gets one of three roles:</p>
<ul>
  <li><strong>Core point:</strong> has ≥ min_samples points within ε. The interior of a cluster.</li>
  <li><strong>Border point:</strong> not dense itself, but within ε of a core point. The cluster's skin.</li>
  <li><strong>Noise:</strong> neither. Belongs to nothing - and that's a feature, not a failure.</li>
</ul>
<div class="diagram">
<svg viewBox="0 0 640 260" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Dense cluster of core points with epsilon circles, border points on the edge, and isolated noise points">
  <rect x="20" y="20" width="600" height="215" rx="8" class="d-box-muted"/>
  <!-- core cluster -->
  <circle cx="180" cy="120" r="52" class="d-line-accent" fill="none" stroke-dasharray="5 4"/>
  <circle cx="180" cy="120" r="6" class="d-dot"/>
  <circle cx="205" cy="100" r="5" class="d-dot"/>
  <circle cx="160" cy="95" r="5" class="d-dot"/>
  <circle cx="150" cy="140" r="5" class="d-dot"/>
  <circle cx="200" cy="145" r="5" class="d-dot"/>
  <circle cx="225" cy="125" r="5" class="d-dot"/>
  <text x="180" y="195" text-anchor="middle" class="d-text-sm">core: ≥ min_samples inside its ε-circle</text>
  <!-- border -->
  <circle cx="262" cy="88" r="5" class="d-dot-muted"/>
  <line x1="262" y1="88" x2="225" y2="112" class="d-grid"/>
  <text x="300" y="70" class="d-text-sm">border: sparse itself,</text>
  <text x="300" y="86" class="d-text-sm">but inside a core's reach</text>
  <!-- noise -->
  <circle cx="480" cy="70" r="5" class="d-dot-muted" opacity="0.55"/>
  <circle cx="545" cy="170" r="5" class="d-dot-muted" opacity="0.55"/>
  <circle cx="430" cy="200" r="5" class="d-dot-muted" opacity="0.55"/>
  <text x="487" y="120" class="d-text-sm">noise: labeled −1,</text>
  <text x="487" y="136" class="d-text-sm">assigned to no cluster</text>
</svg>
<div class="caption">Density defines membership. Clusters grow by chaining core points;
anything unreachable stays noise.</div>
</div>
<p>The algorithm: pick an unvisited point; if it's core, start a cluster and
recursively absorb everything density-reachable from it (cores keep expanding the
frontier; borders join but don't expand). When the flood stops, move to the next
unvisited point. Clusters of <em>any shape</em> emerge - crescents, rings, spirals -
because growth follows density, not distance-to-a-center.</p>

<h2>Choosing eps: the k-distance elbow</h2>
<p>min_samples is easy (rule of thumb: ≥ dimensions + 1, commonly 5–20; higher on
noisy data). eps is the sensitive one - too small fragments everything into noise,
too large merges the world into one blob. The standard picker:</p>
<pre><code class="language-python">import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons

X, _ = make_moons(n_samples=500, noise=0.08, random_state=0)
X = StandardScaler().fit_transform(X)

# sort every point's distance to its k-th neighbor; eps ≈ the elbow
k = 5
dists, _ = NearestNeighbors(n_neighbors=k).fit(X).kneighbors(X)
plt.plot(np.sort(dists[:, -1]))
plt.ylabel(f"distance to {k}th neighbor"); plt.title("read eps at the elbow")
plt.show()

db = DBSCAN(eps=0.22, min_samples=5).fit(X)
labels = db.labels_                      # -1 = noise
print("clusters:", len(set(labels)) - (1 if -1 in labels else 0))
print("noise points:", (labels == -1).sum())
# the two crescents - which K-Means cuts in half - come out perfectly
</code></pre>

<h2>Where DBSCAN wins and where it breaks</h2>
<table>
  <tr><th>Wins</th><th>Breaks</th></tr>
  <tr><td>Arbitrary cluster shapes (moons, rings, spatial traces)</td><td><strong>Varying densities:</strong> one eps can't fit a dense cluster and a sparse one simultaneously (HDBSCAN fixes this)</td></tr>
  <tr><td>k discovered, not declared</td><td>eps is fiddly and scale-dependent (standardize first!)</td></tr>
  <tr><td>Native outlier detection - noise is a first-class output</td><td>High dimensions: density becomes meaningless (curse of dimensionality again)</td></tr>
  <tr><td>Deterministic (up to border ties); robust to outliers by design</td><td>Border points make "probability of membership" ill-defined</td></tr>
</table>
<p>Classic use cases: geospatial hotspots (crimes, check-ins - clusters are
street-shaped, not round), anomaly detection via the noise label
(<a href="#/eda/outliers">Outlier Detection</a>), and any data where "how many
clusters?" is unanswerable in advance. For clusters of varying density, the modern
successor <strong>HDBSCAN</strong> (hierarchical DBSCAN) removes eps entirely and is
usually the better default library choice.</p>

<h2>The three clusterers, side by side</h2>
<table>
  <tr><th></th><th><a href="#/ml/kmeans">K-Means</a></th><th><a href="#/ml/hierarchical-clustering">Hierarchical</a></th><th>DBSCAN</th></tr>
  <tr><td>k required?</td><td>yes</td><td>no (cut later)</td><td>no</td></tr>
  <tr><td>Shapes</td><td>convex blobs</td><td>depends on linkage</td><td>arbitrary</td></tr>
  <tr><td>Noise concept</td><td>none</td><td>none</td><td><strong>yes</strong></td></tr>
  <tr><td>Scale</td><td>millions</td><td>~10⁴</td><td>~10⁵–10⁶ (with indexing)</td></tr>
  <tr><td>Key knob</td><td>k</td><td>linkage + cut</td><td>eps, min_samples</td></tr>
</table>

<h2>Common interview questions</h2>
<div class="qa">
  <div class="qa-q">Q1. Core vs border vs noise - define them.</div>
  <div class="qa-a"><p>Core: ≥ min_samples neighbors within ε. Border: fewer, but lies within ε of some
  core point (joins that cluster, can't extend it). Noise: neither - labeled −1 and
  excluded. Only core points propagate cluster growth.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q2. Why does DBSCAN find crescent shapes that K-Means can't?</div>
  <div class="qa-a"><p>K-Means assigns by distance to a centroid, producing convex cells. DBSCAN grows
  clusters by local density connectivity - a chain of nearby dense points - so any
  connected dense shape, however curved, is one cluster.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q3. Your dataset has one tight cluster and one diffuse cluster. What happens?</div>
  <div class="qa-a"><p>The single global eps fails: small eps shatters the diffuse cluster into noise;
  large eps bleeds the tight cluster into its surroundings. Use HDBSCAN (varying-density
  aware) or cluster the regions separately.</p></div>
</div>

<h2>Quick quiz</h2>
<div class="quiz">
  <div class="quiz-q" data-answer="1">
    <p class="quiz-question">1. A point with label −1 in DBSCAN's output is…</p>
    <button class="quiz-opt">A member of cluster −1</button>
    <button class="quiz-opt">Noise - dense enough for nothing</button>
    <button class="quiz-opt">A core point</button>
    <div class="quiz-explain hidden">−1 is the noise label. Having an explicit "belongs nowhere" verdict is DBSCAN's signature advantage - and a free anomaly detector.</div>
  </div>
  <div class="quiz-q" data-answer="0">
    <p class="quiz-question">2. Doubling eps generally…</p>
    <button class="quiz-opt">Merges clusters and shrinks the noise set</button>
    <button class="quiz-opt">Splits clusters apart</button>
    <button class="quiz-opt">Has no effect</button>
    <div class="quiz-explain hidden">Bigger neighborhoods = more points qualify as core and previously separate dense regions connect. eps is the resolution dial.</div>
  </div>
  <div class="quiz-q" data-answer="2">
    <p class="quiz-question">3. GPS points of taxi pickups along winding streets. Best clusterer?</p>
    <button class="quiz-opt">K-Means</button>
    <button class="quiz-opt">Ward hierarchical</button>
    <button class="quiz-opt">DBSCAN</button>
    <div class="quiz-explain hidden">Street-shaped hotspots are elongated, curved, and mixed with scattered noise - the exact geometry density-based clustering was built for.</div>
  </div>
</div>
`};

CONTENT["ml/evaluation-metrics"] = {
  html: String.raw`
<h1>Model Evaluation Metrics</h1>
<p class="lead">"Is the model good?" is not one question - it's several, and each
metric answers a different one. Choosing the metric <em>is</em> choosing what you
care about. This page is the reference for both classification and regression.</p>

<h2>Classification: it all starts with the confusion matrix</h2>
<table>
  <tr><th></th><th>predicted positive</th><th>predicted negative</th></tr>
  <tr><td><strong>actually positive</strong></td><td>TP (true positive)</td><td>FN (false negative - a <em>miss</em>)</td></tr>
  <tr><td><strong>actually negative</strong></td><td>FP (false positive - a <em>false alarm</em>)</td><td>TN (true negative)</td></tr>
</table>
<p>Every classification metric is a ratio of these four cells. Running example:
a cancer screen on 1,000 patients (50 sick): TP = 45, FN = 5, FP = 90, TN = 860.</p>
<div class="math-box">
$$\text{accuracy} = \frac{TP + TN}{\text{all}} = \frac{905}{1000} = 0.905$$
$$\text{precision} = \frac{TP}{TP + FP} = \frac{45}{135} = 0.33
\qquad
\text{recall} = \frac{TP}{TP + FN} = \frac{45}{50} = 0.90$$
$$F_1 = 2\cdot\frac{\text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}} = 0.48$$
</div>
<p>Read the story these numbers tell: 90.5% accuracy <em>sounds</em> excellent - but a
"nobody is sick" model scores 95%! Meanwhile precision says only 1 in 3 alarms is
real, and recall says we catch 9 of 10 cancers. Three metrics, three truths:</p>
<ul>
  <li><strong>Precision</strong> - "when it alarms, is it right?" Optimize when false
      alarms are costly: spam filters (real mail in junk!), arrest decisions.</li>
  <li><strong>Recall</strong> (sensitivity) - "of the real ones, how many caught?"
      Optimize when misses are costly: cancer, fraud, safety recalls.</li>
  <li><strong>F1</strong> - harmonic mean; punishes imbalance between the two
      (harmonic, so 0.9 &amp; 0.1 give 0.18, not 0.5). The single-number compromise.</li>
</ul>
<p>Precision and recall <strong>trade off through the threshold</strong>: lower it and
you catch more (recall ↑) but alarm more falsely (precision ↓) - the dial you met
in <a href="#/features/imbalanced-data">Handling Imbalanced Data</a>.</p>

<h2>Threshold-free: ROC-AUC and PR-AUC</h2>
<p>To compare <em>models</em> rather than <em>thresholds</em>, integrate over all
thresholds (<a href="#/dataviz/ml-visualizations">curve pictures here</a>):</p>
<ul>
  <li><strong>ROC-AUC</strong> - area under the TPR-vs-FPR curve. Elegant meaning:
      <em>the probability that a random positive scores higher than a random
      negative</em>. 0.5 = coin flip, 1.0 = perfect ranking.</li>
  <li><strong>PR-AUC</strong> - area under precision-vs-recall. Under heavy imbalance
      prefer this: ROC's FPR denominator (all the negatives) is so huge that
      thousands of false positives barely move it, while precision collapses -
      PR-AUC shows the pain, ROC-AUC hides it.</li>
</ul>

<h2>Regression metrics</h2>
<div class="math-box">
$$\mathrm{MAE} = \frac{1}{m}\sum \big|\hat{y}^{(i)} - y^{(i)}\big|
\qquad
\mathrm{RMSE} = \sqrt{\frac{1}{m}\sum \big(\hat{y}^{(i)} - y^{(i)}\big)^2}$$
$$R^2 = 1 - \frac{\sum (y^{(i)} - \hat{y}^{(i)})^2}{\sum (y^{(i)} - \bar{y})^2}$$
</div>
<ul>
  <li><strong>MAE</strong> - average miss in original units ("off by ₹4.2 lakh on
      average"). Robust: one huge error counts once.</li>
  <li><strong>RMSE</strong> - same units, but squares first: large errors dominate.
      RMSE ≫ MAE is a diagnostic - a few big misses are hiding in the average.
      Choose RMSE when big errors are disproportionately bad; MAE when they're not.</li>
  <li><strong>R²</strong> - fraction of variance explained vs always-predict-the-mean.
      0 = no better than the mean; can go <em>negative</em> on test data (worse than
      the mean!). Beware: R² always rises as you add features on training data -
      judge on held-out data.</li>
  <li><strong>MAPE</strong> (percentage error) - intuitive for business, but explodes
      near y = 0 and asymmetrically punishes over-prediction. Use with care.</li>
</ul>

<h2>All of it in code</h2>
<pre><code class="language-python">from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, average_precision_score,
                             mean_absolute_error, mean_squared_error, r2_score)
import numpy as np

# --- classification -------------------------------------------------------
print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred, digits=3))
# report shows precision/recall/F1 PER CLASS - read the minority row!

proba = model.predict_proba(X_test)[:, 1]
print("ROC-AUC:", roc_auc_score(y_true, proba).round(3))
print("PR-AUC :", average_precision_score(y_true, proba).round(3))

# --- regression ------------------------------------------------------------
mae  = mean_absolute_error(y_true_reg, y_pred_reg)
rmse = np.sqrt(mean_squared_error(y_true_reg, y_pred_reg))
print(f"MAE {mae:.2f} | RMSE {rmse:.2f} | R² {r2_score(y_true_reg, y_pred_reg):.3f}")
if rmse &gt; 1.5 * mae:
    print("RMSE >> MAE: a few large errors dominate - inspect the outliers!")
</code></pre>

<h2>Choosing the metric: a decision table</h2>
<table>
  <tr><th>Situation</th><th>Primary metric</th></tr>
  <tr><td>Balanced classes, all errors equal</td><td>accuracy (fine here!)</td></tr>
  <tr><td>Imbalanced, misses are expensive (cancer, fraud)</td><td>recall (+ precision as guardrail), PR-AUC</td></tr>
  <tr><td>Imbalanced, false alarms are expensive (spam)</td><td>precision (+ recall as guardrail)</td></tr>
  <tr><td>Comparing rankers/models across thresholds</td><td>ROC-AUC (balanced), PR-AUC (imbalanced)</td></tr>
  <tr><td>Regression, errors hurt linearly</td><td>MAE</td></tr>
  <tr><td>Regression, big errors hurt disproportionately</td><td>RMSE</td></tr>
  <tr><td>Explaining fit quality to stakeholders</td><td>R² + MAE in original units</td></tr>
</table>

<h2>Common interview questions</h2>
<div class="qa">
  <div class="qa-q">Q1. Precision vs recall - explain with a cancer screen and a spam filter.</div>
  <div class="qa-a"><p>Cancer: a miss (FN) can kill; maximize recall, accept false alarms that a follow-up
  test resolves. Spam: a false alarm (FP) buries real mail; maximize precision, accept some
  spam leaking through. Same math, opposite priorities - set by the cost of each error.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q2. Why can ROC-AUC look great while the model is useless in production?</div>
  <div class="qa-a"><p>At 1:10,000 imbalance, a model can rank well (high AUC) yet at any usable threshold
  produce 100 false positives per true positive - precision ~1%. FPR barely registers those
  FPs against millions of negatives. PR-AUC or precision@k reveal the truth.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q3. Your regression has MAE 5 but RMSE 25. What does that tell you?</div>
  <div class="qa-a"><p>Typical errors are small but rare catastrophic misses exist (squaring amplifies
  them). Investigate the tail: outliers in the target, a missing feature regime, or data
  errors. Reporting RMSE alone would have hidden the "usually fine"; MAE alone the "sometimes
  disastrous".</p></div>
</div>

<h2>Quick quiz</h2>
<div class="quiz">
  <div class="quiz-q" data-answer="1">
    <p class="quiz-question">1. TP=40, FP=10, FN=40, TN=910. Recall is…</p>
    <button class="quiz-opt">0.8</button>
    <button class="quiz-opt">0.5</button>
    <button class="quiz-opt">0.95</button>
    <div class="quiz-explain hidden">Recall = TP/(TP+FN) = 40/80 = 0.5 - half the real positives were missed, despite 95% accuracy. (Precision here is 0.8.)</div>
  </div>
  <div class="quiz-q" data-answer="2">
    <p class="quiz-question">2. Lowering the classification threshold from 0.5 to 0.2 typically…</p>
    <button class="quiz-opt">Raises precision, lowers recall</button>
    <button class="quiz-opt">Raises both</button>
    <button class="quiz-opt">Raises recall, lowers precision</button>
    <div class="quiz-explain hidden">More predictions become positive: you catch more true positives (recall ↑) but admit more false alarms (precision ↓). The threshold slides you along the PR curve.</div>
  </div>
  <div class="quiz-q" data-answer="0">
    <p class="quiz-question">3. ROC-AUC = 0.5 means the model…</p>
    <button class="quiz-opt">Ranks positives no better than random guessing</button>
    <button class="quiz-opt">Is 50% accurate</button>
    <button class="quiz-opt">Has 50% precision</button>
    <div class="quiz-explain hidden">AUC is P(random positive scores above random negative); 0.5 is a coin flip. It says nothing directly about accuracy or precision at any threshold.</div>
  </div>
</div>
`};
