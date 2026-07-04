/* Module 3 - Data Visualization Lab (part 1) */

CONTENT["dataviz/why-visualization"] = {
  html: String.raw`
<h1>Why Visualization Matters</h1>
<p class="lead">In 1973, statistician Francis Anscombe built four tiny datasets that
share <em>identical</em> means, variances, correlations, and regression lines -
yet look nothing alike. They remain the best two-minute argument ever made
for plotting your data.</p>

<h2>Anscombe's quartet: same statistics, different realities</h2>
<p>All four datasets below have: mean of x = 9, mean of y ≈ 7.5, variance of x = 11,
correlation r ≈ 0.816, and the same fitted line \(\hat{y} = 3 + 0.5x\).
Summary statistics say they're the same data. Your eyes say otherwise:</p>
<div class="diagram">
<svg viewBox="0 0 660 420" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Anscombe's quartet: four scatter plots with the same regression line but completely different shapes">
  <!-- panel I: normal linear cloud -->
  <rect x="20" y="25" width="290" height="170" rx="6" class="d-box-muted"/>
  <text x="30" y="45" class="d-text-accent">I - a sensible linear cloud</text>
  <line x1="40" y1="170" x2="290" y2="70" class="d-line-accent"/>
  <circle cx="70" cy="162" r="4" class="d-dot"/><circle cx="100" cy="138" r="4" class="d-dot"/>
  <circle cx="125" cy="150" r="4" class="d-dot"/><circle cx="150" cy="120" r="4" class="d-dot"/>
  <circle cx="175" cy="128" r="4" class="d-dot"/><circle cx="200" cy="100" r="4" class="d-dot"/>
  <circle cx="225" cy="112" r="4" class="d-dot"/><circle cx="250" cy="85" r="4" class="d-dot"/>
  <circle cx="270" cy="95" r="4" class="d-dot"/>
  <!-- panel II: curve -->
  <rect x="345" y="25" width="290" height="170" rx="6" class="d-box-muted"/>
  <text x="355" y="45" class="d-text-accent">II - a perfect curve, not a line</text>
  <line x1="365" y1="170" x2="615" y2="70" class="d-line-accent"/>
  <circle cx="385" cy="168" r="4" class="d-dot"/><circle cx="410" cy="138" r="4" class="d-dot"/>
  <circle cx="435" cy="115" r="4" class="d-dot"/><circle cx="460" cy="100" r="4" class="d-dot"/>
  <circle cx="485" cy="93" r="4" class="d-dot"/><circle cx="510" cy="95" r="4" class="d-dot"/>
  <circle cx="535" cy="105" r="4" class="d-dot"/><circle cx="560" cy="122" r="4" class="d-dot"/>
  <circle cx="585" cy="147" r="4" class="d-dot"/>
  <!-- panel III: line with one outlier -->
  <rect x="20" y="225" width="290" height="170" rx="6" class="d-box-muted"/>
  <text x="30" y="245" class="d-text-accent">III - a perfect line + one outlier</text>
  <line x1="40" y1="370" x2="290" y2="270" class="d-line-accent"/>
  <circle cx="65" cy="372" r="4" class="d-dot"/><circle cx="90" cy="364" r="4" class="d-dot"/>
  <circle cx="115" cy="356" r="4" class="d-dot"/><circle cx="140" cy="348" r="4" class="d-dot"/>
  <circle cx="165" cy="340" r="4" class="d-dot"/><circle cx="190" cy="332" r="4" class="d-dot"/>
  <circle cx="215" cy="324" r="4" class="d-dot"/><circle cx="240" cy="316" r="4" class="d-dot"/>
  <circle cx="205" cy="260" r="5" class="d-dot-muted"/>
  <text x="215" y="258" class="d-text-sm">← the liar</text>
  <!-- panel IV: vertical + outlier -->
  <rect x="345" y="225" width="290" height="170" rx="6" class="d-box-muted"/>
  <text x="355" y="245" class="d-text-accent">IV - no x-variation + one outlier</text>
  <line x1="365" y1="370" x2="615" y2="270" class="d-line-accent"/>
  <circle cx="420" cy="372" r="4" class="d-dot"/><circle cx="420" cy="356" r="4" class="d-dot"/>
  <circle cx="420" cy="342" r="4" class="d-dot"/><circle cx="420" cy="328" r="4" class="d-dot"/>
  <circle cx="420" cy="314" r="4" class="d-dot"/><circle cx="420" cy="300" r="4" class="d-dot"/>
  <circle cx="420" cy="348" r="4" class="d-dot"/><circle cx="420" cy="335" r="4" class="d-dot"/>
  <circle cx="590" cy="262" r="5" class="d-dot-muted"/>
</svg>
<div class="caption">Four datasets, one set of summary statistics. Only panel I is what
the statistics imply. II needs a quadratic model, III needs outlier handling, IV has
essentially no relationship at all.</div>
</div>
<p>The modern sequel is the <em>Datasaurus Dozen</em>: thirteen datasets with identical
statistics where one of them is a drawing of a dinosaur. The lesson has not changed
in fifty years: <strong>statistics summarize; only plots reveal.</strong></p>

<h2>What visualization is <em>for</em></h2>
<p>Three distinct jobs, needing different levels of polish:</p>
<ul>
  <li><strong>Exploration (for you):</strong> fast, ugly, dozens per hour. Spot skew,
      outliers, clusters, data errors. This is most of Module 4 (EDA).</li>
  <li><strong>Diagnosis (for your model):</strong> loss curves, residual plots, confusion
      matrices - is the model learning, and <em>what kind</em> of wrong is it?
      Covered in <a href="#/dataviz/ml-visualizations">Visualizations for ML</a>.</li>
  <li><strong>Communication (for others):</strong> few, polished, honest. One message per
      chart, labeled axes, no distortion. Covered in
      <a href="#/dataviz/paper-visualizations">Visualizations in Research Papers</a>.</li>
</ul>

<h2>Verify Anscombe yourself</h2>
<pre><code class="language-python">import seaborn as sns
import matplotlib.pyplot as plt

df = sns.load_dataset("anscombe")     # ships with seaborn

# identical statistics...
print(df.groupby("dataset").agg(
    x_mean=("x", "mean"), y_mean=("y", "mean"),
    x_var=("x", "var"),
    r=("x", lambda s: df.loc[s.index, ["x", "y"]].corr().iloc[0, 1]),
).round(2))

# ...completely different pictures
sns.lmplot(data=df, x="x", y="y", col="dataset", col_wrap=2,
           ci=None, height=3, scatter_kws={"s": 40})
plt.show()
</code></pre>

<h2>The honesty rules</h2>
<p>Because plots persuade so effectively, they can also lie effectively. The four
classic sins to avoid (and to spot in others' charts):</p>
<table>
  <tr><th>Sin</th><th>Effect</th><th>Rule</th></tr>
  <tr><td>Truncated y-axis on bar charts</td><td>A 2% difference looks like 2×</td><td>Bars start at zero. Always.</td></tr>
  <tr><td>Cherry-picked time windows</td><td>Any trend you want</td><td>Show the full relevant range; justify the window.</td></tr>
  <tr><td>Dual y-axes</td><td>Fake correlations by scaling</td><td>Avoid; use two panels or indexed values.</td></tr>
  <tr><td>3-D / area distortions</td><td>Volume exaggerates ratios</td><td>Encode quantities as lengths/positions, not areas or volumes.</td></tr>
</table>

<h2>Quick quiz</h2>
<div class="quiz">
  <div class="quiz-q" data-answer="1">
    <p class="quiz-question">1. Anscombe's quartet demonstrates that…</p>
    <button class="quiz-opt">Correlation is always misleading</button>
    <button class="quiz-opt">Identical summary statistics can hide completely different data structures</button>
    <button class="quiz-opt">Regression only works on linear data</button>
    <div class="quiz-explain hidden">Same means, variances, r, and regression line - yet a line, a curve, an outlier artifact, and a no-relationship dataset. Summaries compress; plots reveal.</div>
  </div>
  <div class="quiz-q" data-answer="2">
    <p class="quiz-question">2. A bar chart of conversion rates starts its y-axis at 4.0% instead of 0. The problem?</p>
    <button class="quiz-opt">Nothing - it zooms into the interesting region</button>
    <button class="quiz-opt">The bars should be horizontal</button>
    <button class="quiz-opt">Bar length no longer encodes the quantity, so small differences look enormous</button>
    <div class="quiz-explain hidden">A bar visually says "my length IS the value". Truncating the axis breaks that contract - 4.2% vs 4.4% renders as a 2× longer bar. (Line charts may zoom; bars may not.)</div>
  </div>
</div>
`};

CONTENT["dataviz/matplotlib"] = {
  html: String.raw`
<h1>Matplotlib Basics</h1>
<p class="lead">Matplotlib is the foundation every other Python plotting library
builds on. Learn its object model once - Figure, Axes, and the handful of plot
types - and Seaborn, Pandas plotting, and half of Plotly will suddenly make sense.</p>

<h2>The anatomy: Figure and Axes</h2>
<p>The naming trips everyone: a <strong>Figure</strong> is the whole canvas/window;
an <strong>Axes</strong> is <em>one plot</em> living on it (not the x/y lines - those are
<em>axis</em>, singular). A figure can hold many axes (subplots).</p>
<div class="diagram">
<svg viewBox="0 0 640 300" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="A Figure containing two Axes side by side, with title, axis labels, ticks and legend annotated">
  <rect x="20" y="20" width="600" height="255" rx="8" class="d-box"/>
  <text x="30" y="42" class="d-text-accent">Figure - the whole canvas (fig)</text>
  <!-- axes 1 -->
  <rect x="55" y="70" width="240" height="165" rx="4" class="d-box-soft"/>
  <text x="175" y="63" text-anchor="middle" class="d-text-sm">Axes #1 (ax1) - one plot</text>
  <line x1="80" y1="215" x2="275" y2="215" class="d-line"/>
  <line x1="80" y1="215" x2="80" y2="90" class="d-line"/>
  <polyline points="90,200 130,160 170,175 210,120 250,105" class="d-curve"/>
  <text x="177" y="232" text-anchor="middle" class="d-text-sm">xlabel</text>
  <text x="68" y="150" class="d-text-sm" transform="rotate(-90 68 150)">ylabel</text>
  <!-- axes 2 -->
  <rect x="340" y="70" width="240" height="165" rx="4" class="d-box-soft"/>
  <text x="460" y="63" text-anchor="middle" class="d-text-sm">Axes #2 (ax2)</text>
  <line x1="365" y1="215" x2="560" y2="215" class="d-line"/>
  <line x1="365" y1="215" x2="365" y2="90" class="d-line"/>
  <rect x="385" y="150" width="30" height="65" class="d-box-soft"/>
  <rect x="425" y="120" width="30" height="95" class="d-box-soft"/>
  <rect x="465" y="170" width="30" height="45" class="d-box-soft"/>
  <rect x="505" y="100" width="30" height="115" class="d-box-soft"/>
  <text x="320" y="262" text-anchor="middle" class="d-text-sm">fig, (ax1, ax2) = plt.subplots(1, 2)</text>
</svg>
<div class="caption">One Figure, two Axes. Every label, limit, and plot call belongs
to a specific Axes.</div>
</div>
<p>There are two APIs. The quick <code>plt.plot(...)</code> style is fine for one-liners,
but the <strong>object-oriented style</strong> below scales to real work and is the one
used everywhere on this site:</p>
<pre><code class="language-python">import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 200)

fig, ax = plt.subplots(figsize=(8, 4.5))       # create Figure + one Axes
ax.plot(x, np.sin(x), label="sin(x)", linewidth=2)
ax.plot(x, np.cos(x), label="cos(x)", linestyle="--")
ax.set_xlabel("x")
ax.set_ylabel("value")
ax.set_title("The OO pattern: create, plot, label, show")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
</code></pre>
<p>That five-line rhythm - <em>subplots → plot → labels → legend → show</em> -
is 80% of matplotlib. Everything else is variations.</p>

<h2>The core plot types</h2>
<pre><code class="language-python">rng = np.random.default_rng(0)
data = rng.normal(70, 12, 500)                 # e.g. exam scores
cats = ["A", "B", "C", "D"]
vals = [23, 41, 12, 30]

fig, axes = plt.subplots(2, 2, figsize=(10, 7))

# line - trends over an ordered axis (time, epochs)
axes[0, 0].plot(np.cumsum(rng.normal(0, 1, 100)))
axes[0, 0].set_title("line: trend over time")

# histogram - the SHAPE of one variable (bins matter: try several!)
axes[0, 1].hist(data, bins=30, edgecolor="white")
axes[0, 1].set_title("hist: distribution of scores")

# scatter - relationship between two variables
x2 = rng.normal(size=200)
axes[1, 0].scatter(x2, 2 * x2 + rng.normal(0, 1, 200), s=15, alpha=0.6)
axes[1, 0].set_title("scatter: x vs y")

# bar - quantities across categories
axes[1, 1].bar(cats, vals)
axes[1, 1].set_title("bar: counts per category")

plt.tight_layout()
plt.show()
</code></pre>

<h2>Customization that actually matters</h2>
<p>Resist decorating. The adjustments worth knowing are the ones that change
whether a chart can be read:</p>
<pre><code class="language-python">fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(x, np.exp(x * 0.5))

ax.set_yscale("log")            # log scale: THE tool for exponential data
ax.set_xlim(0, 8)               # zoom (fine for lines, never for bar baselines)
ax.axhline(100, color="gray", linestyle=":", label="threshold")   # reference line
ax.annotate("crosses here", xy=(9.2, 100), xytext=(5.5, 400),
            arrowprops=dict(arrowstyle="-&gt;"))
ax.legend()

fig.savefig("figure.png", dpi=200, bbox_inches="tight")   # publication export
</code></pre>
<ul>
  <li><strong>Log scales</strong> straighten exponential growth and reveal structure
      spanning orders of magnitude (loss curves love them).</li>
  <li><strong>Reference lines</strong> (<code>axhline/axvline</code>) give numbers context:
      baseline accuracy, a threshold, "chance level".</li>
  <li><strong>figsize + tight_layout + dpi</strong> are what make figures legible in
      documents instead of cramped thumbnails.</li>
</ul>

<h2>Common interview questions</h2>
<div class="qa">
  <div class="qa-q">Q1. Figure vs Axes vs axis?</div>
  <div class="qa-a"><p>Figure = the whole canvas. Axes = one plot region on it (a figure may contain many).
  axis = the x or y number line of one Axes. <code>plt.subplots(2, 3)</code> returns one Figure
  and a 2×3 array of Axes.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q2. When do you reach for a log scale?</div>
  <div class="qa-a"><p>When values span several orders of magnitude (city populations, word frequencies)
  or grow/decay multiplicatively (training loss, epidemics). On a log axis, exponential
  curves become straight lines and relative changes become comparable everywhere.</p></div>
</div>

<h2>Quick quiz</h2>
<div class="quiz">
  <div class="quiz-q" data-answer="1">
    <p class="quiz-question">1. You want the distribution shape of a single numeric column. First choice?</p>
    <button class="quiz-opt">Scatter plot</button>
    <button class="quiz-opt">Histogram</button>
    <button class="quiz-opt">Pie chart</button>
    <div class="quiz-explain hidden">Histograms show center, spread, skew, modes, and outliers of one variable. Scatter needs two variables; pie charts show parts of a whole.</div>
  </div>
  <div class="quiz-q" data-answer="0">
    <p class="quiz-question">2. fig, axes = plt.subplots(2, 2) - what is axes?</p>
    <button class="quiz-opt">A 2×2 NumPy array of Axes objects</button>
    <button class="quiz-opt">Four Figure objects</button>
    <button class="quiz-opt">A list of x/y axis lines</button>
    <div class="quiz-explain hidden">One Figure, four Axes in a 2×2 grid, accessed as axes[row, col] - each an independent plot.</div>
  </div>
</div>
`};

CONTENT["dataviz/seaborn"] = {
  html: String.raw`
<h1>Seaborn Basics</h1>
<p class="lead">Seaborn is matplotlib with a statistics degree. It speaks DataFrame
natively - you name columns, it handles grouping, coloring, aggregating, and even
confidence intervals. One line of Seaborn often replaces twenty of matplotlib.</p>

<h2>The mental model: map columns to visual roles</h2>
<p>Every Seaborn call has the same grammar: <em>data =</em> a DataFrame, then map
column names to roles - <code>x</code>, <code>y</code>, <code>hue</code> (color),
<code>size</code>, <code>style</code>, <code>col</code>/<code>row</code> (facets).
The library does the rest:</p>
<pre><code class="language-python">import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")     # a classic demo dataset
print(tips.head())                  # total_bill, tip, sex, smoker, day, time, size

# one line: scatter + third variable as color + fourth as marker style
sns.scatterplot(data=tips, x="total_bill", y="tip",
                hue="time", style="smoker")
plt.show()
</code></pre>
<p><code>hue</code> is the killer argument. "Does the pattern differ by group?" is the
most common question in analysis, and <code>hue=</code> answers it in every plot type
without a single loop.</p>

<h2>Distribution plots: the EDA workhorses</h2>
<pre><code class="language-python"># histogram + smooth density estimate, split by a category
sns.histplot(data=tips, x="total_bill", hue="time", kde=True, alpha=0.5)

# box plot: median, quartiles, whiskers, outliers - per category
sns.boxplot(data=tips, x="day", y="total_bill")

# violin: box plot + full distribution shape (spot bimodality!)
sns.violinplot(data=tips, x="day", y="total_bill", inner="quartile")

# ECDF: cumulative view, no bin-choice bias, great for comparisons
sns.ecdfplot(data=tips, x="total_bill", hue="time")
</code></pre>
<p>Box plots summarize five numbers; violins show the whole shape (a two-humped
violin means two subpopulations a box plot would hide); ECDFs let you read
"what fraction is below X" directly. Rotate through all three during EDA -
each reveals what the others smooth over.</p>

<h2>Relationships and the pair plot</h2>
<pre><code class="language-python"># scatter + fitted regression with confidence band
sns.regplot(data=tips, x="total_bill", y="tip")

# EVERY numeric column against every other - the 30-second dataset overview
sns.pairplot(tips, hue="time", corner=True)

# correlation heatmap (details in Module 4 → Correlation Analysis)
corr = tips.select_dtypes("number").corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag",
            vmin=-1, vmax=1, square=True)
</code></pre>
<p><code>pairplot</code> on a new dataset is a ritual: histograms on the diagonal,
scatter plots off it. In one image you spot skew, outliers, clusters, and which
features actually relate to each other.</p>

<h2>Facets: small multiples with col= and row=</h2>
<p>When <code>hue</code> gets crowded, split groups into separate panels that share
axes - the "small multiples" technique, and one of the strongest ideas in all of
visualization:</p>
<pre><code class="language-python"># figure-level functions (relplot / displot / catplot) take col= and row=
sns.relplot(data=tips, x="total_bill", y="tip",
            col="time", row="smoker", hue="day", height=3)

# aggregate with error bars: catplot computes means + 95% CIs for you
sns.catplot(data=tips, x="day", y="total_bill",
            kind="bar", errorbar="ci", height=4)
</code></pre>
<p>Note the two families: <em>axes-level</em> functions (<code>scatterplot</code>,
<code>boxplot</code>…) draw on one matplotlib Axes and mix freely with matplotlib code;
<em>figure-level</em> ones (<code>relplot</code>, <code>displot</code>,
<code>catplot</code>) manage their own figure to enable faceting. If you ever wonder
"why can't I put this on my subplot?" - you grabbed a figure-level function.</p>

<h2>Making it presentable in two lines</h2>
<pre><code class="language-python">sns.set_theme(style="whitegrid", context="talk", palette="deep")
# style: background/grid  ·  context: sizes for paper/notebook/talk/poster
# palette: "deep", "colorblind" (use it!), "vlag"/"coolwarm" for diverging
</code></pre>

<h2>Common interview questions</h2>
<div class="qa">
  <div class="qa-q">Q1. When a violin plot over a box plot?</div>
  <div class="qa-a"><p>When distribution <em>shape</em> matters - especially multimodality. A box plot
  reduces data to quartiles and can look identical for a normal and a two-humped
  distribution; the violin's width profile exposes the difference immediately.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q2. What does hue= do, mechanically?</div>
  <div class="qa-a"><p>It partitions rows by a column's values and renders each partition with a distinct
  color (plus a legend) - the visual equivalent of a groupby applied inside the plot.
  On aggregating plots (barplot, lineplot) it also computes per-group statistics and CIs.</p></div>
</div>

<h2>Quick quiz</h2>
<div class="quiz">
  <div class="quiz-q" data-answer="2">
    <p class="quiz-question">1. Fastest way to see all pairwise relationships in a new numeric dataset?</p>
    <button class="quiz-opt">A for-loop of plt.scatter calls</button>
    <button class="quiz-opt">sns.heatmap of the raw data</button>
    <button class="quiz-opt">sns.pairplot(df)</button>
    <div class="quiz-explain hidden">pairplot builds the full scatter-matrix with distributions on the diagonal in one call - the standard first look at any dataset.</div>
  </div>
  <div class="quiz-q" data-answer="1">
    <p class="quiz-question">2. A violin plot of response times shows two clear bulges. Interpretation?</p>
    <button class="quiz-opt">The data is normally distributed</button>
    <button class="quiz-opt">Two subpopulations exist (e.g. cache hits vs misses) - investigate before averaging</button>
    <button class="quiz-opt">The plot is broken</button>
    <div class="quiz-explain hidden">Bimodality means the mean describes nobody: it lands between the humps. Finding the hidden grouping variable is usually a real discovery.</div>
  </div>
</div>
`};

CONTENT["dataviz/plotly"] = {
  html: String.raw`
<h1>Plotly (Interactive)</h1>
<p class="lead">Matplotlib and Seaborn produce pictures; Plotly produces
<em>instruments</em> - charts you can hover, zoom, filter, and share as living
HTML files. For exploring dense data and for dashboards, interactivity changes
what you can see.</p>

<h2>Why interactivity earns its keep</h2>
<ul>
  <li><strong>Hover tooltips</strong> - "which point is that?" answered instantly;
      no more re-plotting with labels to identify one outlier.</li>
  <li><strong>Zoom &amp; pan</strong> - dense scatter plots of 100k points become
      explorable instead of ink blots.</li>
  <li><strong>Legend toggling</strong> - click a legend entry to hide/show a series;
      compare any subset without new code.</li>
  <li><strong>Standalone HTML export</strong> - send one file; the recipient needs only
      a browser. This is the underrated superpower for sharing analysis.</li>
</ul>

<h2>Plotly Express: the 90% API</h2>
<p><code>plotly.express</code> (px) mirrors Seaborn's grammar - DataFrame in,
column names mapped to roles - so it costs almost nothing to learn:</p>
<pre><code class="language-python">import plotly.express as px

df = px.data.gapminder().query("year == 2007")   # demo: countries in 2007

fig = px.scatter(
    df, x="gdpPercap", y="lifeExp",
    size="pop", color="continent",
    hover_name="country",            # tooltip shows the country name
    log_x=True, size_max=55,
    title="Wealth vs health, 2007 (hover any bubble)",
)
fig.show()                            # interactive in notebook/browser
fig.write_html("wealth_health.html") # shareable standalone file
</code></pre>
<p>Five arguments encode five data dimensions (x, y, size, color, tooltip) -
and every one of them is explorable by the reader, not fixed by the author.</p>

<h2>The greatest hits</h2>
<pre><code class="language-python"># line chart with range slider - time series exploration
stocks = px.data.stocks()
fig = px.line(stocks, x="date", y=["GOOG", "AAPL", "AMZN"])
fig.update_xaxes(rangeslider_visible=True)
fig.show()

# animated scatter: watch 50 years of development unfold
gap = px.data.gapminder()
px.scatter(gap, x="gdpPercap", y="lifeExp", size="pop", color="continent",
           animation_frame="year", log_x=True, size_max=55,
           range_y=[25, 90]).show()

# density heatmap for BIG scatter data (100k+ points)
px.density_heatmap(gap, x="gdpPercap", y="lifeExp", nbinsx=40, nbinsy=40).show()

# quick 3-D scatter (use sparingly - hard to read in print)
px.scatter_3d(df, x="gdpPercap", y="lifeExp", z="pop",
              color="continent", log_x=True).show()
</code></pre>

<h2>When to use which library</h2>
<table>
  <tr><th>Situation</th><th>Reach for</th><th>Why</th></tr>
  <tr><td>Quick EDA on your own</td><td>Seaborn / pandas .plot()</td><td>fastest to type, statistical defaults</td></tr>
  <tr><td>Publication figure (PDF/print)</td><td>Matplotlib (+ Seaborn styling)</td><td>total control, vector export, journal conventions</td></tr>
  <tr><td>Dense data you must explore</td><td>Plotly</td><td>zoom/hover beat any static rendering</td></tr>
  <tr><td>Share findings with non-coders</td><td>Plotly HTML export</td><td>recipients interact in a browser, no setup</td></tr>
  <tr><td>Dashboards / web apps</td><td>Plotly (+ Dash/Streamlit)</td><td>designed for it</td></tr>
</table>
<p>They compose, too: many people explore in Plotly, then rebuild the two figures
worth publishing in matplotlib. Knowing all three, you pick per task -
the grammar (map columns to x/y/color/facet) transfers between them.</p>

<h2>Quick quiz</h2>
<div class="quiz">
  <div class="quiz-q" data-answer="1">
    <p class="quiz-question">1. A stakeholder without Python needs to explore your results themselves. Best delivery?</p>
    <button class="quiz-opt">A PNG of a matplotlib figure</button>
    <button class="quiz-opt">A Plotly chart exported with write_html()</button>
    <button class="quiz-opt">The raw CSV</button>
    <div class="quiz-explain hidden">The HTML file is self-contained and opens in any browser with hover/zoom/legend-filtering intact - exploration without installation.</div>
  </div>
  <div class="quiz-q" data-answer="2">
    <p class="quiz-question">2. Your scatter plot of 300,000 points is an unreadable blob, even in Plotly. Best move?</p>
    <button class="quiz-opt">Increase the marker size</button>
    <button class="quiz-opt">Switch to a pie chart</button>
    <button class="quiz-opt">Use a density heatmap / 2-D histogram (or sample + lower alpha)</button>
    <div class="quiz-explain hidden">Past ~10⁴–10⁵ points, individual markers overplot completely. Binning into a density view (px.density_heatmap) shows where the mass actually is.</div>
  </div>
</div>
`};
