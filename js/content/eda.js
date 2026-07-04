/* Module 4 - Exploratory Data Analysis (part 1) */

CONTENT["eda/workflow"] = {
  html: String.raw`
<h1>EDA Workflow / Checklist</h1>
<p class="lead">Exploratory Data Analysis is the structured interrogation of a dataset
before any modeling. Done well, it prevents the two classic disasters: modeling
garbage, and "discovering" things that are actually data-collection artifacts.</p>

<h2>What EDA is really for</h2>
<p>Coined by statistician John Tukey in the 1970s, EDA answers four questions:</p>
<ol>
  <li><strong>Can I trust this data?</strong> (completeness, correctness, duplicates, leakage)</li>
  <li><strong>What does each variable look like?</strong> (types, distributions, ranges)</li>
  <li><strong>How do variables relate?</strong> (correlations, group differences, interactions)</li>
  <li><strong>What does this imply for modeling?</strong> (transformations, features to build
      or drop, an honest sense of achievable performance)</li>
</ol>
<p>The mindset matters: you are a detective, not a lawyer. A lawyer looks for
evidence supporting a story; a detective follows evidence wherever it goes -
including toward "this column is broken" or "the target leaks into feature X."</p>

<h2>The seven-step checklist</h2>
<p>This ordering - data quality first, target second, relationships last - exists
because each step depends on the previous ones being trustworthy:</p>
<div class="diagram">
<svg viewBox="0 0 660 210" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Seven EDA steps in a pipeline: first look, quality, target, univariate, bivariate, multivariate, summarize">
  <defs>
    <marker id="earr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M0,0 L10,5 L0,10 z" style="fill: var(--text-muted)"/>
    </marker>
  </defs>
  <rect x="15" y="30" width="140" height="52" rx="8" class="d-box-soft"/>
  <text x="85" y="52" text-anchor="middle" class="d-text">1 · First look</text>
  <text x="85" y="70" text-anchor="middle" class="d-text-sm">shape, head, dtypes</text>
  <line x1="155" y1="56" x2="180" y2="56" class="d-line" marker-end="url(#earr)"/>
  <rect x="183" y="30" width="140" height="52" rx="8" class="d-box-soft"/>
  <text x="253" y="52" text-anchor="middle" class="d-text">2 · Quality audit</text>
  <text x="253" y="70" text-anchor="middle" class="d-text-sm">missing, dupes, errors</text>
  <line x1="323" y1="56" x2="348" y2="56" class="d-line" marker-end="url(#earr)"/>
  <rect x="351" y="30" width="140" height="52" rx="8" class="d-box-soft"/>
  <text x="421" y="52" text-anchor="middle" class="d-text">3 · Know the target</text>
  <text x="421" y="70" text-anchor="middle" class="d-text-sm">balance / skew</text>
  <line x1="491" y1="56" x2="516" y2="56" class="d-line" marker-end="url(#earr)"/>
  <rect x="519" y="30" width="130" height="52" rx="8" class="d-box-soft"/>
  <text x="584" y="52" text-anchor="middle" class="d-text">4 · Univariate</text>
  <text x="584" y="70" text-anchor="middle" class="d-text-sm">each column alone</text>
  <line x1="584" y1="82" x2="584" y2="110" class="d-line" marker-end="url(#earr)"/>
  <rect x="519" y="113" width="130" height="52" rx="8" class="d-box-soft"/>
  <text x="584" y="135" text-anchor="middle" class="d-text">5 · Bivariate</text>
  <text x="584" y="153" text-anchor="middle" class="d-text-sm">feature vs target</text>
  <line x1="519" y1="139" x2="494" y2="139" class="d-line" marker-end="url(#earr)"/>
  <rect x="351" y="113" width="140" height="52" rx="8" class="d-box-soft"/>
  <text x="421" y="135" text-anchor="middle" class="d-text">6 · Multivariate</text>
  <text x="421" y="153" text-anchor="middle" class="d-text-sm">correlations, pairs</text>
  <line x1="351" y1="139" x2="326" y2="139" class="d-line" marker-end="url(#earr)"/>
  <rect x="183" y="113" width="140" height="52" rx="8" class="d-box"/>
  <text x="253" y="135" text-anchor="middle" class="d-text">7 · Summarize</text>
  <text x="253" y="153" text-anchor="middle" class="d-text-sm">findings + decisions</text>
</svg>
<div class="caption">The EDA pipeline. Quality before analysis; one variable before pairs;
always end with written findings.</div>
</div>

<h3>Step 1 - First look (5 minutes)</h3>
<pre><code class="language-python">import pandas as pd
df = pd.read_csv("data.csv")

df.shape                    # how much data?
df.head(10)                 # what do actual rows look like?
df.dtypes                   # numbers stored as text? dates as strings?
df.sample(10)               # random rows - heads can be unrepresentative
</code></pre>
<h3>Step 2 - Quality audit</h3>
<pre><code class="language-python">df.isna().mean().sort_values(ascending=False)   # % missing per column
df.duplicated().sum()                            # exact duplicates
df.describe()                                    # impossible mins/maxs?
df.nunique()                                     # constant columns? IDs?
</code></pre>
<p>Also hunt for <strong>leakage</strong>: any column that wouldn't exist at prediction
time (e.g. <code>days_until_churn</code> when predicting churn) will make models look
brilliant and be useless. This is the most expensive thing EDA can catch.</p>
<h3>Step 3 - Know your target</h3>
<pre><code class="language-python">df["target"].value_counts(normalize=True)   # classification: balance?
df["price"].hist(bins=50)                    # regression: skew? outliers?
</code></pre>
<p>A 97/3 class split changes every downstream decision (metrics, resampling -
see <a href="#/features/imbalanced-data">Handling Imbalanced Data</a>). A skewed
regression target usually wants a log transform.</p>
<h3>Steps 4–6 - Univariate, bivariate, multivariate</h3>
<p>Each gets its own chapter:
<a href="#/eda/uni-bi-multivariate">Univariate / Bivariate / Multivariate</a> and
<a href="#/eda/correlation-analysis">Correlation Analysis</a>, with
<a href="#/eda/missing-values">missing values</a> and
<a href="#/eda/outliers">outliers</a> handled along the way.</p>
<h3>Step 7 - Summarize in writing</h3>
<p>End with a short findings file: data quality issues found and how you'll handle
each, the 5–10 most important facts (with the plot that shows each), and the
modeling decisions they imply. If you can't write this summary, the EDA isn't done -
you looked at data, but you didn't learn from it.</p>

<h2>Habits of people who are good at this</h2>
<ul>
  <li><strong>Hypothesize before plotting.</strong> Guess what the histogram will look
      like, then look. Surprises are where discoveries live.</li>
  <li><strong>Chase anomalies.</strong> Every weird spike has a cause: a default value,
      a unit change, a scraping bug, or a real phenomenon. All four are worth knowing.</li>
  <li><strong>Keep a notebook of dead ends too.</strong> "Checked X, nothing there" saves
      your future self a repeat investigation.</li>
  <li><strong>Don't peek at the test set.</strong> EDA on all the data subtly tunes your
      choices to it; keep the final test split untouched.</li>
</ul>

<h2>Quick quiz</h2>
<div class="quiz">
  <div class="quiz-q" data-answer="1">
    <p class="quiz-question">1. Why does the quality audit come before any analysis of relationships?</p>
    <button class="quiz-opt">It's the fastest step</button>
    <button class="quiz-opt">Patterns computed on broken data (duplicates, sentinel values, leaks) are confidently wrong</button>
    <button class="quiz-opt">Tradition</button>
    <div class="quiz-explain hidden">A correlation computed over duplicated rows or -999 sentinels is a statement about the bugs, not the world. Trust first, then explore.</div>
  </div>
  <div class="quiz-q" data-answer="0">
    <p class="quiz-question">2. A feature "account_closed_date" appears in a churn-prediction dataset. Reaction?</p>
    <button class="quiz-opt">Leakage alarm - it's only known after churn happens; drop it</button>
    <button class="quiz-opt">Great feature - keep it</button>
    <button class="quiz-opt">Impute its missing values</button>
    <div class="quiz-explain hidden">It encodes the answer. The model would score ~100% in validation and 0 value in production, where the date doesn't exist yet for current customers.</div>
  </div>
</div>
`};

CONTENT["eda/missing-values"] = {
  html: String.raw`
<h1>Handling Missing Values</h1>
<p class="lead">The question is never just "how do I fill these holes?" -
it's "<em>why</em> are they holes?" The mechanism behind the missingness decides
which fixes are safe and which quietly bias everything downstream.</p>

<h2>First, see the damage</h2>
<pre><code class="language-python">import pandas as pd
import numpy as np

df = pd.read_csv("survey.csv")

pct = (df.isna().mean() * 100).sort_values(ascending=False)
print(pct[pct &gt; 0].round(1))

# do columns go missing TOGETHER? (structure = information)
import seaborn as sns
sns.heatmap(df.isna(), cbar=False)       # rows x columns, dark = missing

# does missingness relate to other variables? (the crucial check)
df.groupby(df["income"].isna())["age"].mean()
</code></pre>
<p>That last line is the underrated one: if rows missing <code>income</code> have a
noticeably different average <code>age</code>, the data isn't missing at random -
and naive fixes will distort your dataset.</p>

<h2>The three mechanisms (this is the concept that matters)</h2>
<table>
  <tr><th>Mechanism</th><th>Meaning</th><th>Example</th><th>Consequence</th></tr>
  <tr><td><strong>MCAR</strong><br>completely at random</td><td>a coin flip decided</td><td>a sensor randomly dropped 2% of readings</td><td>safe to drop rows or impute simply - only statistical power is lost</td></tr>
  <tr><td><strong>MAR</strong><br>at random <em>given other columns</em></td><td>missingness explained by observed data</td><td>younger users skip the income question</td><td>impute using those other columns (model-based); dropping biases the data</td></tr>
  <tr><td><strong>MNAR</strong><br>not at random</td><td>missingness depends on the hidden value itself</td><td>high earners refuse to state income</td><td>no imputation fixes it; the missingness itself is signal - model it</td></tr>
</table>
<p>You can't fully prove which mechanism holds (MNAR by definition hides), but the
group-comparison check above plus domain sense gets you a working diagnosis.</p>

<h2>Option 1: dropping</h2>
<pre><code class="language-python">df.dropna()                          # rows with ANY missing - usually too brutal
df.dropna(subset=["target"])         # rows missing the label: almost always drop
df.drop(columns=["notes"])           # a column that's 90% missing: usually drop
</code></pre>
<p>Dropping rows is fine when losses are small (&lt; ~5%) and plausibly MCAR.
Beware the silent multiplier: with 20 columns each 5% missing independently,
<code>dropna()</code> can delete a third of your dataset.</p>

<h2>Option 2: simple imputation</h2>
<pre><code class="language-python">from sklearn.impute import SimpleImputer

num_imp = SimpleImputer(strategy="median")   # median: robust to skew/outliers
cat_imp = SimpleImputer(strategy="most_frequent")

# add a "was missing" flag BEFORE filling - often predictive by itself!
df["income_missing"] = df["income"].isna().astype(int)
df["income"] = num_imp.fit_transform(df[["income"]])
</code></pre>
<p>Median/mode imputation is fast and fine for mild missingness, but understand the
cost: every filled cell lands exactly on the center, which <strong>shrinks the
column's variance and weakens its correlations</strong>. The indicator column is the
cheap insurance - if missingness carries signal (MNAR!), the model can still use it.</p>

<h2>Option 3: model-based imputation</h2>
<pre><code class="language-python">from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer

# KNN: fill from the k most similar rows
knn_filled = KNNImputer(n_neighbors=5).fit_transform(X)

# Iterative (MICE-style): predict each column from all the others, repeat
mice_filled = IterativeImputer(max_iter=10, random_state=0).fit_transform(X)
</code></pre>
<p>These respect relationships between columns (an imputed income now depends on
age, education…), making them the right tool under MAR. Cost: slower, and you
must be careful to fit them on training data only.</p>
<div class="callout warn">
  <span class="co-title">The leakage rule for imputation</span>
  Fit the imputer on the <strong>training split only</strong>, then apply it to
  validation/test. Computing a fill value from all rows lets test-set information
  seep into training - a small leak, but the habit protects you everywhere
  (same rule as <a href="#/features/scaling">scaling</a>). In scikit-learn, put the
  imputer inside a <code>Pipeline</code> and this happens automatically.
</div>

<h2>Special cases worth knowing</h2>
<ul>
  <li><strong>Time series:</strong> forward-fill (<code>df.ffill()</code>) - carry the last
      known value - respects time's arrow; interpolation
      (<code>df.interpolate()</code>) suits smooth signals.</li>
  <li><strong>Trees &amp; boosting:</strong> XGBoost/LightGBM handle NaN natively (they learn
      which branch missing values should take) - sometimes the best "imputation"
      is none.</li>
  <li><strong>Categoricals:</strong> a literal <code>"missing"</code> category often beats
      mode-filling - it preserves the information that the value was absent.</li>
</ul>

<h2>Decision summary</h2>
<table>
  <tr><th>Situation</th><th>Reasonable default</th></tr>
  <tr><td>&lt; 5% missing, looks MCAR</td><td>drop rows, or median/mode fill</td></tr>
  <tr><td>5–30% missing, MAR-ish</td><td>model-based imputation + missing-indicator flags</td></tr>
  <tr><td>&gt; ~60% missing in a column</td><td>drop the column (keep the indicator if predictive)</td></tr>
  <tr><td>Suspected MNAR</td><td>indicator flag is essential; consider modeling missingness explicitly</td></tr>
  <tr><td>Missing target values</td><td>drop those rows - never impute the label</td></tr>
</table>

<h2>Common interview questions</h2>
<div class="qa">
  <div class="qa-q">Q1. Explain MCAR vs MAR vs MNAR with one example each.</div>
  <div class="qa-a"><p>MCAR: randomness unrelated to anything (lab sample dropped on the floor).
  MAR: explained by observed columns (younger respondents skip the income question).
  MNAR: depends on the missing value itself (high earners hide income). The mechanism
  dictates the safe remedy: drop / model-impute / flag-and-model, respectively.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q2. Why does mean imputation weaken correlations?</div>
  <div class="qa-a"><p>Imputed points sit exactly on the mean line with zero deviation, adding mass at the
  center that co-varies with nothing. Variance shrinks, and covariance with every other
  variable is diluted toward zero - which then biases any model that relies on it.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q3. When is a missing value itself a feature?</div>
  <div class="qa-a"><p>Whenever the act of not-answering carries meaning - skipped salary fields,
  untaken medical tests, optional profile fields. An <code>is_missing</code> indicator lets
  the model exploit that signal even after you impute the column, and it's the only
  defense you have under MNAR.</p></div>
</div>

<h2>Quick quiz</h2>
<div class="quiz">
  <div class="quiz-q" data-answer="2">
    <p class="quiz-question">1. Patients with severe symptoms are too ill to complete a survey question about symptom severity. This is…</p>
    <button class="quiz-opt">MCAR</button>
    <button class="quiz-opt">MAR</button>
    <button class="quiz-opt">MNAR</button>
    <div class="quiz-explain hidden">Missingness depends on the value that's missing (severity itself). Imputing from observed data will systematically underestimate severity - the missing rows are exactly the severe ones.</div>
  </div>
  <div class="quiz-q" data-answer="1">
    <p class="quiz-question">2. Before filling a column, it's often wise to…</p>
    <button class="quiz-opt">Sort the dataframe</button>
    <button class="quiz-opt">Add a binary "was_missing" indicator column</button>
    <button class="quiz-opt">Convert it to strings</button>
    <div class="quiz-explain hidden">The flag preserves the information imputation destroys, costs one column, and frequently turns out to be predictive on its own.</div>
  </div>
  <div class="quiz-q" data-answer="0">
    <p class="quiz-question">3. Where should an imputer learn its fill values?</p>
    <button class="quiz-opt">From the training split only</button>
    <button class="quiz-opt">From the full dataset for accuracy</button>
    <button class="quiz-opt">From the test set</button>
    <div class="quiz-explain hidden">Statistics computed over test rows leak information into training. Pipelines enforce the train-only rule automatically.</div>
  </div>
</div>
`};

CONTENT["eda/outliers"] = {
  html: String.raw`
<h1>Outlier Detection</h1>
<p class="lead">An outlier is a point that doesn't fit the pattern of the rest.
The detection math is easy; the judgment is not - because an outlier is either
an error to remove, or the most informative point in your dataset.</p>

<h2>First: what kind of outlier is it?</h2>
<ul>
  <li><strong>Impossible values</strong> - age 250, negative weight. Data errors;
      fix or nullify (already covered in <a href="#/python/data-cleaning">Data Cleaning</a>).</li>
  <li><strong>Legitimate extremes</strong> - a real $2M house among $300k ones. True data;
      deleting it is falsifying your dataset.</li>
  <li><strong>Interesting anomalies</strong> - the fraudulent transaction, the failing
      machine. Sometimes the outliers <em>are the entire point</em> (anomaly detection).</li>
</ul>
<p>Detection methods flag candidates; this classification decides what to do with them.</p>

<h2>Method 1: the IQR rule (robust default)</h2>
<p>The box-plot rule you met in <a href="#/math/descriptive-statistics">Descriptive
Statistics</a>: anything beyond 1.5 × IQR from the quartiles is flagged.</p>
<div class="math-box">
$$\text{flag if } x \lt Q_1 - 1.5\,\mathrm{IQR} \;\text{ or }\; x \gt Q_3 + 1.5\,\mathrm{IQR},
\qquad \mathrm{IQR} = Q_3 - Q_1$$
</div>
<div class="diagram">
<svg viewBox="0 0 640 240" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Anatomy of a box plot: box from Q1 to Q3 with median line, whiskers to the fences, and outlier points beyond">
  <line x1="40" y1="120" x2="600" y2="120" class="d-grid"/>
  <!-- whiskers -->
  <line x1="120" y1="120" x2="230" y2="120" class="d-line"/>
  <line x1="120" y1="103" x2="120" y2="137" class="d-line"/>
  <line x1="430" y1="120" x2="520" y2="120" class="d-line"/>
  <line x1="520" y1="103" x2="520" y2="137" class="d-line"/>
  <!-- box -->
  <rect x="230" y="88" width="200" height="64" rx="4" class="d-box-soft"/>
  <line x1="300" y1="88" x2="300" y2="152" class="d-line-accent"/>
  <!-- outliers -->
  <circle cx="70" cy="120" r="5" class="d-dot-muted"/>
  <circle cx="565" cy="120" r="5" class="d-dot-muted"/>
  <circle cx="595" cy="120" r="5" class="d-dot-muted"/>
  <!-- labels -->
  <text x="230" y="75" text-anchor="middle" class="d-text-sm">Q1</text>
  <text x="300" y="75" text-anchor="middle" class="d-text-accent">median</text>
  <text x="430" y="75" text-anchor="middle" class="d-text-sm">Q3</text>
  <text x="330" y="178" text-anchor="middle" class="d-text-sm">← IQR →</text>
  <text x="120" y="178" text-anchor="middle" class="d-text-sm">Q1 − 1.5·IQR</text>
  <text x="520" y="178" text-anchor="middle" class="d-text-sm">Q3 + 1.5·IQR</text>
  <text x="580" y="100" text-anchor="middle" class="d-text-sm">outliers</text>
  <text x="320" y="215" text-anchor="middle" class="d-text-sm">whiskers extend to the most extreme points inside the fences</text>
</svg>
<div class="caption">Box-plot anatomy. The 1.5×IQR fences are conventions, not laws -
they flag ~0.7% of perfectly normal data.</div>
</div>
<p>Being quartile-based, this rule is itself immune to the outliers it hunts -
that robustness is why it's the default.</p>

<h2>Method 2: z-scores (fine for symmetric data)</h2>
<div class="math-box">
$$z_i = \frac{x_i - \bar{x}}{s}, \qquad \text{flag if } |z_i| \gt 3$$
</div>
<p>Simple, but two catches: it assumes roughly normal data (on skewed data it
flags the whole tail), and it's circular - a huge outlier inflates \(s\), which can
hide that same outlier ("masking"). The <em>modified z-score</em> using the median
absolute deviation (MAD) fixes the circularity:
\(z^{mod}_i = 0.6745\,(x_i - \tilde{x})/\mathrm{MAD}\), flag beyond 3.5.</p>

<h2>Method 3: multivariate outliers</h2>
<p>The sneakiest kind: each value is normal alone, the <em>combination</em> is absurd -
a 1.9 m person weighing 45 kg passes both single-column checks. You need methods
that look at the joint distribution:</p>
<pre><code class="language-python">import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

rng = np.random.default_rng(0)
height = rng.normal(170, 8, 500)
weight = 0.9 * height - 90 + rng.normal(0, 6, 500)
df = pd.DataFrame({"height": height, "weight": weight})
df.loc[500] = [190, 45]                     # normal alone, absurd together

# Isolation Forest: outliers are points that are EASY to isolate
# with random splits - few cuts needed to fence them off alone
iso = IsolationForest(contamination=0.01, random_state=0)
df["flag"] = iso.fit_predict(df[["height", "weight"]])   # -1 = outlier
print(df[df["flag"] == -1])                 # catches the impossible combo
</code></pre>
<p>Alternatives in the same family: Local Outlier Factor (density-based),
DBSCAN's noise points (<a href="#/ml/dbscan">Module 6</a>), and Mahalanobis
distance (z-score generalized to correlated dimensions).</p>

<h2>What to do with a confirmed outlier</h2>
<table>
  <tr><th>Action</th><th>When</th><th>How</th></tr>
  <tr><td><strong>Fix</strong></td><td>traceable data error</td><td>correct at the source (180 cm entered as 18.0)</td></tr>
  <tr><td><strong>Remove</strong></td><td>error you can't fix; irrelevant to the task</td><td>drop, and report how many/why</td></tr>
  <tr><td><strong>Cap (winsorize)</strong></td><td>legitimate extremes destabilizing a model</td><td><code>df[col].clip(lower=p1, upper=p99)</code></td></tr>
  <tr><td><strong>Transform</strong></td><td>heavy right tail is the norm (prices!)</td><td><code>np.log1p(x)</code> - tames the tail, keeps the ranking</td></tr>
  <tr><td><strong>Use robust models</strong></td><td>you want to keep everything</td><td>trees/median-based losses barely notice outliers; linear + MSE notices intensely</td></tr>
  <tr><td><strong>Study them</strong></td><td>anomalies are the target</td><td>fraud, intrusion, defect detection - flip the problem</td></tr>
</table>
<div class="callout warn">
  <span class="co-title">The cardinal sin</span>
  Deleting outliers <em>because they hurt the model's score</em>, with no evidence
  they're errors, is data manipulation. Every removal needs a documented reason a
  colleague could audit. When in doubt: run the analysis with and without them,
  and report both.
</div>

<h2>Common interview questions</h2>
<div class="qa">
  <div class="qa-q">Q1. Why prefer IQR over z-score for outlier detection?</div>
  <div class="qa-a"><p>Quartiles are unaffected by the extreme values being tested, while the z-score's
  mean and standard deviation are inflated by them (masking). IQR also makes no normality
  assumption, so it behaves sensibly on skewed data.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q2. What's a multivariate outlier? Give an example.</div>
  <div class="qa-a"><p>A point whose individual coordinates are unremarkable but whose combination is -
  a car with 400 hp (fine) and 2.5 L/100km fuel use (fine) simultaneously (impossible).
  Caught by joint-distribution methods: Isolation Forest, LOF, Mahalanobis distance.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q3. How do outliers affect linear regression vs a decision tree?</div>
  <div class="qa-a"><p>Linear regression with squared loss is highly sensitive - one extreme point can
  rotate the whole line (its error is squared). Trees split on rank order, so an outlier
  just falls in an end bucket; its magnitude barely matters. This robustness difference
  should inform both cleaning effort and model choice.</p></div>
</div>

<h2>Quick quiz</h2>
<div class="quiz">
  <div class="quiz-q" data-answer="1">
    <p class="quiz-question">1. Q1 = 40, Q3 = 60. Under the standard rule, which value is flagged?</p>
    <button class="quiz-opt">85</button>
    <button class="quiz-opt">95</button>
    <button class="quiz-opt">65</button>
    <div class="quiz-explain hidden">IQR = 20, upper fence = 60 + 1.5×20 = 90. Only 95 exceeds it.</div>
  </div>
  <div class="quiz-q" data-answer="2">
    <p class="quiz-question">2. House prices have a long right tail of genuine luxury homes. Best treatment before linear regression?</p>
    <button class="quiz-opt">Delete all houses above the fence</button>
    <button class="quiz-opt">Nothing - data is data</button>
    <button class="quiz-opt">Log-transform the price</button>
    <div class="quiz-explain hidden">The extremes are real, so deletion falsifies the data - but untransformed, they dominate the squared loss. log1p compresses the tail while preserving order, usually improving fit for everyone.</div>
  </div>
  <div class="quiz-q" data-answer="0">
    <p class="quiz-question">3. In fraud detection, outliers are…</p>
    <button class="quiz-opt">The signal - the whole point of the analysis</button>
    <button class="quiz-opt">Noise to remove before modeling</button>
    <button class="quiz-opt">Impossible values</button>
    <div class="quiz-explain hidden">Same math, inverted goal: anomaly detection treats the flagged points as the product, not the garbage. Whether an outlier is trash or treasure depends entirely on the question.</div>
  </div>
</div>
`};
