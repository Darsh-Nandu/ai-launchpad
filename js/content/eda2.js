/* Module 4 - Exploratory Data Analysis (part 2) */

CONTENT["eda/uni-bi-multivariate"] = {
  html: String.raw`
<h1>Univariate / Bivariate / Multivariate Analysis</h1>
<p class="lead">Explore in widening circles: one variable at a time, then pairs,
then the whole web at once. Each level has its own standard toolkit - and each
catches problems the other levels can't see.</p>

<h2>Level 1 - Univariate: one column at a time</h2>
<p>For every column, you want five facts: type, center, spread, shape, and weirdness
(missing/outliers/impossible values).</p>
<pre><code class="language-python">import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df = sns.load_dataset("titanic")

# numeric columns: histogram + box, side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
sns.histplot(df, x="age", kde=True, ax=ax1)
sns.boxplot(df, x="age", ax=ax2)

# and the numbers to go with the picture:
print(df["age"].describe())
print("skew:", df["age"].skew().round(2))

# categorical columns: value_counts - always with normalize
print(df["embark_town"].value_counts(normalize=True).round(3))
sns.countplot(df, y="embark_town", order=df["embark_town"].value_counts().index)
</code></pre>
<p>What you're hunting at this level: skew that wants a transform, spikes at
suspicious values (0? -999? exactly 18?), categories with tiny counts (merge them?),
and columns with one dominant value (near-zero variance - barely informative).</p>

<h2>Level 2 - Bivariate: pairs, especially feature-vs-target</h2>
<p>The chart depends on the <em>types</em> of the two variables:</p>
<table>
  <tr><th>Pair</th><th>Plot</th><th>Statistic</th></tr>
  <tr><td>numeric × numeric</td><td>scatter (+ trend line)</td><td>Pearson / Spearman r</td></tr>
  <tr><td>numeric × categorical</td><td>box / violin per category</td><td>group means, t-test/ANOVA</td></tr>
  <tr><td>categorical × categorical</td><td>heatmap of a crosstab</td><td>proportions, chi-squared</td></tr>
</table>
<pre><code class="language-python"># numeric x numeric
sns.scatterplot(df, x="age", y="fare", alpha=0.5)

# numeric x categorical - THE plot for classification EDA:
# how does the feature distribution differ between classes?
sns.violinplot(df, x="survived", y="age")

# categorical x categorical: crosstab + heatmap
ct = pd.crosstab(df["pclass"], df["survived"], normalize="index")
sns.heatmap(ct, annot=True, fmt=".2f", cmap="Blues")
# 1st class: 63% survived · 3rd class: 24% - a strong feature!
</code></pre>
<p>For supervised problems, run <em>every feature against the target</em> first.
Features whose distribution barely changes across target values will contribute
little; features with dramatic separation (like <code>pclass</code> above) are your
future model's backbone. This is also where you sanity-check direction: does the
relationship make real-world sense, or does it smell like leakage?</p>

<h2>Level 3 - Multivariate: the web of interactions</h2>
<p>Two variables can lie to you until a third enters the picture (remember
<a href="#/math/correlation-causation">Simpson's paradox</a>). The standard moves:</p>
<pre><code class="language-python"># 1. pair plot colored by target - the full overview in one command
cols = ["survived", "age", "fare", "pclass"]
sns.pairplot(df[cols].dropna(), hue="survived", corner=True)

# 2. add a third variable to any bivariate plot via hue/size/facets
sns.scatterplot(df, x="age", y="fare", hue="pclass", size="survived")

# 3. facet grids: does the age-survival pattern differ by sex?
g = sns.FacetGrid(df, col="sex", height=3.5)
g.map_dataframe(sns.histplot, x="age", hue="survived", multiple="stack")

# 4. grouped aggregates: interactions in numbers
print(df.pivot_table(values="survived", index="pclass",
                     columns="sex", aggfunc="mean").round(2))
#        female  male
# 1        0.97  0.37     <- sex effect is huge WITHIN every class
# 2        0.92  0.16
# 3        0.50  0.14
</code></pre>
<p>That pivot table is multivariate analysis at its best: three variables, one
glance, and the story ("women and children first, but class bought survival too")
is unmistakable - including the <em>interaction</em>: the sex gap is largest in
2nd class. Interactions like this are exactly what tree models exploit and what
linear models miss unless you hand-craft them
(<a href="#/features/encoding">Feature Engineering</a>).</p>

<h2>The correlation matrix - gateway to the next chapter</h2>
<pre><code class="language-python">corr = df.select_dtypes("number").corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", vmin=-1, vmax=1)
</code></pre>
<p>Reading it properly - including its traps (non-linearity, confounders,
multicollinearity) - is the subject of
<a href="#/eda/correlation-analysis">Correlation Analysis</a>.</p>

<h2>Quick quiz</h2>
<div class="quiz">
  <div class="quiz-q" data-answer="1">
    <p class="quiz-question">1. Feature: monthly_charge (numeric). Target: churned (yes/no). The right bivariate plot?</p>
    <button class="quiz-opt">Scatter plot</button>
    <button class="quiz-opt">Box or violin plot of charge, split by churn status</button>
    <button class="quiz-opt">Pie chart of churn</button>
    <div class="quiz-explain hidden">Numeric × categorical → compare the numeric distribution across categories. If churners' charges sit visibly higher, you've found a real feature.</div>
  </div>
  <div class="quiz-q" data-answer="2">
    <p class="quiz-question">2. Why isn't bivariate analysis enough?</p>
    <button class="quiz-opt">It's too slow</button>
    <button class="quiz-opt">It only works on numeric data</button>
    <button class="quiz-opt">Third variables can reverse or explain away pairwise patterns (confounding, Simpson's paradox)</button>
    <div class="quiz-explain hidden">A pairwise trend can flip sign within every subgroup. Faceting, hue, and pivot tables are the EDA tools that expose this.</div>
  </div>
  <div class="quiz-q" data-answer="0">
    <p class="quiz-question">3. In the Titanic pivot table, "the sex gap differs by class" is an example of…</p>
    <button class="quiz-opt">An interaction effect</button>
    <button class="quiz-opt">Multicollinearity</button>
    <button class="quiz-opt">Data leakage</button>
    <div class="quiz-explain hidden">The effect of one variable depends on the level of another - an interaction. Linear models need explicit interaction terms to capture it; trees find it naturally.</div>
  </div>
</div>
`};

CONTENT["eda/correlation-analysis"] = {
  html: String.raw`
<h1>Correlation Analysis</h1>
<p class="lead">The correlation heatmap is EDA's most popular chart - and its most
misread. This chapter is about extracting real decisions from it: which features
carry signal, which are redundant, and which correlations are traps.</p>

<h2>Making the matrix (properly)</h2>
<pre><code class="language-python">import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

df = sns.load_dataset("penguins").dropna()
num = df.select_dtypes("number")

corr = num.corr()                      # Pearson by default
corr_s = num.corr(method="spearman")   # rank-based: robust, catches monotone curves

# upper-triangle mask: the matrix is symmetric, show each pair once
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
            cmap="vlag", vmin=-1, vmax=1, square=True)
plt.show()
</code></pre>
<p>Compute <em>both</em> Pearson and Spearman. If they disagree substantially for a
pair (Pearson 0.2, Spearman 0.7), the relationship is non-linear or
outlier-distorted - either way, the scatter plot will tell you which. Which brings
up the iron rule from <a href="#/dataviz/why-visualization">Anscombe</a>:
<strong>a correlation you haven't scatter-plotted is a rumor, not a finding.</strong></p>

<h2>Reading it for modeling decisions</h2>
<p>Three different things to scan for, in order:</p>
<ol>
  <li><strong>Feature ↔ target correlations</strong> - the candidates for strong
      predictors. (Remember they only capture <em>linear</em> signal: a U-shaped
      relationship scores ≈ 0 and would be wrongly discarded.)</li>
  <li><strong>Feature ↔ feature near-duplicates</strong> - |r| &gt; ~0.9 means the pair
      carries almost the same information. Redundancy, handled below.</li>
  <li><strong>Suspiciously perfect correlations with the target</strong> - r ≈ 0.99 is
      rarely brilliance; it's usually <em>leakage</em> (the feature encodes the answer).
      Investigate before celebrating.</li>
</ol>

<h2>Multicollinearity: when features duplicate each other</h2>
<p>Highly correlated features don't hurt <em>predictive accuracy</em> much, but they:</p>
<ul>
  <li>make linear-model coefficients unstable and uninterpretable (the credit for
      the shared signal gets split arbitrarily - coefficients can even flip sign
      between runs);</li>
  <li>dilute feature-importance scores across the duplicates;</li>
  <li>waste computation and complicate the model.</li>
</ul>
<p>Pairwise r misses group-wise redundancy (x₃ ≈ x₁ + x₂ can hide behind modest
pairwise values). The proper instrument is <strong>VIF</strong> - the Variance
Inflation Factor: regress each feature on all the others; VIF = 1/(1−R²).</p>
<pre><code class="language-python">from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd

X = num.drop(columns=["body_mass_g"])         # features only
vif = pd.Series(
    [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
    index=X.columns).sort_values(ascending=False)
print(vif.round(1))
# rule of thumb: VIF > 10 = serious redundancy; 5-10 = worth attention
</code></pre>
<p>Remedies: drop one of each redundant pair (keep the more interpretable/available
one), combine them (average, ratio, or <a href="#/features/dimensionality-reduction">PCA</a>),
or use <a href="#/ml/regularization">Ridge regularization</a>, which tolerates
collinearity gracefully.</p>

<h2>Correlation with the target isn't feature importance</h2>
<p>Two humbling facts to keep the heatmap in perspective:</p>
<ul>
  <li>A feature with <em>zero</em> target correlation can be vital <em>in combination</em>
      (XOR-style interactions - each input alone is uninformative, together they
      determine the answer).</li>
  <li>A feature with strong correlation can be useless <em>given another feature</em>
      that carries the same signal better.</li>
</ul>
<p>So use the heatmap to build hypotheses and catch redundancy/leakage - but let
actual models (<a href="#/features/feature-selection">Feature Selection</a>) make the
final keep/drop calls.</p>

<h2>Common interview questions</h2>
<div class="qa">
  <div class="qa-q">Q1. What is multicollinearity and why does it matter?</div>
  <div class="qa-a"><p>Strong linear dependence among features. Predictions survive, but coefficients
  become unstable and uninterpretable, standard errors inflate, and importance scores get
  split across duplicates. Detect with correlation pairs + VIF; fix by dropping, combining,
  or regularizing.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q2. Pearson says 0.1, Spearman says 0.8 for the same pair. What's going on?</div>
  <div class="qa-a"><p>A strongly monotonic but non-linear relationship (e.g. exponential), or outliers
  wrecking Pearson. Spearman works on ranks, so it sees the monotone pattern. Plot it; a
  transform (log) will often linearize the relationship for Pearson too.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q3. A feature correlates 0.98 with your target. Your reaction?</div>
  <div class="qa-a"><p>Suspicion, not excitement. Check whether the feature would exist at prediction
  time or derives from the target (leakage). Real-world signals are rarely that clean;
  leaks always are.</p></div>
</div>

<h2>Quick quiz</h2>
<div class="quiz">
  <div class="quiz-q" data-answer="1">
    <p class="quiz-question">1. Two features have r = 0.95 with each other. For a linear model you should…</p>
    <button class="quiz-opt">Keep both - more features is better</button>
    <button class="quiz-opt">Drop or combine one - they're near-duplicates that destabilize coefficients</button>
    <button class="quiz-opt">Drop both</button>
    <div class="quiz-explain hidden">They carry ~the same information. Keeping both splits credit arbitrarily between unstable coefficients; keeping one loses almost nothing.</div>
  </div>
  <div class="quiz-q" data-answer="0">
    <p class="quiz-question">2. VIF for a feature is 25. Interpretation?</p>
    <button class="quiz-opt">Other features jointly predict it almost perfectly - severe redundancy</button>
    <button class="quiz-opt">It's the most important feature</button>
    <button class="quiz-opt">It has 25 missing values</button>
    <div class="quiz-explain hidden">VIF = 1/(1−R²) = 25 means R² ≈ 0.96 when regressing this feature on the rest. It adds little unique information, and VIF catches group-wise redundancy that pairwise r misses.</div>
  </div>
  <div class="quiz-q" data-answer="2">
    <p class="quiz-question">3. A feature shows r ≈ 0 with the target. Safe to discard?</p>
    <button class="quiz-opt">Yes - zero correlation means zero information</button>
    <button class="quiz-opt">Yes, if the dataset is large</button>
    <button class="quiz-opt">Not necessarily - non-linear or interaction effects are invisible to r</button>
    <div class="quiz-explain hidden">r only sees linear, solo relationships. U-shapes and XOR-like combinations score 0 yet can be decisive. Let model-based selection make the call.</div>
  </div>
</div>
`};

CONTENT["eda/case-study"] = {
  html: String.raw`
<h1>Case Study: Full EDA on the Titanic Dataset</h1>
<p class="lead">Everything from this module, executed start-to-finish on the most
famous teaching dataset: 891 Titanic passengers, one question - who survived,
and what does the data say about why? Run every block; the printed numbers
are the narrative.</p>

<h2>Steps 1–2: first look and quality audit</h2>
<pre><code class="language-python">import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = sns.load_dataset("titanic")
print(df.shape)                    # (891, 15)
print(df.dtypes)
df.head()

# quality audit
print(df.isna().mean().sort_values(ascending=False).head(5).round(3))
# deck        0.772   <- 77% missing: drop column, keep "had_cabin" flag
# age         0.199   <- 20% missing: impute + indicator
# embarked    0.002   <- 2 rows: fill with mode
print("duplicates:", df.duplicated().sum())        # 107 - but wait...
</code></pre>
<p>Those "duplicates" are a judgment call from
<a href="#/python/data-cleaning">Data Cleaning</a>: this dataset has no unique ID,
and two 3rd-class adult males with the same fare are plausibly different people.
We keep them - documented, not ignored. Decisions so far:</p>
<ul>
  <li><code>deck</code>: 77% missing → drop, but keep <code>had_cabin = deck.notna()</code>
      (having a recorded cabin may proxy wealth - an MNAR insight!).</li>
  <li><code>age</code>: 20% missing → median-impute + <code>age_missing</code> flag.</li>
</ul>
<pre><code class="language-python">df["had_cabin"] = df["deck"].notna().astype(int)
df["age_missing"] = df["age"].isna().astype(int)
df["age"] = df["age"].fillna(df["age"].median())
df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])
df = df.drop(columns=["deck"])
</code></pre>

<h2>Step 3: know the target</h2>
<pre><code class="language-python">print(df["survived"].value_counts(normalize=True).round(3))
# 0: 0.616 · 1: 0.384 - mildly imbalanced; accuracy baseline = 61.6%
</code></pre>
<p>Writing down the majority-class baseline <em>now</em> keeps future-you honest:
a model scoring 62% has learned nothing.</p>

<h2>Step 4: univariate highlights</h2>
<pre><code class="language-python">fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
sns.histplot(df, x="age", bins=40, ax=axes[0])          # spike of infants!
sns.histplot(df, x="fare", bins=40, ax=axes[1])         # extreme right skew
sns.countplot(df, x="pclass", ax=axes[2])               # 3rd class dominates

print("fare skew:", df["fare"].skew().round(1))          # ≈ 4.8!
df["log_fare"] = np.log1p(df["fare"])                    # tame the tail
</code></pre>
<p>Three univariate finds: a bump of babies (families aboard - units of survival?),
a fare distribution skewed enough (4.8) to demand a log transform, and a passenger
majority in 3rd class. One impossible-value check:
<code>(df["fare"] == 0).sum()</code> → 15 people with fare 0 - crew? guests?
Flagged for the report.</p>

<h2>Step 5: bivariate - every feature vs survival</h2>
<pre><code class="language-python">for col in ["sex", "pclass", "embarked", "had_cabin", "age_missing"]:
    print(df.groupby(col)["survived"].mean().round(2), "\n")
# sex:        female 0.74 · male 0.19        <- the dominant factor
# pclass:     1: 0.63 · 2: 0.47 · 3: 0.24    <- strong, monotonic
# had_cabin:  1: 0.67 · 0: 0.30              <- the MNAR flag pays off!
# age_missing:1: 0.29 · 0: 0.41              <- even missingness is signal

sns.violinplot(df, x="survived", y="age")    # children survived more
sns.violinplot(df, x="survived", y="log_fare")   # payers survived more
</code></pre>

<h2>Step 6: multivariate - the interactions</h2>
<pre><code class="language-python">print(df.pivot_table(values="survived", index="pclass",
                     columns="sex", aggfunc="mean").round(2))
#        female  male
# 1        0.97  0.37
# 2        0.92  0.16
# 3        0.50  0.14
# -> sex dominates everywhere, but 3rd-class women lost the protection
#    1st/2nd-class women had. Class x sex is a true interaction.

# family size: engineered from siblings + parents columns
df["family_size"] = df["sibsp"] + df["parch"] + 1
print(df.groupby("family_size")["survived"].mean().round(2))
# 1: 0.30 · 2: 0.55 · 3: 0.58 · 4: 0.72 · then collapses: 7+: ~0.0-0.33
# -> non-monotonic! small families beat both loners and large families

corr = df[["survived", "pclass", "age", "log_fare",
           "had_cabin", "family_size"]].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", vmin=-1, vmax=1)
# log_fare ↔ pclass: r = -0.75 - expected redundancy (fare buys class)
</code></pre>

<h2>Step 7: the findings summary</h2>
<div class="callout">
  <span class="co-title">EDA report - Titanic (what we'd hand to the modeling step)</span>
  <strong>Quality:</strong> deck dropped (77% missing, kept had_cabin flag);
  age imputed with indicator; 15 zero-fare rows flagged; no usable unique ID.<br>
  <strong>Target:</strong> 38.4% positive; baseline accuracy 61.6%.<br>
  <strong>Signal (strong → weak):</strong> sex ≫ pclass ≈ had_cabin &gt; log_fare &gt;
  family_size (non-monotonic) &gt; age (children boosted).<br>
  <strong>Interactions:</strong> sex × class is real and large; expect tree models to
  beat plain logistic regression unless interaction terms are added.<br>
  <strong>Transforms:</strong> log(fare) required; family_size worth binning
  (solo / 2–4 / 5+).<br>
  <strong>Redundancy:</strong> fare ↔ pclass (r = −0.75): keep both for trees,
  consider dropping one for linear models.<br>
  <strong>Leakage check:</strong> no post-outcome columns found. ✓
</div>
<p>Notice what EDA delivered before any model existed: honest baselines, engineered
features (<code>family_size</code>, <code>had_cabin</code>), transform decisions, a
model-family recommendation, and audit-ready cleaning decisions. That's the
standard to aim for on every dataset - and Module 5 turns several of these findings
into formal <a href="#/features/encoding">feature engineering</a> techniques.</p>

<h2>Quick quiz</h2>
<div class="quiz">
  <div class="quiz-q" data-answer="1">
    <p class="quiz-question">1. The had_cabin flag (built from a 77%-missing column) turned out predictive. The lesson?</p>
    <button class="quiz-opt">Always drop very-missing columns entirely</button>
    <button class="quiz-opt">Missingness itself can carry signal - encode it before discarding the column</button>
    <button class="quiz-opt">The column should have been median-imputed</button>
    <div class="quiz-explain hidden">Whether a cabin was recorded proxies wealth/class - classic informative missingness (MNAR). The flag preserved that signal even though the raw column was unusable.</div>
  </div>
  <div class="quiz-q" data-answer="2">
    <p class="quiz-question">2. Survival by family size rose from 1→4 then collapsed for 5+. Why does this matter for modeling?</p>
    <button class="quiz-opt">It proves family size is irrelevant</button>
    <button class="quiz-opt">It means the data is corrupted</button>
    <button class="quiz-opt">The relationship is non-monotonic - linear models need it binned; trees can handle it raw</button>
    <div class="quiz-explain hidden">A single linear coefficient can't encode "medium is best". EDA findings like this directly dictate feature engineering choices.</div>
  </div>
  <div class="quiz-q" data-answer="0">
    <p class="quiz-question">3. Why record the 61.6% majority-class baseline during EDA?</p>
    <button class="quiz-opt">Any future model must beat it to have learned anything at all</button>
    <button class="quiz-opt">It's needed to compute correlations</button>
    <button class="quiz-opt">scikit-learn requires it</button>
    <div class="quiz-explain hidden">"79% accuracy" sounds good until you know 61.6% was free. Baselines anchor every later claim of success.</div>
  </div>
</div>
`};
