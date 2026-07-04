/* Module 2 - Python for Data Science */

CONTENT["python/numpy"] = {
  html: String.raw`
<h1>NumPy Essentials</h1>
<p class="lead">NumPy is the reason Python - a slow language - dominates a
performance-hungry field. One idea powers it all: stop looping over numbers
one at a time, and operate on whole arrays at once.</p>

<h2>Why not just Python lists?</h2>
<p>A Python list stores pointers to scattered objects; a NumPy array
(<code>ndarray</code>) stores raw numbers packed side by side in memory, all of one
type (<code>dtype</code>). That layout lets NumPy hand entire operations to optimized
C code - typically <strong>50–200× faster</strong> than a Python loop, and the gap is
exactly why every ML library (Pandas, scikit-learn, PyTorch) is built on arrays:</p>
<pre><code class="language-python">import numpy as np
import time

data = list(range(10_000_000))
arr  = np.arange(10_000_000)

t0 = time.perf_counter()
squares_slow = [x * x for x in data]          # Python loop
t1 = time.perf_counter()
squares_fast = arr * arr                      # one vectorized op
t2 = time.perf_counter()

print(f"loop      : {t1 - t0:.3f}s")
print(f"vectorized: {t2 - t1:.3f}s")          # ~50-100x faster
</code></pre>
<p><strong>The habit to build:</strong> whenever you catch yourself writing
<code>for</code> over numbers, ask "is there an array operation for this?"
There almost always is.</p>

<h2>Creating arrays and knowing their shape</h2>
<pre><code class="language-python">import numpy as np

a = np.array([1.0, 2.0, 3.0])            # from a list        -> shape (3,)
X = np.array([[1, 2, 3],
              [4, 5, 6]])                # 2-D               -> shape (2, 3)

np.zeros((3, 4))          # 3x4 of 0.0     (placeholder for results)
np.ones(5)                # five 1.0
np.arange(0, 10, 2)       # [0 2 4 6 8]    (like range)
np.linspace(0, 1, 5)      # [0. 0.25 0.5 0.75 1.]  (evenly spaced, inclusive)
rng = np.random.default_rng(0)
rng.normal(0, 1, (2, 3))  # random draws, any shape

print(X.shape)    # (2, 3)  - 2 rows (examples), 3 columns (features)
print(X.dtype)    # int64   - one type for the whole array
print(X.ndim)     # 2       - number of axes
</code></pre>
<p><strong>Shape is everything.</strong> Ninety percent of NumPy (and deep learning)
bugs are shape bugs. Get used to narrating shapes in your head:
"(2, 3) times (3,) gives (2,)". The convention from
<a href="#/math/linear-algebra">Linear Algebra</a> holds: axis 0 = rows = examples,
axis 1 = columns = features.</p>

<h2>Indexing and slicing: reading rectangles of data</h2>
<pre><code class="language-python">X = np.arange(20).reshape(4, 5)   # 4 rows x 5 cols: 0..19

X[1, 3]        # single element (row 1, col 3)         -> 8
X[0]           # entire first row                      -> shape (5,)
X[:, 2]        # entire third column                   -> shape (4,)
X[1:3, :2]     # rows 1-2, first two cols              -> shape (2, 2)
X[::-1]        # rows reversed

# boolean masking - the most important indexing trick in data work:
ages = np.array([22, 41, 17, 35, 64, 15])
adults = ages[ages &gt;= 18]              # [22 41 35 64]
ages[ages &lt; 18] = 18                   # clip in place: assign to a mask

# combine conditions with &amp; and | (NOT 'and'/'or'), parentheses required:
mask = (ages &gt; 20) &amp; (ages &lt; 50)
print(ages[mask])
</code></pre>
<p>One subtlety worth knowing early: slices are <strong>views</strong> (they share memory
with the original - modifying a slice modifies the source), while masked selections
are <strong>copies</strong>. If a mysterious "my array changed by itself" bug ever hits
you, this is why; use <code>.copy()</code> when you need independence.</p>

<h2>Broadcasting: arithmetic between different shapes</h2>
<p>NumPy lets you combine arrays of different shapes by <em>virtually stretching</em>
the smaller one - no memory copied. The rule: compare shapes from the right;
each pair of dimensions must be equal, or one of them must be 1 (it gets stretched).</p>
<div class="diagram">
<svg viewBox="0 0 640 240" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="A 3 by 3 matrix plus a 1 by 3 row vector: the row is virtually copied down three times before element-wise addition">
  <!-- matrix -->
  <rect x="30" y="40" width="150" height="120" rx="6" class="d-box"/>
  <line x1="30" y1="80" x2="180" y2="80" class="d-grid"/>
  <line x1="30" y1="120" x2="180" y2="120" class="d-grid"/>
  <line x1="80" y1="40" x2="80" y2="160" class="d-grid"/>
  <line x1="130" y1="40" x2="130" y2="160" class="d-grid"/>
  <text x="105" y="185" text-anchor="middle" class="d-text-sm">X - shape (3, 3)</text>
  <text x="200" y="105" class="d-text" style="font-size:20px">+</text>
  <!-- row vector -->
  <rect x="235" y="76" width="150" height="40" rx="6" class="d-box-soft"/>
  <line x1="285" y1="76" x2="285" y2="116" class="d-grid"/>
  <line x1="335" y1="76" x2="335" y2="116" class="d-grid"/>
  <!-- ghost copies -->
  <rect x="235" y="120" width="150" height="40" rx="6" class="d-box-muted" stroke-dasharray="5 4" opacity="0.65"/>
  <rect x="235" y="164" width="150" height="40" rx="6" class="d-box-muted" stroke-dasharray="5 4" opacity="0.4"/>
  <text x="310" y="228" text-anchor="middle" class="d-text-sm">mean - shape (1, 3), stretched down "for free"</text>
  <text x="405" y="105" class="d-text" style="font-size:20px">=</text>
  <!-- result -->
  <rect x="440" y="40" width="150" height="120" rx="6" class="d-box"/>
  <line x1="440" y1="80" x2="590" y2="80" class="d-grid"/>
  <line x1="440" y1="120" x2="590" y2="120" class="d-grid"/>
  <line x1="490" y1="40" x2="490" y2="160" class="d-grid"/>
  <line x1="540" y1="40" x2="540" y2="160" class="d-grid"/>
  <text x="515" y="185" text-anchor="middle" class="d-text-sm">result - shape (3, 3)</text>
</svg>
<div class="caption">Broadcasting: the (1, 3) row behaves as if copied down each row of
the (3, 3) matrix. This is how you standardize every feature in one line.</div>
</div>
<pre><code class="language-python">X = rng.normal(50, 10, (1000, 3))     # 1000 examples, 3 features

# standardize every column: (X - mean) / std ... in ONE line
X_std = (X - X.mean(axis=0)) / X.std(axis=0)
#        (1000,3) - (3,)  → broadcast → (1000,3)     ✓

print(X_std.mean(axis=0).round(3))    # [0. 0. 0.]
print(X_std.std(axis=0).round(3))     # [1. 1. 1.]
</code></pre>
<p>The <code>axis</code> argument trips everyone at first. Remember it as
<strong>"the axis that gets collapsed"</strong>: <code>X.mean(axis=0)</code> collapses
the rows, leaving one mean per column - exactly what you want for per-feature
statistics.</p>

<h2>The operations ML uses daily</h2>
<pre><code class="language-python"># aggregations
X.sum(), X.min(), X.max(), X.mean(), X.std()
X.argmax()              # INDEX of the max - how classifiers pick a class!
X.max(axis=1)           # per-row max

# element-wise math (all vectorized)
np.exp(a), np.log(a), np.sqrt(a), np.abs(a), np.clip(a, 0, 1)

# linear algebra (Module 1 in code)
A @ B                   # matrix multiply - THE deep learning operation
A.T                     # transpose
np.linalg.inv(A)        # inverse (rarely used directly - slow, unstable)
np.linalg.norm(a)       # vector length

# reshaping
a.reshape(2, 3)         # same data, new shape
a.reshape(-1, 1)        # -1 = "figure this dimension out" → column vector
a.flatten()             # back to 1-D

# stacking datasets
np.vstack([X1, X2])     # stack rows      (more examples)
np.hstack([X1, X2])     # stack columns   (more features)

# where: vectorized if/else
np.where(ages &gt;= 18, "adult", "minor")
</code></pre>

<h2>Putting it together: a tiny end-to-end example</h2>
<p>Everything above in one realistic snippet - compute each student's best subject
without a single loop:</p>
<pre><code class="language-python">rng = np.random.default_rng(42)
scores = rng.integers(40, 100, size=(5, 3))       # 5 students x 3 subjects
subjects = np.array(["math", "physics", "cs"])

print(scores)
print("class average per subject:", scores.mean(axis=0).round(1))
print("each student's average   :", scores.mean(axis=1).round(1))
print("each student's best      :", subjects[scores.argmax(axis=1)])
print("students above class avg :", (scores.mean(axis=1) &gt; scores.mean()).sum())
</code></pre>

<h2>Common interview questions</h2>
<div class="qa">
  <div class="qa-q">Q1. Why is NumPy so much faster than pure Python?</div>
  <div class="qa-a"><p>Contiguous typed memory (no per-element object overhead), operations executed in
  compiled C over whole arrays (no interpreter in the inner loop), and use of SIMD/vectorized
  CPU instructions. Python's flexibility is paid per element; NumPy pays it once per array.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q2. What will <code>(3, 4) + (4,)</code> shapes broadcast to? And <code>(3, 4) + (3,)</code>?</div>
  <div class="qa-a"><p>(3,4) + (4,) → (3,4): the (4,) row is stretched across the 3 rows.
  (3,4) + (3,) → <strong>error</strong>: aligned from the right, 4 vs 3 don't match. To add
  per-row values you must reshape to (3,1) first - a daily real-world fix.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q3. Difference between a view and a copy?</div>
  <div class="qa-a"><p>A view shares the underlying memory (slices return views - mutating them mutates
  the original); a copy is independent (fancy/boolean indexing returns copies). Use
  <code>.copy()</code> to be explicit when independence matters.</p></div>
</div>

<h2>Quick quiz</h2>
<div class="quiz">
  <div class="quiz-q" data-answer="1">
    <p class="quiz-question">1. X has shape (200, 5). What does X.mean(axis=0) return?</p>
    <button class="quiz-opt">A single number</button>
    <button class="quiz-opt">5 numbers - the mean of each column (feature)</button>
    <button class="quiz-opt">200 numbers - the mean of each row</button>
    <div class="quiz-explain hidden">axis=0 collapses the row axis, producing one value per column: shape (5,). Per-example means would be axis=1.</div>
  </div>
  <div class="quiz-q" data-answer="2">
    <p class="quiz-question">2. Which line selects all rows of X where the last column is positive?</p>
    <button class="quiz-opt">X[X &gt; 0, -1]</button>
    <button class="quiz-opt">X[-1][X &gt; 0]</button>
    <button class="quiz-opt">X[X[:, -1] &gt; 0]</button>
    <div class="quiz-explain hidden">Build the mask from the last column (X[:, -1] &gt; 0, one boolean per row), then use it to index the rows.</div>
  </div>
  <div class="quiz-q" data-answer="0">
    <p class="quiz-question">3. a.reshape(-1, 1) turns a shape-(n,) vector into…</p>
    <button class="quiz-opt">A column vector of shape (n, 1)</button>
    <button class="quiz-opt">A row vector of shape (1, n)</button>
    <button class="quiz-opt">A copy with the same shape</button>
    <div class="quiz-explain hidden">-1 means "infer this dimension". scikit-learn requires 2-D feature matrices, so this exact reshape appears in nearly every ML script.</div>
  </div>
</div>
`};

CONTENT["python/pandas"] = {
  html: String.raw`
<h1>Pandas Essentials</h1>
<p class="lead">If NumPy is a grid of numbers, Pandas is a <em>labeled</em> grid -
a spreadsheet you command with code. It's where every real project starts,
because real data arrives as messy tables, not clean matrices.</p>

<h2>The two objects</h2>
<ul>
  <li><strong>Series</strong> - one labeled column (a NumPy array + an index).</li>
  <li><strong>DataFrame</strong> - a table: multiple Series sharing one index.
      This is the object you'll live in.</li>
</ul>
<pre><code class="language-python">import pandas as pd
import numpy as np

df = pd.DataFrame({
    "name":   ["Asha", "Ben", "Chen", "Dia", "Eli", "Farah"],
    "dept":   ["eng", "sales", "eng", "hr", "sales", "eng"],
    "salary": [85, 52, 91, 48, 61, 78],          # thousands
    "years":  [4, 2, 7, 3, 5, 6],
})

df.head()          # first 5 rows - always your first command
df.shape           # (6, 4)
df.dtypes          # type of each column
df.info()          # non-null counts + dtypes: the health report
df.describe()      # count/mean/std/min/quartiles/max of numeric columns
</code></pre>
<p>Those last three commands are the opening ritual of every analysis -
you'll formalize this into a checklist in <a href="#/eda/workflow">EDA Workflow</a>.
In real work the DataFrame comes from a file:
<code>pd.read_csv("data.csv")</code>, <code>pd.read_excel(...)</code>,
<code>pd.read_parquet(...)</code>, or <code>pd.read_sql(...)</code>.</p>

<h2>Selecting data: loc, iloc, and masks</h2>
<p>Two selection dialects, and mixing them up is the #1 beginner bug:</p>
<pre><code class="language-python">df["salary"]                 # one column  -> Series
df[["name", "salary"]]       # several     -> DataFrame

df.loc[2, "salary"]          # BY LABEL:    index label 2, column "salary"
df.iloc[2, 2]                # BY POSITION: 3rd row, 3rd column
df.loc[1:3, "name":"salary"] # label slices INCLUDE the end!
df.iloc[1:3, 0:2]            # position slices exclude it (like Python)

# boolean filtering - same idea as NumPy masks:
df[df["salary"] &gt; 60]
df[(df["dept"] == "eng") &amp; (df["years"] &gt; 5)]
df.query("dept == 'eng' and years &gt; 5")     # same thing, often more readable

# creating columns
df["salary_per_year"] = df["salary"] / df["years"]
df["senior"] = df["years"] &gt;= 5
</code></pre>
<div class="callout warn">
  <span class="co-title">The SettingWithCopyWarning</span>
  Writing <code>df[df.senior]["salary"] = 0</code> modifies a <em>temporary copy</em>
  and silently does nothing to <code>df</code>. Always write through a single
  <code>.loc</code>: <code>df.loc[df.senior, "salary"] = 0</code>. When Pandas shows
  this warning, it is almost always telling you your edit didn't land.
</div>

<h2>Group-by: the most powerful line in data analysis</h2>
<p>"Average salary <em>per department</em>" is a <strong>split → apply → combine</strong>
operation: split rows into groups, apply an aggregation to each, combine the results
into a new table.</p>
<div class="diagram">
<svg viewBox="0 0 660 250" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Split apply combine: one table splits into three department groups, each is aggregated, results combine into a small summary table">
  <rect x="15" y="70" width="120" height="110" rx="6" class="d-box"/>
  <text x="75" y="60" text-anchor="middle" class="d-text-sm">full table</text>
  <line x1="15" y1="106" x2="135" y2="106" class="d-grid"/>
  <line x1="15" y1="142" x2="135" y2="142" class="d-grid"/>
  <text x="75" y="95" text-anchor="middle" class="d-text-sm">eng · 85</text>
  <text x="75" y="130" text-anchor="middle" class="d-text-sm">sales · 52</text>
  <text x="75" y="166" text-anchor="middle" class="d-text-sm">eng · 91 …</text>
  <line x1="140" y1="95" x2="215" y2="55" class="d-line"/>
  <line x1="140" y1="125" x2="215" y2="125" class="d-line"/>
  <line x1="140" y1="155" x2="215" y2="195" class="d-line"/>
  <text x="150" y="35" class="d-text-accent">1 · split</text>
  <rect x="220" y="35" width="115" height="44" rx="6" class="d-box-soft"/>
  <text x="277" y="62" text-anchor="middle" class="d-text-sm">eng: 85, 91, 78</text>
  <rect x="220" y="103" width="115" height="44" rx="6" class="d-box-soft"/>
  <text x="277" y="130" text-anchor="middle" class="d-text-sm">sales: 52, 61</text>
  <rect x="220" y="171" width="115" height="44" rx="6" class="d-box-soft"/>
  <text x="277" y="198" text-anchor="middle" class="d-text-sm">hr: 48</text>
  <line x1="335" y1="57" x2="420" y2="57" class="d-line"/>
  <line x1="335" y1="125" x2="420" y2="125" class="d-line"/>
  <line x1="335" y1="193" x2="420" y2="193" class="d-line"/>
  <text x="345" y="35" class="d-text-accent">2 · apply mean()</text>
  <rect x="425" y="40" width="90" height="34" rx="6" class="d-box-muted"/>
  <text x="470" y="62" text-anchor="middle" class="d-text-sm">84.7</text>
  <rect x="425" y="108" width="90" height="34" rx="6" class="d-box-muted"/>
  <text x="470" y="130" text-anchor="middle" class="d-text-sm">56.5</text>
  <rect x="425" y="176" width="90" height="34" rx="6" class="d-box-muted"/>
  <text x="470" y="198" text-anchor="middle" class="d-text-sm">48.0</text>
  <line x1="515" y1="57" x2="560" y2="105" class="d-line"/>
  <line x1="515" y1="125" x2="560" y2="125" class="d-line"/>
  <line x1="515" y1="193" x2="560" y2="145" class="d-line"/>
  <text x="530" y="35" class="d-text-accent">3 · combine</text>
  <rect x="562" y="88" width="88" height="80" rx="6" class="d-box"/>
  <line x1="562" y1="114" x2="650" y2="114" class="d-grid"/>
  <line x1="562" y1="140" x2="650" y2="140" class="d-grid"/>
  <text x="606" y="107" text-anchor="middle" class="d-text-sm">eng 84.7</text>
  <text x="606" y="133" text-anchor="middle" class="d-text-sm">sales 56.5</text>
  <text x="606" y="159" text-anchor="middle" class="d-text-sm">hr 48.0</text>
</svg>
<div class="caption">groupby = split → apply → combine. One line of code, three conceptual steps.</div>
</div>
<pre><code class="language-python">df.groupby("dept")["salary"].mean()

# several aggregations at once:
df.groupby("dept").agg(
    avg_salary=("salary", "mean"),
    headcount=("name", "count"),
    max_years=("years", "max"),
)

# group by multiple keys, e.g. dept x seniority:
df.groupby(["dept", "senior"])["salary"].mean().unstack()

# pivot tables - groupby dressed as a spreadsheet:
pd.pivot_table(df, values="salary", index="dept",
               columns="senior", aggfunc="mean")
</code></pre>

<h2>Joining tables: merge</h2>
<pre><code class="language-python">locations = pd.DataFrame({
    "dept": ["eng", "sales", "hr"],
    "city": ["Pune", "Mumbai", "Delhi"],
})

# SQL-style join on the shared column
merged = df.merge(locations, on="dept", how="left")
</code></pre>
<p><code>how=</code> works exactly like SQL: <code>"inner"</code> keeps only matches,
<code>"left"</code> keeps every row of the left table (filling misses with
<code>NaN</code>), <code>"outer"</code> keeps everything. After any merge, check
<code>merged.shape</code> - an unexpected row explosion means your join key
wasn't unique.</p>

<h2>The everyday toolbox</h2>
<pre><code class="language-python">df.sort_values("salary", ascending=False)
df["dept"].value_counts()               # frequency table - use constantly
df["dept"].unique()
df.rename(columns={"salary": "salary_k"})
df.drop(columns=["senior"])
df.assign(bonus=lambda d: d["salary"] * 0.1)   # chain-friendly new column

# apply a function to a column (prefer vectorized ops when possible)
df["name_len"] = df["name"].str.len()          # .str = string methods
df["dept_upper"] = df["dept"].str.upper()

# dates get the same treatment via .dt
events = pd.DataFrame({"ts": pd.to_datetime(["2026-01-05", "2026-02-11"])})
events["month"] = events["ts"].dt.month
events["weekday"] = events["ts"].dt.day_name()
</code></pre>

<h2>Common interview questions</h2>
<div class="qa">
  <div class="qa-q">Q1. loc vs iloc?</div>
  <div class="qa-a"><p><code>loc</code> selects by <em>label</em> (and its slices are inclusive of the end);
  <code>iloc</code> selects by <em>integer position</em> (end-exclusive, like Python).
  After filtering or sorting, labels and positions diverge - using the wrong one silently
  grabs the wrong rows.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q2. How would you get the top-3 highest-paid people per department?</div>
  <div class="qa-a"><p><code>df.sort_values("salary", ascending=False).groupby("dept").head(3)</code> -
  sort first, then take the first rows of each group. (Or
  <code>groupby("dept")["salary"].nlargest(3)</code>.)</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q3. A merge unexpectedly grew from 10k rows to 45k. What happened?</div>
  <div class="qa-a"><p>The join key wasn't unique in the right table: each left row matched multiple right
  rows (a many-to-many join). Check <code>right.duplicated("key").sum()</code>, deduplicate
  or aggregate the right table first, or pass <code>validate="one_to_one"</code> /
  <code>"many_to_one"</code> to make Pandas raise on surprises.</p></div>
</div>

<h2>Quick quiz</h2>
<div class="quiz">
  <div class="quiz-q" data-answer="2">
    <p class="quiz-question">1. Which line correctly zeroes the salary of senior employees?</p>
    <button class="quiz-opt">df[df.senior]["salary"] = 0</button>
    <button class="quiz-opt">df["salary"][df.senior] == 0</button>
    <button class="quiz-opt">df.loc[df.senior, "salary"] = 0</button>
    <div class="quiz-explain hidden">Chained indexing (option 1) writes to a temporary copy - the infamous SettingWithCopyWarning. A single .loc does label-based selection and assignment in one step.</div>
  </div>
  <div class="quiz-q" data-answer="0">
    <p class="quiz-question">2. "Median years of experience per department" is…</p>
    <button class="quiz-opt">df.groupby("dept")["years"].median()</button>
    <button class="quiz-opt">df["years"].median().groupby("dept")</button>
    <button class="quiz-opt">df.median().groupby("years")</button>
    <div class="quiz-explain hidden">Split by dept, select the years column, apply median. The groupby must come before the aggregation.</div>
  </div>
  <div class="quiz-q" data-answer="1">
    <p class="quiz-question">3. A left merge of orders (left) with customers (right) produces NaN in customer columns for some rows. Meaning?</p>
    <button class="quiz-opt">Those customers have no orders</button>
    <button class="quiz-opt">Those orders reference a customer id missing from the customers table</button>
    <button class="quiz-opt">The merge failed entirely</button>
    <div class="quiz-explain hidden">A left join keeps all left rows; unmatched ones get NaN on the right side. Orphaned foreign keys like this are a classic data-quality find during EDA.</div>
  </div>
</div>
`};

CONTENT["python/data-cleaning"] = {
  html: String.raw`
<h1>Data Cleaning Basics</h1>
<p class="lead">Real datasets arrive broken: duplicated rows, "N/A" strings pretending
to be data, prices stored as text, three spellings of the same city. Cleaning is
not glamorous, but it's where data scientists genuinely spend most of their time -
and where silent errors are born.</p>

<h2>The mindset: garbage in, garbage out - quietly</h2>
<p>A model trained on dirty data doesn't crash. It trains fine, scores fine on the
equally-dirty test set, and then fails in production. That's what makes cleaning
matter: <strong>errors here don't announce themselves.</strong> The professional habit is
a fixed inspection routine before any analysis - the same five commands, every time:</p>
<pre><code class="language-python">import pandas as pd
import numpy as np

df = pd.read_csv("sales.csv")

df.head(10)              # eyeball actual values
df.info()                # dtypes + non-null counts: the single best overview
df.describe(include="all")
df.isna().sum()          # missing per column
df.duplicated().sum()    # exact duplicate rows
</code></pre>

<h2>Problem 1: duplicates</h2>
<pre><code class="language-python">df = df.drop_duplicates()                        # exact copies
df = df.drop_duplicates(subset=["order_id"],     # same id, keep newest
                        keep="last")
</code></pre>
<p>Think before dropping: two identical rows might be a data-entry accident
(drop) or two genuinely identical purchases (keep!). Duplicates are a question
about the <em>real world</em>, not just the table.</p>

<h2>Problem 2: wrong types</h2>
<p>The classic: a numeric column read as text because one row contains
<code>"12,500"</code> or <code>"unknown"</code>. Everything downstream
(means, models, plots) breaks or lies.</p>
<pre><code class="language-python">df.dtypes                                # spot 'object' where numbers should be

# strip junk, then convert; errors="coerce" turns failures into NaN
df["price"] = (df["price"].astype(str)
               .str.replace(",", "", regex=False)
               .str.replace("$", "", regex=False))
df["price"] = pd.to_numeric(df["price"], errors="coerce")

df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
df["category"] = df["category"].astype("category")   # memory + speed win
</code></pre>
<p>After every <code>errors="coerce"</code>, count the casualties:
<code>df["price"].isna().sum()</code>. Each new NaN was a value that failed to parse -
you want to know how many and why, not discover it later.</p>

<h2>Problem 3: disguised missing values</h2>
<p>Files encode "missing" a dozen creative ways: <code>"N/A"</code>, <code>"?"</code>,
<code>"none"</code>, <code>-999</code>, empty strings, or 0 where 0 is impossible
(a person with height 0). Pandas only recognizes real <code>NaN</code> -
the rest silently poison your statistics:</p>
<pre><code class="language-python"># tell read_csv up front:
df = pd.read_csv("sales.csv", na_values=["N/A", "?", "none", "-999", ""])

# or repair after the fact:
df["height"] = df["height"].replace(0, np.nan)

# then quantify the damage:
missing = df.isna().mean().sort_values(ascending=False)
print((missing * 100).round(1).astype(str) + "%")
</code></pre>
<p>What to <em>do</em> about missing values (drop vs impute, and the traps of each)
is a big enough topic to get its own chapter:
<a href="#/eda/missing-values">Handling Missing Values</a>.</p>

<h2>Problem 4: inconsistent categories and strings</h2>
<p><code>"Delhi"</code>, <code>"delhi "</code>, <code>"DELHI"</code> are three different
groups to a computer. Always check <code>value_counts()</code> on categorical columns -
the misspellings jump out at the bottom of the list:</p>
<pre><code class="language-python">df["city"].value_counts()          # reveals: 'Delhi', 'delhi ', 'DELHI'...

df["city"] = (df["city"].astype(str)
              .str.strip()         # kill stray whitespace
              .str.title())        # unify capitalization

# map known aliases to one canonical name
df["city"] = df["city"].replace({"Bombay": "Mumbai", "Bengaluru": "Bangalore"})
</code></pre>

<h2>Problem 5: impossible values</h2>
<p>Ages of 250, negative quantities, orders dated in the future. Define validity
rules from domain knowledge and check them explicitly:</p>
<pre><code class="language-python">problems = {
    "negative price": (df["price"] &lt; 0).sum(),
    "age &gt; 120":      (df["age"] &gt; 120).sum(),
    "future order":   (df["order_date"] &gt; pd.Timestamp.today()).sum(),
}
print(problems)

# fix = correct if you can, else set to NaN (don't silently drop rows)
df.loc[df["age"] &gt; 120, "age"] = np.nan
</code></pre>
<p>Genuine-but-extreme values (a real $50,000 order among $50 ones) are a different
beast from impossible ones - that judgment call is covered in
<a href="#/eda/outliers">Outlier Detection</a>.</p>

<h2>A reusable cleaning pipeline</h2>
<p>Package the steps into one function so the <em>same</em> cleaning applies to every
new batch of data - reproducibility is the difference between cleaning and tampering:</p>
<pre><code class="language-python">def clean_sales(raw: pd.DataFrame) -&gt; pd.DataFrame:
    df = raw.copy()

    # 1. duplicates
    df = df.drop_duplicates(subset=["order_id"], keep="last")

    # 2. types
    df["price"] = pd.to_numeric(
        df["price"].astype(str).str.replace(r"[$,]", "", regex=True),
        errors="coerce")
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")

    # 3. strings
    df["city"] = df["city"].astype(str).str.strip().str.title()

    # 4. validity rules
    df.loc[df["price"] &lt; 0, "price"] = np.nan
    df = df[df["order_date"] &lt;= pd.Timestamp.today()]

    return df

clean = clean_sales(df)
print(f"rows: {len(df)} -&gt; {len(clean)}")
print(clean.info())
</code></pre>
<div class="callout">
  <span class="co-title">Golden rules of cleaning</span>
  1) Never edit the raw file - clean in code, keep the original untouchable.
  2) Log what each step changed (row counts, NaN counts) so decisions are auditable.
  3) Every "fix" encodes an assumption about the real world - write the assumption
  down. 4) Cleaning decisions made using the <em>test set</em> are a form of leakage;
  fit your rules on training data (more in <a href="#/features/scaling">Module 5</a>).
</div>

<h2>Common interview questions</h2>
<div class="qa">
  <div class="qa-q">Q1. A CSV's numeric column loads as dtype object. Diagnose and fix.</div>
  <div class="qa-a"><p>Some values aren't parseable as numbers (currency symbols, thousands separators,
  "N/A" strings). Inspect with <code>df[col].str.match(r'^-?[\d.]+$') == False</code>-style
  filters or value_counts, strip the junk, then <code>pd.to_numeric(..., errors="coerce")</code>
  and count the resulting NaNs to verify nothing important was lost.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q2. Why is -999 used as a missing marker dangerous?</div>
  <div class="qa-a"><p>It's a legal number, so it silently flows into means, standard deviations, and model
  training - dragging statistics toward nonsense without any error. Convert sentinel values
  to NaN immediately on load (<code>na_values</code>), before any computation touches them.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q3. When is dropping duplicate rows the wrong move?</div>
  <div class="qa-a"><p>When identical rows are legitimate repeated events - two same-priced purchases by
  the same customer on the same day, repeated sensor readings, etc. Deduplicate on a true
  business key (order id) rather than full-row equality, and confirm with the data's owner
  what a duplicate actually means.</p></div>
</div>

<h2>Quick quiz</h2>
<div class="quiz">
  <div class="quiz-q" data-answer="1">
    <p class="quiz-question">1. After pd.to_numeric(col, errors="coerce"), unparseable values become…</p>
    <button class="quiz-opt">0</button>
    <button class="quiz-opt">NaN</button>
    <button class="quiz-opt">They raise an exception</button>
    <div class="quiz-explain hidden">"coerce" converts failures to NaN - which is why you must count NaNs afterwards to see what was sacrificed. errors="raise" (the default) would throw instead.</div>
  </div>
  <div class="quiz-q" data-answer="2">
    <p class="quiz-question">2. A height column contains many 0 values for adults. Best first step?</p>
    <button class="quiz-opt">Leave them - zero is a number</button>
    <button class="quiz-opt">Drop every row with height 0</button>
    <button class="quiz-opt">Treat 0 as disguised missing: convert to NaN, then decide how to handle it</button>
    <div class="quiz-explain hidden">An adult height of 0 is physically impossible - it's a missing-value marker in disguise. Converting to NaN makes the missingness visible and keeps the row's other columns usable.</div>
  </div>
  <div class="quiz-q" data-answer="0">
    <p class="quiz-question">3. Why should cleaning live in a function/script rather than manual edits?</p>
    <button class="quiz-opt">Reproducibility - the same rules apply to new data, and every change is auditable</button>
    <button class="quiz-opt">It runs faster</button>
    <button class="quiz-opt">Pandas requires it</button>
    <div class="quiz-explain hidden">Hand-edited spreadsheets can't be re-applied to next month's export and leave no record of what changed. Code is documentation, audit trail, and automation at once.</div>
  </div>
</div>
`};
