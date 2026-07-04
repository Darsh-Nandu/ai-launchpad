/* Module 1 - Math & Stats Foundations (part 3):
   Distributions, Hypothesis Testing, Confidence Intervals,
   Correlation vs Causation */

CONTENT["math/distributions"] = {
  html: String.raw`
<h1>Distributions</h1>
<p class="lead">A distribution is the complete answer to "what values does this
variable take, and how often?" Learn to recognize the five or six shapes that
cover 95% of real data, and half of statistics becomes pattern-matching.</p>

<h2>What a distribution tells you</h2>
<p>A histogram of your data is an <em>empirical</em> distribution. A named distribution
(normal, binomial, Poisson…) is a <em>mathematical idealization</em> - a formula that
generates such shapes from one or two parameters. Recognizing "this column looks
Poisson" buys you a lot: formulas for its mean and spread, sensible models,
and warnings about what will break.</p>
<p>Two kinds exist: <strong>discrete</strong> distributions assign probability to separate
values (counts: 0, 1, 2, …) via a <em>probability mass function</em> (PMF);
<strong>continuous</strong> ones spread probability over ranges via a <em>probability
density function</em> (PDF), where only areas under the curve - never single points -
have probability.</p>

<h2>The normal (Gaussian) - the celebrity</h2>
<div class="math-box">
$$f(x) = \frac{1}{\sigma\sqrt{2\pi}}\; e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$
</div>
<p>Two parameters: center \(\mu\) and spread \(\sigma\). Heights, measurement errors,
averages of anything - all end up normal. The reason is the
<strong>Central Limit Theorem</strong>: the sum (or mean) of many independent random
effects tends toward a normal distribution <em>regardless of the original shape</em>.
That theorem is why the normal appears everywhere and why so much of statistics
assumes it.</p>
<p>The rule to memorize:</p>
<div class="diagram">
<svg viewBox="0 0 640 290" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Bell curve with 68, 95 and 99.7 percent bands at one, two and three standard deviations">
  <line x1="40" y1="240" x2="600" y2="240" class="d-line"/>
  <!-- bell -->
  <path d="M 60 238 C 170 232, 230 40, 320 40 C 410 40, 470 232, 580 238" class="d-curve"/>
  <!-- sigma markers -->
  <line x1="320" y1="240" x2="320" y2="38" class="d-line" stroke-dasharray="3 3"/>
  <line x1="235" y1="240" x2="235" y2="118" class="d-grid"/>
  <line x1="405" y1="240" x2="405" y2="118" class="d-grid"/>
  <line x1="150" y1="240" x2="150" y2="205" class="d-grid"/>
  <line x1="490" y1="240" x2="490" y2="205" class="d-grid"/>
  <text x="320" y="262" text-anchor="middle" class="d-text-sm">μ</text>
  <text x="235" y="262" text-anchor="middle" class="d-text-sm">μ−σ</text>
  <text x="405" y="262" text-anchor="middle" class="d-text-sm">μ+σ</text>
  <text x="150" y="262" text-anchor="middle" class="d-text-sm">μ−2σ</text>
  <text x="490" y="262" text-anchor="middle" class="d-text-sm">μ+2σ</text>
  <!-- bands -->
  <line x1="235" y1="90" x2="405" y2="90" class="d-line-accent"/>
  <text x="320" y="80" text-anchor="middle" class="d-text-accent">68% within ±1σ</text>
  <line x1="150" y1="140" x2="490" y2="140" class="d-line"/>
  <text x="320" y="132" text-anchor="middle" class="d-text-sm">95% within ±2σ &nbsp;·&nbsp; 99.7% within ±3σ</text>
  <text x="320" y="285" text-anchor="middle" class="d-text-sm">z-score: z = (x − μ)/σ - "how many σ away from the mean?"</text>
</svg>
<div class="caption">The 68–95–99.7 rule. A value 3σ from the mean is a 1-in-370 event -
which is why |z| &gt; 3 is a common outlier flag.</div>
</div>
<p>The <strong>z-score</strong> \(z = (x - \mu)/\sigma\) re-expresses any value as
"standard deviations from the mean" - it's how we standardize features
(<a href="#/features/scaling">Scaling</a>) and detect outliers.</p>

<h2>Binomial - counting successes</h2>
<div class="math-box">
$$P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}
\qquad \mathbb{E}[X] = np, \quad \mathrm{Var}(X) = np(1-p)$$
</div>
<p>The number of successes in \(n\) independent yes/no trials, each with success
probability \(p\). Coin flips, conversion counts ("of 1,000 visitors, how many buy?"),
correct answers on a quiz. As \(n\) grows it looks increasingly normal - the CLT
at work. A/B testing is largely applied binomial statistics.</p>

<h2>Poisson - counting rare events in a window</h2>
<div class="math-box">
$$P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}
\qquad \mathbb{E}[X] = \mathrm{Var}(X) = \lambda$$
</div>
<p>Counts of events in a fixed interval when events are independent and occur at a
steady average rate \(\lambda\): support tickets per hour, typos per page, goals per
match. Signature property: <strong>mean = variance</strong>. If your count data has
variance ≫ mean ("overdispersion"), it isn't Poisson - something is clustering.</p>

<h2>The rest of the cast</h2>
<table>
  <tr><th>Distribution</th><th>Shape / support</th><th>Typical use</th></tr>
  <tr><td><strong>Uniform</strong></td><td>flat over [a, b]</td><td>random initialization, "no prior knowledge", <code>rng.random()</code></td></tr>
  <tr><td><strong>Bernoulli</strong></td><td>single 0/1 trial</td><td>one click/no-click; binomial with n = 1; the output of a binary classifier</td></tr>
  <tr><td><strong>Exponential</strong></td><td>decaying curve on [0, ∞)</td><td>waiting time between Poisson events (time until next ticket)</td></tr>
  <tr><td><strong>Log-normal</strong></td><td>right-skewed, positive</td><td>incomes, city sizes, file sizes - take the log, get a normal</td></tr>
  <tr><td><strong>Student's t</strong></td><td>bell with heavy tails</td><td>small-sample inference; the basis of the t-test</td></tr>
</table>

<h2>See them by sampling</h2>
<pre><code class="language-python">import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(1)
fig, axes = plt.subplots(2, 2, figsize=(10, 7))

# normal: heights of adults (cm)
axes[0, 0].hist(rng.normal(170, 8, 10_000), bins=50)
axes[0, 0].set_title("Normal(170, 8) - heights")

# binomial: conversions among 100 visitors, p = 0.04
axes[0, 1].hist(rng.binomial(100, 0.04, 10_000), bins=range(15))
axes[0, 1].set_title("Binomial(100, 0.04) - conversions")

# poisson: support tickets per hour, λ = 3
axes[1, 0].hist(rng.poisson(3, 10_000), bins=range(13))
axes[1, 0].set_title("Poisson(3) - tickets/hour")

# central limit theorem: means of 50 UNIFORM draws become normal!
means = rng.random((10_000, 50)).mean(axis=1)
axes[1, 1].hist(means, bins=50)
axes[1, 1].set_title("CLT: means of 50 uniforms → bell")

plt.tight_layout()
plt.show()

# the CLT panel is the punchline: the raw data is flat,
# yet averages of it form a near-perfect bell curve.
</code></pre>

<h2>Common interview questions</h2>
<div class="qa">
  <div class="qa-q">Q1. State the Central Limit Theorem and why it matters.</div>
  <div class="qa-a"><p>The mean of \(n\) independent samples from (almost) any distribution approaches a
  normal distribution as \(n\) grows, with spread shrinking like \(\sigma/\sqrt{n}\).
  It's why sample means are trustworthy, why confidence intervals use z/t values, and why
  the normal shows up in data that is a sum of many small effects.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q2. How do you decide between modeling counts as binomial vs Poisson?</div>
  <div class="qa-a"><p>Binomial when there's a fixed number of trials n with success probability p
  (bounded above by n). Poisson when events occur in a continuous window with no natural
  upper limit at rate λ. (Poisson is the limit of binomial with large n, small p, np = λ.)</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q3. Your model's residuals should be normal, but their kurtosis is high. Consequence?</div>
  <div class="qa-a"><p>Extreme errors happen far more often than the normal assumption predicts, so
  prediction intervals are too narrow and squared-error training over-reacts to tail events.
  Fixes: robust losses (Huber), transforming the target, or heavy-tailed error models.</p></div>
</div>

<h2>Quick quiz</h2>
<div class="quiz">
  <div class="quiz-q" data-answer="1">
    <p class="quiz-question">1. Data is normal with μ = 100, σ = 15. Roughly what fraction lies between 70 and 130?</p>
    <button class="quiz-opt">68%</button>
    <button class="quiz-opt">95%</button>
    <button class="quiz-opt">99.7%</button>
    <div class="quiz-explain hidden">70 and 130 are μ ± 2σ → the 95% band of the 68–95–99.7 rule.</div>
  </div>
  <div class="quiz-q" data-answer="2">
    <p class="quiz-question">2. Website errors arrive at a steady average of 2 per hour, independently. The count of errors in one hour is best modeled as…</p>
    <button class="quiz-opt">Normal(2, 2)</button>
    <button class="quiz-opt">Binomial(2, 0.5)</button>
    <button class="quiz-opt">Poisson(2)</button>
    <div class="quiz-explain hidden">Independent events at a constant rate in a time window = Poisson with λ = 2. There's no fixed number of "trials", so binomial doesn't apply.</div>
  </div>
  <div class="quiz-q" data-answer="0">
    <p class="quiz-question">3. Why do averages of skewed data still get bell-shaped as sample size grows?</p>
    <button class="quiz-opt">The Central Limit Theorem</button>
    <button class="quiz-opt">Bessel's correction</button>
    <button class="quiz-opt">The law of small numbers</button>
    <div class="quiz-explain hidden">The CLT applies regardless of the underlying shape (as long as variance is finite) - that's exactly what makes it so powerful.</div>
  </div>
</div>
`};

CONTENT["math/hypothesis-testing"] = {
  html: String.raw`
<h1>Hypothesis Testing &amp; p-value</h1>
<p class="lead">"Our new model scored 2% higher - but is that real, or luck?"
Hypothesis testing is the machinery for answering that question honestly.
The p-value is its most used - and most misunderstood - number.</p>

<h2>The courtroom analogy</h2>
<p>A hypothesis test works like a trial. The defendant is the boring explanation -
<em>"nothing is going on, it's just chance"</em> - called the
<strong>null hypothesis \(H_0\)</strong>. It is presumed innocent. The exciting claim -
<em>"there is a real effect"</em> - is the <strong>alternative \(H_1\)</strong>.
Your data is the evidence. The verdict rules are:</p>
<ul>
  <li>Convict (<strong>reject \(H_0\)</strong>) only if the evidence would be very
      surprising under innocence.</li>
  <li>Otherwise, acquit (<strong>fail to reject \(H_0\)</strong>) - which, importantly,
      is <em>not</em> a declaration of innocence. It means "not enough evidence."</li>
</ul>

<h2>The recipe, step by step</h2>
<ol>
  <li><strong>State \(H_0\)</strong> - e.g. "this coin is fair, \(p = 0.5\)".</li>
  <li><strong>Choose a significance level \(\alpha\)</strong> - the false-conviction
      rate you'll tolerate, conventionally 0.05, <em>before</em> looking at data.</li>
  <li><strong>Compute a test statistic</strong> - a number measuring how far the data
      deviates from what \(H_0\) predicts.</li>
  <li><strong>Compute the p-value</strong> - see below.</li>
  <li><strong>Decide</strong>: p \(\le \alpha\) → reject \(H_0\). Otherwise, fail to reject.</li>
</ol>

<h2>What a p-value actually is</h2>
<div class="callout">
  <span class="co-title">The definition - memorize this exact sentence</span>
  The p-value is the probability of observing data <em>at least as extreme as yours</em>,
  <em>assuming the null hypothesis is true</em>.
  It is \(P(\text{data this extreme} \mid H_0)\) -
  <strong>not</strong> \(P(H_0 \mid \text{data})\), not the probability you're wrong,
  and not the probability the effect is real.
</div>
<p>Small p-value = "if nothing were going on, data like mine would be rare" =
evidence against \(H_0\). It is a measure of <em>surprise under the boring
explanation</em>, nothing more.</p>

<h2>Worked example: is this coin fair?</h2>
<p>You flip a coin 100 times and get <strong>62 heads</strong>. Suspicious?</p>
<p>Under \(H_0\) (fair coin), the number of heads is Binomial(100, 0.5):
mean \(np = 50\), standard deviation \(\sqrt{np(1-p)} = \sqrt{25} = 5\).
Convert the observation to a z-score:</p>
<div class="math-box">
$$z = \frac{62 - 50}{5} = 2.4$$
</div>
<p>62 heads is 2.4 standard deviations above what fairness predicts. From the normal
distribution, the probability of being at least this extreme <em>on either side</em>
(a two-tailed test - we'd have been equally suspicious of 38 heads) is:</p>
<div class="math-box">
$$p\text{-value} = 2 \times P(Z \ge 2.4) \approx 2 \times 0.0082 = 0.016$$
</div>
<p>Since \(0.016 \lt 0.05\), we reject \(H_0\): a fair coin would produce a result
this lopsided only ~1.6% of the time. Note what we did <em>not</em> conclude:
we did not compute "98.4% probability the coin is biased." We only measured
how surprising the data is under fairness.</p>

<h2>The two ways to be wrong</h2>
<table>
  <tr><th></th><th>\(H_0\) actually true<br>(no real effect)</th><th>\(H_0\) actually false<br>(real effect exists)</th></tr>
  <tr><td><strong>Reject \(H_0\)</strong></td><td>❌ <strong>Type I error</strong> (false alarm)<br>probability = \(\alpha\)</td><td>✅ correct - a true discovery<br>probability = <em>power</em></td></tr>
  <tr><td><strong>Fail to reject</strong></td><td>✅ correct</td><td>❌ <strong>Type II error</strong> (missed effect)<br>probability = \(\beta\)</td></tr>
</table>
<p>Lowering \(\alpha\) (being harder to convince) trades Type I errors for Type II
errors. The way to reduce both at once is <strong>more data</strong> - larger samples
shrink the noise (\(\sigma/\sqrt{n}\)), making real effects easier to distinguish
from luck. "Power analysis" is choosing \(n\) before an experiment so that a real
effect of the size you care about would likely be detected.</p>

<h2>In code: comparing two models properly</h2>
<pre><code class="language-python">import numpy as np
from scipy import stats

rng = np.random.default_rng(7)

# accuracy of two models across 30 cross-validation folds
model_a = rng.normal(0.845, 0.020, 30)   # mean 84.5%
model_b = rng.normal(0.860, 0.020, 30)   # mean 86.0% - but noisy!

# H0: the two models have the same mean accuracy
t_stat, p_value = stats.ttest_ind(model_a, model_b)
print(f"t = {t_stat:.2f},  p = {p_value:.4f}")

alpha = 0.05
if p_value &lt;= alpha:
    print("Reject H0: the accuracy difference is unlikely to be luck.")
else:
    print("Fail to reject H0: the difference could easily be noise.")

# --- the coin example, exactly (no normal approximation needed) ---------
p_coin = stats.binomtest(62, n=100, p=0.5).pvalue
print("coin p-value:", round(p_coin, 4))         # ≈ 0.021, same verdict
</code></pre>

<h2>How p-values get abused</h2>
<ul>
  <li><strong>p-hacking:</strong> testing 20 things and reporting the one with p &lt; 0.05.
      With \(\alpha = 0.05\), one in twenty <em>pure-noise</em> tests "succeeds" by design.
      Fix: decide the test beforehand; correct for multiple comparisons.</li>
  <li><strong>"Significant" ≠ "important":</strong> with a million samples, a 0.01%
      accuracy difference gets a tiny p-value. Always report the <em>effect size</em>
      alongside the p-value.</li>
  <li><strong>p = 0.06 ≠ "no effect":</strong> failing to reject is not proof of absence -
      maybe the sample was just too small (low power).</li>
</ul>

<h2>Common interview questions</h2>
<div class="qa">
  <div class="qa-q">Q1. Define the p-value precisely.</div>
  <div class="qa-a"><p>The probability, computed assuming \(H_0\) is true, of obtaining a result at
  least as extreme as the one observed. It is <em>not</em> the probability that \(H_0\)
  is true - that would require Bayes' theorem and a prior.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q2. What are Type I and Type II errors, and what controls them?</div>
  <div class="qa-a"><p>Type I: rejecting a true null (false positive), controlled directly by \(\alpha\).
  Type II: failing to reject a false null (missed effect), probability \(\beta\); reduced by
  larger samples, bigger true effects, or a higher \(\alpha\). Power = 1 − β.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q3. Your A/B test on 40 metrics found 2 significant at α = 0.05. Impressed?</div>
  <div class="qa-a"><p>No - under pure chance you'd <em>expect</em> 40 × 0.05 = 2 false positives.
  This needs a multiple-comparison correction (e.g. Bonferroni: require p &lt; 0.05/40),
  or a pre-registered primary metric.</p></div>
</div>

<h2>Quick quiz</h2>
<div class="quiz">
  <div class="quiz-q" data-answer="1">
    <p class="quiz-question">1. p = 0.03 means…</p>
    <button class="quiz-opt">There's a 3% chance the null hypothesis is true</button>
    <button class="quiz-opt">If the null were true, data this extreme would occur about 3% of the time</button>
    <button class="quiz-opt">The effect has a 97% chance of being real</button>
    <div class="quiz-explain hidden">The p-value conditions on H₀ being true; it says nothing directly about the probability of H₀ itself.</div>
  </div>
  <div class="quiz-q" data-answer="2">
    <p class="quiz-question">2. A spam filter that flags a legitimate email is committing the equivalent of…</p>
    <button class="quiz-opt">A Type II error</button>
    <button class="quiz-opt">No error</button>
    <button class="quiz-opt">A Type I error</button>
    <div class="quiz-explain hidden">H₀ = "email is normal". Flagging it = rejecting a true null = false positive = Type I. Letting real spam through would be Type II.</div>
  </div>
  <div class="quiz-q" data-answer="0">
    <p class="quiz-question">3. A test returns p = 0.20. The correct conclusion is…</p>
    <button class="quiz-opt">The data doesn't provide enough evidence against H₀</button>
    <button class="quiz-opt">H₀ has been proven true</button>
    <button class="quiz-opt">The effect is 20% as large as expected</button>
    <div class="quiz-explain hidden">"Fail to reject" is a verdict of insufficient evidence, never proof of the null - the effect might be real but the sample too small to see it.</div>
  </div>
</div>
`};

CONTENT["math/confidence-intervals"] = {
  html: String.raw`
<h1>Confidence Intervals</h1>
<p class="lead">A single number ("average session: 8.2 minutes") hides how sure you are.
A confidence interval - "8.2 ± 0.6 minutes" - carries the uncertainty with it.
Every error bar in every research paper is one of these.</p>

<h2>The problem: samples wobble</h2>
<p>You want the true average \(\mu\) of a population (all users, ever), but you only
have a sample of \(m\) of them. Your sample mean \(\bar{x}\) is an estimate - and a
different sample would have given a slightly different \(\bar{x}\). How much do such
estimates wobble?</p>
<p>The Central Limit Theorem answers precisely: sample means are approximately normal
around the true \(\mu\), with standard deviation</p>
<div class="math-box">
$$\mathrm{SE} = \frac{\sigma}{\sqrt{m}}$$
</div>
<p>called the <strong>standard error</strong>. Note the \(\sqrt{m}\): to halve your
uncertainty you need <em>four times</em> the data. This diminishing return governs
everything from survey sizes to how many random seeds you should average in an
ML paper.</p>

<h2>Building the interval</h2>
<p>Since \(\bar{x}\) lands within ±1.96 standard errors of \(\mu\) 95% of the time
(the normal's 95% band), the recipe is:</p>
<div class="math-box">
$$\text{95\% CI} \;=\; \bar{x} \;\pm\; 1.96 \cdot \frac{s}{\sqrt{m}}$$
</div>
<p>(For small samples, \(m \lesssim 30\), replace 1.96 by the slightly larger value from
Student's t distribution - it compensates for also having to estimate \(s\).)
Other confidence levels just swap the multiplier: 90% → 1.645, 99% → 2.576.
More confidence = wider interval; there's no free lunch.</p>
<h3>Worked example</h3>
<p>You measure 100 user sessions: \(\bar{x} = 8.2\) min, \(s = 3.0\) min.</p>
<div class="math-box">
$$\mathrm{SE} = \frac{3.0}{\sqrt{100}} = 0.3
\qquad
\text{95\% CI} = 8.2 \pm 1.96 \times 0.3 = [7.61,\; 8.79] \text{ min}$$
</div>
<p>Report: "mean session length 8.2 min (95% CI: 7.6–8.8)." A reader instantly sees
both the estimate and its reliability. If a competitor claims their true mean is
9.5 - that's outside your interval, and you have grounds to doubt the claim
(this is the deep link to hypothesis testing: a 95% CI contains exactly the
\(H_0\) values a two-tailed \(\alpha=0.05\) test would <em>not</em> reject).</p>

<h2>What "95% confident" really means</h2>
<div class="callout warn">
  <span class="co-title">The classic misreading</span>
  Wrong: "there's a 95% probability that \(\mu\) is between 7.61 and 8.79."
  \(\mu\) is a fixed (unknown) number - it's either in that interval or it isn't.
  Right: <strong>the procedure works 95% of the time</strong> - if you repeated the
  whole experiment many times, 95% of the intervals you'd construct would capture
  \(\mu\). Your one interval is a draw from that reliable process.
</div>
<p>The simulation below makes this concrete - and honestly, watching it once
teaches more than any paragraph:</p>
<pre><code class="language-python">import numpy as np
from scipy import stats

rng = np.random.default_rng(42)
TRUE_MU, TRUE_SIGMA = 8.0, 3.0     # the "population" truth (normally unknown!)

captured = 0
n_experiments, m = 10_000, 100

for _ in range(n_experiments):
    sample = rng.normal(TRUE_MU, TRUE_SIGMA, m)
    se = sample.std(ddof=1) / np.sqrt(m)
    lo = sample.mean() - 1.96 * se
    hi = sample.mean() + 1.96 * se
    if lo &lt;= TRUE_MU &lt;= hi:
        captured += 1

print(f"intervals that captured the true mean: {captured / n_experiments:.1%}")
# -> ≈ 95.0%. The *procedure* has a 95% success rate.

# --- the one-liner for real work -----------------------------------------
sample = rng.normal(TRUE_MU, TRUE_SIGMA, 100)
ci = stats.t.interval(0.95, df=len(sample)-1,
                      loc=sample.mean(),
                      scale=stats.sem(sample))
print("95% CI:", np.round(ci, 2))
</code></pre>

<h2>Where you'll meet CIs in ML</h2>
<ul>
  <li><strong>Model comparison:</strong> "Model B: 86.0% ± 1.2 vs Model A: 84.5% ± 1.1" -
      overlapping intervals are a warning that the difference may be noise.</li>
  <li><strong>Papers:</strong> error bars and shaded bands around training curves are CIs
      (or standard errors) across random seeds.</li>
  <li><strong>A/B tests:</strong> conversion lift is always reported with an interval;
      a lift whose CI includes 0 hasn't been demonstrated.</li>
  <li><strong>Bootstrap:</strong> when no formula exists (e.g. CI for a median or an AUC),
      resample your data with replacement thousands of times and take the middle 95%
      of the recomputed statistic - a technique worth knowing by name.</li>
</ul>

<h2>Common interview questions</h2>
<div class="qa">
  <div class="qa-q">Q1. Interpret "95% CI for the lift: [−0.2%, +3.1%]" from an A/B test.</div>
  <div class="qa-a"><p>The data is consistent with effects from slightly negative to +3.1%. Because 0 is
  inside the interval, the test hasn't shown a real improvement at the 5% level - equivalent
  to p &gt; 0.05. Either ship on other grounds or collect more data to narrow the interval.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q2. How does sample size affect the width of a CI?</div>
  <div class="qa-a"><p>Width shrinks with \(1/\sqrt{m}\): quadruple the data to halve the interval.
  This square-root law is why the last decimal point of certainty is astronomically expensive.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q3. Why use Student's t instead of the normal for small samples?</div>
  <div class="qa-a"><p>With small \(m\), the sample standard deviation \(s\) is itself a noisy estimate
  of \(\sigma\). The t distribution's heavier tails widen the interval to account for that
  extra uncertainty; by \(m \approx 30\) it's nearly identical to the normal.</p></div>
</div>

<h2>Quick quiz</h2>
<div class="quiz">
  <div class="quiz-q" data-answer="2">
    <p class="quiz-question">1. To make a confidence interval half as wide, you need roughly…</p>
    <button class="quiz-opt">Twice the data</button>
    <button class="quiz-opt">Half the data</button>
    <button class="quiz-opt">Four times the data</button>
    <div class="quiz-explain hidden">Width ∝ 1/√m, so halving the width requires m to grow by a factor of 4.</div>
  </div>
  <div class="quiz-q" data-answer="1">
    <p class="quiz-question">2. Moving from a 95% CI to a 99% CI (same data) makes the interval…</p>
    <button class="quiz-opt">Narrower</button>
    <button class="quiz-opt">Wider</button>
    <button class="quiz-opt">The same width</button>
    <div class="quiz-explain hidden">More confidence demands a bigger net: the multiplier grows from 1.96 to 2.576.</div>
  </div>
  <div class="quiz-q" data-answer="0">
    <p class="quiz-question">3. "95% confidence" refers to…</p>
    <button class="quiz-opt">The long-run success rate of the interval-building procedure</button>
    <button class="quiz-opt">The probability that the true mean is in this specific interval</button>
    <button class="quiz-opt">The fraction of data inside the interval</button>
    <div class="quiz-explain hidden">The true mean is fixed; it's the intervals that vary from sample to sample. 95% of such intervals capture the truth - as the simulation on this page shows.</div>
  </div>
</div>
`};

CONTENT["math/correlation-causation"] = {
  html: String.raw`
<h1>Correlation vs Causation</h1>
<p class="lead">Ice cream sales and drowning deaths rise together every year.
Ice cream does not cause drowning. Understanding exactly why this happens -
and what it takes to actually prove causation - may be the single most
practical statistical skill you'll ever learn.</p>

<h2>Measuring correlation: Pearson's r</h2>
<p>The correlation coefficient measures how tightly two variables move together
<em>linearly</em>:</p>
<div class="math-box">
$$r = \frac{\sum_i (x_i - \bar{x})(y_i - \bar{y})}
{\sqrt{\sum_i (x_i - \bar{x})^2}\; \sqrt{\sum_i (y_i - \bar{y})^2}}
\qquad -1 \le r \le 1$$
</div>
<p>The numerator is the <em>covariance</em>: for each point, do \(x\) and \(y\) deviate
from their means in the same direction (+) or opposite directions (−)? The
denominator rescales so units don't matter. Interpretation:</p>
<div class="diagram">
<svg viewBox="0 0 640 220" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Four scatter plots showing r near 1, r near 0, r near minus 1, and a strong curved pattern with r near 0">
  <!-- r = 0.95 -->
  <rect x="15" y="30" width="140" height="140" rx="6" class="d-box-muted"/>
  <circle cx="35" cy="150" r="3.5" class="d-dot"/><circle cx="55" cy="135" r="3.5" class="d-dot"/>
  <circle cx="70" cy="115" r="3.5" class="d-dot"/><circle cx="85" cy="108" r="3.5" class="d-dot"/>
  <circle cx="100" cy="85" r="3.5" class="d-dot"/><circle cx="115" cy="70" r="3.5" class="d-dot"/>
  <circle cx="130" cy="55" r="3.5" class="d-dot"/><circle cx="60" cy="122" r="3.5" class="d-dot"/>
  <circle cx="108" cy="92" r="3.5" class="d-dot"/>
  <text x="85" y="195" text-anchor="middle" class="d-text">r ≈ +0.95</text>
  <!-- r = 0 -->
  <rect x="175" y="30" width="140" height="140" rx="6" class="d-box-muted"/>
  <circle cx="195" cy="60" r="3.5" class="d-dot"/><circle cx="230" cy="140" r="3.5" class="d-dot"/>
  <circle cx="250" cy="55" r="3.5" class="d-dot"/><circle cx="285" cy="120" r="3.5" class="d-dot"/>
  <circle cx="210" cy="105" r="3.5" class="d-dot"/><circle cx="265" cy="90" r="3.5" class="d-dot"/>
  <circle cx="295" cy="65" r="3.5" class="d-dot"/><circle cx="220" cy="150" r="3.5" class="d-dot"/>
  <circle cx="245" cy="110" r="3.5" class="d-dot"/>
  <text x="245" y="195" text-anchor="middle" class="d-text">r ≈ 0</text>
  <!-- r = -0.95 -->
  <rect x="335" y="30" width="140" height="140" rx="6" class="d-box-muted"/>
  <circle cx="355" cy="55" r="3.5" class="d-dot"/><circle cx="375" cy="72" r="3.5" class="d-dot"/>
  <circle cx="390" cy="88" r="3.5" class="d-dot"/><circle cx="405" cy="100" r="3.5" class="d-dot"/>
  <circle cx="420" cy="118" r="3.5" class="d-dot"/><circle cx="435" cy="130" r="3.5" class="d-dot"/>
  <circle cx="455" cy="148" r="3.5" class="d-dot"/><circle cx="382" cy="80" r="3.5" class="d-dot"/>
  <circle cx="428" cy="125" r="3.5" class="d-dot"/>
  <text x="405" y="195" text-anchor="middle" class="d-text">r ≈ −0.95</text>
  <!-- nonlinear -->
  <rect x="495" y="30" width="140" height="140" rx="6" class="d-box-muted"/>
  <circle cx="510" cy="60" r="3.5" class="d-dot"/><circle cx="525" cy="100" r="3.5" class="d-dot"/>
  <circle cx="540" cy="130" r="3.5" class="d-dot"/><circle cx="555" cy="145" r="3.5" class="d-dot"/>
  <circle cx="570" cy="140" r="3.5" class="d-dot"/><circle cx="585" cy="115" r="3.5" class="d-dot"/>
  <circle cx="600" cy="80" r="3.5" class="d-dot"/><circle cx="615" cy="52" r="3.5" class="d-dot"/>
  <text x="565" y="195" text-anchor="middle" class="d-text">strong but r ≈ 0!</text>
</svg>
<div class="caption">Pearson's r sees only straight-line relationships. The U-shaped
panel has an obvious pattern yet r ≈ 0 - always plot your data.</div>
</div>
<p>Rules of thumb: |r| &gt; 0.7 strong, 0.3–0.7 moderate, &lt; 0.3 weak - but these
depend heavily on the field. Two warnings baked into the math: r captures only
<em>linear</em> association (see the fourth panel), and a single outlier can
manufacture or destroy a large r. For monotonic-but-curved relationships,
<strong>Spearman's rank correlation</strong> (correlate the <em>ranks</em> instead of
the values) is the robust alternative.</p>

<h2>Why correlation ≠ causation: the four suspects</h2>
<p>When X and Y correlate, there are four possible explanations, and the data alone
cannot tell you which:</p>
<table>
  <tr><th>Explanation</th><th>Structure</th><th>Example</th></tr>
  <tr><td>X causes Y</td><td>X → Y</td><td>practice hours → skill</td></tr>
  <tr><td>Y causes X (reverse causation)</td><td>Y → X</td><td>"police presence correlates with crime" - crime attracts police</td></tr>
  <tr><td>A third variable Z causes both (<strong>confounder</strong>)</td><td>X ← Z → Y</td><td>summer heat → ice cream sales <em>and</em> swimming → drownings</td></tr>
  <tr><td>Pure coincidence</td><td>-</td><td>with thousands of series, some align by chance (per-capita cheese consumption vs bedsheet deaths, r = 0.95)</td></tr>
</table>
<p>The confounder case is the most dangerous because it feels causal. The heat
variable Z creates a rock-solid, perfectly reproducible correlation between two
things with no causal link at all. Controlling for Z (comparing only days of equal
temperature) makes the ice-cream/drowning correlation vanish.</p>

<h3>Simpson's paradox: correlation can even flip</h3>
<p>A treatment can look worse than a placebo overall, yet better within <em>every
subgroup</em> - if doctors gave the treatment mostly to severe cases. Aggregation
across an uneven mix reverses the sign of the relationship. Moral: check whether a
correlation survives when you split the data by plausible confounders. This is a
routine step in serious <a href="#/eda/correlation-analysis">correlation analysis</a>.</p>

<h2>What it takes to prove causation</h2>
<ul>
  <li><strong>Randomized controlled trials (RCT / A/B tests):</strong> randomly assigning
      X breaks every arrow into it - no confounder can survive randomization. This is
      why product teams A/B test instead of mining logs for correlations.</li>
  <li><strong>Natural experiments &amp; quasi-experiments:</strong> when you can't randomize,
      exploit boundaries (policy changes, lotteries) that assign X as-if randomly.</li>
  <li><strong>Controlling for confounders:</strong> regression with the confounder included,
      matching, stratification - works only for confounders you know and measured.</li>
</ul>
<div class="callout">
  <span class="co-title">What this means for ML models</span>
  A predictive model is a correlation machine - and that's fine for prediction:
  ice cream sales genuinely do predict drowning risk. The trap is
  <strong>acting</strong> on the model as if it were causal ("ban ice cream to reduce
  drowning") or expecting it to survive a distribution shift (a model using hospital
  ward as a pneumonia-severity feature collapses when the admission policy changes).
  Predict with correlations; intervene only with causation.
</div>

<h2>Computing it</h2>
<pre><code class="language-python">import numpy as np
from scipy import stats

rng = np.random.default_rng(3)

# a confounder Z (temperature) driving both X and Y
z = rng.normal(25, 7, 500)                    # daily temperature
ice_cream = 20 + 3.0 * z + rng.normal(0, 8, 500)
drownings = 1 + 0.15 * z + rng.normal(0, 1, 500)

r, p = stats.pearsonr(ice_cream, drownings)
print(f"raw correlation:        r = {r:.2f}  (p = {p:.1e})")   # strong!

# control for the confounder: correlate the residuals after removing z
ice_resid = ice_cream - np.poly1d(np.polyfit(z, ice_cream, 1))(z)
drown_resid = drownings - np.poly1d(np.polyfit(z, drownings, 1))(z)
r_partial, _ = stats.pearsonr(ice_resid, drown_resid)
print(f"controlling for temp:   r = {r_partial:.2f}")          # ≈ 0 - gone

# rank-based alternative, robust to outliers and curves:
rho, _ = stats.spearmanr(ice_cream, drownings)
print(f"spearman rho:           {rho:.2f}")
</code></pre>

<h2>Common interview questions</h2>
<div class="qa">
  <div class="qa-q">Q1. Give the possible explanations for a strong observed correlation between X and Y.</div>
  <div class="qa-a"><p>X→Y, Y→X (reverse causation), a confounder Z→both, or coincidence/selection
  effects. Distinguishing them requires intervention (randomization) or careful causal
  assumptions - never the correlation alone.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q2. Pearson r is 0. Can the variables still be related?</div>
  <div class="qa-a"><p>Absolutely - r only measures linear association. y = x² over a symmetric range has
  r = 0 despite perfect dependence. Plot the data; consider Spearman correlation or mutual
  information for non-linear relationships.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q3. Why are A/B tests trusted over correlational log analysis?</div>
  <div class="qa-a"><p>Random assignment guarantees the two groups differ only in the treatment, so any
  outcome difference (beyond noise) must be caused by it. Log analysis compares self-selected
  groups, which differ in countless confounding ways.</p></div>
</div>

<h2>Quick quiz</h2>
<div class="quiz">
  <div class="quiz-q" data-answer="1">
    <p class="quiz-question">1. Cities with more firefighters have more fire damage. Best explanation?</p>
    <button class="quiz-opt">Firefighters cause damage</button>
    <button class="quiz-opt">A confounder - bigger cities (and bigger fires) mean both more firefighters and more damage</button>
    <button class="quiz-opt">The correlation must be coincidence</button>
    <div class="quiz-explain hidden">City size and fire severity drive both variables. Controlling for them would shrink or reverse the correlation - firefighters reduce damage, given the same fire.</div>
  </div>
  <div class="quiz-q" data-answer="2">
    <p class="quiz-question">2. Which relationship would Pearson's r badly understate?</p>
    <button class="quiz-opt">y = 3x + noise</button>
    <button class="quiz-opt">y = −2x + noise</button>
    <button class="quiz-opt">y = (x − 5)² + noise, for x from 0 to 10</button>
    <div class="quiz-explain hidden">The U-shape is symmetric around x = 5: positive and negative co-deviations cancel, so r ≈ 0 despite a strong deterministic relationship.</div>
  </div>
  <div class="quiz-q" data-answer="0">
    <p class="quiz-question">3. A model predicts hospital readmission using ZIP code, and it works. What can you safely conclude?</p>
    <button class="quiz-opt">ZIP code carries predictive signal - but changing someone's ZIP wouldn't change their risk</button>
    <button class="quiz-opt">Living in certain ZIP codes causes readmission</button>
    <button class="quiz-opt">The model is wrong to use ZIP code</button>
    <div class="quiz-explain hidden">ZIP proxies income, environment, access to care - real predictive signal via confounders. Prediction ≠ intervention: the model is useful for risk flags, useless (or harmful) as a causal claim.</div>
  </div>
</div>
`};
