/* Module 6 - Machine Learning Algorithms */

CONTENT["ml/linear-regression"] = {
  html: String.raw`
<h1>Linear Regression</h1>
<p class="lead">The "hello world" of machine learning: fit the best straight line
through data, and in doing so meet the three ideas - model, loss, optimization -
that every other algorithm reuses.</p>

<h2>The idea: a disciplined rule of thumb</h2>
<p>You're guessing house prices. You notice bigger houses cost more, so you make up a
rule of thumb: <em>"price ≈ some rate per square meter, plus a base amount."</em>
That rule is a straight line. Linear regression is just the disciplined version of
this: instead of eyeballing the rate and the base amount, we let the data choose the
values that make the guesses <strong>least wrong on average</strong>.</p>
<p>Picture a scatter plot of area vs. price and a ruler you can slide and rotate.
For any ruler position, measure each point's vertical distance to the ruler
(how wrong the guess is - the <em>residual</em>). Linear regression finds the one
ruler position where those errors, squared and averaged, are as small as possible.</p>

<h2>The model</h2>
<p>With one feature, the model is a line; with \(n\) features, it's the same idea in
higher dimensions (a hyperplane):</p>
<div class="math-box">
$$\hat{y} = w\,x + b
\qquad\text{(one feature)}
\qquad\qquad
\hat{y} = \mathbf{w}^\top \mathbf{x} + b
\qquad\text{(n features)}$$
</div>
<p>Here \(w\) is the slope (how much \(\hat{y}\) changes per unit of \(x\)) and
\(b\) is the intercept/bias (the prediction when all features are 0).
"Training" means finding the \(w, b\) that minimize a <strong>cost function</strong> -
a single number measuring how wrong the model is over the whole dataset.</p>
<h3>Assumptions (asked in every interview)</h3>
<ul>
  <li><strong>Linearity</strong> - the true relationship between \(x\) and \(y\) is roughly linear.</li>
  <li><strong>Independence</strong> - examples don't influence each other.</li>
  <li><strong>Homoscedasticity</strong> - the spread of errors is constant across all \(x\) (no funnel shape).</li>
  <li><strong>Normality of residuals</strong> - errors are roughly bell-shaped (needed for confidence intervals, not for prediction).</li>
  <li><strong>No multicollinearity</strong> - features aren't near-duplicates of each other.</li>
</ul>

<h2>Training: a loss and a way downhill</h2>
<h3>The cost function: Mean Squared Error</h3>
<p>For \(m\) training examples, average the squared residuals (the \(\tfrac{1}{2}\) is
cosmetic - it cancels a 2 when differentiating):</p>
<div class="math-box">
$$J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} \left( \hat{y}^{(i)} - y^{(i)} \right)^2$$
</div>
<p>Squaring does two jobs: errors can't cancel out (negative and positive both count),
and big errors are punished disproportionately. \(J\) is a smooth bowl-shaped
function of \(w\) and \(b\) - it has exactly one minimum.</p>

<h3>Finding the minimum: gradient descent</h3>
<p>Compute the slope of the cost with respect to each parameter, then step downhill.
Repeat until the steps stop changing anything:</p>
<div class="math-box">
$$\frac{\partial J}{\partial w} = \frac{1}{m} \sum_{i=1}^{m} \left( \hat{y}^{(i)} - y^{(i)} \right) x^{(i)}
\qquad
\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} \left( \hat{y}^{(i)} - y^{(i)} \right)$$
$$w \leftarrow w - \alpha \frac{\partial J}{\partial w}
\qquad\qquad
b \leftarrow b - \alpha \frac{\partial J}{\partial b}$$
</div>
<p>\(\alpha\) is the <strong>learning rate</strong>: too small and training crawls,
too large and it overshoots the minimum and diverges. This exact update loop -
predict, measure error, nudge weights downhill - is the same one that trains
neural networks in Module 7. Learn it here, on the easiest possible model.</p>

<details class="disclosure">
<summary>Show full derivation: the closed-form Normal Equation</summary>
<div class="disclosure-body">
<p>Linear regression is special: the minimum can also be computed directly, no loop
needed. Fold the bias into the weights by adding a column of 1s to \(\mathbf{X}\),
so \(\hat{\mathbf{y}} = \mathbf{X}\mathbf{w}\). The cost in matrix form:</p>
$$J(\mathbf{w}) = \frac{1}{2m}\,(\mathbf{X}\mathbf{w} - \mathbf{y})^\top (\mathbf{X}\mathbf{w} - \mathbf{y})$$
<p>Take the gradient with respect to \(\mathbf{w}\) and set it to zero
(the bottom of the bowl is where the slope vanishes):</p>
$$\nabla_{\mathbf{w}} J = \frac{1}{m}\,\mathbf{X}^\top(\mathbf{X}\mathbf{w} - \mathbf{y}) = \mathbf{0}
\;\;\Longrightarrow\;\;
\mathbf{X}^\top\mathbf{X}\,\mathbf{w} = \mathbf{X}^\top\mathbf{y}$$
<p>Solving for \(\mathbf{w}\) gives the <strong>normal equation</strong>:</p>
$$\boxed{\;\mathbf{w} = (\mathbf{X}^\top\mathbf{X})^{-1}\,\mathbf{X}^\top\mathbf{y}\;}$$
<p>Why ever use gradient descent then? Inverting \(\mathbf{X}^\top\mathbf{X}\) costs
roughly \(O(n^3)\) in the number of features and can be numerically unstable when
features are correlated. For a handful of features, use the normal equation;
for thousands (or for any model more complex than linear), gradient descent wins.</p>
</div>
</details>


<div class="diagram">
<svg viewBox="0 0 640 330" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Scatter plot with fitted regression line and dashed residual segments from each point to the line">
  <!-- axes -->
  <line x1="60" y1="280" x2="610" y2="280" class="d-line"/>
  <line x1="60" y1="280" x2="60" y2="40" class="d-line"/>
  <text x="560" y="300" class="d-text-sm">area x</text>
  <text x="30" y="52" class="d-text-sm">price y</text>
  <!-- fitted line -->
  <line x1="80" y1="250" x2="580" y2="80" class="d-line-accent"/>
  <text x="470" y="82" class="d-text-accent">ŷ = w·x + b</text>
  <!-- residuals (dashed) then points -->
  <line x1="140" y1="215" x2="140" y2="230" class="d-line" stroke-dasharray="4 3"/>
  <line x1="200" y1="225" x2="200" y2="209" class="d-line" stroke-dasharray="4 3"/>
  <line x1="260" y1="180" x2="260" y2="189" class="d-line" stroke-dasharray="4 3"/>
  <line x1="330" y1="150" x2="330" y2="165" class="d-line" stroke-dasharray="4 3"/>
  <line x1="400" y1="155" x2="400" y2="141" class="d-line" stroke-dasharray="4 3"/>
  <line x1="470" y1="108" x2="470" y2="117" class="d-line" stroke-dasharray="4 3"/>
  <line x1="530" y1="110" x2="530" y2="97" class="d-line" stroke-dasharray="4 3"/>
  <circle cx="140" cy="215" r="5" class="d-dot"/>
  <circle cx="200" cy="225" r="5" class="d-dot"/>
  <circle cx="260" cy="180" r="5" class="d-dot"/>
  <circle cx="330" cy="150" r="5" class="d-dot"/>
  <circle cx="400" cy="155" r="5" class="d-dot"/>
  <circle cx="470" cy="108" r="5" class="d-dot"/>
  <circle cx="530" cy="110" r="5" class="d-dot"/>
  <!-- residual label -->
  <text x="345" y="162" class="d-text-sm">residual = ŷ − y</text>
  <text x="60" y="322" class="d-text-sm">Training minimizes the average of the squared dashed distances (MSE).</text>
</svg>
<div class="caption">The fitted line and the residuals it is trying to shrink.</div>
</div>

<h2>From scratch in NumPy</h2>
<p>Gradient descent from scratch in ~25 lines of NumPy. Every line maps to a formula above:</p>
<pre><code class="language-python">import numpy as np

# --- toy data: area (m²) vs price (lakh) --------------------------------
rng = np.random.default_rng(42)
x = rng.uniform(40, 200, size=60)                 # 60 houses
y = 0.9 * x + 15 + rng.normal(0, 12, size=60)     # true line + noise

# scale the feature (gradient descent converges much faster when x ~ N(0,1))
x_mean, x_std = x.mean(), x.std()
xs = (x - x_mean) / x_std

# --- gradient descent ----------------------------------------------------
w, b = 0.0, 0.0        # start anywhere; the bowl has one minimum
alpha = 0.1            # learning rate
m = len(xs)

for epoch in range(200):
    y_hat = w * xs + b                       # 1) predict
    error = y_hat - y                        # 2) residuals
    dw = (error * xs).mean()                 # 3) ∂J/∂w
    db = error.mean()                        #    ∂J/∂b
    w -= alpha * dw                          # 4) step downhill
    b -= alpha * db
    if epoch % 50 == 0:
        cost = (error ** 2).mean() / 2
        print(f"epoch {epoch:3d}   cost = {cost:8.2f}")

# convert back to the original (unscaled) feature for interpretation
slope = w / x_std
intercept = b - w * x_mean / x_std
print(f"\nlearned: price ≈ {slope:.2f} · area + {intercept:.2f}")
# ≈ the true 0.9 and 15 we generated the data with
</code></pre>

<details class="disclosure">
<summary>Show: the same model in scikit-learn (3 lines) + R² evaluation</summary>
<div class="disclosure-body">
<pre><code class="language-python">from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

X = x.reshape(-1, 1)                # sklearn expects a 2-D feature matrix
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0)

model = LinearRegression()
model.fit(X_train, y_train)         # internally solves the normal equation

y_pred = model.predict(X_test)
print("slope     :", model.coef_[0])
print("intercept :", model.intercept_)
print("R²        :", r2_score(y_test, y_pred))       # 1.0 = perfect fit
print("RMSE      :", np.sqrt(mean_squared_error(y_test, y_pred)))
</code></pre>
<p>Note that we evaluate on a <em>held-out test set</em> - the model's score on data
it trained on is always flattering and always misleading. Much more on this in
<a href="#/ml/cross-validation">Cross-Validation</a>.</p>
</div>
</details>

<h2>When to use it - pros &amp; cons</h2>
<table>
  <tr><th>Pros</th><th>Cons</th></tr>
  <tr>
    <td>Fast to train, even on huge datasets</td>
    <td>Can only model straight-line relationships (without manual feature engineering)</td>
  </tr>
  <tr>
    <td>Fully interpretable - each weight says how much a feature matters</td>
    <td>Sensitive to outliers (squared error amplifies them)</td>
  </tr>
  <tr>
    <td>No hyperparameters to tune (in the basic form)</td>
    <td>Struggles when features are highly correlated (multicollinearity)</td>
  </tr>
  <tr>
    <td>A strong, honest baseline - always try it first</td>
    <td>Assumptions (constant error variance, independence) are often violated in practice</td>
  </tr>
</table>
<p><strong>Use it when:</strong> the target is a continuous number, you suspect a roughly
linear relationship, and you need interpretability or a baseline. If it underfits,
move to tree ensembles (<a href="#/ml/random-forest">Random Forest</a>,
<a href="#/ml/gradient-boosting">Gradient Boosting</a>) - not straight to deep learning.</p>

<h2>Common interview questions</h2>
<div class="qa">
  <div class="qa-q">Q1. Why minimize <em>squared</em> error rather than absolute error?</div>
  <div class="qa-a"><p>Squared error is smoothly differentiable everywhere (absolute error has a kink at 0),
  has a closed-form solution, and corresponds to maximum-likelihood estimation when the noise
  is Gaussian. The trade-off: it is more sensitive to outliers. When outliers are a problem,
  minimize absolute error (this gives median-like fits) or use Huber loss.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q2. Gradient descent vs the normal equation - when would you pick each?</div>
  <div class="qa-a"><p>Normal equation: small number of features (inverting \(\mathbf{X}^\top\mathbf{X}\)
  is \(O(n^3)\)), one shot, no learning rate. Gradient descent: many features, streaming/huge
  data, and it's the only option for models without closed forms (i.e., almost everything else).</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q3. What does an R² of 0.85 mean?</div>
  <div class="qa-a"><p>The model explains 85% of the variance in the target around its mean.
  R² = 0 means "no better than always predicting the mean"; R² can even be negative on a
  test set if the model is worse than that baseline.</p></div>
</div>
<div class="qa">
  <div class="qa-q">Q4. Your residual plot shows a funnel shape (errors grow with x). What's wrong?</div>
  <div class="qa-a"><p>Heteroscedasticity - the constant-variance assumption is violated. Predictions
  are still unbiased but the confidence intervals are wrong. Common fixes: log-transform the
  target, or use weighted least squares.</p></div>
</div>

<h2>Quick quiz</h2>
<div class="quiz">
  <div class="quiz-q" data-answer="1">
    <p class="quiz-question">1. If the learning rate α is set far too high, gradient descent will…</p>
    <button class="quiz-opt">Converge, just more slowly</button>
    <button class="quiz-opt">Overshoot the minimum and diverge</button>
    <button class="quiz-opt">Get stuck at the starting point</button>
    <div class="quiz-explain hidden">Each step jumps past the minimum to a point with an even steeper slope, so the next jump is bigger - the cost explodes. Too <em>small</em> an α is what makes it slow.</div>
  </div>
  <div class="quiz-q" data-answer="0">
    <p class="quiz-question">2. In ŷ = w·x + b for house prices (x = area in m²), the weight w is best read as…</p>
    <button class="quiz-opt">The price change per extra square meter</button>
    <button class="quiz-opt">The price of an average house</button>
    <button class="quiz-opt">The percentage error of the model</button>
    <div class="quiz-explain hidden">A weight is always "change in prediction per unit change of its feature, holding the rest fixed." That's what makes linear regression so interpretable.</div>
  </div>
  <div class="quiz-q" data-answer="2">
    <p class="quiz-question">3. Why must the model be evaluated on a test set the model never trained on?</p>
    <button class="quiz-opt">Training metrics are too expensive to compute</button>
    <button class="quiz-opt">The test set makes the model train faster</button>
    <button class="quiz-opt">Training scores are inflated - they reward memorization, not generalization</button>
    <div class="quiz-explain hidden">A model can score perfectly on data it has seen by memorizing it. Only performance on unseen data estimates how it will behave in the real world.</div>
  </div>
</div>
`};
