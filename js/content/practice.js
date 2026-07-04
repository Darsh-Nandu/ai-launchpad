/* Module 12 - Practice Zone */

CONTENT["practice/cheat-sheets"] = {
  html: String.raw`
<h1>Cheat Sheets</h1>
<p class="lead">One compressed summary per module - the formulas, rules of thumb,
and decision tables worth having open while you work. Revisit after finishing
each module; print the ones you use. (Tip: your browser's Print → Save as PDF
turns any of these into a one-pager.)</p>

<h2>Math &amp; Stats (Module 1)</h2>
<table>
  <tr><th>Item</th><th>The essence</th></tr>
  <tr><td>Dot product</td><td>\(\mathbf{a}^\top\mathbf{b} = \sum a_i b_i = \lVert a\rVert\lVert b\rVert\cos\theta\) - similarity</td></tr>
  <tr><td>Gradient</td><td>\(\nabla J\) points steepest uphill → update \(\mathbf{w} \leftarrow \mathbf{w} - \alpha\nabla J\)</td></tr>
  <tr><td>Chain rule</td><td>\(\frac{dy}{dx} = \frac{dy}{du}\frac{du}{dx}\) - multiply local sensitivities (= backprop)</td></tr>
  <tr><td>Bayes</td><td>posterior = likelihood × prior / evidence; base rates dominate rare events</td></tr>
  <tr><td>Mean vs median</td><td>mean ≫ median ⇒ right skew ⇒ consider log transform</td></tr>
  <tr><td>68–95–99.7</td><td>fraction of a normal within ±1σ, ±2σ, ±3σ; |z| &gt; 3 = outlier flag</td></tr>
  <tr><td>p-value</td><td>P(data this extreme | H₀ true) - NOT P(H₀ | data)</td></tr>
  <tr><td>95% CI</td><td>\(\bar{x} \pm 1.96\, s/\sqrt{m}\); halve the width = 4× the data</td></tr>
  <tr><td>Correlation traps</td><td>confounders, reverse causation, non-linearity (r misses curves), Simpson's paradox</td></tr>
</table>

<h2>Python / Pandas (Module 2)</h2>
<table>
  <tr><th>Task</th><th>Idiom</th></tr>
  <tr><td>First look</td><td><code>df.head() · df.info() · df.describe() · df.isna().sum()</code></td></tr>
  <tr><td>Filter rows</td><td><code>df[(df.a &gt; 5) &amp; (df.b == "x")]</code> - &amp;/|, parentheses required</td></tr>
  <tr><td>Safe assignment</td><td><code>df.loc[mask, "col"] = value</code> (never chained indexing)</td></tr>
  <tr><td>Aggregate per group</td><td><code>df.groupby("k")["v"].mean()</code> · <code>.agg(new=("v","mean"))</code></td></tr>
  <tr><td>Join</td><td><code>a.merge(b, on="key", how="left")</code>; check shape after!</td></tr>
  <tr><td>Vectorize</td><td>arrays, not loops; <code>axis=0</code> collapses rows (per-column stats)</td></tr>
  <tr><td>Broadcasting</td><td>align shapes from the right; dims equal or 1</td></tr>
</table>

<h2>EDA &amp; Features (Modules 4–5)</h2>
<table>
  <tr><th>Rule</th><th>Why</th></tr>
  <tr><td>Quality audit before analysis</td><td>patterns on broken data are confidently wrong</td></tr>
  <tr><td>Baseline first</td><td>majority class % / predict-the-mean - models must beat it</td></tr>
  <tr><td>Missingness: MCAR/MAR/MNAR</td><td>mechanism decides the fix; add was_missing flags</td></tr>
  <tr><td>Outliers: IQR fences</td><td>[Q1 − 1.5·IQR, Q3 + 1.5·IQR]; fix/cap/log/robust-model - never silent deletion</td></tr>
  <tr><td>Encoding</td><td>one-hot (nominal, low-card) · ordinal (real order) · target enc out-of-fold (high-card)</td></tr>
  <tr><td>Scaling</td><td>standardize for distance/gradient models; trees don't care; <strong>fit on train only</strong></td></tr>
  <tr><td>Imbalance</td><td>fix metrics first (F1/PR-AUC) → class weights → threshold → SMOTE in-pipeline</td></tr>
  <tr><td>Leakage test</td><td>"would this feature exist at prediction time?" - r ≈ 0.99 with target = alarm</td></tr>
</table>

<h2>Classical ML (Module 6)</h2>
<table>
  <tr><th>Model</th><th>One-liner</th><th>Watch out</th></tr>
  <tr><td>Linear/Logistic</td><td>interpretable baseline; convex; linear boundary</td><td>scale + regularize; multicollinearity</td></tr>
  <tr><td>KNN</td><td>vote of k nearest; zero training</td><td>scaling mandatory; dies in high dims</td></tr>
  <tr><td>Decision tree</td><td>readable if/else; nonlinear + interactions free</td><td>overfits unpruned; unstable</td></tr>
  <tr><td>Random forest</td><td>bagged decorrelated trees; strong default</td><td>can't extrapolate; slower inference</td></tr>
  <tr><td>Boosting (XGB/LGBM)</td><td>sequential error-correctors; tabular king</td><td>needs early stopping; rate×trees pairing</td></tr>
  <tr><td>SVM</td><td>max margin + kernels; great in high-dim small data</td><td>O(m²) training; tune C and γ together</td></tr>
  <tr><td>Naive Bayes</td><td>counts + independence lie; instant text baseline</td><td>probabilities over-confident</td></tr>
  <tr><td>K-Means / DBSCAN</td><td>blobs by distance / shapes by density</td><td>k &amp; scaling / eps &amp; varying density</td></tr>
</table>
<p>Evaluation: precision = trust the alarms; recall = catch the cases;
F1 harmonizes; ROC-AUC ranks (PR-AUC under imbalance);
MAE robust, RMSE punishes big misses; RMSE ≫ MAE ⇒ investigate outliers.
Protocol: test set in a vault; CV for all decisions; every fitted step inside
the pipeline; bias (both scores bad) → more capacity; variance (train ≫ val) →
regularize/more data.</p>

<h2>Deep Learning (Modules 7–10)</h2>
<table>
  <tr><th>Item</th><th>The essence</th></tr>
  <tr><td>Layer</td><td>\(\mathbf{a}^{[l]} = g(\mathbf{W}^{[l]}\mathbf{a}^{[l-1]} + \mathbf{b}^{[l]})\); no activation = collapses to linear</td></tr>
  <tr><td>Activations</td><td>hidden: ReLU (GELU in Transformers); output: sigmoid/softmax/linear</td></tr>
  <tr><td>Losses</td><td>CE for classes (feed logits - fused!), MSE/Huber for regression</td></tr>
  <tr><td>Backprop</td><td>δ flows back through Wᵀ, gated by g′; grad(W) = δ · aᵀ</td></tr>
  <tr><td>Init</td><td>He for ReLU (√(2/n)), Xavier for tanh; never constants</td></tr>
  <tr><td>Optimizer</td><td>AdamW lr=1e-3 default; tune lr first, log scale; clip RNN grads</td></tr>
  <tr><td>Regularization</td><td>early stopping always; dropout on dense; BN in CNNs; augment images</td></tr>
  <tr><td>PyTorch liturgy</td><td>zero_grad → forward → loss → backward → step; train()/eval(); no_grad()</td></tr>
  <tr><td>CNN sizing</td><td>O = ⌊(N + 2P − F)/S⌋ + 1; 3×3 "same"; halve space ↔ double channels</td></tr>
  <tr><td>Vanishing fix kit</td><td>ReLU + He + BatchNorm + residual skips (+ gates in time)</td></tr>
  <tr><td>Attention</td><td>softmax(QKᵀ/√d)V - similarity-weighted average; O(T²); needs positions</td></tr>
  <tr><td>Transfer</td><td>pretrain → freeze or fine-tune at tiny lr; LoRA for LLMs; beware forgetting</td></tr>
</table>

<h2>The universal debugging ladder</h2>
<ol>
  <li>Overfit 10 examples to ~zero loss (if you can't, the code is broken).</li>
  <li>Check shapes and one batch's values at every stage.</li>
  <li>Plot train vs validation loss - bias or variance? (This decides everything.)</li>
  <li>Compare against the dumb baseline before celebrating.</li>
  <li>When results look too good - hunt the leak before publishing.</li>
</ol>
`};

CONTENT["practice/glossary"] = {
  html: String.raw`
<h1>Glossary of ML/DL Terms</h1>
<p class="lead">Sixty-odd terms you'll meet in papers, docs, and interviews -
one honest sentence each, with links into the chapters that teach them properly.
Use your browser's find (Ctrl/Cmd-F) to jump.</p>

<h2>A–C</h2>
<table>
  <tr><td><strong>Activation function</strong></td><td>The non-linearity between layers (ReLU, sigmoid…); without it depth collapses to a linear model. <a href="#/dl/activation-functions">→</a></td></tr>
  <tr><td><strong>Attention</strong></td><td>Mechanism that computes similarity-weighted averages over a sequence, letting models focus per-step. <a href="#/sequence/attention">→</a></td></tr>
  <tr><td><strong>AUC</strong></td><td>Area under the ROC curve; probability a random positive outranks a random negative. <a href="#/ml/evaluation-metrics">→</a></td></tr>
  <tr><td><strong>Autoencoder</strong></td><td>Network trained to reconstruct its input through a bottleneck; yields compression and anomaly detection. <a href="#/advanced/autoencoders">→</a></td></tr>
  <tr><td><strong>Autoregression</strong></td><td>Generating a sequence by repeatedly predicting the next element from previous outputs (GPT's loop). <a href="#/sequence/rnn">→</a></td></tr>
  <tr><td><strong>Backpropagation</strong></td><td>The chain rule applied backwards through a network to get every weight's gradient in one sweep. <a href="#/dl/backpropagation">→</a></td></tr>
  <tr><td><strong>Bagging</strong></td><td>Training models on bootstrap samples and averaging - variance reduction (random forests). <a href="#/ml/random-forest">→</a></td></tr>
  <tr><td><strong>Batch normalization</strong></td><td>Re-standardizing activations per mini-batch between layers; stabilizes and accelerates training. <a href="#/dl/dl-regularization">→</a></td></tr>
  <tr><td><strong>Bias (model)</strong></td><td>Systematic error from a model too rigid for the truth - underfitting's cause. <a href="#/ml/bias-variance">→</a></td></tr>
  <tr><td><strong>Bias (neuron)</strong></td><td>The learnable offset b in wx + b, shifting the decision threshold. <a href="#/dl/perceptron">→</a></td></tr>
  <tr><td><strong>Boosting</strong></td><td>Sequentially training models on the ensemble's remaining errors - bias reduction (XGBoost). <a href="#/ml/gradient-boosting">→</a></td></tr>
  <tr><td><strong>Confusion matrix</strong></td><td>Actual × predicted class table; the source of precision, recall, and error analysis. <a href="#/ml/evaluation-metrics">→</a></td></tr>
  <tr><td><strong>Cross-entropy</strong></td><td>Classification loss = −log(probability given to the truth); punishes confident wrongness brutally. <a href="#/dl/loss-functions">→</a></td></tr>
  <tr><td><strong>Cross-validation</strong></td><td>Rotating train/validation folds to get performance estimates with error bars. <a href="#/ml/cross-validation">→</a></td></tr>
  <tr><td><strong>Curse of dimensionality</strong></td><td>In high dimensions data gets sparse and distances uninformative; distance-based methods degrade. <a href="#/ml/knn">→</a></td></tr>
</table>

<h2>D–G</h2>
<table>
  <tr><td><strong>Data leakage</strong></td><td>Information unavailable at prediction time sneaking into training - great validation, useless production. <a href="#/eda/workflow">→</a></td></tr>
  <tr><td><strong>Decision boundary</strong></td><td>The surface where a classifier switches verdicts; its shape reveals the model's character. <a href="#/dataviz/ml-visualizations">→</a></td></tr>
  <tr><td><strong>Dropout</strong></td><td>Randomly silencing units during training; forces redundancy, approximates an ensemble. <a href="#/dl/dl-regularization">→</a></td></tr>
  <tr><td><strong>Early stopping</strong></td><td>Halting training when validation loss stops improving; free regularization. <a href="#/dl/dl-regularization">→</a></td></tr>
  <tr><td><strong>Embedding</strong></td><td>A learned dense vector representing a discrete item (word, user), placing similar items nearby. <a href="#/sequence/why-sequence-models">→</a></td></tr>
  <tr><td><strong>Ensemble</strong></td><td>Combining multiple models' predictions; averaging kills variance, sequencing kills bias. <a href="#/ml/random-forest">→</a></td></tr>
  <tr><td><strong>Epoch</strong></td><td>One full pass through the training set. <a href="#/dl/gradient-descent-variants">→</a></td></tr>
  <tr><td><strong>F1 score</strong></td><td>Harmonic mean of precision and recall - the single-number compromise for imbalanced classes. <a href="#/ml/evaluation-metrics">→</a></td></tr>
  <tr><td><strong>Feature engineering</strong></td><td>Transforming raw columns into signals models can use - encoding, scaling, constructing, selecting. <a href="#/features/encoding">→</a></td></tr>
  <tr><td><strong>Fine-tuning</strong></td><td>Continuing training of a pretrained model on your task at a small learning rate. <a href="#/transformers/transfer-learning">→</a></td></tr>
  <tr><td><strong>GAN</strong></td><td>Generator vs discriminator in a minimax game until fakes are indistinguishable. <a href="#/advanced/gans">→</a></td></tr>
  <tr><td><strong>Gradient</strong></td><td>Vector of all partial derivatives - the local direction of steepest loss increase. <a href="#/math/calculus">→</a></td></tr>
  <tr><td><strong>Gradient clipping</strong></td><td>Capping gradient norms per step to survive explosions (standard in RNNs/Transformers). <a href="#/dl/vanishing-gradients">→</a></td></tr>
  <tr><td><strong>Gradient descent</strong></td><td>Repeatedly stepping weights against the gradient to minimize a loss. <a href="#/ml/linear-regression">→</a></td></tr>
</table>

<h2>H–O</h2>
<table>
  <tr><td><strong>Hidden state</strong></td><td>An RNN's running memory vector, updated each timestep. <a href="#/sequence/rnn">→</a></td></tr>
  <tr><td><strong>Hyperparameter</strong></td><td>A setting chosen before training (k, depth, learning rate) - tuned by search + CV. <a href="#/ml/hyperparameter-tuning">→</a></td></tr>
  <tr><td><strong>Imbalanced data</strong></td><td>Heavily unequal class frequencies; breaks accuracy, demands F1/PR-AUC, weights, resampling. <a href="#/features/imbalanced-data">→</a></td></tr>
  <tr><td><strong>Inference</strong></td><td>Using a trained model to predict (vs training); cheaper - no gradients, no activation cache. <a href="#/dl/forward-propagation">→</a></td></tr>
  <tr><td><strong>KL divergence</strong></td><td>Asymmetric measure of how one distribution differs from another; the VAE's latent regularizer.</td></tr>
  <tr><td><strong>Latent space</strong></td><td>The compressed internal representation space (autoencoder bottleneck, GAN noise input). <a href="#/advanced/autoencoders">→</a></td></tr>
  <tr><td><strong>Learning rate</strong></td><td>Step size α of gradient descent - the single most important hyperparameter. <a href="#/dl/optimizers">→</a></td></tr>
  <tr><td><strong>Logits</strong></td><td>Raw pre-softmax/sigmoid scores; modern losses expect them (fused, stable). <a href="#/dl/loss-functions">→</a></td></tr>
  <tr><td><strong>LoRA</strong></td><td>Fine-tuning via tiny low-rank weight corrections - ~1% of parameters, near-full quality. <a href="#/transformers/transfer-learning">→</a></td></tr>
  <tr><td><strong>LSTM</strong></td><td>Gated RNN with a protected additive cell state; solved vanishing gradients across time. <a href="#/sequence/lstm">→</a></td></tr>
  <tr><td><strong>Mini-batch</strong></td><td>The subset of examples per gradient step (32–512); noise ∝ 1/√B. <a href="#/dl/gradient-descent-variants">→</a></td></tr>
  <tr><td><strong>Multicollinearity</strong></td><td>Features that nearly duplicate each other; destabilizes linear coefficients (check VIF). <a href="#/eda/correlation-analysis">→</a></td></tr>
  <tr><td><strong>One-hot encoding</strong></td><td>One binary column per category - no fake order, at the cost of width. <a href="#/features/encoding">→</a></td></tr>
  <tr><td><strong>Overfitting</strong></td><td>Memorizing training noise; great train scores, poor generalization - the variance disease. <a href="#/ml/bias-variance">→</a></td></tr>
</table>

<h2>P–Z</h2>
<table>
  <tr><td><strong>PCA</strong></td><td>Projection onto the top eigenvectors of the covariance - max-variance linear compression. <a href="#/features/dimensionality-reduction">→</a></td></tr>
  <tr><td><strong>Positional encoding</strong></td><td>Vectors added to embeddings so order-blind attention knows token positions. <a href="#/transformers/positional-encoding">→</a></td></tr>
  <tr><td><strong>Precision / Recall</strong></td><td>Of the flagged, how many real? / Of the real, how many caught? The imbalance duo. <a href="#/ml/evaluation-metrics">→</a></td></tr>
  <tr><td><strong>Pretraining</strong></td><td>Self-supervised training on massive unlabeled data before task adaptation. <a href="#/transformers/bert-gpt">→</a></td></tr>
  <tr><td><strong>p-value</strong></td><td>P(data at least this extreme | null hypothesis true). Not the probability the null is true. <a href="#/math/hypothesis-testing">→</a></td></tr>
  <tr><td><strong>Receptive field</strong></td><td>The input region influencing one unit's output; grows with depth in CNNs. <a href="#/cnn/why-cnns">→</a></td></tr>
  <tr><td><strong>Regularization</strong></td><td>Any pressure toward simpler models: L1/L2, dropout, early stopping, augmentation. <a href="#/ml/regularization">→</a></td></tr>
  <tr><td><strong>Residual connection</strong></td><td>y = F(x) + x - the skip that lets gradients through any depth (ResNet, Transformers). <a href="#/cnn/classic-architectures">→</a></td></tr>
  <tr><td><strong>RLHF</strong></td><td>Reinforcement learning from human feedback - aligning language models with preferences. <a href="#/transformers/bert-gpt">→</a></td></tr>
  <tr><td><strong>Self-attention</strong></td><td>A sequence attending to itself via Q/K/V projections - the Transformer's engine. <a href="#/transformers/self-attention">→</a></td></tr>
  <tr><td><strong>Self-supervised learning</strong></td><td>Manufacturing labels from raw data (mask a word, predict it) - pretraining's fuel. <a href="#/transformers/bert-gpt">→</a></td></tr>
  <tr><td><strong>SMOTE</strong></td><td>Synthesizing minority-class points by interpolating neighbors - in-pipeline only! <a href="#/features/imbalanced-data">→</a></td></tr>
  <tr><td><strong>Softmax</strong></td><td>Exponentiate-and-normalize: scores → a probability distribution. <a href="#/dl/activation-functions">→</a></td></tr>
  <tr><td><strong>Teacher forcing</strong></td><td>Training decoders on ground-truth previous tokens instead of their own guesses. <a href="#/sequence/seq2seq">→</a></td></tr>
  <tr><td><strong>Tokenization</strong></td><td>Splitting text into model units (words/subwords) mapped to ids - the LLM's first step.</td></tr>
  <tr><td><strong>Transfer learning</strong></td><td>Reusing knowledge from one task/model as the starting point for another. <a href="#/advanced/cnn-transfer-learning">→</a></td></tr>
  <tr><td><strong>t-SNE / UMAP</strong></td><td>Non-linear 2-D maps preserving neighborhoods - for eyes, not features. <a href="#/features/dimensionality-reduction">→</a></td></tr>
  <tr><td><strong>Underfitting</strong></td><td>Model too simple for the signal - poor scores everywhere; the bias disease. <a href="#/ml/bias-variance">→</a></td></tr>
  <tr><td><strong>Vanishing gradients</strong></td><td>Error signals decaying exponentially through depth/time; fixed by ReLU, init, norm, skips, gates. <a href="#/dl/vanishing-gradients">→</a></td></tr>
  <tr><td><strong>Variance (model)</strong></td><td>Sensitivity of the learned model to which sample it saw - overfitting's cause. <a href="#/ml/bias-variance">→</a></td></tr>
  <tr><td><strong>Weight decay</strong></td><td>L2-style shrinkage of weights each step; AdamW applies it decoupled and correctly. <a href="#/dl/optimizers">→</a></td></tr>
  <tr><td><strong>z-score</strong></td><td>(x − μ)/σ - "how many standard deviations from the mean"; the standardization unit. <a href="#/math/distributions">→</a></td></tr>
</table>
`};

CONTENT["practice/quizzes"] = {
  html: String.raw`
<h1>Quiz per Module</h1>
<p class="lead">One synthesis question per module - deliberately cross-cutting,
harder than the per-chapter quizzes. Score honestly; every miss links back to
the chapter to reread. (Chapter-level quizzes live at the bottom of each topic
page.)</p>

<h2>The gauntlet: 12 questions, one per module</h2>
<div class="quiz">
  <div class="quiz-q" data-answer="2">
    <p class="quiz-question">M1 · A medical test is 95% accurate; the disease hits 1 in 500. A patient tests positive. The posterior probability is closest to…</p>
    <button class="quiz-opt">95%</button>
    <button class="quiz-opt">50%</button>
    <button class="quiz-opt">4%</button>
    <div class="quiz-explain hidden">Per 10,000 people: ~20 sick → 19 true positives; ~9,980 healthy → ~499 false positives. 19/(19+499) ≈ 3.7%. Base rates rule. Reread <a href="#/math/probability">Probability Basics</a>.</div>
  </div>
  <div class="quiz-q" data-answer="1">
    <p class="quiz-question">M2 · X.shape is (1000, 8). X.mean(axis=0).shape and (X - X.mean(axis=0)).shape are…</p>
    <button class="quiz-opt">(1000,) and an error</button>
    <button class="quiz-opt">(8,) and (1000, 8) - broadcasting stretches the means across rows</button>
    <button class="quiz-opt">(8,) and (8, 8)</button>
    <div class="quiz-explain hidden">axis=0 collapses rows → one mean per column; subtraction broadcasts (8,) across all 1000 rows. This is per-feature centering in one line. <a href="#/python/numpy">NumPy Essentials</a>.</div>
  </div>
  <div class="quiz-q" data-answer="0">
    <p class="quiz-question">M3 · A colleague's bar chart of model accuracies starts its y-axis at 88%. Models score 89% and 91%. The chart's crime?</p>
    <button class="quiz-opt">The 91% bar looks ~3× taller for a 2-point difference - truncated baselines break the length encoding</button>
    <button class="quiz-opt">Bar charts can't show accuracies</button>
    <button class="quiz-opt">The colors are wrong</button>
    <div class="quiz-explain hidden">Bars encode value as length from zero; truncation turns 2 points into a visual landslide. Zoom with dots/lines, never bars. <a href="#/dataviz/why-visualization">Why Visualization Matters</a>.</div>
  </div>
  <div class="quiz-q" data-answer="2">
    <p class="quiz-question">M4 · During EDA you find a feature correlating 0.997 with the churn target. Your first hypothesis?</p>
    <button class="quiz-opt">You've found the perfect predictor - ship it</button>
    <button class="quiz-opt">The feature needs scaling</button>
    <button class="quiz-opt">Leakage - it probably encodes the outcome (e.g. computed after churn happened)</button>
    <div class="quiz-explain hidden">Real-world signals are never that clean; leaks always are. Verify the feature exists at prediction time before anything else. <a href="#/eda/workflow">EDA Workflow</a>.</div>
  </div>
  <div class="quiz-q" data-answer="1">
    <p class="quiz-question">M5 · A pipeline scales features, then SMOTE-oversamples, then cross-validates - all correctly inside the CV folds. Which step must additionally NEVER touch the test set?</p>
    <button class="quiz-opt">Scaling - the test set should be scaled with its own statistics</button>
    <button class="quiz-opt">SMOTE - the test set must keep the real class ratio (scaling DOES apply to test, using train-fitted parameters)</button>
    <button class="quiz-opt">Neither may be applied to test data in any form</button>
    <div class="quiz-explain hidden">Transformations (scaling) are applied to test using train-fitted parameters; resampling changes the data distribution and belongs to training only. <a href="#/features/imbalanced-data">Imbalanced Data</a>.</div>
  </div>
  <div class="quiz-q" data-answer="0">
    <p class="quiz-question">M6 · Train accuracy 0.99, CV accuracy 0.78, with a gradient-boosted model of depth 12, 3000 trees, no early stopping. Two changes with the best expected payoff?</p>
    <button class="quiz-opt">Early stopping on a validation set + shallower trees (depth 3–6)</button>
    <button class="quiz-opt">More trees + higher learning rate</button>
    <button class="quiz-opt">Remove cross-validation + train longer</button>
    <div class="quiz-explain hidden">A 21-point gap is textbook variance: cap the sequential noise-chasing (early stopping) and per-tree complexity. <a href="#/ml/gradient-boosting">Gradient Boosting</a>, <a href="#/ml/bias-variance">Bias-Variance</a>.</div>
  </div>
  <div class="quiz-q" data-answer="2">
    <p class="quiz-question">M7 · A 40-layer sigmoid MLP trains no better than a 3-layer one; early-layer weights barely move. The modern four-part fix kit?</p>
    <button class="quiz-opt">Bigger batches, more epochs, more data, wider layers</button>
    <button class="quiz-opt">Lower learning rate, more dropout, smaller init, MSE loss</button>
    <button class="quiz-opt">ReLU activations, He init, batch norm, residual connections</button>
    <div class="quiz-explain hidden">Vanishing gradients: every fix attacks a factor in the backprop product - slopes of 1, unit-variance init, per-layer renormalization, additive skip paths. <a href="#/dl/vanishing-gradients">Vanishing Gradients</a>.</div>
  </div>
  <div class="quiz-q" data-answer="1">
    <p class="quiz-question">M8 · Conv2d(64, 128, kernel_size=3, stride=2, padding=1) applied to a 16×16×64 tensor yields…</p>
    <button class="quiz-opt">16×16×128</button>
    <button class="quiz-opt">8×8×128</button>
    <button class="quiz-opt">7×7×128</button>
    <div class="quiz-explain hidden">O = ⌊(16 + 2 − 3)/2⌋ + 1 = 8; channels = number of filters = 128. Halve space, double channels. <a href="#/cnn/filters-padding-stride">Filters, Padding, Stride</a>.</div>
  </div>
  <div class="quiz-q" data-answer="0">
    <p class="quiz-question">M9 · LSTMs beat vanilla RNNs on long sequences because their cell state is updated by…</p>
    <button class="quiz-opt">Gated element-wise erase-and-add - no repeated matrix multiplication for gradients to die in</button>
    <button class="quiz-opt">A much higher learning rate</button>
    <button class="quiz-opt">Processing the sequence in both directions</button>
    <div class="quiz-explain hidden">The conveyor-belt Jacobian is diag(fₜ), not W_hh^T - with gates near 1, gradients cross 100 steps intact. Same additive-highway idea as ResNet. <a href="#/sequence/lstm">LSTM</a>.</div>
  </div>
  <div class="quiz-q" data-answer="1">
    <p class="quiz-question">M10 · In softmax(QKᵀ/√d)V, removing the √d scaling would…</p>
    <button class="quiz-opt">Change nothing - it's cosmetic</button>
    <button class="quiz-opt">Let score variance grow with dimension, saturating the softmax and killing its gradients</button>
    <button class="quiz-opt">Make attention twice as fast</button>
    <div class="quiz-explain hidden">Dot products of d-dim vectors have variance ∝ d; unscaled, one weight ≈ 1 and the rest ≈ 0, with near-zero gradients. <a href="#/transformers/self-attention">Self-Attention</a>.</div>
  </div>
  <div class="quiz-q" data-answer="2">
    <p class="quiz-question">M11 · You must detect novel fraud patterns with zero labeled fraud examples. The architecture-shaped answer is…</p>
    <button class="quiz-opt">Fine-tune BERT</button>
    <button class="quiz-opt">A GAN generating fraud</button>
    <button class="quiz-opt">An autoencoder trained on normal transactions; flag high reconstruction error</button>
    <div class="quiz-explain hidden">One-class learning: model normality, let abnormality expose itself. No labels needed - the escape hatch from impossible imbalance. <a href="#/advanced/autoencoders">Autoencoders</a>.</div>
  </div>
  <div class="quiz-q" data-answer="0">
    <p class="quiz-question">M12 · Your deployed model's average predicted probability shifted from 0.21 to 0.38 this week; true labels won't arrive for a month. You conclude…</p>
    <button class="quiz-opt">Something changed - input drift or a pipeline bug; investigate now, don't wait for labels</button>
    <button class="quiz-opt">Nothing can be known without labels</button>
    <button class="quiz-opt">The model improved</button>
    <div class="quiz-explain hidden">Prediction-distribution monitoring is the early smoke alarm precisely because it needs no labels. <a href="#/advanced/deployment">Deployment Basics</a>.</div>
  </div>
</div>

<h2>How to use your score</h2>
<table>
  <tr><th>Score</th><th>Reading</th></tr>
  <tr><td>10–12</td><td>You've genuinely integrated the course. Build projects; the <a href="#/practice/cheat-sheets">cheat sheets</a> are now your working reference.</td></tr>
  <tr><td>7–9</td><td>Solid. Each miss names its chapter - reread those, redo their quizzes.</td></tr>
  <tr><td>≤ 6</td><td>Normal after one pass! Revisit modules in order; the second pass is where it clicks.</td></tr>
</table>
<p>And a graduation note: everything here was designed to be <em>built upon</em>.
The from-scratch network you wrote in Module 7, the CNN you trained in
Module 8, the attention you implemented in Module 9 - those are the actual
foundations of the systems making headlines. You didn't just read about them.
You built them. 🎓</p>
`};
