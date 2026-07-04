/* Curated external resources per topic: research papers, videos, books,
   and interactive tools. Rendered automatically by app.js as a
   "Go deeper" section at the end of each topic page.
   Format: [type, title, source/author, url]
   Types: "paper" | "video" | "read" | "interactive" */

var RESOURCES = {

  /* ---------- Module 0 ---------- */
  "getting-started/how-to-use": [
    ["read", "Machine Learning Crash Course", "Google, free course with exercises", "https://developers.google.com/machine-learning/crash-course"],
    ["read", "Kaggle Learn", "free hands-on micro-courses with in-browser notebooks", "https://www.kaggle.com/learn"],
    ["video", "The Complete Mathematics of Neural Networks and Deep Learning", "a taste of where this course leads", "https://www.youtube.com/watch?v=Ixl3nykKG9M"]
  ],
  "getting-started/roadmap": [
    ["read", "Dive into Deep Learning (D2L)", "free interactive book, code in PyTorch", "https://d2l.ai/"],
    ["read", "An Introduction to Statistical Learning (ISLR)", "James, Witten, Hastie, Tibshirani, free PDF", "https://www.statlearning.com/"],
    ["read", "Deep Learning", "Goodfellow, Bengio, Courville, free online", "https://www.deeplearningbook.org/"]
  ],

  /* ---------- Module 1: Math & Stats ---------- */
  "math/linear-algebra": [
    ["video", "Essence of Linear Algebra", "3Blue1Brown, the definitive visual series", "https://www.3blue1brown.com/topics/linear-algebra"],
    ["read", "MIT 18.06 Linear Algebra", "Gilbert Strang, full OpenCourseWare course", "https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/"],
    ["read", "Linear Algebra chapter, Deep Learning book", "Goodfellow et al., the ML-focused summary", "https://www.deeplearningbook.org/contents/linear_algebra.html"]
  ],
  "math/calculus": [
    ["video", "Essence of Calculus", "3Blue1Brown video series", "https://www.3blue1brown.com/topics/calculus"],
    ["video", "Gradient descent, how neural networks learn", "3Blue1Brown", "https://www.youtube.com/watch?v=IHZwWFHWa-w"],
    ["read", "Khan Academy: Multivariable Calculus", "partial derivatives and gradients, free", "https://www.khanacademy.org/math/multivariable-calculus"]
  ],
  "math/probability": [
    ["interactive", "Seeing Theory", "Brown University, beautiful interactive probability", "https://seeing-theory.brown.edu/"],
    ["video", "Statistics 110: Probability", "Joe Blitzstein, Harvard, full lecture series", "https://projects.iq.harvard.edu/stat110/youtube"],
    ["video", "Bayes theorem, the geometry of changing beliefs", "3Blue1Brown", "https://www.youtube.com/watch?v=HZGCoVF3YvM"]
  ],
  "math/descriptive-statistics": [
    ["video", "StatQuest: Statistics Fundamentals playlist", "Josh Starmer, gentle and precise", "https://www.youtube.com/playlist?list=PLblh5JKOoLUK0FLuzwntyYI10UQFUhsY9"],
    ["read", "Khan Academy: Statistics and Probability", "free full course", "https://www.khanacademy.org/math/statistics-probability"],
    ["interactive", "Seeing Theory: Basic Probability", "interactive mean/variance visuals", "https://seeing-theory.brown.edu/basic-probability/index.html"]
  ],
  "math/distributions": [
    ["video", "But what is the Central Limit Theorem?", "3Blue1Brown", "https://www.youtube.com/watch?v=zeJD6dqJ5lo"],
    ["interactive", "Seeing Theory: Probability Distributions", "interactive distribution explorer", "https://seeing-theory.brown.edu/probability-distributions/index.html"],
    ["video", "StatQuest: The Normal Distribution, Clearly Explained", "Josh Starmer", "https://www.youtube.com/watch?v=rzFX5NWojp0"]
  ],
  "math/hypothesis-testing": [
    ["video", "StatQuest: Hypothesis Testing and The Null Hypothesis", "Josh Starmer", "https://www.youtube.com/watch?v=0oc49DyA3hU"],
    ["paper", "The ASA Statement on p-Values", "Wasserstein & Lazar, 2016, what p-values do and do not mean", "https://www.tandfonline.com/doi/full/10.1080/00031305.2016.1154108"],
    ["interactive", "Seeing Theory: Frequentist Inference", "interactive testing and intervals", "https://seeing-theory.brown.edu/frequentist-inference/index.html"]
  ],
  "math/confidence-intervals": [
    ["video", "StatQuest: Confidence Intervals, Clearly Explained", "Josh Starmer", "https://www.youtube.com/watch?v=TqOeMYtOc1w"],
    ["interactive", "Seeing Theory: Confidence Intervals", "watch the coverage simulation live", "https://seeing-theory.brown.edu/frequentist-inference/index.html"],
    ["read", "Bootstrap confidence intervals", "scikit-learn adjacent, Efron's method explained simply", "https://www.textbook.ds100.org/ch/18/hyp_studentized.html"]
  ],
  "math/correlation-causation": [
    ["interactive", "Spurious Correlations", "Tyler Vigen, absurd real correlations gallery", "https://www.tylervigen.com/spurious-correlations"],
    ["read", "The Book of Why", "Judea Pearl, the modern causality bible (book site)", "http://bayes.cs.ucla.edu/WHY/"],
    ["interactive", "Guess the Correlation", "train your eye for r on real scatter plots", "https://www.guessthecorrelation.com/"]
  ],

  /* ---------- Module 2: Python ---------- */
  "python/numpy": [
    ["read", "NumPy Quickstart", "official tutorial", "https://numpy.org/doc/stable/user/quickstart.html"],
    ["read", "CS231n Python/NumPy Tutorial", "Stanford, the ML-oriented crash course", "https://cs231n.github.io/python-numpy-tutorial/"],
    ["read", "From Python to NumPy", "Nicolas Rougier, free book on vectorized thinking", "https://www.labri.fr/perso/nrougier/from-python-to-numpy/"]
  ],
  "python/pandas": [
    ["read", "10 Minutes to Pandas", "official quickstart", "https://pandas.pydata.org/docs/user_guide/10min.html"],
    ["read", "Kaggle Learn: Pandas", "free hands-on exercises", "https://www.kaggle.com/learn/pandas"],
    ["read", "Python Data Science Handbook, ch. 3", "Jake VanderPlas, free online", "https://jakevdp.github.io/PythonDataScienceHandbook/03.00-introduction-to-pandas.html"]
  ],
  "python/data-cleaning": [
    ["paper", "Tidy Data", "Hadley Wickham, Journal of Statistical Software 2014, the classic on data shape", "https://vita.had.co.nz/papers/tidy-data.pdf"],
    ["read", "Kaggle Learn: Data Cleaning", "free course with messy real datasets", "https://www.kaggle.com/learn/data-cleaning"],
    ["read", "Working with missing data", "official pandas guide", "https://pandas.pydata.org/docs/user_guide/missing_data.html"]
  ],

  /* ---------- Module 3: Visualization ---------- */
  "dataviz/why-visualization": [
    ["paper", "Graphs in Statistical Analysis", "Anscombe, 1973, the quartet's original paper", "https://www.jstor.org/stable/2682899"],
    ["paper", "Same Stats, Different Graphs (Datasaurus Dozen)", "Matejka & Fitzmaurice, 2017", "https://www.research.autodesk.com/publications/same-stats-different-graphs/"],
    ["read", "The Visual Display of Quantitative Information", "Edward Tufte, the field's classic book", "https://www.edwardtufte.com/tufte/books_vdqi"]
  ],
  "dataviz/matplotlib": [
    ["read", "Matplotlib official tutorials", "start with 'The Lifecycle of a Plot'", "https://matplotlib.org/stable/tutorials/index.html"],
    ["read", "Scientific Visualization: Python + Matplotlib", "Nicolas Rougier, free book", "https://github.com/rougier/scientific-visualization-book"]
  ],
  "dataviz/seaborn": [
    ["read", "Seaborn official tutorial", "the gallery + tutorial are excellent", "https://seaborn.pydata.org/tutorial.html"],
    ["read", "Python Data Science Handbook, ch. 4", "Jake VanderPlas, visualization chapter", "https://jakevdp.github.io/PythonDataScienceHandbook/04.00-introduction-to-matplotlib.html"]
  ],
  "dataviz/plotly": [
    ["read", "Plotly Python documentation", "official, example-driven", "https://plotly.com/python/"],
    ["read", "Plotly Express overview", "the high-level API in one page", "https://plotly.com/python/plotly-express/"]
  ],
  "dataviz/chart-picker": [
    ["paper", "Graphical Perception", "Cleveland & McGill, 1984, the encoding-accuracy experiments", "https://www.jstor.org/stable/2288400"],
    ["interactive", "From Data to Viz", "decision tree from your data type to the right chart", "https://www.data-to-viz.com/"],
    ["read", "Visual Vocabulary", "Financial Times chart-choosing poster", "https://github.com/Financial-Times/chart-doctor/tree/main/visual-vocabulary"]
  ],
  "dataviz/ml-visualizations": [
    ["interactive", "A Visual Introduction to Machine Learning", "R2D3, scrollytelling masterpiece", "http://www.r2d3.us/visual-intro-to-machine-learning-part-1/"],
    ["read", "scikit-learn: Visualizations", "official plotting API for ROC, confusion matrices, boundaries", "https://scikit-learn.org/stable/visualizations.html"]
  ],
  "dataviz/paper-visualizations": [
    ["paper", "Ten Simple Rules for Better Figures", "Rougier, Droettboom, Bourne, PLOS Comp Bio 2014", "https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003833"],
    ["read", "Distill.pub", "the gold standard of interactive ML exposition", "https://distill.pub/"]
  ],

  /* ---------- Module 4: EDA ---------- */
  "eda/workflow": [
    ["read", "R for Data Science: EDA chapter", "Wickham & Grolemund, language-agnostic wisdom", "https://r4ds.hadley.nz/eda"],
    ["read", "Kaggle Learn: Intro to Machine Learning", "EDA-to-model workflow with exercises", "https://www.kaggle.com/learn/intro-to-machine-learning"]
  ],
  "eda/missing-values": [
    ["paper", "Inference and Missing Data", "Rubin, Biometrika 1976, where MCAR/MAR/MNAR were born", "https://doi.org/10.1093/biomet/63.3.581"],
    ["read", "Flexible Imputation of Missing Data", "Stef van Buuren, free online book", "https://stefvanbuuren.name/fimd/"],
    ["read", "scikit-learn: Imputation of missing values", "official guide to Simple/KNN/Iterative imputers", "https://scikit-learn.org/stable/modules/impute.html"]
  ],
  "eda/outliers": [
    ["paper", "Isolation Forest", "Liu, Ting, Zhou, ICDM 2008", "https://doi.org/10.1109/ICDM.2008.17"],
    ["read", "scikit-learn: Outlier detection", "official comparison of methods", "https://scikit-learn.org/stable/modules/outlier_detection.html"],
    ["read", "PyOD documentation", "the dedicated outlier-detection library, 40+ algorithms", "https://pyod.readthedocs.io/"]
  ],
  "eda/uni-bi-multivariate": [
    ["read", "Penn State STAT 500 online notes", "applied statistics with worked examples", "https://online.stat.psu.edu/stat500/"],
    ["read", "Seaborn tutorial: Visualizing distributions", "the practical toolkit for levels 1 and 2", "https://seaborn.pydata.org/tutorial/distributions.html"]
  ],
  "eda/correlation-analysis": [
    ["interactive", "Guess the Correlation", "calibrate your intuition for r", "https://www.guessthecorrelation.com/"],
    ["read", "statsmodels: Variance Inflation Factor", "VIF documentation with examples", "https://www.statsmodels.org/stable/generated/statsmodels.stats.outliers_influence.variance_inflation_factor.html"]
  ],
  "eda/case-study": [
    ["interactive", "Kaggle: Titanic competition", "run your own EDA and submit predictions", "https://www.kaggle.com/c/titanic"],
    ["read", "Kaggle: Titanic Tutorial notebook", "Alexis Cook, the standard walkthrough", "https://www.kaggle.com/code/alexisbcook/titanic-tutorial"]
  ],

  /* ---------- Module 5: Feature Engineering ---------- */
  "features/encoding": [
    ["read", "scikit-learn: Preprocessing categorical features", "official guide", "https://scikit-learn.org/stable/modules/preprocessing.html#encoding-categorical-features"],
    ["read", "Category Encoders documentation", "target/ordinal/hashing encoders, ready-made", "https://contrib.scikit-learn.org/category_encoders/"],
    ["read", "Kaggle Learn: Data Leakage", "the leakage lesson every target-encoder needs", "https://www.kaggle.com/code/alexisbcook/data-leakage"]
  ],
  "features/scaling": [
    ["read", "Compare the effect of different scalers", "scikit-learn example, scalers vs outliers visually", "https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html"],
    ["read", "Google ML Crash Course: Normalization", "short and practical", "https://developers.google.com/machine-learning/data-prep/transform/normalization"]
  ],
  "features/imbalanced-data": [
    ["paper", "SMOTE: Synthetic Minority Over-sampling Technique", "Chawla et al., JAIR 2002", "https://arxiv.org/abs/1106.1813"],
    ["read", "imbalanced-learn documentation", "SMOTE variants, pipelines, and pitfalls", "https://imbalanced-learn.org/stable/"],
    ["video", "StatQuest: ROC and AUC, Clearly Explained", "the metrics side of imbalance", "https://www.youtube.com/watch?v=4jRBRDbJemM"]
  ],
  "features/feature-selection": [
    ["paper", "An Introduction to Variable and Feature Selection", "Guyon & Elisseeff, JMLR 2003, the classic survey", "https://www.jmlr.org/papers/v3/guyon03a.html"],
    ["read", "scikit-learn: Feature selection", "official guide to filters, RFE, SelectFromModel", "https://scikit-learn.org/stable/modules/feature_selection.html"]
  ],
  "features/dimensionality-reduction": [
    ["paper", "Visualizing Data using t-SNE", "van der Maaten & Hinton, JMLR 2008", "https://www.jmlr.org/papers/v9/vandermaaten08a.html"],
    ["paper", "UMAP: Uniform Manifold Approximation and Projection", "McInnes, Healy, Melville, 2018", "https://arxiv.org/abs/1802.03426"],
    ["interactive", "How to Use t-SNE Effectively", "Distill.pub, play with perplexity live", "https://distill.pub/2016/misread-tsne/"],
    ["video", "StatQuest: PCA main ideas", "Josh Starmer", "https://www.youtube.com/watch?v=FgakZw6K1QQ"]
  ],

  /* ---------- Module 6: ML Algorithms ---------- */
  "ml/learning-paradigms": [
    ["read", "Reinforcement Learning: An Introduction", "Sutton & Barto, the RL bible, free PDF", "http://incompleteideas.net/book/the-book.html"],
    ["read", "Google ML Crash Course", "supervised learning foundations, free", "https://developers.google.com/machine-learning/crash-course"]
  ],
  "ml/linear-regression": [
    ["read", "ISLR, Chapter 3: Linear Regression", "the definitive textbook treatment, free PDF", "https://www.statlearning.com/"],
    ["video", "StatQuest: Linear Regression, Clearly Explained", "Josh Starmer", "https://www.youtube.com/watch?v=7ArmBVF2dCs"]
  ],
  "ml/logistic-regression": [
    ["video", "StatQuest: Logistic Regression", "Josh Starmer", "https://www.youtube.com/watch?v=yIYKR4sgzI8"],
    ["read", "ISLR, Chapter 4: Classification", "free PDF", "https://www.statlearning.com/"],
    ["read", "scikit-learn: Logistic regression", "official user guide", "https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression"]
  ],
  "ml/knn": [
    ["video", "StatQuest: K-nearest neighbors, Clearly Explained", "Josh Starmer", "https://www.youtube.com/watch?v=HVXime0nQeI"],
    ["read", "ISLR, Section 2.2.3 + Chapter 4", "KNN and the bias-variance view, free PDF", "https://www.statlearning.com/"]
  ],
  "ml/decision-trees": [
    ["video", "StatQuest: Decision and Classification Trees", "Josh Starmer", "https://www.youtube.com/watch?v=_L39rN6gz7Y"],
    ["interactive", "A Visual Introduction to Machine Learning", "R2D3, decision trees explained by scrolling", "http://www.r2d3.us/visual-intro-to-machine-learning-part-1/"],
    ["read", "ISLR, Chapter 8: Tree-Based Methods", "free PDF", "https://www.statlearning.com/"]
  ],
  "ml/random-forest": [
    ["paper", "Random Forests", "Leo Breiman, Machine Learning 2001, the original", "https://doi.org/10.1023/A:1010933404324"],
    ["video", "StatQuest: Random Forests Part 1", "Josh Starmer", "https://www.youtube.com/watch?v=J4Wdy0Wc_xQ"]
  ],
  "ml/gradient-boosting": [
    ["paper", "XGBoost: A Scalable Tree Boosting System", "Chen & Guestrin, KDD 2016", "https://arxiv.org/abs/1603.02754"],
    ["paper", "Greedy Function Approximation: A Gradient Boosting Machine", "Friedman, Annals of Statistics 2001", "https://doi.org/10.1214/aos/1013203451"],
    ["video", "StatQuest: Gradient Boost Part 1", "Josh Starmer, 4-part series", "https://www.youtube.com/watch?v=3CC4N4z3GJc"]
  ],
  "ml/svm": [
    ["paper", "Support-Vector Networks", "Cortes & Vapnik, Machine Learning 1995", "https://doi.org/10.1007/BF00994018"],
    ["video", "StatQuest: Support Vector Machines", "Josh Starmer", "https://www.youtube.com/watch?v=efR1C6CvhmE"],
    ["video", "MIT 6.034: Support Vector Machines", "Patrick Winston, a legendary lecture", "https://www.youtube.com/watch?v=_PwhiWxHK8o"]
  ],
  "ml/naive-bayes": [
    ["video", "StatQuest: Naive Bayes, Clearly Explained", "Josh Starmer", "https://www.youtube.com/watch?v=O2L2Uv9pdDA"],
    ["read", "scikit-learn: Naive Bayes", "official guide to the three variants", "https://scikit-learn.org/stable/modules/naive_bayes.html"]
  ],
  "ml/kmeans": [
    ["interactive", "Visualizing K-Means Clustering", "Naftali Harris, step through the algorithm live", "https://www.naftaliharris.com/blog/visualizing-k-means-clustering/"],
    ["video", "StatQuest: K-means clustering", "Josh Starmer", "https://www.youtube.com/watch?v=4b5d3muPQmA"],
    ["paper", "k-means++: The Advantages of Careful Seeding", "Arthur & Vassilvitskii, SODA 2007", "https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf"]
  ],
  "ml/hierarchical-clustering": [
    ["video", "StatQuest: Hierarchical Clustering", "Josh Starmer", "https://www.youtube.com/watch?v=7xHsRkOdVwo"],
    ["read", "SciPy: Hierarchical clustering reference", "linkage methods and dendrograms", "https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html"]
  ],
  "ml/dbscan": [
    ["paper", "A Density-Based Algorithm for Discovering Clusters (DBSCAN)", "Ester, Kriegel, Sander, Xu, KDD 1996", "https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf"],
    ["interactive", "Visualizing DBSCAN Clustering", "Naftali Harris, interactive eps playground", "https://www.naftaliharris.com/blog/visualizing-dbscan-clustering/"],
    ["read", "HDBSCAN documentation", "the modern varying-density successor", "https://hdbscan.readthedocs.io/"]
  ],
  "ml/evaluation-metrics": [
    ["video", "StatQuest: ROC and AUC, Clearly Explained", "Josh Starmer", "https://www.youtube.com/watch?v=4jRBRDbJemM"],
    ["read", "scikit-learn: Model evaluation", "the full metrics reference", "https://scikit-learn.org/stable/modules/model_evaluation.html"],
    ["read", "Google ML Crash Course: Classification metrics", "precision/recall with interactive widgets", "https://developers.google.com/machine-learning/crash-course/classification"]
  ],
  "ml/cross-validation": [
    ["video", "StatQuest: Cross Validation", "Josh Starmer", "https://www.youtube.com/watch?v=fSytzGwwBVw"],
    ["read", "scikit-learn: Cross-validation", "official guide including time-series and group splits", "https://scikit-learn.org/stable/modules/cross_validation.html"]
  ],
  "ml/bias-variance": [
    ["read", "Understanding the Bias-Variance Tradeoff", "Scott Fortmann-Roe, the classic essay", "https://scott.fortmann-roe.com/docs/BiasVariance.html"],
    ["video", "StatQuest: Bias and Variance", "Josh Starmer", "https://www.youtube.com/watch?v=EuBBz3bI-aA"],
    ["paper", "Reconciling modern machine-learning practice and the bias-variance trade-off", "Belkin et al., 2019, double descent for the curious", "https://arxiv.org/abs/1812.11118"]
  ],
  "ml/regularization": [
    ["paper", "Regression Shrinkage and Selection via the Lasso", "Tibshirani, 1996", "https://doi.org/10.1111/j.2517-6161.1996.tb02080.x"],
    ["video", "StatQuest: Ridge Regression", "Josh Starmer, followed by the Lasso video", "https://www.youtube.com/watch?v=Q81RR3yKn30"],
    ["read", "ISLR, Chapter 6: Linear Model Selection and Regularization", "free PDF", "https://www.statlearning.com/"]
  ],
  "ml/hyperparameter-tuning": [
    ["paper", "Random Search for Hyper-Parameter Optimization", "Bergstra & Bengio, JMLR 2012", "https://www.jmlr.org/papers/v13/bergstra12a.html"],
    ["read", "Optuna documentation", "modern Bayesian optimization in practice", "https://optuna.org/"],
    ["read", "scikit-learn: Tuning the hyper-parameters", "GridSearchCV and RandomizedSearchCV guide", "https://scikit-learn.org/stable/modules/grid_search.html"]
  ],

  /* ---------- Module 7: Deep Learning ---------- */
  "dl/perceptron": [
    ["paper", "The Perceptron: A Probabilistic Model", "Rosenblatt, Psychological Review 1958, where it all began", "https://doi.org/10.1037/h0042519"],
    ["video", "But what is a neural network?", "3Blue1Brown, chapter 1", "https://www.youtube.com/watch?v=aircAruvnKk"]
  ],
  "dl/mlp": [
    ["interactive", "TensorFlow Playground", "train an MLP in your browser, watch XOR fall", "https://playground.tensorflow.org/"],
    ["video", "But what is a neural network?", "3Blue1Brown", "https://www.youtube.com/watch?v=aircAruvnKk"],
    ["paper", "Approximation by Superpositions of a Sigmoidal Function", "Cybenko, 1989, the universal approximation theorem", "https://doi.org/10.1007/BF02551274"]
  ],
  "dl/activation-functions": [
    ["paper", "Gaussian Error Linear Units (GELUs)", "Hendrycks & Gimpel, 2016", "https://arxiv.org/abs/1606.08415"],
    ["paper", "Deep Sparse Rectifier Neural Networks", "Glorot, Bordes, Bengio, 2011, the ReLU case", "https://proceedings.mlr.press/v15/glorot11a.html"],
    ["read", "Deep Learning book, Chapter 6", "Goodfellow et al., free online", "https://www.deeplearningbook.org/contents/mlp.html"]
  ],
  "dl/loss-functions": [
    ["paper", "Focal Loss for Dense Object Detection", "Lin et al., ICCV 2017", "https://arxiv.org/abs/1708.02002"],
    ["read", "PyTorch: Loss functions reference", "every loss with its expected inputs", "https://pytorch.org/docs/stable/nn.html#loss-functions"]
  ],
  "dl/forward-propagation": [
    ["read", "CS231n: Neural Networks Part 1", "Stanford notes on architecture and forward pass", "https://cs231n.github.io/neural-networks-1/"],
    ["video", "The spelled-out intro to neural networks (micrograd)", "Andrej Karpathy, build autograd from zero", "https://www.youtube.com/watch?v=VMj-3S1tku0"]
  ],
  "dl/backpropagation": [
    ["paper", "Learning representations by back-propagating errors", "Rumelhart, Hinton, Williams, Nature 1986", "https://doi.org/10.1038/323533a0"],
    ["video", "What is backpropagation really doing?", "3Blue1Brown, chapters 3 and 4", "https://www.youtube.com/watch?v=Ilg3gGewQ5U"],
    ["video", "The spelled-out intro to neural networks (micrograd)", "Andrej Karpathy, backprop by hand then in code", "https://www.youtube.com/watch?v=VMj-3S1tku0"],
    ["read", "CS231n: Backpropagation intuitions", "Stanford notes", "https://cs231n.github.io/optimization-2/"]
  ],
  "dl/gradient-descent-variants": [
    ["read", "An overview of gradient descent optimization algorithms", "Sebastian Ruder, the canonical survey", "https://www.ruder.io/optimizing-gradient-descent/"],
    ["video", "Gradient descent, how neural networks learn", "3Blue1Brown", "https://www.youtube.com/watch?v=IHZwWFHWa-w"]
  ],
  "dl/optimizers": [
    ["paper", "Adam: A Method for Stochastic Optimization", "Kingma & Ba, ICLR 2015", "https://arxiv.org/abs/1412.6980"],
    ["paper", "Decoupled Weight Decay Regularization (AdamW)", "Loshchilov & Hutter, ICLR 2019", "https://arxiv.org/abs/1711.05101"],
    ["interactive", "Why Momentum Really Works", "Distill.pub, interactive momentum physics", "https://distill.pub/2017/momentum/"]
  ],
  "dl/weight-initialization": [
    ["paper", "Understanding the difficulty of training deep feedforward networks", "Glorot & Bengio, 2010, Xavier init", "https://proceedings.mlr.press/v9/glorot10a.html"],
    ["paper", "Delving Deep into Rectifiers", "He et al., 2015, He init + PReLU", "https://arxiv.org/abs/1502.01852"],
    ["interactive", "Initializing neural networks", "deeplearning.ai AI Notes, interactive demo", "https://www.deeplearning.ai/ai-notes/initialization/"]
  ],
  "dl/vanishing-gradients": [
    ["paper", "Learning long-term dependencies with gradient descent is difficult", "Bengio, Simard, Frasconi, 1994", "https://doi.org/10.1109/72.279181"],
    ["paper", "Deep Residual Learning for Image Recognition", "He et al., 2015, the ResNet fix", "https://arxiv.org/abs/1512.03385"]
  ],
  "dl/dl-regularization": [
    ["paper", "Dropout: A Simple Way to Prevent Neural Networks from Overfitting", "Srivastava et al., JMLR 2014", "https://www.jmlr.org/papers/v15/srivastava14a.html"],
    ["paper", "Batch Normalization", "Ioffe & Szegedy, ICML 2015", "https://arxiv.org/abs/1502.03167"]
  ],
  "dl/nn-from-scratch": [
    ["read", "Neural Networks and Deep Learning", "Michael Nielsen, free book, builds everything from scratch", "http://neuralnetworksanddeeplearning.com/"],
    ["video", "The spelled-out intro to neural networks (micrograd)", "Andrej Karpathy", "https://www.youtube.com/watch?v=VMj-3S1tku0"],
    ["read", "Yes you should understand backprop", "Andrej Karpathy, why this exercise matters", "https://karpathy.medium.com/yes-you-should-understand-backprop-e2f06eab496b"]
  ],
  "dl/nn-in-pytorch": [
    ["read", "PyTorch: Deep Learning 60 Minute Blitz", "the official onboarding tutorial", "https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html"],
    ["read", "A Recipe for Training Neural Networks", "Andrej Karpathy, the debugging bible", "https://karpathy.github.io/2019/04/25/recipe/"]
  ],

  /* ---------- Module 8: CNNs ---------- */
  "cnn/why-cnns": [
    ["read", "CS231n: Convolutional Neural Networks", "Stanford, the standard course notes", "https://cs231n.github.io/convolutional-networks/"],
    ["paper", "Receptive fields, binocular interaction and functional architecture in the cat's visual cortex", "Hubel & Wiesel, 1962, the biological inspiration", "https://doi.org/10.1113/jphysiol.1962.sp006837"],
    ["video", "But what is a convolution?", "3Blue1Brown", "https://www.youtube.com/watch?v=KuXjwB4LzSA"]
  ],
  "cnn/convolution": [
    ["interactive", "Image Kernels Explained Visually", "Setosa, apply filters to images live", "https://setosa.io/ev/image-kernels/"],
    ["video", "But what is a convolution?", "3Blue1Brown", "https://www.youtube.com/watch?v=KuXjwB4LzSA"],
    ["interactive", "CNN Explainer", "Georgia Tech, walk through a live CNN layer by layer", "https://poloclub.github.io/cnn-explainer/"]
  ],
  "cnn/filters-padding-stride": [
    ["paper", "A guide to convolution arithmetic for deep learning", "Dumoulin & Visin, 2016, every case with animations", "https://arxiv.org/abs/1603.07285"],
    ["interactive", "Convolution arithmetic animations", "the companion GIF gallery", "https://github.com/vdumoulin/conv_arithmetic"]
  ],
  "cnn/pooling": [
    ["read", "CS231n: Pooling layer notes", "Stanford", "https://cs231n.github.io/convolutional-networks/#pool"],
    ["paper", "Striving for Simplicity: The All Convolutional Net", "Springenberg et al., 2014, do we even need pooling?", "https://arxiv.org/abs/1412.6806"]
  ],
  "cnn/classic-architectures": [
    ["paper", "ImageNet Classification with Deep CNNs (AlexNet)", "Krizhevsky, Sutskever, Hinton, NeurIPS 2012", "https://proceedings.neurips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html"],
    ["paper", "Very Deep Convolutional Networks (VGG)", "Simonyan & Zisserman, 2014", "https://arxiv.org/abs/1409.1556"],
    ["paper", "Deep Residual Learning (ResNet)", "He, Zhang, Ren, Sun, 2015", "https://arxiv.org/abs/1512.03385"],
    ["paper", "Going Deeper with Convolutions (Inception)", "Szegedy et al., 2014", "https://arxiv.org/abs/1409.4842"]
  ],
  "cnn/cnn-project": [
    ["read", "PyTorch: Training a Classifier (CIFAR-10)", "the official tutorial this chapter extends", "https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html"],
    ["interactive", "CNN Explainer", "inspect a trained CNN's every activation", "https://poloclub.github.io/cnn-explainer/"]
  ],

  /* ---------- Module 9: Sequence Models ---------- */
  "sequence/why-sequence-models": [
    ["read", "The Unreasonable Effectiveness of Recurrent Neural Networks", "Andrej Karpathy, the classic post", "https://karpathy.github.io/2015/05/21/rnn-effectiveness/"],
    ["paper", "Efficient Estimation of Word Representations (word2vec)", "Mikolov et al., 2013, embeddings", "https://arxiv.org/abs/1301.3781"]
  ],
  "sequence/rnn": [
    ["read", "The Unreasonable Effectiveness of Recurrent Neural Networks", "Andrej Karpathy", "https://karpathy.github.io/2015/05/21/rnn-effectiveness/"],
    ["video", "StatQuest: Recurrent Neural Networks", "Josh Starmer", "https://www.youtube.com/watch?v=AsNTP8Kwu80"]
  ],
  "sequence/lstm": [
    ["read", "Understanding LSTM Networks", "Christopher Olah, the single best LSTM explanation ever written", "https://colah.github.io/posts/2015-08-Understanding-LSTMs/"],
    ["paper", "Long Short-Term Memory", "Hochreiter & Schmidhuber, Neural Computation 1997", "https://doi.org/10.1162/neco.1997.9.8.1735"],
    ["video", "StatQuest: Long Short-Term Memory", "Josh Starmer", "https://www.youtube.com/watch?v=YCzL96nL7j0"]
  ],
  "sequence/gru": [
    ["paper", "Learning Phrase Representations using RNN Encoder-Decoder (GRU)", "Cho et al., 2014", "https://arxiv.org/abs/1406.1078"],
    ["paper", "Empirical Evaluation of Gated Recurrent Neural Networks", "Chung et al., 2014, the GRU-vs-LSTM study", "https://arxiv.org/abs/1412.3555"]
  ],
  "sequence/seq2seq": [
    ["paper", "Sequence to Sequence Learning with Neural Networks", "Sutskever, Vinyals, Le, NeurIPS 2014", "https://arxiv.org/abs/1409.3215"],
    ["read", "Visualizing seq2seq Models with Attention", "Jay Alammar, animated walkthrough", "https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/"]
  ],
  "sequence/attention": [
    ["paper", "Neural Machine Translation by Jointly Learning to Align and Translate", "Bahdanau, Cho, Bengio, 2014, attention's birth", "https://arxiv.org/abs/1409.0473"],
    ["read", "Attention and Augmented Recurrent Neural Networks", "Olah & Carter, Distill.pub, interactive", "https://distill.pub/2016/augmented-rnns/"],
    ["paper", "Effective Approaches to Attention-based NMT", "Luong, Pham, Manning, 2015", "https://arxiv.org/abs/1508.04025"]
  ],

  /* ---------- Module 10: Transformers ---------- */
  "transformers/self-attention": [
    ["paper", "Attention Is All You Need", "Vaswani et al., NeurIPS 2017, THE paper", "https://arxiv.org/abs/1706.03762"],
    ["read", "The Illustrated Transformer", "Jay Alammar, the most-read explanation on the internet", "https://jalammar.github.io/illustrated-transformer/"],
    ["video", "Attention in transformers, visually explained", "3Blue1Brown", "https://www.youtube.com/watch?v=eMlx5fFNoYc"]
  ],
  "transformers/positional-encoding": [
    ["paper", "RoFormer: Enhanced Transformer with Rotary Position Embedding (RoPE)", "Su et al., 2021, what modern LLMs use", "https://arxiv.org/abs/2104.09864"],
    ["read", "Transformer Architecture: The Positional Encoding", "Amirhossein Kazemnejad, the deep dive on sinusoids", "https://kazemnejad.com/blog/transformer_architecture_positional_encoding/"]
  ],
  "transformers/architecture": [
    ["paper", "Attention Is All You Need", "Vaswani et al., 2017", "https://arxiv.org/abs/1706.03762"],
    ["read", "The Annotated Transformer", "Harvard NLP, the paper reimplemented line by line", "https://nlp.seas.harvard.edu/annotated-transformer/"],
    ["video", "Let's build GPT: from scratch, in code", "Andrej Karpathy, 2 hours that change how you see LLMs", "https://www.youtube.com/watch?v=kCc8FmEb1nY"]
  ],
  "transformers/bert-gpt": [
    ["paper", "BERT: Pre-training of Deep Bidirectional Transformers", "Devlin et al., 2018", "https://arxiv.org/abs/1810.04805"],
    ["paper", "Language Models are Few-Shot Learners (GPT-3)", "Brown et al., 2020", "https://arxiv.org/abs/2005.14165"],
    ["read", "The Illustrated BERT", "Jay Alammar", "https://jalammar.github.io/illustrated-bert/"],
    ["paper", "Training language models to follow instructions (InstructGPT/RLHF)", "Ouyang et al., 2022", "https://arxiv.org/abs/2203.02155"]
  ],
  "transformers/transfer-learning": [
    ["paper", "LoRA: Low-Rank Adaptation of Large Language Models", "Hu et al., 2021", "https://arxiv.org/abs/2106.09685"],
    ["paper", "Universal Language Model Fine-tuning (ULMFiT)", "Howard & Ruder, 2018, transfer learning arrives in NLP", "https://arxiv.org/abs/1801.06146"],
    ["read", "Hugging Face NLP Course", "free, hands-on fine-tuning with Transformers", "https://huggingface.co/learn/nlp-course"]
  ],

  /* ---------- Module 11: Advanced ---------- */
  "advanced/autoencoders": [
    ["paper", "Auto-Encoding Variational Bayes (VAE)", "Kingma & Welling, 2013", "https://arxiv.org/abs/1312.6114"],
    ["read", "From Autoencoder to Beta-VAE", "Lilian Weng, the definitive blog treatment", "https://lilianweng.github.io/posts/2018-08-12-vae/"]
  ],
  "advanced/gans": [
    ["paper", "Generative Adversarial Networks", "Goodfellow et al., NeurIPS 2014", "https://arxiv.org/abs/1406.2661"],
    ["paper", "Unsupervised Representation Learning with DCGANs", "Radford, Metz, Chintala, 2015, the training recipes", "https://arxiv.org/abs/1511.06434"],
    ["interactive", "GAN Lab", "Georgia Tech, train a GAN in your browser and watch the duel", "https://poloclub.github.io/ganlab/"]
  ],
  "advanced/cnn-transfer-learning": [
    ["paper", "How transferable are features in deep neural networks?", "Yosinski et al., NeurIPS 2014, layer-by-layer transferability", "https://arxiv.org/abs/1411.1792"],
    ["read", "PyTorch: Transfer Learning tutorial", "the official fine-tune-a-ResNet walkthrough", "https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html"],
    ["read", "CS231n: Transfer Learning notes", "Stanford's practical recommendations", "https://cs231n.github.io/transfer-learning/"]
  ],
  "advanced/deployment": [
    ["paper", "Hidden Technical Debt in Machine Learning Systems", "Sculley et al., NeurIPS 2015, the famous 'ML is 5% of the system' paper", "https://proceedings.neurips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html"],
    ["read", "Made With ML: MLOps course", "Goku Mohandas, free end-to-end production ML", "https://madewithml.com/"],
    ["read", "FastAPI documentation", "the serving framework used in this chapter", "https://fastapi.tiangolo.com/"]
  ],

  /* ---------- Module 12: Practice ---------- */
  "practice/cheat-sheets": [
    ["read", "An Introduction to Statistical Learning", "free PDF, the classical-ML reference to keep", "https://www.statlearning.com/"],
    ["read", "Dive into Deep Learning", "free, the deep-learning reference to keep", "https://d2l.ai/"]
  ],
  "practice/glossary": [
    ["read", "Google Machine Learning Glossary", "the most complete free ML glossary", "https://developers.google.com/machine-learning/glossary"],
    ["read", "Papers With Code: Methods", "every technique linked to its papers and code", "https://paperswithcode.com/methods"]
  ],
  "practice/quizzes": [
    ["interactive", "Kaggle Competitions", "the real final exam: apply everything on live problems", "https://www.kaggle.com/competitions"],
    ["read", "Chip Huyen: ML Interviews Book", "free, for turning this knowledge into job offers", "https://huyenchip.com/ml-interviews-book/"]
  ]
};
