/* ============================================================
   Curriculum definition - single source of truth for the
   sidebar, routing, prev/next order, and progress tracking.
   To add a topic later: add it here, then register its page in
   a js/content/*.js file under CONTENT["<module>/<topic>"].
   ============================================================ */

/* Registry that content files write into. Key: "moduleId/topicSlug". */
var CONTENT = {};

var CURRICULUM = [
  {
    id: "getting-started",
    num: 0,
    title: "Getting Started",
    desc: "How the course works and the full roadmap.",
    topics: [
      { slug: "how-to-use", title: "How to use this site" },
      { slug: "roadmap", title: "Prerequisites & Roadmap" }
    ]
  },
  {
    id: "math",
    num: 1,
    title: "Math & Stats Foundations",
    desc: "The linear algebra, calculus and statistics every ML topic builds on.",
    topics: [
      { slug: "linear-algebra", title: "Linear Algebra for ML" },
      { slug: "calculus", title: "Calculus for ML" },
      { slug: "probability", title: "Probability Basics" },
      { slug: "descriptive-statistics", title: "Descriptive Statistics" },
      { slug: "distributions", title: "Distributions" },
      { slug: "hypothesis-testing", title: "Hypothesis Testing & p-value" },
      { slug: "confidence-intervals", title: "Confidence Intervals" },
      { slug: "correlation-causation", title: "Correlation vs Causation" }
    ]
  },
  {
    id: "python",
    num: 2,
    title: "Python for Data Science",
    desc: "NumPy, Pandas and data cleaning - the daily tools.",
    topics: [
      { slug: "numpy", title: "NumPy Essentials" },
      { slug: "pandas", title: "Pandas Essentials" },
      { slug: "data-cleaning", title: "Data Cleaning Basics" }
    ]
  },
  {
    id: "dataviz",
    num: 3,
    title: "Data Visualization Lab",
    desc: "Matplotlib to Plotly, plus which chart to use when.",
    topics: [
      { slug: "why-visualization", title: "Why Visualization Matters" },
      { slug: "matplotlib", title: "Matplotlib Basics" },
      { slug: "seaborn", title: "Seaborn Basics" },
      { slug: "plotly", title: "Plotly (Interactive)" },
      { slug: "chart-picker", title: "Chart-Picker Guide" },
      { slug: "ml-visualizations", title: "Visualizations for ML" },
      { slug: "paper-visualizations", title: "Visualizations in Research Papers" }
    ]
  },
  {
    id: "eda",
    num: 4,
    title: "Exploratory Data Analysis",
    desc: "A repeatable workflow for understanding any dataset.",
    topics: [
      { slug: "workflow", title: "EDA Workflow / Checklist" },
      { slug: "missing-values", title: "Handling Missing Values" },
      { slug: "outliers", title: "Outlier Detection" },
      { slug: "uni-bi-multivariate", title: "Univariate / Bivariate / Multivariate" },
      { slug: "correlation-analysis", title: "Correlation Analysis" },
      { slug: "case-study", title: "Case Study: Full EDA" }
    ]
  },
  {
    id: "features",
    num: 5,
    title: "Feature Engineering",
    desc: "Turning raw columns into signals models can learn from.",
    topics: [
      { slug: "encoding", title: "Encoding Categorical Variables" },
      { slug: "scaling", title: "Scaling & Normalization" },
      { slug: "imbalanced-data", title: "Handling Imbalanced Data" },
      { slug: "feature-selection", title: "Feature Selection Methods" },
      { slug: "dimensionality-reduction", title: "Dimensionality Reduction (PCA, t-SNE, UMAP)" }
    ]
  },
  {
    id: "ml",
    num: 6,
    title: "Machine Learning Algorithms",
    desc: "The classic algorithms: theory, math, code, and when to use each.",
    topics: [
      { slug: "learning-paradigms", title: "Supervised vs Unsupervised vs RL" },
      { slug: "linear-regression", title: "Linear Regression" },
      { slug: "logistic-regression", title: "Logistic Regression" },
      { slug: "knn", title: "K-Nearest Neighbors" },
      { slug: "decision-trees", title: "Decision Trees" },
      { slug: "random-forest", title: "Random Forest" },
      { slug: "gradient-boosting", title: "Gradient Boosting (XGBoost, LightGBM)" },
      { slug: "svm", title: "Support Vector Machines" },
      { slug: "naive-bayes", title: "Naive Bayes" },
      { slug: "kmeans", title: "K-Means Clustering" },
      { slug: "hierarchical-clustering", title: "Hierarchical Clustering" },
      { slug: "dbscan", title: "DBSCAN" },
      { slug: "evaluation-metrics", title: "Model Evaluation Metrics" },
      { slug: "cross-validation", title: "Cross-Validation" },
      { slug: "bias-variance", title: "Bias-Variance Tradeoff" },
      { slug: "regularization", title: "Regularization (L1/L2)" },
      { slug: "hyperparameter-tuning", title: "Hyperparameter Tuning" }
    ]
  },
  {
    id: "dl",
    num: 7,
    title: "Deep Learning Foundations",
    desc: "From a single neuron to a full network trained from scratch.",
    topics: [
      { slug: "perceptron", title: "Biological Neuron → Perceptron" },
      { slug: "mlp", title: "Multi-Layer Perceptron / ANN" },
      { slug: "activation-functions", title: "Activation Functions" },
      { slug: "loss-functions", title: "Loss Functions" },
      { slug: "forward-propagation", title: "Forward Propagation" },
      { slug: "backpropagation", title: "Backpropagation" },
      { slug: "gradient-descent-variants", title: "Gradient Descent Variants" },
      { slug: "optimizers", title: "Optimizers (SGD, Adam, …)" },
      { slug: "weight-initialization", title: "Weight Initialization" },
      { slug: "vanishing-gradients", title: "Vanishing / Exploding Gradients" },
      { slug: "dl-regularization", title: "Regularization in DL" },
      { slug: "nn-from-scratch", title: "Neural Network from Scratch (NumPy)" },
      { slug: "nn-in-pytorch", title: "The Same Network in PyTorch" }
    ]
  },
  {
    id: "cnn",
    num: 8,
    title: "Convolutional Neural Networks",
    desc: "How machines see: convolutions, pooling, classic architectures.",
    topics: [
      { slug: "why-cnns", title: "Why CNNs for Images" },
      { slug: "convolution", title: "The Convolution Operation" },
      { slug: "filters-padding-stride", title: "Filters, Padding, Stride" },
      { slug: "pooling", title: "Pooling Layers" },
      { slug: "classic-architectures", title: "Classic Architectures" },
      { slug: "cnn-project", title: "CNN Code Walkthrough" }
    ]
  },
  {
    id: "sequence",
    num: 9,
    title: "Sequence Models",
    desc: "RNNs, LSTMs, GRUs and the road to attention.",
    topics: [
      { slug: "why-sequence-models", title: "Why Sequence Models" },
      { slug: "rnn", title: "Recurrent Neural Networks" },
      { slug: "lstm", title: "LSTM" },
      { slug: "gru", title: "GRU" },
      { slug: "seq2seq", title: "Sequence-to-Sequence Models" },
      { slug: "attention", title: "Attention Mechanism" }
    ]
  },
  {
    id: "transformers",
    num: 10,
    title: "Transformers & Modern Architectures",
    desc: "Self-attention, the Transformer, BERT/GPT and fine-tuning.",
    topics: [
      { slug: "self-attention", title: "Self-Attention & Multi-Head Attention" },
      { slug: "positional-encoding", title: "Positional Encoding" },
      { slug: "architecture", title: "Transformer Architecture" },
      { slug: "bert-gpt", title: "BERT / GPT Overview" },
      { slug: "transfer-learning", title: "Transfer Learning & Fine-tuning" }
    ]
  },
  {
    id: "advanced",
    num: 11,
    title: "Advanced / Extra Topics",
    desc: "Autoencoders, GANs, transfer learning and deployment.",
    topics: [
      { slug: "autoencoders", title: "Autoencoders" },
      { slug: "gans", title: "GANs" },
      { slug: "cnn-transfer-learning", title: "Transfer Learning in CNNs" },
      { slug: "deployment", title: "Model Deployment Basics" }
    ]
  },
  {
    id: "practice",
    num: 12,
    title: "Practice Zone",
    desc: "Cheat sheets, glossary and quizzes for revision.",
    topics: [
      { slug: "cheat-sheets", title: "Cheat Sheets" },
      { slug: "glossary", title: "Glossary of ML/DL Terms" },
      { slug: "quizzes", title: "Quiz per Module" }
    ]
  }
];
