import { RoadmapTrack, ResourceItem, VideoItem, ProjectItem } from "./types";

export const ROADMAP_TRACKS: RoadmapTrack[] = [
  {
    id: "ml-fundamentals",
    iconName: "BrainCircuit",
    title: "Machine Learning Fundamentals",
    description: "Master classical machine learning algorithms, statistical formulation, model evaluation, and regularizations.",
    difficulty: "Beginner",
    duration: "4 Weeks",
    syllabus: [
      {
        week: "Week 1",
        title: "Introduction to Supervised Learning",
        topics: ["Linear Regression & Gradient Descent", "Cost Functions", "Locally Weighted Regression", "Lasso (L1) & Ridge (L2) Regularization"],
      },
      {
        week: "Week 2",
        title: "Classification Algorithms",
        topics: ["Logistic Regression & Sigmoid", "Decision Trees & Entropy", "Support Vector Machines (SVM) & Kernel Tricks", "Generative vs Discriminative"],
      },
      {
        week: "Week 3",
        title: "Ensemble Methods & Unsupervised Learning",
        topics: ["Random Forests & Bagging", "AdaBoost & XGBoost", "K-Means Clustering", "Principal Component Analysis (PCA)"],
      },
      {
        week: "Week 4",
        title: "Model Diagnostics & Deployment",
        topics: ["Precision, Recall, ROC-AUC", "Overfitting, Underfitting & Bias-Variance Tradeoff", "Cross-Validation Techniques", "Pruning & Hyperparameter Tuning"],
      },
    ],
  },
  {
    id: "deep-learning",
    iconName: "Cpu",
    title: "Deep Learning & Neural Networks",
    description: "Dive deep into multilayer perceptrons, backpropagation, convolution structures, and recurrent neural nets.",
    difficulty: "Intermediate",
    duration: "6 Weeks",
    syllabus: [
      {
        week: "Week 1",
        title: "Neural Network Anatomy",
        topics: ["Neurons, Weights, Biases", "Activation Functions (ReLU, GELU, Softmax)", "Forward Propagation", "Matrix Formulations"],
      },
      {
        week: "Week 2",
        title: "Backpropagation & Optimization",
        topics: ["Anatomy of gradients & chain rule", "Loss Functions (Cross-Entropy, MSE)", "Adam, SGD, and RMSProp optimizers", "Learning Rate Schedulers"],
      },
      {
        week: "Week 3",
        title: "Regularizing Deep Networks",
        topics: ["Dropout layers", "Batch Normalization & Layer Normalization", "Vanishing & Exploding Gradients", "Weight Initialization (He, Xavier)"],
      },
      {
        week: "Week 4",
        title: "Convolutional Neural Networks (CNNs)",
        topics: ["Convolutions & Pooling operations", "Feature Map Extraction", "ResNet & VGG architectures", "Data Augmentation"],
      },
      {
        week: "Week 5",
        title: "Recurrent Networks & Sequential Models",
        topics: ["RNNs & Gradient decay", "LSTMs & GRUs", "Sequence-to-Sequence Modeling", "Bidirectional RNNs"],
      },
      {
        week: "Week 6",
        title: "Frameworks & Tensor Mathematics",
        topics: ["PyTorch Autograd", "Custom Datasets & Loaders", "GPU hardware acceleration indices", "Tensor manipulation math"],
      },
    ],
  },
  {
    id: "nlp",
    iconName: "MessageCircle",
    title: "Natural Language Processing",
    description: "Go from bag-of-words representational basics to contemporary transformer-based linguistic modules.",
    difficulty: "Intermediate",
    duration: "5 Weeks",
    syllabus: [
      {
        week: "Week 1",
        title: "Classical Text Representations",
        topics: ["Tokenization & Lemmatization", "TF-IDF Vectorization", "N-Gram models", "Regular Expressions & Parsing"],
      },
      {
        week: "Week 2",
        title: "Dense Word Vectors",
        topics: ["Word2Vec (Skip-gram & CBOW)", "GloVe Embeddings", "Cosine Similarity Metric", "Embedding Layer Weights"],
      },
      {
        week: "Week 3",
        title: "Sequence-to-Sequence & Attention",
        topics: ["Encoder-Decoder architecture", "Sutskever Seq2Seq model", "Bahdanau Additive Attention", "Luong Multiplicative Attention"],
      },
      {
        week: "Week 4",
        title: "The Transformer Revolution",
        topics: ["Self-Attention Mechanics", "Multi-Head Attention", "Positional Encoding formulation", "Scaled Dot-Product Formula"],
      },
      {
        week: "Week 5",
        title: "Pre-trained Language Models",
        topics: ["BERT (Encoder-only Masking)", "GPT (Decoder-only Causal Masking)", "Fine-tuning linguistic adapters", "Hugging Face Transformers library"],
      },
    ],
  },
  {
    id: "computer-vision",
    iconName: "Eye",
    title: "Computer Vision",
    description: "Understand digital visual processing, classical feature maps, object localization, and segmentative masks.",
    difficulty: "Intermediate",
    duration: "5 Weeks",
    syllabus: [
      {
        week: "Week 1",
        title: "Digital Image Basics & Kernels",
        topics: ["Pixel matrix math", "Sobel Filters & Edge detection", "Gaussian Blurring", "Histogram Equalization"],
      },
      {
        week: "Week 2",
        title: "Object Detection Architectures",
        topics: ["Bounding Box regressors", "IoU (Intersection over Union)", "YOLO (You Only Look Once) Realtime framework", "Anchor boxes"],
      },
      {
        week: "Week 3",
        title: "Two-Stage Detectors & Anchor Optimization",
        topics: ["R-CNN, Fast R-CNN, Faster R-CNN", "Region Proposal Networks (RPN)", "RoI Pooling & RoI Align"],
      },
      {
        week: "Week 4",
        title: "Visual Image Segmentation",
        topics: ["Semantic vs Instance Segmentation", "U-Net architecture for biomedical assets", "Mask R-CNN segmentation branches"],
      },
      {
        week: "Week 5",
        title: "Vision Transformers (ViT)",
        topics: ["Patch extraction from images", "Linear projection of patches", "Self-Attention in visual landscapes", "Hybrid CNN-Transformer models"],
      },
    ],
  },
  {
    id: "generative-ai",
    iconName: "Sparkles",
    title: "Generative AI & LLMs",
    description: "Explore the bleeding edge: autoregressive token decoding, prompt engineering, RLHF, and vector databases.",
    difficulty: "Advanced",
    duration: "8 Weeks",
    syllabus: [
      {
        week: "Week 1",
        title: "LLM Architectures & Mechanics",
        topics: ["Autoregressive decoding", "KV-Caching", "Top-k, Top-p, Temperature parameters", "Rotary Positional Embeddings (RoPE)"],
      },
      {
        week: "Week 2",
        title: "Prompt Engineering & In-Context Learning",
        topics: ["Few-Shot Prompting", "Chain-of-Thought (CoT) reasoning", "ReAct framework (Reason + Action)", "Structured JSON outputs"],
      },
      {
        week: "Week 3",
        title: "Retrieval-Augmented Generation (RAG)",
        topics: ["Document chunking strategies", "Dense retrieval & vector search", "Pinecone, ChromaDB & pgvector", "Reranker models"],
      },
      {
        week: "Week 4",
        title: "Instruction Fine-Tuning & PEFT",
        topics: ["Supervised Fine-Tuning (SFT)", "LoRA (Low-Rank Adaptation)", "QLoRA (Quantized LoRA)", "Prefix Tuning & Prompt Tuning"],
      },
      {
        week: "Week 5",
        title: "Alignment & RLHF",
        topics: ["Reward Model formulation", "PPO (Proximal Policy Optimization)", "DPO (Direct Preference Optimization)", "KTO Alignment algorithms"],
      },
      {
        week: "Week 6",
        title: "Agentic Workflows & Multi-Agent Frameworks",
        topics: ["Tool Use & Function Calling", "LangChain & LangGraph structures", "CrewAI or Autogen coordination", "Planner-Executor loops"],
      },
      {
        week: "Week 7",
        title: "Quantization & Efficient Inference",
        topics: ["GGUF, AWQ, and GPTQ formats", "vLLM serving engine", "Speculative decoding", "BitsAndBytes parameters"],
      },
      {
        week: "Week 8",
        title: "Evaluation & AI Safeguarding",
        topics: ["MMLU, GSM8k benchmarking datasets", "LLM-as-a-Judge evaluative metrics", "NeMo Guardrails & Llama Guard", "RLAIF (AI Feedback Alignment)"],
      },
    ],
  },
];

export const COURSES: ResourceItem[] = [
  {
    id: "course-1",
    name: "Practical Deep Learning for Coders",
    description: "Jeremy Howard's world-famous top-down code-first course that gets you training world-class models within hours using fast.ai and PyTorch.",
    tags: ["Free", "Beginner", "Python", "PyTorch"],
    url: "https://course.fast.ai/",
  },
  {
    id: "course-2",
    name: "Machine Learning Specialization",
    description: "Andrew Ng's classic, remastered masterclass on classical supervised and unsupervised ML, including deep math and production vectors.",
    tags: ["Free Tier", "Beginner", "Math", "Supervised"],
    url: "https://www.coursera.org/specializations/machine-learning-introduction",
  },
  {
    id: "course-3",
    name: "Deep Learning Specialization",
    description: "The gold standard Coursera suite by Andrew Ng covering CNNs, Sequence models, hyperparameter optimization, and transformer building blocks.",
    tags: ["Free Tier", "Intermediate", "Neural Nets"],
    url: "https://www.coursera.org/specializations/deep-learning",
  },
  {
    id: "course-4",
    name: "Stanford CS231n: Deep Learning for Computer Vision",
    description: "Stanford University's premier, deeply technical visual processing curriculum. Delves massive neural mathematical architectures.",
    tags: ["Free", "Advanced", "CNNs", "Stanford"],
    url: "http://cs231n.stanford.edu/",
  },
  {
    id: "course-5",
    name: "Hugging Face Course",
    description: "Step-by-step master guide for training NLP models, utilizing the Hugging Face ecosystem (Transformers, Datasets, Tokenizers, and Hub).",
    tags: ["Free", "Intermediate", "Transformers", "NLP"],
    url: "https://huggingface.co/learn/nlp-course",
  },
];

export const BOOKS: ResourceItem[] = [
  {
    id: "book-1",
    name: "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow",
    description: "Aurélien Géron's definitive practical manual that walks you through complete design steps for real-world production models.",
    tags: ["Comprehensive", "Codes", "Python"],
    url: "https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/",
  },
  {
    id: "book-2",
    name: "Deep Learning",
    description: "The 'Bible of Deep Learning' written by Ian Goodfellow, Yoshua Bengio, and Aaron Courville. Highly mathematical and theoretical.",
    tags: ["Math", "Advanced", "Theory"],
    url: "https://www.deeplearningbook.org/",
  },
  {
    id: "book-3",
    name: "The Hundred-Page Machine Learning Book",
    description: "Andriy Burkov's masterpiece of high-density compression—covers essential algorithms in exactly one hundred math-rich, readable pages.",
    tags: ["Core Concepts", "Beginner Friendly"],
    url: "http://themlbook.com/",
  },
  {
    id: "book-4",
    name: "RLHF & Reinforcement Learning Guides",
    description: "A highly curated handbook collection compiled by Hugging Face explaining deep human-preference alignment and proximal rewards.",
    tags: ["Alignment", "RLHF", "Advanced"],
    url: "https://huggingface.co/blog/rlhf",
  },
];

export const PAPERS: ResourceItem[] = [
  {
    id: "paper-1",
    name: "Attention Is All You Need (2017)",
    description: "Vaswani et al. introduced the revolutionary Multi-Head Self-Attention Transformer structure, replacing RNNs and creating modern Gen AI.",
    tags: ["Historic", "Transformers", "NLP"],
    url: "https://arxiv.org/abs/1706.03762",
  },
  {
    id: "paper-2",
    name: "Deep Residual Learning for Image Recognition (ResNet, 2015)",
    description: "Kaiming He et al. solved the vanishing gradient problem in deep networks using skip connections, creating 100+ layer architectures.",
    tags: ["CNN", "Residuals", "Vision"],
    url: "https://arxiv.org/abs/1512.03385",
  },
  {
    id: "paper-3",
    name: "Generative Adversarial Nets (GANs, 2014)",
    description: "Ian Goodfellow et al. introduced the minimax game theory setup pairing a Generator against a Discriminator to synthesize deep features.",
    tags: ["Generative", "Game Theory"],
    url: "https://arxiv.org/abs/1406.2661",
  },
  {
    id: "paper-4",
    name: "BERT: Pre-training of Deep Bidirectional Transformers (2018)",
    description: "Devlin et al. introduced bidirectional Transformer representations trained with Masked LM objectives, sweeping modern benchmarks.",
    tags: ["BERT", "Encoder", "NLP"],
    url: "https://arxiv.org/abs/1810.04805",
  },
  {
    id: "paper-5",
    name: "Language Models are Few-Shot Learners (GPT-3, 2020)",
    description: "Brown et al. showed that massive scale increases context competence, allowing zero/few shot transfers without direct parameter updates.",
    tags: ["LLM", "Few-Shot", "Scales"],
    url: "https://arxiv.org/abs/2005.14165",
  },
];

export const TOOLS: ResourceItem[] = [
  {
    id: "tool-1",
    name: "Google Colab",
    description: "Free in-browser Jupyter notebook environment with free access to high-end Nvidia GPUs (T4) – perfect for fast experimental pipelines.",
    tags: ["GPU", "Cloud", "Free"],
    url: "https://colab.research.google.com/",
  },
  {
    id: "tool-2",
    name: "Kaggle",
    description: "The global hub for data scientists. Hosts world-class predictive model challenges, massive public datasets, and GPU kernels.",
    tags: ["Competitions", "Datasets"],
    url: "https://www.kaggle.com/",
  },
  {
    id: "tool-3",
    name: "Weights & Biases (W&B)",
    description: "Production MLOps platform to track hyperparameters, log multi-epoch training metrics, visualize system resource utilization, and compare runs.",
    tags: ["MLOps", "Analytics", "Logging"],
    url: "https://wandb.ai/",
  },
  {
    id: "tool-4",
    name: "Hugging Face",
    description: "The 'GitHub of Machine Learning'. Access weights of open-source models, download massive token datasets, and deploy web demos.",
    tags: ["Models Hub", "Datasets", "Inference"],
    url: "https://huggingface.co/",
  },
  {
    id: "tool-5",
    name: "LangChain",
    description: "Powerful framework designed to easily bind Large Language Models with secondary APIs, structured agents, routers, and persistent vector storages.",
    tags: ["Agents", "LLMOps", "Chains"],
    url: "https://www.langchain.com/",
  },
];

export const VIDEOS: VideoItem[] = [
  {
    id: "vid-1",
    youtubeId: "VMj-3S1tku0",
    title: "Neural Networks: Zero to Hero — Intro to Backpropagation & Building micrograd",
    channel: "Andrej Karpathy",
    badge: "2h 24m",
  },
  {
    id: "vid-2",
    youtubeId: "PaFPbb6Z_fM",
    playlistId: "PLtBw6nj87gUY5bU_jG0F_NUpzZHe_S-81",
    title: "Machine Learning Playlist: Visualizing Core Statistical Foundations",
    channel: "StatQuest with Josh Starmer",
    badge: "Playlist",
  },
  {
    id: "vid-3",
    youtubeId: "8SF_h3xF378",
    title: "Practical Deep Learning for Coders: Part 1 Lesson 1 Introduction",
    channel: "fast.ai",
    badge: "1h 30m",
  },
  {
    id: "vid-4",
    youtubeId: "VqpS0kO7SCo",
    title: "Complete Python Machine Learning Roadmap & Concepts Tutorial",
    channel: "CampusX (Hindi)",
    badge: "2h 40m",
  },
  {
    id: "vid-5",
    youtubeId: "mEsleV16qdo",
    title: "Generative AI Full Course: LangChain, & Vector Databases for Developers",
    channel: "freeCodeCamp.org",
    badge: "4h 50m",
  },
  {
    id: "vid-6",
    youtubeId: "nE28_C1QLsk",
    title: "LangChain Crash Course for Beginners with Python Solutions",
    channel: "Patrick Loeber",
    badge: "45m",
  },
  {
    id: "vid-7",
    youtubeId: "ZftI2fEz0Fw",
    playlistId: "PLKnIAq7c0vKE-ZfMEb-TAn38VGrfXg_r6",
    title: "100 Days of Deep Learning: Complete Course & Neural Architectures",
    channel: "CampusX (Hindi)",
    badge: "Playlist",
  },
];

export const PROJECTS: ProjectItem[] = [
  // Beginner
  {
    id: "proj-beg-1",
    name: "Diabetes Risk Estimator",
    whatYouWillLearn: "Build a binary classification regression model, clean clinical datasets, handle NaN metrics, and compute precision-recall score analytics.",
    concepts: ["Logistic Regression", "Data Splitting", "ROC-AUC Curves"],
    estimatedTime: "6-8 Hours",
    difficulty: "Beginner",
  },
  {
    id: "proj-beg-2",
    name: "House Prices Predictor (Kaggle Classic)",
    whatYouWillLearn: "Preprocess complex continuous data rows, impute missing columns, apply Log-Transforms, and use simple Lasso/Ridge regularizations.",
    concepts: ["Linear Regression", "Lasso/Ridge", "Feature Scaling"],
    estimatedTime: "10 Hours",
    difficulty: "Beginner",
  },
  {
    id: "proj-beg-3",
    name: "Customer Clustered Segments with K-Means",
    whatYouWillLearn: "Apply numerical feature engineering on online retail transactions, determine optimal cluster numbers using the Elbow Curve method.",
    concepts: ["K-Means", "Silhouette Score", "Euclidean Distance"],
    estimatedTime: "6 Hours",
    difficulty: "Beginner",
  },
  {
    id: "proj-beg-4",
    name: "Interactive Spambox Classifier",
    whatYouWillLearn: "Master elementary NLP steps: lowercase mapping, punctuation stripping, token bags matching, and basic Naive Bayes probability modeling.",
    concepts: ["Naive Bayes", "CountVectorizer", "Tokenization"],
    estimatedTime: "8 Hours",
    difficulty: "Beginner",
  },
  // Intermediate
  {
    id: "proj-int-1",
    name: "Custom neural micrograd Engine",
    whatYouWillLearn: "Construct a custom autograd scalar solver in NumPy to backpropagate weight matrices across MLP nodes. Deeply understand mathematics.",
    concepts: ["PyTorch Core", "Gradient Chains", "NumPy Math"],
    estimatedTime: "15 Hours",
    difficulty: "Intermediate",
  },
  {
    id: "proj-int-2",
    name: "Pneumonia Scanner from Chest X-Rays",
    whatYouWillLearn: "Utilize transfer-learning (ResNet-18) in PyTorch, handle severe class imbalances with weighted cross-entropy loss, and print confusion metrics.",
    concepts: ["Transfer Learning", "CNNs", "PyTorch Optimizer"],
    estimatedTime: "18 Hours",
    difficulty: "Intermediate",
  },
  {
    id: "proj-int-3",
    name: "Real-time Multi-Object Tracker (YOLOv8)",
    whatYouWillLearn: "Connect a local OpenCV digital stream to a pre-trained YOLOv8 layer, extract visual bounding-boxes, and count custom object classes.",
    concepts: ["YOLOv8", "OpenCV Video", "Bounding Boxes"],
    estimatedTime: "12 Hours",
    difficulty: "Intermediate",
  },
  {
    id: "proj-int-4",
    name: "LSTM Character-Level Language Generator",
    whatYouWillLearn: "Train an LSTM network directly in text chunks, output characters sequentially by sampling probability tables configured under Temperature limits.",
    concepts: ["LSTMs", "Character Embeddings", "Temperature"],
    estimatedTime: "16 Hours",
    difficulty: "Intermediate",
  },
  // Advanced
  {
    id: "proj-adv-1",
    name: "Mini-GPT Transformer model from scratch",
    whatYouWillLearn: "Create complete self-attention modules, causal mask matrices, layer normalizations, token position estimators, and train on Shakespeare verses.",
    concepts: ["Multihead Attention", "Transformer Block", "Causal Masks"],
    estimatedTime: "30 Hours",
    difficulty: "Advanced",
  },
  {
    id: "proj-adv-2",
    name: "RAG Copilot with PDF ingestion",
    whatYouWillLearn: "Write a complete doc processor using LangChain, parse raw PDF layers, create text chunks, index embeddings inside ChromaDB, and prompt with LLM context.",
    concepts: ["ChromaDB", "Vector Search", "Prompt Ingestion"],
    estimatedTime: "24 Hours",
    difficulty: "Advanced",
  },
  {
    id: "proj-adv-3",
    name: "LoRA Adapters Fine-tuning for Sentiment Analysis",
    whatYouWillLearn: "Load a quantized Llama-3 model using Hugging Face PEFT, configure rank adapters, fine-tune downstream weights safely on a consumer GPU.",
    concepts: ["LoRA / QLoRA", "PEFT fine-tuning", "Quantization"],
    estimatedTime: "25 Hours",
    difficulty: "Advanced",
  },
  {
    id: "proj-adv-4",
    name: "Multi-Agent Research Coordinator",
    whatYouWillLearn: "Build structured multi-agent coordination with LangGraph, where agent roles (Searcher, Writer, Reviewer) execute sequential plan loops with clean feedback.",
    concepts: ["Multi-Agents", "State Handshakes", "LangGraph Loops"],
    estimatedTime: "28 Hours",
    difficulty: "Advanced",
  },
];
