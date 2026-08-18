/**
 * data.js
 * Centralized Single Source of Truth for Research & Experimentation Metrics.
 * Exact Data Integrity: 4.18 yrs MAE, Dual Ensemble (DEX + Hybrid Head), DeepFake 25K (7.20 MAE).
 */

export const PROJECT_CONFIG = {
  name: "AGE-X",
  title: "AI Facial Age Prediction",
  subtitle: "Research & Experimentation Console",
  version: "v3.3.0-research",
  status: {
    model: "ONLINE",
    inference: "READY",
    cuda: "ACTIVE",
    checkpoint: "DUAL ENSEMBLE (EFFNETV2-S DEX + HYBRID HEAD)",
  }
};

export const KPI_METRICS = [
  {
    id: "mae",
    label: "FINAL ENSEMBLE MAE",
    value: "4.18",
    unit: "yrs",
    direction: "Lower is better",
    delta: "Dual Ensemble (DEX + Hybrid)",
    status: "positive",
    highlight: true,
  },
  {
    id: "dataset",
    label: "MASTER CORPUS",
    value: "276.2K",
    unit: "faces",
    direction: "Curated dataset",
    delta: "2.33L Master Manifest",
    status: "neutral",
  },
  {
    id: "accuracy",
    label: "ACCURACY (±7 YEARS)",
    value: "82.0%",
    unit: "",
    direction: "Higher is better",
    delta: "Acc (±5 Yrs): 67.0%",
    status: "positive",
  },
  {
    id: "range",
    label: "EVALUATION RANGE",
    value: "1–100",
    unit: "yrs",
    direction: "Full demographic span",
    delta: "100-way DEX continuous distribution",
    status: "neutral",
  }
];

export const PROBLEM_CARDS = [
  {
    id: "imbalance",
    num: "01",
    title: "Long-Tailed Age Distribution",
    summary: "Significant sample density disparity across demographics.",
    desc: "Real-world facial datasets are heavily biased toward the 20–45 age cohort (>60% of samples). Senior (61–75) and elderly (76–100) demographics represent less than 9% of total volume, causing standard models to under-fit tail cohorts.",
    metric: "61.1% in ages 20–45 vs 1.4% in 76–100"
  },
  {
    id: "uncertainty",
    num: "02",
    title: "Age Regression Uncertainty",
    summary: "Biological appearance diverges non-linearly from chronological age.",
    desc: "Single scalar regression (L1/MSE) penalizes natural physiological divergence equally. Direct single-target regression fails to capture distribution variance caused by genetics, lighting, cosmetics, expression, and periorbital bone structure.",
    metric: "High epistemic variance in single L1/MSE loss"
  },
  {
    id: "generalization",
    num: "03",
    title: "Tail-Age Generalization Gap",
    summary: "Severe error escalation at extreme ends of the lifespan.",
    desc: "When evaluated on held-out test splits without specialized inductive biases, standard architectures exhibit severe regression collapse, over-predicting young subjects and under-predicting elderly individuals.",
    metric: "Baseline elderly MAE exceeded 12.8 yrs"
  }
];

export const DATASET_PIPELINE = [
  {
    stage: "Stage 01",
    name: "Original Base Dataset",
    count: "210,148",
    unit: "images",
    role: "Core face pool with high natural variation.",
    pct: "76.1%"
  },
  {
    stage: "Stage 02",
    name: "UTKFace Supplement",
    count: "+23,708",
    unit: "images",
    role: "Balanced ethnicity and childhood age distribution.",
    pct: "+8.6%"
  },
  {
    stage: "Stage 03",
    name: "IMDB Targeted Supplement",
    count: "+42,424",
    unit: "images",
    role: "High-resolution senior & elder demographic augmentation.",
    pct: "+15.3%"
  },
  {
    stage: "Master",
    name: "Master Combined Corpus",
    count: "276,280",
    unit: "images",
    role: "Unified locked dataset with stratified 80/10/10 split.",
    pct: "100.0%",
    isMaster: true
  }
];

export const DEMOGRAPHIC_COHORTS = [
  { bracket: "1–12", label: "Children", count: 10259, pct: 3.71, sparse: false },
  { bracket: "13–19", label: "Teens", count: 18437, pct: 6.67, sparse: false },
  { bracket: "20–35", label: "Young Adults", count: 115002, pct: 41.63, dense: true },
  { bracket: "36–45", label: "Adults", count: 53828, pct: 19.48, dense: true },
  { bracket: "46–60", label: "Middle Age", count: 53402, pct: 19.33, dense: false },
  { bracket: "61–75", label: "Seniors", count: 21341, pct: 7.72, sparse: true },
  { bracket: "76–100", label: "Elderly", count: 4011, pct: 1.45, sparse: true, highlight: true }
];

export const MODEL_TOURNAMENT = [
  {
    id: "EXP-01",
    architecture: "ResNet-50 + L1 Regression",
    backbone: "ResNet-50",
    head: "Single Scalar L1 Linear",
    strategy: "Standard SGD Baseline",
    mae: 7.62,
    acc7: "58.4%",
    isChampion: false,
    notes: "High outlier sensitivity; severe regression collapse at boundaries."
  },
  {
    id: "EXP-DF25K",
    architecture: "DeepFake-25K Fine-Tuned",
    backbone: "ResNet / MobileNet",
    head: "Regression Head",
    strategy: "Trained on 25k data for 8 epochs",
    mae: 7.20,
    acc7: "61.2%",
    isChampion: false,
    notes: "Fine-tuned on 25,000 deepfake/external face images across 8 epochs; baseline domain transfer."
  },
  {
    id: "EXP-15",
    architecture: "MobileNetV3 Lightweight",
    backbone: "MobileNetV3-Large",
    head: "Linear + Smooth L1",
    strategy: "Lightweight architecture trial",
    mae: 6.45,
    acc7: "66.8%",
    isChampion: false,
    notes: "Fast edge-inference, but constrained capacity on complex facial wrinkle patterns."
  },
  {
    id: "EXP-20",
    architecture: "ResNet-50 + Huber Loss",
    backbone: "ResNet-50",
    head: "Huber Regression Head",
    strategy: "Robust loss optimization",
    mae: 5.11,
    acc7: "75.4%",
    isChampion: false,
    notes: "Reduced outlier sensitivity; 2.51 yrs MAE gain over naive L1 baseline."
  },
  {
    id: "EXP-23",
    architecture: "EfficientNetV2-S + Hybrid Head",
    backbone: "EfficientNetV2-S",
    head: "DEX Softmax + Auxiliary Regression",
    strategy: "Multi-task loss formulation",
    mae: 4.67,
    acc7: "78.1%",
    isChampion: false,
    notes: "Fused progressive feature pyramid with auxiliary loss stabilization."
  },
  {
    id: "EXP-25",
    architecture: "EfficientNetV2-S + DEX Head",
    backbone: "EfficientNetV2-S",
    head: "100-Way Discrete DEX Softmax",
    strategy: "Expectation-over-probabilities",
    mae: 4.64,
    acc7: "78.2%",
    isChampion: false,
    notes: "Softmax expected value formulation eliminates gradient saturation."
  },
  {
    id: "DUAL-ENS",
    architecture: "Dual Ensemble (DEX + Hybrid Head)",
    backbone: "Dual EfficientNetV2-S (Model A + Model B)",
    head: "DEX (100-Way) + Hybrid Fusion + 2-View TTA",
    strategy: "Model A (DEX) + Model B (Hybrid) + Flip TTA",
    mae: 4.18,
    acc7: "82.0%",
    isChampion: true,
    badge: "Dual Ensemble Champion",
    notes: "Official Champion: 2 Combined Models (Model A DEX + Model B Hybrid) averaged with 2-View Mirror TTA achieving 4.18 yrs MAE."
  }
];

export const EXPERIMENT_TIMELINE = [
  {
    id: "EXP-01",
    phase: "Phase 1: Baseline",
    model: "ResNet-50 (L1 Regression)",
    mae: "7.62 yrs",
    delta: "Starting Baseline",
    insight: "Scalar regression is unstable for facial aging due to non-Gaussian noise and age label ambiguity."
  },
  {
    id: "EXP-DF",
    phase: "Phase 2: DeepFake 25K Trial",
    model: "DeepFake-25K (Fine-Tuned 8 Epochs)",
    mae: "7.20 yrs",
    delta: "-0.42 yrs",
    insight: "Fine-tuning on 25,000 external face samples for 8 epochs showed transferability limitations."
  },
  {
    id: "EXP-15",
    phase: "Phase 3: Mobile Exploration",
    model: "MobileNetV3 Lightweight",
    mae: "6.45 yrs",
    delta: "-0.75 yrs",
    insight: "Proved deep feature hierarchies are strictly necessary for fine periorbital wrinkle representations."
  },
  {
    id: "EXP-20",
    phase: "Phase 4: Loss Function Engineering",
    model: "ResNet-50 + Huber Loss",
    mae: "5.11 yrs",
    delta: "-1.34 yrs",
    insight: "Robust Huber loss effectively suppressed gradient explosion caused by mislabeled dataset outliers."
  },
  {
    id: "EXP-23",
    phase: "Phase 5: Multi-Task Architecture",
    model: "EfficientNetV2-S (Hybrid Head)",
    mae: "4.67 yrs",
    delta: "-0.44 yrs",
    insight: "Simultaneous optimization of continuous regression and discrete age buckets stabilized backpropagation."
  },
  {
    id: "EXP-25",
    phase: "Phase 6: Distribution Formulation",
    model: "EfficientNetV2-S (DEX Expected Age)",
    mae: "4.64 yrs",
    delta: "-0.03 yrs",
    insight: "Softmax probability distribution sum(p_i * i) eliminated boundary clipping and improved smooth gradients."
  },
  {
    id: "DUAL-ENS",
    phase: "Phase 7: Dual Ensemble Champion",
    model: "Dual Ensemble (EffNetV2-S DEX + Hybrid) + 2-View TTA",
    mae: "4.18 yrs",
    delta: "-0.46 yrs",
    isChampion: true,
    insight: "Combining Model A (DEX Head) and Model B (Hybrid Head) with 2-View Mirror TTA achieved state-of-the-art 4.18 yrs MAE."
  }
];

export const TRAINING_CONVERGENCE = [
  { epoch: 1, trainMae: 7.84, valMae: 5.73, acc5: 54.58 },
  { epoch: 2, trainMae: 5.30, valMae: 4.90, acc5: 63.30 },
  { epoch: 3, trainMae: 4.61, valMae: 4.82, acc5: 64.44 },
  { epoch: 4, trainMae: 3.99, valMae: 4.69, acc5: 65.92 },
  { epoch: 5, trainMae: 3.62, valMae: 4.64, acc5: 67.00 }
];

export const DEMOGRAPHIC_PERFORMANCE = [
  {
    bracket: "01–12",
    cohort: "Children / Pediatric",
    mae: 2.63,
    acc5: 85.93,
    acc7: 88.32,
    testSamples: 334,
    grade: "Exceptional",
    intensity: 0.95,
    highlight: true,
    comment: "Bone structure and cranial geometry changes provide sharp visual distinction."
  },
  {
    bracket: "13–19",
    cohort: "Teens",
    mae: 5.31,
    acc5: 59.92,
    acc7: 73.45,
    testSamples: 1027,
    grade: "Good",
    intensity: 0.65,
    comment: "High variance due to puberty acceleration, styling, and makeup."
  },
  {
    bracket: "20–30",
    cohort: "Young Adults",
    mae: 3.74,
    acc5: 76.13,
    acc7: 86.80,
    testSamples: 4894,
    grade: "Exceptional",
    intensity: 0.90,
    highlight: true,
    comment: "Dense training data enables sub-3.8 year precision."
  },
  {
    bracket: "31–45",
    cohort: "Adults",
    mae: 3.98,
    acc5: 71.21,
    acc7: 85.29,
    testSamples: 6451,
    grade: "High Accuracy",
    intensity: 0.88,
    highlight: true,
    comment: "Consistent facial landmarks and fine line emergence patterns."
  },
  {
    bracket: "46–60",
    cohort: "Middle Age",
    mae: 5.13,
    acc5: 60.94,
    acc7: 75.19,
    testSamples: 2652,
    grade: "Good",
    intensity: 0.70,
    comment: "Skin elasticity variability and lighting conditions increase variance."
  },
  {
    bracket: "61–75",
    cohort: "Seniors",
    mae: 5.97,
    acc5: 56.86,
    acc7: 71.35,
    testSamples: 897,
    grade: "Moderate",
    intensity: 0.55,
    comment: "Sparse demographic distribution impacts generalization accuracy."
  },
  {
    bracket: "76–100",
    cohort: "Elderly",
    mae: 8.59,
    acc5: 46.85,
    acc7: 58.27,
    testSamples: 143,
    grade: "Data-Sparse",
    intensity: 0.35,
    isWeakness: true,
    comment: "Extreme long-tail scarcity (<1.5% master corpus). High biological appearance variance."
  }
];

export const ERROR_TOLERANCE = [
  { tolerance: "±3 years", percentage: 48.76, label: "Exact / Tight Demographic Alignment", barWidth: 48.76 },
  { tolerance: "±5 years", percentage: 67.00, label: "Industrial Verification Tolerance (Acc ±5)", barWidth: 67.00 },
  { tolerance: "±7 years", percentage: 82.00, label: "Benchmark Passing Criterion (80%+)", barWidth: 82.00, isBenchmark: true },
  { tolerance: "±10 years", percentage: 91.68, label: "Demographic Bracket Reliability", barWidth: 91.68 }
];

export const ARCHITECTURE_WHY = [
  {
    name: "EfficientNetV2-S Backbone",
    tag: "Dual Convolutional Feature Extractor",
    desc: "Optimized accuracy-to-compute tradeoff utilizing Fused-MBConv stages. Provides progressive training stability and rich multi-scale feature hierarchies on 320x320 facial patches."
  },
  {
    name: "Model A: DEX 100-Way Softmax Head",
    tag: "Expectation Distribution Formulation",
    desc: "Transforms continuous age prediction into a 100-way discrete probability distribution, computing expected age sum(p_i * i). Prevents single-outlier loss explosion and avoids vanishing gradients."
  },
  {
    name: "Model B: Hybrid Dual Head",
    tag: "Discrete + Continuous Multi-Task Anchor",
    desc: "Combines 100-class categorical cross-entropy with auxiliary continuous regression anchor. Stabilizes backpropagation and complements Model A's probability curve."
  },
  {
    name: "2-View Mirror Test-Time Augmentation (TTA)",
    tag: "Inference-Time Noise Reduction",
    desc: "Evaluates both original and horizontally flipped images through Model A and Model B in parallel. Averaging all 4 predictions cancels asymmetric facial illumination and head-pose variance."
  }
];

export const RESEARCH_INSIGHTS = [
  {
    num: "01",
    title: "More data does not automatically mean better performance.",
    desc: "Adding uncurated celebrity datasets degraded baseline validation MAE due to heavy cosmetics, airbrushing, and studio lighting artifacts. Balanced demographic representation is strictly superior to raw sample count."
  },
  {
    num: "02",
    title: "Specialization improves weak cohorts but risks catastrophic forgetting.",
    desc: "Training dedicated age-expert models on elderly subgroups reduced 76–100 MAE to 6.1 yrs, but severely degraded child/teen accuracy by 4.2 yrs when tested globally. Dual joint ensemble fusion is required to preserve general representations."
  },
  {
    num: "03",
    title: "Dual complementary heads outperform single model regression.",
    desc: "Combining Model A (DEX Expected Age) with Model B (Hybrid Head) and 2-View Mirror TTA reduced overall MAE to 4.18 yrs by smoothing prediction variance across ambiguous boundary ages."
  },
  {
    num: "04",
    title: "Tail-age prediction remains the fundamental computer vision frontier.",
    desc: "Biological appearance divergence increases exponentially after age 60. Factors like sun exposure, lifestyle, dental structure, and genetics cause two 70-year-olds to appear biologically 15 years apart, establishing an inherent epistemic uncertainty limit."
  }
];

export const LIMITATIONS = [
  {
    title: "Sparse 76–100 Demographic Data",
    desc: "Public and curated datasets contain under 2% representation for elderly subjects, constraining statistical certainty in high-age brackets."
  },
  {
    title: "Appearance vs Chronological Age Discrepancy",
    desc: "Facial age estimation measures apparent biological age; genetics, health, and cosmetic interventions cause natural deviation from chronological birth records."
  },
  {
    title: "Non-Uniform Demographic Distribution",
    desc: "The master corpus exhibits historical regional imbalance; while UTKFace mitigates racial bias, marginal domain shifts persist in extreme lighting."
  },
  {
    title: "Domain Adaptation on Heavy Occlusions",
    desc: "Thick eyewear, dense beards, religious headwear, and severe off-axis yaw angles (>45 deg) increase prediction variance by up to ±2.1 years."
  }
];
