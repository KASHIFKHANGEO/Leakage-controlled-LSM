# Leakage-Controlled Evaluation of Machine Learning Landslide Susceptibility

This repository provides the complete Python implementation used for landslide susceptibility assessment (LSA) based on eight machine learning model families, four validation protocols, and multiple preprocessing and feature-engineering strategies. The workflow integrates Sentinel-2 monthly composites, CHIRPS monthly rainfall, half-hourly IMERG storm triggers, and 30 m SRTM terrain data under a connected-component leakage-control framework, and evaluates how much reported predictive skill survives independent validation.

The code is structured to ensure transparency and reproducibility of the methodological framework presented in the associated journal article.

## Methods Implemented

The repository includes the following components:

**Feature engineering and input data**

- Spatiotemporal feature vector construction
  - 12-month lookback window (60 features)
  - Pre-event change features (5 features)
  - Statistical summaries — mean, std, min, max, trend (25 features)
- Terrain and thematic conditioning factors (13 features)
- IMERG half-hourly rainfall trigger features (17 features)
  - Peak rolling accumulations (1, 3, 6, 12, 24 h)
  - Storm segmentation (count, total, duration, intensity)
  - Antecedent rainfall totals (7, 15, 30, 45 days)
  - Decay-weighted antecedent precipitation index
- Feature sets
  - **A** — temporal only (90 columns)
  - **B** — terrain only (13 columns)
  - **C** — terrain + temporal (103 columns)
  - **E** — terrain + IMERG triggers (30 columns)
  - **F** — all predictors (120 columns)

**Negative-sampling designs**

- **V0** — unmatched draw from stable-slope pool
- **V1** — calendar-month matched
- **V2** — season-and-year matched (adopted throughout)

**Machine learning models**

- Logistic Regression (LR)
- Random Forest (RF)
- Gradient Boosting Decision Trees (GBT)
- Multilayer Perceptron (MLP)
- CatBoost
- TabNet (attentive tabular network)
- FT-Transformer (tabular transformer)
- ST-CoupleNet (static-temporal coupling network with monotonic rainfall penalty)

**Validation strategies**

- **P1** — Random 5-fold cross-validation (leaky reference)
- **P2** — Spatial block cross-validation (5-fold on 0.1° cells)
- **P3** — Temporal point-split (forward-in-time, point-disjoint)
- **P4** — Cluster-disjoint cross-validation (points + 0.1° cells + source events held disjoint simultaneously via connected-component clustering)

**Structured ablations**

- ST-CoupleNet ablation on the temporal point-split
  - Full model
  - No attention (mean-pooled temporal branch)
  - No learned coupling (additive combination only)
  - No static branch (temporal-only)
  - No temporal branch (terrain-only)
  - Rainfall-only temporal channel
  - NDVI channel removed

**Evaluation metrics**

- AUC–ROC
- Accuracy
- Precision
- Recall
- Specificity
- F1-score
- Matthews Correlation Coefficient (MCC)
- Confusion-matrix statistics (TP, TN, FP, FN)
- DeLong tests for correlated ROC curves
- Bootstrap confidence intervals (2,000 resamples)
  - Sample-level
  - Cell-clustered

**Spatial dependence diagnostics**

- Empirical terrain variogram (60,000 random point pairs)
- Declustering-based Kish effective sample size

**Landslide susceptibility mapping**

- Pixel-based prediction on the 30 m terrain grid
- Five-class susceptibility zoning (Very Low → Very High) by natural breaks
- Area of applicability masking based on training-cluster distance


