# MLP-AE-BiLSTM
🔹 1. Image Preprocessing
Input images undergo enhancement and noise removal:
Histogram equalization (histeq)
Morphological operations: opening, closing, erosion, dilation

🔹 2. Fused Feature Extraction (fused_feature.m)

Multiple complementary features are extracted and fused:

CSIFT features (scale, octave, layer, location, metric)
HOG features (texture/gradient)
Vegetation indices: NDVI, GNDVI, SVI
Statistical feature: kurtosis
_______________________________________________________________
proposed.m

P-LDA → Normalization → Autoencoder → BiLSTM → Classification

🔹 Step-by-step (brief)

Stage 1: P-LDA (Dimensionality Reduction)
Computes between-class & within-class scatter
Adds small perturbation for stability
Projects data to low-dimensional discriminative space
➝ Improves class separability

Stage 2: Z-score Normalization
Standardizes features (mean = 0, std = 1)

Stage 3: MLP Autoencoder
Learns compressed representation (latent features)
Encoder → bottleneck → decoder
➝ Extracts meaningful feature representations

Stage 4: Feature Extraction
Uses encoder output as final features

Stage 5: Sequence Preparation
Converts features into sequences for BiLSTM

Stage 6–7: BiLSTM Model
Bidirectional LSTM for classification
Uses optimized hyperparameters using (ObiLSTM_WFS.m)

Stage 8–9: Training & Testing
Train model and predict test labels
Stage 10: Evaluation
Outputs metrics: Accuracy, Precision, Recall, F1, MCC
