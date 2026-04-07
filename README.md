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

