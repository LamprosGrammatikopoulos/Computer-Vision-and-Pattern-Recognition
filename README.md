# Visual Search of an Image Collection

A Content-Based Image Retrieval (CBIR) system developed for the **EEE3032 – Computer Vision & Pattern Recognition** module at the **University of Surrey**. The project implements and evaluates multiple handcrafted image descriptors, dimensionality reduction techniques, and similarity metrics for visual image retrieval using the **MSRC-v2** image dataset.

---

## Overview

This project investigates how different image descriptors and similarity measures affect the performance of a visual search system.

Given a query image, the system extracts feature descriptors, compares them with descriptors from every image in the dataset, and returns the most visually similar images ranked by similarity.

The project progressively extends a baseline Global Colour Histogram implementation with more advanced computer vision techniques including spatial descriptors, texture features, PCA, and multiple histogram distance metrics.

---

## Features

- Global RGB Colour Histograms
- Spatial Grid descriptors (2×2, 3×3, 4×4)
- Edge Orientation Histograms (EOH)
- Combined Colour + Texture descriptors
- Principal Component Analysis (PCA)
- Euclidean and Mahalanobis retrieval
- 12 different similarity/distance metrics
- Precision–Recall evaluation
- Mean Average Precision (mAP)
- Confusion Matrix generation
- Query visualization
- Descriptor comparison framework

---

## Dataset

The system uses the **Microsoft Research Cambridge (MSRC-v2)** dataset.

- **591 images**
- **20 object categories**
- Images include buildings, trees, sheep, cars, bicycles, people, dogs, birds, bookshelves, boats and more.

---

## Project Structure

```
cvpr_computedescriptors.py
cvpr_visualsearch.py
cvpr_compare.py
cvpr_pca_utils.py
run_pca_comparison.py

descriptors/
plots/
results/
```

---

## Implemented Image Descriptors

### 1. Global Colour Histogram

- RGB histogram descriptors
- Multiple quantization levels
- Normalized histograms
- Euclidean distance baseline

Tested quantization levels:

- 4 bins
- 8 bins
- 12 bins
- 16 bins
- 20 bins
- 24 bins
- 28 bins
- 32 bins

---

### 2. Spatial Grid Descriptors

Images are divided into spatial cells before extracting features.

Grid sizes evaluated:

- 2×2
- 3×3
- 4×4

This preserves spatial information that is lost by global histograms.

---

### 3. Edge Orientation Histograms

Texture descriptors are extracted using:

- Sobel gradients
- Gradient orientation
- Gradient magnitude weighting
- Orientation histogram normalization

Orientation bins tested:

- 4
- 8
- 12
- 16

---

### 4. Combined Colour + Texture Descriptors

The project also evaluates descriptors that combine:

- Spatial Grid
- Global Colour Histogram
- Edge Orientation Histogram

These descriptors provide significantly higher retrieval performance than colour alone.

---

## PCA Experiments

Principal Component Analysis was implemented from scratch using Eigenvalue Decomposition.

Experiments evaluate:

- 70% variance
- 75%
- 80%
- 85%
- 90%
- 95%
- 99%

using both

- Euclidean distance
- Mahalanobis distance

to study the trade-off between dimensionality reduction and retrieval accuracy.

---

## Distance Metrics

The following similarity metrics are implemented:

- Euclidean
- Manhattan
- Minkowski
- Chebyshev
- Cosine
- Canberra
- Mahalanobis
- Chi-Square
- Histogram Intersection
- Bhattacharyya
- Hellinger
- Correlation

---

## Evaluation

The retrieval system is evaluated using:

- Precision
- Recall
- Precision–Recall Curves
- Average Precision (AP)
- Mean Average Precision (mAP)
- mAP@10
- Full mAP
- Confusion Matrices

Both qualitative retrieval examples and quantitative evaluations are provided.

---

## Results

Key findings include:

- **8-bin Global Colour Histograms** provided the best baseline performance.
- Spatial Grid descriptors significantly outperformed global colour histograms.
- The **3×3 Spatial Grid + Edge Orientation Histogram** achieved the highest overall retrieval accuracy.
- Combining colour and texture produced more balanced performance across image classes.
- PCA reduced descriptor dimensionality by over **99%** while maintaining or even improving retrieval accuracy.
- The best-performing distance metrics were:
  - Bhattacharyya
  - Hellinger
  - Chi-Square
  - Manhattan
  - Histogram Intersection

---

## Technologies

- Python
- NumPy
- OpenCV
- SciPy
- Matplotlib
- scikit-learn

---

## Example Workflow

1. Compute image descriptors for every image.
2. Store descriptors on disk.
3. Select a query image.
4. Compare descriptors using a selected distance metric.
5. Rank images by similarity.
6. Display the top retrieved images.
7. Evaluate using mAP, Precision-Recall curves, and Confusion Matrices.

---

## Future Improvements

Potential extensions include:

- SIFT descriptors
- SURF descriptors
- ORB descriptors
- Bag of Visual Words (BoVW)
- Spatial Pyramid Matching
- Deep CNN embeddings (ResNet, CLIP)
- FAISS approximate nearest neighbour search
- Interactive retrieval GUI

---

## Author

**Lampros Grammatikopoulos**

MSc Artificial Intelligence  
University of Surrey

---

## License

This project was developed for academic coursework purposes.
