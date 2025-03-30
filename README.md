# Breast Cancer Molecular Subtyping Analysis

This project performs comprehensive analysis of breast cancer gene expression data using the PAM50 classifier, including exploratory data analysis, clustering, and classification.

## Dataset Overview

The dataset contains:
- 1016 breast cancer samples
- Expression values for 50 PAM50 classifier genes
- Molecular subtype classifications:
  - luminal-A: 543 samples
  - luminal-B: 201 samples
  - basal-like: 190 samples
  - HER2-enriched: 82 samples

## Analysis Pipeline

### 1. Data Preprocessing
- Handling missing values (median imputation)
- Outlier treatment (IQR-based clipping)
- Standardization (StandardScaler)

### 2. Exploratory Analysis
- Gene expression distribution visualization (boxplots)
- Principal Component Analysis (PCA) for dimensionality reduction
- Hierarchical clustering (dendrogram visualization)

### 3. Clustering Techniques
- Agglomerative Hierarchical Clustering
- K-Means clustering with elbow method for optimal cluster determination

### 4. Classification Models
- K-Nearest Neighbors (KNN) with cross-validation
- Decision Tree classifier
- Naive Bayes classifier
- Performance metrics (accuracy, precision, recall, F1-score)

## Requirements

- Python 3.x
- Jupyter Notebook
- Required packages:
pandas
numpy
matplotlib
seaborn
scikit-learn
scipy



## How to Run

1. Clone this repository
2. Install requirements: `pip install -r requirements.txt`
3. Open the Jupyter notebook: `jupyter notebook breast_cancer_analysis.ipynb`
4. Run all cells

## Key Visualizations

1. **Gene Expression Boxplots**: Shows distribution of expression values for all 50 genes
2. **PCA Plot**: 2D visualization of samples colored by molecular subtype
3. **Dendrogram**: Hierarchical clustering of samples
4. **Cluster Visualizations**: Results from both hierarchical and K-means clustering
5. **Elbow Method Plot**: For determining optimal K in K-means

## Results

Classification performance metrics are saved to `classification_results.csv`, including:
- Accuracy
- Precision
- Recall
- F1-score

For all evaluated models:
- KNN (with varying k values)
- Decision Tree
- Naive Bayes

## Future Work

Potential extensions:
- Incorporate clinical data for survival analysis
- Implement more advanced classification models (SVM, Random Forest)
- Perform differential expression analysis between subtypes
- Add interactive visualizations

## References

- [PAM50 Classifier](https://en.wikipedia.org/wiki/PAM50)
- [Original PAM50 paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3040961/)
- [scikit-learn documentation](https://scikit-learn.org/stable/)
