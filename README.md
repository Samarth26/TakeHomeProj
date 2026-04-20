# Census Income Classification & Segmentation

## Introduction

This is a submission of a take home project. First preference would be to use the Docker setup to run the code and reproduce the results. However, if you prefer, you can also pip install the requirements and run the python scripts directly. 

## Docker

Build and run both scripts in one command:

```bash
docker build -t census-ml .
docker run -v $(pwd)/images:/app/images -v $(pwd)/outs:/app/outs census-ml
```

All output plots will be saved to the local `images/` directory.
All logs will be saved to the local `outs/` directory.

## Requirements

Python 3.11+

```bash
pip install -r requirements.txt
```

## Running the Code

**Classification model** (income prediction ≤50K vs >50K):
```bash
python pipeline.py
```

**Segmentation model** (demographic clustering):
```bash
python segmentation.py
```

Both scripts expect `census-bureau.data` and `census-bureau.columns` to be in the same directory.


## Outputs

`pipeline.py` produces:
- `numerical_distributions.png` — numeical variables distribution by label
- `feature_importance_baseline.png` — baseline model feature importance
- `feature_importance_selected.png` — selected-feature model feature importance
- `feature_importance_colinearity_check.png` — collinearity check feature importance
- `pipeline.log` — prints from the pipeline script

`segmentation.py` produces:
- `pca_explained_variance.png` — variance explained by PCA components
- `pca_2d.png` — 2D PCA projection
- `tsne_raw.png` — t-SNE projection
- `tsne_clusters.png` — t-SNE colored by cluster
- `kmeans_elbow_silhouette.png` — elbow and silhouette plots
- `cluster_profile_numerical.png` — numerical feature heatmap per cluster
- `cluster_profile_categorical.png` — categorical feature heatmap per cluster
-  `segmentation.log` — prints from the segmentation script

## Project Report

See `Analysis.pdf` for the full project report including data exploration, model architecture, evaluation, findings, and business recommendations.

Notebooks (`Obj_1.ipynb`, `clean_Obj_1.ipynb`, `Segmentation.ipynb`) contain the exploratory analysis and iterative model development process.
