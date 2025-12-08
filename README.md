# Gaussian Naive Bayes Classifier — DATA 2060 Project  

## Table of Contents
- [Overview](#overview)
- [Environment-Setup](#environment-setup)
- [Model-Description](#model-description)
- [Results](#results)
- [Model Performance Comparison](#model-performance-comparison)
- [Authors](#authors)
---

## Overview

This repository implements a **Gaussian Naive Bayes (GNB)** classifier **from scratch** and compares it against **scikit-learn’s GaussianNB** on the Kaggle Diabetes dataset.

The project includes:

- Custom GNB implementation  
- Full unit testing suite using pytest  
- Model evaluation on real medical data  
- Performance comparison with sklearn  
- Reproducible Conda environment  
- Visualizations and metrics in notebook
- 
---

## Environment Setup

### Conda Environment (`environment.yaml`)

```yaml
name: data2060
channels:
  - conda-forge
dependencies:
  - python = 3.12.11
  - matplotlib = 3.10.5
  - pandas = 2.3.2
  - scikit-learn = 1.7.1
  - numpy = 2.3.2
  - pytorch = 2.7.1
  - jupyter
  - pytest = 8.4.1
  - quadprog
  - seaborn = 0.13.2
prefix: /opt/conda
```

## Model Description

### Gaussian Naive Bayes — Training

For each class k:

**Prior probability**
$P(y=k) = \frac{n_k}{N}$

**Feature mean and variance**
$$\mu_{k,j} = \text{mean}(x_j \mid y=k), \quad
\sigma_{k,j}^2 = \text{var}(x_j \mid y=k)$$

---

### Gaussian Naive Bayes — Prediction

For a sample \(x\), compute the joint log-likelihood:

$$\log P(y=k \mid x) =
\log P(y=k)
-\frac12 \sum_j \left[
\log(2\pi\sigma^2_{k,j}) +
\frac{(x_j - \mu_{k,j})^2}{\sigma^2_{k,j}}
\right]$$

Predict the class with the **maximum posterior probability**.

## Results

Evaluation on the Diabetes dataset yields:

| Model            | Accuracy | Precision | Recall  | F1 Score |
|------------------|----------|-----------|---------|----------|
| **Sklearn GNB**  | 0.7727   | 0.6429    | 0.5745  | 0.6067   |
| **Custom GNB**   | 0.7727   | 0.6429    | 0.5745  | 0.6067   |

Custom implementation **matches sklearn exactly**  
Confusion matrices available in the notebook  


## Model Performance Comparison

To benchmark our custom Gaussian Naive Bayes implementation, we compared its performance against several classical machine-learning models on the Diabetes dataset.

| Model            | Accuracy | Precision | Recall  | F1 Score |
|------------------|----------|-----------|---------|----------|
| **Sklearn GNB**  | 0.7727   | 0.6429    | 0.5745  | 0.6067   |
| **Custom GNB**   | 0.7727   | 0.6429    | 0.5745  | 0.6067   |
| **Logistic Regression**   | 0.8052   | 0.7179    | 0.5957  | 0.6512   |
| **Decision Tree**   | 0.7013   | 0.5091    | 0.5957  | 0.5490   |

Key Observations
1. Our Custom GNB matches sklearn’s GNB exactly, confirming correctness.
2. Logistic Regression achieves the best overall performance, especially in accuracy and F1.
3. Decision Tree overfits slightly (higher recall but low precision), resulting in the weakest F1 score.
4. GNB remains a strong, simple baseline with competitive performance given its assumptions.


## Authors

| Name | Contact |
|------|---------|
| **Lin Zhou** | lin_zhou@brown.edu |
| **Linzhuo Zhang** | linzhuo_zhang@brown.edu |
| **Zhiwei He** | zhiwei_he@brown.edu |
| **Xiaoqing Yao** | xiaoqing_yao@brown.edu |



