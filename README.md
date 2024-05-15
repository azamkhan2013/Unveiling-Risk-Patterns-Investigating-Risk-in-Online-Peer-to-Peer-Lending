# Unveiling-Risk-Patterns-Investigating-Risk-in-Online-Peer-to-Peer-Lending


## Overview

This repository contains the code and findings from my dissertation titled "Unveiling Risk Patterns: Investigating Risk in Online Peer-to-Peer Lending," completed at the University of Essex under the supervision of Dr. Dmitry V. Savostyanov. The study focuses on the risk assessment in peer-to-peer (P2P) lending platforms, utilizing machine learning models to predict loan defaults.

## Table of Contents

- [Introduction](#introduction)
- [Literature Review](#literature-review)
- [Data Exploration and Insights](#data-exploration-and-insights)
- [Methodology](#methodology)
- [Results](#results)
- [Conclusions](#conclusions)
- [Usage](#usage)
- [Requirements](#requirements)
- [Acknowledgments](#acknowledgments)

## Introduction

P2P lending platforms like LendingClub have revolutionized the lending space by connecting lenders directly with borrowers. However, this bypasses traditional bank safety checks, necessitating robust risk analysis. This dissertation investigates the use of machine learning algorithms for predicting loan defaults on P2P lending platforms.

## Literature Review

The literature review discusses the growth of P2P lending and its significance in the post-recession era, highlighting the benefits and risks from both lender and borrower perspectives. Several studies employing various machine learning models for credit risk assessment in P2P lending are reviewed.

## Data Exploration and Insights

We utilized the LendingClub dataset from 2007 to 2018, comprising over 2 million observations and 151 features. Our analysis included:

- Univariate Analysis: Examining individual variable characteristics.
- Bivariate Analysis: Studying relationships between variables.
- Feature Extraction: Selecting relevant features for the predictive models.

## Methodology

### Handling Imbalanced Data

To address the class imbalance in our dataset (87% Fully Paid, 13% Charged Off), we employed:

- Class Weights: Penalizing incorrect predictions in the minority class more heavily.
- Random Under Sampling (RUS): Balancing the class distribution by undersampling the majority class.

### Data Splitting and Scaling

- Train-Test Split: Stratified partition to maintain class proportions in both training and test sets.
- Feature Scaling: MinMaxScaler to normalize feature values.

### Machine Learning Models

We evaluated four machine learning models:

1. **Decision Tree**
2. **Random Forest**
3. **Gradient Boosting**
4. **XGBoost**

### Evaluation Metrics

Given the imbalanced data, we used the following metrics:

- Recall
- F2-Score
- AUC-PR (Area Under the Precision-Recall Curve)
- Classification Report

## Results

The models' performance varied, with Random Forest achieving the highest recall and F2-Score, while XGBoost had the best AUC-PR. Detailed results and performance metrics are provided in the Results section.

## Conclusions

Random Forest and XGBoost models demonstrated robust performance for predicting loan defaults on P2P lending platforms. Effective sampling techniques are crucial for handling imbalanced datasets and improving model reliability.




## Requirements

- `numpy==1.20.3`
- `pandas==1.2.4`
- `scikit-learn==0.24.2`
- `xgboost==1.4.2`
- `matplotlib==3.4.2`
- `seaborn==0.11.1`
- `jupyter==1.0.0`

Install the required packages using:
```sh
pip install -r requirements.txt

