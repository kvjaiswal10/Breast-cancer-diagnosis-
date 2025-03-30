# Breast-cancer-diagnosis-

## Overview
This project implements a Support Vector Machine (SVM) model to classify breast cancer as benign or malignant based on extracted features. The dataset used for this project is the Wisconsin Diagnostic Breast Cancer (WDBC) dataset.

## Dataset
- **Input Data**: [`wdbc.data`](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
- **Columns**:
  - `ID`: Patient ID (removed during preprocessing)
  - `Diagnosis`: Malignant (M) or Benign (B), mapped to 1 and 0 respectively
  - `Feature_1` to `Feature_30`: Numeric features extracted from cell nuclei images

## Preprocessing Steps
1. Load the dataset and assign appropriate column names.
2. Remove the `ID` column as it is not a predictive feature.
3. Convert the `Diagnosis` column to numeric values (`M` → `1`, `B` → `0`).
4. Check for missing values and handle them if necessary.
5. Split the dataset into training (80%) and testing (20%) sets.
6. Standardize the features using `StandardScaler`.

## Model
- **Algorithm**: Support Vector Machine (SVM)
- **Kernel**: Radial Basis Function (RBF)
- **Hyperparameters**:
  - `C=1.0`: Regularization parameter
  - `gamma="scale"`: Kernel coefficient
- **Evaluation Metrics**:
  - Accuracy
  - Classification Report (Precision, Recall, F1-Score)
  - Confusion Matrix


