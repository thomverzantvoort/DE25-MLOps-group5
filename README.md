## Design \& implementation

### ML Pipeline (Lab 4/5)

Google Vertex AI implementation which exists more or less of:

1. Data Ingestion
2. Train Test split
3. Model 1 Training
4. Model 2 Training
5. Model Comparing
6. Model Deployment Evaluation (best trained model vs current model)
7. Model Checking Approval
8. Trigger CI/CD Pipeline

### CI/CD (Lab 3)

A set of CI-CD pipelines (with Google Cloud Build) is used to automate:

1. building of training pipeline components, prediction service component, and prediction UI component,
2. compiling and publishing the pipeline, and
3. (re)-executing the pipeline (i.e., continuous training)

### Prediction and Serving Components (Lab 1-2)

This exists of 2 parts:

* Consumer App / Frontend (predicition-UI component)
* Serving Infrastructure, host the model and expose it as API (predicition-API component)

## ML Case

https://www.kaggle.com/datasets/nabihazahid/spotify-dataset-for-churn-analysis/data
https://www.kaggle.com/code/lauraalonso/spotify-2025-eda-prediction-models
https://github.com/Laura-Alonso/Spotify-Analysis-Dataset-2025



### About the Dataset:

Each row represents a unique Spotify user.
The outcome variable: is\_churned → Target variable (0 = Active, 1 = Churned)

The goal of this project is to predict whether a Spotify user will churn (cancel their subscription) or remain active.

Features (This will be the input for the Prediction UI):

* age → User age
* listening\_time → Minutes spent listening per day
* songs\_played\_per\_day → Number of songs played daily
* skip\_rate → Percentage of songs skipped
* ads\_listened\_per\_week → Number of ads heard per week

In the notebook we have 4 models to choose from:

* Logistic Regression
* Decision Tree (highest recall)
* Random Forest
* XGBoost (best balanced)

## Tasks

### Asisgnment (Code)

* MLOpsPipeline (Notebook)
* Setup Google Cloud Project
* CI/CD Pipeline
* Connecting parts together
* Application (Prediction UI)
* Application (Prediction API)

### Report

* Overview of the ML Application
* Design and Implementation of the MLOps System
* Reflection on the Design and Implementation of the MLOps System
