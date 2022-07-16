# PRD : Credit Card Default

## Introduction
This notebook was created to learn basic techniques of data manipulation and machine learning. The idea is to use the dataset UCI_Credit_Card to improve basic skills of data cleaning, data analysis, data visualization and machine learning. It is primarily intended to help myself understanding what to do and how. Any feedback is welcome.

## Problem Alignment
### The Problem

To increase business opportunities by giving approval to more credit card applications, although at the same time the institution will also want to lower the credit default numbers. We need to come up with a solution of how to predict if an application will likely to default or not in the future. Based on information on default payments, demographic factors, credit data, history of payment and bill statements of credit card clients in Taiwan from April 2005 to September 2005, we will develop a robust model to predict if an application will be default or not to minimise the risk of future accounts default.

### High-Level Approach

We intend to create a predictor engine using model selection, to compare a few models and select the best one. 

### Goals & Success

The engine will be presented in a visual interface, which will give a prediction number 1 (Will default) or 0 (Will not default) to any given customer data feed into the engine.

## Solution Alignment

### Key Solution

In this project we will use “Default Payments of Credit Card Clients in Taiwan from 2005 Dataset”. The dataset provided by UCI Machine Learning (https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients). Consists of 25 variables. We will compare two or three models and select the best one based on several metrics such as accuracy score, ROC-AUC and recall score.

### Launch Readiness

### Key Milestones
Date | Milestone | Description
-----|-----------|-------------
17-Jun-2022 | Feature Update | EDA, Pre-processing, Feature Selection, Data ready for modelling
01-Jul-2022 | Modelling | Expected metric result: xx%
22-Jul-2022 | Serving Ready | Backend and model image is ready to deploy

## Artefacts
Dataset: https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset 
## References
https://www.kaggle.com/code/lucabasa/credit-card-default-a-very-pedagogical-notebook