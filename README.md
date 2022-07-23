# PRD : Credit Card Default

## Introduction
This notebook was created to learn basic techniques of data manipulation and machine learning. The idea is to use the dataset UCI_Credit_Card to improve basic skills of data cleaning, data analysis, data visualization and machine learning. It is primarily intended to help myself understanding what to do and how. Any feedback is welcome.

## Problem Alignment

This is Classification Machine Learning case. The result will be default or not as our target variables. The business problem we are trying to solve is to create model that can predict the next payment will be about default or not, using last 6 months transaction.

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

## Variables description
There are 25 variables:

ID: ID of each client
LIMIT_BAL: Amount of given credit in NT dollars (includes individual and family/supplementary credit
SEX: Gender (1=male, 2=female)
EDUCATION: (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown)
MARRIAGE: Marital status (1=married, 2=single, 3=others)
AGE: Age in years
PAY_0: Repayment status in September, 2005 (-1=pay duly, 1=payment delay for one month, 2=payment delay for two months, … 8=payment delay for eight months, 9=payment delay for nine months and above)
PAY_2: Repayment status in August, 2005 (scale same as above)
PAY_3: Repayment status in July, 2005 (scale same as above)
PAY_4: Repayment status in June, 2005 (scale same as above)
PAY_5: Repayment status in May, 2005 (scale same as above)
PAY_6: Repayment status in April, 2005 (scale same as above)
BILL_AMT1: Amount of bill statement in September, 2005 (NT dollar)
BILL_AMT2: Amount of bill statement in August, 2005 (NT dollar)
BILL_AMT3: Amount of bill statement in July, 2005 (NT dollar)
BILL_AMT4: Amount of bill statement in June, 2005 (NT dollar)
BILL_AMT5: Amount of bill statement in May, 2005 (NT dollar)
BILL_AMT6: Amount of bill statement in April, 2005 (NT dollar)
PAY_AMT1: Amount of previous payment in September, 2005 (NT dollar)
PAY_AMT2: Amount of previous payment in August, 2005 (NT dollar)
PAY_AMT3: Amount of previous payment in July, 2005 (NT dollar)
PAY_AMT4: Amount of previous payment in June, 2005 (NT dollar)
PAY_AMT5: Amount of previous payment in May, 2005 (NT dollar)
PAY_AMT6: Amount of previous payment in April, 2005 (NT dollar)
default.payment.next.month: Default payment (1=yes, 0=no)

## Artefacts
Dataset: https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset

![workflow_plds](https://user-images.githubusercontent.com/65161523/180449653-40990757-87fe-4f75-bf1f-9248ea5b97e1.jpg)

## References
https://www.kaggle.com/code/lucabasa/credit-card-default-a-very-pedagogical-notebook
