# PS4E4 - Abalone Regression 🐚🦢

## This is my current submission to the Regression with an Abalone Dataset competition in Kaggle Playground Series.

### My submission: https://www.kaggle.com/code/ednaxx/ps4e4-abalone-regression

#### Current best Public Score: 0.14804

## Overview

"Welcome to the 2024 Kaggle Playground Series! We plan to continue in the spirit of previous playgrounds, providing interesting an approachable datasets for our community to practice their machine learning skills, and anticipate a competition each month.

Your Goal: The goal of this competition is to predict the age of abalone from various physical measurements."

## Evaluation

The evaluation metric for this competition is [**Root Mean Squared Logarithmic Error**](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.root_mean_squared_log_error.html).

The **RMSLE** is calculated as:

$$ \sqrt{\frac{1}{n} \sum_{i=1}^n (log(1 + ŷ_i) - log(1 + y_i))^2} $$
