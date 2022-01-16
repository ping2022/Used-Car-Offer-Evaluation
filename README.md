# Used Car Offer Evaluation
Summer Project Using R


# Dataset
Kaggle: https://www.kaggle.com/smritisingh1997/car-salescsv


# Introduction
The objective of this project is to predict price value for used car using different machine learning algorithms in R.

# Models
We take log transform on price, use following models to predict price with 7 features. And finally we obtain results as below:

- linear regression: RMSE of 0.299 under OLS and 0.3147 with 10-fold cross validation.
- lasso: the best model in terms of RMSE is a simple OLS model with no penalty (lamba = 0)
- knn: RMSE of 0.3447473 with knn
- boosting: RMSE of 0.265289 with boosting
- regression tree: RMSE of 0.3897077 with regression tree
- bagging: RMSE of 0.3428916 with bagging
- random forest: RMSE of 0.2373476 with random forest

Looks like our 7 predictor Random Forest model performs the best.
