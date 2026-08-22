# Experiment 02 — Early Propagation Prediction

## Objective

To determine whether early propagation behavior can be used to predict the eventual size of a misinformation cascade.

The experiment evaluates propagation information available after 1 hour, 6 hours, and 24 hours.

## Dataset

Dataset: FibVID

Propagation cascades: 295

Propagation records: 221,253

Target:

`log_final_cascade_size`

The logarithmic transformation was used because final cascade sizes are highly right-skewed.

## Input Features

For each temporal window, the model uses:

- Number of tweets
- Number of unique users
- Maximum propagation depth
- Total likes
- Total retweets
- Engagement

No final cascade information was used as an input feature.

Target leakage was explicitly prevented by removing features derived from final cascade size.

## Models Evaluated

1. Median baseline
2. Ridge Regression
3. Random Forest Regression

## Initial Holdout Results

### 1-hour

Random Forest:

- Log MAE: 1.4379
- Log RMSE: 1.7278
- Log R²: 0.5255

### 6-hour

Random Forest:

- Log MAE: 1.3286
- Log RMSE: 1.5471
- Log R²: 0.6196

### 24-hour

Random Forest:

- Log MAE: 1.3262
- Log RMSE: 1.5631
- Log R²: 0.6117

## 5-Fold Cross-Validation

| Window   | Mean R² | Std R² | Mean MAE | Mean RMSE |
| -------- | ------: | -----: | -------: | --------: |
| 1 hour   |  0.2999 | 0.1900 |   1.5796 |    1.9822 |
| 6 hours  |  0.3705 | 0.1914 |   1.4860 |    1.8726 |
| 24 hours |  0.3891 | 0.1878 |   1.4378 |    1.8440 |

## Interpretation

The results demonstrate that early propagation behavior contains predictive information about eventual cascade size.

The 1-hour model provides the earliest warning but has the lowest predictive performance.

The 6-hour model provides a useful balance between early intervention and prediction quality.

The 24-hour model achieves the strongest average cross-validated performance, but provides a less timely intervention opportunity.

The relatively high standard deviation across folds indicates variability in prediction performance. This is expected given the relatively small number of available propagation cascades and the highly skewed cascade-size distribution.

## Model Selection Decision

Random Forest was selected as the propagation prediction model.

A deeper GNN-based propagation predictor was not adopted at this stage because the available cascade-level sample size is limited and the current Random Forest already demonstrates meaningful predictive signal.

The system will therefore use:

- 1-hour Random Forest for early warning
- 6-hour Random Forest as the primary intervention prediction model
- 24-hour Random Forest as a later high-confidence forecast

## Research Significance

The experiment establishes that misinformation propagation can be treated as a forecasting problem rather than only a fake/real classification problem.

This provides the foundation for the intervention simulation component of the proposed system.
