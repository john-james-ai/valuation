# A Protocol for Robust Time Series Model Selection and Evaluation

The development of a high-performing and generalizable time series forecasting model requires a rigorous and systematic evaluation protocol. The primary objective of such a protocol is to mitigate the dual risks of overfitting to the training data and selection bias during hyperparameter optimization. This document specifies an end-to-end methodology for training, validating, and testing multiple candidate algorithms to select a final, production-ready model.

***

## 1. Temporal Data Partitioning

The foundational step is a strict, chronologically-ordered partitioning of the dataset into three distinct subsets. Maintaining temporal order is critical to prevent data leakage, where information from the future is used to train or select a model, leading to an overly optimistic and invalid estimate of its performance.

* **Training Set (e.g., 70%):** This is the largest partition and is used exclusively for the initial parameter estimation of each candidate model. The model learns the underlying patterns, trends, seasonality, and feature relationships from this data.
* **Validation Set (e.g., 15%):** This subsequent partition is used for hyperparameter optimization. By evaluating the performance of different model configurations on this unseen data, we can tune the model's hyperparameters to improve its ability to generalize. The error metric on this set guides the tuning process.
* **Test Set (Hold-out, e.g., 15%):** This final partition is held in reserve and must remain completely untouched during all training and validation phases. Its sole purpose is to provide a final, unbiased estimate of the selected model's generalization error. Performance on this set determines the ultimate winner of the model selection process.

***

## 2. Candidate Model Specification

A multi-algorithm approach is employed to explore different inductive biases and model architectures. A typical selection of candidates would include models from distinct classes to ensure a comprehensive search of potential solutions.

* **Statistical Models (e.g., SARIMA):** These models serve as a robust baseline. They are highly effective for time series that exhibit clear autocorrelation structures and regular seasonality. Their parameters are interpretable, providing insight into the underlying data-generating process.
* **Gradient Boosted Trees (e.g., XGBoost, LightGBM):** These are powerful, non-linear ensemble models. They excel at capturing complex interactions and can readily incorporate a large number of exogenous features (e.g., promotions, holidays, economic indicators) and lagged variables.
* **Decompositional Models (e.g., Prophet):** These models are designed to be robust to missing data and trend changes. They perform well on time series with multiple, strong seasonalities (e.g., weekly, yearly) and a large number of irregular events like holidays.

***

## 3. Hyperparameter Optimization via Validation

For each candidate algorithm, a search for the optimal hyperparameter configuration is conducted. This process is governed by the model's performance on the **validation set**.

The objective is to find the set of hyperparameters that minimizes a pre-defined error metric (e.g., Mean Absolute Error, Root Mean Squared Error) on the validation data. Common search strategies include Grid Search, Random Search, or more sophisticated methods like Bayesian Optimization.

The output of this stage is a single "champion" model for each algorithm class—the specific configuration (e.g., the best-tuned XGBoost model) that demonstrated the lowest generalization error during the validation phase.

***

## 4. Final Model Selection on the Hold-out Test Set

The champion model from each algorithm class is now subjected to a final evaluation on the **hold-out test set**.

This is a one-time-only contest. Each optimized model generates a forecast for the test period, and its performance is calculated. The model that achieves the lowest error on this final, unseen data is selected as the definitive, production-ready model.

The error score from this final test serves as the most reliable estimate of the model's expected performance in a real-world production environment. No further tuning or model adjustments should be made after this step, as doing so would invalidate the test set.

***

## 5. Advanced Validation: Rolling Forecast Origin

For a more robust estimate of generalization error, especially with non-stationary data, a simple train-validate-test split can be augmented with a **rolling forecast origin** methodology (also known as time series cross-validation). In this procedure, the model is iteratively retrained on an expanding or sliding window of training data and evaluated on subsequent time steps. By averaging the performance across these multiple forecast windows, we can obtain a more stable and reliable measure of the model's true predictive power, better simulating a real-world deployment scenario where the model is periodically updated with new data.