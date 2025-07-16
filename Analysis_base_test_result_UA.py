#!/usr/bin/env python
"""Analysis_base_test_result_UA.py

Post-processing analysis for the Ukrainian EMOBENCH-UA experiments.

Methodology:
1.  Loads the artifacts saved by `Base_test_emotions_hybrid_UA.py`.
2.  Ablation Study: Trains a classifier ONLY on the generated Kunchenko
    features to measure their standalone predictive power.
3.  Significance Test: Performs a bootstrap test to verify if the
    improvement of the hybrid model over the baseline is statistically
    significant.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

def calculate_macro_f1(y_true, y_pred):
    """Calculates macro-F1 for a multi-label task."""
    n_classes = y_true.shape[1]
    class_f1_scores = []
    for i in range(n_classes):
        tp = np.sum((y_true[:, i] == 1) & (y_pred[:, i] == 1))
        fp = np.sum((y_true[:, i] == 0) & (y_pred[:, i] == 1))
        fn = np.sum((y_true[:, i] == 1) & (y_pred[:, i] == 0))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        class_f1_scores.append(f1)
    return np.mean(class_f1_scores)

def run_ablation_study(X_train, y_train, X_test, y_test):
    """Trains a classifier on Kunchenko features alone."""
    print("\n" + "="*60 + "\nABLATION STUDY\n" + "="*60)
    print("Training a classifier ONLY on the 7 Kunchenko features...")

    pipeline_ablation = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', OneVsRestClassifier(LogisticRegression(solver='liblinear', random_state=42, C=1.0)))
    ])
    pipeline_ablation.fit(X_train, y_train)
    y_pred_ablation = pipeline_ablation.predict(X_test)
    f1_ablation = calculate_macro_f1(y_test, y_pred_ablation)

    print("\nResult:")
    print(f"Macro F1 on Kunchenko-only features: {f1_ablation:.4f}")
    if f1_ablation > 0.2:
        print("CONCLUSION: The Kunchenko features carry significant predictive power on their own,")
        print("as the classification quality is substantially higher than random chance.")
    else:
        print("CONCLUSION: The Kunchenko features are weak predictors on their own; their value is in combination.")

def run_bootstrap_test(y_true, y_pred_model1, y_pred_model2, n_iterations=1000):
    """Performs a bootstrap test for statistical significance."""
    print("\n" + "="*60 + "\nSTATISTICAL SIGNIFICANCE TEST (BOOTSTRAP)\n" + "="*60)
    print(f"Running {n_iterations} iterations to compare the Hybrid model vs. Baseline...")

    original_diff = calculate_macro_f1(y_true, y_pred_model2) - calculate_macro_f1(y_true, y_pred_model1)
    size = len(y_true)
    diff_scores = []
    for _ in range(n_iterations):
        indices = resample(np.arange(size))
        f1_model1 = calculate_macro_f1(y_true[indices], y_pred_model1[indices])
        f1_model2 = calculate_macro_f1(y_true[indices], y_pred_model2[indices])
        diff_scores.append(f1_model2 - f1_model1)

    diff_scores = np.array(diff_scores)
    p_value = np.mean(diff_scores <= 0)
    alpha = 0.05
    lower_bound = np.percentile(diff_scores, 100 * (alpha / 2))
    upper_bound = np.percentile(diff_scores, 100 * (1 - alpha / 2))

    print("\nResults:")
    print(f"Original Macro F1 Improvement: {original_diff:+.4f}")
    print(f"95% Confidence Interval for Improvement: [{lower_bound:+.4f}, {upper_bound:+.4f}]")
    print(f"p-value: {p_value:.4f}")

    if p_value < 0.05:
        print("\nCONCLUSION: p-value < 0.05. The improvement is STATISTICALLY SIGNIFICANT.")
        print("The probability of observing such an improvement by chance is very low.")
    else:
        print("\nCONCLUSION: p-value >= 0.05. The improvement is NOT statistically significant.")
        print("We cannot confidently reject the hypothesis that the improvement was due to chance.")

def main():
    """Main function to load results and run analyses."""
    try:
        data = np.load("experiment_outputs.npz")
    except FileNotFoundError:
        print("Error: 'experiment_outputs.npz' not found.")
        print("Please run '1_ukr_emotions_hybrid_model.py' first to generate this file.")
        return

    run_ablation_study(
        X_train=data['X_train_kunchenko'], y_train=data['y_train'],
        X_test=data['X_test_kunchenko'], y_test=data['y_test']
    )
    run_bootstrap_test(
        y_true=data['y_test'],
        y_pred_model1=data['y_pred_base'],
        y_pred_model2=data['y_pred_hybrid']
    )

if __name__ == "__main__":
    main()