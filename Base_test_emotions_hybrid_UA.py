#!/usr/bin/env python
"""Base_test_emotions_hybrid_UA.py

Main experiment for the Ukrainian EMOBENCH-UA dataset.

Methodology:
1.  Establishes a baseline by applying optimized thresholds to the raw
    probabilities of the `ukr-detect/ukr-emotions-classifier` model.
2.  Generates "statistical-geometric" features using the Kunchenko
    decomposition method with optimized basis functions (powers=(0.5, 1/3, 0.25)).
3.  Trains a hybrid model (Logistic Regression) on the combined probabilities
    and Kunchenko features.
4.  Compares the hybrid model against the baseline.
5.  Saves predictions and features to `experiment_outputs.npz` for further analysis,
    such as ablation studies and significance testing.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from tqdm.auto import tqdm
from typing import Tuple

# --- Kunchenko Feature Generator (Refactored) ---
class KunchenkoFeatureGenerator(BaseEstimator, TransformerMixin):
    def __init__(self, powers: Tuple[float, ...], lambda_reg=0.01, epsilon=1e-8):
        self.powers = powers
        self.lambda_reg = lambda_reg
        self.epsilon = epsilon

    def fit(self, X, y):
        self.basis_functions_ = [
            lambda x, p=p, eps=self.epsilon: np.sign(x) * (np.abs(x) + eps)**p
            for p in self.powers
        ]
        self.n_basis_funcs_ = len(self.basis_functions_)
        self.classes_ = np.arange(y.shape[1])
        n_features = X.shape[1]
        self.models_ = {}
        print("Training Kunchenko feature generator...")
        for c in self.classes_:
            pure_indices = np.where((y.sum(axis=1) == 1) & (y[:, c] == 1))[0]
            X_class = X[pure_indices] if len(pure_indices) >= 2 else X[np.where(y[:, c] == 1)[0]]
            if len(X_class) < 2: continue

            class_model = {}
            for feature_idx in range(n_features):
                signals = X_class[:, feature_idx]
                basis = self._apply_basis(signals)
                E_x, E_phi = np.mean(signals), np.mean(basis, axis=0)
                centered_signals, centered_basis = signals - E_x, basis - E_phi
                F = centered_basis.T @ centered_basis / len(signals) + self.lambda_reg * np.eye(self.n_basis_funcs_)
                b = centered_basis.T @ centered_signals / len(signals)
                try: K = np.linalg.solve(F, b)
                except np.linalg.LinAlgError: K = np.linalg.pinv(F) @ b
                class_model[feature_idx] = {'K': K, 'E_x': E_x, 'E_phi': E_phi}
            self.models_[c] = class_model
        return self

    def _apply_basis(self, signal_data):
        return np.stack([func(signal_data) for func in self.basis_functions_], axis=-1)

    def transform(self, X):
        print("Generating Kunchenko features...")
        features = np.zeros((X.shape[0], len(self.classes_)))
        for i in tqdm(range(X.shape[0]), desc="Generating Kunchenko Features"):
            for c_idx, c in enumerate(self.classes_):
                if c not in self.models_: continue
                total_error = 0
                for feature_idx in range(X.shape[1]):
                    signal_1d = X[i, feature_idx]
                    model = self.models_[c][feature_idx]
                    basis_matrix = self._apply_basis(signal_1d)
                    reconstructed_signal = model['E_x'] + (basis_matrix - model['E_phi']) @ model['K']
                    total_error += (signal_1d - reconstructed_signal)**2
                features[i, c_idx] = np.log(total_error / X.shape[1] + self.epsilon)
        return features

# --- Utility Functions ---
def get_outputs_batched(texts, model, tokenizer, device, batch_size=32, get_embeds=False):
    print(f"Processing {len(texts)} texts in batches of {batch_size}...")
    all_outputs = []
    model.eval()
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing Batches"):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=128).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            output = outputs.last_hidden_state[:, 0, :].cpu().numpy() if get_embeds else torch.sigmoid(outputs.logits).cpu().numpy()
            all_outputs.append(output)
    return np.vstack(all_outputs)

def calculate_metrics(y_true, y_pred):
    metrics = {}
    n_classes = y_true.shape[1]
    class_f1_scores = []
    for i in range(n_classes):
        tp = np.sum((y_true[:, i] == 1) & (y_pred[:, i] == 1))
        fp = np.sum((y_true[:, i] == 0) & (y_pred[:, i] == 1))
        fn = np.sum((y_true[:, i] == 1) & (y_pred[:, i] == 0))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        support = np.sum(y_true[:, i])
        metrics[i] = {'precision': precision, 'recall': recall, 'f1-score': f1, 'support': int(support)}
        class_f1_scores.append(f1)
    macro_f1 = np.mean(class_f1_scores)
    return metrics, macro_f1

def print_classification_report(metrics, target_names, title="Classification Report"):
    print(f"\n--- {title} ---")
    print(f"{'':<12} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}")
    print("-" * 55)
    for i, name in enumerate(target_names):
        m = metrics[i]
        print(f"{name:<12} {m['precision']:>10.2f} {m['recall']:>10.2f} {m['f1-score']:>10.2f} {m['support']:>10}")
    print("-" * 55)

def main():
    # Setup
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    model_name = "ukr-detect/ukr-emotions-classifier"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_clf = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model_emb = AutoModel.from_pretrained(model_name).to(device)
    dataset = load_dataset("ukr-detect/ukr-emotions-binary")
    label_order = [model_clf.config.id2label[i] for i in range(len(model_clf.config.id2label))]

    def prepare_data(dataset_split, order):
        df = dataset_split.to_pandas()
        texts = df['text'].tolist()
        base_emotions = [em for em in order if em != 'None']
        df['None'] = (df[base_emotions].sum(axis=1) == 0).astype(int)
        return texts, df[order].to_numpy()

    X_train_texts, y_train = prepare_data(dataset['train'], label_order)
    X_test_texts, y_test = prepare_data(dataset['test'], label_order)

    # --- EXPERIMENT A: BASELINE ON RAW PROBABILITIES ---
    print("\n" + "="*60 + "\nEXPERIMENT A: BASELINE ON RAW MODEL PROBABILITIES\n" + "="*60)
    X_test_probs = get_outputs_batched(X_test_texts, model_clf, tokenizer, device)
    thresholds_dict = {"Joy": 0.35, "Fear": 0.5, "Anger": 0.25, "Sadness": 0.5, "Disgust": 0.3, "Surprise": 0.25, "None": 0.35}
    thresholds = np.array([thresholds_dict[label] for label in label_order])
    y_pred_base = (X_test_probs >= thresholds).astype(int)
    metrics_base, f1_base = calculate_metrics(y_test, y_pred_base)
    print(f"\nMacro F1 with optimized thresholds: {f1_base:.4f}")
    print_classification_report(metrics_base, label_order, "Baseline Model Report")

    # --- EXPERIMENT B: HYBRID MODEL CLASSIFICATION ---
    print("\n" + "="*60 + "\nEXPERIMENT B: HYBRID MODEL CLASSIFICATION\n" + "="*60)
    X_train_probs = get_outputs_batched(X_train_texts, model_clf, tokenizer, device)
    X_train_emb = get_outputs_batched(X_train_texts, model_emb, tokenizer, device, get_embeds=True)
    X_test_emb = get_outputs_batched(X_test_texts, model_emb, tokenizer, device, get_embeds=True)

    # Using optimal powers found in grid search: n=4, alpha=0.0 -> (1/2, 1/3, 1/4)
    # The original formula for n=4, alpha=0 is 1/2, 1/3, 1/4.
    # The best search result was n=4, alpha=0.0, which means powers 1/2, 1/3, 1/4.
    # Let's use the best found configuration.
    optimal_powers = (0.5, 1/3, 0.25) 
    print(f"Using optimal basis functions with powers: {optimal_powers}")
    kunchenko_extractor = KunchenkoFeatureGenerator(powers=optimal_powers)
    kunchenko_extractor.fit(X_train_emb, y_train)
    X_train_kunchenko = kunchenko_extractor.transform(X_train_emb)
    X_test_kunchenko = kunchenko_extractor.transform(X_test_emb)
    
    X_train_hybrid = np.hstack([X_train_probs, X_train_kunchenko])
    X_test_hybrid = np.hstack([X_test_probs, X_test_kunchenko])
    
    pipeline_hybrid = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', OneVsRestClassifier(LogisticRegression(solver='liblinear', random_state=42, C=1.0)))
    ])
    pipeline_hybrid.fit(X_train_hybrid, y_train)
    y_pred_hybrid = pipeline_hybrid.predict(X_test_hybrid)
    
    metrics_hybrid, f1_hybrid = calculate_metrics(y_test, y_pred_hybrid)
    print(f"\nMacro F1 on hybrid features: {f1_hybrid:.4f}")
    print_classification_report(metrics_hybrid, label_order, "Hybrid Model Report")
    
    # --- FINAL SUMMARY ---
    print("\n" + "="*60 + "\nCOMPARATIVE RESULTS\n" + "="*60)
    print(f"Baseline Macro F1 (with thresholds): {f1_base:.4f}")
    print(f"Hybrid Model Macro F1 (Probs + Kunchenko): {f1_hybrid:.4f}")
    improvement = f1_hybrid - f1_base
    print(f"\nImprovement: {improvement:+.4f}")
    if improvement > 0.001:
        print("\nCONCLUSION: The hybrid method provided valuable information, improving the Macro F1 score.")
    else:
        print("\nCONCLUSION: The hybrid method did not yield a significant improvement.")

    # --- SAVE RESULTS FOR FURTHER ANALYSIS ---
    print("\n" + "="*60 + "\nSAVING ARTIFACTS FOR ANALYSIS\n" + "="*60)
    output_filename = "experiment_outputs.npz"
    np.savez_compressed(
        output_filename,
        X_train_kunchenko=X_train_kunchenko, y_train=y_train,
        X_test_kunchenko=X_test_kunchenko, y_test=y_test,
        y_pred_base=y_pred_base, y_pred_hybrid=y_pred_hybrid
    )
    print(f"Experiment artifacts saved to: {output_filename}")


if __name__ == "__main__":
    main()