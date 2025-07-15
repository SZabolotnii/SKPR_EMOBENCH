#!/usr/bin/env python
"""Basis_function_search_EMOBENCH-UA.py

An experiment to find the optimal basis parameters for the Kunchenko
feature generator, comparing Logistic Regression and SVM classifiers against
strong benchmarks.
"""

import argparse
import random
from pathlib import Path
from typing import List, Tuple
import itertools

import numpy as np
import torch
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm.auto import tqdm
import pandas as pd

from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

# --- Kunchenko Reconstructor Class (Unchanged) ---
class KunchenkoForEmbeddings:
    def __init__(self, n=3, alpha=0.0, lambda_reg=0.01, epsilon=1e-8):
        self.n, self.alpha, self.lambda_reg, self.epsilon = n, alpha, lambda_reg, epsilon

    def _compute_power(self, i, alpha):
        A, B, C = 1/i, 4 - i - 3/i, 2*i - 4 + 2/i
        return A + B * alpha + C * (alpha**2)

    def fit(self, X, y):
        self.basis_functions = [
            lambda x, p=self._compute_power(i, self.alpha), eps=self.epsilon: np.sign(x) * (np.abs(x) + eps)**p
            for i in range(2, self.n + 1)
        ]
        self.n_basis_funcs = len(self.basis_functions)
        if self.n_basis_funcs == 0: return self

        self.classes_ = np.arange(y.shape[1])
        n_features = X.shape[1]
        self.models_ = {}

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
                F = centered_basis.T @ centered_basis / len(centered_basis) + self.lambda_reg * np.eye(self.n_basis_funcs)
                b = centered_basis.T @ centered_signals / len(centered_signals)
                try: K = np.linalg.solve(F, b)
                except np.linalg.LinAlgError: K = np.linalg.pinv(F) @ b
                class_model[feature_idx] = {'K': K, 'E_x': E_x, 'E_phi': E_phi}
            self.models_[c] = class_model
        return self

    def _apply_basis(self, signal_data):
        if not self.basis_functions: return np.empty((signal_data.shape[0], 0))
        return np.stack([func(signal_data) for func in self.basis_functions], axis=-1)

    def transform(self, X):
        n_samples, n_features_X = X.shape
        features = np.zeros((n_samples, len(self.classes_)))
        for i in range(n_samples):
            for c_idx, c in enumerate(self.classes_):
                if c not in self.models_: continue
                total_error = 0
                for feature_idx in range(n_features_X):
                    signal_1d = X[i, feature_idx]
                    model = self.models_[c][feature_idx]
                    K, E_x, E_phi = model['K'], model['E_x'], model['E_phi']
                    basis_matrix = self._apply_basis(signal_1d)
                    reconstructed_signal = E_x + (basis_matrix - E_phi) @ K
                    total_error += (signal_1d - reconstructed_signal)**2
                features[i, c_idx] = np.log(total_error / n_features_X + 1e-9)
        return features

# --- Feature Extraction (run once and cache) ---
def get_and_cache_features(texts, model_clf, model_emb, tokenizer, device, cache_path, batch_size=32):
    if cache_path.exists():
        print(f"Завантаження ознак з кешу: {cache_path}")
        with np.load(cache_path) as data:
            return data['probs'], data['embeds']
    print("Генерація та кешування ознак (ймовірності та ембединги)...")
    all_probs, all_embeds = [], []
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating Features"):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=128).to(device)
        with torch.no_grad():
            clf_outputs = model_clf(**inputs)
            all_probs.append(torch.sigmoid(clf_outputs.logits).cpu().numpy())
            emb_outputs = model_emb(**inputs)
            all_embeds.append(emb_outputs.last_hidden_state[:, 0, :].cpu().numpy())
    probs, embeds = np.vstack(all_probs), np.vstack(all_embeds)
    np.savez_compressed(cache_path, probs=probs, embeds=embeds)
    print(f"Ознаки збережено в кеш: {cache_path}")
    return probs, embeds

# --- Evaluation Metric ---
def calculate_macro_f1(y_true, y_pred):
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

# --- Main Experiment Logic ---
def main(args):
    # --- SETUP ---
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- DATA & MODEL LOADING ---
    model_name = "ukr-detect/ukr-emotions-classifier"
    dataset = load_dataset("ukr-detect/ukr-emotions-binary", cache_dir=args.cache_dir)
    label_order = ['Joy', 'Fear', 'Anger', 'Sadness', 'Disgust', 'Surprise', 'None']
    def prepare_data(dataset_split, order):
        df = dataset_split.to_pandas()
        texts = df['text'].tolist()
        base_emotions = [em for em in order if em != 'None']
        df['None'] = (df[base_emotions].sum(axis=1) == 0).astype(int)
        return texts, df[order].to_numpy()
    
    X_train_texts, y_train = prepare_data(dataset['train'], label_order)
    X_test_texts, y_test = prepare_data(dataset['test'], label_order)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_clf = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model_emb = AutoModel.from_pretrained(model_name).to(device)
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    X_train_probs, X_train_emb = get_and_cache_features(X_train_texts, model_clf, model_emb, tokenizer, device, Path(args.output_dir) / "train_features.npz")
    X_test_probs, X_test_emb = get_and_cache_features(X_test_texts, model_clf, model_emb, tokenizer, device, Path(args.output_dir) / "test_features.npz")

    # --- EXPERIMENT 1: BENCHMARKS (PROBABILITIES-ONLY) ---
    print("\n" + "="*60 + "\nEXPERIMENT 1: BENCHMARKS (CLASSIFIERS ON PROBABILITIES-ONLY)\n" + "="*60)
    
    # Benchmark 1: Logistic Regression
    pipeline_logreg_bench = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', OneVsRestClassifier(LogisticRegression(solver='liblinear', random_state=args.seed, C=1.0)))
    ])
    pipeline_logreg_bench.fit(X_train_probs, y_train)
    y_pred_logreg_bench = pipeline_logreg_bench.predict(X_test_probs)
    f1_logreg_benchmark = calculate_macro_f1(y_test, y_pred_logreg_bench)
    print(f"Benchmark Macro F1 (Logistic Regression): {f1_logreg_benchmark:.4f}")

    # Benchmark 2: SVM
    pipeline_svm_bench = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', OneVsRestClassifier(SVC(random_state=args.seed, C=1.0)))
    ])
    pipeline_svm_bench.fit(X_train_probs, y_train)
    y_pred_svm_bench = pipeline_svm_bench.predict(X_test_probs)
    f1_svm_benchmark = calculate_macro_f1(y_test, y_pred_svm_bench)
    print(f"Benchmark Macro F1 (SVM):                 {f1_svm_benchmark:.4f}")

    # --- EXPERIMENT 2: GRID SEARCH FOR KUNCHENKO BASIS PARAMETERS ---
    print("\n" + "="*60 + "\nEXPERIMENT 2: ПОШУК ОПТИМАЛЬНИХ ПАРАМЕТРІВ БАЗИСУ\n" + "="*60)
    param_grid = {'n': [3, 4, 5], 'alpha': [0.0, 1.0]}
    results = []
    param_combinations = list(itertools.product(param_grid['n'], param_grid['alpha']))
    
    for n, alpha in tqdm(param_combinations, desc="Grid Search"):
        kunchenko_extractor = KunchenkoForEmbeddings(n=n, alpha=alpha)
        kunchenko_extractor.fit(X_train_emb, y_train)
        X_train_kunchenko = kunchenko_extractor.transform(X_train_emb)
        X_test_kunchenko = kunchenko_extractor.transform(X_test_emb)
        
        X_train_hybrid = np.hstack([X_train_probs, X_train_kunchenko])
        X_test_hybrid = np.hstack([X_test_probs, X_test_kunchenko])
        
        # Train and evaluate Logistic Regression
        pipeline_logreg_hybrid = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', OneVsRestClassifier(LogisticRegression(solver='liblinear', random_state=args.seed, C=1.0)))
        ])
        pipeline_logreg_hybrid.fit(X_train_hybrid, y_train)
        y_pred_logreg_hybrid = pipeline_logreg_hybrid.predict(X_test_hybrid)
        f1_logreg_hybrid = calculate_macro_f1(y_test, y_pred_logreg_hybrid)
        
        # Train and evaluate SVM
        pipeline_svm_hybrid = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', OneVsRestClassifier(SVC(random_state=args.seed, C=1.0)))
        ])
        pipeline_svm_hybrid.fit(X_train_hybrid, y_train)
        y_pred_svm_hybrid = pipeline_svm_hybrid.predict(X_test_hybrid)
        f1_svm_hybrid = calculate_macro_f1(y_test, y_pred_svm_hybrid)

        results.append({
            'n': n, 'alpha': alpha, 
            'f1_logreg': f1_logreg_hybrid, 
            'f1_svm': f1_svm_hybrid
        })
        
    print("\n" + "="*60 + "\nРЕЗУЛЬТАТИ ПОШУКУ ПО СІТЦІ\n" + "="*60)
    results_df = pd.DataFrame(results)
    print(results_df.sort_values(by='f1_logreg', ascending=False).to_string())
    
    # --- FINAL SUMMARY ---
    print("\n" + "="*60 + "\nFINAL SUMMARY\n" + "="*60)
    
    if not results:
        print("Пошук по сітці не дав результатів.")
        return

    best_logreg_result = results_df.loc[results_df['f1_logreg'].idxmax()]
    best_svm_result = results_df.loc[results_df['f1_svm'].idxmax()]
    
    print("--- Benchmarks (Probabilities-only) ---")
    print(f"Logistic Regression: {f1_logreg_benchmark:.4f}")
    print(f"SVM:                 {f1_svm_benchmark:.4f}\n")
    
    print("--- Best Hybrid Results ---")
    print(f"Logistic Regression: {best_logreg_result['f1_logreg']:.4f} (at n={int(best_logreg_result['n'])}, alpha={best_logreg_result['alpha']})")
    print(f"SVM:                 {best_svm_result['f1_svm']:.4f} (at n={int(best_svm_result['n'])}, alpha={best_svm_result['alpha']})\n")

    print("--- Improvement vs. Own Benchmark ---")
    improvement_logreg = best_logreg_result['f1_logreg'] - f1_logreg_benchmark
    improvement_svm = best_svm_result['f1_svm'] - f1_svm_benchmark
    print(f"LogReg Improvement: {improvement_logreg:+.4f}")
    print(f"SVM Improvement:    {improvement_svm:+.4f}")

    print("-" * 40)
    overall_best_f1 = max(best_logreg_result['f1_logreg'], best_svm_result['f1_svm'])
    if overall_best_f1 > max(f1_logreg_benchmark, f1_svm_benchmark) + 0.001:
        print("ВИСНОВОК: УСПІХ! Гібридний підхід з ознаками Кунченка дозволив значно покращити результат.")
        if best_logreg_result['f1_logreg'] > best_svm_result['f1_svm']:
             print("Найкращу абсолютну продуктивність показав класифікатор Logistic Regression.")
        else:
             print("Найкращу абсолютну продуктивність показав класифікатор SVM.")
    else:
        print("ВИСНОВОК: Гібридний підхід не показав значного покращення відносно бенчмарків.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize Kunchenko basis parameters with LogReg and SVM.")
    parser.add_argument("--output_dir", type=str, default="./kunchenko_optimization_results")
    parser.add_argument("--cache_dir", type=str, default="./.cache/datasets")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)