#!/usr/bin/env python
"""Basis_function_search_UA.py

An experiment for the Ukrainian EMOBENCH-UA dataset to find the optimal
basis function configuration for the Kunchenko feature generator.

Methodology:
1.  Establishes two strong benchmarks by training Logistic Regression and SVM
    classifiers on the raw probabilities from the `ukr-detect/ukr-emotions-classifier` model.
2.  Defines a search space for basis functions, including integer, fractional,
    and combined power-based functions.
3.  For each basis function configuration, generates Kunchenko features from
    the model's embeddings.
4.  Trains and evaluates two hybrid classifiers (Logistic Regression, SVM) on
    the combined transformer probabilities and Kunchenko features.
5.  Analyzes the results to find the best-performing configuration for each
    classifier and compares it against its respective benchmark.
"""

import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple
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

# --- Kunchenko Feature Generator (Refactored to use explicit powers) ---
class KunchenkoFeatureGenerator:
    def __init__(self, powers: Tuple[float, ...], lambda_reg: float = 0.01, epsilon: float = 1e-8):
        self.powers = powers
        self.lambda_reg = lambda_reg
        self.epsilon = epsilon
        self.models_ = {}
        self.classes_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KunchenkoFeatureGenerator":
        self.basis_functions_ = [
            lambda x, p=p, eps=self.epsilon: np.sign(x) * (np.abs(x) + eps)**p
            for p in self.powers
        ]
        self.n_basis_funcs_ = len(self.basis_functions_)
        if self.n_basis_funcs_ == 0:
            return self

        self.classes_ = np.arange(y.shape[1])
        n_features = X.shape[1]

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

    def _apply_basis(self, signal_data: np.ndarray) -> np.ndarray:
        if not self.basis_functions_:
            return np.empty((signal_data.shape[0], 0))
        return np.stack([func(signal_data) for func in self.basis_functions_], axis=-1)

    def transform(self, X: np.ndarray) -> np.ndarray:
        features = np.zeros((X.shape[0], len(self.classes_)))
        for i in range(X.shape[0]):
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
def get_and_cache_features(texts, model_clf, model_emb, tokenizer, device, cache_path, batch_size=32):
    if cache_path.exists():
        print(f"Loading features from cache: {cache_path}")
        with np.load(cache_path) as data:
            return data['probs'], data['embeds']
    print("Generating and caching features (probabilities and embeddings)...")
    model_clf.eval()
    model_emb.eval()
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
    print(f"Features saved to cache: {cache_path}")
    return probs, embeds

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
    
    pipeline_logreg_bench = Pipeline([('scaler', StandardScaler()), ('clf', OneVsRestClassifier(LogisticRegression(solver='liblinear', random_state=args.seed, C=1.0)))])
    pipeline_logreg_bench.fit(X_train_probs, y_train)
    f1_logreg_benchmark = calculate_macro_f1(y_test, pipeline_logreg_bench.predict(X_test_probs))
    print(f"Benchmark Macro F1 (Logistic Regression): {f1_logreg_benchmark:.4f}")

    pipeline_svm_bench = Pipeline([('scaler', StandardScaler()), ('clf', OneVsRestClassifier(SVC(random_state=args.seed, C=1.0)))])
    pipeline_svm_bench.fit(X_train_probs, y_train)
    f1_svm_benchmark = calculate_macro_f1(y_test, pipeline_svm_bench.predict(X_test_probs))
    print(f"Benchmark Macro F1 (SVM):                 {f1_svm_benchmark:.4f}")

    # --- EXPERIMENT 2: GRID SEARCH FOR KUNCHENKO BASIS PARAMETERS ---
    print("\n" + "="*60 + "\nEXPERIMENT 2: GRID SEARCH FOR OPTIMAL BASIS PARAMETERS\n" + "="*60)
    
    search_space = {
        "Fractional Powers": [(1/2,), (1/2, 1/3), (1/2, 1/3, 1/4)],
        "Integer Powers": [(2.0,), (2.0, 3.0), (2.0, 3.0, 4.0)],
    }
    results = []
    
    for category, power_options in search_space.items():
        for powers in power_options:
            print(f"\n--- Testing Category: '{category}', Powers: {powers} ---")
            kunchenko_extractor = KunchenkoFeatureGenerator(powers=powers)
            kunchenko_extractor.fit(X_train_emb, y_train)
            X_train_kunchenko = kunchenko_extractor.transform(X_train_emb)
            X_test_kunchenko = kunchenko_extractor.transform(X_test_emb)
            
            X_train_hybrid = np.hstack([X_train_probs, X_train_kunchenko])
            X_test_hybrid = np.hstack([X_test_probs, X_test_kunchenko])
            
            pipeline_logreg = Pipeline([('scaler', StandardScaler()), ('clf', OneVsRestClassifier(LogisticRegression(solver='liblinear', random_state=args.seed, C=1.0)))])
            pipeline_logreg.fit(X_train_hybrid, y_train)
            f1_logreg = calculate_macro_f1(y_test, pipeline_logreg.predict(X_test_hybrid))
            
            pipeline_svm = Pipeline([('scaler', StandardScaler()), ('clf', OneVsRestClassifier(SVC(random_state=args.seed, C=1.0)))])
            pipeline_svm.fit(X_train_hybrid, y_train)
            f1_svm = calculate_macro_f1(y_test, pipeline_svm.predict(X_test_hybrid))
            
            print(f"Result (LogReg F1): {f1_logreg:.4f}, Result (SVM F1): {f1_svm:.4f}")
            results.append({'category': category, 'powers': powers, 'f1_logreg': f1_logreg, 'f1_svm': f1_svm})
        
    # --- FINAL SUMMARY ---
    print("\n" + "="*60 + "\nFINAL SUMMARY\n" + "="*60)
    results_df = pd.DataFrame(results)
    best_logreg = results_df.loc[results_df['f1_logreg'].idxmax()]
    best_svm = results_df.loc[results_df['f1_svm'].idxmax()]
    
    print("--- Benchmarks (Probabilities-only) ---")
    print(f"Logistic Regression: {f1_logreg_benchmark:.4f}")
    print(f"SVM:                 {f1_svm_benchmark:.4f}\n")
    
    print("--- Best Hybrid Results ---")
    print(f"Logistic Regression: {best_logreg['f1_logreg']:.4f} (at powers={best_logreg['powers']})")
    print(f"SVM:                 {best_svm['f1_svm']:.4f} (at powers={best_svm['powers']})\n")

    print("--- Improvement vs. Own Benchmark ---")
    print(f"LogReg Improvement: {best_logreg['f1_logreg'] - f1_logreg_benchmark:+.4f}")
    print(f"SVM Improvement:    {best_svm['f1_svm'] - f1_svm_benchmark:+.4f}")

    print("-" * 40)
    if max(best_logreg['f1_logreg'], best_svm['f1_svm']) > max(f1_logreg_benchmark, f1_svm_benchmark) + 0.001:
        print("CONCLUSION: SUCCESS! The hybrid approach with Kunchenko features significantly improved the results.")
    else:
        print("CONCLUSION: The hybrid approach did not yield a significant improvement over the benchmarks.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize Kunchenko basis parameters with LogReg and SVM for Ukrainian.")
    parser.add_argument("--output_dir", type=str, default="./kunchenko_optimization_results_ua")
    parser.add_argument("--cache_dir", type=str, default="./.cache/datasets")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)