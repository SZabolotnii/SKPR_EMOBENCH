#!/usr/bin/env python
"""Supervised_vs_unsupervised_EMOBENCH-UA.py

A final, elegant experiment comparing two dimensionality reduction techniques
with the same number of output features:
1. Unsupervised: Principal Component Analysis (PCA).
2. Supervised: Kunchenko error-based feature generation.
It also includes t-SNE visualizations to inspect the feature spaces, optimized
for high-quality, black-and-white scientific publications.
"""

import argparse
import random
from pathlib import Path
import numpy as np
import torch
from datasets import load_dataset
from sklearn.svm import SVC
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm.auto import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple

from transformers import AutoModel, AutoTokenizer

# --- Kunchenko Reconstructor for ERROR-based features (Unchanged) ---
class KunchenkoErrorExtractor:
    def __init__(self, n=4, alpha=1, lambda_reg=0.01, epsilon=1e-8):
        self.n, self.alpha, self.lambda_reg, self.epsilon = n, alpha, lambda_reg, epsilon
        self.models_ = {}
        self.classes_ = None

    def _compute_power(self, i, alpha):
        A, B, C = 1/i, 4 - i - 3/i, 2*i - 4 + 2/i
        return A + B * alpha + C * (alpha**2)

    def _create_basis_functions(self):
        return [
            lambda x, p=self._compute_power(i, self.alpha), eps=self.epsilon: np.sign(x) * (np.abs(x) + eps)**p
            for i in range(2, self.n + 1)
        ]

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        basis_functions = self._create_basis_functions()
        n_basis_funcs = len(basis_functions)

        for c in self.classes_:
            X_class = X[y == c]
            if len(X_class) < 2: continue
            
            class_model = {}
            for feature_idx in range(X.shape[1]):
                signals = X_class[:, feature_idx]
                basis = np.stack([func(signals) for func in basis_functions], axis=-1)
                E_x, E_phi = np.mean(signals), np.mean(basis, axis=0)
                centered_signals, centered_basis = signals - E_x, basis - E_phi
                F = centered_basis.T @ centered_basis / len(signals) + self.lambda_reg * np.eye(n_basis_funcs)
                b = centered_basis.T @ centered_signals / len(signals)
                try: K = np.linalg.solve(F, b)
                except np.linalg.LinAlgError: K = np.linalg.pinv(F) @ b
                class_model[feature_idx] = {'K': K, 'E_x': E_x, 'E_phi': E_phi, 'basis_functions': basis_functions}
            self.models_[c] = class_model
        return self

    def transform(self, X):
        print("Generating Kunchenko ERROR features...")
        features = np.zeros((X.shape[0], len(self.classes_)))
        for i in tqdm(range(X.shape[0]), desc="Generating Error Features"):
            for c_idx, c in enumerate(self.classes_):
                if c not in self.models_: continue
                total_error = 0
                for feature_idx in range(X.shape[1]):
                    model = self.models_[c][feature_idx]
                    basis_functions = model['basis_functions']
                    basis_matrix = np.stack([func(X[i, feature_idx]) for func in basis_functions])
                    reconstructed_signal = model['E_x'] + (basis_matrix - model['E_phi']) @ model['K']
                    total_error += (X[i, feature_idx] - reconstructed_signal)**2
                features[i, c_idx] = np.log(total_error / X.shape[1] + 1e-9)
        return features

# --- Feature Extraction (run once and cache) ---
def get_and_cache_embeddings(texts, model_emb, tokenizer, device, cache_path, batch_size=32):
    if cache_path.exists():
        print(f"Завантаження ембедингів з кешу: {cache_path}")
        return np.load(cache_path)['embeds']
    print("Генерація та кешування ембедингів...")
    all_embeds = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating Embeddings"):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=128).to(device)
        with torch.no_grad():
            emb_outputs = model_emb(**inputs)
            all_embeds.append(emb_outputs.last_hidden_state[:, 0, :].cpu().numpy())
    embeds = np.vstack(all_embeds)
    np.savez_compressed(cache_path, embeds=embeds)
    print(f"Ембединги збережено в кеш: {cache_path}")
    return embeds

# --- MODIFIED Visualization Function for Black-and-White Publications ---
def plot_tsne(X: np.ndarray, y: np.ndarray, title: str, label_names: List[str], filename: str):
    print(f"Generating publication-quality t-SNE plot for '{title}'...")
    # 1. Standardize features before t-SNE
    X_scaled = StandardScaler().fit_transform(X)
    
    # 2. Compute t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000, n_jobs=-1)
    X_tsne = tsne.fit_transform(X_scaled)
    
    # 3. Створюємо палітру градацій сірого (від темного до світлого)
    #    і набір чітких маркерів
    gray_palette = sns.color_palette("Greys_r", n_colors=len(label_names))
    markers = ['o', 's', '^', 'D', 'P', 'X', '*'] # Circle, Square, Triangle, Diamond, Pentagon, X, Star
    
    if len(label_names) > len(markers):
        markers = markers * (len(label_names) // len(markers) + 1)
        
    # 4. Налаштування стилю та розміру графіку
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(8, 6))
    
    # 5. Малюємо точки для кожного класу окремо, використовуючи свій маркер та відтінок сірого
    for i, label in enumerate(label_names):
        class_points = X_tsne[y == i]
        plt.scatter(
            class_points[:, 0], class_points[:, 1],
            marker=markers[i],
            label=label,
            facecolor=gray_palette[i], # Основний колір - відтінок сірого
            edgecolor='black',         # Контур - завжди чорний
            linewidths=0.5,            # Тонкий контур
            alpha=0.8,                 # Трохи більша насиченість для чіткості
            s=60                       # Збільшений розмір маркерів
        )
        
    # 6. Фіналізація графіку з якісними підписами
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    
    # Створюємо чисту легенду
    plt.legend(title="Emotions", fontsize=11, title_fontsize=12, markerscale=1.2, frameon=True, facecolor='white', framealpha=0.8)
    
    # 7. Зберігаємо графік у високій якості
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {filename}")
    plt.close()

# --- Main Experiment Logic ---
def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    model_name = "ukr-detect/ukr-emotions-classifier"
    dataset = load_dataset("ukr-detect/ukr-emotions-binary", cache_dir=args.cache_dir)
    def prepare_data(dataset_split):
        df = dataset_split.to_pandas()
        texts = df['text'].tolist()
        label_order = ['Joy', 'Fear', 'Anger', 'Sadness', 'Disgust', 'Surprise']
        df['None'] = (df[label_order].sum(axis=1) == 0).astype(int)
        label_order.append('None')
        labels = np.argmax(df[label_order].to_numpy(), axis=1)
        return texts, labels, label_order
    
    X_train_texts, y_train, label_names = prepare_data(dataset['train'])
    X_test_texts, y_test, _ = prepare_data(dataset['test'])

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_emb = AutoModel.from_pretrained(model_name).to(device)
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    X_train_emb = get_and_cache_embeddings(X_train_texts, model_emb, tokenizer, device, Path(args.output_dir) / "train_embeddings.npz")
    X_test_emb = get_and_cache_embeddings(X_test_texts, model_emb, tokenizer, device, Path(args.output_dir) / "test_embeddings.npz")

    n_features = len(label_names)

    print("\n--- Generating PCA features (Unsupervised) ---")
    pca = PCA(n_components=n_features)
    X_train_pca = pca.fit_transform(X_train_emb)
    X_test_pca = pca.transform(X_test_emb)
    
    print("\n--- Generating Kunchenko Error features (Supervised) ---")
    # Using optimized parameters from previous experiments
    error_extractor = KunchenkoErrorExtractor(n=4, alpha=1.0).fit(X_train_emb, y_train)
    X_train_error = error_extractor.transform(X_train_emb)
    X_test_error = error_extractor.transform(X_test_emb)
    
    results = {}
    
    def run_svm(name, X_train, y_train, X_test, y_test):
        print(f"\n--- Running SVM for: {name} ---")
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(class_weight='balanced', random_state=args.seed))
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='macro')
        print(f"Macro F1: {f1:.4f}")
        results[name] = f1

    run_svm(f"PCA-Only ({n_features} features)", X_train_pca, y_train, X_test_pca, y_test)
    run_svm(f"Kunchenko Error-Only ({n_features} features)", X_train_error, y_train, X_test_error, y_test)
    
    X_train_hybrid = np.hstack([X_train_pca, X_train_error])
    X_test_hybrid = np.hstack([X_test_pca, X_test_error])
    run_svm(f"Hybrid (PCA + Error, {2*n_features} features)", X_train_hybrid, y_train, X_test_hybrid, y_test)

    # --- VISUALIZATION ---
    print("\n" + "="*60 + "\nVISUALIZATION (using test set features)\n" + "="*60)
    plot_dir = Path(args.output_dir) / "plots"
    plot_dir.mkdir(exist_ok=True)
    
    plot_tsne(X_test_pca, y_test, f"t-SNE of PCA Features ({n_features} dims)", label_names, plot_dir / "tsne_pca.png")
    plot_tsne(X_test_error, y_test, f"t-SNE of Kunchenko Error Features ({n_features} dims)", label_names, plot_dir / "tsne_kunchenko.png")
    plot_tsne(X_test_hybrid, y_test, f"t-SNE of Hybrid Features ({2*n_features} dims)", label_names, plot_dir / "tsne_hybrid.png")

    # --- FINAL SUMMARY ---
    print("\n" + "="*60 + "\nSUPERVISED VS UNSUPERVISED DIMENSIONALITY REDUCTION\n" + "="*60)
    summary_df = pd.DataFrame.from_dict(results, orient='index', columns=['Macro F1'])
    print(summary_df.sort_values(by='Macro F1', ascending=False).to_string())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Supervised vs Unsupervised feature generation.")
    parser.add_argument("--output_dir", type=str, default="./comparison_results")
    parser.add_argument("--cache_dir", type=str, default="./.cache/datasets")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)