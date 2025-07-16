#!/usr/bin/env python
"""Classic_vs_kunchenko_EN.py

A focused experiment to compare classic ML features (TF-IDF) against
Kunchenko features and their hybrid combination, using SVM as the classifier.

Methodology:
1. Train an SVM on TF-IDF features (reproducing the SOTA baseline).
2. Train an SVM on Kunchenko features alone.
3. Train an SVM on the combined TF-IDF + Kunchenko features.
"""

import argparse
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from tqdm.auto import tqdm

from transformers import AutoModel, AutoTokenizer

# --- Kunchenko Reconstructor Class (Unchanged) ---
class KunchenkoReconstructor:
    def __init__(self, powers: Tuple[float, ...] = (0.5, 1.0 / 3), reg: float = 1e-2):
        self.powers, self.reg = powers, reg
        self._K, self._mu_phi, self._mu_x = None, None, None
    def _phi(self, X: np.ndarray) -> np.ndarray:
        return np.concatenate([np.power(np.abs(X), p) * np.sign(X) for p in self.powers], axis=1)
    def fit(self, X: np.ndarray) -> "KunchenkoReconstructor":
        n, D = X.shape
        Phi = self._phi(X)
        self._mu_x, self._mu_phi = X.mean(axis=0), Phi.mean(axis=0)
        Xc, Phic = X - self._mu_x, Phi - self._mu_phi
        F = (Phic.T @ Phic) / n + self.reg * np.eye(Phic.shape[1])
        b = (Phic.T @ Xc) / n
        self._K = np.linalg.solve(F, b)
        return self
    def mse(self, X: np.ndarray) -> np.ndarray:
        if self._K is None: raise RuntimeError("Model not fitted.")
        Y = (self._phi(X) - self._mu_phi) @ self._K + self._mu_x
        return ((X - Y) ** 2).mean(axis=1)

# --- Data Loading (Unchanged) ---
def load_emo_event_en(cache_dir: str) -> Dataset:
    base_url = "https://raw.githubusercontent.com/fmplaza/EmoEvent/master/splits/en/"
    data_urls = {'train': f'{base_url}train.tsv', 'test': f'{base_url}test.tsv', 'validation': f'{base_url}dev.tsv'}
    ds = load_dataset("csv", data_files=data_urls, delimiter="\t", column_names=["id", "event", "tweet", "offensive", "label"], skiprows=1, cache_dir=cache_dir)
    label_names = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "others"]
    label2id = {name: i for i, name in enumerate(label_names)}
    def unify(ex):
        text, label_str = ex["tweet"], ex["label"]
        if not isinstance(text, str) or not text.strip(): return None
        label_id = label2id.get(label_str, -1)
        if label_id == -1: return None
        return {"text": text, "label": label_id}
    processed_ds = ds.map(unify)
    filtered_ds = processed_ds.filter(lambda ex: ex is not None and ex["text"] is not None and ex["label"] is not None)
    return filtered_ds.remove_columns(["id", "event", "tweet", "offensive"])

# --- Embedding Generation ---
@torch.inference_mode()
def get_base_embeddings(ds: Dataset, model, tokenizer, device, batch_size=32) -> np.ndarray:
    model.eval().to(device)
    all_embeds = []
    for i in tqdm(range(0, len(ds), batch_size), desc="Generating Embeddings"):
        batch = ds[i:i+batch_size]
        toks = tokenizer(batch["text"], padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        all_embeds.append(model(**toks).last_hidden_state[:, 0, :].cpu().numpy())
    return np.vstack(all_embeds)

# --- Main Experiment Logic ---
def main(args):
    # Setup
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Data
    full_ds = load_emo_event_en(args.cache_dir)
    train_ds = concatenate_datasets([full_ds["train"], full_ds["validation"]])
    test_ds = full_ds["test"]
    
    X_train_text = train_ds["text"]
    X_test_text = test_ds["text"]
    y_train = np.array(train_ds["label"])
    y_test = np.array(test_ds["label"])
    label_names = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "others"]

    # --- EXPERIMENT A: SVM on TF-IDF (SOTA REPRODUCTION) ---
    print("\n" + "="*60 + "\nEXPERIMENT 1: SVM on TF-IDF Features (SOTA Reproduction)\n" + "="*60)
    tfidf_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ('svm', SVC(class_weight='balanced', random_state=args.seed))
    ])
    tfidf_pipeline.fit(X_train_text, y_train)
    y_pred_tfidf = tfidf_pipeline.predict(X_test_text)
    f1_tfidf = f1_score(y_test, y_pred_tfidf, average="macro")
    print(f"\n--- TF-IDF + SVM RESULTS ---")
    print(f"Macro F1: {f1_tfidf:.4f}")
    print(classification_report(y_test, y_pred_tfidf, target_names=label_names, digits=4))

    # --- Generate Embeddings for Kunchenko Features ---
    print("\n" + "="*60 + "\nGenerating Embeddings for Kunchenko Features\n" + "="*60)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model_base = AutoModel.from_pretrained(args.model_name)
    X_train_emb = get_base_embeddings(train_ds, model_base, tokenizer, device)
    X_test_emb = get_base_embeddings(test_ds, model_base, tokenizer, device)

    # --- EXPERIMENT B: SVM on Kunchenko Features Only ---
    print("\n" + "="*60 + "\nEXPERIMENT 2: SVM on Kunchenko Features Only\n" + "="*60)
    kun_models = [KunchenkoReconstructor().fit(X_train_emb[y_train == c]) for c in range(len(label_names))]
    X_train_kunchenko = np.stack([m.mse(X_train_emb) for m in kun_models], axis=1)
    X_test_kunchenko = np.stack([m.mse(X_test_emb) for m in kun_models], axis=1)
    
    svm_kunchenko = SVC(class_weight='balanced', random_state=args.seed).fit(X_train_kunchenko, y_train)
    y_pred_kunchenko = svm_kunchenko.predict(X_test_kunchenko)
    f1_kunchenko = f1_score(y_test, y_pred_kunchenko, average="macro")
    print(f"\n--- KUNCHENKO-ONLY + SVM RESULTS ---")
    print(f"Macro F1: {f1_kunchenko:.4f}")
    print(classification_report(y_test, y_pred_kunchenko, target_names=label_names, digits=4))

    # --- EXPERIMENT C: SVM on Hybrid Features ---
    print("\n" + "="*60 + "\nEXPERIMENT 3: SVM on Hybrid (TF-IDF + Kunchenko) Features\n" + "="*60)
    X_train_tfidf_feats = tfidf_pipeline.named_steps['tfidf'].transform(X_train_text)
    X_test_tfidf_feats = tfidf_pipeline.named_steps['tfidf'].transform(X_test_text)
    
    X_train_hybrid = np.hstack([X_train_tfidf_feats.toarray(), X_train_kunchenko])
    X_test_hybrid = np.hstack([X_test_tfidf_feats.toarray(), X_test_kunchenko])
    
    svm_hybrid = SVC(class_weight='balanced', random_state=args.seed).fit(X_train_hybrid, y_train)
    y_pred_hybrid = svm_hybrid.predict(X_test_hybrid)
    f1_hybrid = f1_score(y_test, y_pred_hybrid, average="macro")
    print(f"\n--- HYBRID + SVM RESULTS ---")
    print(f"Macro F1: {f1_hybrid:.4f}")
    print(classification_report(y_test, y_pred_hybrid, target_names=label_names, digits=4))

    # --- FINAL SUMMARY ---
    print("\n" + "="*60 + "\nFINAL SUMMARY\n" + "="*60)
    print(f"1. SVM on TF-IDF (SOTA Baseline): {f1_tfidf:.4f}")
    print(f"2. SVM on Kunchenko-Only:       {f1_kunchenko:.4f}")
    print(f"3. SVM on Hybrid Features:      {f1_hybrid:.4f}")
    print("-" * 45)
    print(f"Improvement (Hybrid vs TF-IDF): {f1_hybrid - f1_tfidf:+.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare classic vs Kunchenko features with SVM.")
    parser.add_argument("--model_name", type=str, default="roberta-base", help="Base model for embeddings.")
    parser.add_argument("--cache_dir", type=str, default="./.cache/datasets")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    main(args)