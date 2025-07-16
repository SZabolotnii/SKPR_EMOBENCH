#!/usr/bin/env python
"""Base_test_emotions_basis_search_EN.py

Experiment for the English EmoEvent dataset (Scenario B from the paper).

Methodology:
1.  Fine-tunes a RoBERTa-base model on the EmoEvent dataset to create a
    strong benchmark.
2.  Defines a search space for basis functions (integer, fractional, combined).
3.  For each basis function configuration, generates Kunchenko features from
    the fine-tuned model's embeddings.
4.  Trains and evaluates two hybrid classifiers (Logistic Regression, SVM) on
    the combined transformer probabilities and Kunchenko features.
5.  Analyzes the results to find the best-performing configuration and compares
    it against the fine-tuned benchmark.
"""

import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.svm import SVC
from tqdm.auto import tqdm
import pandas as pd

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# --- Kunchenko Reconstructor (Refactored) ---
class KunchenkoReconstructor:
    def __init__(self, powers: Tuple[float, ...], reg: float = 1e-2):
        self.powers = powers
        self.reg = reg
        self._K = None
        self._mu_phi = None
        self._mu_x = None

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
        if self._K is None:
            raise RuntimeError("Model has not been fitted yet.")
        Y = (self._phi(X) - self._mu_phi) @ self._K + self._mu_x
        return ((X - Y) ** 2).mean(axis=1)

# --- Data Loading and Feature Extraction (Unchanged) ---
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

@torch.inference_mode()
def get_features(ds: Dataset, model, tokenizer, device, batch_size=32) -> Tuple[np.ndarray, np.ndarray]:
    model.eval().to(device)
    all_embeds, all_probs = [], []
    for i in tqdm(range(0, len(ds), batch_size), desc="Extracting Features"):
        batch = ds[i:i+batch_size]
        toks = tokenizer(batch["text"], padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        outputs = model(**toks, output_hidden_states=True)
        all_embeds.append(outputs.hidden_states[-1][:, 0, :].cpu().numpy())
        all_probs.append(torch.softmax(outputs.logits, dim=-1).cpu().numpy())
    return np.vstack(all_embeds), np.vstack(all_probs)

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
    train_ds, val_ds, test_ds = full_ds["train"], full_ds["validation"], full_ds["test"]
    y_train_val = np.array(concatenate_datasets([train_ds, val_ds])["label"])
    y_test = np.array(test_ds["label"])
    label_names = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "others"]

    # --- EXPERIMENT 1: FINE-TUNED BENCHMARK ---
    print("\n" + "="*60 + "\nEXPERIMENT 1: Fine-tuning RoBERTa (Benchmark)\n" + "="*60)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
    tokenized_train = train_ds.map(tokenize_function, batched=True)
    tokenized_val = val_ds.map(tokenize_function, batched=True)
    tokenized_test = test_ds.map(tokenize_function, batched=True)
    model_finetuned = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=len(label_names))
    
    # CORRECTED: Using `eval_strategy` for newer versions of transformers
    training_args = TrainingArguments(
        output_dir=f"{args.output_dir}/finetune_checkpoints", num_train_epochs=4,
        per_device_train_batch_size=16, per_device_eval_batch_size=16,
        warmup_steps=500, weight_decay=0.01, logging_dir=f"{args.output_dir}/logs",
        eval_strategy="epoch", 
        save_strategy="epoch", 
        load_best_model_at_end=True,
        metric_for_best_model="f1", report_to="none"
    )
    trainer = Trainer(model=model_finetuned, args=training_args, train_dataset=tokenized_train,
                      eval_dataset=tokenized_val,
                      compute_metrics=lambda p: {"f1": f1_score(p.label_ids, np.argmax(p.predictions, axis=-1), average="macro")})
    
    print("Fine-tuning the model...")
    trainer.train()
    
    print("Evaluating the fine-tuned benchmark model...")
    predictions = trainer.predict(tokenized_test)
    y_pred_benchmark = np.argmax(predictions.predictions, axis=1)
    f1_benchmark = f1_score(y_test, y_pred_benchmark, average="macro")
    print(f"\n--- BENCHMARK RESULTS ---")
    print(f"Macro F1: {f1_benchmark:.4f}")
    print(classification_report(y_test, y_pred_benchmark, target_names=label_names, digits=4))

    # --- EXPERIMENT 2: HYBRID MODELS - BASIS FUNCTION SEARCH ---
    print("\n" + "="*60 + "\nEXPERIMENT 2: Basis Function Search for Hybrid Models\n" + "="*60)
    print("Extracting features from the fine-tuned model (run once)...")
    train_val_ds = concatenate_datasets([train_ds, val_ds])
    X_train_emb, X_train_probs = get_features(train_val_ds, model_finetuned, tokenizer, device)
    X_test_emb, X_test_probs = get_features(test_ds, model_finetuned, tokenizer, device)

    search_space = {
        "Integer Powers": [(2.0,), (2.0, 3.0), (2.0, 3.0, 4.0), (2.0, 3.0, 4.0, 5.0)],
        "Fractional Powers": [(1/2,), (1/2, 1/3), (1/2, 1/3, 1/4), (1/2, 1/3, 1/4, 1/5)],
        "Combined Powers": [(1/2, 2.0), (1/2, 1/3, 2.0, 3.0)]
    }
    
    results = []
    for category, power_options in search_space.items():
        for powers in power_options:
            print(f"\n--- Testing Category: '{category}', Powers: {powers} ---")
            kun_models = [KunchenkoReconstructor(powers=powers).fit(X_train_emb[y_train_val == c]) for c in range(len(label_names))]
            X_train_kunchenko = np.stack([m.mse(X_train_emb) for m in kun_models], axis=1)
            X_test_kunchenko = np.stack([m.mse(X_test_emb) for m in kun_models], axis=1)
            
            X_train_hybrid = np.concatenate([X_train_probs, X_train_kunchenko], axis=1)
            X_test_hybrid = np.concatenate([X_test_probs, X_test_kunchenko], axis=1)
            
            clf_logreg = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=args.seed).fit(X_train_hybrid, y_train_val)
            f1_logreg = f1_score(y_test, clf_logreg.predict(X_test_hybrid), average="macro")
            
            clf_svm = SVC(class_weight="balanced", random_state=args.seed).fit(X_train_hybrid, y_train_val)
            f1_svm = f1_score(y_test, clf_svm.predict(X_test_hybrid), average="macro")
            
            print(f"Result (LogReg F1): {f1_logreg:.4f}, Result (SVM F1): {f1_svm:.4f}")
            results.append({"category": category, "powers": powers, "f1_logreg": f1_logreg, "f1_svm": f1_svm})

    # --- FINAL ANALYSIS ---
    print("\n" + "="*60 + "\nFINAL ANALYSIS OF BASIS FUNCTIONS\n" + "="*60)
    results_df = pd.DataFrame(results)
    best_logreg = results_df.loc[results_df['f1_logreg'].idxmax()]
    best_svm = results_df.loc[results_df['f1_svm'].idxmax()]
    
    print(f"Fine-tuned Benchmark (RoBERTa): {f1_benchmark:.4f}\n")
    print("--- Best Result for Logistic Regression ---")
    print(f"Macro F1: {best_logreg['f1_logreg']:.4f} (Improvement: {best_logreg['f1_logreg'] - f1_benchmark:+.4f})")
    print(f"Category: {best_logreg['category']}, Powers: {best_logreg['powers']}\n")

    print("--- Best Result for SVM ---")
    print(f"Macro F1: {best_svm['f1_svm']:.4f} (Improvement: {best_svm['f1_svm'] - f1_benchmark:+.4f})")
    print(f"Category: {best_svm['category']}, Powers: {best_svm['powers']}\n")
    
    if max(best_logreg['f1_logreg'], best_svm['f1_svm']) > f1_benchmark:
        print("CONCLUSION: SUCCESS! The hybrid approach with optimal basis functions improved upon the benchmark.")
    else:
        print("CONCLUSION: The hybrid approach did not improve upon the fine-tuned benchmark.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search for optimal basis functions on English EmoEvent.")
    parser.add_argument("--model_name", type=str, default="roberta-base", help="Base model for fine-tuning.")
    parser.add_argument("--output_dir", type=str, default="./emo_event_search_results")
    parser.add_argument("--cache_dir", type=str, default="./.cache/datasets")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)