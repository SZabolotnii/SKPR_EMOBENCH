#!/usr/bin/env python
"""RoBERTa_trained_EmoEvent.py

A focused experiment to improve a fine-tuned model's performance on the
English EmoEvent dataset using feature engineering.

Methodology:
1. Fine-tune a RoBERTa model to create a strong benchmark.
2. Apply the Kunchenko feature generation method to the embeddings from the
   fine-tuned model.
3. Train hybrid classifiers (Logistic Regression, SVM) on the combined
   features and compare their performance against the benchmark.
"""

import argparse
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.svm import SVC
from tqdm.auto import tqdm

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

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

# --- Data Loading (Focused on English EmoEvent) ---
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
    final_ds = filtered_ds.remove_columns(["id", "event", "tweet", "offensive"])
    print("Структура фінального датасету:", final_ds)
    return final_ds

# --- Feature Extraction ---
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
    train_ds = full_ds["train"]
    val_ds = full_ds["validation"]
    test_ds = full_ds["test"]
    y_train_val = np.array(concatenate_datasets([train_ds, val_ds])["label"])
    y_test = np.array(test_ds["label"])
    label_names = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "others"]

    # --- EXPERIMENT 1: FINE-TUNED BENCHMARK ---
    print("\n" + "="*60 + "\nEXPERIMENT 1: Fine-tuning RoBERTa (Benchmark)\n" + "="*60)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    tokenized_train_ds = train_ds.map(tokenize_function, batched=True)
    tokenized_val_ds = val_ds.map(tokenize_function, batched=True)
    tokenized_test_ds = test_ds.map(tokenize_function, batched=True)

    model_finetuned = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=len(label_names))

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {"f1": f1_score(labels, predictions, average="macro")}

    training_args = TrainingArguments(
        output_dir=f"{args.output_dir}/finetune_checkpoints",
        num_train_epochs=4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f"{args.output_dir}/logs",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="none",
    )

    trainer = Trainer(
        model=model_finetuned,
        args=training_args,
        train_dataset=tokenized_train_ds,
        eval_dataset=tokenized_val_ds,
        compute_metrics=compute_metrics,
    )

    print("Fine-tuning the model...")
    trainer.train()

    print("Evaluating the fine-tuned benchmark model...")
    predictions = trainer.predict(tokenized_test_ds)
    y_pred_benchmark = np.argmax(predictions.predictions, axis=1)
    f1_benchmark = f1_score(y_test, y_pred_benchmark, average="macro")
    print(f"\n--- BENCHMARK RESULTS ---")
    print(f"Macro F1: {f1_benchmark:.4f}")
    print(classification_report(y_test, y_pred_benchmark, target_names=label_names, digits=4))

    # --- EXPERIMENT 2: HYBRID MODELS ---
    print("\n" + "="*60 + "\nEXPERIMENT 2: Hybrid Models (RoBERTa + Kunchenko)\n" + "="*60)

    print("Extracting features from the fine-tuned model...")
    train_val_ds_combined = concatenate_datasets([train_ds, val_ds])
    X_train_finetuned_emb, X_train_finetuned_probs = get_features(train_val_ds_combined, model_finetuned, tokenizer, device)
    X_test_finetuned_emb, X_test_finetuned_probs = get_features(test_ds, model_finetuned, tokenizer, device)

    print("Generating Kunchenko features from fine-tuned embeddings...")
    kun_models = [KunchenkoReconstructor().fit(X_train_finetuned_emb[y_train_val == c]) for c in range(len(label_names))]
    X_train_kunchenko = np.stack([m.mse(X_train_finetuned_emb) for m in kun_models], axis=1)
    X_test_kunchenko = np.stack([m.mse(X_test_finetuned_emb) for m in kun_models], axis=1)

    X_train_hybrid = np.concatenate([X_train_finetuned_probs, X_train_kunchenko], axis=1)
    X_test_hybrid = np.concatenate([X_test_finetuned_probs, X_test_kunchenko], axis=1)

    # --- Hybrid Classifier 1: Logistic Regression ---
    print("\nTraining Hybrid Classifier (Logistic Regression)...")
    clf_logreg = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=args.seed).fit(X_train_hybrid, y_train_val)
    y_pred_logreg = clf_logreg.predict(X_test_hybrid)
    f1_logreg = f1_score(y_test, y_pred_logreg, average="macro")
    print(f"\n--- HYBRID RESULTS (LOGISTIC REGRESSION) ---")
    print(f"Macro F1: {f1_logreg:.4f}")
    print(classification_report(y_test, y_pred_logreg, target_names=label_names, digits=4))

    # --- Hybrid Classifier 2: SVM ---
    print("\nTraining Hybrid Classifier (SVM)...")
    clf_svm = SVC(class_weight="balanced", random_state=args.seed).fit(X_train_hybrid, y_train_val)
    y_pred_svm = clf_svm.predict(X_test_hybrid)
    f1_svm = f1_score(y_test, y_pred_svm, average="macro")
    print(f"\n--- HYBRID RESULTS (SVM) ---")
    print(f"Macro F1: {f1_svm:.4f}")
    print(classification_report(y_test, y_pred_svm, target_names=label_names, digits=4))


    # --- FINAL SUMMARY ---
    print("\n" + "="*60 + "\nFINAL SUMMARY\n" + "="*60)
    print(f"Fine-tuned Benchmark (RoBERTa):   {f1_benchmark:.4f}")
    print(f"Hybrid Model (LogReg):            {f1_logreg:.4f}")
    print(f"Hybrid Model (SVM):               {f1_svm:.4f}")
    print("-" * 40)
    improvement_logreg = f1_logreg - f1_benchmark
    improvement_svm = f1_svm - f1_benchmark
    print(f"Improvement (LogReg vs Benchmark): {improvement_logreg:+.4f}")
    print(f"Improvement (SVM vs Benchmark):    {improvement_svm:+.4f}")

    if improvement_logreg > 0 or improvement_svm > 0:
        print("\nCONCLUSION: SUCCESS! The hybrid feature engineering approach improved upon the fine-tuned benchmark.")
    else:
        print("\nCONCLUSION: The hybrid models did not improve upon the fine-tuned benchmark.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SOTA Challenge on English EmoEvent.")
    parser.add_argument("--model_name", type=str, default="roberta-base", help="English base model.")
    parser.add_argument("--output_dir", type=str, default="./RoBERTa_trained_EmoEvent_results")
    parser.add_argument("--cache_dir", type=str, default="./.cache/datasets")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    main(args)