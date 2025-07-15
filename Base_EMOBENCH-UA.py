import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import warnings
from tqdm import tqdm
import pandas as pd

warnings.filterwarnings("ignore")

def get_outputs_batched(texts, model, tokenizer, device, batch_size=32, get_embeds=False):
    print(f"Обробка текстів пакетами по {batch_size}...")
    all_outputs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing Batches"):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=128).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            if get_embeds:
                output = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            else:
                output = torch.sigmoid(outputs.logits).cpu().numpy()
            all_outputs.append(output)
    return np.vstack(all_outputs)

class KunchenkoForEmbeddings(BaseEstimator, TransformerMixin):
    def __init__(self, n=2, alpha=0.75, lambda_reg=0.01, epsilon=1e-8):
        self.n = n; self.alpha = alpha; self.lambda_reg = lambda_reg; self.epsilon = epsilon
    def _compute_power(self, i, alpha):
        A = 1/i; B = 4 - i - (3/i); C = 2*i - 4 + (2/i)
        return A + B*alpha + C*(alpha**2)
    def fit(self, X, y):
        self.basis_functions = []
        for i in range(2, self.n + 1):
            p = self._compute_power(i, self.alpha)
            self.basis_functions.append(lambda x, p=p, eps=self.epsilon: np.sign(x) * (np.abs(x) + eps)**p)
        self.n_basis_funcs = len(self.basis_functions)
        self.classes_ = np.arange(y.shape[1])
        n_features = X.shape[1]
        self.models_ = {}
        print("Навчання генератора ознак Кунченка...")
        for c in self.classes_:
            pure_class_indices = np.where((y.sum(axis=1) == 1) & (y[:, c] == 1))[0]
            if len(pure_class_indices) < 2:
                all_class_indices = np.where(y[:, c] == 1)[0]
                if len(all_class_indices) < 2: continue
                X_class = X[all_class_indices]
            else:
                X_class = X[pure_class_indices]
            class_model = {}
            for feature_idx in range(n_features):
                all_signals = X_class[:, feature_idx]
                all_basis = self._apply_basis(all_signals)
                E_x = np.mean(all_signals); E_phi = np.mean(all_basis, axis=0)
                centered_signals = all_signals - E_x; centered_basis = all_basis - E_phi
                F = centered_basis.T @ centered_basis / len(centered_basis)
                b = centered_basis.T @ centered_signals / len(centered_signals)
                F_reg = F + self.lambda_reg * np.eye(self.n_basis_funcs)
                try: K = np.linalg.solve(F_reg, b)
                except np.linalg.LinAlgError: K = np.linalg.pinv(F_reg) @ b
                class_model[feature_idx] = {'K': K, 'E_x': E_x, 'E_phi': E_phi}
            self.models_[c] = class_model
        return self
    def _apply_basis(self, signal_data):
        return np.stack([func(signal_data) for func in self.basis_functions], axis=-1)
    def transform(self, X):
        print("Генерація ознак Кунченка...")
        n_samples = X.shape[0]
        features = np.zeros((n_samples, len(self.classes_)))
        for i in tqdm(range(n_samples), desc="Generating Kunchenko Features"):
            for c_idx, c in enumerate(self.classes_):
                if c not in self.models_: continue
                total_error = 0
                for feature_idx in range(X.shape[1]):
                    signal_1d = X[i, feature_idx]
                    model = self.models_[c][feature_idx]
                    K, E_x, E_phi = model['K'], model['E_x'], model['E_phi']
                    basis_matrix = self._apply_basis(signal_1d)
                    reconstructed_signal = E_x + (basis_matrix - E_phi) @ K
                    total_error += (signal_1d - reconstructed_signal)**2
                features[i, c_idx] = np.log(total_error / X.shape[1] + 1e-9)
        return features

# =================================================================
# НОВІ ФУНКЦІЇ ДЛЯ РУЧНОГО РОЗРАХУНКУ ТА ВИВОДУ МЕТРИК
# =================================================================
def calculate_metrics(y_true, y_pred):
    """Обчислює precision, recall, f1 для кожного класу та macro-f1."""
    metrics = {}
    n_classes = y_true.shape[1]
    
    for i in range(n_classes):
        tp = np.sum((y_true[:, i] == 1) & (y_pred[:, i] == 1))
        fp = np.sum((y_true[:, i] == 0) & (y_pred[:, i] == 1))
        fn = np.sum((y_true[:, i] == 1) & (y_pred[:, i] == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        support = np.sum(y_true[:, i])
        
        metrics[i] = {'precision': precision, 'recall': recall, 'f1-score': f1, 'support': support}
        
    macro_f1 = np.mean([metrics[i]['f1-score'] for i in range(n_classes)])
    return metrics, macro_f1

def print_classification_report(metrics, target_names):
    """Виводить звіт у форматі, схожому на scikit-learn."""
    print(f"{'':<12} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}")
    print("-" * 55)
    for i, name in enumerate(target_names):
        m = metrics[i]
        print(f"{name:<12} {m['precision']:>10.2f} {m['recall']:>10.2f} {m['f1-score']:>10.2f} {m['support']:>10}")
    print("-" * 55)

def main():
    # ... (код завантаження даних без змін) ...
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Використовується Apple M3 MAX GPU (MPS).")
    else: device = torch.device("cpu")
    model_name = "ukr-detect/ukr-emotions-classifier"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_clf = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model_emb = AutoModel.from_pretrained(model_name).to(device)
    dataset = load_dataset("ukr-detect/ukr-emotions-binary")
    id2label = model_clf.config.id2label
    label_order = [id2label[i] for i in range(len(id2label))]
    def prepare_data(dataset_split, order):
        df = dataset_split.to_pandas()
        texts = df['text'].tolist()
        base_emotions = [em for em in order if em != 'None']
        df['None'] = (df[base_emotions].sum(axis=1) == 0).astype(int)
        labels = df[order].to_numpy()
        return texts, labels
    X_train_texts, y_train = prepare_data(dataset['train'], label_order)
    X_test_texts, y_test = prepare_data(dataset['test'], label_order)

    # --- Етап 1: Бенчмарк ---
    print("\n" + "="*60)
    print("ЕКСПЕРИМЕНТ А: БЕНЧМАРК НА ВИХІДНИХ ЙМОВІРНОСТЯХ")
    print("="*60)
    
    X_test_probs = get_outputs_batched(X_test_texts, model_clf, tokenizer, device, get_embeds=False)
    thresholds_dict = {"Joy": 0.35, "Fear": 0.5, "Anger": 0.25, "Sadness": 0.5, "Disgust": 0.3, "Surprise": 0.25, "None": 0.35}
    thresholds = np.array([thresholds_dict[label] for label in label_order])
    y_pred_base = (X_test_probs >= thresholds).astype(int)
    
    metrics_base, f1_base = calculate_metrics(y_test, y_pred_base)
    print(f"F1-macro з оптимальними порогами: {f1_base:.4f}")
    print_classification_report(metrics_base, label_order)

    # --- Етап 2: Гібридна модель ---
    print("\n" + "="*60)
    print("ЕКСПЕРИМЕНТ Б: КЛАСИФІКАЦІЯ НА ГІБРИДНИХ ОЗНАКАХ")
    print("="*60)
    
    X_train_probs = get_outputs_batched(X_train_texts, model_clf, tokenizer, device, get_embeds=False)
    X_train_emb = get_outputs_batched(X_train_texts, model_emb, tokenizer, device, get_embeds=True)
    X_test_emb = get_outputs_batched(X_test_texts, model_emb, tokenizer, device, get_embeds=True)
    
    # kunchenko_extractor = KunchenkoForEmbeddings(n=3, alpha=0)
    # kunchenko_extractor = KunchenkoForEmbeddings(n=4, alpha=1.0)
    kunchenko_extractor = KunchenkoForEmbeddings(n=2, alpha=0.75)
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
    print(f"F1-macro на гібридних ознаках: {f1_hybrid:.4f}")
    print_classification_report(metrics_hybrid, label_order)
    
    # --- Фінальний висновок ---
    print("\n" + "="*60)
    print("ПОРІВНЯННЯ РЕЗУЛЬТАТІВ")
    print("="*60)
    print(f"F1-macro на вихідних ймовірностях (з порогами): {f1_base:.4f}")
    print(f"F1-macro на гібридних ознаках (ймовірності + наші 7): {f1_hybrid:.4f}")
    
    improvement = f1_hybrid - f1_base
    if improvement > 0.001:
        print(f"\nВИСНОВОК: Наш метод 'уточнення' ДОДАВ цінну інформацію, покращивши F1-macro на {improvement:.4f}!")
    else:
        print("\nВИСНОВОК: Наш метод не дав значного покращення. Вихідних ймовірностей достатньо.")

    # --- Збереження результатів для подальшого аналізу ---
    print("\n" + "="*60)
    print("ЗБЕРЕЖЕННЯ РЕЗУЛЬТАТІВ ДЛЯ ПОДАЛЬШОГО АНАЛІЗУ")
    print("="*60)
    
    output_filename = "experiment_outputs.npz"
    np.savez_compressed(
        output_filename,
        # Дані для Ablation Study
        X_train_kunchenko=X_train_kunchenko,
        y_train=y_train,
        X_test_kunchenko=X_test_kunchenko,
        # Дані для тесту на значущість
        y_test=y_test,
        y_pred_base=y_pred_base,
        y_pred_hybrid=y_pred_hybrid
    )
    print(f"Результати збережено у файл: {output_filename}")


if __name__ == "__main__":
    main()