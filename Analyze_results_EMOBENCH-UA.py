import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def calculate_metrics(y_true, y_pred):
    """Обчислює precision, recall, f1 для кожного класу та macro-f1."""
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
        
        class_f1_scores.append(f1)
        
    macro_f1 = np.mean(class_f1_scores)
    return macro_f1

def run_ablation_study(X_train, y_train, X_test, y_test):
    """
    Проводить експеримент, навчаючи класифікатор лише на ознаках Кунченка.
    """
    print("\n" + "="*60)
    print("ДОСЛІДЖЕННЯ 'ЧОРНОЇ СКРИНЬКИ' (ABLATION STUDY)")
    print("="*60)
    print("Навчаємо класифікатор ТІЛЬКИ на 7 ознаках Кунченка...")

    pipeline_ablation = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', OneVsRestClassifier(LogisticRegression(solver='liblinear', random_state=42, C=1.0)))
    ])

    pipeline_ablation.fit(X_train, y_train)
    y_pred_ablation = pipeline_ablation.predict(X_test)

    # ВИПРАВЛЕНО: Використовуємо нашу ручну функцію для розрахунку macro F1
    f1_ablation = calculate_metrics(y_test, y_pred_ablation)

    print(f"\nРезультат:")
    print(f"F1-macro на ознаках Кунченка (без ймовірностей): {f1_ablation:.4f}")
    
    if f1_ablation > 0.2:
        print("ВИСНОВОК: Ознаки Кунченка самі по собі несуть значну інформацію,")
        print("оскільки якість класифікації на них значно вища за випадкову.")
    else:
        print("ВИСНОВОК: Ознаки Кунченка самі по собі слабкі, їхня цінність проявляється лише в комбінації.")


def run_bootstrap_test(y_true, y_pred_model1, y_pred_model2, n_iterations=1000):
    """
    Проводить бутстреп-тест для оцінки статистичної значущості різниці.
    """
    print("\n" + "="*60)
    print("ТЕСТ НА СТАТИСТИЧНУ ЗНАЧУЩІСТЬ (БУТСТРЕП)")
    print("="*60)
    print(f"Проводимо {n_iterations} ітерацій для порівняння Гібридної моделі та Бенчмарку...")

    # ВИПРАВЛЕНО: Використовуємо нашу ручну функцію для розрахунку macro F1
    original_diff = calculate_metrics(y_true, y_pred_model2) - calculate_metrics(y_true, y_pred_model1)

    size = len(y_true)
    diff_scores = []
    for i in range(n_iterations):
        indices = resample(np.arange(size))
        
        f1_model1 = calculate_metrics(y_true[indices], y_pred_model1[indices])
        f1_model2 = calculate_metrics(y_true[indices], y_pred_model2[indices])
        
        diff_scores.append(f1_model2 - f1_model1)

    diff_scores = np.array(diff_scores)
    p_value = np.mean(diff_scores <= 0)
    
    alpha = 0.05
    lower_bound = np.percentile(diff_scores, 100 * (alpha / 2))
    upper_bound = np.percentile(diff_scores, 100 * (1 - alpha / 2))

    print("\nРезультати:")
    print(f"Оригінальне покращення F1-macro: {original_diff:+.4f}")
    print(f"95% довірчий інтервал для покращення: [{lower_bound:+.4f}, {upper_bound:+.4f}]")
    print(f"p-value: {p_value:.4f}")

    if p_value < 0.05:
        print("\nВИСНОВОК: p-value < 0.05. Покращення є СТАТИСТИЧНО ЗНАЧУЩИМ.")
        print("Це означає, що ймовірність отримати таке покращення випадково дуже низька.")
    else:
        print("\nВИСНОВОК: p-value >= 0.05. Покращення НЕ є статистично значущим.")
        print("Ми не можемо з упевненістю відкинути гіпотезу, що покращення було випадковим.")


def main():
    """
    Головна функція для завантаження результатів та запуску аналізу.
    """
    try:
        data = np.load("experiment_outputs.npz")
    except FileNotFoundError:
        print("Помилка: файл 'experiment_outputs.npz' не знайдено.")
        print("Будь ласка, спочатку запустіть основний скрипт SKPR_EMOBENCH-UA.py, щоб згенерувати цей файл.")
        return

    run_ablation_study(
        X_train=data['X_train_kunchenko'],
        y_train=data['y_train'],
        X_test=data['X_test_kunchenko'],
        y_test=data['y_test']
    )

    run_bootstrap_test(
        y_true=data['y_test'],
        y_pred_model1=data['y_pred_base'],
        y_pred_model2=data['y_pred_hybrid']
    )


if __name__ == "__main__":
    main()