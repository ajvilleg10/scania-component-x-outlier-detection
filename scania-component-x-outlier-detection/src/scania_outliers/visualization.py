from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


def _prepare_path(output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _boxplot(ax, data, labels, **kwargs):
    """Matplotlib compatibility across Colab versions."""
    try:
        return ax.boxplot(data, tick_labels=labels, **kwargs)
    except TypeError:  # Matplotlib < 3.9
        return ax.boxplot(data, labels=labels, **kwargs)


def save_missing_values_plot(missing_report: list[dict], output_path: str | Path, top_n: int = 30) -> None:
    import matplotlib.pyplot as plt

    rows = sorted(missing_report, key=lambda r: float(r.get("missing_ratio", 0.0)), reverse=True)[:top_n]
    if not rows:
        return
    columns = [str(r["column"]) for r in rows]
    ratios = [float(r.get("missing_ratio", 0.0)) for r in rows]
    path = _prepare_path(output_path)

    fig, ax = plt.subplots(figsize=(10, max(4, 0.27 * len(columns))))
    ax.barh(columns, ratios)
    ax.set_xlabel("Proporción de valores faltantes")
    ax.set_ylabel("Variable")
    ax.set_title("Variables con mayor proporción de valores faltantes")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_label_distribution_plot(label_counts: dict[str, dict[int, int]], output_path: str | Path) -> None:
    import matplotlib.pyplot as plt

    splits = list(label_counts.keys())
    if not splits:
        return
    normals = [int(label_counts[s].get(0, 0)) for s in splits]
    positives = [int(label_counts[s].get(1, 0)) for s in splits]
    x = np.arange(len(splits))
    width = 0.36
    path = _prepare_path(output_path)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, normals, width, label="Normal")
    ax.bar(x + width / 2, positives, width, label="Outlier / reparación")
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_ylabel("Número de vehículos")
    ax.set_title("Distribución de etiquetas por partición")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_feature_distributions(values: np.ndarray, feature_names: list[str], output_path: str | Path, bins: int = 30) -> None:
    import matplotlib.pyplot as plt

    if values.size == 0 or not feature_names:
        return
    n = min(values.shape[1], len(feature_names), 6)
    rows = int(np.ceil(n / 2))
    path = _prepare_path(output_path)
    fig, axes = plt.subplots(rows, 2, figsize=(12, 3.4 * rows))
    axes = np.atleast_1d(axes).ravel()
    for i in range(n):
        arr = values[:, i]
        arr = arr[np.isfinite(arr)]
        axes[i].hist(arr, bins=bins)
        axes[i].set_title(feature_names[i])
        axes[i].set_xlabel("Valor")
        axes[i].set_ylabel("Frecuencia")
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Distribuciones de una muestra de variables operacionales")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_feature_boxplots(values: np.ndarray, feature_names: list[str], output_path: str | Path) -> None:
    import matplotlib.pyplot as plt

    if values.size == 0 or not feature_names:
        return
    n = min(values.shape[1], len(feature_names), 8)
    data = [values[:, i][np.isfinite(values[:, i])] for i in range(n)]
    path = _prepare_path(output_path)
    fig, ax = plt.subplots(figsize=(max(9, 1.15 * n), 5.5))
    _boxplot(ax, data, feature_names[:n], showfliers=False)
    ax.set_title("Box plots de variables operacionales seleccionadas")
    ax.set_ylabel("Valor")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_correlation_matrix(values: np.ndarray, feature_names: list[str], output_path: str | Path) -> None:
    import matplotlib.pyplot as plt

    if values.size == 0 or values.shape[1] < 2:
        return
    valid_cols = []
    valid_names = []
    for i, name in enumerate(feature_names[: values.shape[1]]):
        col = values[:, i]
        if np.sum(np.isfinite(col)) >= 3 and np.nanstd(col) > 0:
            valid_cols.append(col)
            valid_names.append(name)
    if len(valid_cols) < 2:
        return
    matrix = np.column_stack(valid_cols)
    # Correlation is calculated only on the bounded visualization sample, never on the full driver dataset.
    filled = matrix.copy()
    medians = np.nanmedian(filled, axis=0)
    for j in range(filled.shape[1]):
        filled[~np.isfinite(filled[:, j]), j] = medians[j] if np.isfinite(medians[j]) else 0.0
    corr = np.corrcoef(filled, rowvar=False)
    path = _prepare_path(output_path)
    fig, ax = plt.subplots(figsize=(max(7, 0.65 * len(valid_names)), max(6, 0.6 * len(valid_names))))
    image = ax.imshow(corr, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(valid_names)))
    ax.set_yticks(np.arange(len(valid_names)))
    ax.set_xticklabels(valid_names, rotation=90)
    ax.set_yticklabels(valid_names)
    ax.set_title("Matriz de correlación de variables seleccionadas")
    fig.colorbar(image, ax=ax, label="Correlación")
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def save_histogram(values: Iterable[float], output_path: str | Path, title: str, xlabel: str, bins: int = 30) -> None:
    import matplotlib.pyplot as plt

    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return
    path = _prepare_path(output_path)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(arr, bins=bins)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frecuencia")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_preprocessing_feature_summary(selected: int, dropped_missing: int, dropped_constant: int, output_path: str | Path) -> None:
    import matplotlib.pyplot as plt

    labels = ["Seleccionadas", "Descartadas por faltantes", "Descartadas constantes"]
    values = [selected, dropped_missing, dropped_constant]
    path = _prepare_path(output_path)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, values)
    ax.set_ylabel("Número de variables")
    ax.set_title("Resumen de selección de variables")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_windows_by_split_plot(split_metadata: dict[str, dict], output_path: str | Path) -> None:
    import matplotlib.pyplot as plt

    splits = list(split_metadata.keys())
    counts = [int(split_metadata[s].get("n_windows", 0)) for s in splits]
    if not splits:
        return
    path = _prepare_path(output_path)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(splits, counts)
    ax.set_ylabel("Número de ventanas")
    ax.set_title("Ventanas temporales generadas por partición")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_training_history_plot(history: dict, output_path: str | Path, model_name: str) -> None:
    import matplotlib.pyplot as plt

    train = np.asarray(history.get("train_loss", []), dtype=float)
    val = np.asarray(history.get("val_loss", []), dtype=float)
    if train.size == 0:
        return
    epochs = np.arange(1, len(train) + 1)
    path = _prepare_path(output_path)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train, marker="o", label="Train loss")
    if val.size:
        ax.plot(np.arange(1, len(val) + 1), val, marker="o", label="Validation loss")
    ax.set_xlabel("Época")
    ax.set_ylabel("MSE")
    ax.set_title(f"Curva de entrenamiento - {model_name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix_plot(y_true, y_pred, output_path: str | Path, model_name: str) -> None:
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    mask = y_true >= 0
    y_true, y_pred = y_true[mask], y_pred[mask]
    if y_true.size == 0:
        return
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    path = _prepare_path(output_path)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    image = ax.imshow(cm)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal", "Outlier"])
    ax.set_yticklabels(["Normal", "Outlier"])
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Etiqueta de referencia")
    ax.set_title(f"Matriz de confusión - {model_name}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_precision_recall_curve(y_true, scores, output_path: str | Path, model_name: str) -> None:
    import matplotlib.pyplot as plt
    from sklearn.metrics import average_precision_score, precision_recall_curve

    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    mask = (y_true >= 0) & np.isfinite(scores)
    y_true, scores = y_true[mask], scores[mask]
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return
    precision, recall, _ = precision_recall_curve(y_true, scores)
    ap = average_precision_score(y_true, scores)
    path = _prepare_path(output_path)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, label=f"PR-AUC={ap:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Curva Precision-Recall - {model_name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_roc_curve(y_true, scores, output_path: str | Path, model_name: str) -> None:
    import matplotlib.pyplot as plt
    from sklearn.metrics import auc, roc_curve

    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    mask = (y_true >= 0) & np.isfinite(scores)
    y_true, scores = y_true[mask], scores[mask]
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    path = _prepare_path(output_path)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, label=f"ROC-AUC={roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"Curva ROC - {model_name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_score_distribution(y_true, scores, output_path: str | Path, model_name: str) -> None:
    import matplotlib.pyplot as plt

    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    mask = (y_true >= 0) & np.isfinite(scores)
    y_true, scores = y_true[mask], scores[mask]
    if y_true.size == 0:
        return
    path = _prepare_path(output_path)
    fig, ax = plt.subplots(figsize=(8, 5))
    normal = scores[y_true == 0]
    outlier = scores[y_true == 1]
    if normal.size:
        ax.hist(normal, bins=30, alpha=0.65, label="Normal")
    if outlier.size:
        ax.hist(outlier, bins=30, alpha=0.65, label="Outlier")
    ax.set_xlabel("Outlier score")
    ax.set_ylabel("Frecuencia")
    ax.set_title(f"Distribución de scores por clase - {model_name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_score_boxplot(y_true, scores, output_path: str | Path, model_name: str) -> None:
    import matplotlib.pyplot as plt

    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    mask = (y_true >= 0) & np.isfinite(scores)
    y_true, scores = y_true[mask], scores[mask]
    groups = []
    labels = []
    for label, name in [(0, "Normal"), (1, "Outlier")]:
        arr = scores[y_true == label]
        if arr.size:
            groups.append(arr)
            labels.append(name)
    if not groups:
        return
    path = _prepare_path(output_path)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    _boxplot(ax, groups, labels, showfliers=False)
    ax.set_ylabel("Outlier score")
    ax.set_title(f"Box plot de scores - {model_name}")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_model_comparison_plot(rows: list[dict], output_path: str | Path, level: str = "vehicle") -> None:
    import matplotlib.pyplot as plt

    filtered = [r for r in rows if str(r.get("level")) == level]
    if not filtered:
        return
    models = [str(r.get("model")) for r in filtered]
    metrics = ["precision", "recall", "f1_score", "pr_auc", "roc_auc"]
    x = np.arange(len(models))
    width = 0.15
    path = _prepare_path(output_path)
    fig, ax = plt.subplots(figsize=(11, 6))
    for idx, metric in enumerate(metrics):
        vals = [float(r.get(metric)) if r.get(metric) is not None else np.nan for r in filtered]
        ax.bar(x + (idx - 2) * width, vals, width, label=metric)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Valor")
    ax.set_title("Comparación de métricas por modelo (nivel vehículo)")
    ax.legend(ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def save_runtime_comparison_plot(rows: list[dict], output_path: str | Path, level: str = "vehicle") -> None:
    import matplotlib.pyplot as plt

    filtered = [r for r in rows if str(r.get("level")) == level]
    if not filtered:
        return
    models = [str(r.get("model")) for r in filtered]
    train_times = [float(r.get("training_time_seconds") or 0.0) for r in filtered]
    infer_times = [float(r.get("inference_time_seconds") or 0.0) for r in filtered]
    x = np.arange(len(models))
    width = 0.36
    path = _prepare_path(output_path)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, train_times, width, label="Entrenamiento")
    ax.bar(x + width / 2, infer_times, width, label="Inferencia")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15)
    ax.set_ylabel("Segundos")
    ax.set_title("Tiempo de ejecución por modelo")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_cross_run_metric_plot(rows: list[dict], metric: str, output_path: str | Path) -> None:
    """Plot a vehicle-level metric across debug/full runs for every model.

    This chart is intended for the TFM development background. It does not
    replace the final full-run comparison.
    """
    import matplotlib.pyplot as plt

    filtered = [
        row for row in rows
        if str(row.get("level", "vehicle")) == "vehicle" and row.get(metric) is not None
    ]
    if not filtered:
        return

    def run_key(name: str):
        import re
        match = re.fullmatch(r"debug_(\d+)", name)
        if match:
            return (0, int(match.group(1)))
        if name == "full":
            return (1, 10**12)
        return (2, name)

    run_names = sorted({str(row.get("run_name")) for row in filtered}, key=run_key)
    model_names = sorted({str(row.get("model")) for row in filtered})
    x = np.arange(len(run_names))
    path = _prepare_path(output_path)
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for model in model_names:
        by_run = {
            str(row.get("run_name")): float(row[metric])
            for row in filtered
            if str(row.get("model")) == model and row.get(metric) is not None
        }
        y = [by_run.get(run_name, np.nan) for run_name in run_names]
        ax.plot(x, y, marker="o", label=model)

    ax.set_xticks(x)
    ax.set_xticklabels(run_names, rotation=15)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Ejecución")
    ax.set_ylabel(metric)
    ax.set_title(f"Evolución de {metric} entre ejecuciones")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def save_original_class_distribution_plot(class_counts: dict[str, dict[int, int]], output_path: str | Path) -> None:
    """Plot the official 0..4 temporal class distribution before binarization."""
    import matplotlib.pyplot as plt

    sources = list(class_counts.keys())
    if not sources:
        return
    classes = sorted({int(c) for counts in class_counts.values() for c in counts})
    if not classes:
        return
    path = _prepare_path(output_path)
    x = np.arange(len(classes))
    width = 0.8 / max(len(sources), 1)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for idx, source in enumerate(sources):
        values = [int(class_counts[source].get(cls, 0)) for cls in classes]
        offset = (idx - (len(sources) - 1) / 2) * width
        ax.bar(x + offset, values, width, label=source)
    ax.set_xticks(x)
    ax.set_xticklabels([str(cls) for cls in classes])
    ax.set_xlabel("Clase temporal oficial")
    ax.set_ylabel("Número de vehículos")
    ax.set_title("Distribución original de clases temporales")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
