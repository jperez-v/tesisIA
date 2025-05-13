"""
utils/analysis/analysis.py
────────────────────────────────────────────────────────────────────────────
Herramientas de post‑análisis para modelos Keras. 100% compatibles con:

•  tf.data.Dataset  →  (X, y)  o  (X, y, idx)
•  Tuplas NumPy     →  (X_val, y_val)

Incluye función para recuperar los índices originales de las muestras
mal clasificadas cuando el índice se haya añadido en el Dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)


class ExperimentAnalyzer:
    # ------------------------------------------------------------------ #
    #  CONSTRUCTOR
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        model,
        history,
        val_data,
        class_names: list[str] | None = None,
        effects: np.ndarray | None = None,
    ):
        """
        Parameters
        ----------
        model        : tf.keras.Model ya entrenado.
        history      : Objeto retornado por `model.fit()` o bien `history.history`.
        val_data     : • tf.data.Dataset -> (X, y)  o  (X, y, idx)
                       • tupla (X_val, y_val)  (NumPy)  *sin índice*.
        class_names  : Lista opcional con nombres legibles de las clases.
        """
        self.model = model
        self.history = history.history if hasattr(history, "history") else history
        self.class_names = class_names
        self.effects = effects

        # Convierte val_data a arrays (X, y, idx | None)
        self.X_val, self.y_val, self.idx_val = self._dataset_to_numpy(val_data)


    # ------------------------------------------------------------------ #
    #  MÉTODOS PÚBLICOS
    # ------------------------------------------------------------------ #
    def plot_training_curves(self) -> None:
        """Gráfica de pérdida y exactitud (train / val)."""
        epochs = range(1, len(self.history["loss"]) + 1)

        plt.figure(figsize=(12, 4))

        # — Loss —
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.history["loss"], label="Train Loss")
        plt.plot(epochs, self.history["val_loss"], label="Val Loss")
        plt.title("Loss por Época")
        plt.xlabel("Época"); plt.ylabel("Loss"); plt.legend()

        # — Accuracy —
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.history["accuracy"], label="Train Acc")
        plt.plot(epochs, self.history["val_accuracy"], label="Val Acc")
        plt.title("Accuracy por Época")
        plt.xlabel("Época"); plt.ylabel("Accuracy"); plt.legend()

        plt.tight_layout(); plt.show()

    # ------------------------------------------------------------------ #
    def confusion_matrix(self, normalize: str | None = None):
        """
        Dibuja la matriz de confusión.

        normalize : 'true', 'pred', 'all' o None.
        Siempre fuerza el número de etiquetas a coincidir con `class_names`
        (si se proporcionan) para evitar desajustes de ticks.
        """
        y_pred = self._predict_classes(self.X_val)

        # — Selección de labels —
        if self.class_names is not None:
            n_labels = len(self.class_names)
            labels = list(range(n_labels))
        else:
            labels = np.unique(np.concatenate([self.y_val, y_pred])).tolist()
        
        cm = confusion_matrix(self.y_val, y_pred,
                              labels=labels, normalize=normalize)

        disp = ConfusionMatrixDisplay(cm,
                                      display_labels=self.class_names or labels)
        fig, ax = plt.subplots(figsize=(7, 7))
        disp.plot(ax=ax, cmap="Blues", xticks_rotation=90, colorbar=False)

        norm_txt = f" (normalizada={normalize})" if normalize else ""
        ax.set_title(f"Matriz de Confusión{norm_txt}")
        plt.show()

    # ------------------------------------------------------------------ #
    def classification_report(self) -> None:
        """Imprime precisión, recall y F1 por clase."""
        y_pred = self._predict_classes(self.X_val)
        report = classification_report(
            self.y_val,
            y_pred,
            target_names=self.class_names,
            digits=4,
            zero_division=0,
        )
        print("\n📄 Classification Report\n")
        print(report)

    # ------------------------------------------------------------------ #
    def misclassified_indices(self) -> list[int]:
        """
        Retorna los índices originales (en el HDF5) de las muestras cuya
        predicción es incorrecta.  Requiere que `val_data` incluya índices.
        """
        if self.idx_val is None:
            raise ValueError(
                "El Dataset de validación no incluye índices. "
                "Inicializa tu Dataset con `include_index=True`."
            )
        y_pred = self._predict_classes(self.X_val)
        mask = y_pred != self.y_val
        return self.idx_val[mask].tolist()

    # ------------------------------------------------------------------ #
    #  MÉTODOS PRIVADOS / HELPERS
    # ------------------------------------------------------------------ #
    @staticmethod
    def _dataset_to_numpy(val_data):
        """
        Convierte:
          • tf.data.Dataset → concatena lotes en arrays NumPy.
          • (X_val, y_val)  → Devuelve tal cual + idx=None.
        Devuelve (X, y, idx_or_None).
        """
        if isinstance(val_data, tuple):
            X, y = val_data
            return X, y, None

        xs, ys, idxs = [], [], []
        for batch in val_data:
            # Permite batches (X, y)   o   (X, y, idx)
            if len(batch) == 3:
                x, y, idx = batch
                idxs.append(idx.numpy())
            else:
                x, y = batch
            xs.append(x.numpy());  ys.append(y.numpy())
        X = np.concatenate(xs)
        Y = np.concatenate(ys)
        IDX = np.concatenate(idxs) if idxs else None
        return X, Y, IDX

    # ------------------------------------------------------------------ #
    def _predict_classes(self, X, batch_size: int = 512):
        """Aplica el modelo y devuelve argmax sobre el eje de clases."""
        probs = self.model.predict(X, batch_size=batch_size, verbose=0)
        return np.argmax(probs, axis=-1)


    # ------------------------------------------------------------------ #
    def effect_diagnostics(self, field: str):
        """
        Visualiza cómo un *effect* influye en los fallos de clasificación.
        field debe ser una columna de self.effects
        """
        if self.effects is None:
            raise ValueError("No se pasó el structured array 'effects'.")

        if field not in self.effects.dtype.names:
            raise ValueError(f"'{field}' no existe en Effects.")

        y_pred = self._predict_classes(self.X_val)
        correct = y_pred == self.y_val
        df = pd.DataFrame({
            field: self.effects[field],
            "correct": correct
        })

        # Categórico si ≤10 valores únicos; continuo en caso contrario
        if df[field].dtype.kind in "iu" and df[field].nunique() <= 10:
            plt.figure(figsize=(6,4))
            sns.countplot(x=field, hue="correct", data=df, palette="Set2")
            plt.title(f"Errores vs {field}")
            plt.ylabel("nº de señales"); plt.show()
        else:  # continuo
            plt.figure(figsize=(6,4))
            sns.histplot(data=df, x=field, hue="correct",
                         bins=20, element="step", stat="density", common_norm=False)
            plt.title(f"Distribución de {field} (correct / fail)")
            plt.show()