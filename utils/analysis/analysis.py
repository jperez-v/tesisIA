from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from pathlib import Path
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
        model: tf.keras.Model,
        val_data: tf.data.Dataset,
        cfg: dict,
        history = None,
        effects: np.ndarray | None = None,
    ):
        """
        Parámetros
        ----------
        model        : tf.keras.Model ya entrenado.
        history      : Objeto retornado por `model.fit()` o bien un dict de historial.
        val_data     : tf.data.Dataset -> (X, y_onehot, idx) por batch.
        cfg          : dict de configuración del experimento
        effects      : Structured array con efectos de validación.
        """
        self.model = model
        self.history = history.history if hasattr(history, "history") else history
        self.cfg = cfg
        self.class_names = self.cfg["dataset"].get("class_names")
        self.effects = effects
        
        self.BASE_DIR = Path('/content/drive/MyDrive/structure')
        self.output_dir = self.BASE_DIR / self.cfg.get('experiment').get('output_root') / self.cfg.get('experiment').get('output_subdir') / 'reports'
    
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        
        # Mapper: (x, y_onehot) o (x, y_onehot, idx) -> (x, y_idx, idx_or_None)
        def to_labels_and_idx(*batch):
            x = batch[0]
            y_onehot = batch[1]
            idx = batch[2] if len(batch) == 3 else None
            y_idx = tf.argmax(y_onehot, axis=-1)
            return (x, y_idx, idx) if idx is not None else (x, y_idx)

        # Aplicar siempre map al dataset de validación
        val_data = val_data.map(
            to_labels_and_idx,
            num_parallel_calls=tf.data.AUTOTUNE
        )

        # Convertir a NumPy arrays
        self.X_val, self.y_val, self.idx_val = self._dataset_to_numpy(val_data)

    # ------------------------------------------------------------------ #
    #  MÉTODOS PÚBLICOS
    # ------------------------------------------------------------------ #
    def plot_training_curves(self) -> None:
        """Gráfica de pérdida y exactitud (train / val), ejes iniciando en cero."""
        if not self.history:
            print("** No es posible graficar las curvas de entrenamiento. No se ha proporcionado el parámetro 'history' **")
        epochs = range(1, len(self.history["loss"]) + 1)

        plt.figure(figsize=(12, 4))

        # — Loss —
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.history["loss"], label="Train Loss")
        plt.plot(epochs, self.history["val_loss"], label="Val Loss")
        plt.title("Loss por Época")
        plt.xlabel("Época")
        plt.ylabel("Loss")
        plt.ylim(bottom=0)
        plt.legend()

        # — Accuracy —
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.history["accuracy"], label="Train Acc")
        plt.plot(epochs, self.history["val_accuracy"], label="Val Acc")
        plt.title("Accuracy por Época")
        plt.xlabel("Época")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1)
        plt.legend()

        plt.tight_layout()
        plt.show()
        
    def confusion_matrix(self, normalize: str | None = None) -> None:
        """
        Dibuja la matriz de confusión.

        normalize : 'true', 'pred', 'all' o None.
        """
        y_pred = self._predict_classes(self.X_val)

        # Definir labels
        if self.class_names is not None:
            labels = list(range(len(self.class_names)))
        else:
            labels = np.unique(np.concatenate([self.y_val, y_pred])).tolist()

        cm = confusion_matrix(
            self.y_val, y_pred,
            labels=labels,
            normalize=normalize
        )

        disp = ConfusionMatrixDisplay(
            cm,
            display_labels=self.class_names or labels
        )
        fig, ax = plt.subplots(figsize=(7, 7))
        disp.plot(ax=ax, cmap="Blues", xticks_rotation=90, colorbar=False)
        title = "Matriz de Confusión"
        if normalize:
            title += f" (normalizada={normalize})"
        ax.set_title(title)
        
        # Guardar
        file_path = self.output_dir / "confusion_matrix.png"
        fig.savefig(file_path, dpi=300)
        print(f"🔖 Gráfica guardada en: {file_path}")
        
        plt.show()

    def classification_report(self) -> None:
        """Imprime precisión, recall y F1 por clase."""
        y_pred = self._predict_classes(self.X_val)
        report = classification_report(
            self.y_val,
            y_pred,
            target_names=self.class_names,
            digits=4,
            zero_division=0
        )
        print("\n📄 Classification Report\n")
        print(report)

    def misclassified_indices(self) -> list[int]:
        """
        Retorna los índices originales de las muestras mal clasificadas.
        """
        if self.idx_val is None:
            raise ValueError(
                "El Dataset de validación no incluye índices."
            )
        y_pred = self._predict_classes(self.X_val)
        mask = y_pred != self.y_val
        return self.idx_val[mask].tolist()

    def effect_diagnostics(self, field: str, bins: int = 10) -> None:
        """
        Visualiza cómo un efecto influye en los fallos de clasificación
        usando barras apiladas normalizadas (éxito / error suman 1),
        y opcionalmente guarda la gráfica en PNG.

        Parámetros
        ----------
        field : str
            Nombre del campo en self.effects para analizar.
        bins : int, opcional
            Número de bins a usar si el campo es continuo.
        save_path : str | None, opcional
            Ruta de archivo donde guardar la figura (PNG). Si es None, no guarda.
        """
        if self.effects is None:
            raise ValueError("No se proporcionó 'effects'.")
        if field not in self.effects.dtype.names:
            raise ValueError(f"'{field}' no existe en Effects.")

        # Predicciones y mask de correctas
        y_pred = self._predict_classes(self.X_val)
        correct = y_pred == self.y_val

        df = pd.DataFrame({ field: self.effects[field], "correct": correct })
        is_cat = df[field].dtype.kind in "iu" and df[field].nunique() <= 10

        # Preparamos figura
        fig, ax = plt.subplots(figsize=(8, 5))

        if is_cat:
            prop = (
                df
                .groupby(field)["correct"]
                .value_counts(normalize=True)
                .unstack(fill_value=0)
                .rename(columns={True: "success", False: "error"})
            )
            prop.plot(kind="bar", stacked=True, edgecolor="black", ax=ax)
            ax.set_xlabel(field)
            title = f"Éxito vs Error por {field} (normalizado)"
        else:
            df["bin"] = pd.cut(df[field], bins=bins)
            prop = (
                df
                .groupby("bin", observed=True)["correct"]
                .value_counts(normalize=True)
                .unstack(fill_value=0)
                .rename(columns={True: "success", False: "error"})
            )
            prop.plot(kind="bar", stacked=True, edgecolor="black", ax=ax)
            ax.set_xlabel(f"{field} (binned)")
            ax.set_xticklabels(prop.index.astype(str), rotation=45, ha="right")
            title = f"Éxito vs Error en bins de {field} (normalizado)"

        ax.set_ylabel("Proporción")
        ax.set_title(title)
        ax.legend(["Éxito", "Error"], loc="upper right")
        ax.set_ylim(0, 1)
        plt.tight_layout()

        # Guardar
        file_path = self.output_dir /  f"report_{field}.png"
        fig.savefig(file_path, dpi=300, bbox_inches="tight")
        print(f"🔖 Gráfica guardada en: {file_path}")

        plt.show()

    # ------------------------------------------------------------------ #
    #  MÉTODOS PRIVADOS
    # ------------------------------------------------------------------ #
    @staticmethod
    def _dataset_to_numpy(val_data):
        """
        Convierte tf.data.Dataset a arrays NumPy.
        Devuelve (X, y, idx_or_None).
        """
        xs, ys, idxs = [], [], []
        for batch in val_data:
            if len(batch) == 3:
                x, y, idx = batch
                idxs.append(idx.numpy())
            else:
                x, y = batch
            xs.append(x.numpy()); ys.append(y.numpy())
        X = np.concatenate(xs)
        Y = np.concatenate(ys)
        IDX = np.concatenate(idxs) if idxs else None
        return X, Y, IDX

    def _predict_classes(self, X, batch_size: int = 512) -> np.ndarray:
        """Predice clases y devuelve argmax sobre probas softmax."""
        probs = self.model.predict(X, batch_size=batch_size, verbose=0)
        return np.argmax(probs, axis=-1)
