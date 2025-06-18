from __future__ import annotations

import json
import os
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report as sk_cr,
)



class ExperimentAnalyzer:
    # ------------------------------------------------------------------ #
    #  CONSTRUCTOR
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        model: tf.keras.Model,
        test_data: tf.data.Dataset,
        cfg: dict,
        repeat_index: int,
        history=None,
        effects: np.ndarray | None = None,
        show_plots: bool = False,
        fold_index: int | None = None,
    ):
        """
        Parámetros
        ----------
        model   : tf.keras.Model ya entrenado.
        history : Objeto retornado por `model.fit()` o bien un dict de historial.
        test_data: tf.data.Dataset -> (X, y_onehot, idx) por batch.
        cfg     : dict de configuración del experimento.
        effects : Structured array con efectos de validación.
        """
        self.model = model
        self.history = history.history if hasattr(history, "history") else history
        self.cfg = cfg
        self.repeat_index = repeat_index
        self.class_names = self.cfg["dataset"].get("class_names")
        self.effects = effects
        self.show_plots = show_plots
        
        # Inclusión fold_index si kfold está activado
        k = cfg["dataset"].get("k_folds")
        if k is not None and k > 1:
            if fold_index is None:
                raise ValueError("Es necesario indicar el índice de kfold para generar correctamente los reportes.")
            self.fold_index = int(fold_index)
        else:
            self.fold_index = None
        
        
        self.BASE_DIR = Path('/content/drive/MyDrive/structure')
        exp_cfg = self.cfg.get('experiment', {})
        self.output_dir = (
            self.BASE_DIR
            / exp_cfg.get('output_root', '')
            / exp_cfg.get('output_subdir', '')
            / 'reports'
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Mapper: convierte (x, y_onehot, [idx]) -> (x, y_idx, [idx])
        def to_labels_and_idx(*batch):
            x = batch[0]
            y_onehot = batch[1]
            idx = batch[2] if len(batch) == 3 else None
            y_idx = tf.argmax(y_onehot, axis=-1)
            return (x, y_idx, idx) if idx is not None else (x, y_idx)

        # Mapea etiquetas a índices directamente
        test_data = test_data.map(
            to_labels_and_idx,
            num_parallel_calls=tf.data.AUTOTUNE
        )

        # Convertir a NumPy arrays
        self.X_val, self.y_val, self.idx_val = self._dataset_to_numpy(test_data)

    # ------------------------------------------------------------------ #
    #  MÉTODOS PÚBLICOS
    # ------------------------------------------------------------------ #
    def plot_training_curves(self) -> None:
        """Gráfica de pérdida y exactitud (train / val), ejes iniciando en cero."""
        if not self.history:
            print("** No es posible graficar curvas. Parámetro 'history' no proporcionado **")
            return

        epochs = range(1, len(self.history.get("loss", [])) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # — Loss —
        ax = axes[0]
        ax.plot(epochs, self.history.get("loss", []), label="Train Loss")
        ax.plot(epochs, self.history.get("val_loss", []), label="Val Loss")
        ax.set_title("Loss por Época")
        ax.set_xlabel("Época")
        ax.set_ylabel("Loss")
        ax.set_ylim(bottom=0)
        ax.legend()
        ax.grid(True)

        # — Accuracy —
        ax = axes[1]
        ax.plot(epochs, self.history.get("accuracy", []), label="Train Acc")
        ax.plot(epochs, self.history.get("val_accuracy", []), label="Val Acc")
        ax.set_title("Accuracy por Época")
        ax.set_xlabel("Época")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True)

        plt.tight_layout()
        plt.show()

    def confusion_matrix(self, normalize: str | None = None, show_plots: bool = True) -> None:
        """Dibuja y guarda la matriz de confusión opcionalmente normalizada."""
        y_pred = self._predict_classes(self.X_val)
        labels = (
            list(range(len(self.class_names)))
            if self.class_names
            else np.unique(np.concatenate([self.y_val, y_pred])).tolist()
        )
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

        if self.show_plots == True:
            plt.show()
            
        # Guardar
        file_path = self.output_dir / "confusion_matrix.png"
        fig.savefig(file_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"🔖 Guardado: {file_path}")
        
        

    def classification_report(self) -> None:
        """
        Evalúa el modelo sobre el set de validación, guarda en JSON el loss, la accuracy
        de la evaluación y el classification_report completo, e imprime un resumen en consola.
        Se ajusta para convertir y_val de índices a one-hot antes de evaluar.
        """
        
        # 1) Preparar y_val en one-hot para evaluación
        num_classes = self.model.output_shape[-1]
        y_true_oh = to_categorical(self.y_val, num_classes=num_classes)
    
        # 2) Evaluación en el set de validación
        batch_size = int(self.cfg.get("training", {}).get("batch_size", 32))
        results = self.model.evaluate(self.X_val, y_true_oh,
                                      batch_size=batch_size,
                                      verbose=0)
        val_loss = float(results[0])
        val_acc  = float(results[1]) if len(results) > 1 else None
    
        # 3) Generar el dict completo del classification_report
        y_pred = self._predict_classes(self.X_val)
        report_dict = sk_cr(
            self.y_val,           # etiquetas en índice
            y_pred,
            target_names=self.class_names,
            output_dict=True,
            zero_division=0
        )
    
        # 4) Construir objeto JSON con metadata y resultados
        out = {
            "experiment": {
                "name":      self.cfg["experiment"]["name"],
                "timestamp": datetime.now().isoformat(),
                "repeat_index": self.repeat_index,
            },
            "evaluation": {
                "loss":     val_loss,
                "accuracy": val_acc
            },
            "classification_report": report_dict
        }
        
        # 4.1) Incluir fold_index (si es kfold)
        if self.fold_index is not None:
            out["experiment"]["fold_index"] = int(self.fold_index)
    
        # 5) Guardar JSON
        json_path = self.output_dir / "classification_report.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)

        print(f"🔖 JSON de métricas y evaluación guardado en: {json_path}")
    
        # 6) Imprimir resumen en consola
        text_report = sk_cr(
            self.y_val,
            y_pred,
            target_names=self.class_names,
            digits=4,
            zero_division=0
        )
        print("\n📄 Classification Report Summary\n")
        print(text_report)
        print(f"\nEval loss: {val_loss:.4f}"
              f"{', Eval accuracy: '+format(val_acc, '.4f') if val_acc is not None else ''}")


    def misclassified_indices(self) -> list[int]:
        """Índices originales de muestras mal clasificadas."""
        if self.idx_val is None:
            raise ValueError("Dataset de validación sin 'idx'.")
        y_pred = self._predict_classes(self.X_val)
        mask = y_pred != self.y_val
        return self.idx_val[mask].tolist()
    
    def effect_report(self,
                      fields: list[str] = ['num_taps', 'snr_db', 'phase_offset', 'roll_off'],
                      save_csv: bool = False,
                      bins: int = 10) -> None:
        """
        Genera un único informe JSON de effects para varios campos, 
        y guarda además una gráfica PNG por campo.
        
        - fields: lista de campos a analizar en self.effects. 
                  Si es None, usa ['num_taps', 'snr_db', 'phase_offset', 'roll_off'].
        - bins: número de bins si el campo es continuo.
        
        No retorna nada; guarda:
          • effects_report.json      ← JSON con todos los fields,
          • report_<field>.png       ← gráfica de barras apiladas,
          • effects_report_<field>.csv ← CSV con datos de proporciones.
        """
    
        if self.effects is None:
            raise ValueError("No se proporcionó 'effects'.")
    
        if fields is None:
            raise ValueError("La lista 'fields' está vacía.")
    
        # Predicción y máscara de correctas
        y_pred = self._predict_classes(self.X_val)
        correct = y_pred == self.y_val
    
        # Construir dict principal
        report: dict = {
            "experiment": {
                "name": self.cfg["experiment"]["name"],
                "timestamp": datetime.now().isoformat(),
                "repeat_index": self.repeat_index,
            },
            "effects": {}
        }
        # Incluir fold_index (si es kfold)
        if self.fold_index is not None:
            report["experiment"]["fold_index"] = int(self.fold_index)
             

        for field in fields:
            if field not in self.effects.dtype.names:
                raise ValueError(f"'{field}' no existe en effects.")
    
            # DataFrame con el efecto y acierto/fallo
            df = pd.DataFrame({field: self.effects[field], "correct": correct})
    
            # Categórico vs continuo
            is_cat = df[field].dtype.kind in "iu" and df[field].nunique() <= 10
    
            # Agrupar y calcular proporciones
            if is_cat:
                grp = (
                    df.groupby(field)["correct"]
                      .value_counts(normalize=True)
                      .unstack(fill_value=0)
                      .rename(columns={True: "Éxito", False: "Error"})
                )
                x_label = field
            else:
                df["bin"] = pd.cut(df[field], bins=bins)
                grp = (
                    df.groupby("bin", observed=True)["correct"]
                      .value_counts(normalize=True)
                      .unstack(fill_value=0)
                      .rename(columns={True: "Éxito", False: "Error"})
                )
                x_label = f"{field} (binned)"
    
            # Asegurar orden: éxito abajo, error arriba
            grp = grp[["Éxito", "Error"]]
    
            # --- 1) Guardar CSV de datos ---
            if save_csv:
                csv_path = self.output_dir / f"effects_report_{field}.csv"
                grp.to_csv(csv_path, index_label=(field if is_cat else "bin"))
                print(f"🔖 Datos CSV guardados en: {csv_path}")
    
            # --- 2) Generar y guardar gráfica ---
            fig, ax = plt.subplots(figsize=(8, 5))
            grp.plot(
                kind="bar",
                stacked=True,
                edgecolor="black",
                color=["green", "red"],
                ax=ax
            )
            ax.set_xticks(np.arange(len(grp.index)))
            ax.set_xticklabels([str(i) for i in grp.index], rotation=45, ha="right")
            ax.set_xlabel(x_label)
            ax.set_ylabel("Proporción")
            ax.set_ylim(0, 1)
            ax.set_title(f"Error vs Éxito por {field} (normalizado)")
            ax.legend(loc="upper right")
            plt.tight_layout()
            
            if self.show_plots == True:
                plt.show()
    
            png_path = self.output_dir / f"report_{field}.png"
            fig.savefig(png_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"🔖 Gráfico guardado en: {png_path}")
    
            # --- 3) Añadir al JSON ---
            report["effects"][field] = {
                "is_categorical": bool(is_cat),
                "bins": None if is_cat else bins,
                "values": {
                    str(idx): {
                        "Éxito": float(row["Éxito"]),
                        "Error": float(row["Error"])
                    }
                    for idx, row in grp.iterrows()
                }
            }
    
        # --- 4) Guardar JSON único ---
        json_path = self.output_dir / "effects_report.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"🔖 Effects report JSON guardado en: {json_path}")


    # ------------------------------------------------------------------ #
    #  MÉTODOS PRIVADOS
    # ------------------------------------------------------------------ #
    @staticmethod
    def _dataset_to_numpy(val_data):
        """Convierte tf.data.Dataset a NumPy arrays: X, y, [idx]."""
        xs, ys, idxs = [], [], []
        for batch in val_data:
            if len(batch) == 3:
                x, y, idx = batch
                idxs.append(idx.numpy())
            else:
                x, y = batch
            xs.append(x.numpy())
            ys.append(y.numpy())
        X = np.concatenate(xs)
        Y = np.concatenate(ys)
        IDX = np.concatenate(idxs) if idxs else None
        return X, Y, IDX

    def _predict_classes(self, X, batch_size: int = 128) -> np.ndarray:
        """Predice clases y devuelve argmax sobre probabilidades softmax."""
        probs = self.model.predict(X, batch_size=batch_size, verbose=0)
        return np.argmax(probs, axis=-1)
