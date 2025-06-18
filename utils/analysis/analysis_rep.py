import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import t
from IPython.display import display
import seaborn as sns


class ExperimentRepAnalyzer:
    """
    Agrega y resume los resultados de un experimento con Repeticiones + (opcionalmente) K-Fold.
    Incluye:
      - Estadísticas de Loss y Accuracy por Fold (media, std, IC).
      - Estadísticas de clasificación por clase (precision, recall, f1 con IC).
      - Heatmaps de métricas por (Repetición x Fold).
      - Gráficas de barras por clase para métricas de clasificación.
      - Distribución de soporte por clase.
      - Dashboard que consolida las principales visualizaciones.
    """

    def __init__(self, cfg: dict):
        """
        cfg: dict de configuración del experimento, debe incluir 'experiment': {
             'output_root', 'output_subdir'}
        """
        # --- Inicialización de rutas y flags ---
        self.BASE_DIR = Path('/content/drive/MyDrive/structure')
        self.cfg = cfg
        exp_cfg = self.cfg.get('experiment', {})

        self.root_exp_dir = (
            self.BASE_DIR
            / exp_cfg.get('output_root')
            / exp_cfg.get('output_subdir')
        )

        k = self.cfg["dataset"].get("k_folds", None)
        self.is_k_fold = bool(k and k > 1)

        # --- Construcción de la lista de JSONs existentes ---
        if self.is_k_fold:
            self.report_paths = sorted(
                [
                    fold_dir / 'reports' / 'classification_report.json'
                    for rep_dir in self.root_exp_dir.iterdir()
                    if rep_dir.is_dir() and rep_dir.name.startswith("rep_")
                    for fold_dir in rep_dir.iterdir()
                    if (fold_dir.is_dir() and fold_dir.name.startswith("fold_")
                        and (fold_dir / 'reports' / 'classification_report.json').exists())
                ]
            )
        else:
            self.report_paths = sorted(
                [
                    rep_dir / 'reports' / 'classification_report.json'
                    for rep_dir in self.root_exp_dir.iterdir()
                    if (rep_dir.is_dir() and rep_dir.name.startswith("rep_")
                        and (rep_dir / 'reports' / 'classification_report.json').exists())
                ]
            )

        if not self.report_paths:
            raise FileNotFoundError(f"No se encontraron reports en {self.root_exp_dir}")

        # --- Lectura de cada JSON para construir un DataFrame “long” ---
        # Cada fila: rep, fold, loss, accuracy
        self.reports = []
        for json_path in self.report_paths:
            j = json.loads(json_path.read_text(encoding='utf-8'))
            rep_idx = j['experiment']['repeat_index']
            fold_idx = j['experiment'].get('fold_index', 0)
            loss = j['evaluation']['loss']
            acc  = j['evaluation'].get('accuracy', None)
            self.reports.append({
                'rep': rep_idx,
                'fold': fold_idx,
                'loss': loss,
                'accuracy': acc,
                'json_path': json_path
            })

        self.df_all = (
            pd.DataFrame(self.reports)
              .sort_values(['rep', 'fold'])
              .reset_index(drop=True)
        )

        self.k = int(self.df_all['fold'].nunique())
        self.num_reps = int(self.df_all['rep'].nunique())

    # ----------------------
    # Métodos Privados Útiles
    # ----------------------
    def _t_confidence_interval(self, data: np.ndarray, confidence: float = 0.95):
        """
        Dado array “data”, retorna:
          (half_width_ci, lower_bound, upper_bound).
        Usa t-Student con n-1 grados de libertad.
        Si len(data)<2, retorna (0, media, media).
        """
        n = len(data)
        mean = data.mean()
        if n < 2 or np.allclose(data, mean):
            return 0.0, mean, mean
        std = data.std(ddof=1)
        se = std / np.sqrt(n)
        alpha = 1 - confidence
        dfree = n - 1
        tcrit = t.ppf(1 - alpha/2, dfree)
        ci = tcrit * se
        return ci, mean - ci, mean + ci

    # ----------------------
    # Estadísticas: Loss / Accuracy
    # ----------------------
    def aggregate_evaluation(self, confidence: float = 0.95) -> pd.DataFrame:
        """
        Para cada fold, calcula:
          - loss_mean, loss_std, loss_ci_lower, loss_ci_upper
          - accuracy_mean, accuracy_std, accuracy_ci_lower, accuracy_ci_upper
          - n_reps (cuántas repeticiones efectivas tuvo ese fold)
        Retorna DataFrame indexado por 'fold'.
        """
        records = []
        for fold_idx, grp in self.df_all.groupby('fold'):
            losses = grp['loss'].to_numpy()
            accs   = grp['accuracy'].to_numpy()

            # Loss
            loss_mean = losses.mean()
            loss_std  = losses.std(ddof=1) if len(losses) > 1 else 0.0
            loss_ci, loss_lb, loss_ub = self._t_confidence_interval(losses, confidence)

            # Accuracy
            acc_mean = accs.mean()
            acc_std  = accs.std(ddof=1) if len(accs) > 1 else 0.0
            acc_ci, acc_lb, acc_ub = self._t_confidence_interval(accs, confidence)

            records.append({
                'fold': fold_idx,
                'loss_mean': loss_mean,
                'loss_std': loss_std,
                'loss_ci_lower': loss_lb,
                'loss_ci_upper': loss_ub,
                'accuracy_mean': acc_mean,
                'accuracy_std': acc_std,
                'accuracy_ci_lower': acc_lb,
                'accuracy_ci_upper': acc_ub,
                'n_reps': len(grp)
            })

        df_summary = pd.DataFrame.from_records(records).set_index('fold').sort_index()
        return df_summary

    # ----------------------
    # Estadísticas: Clasificación por Clase
    # ----------------------
    def aggregate_classification(self, confidence: float = 0.95) -> pd.DataFrame:
        """
        Para cada clase en todos los classification_report.json, calcula:
          - precision_mean, precision_std, precision_ci_lower, precision_ci_upper
          - recall_mean, recall_std, recall_ci_lower, recall_ci_upper
          - f1_mean, f1_std, f1_ci_lower, f1_ci_upper
          - support_mean, support_std
        Retorna DataFrame indexado por 'class'.
        """
        records = []
        for json_path in self.report_paths:
            j = json.loads(json_path.read_text(encoding='utf-8'))
            cr = j.get('classification_report', {})

            for cls_label, metrics in cr.items():
                if cls_label in ['accuracy', 'macro avg', 'weighted avg']:
                    continue
                records.append({
                    'class': cls_label,
                    'precision': metrics.get('precision', np.nan),
                    'recall':    metrics.get('recall', np.nan),
                    'f1_score':  metrics.get('f1-score', np.nan),
                    'support':   metrics.get('support', np.nan)
                })

        if not records:
            return pd.DataFrame()

        df_cr = pd.DataFrame.from_records(records)
        stats = []
        for cls_label, grp in df_cr.groupby('class'):
            pres = grp['precision'].to_numpy()
            recs = grp['recall'].to_numpy()
            f1s  = grp['f1_score'].to_numpy()
            sups = grp['support'].to_numpy()

            # Precision
            p_mean = pres.mean()
            p_std  = pres.std(ddof=1) if len(pres) > 1 else 0.0
            p_ci, p_lb, p_ub = self._t_confidence_interval(pres, confidence)
            # Recall
            r_mean = recs.mean()
            r_std  = recs.std(ddof=1) if len(recs) > 1 else 0.0
            r_ci, r_lb, r_ub = self._t_confidence_interval(recs, confidence)
            # F1
            f_mean = f1s.mean()
            f_std  = f1s.std(ddof=1) if len(f1s) > 1 else 0.0
            f_ci, f_lb, f_ub = self._t_confidence_interval(f1s, confidence)
            # Support
            s_mean = sups.mean()
            s_std  = sups.std(ddof=1) if len(sups) > 1 else 0.0

            stats.append({
                'class': cls_label,
                'precision_mean': p_mean,
                'precision_std': p_std,
                'precision_ci_lower': p_lb,
                'precision_ci_upper': p_ub,
                'recall_mean': r_mean,
                'recall_std': r_std,
                'recall_ci_lower': r_lb,
                'recall_ci_upper': r_ub,
                'f1_mean': f_mean,
                'f1_std': f_std,
                'f1_ci_lower': f_lb,
                'f1_ci_upper': f_ub,
                'support_mean': s_mean,
                'support_std': s_std
            })

        df_stats = pd.DataFrame.from_records(stats).set_index('class').sort_index()
        return df_stats

    # ----------------------
    # Gráficas: Heatmaps / Barras
    # ----------------------
    def plot_evaluation(self, confidence: float = 0.95) -> None:
        """
        Grafica, para cada fold, la media de Loss y Accuracy a través de las repeticiones,
        con barras de error que representan los intervalos de confianza.
        """
        # Obtenemos el DataFrame con estadísticas por fold
        df_summary = self.aggregate_evaluation(confidence=confidence)
        folds = df_summary.index.to_list()

        # Extraer medias y límites del IC para loss
        loss_means = df_summary['loss_mean']
        loss_lbs   = df_summary['loss_ci_lower']
        loss_ubs   = df_summary['loss_ci_upper']
        # Para la barra de error usamos (mean - lower_bound)
        loss_err = loss_means - loss_lbs

        # Extraer medias y límites del IC para accuracy
        acc_means = df_summary['accuracy_mean']
        acc_lbs   = df_summary['accuracy_ci_lower']
        acc_ubs   = df_summary['accuracy_ci_upper']
        acc_err  = acc_means - acc_lbs

        # Márgenes para los ejes (tomamos el máximo de ub + 10% de margen)
        loss_max    = (loss_ubs).max()
        loss_margin = loss_max * 0.10
        acc_max     = (acc_ubs).max()
        acc_margin  = acc_max * 0.10

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # --- PLOT LOSS con intervalo de confianza ---
        axes[0].errorbar(
            folds,
            loss_means,
            yerr=loss_err,
            fmt='o-',
            capsize=5,
            label='Loss ± IC'
        )
        axes[0].set_title('Loss con IC por Fold')
        axes[0].set_xlabel('Fold')
        axes[0].set_ylabel('Loss')
        axes[0].set_ylim(0, loss_max + loss_margin)
        axes[0].legend()

        # --- PLOT ACCURACY con intervalo de confianza ---
        axes[1].errorbar(
            folds,
            acc_means,
            yerr=acc_err,
            fmt='o-',
            capsize=5,
            label='Accuracy ± IC'
        )
        axes[1].set_title('Accuracy con IC por Fold')
        axes[1].set_xlabel('Fold')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_ylim(0, acc_max + acc_margin)
        axes[1].legend()

        plt.tight_layout()
        plt.show()

    def plot_heatmap_metric(self, metric: str = 'accuracy', cmap: str = 'viridis') -> None:
        """
        Dibuja un heatmap donde:
          - Eje X = fold
          - Eje Y = repeticiones (rep)
          - Valor = métrica (p.ej. 'accuracy' o 'loss')
        Útil para visualizar la variabilidad conjunta (rep x fold).
        """
        if metric not in ['accuracy', 'loss']:
            raise ValueError("metric debe ser 'accuracy' o 'loss'")

        # Pivot table: index=rep, columns=fold, values=metric
        pivot = self.df_all.pivot(index='rep', columns='fold', values=metric)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".3f",
            cmap=cmap,
            cbar_kws={'label': metric},
            linewidths=0.5
        )
        plt.title(f"Heatmap de `{metric}` por Rep vs Fold")
        plt.xlabel("Fold")
        plt.ylabel("Repetición")
        plt.tight_layout()
        plt.show()

    def plot_classification_bars(self, metric: str = 'f1', confidence: float = 0.95) -> None:
        """
        Grafica, para cada clase, la media ± IC de la métrica seleccionada
        (puede ser 'precision', 'recall' o 'f1'), según lo calculado en aggregate_classification().
        """
        df_cls = self.aggregate_classification(confidence=confidence)
        if df_cls.empty:
            print("No hay datos de clasificación para graficar.")
            return

        if metric not in ['precision', 'recall', 'f1']:
            raise ValueError("metric debe ser 'precision', 'recall' o 'f1'")

        # Construir columnas para la métrica elegida
        mean_col = f"{metric}_mean"
        std_col  = f"{metric}_std"
        lb_col   = f"{metric}_ci_lower"
        ub_col   = f"{metric}_ci_upper"

        classes = df_cls.index.to_list()
        means = df_cls[mean_col].to_numpy()
        lbs   = df_cls[lb_col].to_numpy()
        # Para errorbars, usamos mean - lb
        errs  = means - lbs

        x = np.arange(len(classes))
        plt.figure(figsize=(10, 6))
        plt.bar(x, means, yerr=errs, capsize=5, alpha=0.7)
        plt.xticks(x, classes, rotation=45, ha='right')
        plt.ylabel(metric.capitalize())
        plt.title(f"{metric.capitalize()} promedio por clase ± IC ({int(confidence*100)}%)")
        plt.tight_layout()
        plt.show()

    def plot_support_distribution(self, confidence: float = 0.95) -> None:
        """
        Muestra un gráfico de barras con el soporte (support) promedio por clase
        y su desviación estándar, a partir de aggregate_classification().
        """
        df_cls = self.aggregate_classification(confidence=confidence)
        if df_cls.empty:
            print("No hay datos de clasificación para graficar soportes.")
            return

        means = df_cls['support_mean']
        stds  = df_cls['support_std']
        classes = df_cls.index.to_list()
        x = np.arange(len(classes))

        plt.figure(figsize=(10, 6))
        plt.bar(x, means, yerr=stds, capsize=5, alpha=0.7, color='skyblue')
        plt.xticks(x, classes, rotation=45, ha='right')
        plt.ylabel("Support (promedio)")
        plt.title("Distribución de Support por Clase (media ± std)")
        plt.tight_layout()
        plt.show()

    def show_dashboard(self, confidence: float = 0.95) -> None:
        """
        Despliega un “dashboard” con las principales visualizaciones:
          1) Tabla de aggregate_evaluation (loss/accuracy por fold con IC).
          2) Heatmap de accuracy por rep x fold.
          3) Gráfico de barras de F1-score por clase.
          4) Gráfico de distribución de support por clase.
        """
        print("\n=== Tabla de Evaluación (Fold) ===")
        df_eval = self.aggregate_evaluation(confidence=confidence)
        display(df_eval)  # Si estás en Jupyter/Colab
        print("\n")

        print("=== Heatmap de Accuracy por Rep x Fold ===")
        self.plot_heatmap_metric(metric='accuracy')

        print("=== Heatmap de Loss por Rep x Fold ===")
        self.plot_heatmap_metric(metric='loss')

        print("\n=== Barras de F1-score por Clase con IC ===")
        self.plot_classification_bars(metric='f1', confidence=confidence)

        print("\n=== Distribución de Support por Clase ===")
        self.plot_support_distribution(confidence=confidence)

    def report_summary(self, confidence: float = 0.95) -> None:
        """
        Imprime en consola:
         1) Tabla de evaluation por fold (media, std e IC).
         2) Tabla de clasificación por clase (media, std e IC).
        """
        print("\n=== Resumen de Evaluación por Fold ===")
        print(self.aggregate_evaluation(confidence=confidence))

        print("\n=== Métricas promedio de Clasificación (Prec / Rec / F1) ===")
        print(self.aggregate_classification(confidence=confidence))
