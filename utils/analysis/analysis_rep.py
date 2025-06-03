import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import t


class ExperimentRepAnalyzer:
    """
    Agrega y resume los resultados de un experimento K-Fold o sin este leyendo los JSON de cada fold.
    """
    def __init__(self, cfg: dict):
        """
        cfg: dict de configuración del experimento, debe incluir 'experiment': {
             'output_root', 'output_subdir'}
        """
        self.BASE_DIR = Path('/content/drive/MyDrive/structure')
        self.cfg = cfg
        exp_cfg = self.cfg.get('experiment', {})

        # Directorio donde están las carpetas rep_<i>/(fold_<j>/)reports/classification_report.json
        self.root_exp_dir = (
            self.BASE_DIR
            / exp_cfg.get('output_root')
            / exp_cfg.get('output_subdir')
        )

        # Discriminar entre estructura k-fold (rep_0/fold_1) y estructura rep (rep_0/)
        k = self.cfg["dataset"].get("k_folds")
        self.is_k_fold = bool(k and k > 1)

        if self.is_k_fold:
            # Caso K-Fold: buscamos rep_<i>/fold_<j>/reports/classification_report.json
            self.report_paths = sorted(
                [
                    fold_dir / 'reports' / 'classification_report.json'
                    for rep_dir in self.root_exp_dir.iterdir()
                    if rep_dir.is_dir() and rep_dir.name.startswith("rep_")
                    for fold_dir in rep_dir.iterdir()
                    if fold_dir.is_dir() and fold_dir.name.startswith("fold_")
                    and (fold_dir / 'reports' / 'classification_report.json').exists()
                ]
            )
        else:
            # Caso solo repeticiones: buscamos rep_<i>/reports/classification_report.json
            self.report_paths = sorted(
                [
                    rep_dir / "reports" / "classification_report.json"
                    for rep_dir in self.root_exp_dir.iterdir()
                    if rep_dir.is_dir()
                    and rep_dir.name.startswith("rep_")
                    and (rep_dir / "reports" / "classification_report.json").exists()
                ]
            )

        if len(self.report_paths) == 0:
            raise FileNotFoundError(f"No se encontraron reports en {self.root_exp_dir}")

        # Leer cada JSON y extraer rep, fold, loss, accuracy
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
                'accuracy': acc
            })

        # Convertimos la lista a DataFrame para agregaciones y gráficas
        self.df_all = (
            pd.DataFrame(self.reports)
              .sort_values(['rep', 'fold'])
              .reset_index(drop=True)
        )

        # Número de folds distintos (si no hay k-fold, fold será siempre 0 → k=1)
        self.k = int(self.df_all['fold'].nunique())
        # Número de repeticiones
        self.num_reps = int(self.df_all['rep'].nunique())

    def aggregate_evaluation(self) -> pd.DataFrame:
        """
        Devuelve un DataFrame con, para cada fold:
          - loss_mean, loss_std, accuracy_mean, accuracy_std
        Agrupando sobre todas las repeticiones.
        Índice: fold (int)
        Columnas: loss_mean, loss_std, accuracy_mean, accuracy_std
        """
        df_fold = self.df_all.groupby('fold')[['loss', 'accuracy']].agg(['mean', 'std'])
        # Renombrar columnas (MultiIndex → nombres planos)
        df_fold.columns = ['loss_mean', 'loss_std', 'accuracy_mean', 'accuracy_std']
        return df_fold

    def aggregate_classification(self) -> pd.DataFrame:
        """
        Promedia métricas de clasificación por clase sobre todos los informes JSON.
        Recorre self.report_paths para leer cada JSON y extraer 'classification_report',
        luego agrupa por clase y promedia precision, recall y f1-score.
        """
        records = []
        for json_path in self.report_paths:
            j = json.loads(json_path.read_text(encoding='utf-8'))
            # Extraer índices (para saber de qué rep y fold proviene)
            rep_idx = j['experiment']['repeat_index']
            fold_idx = j['experiment'].get('fold_index', 0)
            cr = j.get('classification_report', {})

            # Iterar sobre cada clase dentro del classification_report
            for cls_label, metrics in cr.items():
                if cls_label in ['accuracy', 'macro avg', 'weighted avg']:
                    continue
                records.append({
                    'rep': rep_idx,
                    'fold': fold_idx,
                    'class': cls_label,
                    'precision': metrics.get('precision', np.nan),
                    'recall':    metrics.get('recall', np.nan),
                    'f1_score':  metrics.get('f1-score', np.nan)
                })

        if not records:
            # Si ningún report tenía classification_report, devolvemos DataFrame vacío
            return pd.DataFrame()

        df_cr = pd.DataFrame.from_records(records)
        # Agrupar por clase y calcular promedio sobre rep y fold
        df_avg = df_cr.groupby('class')[['precision', 'recall', 'f1_score']].mean()
        return df_avg

    def plot_evaluation(self) -> None:
        """
        Grafica, para cada fold, la media de Loss y Accuracy a través de las repeticiones,
        con barras de error que representan la desviación estándar.
        """
        df_summary = self.aggregate_evaluation()
        folds = df_summary.index.to_list()  # lista de índices de fold

        # Extraer medias y desviaciones
        loss_means = df_summary['loss_mean']
        loss_stds  = df_summary['loss_std']
        acc_means  = df_summary['accuracy_mean']
        acc_stds   = df_summary['accuracy_std']

        # Márgenes para los ejes
        loss_max    = (loss_means + loss_stds).max()
        loss_margin = loss_max * 0.10
        acc_max     = (acc_means + acc_stds).max()
        acc_margin  = acc_max * 0.10

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # PLOT LOSS
        axes[0].errorbar(
            folds,
            loss_means,
            yerr=loss_stds,
            fmt='o-',
            capsize=5
        )
        axes[0].set_title('Loss promedio por Fold')
        axes[0].set_xlabel('Fold')
        axes[0].set_ylabel('Loss')
        axes[0].set_ylim(0, loss_max + loss_margin)

        # PLOT ACCURACY
        axes[1].errorbar(
            folds,
            acc_means,
            yerr=acc_stds,
            fmt='o-',
            capsize=5
        )
        axes[1].set_title('Accuracy promedio por Fold')
        axes[1].set_xlabel('Fold')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_ylim(0, acc_max + acc_margin)

        plt.tight_layout()
        plt.show()

    def plot_accuracy_summary(self, confidence: float = 0.95) -> None:
        """
        Grafica una sola barra con:
         - La media de accuracy por repetición (es decir, primero promedia los k folds de cada rep).
         - El intervalo de confianza (t-Student) de esa media a través de las repeticiones.
        """
        # 1) Para cada repetición, calculamos la media de accuracy en sus k folds
        df = self.df_all.copy()
        rep_means = df.groupby('rep')['accuracy'].mean().to_numpy()
        n_reps = len(rep_means)

        # 2) Estadísticos de esas medias
        mean_acc = rep_means.mean()
        std_acc  = rep_means.std(ddof=1)
        se_acc   = std_acc / np.sqrt(n_reps)

        alpha = 1 - confidence
        dfree = n_reps - 1
        tcrit = t.ppf(1 - alpha/2, dfree)
        ci    = tcrit * se_acc

        lb = mean_acc - ci
        ub = mean_acc + ci

        # 3) Plot
        fig, ax = plt.subplots(figsize=(4, 6))
        ax.bar(
            0, mean_acc,
            yerr=ci,
            capsize=10,
            width=0.6,
            label=f"{n_reps} repeticiones"
        )
        ax.set_xticks([0])
        ax.set_xticklabels(["Accuracy"])
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Media de Accuracy con {int(confidence*100)}% IC")
        ax.legend()

        # Espacio extra arriba
        ax.set_ylim(0, ub + 0.05)

        # Anotar valores
        ax.text(
            0, mean_acc + 0.01,
            f"Media: {mean_acc:.3f}",
            ha='center', va='bottom', fontweight='bold'
        )
        ax.text(
            0, lb - 0.01,
            f"LI: {lb:.3f}",
            ha='center', va='top', color='gray'
        )
        ax.text(
            0, ub + 0.01,
            f"LS: {ub:.3f}",
            ha='center', va='bottom', color='gray'
        )

        plt.tight_layout()
        plt.show()

    def report_summary(self) -> None:
        """
        Imprime en consola:
         1) Resumen de evaluación por Fold (media y desviación).
         2) Métricas promedio de clasificación por clase.
        """
        print("\n=== Resumen de Evaluación por Fold ===")
        print(self.aggregate_evaluation())

        print("\n=== Métricas promedio de Clasificación (Prec / Rec / F1) ===")
        print(self.aggregate_classification())
