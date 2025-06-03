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
             'name', 'output_root', 'output_subdir'}
        """
        self.BASE_DIR = Path('/content/drive/MyDrive/structure')
        self.cfg = cfg
        exp_cfg = self.cfg.get('experiment', {})

        # Directorio donde están las carpetas de folds
        self.root_exp_dir = (
            self.BASE_DIR
            / exp_cfg.get('output_root')
            / exp_cfg.get('output_subdir')
        )
        
        # Discriminar entre estructura k-fold (rep_0/fold_1) y estructura rep (rep_0/)
        k = self.cfg["dataset"].get("k_folds")
        if k is not None and k > 1:
            # Buscar subdirectorios de k-fold
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
            self.report_paths = sorted(
                [
                    rep_dir / "reports" / "classification_report.json"
                    for rep_dir in self.root_exp_dir.iterdir()
                    if rep_dir.is_dir()
                    and rep_dir.name.startswith("rep_")
                    and (rep_dir / "reports" / "classification_report.json").exists()
                ]
            )
            

        self.k = len(self.report_paths)
        if self.k == 0:
            raise FileNotFoundError(f"No se encontraron reports en {self.root_exp_dir}")

        # Cargar todos los JSON de cada fold
        self.reports = [
            json.loads(path.read_text(encoding='utf-8'))
            for path in self.report_paths
        ]

    def aggregate_evaluation(self) -> pd.DataFrame:
        """Devuelve un DataFrame con loss y accuracy por fold, más la media y desviación estándar."""
        data = []
        for r in self.reports:
            fold = r['experiment'].get('fold_index')
            loss = r['evaluation']['loss']
            acc  = r['evaluation'].get('accuracy')
            data.append({'fold': fold, 'loss': loss, 'accuracy': acc})
        df = pd.DataFrame(data).set_index('fold').sort_index()
        stats = df.agg(['mean', 'std'])
        summary = pd.concat([df, stats.rename(index={'mean': 'mean', 'std': 'std'})])
        return summary

    def aggregate_classification(self) -> pd.DataFrame:
        """Promedia métricas de clasificación por clase sobre todos los folds."""
        dfs = []
        for r in self.reports:
            cr = pd.DataFrame(r['classification_report']).T
            cr = cr.drop(index=['accuracy', 'macro avg', 'weighted avg'], errors='ignore')
            dfs.append(cr[['precision', 'recall', 'f1-score']])
        combined = pd.concat(dfs, keys=range(self.k), names=['fold', 'class'])
        avg = combined.groupby('class').mean()
        return avg

    def plot_evaluation(self) -> None:
        """Grafica loss y accuracy por fold con barras de error (desviación estándar)."""
        summary = self.aggregate_evaluation()
        fold_df = summary.head(self.k)

        losses = fold_df['loss']
        accs   = fold_df['accuracy']

        std_loss = losses.std(ddof=1)
        std_acc  = accs.std(ddof=1)

        loss_max   = losses.max()
        acc_max    = accs.max()
        loss_margin = loss_max * 0.10
        acc_margin  = acc_max  * 0.10

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].errorbar(range(self.k), losses,
                         yerr=std_loss, fmt='o-', capsize=5)
        axes[0].set_title('Loss por Fold')
        axes[0].set_xlabel('Fold')
        axes[0].set_ylabel('Loss')
        axes[0].set_ylim(0, loss_max + loss_margin)

        axes[1].errorbar(range(self.k), accs,
                         yerr=std_acc, fmt='o-', capsize=5)
        axes[1].set_title('Accuracy por Fold')
        axes[1].set_xlabel('Fold')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_ylim(0, acc_max + acc_margin)

        plt.tight_layout()
        plt.show()

    def plot_accuracy_summary(self, confidence: float = 0.95) -> None:
        """Grafica una sola barra con la media de accuracy y su intervalo de confianza,
        e imprime media, límite inferior y superior."""
        # Extraer las k accuracies
        summary    = self.aggregate_evaluation()
        accuracies = summary.head(self.k)['accuracy']
        k          = self.k
    
        # Estadísticos
        mean_acc = accuracies.mean()
        std_acc  = accuracies.std(ddof=1)
        se_acc   = std_acc / np.sqrt(k)
    
        # Valor crítico t para el intervalo
        alpha  = 1 - confidence
        df     = k - 1
        tcrit  = t.ppf(1 - alpha/2, df)
        ci     = tcrit * se_acc
    
        # Cálculo de límites
        lb = mean_acc - ci
        ub = mean_acc + ci
    
        # Plot
        fig, ax = plt.subplots(figsize=(4, 6))
        ax.bar(
            0, mean_acc,
            yerr=ci,
            capsize=10,
            width=0.6,
            label=f"{k}-fold CV"
        )
        ax.set_xticks([0])
        ax.set_xticklabels(["Accuracy"])
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Media de Accuracy con {int(confidence*100)}% IC")
        ax.legend()
    
        # Un poco de espacio extra arriba
        ax.set_ylim(0, ub + 0.05)
    
        # Anotar valores
        # Media
        ax.text(
            0, mean_acc + 0.01,
            f"Media: {mean_acc:.3f}",
            ha='center', va='bottom', fontweight='bold'
        )
        # Límite inferior
        ax.text(
            0, lb - 0.01,
            f"LI: {lb:.3f}",
            ha='center', va='top', color='gray'
        )
        # Límite superior
        ax.text(
            0, ub + 0.01,
            f"LS: {ub:.3f}",
            ha='center', va='bottom', color='gray'
        )
    
        plt.tight_layout()
        plt.show()

    def report_summary(self) -> None:
        """Imprime en consola el resumen de evaluación y clasificación agregados."""
        print("\n=== Resumen de evaluación por Fold ===")
        print(self.aggregate_evaluation())
        print("\n=== Métricas promedio de Clasificación ===")
        print(self.aggregate_classification())
