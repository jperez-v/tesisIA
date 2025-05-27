from pathlib import Path
from tabulate import tabulate

# Rutas base en Drive 
CONFIG_ROOT = Path('/content/drive/MyDrive/structure/configs')
MODELS_ROOT = Path('/content/drive/MyDrive/structure/models')
DATA_ROOT   = Path('/content/drive/MyDrive/structure/datasets')


def print_exp_configuration(cfg: dict):
    """
    Muestra de forma legible la configuración del experimento:
      1) EXPERIMENTO   (nombre, descripción, módulo, clase, salida)
      2) DATASET       (fuente, ruta/ID, splits)
      3) MODELO        (parámetros del YAML)
      4) ENTRENAMIENTO (batch, epochs, lr, etc.)
      5) RUTAS         (logs, checkpoints, outputs)
    """

    # ───────────────────── 1) EXPERIMENTO ─────────────────────
    exp = cfg["experiment"]
    exp_info = [
        ["Nombre",        exp.get("name", "-")],
        ["Descripción",   exp.get("description", "-")],
        ["Módulo modelo", exp.get("model_module", "-")],
        ["Clase modelo",  exp.get("model_class", "-")],
        ["Output root",   exp.get("output_root", "-")],
        ["Output subdir", exp.get("output_subdir", "-")],
    ]
    print("\n=== EXPERIMENTO ===")
    print(tabulate(exp_info, headers=["Campo", "Valor"], tablefmt="fancy_grid"))

    # ───────────────────── 2) DATASET ────────────────────────
    ds = cfg["dataset"]
    if ds["source"] == "kaggle":
        src_label = "Kaggle"
        ruta  = ds["kaggle"].get("dataset_id", "-")
        extra = ds["kaggle"].get("download_dir", "-")
    else:                                # source == 'local'
        src_label = "Local"
        ruta  = ds.get("local_path", "-")
        extra = "-"

    ds_info = [
        ["Fuente",       src_label],
        ["Ruta / ID",    ruta],
        ["Dir descarga", extra],
        ["test_pct",      ds.get("test_pct", "-")],
        ["train_pct",    ds.get("train_pct", "-")],
        ["k_fold",   ds.get("k_fold", "-")],
        ["class_names",  ds.get("class_names", "-")],
    ]
    print("\n=== DATASET ===")
    print(tabulate(ds_info, headers=["Campo", "Valor"], tablefmt="fancy_grid"))

    # ───────────────────── 3) PARÁMETROS DE MODELO ────────────
    mp = cfg.get("model", {}).get("params", {})
    model_table = [[k, mp[k]] for k in sorted(mp)]
    print("\n=== PARÁMETROS DE MODELO ===")
    print(tabulate(model_table, headers=["Parámetro", "Valor"], tablefmt="fancy_grid"))

    # ───────────────────── 4) ENTRENAMIENTO ───────────────────
    tr = cfg.get("training", {})
    train_table = [[k, tr[k]] for k in sorted(tr)]
    print("\n=== HÍPERPARÁMETROS DE ENTRENAMIENTO ===")
    print(tabulate(train_table, headers=["Parámetro", "Valor"], tablefmt="fancy_grid"))

    # ───────────────────── 5) RUTAS & OUTPUTS ─────────────────
    paths = {
        "logs_dir":        cfg.get("paths", {}).get("logs_dir", "-"),
        "checkpoints_dir": cfg.get("paths", {}).get("checkpoints_dir", "-"),
        "output_root":     exp.get("output_root", "-"),
        "output_subdir":   exp.get("output_subdir", "-"),
    }
    paths_table = [[k, paths[k]] for k in sorted(paths)]
    print("\n=== RUTAS & OUTPUTS ===")
    print(tabulate(paths_table, headers=["Directorio", "Ruta"], tablefmt="fancy_grid"))
