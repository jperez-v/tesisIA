"""
Carga la configuración de un experimento, convierte notebooks a .py si
es necesario, importa dinámicamente el modelo y prepara los datos
(locales o descargados de Kaggle) en formato NumPy o tf.data.Dataset.

"""

import yaml                          # Lectura y mezcla de archivos YAML
import sys, subprocess               # Conversión IPython Notebook → .py
from pathlib import Path             # Manejo robusto de rutas
from importlib import import_module  # Import dinámico del modelo
from inspect import signature        # Validación de parámetros del modelo

from utils.data.hdf5_dataset import HDF5Dataset  # Clase de carga de datos

import yaml
from pathlib import Path
import sys
import subprocess
import numpy as np
import h5py
from sklearn.model_selection import KFold, train_test_split
from importlib import import_module
from inspect import signature, Parameter, _empty
from utils.data.hdf5_dataset import HDF5Dataset



# Rutas base (ajusta a tu estructura en Google Drive si cambia)
CONFIG_ROOT = Path("/content/drive/MyDrive/structure/configs")
MODELS_ROOT = Path("/content/drive/MyDrive/structure/models")
DATA_ROOT   = Path("/content/drive/MyDrive/structure/datasets")

def load_config(exp_name:str):
    exp_path = CONFIG_ROOT / "experiments" / f"{exp_name}.yaml"
    exp_cfg  = yaml.safe_load(exp_path.read_text())

    if "_base_" in exp_cfg:                                # herencia opcional
        base_cfg = yaml.safe_load((CONFIG_ROOT / exp_cfg["_base_"]).read_text())
        cfg = {**base_cfg, **exp_cfg}                      # exp > default
    else:
        cfg = exp_cfg
    return cfg

def load_experiment(
    exp_name: str,
    repeat_index: int,
    fold_index: int | None = None,
    ):
    """
    Devuelve:
        cfg          → dict  (configuración combinada)
        ModelClass   → type  (sub‑clase de tu BaseTFModel)
        model_params → dict  (params filtrados para __init__)
        full_dataset 
        train_data   → (X,Y) tf.data.Dataset
        val_data     → idem
        val_indices
    """

    # ─────────────────── 1) Leer YAML ──────────────────────────
    cfg = load_config(exp_name=exp_name)

    # ─────────────────── 2) Notebook (.ipynb) → Python (.py) ──────────────
    model_module = cfg["experiment"]["model_module"]
    ipynb_path   = MODELS_ROOT / f"{model_module}.ipynb"
    py_path      = MODELS_ROOT / f"{model_module}.py"

    if ipynb_path.exists() and (
        not py_path.exists() or ipynb_path.stat().st_mtime > py_path.stat().st_mtime
    ):
        subprocess.run(
            [sys.executable, "-m", "nbconvert", "--to", "python", str(ipynb_path)],
            check=True,
        )

    # ─────────────────── 3) Import dinámico del modelo ────────────────────
    sys.path.append(str(MODELS_ROOT))
    module      = import_module(model_module)
    ModelClass  = getattr(module, cfg["experiment"]["model_class"])

    # ─────────────────── 4) Filtrar parámetros válidos ────────────────────
    sig          = signature(ModelClass.__init__)
    raw_params   = cfg["model"]["params"]
    model_params = {k: v for k, v in raw_params.items()}

    # ─────────────────── 5) Preparar Dataset (local o Kaggle) ─────────────
    ds_cfg = cfg["dataset"]
    exp_cfg = cfg["experiment"]
    common_ds_kwargs = dict(
        test_pct   = ds_cfg["test_pct"],
        train_pct  = ds_cfg["train_pct"],
        repeat_index = repeat_index if exp_cfg["repeats"] else None ,
        k_folds    = ds_cfg["k_folds"] or None,
        fold_index = fold_index if ds_cfg["k_folds"] else None ,
        seed       = cfg["training"]["seed"],
        keys       = ds_cfg["keys"],
    )
    
    # // Modificar subdirectorio de acuerdo a número actual de repetición  \\
    cfg["experiment"]["output_subdir"] = cfg["experiment"]["output_subdir"] + "/" + f"rep_{repeat_index}"
    
    # // Modificar subdirectorio si k-fold está configurado \\
    k = cfg["dataset"].get("k_folds")
    if k is not None and k > 1:
        cfg["experiment"]["output_subdir"] = cfg["experiment"]["output_subdir"] + "/" + f"fold_{fold_index}"


    if ds_cfg["source"] == "kaggle":
        full_ds = HDF5Dataset(
            kaggle_dataset_id  = ds_cfg["kaggle"]["dataset_id"],
            local_download_dir = ds_cfg["kaggle"]["download_dir"],
            **common_ds_kwargs,
        )

    else:  # source == 'local'
        file_path = DATA_ROOT.parent / ds_cfg["local_path"]   # datasets/raw/…
        full_ds   = HDF5Dataset(file_path=str(file_path), **common_ds_kwargs)

    # tf.data.Dataset
    bs = cfg["training"].get("batch_size", 32)
    train_tf = full_ds.to_tf_dataset("train", batch_size=bs, shuffle=True)
    val_tf   = full_ds.to_tf_dataset("val",   batch_size=bs, shuffle=False)

    test_tf_idx = (
        None if full_ds.test_idx is None
        else full_ds.to_tf_dataset("test", batch_size=bs,
        shuffle=False, include_index=True)
        )

    train_data, val_data, test_data_idx = train_tf, val_tf, test_tf_idx

    # ─────────────────── 6) Return ────────────────────────────────────────
    return cfg, ModelClass, model_params, full_ds, train_data, val_data, test_data_idx
