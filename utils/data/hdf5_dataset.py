"""
Carga un .hdf5 (local o descargado desde Kaggle) y genera:
  •  Atributos X, Y, Z en NumPy.
  •  Índices de train/val por porcentaje o K‑Fold.
  •  Métodos helper para obtener (X_split, Y_split) o tf.data.Dataset.

No depende de PyTorch.
"""

from __future__ import annotations
import os
from pathlib import Path
import h5py
import numpy as np
from sklearn.model_selection import KFold

try:
    # solo necesario si usas source='kaggle'
    os.environ['KAGGLE_USERNAME'] = 'ilikepizzaanddrones'
    os.environ['KAGGLE_KEY']      = 'b7d0370fced8eb934d226172fff8221f'
    from kaggle import KaggleApi
except ModuleNotFoundError:
    KaggleApi = None

class HDF5Dataset:
    def __init__(
        self,
        *,
        # --- local
        file_path: str | Path | None = None,
        # --- Kaggle
        kaggle_dataset_id: str | None = None,
        local_download_dir: str | Path = "datasets/raw",
        # --- split
        split: str = "train",          # "train" | "val"
        train_pct: float = 0.8,
        k_folds: int | None = None,
        fold_index: int = 0,
        seed: int = 42,
        # --- keys dentro del HDF5
        keys: dict | None = None,
    ):
        # ─────────────────── 0) Descargar de Kaggle (opcional) ───────────────────
        if kaggle_dataset_id:
            if KaggleApi is None:
                raise ImportError("pip install kaggle  (librería faltante)")
            api = KaggleApi(); api.authenticate()

            local_download_dir = Path(local_download_dir)
            local_download_dir.mkdir(parents=True, exist_ok=True)

            print(f"⬇️  Descargando «{kaggle_dataset_id}» …")
            api.dataset_download_files(
                kaggle_dataset_id,
                path=str(local_download_dir),
                unzip=True,
                quiet=False,
            )
            h5_files = sorted(local_download_dir.rglob("*.hdf5"))
            if not h5_files:
                raise FileNotFoundError("No se encontró ningún .hdf5 en el zip")
            file_path = h5_files[0]
            print(f"✅ Usando archivo HDF5: {file_path}")

        if file_path is None or not Path(file_path).is_file():
            raise FileNotFoundError(f"HDF5 inexistente: {file_path}")

        # ─────────────────── 1) Leer en memoria ───────────────────
        self.keys = keys or {"X": "X", "Y": "Y", "Z": "Z"}
        with h5py.File(file_path, "r") as f:
            X = f[self.keys["X"]][:]
            Y = f[self.keys["Y"]][:]
            Z = f[self.keys["Z"]][:]
            
            # -------- Effects --------------------------------------------------
            if "Effects" in f:
                grp = f["Effects"]
                dtype = [(name, grp[name].dtype) for name in grp.keys()]
                eff = np.empty(len(X), dtype=dtype)
                for name in grp.keys():
                    eff[name] = grp[name][:]
                self.Effects = eff            # structured array
            else:
                self.Effects = None           # << siempre creado
        
        self.X = X
        self.Y = Y
        self.Z = Z
        
        # ─────────────────── 2) Índices de split ───────────────────
        rng = np.random.RandomState(seed)
        indices = np.arange(len(self.X))

        if k_folds and k_folds > 1:
            kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
            train_idx, val_idx = list(kf.split(indices))[fold_index]
            self.train_idx, self.val_idx = train_idx, val_idx
        else:
            rng.shuffle(indices)
            cut = int(len(indices) * train_pct)
            self.train_idx, self.val_idx = indices[:cut], indices[cut:]

        self.split = split.lower()

    # ─────────────────── helpers ───────────────────
    def get_arrays(self, split: str | None = None):
        """Devuelve (X_split, Y_split)"""
        split = (split or self.split).lower()
        idx   = self.train_idx if split == "train" else self.val_idx
        return self.X[idx], self.Y[idx]

    def to_tf_dataset(
            self,
            split: str | None = None,
            batch_size: int = 32,
            shuffle: bool = True,
            buffer_size: int | None = None,
            prefetch: bool = True,
            include_index: bool | None = None,
        ):
        """
        Devuelve un tf.data.Dataset listo para Keras.
    
        Args
        ----
        split          : "train" | "val" | None (usa self.split por defecto)
        batch_size     : tamaño de lote para .batch()
        shuffle        : baraja el dataset (solo si True)
        buffer_size    : tamaño del buffer para .shuffle()
        prefetch       : aplica .prefetch(AUTOTUNE) si True
        include_index  : • True  → añade el índice original como 3er tensor
                         • False → devuelve solo (X, y)
                         • None  → incluye el índice SOLO si split == "val"
        """
        import tensorflow as tf
    
        split = (split or self.split).lower()
        X_split, Y_split = self.get_arrays(split)
        idx_split = (
            self.train_idx if split == "train" else self.val_idx
        )  # índices absolutos
    

        if include_index:
            ds = tf.data.Dataset.from_tensor_slices((X_split, Y_split, idx_split))
        else:
            ds = tf.data.Dataset.from_tensor_slices((X_split, Y_split))
    
        # Barajado, batching, prefetch
        if shuffle:
            ds = ds.shuffle(buffer_size or len(X_split), seed=123, reshuffle_each_iteration=True)
        ds = ds.batch(batch_size)
        if prefetch:
            ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    # ------------------------------------------------------------------ #
    def get_effects(
        self,
        split: str | None = None,
        fields: list[str] | None = None,
    ):
        """
        Devuelve un *structured array* con los efectos (grupo `Effects`)
        alineados al split solicitado.

        Parameters
        ----------
        split   : {"train", "val", None}
                  None → usa el split con el que se creó la instancia.
        fields  : lista opcional de nombres de columna para filtrar,
                  p.ej. ["snr_db", "num_taps"].  Si se omite devuelve
                  todas las columnas.

        Returns
        -------
        np.ndarray  (structured)
            Efectos correspondientes al split, con los dtypes originales.

        Raises
        ------
        ValueError
            Si el HDF5 no contiene grupo 'Effects'.
        """
        if self.Effects is None:
            raise ValueError("Este archivo HDF5 no contiene grupo 'Effects'.")

        split = (split or self.split).lower()
        idx = self.train_idx if split == "train" else self.val_idx

        eff = self.Effects[idx]          # vista alineada
        if fields is not None:
            eff = eff[fields].copy()     # sub‑vista con columnas pedidas
        return eff
