"""
Carga un .hdf5 (local o descargado desde Kaggle) y genera:
  •  Atributos X, Y, Z en NumPy.
  • Split independiente de *test* (stratificado)
  • Validación con *StratifiedKFold* sobre el bloque train+val
  •  Índices de train/val por porcentaje o K‑Fold.
  •  Métodos helper para obtener (X_split, Y_split) o tf.data.Dataset.

"""

from __future__ import annotations
import os
from pathlib import Path

import h5py
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold

try:
    # solo necesario si usas source='kaggle'
    os.environ['KAGGLE_USERNAME'] = 'ilikepizzaanddrones'
    os.environ['KAGGLE_KEY']      = 'b7d0370fced8eb934d226172fff8221f'
    from kaggle import KaggleApi
except ModuleNotFoundError:
    KaggleApi = None

class HDF5Dataset:
    """Carga en memoria un *.hdf5* con X/Y[/Z] y provee *splits* flexibles."""

    # ──────────────────────────────────────────────────────────────────
    def __init__(
        self,
        *,
        # --- local
        file_path: str | Path | None = None,
        # --- Kaggle
        kaggle_dataset_id: str | None = None,
        local_download_dir: str | Path = "datasets/raw",
        # --- split
        split: str = "train",  # "train" | "val" | "test"
        test_pct: float = 0.15,  # proporción para test (0 → sin test)
        train_pct: float = 0.8,  # proporción *dentro* de train+val
        k_folds: int | None = None,
        fold_index: int | None = None,
        # --- repetitions
        repeat_index: int,
        seed: int = 42,
        # --- keys dentro del HDF5
        keys: dict | None = None,
    ) -> None:
        self.split = split.lower()
        
        # ╭─────────────────── SEED ───────────────────╮
        self.seed = seed + repeat_index * 1000 + (fold_index or 0)
            
        # ╭─────────────────── 0) Descarga de Kaggle ───────────────────╮
        if kaggle_dataset_id:
            if KaggleApi is None:
                raise ImportError("pip install kaggle  (librería faltante)")
            
            # Verificar si ya existe el archivo HDF5
            local_download_dir = Path(local_download_dir)
            local_download_dir.mkdir(parents=True, exist_ok=True)
            
            h5_files = sorted(local_download_dir.rglob("*.hdf5"))
            
            if not h5_files:  # Solo descargar si no existe
                print(f"⬇️  Descargando «{kaggle_dataset_id}» …")
                api = KaggleApi(); api.authenticate()
                api.dataset_download_files(
                    kaggle_dataset_id,
                    path=str(local_download_dir),
                    unzip=True,
                    quiet=False,
                )
                
            h5_files = sorted(local_download_dir.rglob("*.hdf5"))
            if not h5_files:
                raise FileNotFoundError("No se encontró ningún .hdf5 en el zip")
                    
            if len(h5_files) > 1:
                raise ValidationError("Existe más de un dataset en el directorio. No es posible diferenciar correctamente el dataset a emplear.")

            file_path = h5_files[0]
    

        if file_path is None or not Path(file_path).is_file():
            raise FileNotFoundError(f"HDF5 inexistente: {file_path}")
        
        print(f"✅ Usando archivo HDF5: {file_path}")

        # ╭─────────────────── 1) Lectura a NumPy ───────────────────────╮
        self.keys = keys or {"X": "X", "Y": "Y", "Z": "Z"}
        with h5py.File(file_path, "r") as f:
            self.X = f[self.keys["X"]][:]
            self.Y = f[self.keys["Y"]][:]
            self.Z = f[self.keys["Z"]][:]
            
            # -------- Effects --------------------------------------------------
            if "Effects" in f:
                grp = f["Effects"]
                dtype = [(name, grp[name].dtype) for name in grp.keys()]
                eff = np.empty(len(self.X), dtype=dtype)
                for name in grp.keys():
                    eff[name] = grp[name][:]
                self.Effects = eff            # structured array
            else:
                self.Effects = None           # << siempre creado
        

        # ╭─────────────────── 2) Índices de split ──────────────────────╮
        rng = np.random.RandomState(self.seed)
        indices = np.arange(len(self.X))

        # 2.1) Split TEST (fijo, estratificado) ─────────────────────────
        if test_pct <= 0:
            raise ValueError(f"El porcentaje de testeo no puede ser menor o igual a cero. (test_pct={test_pct})")

        trainval_idx, test_idx = train_test_split(
            indices,
            test_size=test_pct,
            stratify=self.Y,
            random_state=self.seed,
        )
        self.test_idx = np.array(test_idx, dtype=np.int64)
            
        # 2.2) Dentro de trainval → CV o split porcentual ───────────────
        if k_folds is not None and k_folds > 1:
            if fold_index is None:
                raise ValueError("Debes indicar fold_index (0‑based) cuando k_folds>1")
            if not (0 <= fold_index < k_folds):
                raise ValueError(f"fold_index={fold_index} fuera de rango para k_folds={k_folds}")

            # ── Estratificación: aseguramos etiquetas en formato entero ──
            if self.Y.ndim > 1:                       # one‑hot → entero
                y_strat = np.argmax(self.Y, axis=-1)
            else:                                     # ya es vector 1‑D
                y_strat = self.Y

            skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=self.seed)
            splits = list(skf.split(trainval_idx, y_strat[trainval_idx]))
            train_rel, val_rel = splits[fold_index]
            self.train_idx = trainval_idx[train_rel]
            self.val_idx   = trainval_idx[val_rel]
        else:
            rng.shuffle(trainval_idx)
            cut = int(len(trainval_idx) * train_pct)
            self.train_idx = trainval_idx[:cut]
            self.val_idx   = trainval_idx[cut:]


    # ───────────────────── helpers ─────────────────────
    def _sel(self, idx: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
        if idx is None:
            raise ValueError("Este split no está disponible (quizá test_pct=0?)")
        return self.X[idx], self.Y[idx]

    def get_arrays(self, split: str | None = None):
        """Devuelve *(X_split, Y_split)* para «train», «val» o «test»."""
        split = (split or self.split).lower()
        if split == "train":
            return self._sel(self.train_idx)
        if split == "val":
            return self._sel(self.val_idx)
        if split == "test":
            return self._sel(self.test_idx)
        raise ValueError("split debe ser 'train', 'val' o 'test'")
        
    # ------------------------------------------------------------------
    def to_tf_dataset(
        self,
        split: str | None = None,
        batch_size: int = 32,
        shuffle: bool = True,
        include_index: bool = False,
        buffer_size: int | None = None,
        prefetch: bool = True,
    ):
        """Convierte cualquier split en un *tf.data.Dataset* listo para Keras."""
        import tensorflow as tf

        Xs, Ys = self.get_arrays(split)
        idx = (
            self.train_idx if split == "train" else
            self.val_idx   if split == "val"   else
            self.test_idx
        )

        if include_index:
            ds = tf.data.Dataset.from_tensor_slices((Xs, Ys, idx))
        else:
            ds = tf.data.Dataset.from_tensor_slices((Xs, Ys))

        if shuffle and split == "train":  # sólo barajamos entreno
            ds = ds.shuffle(buffer_size or len(Xs), seed=self.seed, reshuffle_each_iteration=True)
        ds = ds.batch(batch_size)
        if prefetch:
            ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    # ------------------------------------------------------------------ #
    def get_effects(self, split: str | None = None, *, fields: list[str] | None = None):
        """Devuelve un *structured array* de efectos alineado al split."""
        if self.Effects is None:
            raise ValueError("El archivo HDF5 no contiene grupo 'Effects'.")
        split = (split or self.split).lower()
        idx = (
            self.train_idx if split == "train" else
            self.val_idx   if split == "val"   else
            self.test_idx
        )
        eff = self.Effects[idx]  # vista alineada
        if fields is not None:
            eff = eff[fields].copy()
        return eff
