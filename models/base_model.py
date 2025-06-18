import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau
from pathlib import Path
from tensorflow.keras.losses import CategoricalCrossentropy
import datetime

class EpochLoggerCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_freq=5, timestamp_format='%d-%m-%Y %H:%M:%S'):
        super().__init__()
        self.log_freq = log_freq
        self.timestamp_format = timestamp_format
      
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.log_freq == 0:
            ts = datetime.datetime.now().strftime(self.timestamp_format)
            metrics = " - ".join([f"{k}: {v:.6f}" for k, v in logs.items()])
            print(f"[{ts}] Epoch {epoch + 1}: {metrics}")

class BaseTFModel:
    """
    Clase base para modelos Keras.
    Recibe `config` y `model_params` del YAML, gestiona callbacks,
    checkpoints y retoma entrenamiento automáticamente.
    """

    def __init__(self, config: dict, **model_params):
        """
        Args:
            config (dict): configuración completa cargada desde YAML
            model_params: kwargs propios de la arquitectura (input_size, dropout, etc.)
        """
        self.cfg          = config
        self.model_params = model_params

        # Rutas en Drive
        self.BASE_DIR = Path('/content/drive/MyDrive/structure')
        exp = self.cfg['experiment']
        self.out_dir = self.BASE_DIR / exp['output_root'] / exp['output_subdir']
        (self.out_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
        (self.out_dir / 'logs').mkdir(parents=True, exist_ok=True)

        # Construir y compilar el modelo
        self.model = self.build_model()
        print(" "*5, "✔️ Modelo Keras inicializado")

        tr = self.cfg['training']
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=float(tr['learning_rate'])
        )
        self.model.compile(
            optimizer=optimizer,
            loss=CategoricalCrossentropy(),
            metrics=['accuracy']
        )

    def build_model(self) -> tf.keras.Model:
        """
        Subclase debe usar `self.model_params` y `self.cfg` para construir
        la red y devolver un tf.keras.Model.
        """
        raise NotImplementedError("Implementa build_model() en tu subclase")

    def get_callbacks(self):
        tr = self.cfg['training']
        cbks = [
            # 1) TensorBoard
            TensorBoard(log_dir=str(self.out_dir / 'logs'), update_freq='epoch'),

            # 2) EarlyStopping
            EarlyStopping(monitor='val_accuracy',
                          patience=int(tr.get('patience', 5)),
                          restore_best_weights=True),

            # 3) ModelCheckpoint en formato .keras
            ModelCheckpoint(
                filepath=str(self.out_dir / 'checkpoints' / 'epoch_{epoch:02d}.keras'),
                monitor='val_accuracy',
                save_best_only=bool(tr.get('save_best_only', True)),
                save_weights_only=False,
                verbose=0
            ),
            
            # 4) Reduce learning rate on plateau
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=6,
                cooldown=2,
                min_delta=1e-4,
                min_lr=2e-5,
                verbose=0
            ),
            
            EpochLoggerCallback(
                log_freq=5
            ),
            
        ]
        return cbks

    def fit(self, train_data, val_data):
        """
        - Desempaqueta train_data y val_data
        - Carga último checkpoint .keras
        - Lanza `model.fit(..., initial_epoch=...)`
        """
        # 1) Desempaquetar datos
        if isinstance(train_data, tuple):
            X_train, y_train = train_data
        else:
            X_train, y_train = train_data, None

        if isinstance(val_data, tuple):
            X_val, y_val = val_data
            val_arg = (X_val, y_val)
        else:
            val_arg = val_data

        # 2) Buscar último checkpoint (.keras)
        ckpt_dir = self.out_dir / 'checkpoints'
        ckpt_files = sorted(ckpt_dir.glob('epoch_*.keras'))
        initial_epoch = 0
        if ckpt_files:
            last_ckpt = ckpt_files[-1]
            print(f"🔄 Cargando checkpoint previo: {last_ckpt.name}")
            self.load_weights(str(last_ckpt))
            # 'epoch_XX.keras' → XX
            initial_epoch = int(last_ckpt.stem.split('_')[1])

        # 3) Preparar parámetros
        tr = self.cfg['training']
        epochs     = int(tr['epochs'])
        batch_size = int(tr['batch_size'])

        # 4) Entrenar / retomar
        history = self.model.fit(
            X_train, y_train,
            validation_data=val_arg,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=self.get_callbacks(),
            initial_epoch=initial_epoch,
            verbose=0
        )
        return history

    def load_weights(self, path: str):
        """Carga pesos guardados (.keras)."""
        self.model.load_weights(path)
        print(" "*7, f"✔️ Pesos cargados desde {path}")

    def cleanup_old_checkpoints(self):
        """Elimina todos los checkpoints excepto el más reciente."""
        ckpt_dir = self.out_dir / 'checkpoints'
        ckpt_files = sorted(ckpt_dir.glob('epoch_*.keras'))
          
        if len(ckpt_files) > 1:
            # Mantener solo el último, eliminar el resto
            for old_ckpt in ckpt_files[:-1]:
                old_ckpt.unlink()
                print(" "*10, f"🗑️ Eliminado checkpoint: {old_ckpt.name}")
