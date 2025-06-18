import tensorflow as tf
from tensorflow.keras import layers
from models.base_model import BaseTFModel

class AttentionPool(layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.W = layers.Dense(units, activation='tanh')
        self.v = layers.Dense(1)

    def call(self, features):
        score = self.v(self.W(features))         # [B, T, 1]
        weights = tf.nn.softmax(score, axis=1)   # atención temporal
        return tf.reduce_sum(features * weights, axis=1)

class NN(BaseTFModel):
    def build_model(self):
        mp   = self.model_params             # ← dict model.params del YAML
        P    = lambda k, d=None: mp.get(k, d)

        seq_len   = int(P('seq_len', 4096))
        n_class   = int(P('n_classes', 6))

        # --- front-end CNN parametrizable ----------------------------
        f, k, s   = P('conv_filters'), P('conv_kernels'), P('conv_strides')
        pool_s    = int(P('pool_stride', 2))

        inp = layers.Input(shape=(seq_len, 2), name='IQ')
        x   = inp
        for i, (fi, ki, si) in enumerate(zip(f, k, s)):
            x = layers.Conv1D(fi, ki, strides=si, padding='same',
                              activation='relu', name=f'conv{i+1}')(x)
            x = layers.BatchNormalization()(x)
            if i == 1:                        # max-pool tras el 2.º bloque
                x = layers.MaxPooling1D(pool_s)(x)

        # --- Bi-LSTM (1 o N capas) -----------------------------------
        lstm_units = P('bi_lstm_units', 256)
        if isinstance(lstm_units, (list, tuple)):
            for j, u in enumerate(lstm_units):
                rs = (j < len(lstm_units) - 1)  # return_sequences=True salvo última
                x  = layers.Bidirectional(
                        layers.LSTM(u, return_sequences=rs,
                                    dropout=0.0, recurrent_dropout=0.25),
                        name=f'bilstm{j+1}')(x)
        else:
            x = layers.Bidirectional(
                    layers.LSTM(lstm_units, return_sequences=True))(x)

        # --- Atención + clasificador ---------------------------------
        x   = AttentionPool(int(P('atten_units', 128)))(x)
        x   = layers.Dense(int(P('dense_units', 128)), activation=P('act_dense', 'relu'))(x)
        x   = layers.Dropout(P('dropout', 0.35))(x)
        out = layers.Dense(n_class, activation='softmax')(x)

        return tf.keras.Model(inp, out)
