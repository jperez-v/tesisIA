import tensorflow as tf
from models.base_model import BaseTFModel
from tensorflow.keras import layers

class NN(BaseTFModel):
    def build_model(self):
        mp = self.model_params
        seq_len   = int(mp.get('seq_len', 4096))
        n_classes = int(mp.get('n_classes', 7))

        inp = layers.Input(shape=(seq_len, 2), name='IQ_input')

        # 1) Inception-1D
        def inception_block(x, filters):
            c1 = layers.Conv1D(filters, 3, padding='same', activation='relu')(x)
            c2 = layers.Conv1D(filters, 5, padding='same', activation='relu')(x)
            c3 = layers.Conv1D(filters, 7, padding='same', activation='relu')(x)
            return layers.Concatenate()([c1, c2, c3])

        x = inception_block(inp, mp.get('inception_filters', 64))
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(mp.get('pool_size', 4))(x)

        # 2) Bloques residuales dilatados
        def res_dilated_block(x, filters, rate):
            y = layers.Conv1D(filters, 3, padding='same',
                              dilation_rate=rate, activation='relu')(x)
            y = layers.BatchNormalization()(y)
            y = layers.Conv1D(filters, 3, padding='same',
                              dilation_rate=rate)(y)
            y = layers.BatchNormalization()(y)
            if x.shape[-1] != filters:
                x = layers.Conv1D(filters, 1, padding='same')(x)
            return layers.Activation('relu')(layers.Add()([x, y]))

        for f, d in zip(mp.get('res_filters', [128, 256]),
                        mp.get('dilation_rates', [1, 2])):
            x = res_dilated_block(x, f, d)
            x = layers.MaxPooling1D(mp.get('pool_size', 4))(x)

        # 3) Transformer Encoder Block
        attn = layers.MultiHeadAttention(
            num_heads=mp.get('num_heads', 4),
            key_dim=mp.get('key_dim', 64)
        )(x, x)
        attn = layers.Dropout(mp.get('attn_dropout', 0.1))(attn)
        x = layers.LayerNormalization(epsilon=1e-6)(x + attn)

        ffn = layers.Dense(mp.get('ffn_units', 256), activation='relu')(x)
        ffn = layers.Dense(x.shape[-1])(ffn)
        x   = layers.LayerNormalization(epsilon=1e-6)(x + ffn)

        # 4) Pooling y Clasificador
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(mp.get('dense_units', 256), activation='relu')(x)
        x = layers.Dropout(mp.get('dropout', 0.5))(x)
        out = layers.Dense(n_classes, activation='softmax')(x)

        return tf.keras.Model(inputs=inp, outputs=out)
