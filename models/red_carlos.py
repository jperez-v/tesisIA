# models/amc_model.py

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from models.base_model import BaseTFModel

class NN(BaseTFModel):

    def build_model(self):
        # --- Squeeze-and-Excitation block ---
        def squeeze_excite_block(input_tensor, ratio=16):
            filters = input_tensor.shape[-1]
            se = layers.GlobalAveragePooling1D()(input_tensor)
            se = layers.Dense(filters // ratio, activation='relu')(se)
            se = layers.Dense(filters, activation='sigmoid')(se)
            se = layers.Reshape((1, filters))(se)
            return layers.Multiply()([input_tensor, se])

        # --- Inception block con residual y SE ---
        def inception_res_block(input_tensor, filters, l2_reg=1e-4):
            tower1 = layers.Conv1D(filters, 4, padding='same', activation='relu',
                                   kernel_regularizer=regularizers.l2(l2_reg))(input_tensor)
            tower2 = layers.Conv1D(filters, 6, padding='same', activation='relu',
                                   kernel_regularizer=regularizers.l2(l2_reg))(input_tensor)
            tower3 = layers.Conv1D(filters, 8, padding='same', activation='relu',
                                   kernel_regularizer=regularizers.l2(l2_reg))(input_tensor)

            concat = layers.Concatenate()([tower1, tower2, tower3])
            concat = layers.BatchNormalization()(concat)
            se = squeeze_excite_block(concat)

            # Ajuste de dimensiones si es necesario para residual
            if input_tensor.shape[-1] != se.shape[-1]:
                input_tensor = layers.Conv1D(se.shape[-1], 1, padding='same')(input_tensor)

            out = layers.Add()([input_tensor, se])
            out = layers.Activation('relu')(out)
            return out

        # --- Multi-head Attention block ---
        
        def attention_block(input_tensor, num_heads=2, key_dim = 16):
            norm = layers.LayerNormalization()(input_tensor)
            attn_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(norm, norm)
            attn_output = layers.Dropout(0.1)(attn_output)
            out = layers.Add()([input_tensor, attn_output])
            return out

        # 1) Parámetros del modelo
        mp = self.model_params
        seq_len = int(mp.get('seq_len', 4096))
        n_classes = int(mp.get('output_size', mp.get('n_classes', 7)))
        filtros = int(mp.get('filters', 32))
        num_heads = int(mp.get('num_heads', 2))
        key_dim = int(mp.get('key_dim', 16))
        regularizador = float(mp.get('regularizer', 2e-4))
        densa = int(mp.get('dense', 32))
        dropout = float(mp.get('dropout', 0.3))
        pooling = int(mp.get('pooling', 2))



        # 2) Entrada
        inp = layers.Input(shape=(seq_len, 2), name='IQ_input')

        # 3) Bloques iniciales
        x = layers.Conv1D(filtros, kernel_size=8, strides=2, padding='same', activation='relu')(inp)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(pool_size=pooling)(x)

        # 4) Bloques Inception + SE + Residual
        x = inception_res_block(x, filters=filtros, l2_reg = regularizador)
        x = inception_res_block(x, filters=filtros, l2_reg = regularizador)

        # 5) Bloque de Atención
        x = attention_block(x, num_heads=num_heads)

        # 6) Clasificación
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(densa, activation='relu', kernel_regularizer=regularizers.l2(regularizador))(x)
        x = layers.LayerNormalization()(x)
        x = layers.Dropout(dropout)(x)
        outputs = layers.Dense(n_classes, activation='softmax')(x)

        # 7) Modelo final
        model = models.Model(inputs=inp, outputs=outputs)
        return model
