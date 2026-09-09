import tensorflow as tf
from tensorflow import keras

@keras.saving.register_keras_serializable(package='SecureBlockNet')
class TemporalAttention(keras.layers.Layer):
    def __init__(self, attention_units=64, return_attention=False, **kwargs):
        super().__init__(**kwargs)
        self.attention_units = attention_units
        self.return_attention = return_attention
        self.proj = keras.layers.Dense(attention_units, activation='tanh')
        self.score = keras.layers.Dense(1, use_bias=False)

    def call(self, inputs, mask=None):
        e = self.score(self.proj(inputs))
        if mask is not None:
            mask = tf.cast(mask[:, :, None], e.dtype)
            e = e + (1.0 - mask) * tf.constant(-1e9, dtype=e.dtype)
        alpha = tf.nn.softmax(e, axis=1)
        context = tf.reduce_sum(alpha * inputs, axis=1)
        return (context, tf.squeeze(alpha, axis=-1)) if self.return_attention else context

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'attention_units': self.attention_units, 'return_attention': self.return_attention})
        return cfg
