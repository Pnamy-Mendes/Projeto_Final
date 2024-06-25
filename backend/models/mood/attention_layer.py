# attention_layer.py
import tensorflow as tf
from tensorflow.keras.layers import Layer

class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', shape=(input_shape[-1],), initializer='uniform', trainable=True)
        self.b = self.add_weight(name='attention_bias', shape=(input_shape[1],), initializer='uniform', trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        e = tf.nn.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        a = tf.nn.softmax(e)
        output = x * tf.expand_dims(a, -1)
        return tf.reduce_sum(output, axis=1)

def add_attention_layer(inputs):
    return AttentionLayer()(inputs)
