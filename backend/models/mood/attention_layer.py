import tensorflow as tf
from tensorflow.keras.layers import Layer

class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='att_weight', shape=(input_shape[-1], input_shape[-1]), initializer='uniform', trainable=True)
        self.b = self.add_weight(name='att_bias', shape=(input_shape[-1],), initializer='uniform', trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        x = tf.tensordot(inputs, self.W, axes=1)
        x += self.b
        x = tf.nn.tanh(x)
        att_weights = tf.nn.softmax(x, axis=1)
        return att_weights * inputs

    def compute_output_shape(self, input_shape):
        return input_shape
