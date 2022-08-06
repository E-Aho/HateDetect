import numpy as np
import tensorflow as tf


class PackedTensor(tf.experimental.BatchableExtensionType):
    __name__ = 'extension_type_colab.PackedTensor'

    output_0: tf.Tensor
    output_1: tf.Tensor
    output_2: tf.Tensor

    # shape and dtype hold no meaning in this context, so we use a dummy
    # to stop Keras from complaining

    shape = property(lambda self: self.output_0.shape)
    dtype = property(lambda self: self.output_0.dtype)

    class Spec:

        def __init__(self, shape, dtype=tf.float32):
            self.output_0 = tf.TensorSpec(shape, dtype)
            self.output_1 = tf.TensorSpec(shape, dtype)
            self.output_2 = tf.TensorSpec(shape, dtype)

        # shape and dtype hold no meaning in this context, so we use a dummy
        # to stop Keras from complaining
        shape: tf.TensorShape = tf.constant(1.).shape
        dtype: tf.DType = tf.constant(1.).dtype

    def numpy(self):
        return np.column_stack([
            self.output_0.numpy(),
            self.output_1.numpy(),
            self.output_2.numpy(),
        ])


class PackingLayer(tf.keras.layers.Layer):

    def call(self, inputs, training=None):
        first_out, second_out, third_out = inputs
        packed_output = PackedTensor(first_out, second_out, third_out)
        return packed_output


@tf.experimental.dispatch_for_api(tf.shape)
def packed_shape(input: PackedTensor, out_type=tf.int32, name=None):
    return tf.shape(input.col_ids)


@tf.experimental.dispatch_for_api(tf.cast)
def packed_cast(x: PackedTensor, dtype: str, name=None):
    return x