from typing import List, Generator

import numpy as np
import pandas
import pandas as pd
import tensorflow as tf

# Based on a workaround found from StackOverflow:
# https://stackoverflow.com/questions/51680818/keras-custom-loss-as-a-function-of-multiple-outputs


class PackedTensor(tf.experimental.BatchableExtensionType):
    """Extension of a tensor that contains 3 sub-tensors of different shapes.
    Used to return 3 different items from the model, so that a normal loss function can be used as part of the model

    In this project, the 3 types are just
    1) Output tokens
    2) Attention tensor
    3) Attention mask
    """
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

    def get_values(self) -> List[np.ndarray]:
        o0 = self.output_0.numpy()
        o1 = self.output_1.numpy()
        o2 = self.output_2.numpy()

        return [o0, o1, o2]


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


def predict_from_ear_model(x: Generator, steps: int, model: tf.keras.Model) -> pandas.DataFrame:
    pred_out = []
    attn_out = []
    y_in = []
    tkn_in = []
    mask_in = []
    for _ in range(steps):
        x_batch, y_batch = next(x)
        token_batch, mask_batch = x_batch
        out = model(x_batch)
        values = out.get_values()
        pred_out.append(values[0])
        attn_out.append(values[1])
        y_in.append(y_batch)
        tkn_in.append(token_batch)
        mask_in.append(mask_batch)

    pred_out = np.concatenate(pred_out).tolist()
    attn_out = np.concatenate(attn_out).tolist()
    tkn_out = np.concatenate(tkn_in).tolist()
    mask_out = np.concatenate(mask_in).tolist()
    y_ret = np.concatenate(y_in).tolist()

    out_df = pd.DataFrame({
        "tokens": tkn_out,
        "mask": mask_out,
        "labels": y_ret,
        "prediction": pred_out,
        "attention": attn_out,
    })

    return out_df
