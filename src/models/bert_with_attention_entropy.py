from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.losses import Loss
from transformers import RobertaConfig

from datasets.hatexplain import hatexplain_dataset_path
from src.models import BASE_MODEL_MODEL_PATH
from src.models.abstract_base_model import AbstractModel
from src.models.data_utils import HatexplainDataset

learning_rate = 100
n_head = 12
max_token_len = 113

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


# these two functions have no meaning, but need dummy implementations
# to stop Keras from complaining
@tf.experimental.dispatch_for_api(tf.shape)
def packed_shape(input: PackedTensor, out_type=tf.int32, name=None):
    return tf.shape(input.col_ids)


@tf.experimental.dispatch_for_api(tf.cast)
def packed_cast(x: PackedTensor, dtype: str, name=None):
    return x


@tf.function
def entropy_of_tensor(t: tf.Tensor):
    t = tf.boolean_mask(t, tf.math.greater(t, tf.zeros_like(t))) # filter out zero values
    tlogt = t * tf.math.log(t)
    return - tf.math.reduce_sum(tlogt, axis=0)
@tf.function
def calculate_masked_entropy_for_subbatch(a_3d: tf.Tensor, one_d_attention_mask: tf.Tensor):
    bool_mask = tf.cast(one_d_attention_mask, tf.bool)

    masked_attention = tf.boolean_mask(
        tf.boolean_mask(a_3d, mask=bool_mask, axis=2),
    mask=bool_mask, axis=1)

    return tf.reduce_mean(
        tf.map_fn(entropy_of_tensor, masked_attention)
    )

@tf.function
def calculate_masked_entropy(attentions: tf.Tensor, attention_mask: tf.Tensor) -> tf.Tensor:

    return tf.reduce_mean(tf.map_fn(
            lambda batch: calculate_masked_entropy_for_subbatch(batch[0], batch[1]),
            [attentions, attention_mask], fn_output_signature=tf.float32
    ))


class AttentionEntropyLoss(Loss):
    def __init__(self, chi: float = 0.1):
        super().__init__()
        self.chi = chi
        self.loss_fn = tf.keras.losses.CategoricalCrossentropy()


    def call(self, y_true, y_pred):
        predicted_labels, attentions, attention_mask = y_pred.output_0, y_pred.output_1, y_pred.output_2

        cat_loss = self.loss_fn(y_true, predicted_labels)
        entropy_loss = calculate_masked_entropy(attentions, attention_mask)

        _o = cat_loss + tf.scalar_mul(self.chi, entropy_loss)
        return _o



class PackingLayer(tf.keras.layers.Layer):
    def call(self, inputs, training=None):
        first_out, second_out, third_out = inputs
        packed_output = PackedTensor(first_out, second_out, third_out)
        return packed_output


# def attention_entropy_loss(self, input_ids, output_ids, attention, chi: float):
#     n_layers, n_heads = self.bert.config.num_hidden_layers, self.bert.config.num_attention_heads
#     attention_entropy = tf.zeros(n_layers, n_heads)

class BertModelWithAttentionEntropy(AbstractModel):
    def __init__(self, input_shape: tuple, output_shape: tuple):
        model_pretrained_name = "roberta-base"
        save_path = BASE_MODEL_MODEL_PATH
        self.dropout_rate = 0.2
        model_config = RobertaConfig.from_pretrained(
            model_pretrained_name,
            output_hidden_states=True,
            output_attentions=True,
        )

        super().__init__(
            input_shape=input_shape,
            output_shape=output_shape,
            model_name=model_pretrained_name,
            model_config=model_config,
            save_path=save_path,
        )

        self.loss = AttentionEntropyLoss(chi=0.2)
        self.metrics = []
        # logs = Path("logs") / datetime.now().strftime("%Y%m%d-%H%M%S")
        # # self.callbacks.append(
        #     tf.keras.callbacks.TensorBoard(log_dir=logs,
        #                                   histogram_freq=1, update_freq="batch", profile_batch=1)
        #
        # )
        self.optimizer = "adam"

    def get_opt(self, learning_rate: float, ):
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def get_model(self, input_shape: tuple, output_shape: tuple, ) -> tf.keras.Model:
        input_ids = tf.keras.layers.Input(shape=input_shape, dtype=tf.int32, name="input_ids")

        input_attention_mask = tf.keras.layers.Input(
            shape=input_shape,
            dtype=tf.int32,
            name="attention_mask"
        )

        bert_model = self.bert([input_ids, input_attention_mask])
        last_hidden_state = bert_model.last_hidden_state
        attentions = bert_model.attentions
        sum_attentions = tf.add_n(attentions)
        head_averaged_attentions = tf.scalar_mul(1/12, sum_attentions)
        cls_token_out = last_hidden_state[:, 0, :]  # the CLS token is used to represent sentence level classification
        dropout = tf.keras.layers.Dropout(self.dropout_rate)(cls_token_out)

        output = tf.keras.layers.Dense(output_shape, activation="softmax")(dropout)  # 3 for non-split training
        packed_layer = PackingLayer()(
            [output, head_averaged_attentions, input_attention_mask]
        )

        model = tf.keras.models.Model(inputs=[input_ids, input_attention_mask], outputs=packed_layer)
        return model

    def fine_tune_and_train_mdl(self, dataset: HatexplainDataset, learning_rate: float, n_epochs: int, ):
        model = self.model
        num_epochs = n_epochs

        model.compile(
            optimizer=self.optimizer,
            metrics=self.metrics,
            loss=AttentionEntropyLoss(chi=0.2),
            run_eagerly=False,
        )

        train_data = dataset.get_train_generator()
        test_data = dataset.get_test_generator()

        model.fit(
            train_data,
            validation_data=test_data,
            batch_size=dataset.batch_size,
            epochs=num_epochs,
            steps_per_epoch=dataset.train_batches,
            validation_steps=dataset.test_batches,
            use_multiprocessing=False,
            callbacks=self.callbacks,
            max_queue_size=20,
            workers=1,
        )

        return model


if __name__ == "__main__":
    hatexplain_dataset = HatexplainDataset(
        pd.read_parquet(hatexplain_dataset_path), p_test=0.2
    )

    base_mdl = BertModelWithAttentionEntropy(
        input_shape=hatexplain_dataset.model_input_size,
        output_shape=hatexplain_dataset.model_output_size,
    )

    trained_mdl = base_mdl.fine_tune_and_train_mdl(
        dataset=hatexplain_dataset,
        learning_rate=learning_rate,
        n_epochs=1,
    )
