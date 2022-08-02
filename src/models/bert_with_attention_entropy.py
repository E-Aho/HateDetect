from typing import Union

import keras.metrics
import pandas as pd
import tensorflow as tf
from transformers import RobertaConfig

from datasets.hatexplain import hatexplain_dataset_path
from src.models import BASE_MODEL_MODEL_PATH
from src.models.abstract_base_model import AbstractModel
from src.models.attention_entropy_loss import AttentionEntropyLoss, CompatibleMetric
from src.models.data_utils import HatexplainDataset
from src.models.model_utils import PackingLayer

learning_rate = 100
n_head = 12
max_token_len = 113


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

        self.loss = AttentionEntropyLoss(phi=0.2)
        self.metrics = [CompatibleMetric(metric_fn=keras.metrics.categorical_accuracy, name="categorical_accuracy")]
        self.optimizer = self.get_opt

    def get_opt(self, learning_rate: float):
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

    def fine_tune_and_train_mdl(
            self, dataset: HatexplainDataset,
            first_lr: float,
            second_lr: float,
            n_epochs: int,
            phi: float,
    ):
        model = self.model
        num_epochs = n_epochs

        train_data = dataset.get_train_generator()
        test_data = dataset.get_test_generator()

        # Set up to train outer layer with high learning rate
        model.compile(
            optimizer=self.optimizer(learning_rate=first_lr),
            metrics=self.metrics,
            loss=AttentionEntropyLoss(phi=None),  # do not use entropy loss when just tuning the outer loss
            run_eagerly=False,
        )

        for layer in model.layers:
            layer.trainable = False

        for layer_i in range(5, len(model.layers)):
            model.layers[layer_i].trainable = True

        print("\n\n~~ Training ~~\n\n")

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

        # Fine tune on sub data
        model.compile(
            optimizer=self.optimizer(learning_rate=second_lr),
            metrics=self.metrics,
            loss=AttentionEntropyLoss(phi=phi),
            run_eagerly=False,
        )

        for layer in model.layers:
            layer.trainable = True

        print("\n\n--Done training head, fine tuning full model--\n\n")
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
        first_lr=1e-3,
        second_lr=1e-5,
        n_epochs=1,
        phi=3,
    )
