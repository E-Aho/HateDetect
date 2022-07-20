import abc
import gc
from abc import ABC
from collections import Generator
from pathlib import Path
from typing import List

import keras
import pandas as pd
import tensorflow as tf

from transformers import (
    RobertaTokenizer,
    TFRobertaModel,
)

from datasets.hatexplain import hatexplain_dataset_path
from src.models import BASE_MODEL_MODEL_PATH
from src.models.data_utils import HatexplainDataset


class GarbageCollectCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        tf.keras.backend.clear_session()


class AbstractModel(ABC):
    def __init__(
            self,
            input_shape: tuple,
            output_shape: tuple,
            model_name: str,
            save_path: Path,
            tokenizer_name: str = None,
    ):
        if tokenizer_name is None:
            tokenizer_name = model_name

        self.tokenizer = RobertaTokenizer.from_pretrained(
            tokenizer_name
        )

        self.bert = TFRobertaModel.from_pretrained(
            model_name
        )
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.model = self.get_model(input_shape=input_shape, output_shape=output_shape,)
        self.save_path = save_path
        self.callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=self.get_save_location(),
                save_weights_only=False,
                monitor='val_categorical_accuracy',
                save_best_only=True,
            ),
            GarbageCollectCallback(),

        ]

    @abc.abstractmethod
    def get_model(self, input_shape: tuple, output_shape: tuple,) -> tf.keras.Model:
        pass

    def get_save_location(self):
        paths = self.save_path.glob("ver_*")
        existing_file_names = [p.stem for p in paths]
        if not existing_file_names:
            return self.save_path / "ver_0"

        existing_versions = [p.split("ver_")[1] for p in existing_file_names]
        next_iter = 1 + max([int(p) if p.isnumeric() else 0 for p in existing_versions])

        return self.save_path / f"ver_{next_iter}"

    def make_predictions(self, data_generators: Generator, data_steps: List[int]):
        all_outs = []
        for generator in data_generators:
            outputs = self.model.predict(generator, steps=data_steps)
            row_based_outputs = None  # spread tensor to result
            all_outs.append(outputs)

        # TODO: Convert back to format and do loss calculations, etc


class BaseBertModel(AbstractModel):

    def __init__(self, input_shape: tuple, output_shape: tuple):
        model_pretrained_name = "roberta-base"
        save_path = BASE_MODEL_MODEL_PATH
        super().__init__(
            input_shape=input_shape,
            output_shape=output_shape,
            model_name=model_pretrained_name,
            save_path=save_path,
        )

        self.loss = tf.keras.losses.categorical_crossentropy()
        self.metrics = [tf.keras.metrics.categorical_accuracy(), ]
        self.optimizer = tf.keras.optimizers.Adam
        self.dropout_rate = 0.2

    def get_opt(self, learning_rate: float,) -> tf.keras.optimizers:
        return self.optimizer(learning_rate=learning_rate)

    def get_model(self, input_shape: tuple, output_shape: tuple,) -> tf.keras.Model:
        input_ids = tf.keras.layers.Input(shape=input_shape, dtype=tf.int32, name="input_ids")

        input_attention_mask = tf.keras.layers.Input(
            shape=input_shape,
            dtype=tf.int32,
            name="attention_mask"
        )

        bert_model = self.bert([input_ids, input_attention_mask])

        last_hidden_state = bert_model.last_hidden_state
        cls_token_out = last_hidden_state[:, 0, :]  # the CLS token is used to represent sentence level classification
        dropout = tf.keras.layers.Dropout(self.dropout_rate)(cls_token_out)

        output = tf.keras.layers.Dense(output_shape, activation="softmax")(dropout)  # 3 for non-split training

        model = tf.keras.models.Model(inputs=[input_ids, input_attention_mask], outputs=output)
        return model

    def fine_tune_and_train_mdl(self, dataset: HatexplainDataset, learning_rate: float, n_epochs: int,):
        model = self.model
        num_epochs = n_epochs

        model.compile(
            optimizer=self.get_opt(learning_rate),
            loss=self.loss,
            metrics=self.metrics,
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


def train_mdl(mdl: BaseBertModel, dataset: HatexplainDataset, learning_rate: float, n_epochs: int, ):
    pass

def fine_tune_and_train():
    pass #todo


if __name__ == "__main__":
    p_test = 0.2
    learning_rate = 1e-5
    epochs = 8

    # TODO: Train fully conected layers with higher learning rate (1e-3)

    hatexplain_dataset = HatexplainDataset(pd.read_parquet(hatexplain_dataset_path), p_test=p_test)

    base_mdl = BaseBertModel(
        input_shape=hatexplain_dataset.model_input_size,
        output_shape=hatexplain_dataset.model_output_size,
    )

    train_mdl(
        base_mdl,
        dataset=hatexplain_dataset,
        learning_rate=learning_rate,
        n_epochs=epochs,
    )




