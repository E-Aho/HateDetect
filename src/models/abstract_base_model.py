import abc
import gc
from abc import ABC
from collections import Generator
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd
import tensorflow as tf

from transformers import (
    RobertaTokenizer,
    TFRobertaModel, PretrainedConfig, BertConfig, AutoConfig,
)

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
            model_config: BertConfig = None,
    ):
        if tokenizer_name is None:
            tokenizer_name = model_name

        if model_config is None:
            model_config = AutoConfig.from_pretrained(
                model_name
            )


        self.tokenizer = RobertaTokenizer.from_pretrained(
            tokenizer_name
        )

        self.bert = TFRobertaModel.from_pretrained(
            model_name,
            config=model_config,
        )
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.model = self.get_model(input_shape=input_shape, output_shape=output_shape,)
        self.save_path = save_path
        self.callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=self.get_save_location(),
                save_weights_only=False,
                monitor="val_categorical_accuracy",
                save_best_only=True,
            ),
            GarbageCollectCallback(),
        ]

    def get_opt(self, learning_rate: float,):
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)

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


def train_mdl(mdl: AbstractModel, dataset: HatexplainDataset, learning_rate: float, n_epochs: int, ):
    pass


def fine_tune_and_train():
    pass  # todo


def join_predictions_to_source_data(data: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
    data["pred_hate"] = predictions[:, 0]
    data["pred_normal"] = predictions[:, 1]
    data["pred_offensive"] = predictions[:, 2]
    data["pred_split"] = predictions[:, 3]
    return data


def make_predictions(model, data_generators: List[Generator], data_steps: List[int], data_sources: List[pd.DataFrame]):
    all_outs = []
    out_dfs = []
    for i in range(len(data_generators)):

        outputs = model.predict(data_generators[i], steps=data_steps[i])
        all_outs.append(outputs)
        df = join_predictions_to_source_data(data_sources[i], outputs)
        out_dfs.append(df)

    result_df = pd.concat(out_dfs)
    return result_df

    # TODO: do loss calculations etc (currently in form of nd arrays)
