import os
import pathlib
import shutil
from typing import Tuple, Generator, List, Union

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub

import transformers
from transformers import (
    RobertaConfig,
    RobertaTokenizer,
    TFRobertaForSequenceClassification,
    set_seed, create_optimizer,
)

from datasets.hatexplain import hatexplain_dataset_path
from src.models import RNG_SEED

DATASET_MAX_SIZE = None
BATCH_SIZE = 32
P_TEST = 0.3
P_VAL = 0

x_cols = ["input_ids", "attention_mask"]
y_cols = ["hate", "normal", "offensive", "split"]


def pivot_nested_list(input_arr: List[List]) -> List[np.array]:
    """Pivots a list of lists, so that each 'column' is its own array"""
    return [np.array([input_arr[i][j] for i in range(len(input_arr))]) for j in range(len(input_arr[0]))]


def dataset_generator(df: pd.DataFrame):
    x, y = [], []
    while True:
        for _, row in df.iterrows():
            x_row = list([tf.convert_to_tensor(x, tf.int32) for x in row[x_cols]])
            y_row = list([tf.cast(y, tf.float32) for y in row[y_cols]])
            x.append(x_row)
            y.append(y_row)
            if len(x) >= BATCH_SIZE:
                x_arr = pivot_nested_list(x)
                y_arr = np.array(y)
                yield x_arr, y_arr
                x, y = [], []


class HatexplainDataset:
    def __init__(self, initial_df: pd.DataFrame, p_test: float, p_val: float=0.0):
        self.batch_size = BATCH_SIZE
        self.initial_dataframe = initial_df
        self.dataset_size = len(self.initial_dataframe)
        self.token_width = len(initial_df["input_ids"][0])
        self.num_batches = np.ceil(self.dataset_size / self.batch_size)

        def __get_datasets__(df: pd.DataFrame) -> Tuple[pd.DataFrame, ...]:
            shuffled_df = df.sample(frac=1, random_state=RNG_SEED)
            # Add attention head empty output here

            # ds = tf.data.Dataset.from_tensor_slices(shuffled_df.to_dict(orient="list"))

            # Get lengths of dataset for tf
            p_train = 1 - (p_val + p_test)
            n_train = int(p_train * self.dataset_size)
            n_val = int(p_val * self.dataset_size)
            n_test = self.dataset_size - (n_val + n_train)

            train_df = shuffled_df.iloc[:n_train]
            test_df = shuffled_df.iloc[n_train:n_test]

            if n_val < 1:
                return train_df, test_df
            else:
                val_df = shuffled_df.iloc[n_train + n_test:]
                return train_df, test_df, val_df

        all_splits = __get_datasets__(self.initial_dataframe)
        self.train_df = all_splits[0]
        self.test_df = all_splits[1]
        if len(all_splits) <= 2:
            self.val_df = None
        else:
            self.val_df = all_splits[2]

    def get_train_generator(self) -> Generator:
        return dataset_generator(self.train_df)

    def get_test_generator(self) -> Generator:
        return dataset_generator(self.test_df)

    def get_val_generator(self) -> Union[None, iter]:
        if self.val_df is None:
            return None
        return dataset_generator(self.val_df)


class DataLoader:
    def __init__(self, location: pathlib.Path):
        self.path = location
        self.dataset = self.__init_dataset__(location)

    def __init_dataset__(self, location: pathlib.Path) -> HatexplainDataset:

        pd_df = pd.read_parquet(location)
        ds = HatexplainDataset(pd_df, p_test=0.3)
        return ds

    #
    # def get_dataset(self, p_test: float, p_val: float = 0, batch_size: int = 32,) -> Tuple[tf.data.Dataset, ...]:
    #     if p_test + p_val >= 1.0:
    #         raise Exception(f"P_test and p_train are too large. Must add to less than 1.\nP_test={p_test}, p_train={p_val}")
    #     if p_test < 0 or p_val < 0:
    #         raise Exception(f"Proportions must be greater than 0.\nP_test={p_test}, p_train={p_val}")
    #
    #     p_train = 1 - (p_val + p_test)
    #     n_train = int(p_train * self.dataset_size)
    #     n_val = int(p_val * self.dataset_size)
    #     n_test = self.dataset_size - (n_val + n_train)
    #     shuffled_dataset = self.dataset.shuffle(
    #         buffer_size=self.dataset_size, seed=RNG_SEED,
    #         reshuffle_each_iteration=False
    #     )
    #
    #     if n_val > 0:
    #         return (
    #             shuffled_dataset.take(n_train).batch(batch_size=batch_size, drop_remainder=False),
    #             shuffled_dataset.skip(n_train).take(n_test).batch(batch_size=batch_size, drop_remainder=False),
    #             shuffled_dataset.skip(n_train+n_test).batch(batch_size=batch_size, drop_remainder=False)
    #         )
    #     else:
    #         return (
    #             shuffled_dataset.take(n_train).batch(batch_size=batch_size, drop_remainder=False),
    #             shuffled_dataset.skip(n_train).batch(batch_size=batch_size, drop_remainder=False)
    #         )


class BaseBertModel:

    def __init__(self, dataloader: DataLoader):
        self.model_pretrained_name = "roberta-base"
        self.tokenizer = RobertaTokenizer.from_pretrained(
            self.model_pretrained_name,
        )

        self.bert_model = TFRobertaForSequenceClassification.from_pretrained(
            self.model_pretrained_name,
        )

        self.dataset = dataloader.dataset
        self.dataloader = dataloader
        self.token_width = self.dataset.token_width

    def get_model(self) -> TFRobertaForSequenceClassification:
        return self.bert_model



def train_mdl(mdl: BaseBertModel):
    model = mdl.get_model()
    batch_size = BATCH_SIZE
    num_epochs = 5

    bert_model = model
    input_ids = tf.keras.layers.Input(shape=(mdl.dataset.token_width,), dtype=tf.int32, name="input_ids")
    input_attention_mask = tf.keras.layers.Input(shape=(mdl.dataset.token_width,), dtype=tf.int32, name="attention_mask")
    bert_out = bert_model([input_ids, input_attention_mask])[0]
    output = tf.keras.layers.Dense(4, activation="softmax")(bert_out)  # 3 for non-split training
    model = tf.keras.models.Model(inputs=[input_ids, input_attention_mask], outputs=output)

    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    train_data = mdl.dataset.get_train_generator()
    test_data = mdl.dataset.get_test_generator()
    model.fit(
        train_data, validation_data=test_data,
        epochs=num_epochs, batch_size=batch_size, steps_per_epoch=mdl.dataset.num_batches,
    )


def entropy(p):
    plogp = p * tf.math.log(p)
    plogp[p == 0] = 0
    return tf.math.cumsum(plogp)


if __name__ == "__main__":
    dataloader = DataLoader(hatexplain_dataset_path)
    base_mdl = BaseBertModel(dataloader)

    train_mdl(base_mdl)


