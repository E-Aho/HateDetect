from typing import List, Generator, Union

import numpy as np
import pandas as pd
import tensorflow as tf

from src.models import RNG_SEED

BATCH_SIZE = 64
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
        # df = df.sample(frac=1, random_state=RNG_SEED) # shuffle order of split each epoch
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

        if x:
            yield pivot_nested_list(x), np.array(y)
            x, y = [], []



def prepare_dataset_splits(
        df: pd.DataFrame, n_train: int, n_test: int, n_val: int = 0, rng_seed=RNG_SEED,
):
    # Add attention head empty output here
    shuffled_df = df.sample(frac=1, random_state=rng_seed)
    train_df = shuffled_df.iloc[:n_train]
    test_df = shuffled_df.iloc[n_train: n_test+n_train]

    if n_val < 1:
        return train_df, test_df
    else:
        val_df = shuffled_df.iloc[n_train + n_test:]
        return train_df, test_df, val_df


class HatexplainDataset:
    def __init__(self, initial_df: pd.DataFrame, p_test: float, p_val: float=0.0):
        self.batch_size = BATCH_SIZE
        self.initial_dataframe = initial_df
        self.dataset_size = len(self.initial_dataframe)
        self.token_width = len(initial_df["input_ids"][0])
        self.model_input_size = (self.token_width,)
        self.model_output_size = 4

        p_train = 1 - (p_val + p_test)

        n_train = int(np.floor(p_train * self.dataset_size))
        n_test = int(np.floor(p_test * self.dataset_size))
        n_val = int(np.floor(p_val * self.dataset_size))

        data_splits = prepare_dataset_splits(self.initial_dataframe, n_train=n_train, n_test=n_test, n_val=n_val)
        self.train_batches = n_train // self.batch_size
        self.test_batches = n_test // self.batch_size

        self.train_df = data_splits[0]
        self.test_df = data_splits[1]

        print(f"LENGTH OF TRAIN: {len(self.train_df)}")
        print(f"LENGTH OF TEST: {len(self.test_df)}")
        print(f"BATCH N: {self.train_batches, self.test_batches}")

        if len(data_splits) <= 2:
            self.val_df = None
        else:
            self.val_df = data_splits[2]

    def get_train_generator(self) -> Generator:
        return dataset_generator(self.train_df)

    def get_test_generator(self) -> Generator:
        return dataset_generator(self.test_df)

    def get_val_generator(self) -> Union[None, iter]:
        if self.val_df is None:
            return None
        return dataset_generator(self.val_df)


def entropy(p):
    plogp = p * tf.math.log(p)
    plogp[p == 0] = 0
    return tf.math.cumsum(plogp)