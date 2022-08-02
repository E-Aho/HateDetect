import argparse
import random

import keras.callbacks
import pandas as pd
import wandb

from datasets.hatexplain import hatexplain_dataset_path
from src.models import RNG_SEED
from src.models.bert_with_attention_entropy import BertModelWithAttentionEntropy
from src.models.constants import WANDB_PROJECT_NAME
from src.models.data_utils import HatexplainDataset

import tensorflow as tf


class WandBCallback(keras.callbacks.Callback):
    def __init__(self, wb_api):
        super().__init__()
        self.wb_api = wb_api

    def on_epoch_end(self, epoch, logs):
        print(f"logging epoch results: val_acc = {logs['val_categorical_accuracy']}")
        self.wb_api.log({
            "epoch": logs["epoch"],
            "loss": logs["loss"],
            "val_loss": logs["val_loss"],
            "categorical_accuracy": logs["categorical_accuracy"],
            "val_categorical_accuracy": logs["val_categorical_accuracy"],
        })


def train():
    wandb.init(project=WANDB_PROJECT_NAME)
    run_config = wandb.config

    dataset = HatexplainDataset(
        pd.read_parquet(hatexplain_dataset_path),
        p_test=0.25,
        batch_size=run_config["batch_size"],
        rng_seed=random.randint(1, 10**7),
    )

    model = BertModelWithAttentionEntropy(
      input_shape=dataset.model_input_size,
      output_shape=dataset.model_output_size,
      epochs=run_config["epochs"],
      head_learning_rate=run_config["head_learning_rate"],
      fine_tune_learning_rate=run_config["fine_tune_learning_rate"],
      phi=run_config["phi"],
      batch_size=run_config["batch_size"],
      dropout_rate=run_config["dropout_rate"],
    )

    trained_mdl = model.fine_tune_and_train_mdl(
        dataset=dataset,
        custom_callback=WandBCallback(wandb),
    )


def sweep_run():

    random.seed = RNG_SEED

    sweep_config = {
        "name": "first_sweep",
        "method": "grid",
        "parameters": {
            "epochs": {
                "value": 20
            },
            "head_learning_rate": {
                "values": [1e-2, 1e-3, 1e-4]
            },
            "fine_tune_learning_rate": {
                "values": [1e-3, 1e-4, 1e-5]
            },
            "phi": {
                "values": [10, 1, 0.1, 1e-2]
            },
            "batch_size": {
                "values": [32, 64, 128]
            },
            "dropout_rate": {
                "values": [0.1]
            }
        }
    }
    sweep_id = wandb.sweep(sweep_config)
    wandb.agent(sweep_id=sweep_id, function=train, count=None)


if __name__ == "__main__":
    sweep_run()
