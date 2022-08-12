import argparse
import random
from math import exp, log
from pathlib import Path
from typing import Tuple

import keras.callbacks
import wakepy
from keras import backend as K
import pandas as pd
import wandb

from datasets.hatexplain import hatexplain_dataset_path, hatexplain_twitter_roberta_path, hatexplain_sentiment_path, \
    hatexplain_emotion_path
from src.models import RNG_SEED, BASE_MODEL_MODEL_PATH, TWITTER_MODEL_PATH, SENTIMENT_MODEL_PATH, EMOTION_MODEL_PATH
from src.models.bert_with_attention_entropy import BertModelWithAttentionEntropy
from src.models.model_utils import predict_from_ear_model
from src.models.constants import WANDB_ENTROPY_COMPARE_PROJECT, WANDB_MODEL_COMPARE_PROJECT
from src.models.data_utils import HatexplainDataset


class ModelVariant:
    def __init__(self, model_name: str, dataset_path: Path, from_pt: bool, save_path):
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.from_pt = from_pt
        self.save_path = save_path


base_model = ModelVariant("roberta-base", hatexplain_dataset_path, False, BASE_MODEL_MODEL_PATH)
twitter_model = ModelVariant("cardiffnlp/twitter-roberta-base-emotion", hatexplain_twitter_roberta_path, True, TWITTER_MODEL_PATH)
sentiment_model = ModelVariant("cardiffnlp/twitter-roberta-base-sentiment-latest", hatexplain_sentiment_path, True, SENTIMENT_MODEL_PATH)
emotion_model = ModelVariant("cardiffnlp/twitter-roberta-base-emotion", hatexplain_emotion_path, True, EMOTION_MODEL_PATH)


def get_model_variant(conf_name: str) -> ModelVariant:
    return {
        "base": base_model,
        "twitter-base": twitter_model,
        "sentiment": sentiment_model,
        "emotion": emotion_model,
    }.get(conf_name)


class WandBCallback(keras.callbacks.Callback):
    def __init__(self, wb_api):
        super().__init__()
        self.wb_api = wb_api

    def on_epoch_end(self, epoch, logs=None):
        self.wb_api.log({
            "loss": logs["loss"],
            "val_loss": logs["val_loss"],
            "categorical_accuracy": logs["categorical_accuracy"],
            "val_categorical_accuracy": logs["val_categorical_accuracy"],
        })


def train():
    K.clear_session()
    wandb.init(project=WANDB_ENTROPY_COMPARE_PROJECT)
    run_config = wandb.config

    model_variant = get_model_variant(run_config["variant"])

    dataset = HatexplainDataset(
        pd.read_parquet(model_variant.dataset_path),
        p_test=0.2,
        p_val=0.1,
        batch_size=run_config["batch_size"],
    )

    if run_config["phi"] <= 0:
        phi = None
    else:
        phi = run_config["phi"]

    model = BertModelWithAttentionEntropy(
      input_shape=dataset.model_input_size,
      output_shape=dataset.model_output_size,
      epochs=run_config["epochs"],
      head_learning_rate=run_config["head_learning_rate"],
      fine_tune_learning_rate=run_config["fine_tune_learning_rate"],
      phi=phi,
      batch_size=run_config["batch_size"],
      dropout_rate=run_config["dropout_rate"],
      base_model_name=model_variant.model_name,
      from_pt=model_variant.from_pt,
      save_path=model_variant.save_path,
    )

    trained_mdl = model.fine_tune_and_train_mdl(
        dataset=dataset,
        custom_callback=WandBCallback(wandb),
    )

    val_metrics = trained_mdl.evaluate(
        dataset.get_val_generator(),
        steps=dataset.val_batches,
        return_dict=True,
    )

    wandb.log(
        {f"final_{x}": y for x, y in val_metrics.items()}
    )

    # prediction_output = predict_from_ear_model(
    #     dataset.get_val_generator(),
    #     steps=dataset.val_batches,
    #     model=trained_mdl,
    # )

    # pd.DataFrame(prediction_output).to_csv(model.save_path/"val_predictions.csv")


def sweep_run():

    def get_sweep_config(
            sweep_name: str,
            desc: str,
            model_var: str,
    ):
        return {
            "name": f"{sweep_name}",
            "description": f"{desc}",
            "method": "bayes",
            "metric": {
                "name": "val_categorical_accuracy",
                "goal": "maximize",
            },
            "parameters": {
                "variant": {
                    "value": f"{model_var}"
                },
                "epochs": {
                    "values": [10]
                },
                "head_learning_rate": {
                    "distribution": "log_uniform_values",
                    "min": 1e-3,
                    "max": 1e-1
                    # "values": [1e-3, 3e-3, 1e-2]
                },
                "fine_tune_learning_rate": {
                    "distribution": "log_uniform_values",
                    "min": 1e-4,
                    "max": 1e-2,
                    # "values": [1e-5, 1e-4, 1e-3]
                },
                "phi": {
                    "distribution": "log_uniform_values",  # To test this for non-EAR, set phi to None
                    "min": 1e-3,
                    "max": 2e-2
                    # "values": [1e-2, 3e-2, 1e-1]
                },
                "batch_size": {
                    "values": [32]
                },
                "dropout_rate": {
                    "distribution": "uniform",
                    "min": 0.1,
                    "max": 0.4
                    # "values": [0.15]

                }
            }
        }

    random.seed = RNG_SEED

    base_sweep_config = get_sweep_config(
        sweep_name="base_model_baesian_1",
        desc="Base sweep for RoBERTA Base",
        model_var="base"
    )
    #
    # no_ear_sweep_config = get_sweep_config(
    #     sweep_name="no_ear_model_sweep_1",
    #     desc="Base model sweep with no EAR",
    #     model_var="base"
    # )
    #
    # no_ear_sweep_config["parameters"]["phi"] = {"value": -1.0}

    twitter_sweep_config = get_sweep_config(
        sweep_name="twitter_model_sweep_1",
        desc="Twitter bayesian sweep 1",
        model_var="twitter-base"
    )

    sentiment_sweep_config = get_sweep_config(
        sweep_name="sentiment_model_sweep_1",
        desc="Sentiment bayesian sweep 1",
        model_var="sentiment"
    )

    emotion_sweep_config = get_sweep_config(
        sweep_name="emotion_model_sweep_1",
        desc="Emotion bayesian sweep 1",
        model_var="emotion"
    )

    # wandb.login(key="946c560dfb68fe3e05de34ebb4da31536d1fe7bf")
    # sweep_id = wandb.sweep(base_sweep_config, project=WANDB_ENTROPY_COMPARE_PROJECT)
    # wandb.agent(sweep_id="t4ozj0jb", function=train)

    # no_ear_sweep_id = wandb.sweep(no_ear_sweep_config, project=WANDB_ENTROPY_COMPARE_PROJECT)
    # wandb.agent(sweep_id=no_ear_sweep_id, function=train)


    base_baesian_id = wandb.sweep(base_sweep_config, project=WANDB_MODEL_COMPARE_PROJECT)  # dg0eux06
    base_baesian_id = "dg0eux06"

    twitter_sweep_id = wandb.sweep(twitter_sweep_config, project=WANDB_MODEL_COMPARE_PROJECT)  # rizh7f8s
    sentiment_sweep_id = wandb.sweep(sentiment_sweep_config, project=WANDB_MODEL_COMPARE_PROJECT)  # lviwc9u6
    emotion_sweep_config = wandb.sweep(emotion_sweep_config, project=WANDB_MODEL_COMPARE_PROJECT)  # xfxlwe2d

    twitter_sweep_id = "rizh7f8s"
    sentiment_sweep_id = "lviwc9u6"
    emotion_sweep_config = "xfxlwe2d"

    run_count = 1
    # wandb.agent(sweep_id=base_baesian_id, function=train, count=run_count)
    wandb.agent(sweep_id=twitter_sweep_id, function=train, count=run_count)
    # wandb.agent(sweep_id=sentiment_sweep_id, function=train, count=run_count)
    # wandb.agent(sweep_id=emotion_sweep_config, function=train, count=run_count)

    run_count = 8
    # wandb.agent(sweep_id=base_baesian_id, function=train, count=run_count)
    # wandb.agent(sweep_id=twitter_sweep_id, function=train, count=run_count)
    wandb.agent(sweep_id=sentiment_sweep_id, function=train, count=run_count)
    wandb.agent(sweep_id=emotion_sweep_config, function=train, count=run_count)


if __name__ == "__main__":
    with wakepy.keepawake(keep_screen_awake=False):
        sweep_run()
