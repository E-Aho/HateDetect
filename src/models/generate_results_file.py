from pathlib import Path
from typing import Tuple, Dict

import pandas as pd

from src.models.attention_entropy_loss import CompatibleMetric, AttentionEntropyLoss
from src.models.bert_with_attention_entropy import BertModelWithAttentionEntropy
from src.models.data_utils import HatexplainDataset
from src.models.hyperparameter_sweep import ModelVariant, base_model, twitter_model, sentiment_model, emotion_model
import tensorflow as tf

from src.models.model_utils import predict_from_ear_model, PackingLayer

RESULTS_FILE = Path("results")

def generate_results_for_model(run_config: Dict, checkpoint_path: Path, model_variant: ModelVariant, out_name: str):

    dataset = HatexplainDataset(
        pd.read_parquet(model_variant.dataset_path),
        p_test=0.2,
        p_val=0.1,
        batch_size=run_config["batch_size"],
    )

    model = BertModelWithAttentionEntropy(
      input_shape=dataset.model_input_size,
      output_shape=dataset.model_output_size,
      epochs=10,
      head_learning_rate=run_config["head_learning_rate"],
      fine_tune_learning_rate=run_config["fine_tune_learning_rate"],
      phi=run_config["phi"],
      batch_size=run_config["batch_size"],
      dropout_rate=run_config["dropout_rate"],
      base_model_name=model_variant.model_name,
      from_pt=model_variant.from_pt,
      save_path=model_variant.save_path,
    )

    model.model.load_weights(checkpoint_path/"variables"/"variables")

    result_df = predict_from_ear_model(x=dataset.get_val_generator(), steps=dataset.val_batches, model=model.model)
    simple_results = result_df[["tokens", "labels", "prediction"]]
    simple_results[["t_hate", "t_normal", "t_offensive", "t_split"]] = pd.DataFrame(simple_results.labels.tolist(), index=simple_results.index)
    simple_results[["p_hate", "p_normal", "p_offensive", "p_split"]] = pd.DataFrame(simple_results.prediction.tolist(), index=simple_results.index)

    simple_results.to_csv(RESULTS_FILE/f"{out_name}_simple.csv")
    result_df.to_parquet(RESULTS_FILE/f"{out_name}_full.parquet")


if __name__ == "__main__":

    BEST_SENTIMENT = {
        "batch_size": 32,
        "dropout_rate": 0.17116138498335182,
        "epochs": 10,
        "fine_tune_learning_rate": 0.00032784376760056457,
        "head_learning_rate": 0.0048805961553678905,
        "phi": 0.00789578446841281,
    }
    BEST_EMOTION = {
        "batch_size": 32,
        "dropout_rate": 0.11964834257592596,
        "epochs": 10,
        "fine_tune_learning_rate": 0.006722184593747569,
        "head_learning_rate": 0.0013447188857698632,
        "phi": 0.0023978490206497767,
    }
    BEST_TWITTER = {
        "batch_size": 32,
        "dropout_rate": 0.25163495511605594,
        "epochs": 10,
        "fine_tune_learning_rate":  0.00013676752225509297,
        "head_learning_rate": 0.005847387020815067,
        "phi": 0.006660372873379781,
    }
    BEST_BASE = {
        "batch_size": 32,
        "dropout_rate": 0.1630668350495539,
        "epochs": 10,
        "fine_tune_learning_rate": 0.004696244660723228,
        "head_learning_rate": 0.009693210660178166,
        "phi": 0.00429884332331886,
    }

    model_path = Path("saved_models/base_roberta")
    SENTIMENT = model_path / "ver_26"
    EMOTION = model_path / "ver_61"
    TWITTER_BASE = model_path / "ver_40"
    BASE = model_path / "ver_29"

    generate_results_for_model(
        run_config=BEST_BASE,
        checkpoint_path=BASE,
        model_variant=base_model,
        out_name="base_pred",
    )

    generate_results_for_model(
        run_config=BEST_TWITTER,
        checkpoint_path=TWITTER_BASE,
        model_variant=twitter_model,
        out_name="twitter_pred"
    )

    generate_results_for_model(
        run_config=BEST_SENTIMENT,
        checkpoint_path=SENTIMENT,
        model_variant=sentiment_model,
        out_name="sentiment_pred"
    )

    generate_results_for_model(
        run_config=BEST_EMOTION,
        checkpoint_path=EMOTION,
        model_variant=emotion_model,
        out_name="emotion_pred"
    )

