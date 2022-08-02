import pandas as pd

from datasets.hatexplain import hatexplain_dataset_path
from src.models.abstract_base_model import make_predictions
from src.models.bert_base import BaseBertModel
from src.models.data_utils import HatexplainDataset

if __name__ == "__main__":
    p_test = 0.2
    learning_rate = 1e-5
    epochs = 8

    # TODO: Train fully connected layers with higher learning rate (1e-3)

    hatexplain_dataset = HatexplainDataset(pd.read_parquet(hatexplain_dataset_path), p_test=p_test)

    base_mdl = BaseBertModel(
        input_shape=hatexplain_dataset.model_input_size,
        output_shape=hatexplain_dataset.model_output_size,
    )

    trained_mdl = base_mdl.fine_tune_and_train_mdl(
        dataset=hatexplain_dataset,
        learning_rate=learning_rate,
        n_epochs=8,
    )

    prediction_df = make_predictions(
        trained_mdl,
        data_generators=hatexplain_dataset.get_generators(),
        data_steps=hatexplain_dataset.get_steps(),
        data_sources=hatexplain_dataset.get_data_sources(),
    )

    prediction_df.to_csv(base_mdl.save_path/"predictions.csv", header=True)




