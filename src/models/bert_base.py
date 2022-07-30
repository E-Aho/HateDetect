import tensorflow as tf

from src.models import BASE_MODEL_MODEL_PATH
from src.models.abstract_base_model import AbstractModel
from src.models.data_utils import HatexplainDataset


class BaseBertModel(AbstractModel):

    def __init__(self, input_shape: tuple, output_shape: tuple):
        model_pretrained_name = "roberta-base"
        save_path = BASE_MODEL_MODEL_PATH
        self.dropout_rate = 0.2

        super().__init__(
            input_shape=input_shape,
            output_shape=output_shape,
            model_name=model_pretrained_name,
            save_path=save_path,
        )

        self.loss = "categorical_crossentropy"
        self.metrics = ["categorical_accuracy"]

    def get_opt(self, learning_rate: float,):
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)

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
