from decimal import Decimal
from typing import List
from unittest.mock import MagicMock

import numpy as np
import tensorflow as tf

from src.models.model_utils import PackedTensor, predict_from_ear_model


class TestPackedTensor:
    def test_to_numpy_works_as_expected(self):

        prediction_input = np.array([
            [.1, .2, .3, .4],
            [.1, .4, .7, .9]
        ])

        attention_input = np.array([
            [[1, 1], [2, 2]],
            [[3, 3], [4, 4]]
        ])

        mask_input = np.array([
            [[0, 1], [1, 1]],
            [[1, 1], [0, 0]],
        ])

        packed_tensor = PackedTensor(
            tf.convert_to_tensor(prediction_input),
            tf.convert_to_tensor(attention_input),
            tf.convert_to_tensor(mask_input)
        )

        out = packed_tensor.get_values()
        expected_out = [prediction_input, attention_input, mask_input]

        for i in range(len(out)):
            assert np.array_equal(out[i], expected_out[i])


class TestPredictFromEARModel:

    """Simple test to make sure the hacky way of iterating predictions works as expected"""

    def get_mock_model(self, expected_outputs: List[PackedTensor]):
        mock_model = MagicMock()
        mock_model.side_effect = expected_outputs
        return mock_model

    def make_generator(self, x_in: dict, y_in: dict):
        batch_count = len(x_in["token"])
        batch_width = len(x_in["token"][0])
        for b in range(batch_count):
            yield [
                [
                    tf.convert_to_tensor(x_in["token"][b][i], tf.int32),
                    tf.convert_to_tensor(x_in["mask"][b][i], tf.int32)
                ] for i in range(batch_width)
            ], [
                tf.convert_to_tensor(y_in["pred"][b][i]) for i in range(batch_width)
            ]

    def test_for_simple_case_returns_as_expected(self):

        token_in = [
            [[1, 2, 3, 2, 1], [2, 1, 2, 1, 2]],
            [[1, 2, 3, 3, 2], [2, 11, 2, 5, 6]],
            [[5, 1, 9, 2, 1], [9, 1, 2, 0, 2]],
        ]
        mask_in = [
            [[1, 1, 0, 0, 0], [1, 1, 1, 0, 0]],
            [[0, 1, 1, 0, 0], [1, 1, 1, 0, 1]],
            [[1, 1, 0, 1, 0], [0, 1, 0, 0, 1]],
        ]
        true_out = [
            [[1, 0, 0, 0], [1, 0, 0, 0]],
            [[0, 1, 0, 0], [0, 0, 1, 0]],
            [[0, 0, 0, 1], [1, 0, 0, 0]]
        ]
        predicted_out = [
            [[1, 0, 0, 0], [1, 0, 0, 0]],
            [[0, 0, 0, 1], [0, 0, 1, 0]],
            [[0, 0, 1, 0], [0, 0, 0, 1]]
        ]
        attn_out = [
            [[.1, .1, .1, .1, .1], [.1, .1, .1, .1, .3]],
            [[.5, .1, .2, .1, .1], [.1, .1, .8, .1, .1]],
            [[.1, .1, .1, .9, .1], [.1, .4, .1, .1, .1]],
        ]

        x_in = {"token": token_in, "mask": mask_in}
        y_in = {"pred": true_out}

        generator = self.make_generator(x_in, y_in)

        expected_outputs = [
            PackedTensor(
                tf.convert_to_tensor(predicted_out[i], tf.float32),
                tf.convert_to_tensor(attn_out[i], tf.float32),
                tf.convert_to_tensor(mask_in[i], tf.float32)
            )
            for i in range(len(attn_out))
        ]

        mock_model = self.get_mock_model(expected_outputs)

        returned_df = predict_from_ear_model(
            generator,
            steps=len(token_in),
            model=mock_model
        )

        def round_list(l) -> List:
            return [round(x, 1) for x in l]

        for b in range(len(token_in)):

            assert returned_df["tokens"][b*2] == token_in[b][0]
            assert returned_df["tokens"][b*2 + 1] == token_in[b][1]

            assert returned_df["mask"][b*2] == mask_in[b][0]
            assert returned_df["mask"][b*2 + 1] == mask_in[b][1]

            assert returned_df["labels"][b*2] == true_out[b][0]
            assert returned_df["labels"][b*2 + 1] == true_out[b][1]

            assert returned_df["prediction"][b*2] == round_list(predicted_out[b][0])
            assert returned_df["prediction"][b*2 + 1] == round_list(predicted_out[b][1])

            assert round_list(returned_df["attention"][b * 2]) == attn_out[b][0]
            assert round_list(returned_df["attention"][b * 2 + 1]) == attn_out[b][1]
