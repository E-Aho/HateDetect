import tensorflow as tf
from keras.losses import Loss


@tf.function
def entropy_of_tensor(t: tf.Tensor):
    t = tf.boolean_mask(t, tf.math.greater(t, tf.zeros_like(t)))  # filter out zero values
    tlogt = t * tf.math.log(t)
    return - tf.math.reduce_sum(tlogt, axis=0)


@tf.function
def calculate_masked_entropy_for_subbatch(a_3d: tf.Tensor, one_d_attention_mask: tf.Tensor):
    bool_mask = tf.cast(one_d_attention_mask, tf.bool)

    masked_attention = tf.boolean_mask(
        tf.boolean_mask(a_3d, mask=bool_mask, axis=2), mask=bool_mask, axis=1
    )  # mask both axes of attention-attention mapping

    return tf.reduce_mean(
        tf.map_fn(entropy_of_tensor, masked_attention)
    )


@tf.function
def calculate_masked_entropy(attentions: tf.Tensor, attention_mask: tf.Tensor) -> tf.Tensor:

    return tf.reduce_mean(tf.map_fn(
            lambda batch: calculate_masked_entropy_for_subbatch(batch[0], batch[1]),
            [attentions, attention_mask], fn_output_signature=tf.float32
    ))


class AttentionEntropyLoss(Loss):
    def __init__(self, chi: float = 0.1):
        super().__init__()
        self.chi = chi
        self.loss_fn = tf.keras.losses.CategoricalCrossentropy()

    def call(self, y_true, y_pred):
        predicted_labels, attentions, attention_mask = y_pred.output_0, y_pred.output_1, y_pred.output_2

        cat_loss = self.loss_fn(y_true, predicted_labels)
        entropy_loss = calculate_masked_entropy(attentions, attention_mask)

        _o = cat_loss + tf.scalar_mul(self.chi, entropy_loss)
        return _o
