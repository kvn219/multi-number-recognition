import numpy as np
import tensorflow as tf
from tensorflow.python.ops.nn_ops import softmax, sparse_softmax_cross_entropy_with_logits

results_fmt = '[{:%H:%M:%S}] EPOCH {}/{} LOSS {:.3f}, DEV: {:.3f}'

train_eval = "[{:%H:%M:%S}] Step {:5d}/{}, Loss: {:.6f}, Train: {:.3f}, Valid: {:.3f}"

eval_fmt = '[{:%H:%M:%S}] Step: {:5d}, Loss: {:09.5f}, Seq Acc: {:.2f}, Dig Acc: {:.2f}'


def get_incoming_shape(incoming):
    """ Returns the incoming data shape """
    if isinstance(incoming, tf.Tensor):
        return incoming.get_shape().as_list()
    elif type(incoming) in [np.array, list, tuple]:
        return np.shape(incoming)
    else:
        raise Exception("Invalid incoming layer.")


def accuracy(correct_in_image):
    return tf.reduce_mean(tf.cast(tf.equal(tf.reduce_sum(tf.cast(correct_in_image, tf.int32), axis=1), 5), tf.float32))


def sequence_accuracy(predictions, labels):
    with tf.name_scope("Sequence"):
        scores = score_digits_in_image(predictions)
        correct_digits = correct_digits_in_image(scores, labels)
        return accuracy(correct_digits)


def score_digits_in_image(predictions):
    with tf.name_scope("ScoreDigits"):
        return tf.transpose(tf.argmax(predictions, 2))


def correct_digits_in_image(scores_in_image, labels):
    with tf.name_scope("CorrectDigits"):
        return tf.equal(scores_in_image, labels[:, 1:])


def digit_accuracy(correct_in_image, labels):
    with tf.name_scope("Digit"):
        shape = get_incoming_shape(labels[:, 1:])
        digits = tf.cast(correct_in_image, tf.float32)
        return tf.div(tf.div(tf.reduce_sum(digits), shape[0]), shape[1])


def reduce_cross_entropy(logits, lablels):
    with tf.name_scope('CrossEntropy'):
        return tf.reduce_mean(sparse_softmax_cross_entropy_with_logits(logits, lablels))


def error(logits, labels):
    with tf.name_scope("Error"):
        return tf.reduce_sum([reduce_cross_entropy(logits[i], labels[:, i + 1]) for i in range(5)])


def predict(logits):
    with tf.name_scope("PredictSequence"):
        return tf.pack([softmax(logits[i]) for i in range(5)])


def get_size(x):
    return x.get_shape().as_list()
