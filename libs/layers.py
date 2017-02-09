import numpy as np
import tensorflow as tf
from libs.utilities import get_size

FLAGS = tf.app.flags.FLAGS

seed = 99

xavier_init = tf.contrib.layers.xavier_initializer_conv2d(seed=5)
bias_init = tf.constant_initializer(.01, dtype=tf.float32)


def convolution(x, dims, filter=3, stride=1, name=None, verbose=False, batch_norm=False, mode="train"):
    dim_in, dim_out = dims
    shape = [filter, filter, dim_in, dim_out]
    input_dim = get_size(x)

    with tf.variable_scope(name) as scope:
        w = tf.get_variable('ConvWeights', shape=shape, dtype=tf.float32, initializer=xavier_init)
        b = tf.get_variable('ConvBiases', shape=[dim_out], dtype=tf.float32, initializer=bias_init)
        x = tf.nn.conv2d(x, w, [1, stride, stride, 1], padding='SAME')
        is_training = mode in ['test', 'valid']

        # Batch norm
        if batch_norm and mode == "train":
            x = tf.contrib.layers.batch_norm(x, is_training=is_training, trainable=True)

        # Verbosity
        if verbose:
            print(scope.name)
            print("Input: {}".format(input_dim))
            print("Filter: {}".format(get_size(w)[:2]))
            print("Stride: {}".format(stride))
            print("Hidden units: {}".format(get_size(w)[2:]))
            print("Output: {}".format(get_size(x)))
            print()

        return tf.nn.relu(tf.nn.bias_add(x, b), name="Relu")


def affine(x, dim_out, name=None, is_last=False):
    size_arr = get_size(x)
    dim_in = np.prod(np.asarray(size_arr[1:]))
    dims = [dim_in, dim_out]

    with tf.variable_scope(name) as scope:
        x = tf.reshape(x, [-1, dim_in])
        w = tf.get_variable('DenseWeights', shape=dims, dtype=tf.float32, initializer=xavier_init)
        b = tf.get_variable('DenseBiases', shape=[dim_out], dtype=tf.float32, initializer=bias_init)
        fc = tf.add(tf.matmul(x, w), b)

        if not is_last:
            fc = tf.nn.relu(fc)
            return fc
        return fc


def dropout(x, keep_prob, name):
    return tf.nn.dropout(x, keep_prob, seed=seed, name=name)


def connect(x, dim_out, name=None):
    with tf.variable_scope(name) as scope:
        size_arr = get_size(x)
        dim_in = np.prod(np.asarray(size_arr[1:]))
        dims = [dim_in, dim_out]
        x = tf.reshape(x, [-1, dim_in])
        w = tf.get_variable('ProbWeights', shape=dims, dtype=tf.float32, initializer=xavier_init)
        b = tf.get_variable('ProbBiases', shape=[dim_out], dtype=tf.float32, initializer=bias_init)
        return tf.add(tf.matmul(x, w), b)
