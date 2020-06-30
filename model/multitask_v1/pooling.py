import tensorflow as tf
from model.common import shape_list, dense_relu, dense_tanh, split_heads, combine_last_two_dimensions
import sys


VAR2STD_EPSILON = 1e-12


def statistics_pooling_v2(features, feat_length, endpoints, params, is_training):
    """Statistics pooling
    Note that we need to take care of the zeros in the variance since the sqrt on 0 will lead to NaN.

    Args:
        features: A tensor with shape [batch, length, dim].
        feat_length: The length of each utterance.
        endpoints: Outputs of different parts of the network.
        params:
        is_training:
    :return:
        Statistics pooling result [mean, stddev] with shape [batch, dim].
    """
    with tf.variable_scope("stat_pooling"):
        feat_shape = shape_list(features)
        frame_index = tf.tile(tf.expand_dims(tf.range(feat_shape[1]), axis=0), [feat_shape[0], 1])
        feat_length = tf.expand_dims(feat_length, axis=1)
        feat_length_new = tf.tile(feat_length, [1, feat_shape[1]])
        mask = tf.expand_dims(tf.to_float(tf.less(frame_index, feat_length_new)), axis=2)
        feat_length = tf.to_float(tf.expand_dims(feat_length, axis=2))
        mean = tf.reduce_sum(features * mask, axis=1, keep_dims=True) / (feat_length + 1e-16)
        variance = tf.reduce_sum(tf.squared_difference(features, mean) * mask, axis=1, keep_dims=True) / (feat_length + 1e-16)

        mean = tf.squeeze(mean, 1)
        variance = tf.squeeze(variance, 1)

        mask = tf.to_float(tf.less_equal(variance, VAR2STD_EPSILON))
        variance = (1.0 - mask) * variance + mask * VAR2STD_EPSILON
        stddev = tf.sqrt(variance)
        stat_pooling = tf.concat([mean, stddev], 1, name="concat")

    return stat_pooling


if __name__ == "__main__":
    num_labels = 10
    num_data = 100
    num_length = 1000
    num_dim = 1500
    features = tf.placeholder(tf.float32, shape=[None, None, num_dim], name="features")
    feat_length = tf.placeholder(tf.int32, shape=[None], name="feat_length")
    from collections import OrderedDict
    endpoints = OrderedDict()
    from misc.utils import ParamsPlain

    # Self-attention
    params = ParamsPlain()

    stat_pooling = statistics_pooling_v2(features, feat_length, endpoints, params, True)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        import numpy as np
        features_val = np.random.rand(num_data, num_length, num_dim).astype(np.float32)
        features_val[0, :, :] = 0
        length_val = np.random.randint(100, 1001, size=(num_data))
        stat_pooling_tf = sess.run(stat_pooling, feed_dict={features: features_val,
                                                            feat_length: length_val})

        def compute_stat_pooling(features, length):
            num_data, l, dim = features.shape
            assert num_data == length.shape[0]
            mean = np.zeros((num_data, dim))
            stddev = np.zeros((num_data, dim))
            for i in range(num_data):
                for j in range(length[i]):
                    mean[i, :] += features[i, j, :]
                    stddev[i, :] += np.square(features[i, j, :])
                mean[i, :] /= length[i]
                stddev[i, :] /= length[i]
                stddev[i, :] = np.sqrt(np.maximum(stddev[i, :] - np.square(mean[i, :]), 1e-12))
            return np.concatenate([mean, stddev], axis=1)

        stat_pooling_np = compute_stat_pooling(features_val, length_val)
        assert np.allclose(stat_pooling_tf, stat_pooling_np)
