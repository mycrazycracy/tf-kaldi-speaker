import tensorflow as tf
from collections import OrderedDict
import random
from six.moves import range
import numpy as np

def shape_list(x):
    """Return list of dims, statically where possible."""
    x = tf.convert_to_tensor(x)

    # If unknown rank, return dynamic shape
    if x.get_shape().dims is None:
        return tf.shape(x)

    static = x.get_shape().as_list()
    shape = tf.shape(x)

    ret = []
    for i in range(len(static)):
        dim = static[i]
        if dim is None:
            dim = shape[i]
        ret.append(dim)
    return ret


def prelu(x, name="prelu", shared=False):
    """Parametric ReLU

    Args:
        x: the input tensor.
        name: the name of this operation.
        shared: use a shared alpha for all channels.
    """
    alpha_size = 1 if shared else x.get_shape()[-1]
    with tf.variable_scope(name):
        alpha = tf.get_variable('alpha', alpha_size,
                               initializer=tf.constant_initializer(0.01),
                               dtype=tf.float32)
        pos = tf.nn.relu(x)
        neg = alpha * (x - abs(x)) * 0.5
    return pos + neg


def l2_scaling(x, scaling_factor, epsilon=1e-12, name="l2_norm"):
    """Feature normalization before re-scaling along the last axis.
       This function is the similar to tf.nn.l2_normalize, but scale to a scaling_factor rather than 1.

    Args:
        x: The input features.
        scaling_factor: The scaling factor.
    :return: Normalized and re-scaled features.
    """
    with tf.name_scope(name):
        square_sum = tf.reduce_sum(tf.square(x), axis=-1, keep_dims=True)
        x_inv_norm = tf.rsqrt(tf.maximum(square_sum, epsilon)) * scaling_factor
        x_scale = x * x_inv_norm
    return x_scale


def pairwise_euc_distances(embeddings, squared=False):
    """Compute the 2D matrix of squared euclidean distances between all the embeddings.

    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        squared: || x1 - x2 ||^2 or || x1 - x2 ||
    :return: pairwise_square_distances: tensor of shape (batch_size, batch_size)
    """
    # Get the dot product between all embeddings
    # shape (batch_size, batch_size)
    dot_product = tf.matmul(embeddings, tf.transpose(embeddings))

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)
    square_norm = tf.diag_part(dot_product)

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = tf.expand_dims(square_norm, 1) - 2.0 * dot_product + tf.expand_dims(square_norm, 0)

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = tf.maximum(distances, 0.0)

    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = tf.to_float(tf.equal(distances, 0.0))
        distances = distances + mask * 1e-16
        distances = tf.sqrt(distances)
        distances = distances * (1.0 - mask)

    return distances


def pairwise_cos_similarity(embeddings, epsilon=1e-12):
    """Compute the 2D matrix of cosine similarity between all the embeddings.

    Args:
        embeddings: input tensors.
    :return: pairwise_cos: tensor of shape (batch_size, batch_size)
    """
    dot_product = tf.matmul(embeddings, tf.transpose(embeddings))
    square_sum = tf.reduce_sum(tf.square(embeddings), axis=-1, keep_dims=True)
    inv_norm = tf.rsqrt(tf.maximum(square_sum, epsilon))
    inv_norm = tf.matmul(inv_norm, tf.transpose(inv_norm))
    cos_similarity = tf.multiply(dot_product, inv_norm)
    cos_similarity = tf.clip_by_value(cos_similarity, -1, 1)
    return cos_similarity


def dense_bn_relu(features, num_nodes, endpoints, params, is_training=None, name="dense"):
    """Dense + bn + relu

    Args:
        features: The input features.
        num_nodes: The number of the nodes in this layer.
        endpoints: The endpoitns.
        params: Parameters.
        is_training:
        name:
    :return: The output of the layer. The endpoints also contains the intermediate outputs of this layer.
    """
    relu = tf.nn.relu
    if "network_relu_type" in params.dict:
        if params.network_relu_type == "prelu":
            relu = prelu
        if params.network_relu_type == "lrelu":
            relu = tf.nn.leaky_relu

    with tf.variable_scope(name):
        features = tf.layers.dense(features,
                                   num_nodes,
                                   activation=None,
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(params.weight_l2_regularizer),
                                   name="%s_dense" % name)
        endpoints["%s_dense" % name] = features
        features = tf.layers.batch_normalization(features,
                                                 momentum=params.batchnorm_momentum,
                                                 training=is_training,
                                                 name="%s_bn" % name)
        endpoints["%s_bn" % name] = features
        features = relu(features, name='%s_relu' % name)
        endpoints["%s_relu" % name] = features
    return features


def dense(features, num_nodes, endpoints, params, is_training=None, name="dense"):
    """Dense connected layer (affine)

    Args:
        features: The input features.
        num_nodes: The number of the nodes in this layer.
        endpoints: The endpoitns.
        params: Parameters.
        is_training:
        name:
    :return: The output of the layer. The endpoints also contains the intermediate outputs of this layer.
    """
    with tf.variable_scope(name):
        features = tf.layers.dense(features,
                                   num_nodes,
                                   activation=None,
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(params.weight_l2_regularizer),
                                   name="%s_dense" % name)
        endpoints["%s_dense" % name] = features
    return features


def dense_relu(features, num_nodes, endpoints, params, is_training=None, name="dense"):
    """Dense connected layer (affine+relu)

    Args:
        features: The input features.
        num_nodes: The number of the nodes in this layer.
        endpoints: The endpoitns.
        params: Parameters.
        is_training:
        name:
    :return: The output of the layer. The endpoints also contains the intermediate outputs of this layer.
    """
    relu = tf.nn.relu
    if "network_relu_type" in params.dict:
        if params.network_relu_type == "prelu":
            relu = prelu
        if params.network_relu_type == "lrelu":
            relu = tf.nn.leaky_relu

    with tf.variable_scope(name):
        features = tf.layers.dense(features,
                                   num_nodes,
                                   activation=None,
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(params.weight_l2_regularizer),
                                   name="%s_dense" % name)
        endpoints["%s_dense" % name] = features
        features = relu(features, name='%s_relu' % name)
        endpoints["%s_relu" % name] = features
    return features


def dense_tanh(features, num_nodes, endpoints, params, is_training=None, name="dense"):
    """Dense connected layer (affine + tanh)

    Args:
        features: The input features.
        num_nodes: The number of the nodes in this layer.
        endpoints: The endpoitns.
        params: Parameters.
        is_training:
        name:
    :return: The output of the layer. The endpoints also contains the intermediate outputs of this layer.
    """
    with tf.variable_scope(name):
        features = tf.layers.dense(features,
                                   num_nodes,
                                   activation=None,
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(params.weight_l2_regularizer),
                                   name="%s_dense" % name)
        endpoints["%s_dense" % name] = features
        features = tf.nn.tanh(features, name='%s_tanh' % name)
        endpoints["%s_tanh" % name] = features
    return features


def split_last_dimension(x, n):
    """Reshape x so that the last dimension becomes two dimensions.
    The first of these two dimensions is n.

    Args:
        x: a Tensor with shape [..., m]
        n: an integer.

    Returns:
        a Tensor with shape [..., n, m/n]
    """
    x_shape = shape_list(x)
    m = x_shape[-1]
    if isinstance(m, int) and isinstance(n, int):
        assert m % n == 0
    return tf.reshape(x, x_shape[:-1] + [n, m // n])


def split_heads(x, num_heads):
    """Split channels (dimension 2) into multiple heads (becomes dimension 1).

    Args:
        x: a Tensor with shape [batch, length, channels]
        num_heads: an integer
    Returns:
        a Tensor with shape [batch, num_heads, length, channels / num_heads]
    """
    return tf.transpose(split_last_dimension(x, num_heads), [0, 2, 1, 3])


def combine_last_two_dimensions(x):
    """Reshape x so that the last two dimension become one.

    Args:
        x: a Tensor with shape [..., a, b]
    Returns:
        a Tensor with shape [..., ab]
    """
    x_shape = shape_list(x)
    return tf.reshape(x, x_shape[:-2] + [x_shape[-2] * x_shape[-1]])
