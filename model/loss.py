import tensorflow as tf
import numpy as np
from model.common import shape_list, pairwise_euc_distances, pairwise_cos_similarity
from six.moves import range
from model.common import l2_scaling
from collections import OrderedDict


def softmax(features, labels, num_outputs, params, is_training=None, reuse_variables=None, name="softmax"):
    """Vanilla softmax loss.

    Args:
        features: A tensor with shape [batch, dim].
        labels: A tensor with shape [batch].
        num_outputs: The number of classes.
        params: params.weight_l2_regularizer used for L2 regularization.
        is_training: Not used in this case
        reuse_variables: Share the created variables or not.
        name:
    :return: A scalar tensor which is the loss value.
    """
    assert len(shape_list(features)) == len(shape_list(labels)) + 1

    endpoints = OrderedDict()
    tf.logging.info("Softmax is used.")
    weight_l2_regularizer = params.weight_l2_regularizer
    if "output_weight_l2_regularizer" in params.dict:
        weight_l2_regularizer = params.output_weight_l2_regularizer
    with tf.variable_scope(name, reuse=reuse_variables):
        logits = tf.layers.dense(features,
                                   num_outputs,
                                   activation=None,
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_l2_regularizer),
                                   name="output")
        loss = tf.losses.sparse_softmax_cross_entropy(labels, logits)
        endpoints["logits"] = logits
        endpoints["labels"] = labels
        tf.summary.scalar("softmax_loss", loss)

    if 'aux_loss_func' in params.dict:
        loss_aux, endpoints_aux = aux_loss_func(features, labels, num_outputs, params,
                                                is_training=is_training,
                                                reuse_variables=reuse_variables,
                                                name=name)
        loss += loss_aux
        endpoints.update(endpoints_aux)

    return loss, endpoints


def asoftmax(features, labels, num_outputs, params, is_training=None, reuse_variables=None, name="softmax"):
    """Compute angular softmax.
    SphereFace: Deep Hypersphere Embedding for Face Recognition
    https://arxiv.org/abs/1704.08063

    Hint:
    For marginal softmax, the target logits will be significantly smaller than other logits in the beginning
    of the training, making the probability quite small. The cross entropy may encounter some numerical problem
    (prob --> 0, -xent --> inf). Due to this reason, applying margin after pretraining may be a good choice.
    After pretraining, the target logit will be larger than other logits, and the margin won't eliminate
    the probability in that case. So we apply annealing to all marginal softmax (asoftmax, amsoftmax, arcsoftmax, etc.).

    Args:
        features: A tensor with shape [batch, dim].
        labels: A tensor with shape [batch].
        num_outputs: The number of classes.
        params: params.weight_l2_regularizer: the L2 regularization.
                params.asoftmax_m: The m value.
                params.asoftmax_lambda_min,
                params.asoftmax_lambda_base,
                params.asoftmax_lambda_gamma,
                params.asoftmax_lambda_power,
                params.asoftmax_norm, params.asoftmax_s: If asoftmax_norm is True, asoftmax_s must be specified.
                                                         This means we normalize the length of the features, and do the
                                                         scaling on the cosine similarity.
                params.global_step:  All used to tune lambda.
        is_training: Not used in this case
        reuse_variables: Reuse variables.
        name:
    :return: The A-softmax loss.
    """
    assert len(shape_list(features)) == len(shape_list(labels)) + 1
    # Convert the parameters to float
    params.asoftmax_lambda_min = float(params.asoftmax_lambda_min)
    params.asoftmax_lambda_base = float(params.asoftmax_lambda_base)
    params.asoftmax_lambda_gamma = float(params.asoftmax_lambda_gamma)
    params.asoftmax_lambda_power = float(params.asoftmax_lambda_power)

    tf.logging.info("Angular softmax is used.")
    tf.logging.info("The margin in the angular softmax is %d" % params.asoftmax_m)

    endpoints = OrderedDict()
    weight_l2_regularizer = params.weight_l2_regularizer
    if "output_weight_l2_regularizer" in params.dict:
        weight_l2_regularizer = params.output_weight_l2_regularizer
    # We keep the name of the variable_scope as the same with softmax loss to enable fine-tuning.
    with tf.variable_scope(name, reuse=reuse_variables):
        # There are 1 variables in angular softmax: the weight matrix w.
        # The name of w is selected as the same with softmax to enable fine-tuning.
        w = tf.get_variable("output/kernel", [shape_list(features)[1], num_outputs], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer(),
                            regularizer=tf.contrib.layers.l2_regularizer(weight_l2_regularizer))
        params.dict["softmax_w"] = w
        w_norm = tf.nn.l2_normalize(w, dim=0)

        # If ||x|| is scaled, ||x|| = s
        # Then the logits is s*cos(theta), else ||x||cos(theta)
        logits = tf.matmul(features, w_norm)

        if params.asoftmax_m == 1:
            # logits = ||x|| * cos(theta) or s * cos(theta)
            loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
            endpoints["logits"] = logits
            endpoints["labels"] = labels
            return loss, endpoints

        ordinal = tf.to_int32(tf.range(shape_list(features)[0]))
        ordinal_labels = tf.stack([ordinal, labels], axis=1)
        sel_logits = tf.gather_nd(logits, ordinal_labels)

        # The angle between x and the target w_i.
        eps = 1e-12
        features_norm = tf.maximum(tf.norm(features, axis=1), eps)
        cos_theta_i = tf.div(sel_logits, features_norm)
        cos_theta_i = tf.clip_by_value(cos_theta_i, -1+eps, 1-eps)  # for numerical steady

        # Phi(theta, m) = (-1)^k * cos(2*theta) - 2 * k
        # Phi(theta_i, m) is a monotonical function about theta given m
        if params.asoftmax_m == 2:
            phi = 2 * tf.multiply(tf.sign(cos_theta_i), tf.square(cos_theta_i)) - 1
        elif params.asoftmax_m == 4:
            cos_th2 = tf.square(cos_theta_i)
            cos_th4 = tf.pow(cos_theta_i, 4)
            sign0 = tf.sign(cos_theta_i)
            sign3 = tf.multiply(tf.sign(2 * cos_th2 - 1), sign0)
            sign4 = 2 * sign0 + sign3 - 3
            phi = sign3 * (8 * cos_th4 - 8 * cos_th2 + 1) + sign4
        else:
            raise NotImplementedError("[ERROR] m=%d is not unsupported." % params.asoftmax_m)

        # logits = ||x||cos(m * theta)
        scaled_logits = tf.multiply(phi, features_norm)

        asoftmax_lambda = tf.maximum(params.asoftmax_lambda_min,
                                     params.asoftmax_lambda_base * (1.0 + params.asoftmax_lambda_gamma * tf.to_float(params.global_step)) ** (-params.asoftmax_lambda_power))
        fa = 1.0 / (1.0 + asoftmax_lambda)
        fs = 1.0 - fa
        logits_asoftmax = tf.add(logits,
                                    tf.scatter_nd(ordinal_labels,
                                                  tf.subtract(scaled_logits, sel_logits),
                                                  tf.shape(logits, out_type=tf.int32)))
        updated_logits = fs * logits + fa * logits_asoftmax

        tf.summary.scalar("asoftmax_lambda", asoftmax_lambda)
        tf.summary.scalar("asoftmax_m", params.asoftmax_m)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=updated_logits)
        endpoints["logits"] = logits
        endpoints["labels"] = labels
        tf.summary.scalar("asoftmax_loss", loss)

    if 'aux_loss_func' in params.dict:
        loss_aux, endpoints_aux = aux_loss_func(features, labels, num_outputs, params,
                                                is_training=is_training,
                                                reuse_variables=reuse_variables,
                                                name=name)
        loss += loss_aux
        endpoints.update(endpoints_aux)

    return loss, endpoints


def additive_margin_softmax(features, labels, num_outputs, params, is_training=None, reuse_variables=None, name="softmax"):
    """Additive margin softmax.
    link: https://arxiv.org/abs/1801.05599
    ref: https://github.com/Joker316701882/Additive-Margin-Softmax
    Although it is claimed that there is no need to use annealing scheme, I also add lambda to tune the weight of
    the modified logits.

    Args:
        features: A tensor with shape [batch, dim].
        labels: A tensor with shape [batch].
        num_outputs: The number of classes.
        params: params.weight_l2_regularizer: the L2 regularization.
                params.amsoftmax_m: the margin. (0.25-0.5)
                params.amsoftmax_norm, params.amsoftmax_s: If amsoftmax_norm is True, amsoftmax_s must be specified.
                                                         This means we normalize the length of the features, and do the
                                                         scaling on the cosine similarity.
        is_training: Not used in this case.
        reuse_variables: Reuse variables.
        name:
    """
    assert len(shape_list(features)) == len(shape_list(labels)) + 1
    # Convert the parameters to float
    params.amsoftmax_lambda_min = float(params.amsoftmax_lambda_min)
    params.amsoftmax_lambda_base = float(params.amsoftmax_lambda_base)
    params.amsoftmax_lambda_gamma = float(params.amsoftmax_lambda_gamma)
    params.amsoftmax_lambda_power = float(params.amsoftmax_lambda_power)
    params.amsoftmax_m = float(params.amsoftmax_m)

    tf.logging.info("Additive margin softmax is used.")
    tf.logging.info("The margin in the additive margin softmax is %f" % params.amsoftmax_m)

    endpoints = OrderedDict()
    weight_l2_regularizer = params.weight_l2_regularizer
    if "output_weight_l2_regularizer" in params.dict:
        weight_l2_regularizer = params.output_weight_l2_regularizer
    with tf.variable_scope(name, reuse=reuse_variables):
        w = tf.get_variable("output/kernel", [shape_list(features)[1], num_outputs], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer(),
                            regularizer=tf.contrib.layers.l2_regularizer(weight_l2_regularizer))
        params.dict["softmax_w"] = w

        w_norm = tf.nn.l2_normalize(w, dim=0)
        logits = tf.matmul(features, w_norm)

        ordinal = tf.to_int32(tf.range(shape_list(features)[0]))
        ordinal_labels = tf.stack([ordinal, labels], axis=1)
        sel_logits = tf.gather_nd(logits, ordinal_labels)

        # The angle between x and the target w_i.
        eps = 1e-12
        features_norm = tf.maximum(tf.norm(features, axis=1), eps)
        cos_theta_i = tf.div(sel_logits, features_norm)
        cos_theta_i = tf.clip_by_value(cos_theta_i, -1+eps, 1-eps)  # for numerical steady
        phi_i = cos_theta_i - params.amsoftmax_m

        # logits = ||x||(cos(theta) - m)
        scaled_logits = tf.multiply(phi_i, features_norm)

        logits_amsoftmax = tf.add(logits,
                                  tf.scatter_nd(ordinal_labels,
                                                tf.subtract(scaled_logits, sel_logits),
                                                tf.shape(logits, out_type=tf.int32)))

        amsoftmax_lambda = tf.maximum(params.amsoftmax_lambda_min,
                                      params.amsoftmax_lambda_base * (1.0 + params.amsoftmax_lambda_gamma * tf.to_float(
                                          params.global_step)) ** (-params.amsoftmax_lambda_power))
        fa = 1.0 / (1.0 + amsoftmax_lambda)
        fs = 1.0 - fa
        updated_logits = fs * logits + fa * logits_amsoftmax

        tf.summary.scalar("amsoftmax_m", params.amsoftmax_m)
        tf.summary.scalar("amsoftmax_lambda", amsoftmax_lambda)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=updated_logits)
        endpoints["logits"] = logits
        endpoints["labels"] = labels
        tf.summary.scalar("amsoftmax_loss", loss)

    if 'aux_loss_func' in params.dict:
        loss_aux, endpoints_aux = aux_loss_func(features, labels, num_outputs, params,
                                                is_training=is_training,
                                                reuse_variables=reuse_variables,
                                                name=name)
        loss += loss_aux
        endpoints.update(endpoints_aux)

    return loss, endpoints


def additive_angular_margin_softmax(features, labels, num_outputs, params, is_training=None, reuse_variables=None, name="softmax"):
    """Additive angular margin softmax (ArcFace)
    link: https://arxiv.org/abs/1801.07698
    Annealing scheme is also added.

    Args:
        features: A tensor with shape [batch, dim].
        labels: A tensor with shape [batch].
        num_outputs: The number of classes.
        params: params.weight_l2_regularizer: the L2 regularization.
                params.arcsoftmax_m: the angular margin (0.4-0.55)
                params.arcsoftmax_norm, params.arcsoftmax_s: If arcsoftmax_norm is True, arcsoftmax_s must be specified.
                                                         This means we normalize the length of the features, and do the
                                                         scaling on the cosine similarity.
        is_training: Not used in this case.
        reuse_variables: Reuse variables.
        name:
    """
    assert len(shape_list(features)) == len(shape_list(labels)) + 1
    # Convert the parameters to float
    params.arcsoftmax_lambda_min = float(params.arcsoftmax_lambda_min)
    params.arcsoftmax_lambda_base = float(params.arcsoftmax_lambda_base)
    params.arcsoftmax_lambda_gamma = float(params.arcsoftmax_lambda_gamma)
    params.arcsoftmax_lambda_power = float(params.arcsoftmax_lambda_power)
    params.arcsoftmax_m = float(params.arcsoftmax_m)

    tf.logging.info("Additive angular margin softmax is used.")
    tf.logging.info("The margin in the additive angular margin softmax is %f" % params.arcsoftmax_m)

    endpoints = OrderedDict()
    weight_l2_regularizer = params.weight_l2_regularizer
    if "output_weight_l2_regularizer" in params.dict:
        weight_l2_regularizer = params.output_weight_l2_regularizer
    with tf.variable_scope(name, reuse=reuse_variables):
        w = tf.get_variable("output/kernel", [shape_list(features)[1], num_outputs], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer(),
                            regularizer=tf.contrib.layers.l2_regularizer(weight_l2_regularizer))
        params.dict["softmax_w"] = w

        w_norm = tf.nn.l2_normalize(w, dim=0)
        logits = tf.matmul(features, w_norm)

        ordinal = tf.to_int32(tf.range(shape_list(features)[0]))
        ordinal_labels = tf.stack([ordinal, labels], axis=1)
        sel_logits = tf.gather_nd(logits, ordinal_labels)

        # The angle between x and the target w_i.
        eps = 1e-12
        features_norm = tf.maximum(tf.norm(features, axis=1), eps)
        cos_theta_i = tf.div(sel_logits, features_norm)
        cos_theta_i = tf.clip_by_value(cos_theta_i, -1+eps, 1-eps)  # for numerical steady

        # Since 0 < theta < pi, sin(theta) > 0. sin(theta) = sqrt(1 - cos(theta)^2)
        # cos(theta + m) = cos(theta)cos(m) - sin(theta)sin(m)
        sin_theta_i_sq = 1 - tf.square(cos_theta_i)
        sin_theta_i = tf.sqrt(tf.maximum(sin_theta_i_sq, 1e-12))
        cos_theta_plus_m_i = cos_theta_i * tf.cos(params.arcsoftmax_m) - sin_theta_i * tf.sin(params.arcsoftmax_m)

        # Since theta \in [0, pi], theta + m > pi means cos(theta) < cos(pi - m)
        # If theta + m < pi, Phi(theta) = cos(theta + m).
        # If theta + m > pi, Phi(theta) = -cos(theta + m) - 2
        phi_i = tf.where(tf.greater(cos_theta_i, tf.cos(np.pi - params.arcsoftmax_m)),
                         cos_theta_plus_m_i,
                         -cos_theta_plus_m_i - 2)

        # logits = ||x||(cos(theta + m))
        scaled_logits = tf.multiply(phi_i, features_norm)

        logits_arcsoftmax = tf.add(logits,
                                   tf.scatter_nd(ordinal_labels,
                                                 tf.subtract(scaled_logits, sel_logits),
                                                 tf.shape(logits, out_type=tf.int32)))

        arcsoftmax_lambda = tf.maximum(params.arcsoftmax_lambda_min,
                                       params.arcsoftmax_lambda_base * (1.0 + params.arcsoftmax_lambda_gamma * tf.to_float(
                                           params.global_step)) ** (-params.arcsoftmax_lambda_power))
        fa = 1.0 / (1.0 + arcsoftmax_lambda)
        fs = 1.0 - fa
        updated_logits = fs * logits + fa * logits_arcsoftmax

        tf.summary.scalar("arcsoftmax_m", params.arcsoftmax_m)
        tf.summary.scalar("arcsoftmax_lambda", arcsoftmax_lambda)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=updated_logits)
        endpoints["logits"] = logits
        endpoints["labels"] = labels
        tf.summary.scalar("arcsoftmax_loss", loss)

    if 'aux_loss_func' in params.dict:
        loss_aux, endpoints_aux = aux_loss_func(features, labels, num_outputs, params,
                                                is_training=is_training,
                                                reuse_variables=reuse_variables,
                                                name=name)
        loss += loss_aux
        endpoints.update(endpoints_aux)

    return loss, endpoints


def semihard_triplet_loss(features, labels, num_outputs, params, is_training=None, reuse_variables=None, name="triplet_loss"):
    """ Computes the triplet loss with semi-hard negative mining.
        This is already implemented in TF 1.10.
        Use the TF implementation (https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/contrib/losses/python/metric_learning/metric_loss_ops.py)

        The loss encourages the positive distances (between a pair of embeddings with
        the same labels) to be smaller than the minimum negative distance among
        which are at least greater than the positive distance plus the margin constant
        (called semi-hard negative) in the mini-batch. If no such negative exists,
        uses the largest negative distance instead.
        See: https://arxiv.org/abs/1503.03832.

        This implementation uses tf.tile which make the matrix HUGE. It may fail when the batch size is too large,

    Args:
        features: A tensor with shape [batch, dim]. The L2 normalization should be applied before the loss computation.
        labels: A tensor with shape [batch].
        num_outputs: Not used in this case.
        params: params.margin: the margin of the samples.
                params.triplet_loss_squared:
        is_training: Not used in this case
        reuse_variables: Not used.
        name:
    :return: The semi-hard triplet loss.
    """
    assert len(shape_list(features)) == len(shape_list(labels)) + 1
    endpoints = OrderedDict()
    tf.logging.info("Semi-hard triplet loss is used.")
    with tf.variable_scope(name, reuse=reuse_variables):
        # Reshape [batch_size] label tensor to a [batch_size, 1] label tensor.
        lshape = tf.shape(labels)
        assert lshape.shape == 1
        batch_size = lshape[0]
        labels = tf.reshape(labels, [lshape[0], 1])
        margin = tf.to_float(params.margin)

        # Build pairwise squared distance matrix.
        distances = pairwise_euc_distances(features, params.triplet_loss_squared)

        # Build pairwise binary adjacency matrix.
        adjacency = tf.equal(labels, tf.transpose(labels))
        # Invert so we can select negatives only.
        adjacency_not = tf.logical_not(adjacency)

        # Compute the mask.
        distances_tile = tf.tile(distances, [batch_size, 1])

        # mask: valid negative, f_ap < f_an
        # For i-th block, m(x, y) = 1 when label(x) != label(y) and d_xy  > d_xi
        # The i-th block is for triplet (x, i, y)
        mask = tf.logical_and(
            tf.tile(adjacency_not, [batch_size, 1]),
            tf.greater(
                distances_tile, tf.reshape(
                    tf.transpose(distances), [-1, 1])))

        # Before reshape:
        # In i-th block, mask_final(x) means for triplet (x, i, y) exists a negative y that d_xy > d_xi
        # After reshape:
        # mask_final(i, x) means triplet (x, i, y) exists a negative y that d_xy > d_xi
        # After transpose:
        # mask_final(x, i) means triplet (x, i, y) exists a negative y that d_xy > d_xi
        mask_final = tf.reshape(
            tf.greater(
                tf.reduce_sum(
                    tf.cast(mask, dtype=tf.float32), 1, keep_dims=True),
                0.0), [batch_size, batch_size])
        mask_final = tf.transpose(mask_final)

        adjacency_not = tf.cast(adjacency_not, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)

        def _masked_maximum(data, mask, dim=1):
            """Computes the axis wise maximum over chosen elements.
            Args:
                data: 2-D float `Tensor` of size [n, m].
                mask: 2-D Boolean `Tensor` of size [n, m].
                dim: The dimension over which to compute the maximum.
            Returns:
                masked_maximums: N-D `Tensor`.
                  The maximized dimension is of size 1 after the operation.
            """
            axis_minimums = tf.reduce_min(data, dim, keep_dims=True)
            masked_maximums = tf.reduce_max(tf.multiply(data - axis_minimums, mask), dim,
                                            keep_dims=True) + axis_minimums
            return masked_maximums

        def _masked_minimum(data, mask, dim=1):
            """Computes the axis wise minimum over chosen elements.
            Args:
                data: 2-D float `Tensor` of size [n, m].
                mask: 2-D Boolean `Tensor` of size [n, m].
                dim: The dimension over which to compute the minimum.
            Returns:
                masked_minimums: N-D `Tensor`.
                  The minimized dimension is of size 1 after the operation.
            """
            axis_maximums = tf.reduce_max(data, dim, keep_dims=True)
            masked_minimums = tf.reduce_min(tf.multiply(data - axis_maximums, mask), dim,
                                            keep_dims=True) + axis_maximums
            return masked_minimums

        # negatives_outside: smallest D_an where D_an > D_ap.
        # negatives_outside(i, x) is the minimum d_xy when y is negative and d_xy > d_xi
        # After transpose: negatives_outside(x, i) means the minimum d_xy of a valid semi-hard negative for triplet (x, i, y)
        negatives_outside = tf.reshape(
            _masked_minimum(distances_tile, mask), [batch_size, batch_size])
        negatives_outside = tf.transpose(negatives_outside)

        # negatives_inside: largest D_an.
        # For x, negatives_inside(x) is the maximum d_xy for a valid negative
        negatives_inside = tf.tile(
            _masked_maximum(distances, adjacency_not), [1, batch_size])

        # If (x, i) has no semi-hard negative, use the negative y with a maximum d_xy
        semi_hard_negatives = tf.where(
            mask_final, negatives_outside, negatives_inside)

        # margin + d_xi - min(d_xy), y is a valid semi-hard negative for triplet (x, i, y)
        loss_mat = tf.add(margin, distances - semi_hard_negatives)

        # Next, we should find valid positive (x, i)
        mask_positives = tf.cast(
            adjacency, dtype=tf.float32) - tf.diag(
            tf.ones([batch_size]))

        # In lifted-struct, the authors multiply 0.5 for upper triangular
        #   in semihard, they take all positive pairs except the diagonal.
        num_positives = tf.maximum(tf.reduce_sum(mask_positives), 1e-16)

        # compute loss on the valid triplet (x, i, y) with semi-hard negative (x, y)
        loss = tf.truediv(
            tf.reduce_sum(
                tf.maximum(
                    tf.multiply(loss_mat, mask_positives), 0.0)),
            num_positives,
            name='triplet_semihard_loss')
        endpoints["loss"] = loss
        endpoints["labels"] = labels
        tf.summary.scalar("triplet_loss", loss)
        return loss, endpoints


def angular_triplet_loss(features, labels, num_outputs, params, is_training=None, reuse_variables=None, name="angular_triplet_loss"):
    """ Computes the triplet loss.
        Online mining. Refer to https://omoindrot.github.io/triplet-loss

        Args:
            features: A tensor with shape [batch, dim].
            labels: A tensor with shape [batch].
            num_outputs: Not used in this case.
            params: params.margin: the margin of the samples.
                    params.triplet_type: "all", "hard".
                    params.loss_type: "asoftmax", "additive_margin_softmax", "additive_angular_margin_softmax"
            is_training: Not used in this case
            reuse_variables: Not used.
            name:
        :return: The triplet loss.
    """
    assert len(shape_list(features)) == len(shape_list(labels)) + 1
    assert params.triplet_type == "all" or params.triplet_type == "hard"
    assert params.loss_type in ["asoftmax", "additive_margin_softmax", "additive_angular_margin_softmax"]
    params.margin = float(params.margin)
    # Define the minimum positive similarity. loss = max(x, eps)
    eps = 1e-12

    tf.logging.info("Angular triplet is used.")
    tf.logging.info("The angular triple loss type is %s" % params.loss_type)
    tf.logging.info("Using %s samples." % params.triplet_type)
    tf.logging.info("The margin in the angular triplet loss is %f" % params.margin)

    endpoints = OrderedDict()
    with tf.variable_scope(name, reuse=reuse_variables):
        pairwise_cos = pairwise_cos_similarity(features)

        # The loss is d_n - d_p
        # asoftmax: d_p = Phi(m * theta), d_n = Phi(theta)
        # amsoftmax: d_p = cos(theta) - m, d_n = cos(theta)
        # arcsoftmax: d_p = cos(theta + m), d_n = cos(theta)
        def _get_positive_dist(pairwise_dist, loss_type, margin):
            if loss_type == "asoftmax":
                if int(margin) == 1:
                    pairwise_dist = pairwise_dist
                elif int(margin) == 2:
                    pairwise_dist = 2 * tf.multiply(tf.sign(pairwise_dist), tf.square(pairwise_dist)) - 1
                elif int(margin) == 4:
                    cos_th2 = tf.square(pairwise_dist)
                    cos_th4 = tf.pow(pairwise_dist, 4)
                    sign0 = tf.sign(pairwise_dist)
                    sign3 = tf.multiply(tf.sign(2 * cos_th2 - 1), sign0)
                    sign4 = 2 * sign0 + sign3 - 3
                    pairwise_dist = sign3 * (8 * cos_th4 - 8 * cos_th2 + 1) + sign4
                else:
                    raise NotImplementedError("[ERROR] m=%d is not unsupported if asoftmax is selected." % margin)
            elif loss_type == "additive_margin_softmax":
                pairwise_dist = pairwise_dist - margin
            else:
                # Convert to cos(theta+m)
                new_pairwise_dist = pairwise_dist * tf.cos(margin) - tf.sqrt(1 - pairwise_dist ** 2) * tf.sin(margin)
                pairwise_dist = tf.where(tf.less_equal(pairwise_dist, tf.cos(np.pi - margin)),
                                         -new_pairwise_dist - 2,
                                         new_pairwise_dist)
            return pairwise_dist

        def _get_negative_dist(pairwise_dist, loss_type):
            return pairwise_dist

        pairwise_positive_dist = _get_positive_dist(pairwise_cos, params.loss_type, params.margin)
        pairwise_negative_dist = _get_negative_dist(pairwise_cos, params.loss_type)

        def _get_triplet_mask(labels):
            indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
            indices_not_equal = tf.logical_not(indices_equal)
            i_not_equal_j = tf.expand_dims(indices_not_equal, 2)
            i_not_equal_k = tf.expand_dims(indices_not_equal, 1)
            j_not_equal_k = tf.expand_dims(indices_not_equal, 0)
            distinct_indices = tf.logical_and(tf.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)

            label_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
            i_equal_j = tf.expand_dims(label_equal, 2)
            i_equal_k = tf.expand_dims(label_equal, 1)
            valid_labels = tf.logical_and(i_equal_j, tf.logical_not(i_equal_k))

            mask = tf.logical_and(distinct_indices, valid_labels)
            return mask

        def _get_anchor_positive_triplet_mask(labels):
            indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
            indices_not_equal = tf.logical_not(indices_equal)
            labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
            mask = tf.logical_and(indices_not_equal, labels_equal)
            return mask

        def _get_anchor_negative_triplet_mask(labels):
            labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
            mask = tf.logical_not(labels_equal)
            return mask

        if params.triplet_type == "all":
            anchor_positive_dist = tf.expand_dims(pairwise_positive_dist, 2)
            anchor_negative_dist = tf.expand_dims(pairwise_negative_dist, 1)
            triplet_loss = anchor_negative_dist - anchor_positive_dist

            # Put zero in the invalid triplets
            mask = tf.to_float(_get_triplet_mask(labels))
            triplet_loss = tf.maximum(tf.multiply(mask, triplet_loss), 0.0)

            # Count the number of positive triplets (that takes effects)
            valid_triplets = tf.to_float(tf.greater(triplet_loss, eps))
            num_positive_triplets = tf.reduce_sum(valid_triplets)
            num_valid_triplets = tf.reduce_sum(mask)
            fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)
            tf.summary.scalar("fraction_positive_triplets", fraction_positive_triplets)
            tf.summary.scalar("total_triplets", num_valid_triplets)
            tf.summary.scalar("positive_triplets", num_positive_triplets)
            triplet_loss = tf.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-16)
        else:
            # The hardest sampling: the maximum d_n and the minimum d_p.
            # The value could be positive and negative.
            mask_anchor_positive = tf.to_float(_get_anchor_positive_triplet_mask(labels))
            max_anchor_positive_dist = tf.reduce_max(pairwise_positive_dist, axis=1, keep_dims=True)
            anchor_positive_dist = pairwise_positive_dist * mask_anchor_positive + max_anchor_positive_dist * (1.0 - mask_anchor_positive)
            hardest_positive_dist = tf.reduce_min(anchor_positive_dist, axis=1, keep_dims=True)

            mask_anchor_negative = tf.to_float(_get_anchor_negative_triplet_mask(labels))
            max_anchor_negative_dist = tf.reduce_min(pairwise_positive_dist, axis=1, keep_dims=True)
            anchor_negative_dist = pairwise_negative_dist * mask_anchor_negative + max_anchor_negative_dist * (1.0 - mask_anchor_negative)
            hardest_negative_dist = tf.reduce_max(anchor_negative_dist, axis=1, keep_dims=True)

            triplet_loss = tf.maximum(hardest_negative_dist - hardest_positive_dist, 0.0)
            triplet_loss = tf.reduce_mean(triplet_loss)

        tf.summary.scalar("triplet_margin", params.margin)
        endpoints["loss"] = triplet_loss
        endpoints["labels"] = labels
        tf.summary.scalar("triplet_loss", triplet_loss)
        return triplet_loss, endpoints


def e2e_valid_loss(features, labels, num_outputs, params, is_training=None, reuse_variables=None, name="valid_e2e_loss"):
    """End2end loss for valid set. Same with softmax generalized End2End loss.

    Warning: The function require the features are arrange according to speaker, like:
             [s1, s1, s1, s1, s2, s2, s2, s2, ..., sN, sN, sN, sN].
             It is the user's response to ensure the order. If it is not the case, the loss will be incorrect.

    Args:
        features: A tensor with shape [batch, dim] WITHOUT L2 normalization, where batch = num_speakers * num_segments
        labels: A tensor with shape [batch]. Not used in this case. But the features should be arranged carefully!
        num_outputs: Not used in this case
        params: params.num_speakers_per_batch and params.num_segments_per_speaker are very important.
                Make sure their values are consistent with the features.
                params.init_end2end_w, params.init_end2end_b are the initial values for w and b.
        is_training: Not used in this case
        reuse_variables: Share the created variables or not.
        name:
    :return: GE2E loss.
    """
    assert len(shape_list(features)) == len(shape_list(labels)) + 1
    assert "num_valid_speakers_per_batch" in params.dict and "num_valid_segments_per_speaker" in params.dict, \
        "Valid parameters should be set if E2E loss is selected"
    s = 20

    tf.logging.info("Softmax generalized e2e loss for valid set.")
    tf.logging.info("The scaling is set to 20.")

    endpoints = OrderedDict()
    with tf.variable_scope(name, reuse=reuse_variables):
        features = l2_scaling(features, 1)

        # The inputs contain N speakers and M segments per speaker.
        num_features = shape_list(features)[0]
        dim = shape_list(features)[1]
        num_speakers = params.num_valid_speakers_per_batch
        num_segments_per_speaker = params.num_valid_segments_per_speaker

        features_reshape = tf.reshape(features, shape=[num_speakers, num_segments_per_speaker, dim])
        # Centers for each speaker
        center = l2_scaling(tf.reduce_mean(features_reshape, axis=1), 1)
        center_ex = l2_scaling(tf.reshape(tf.reduce_sum(features_reshape, axis=1, keep_dims=True) - features_reshape,
                                          shape=[num_features, dim]), 1)

        # Similarity matrix
        similarity = tf.matmul(features, tf.transpose(center))
        # Special similarity that using the centers excluding the sample itself.
        similarity_ex = tf.reduce_sum(features * center_ex, axis=1)

        # Combined the two similarities
        # [0, 1, 2, ..., N-1] repeat num_features times
        label_ref = tf.tile(tf.expand_dims(tf.range(num_speakers), axis=0), [num_features, 1])
        # [0, 0, .., 0, 1, 1, ..., N-1, ..., N-1]
        label_new = tf.reshape(tf.tile(tf.expand_dims(tf.range(num_speakers), axis=1), [1, num_segments_per_speaker]),
                               [num_features, 1])
        # mask[i, j] == 1 when feature i belongs to class j
        mask = tf.to_float(tf.equal(label_new, label_ref))
        similarity = similarity * (1 - mask) + tf.expand_dims(similarity_ex, axis=1) * mask

        similarity = s * similarity
        index = np.zeros((num_speakers * num_segments_per_speaker, 2), dtype=np.int32)
        for i in range(num_speakers):
            for j in range(num_segments_per_speaker):
                index[i * num_segments_per_speaker + j, 0] = i * num_segments_per_speaker + j
                index[i * num_segments_per_speaker + j, 1] = i

        loss = tf.losses.sparse_softmax_cross_entropy(index[:, 1], similarity)
        endpoints["loss"] = loss
        endpoints["labels"] = labels
        return loss, endpoints


def generalized_angular_triplet_loss(features, labels, num_outputs, params, is_training=None, reuse_variables=None, name="softmax"):
    """Angular triplet loss with the center of different classes.

    The center is estimated using:
        1. learnable parameters
        2. moving average

    The hard samples mining:
        topn = 0: use all (only violate the margin)
        topn = 1: hardest
        topn = k: top-k hardest

        Args:
            features: A tensor with shape [batch, dim] WITHOUT L2 normalization, where batch = num_speakers * num_segments
            labels: A tensor with shape [batch]. Not used in this case. But the features should be arranged carefully!
            num_outputs: #classes
            params:
            is_training: Update the centers or not.
            reuse_variables: Share the created variables or not.
            name:
        :return: loss.
    """
    assert len(shape_list(features)) == len(shape_list(labels)) + 1
    assert params.triplet_center == "learnable" or params.triplet_center == "average"
    assert params.loss_compute == "raw" or params.loss_compute == "softplus"
    params.margin = float(params.margin)
    params.target_margin = float(params.target_margin)
    params.triplet_topn = int(params.triplet_topn)
    params.triplet_loss_weight = float(params.triplet_loss_weight)
    params.center_loss_weight = float(params.center_loss_weight)
    params.between_loss_weight = float(params.between_loss_weight)
    assert params.l2_loss_weight == 0.0, "The weight decay is applied by regularization term, not the loss!"

    tf.logging.info("Generalized angular triplet loss is used.")
    tf.logging.info("Use %s to compute the loss." % params.loss_compute)
    tf.logging.info("Margin is %f" % params.margin)

    endpoints = OrderedDict()
    weight_l2_regularizer = params.weight_l2_regularizer
    if "output_weight_l2_regularizer" in params.dict:
        weight_l2_regularizer = params.output_weight_l2_regularizer

    # Use the scope to reuse the parameters from other softmax-like loss
    with tf.variable_scope(name, reuse=reuse_variables):
        if params.triplet_center == "learnable":
            # The centers are considered as the parameters of the model.
            w = tf.get_variable("output/kernel", [shape_list(features)[1], num_outputs], dtype=tf.float32,
                                initializer=tf.contrib.layers.xavier_initializer(),
                                regularizer=tf.contrib.layers.l2_regularizer(weight_l2_regularizer))
        else:
            # Update the centers on-the-fly, similar to batch normalization.
            # The center is not trainable while is updated during the training.
            w = tf.get_variable("output/kernel", [shape_list(features)[1], num_outputs], dtype=tf.float32,
                                initializer=tf.contrib.layers.xavier_initializer(),
                                trainable=False)
        # Norm features
        features_norm = tf.nn.l2_normalize(features, dim=1)

        w_update = w
        if params.triplet_center == "average" and is_training:
            params.triplet_center_momentum = float(params.triplet_center_momentum)
            # Update the centers before the cosine if training is True
            decay = 1 - params.triplet_center_momentum

            # w_norm_first = tf.nn.l2_normalize(w, dim=0)
            # w_norm_first = tf.transpose(w_norm_first)
            # sel_w_norm = tf.gather(w_norm_first, labels)
            # update_delta = (sel_w_norm - features_norm) * decay
            # w_update = w.assign(tf.transpose(w_norm_first -
            #                                  tf.scatter_nd(tf.expand_dims(labels, axis=1), update_delta,
            #                                                tf.shape(w_norm_first, out_type=tf.int32))))

            w_first = tf.transpose(w)
            sel_w = tf.gather(w_first, labels)
            update_delta = (sel_w - features) * decay
            w_update = w.assign(tf.transpose(w_first -
                                             tf.scatter_nd(tf.expand_dims(labels, axis=1), update_delta,
                                                           tf.shape(w_first, out_type=tf.int32))))

            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, w_update)

        # Norm centers
        # Now the centers have been updated.
        w_norm = tf.nn.l2_normalize(w_update, dim=0)

        eps = 1e-12

        if params.loss_compute == "raw":
            dist = tf.reduce_sum(tf.square(tf.expand_dims(features_norm, axis=1) -
                                           tf.expand_dims(tf.transpose(w_norm), axis=0)), axis=2)
            ordinal = tf.to_int32(tf.range(shape_list(features)[0]))
            ordinal_labels = tf.stack([ordinal, labels], axis=1)
            target_dist = tf.gather_nd(dist, ordinal_labels)
            mask = tf.scatter_nd(ordinal_labels, tf.ones_like(target_dist, dtype=tf.float32),
                                 shape=tf.shape(dist))
            new_dist = dist * (1 - mask) + (tf.reduce_max(dist, axis=1, keep_dims=True) + dist) * mask

            # We only care about the `hardest` sample whose distance to the center it belongs is bad.
            target_mask = tf.to_float(tf.greater(target_dist, params.target_margin))

            if params.triplet_topn == 1:
                nontarget_dist = tf.reduce_min(new_dist, axis=1)
                triplet_loss = tf.maximum(params.margin + target_dist - nontarget_dist, 1e-16)
                triplet_loss = target_mask * triplet_loss
                triplet_loss = tf.reduce_sum(triplet_loss) / (tf.reduce_sum(tf.to_float(tf.greater(triplet_loss, eps))) + eps)
            elif params.triplet_topn == 0:
                triplet_loss = tf.maximum(params.margin + tf.expand_dims(target_dist, 1) - new_dist, 1e-16)
                triplet_loss = triplet_loss * (1 - mask)
                triplet_loss = tf.expand_dims(target_mask, 1) * triplet_loss
                triplet_loss = tf.reduce_sum(triplet_loss) / (tf.reduce_sum(tf.to_float(tf.greater(triplet_loss, eps))) + eps)
            else:
                nontarget_dist, _ = tf.nn.top_k(-1 * new_dist, params.triplet_topn, sorted=False)
                nontarget_dist = nontarget_dist * -1
                triplet_loss = tf.maximum(params.margin + tf.expand_dims(target_dist, 1) - nontarget_dist, 1e-16)
                triplet_loss = tf.expand_dims(target_mask, 1) * triplet_loss
                triplet_loss = tf.reduce_sum(triplet_loss) / (tf.reduce_sum(tf.to_float(tf.greater(triplet_loss, eps))) + eps)

            # Normalize with the valid samples
            center_loss = tf.reduce_sum(target_mask * target_dist) / (tf.reduce_sum(target_mask) + eps)
            # # Normalize with all the samples
            # center_loss = tf.reduce_sum(target_mask * target_dist) / tf.to_float(shape_list(target_dist)[0])

            # To save memory, use inner product rather than norm
            # between_loss = tf.reduce_sum(np.square((tf.expand_dims(tf.transpose(w_norm), axis=0) -
            #                                         tf.expand_dims(tf.transpose(w_norm), axis=1))), axis=2)
            # between_loss should be around 2 during training.
            between_loss = 2 - 2 * tf.matmul(tf.transpose(w_norm), w_norm)
            between_loss = -1 * tf.reduce_sum((1.0 - tf.eye(num_outputs, dtype=tf.float32)) * between_loss) / (num_outputs * (num_outputs - 1))
        else:
            raise NotImplementedError("Not implemented.")
            # cos_theta = tf.matmul(features_norm, w_norm)
            #
            # # Cosine with the target center
            # ordinal = tf.to_int32(tf.range(shape_list(features)[0]))
            # ordinal_labels = tf.stack([ordinal, labels], axis=1)
            # target_cos_theta = tf.gather_nd(cos_theta, ordinal_labels)
            #
            # # Find the negative
            # mask = tf.scatter_nd(ordinal_labels, tf.ones_like(target_cos_theta, dtype=tf.float32),
            #                      shape=tf.shape(cos_theta))
            # cos_theta_mask = cos_theta - 2 * mask
            #
            # if params.triplet_topn == 1:
            #     # Hardest sample
            #     nontarget_cos_theta = tf.reduce_max(cos_theta_mask, axis=1)
            #     margin_mask = tf.to_float(tf.greater(params.margin + nontarget_cos_theta - target_cos_theta, eps))
            #     if params.triplet_norm_hard:
            #         triplet_loss = tf.reduce_sum(tf.log(1 + tf.exp(params.margin + nontarget_cos_theta - target_cos_theta))) / (tf.reduce_sum(margin_mask) + eps)
            #     else:
            #         triplet_loss = tf.reduce_sum(
            #             tf.log(1 + tf.exp(params.margin + nontarget_cos_theta - target_cos_theta))) / tf.to_float(shape_list(features)[0])
            # elif params.triplet_topn == 0:
            #     # Choose hard using margin
            #     triplet_loss = tf.log(1 + tf.exp(params.margin + cos_theta - tf.expand_dims(target_cos_theta, 1)))
            #     margin_mask = tf.to_float(tf.greater(params.margin + cos_theta - tf.expand_dims(target_cos_theta, 1), eps))
            #     nontarget_mask = (1 - mask) * margin_mask
            #     if params.triplet_norm_hard:
            #         triplet_loss = tf.reduce_sum(nontarget_mask * triplet_loss) / (tf.reduce_sum(nontarget_mask) + 1e-16)
            #     else:
            #         triplet_loss = tf.reduce_sum(nontarget_mask * triplet_loss) / tf.to_float(shape_list(features)[0] * (num_outputs-1))
            # else:
            #     # Top n hardest samples
            #     nontarget_cos_theta, _ = tf.nn.top_k(cos_theta_mask, params.triplet_topn, sorted=False)
            #     margin_mask = tf.to_float(tf.greater(params.margin + nontarget_cos_theta - tf.expand_dims(target_cos_theta, 1), eps))
            #     if params.triplet_norm_hard:
            #         triplet_loss = tf.reduce_sum(tf.log(1 + tf.exp(params.margin + nontarget_cos_theta - tf.expand_dims(target_cos_theta, 1)))) / (tf.reduce_sum(margin_mask) + eps)
            #     else:
            #         triplet_loss = tf.reduce_sum(
            #             tf.log(1 + tf.exp(params.margin + nontarget_cos_theta - tf.expand_dims(target_cos_theta, 1)))) / tf.to_float(shape_list(features)[0] * params.triplet_topn)
            #
            #     # Within-class loss (center loss)
            #     center_loss = -1 * tf.reduce_mean(target_cos_theta)
            #
            #     # Between-class loss
            #     between_loss = tf.reduce_sum((1 - tf.eye(num_outputs, dtype=tf.float32)) *
            #                                  tf.matmul(tf.transpose(w_norm), w_norm)) / (num_outputs * (num_outputs - 1))

        # L2 loss
        weight_loss = tf.norm(w_update)

        tf.summary.scalar("triplet_loss", triplet_loss)
        tf.summary.scalar("center_loss", center_loss)
        tf.summary.scalar("between_loss", between_loss)
        tf.summary.scalar("weight_loss", weight_loss)

        loss = params.triplet_loss_weight * triplet_loss + params.center_loss_weight * center_loss + params.between_loss_weight * between_loss + params.l2_loss_weight * weight_loss

        endpoints["average_centers"] = w
        endpoints["triplet_loss"] = triplet_loss
        endpoints["center_loss"] = center_loss
        endpoints["between_loss"] = between_loss
        endpoints["l2_loss"] = weight_loss
        tf.summary.scalar("gtriplet_loss", loss)
        return loss, endpoints

# def ge2e_loss(features, labels, num_outputs, params, is_training=None, reuse_variables=None):
#     """Generalized End2End loss.
#
#     Warning: The function require the features are arrange according to speaker, like:
#              [s1, s1, s1, s1, s2, s2, s2, s2, ..., sN, sN, sN, sN].
#              It is the user's response to ensure the order. If it is not the case, the loss will be incorrect.
#
#     Args:
#         features: A tensor with shape [batch, dim] WITHOUT L2 normalization, where batch = num_speakers * num_segments
#         labels: A tensor with shape [batch]. Not used in this case. But the features should be arranged carefully!
#         num_outputs: Not used in this case
#         params: params.num_speakers_per_batch and params.num_segments_per_speaker are very important.
#                 Make sure their values are consistent with the features.
#                 params.init_end2end_w, params.init_end2end_b are the initial values for w and b.
#         is_training: Not used in this case
#         reuse_variables: Share the created variables or not.
#     :return: GE2E loss.
#     """
#     with tf.variable_scope("ge2e", reuse=reuse_variables):
#         # There are 2 variables in the End2End loss
#         w = tf.get_variable("w", initializer=float(params.init_end2end_w), dtype=tf.float32)
#         b = tf.get_variable("b", initializer=float(params.init_end2end_b), dtype=tf.float32)
#
#         tf.summary.scalar("w", w)
#         tf.summary.scalar("b", b)
#
#         # The inputs contain N speakers and M segments per speaker.
#         num_features = shape_list(features)[0]
#         dim = shape_list(features)[1]
#         # num_segments_per_speaker = tf.reduce_sum(
#         #     tf.to_int32(tf.equal(tf.expand_dims(labels, 1), tf.expand_dims(labels, 0))), axis=1)[0]
#         # num_speakers = num_features / num_segments_per_speaker
#         num_segments_per_speaker = params.num_segments_per_speaker
#         num_speakers = params.num_speakers_per_batch
#         assert num_segments_per_speaker != 1
#
#         features_reshape = tf.reshape(features, shape=[num_speakers, num_segments_per_speaker, dim])
#
#         # Centers for each speaker
#         center = l2_normalize(tf.reduce_mean(features_reshape, axis=1))
#         # Centers that exclude each sample
#         center_ex = l2_normalize(tf.reshape(tf.reduce_sum(features_reshape, axis=1, keep_dims=True) - features_reshape,
#                                             shape=[num_features, dim])
#                                  / (num_segments_per_speaker - 1))
#
#         # Similarity matrix
#         similarity = tf.matmul(features, tf.transpose(center))
#         # Special similarity that using the centers excluding the sample itself.
#         similarity_ex = tf.reduce_sum(features * center_ex, axis=1)
#
#         # Combined the two similarities
#         # [0, 1, 2, ..., N-1] repeat num_features times
#         label_ref = tf.tile(tf.expand_dims(tf.range(num_speakers), axis=0), [num_features, 1])
#         # [0, 0, .., 0, 1, 1, ..., N-1, ..., N-1]
#         label_new = tf.reshape(tf.tile(tf.expand_dims(tf.range(num_speakers), axis=1), [1, num_segments_per_speaker]),
#                                [num_features, 1])
#         # mask[i, j] == 1 when feature i belongs to class j
#         mask = tf.to_float(tf.equal(label_new, label_ref))
#         similarity = similarity * (1 - mask) + tf.expand_dims(similarity_ex, axis=1) * mask
#
#         similarity = tf.abs(w) * similarity + b
#         index = np.zeros((num_speakers * num_segments_per_speaker, 2), dtype=np.int32)
#         for i in range(num_speakers):
#             for j in range(num_segments_per_speaker):
#                 index[i * num_segments_per_speaker + j, 0] = i * num_segments_per_speaker + j
#                 index[i * num_segments_per_speaker + j, 1] = i
#
#         if params.ge2e_loss_type == "softmax":
#             # loss = -tf.reduce_sum(similarity_true - tf.reduce_logsumexp(similarity, axis=1))
#             # Use tf sparse_softmax_cross_entropy to compute the softmax value more efficiently.
#             loss = tf.losses.sparse_softmax_cross_entropy(index[:, 1], similarity)
#         elif params.ge2e_loss_type == "contrastive":
#             similarity_true = tf.gather_nd(similarity, index)
#             similarity_neg = tf.reduce_max(tf.multiply(1-mask, tf.sigmoid(similarity)), axis=1)
#             loss = tf.truediv(tf.reduce_sum(1 - tf.sigmoid(similarity_true) + similarity_neg),
#                               tf.to_float(num_features))
#         else:
#             raise ValueError("The GE2E only support softmax and contrastive loss")
#
#         return loss


def aux_loss_func(features, labels, num_outputs, params, is_training=None, reuse_variables=None, name="auxiliary_loss"):
    """Auxiliary loss function.

        Args:
            features: A tensor with shape [batch, dim].
            labels: A tensor with shape [batch].
            num_outputs: The number of classes.
            params: params.weight_l2_regularizer used for L2 regularization.
            is_training: Not used in this case
            reuse_variables: Share the created variables or not.
            name:
        :return: A scalar tensor which is the loss value.
    """
    # params.aux_loss_func is a list that contains all the auxiliary losses
    loss = 0
    endpoints = OrderedDict()

    for loss_func in params.aux_loss_func:
        if loss_func == "ring_loss":
            ring_loss_init = float(params.ring_loss_init)
            ring_loss_lambda = float(params.ring_loss_lambda)
            tf.logging.info("Ring loss is added.")
            tf.logging.info("Ring loss lambda is %f" % ring_loss_lambda)
            with tf.variable_scope(name+'_ringloss', reuse=reuse_variables):
                r = tf.get_variable("r", dtype=tf.float32,
                                    initializer=ring_loss_init,
                                    trainable=True)
                ring_loss = ring_loss_lambda * tf.reduce_mean(tf.square(tf.norm(features, axis=1) - r))
                loss += ring_loss
                endpoints["ring_loss_r"] = r
                tf.summary.scalar("ring_loss_r", r)
                tf.summary.scalar("ring_loss", ring_loss)
        elif loss_func == "mhe_loss":
            mhe_lambda = float(params.mhe_lambda)
            # MHE need to load the softmax weights
            tf.logging.info("MHE is added.")
            tf.logging.info("MHE lambda is %f" % mhe_lambda)
            with tf.variable_scope(name, reuse=True):
                w = tf.get_variable("output/kernel")
                # Norm w first
                w_norm = tf.nn.l2_normalize(w, dim=0)
                trans_w = tf.transpose(w_norm)
                sel_w = tf.gather(trans_w, labels)
                # The inner product may be faster than the element-wise operations.
                mhe_loss = mhe_lambda * (1.0 / (tf.reduce_mean(2.0 - 2.0 * tf.matmul(sel_w, w_norm)) + 1e-6))
                # mhe_loss = mhe_lambda * (1.0 / (tf.reduce_mean(tf.reduce_sum(tf.square(tf.expand_dims(sel_w, axis=1) - tf.expand_dims(trans_w, axis=0)), axis=2)) + 1e-6))
                loss += mhe_loss
                tf.summary.scalar("mhe_loss", mhe_loss)
                endpoints["w"] = w
            pass
        else:
            raise NotImplementedError("Unsupported loss function %s" % loss_func)
    return loss, endpoints


if __name__ == "__main__":
    from misc.utils import ParamsPlain
    params = ParamsPlain()
    num_labels = 100
    num_speakers = 10
    num_segments_per_speaker = 1
    num_data = num_speakers * num_segments_per_speaker
    num_dim = 512
    labels = tf.placeholder(tf.int32, shape=[None], name="labels")
    embeddings = tf.placeholder(tf.float32, shape=[None, num_dim], name="embeddings")

    embeddings_val = np.random.rand(num_data, num_dim).astype(np.float32)
    labels_val = np.arange(0, num_data, dtype=np.int32)

    params.dict["aux_loss_func"] = ["ring_loss"]
    params.dict["ring_loss_init"] = 0.1
    params.dict["ring_loss_lambda"] = 0.01

    from model.test_utils import compute_amsoftmax, compute_ring_loss, compute_mhe
    loss, endpoints = aux_loss_func(embeddings, labels, num_labels, params)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        loss_tf = sess.run(loss, feed_dict={embeddings: embeddings_val,
                                            labels: labels_val})
        endpoints_val = sess.run(endpoints, feed_dict={embeddings: embeddings_val,
                                            labels: labels_val})
        loss_np = compute_ring_loss(embeddings_val, params, params.ring_loss_init)
        assert np.allclose(loss_tf, loss_np)

    params.dict["weight_l2_regularizer"] = 1e-5
    params.dict["amsoftmax_lambda_min"] = 10
    params.dict["amsoftmax_lambda_base"] = 1000
    params.dict["amsoftmax_lambda_gamma"] = 1
    params.dict["amsoftmax_lambda_power"] = 4
    params.dict["feature_norm"] = False
    params.dict["amsoftmax_m"] = 0.1
    params.dict["global_step"] = 1000
    params.dict["aux_loss_func"] = ["mhe_loss"]
    params.dict["mhe_lambda"] = 0.1

    loss, endpoints = additive_margin_softmax(embeddings, labels, num_labels, params, is_training=True, reuse_variables=tf.AUTO_REUSE)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        loss_tf, endpoints_val = sess.run([loss, endpoints], feed_dict={embeddings: embeddings_val,
                                                                 labels: labels_val})
        loss_np = compute_amsoftmax(embeddings_val, labels_val, params, endpoints_val["w"])
        loss_mhe_np = compute_mhe(labels_val, params, endpoints_val["w"])
        assert np.allclose(loss_tf, loss_np + loss_mhe_np)

    # params.dict["triplet_center"] = "learnable"
    # params.dict["loss_compute"] = "raw"
    # params.dict["triplet_topn"] = 5
    # params.dict["margin"] = 0.1
    # params.dict["target_margin"] = 1
    # params.dict["triplet_loss_weight"] = 1
    # params.dict["center_loss_weight"] = 1
    # params.dict["between_loss_weight"] = 0
    # params.dict["l2_loss_weight"] = 0
    # params.dict["weight_l2_regularizer"] = 1e-2
    #
    # import numpy as np
    # labels_val = np.arange(0, 10, dtype=np.int32)
    #
    # with tf.name_scope("loss") as scope:
    #     w_init = np.random.randn(num_dim, num_labels).astype(np.float32)
    #     w = tf.get_variable("softmax/output/kernel", initializer=w_init, dtype=tf.float32)
    #     loss, endpoints = generalized_angular_triplet_loss(embeddings, labels, num_labels, params, True, reuse_variables=True)
    #     with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope)):
    #         loss = tf.identity(loss)
    #
    # from model.test_utils import compute_generalized_triplet_loss
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     # No. 1
    #     embeddings_val = np.random.rand(num_data, num_dim).astype(np.float32)
    #     embeddings_val[0, :] = np.transpose(w_init[:, labels_val[0]])
    #     embeddings_val[1, :] = -1 * np.transpose(w_init[:, labels_val[1]])
    #     embeddings_val[2, :] = np.transpose(w_init[:, labels_val[1]])
    #     embeddings_val[3, :] = -1 * np.transpose(w_init[:, labels_val[1]])
    #     loss_np, w_update = compute_generalized_triplet_loss(embeddings_val, w_init, labels_val, params, num_labels)
    #     loss_tf, endpoints_val = sess.run([loss, endpoints], feed_dict={embeddings: embeddings_val,
    #                                                                       labels: labels_val})
    #
    #     import pdb
    #     pdb.set_trace()
    #     assert np.allclose(loss_np["loss"], loss_tf)
    #     assert np.allclose(loss_np["triplet_loss"], endpoints_val["triplet_loss"])
    #     assert np.allclose(loss_np["center_loss"], endpoints_val["center_loss"])
    #     assert np.allclose(loss_np["between_loss"], endpoints_val["between_loss"])
    #     assert np.allclose(endpoints_val["average_centers"], w_update)
    #
    #     # No. 2
    #     embeddings_val = np.random.rand(num_data, num_dim).astype(np.float32)
    #     embeddings_val[0, :] = np.transpose(w_update[:, labels_val[0]])
    #     loss_np, w_update = compute_generalized_triplet_loss(embeddings_val, w_update, labels_val, params, num_labels)
    #     loss_tf, endpoints_val = sess.run([loss, endpoints], feed_dict={embeddings: embeddings_val,
    #                                                                     labels: labels_val})
    #     assert np.allclose(loss_np["loss"], loss_tf)
    #     assert np.allclose(loss_np["triplet_loss"], endpoints_val["triplet_loss"])
    #     assert np.allclose(loss_np["center_loss"], endpoints_val["center_loss"])
    #     assert np.allclose(loss_np["between_loss"], endpoints_val["between_loss"])
    #     assert np.allclose(endpoints_val["average_centers"], w_update)
    #
    #     # No. 3
    #     embeddings_val = np.random.rand(num_data, num_dim).astype(np.float32)
    #     embeddings_val[0, :] = -1 * np.transpose(w_update[:, labels_val[0]])
    #     embeddings_val[1, :] = np.transpose(w_update[:, labels_val[0]])
    #     loss_np, w_update = compute_generalized_triplet_loss(embeddings_val, w_update, labels_val, params, num_labels)
    #     loss_tf, endpoints_val = sess.run([loss, endpoints], feed_dict={embeddings: embeddings_val,
    #                                                                     labels: labels_val})
    #     assert np.allclose(loss_np["loss"], loss_tf)
    #     assert np.allclose(loss_np["triplet_loss"], endpoints_val["triplet_loss"])
    #     assert np.allclose(loss_np["center_loss"], endpoints_val["center_loss"])
    #     assert np.allclose(loss_np["between_loss"], endpoints_val["between_loss"])
    #     assert np.allclose(endpoints_val["average_centers"], w_update)
