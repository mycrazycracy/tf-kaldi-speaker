import tensorflow as tf
from model.pooling import statistics_pooling, self_attention, ghost_vlad
from model.common import prelu, shape_list
from collections import OrderedDict
from six.moves import range


def tdnn(features, params, is_training=None, reuse_variables=None, aux_features=None):
    """Build a TDNN network.
    The structure is similar to Kaldi, while it uses bn+relu rather than relu+bn.
    And there is no dilation used, so it has more parameters than Kaldi x-vector.

    Args:
        features: A tensor with shape [batch, length, dim].
        params: Configuration loaded from a JSON.
        is_training: True if the network is used for training.
        reuse_variables: True if the network has been built and enable variable reuse.
        aux_features: Auxiliary features (e.g. linguistic features or bottleneck features).
    :return:
        features: The output of the last layer.
        endpoints: An OrderedDict containing output of every components. The outputs are in the order that they add to
                   the network. Thus it is convenient to split the network by a output name
    """
    # ReLU is a normal choice while other activation function is possible.
    relu = tf.nn.relu
    if "network_relu_type" in params.dict:
        if params.network_relu_type == "prelu":
            relu = prelu
        if params.network_relu_type == "lrelu":
            relu = tf.nn.leaky_relu

    endpoints = OrderedDict()
    with tf.variable_scope("tdnn", reuse=reuse_variables):
        # Convert to [b, 1, l, d]
        features = tf.expand_dims(features, 1)

        # Layer 1: [-2,-1,0,1,2] --> [b, 1, l-4, 512]
        # conv2d + batchnorm + relu
        features = tf.layers.conv2d(features,
                                512,
                                (1, 5),
                                activation=None,
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(params.weight_l2_regularizer),
                                name='tdnn1_conv')
        endpoints["tdnn1_conv"] = features
        features = tf.layers.batch_normalization(features,
                                                 momentum=params.batchnorm_momentum,
                                                 training=is_training,
                                                 name="tdnn1_bn")
        endpoints["tdnn1_bn"] = features
        features = relu(features, name='tdnn1_relu')
        endpoints["tdnn1_relu"] = features

        # Layer 2: [-2, -1, 0, 1, 2] --> [b ,1, l-4, 512]
        # conv2d + batchnorm + relu
        # This is slightly different with Kaldi which use dilation convolution
        features = tf.layers.conv2d(features,
                                    512,
                                    (1, 5),
                                    activation=None,
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(params.weight_l2_regularizer),
                                    name='tdnn2_conv')
        endpoints["tdnn2_conv"] = features
        features = tf.layers.batch_normalization(features,
                                                 momentum=params.batchnorm_momentum,
                                                 training=is_training,
                                                 name="tdnn2_bn")
        endpoints["tdnn2_bn"] = features
        features = relu(features, name='tdnn2_relu')
        endpoints["tdnn2_relu"] = features

        # Layer 3: [-3, -2, -1, 0, 1, 2, 3] --> [b, 1, l-6, 512]
        # conv2d + batchnorm + relu
        # Still, use a non-dilation one
        features = tf.layers.conv2d(features,
                                    512,
                                    (1, 7),
                                    activation=None,
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(params.weight_l2_regularizer),
                                    name='tdnn3_conv')
        endpoints["tdnn3_conv"] = features
        features = tf.layers.batch_normalization(features,
                                                 momentum=params.batchnorm_momentum,
                                                 training=is_training,
                                                 name="tdnn3_bn")
        endpoints["tdnn3_bn"] = features
        features = relu(features, name='tdnn3_relu')
        endpoints["tdnn3_relu"] = features

        # Convert to [b, l, 512]
        features = tf.squeeze(features, axis=1)
        # The output of the 3-rd layer can simply be rank 3.
        endpoints["tdnn3_relu"] = features

        # Layer 4: [b, l, 512] --> [b, l, 512]
        features = tf.layers.dense(features,
                                   512,
                                   activation=None,
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(params.weight_l2_regularizer),
                                   name="tdnn4_dense")
        endpoints["tdnn4_dense"] = features
        features = tf.layers.batch_normalization(features,
                                                 momentum=params.batchnorm_momentum,
                                                 training=is_training,
                                                 name="tdnn4_bn")
        endpoints["tdnn4_bn"] = features
        features = relu(features, name='tdnn4_relu')
        endpoints["tdnn4_relu"] = features

        # Layer 5: [b, l, x]
        if "num_nodes_pooling_layer" not in params.dict:
            # The default number of nodes before pooling
            params.dict["num_nodes_pooling_layer"] = 1500

        features = tf.layers.dense(features,
                                   params.num_nodes_pooling_layer,
                                   activation=None,
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(params.weight_l2_regularizer),
                                   name="tdnn5_dense")
        endpoints["tdnn5_dense"] = features
        features = tf.layers.batch_normalization(features,
                                                 momentum=params.batchnorm_momentum,
                                                 training=is_training,
                                                 name="tdnn5_bn")
        endpoints["tdnn5_bn"] = features
        features = relu(features, name='tdnn5_relu')
        endpoints["tdnn5_relu"] = features

        # Pooling layer
        # If you add new pooling layer, modify this code.
        # Statistics pooling
        # [b, l, 1500] --> [b, x]
        if params.pooling_type == "statistics_pooling":
            features = statistics_pooling(features, aux_features, endpoints, params, is_training)
        elif params.pooling_type == "self_attention":
            features = self_attention(features, aux_features, endpoints, params, is_training)
        elif params.pooling_type == "ghost_vlad":
            features = ghost_vlad(features, aux_features, endpoints, params, is_training)
        # elif params.pooling_type == "aux_attention":
        #     features = aux_attention(features, aux_features, endpoints, params, is_training)
        else:
            raise NotImplementedError("Not implement %s pooling" % params.pooling_type)
        endpoints['pooling'] = features

        # Utterance-level network
        # Layer 6: [b, 512]
        features = tf.layers.dense(features,
                                   512,
                                   activation=None,
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(params.weight_l2_regularizer),
                                   name='tdnn6_dense')
        endpoints['tdnn6_dense'] = features
        features = tf.layers.batch_normalization(features,
                                                 momentum=params.batchnorm_momentum,
                                                 training=is_training,
                                                 name="tdnn6_bn")
        endpoints["tdnn6_bn"] = features
        features = relu(features, name='tdnn6_relu')
        endpoints["tdnn6_relu"] = features

        # Layer 7: [b, x]
        if "num_nodes_last_layer" not in params.dict:
            # The default number of nodes in the last layer
            params.dict["num_nodes_last_layer"] = 512

        features = tf.layers.dense(features,
                                   params.num_nodes_last_layer,
                                   activation=None,
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(params.weight_l2_regularizer),
                                   name='tdnn7_dense')
        endpoints['tdnn7_dense'] = features

        if "last_layer_no_bn" not in params.dict:
            params.last_layer_no_bn = False

        if not params.last_layer_no_bn:
            features = tf.layers.batch_normalization(features,
                                                     momentum=params.batchnorm_momentum,
                                                     training=is_training,
                                                     name="tdnn7_bn")
            endpoints["tdnn7_bn"] = features

        if "last_layer_linear" not in params.dict:
            params.last_layer_linear = False

        if not params.last_layer_linear:
            # If the last layer is linear, no further activation is needed.
            features = relu(features, name='tdnn7_relu')
            endpoints["tdnn7_relu"] = features

    return features, endpoints


if __name__ == "__main__":
    num_labels = 10
    num_speakers = 10
    num_segments_per_speaker = 10
    num_data = 100
    num_length = 100
    num_dim = 512
    features = tf.placeholder(tf.float32, shape=[None, None, num_dim], name="features")
    labels = tf.placeholder(tf.int32, shape=[None], name="labels")
    embeddings = tf.placeholder(tf.float32, shape=[None, num_dim], name="embeddings")

    import numpy as np
    features_val = np.random.rand(num_data, num_length, num_dim).astype(np.float32)
    features_val[2, :, :] = 1e-8 * features_val[2, :, :]
    features_val[3, :, :] = 100 * features_val[3, :, :]
    labels_val = np.random.randint(0, num_labels, size=(num_data,)).astype(np.int32)

    from misc.utils import ParamsPlain
    params = ParamsPlain()
    params.dict["weight_l2_regularizer"] = 1e-5
    params.dict["batchnorm_momentum"] = 0.99
    params.dict["pooling_type"] = "statistics_pooling"
    params.dict["last_layer_linear"] = False
    params.dict["output_weight_l2_regularizer"] = 1e-4
    params.dict["network_relu_type"] = "prelu"

    # If the norm (s) is too large, after applying the margin, the softmax value would be extremely small
    params.dict["asoftmax_lambda_min"] = 10
    params.dict["asoftmax_lambda_base"] = 1000
    params.dict["asoftmax_lambda_gamma"] = 1
    params.dict["asoftmax_lambda_power"] = 4

    params.dict["amsoftmax_lambda_min"] = 10
    params.dict["amsoftmax_lambda_base"] = 1000
    params.dict["amsoftmax_lambda_gamma"] = 1
    params.dict["amsoftmax_lambda_power"] = 4

    params.dict["arcsoftmax_lambda_min"] = 10
    params.dict["arcsoftmax_lambda_base"] = 1000
    params.dict["arcsoftmax_lambda_gamma"] = 1
    params.dict["arcsoftmax_lambda_power"] = 4

    params.dict["feature_norm"] = True
    params.dict["feature_scaling_factor"] = 20

    from model.common import l2_scaling
    outputs, endpoints = tdnn(features, params, is_training=True, reuse_variables=False)
    outputs = l2_scaling(outputs, params.feature_scaling_factor)
    outputs_norm = tf.norm(outputs, axis=1)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        [outputs_val, outputs_norm_val] = sess.run([outputs, outputs_norm], feed_dict={features: features_val})
        assert np.allclose(np.sqrt(np.sum(outputs_val ** 2, axis=1)), params.feature_scaling_factor)
        assert np.allclose(outputs_norm_val, params.feature_scaling_factor)

    # Test loss functions
    # It only works on debug mode, since the loss is asked to output weights for our numpy computation.
    from model.loss import asoftmax, additive_margin_softmax, additive_angular_margin_softmax
    from model.test_utils import compute_asoftmax, compute_amsoftmax, compute_arcsoftmax

    params.dict["global_step"] = 1
    print("Asoftmax")
    for scaling in [True, False]:
        for m in [1, 2, 4]:
            print("m=%d" % m)
            params.dict["feature_norm"] = scaling
            params.dict["feature_scaling_factor"] = 0.1
            params.dict["asoftmax_m"] = m
            outputs = embeddings
            if params.dict["feature_norm"]:
                outputs = l2_scaling(outputs, params.feature_scaling_factor)
            loss = asoftmax(outputs, labels, num_labels, params, is_training=True, reuse_variables=tf.AUTO_REUSE)
            grads = tf.gradients(loss, embeddings)
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                w_val = sess.run(params.softmax_w)

                # very large embedding, very small embedding, angle close to 0 and pi
                # The norm cannot be too large, since the precision in softmax and log will screw things up.
                embeddings_val = np.random.rand(num_data, num_dim).astype(np.float32)
                embeddings_val[0, :] = w_val[:, labels_val[0]] + 1e-5
                embeddings_val[1, :] = -1 * w_val[:, labels_val[1]] + 1e-5
                embeddings_val[2, :] = 1e-4 * embeddings_val[2, :]
                embeddings_val[3, :] = 10 * embeddings_val[3, :]

                loss_val, grads_val = sess.run([loss, grads], feed_dict={embeddings: embeddings_val,
                                                                         labels: labels_val})
                loss_np = compute_asoftmax(embeddings_val, labels_val, params, w_val)
                assert not np.any(np.isnan(grads_val)), "Gradient should not be nan"
                assert np.allclose(loss_val, loss_np)

    params.dict["global_step"] = 1000
    print("Additive margin softmax")
    for scaling in [False, True]:
        for m in [0, 0.1, 0.5]:
            print("m=%f" % m)
            params.dict["feature_norm"] = scaling
            params.dict["feature_scaling_factor"] = 0.1
            params.dict["amsoftmax_m"] = m
            outputs = embeddings
            if params.dict["feature_norm"]:
                outputs = l2_scaling(outputs, params.feature_scaling_factor)
            loss = additive_margin_softmax(outputs, labels, num_labels, params, is_training=True,
                                           reuse_variables=tf.AUTO_REUSE)
            grads = tf.gradients(loss, embeddings)
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                w_val = sess.run(params.softmax_w)

                # very large embedding, very small embedding, angle close to 0 and pi
                # The norm cannot be too large, since the precision in softmax and log will screw things up.
                embeddings_val = np.random.rand(num_data, num_dim).astype(np.float32)
                embeddings_val[0, :] = w_val[:, labels_val[0]] + 1e-5
                embeddings_val[1, :] = -1 * w_val[:, labels_val[1]] + 1e-5
                embeddings_val[2, :] = 1e-4 * embeddings_val[2, :]

                loss_val, grads_val = sess.run([loss, grads], feed_dict={embeddings: embeddings_val,
                                                                         labels: labels_val})
                loss_np = compute_amsoftmax(embeddings_val, labels_val, params, w_val)
                assert not np.any(np.isnan(grads_val)), "Gradient should not be nan"
                assert np.allclose(loss_val, loss_np)

    print("Additive angular margin softmax")
    for scaling in [False, True]:
        for m in [0, 0.1, 0.5]:
            print("m=%f" % m)
            params.dict["feature_norm"] = scaling
            params.dict["feature_scaling_factor"] = 0.1
            params.dict["arcsoftmax_m"] = m
            outputs = embeddings
            if params.dict["feature_norm"]:
                outputs = l2_scaling(outputs, params.feature_scaling_factor)
            loss = additive_angular_margin_softmax(outputs, labels, num_labels, params, is_training=True, reuse_variables=tf.AUTO_REUSE)
            grads = tf.gradients(loss, embeddings)
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                w_val = sess.run(params.softmax_w)

                # very large embedding, very small embedding, angle close to 0 and pi
                # The norm cannot be too large, since the precision in softmax and log will screw things up.
                embeddings_val = np.random.rand(num_data, num_dim).astype(np.float32)
                embeddings_val[0, :] = w_val[:, labels_val[0]] + 1e-5
                embeddings_val[1, :] = -1 * w_val[:, labels_val[1]] + 1e-5
                embeddings_val[2, :] = 1e-4 * embeddings_val[2, :]

                loss_val, grads_val = sess.run([loss, grads], feed_dict={embeddings: embeddings_val,
                                                                         labels: labels_val})
                loss_np = compute_arcsoftmax(embeddings_val, labels_val, params, w_val)
                assert not np.any(np.isnan(grads_val)), "Gradient should not be nan"
                assert np.allclose(loss_val, loss_np)

    # # semihard sampling triplet loss
    # from model.loss import semihard_triplet_loss
    # params.dict["num_speakers_per_batch"] = num_speakers
    # params.dict["num_segments_per_speaker"] = num_segments_per_speaker
    # params.dict["margin"] = 0.2
    # for squared in [True, False]:
    #     params.dict["triplet_loss_squared"] = squared
    #     loss = semihard_triplet_loss(embeddings, labels, 10, params)
    #     grads = tf.gradients(loss, embeddings)
    #
    #     embeddings_val = np.random.rand(num_data, num_dim).astype(np.float32)
    #     labels_val = np.zeros(num_data, dtype=np.int32)
    #     for i in range(num_speakers):
    #         labels_val[i * num_segments_per_speaker:(i + 1) * num_segments_per_speaker] = i
    #     embeddings_val[-1, :] = embeddings_val[-2, :]
    #
    #     from model.test_utils import compute_triplet_loss
    #     loss_np = compute_triplet_loss(embeddings_val, labels_val, params.margin, params.triplet_loss_squared)
    #
    #     with tf.Session() as sess:
    #         sess.run(tf.global_variables_initializer())
    #         loss_val, grad_val = sess.run([loss, grads], feed_dict={embeddings: embeddings_val,
    #                                                                 labels: labels_val})
    #         assert not np.any(np.isnan(grad_val)), "Gradient should not be nan"
    #         assert np.allclose(loss_val, loss_np)

    from model.common import pairwise_cos_similarity
    from model.loss import angular_triplet_loss
    from model.test_utils import pairwise_cos_similarity_np, asoftmax_angular_triplet_loss
    from model.test_utils import amsoftmax_angular_triplet_loss, arcsoftmax_angular_triplet_loss

    embeddings_val = np.random.rand(num_data, num_dim).astype(np.float32)
    embeddings_val[1, :] = embeddings_val[0, :]
    embeddings_val[2, :] = -embeddings_val[0, :]
    labels_val = np.zeros(num_data, dtype=np.int32)
    for i in range(num_speakers):
        labels_val[i * num_segments_per_speaker:(i + 1) * num_segments_per_speaker] = i
    features = l2_scaling(embeddings, 1.0)
    cos_sim = pairwise_cos_similarity(features)
    with tf.Session() as sess:
        cos_sim_tf = sess.run(cos_sim, feed_dict={embeddings: embeddings_val})
        cos_sim_np = pairwise_cos_similarity_np(embeddings_val)
        assert np.allclose(cos_sim_np, cos_sim_tf)

    # The following test may fail due to the precision. Does not really matter.
    # asoftmax
    print("asoftmax triplet loss")
    for triplet_type in ["all", "hard"]:
        for margin in [1, 2, 4]:
            params.dict["margin"] = margin
            params.dict["triplet_type"] = triplet_type
            params.dict["loss_type"] = "asoftmax"
            loss = angular_triplet_loss(embeddings, labels, None, params)
            with tf.Session() as sess:
                loss_tf = sess.run(loss, feed_dict={embeddings: embeddings_val,
                                                    labels: labels_val})
                loss_np = asoftmax_angular_triplet_loss(embeddings_val, labels_val, margin, triplet_type)
                assert np.allclose(loss_np, loss_tf)

    # amsoftmax
    print("amsoftmax triplet loss")
    for triplet_type in ["all", "hard"]:
        margin = 0.1
        params.dict["margin"] = margin
        params.dict["triplet_type"] = triplet_type
        params.dict["loss_type"] = "additive_margin_softmax"
        loss = angular_triplet_loss(embeddings, labels, None, params)
        with tf.Session() as sess:
            loss_tf = sess.run(loss, feed_dict={embeddings: embeddings_val,
                                                labels: labels_val})
            loss_np = amsoftmax_angular_triplet_loss(embeddings_val, labels_val, margin, triplet_type)
            assert np.allclose(loss_np, loss_tf)

    # arcsoftmax
    print("arcsoftmax triplet loss")
    for triplet_type in ["all", "hard"]:
        margin = 0.3
        params.dict["margin"] = margin
        params.dict["triplet_type"] = triplet_type
        params.dict["loss_type"] = "additive_angular_margin_softmax"
        loss = angular_triplet_loss(embeddings, labels, None, params)
        with tf.Session() as sess:
            loss_tf = sess.run(loss, feed_dict={embeddings: embeddings_val,
                                                labels: labels_val})
            loss_np = arcsoftmax_angular_triplet_loss(embeddings_val, labels_val, margin, triplet_type)
            assert np.allclose(loss_np, loss_tf, rtol=1e-04)

    from model.loss import e2e_valid_loss
    from model.test_utils import compute_ge2e_loss
    print("E2e valid loss")
    params.dict["num_valid_speakers_per_batch"] = num_speakers
    params.dict["num_valid_segments_per_speaker"] = num_segments_per_speaker
    loss = e2e_valid_loss(embeddings, labels, None, params)
    with tf.Session() as sess:
        loss_tf = sess.run(loss, feed_dict={embeddings: embeddings_val,
                                                labels: labels_val})
        loss_np = compute_ge2e_loss(embeddings_val, labels_val, 20, 0, "softmax")
        assert np.allclose(loss_np, loss_tf)
