# Build the speaker and phone networks.
# In this framework, they are both TDNN with different settings.
# The speaker network is a hard-coded TDNN and the phone network is specified by the parameters.
# Of course, the speaker network can be modified (e.g. to a larger network). Meanwhile, the parameters for the
# phone network should be modified as well so that the architecure is consistent with the speaker network.
# TODO: we can make the speaker network also controlled by config file which is not too difficult.

import tensorflow as tf
from model.multitask_v1.pooling import statistics_pooling_v2
from model.common import l2_scaling, shape_list, prelu


def build_speaker_encoder(features, phone_labels, feature_length, params, endpoints, reuse_variables, is_training=False):
    """Build encoder for speaker latent variable.
    Use the same tdnn network with x-vector.

    Args:
        features: the input features.
        phone_labels: the phone labels (i.e. alignment). will be used in the future.
        feature_length: the length of each feature.
        params: the parameters.
        endpoints: will be updated during building.
        reuse_variables: if true, reuse the existing variables.
        is_training: used in batchnorm
    :return: sampled_zs, mu_zs, logvar_zs
    """
    relu = tf.nn.relu
    if "network_relu_type" in params.dict:
        if params.network_relu_type == "prelu":
            relu = prelu
        if params.network_relu_type == "lrelu":
            relu = tf.nn.leaky_relu

    with tf.variable_scope("encoder", reuse=reuse_variables):
        # Layer 1: [-2,-1,0,1,2] --> [b, 1, l-4, 512]
        # conv2d + batchnorm + relu
        features = tf.layers.conv2d(features,
                                    512,
                                    (1, 5),
                                    activation=None,
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                        params.weight_l2_regularizer),
                                    name='conv1')
        endpoints["conv1"] = features
        features = tf.layers.batch_normalization(features,
                                                 momentum=params.batchnorm_momentum,
                                                 training=is_training,
                                                 name="bn1")
        endpoints["bn1"] = features
        features = relu(features, name='relu1')
        endpoints["relu1"] = features

        # Layer 2: [-2, -1, 0, 1, 2] --> [b ,1, l-4, 512]
        # conv2d + batchnorm + relu
        # This is slightly different with Kaldi which use dilation convolution
        features = tf.layers.conv2d(features,
                                    512,
                                    (1, 5),
                                    activation=None,
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                        params.weight_l2_regularizer),
                                    name='conv2')
        endpoints["conv2"] = features
        features = tf.layers.batch_normalization(features,
                                                 momentum=params.batchnorm_momentum,
                                                 training=is_training,
                                                 name="bn2")
        endpoints["bn2"] = features
        features = relu(features, name='relu2')
        endpoints["relu2"] = features

        # Layer 3: [-3, -2, -1, 0, 1, 2, 3] --> [b, 1, l-6, 512]
        # conv2d + batchnorm + relu
        # Still, use a non-dilation one
        features = tf.layers.conv2d(features,
                                    512,
                                    (1, 7),
                                    activation=None,
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                        params.weight_l2_regularizer),
                                    name='conv3')
        endpoints["conv3"] = features
        features = tf.layers.batch_normalization(features,
                                                 momentum=params.batchnorm_momentum,
                                                 training=is_training,
                                                 name="bn3")
        endpoints["bn3"] = features
        features = relu(features, name='relu3')
        endpoints["relu3"] = features

        # Convert to [b, l, 512]
        features = tf.squeeze(features, axis=1)
        # The output of the 3-rd layer can simply be rank 3.
        endpoints["relu3"] = features

        # Layer 4: [b, l, 512] --> [b, l, 512]
        features = tf.layers.dense(features,
                                   512,
                                   activation=None,
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                       params.weight_l2_regularizer),
                                   name="dense4")
        endpoints["dense4"] = features
        features = tf.layers.batch_normalization(features,
                                                 momentum=params.batchnorm_momentum,
                                                 training=is_training,
                                                 name="bn4")
        endpoints["bn4"] = features
        features = relu(features, name='relu4')
        endpoints["relu4"] = features

        # Layer 5: [b, l, x]
        if "num_nodes_pooling_layer" not in params.dict:
            # The default number of nodes before pooling
            params.dict["num_nodes_pooling_layer"] = 1500

        features = tf.layers.dense(features,
                                   params.num_nodes_pooling_layer,
                                   activation=None,
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                       params.weight_l2_regularizer),
                                   name="dense5")
        endpoints["dense5"] = features
        features = tf.layers.batch_normalization(features,
                                                 momentum=params.batchnorm_momentum,
                                                 training=is_training,
                                                 name="bn5")
        endpoints["bn5"] = features
        features = relu(features, name='relu5')
        endpoints["relu5"] = features

        # Here, we need to slice the feature since the original feature is expanded by the larger context between
        # the speaker and phone context. I make a hypothesis that the phone context will be larger.
        # So the speaker network need to slicing.
        if (params.speaker_left_context < params.phone_left_context and
                params.speaker_right_context < params.phone_right_context):
            features = features[:, params.phone_left_context - params.speaker_left_context:
                                   params.speaker_right_context - params.phone_right_context, :]
        else:
            raise NotImplementedError("The speake and phone context is not supported now.")

        # Make sure we've got the right feature
        with tf.control_dependencies([tf.assert_equal(shape_list(features)[1], shape_list(phone_labels)[1])]):
            # Pooling layer
            # The length of utterances may be different.
            # The original pooling use all the frames which is not appropriate for this case.
            # So we create a new function (I don't want to change the original one).
            if params.pooling_type == "statistics_pooling":
                features = statistics_pooling_v2(features, feature_length, endpoints, params, is_training)
            else:
                raise NotImplementedError("Not implement %s pooling" % params.pooling_type)
            endpoints['pooling'] = features

        # Utterance-level network
        # Layer 6: [b, 512]
        features = tf.layers.dense(features,
                                   512,
                                   activation=None,
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                       params.weight_l2_regularizer),
                                   name='dense6')
        endpoints['dense6'] = features
        features = tf.layers.batch_normalization(features,
                                                 momentum=params.batchnorm_momentum,
                                                 training=is_training,
                                                 name="bn6")
        endpoints["bn6"] = features
        features = relu(features, name='relu6')
        endpoints["relu6"] = features

        # Layer 7: [b, x]
        if "speaker_dim" not in params.dict:
            # The default number of nodes in the last layer
            params.dict["speaker_dim"] = 512

        # We need mean and logvar.
        mu = tf.layers.dense(features,
                             params.speaker_dim,
                             activation=None,
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(params.weight_l2_regularizer),
                             name="zs_dense")
        endpoints['zs_mu_dense'] = mu

        if "spk_last_layer_no_bn" not in params.dict:
            params.spk_last_layer_no_bn = False

        if not params.spk_last_layer_no_bn:
            mu = tf.layers.batch_normalization(mu,
                                               momentum=params.batchnorm_momentum,
                                               training=is_training,
                                               name="zs_bn")
            endpoints['zs_mu_bn'] = mu

        if "spk_last_layer_linear" not in params.dict:
            params.spk_last_layer_linear = False

        if not params.spk_last_layer_linear:
            mu = relu(mu, name="zs_mu_relu")
            endpoints['zs_mu_relu'] = mu

        # We do not compute logvar in this version.
        # Set logvar=0 ==> var=1
        logvar = 0

        # epsilon = tf.random_normal(tf.shape(mu), name='zs_epsilon')
        # sample = mu + tf.exp(0.5 * logvar) * epsilon
        sample = mu

    return sample, mu, logvar


def build_phone_encoder(features, speaker_labels, feature_length, params, endpoints, reuse_variables, is_training=False):
    """Build encoder for phone latent variable.
    Use the tdnn and share the same structure in the lower layers.

    Args:
        features: the input features.
        speaker_labels: the speaker labels (i.e. the speaker index). may be used in the future.
        feature_length: the length of each feature.
        params: the parameters.
        endpoints: will be updated during building.
        reuse_variables: if true, reuse the existing variables
        is_training: used in batchnorm.
    :return: sampled_zs, mu_zs, logvar_zs
    """
    relu = tf.nn.relu
    if "network_relu_type" in params.dict:
        if params.network_relu_type == "prelu":
            relu = prelu
        if params.network_relu_type == "lrelu":
            relu = tf.nn.leaky_relu

    # # This is moved to the model config file.
    # # Acoustic network params:
    # # Most share 4 layers with x-vector network.
    # # [-2,2], [-2,2], [-3,3], [0], [-4,0,4]
    # # The last fully-connected layer is appended as the phonetic embedding
    # layer_size = [512, 512, 512, 512, 512]
    # kernel_size = [5, 5, 7, 1, 3]
    # dilation_size = [1, 1, 1, 1, 4]

    num_layers = len(params.phone_kernel_size)
    layer_index = 0
    if params.num_shared_layers > 0:
        # We may share the lower layers of the two tasks.
        # Go through the shared layers between the speaker and phone networks.
        assert params.num_shared_layers < num_layers
        with tf.variable_scope("encoder", reuse=True):
            for i in range(params.num_shared_layers):
                if params.phone_kernel_size[layer_index] > 1:
                    if len(shape_list(features)) == 3:
                        # Add a dummy dim to support 2d conv
                        features = tf.expand_dims(features, axis=1)
                    features = tf.layers.conv2d(features,
                                                params.phone_layer_size[layer_index],
                                                (1, params.phone_kernel_size[layer_index]),
                                                activation=None,
                                                dilation_rate=(1, params.phone_dilation_size[layer_index]),
                                                kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                                    params.weight_l2_regularizer),
                                                name='conv%d' % (layer_index + 1))
                elif params.phone_kernel_size[layer_index] == 1:
                    if len(shape_list(features)) == 4:
                        # Remove a dummy dim to do dense layer
                        features = tf.squeeze(features, axis=1)
                    features = tf.layers.dense(features,
                                               params.phone_layer_size[layer_index],
                                               activation=None,
                                               kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                                   params.weight_l2_regularizer),
                                               name="dense%d" % (layer_index + 1))

                features = tf.layers.batch_normalization(features,
                                                         momentum=params.batchnorm_momentum,
                                                         training=is_training,
                                                         name="bn%d" % (layer_index + 1))
                features = relu(features, name='relu%d' % (layer_index + 1))
                layer_index += 1

    with tf.variable_scope("encoder_phone", reuse=reuse_variables):
        # In the unshared part, the endpoints should be updated.
        while layer_index < num_layers:
            if params.phone_kernel_size[layer_index] > 1:
                if len(shape_list(features)) == 3:
                    features = tf.expand_dims(features, axis=1)
                features = tf.layers.conv2d(features,
                                            params.phone_layer_size[layer_index],
                                            (1, params.phone_kernel_size[layer_index]),
                                            activation=None,
                                            dilation_rate=(1, params.phone_dilation_size[layer_index]),
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                                params.weight_l2_regularizer),
                                            name='phn_conv%d' % (layer_index + 1))
                endpoints["phn_conv%d" % (layer_index + 1)] = features
            elif params.phone_kernel_size[layer_index] == 1:
                if len(shape_list(features)) == 4:
                    features = tf.squeeze(features, axis=1)
                features = tf.layers.dense(features,
                                           params.phone_layer_size[layer_index],
                                           activation=None,
                                           kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                               params.weight_l2_regularizer),
                                           name="phn_dense%d" % (layer_index + 1))
                endpoints["phn_dense%d" % (layer_index + 1)] = features

            features = tf.layers.batch_normalization(features,
                                                     momentum=params.batchnorm_momentum,
                                                     training=is_training,
                                                     name="phn_bn%d" % (layer_index + 1))
            endpoints["phn_bn%d" % (layer_index + 1)] = features
            features = relu(features, name='phn_relu%d' % (layer_index + 1))
            endpoints["phn_relu%d" % (layer_index + 1)] = features
            layer_index += 1

        # The last layer
        if len(shape_list(features)) == 4:
            features = tf.squeeze(features, axis=1)

        # Similar with the speaker network, we may need to slice the feature due to the different context between
        # the speaker and phone network. At this moment, I just make a hypothesis that the phone context will be
        # larger which means there is no need to slice for the phone network
        if (params.speaker_left_context > params.phone_left_context and
                params.speaker_right_context > params.phone_right_context):
            raise NotImplementedError("The speake and phone context is not supported now.")
            # features = features[:, params.speaker_left_context - params.phone_left_context:
            #                        params.phone_right_context - params.speaker_right_context, :]

        # # We do not validate the length because this will introduce the alignment -- phn_labels, which
        # # is unnecessary when doing the phone inference.
        # with tf.control_dependencies([tf.assert_equal(shape_list(features)[1], shape_list(self.phn_labels)[1])]):
        #     features = tf.identity(features)

        if "phone_dim" not in params.dict:
            params.dict["phone_dim"] = 512
        mu = tf.layers.dense(features,
                             params.phone_dim,
                             activation=None,
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(params.weight_l2_regularizer),
                             name="zp_dense")
        endpoints['zp_mu_dense'] = mu
        mu = tf.layers.batch_normalization(mu,
                                           momentum=params.batchnorm_momentum,
                                           training=is_training,
                                           name="zp_bn")
        endpoints['zp_mu_bn'] = mu
        mu = relu(mu, name='zp_mu_relu')
        endpoints['zp_mu_relu'] = mu

        logvar = 0
        # epsilon = tf.random_normal(tf.shape(mu), name='zp_epsilon')
        # sample = mu + tf.exp(0.5 * logvar) * epsilon
        sample = mu

    return sample, mu, logvar
