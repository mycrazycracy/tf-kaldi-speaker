import tensorflow as tf
import os
import re
import sys
import time
import numpy as np
from model.common import shape_list
from six.moves import range

from model.multitask_v1.common import make_phone_masks
from model.loss import softmax, asoftmax, additive_margin_softmax, additive_angular_margin_softmax
from misc.utils import substring_in_list, activation_summaries, remove_params_prefix, add_dict_prefix
from collections import OrderedDict
from dataset.data_loader import DataOutOfRange
from dataset.multitask.data_loader_v2 import KaldiDataRandomQueueV2, KaldiDataSeqQueueV2
from model.multitask_v1.tdnn import build_speaker_encoder, build_phone_encoder


loss_network = {"softmax": softmax,
                "asoftmax": asoftmax,
                "additive_margin_softmax": additive_margin_softmax,
                "additive_angular_margin_softmax": additive_angular_margin_softmax}


class BaseMT(object):
    """Handle the training, validation and prediction

    Trainer is a simple class that deals with examples having feature-label structure.
    """

    def __init__(self, params, model_dir, dim, num_speakers=None, num_phones=None, single_cpu=False):
        """
        Args:
            params: Parameters loaded from JSON.
            model_dir: The model directory.
            dim: The dimension of the feature.
            num_speakers: The total number of speakers. Used in softmax-like network. Can be None.
            num_phones: The number of phones. Can be None.
            single_cpu: Run Tensorflow on one cpu. (default = False)
        """
        self.params = params

        if single_cpu:
            self.sess_config = tf.ConfigProto(intra_op_parallelism_threads=1,
                                              inter_op_parallelism_threads=1,
                                              device_count={'CPU': 1},
                                              allow_soft_placement=True)
        else:
            self.sess_config = tf.ConfigProto(allow_soft_placement=True)
        self.sess = tf.Session(config=self.sess_config)
        self.model = os.path.join(model_dir, "nnet")

        self.train_summary = None
        self.valid_summary = None

        self.spk_embeddings = None
        self.spk_logits = None
        self.phn_embeddings = None
        self.phn_posteriors = None
        self.phn_logits_subset = None

        self.endpoints = OrderedDict()

        self.optimizer = None
        self.loss = {}
        self.total_loss = None

        self.train_op = None
        self.train_ops = {}
        self.valid_ops = {}

        self.saver = None
        self.summary_writer = None
        self.valid_summary_writer = None

        self.is_built = False
        self.is_loaded = False

        # Required to build the train and valid network
        self.dim = dim
        self.num_speakers = num_speakers
        self.num_phones = num_phones

        self.features = tf.placeholder(tf.float32, shape=[None, None, dim], name="train_features")
        # self.recon = tf.placeholder(tf.float32, shape=[None, None, None, dim], name="recon_features")
        self.spk_labels = tf.placeholder(tf.int32, shape=[None, ], name="spk_labels")
        self.phn_labels = tf.placeholder(tf.int32, shape=[None, None], name="phn_labels")
        # self.vad = tf.placeholder(tf.float32, shape=[None, None], name="vad")  # Not used at this moment
        self.feat_length = tf.placeholder(tf.int32, shape=[None, ], name="lengths")
        # phn_masks is used to choose the valid phone examples
        # and follows the format of gather_nd. [[index1], [index2], ...]
        self.phn_masks = tf.placeholder(tf.int32, shape=[None, 2], name="phn_masks")
        self.global_step = tf.placeholder(tf.int64, name="global_step")
        self.params.dict["global_step"] = self.global_step

        # TODO: We have multiple networks. How many learning rates do we need?
        self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")

        # The discriminant loss network
        self.spk_loss_network = None
        self.phn_loss_network = None

        # The training statistics
        self.spk_training_count = np.zeros((num_speakers), dtype=np.int64)
        self.phn_training_count = np.zeros((num_phones), dtype=np.int64)
        self.phn_expansion_count = np.zeros((2), dtype=np.int64)



        # TODO: The context of the speaker and phone networks can be different.
        # TODO: We make a hypothesis that the context of the phone network will be larger than the speaker network.
        # TODO: If this is not true, the feature slicing should be differnet in the network building.
        # We need to expand the features to make #posteriors == #alignments
        # And convert to [b, 1, l, d]
        assert(self.params.phone_left_context > self.params.speaker_left_context and
               self.params.phone_right_context > self.params.speaker_right_context), \
            "The speaker context is expected to be smaller than the phone context (which may be not true). " \
            "If larger speaker context is used, change the feature expansion code."

        # The loaded feautre is already expanded. Nothing to do here.
        self.expand_features = tf.expand_dims(self.features, axis=1)
        return

    def reset(self):
        """Reset the graph so we can create new input pipeline or graph. (Or for other purposes)"""
        try:
            self.sess.close()
        except tf.errors.OpError:
            # Maybe the session is closed before
            pass
        tf.reset_default_graph()
        # The session should be created again after the graph is reset.
        self.sess = tf.Session(config=self.sess_config)
        # After the graph is reset, the flag should be set
        self.is_built = False
        self.is_loaded = False
        # After reset the graph, it is important to reset the seed.
        tf.set_random_seed(self.params.seed)

        # Reset some variables. The previous ones have become invalid due to the graph reset.
        self.saver = None
        self.summary_writer = None
        self.valid_summary_writer = None

    def close(self):
        """Close the session we opened."""
        try:
            self.sess.close()
        except tf.errors.OpError:
            pass

    def load(self):
        """Load the saved variables.

        If the variables have values, the current values will be changed to the saved ones
        :return The step of the saved model.
        """
        tf.logging.info("Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(self.model)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            step = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            self.saver.restore(self.sess, os.path.join(self.model, ckpt_name))
            tf.logging.info("Succeed to load checkpoint {}".format(ckpt_name))
        else:
            sys.exit("Failed to find a checkpoint in {}".format(self.model))
        self.is_loaded = True
        return step

    def save(self, step):
        """Save the model.

        Args:
            step: The global step.
        """
        self.saver.save(self.sess, os.path.join(self.model, "model"), global_step=step)

    def build(self, mode, noupdate_var_list=None):
        """ Build a network.

        Currently, I use placeholder in the graph and feed data during sess.run. So no need to parse
        features and labels.

        Args:
            mode: `train`, `valid` or `predict`.
            noupdate_var_list: In the fine-tuning, some variables are fixed. The list contains their names (or part of their names).
                               We use `noupdate` rather than `notrain` because some variables are not trainable, e.g.
                               the mean and var in the batchnorm layers.
        """
        assert(mode == "train" or mode == "valid" or mode == "predict")
        reuse_variables = True if self.is_built else None

        # Create a new path for prediction, since the training may build a tower the support multi-GPUs
        if mode == "predict":
            with tf.name_scope("predict") as scope:
                self.endpoints.clear()
                _, _, _ = build_speaker_encoder(self.expand_features,
                                                self.phn_labels,
                                                self.feat_length,
                                                self.params,
                                                self.endpoints,
                                                reuse_variables,
                                                is_training=False)
                _, mu_zp, _ = build_phone_encoder(self.expand_features,
                                                  self.spk_labels,
                                                  self.feat_length,
                                                  self.params,
                                                  self.endpoints,
                                                  reuse_variables,
                                                  is_training=False)

                # Sometimes we need the posteriors of the phones, so we have the build the softmax layer
                # If marginal angular softmax is used, correct the margin to compute the logits
                if self.params.phn_loss_type == "softmax":
                    pass
                elif self.params.phn_loss_type == "asoftmax":
                    self.params.phn_asoftmax_m = 1
                elif self.params.phn_loss_type == "additive_margin_softmax":
                    self.params.phn_amsoftmax_m = 0
                elif self.params.phn_loss_type == "additive_angular_margin_softmax":
                    self.params.phn_arcsoftmax_m = 0
                params = remove_params_prefix(self.params, "phn")

                dummy_labels = tf.zeros(shape_list(mu_zp)[:2], dtype=tf.int32)
                _, endpoints_phn_loss = loss_network[self.params.phn_loss_type](mu_zp, dummy_labels,
                                                                                self.num_phones, params,
                                                                                is_training=False,
                                                                                reuse_variables=reuse_variables,
                                                                                name="phn_softmax")
                self.endpoints.update(add_dict_prefix(endpoints_phn_loss, "phn"))

                # Try to use double float in the decoding process
                self.endpoints["phn_logits"] = tf.cast(self.endpoints["phn_logits"], dtype=tf.float64)

                # The posteriors node.
                self.endpoints["phn_post"] = tf.nn.softmax(self.endpoints["phn_logits"], axis=-1)
                # Compute a special log-posteriors node.
                self.endpoints["log-output"] = tf.nn.log_softmax(self.endpoints["phn_logits"], axis=-1)

                tf.logging.info("The parameters have changed. Do not train the network!")
                if self.saver is None:
                    self.saver = tf.train.Saver()
            return

        if mode == "valid":
            tf.logging.info("Building valid network...")
            with tf.name_scope("valid") as scope:
                self.endpoints.clear()
                _, mu_zs, _ = build_speaker_encoder(self.expand_features,
                                                    self.phn_labels,
                                                    self.feat_length,
                                                    self.params,
                                                    self.endpoints,
                                                    reuse_variables,
                                                    is_training=False)
                _, mu_zp, _ = build_phone_encoder(self.expand_features,
                                                  self.spk_labels,
                                                  self.feat_length,
                                                  self.params,
                                                  self.endpoints,
                                                  reuse_variables,
                                                  is_training=False)

                with tf.control_dependencies([tf.assert_equal(shape_list(mu_zp)[1], shape_list(self.phn_labels)[1])]):
                    mu_zp_subset = tf.gather_nd(mu_zp, self.phn_masks)

                phn_labels_subset = tf.gather_nd(self.phn_labels, self.phn_masks)

                # The speaker discriminator
                if self.params.spk_loss_type == "softmax":
                    pass
                elif self.params.spk_loss_type == "asoftmax":
                    train_spk_margin = self.params.spk_asoftmax_m
                    self.params.spk_asoftmax_m = 1
                elif self.params.spk_loss_type == "additive_margin_softmax":
                    train_spk_margin = self.params.spk_amsoftmax_m
                    self.params.spk_amsoftmax_m = 0
                elif self.params.spk_loss_type == "additive_angular_margin_softmax":
                    train_spk_margin = self.params.spk_arcsoftmax_m
                    self.params.spk_arcsoftmax_m = 0
                else:
                    pass
                params = remove_params_prefix(self.params, "spk")
                valid_spk_loss, endpoints_spk_loss = loss_network[self.params.spk_loss_type](mu_zs, self.spk_labels,
                                                                                             self.num_speakers, params,
                                                                                             is_training=False,
                                                                                             reuse_variables=reuse_variables,
                                                                                             name="spk_softmax")
                self.endpoints.update(add_dict_prefix(endpoints_spk_loss, "spk"))

                # Restore the original parameters
                if self.params.spk_loss_type == "softmax":
                    pass
                elif self.params.spk_loss_type == "asoftmax":
                    self.params.spk_asoftmax_m = train_spk_margin
                elif self.params.spk_loss_type == "additive_margin_softmax":
                    self.params.spk_amsoftmax_m = train_spk_margin
                elif self.params.spk_loss_type == "additive_angular_margin_softmax":
                    self.params.spk_arcsoftmax_m = train_spk_margin
                else:
                    pass

                # Then the phone discriminator
                if self.params.phn_loss_type == "softmax":
                    pass
                elif self.params.phn_loss_type == "asoftmax":
                    train_phn_margin = self.params.phn_asoftmax_m
                    self.params.phn_asoftmax_m = 1
                elif self.params.phn_loss_type == "additive_margin_softmax":
                    train_phn_margin = self.params.phn_amsoftmax_m
                    self.params.phn_amsoftmax_m = 0
                elif self.params.phn_loss_type == "additive_angular_margin_softmax":
                    train_phn_margin = self.params.phn_arcsoftmax_m
                    self.params.phn_arcsoftmax_m = 0
                else:
                    pass
                params = remove_params_prefix(self.params, "phn")
                valid_phn_loss, endpoints_phn_loss = loss_network[self.params.phn_loss_type](mu_zp_subset, phn_labels_subset,
                                                                                             self.num_phones, params,
                                                                                             is_training=False,
                                                                                             reuse_variables=reuse_variables,
                                                                                             name="phn_softmax")
                self.endpoints.update(add_dict_prefix(endpoints_phn_loss, "phn"))

                # Restore the original parameters
                if self.params.phn_loss_type == "softmax":
                    pass
                elif self.params.phn_loss_type == "asoftmax":
                    self.params.phn_asoftmax_m = train_phn_margin
                elif self.params.phn_loss_type == "additive_margin_softmax":
                    self.params.phn_amsoftmax_m = train_phn_margin
                elif self.params.phn_loss_type == "additive_angular_margin_softmax":
                    self.params.phn_arcsoftmax_m = train_phn_margin
                else:
                    pass

                # During validation, the embedding is the input of the loss layers
                self.spk_embeddings = mu_zs
                self.phn_embeddings = mu_zp
                self.spk_logits = self.endpoints["spk_logits"]
                self.phn_logits_subset = self.endpoints["phn_logits"]

                # TODO: define the loss
                valid_loss = self.params.spk_loss_weight * valid_spk_loss + self.params.phn_loss_weight * valid_phn_loss

                mean_valid_loss, mean_valid_loss_op = tf.metrics.mean(valid_loss)
                mean_valid_spk_loss, mean_valid_spk_loss_op = tf.metrics.mean(valid_spk_loss)
                mean_valid_spk_acc, mean_valid_spk_acc_op = tf.metrics.accuracy(labels=self.spk_labels,
                                                                                predictions=tf.argmax(self.spk_logits, axis=-1))
                mean_valid_phn_loss, mean_valid_phn_loss_op = tf.metrics.mean(valid_phn_loss)
                mean_valid_phn_acc, mean_valid_phn_acc_op = tf.metrics.accuracy(labels=phn_labels_subset,
                                                                                predictions=tf.argmax(self.phn_logits_subset, axis=-1))
                valid_ops = {"valid_loss": mean_valid_loss,
                             "valid_loss_op": mean_valid_loss_op,
                             "valid_spk_loss": mean_valid_spk_loss,
                             "valid_spk_loss_op": mean_valid_spk_loss_op,
                             "valid_spk_acc": mean_valid_spk_acc,
                             "valid_spk_acc_op": mean_valid_spk_acc_op,
                             "valid_phn_loss": mean_valid_phn_loss,
                             "valid_phn_loss_op": mean_valid_phn_loss_op,
                             "valid_phn_acc": mean_valid_phn_acc,
                             "valid_phn_acc_op": mean_valid_phn_acc_op}
                self.valid_ops.update(valid_ops)
                valid_loss_summary = tf.summary.scalar("loss", mean_valid_loss)
                valid_spk_loss_summary = tf.summary.scalar("speaker_loss", mean_valid_spk_loss)
                valid_spk_acc_summary = tf.summary.scalar("speaker_acc", mean_valid_spk_acc)
                valid_phn_loss_summary = tf.summary.scalar("phone_loss", mean_valid_phn_loss)
                valid_phn_acc_summary = tf.summary.scalar("phone_acc", mean_valid_phn_acc)
                self.valid_summary = tf.summary.merge([valid_loss_summary,
                                                       valid_spk_loss_summary,
                                                       valid_spk_acc_summary,
                                                       valid_phn_loss_summary,
                                                       valid_phn_acc_summary])
                if self.saver is None:
                    self.saver = tf.train.Saver(max_to_keep=self.params.keep_checkpoint_max)
                if self.valid_summary_writer is None:
                    self.valid_summary_writer = tf.summary.FileWriter(os.path.join(self.model, "eval"), self.sess.graph)
            return

        tf.logging.info("Building training network...")
        if "optimizer" not in self.params.dict:
            # The default optimizer is sgd
            self.params.dict["optimizer"] = "sgd"

        if self.params.optimizer == "sgd":
            if "momentum" in self.params.dict:
                sys.exit("Using sgd as the optimizer and you should not specify the momentum.")
            tf.logging.info("***** Using SGD as the optimizer.")
            opt = tf.train.GradientDescentOptimizer(self.learning_rate, name="optimizer")
        elif self.params.optimizer == "momentum":
            # SGD with momentum
            # It is also possible to use other optimizers, e.g. Adam.
            tf.logging.info("***** Using Momentum as the optimizer.")
            opt = tf.train.MomentumOptimizer(self.learning_rate, self.params.momentum, use_nesterov=self.params.use_nesterov, name="optimizer")
        elif self.params.optimizer == "adam":
            tf.logging.info("***** Using Adam as the optimizer.")
            opt = tf.train.AdamOptimizer(self.learning_rate, name="optimizer")
        else:
            sys.exit("Optimizer %s is not supported." % self.params.optimizer)
        self.optimizer = opt

        with tf.name_scope("train") as scope:
            self.endpoints.clear()
            _, mu_zs, _ = build_speaker_encoder(self.expand_features,
                                                self.phn_labels,
                                                self.feat_length,
                                                self.params,
                                                self.endpoints,
                                                reuse_variables,
                                                is_training=True)
            _, mu_zp, _ = build_phone_encoder(self.expand_features,
                                              self.spk_labels,
                                              self.feat_length,
                                              self.params,
                                              self.endpoints,
                                              reuse_variables,
                                              is_training=True)

            with tf.control_dependencies([tf.assert_equal(shape_list(mu_zp)[1], shape_list(self.phn_labels)[1])]):
                # The length of the outputs should be the same with the length of the alignments
                mu_zp_subset = tf.gather_nd(mu_zp, self.phn_masks)

            phn_labels_subset = tf.gather_nd(self.phn_labels, self.phn_masks)
            self.endpoints["mu_zp_subset"] = mu_zp_subset
            self.endpoints["phn_labels_subset"] = phn_labels_subset

            # TODO: we should use the sampled data or the mean?
            tf.logging.info("Speaker loss")
            params = remove_params_prefix(self.params, "spk")
            spk_disc_loss, endpoints_spk_loss = loss_network[self.params.spk_loss_type](mu_zs, self.spk_labels,
                                                                                        self.num_speakers, params,
                                                                                        is_training=True,
                                                                                        reuse_variables=reuse_variables,
                                                                                        name="spk_softmax")
            self.endpoints.update(add_dict_prefix(endpoints_spk_loss, "spk"))

            tf.logging.info("Phone loss")
            params = remove_params_prefix(self.params, "phn")
            phn_disc_loss, endpoints_phn_loss = loss_network[self.params.phn_loss_type](mu_zp_subset, phn_labels_subset,
                                                                                        self.num_phones, params,
                                                                                        is_training=True,
                                                                                        reuse_variables=reuse_variables,
                                                                                        name="phn_softmax")
            self.endpoints.update(add_dict_prefix(endpoints_phn_loss, "phn"))

            # TODO: Define the loss
            loss = self.params.spk_loss_weight * spk_disc_loss + self.params.phn_loss_weight * phn_disc_loss

            regularization_loss = tf.losses.get_regularization_loss()
            total_loss = loss + regularization_loss
            penalty_loss = tf.get_collection("PENALTY")
            if len(penalty_loss) != 0:
                penalty_loss = tf.reduce_sum(penalty_loss)
                total_loss += penalty_loss
                self.train_summary.append(tf.summary.scalar("penalty_term", penalty_loss))

            self.train_summary = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
            self.train_summary.append(tf.summary.scalar("loss", loss))
            self.train_summary.append(tf.summary.scalar("regularization_loss", regularization_loss))

            self.total_loss = total_loss
            self.train_summary.append(tf.summary.scalar("total_loss", total_loss))
            self.train_summary.append(tf.summary.scalar("learning_rate", self.learning_rate))

            # The gradient ops is inside the scope to support multi-gpus
            if noupdate_var_list is not None:
                old_batchnorm_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope)
                batchnorm_update_ops = []
                for op in old_batchnorm_update_ops:
                    if not substring_in_list(op.name, noupdate_var_list):
                        batchnorm_update_ops.append(op)
                        tf.logging.info("[Info] Update %s" % op.name)
                    else:
                        tf.logging.info("[Info] Op %s will not be executed" % op.name)
            else:
                batchnorm_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope)

            if noupdate_var_list is not None:
                variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                train_var_list = []

                for v in variables:
                    if not substring_in_list(v.name, noupdate_var_list):
                        train_var_list.append(v)
                        tf.logging.info("[Info] Train %s" % v.name)
                    else:
                        tf.logging.info("[Info] Var %s will not be updated" % v.name)
                grads = opt.compute_gradients(total_loss, var_list=train_var_list)
            else:
                grads = opt.compute_gradients(total_loss)

            # Once the model has been built (even for a tower), we set the flag
            self.is_built = True

        # There are some things we can do to the gradients, i.e. learning rate scaling.
        # Here we only apply gradient clipping
        if self.params.clip_gradient:
            grads, vars = zip(*grads)  # compute gradients of variables with respect to loss
            grads_clip, _ = tf.clip_by_global_norm(grads, self.params.clip_gradient_norm)  # l2 norm clipping

            # # we follow the instruction in ge2e paper to scale the learning rate for w and b
            # # Actually, I wonder that we can just simply set a large value for w (e.g. 20) and fix it.
            # if self.loss_type == "ge2e":
            #     # The parameters w and b must be the last variables in the gradients
            #     grads_clip = grads_clip[:-2] + [0.01 * grad for grad in grads_clip[-2:]]
            #     # Simply check the position of w and b
            #     for var in vars[-2:]:
            #         assert("w" in var.name or "b" in var.name)
            grads = zip(grads_clip, vars)

        self.train_summary.append(activation_summaries(self.endpoints))
        for var in tf.trainable_variables():
            self.train_summary.append(tf.summary.histogram(var.op.name, var))
        self.train_summary = tf.summary.merge(self.train_summary)

        with tf.control_dependencies(batchnorm_update_ops):
            self.train_op = opt.apply_gradients(grads)

        # We want to inspect other values during training?
        # self.train_ops["vae_loss"] = vae_loss
        self.train_ops["spk_disc_loss"] = spk_disc_loss
        self.train_ops["phn_disc_loss"] = phn_disc_loss
        self.train_ops["loss"] = loss
        self.train_ops["total_loss"] = total_loss
        # self.train_ops.update(vae_all_losses)

        # The model saver
        if self.saver is None:
            self.saver = tf.train.Saver(max_to_keep=self.params.keep_checkpoint_max)

        # The training summary writer
        if self.summary_writer is None:
            self.summary_writer = tf.summary.FileWriter(self.model, self.sess.graph)
        return

    # def _build_decoder(self, zs, zp, recon):
    #     """Build the decoder to reconstruct features
    #
    #     Args:
    #         zs: The sampled speaker variable
    #         zp: The sampled phone variable
    #         recon:
    #     :return: The sampled output, mean and logvar
    #     """
    #     return sampled_recon, mu, logvar
    #
    # def vae_loss(self, mu, logvar, recon):
    #     return loss, all_losses

    def train(self, data_dir, ali_dir, spklist, learning_rate, aux_data=None):
        """Train the model.

        Args:
            data_dir: The training data directory.
            ali_dir: The ali directory.
            spklist: The spklist is a file map speaker name to the index.
            learning_rate: The learning rate is passed by the main program. The main program can easily tune the
                           learning rate according to the validation accuracy or anything else.
            aux_data: The auxiliary data (maybe useful in child class.)
        """
        # initialize all variables
        self.sess.run(tf.global_variables_initializer())

        # curr_step is the real step the training at.
        curr_step = 0

        # Load the model if we have
        if os.path.isfile(os.path.join(self.model, "checkpoint")):
            curr_step = self.load()

        left_context = max(self.params.phone_left_context, self.params.speaker_left_context)
        right_context = max(self.params.phone_right_context, self.params.speaker_right_context)

        # The data loader
        data_loader = KaldiDataRandomQueueV2(data_dir,
                                             ali_dir,
                                             spklist,
                                             num_parallel=self.params.num_parallel_datasets,
                                             max_qsize=self.params.max_queue_size,
                                             left_context=left_context,
                                             right_context=right_context,
                                             num_speakers=self.params.num_speakers_per_batch,
                                             num_segments=self.params.num_segments_per_speaker,
                                             min_len=self.params.min_segment_len,
                                             max_len=self.params.max_segment_len,
                                             shuffle=True)
        data_loader.start()

        epoch = int(curr_step / self.params.num_steps_per_epoch)
        for step in range(curr_step % self.params.num_steps_per_epoch, self.params.num_steps_per_epoch):
            try:
                # Load the data and subsample for phone loss
                features, vad, ali, length, labels, resample, valid_pos = data_loader.fetch()
                assert(features.shape[1] == ali.shape[1] + left_context + right_context)
                phn_masks = make_phone_masks(length, resample, self.params.num_frames_per_utt)
                self._training_egs_stat(labels, ali, valid_pos, phn_masks)

                if step % self.params.save_summary_steps == 0 or step % self.params.show_training_progress == 0:
                    train_ops = [self.train_ops, self.train_op]
                    if step % self.params.save_summary_steps == 0:
                        train_ops.append(self.train_summary)
                    start_time = time.time()
                    train_val = self.sess.run(train_ops, feed_dict={self.features: features,
                                                                    self.spk_labels: labels,
                                                                    self.phn_labels: ali,
                                                                    self.feat_length: length,
                                                                    self.phn_masks: phn_masks,
                                                                    self.global_step: curr_step,
                                                                    self.learning_rate: learning_rate})
                    end_time = time.time()
                    tf.logging.info(
                        "Epoch: [%2d] step: [%2d/%2d] time: %.4f s/step, spk loss: %f, phn loss: %f, loss: %f, total loss: %f" %
                          (epoch, step, self.params.num_steps_per_epoch, end_time - start_time,
                           train_val[0]["spk_disc_loss"], train_val[0]["phn_disc_loss"],
                           train_val[0]["loss"], train_val[0]["total_loss"]))
                    if step % self.params.save_summary_steps == 0:
                        self.summary_writer.add_summary(train_val[-1], curr_step)
                else:
                    # Only compute optimizer.
                    _ = self.sess.run(self.train_op, feed_dict={self.features: features,
                                                                self.spk_labels: labels,
                                                                self.phn_labels: ali,
                                                                self.feat_length: length,
                                                                self.phn_masks: phn_masks,
                                                                self.global_step: curr_step,
                                                                self.learning_rate: learning_rate})
                if step % self.params.save_checkpoints_steps == 0 and curr_step != 0:
                    self.save(curr_step)
                curr_step += 1
            except DataOutOfRange:
                tf.logging.info("Finished reading features.")
                break

        self._print_egs_stat()
        self._save_egs_stat()
        data_loader.stop()
        self.save(curr_step)
        return

    def train_tune_lr(self, data_dir, ali_dir, spklist, tune_period=100, aux_data=None):
        """Tune the learning rate.

        According to: https://www.kdnuggets.com/2017/11/estimating-optimal-learning-rate-deep-neural-network.html

        Args:
            data_dir: The data directory.
            ali_dir: The ali directory.
            spklist: The spklist is a file map speaker name to the index.
            tune_period: How many steps per learning rate.
            aux_data: The auxiliary data directory.
        """
        # initialize all variables
        self.sess.run(tf.global_variables_initializer())

        # We need to load the model sometimes, since we may try to find the learning rate for fine-tuning.
        if os.path.isfile(os.path.join(self.model, "checkpoint")):
            self.load()

        data_loader = KaldiDataRandomQueueV2(data_dir,
                                             ali_dir,
                                             spklist,
                                             num_parallel=self.params.num_parallel_datasets,
                                             max_qsize=self.params.max_queue_size,
                                             left_context=max(self.params.phone_left_context, self.params.speaker_left_context),
                                             right_context=max(self.params.phone_right_context, self.params.speaker_right_context),
                                             num_speakers=self.params.num_speakers_per_batch,
                                             num_segments=self.params.num_segments_per_speaker,
                                             min_len=self.params.min_segment_len,
                                             max_len=self.params.max_segment_len,
                                             shuffle=True)
        data_loader.start()

        # The learning rate normally varies from 1e-5 to 1
        # Some common values:
        # 1. factor = 1.15
        #    tune_period = 200
        #    tune_times = 100
        init_learning_rate = 1e-5
        factor = 1.15
        tune_times = 100

        fp_lr = open(os.path.join(self.model, "learning_rate_tuning"), "w")
        for step in range(tune_period * tune_times):
            lr = init_learning_rate * (factor ** (step // tune_period))
            try:
                features, vad, ali, length, labels, resample, valid_pos = data_loader.fetch()
                phn_masks = make_phone_masks(length, resample, self.params.num_frames_per_utt)

                if step % tune_period == 0:
                    train_ops = [self.train_ops, self.train_op, self.train_summary]
                    # train_ops = [self.train_ops, self.train_op]
                    start_time = time.time()
                    train_val = self.sess.run(train_ops, feed_dict={self.features: features,
                                                                    self.spk_labels: labels,
                                                                    self.phn_labels: ali,
                                                                    self.feat_length: length,
                                                                    self.phn_masks: phn_masks,
                                                                    self.global_step: 0,
                                                                    self.learning_rate: lr})
                    end_time = time.time()
                    tf.logging.info(
                        "Step: %2d time: %.4f s/step, spk loss: %f, phn loss: %f, loss: %f, total loss: %f" %
                        (step, end_time - start_time,
                         train_val[0]["spk_disc_loss"], train_val[0]["phn_disc_loss"],
                         train_val[0]["loss"], train_val[0]["total_loss"]))
                    fp_lr.write("%d %f %f\n" % (step, lr, train_val[0]["loss"]))
                    self.summary_writer.add_summary(train_val[-1], step)
                else:
                    _ = self.sess.run(self.train_op, feed_dict={self.features: features,
                                                                self.spk_labels: labels,
                                                                self.phn_labels: ali,
                                                                self.feat_length: length,
                                                                self.phn_masks: phn_masks,
                                                                self.global_step: 0,
                                                                self.learning_rate: lr})
            except DataOutOfRange:
                tf.logging.info("Finished reading features.")
                break
        data_loader.stop()
        fp_lr.close()
        return

    def valid(self, data_dir, ali_dir, spklist, batch_type="softmax", output_embeddings=False, aux_data=None):
        """Evaluate on the validation set

        Args:
            data_dir: The data directory.
            ali_dir: The ali directory.
            spklist: The spklist is a file map speaker name to the index.
            batch_type: `softmax` or `end2end`. The batch is `softmax-like` or `end2end-like`.
                        If the batch is `softmax-like`, each sample are from different speakers;
                        if the batch is `end2end-like`, the samples are from N speakers with M segments per speaker.
            output_embeddings: Set True to output the corresponding embeddings and labels of the valid set.
                               If output_embeddings, an additional valid metric (e.g. EER) should be computed outside
                               the function.
            aux_data: The auxiliary data directory.

        :return: valid_loss, embeddings and labels (None if output_embeddings is False).
        """
        # Initialization will reset all the variables in the graph.
        # The local variables are also need to be initialized for metrics function.
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())
        assert batch_type == "softmax" or batch_type == "end2end", "The batch_type can only be softmax or end2end"

        curr_step = 0
        # Load the model. The valid function can only be called after training (of course...)
        if os.path.isfile(os.path.join(self.model, "checkpoint")):
            curr_step = self.load()
        else:
            tf.logging.info("[Warning] Cannot find model in %s. Random initialization is used in validation." % self.model)

        embeddings_val = None
        labels_val = None
        num_batches = 0

        left_context = max(self.params.phone_left_context, self.params.speaker_left_context)
        right_context = max(self.params.phone_right_context, self.params.speaker_right_context)

        if output_embeddings:
            # If we want to output embeddings, the features should be loaded in order
            data_loader = KaldiDataSeqQueueV2(data_dir,
                                              ali_dir,
                                              spklist,
                                              num_parallel=2,
                                              max_qsize=10,
                                              left_context=left_context,
                                              right_context=right_context,
                                              batch_size=self.params.num_speakers_per_batch * self.params.num_segments_per_speaker,
                                              min_len=self.params.min_segment_len,
                                              max_len=self.params.max_segment_len,
                                              shuffle=False)
            data_loader.start()

            tf.logging.info("Generate valid embeddings.")
            # In this mode, the embeddings and labels will be saved and output. It needs more memory and takes longer
            # to process these values.
            while True:
                try:
                    if num_batches % 100 == 0:
                        tf.logging.info("valid step: %d" % num_batches)
                    features, vad, ali, length, labels, resample, valid_pos = data_loader.fetch()
                    valid_emb_val = self.sess.run(self.spk_embeddings, feed_dict={self.features: features,
                                                                                  self.feat_length: length,
                                                                                  self.phn_labels: ali})
                    # Save the embeddings and labels
                    if embeddings_val is None:
                        embeddings_val = valid_emb_val
                        labels_val = labels
                    else:
                        embeddings_val = np.concatenate((embeddings_val, valid_emb_val), axis=0)
                        labels_val = np.concatenate((labels_val, labels), axis=0)
                    num_batches += 1
                except DataOutOfRange:
                    break
            data_loader.stop()

        if batch_type == "softmax":
            data_loader = KaldiDataSeqQueueV2(data_dir,
                                              ali_dir,
                                              spklist,
                                              num_parallel=2,
                                              max_qsize=10,
                                              left_context=left_context,
                                              right_context=right_context,
                                              batch_size=self.params.num_speakers_per_batch * self.params.num_segments_per_speaker,
                                              min_len=self.params.min_segment_len,
                                              max_len=self.params.max_segment_len,
                                              shuffle=True)
        elif batch_type == "end2end":
            # The num_valid_speakers_per_batch and num_valid_segments_per_speaker are only required when
            # End2End loss is used. Since we switch the loss function to softmax generalized e2e loss
            # when the e2e loss is used.
            assert "num_valid_speakers_per_batch" in self.params.dict and "num_valid_segments_per_speaker" in self.params.dict, \
                "Valid parameters should be set if E2E loss is selected"
            data_loader = KaldiDataRandomQueueV2(data_dir,
                                                 ali_dir,
                                                 spklist,
                                                 num_parallel=2,
                                                 max_qsize=10,
                                                 left_context=left_context,
                                                 right_context=right_context,
                                                 num_speakers=self.params.num_valid_speakers_per_batch,
                                                 num_segments=self.params.num_valid_segments_per_speaker,
                                                 min_len=self.params.min_segment_len,
                                                 max_len=self.params.max_segment_len,
                                                 shuffle=True)
        else:
            raise ValueError

        data_loader.start()
        num_batches = 0
        for _ in range(self.params.valid_max_iterations):
            try:
                if num_batches % 100 == 0:
                    tf.logging.info("valid step: %d" % num_batches)
                features, vad, ali, length, labels, resample, valid_pos = data_loader.fetch()
                # phn_masks = make_phone_masks(length, resample, self.params.num_frames_per_utt)
                phn_masks = make_phone_masks(length, resample, -1)
                valid_ops = [self.valid_ops["valid_loss_op"],
                             self.valid_ops["valid_spk_loss_op"],
                             self.valid_ops["valid_phn_loss_op"],
                             self.valid_ops["valid_spk_acc_op"],
                             self.valid_ops["valid_phn_acc_op"]]
                _ = self.sess.run(valid_ops, feed_dict={self.features: features,
                                                        self.spk_labels: labels,
                                                        self.phn_labels: ali,
                                                        self.feat_length: length,
                                                        self.phn_masks: phn_masks,
                                                        self.global_step: curr_step})
                num_batches += 1
            except DataOutOfRange:
                break
        data_loader.stop()

        loss, spk_loss, phn_loss, spk_acc, phn_acc, summary = self.sess.run([self.valid_ops["valid_loss"],
                                                                             self.valid_ops["valid_spk_loss"],
                                                                             self.valid_ops["valid_phn_loss"],
                                                                             self.valid_ops["valid_spk_acc"],
                                                                             self.valid_ops["valid_phn_acc"],
                                                                             self.valid_summary])
        # We only save the summary for the last batch.
        self.valid_summary_writer.add_summary(summary, curr_step)
        # The valid loss is averaged over all the batches.
        tf.logging.info("[Validation %d batches] valid loss: %f, spk loss: %f, phone loss: %f, spk acc: %.4f, phone acc: %.4f" %
                        (num_batches, loss, spk_loss, phn_loss, spk_acc, phn_acc))

        # The output embeddings and labels can be used to compute EER or other metrics
        return loss, embeddings_val, labels_val

    def predict_speaker(self, node, features, ali, length):
        """Output the speaker-related embedding of the specified node

        We may use ali when inferring the speaker embeddings.

        Args:
            node: The node of the network.
            features: A matrix which could be [l, d] or [b, l, d]
            ali: The alignment. [l], [b, l]
            length: The length of each segment. [1] or [b]
        :return: A numpy array which is the embeddings. [d] or [b, d]
        """
        curr_step = 0
        if not self.is_loaded:
            if os.path.isfile(os.path.join(self.model, "checkpoint")):
                curr_step = self.load()
            else:
                sys.exit("Cannot find model in %s" % self.model)
        rank = len(features.shape)
        assert(rank == 2 or rank == 3)
        assert(rank == len(ali.shape)+1)
        # Expand the feature if the rank is 2
        if rank == 2:
            features = np.expand_dims(features, axis=0)
            ali = np.expand_dims(ali, axis=0)

        # Feature expansion
        left_context = max(self.params.phone_left_context, self.params.speaker_left_context)
        right_context = max(self.params.phone_right_context, self.params.speaker_right_context)
        features = np.concatenate([np.tile(features[:, 0, :], [1, left_context, 1]), features,
                                   np.tile(features[:, -1, :], [1, right_context, 1])], axis=1)

        embeddings = self.sess.run(self.endpoints[node], feed_dict={self.features: features,
                                                                    self.phn_labels: ali,
                                                                    self.feat_length: length,
                                                                    self.global_step: curr_step})

        # If embeddings are speaker embedding, its shape should be [b, d];
        # or the shape could be [b, l, d].
        # If b == 1, squeeze the axis to [d] or [l, d]
        if rank == 2:
            embeddings = np.squeeze(embeddings, axis=0)
        return embeddings

    def predict_phone(self, node, features, length):
        """Output the phone-related embedding of the specified node.
        Also used for log-output (posterior)

        Args:
            node: The node of the network.
            features: A matrix which could be [l, d] or [b, l, d]
            length: The length of each segment. [1] or [b]
        :return: A numpy array. [l, d] or [b, l, d]
        """
        curr_step = 0
        if not self.is_loaded:
            if os.path.isfile(os.path.join(self.model, "checkpoint")):
                curr_step = self.load()
            else:
                sys.exit("Cannot find model in %s" % self.model)
        rank = len(features.shape)
        assert (rank == 2 or rank == 3)
        # Expand the feature if the rank is 2
        if rank == 2:
            features = np.expand_dims(features, axis=0)

        # Feature expansion
        left_context = max(self.params.phone_left_context, self.params.speaker_left_context)
        right_context = max(self.params.phone_right_context, self.params.speaker_right_context)
        features = np.concatenate([np.tile(features[:, 0, :], [1, left_context, 1]), features,
                                   np.tile(features[:, -1, :], [1, right_context, 1])], axis=1)

        embeddings = self.sess.run(self.endpoints[node], feed_dict={self.features: features,
                                                                    self.feat_length: length,
                                                                    self.global_step: curr_step})
        if rank == 2:
            embeddings = np.squeeze(embeddings, axis=0)
        return embeddings

    def _training_egs_stat(self, labels, ali, pos, phn_masks):
        """Compute the statistics of the training, i.e. the count of speakers and phones selected.

        Args:
            labels: The batch speaker labels
            ali: The batch phone labels (alignments)
            pos: [start, end]. If the start <= index < end, it can be inferred w/o feature expansion
            phn_masks: The masks to choose the alignments.
        :return: Save the statistics in self.spk_training_count, self.phn_training_count
        """
        # Note that we cannot use batch add since some classes may be sampled more than once in a batch.
        for k in labels:
            self.spk_training_count[k] += 1
        for k in phn_masks:
            self.phn_training_count[ali[k[0], k[1]]] += 1
            if pos[k[0], 0] <= k[1] < pos[k[0], 1]:
                self.phn_expansion_count[0] += 1
            else:
                self.phn_expansion_count[1] += 1

    def _print_egs_stat(self):
        total_spk_egs = np.sum(self.spk_training_count)
        per_spk = self.spk_training_count * 100.0 / total_spk_egs
        tf.logging.info("The min speaker %d has %.4f%% training egs, and the max speaker %d has %.4f%%" % (
            np.argmin(per_spk), np.min(per_spk), np.argmax(per_spk), np.max(per_spk)))

        total_phn_egs = np.sum(self.phn_training_count)
        per_phone = self.phn_training_count * 100.0 / total_phn_egs
        tf.logging.info("The min phone %d has %.4f%% training egs, and the max phone %d has %.4f%%" % (
            np.argmin(per_phone), np.min(per_phone), np.argmax(per_phone), np.max(per_phone)))

        total_frames = np.sum(self.phn_expansion_count)
        assert(total_frames == total_phn_egs)
        tf.logging.info("There are %.4f%% frames can be inferred w/o feature expansion, %.4f%% with feature expansion" %
                        (self.phn_expansion_count[0] * 100.0 / total_phn_egs,
                         self.phn_expansion_count[1] * 100.0 / total_phn_egs))

    def _save_egs_stat(self):
        fp_spk = open(os.path.join(self.model, "speaker_egs"), "w")
        for i in range(self.spk_training_count.shape[0]):
            fp_spk.write("%d\n" % self.spk_training_count[i])
        fp_spk.close()
        fp_phn = open(os.path.join(self.model, "phone_egs"), "w")
        for i in range(self.phn_training_count.shape[0]):
            fp_phn.write("%d\n" % self.phn_training_count[i])
        fp_phn.close()
