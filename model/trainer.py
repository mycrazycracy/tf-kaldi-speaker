import tensorflow as tf
import os
import re
import sys
import time
import numpy as np
from model.common import l2_scaling
from model.tdnn import tdnn
from model.loss import softmax
from model.loss import asoftmax, additive_margin_softmax, additive_angular_margin_softmax
from model.loss import semihard_triplet_loss, angular_triplet_loss, e2e_valid_loss, generalized_angular_triplet_loss
from dataset.data_loader import KaldiDataRandomQueue, KaldiDataSeqQueue, DataOutOfRange
from misc.utils import substring_in_list, activation_summaries
from six.moves import range


class Trainer(object):
    """Handle the training, validation and prediction

    Trainer is a simple class that deals with examples having feature-label structure.
    TODO: Add different Trainers to deal with feature+aux_feature - label+aux_label structure.
    """

    def __init__(self, params, model_dir, single_cpu=False):
        """
        Args:
            params: Parameters loaded from JSON.
            model_dir: The model directory.
            single_cpu: Run Tensorflow on one cpu. (default = False)
        """

        # The network configuration is set while the loss is left to the build function.
        # I think we can switch different loss functions during training epochs.
        # Then simple re-build the network can give us a different loss. The main network won't change at that case.
        self.network_type = params.network_type
        if params.network_type == "tdnn":
            self.network = tdnn
        else:
            raise NotImplementedError("Not implement %s network" % params.network_type)
        self.loss_type = None
        self.loss_network = None

        # We have to save all the parameters since the different models may need different parameters
        self.params = params

        if single_cpu:
            self.sess_config = tf.ConfigProto(intra_op_parallelism_threads=1,
                                              inter_op_parallelism_threads=1,
                                              device_count={'CPU': 1},
                                              allow_soft_placement=True)
        else:
            self.sess_config = tf.ConfigProto(allow_soft_placement=True)
        self.sess = tf.Session(config=self.sess_config)

        # The model is saved in model/nnet and the evaluation result is saved in model/nnet/eval
        self.model = os.path.join(model_dir, "nnet")

        # The global step. Note that we don't use tf.train.create_global_step because we may extend the code to
        # support adversarial training, in which the global step increases by 1 after `several` updates on the critic
        # and encoder. The internal global_step should be carefully handled in that case. So just a placeholder here,
        # and use a counter to feed in this value is also an option.
        self.global_step = None

        # The learning rate is just a placeholder. I use placeholder because it gives me flexibility to dynamically
        # change the learning rate during training.
        self.learning_rate = None

        # Summary for the training and validation
        self.train_summary = None
        self.valid_summary = None

        # The output predictions. Useful in the prediction mode.
        self.embeddings = None
        self.endpoints = None

        # The optimizer used in the training.
        # The total loss is useful if we want to change the gradient or variables to optimize (e.g. in fine-tuning)
        self.optimizer = None
        self.total_loss = None

        # Training operation. This is called at each step
        self.train_op = None

        # Dicts for training and validation inspection.
        # In the basic condition, the train_ops contains optimization and training loss.
        # And valid loss in the valid_ops. It is possible to add other variables to the dictionaries.
        # Note that the valid loss should be computed from tf.metric.mean, so the valid_ops also has the update ops.
        # In some steps, the train_ops is required to combine with train_summary to get the summary string.
        # These ops are only executed once after several steps (for inspection).
        self.train_ops = {}
        self.valid_ops = {}

        # Model saver and summary writers
        # We don't create the saver or writer here, because after reset, they will be unavailable.
        self.saver = None
        self.summary_writer = None
        self.valid_summary_writer = None

        # This is an indicator to tell whether the model is built. After building the model, we can only use `reuse`
        # to refer to different part of the model.
        self.is_built = False
        self.is_loaded = False

        # In train, valid and prediction modes, we need the inputs. If tf.data is used, the input can be a node in
        # the graph. However, we may also use feed_dict mechanism to feed data, in which case the placeholder is placed
        # in the graph.
        # Now we define the placeholder in the build routines.
        self.train_features = None
        self.train_labels = None
        self.valid_features = None
        self.valid_labels = None
        self.pred_features = None

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

    def entire_network(self, features, params, is_training, reuse_variables):
        """The definition of the entire network.
        Sometimes, feature normalization is applied after the main network.
        We combine them together (except for the loss layer).

        Args:
            features: The network input.
            params: The parameters.
            is_training: True if the network is for training.
            reuse_variables: Share variables.
        :return: The network output and the endpoints (for other usage).
        """
        features, endpoints = self.network(features, params, is_training, reuse_variables)
        endpoints["output"] = features
        # Add more components (post-processing) after the main network.
        if "feature_norm" in params.dict and params.feature_norm:
            assert "feature_scaling_factor" in params.dict, "If feature normalization is applied, scaling factor is necessary."
            features = l2_scaling(features, params.feature_scaling_factor)
            endpoints["output"] = features

        return features, endpoints

    def build(self, mode, dim, loss_type=None, num_speakers=None, noupdate_var_list=None):
        """ Build a network.

        Currently, I use placeholder in the graph and feed data during sess.run. So no need to parse
        features and labels.

        Args:
            mode: `train`, `valid` or `predict`.
            dim: The dimension of the feature.
            loss_type: Which loss function do we use. Could be None when mode == predict
            num_speakers: The total number of speakers. Used in softmax-like network
            noupdate_var_list: In the fine-tuning, some variables are fixed. The list contains their names (or part of their names).
                               We use `noupdate` rather than `notrain` because some variables are not trainable, e.g.
                               the mean and var in the batchnorm layers.
        """
        assert(mode == "train" or mode == "valid" or mode == "predict")
        is_training = (mode == "train")
        reuse_variables = True if self.is_built else None

        # Create a new path for prediction, since the training may build a tower the support multi-GPUs
        if mode == "predict":
            self.pred_features = tf.placeholder(tf.float32, shape=[None, None, dim], name="pred_features")
            with tf.name_scope("predict") as scope:
                tf.logging.info("Extract embedding from node %s" % self.params.embedding_node)
                # There is no need to do L2 normalization in this function, because we can do the normalization outside,
                # or simply a cosine similarity can do it.
                # Note that the output node may be different if we use different loss function. For example, if the
                # softmax is used, the output of 2-last layer is used as the embedding. While if the end2end loss is
                # used, the output of the last layer may be a better choice. So it is impossible to specify the
                # embedding node inside the network structure. The configuration will tell the network to output the
                # correct activations as the embeddings.
                _, endpoints = self.entire_network(self.pred_features, self.params, is_training, reuse_variables)
                self.embeddings = endpoints[self.params.embedding_node]
                if self.saver is None:
                    self.saver = tf.train.Saver()
            return

        # global_step should be defined before loss function since some loss functions use this value to tune
        # some internal parameters.
        if self.global_step is None:
            self.global_step = tf.placeholder(tf.int32, name="global_step")
            self.params.dict["global_step"] = self.global_step

        # If new loss function is added, please modify the code.
        self.loss_type = loss_type
        if loss_type == "softmax":
            self.loss_network = softmax
        elif loss_type == "asoftmax":
            self.loss_network = asoftmax
        elif loss_type == "additive_margin_softmax":
            self.loss_network = additive_margin_softmax
        elif loss_type == "additive_angular_margin_softmax":
            self.loss_network = additive_angular_margin_softmax
        elif loss_type == "semihard_triplet_loss":
            self.loss_network = semihard_triplet_loss
        elif loss_type == "angular_triplet_loss":
            self.loss_network = angular_triplet_loss
        elif loss_type == "generalized_angular_triplet_loss":
            self.loss_network = generalized_angular_triplet_loss
        else:
            raise NotImplementedError("Not implement %s loss" % self.loss_type)

        if mode == "valid":
            tf.logging.info("Building valid network...")
            self.valid_features = tf.placeholder(tf.float32, shape=[None, None, dim], name="valid_features")
            self.valid_labels = tf.placeholder(tf.int32, shape=[None,], name="valid_labels")
            with tf.name_scope("valid") as scope:
                # We can adjust some parameters in the config when we do validation
                # TODO: I'm not sure whether it is necssary to change the margin for the valid set.
                # TODO: compare the performance!
                # Change the margin for the valid set.
                if loss_type == "softmax":
                    pass
                elif loss_type == "asoftmax":
                    train_margin = self.params.asoftmax_m
                    self.params.asoftmax_m = 1
                elif loss_type == "additive_margin_softmax":
                    train_margin = self.params.amsoftmax_m
                    self.params.amsoftmax_m = 0
                elif loss_type == "additive_angular_margin_softmax":
                    train_margin = self.params.arcsoftmax_m
                    self.params.arcsoftmax_m = 0
                elif loss_type == "angular_triplet_loss":
                    # Switch loss to e2e_valid_loss
                    train_loss_network = self.loss_network
                    self.loss_network = e2e_valid_loss
                else:
                    pass

                if "aux_loss_func" in self.params.dict:
                    # No auxiliary losses during validation.
                    train_aux_loss_func = self.params.aux_loss_func
                    self.params.aux_loss_func = []

                features, endpoints = self.entire_network(self.valid_features, self.params, is_training, reuse_variables)
                valid_loss, endpoints_loss = self.loss_network(features, self.valid_labels, num_speakers, self.params, is_training, reuse_variables)
                endpoints.update(endpoints_loss)

                if "aux_loss_func" in self.params.dict:
                    self.params.aux_loss_func = train_aux_loss_func

                # Change the margin back!!!
                if loss_type == "softmax":
                    pass
                elif loss_type == "asoftmax":
                    self.params.asoftmax_m = train_margin
                elif loss_type == "additive_margin_softmax":
                    self.params.amsoftmax_m = train_margin
                elif loss_type == "additive_angular_margin_softmax":
                    self.params.arcsoftmax_m = train_margin
                elif loss_type == "angular_triplet_loss":
                    self.loss_network = train_loss_network
                else:
                    pass

                # We can evaluate other stuff in the valid_ops. Just add the new values to the dict.
                # We may also need to check other values expect for the loss. Leave the task to other functions.
                # During validation, I compute the cosine EER for the final output of the network.
                self.embeddings = endpoints["output"]
                self.endpoints = endpoints

                self.valid_ops["raw_valid_loss"] = valid_loss
                mean_valid_loss, mean_valid_loss_op = tf.metrics.mean(valid_loss)
                self.valid_ops["valid_loss"] = mean_valid_loss
                self.valid_ops["valid_loss_op"] = mean_valid_loss_op
                valid_loss_summary = tf.summary.scalar("loss", mean_valid_loss)
                self.valid_summary = tf.summary.merge([valid_loss_summary])
                if self.saver is None:
                    self.saver = tf.train.Saver(max_to_keep=self.params.keep_checkpoint_max)
                if self.valid_summary_writer is None:
                    self.valid_summary_writer = tf.summary.FileWriter(os.path.join(self.model, "eval"), self.sess.graph)
            return

        tf.logging.info("Building training network...")
        self.train_features = tf.placeholder(tf.float32, shape=[None, None, dim], name="train_features")
        self.train_labels = tf.placeholder(tf.int32, shape=[None, ], name="train_labels")
        self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")

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

        # Use name_space here. Create multiple name_spaces if multi-gpus
        # There is a copy in `set_trainable_variables`
        with tf.name_scope("train") as scope:
            features, endpoints = self.entire_network(self.train_features, self.params, is_training, reuse_variables)
            loss, endpoints_loss = self.loss_network(features, self.train_labels, num_speakers, self.params, is_training, reuse_variables)
            self.endpoints = endpoints

            endpoints.update(endpoints_loss)
            regularization_loss = tf.losses.get_regularization_loss()
            total_loss = loss + regularization_loss

            # train_summary contains all the summeries we want to inspect.
            # Get the summaries define in the network and loss function.
            # The summeries in the network and loss function are about the network variables.
            self.train_summary = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
            self.train_summary.append(tf.summary.scalar("loss", loss))
            self.train_summary.append(tf.summary.scalar("regularization_loss", regularization_loss))

            # We may have other losses (i.e. penalty term in attention layer)
            penalty_loss = tf.get_collection("PENALTY")
            if len(penalty_loss) != 0:
                penalty_loss = tf.reduce_sum(penalty_loss)
                total_loss += penalty_loss
                self.train_summary.append(tf.summary.scalar("penalty_term", penalty_loss))

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

        if self.params.clip_gradient:
            grads, vars = zip(*grads)  # compute gradients of variables with respect to loss
            grads_clip, _ = tf.clip_by_global_norm(grads, self.params.clip_gradient_norm)  # l2 norm clipping

            # we follow the instruction in ge2e paper to scale the learning rate for w and b
            # Actually, I wonder that we can just simply set a large value for w (e.g. 20) and fix it.
            if self.loss_type == "ge2e":
                # The parameters w and b must be the last variables in the gradients
                grads_clip = grads_clip[:-2] + [0.01 * grad for grad in grads_clip[-2:]]
                # Simply check the position of w and b
                for var in vars[-2:]:
                    assert("w" in var.name or "b" in var.name)
            grads = zip(grads_clip, vars)

        # There are some things we can do to the gradients, i.e. learning rate scaling.

        # # The values and gradients are added to summeries
        # for grad, var in grads:
        #     if grad is not None:
        #         self.train_summary.append(tf.summary.histogram(var.op.name + '/gradients', grad))
        #         self.train_summary.append(tf.summary.scalar(var.op.name + '/gradients_norm', tf.norm(grad)))

        self.train_summary.append(activation_summaries(endpoints))
        for var in tf.trainable_variables():
            self.train_summary.append(tf.summary.histogram(var.op.name, var))
        self.train_summary = tf.summary.merge(self.train_summary)

        with tf.control_dependencies(batchnorm_update_ops):
            self.train_op = opt.apply_gradients(grads)

        # We want to inspect other values during training?
        self.train_ops["loss"] = total_loss
        self.train_ops["raw_loss"] = loss

        # The model saver
        if self.saver is None:
            self.saver = tf.train.Saver(max_to_keep=self.params.keep_checkpoint_max)

        # The training summary writer
        if self.summary_writer is None:
            self.summary_writer = tf.summary.FileWriter(self.model, self.sess.graph)
        return

    def train(self, data, spklist, learning_rate, aux_data=None):
        """Train the model.

        Args:
            data: The training data directory.
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

        # The data loader
        data_loader = KaldiDataRandomQueue(data, spklist,
                                           num_parallel=self.params.num_parallel_datasets,
                                           max_qsize=self.params.max_queue_size,
                                           num_speakers=self.params.num_speakers_per_batch,
                                           num_segments=self.params.num_segments_per_speaker,
                                           min_len=self.params.min_segment_len,
                                           max_len=self.params.max_segment_len,
                                           shuffle=True)
        data_loader.start()

        epoch = int(curr_step / self.params.num_steps_per_epoch)
        for step in range(curr_step % self.params.num_steps_per_epoch, self.params.num_steps_per_epoch):
            try:
                if step % self.params.save_summary_steps == 0 or step % self.params.show_training_progress == 0:
                    train_ops = [self.train_ops, self.train_op]
                    if step % self.params.save_summary_steps == 0:
                        train_ops.append(self.train_summary)
                    start_time = time.time()
                    features, labels = data_loader.fetch()
                    train_val = self.sess.run(train_ops, feed_dict={self.train_features: features,
                                                                    self.train_labels: labels,
                                                                    self.global_step: curr_step,
                                                                    self.learning_rate: learning_rate})
                    end_time = time.time()
                    tf.logging.info(
                        "Epoch: [%2d] step: [%2d/%2d] time: %.4f s/step, raw loss: %f, total loss: %f"
                        % (epoch, step, self.params.num_steps_per_epoch, end_time - start_time,
                           train_val[0]["raw_loss"], train_val[0]["loss"]))
                    if step % self.params.save_summary_steps == 0:
                        self.summary_writer.add_summary(train_val[-1], curr_step)
                else:
                    # Only compute optimizer.
                    features, labels = data_loader.fetch()
                    _ = self.sess.run(self.train_op, feed_dict={self.train_features: features,
                                                                self.train_labels: labels,
                                                                self.global_step: curr_step,
                                                                self.learning_rate: learning_rate})

                if step % self.params.save_checkpoints_steps == 0 and curr_step != 0:
                    self.save(curr_step)
                curr_step += 1
            except DataOutOfRange:
                tf.logging.info("Finished reading features.")
                break

        data_loader.stop()
        self.save(curr_step)

        return

    def train_tune_lr(self, data, spklist, tune_period=100, aux_data=None):
        """Tune the learning rate.

        According to: https://www.kdnuggets.com/2017/11/estimating-optimal-learning-rate-deep-neural-network.html

        Args:
            data: The training data directory.
            spklist: The spklist is a file map speaker name to the index.
            tune_period: How many steps per learning rate.
            aux_data: The auxiliary data directory.
        """
        # initialize all variables
        self.sess.run(tf.global_variables_initializer())

        # We need to load the model sometimes, since we may try to find the learning rate for fine-tuning.
        if os.path.isfile(os.path.join(self.model, "checkpoint")):
            self.load()

        data_loader = KaldiDataRandomQueue(data, spklist,
                                           num_parallel=self.params.num_parallel_datasets,
                                           max_qsize=self.params.max_queue_size,
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
                if step % tune_period == 0:
                    train_ops = [self.train_ops, self.train_op, self.train_summary]
                    # train_ops = [self.train_ops, self.train_op]
                    start_time = time.time()
                    features, labels = data_loader.fetch()
                    train_val = self.sess.run(train_ops, feed_dict={self.train_features: features,
                                                                    self.train_labels: labels,
                                                                    self.global_step: 0,
                                                                    self.learning_rate: lr})
                    end_time = time.time()
                    tf.logging.info(
                        "Epoch: step: %2d, time: %.4f s/step, lr: %f, raw loss: %f, total loss: %f" \
                        % (step, end_time - start_time, lr,
                           train_val[0]["raw_loss"], train_val[0]["loss"]))
                    fp_lr.write("%d %f %f\n" % (step, lr, train_val[0]["loss"]))
                    self.summary_writer.add_summary(train_val[-1], step)
                else:
                    features, labels = data_loader.fetch()
                    _ = self.sess.run(self.train_op, feed_dict={self.train_features: features,
                                                                self.train_labels: labels,
                                                                self.global_step: 0,
                                                                self.learning_rate: lr})
            except DataOutOfRange:
                tf.logging.info("Finished reading features.")
                break
        data_loader.stop()
        fp_lr.close()
        return

    def valid(self, data, spklist, batch_type="softmax", output_embeddings=False, aux_data=None):
        """Evaluate on the validation set

        Args:
            data: The training data directory.
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

        if output_embeddings:
            # If we want to output embeddings, the features should be loaded in order
            data_loader = KaldiDataSeqQueue(data, spklist,
                                            num_parallel=2,
                                            max_qsize=10,
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
                    features, labels = data_loader.fetch()
                    valid_emb_val, valid_labels_val = self.sess.run([self.embeddings, self.valid_labels], feed_dict={self.valid_features: features,
                                                                                                                     self.valid_labels: labels,
                                                                                                                     self.global_step: curr_step})
                    # Save the embeddings and labels
                    if embeddings_val is None:
                        embeddings_val = valid_emb_val
                        labels_val = valid_labels_val
                    else:
                        embeddings_val = np.concatenate((embeddings_val, valid_emb_val), axis=0)
                        labels_val = np.concatenate((labels_val, valid_labels_val), axis=0)
                    num_batches += 1
                except DataOutOfRange:
                    break
            data_loader.stop()

        if batch_type == "softmax":
            data_loader = KaldiDataSeqQueue(data, spklist,
                                            num_parallel=2,
                                            max_qsize=10,
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
            data_loader = KaldiDataRandomQueue(data, spklist,
                                               num_parallel=2,
                                               max_qsize=10,
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
                features, labels = data_loader.fetch()
                _ = self.sess.run(self.valid_ops["valid_loss_op"], feed_dict={self.valid_features: features,
                                                                                      self.valid_labels: labels,
                                                                                      self.global_step: curr_step})
                num_batches += 1
            except DataOutOfRange:
                break
        data_loader.stop()

        loss, summary = self.sess.run([self.valid_ops["valid_loss"], self.valid_summary])
        # We only save the summary for the last batch.
        self.valid_summary_writer.add_summary(summary, curr_step)
        # The valid loss is averaged over all the batches.
        tf.logging.info("[Validation %d batches] valid loss: %f" % (num_batches, loss))

        # The output embeddings and labels can be used to compute EER or other metrics
        return loss, embeddings_val, labels_val

    def predict(self, features):
        """Output the embeddings

        :return: A numpy array which is the embeddings
        """
        if not self.is_loaded:
            if os.path.isfile(os.path.join(self.model, "checkpoint")):
                self.load()
            else:
                sys.exit("Cannot find model in %s" % self.model)
        rank = len(features.shape)
        assert(rank == 2 or rank == 3)
        # Expand the feature if the rank is 2
        if rank == 2:
            features = np.expand_dims(features, axis=0)
        embeddings = self.sess.run(self.embeddings, feed_dict={self.pred_features: features})
        if rank == 2:
            embeddings = np.squeeze(embeddings, axis=0)
        return embeddings

    def set_trainable_variables(self, variable_list=None):
        """Set the variables which we want to optimize.
        The optimizer will only optimize the variables which contain sub-string in the variable list.
        Basically, this is copied from the training path in `build`.

        The batchnorm statistics can always be updated?

        Args:
            variable_list: The model variable contains sub-string in the list will be optimized.
                           If None, all variables will be optimized.
        """
        add_train_summary = []
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        trainable_variables = []
        if variable_list is None:
            tf.logging.info("[Info] Add all trainable variables to the optimizer.")
            trainable_variables = None
        else:
            for v in variables:
                if substring_in_list(v.name, variable_list):
                    trainable_variables.append(v)
                    tf.logging.info("[Info] Add %s to trainable list" % v.name)

        with tf.name_scope("train") as scope:
            grads = self.optimizer.compute_gradients(self.total_loss, var_list=trainable_variables)

        if self.params.clip_gradient:
            grads, vars = zip(*grads)  # compute gradients of variables with respect to loss
            grads_clip, _ = tf.clip_by_global_norm(grads, self.params.clip_gradient_norm)  # l2 norm clipping
            grads = zip(grads_clip, vars)

        # # The values and gradients are added to summeries
        # for grad, var in grads:
        #     if grad is not None:
        #         add_train_summary.append(tf.summary.histogram(var.op.name + '/gradients', grad))
        #         add_train_summary.append(tf.summary.scalar(var.op.name + '/gradients_norm', tf.norm(grad)))

        if variable_list is None:
            trainable_variables = tf.trainable_variables()
        for var in trainable_variables:
            add_train_summary.append(tf.summary.histogram(var.op.name, var))
        self.train_summary = tf.summary.merge([self.train_summary, tf.summary.merge(add_train_summary)])

        batchnorm_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope)
        with tf.control_dependencies(batchnorm_update_ops):
            self.train_op = self.optimizer.apply_gradients(grads)

    def get_finetune_model(self, excluded_list):
        """Start from a pre-trained model and other parameters are initialized using default initializer.
        Actually, this function is only called at the first epoch of the fine-tuning, because in succeeded epochs,
        we need to fully load the model rather than loading part of the graph.

        The pre-trained model is saved in the model directory as index 0.
        Backup the pre-trained model and save the new model (with random initialized parameters) as index 0 instead.

        Args:
            excluded_list: A list. Do NOT restore the parameters in the exclude_list. This is useful in fine-truning
                          an existing model. We load a part of the pre-trained model and leave the other part
                          randomly initialized.
        Deprecated:
            data: The training data directory.
            spklist: The spklist is a file map speaker name to the index.
            learning_rate: The learning rate is passed by the main program. The main program can easily tune the
                           learning rate according to the validation accuracy or anything else.
        """
        # initialize all variables
        self.sess.run(tf.global_variables_initializer())

        # Load parts of the model
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        restore_variables = []
        for v in variables:
            if not substring_in_list(v.name, excluded_list):
                restore_variables.append(v)
            else:
                tf.logging.info("[Info] Ignore %s when loading the checkpoint" % v.name)
        finetune_saver = tf.train.Saver(var_list=restore_variables)
        ckpt = tf.train.get_checkpoint_state(self.model)
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        finetune_saver.restore(self.sess, os.path.join(self.model, ckpt_name))

        # Backup the old files
        import glob, shutil
        model_checkpoint_path = ckpt.model_checkpoint_path
        for filename in glob.glob(model_checkpoint_path + "*"):
            shutil.copyfile(filename, filename + '.bak')

        # Save the new model. The new model is basically the same with the pre-trained one, while parameters
        # NOT in the pre-trained model are random initialized.
        # Set the step to 0.
        self.save(0)
        return

    def insight(self, data, spklist, batch_type="softmax", output_embeddings=False, aux_data=None):
        """Just use to debug the network
        """
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())
        assert batch_type == "softmax" or batch_type == "end2end", "The batch_type can only be softmax or end2end"

        embeddings_val = None
        labels_val = None

        self.load()

        if output_embeddings:
            # If we want to output embeddings, the features should be loaded in order
            data_loader = KaldiDataSeqQueue(data, spklist,
                                            num_parallel=2,
                                            max_qsize=10,
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
                    features, labels = data_loader.fetch()
                    valid_emb_val, valid_labels_val, endpoints_val = self.sess.run([self.embeddings, self.valid_labels, self.endpoints], feed_dict={self.valid_features: features,
                                                                                                                     self.valid_labels: labels})

                    # acc = np.sum(np.equal(np.argmax(endpoints_val['logits'], axis=1), labels, dtype=np.float)) / float(
                    #     labels.shape[0])
                    # print("Acc: %f" % acc)

                    # Save the embeddings and labels
                    if embeddings_val is None:
                        embeddings_val = valid_emb_val
                        labels_val = valid_labels_val
                    else:
                        embeddings_val = np.concatenate((embeddings_val, valid_emb_val), axis=0)
                        labels_val = np.concatenate((labels_val, valid_labels_val), axis=0)
                except DataOutOfRange:
                    break
            data_loader.stop()

        if batch_type == "softmax":
            data_loader = KaldiDataSeqQueue(data, spklist,
                                            num_parallel=2,
                                            max_qsize=10,
                                            batch_size=self.params.num_speakers_per_batch * self.params.num_segments_per_speaker*10,
                                            min_len=self.params.min_segment_len,
                                            max_len=self.params.max_segment_len,
                                            shuffle=True)
        elif batch_type == "end2end":
            # The num_valid_speakers_per_batch and num_valid_segments_per_speaker are only required when
            # End2End loss is used. Since we switch the loss function to softmax generalized e2e loss
            # when the e2e loss is used.
            assert "num_valid_speakers_per_batch" in self.params.dict and "num_valid_segments_per_speaker" in self.params.dict, \
                "Valid parameters should be set if E2E loss is selected"
            data_loader = KaldiDataRandomQueue(data, spklist,
                                               num_parallel=2,
                                               max_qsize=10,
                                               num_speakers=self.params.num_valid_speakers_per_batch,
                                               num_segments=self.params.num_valid_segments_per_speaker,
                                               min_len=self.params.min_segment_len,
                                               max_len=self.params.max_segment_len,
                                               shuffle=True)
        else:
            raise ValueError

        data_loader.start()

        while True:
            try:
                features, labels = data_loader.fetch()
                _, endpoints_val = self.sess.run([self.valid_ops["valid_loss_op"], self.endpoints], feed_dict={self.valid_features: features,
                                                                                                                 self.valid_labels: labels})
            except DataOutOfRange:
                break
        data_loader.stop()
        loss = self.sess.run(self.valid_ops["valid_loss"])
        tf.logging.info("Shorter segments are used to test the valid loss (%d-%d)" % (self.params.min_segment_len, self.params.max_segment_len))
        tf.logging.info("Loss: %f" % loss)


        # while True:
        #     try:
        #         features, labels = data_loader.fetch()
        #         valid_ops, endpoints_val = self.sess.run([self.valid_ops, self.endpoints], feed_dict={self.valid_features: features,
        #                                                                                                          self.valid_labels: labels})
        #         loss = valid_ops["valid_loss"]
        #     except DataOutOfRange:
        #         break
        # data_loader.stop()
        # tf.logging.info("Loss: %f" % loss)

        acc = np.sum(np.equal(np.argmax(endpoints_val['logits'], axis=1), labels, dtype=np.float)) / float(labels.shape[0])
        print("Acc: %f" % acc)

        import pdb
        pdb.set_trace()
        # from model.test_utils import softmax
        # with tf.variable_scope("softmax", reuse=True):
        #     test = tf.get_variable("output/kernel")
        #     test_val = self.sess.run(test)
        return loss, embeddings_val, labels_val
