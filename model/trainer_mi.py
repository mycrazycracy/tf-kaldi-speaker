import tensorflow as tf
import os
import sys
import time
import numpy as np
from model.trainer import Trainer
from dataset.data_loader import KaldiMultiDataRandomQueue, KaldiMultiDataSeqQueue, DataOutOfRange
from model.common import l2_scaling
from model.loss import softmax
from model.loss import asoftmax, additive_margin_softmax, additive_angular_margin_softmax
from model.loss import semihard_triplet_loss, angular_triplet_loss, e2e_valid_loss
from misc.utils import substring_in_list, activation_summaries
from six.moves import range


class TrainerMultiInput(Trainer):
    """Trainer for multiple inputs.

    The class supports multiple features as the inputs and multiple labels as the outputs.
    Useful when we involving bottleneck features, linguistic features or other auxiliary features.
    """
    def __init__(self, params, model_dir, single_cpu=False):
        """
        Args:
            params: Parameters loaded from JSON.
            model_dir: The model directory.
            single_cpu: Run Tensorflow on one cpu. (default = False)
        """
        super(TrainerMultiInput, self).__init__(params, model_dir, single_cpu)

        # In this class, we need auxiliary features to do the feed-forward operation.
        # The auxiliary features are dictionary that contains multiple possible features.
        # When building the network, the auxiliary features are access by their names.
        # To support more features (inputs), please extend the list below.
        self.train_aux_features = {}
        self.valid_aux_features = {}
        self.pred_aux_features = {}

    def entire_network(self, features, params, is_training, reuse_variables):
        """The definition of the entire network.

        Args:
            features: dict, features["features"] and features["aux_features"]
            params: The parameters.
            is_training: True if the network is for training.
            reuse_variables: Share variables.
        :return: The network output and the endpoints (for other usage).
        """
        features, endpoints = self.network(features["features"], params, is_training, reuse_variables,
                                           aux_features=features["aux_features"])
        endpoints["output"] = features
        # Add more components (post-processing) after the main network.
        if "feature_norm" in params.dict and params.feature_norm:
            assert "feature_scaling_factor" in params.dict, "If feature normalization is applied, scaling factor is necessary."
            features = l2_scaling(features, params.feature_scaling_factor)
            endpoints["output"] = features

        return features, endpoints

    def build(self, mode, dim, loss_type=None, num_speakers=None, noupdate_var_list=None):
        """ Build a network.

        This class accept multiple network inputs so that we can use bottleneck features, linguistic features, etc,
        as the network inputs.

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

            # We also need to initialize other features.
            # We need to specify the dim of the auxiliary features.
            assert "aux_feature_dim" in self.params.dict, "The dim of auxiliary features must be specified as a dict."
            for name in self.params.aux_feature_dim:
                self.pred_aux_features[name] = tf.placeholder(tf.float32,
                                                              shape=[None, None, self.params.aux_feature_dim[name]],
                                                              name="pred_" + name)
            pred_features = {"features": self.pred_features,
                             "aux_features": self.pred_aux_features}

            with tf.name_scope("predict") as scope:
                tf.logging.info("Extract embedding from node %s" % self.params.embedding_node)
                _, endpoints = self.entire_network(pred_features, self.params, is_training, reuse_variables)
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
        else:
            raise NotImplementedError("Not implement %s loss" % self.loss_type)

        if mode == "valid":
            tf.logging.info("Building valid network...")

            assert "aux_feature_dim" in self.params.dict, "The dim of auxiliary features must be specified as a dict."
            for name in self.params.aux_feature_dim:
                self.valid_aux_features[name] = tf.placeholder(tf.float32,
                                                               shape=[None, None, self.params.aux_feature_dim[name]],
                                                               name="valid_" + name)
            self.valid_features = tf.placeholder(tf.float32, shape=[None, None, dim], name="valid_features")
            valid_features = {"features": self.valid_features,
                              "aux_features": self.valid_aux_features}

            self.valid_labels = tf.placeholder(tf.int32, shape=[None, ], name="valid_labels")

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

                features, endpoints = self.entire_network(valid_features, self.params, is_training, reuse_variables)
                valid_loss = self.loss_network(features, self.valid_labels, num_speakers, self.params, is_training, reuse_variables)

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

        assert "aux_feature_dim" in self.params.dict, "The dim of auxiliary features must be specified as a dict."
        for name in self.params.aux_feature_dim:
            self.train_aux_features[name] = tf.placeholder(tf.float32,
                                                           shape=[None, None, self.params.aux_feature_dim[name]],
                                                           name="train_" + name)
        train_features = {"features": self.train_features,
                          "aux_features": self.train_aux_features}

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
            features, endpoints = self.entire_network(train_features, self.params, is_training, reuse_variables)
            loss = self.loss_network(features, self.train_labels, num_speakers, self.params, is_training, reuse_variables)
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
            aux_data: The auxiliary data directory.
        """
        # initialize all variables
        self.sess.run(tf.global_variables_initializer())

        # curr_step is the real step the training at.
        curr_step = 0

        # Load the model if we have
        if os.path.isfile(os.path.join(self.model, "checkpoint")):
            curr_step = self.load()

        # The data loader
        data_loader = KaldiMultiDataRandomQueue(data, aux_data, spklist,
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
                features, labels = data_loader.fetch()
                feed_dict = {self.train_features: features["features"],
                             self.train_labels: labels,
                             self.global_step: curr_step,
                             self.learning_rate: learning_rate}
                for name in features:
                    if name == "features":
                        continue
                    feed_dict[self.train_aux_features[name]] = features[name]

                if step % self.params.save_summary_steps == 0 or step % self.params.show_training_progress == 0:
                    train_ops = [self.train_ops, self.train_op]
                    if step % self.params.save_summary_steps == 0:
                        train_ops.append(self.train_summary)
                    start_time = time.time()
                    train_val = self.sess.run(train_ops, feed_dict=feed_dict)
                    end_time = time.time()
                    tf.logging.info(
                        "Epoch: [%2d] step: [%2d/%2d] time: %.4f s/step, raw loss: %f, total loss: %f"
                        % (epoch, step, self.params.num_steps_per_epoch, end_time - start_time,
                           train_val[0]["raw_loss"], train_val[0]["loss"]))
                    if step % self.params.save_summary_steps == 0:
                        self.summary_writer.add_summary(train_val[-1], curr_step)
                else:
                    # Only compute optimizer.
                    _ = self.sess.run(self.train_op, feed_dict=feed_dict)

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

        I think it is better to use sgd to test the learning rate.

        According to: https://www.kdnuggets.com/2017/11/estimating-optimal-learning-rate-deep-neural-network.html

        Args:
            data: The training data directory.
            spklist: The spklist is a file map speaker name to the index.
            tune_period: How many steps per learning rate.
            aux_data: The auxiliary data directory.
        """
        # initialize all variables
        self.sess.run(tf.global_variables_initializer())

        data_loader = KaldiMultiDataRandomQueue(data, aux_data, spklist,
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
        #    tune_period = 100
        #    tune_times = 100
        init_learning_rate = 1e-5
        factor = 1.15
        tune_times = 100

        fp_lr = open(os.path.join(self.model, "learning_rate_tuning"), "w")
        for step in range(tune_period * tune_times):
            lr = init_learning_rate * (factor ** (step / tune_period))
            features, labels = data_loader.fetch()
            feed_dict = {self.train_features: features["features"],
                         self.train_labels: labels,
                         self.global_step: 0,
                         self.learning_rate: lr}
            for name in features:
                if name == "features":
                    continue
                feed_dict[self.train_aux_features[name]] = features[name]

            try:
                if step % tune_period == 0:
                    train_ops = [self.train_ops, self.train_op, self.train_summary]
                    start_time = time.time()

                    train_val = self.sess.run(train_ops, feed_dict=feed_dict)
                    end_time = time.time()
                    tf.logging.info(
                        "Epoch: step: %2d time: %.4f s/step, lr: %f, raw loss: %f, total loss: %f" \
                        % (step, end_time - start_time, lr,
                           train_val[0]["raw_loss"], train_val[0]["loss"]))
                    fp_lr.write("%d %f %f\n" % (step, lr, train_val[0]["loss"]))
                    self.summary_writer.add_summary(train_val[-1], step)
                else:
                    _ = self.sess.run(self.train_op, feed_dict=feed_dict)
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
            data_loader = KaldiMultiDataSeqQueue(data, aux_data, spklist,
                                                 num_parallel=1,
                                                 max_qsize=10,
                                                 batch_size=self.params.num_speakers_per_batch * self.params.num_segments_per_speaker,
                                                 min_len=self.params.min_segment_len,
                                                 max_len=self.params.max_segment_len,
                                                 shuffle=False)
            data_loader.start()

            # In this mode, the embeddings and labels will be saved and output. It needs more memory and takes longer
            # to process these values.
            while True:
                try:
                    if num_batches % 100 == 0:
                        tf.logging.info("valid step: %d" % num_batches)
                    features, labels = data_loader.fetch()
                    feed_dict = {self.valid_features: features["features"],
                                 self.valid_labels: labels,
                                 self.global_step: curr_step}
                    for name in features:
                        if name == "features":
                            continue
                        feed_dict[self.valid_aux_features[name]] = features[name]

                    valid_emb_val, valid_labels_val = self.sess.run([self.embeddings, self.valid_labels], feed_dict=feed_dict)
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
            data_loader = KaldiMultiDataSeqQueue(data, aux_data, spklist,
                                                 num_parallel=2,
                                                 max_qsize=10,
                                                 batch_size=self.params.num_speakers_per_batch * self.params.num_segments_per_speaker,
                                                 min_len=self.params.min_segment_len,
                                                 max_len=self.params.max_segment_len,
                                                 shuffle=True)
        elif batch_type == "end2end":
            data_loader = KaldiMultiDataRandomQueue(data, aux_data, spklist,
                                                    num_parallel=2,
                                                    max_qsize=10,
                                                    num_speakers=self.params.num_speakers_per_batch,
                                                    num_segments=self.params.num_segments_per_speaker,
                                                    min_len=self.params.min_segment_len,
                                                    max_len=self.params.max_segment_len,
                                                    shuffle=True)
        else:
            raise ValueError

        data_loader.start()
        for _ in range(self.params.valid_max_iterations):
            try:
                if num_batches % 100 == 0:
                    tf.logging.info("valid step: %d" % num_batches)
                features, labels = data_loader.fetch()
                feed_dict = {self.valid_features: features["features"],
                             self.valid_labels: labels,
                             self.global_step: curr_step}
                for name in features:
                    if name == "features":
                        continue
                    feed_dict[self.valid_aux_features[name]] = features[name]

                _ = self.sess.run(self.valid_ops["valid_loss_op"], feed_dict=feed_dict)
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

        rank = len(features["features"].shape)
        assert (rank == 2 or rank == 3)
        # Expand the feature if the rank is 2
        if rank == 2:
            for name in features:
                features[name] = np.expand_dims(features[name], axis=0)

        feed_dict = {self.pred_features: features["features"]}
        for name in features:
            if name == "features":
                continue
            feed_dict[self.pred_aux_features[name]] = features[name]
            # The shape of the features should be the same except for the last dimension.
            assert(features["features"].shape[:-1] == features[name].shape[:-1])

        embeddings = self.sess.run(self.embeddings, feed_dict=feed_dict)
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

        # The values and gradients are added to summeries
        for grad, var in grads:
            if grad is not None:
                add_train_summary.append(tf.summary.histogram(var.op.name + '/gradients', grad))
                add_train_summary.append(tf.summary.scalar(var.op.name + '/gradients_norm', tf.norm(grad)))

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
