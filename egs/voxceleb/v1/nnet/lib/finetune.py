# Fine-tune a pre-trained model to a new model.
# Optimize the entire network after loading the pre-trained model.
# This require that there is no new parameter introduced in the model.
# This is appropriate to train an End-to-end triplet loss (or other) network from a softmax pre-trained network where
# no additional layer is involved.
# If a new softmax layer is added to the pre-trained layer, it is better to train the new softmax first and
# then update the entire network.

import os
import argparse
import random
import sys
import tensorflow as tf
import numpy as np
from misc.utils import get_pretrain_model, load_lr
from misc.utils import ValidLoss, save_codes_and_config, compute_cos_pairwise_eer, load_valid_loss
from model.trainer import Trainer
from dataset.data_loader import KaldiDataRandomQueue
from dataset.kaldi_io import FeatureReader
from six.moves import range

# We don't need to use a `continue` option here, because if we want to resume training, we should simply use train.py.
# In the beginning of finetuning, we want to restore a part of the model rather than the entire graph.
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--cont", action="store_true", help="Continue training from an existing model.")
parser.add_argument("--checkpoint", type=str, default="-1", help="The checkpoint in the pre-trained model. The default is to load the BEST checkpoint (according to valid_loss)")
parser.add_argument("--config", type=str, help="The configuration file.")
parser.add_argument("train_dir", type=str, help="The data directory of the training set.")
parser.add_argument("train_spklist", type=str, help="The spklist file maps the TRAINING speakers to the indices.")
parser.add_argument("valid_dir", type=str, help="The data directory of the validation set.")
parser.add_argument("valid_spklist", type=str, help="The spklist maps the VALID speakers to the indices.")
parser.add_argument("pretrain_model", type=str, help="The pre-trained model directory.")
parser.add_argument("finetune_model", type=str, help="The fine-tuned model directory")


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    args = parser.parse_args()
    params = save_codes_and_config(args.cont, args.finetune_model, args.config)

    # Set the random seed. The random operations may appear in data input, batch forming, etc.
    tf.set_random_seed(params.seed)
    random.seed(params.seed)
    np.random.seed(params.seed)

    # The model directory always has a folder named nnet
    model_dir = os.path.join(args.finetune_model, "nnet")

    if args.cont:
        # If we continue training, we can figure out how much steps the model has been trained,
        # using the index of the checkpoint
        import re

        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            step = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
        else:
            sys.exit("Cannot load checkpoint from %s" % model_dir)
        start_epoch = int(step / params.num_steps_per_epoch)
    else:
        # Load the pre-trained model to the target model directory.
        # The pre-trained model will be copied as the fine-tuned model and can be loaded from the new directory.
        # The pre-trained model is now just like an initialized model.
        get_pretrain_model(os.path.join(args.pretrain_model, "nnet"),
                           os.path.join(args.finetune_model, "nnet"),
                           args.checkpoint)
        start_epoch = 0

    learning_rate = params.learning_rate
    learning_rate_array = []
    if os.path.isfile(str(learning_rate)):
        with open(str(learning_rate), "r") as f:
            for line in f.readlines():
                learning_rate_array.append(float(line.strip()))
        # The size of the file should be large enough
        assert len(learning_rate_array) > params.num_epochs, "The learning rate file is shorter than the num of epochs."
        tf.logging.info("Using specified learning rate decay strategy.")
    else:
        # The learning rate is determined by the training process. However, if we continue training,
        # the code doesn't know the previous learning rate if it is tuned using the validation set.
        # To solve that, just save the learning rate to an individual file.
        if os.path.isfile(os.path.join(model_dir, "learning_rate")):
            learning_rate_array = load_lr(os.path.join(model_dir, "learning_rate"))
            assert len(learning_rate_array) == start_epoch + 1, "Not enough learning rates in the learning_rate file."
        else:
            learning_rate_array = [float(learning_rate)] * (start_epoch + 1)

    dim = FeatureReader(args.train_dir).get_dim()
    with open(os.path.join(model_dir, "feature_dim"), "w") as f:
        f.write("%d\n" % dim)
    num_total_train_speakers = KaldiDataRandomQueue(args.train_dir, args.train_spklist).num_total_speakers
    tf.logging.info("There are %d speakers in the training set and the dim is %d" % (num_total_train_speakers, dim))

    min_valid_loss = ValidLoss()
    if os.path.isfile(os.path.join(model_dir, "valid_loss")):
        min_valid_loss = load_valid_loss(os.path.join(model_dir, "valid_loss"))

    # The trainer is used to control the training process
    trainer = Trainer(params, args.finetune_model)
    trainer.build("train",
                  dim=dim,
                  loss_type=params.loss_func,
                  num_speakers=num_total_train_speakers,
                  noupdate_var_list=params.noupdate_var_list)
    trainer.build("valid",
                  dim=dim,
                  loss_type=params.loss_func,
                  num_speakers=num_total_train_speakers)

    if "early_stop_epochs" not in params.dict:
        params.dict["early_stop_epochs"] = 5
    if "min_learning_rate" not in params.dict:
        params.dict["min_learning_rate"] = 1e-5

    if start_epoch == 0:
        # Load the pre-trained model and transfer to current model
        trainer.get_finetune_model(params.noload_var_list)

        # Before any further training, we evaluate the performance of the current model
        valid_loss, valid_embeddings, valid_labels = trainer.valid(args.valid_dir, args.valid_spklist,
                                                                   batch_type=params.batch_type,
                                                                   output_embeddings=True)
        eer = compute_cos_pairwise_eer(valid_embeddings, valid_labels)
        tf.logging.info("In the beginning: Valid EER: %f" % eer)

    for epoch in range(start_epoch, params.num_epochs):
        trainer.train(args.train_dir, args.train_spklist, learning_rate_array[epoch])
        valid_loss, valid_embeddings, valid_labels = trainer.valid(args.valid_dir, args.valid_spklist,
                                                                   batch_type=params.batch_type,
                                                                   output_embeddings=True)
        eer = compute_cos_pairwise_eer(valid_embeddings, valid_labels)
        tf.logging.info("[INFO] Valid EER: %f" % eer)

        # Tune the learning rate if necessary.
        if not os.path.isfile(str(learning_rate)):
            new_learning_rate = learning_rate_array[epoch]
            if valid_loss < min_valid_loss.min_loss:
                min_valid_loss.min_loss = valid_loss
                min_valid_loss.min_loss_epoch = epoch
            else:
                if epoch - min_valid_loss.min_loss_epoch >= params.reduce_lr_epochs:
                    new_learning_rate /= 2
                    # If the valid loss in the next epoch still does not reduce, the learning rate will keep reducing.
                    tf.logging.info("After epoch %d, no improvement. Reduce the learning rate to %f" % (
                        min_valid_loss.min_loss_epoch, new_learning_rate))
                    # min_valid_loss.min_loss = valid_loss
                    min_valid_loss.min_loss_epoch += (params.reduce_lr_epochs / 2)
            learning_rate_array.append(new_learning_rate)

        if epoch == 0:
            # If this is the first epoch, the first learning rate should be recorded
            with open(os.path.join(model_dir, "learning_rate"), "a") as f:
                f.write("0 %.8f\n" % learning_rate_array[0])

        # Save the learning rate and loss for each epoch.
        with open(os.path.join(model_dir, "learning_rate"), "a") as f:
            f.write("%d %.8f\n" % (epoch + 1, learning_rate_array[epoch + 1]))
        with open(os.path.join(model_dir, "valid_loss"), "a") as f:
            f.write("%d %f %f\n" % (epoch, valid_loss, eer))

        # If the learning rate is too small, the training is actually get stuck.
        # Also early stop is applied.
        if learning_rate_array[epoch + 1] < (params.min_learning_rate - 1e-12) or \
                epoch - min_valid_loss.min_loss_epoch >= params.early_stop_epochs:
            break

    # Close the session before we exit.
    trainer.close()


