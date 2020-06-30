import os
import argparse
import random
import sys
import numpy, scipy, sklearn
import tensorflow as tf
import numpy as np
from misc.utils import ValidLoss, load_lr, load_valid_loss, save_codes_and_config, compute_cos_pairwise_eer
from model.trainer_mi import TrainerMultiInput
from dataset.data_loader import KaldiDataRandomQueue
from dataset.kaldi_io import FeatureReader
from six.moves import range

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--cont", action="store_true", help="Continue training from an existing model.")
parser.add_argument("--config", type=str, help="The configuration file.")
parser.add_argument("train_dir", type=str, help="The data directory of the training set.")
parser.add_argument("train_aux_dir", type=str, help="The auxiliary data directory containing all the auxiliary features.")
parser.add_argument("train_spklist", type=str, help="The spklist file maps the TRAINING speakers to the indices.")
parser.add_argument("valid_dir", type=str, help="The data directory of the validation set.")
parser.add_argument("valid_aux_dir", type=str, help="The auxiliary directory of the validation set.")
parser.add_argument("valid_spklist", type=str, help="The spklist maps the VALID speakers to the indices.")
parser.add_argument("model", type=str, help="The output model directory.")


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    args = parser.parse_args()
    params = save_codes_and_config(args.cont, args.model, args.config)

    # The model directory always has a folder named nnet
    model_dir = os.path.join(args.model, "nnet")

    # Set the random seed. The random operations may appear in data input, batch forming, etc.
    tf.set_random_seed(params.seed)
    random.seed(params.seed)
    np.random.seed(params.seed)

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

    # Load the history valid loss
    min_valid_loss = ValidLoss()
    if os.path.isfile(os.path.join(model_dir, "valid_loss")):
        min_valid_loss = load_valid_loss(os.path.join(model_dir, "valid_loss"))

    # The trainer is used to control the training process
    trainer = TrainerMultiInput(params, args.model)
    trainer.build("train",
                  dim=dim,
                  loss_type=params.loss_func,
                  num_speakers=num_total_train_speakers)
    trainer.build("valid",
                  dim=dim,
                  loss_type=params.loss_func,
                  num_speakers=num_total_train_speakers)

    if "early_stop_epochs" not in params.dict:
        params.dict["early_stop_epochs"] = 10
    if "min_learning_rate" not in params.dict:
        params.dict["min_learning_rate"] = 1e-5

    for epoch in range(start_epoch, params.num_epochs):
        trainer.train(args.train_dir, args.train_spklist, learning_rate_array[epoch], aux_data=args.train_aux_dir)
        valid_loss, valid_embeddings, valid_labels = trainer.valid(args.valid_dir, args.valid_spklist,
                                                                   batch_type=params.batch_type,
                                                                   output_embeddings=True,
                                                                   aux_data=args.valid_aux_dir)

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
                    tf.logging.info("After epoch %d, no improvement. Reduce the learning rate to %.8f" % (
                        min_valid_loss.min_loss_epoch, new_learning_rate))
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

        if not os.path.isfile(str(learning_rate)):
            # If the learning rate is too small, the training is actually get stuck.
            # Also early stop is applied.
            # This is only applied when the learning rate is not specified.
            if learning_rate_array[epoch + 1] < (params.min_learning_rate - 1e-12) or \
                    epoch - min_valid_loss.min_loss_epoch >= params.early_stop_epochs:
                break

    # Close the session before we exit.
    trainer.close()
