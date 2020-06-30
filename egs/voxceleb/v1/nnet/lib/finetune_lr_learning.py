import os
import argparse
import random
import sys
import tensorflow as tf
import numpy as np
from misc.utils import get_pretrain_model
from misc.utils import ValidLoss, save_codes_and_config, compute_cos_pairwise_eer
from model.trainer import Trainer
from dataset.data_loader import KaldiDataRandomQueue
from dataset.kaldi_io import FeatureReader
from six.moves import range

# We don't need to use a `continue` option here, because if we want to resume training, we should simply use train.py.
# In the beginning of finetuning, we want to restore a part of the model rather than the entire graph.
parser = argparse.ArgumentParser()
parser.add_argument("--tune_period", type=int, default=100, help="How many steps per learning rate.")
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
    params = save_codes_and_config(False, args.finetune_model, args.config)

    # Load the pre-trained model to the target model directory.
    # The pre-trained model will be copied as the fine-tuned model and can be loaded from the new directory.
    # The pre-trained model is now just like an initialized model.
    get_pretrain_model(os.path.join(args.pretrain_model, "nnet"),
                       os.path.join(args.finetune_model, "nnet"),
                       args.checkpoint)

    # The model directory always has a folder named nnet
    model_dir = os.path.join(args.finetune_model, "nnet")

    # Set the random seed. The random operations may appear in data input, batch forming, etc.
    tf.set_random_seed(params.seed)
    random.seed(params.seed)
    np.random.seed(params.seed)

    dim = FeatureReader(args.train_dir).get_dim()
    with open(os.path.join(model_dir, "feature_dim"), "w") as f:
        f.write("%d\n" % dim)

    num_total_train_speakers = KaldiDataRandomQueue(args.train_dir, args.train_spklist).num_total_speakers
    tf.logging.info("There are %d speakers in the training set and the dim is %d" % (num_total_train_speakers, dim))

    min_valid_loss = ValidLoss()

    # The trainer is used to control the training process
    trainer = Trainer(params, args.finetune_model)
    trainer.build("train",
                  dim=dim,
                  loss_type=params.loss_func,
                  num_speakers=num_total_train_speakers)
    trainer.build("valid",
                  dim=dim,
                  loss_type=params.loss_func,
                  num_speakers=num_total_train_speakers)

    # Load the pre-trained model and transfer to current model
    trainer.get_finetune_model(params.noload_var_list)

    trainer.train_tune_lr(args.train_dir, args.train_spklist, args.tune_period)
    trainer.close()
    tf.logging.info("Finish tuning.")
