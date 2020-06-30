import os
import argparse
import random
import sys
import tensorflow as tf
import numpy as np
from misc.utils import ValidLoss, load_valid_loss, save_codes_and_config, compute_cos_pairwise_eer
from dataset.multitask.data_loader_v2 import KaldiDataRandomQueueV2
from dataset.kaldi_io import FeatureReaderV2
from six.moves import range

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--cont", action="store_true", help="Continue training from an existing model.")
parser.add_argument("--tune_period", type=int, default=100, help="How many steps per learning rate.")
parser.add_argument("--config", type=str, help="The configuration file.")
parser.add_argument("train_data_dir", type=str, help="The data directory of the training set.")
parser.add_argument("train_ali_dir", type=str, help="The ali directory of the training set.")
parser.add_argument("train_spklist", type=str, help="The spklist file maps the TRAINING speakers to the indices.")
parser.add_argument("model", type=str, help="The output model directory.")


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    args = parser.parse_args()
    params = save_codes_and_config(False, args.model, args.config)

    # The model directory always has a folder named nnet
    model_dir = os.path.join(args.model, "nnet")

    # Set the random seed. The random operations may appear in data input, batch forming, etc.
    tf.set_random_seed(params.seed)
    random.seed(params.seed)
    np.random.seed(params.seed)

    start_epoch = 0

    feat_reader = FeatureReaderV2(args.train_data_dir, args.train_ali_dir)
    dim = feat_reader.get_dim()

    feat_reader = KaldiDataRandomQueueV2(args.train_data_dir, args.train_ali_dir, args.train_spklist)
    num_total_speakers = feat_reader.num_total_speakers
    num_total_phones = feat_reader.num_total_phones

    from model.multitask_v1.base_v1 import BaseMT

    trainer = BaseMT(params, args.model, dim, num_total_speakers, num_total_phones)
    trainer.build("train")
    trainer.train_tune_lr(args.train_data_dir, args.train_ali_dir, args.train_spklist, args.tune_period)
    trainer.close()
    tf.logging.info("Finish tuning.")
