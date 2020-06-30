import os
import argparse
import random
import sys
import numpy, scipy, sklearn
import tensorflow as tf
import numpy as np
from misc.utils import save_codes_and_config, compute_cos_pairwise_eer
from model.trainer import Trainer
from dataset.data_loader import KaldiDataRandomQueue
from dataset.kaldi_io import FeatureReader


parser = argparse.ArgumentParser()
parser.add_argument("data_dir", type=str, help="The data directory of the dataset.")
parser.add_argument("data_spklist", type=str, help="The spklist maps the speakers to the indices.")
parser.add_argument("model", type=str, help="The output model directory.")


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    args = parser.parse_args()
    params = save_codes_and_config(True, args.model, None)

    # The model directory always has a folder named nnet
    model_dir = os.path.join(args.model, "nnet")

    # Set the random seed. The random operations may appear in data input, batch forming, etc.
    tf.set_random_seed(params.seed)
    random.seed(params.seed)
    np.random.seed(params.seed)

    dim = FeatureReader(args.data_dir).get_dim()
    with open(args.data_spklist, 'r') as f:
        num_total_train_speakers = len(f.readlines())
    trainer = Trainer(params, args.model)
    trainer.build("valid",
                  dim=dim,
                  loss_type=params.loss_func,
                  num_speakers=num_total_train_speakers)
    # valid_loss, valid_embeddings, valid_labels = trainer.valid(args.data_dir, args.data_spklist,
    #                                                            batch_type=params.batch_type,
    #                                                            output_embeddings=True)

    valid_loss, valid_embeddings, valid_labels = trainer.insight(args.data_dir, args.data_spklist,
                                                     batch_type=params.batch_type,
                                                     output_embeddings=True)
    eer = compute_cos_pairwise_eer(valid_embeddings, valid_labels)
    tf.logging.info("EER: %f" % eer)
    trainer.close()
