import argparse
import numpy as np
import os
import sys
from misc.utils import Params
from dataset.kaldi_io import open_or_fd, read_mat_ark, read_vec_int, write_mat
from six.moves import range

parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gpu", type=int, default=-1, help="The GPU id. GPU disabled if -1.")
parser.add_argument("-s", "--chunk-size", type=int, default=100000,
                    help="The length of the segments used to extract the embeddings."
                    "This is useful if the utterance is too long.")
parser.add_argument("prior", type=str, help="The prior of the senones.")
parser.add_argument("model_dir", type=str, help="The model directory.")
parser.add_argument("rspecifier", type=str, help="Kaldi feature rspecifier (or ark file).")
parser.add_argument("wspecifier", type=str, help="Kaldi output wspecifier (or ark file).")
args = parser.parse_args()

if args.gpu == -1:
    # Disable GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# In the GPU situation, it is difficult to know how to specify the GPU id.
# If the program is launched locally, you can set CUDA_VISIBLE_DEVICES to the id.
# However, if SGE is used, we cannot simply set CUDA_VISIBLE_DEVICES.
# So it is better to specify the GPU id outside the program.
# Give an arbitrary number (except for -1) to --gpu can enable it. Leave it blank if you want to disable gpu.

import tensorflow as tf


if __name__ == '__main__':
    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.INFO)
    nnet_dir = os.path.join(args.model_dir, "nnet")

    config_json = os.path.join(args.model_dir, "nnet/config.json")
    if not os.path.isfile(config_json):
        sys.exit("Cannot find params.json in %s" % config_json)
    params = Params(config_json)

    with open(os.path.join(nnet_dir, "feature_dim"), "r") as f:
        dim = int(f.readline().strip())
    with open(os.path.join(nnet_dir, "num_speakers"), "r") as f:
        num_total_speakers = int(f.readline().strip())
    with open(os.path.join(nnet_dir, "num_phones"), "r") as f:
        num_total_phones = int(f.readline().strip())

    from model.multitask_v1.base_v1 import BaseMT

    trainer = BaseMT(params, args.model_dir, dim, num_total_speakers, num_total_phones, single_cpu=True)
    trainer.build("predict")
    # Special node: we need the log-probability of the output layer.
    node = "log-output"

    # Load the prior from file
    with open(args.prior, "r") as f:
        prior_vec = np.array([float(i) for i in f.readline().strip().strip("[]").strip().split(" ")], dtype=np.float64)
    # Sanity check on the prior and convert to log
    assert(np.allclose(np.sum(prior_vec), 1.0))
    # Sine the prior is floored during training, it is safe to apply log on the prior
    log_prior_vec = np.log(prior_vec)[np.newaxis, :]

    if args.rspecifier.rsplit(".", 1)[1] == "scp":
        # The rspecifier cannot be scp
        sys.exit("The rspecifier must be ark or input pipe.")

    num_done = 0

    fp_out = open_or_fd(args.wspecifier, "wb")
    for index, (key, feature) in enumerate(read_mat_ark(args.rspecifier)):
        if feature.shape[0] > args.chunk_size:
            # feature_array = []
            # ali_array = []
            # feature_length = []
            # num_chunks = int(np.ceil(float(feature.shape[0]) / args.chunk_size))
            # tf.logging.info("[INFO] Key %s length %d > %d, split to %d segments." % (
            #                 key, feature.shape[0], args.chunk_size, num_chunks))
            # for i in range(num_chunks):
            #     start = i * args.chunk_size
            #     this_chunk_size = args.chunk_size if feature.shape[0] - start > args.chunk_size else feature.shape[
            #                                                                                              0] - start
            #     feature_length.append(this_chunk_size)
            #     feature_array.append(feature[start:start + this_chunk_size])
            #
            # # Except for the last feature, the length of other features should be the same (=chunk_size)
            # log_prob = trainer.predict_phone(node,
            #                                    np.array(feature_array[:-1], dtype=np.float32),
            #                                    feature_length[:-1])
            # log_prob_last = trainer.predict_phone(node, feature_array[-1], [feature_length[-1]])
            #
            # log_prob = np.reshape(log_prob, [log_prob.shape[0] * log_prob.shape[1], log_prob.shape[2]])
            # log_prob = np.concatenate([log_prob, log_prob_last], axis=0)
            # assert(log_prob.shape[0] == feature.shape[0] and log_prob.shape[1] == prior_vec.shape[0])

            raise NotImplementedError("Do not let the utterance to be split.")
        else:
            tf.logging.info("[INFO] Key %s length %d." % (key, feature.shape[0]))
            log_prob = trainer.predict_phone(node, feature, [feature.shape[0]])
            assert(log_prob.shape[0] == feature.shape[0] and log_prob.shape[1] == prior_vec.shape[0])

        # Convert to log-posteriors to log-likelihood
        log_like = log_prob - log_prior_vec
        write_mat(fp_out, log_like, key=key)
        num_done += 1

    fp_out.close()
    trainer.close()
    tf.logging.info("Compute %d log-likelihood." % (num_done))
