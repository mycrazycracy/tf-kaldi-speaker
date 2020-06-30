import argparse
import numpy as np
import os
import sys
import numpy, scipy, sklearn
from model.trainer_mi import TrainerMultiInput
from misc.utils import Params
from dataset.kaldi_io import open_or_fd, read_mat_ark, write_vec_flt, read_key, read_mat

parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gpu", type=int, default=-1, help="The GPU id. GPU disabled if -1.")
parser.add_argument("-m", "--min-chunk-size", type=int, default=25, help="The minimum length of the segments. Any segment shorted than this value will be ignored.")
parser.add_argument("-s", "--chunk-size", type=int, default=10000, help="The length of the segments used to extract the embeddings. "
                                                         "Segments longer than this value will be splited before extraction. "
                                                         "Then the splited embeddings will be averaged to get the final embedding. "
                                                         "L2 normalizaion will be applied before the averaging if specified.")
parser.add_argument("-n", "--normalize", action="store_true", help="Normalize the embedding before averaging and output.")
parser.add_argument("--node", type=str, default="", help="The node to output the embeddings.")
parser.add_argument("model_dir", type=str, help="The model directory.")
parser.add_argument("rspecifier", type=str, help="Kaldi feature rspecifier (or ark file).")
parser.add_argument("aux_rspecifier", type=str, help="Multiple auxiliary rspecifier, split by comma. For example, aux1:ark:rspecifier1.ark;aux2:ark:rspecifier2.ark")
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

    # Change the output node if necessary
    if len(args.node) != 0:
        params.embedding_node = args.node
    tf.logging.info("Extract embedding from %s" % params.embedding_node)

    trainer = TrainerMultiInput(params, args.model_dir, single_cpu=True)

    with open(os.path.join(nnet_dir, "feature_dim"), "r") as f:
        dim = int(f.readline().strip())
    trainer.build("predict", dim=dim)

    if args.rspecifier.rsplit(".", 1)[1] == "scp":
        # The rspecifier cannot be scp
        sys.exit("The rspecifier must be ark or input pipe")

    aux_rspecifier_split = args.aux_rspecifier.split(";")
    aux_rspecifiers = {}
    n_fields = len(aux_rspecifier_split)
    for rs in aux_rspecifier_split:
        name, p = rs.split(":", 1)
        if p.rsplit(".", 1)[1] == "scp":
            sys.exit("The auxiliary rspecifier must be ark or input pipe, got %s" % p)
        aux_rspecifiers[name] = open_or_fd(p)

    fp_out = open_or_fd(args.wspecifier)
    for index, (key, feature) in enumerate(read_mat_ark(args.rspecifier)):
        if feature.shape[0] < args.min_chunk_size:
            tf.logging.info("[INFO] Key %s length too short, %d < %d, skip." % (key, feature.shape[0], args.min_chunk_size))
            continue

        aux_features = {"features": feature}
        for name in aux_rspecifiers:
            aux_key = read_key(aux_rspecifiers[name])
            assert aux_key == key, "The key %s dosen't match key %s" % (key, aux_key)
            aux_features[name] = read_mat(aux_rspecifiers[name])

        if feature.shape[0] > args.chunk_size:
            # Now we have multiple features
            features_array = {}
            for name in aux_features:
                features_array[name] = []
            feature_length = []
            num_chunks = int(np.ceil(float(feature.shape[0] - args.chunk_size) / (args.chunk_size / 2))) + 1
            tf.logging.info("[INFO] Key %s length %d > %d, split to %d segments." % (key, feature.shape[0], args.chunk_size, num_chunks))
            for i in range(num_chunks):
                start = i * (args.chunk_size / 2)
                this_chunk_size = args.chunk_size if feature.shape[0] - start > args.chunk_size else feature.shape[0] - start
                feature_length.append(this_chunk_size)
                for name in aux_features:
                    features_array[name].append(aux_features[name][start:start + this_chunk_size])

            feature_length = np.expand_dims(np.array(feature_length), axis=1)
            # Except for the last feature, the length of other features should be the same (=chunk_size)
            features_array_expect_last = {}
            for name in aux_features:
                features_array_expect_last[name] = np.array(features_array[name][:-1], dtype=np.float32)
            features_array_last = {}
            for name in aux_features:
                features_array_last[name] = features_array[name][-1]

            embeddings = trainer.predict(features_array_expect_last)
            embedding_last = trainer.predict(features_array_last)
            embeddings = np.concatenate([embeddings, np.expand_dims(embedding_last, axis=0)], axis=0)
            if args.normalize:
                embeddings /= np.sqrt(np.sum(np.square(embeddings), axis=1, keepdims=True))
            embedding = np.sum(embeddings * feature_length, axis=0) / np.sum(feature_length)
        else:
            tf.logging.info("[INFO] Key %s length %d." % (key, feature.shape[0]))
            embedding = trainer.predict(aux_features)

        if args.normalize:
            embedding /= np.sqrt(np.sum(np.square(embedding)))
        write_vec_flt(fp_out, embedding, key=key)
    fp_out.close()
    trainer.close()
