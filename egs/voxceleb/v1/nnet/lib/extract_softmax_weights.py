import argparse
import numpy as np
import os
import sys
import numpy, scipy, sklearn
from model.trainer import Trainer
from misc.utils import Params
from dataset.kaldi_io import FeatureReader, open_or_fd, read_mat_ark, write_vec_flt
from six.moves import range
from dataset.data_loader import KaldiDataRandomQueue, KaldiDataSeqQueue, DataOutOfRange
from bhtsne import run_bh_tsne
import matplotlib.pyplot as plt
from collections import OrderedDict

# Example:
# python nnet/lib/extract_softmax_weights.py
#   exp/xvector_nnet data/train data/train/spklist weights embeddings pics

plt.switch_backend('agg')
parser = argparse.ArgumentParser()
parser.add_argument("model_dir", type=str, help="The model directory.")
parser.add_argument("data_dir", type=str, help="The data directory of the training set.")
parser.add_argument("data_spklist", type=str, help="The spklist file maps the TRAINING speakers to the indices.")
parser.add_argument("weights", type=str, help="The output weights")
parser.add_argument("embeddings", type=str, help="Embeddings (label vector).")
parser.add_argument("embedding_pic", type=str, help="The output pic")

args = parser.parse_args()
import tensorflow as tf

if __name__ == '__main__':
    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.INFO)

    nnet_dir = os.path.join(args.model_dir, "nnet")
    config_json = os.path.join(args.model_dir, "nnet/config.json")
    if not os.path.isfile(config_json):
        sys.exit("Cannot find params.json in %s" % config_json)
    params = Params(config_json)
    
    # params.loss_func = "generalized_angular_triplet_loss"
    # params.dict["triplet_center"] = "average"
    # params.dict["triplet_center_momentum"] = 0.9
    # params.dict["loss_compute"] = "softplus"
    # params.dict["margin"] = 0.1

    num_total_train_speakers = KaldiDataRandomQueue(args.data_dir, args.data_spklist).num_total_speakers
    dim = FeatureReader(args.data_dir).get_dim()
    trainer = Trainer(params, args.model_dir, single_cpu=True)
    trainer.build("valid",
                  dim=dim,
                  loss_type=params.loss_func,
                  num_speakers=num_total_train_speakers)
    # trainer.build("predict", dim=dim)

    # Load the model and output embeddings
    trainer.sess.run(tf.global_variables_initializer())
    trainer.sess.run(tf.local_variables_initializer())

    # load the weights
    curr_step = trainer.load()
    with tf.variable_scope("softmax", reuse=True):
        kernel = tf.get_variable("output/kernel", shape=[trainer.embeddings.get_shape()[-1], num_total_train_speakers])
        kernel_val = trainer.sess.run(kernel)
    weights = np.transpose(kernel_val)

    embeddings_val = None
    labels_val = None
    data_loader = KaldiDataSeqQueue(args.data_dir, args.data_spklist,
                                    num_parallel=1,
                                    max_qsize=10,
                                    batch_size=params.num_speakers_per_batch * params.num_segments_per_speaker,
                                    min_len=params.min_segment_len,
                                    max_len=params.max_segment_len,
                                    shuffle=False)
    data_loader.start()
    while True:
        try:
            features, labels = data_loader.fetch()
            valid_emb_val, valid_labels_val, _ = trainer.sess.run([trainer.embeddings, trainer.valid_labels, trainer.valid_ops["valid_loss_op"]],
                                                                feed_dict={trainer.valid_features: features,
                                                                           trainer.valid_labels: labels,
                                                                           trainer.global_step: curr_step})
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

    loss = trainer.sess.run(trainer.valid_ops["valid_loss"])
    tf.logging.info("Loss: %f" % loss)

    np.savetxt(args.weights, weights, fmt="%.4e")

    num_samples = embeddings_val.shape[0]
    dim_embedding = embeddings_val.shape[1]
    with open(args.embeddings, "w") as f:
        for i in range(num_samples):
            f.write("%d" % labels_val[i])
            for j in range(dim_embedding):
                f.write(" %.4e" % embeddings_val[i, j])
            f.write("\n")

    trainer.close()

    # Draw the pic
    # Normalize
    weights /= np.sqrt(np.sum(weights ** 2, axis=1, keepdims=True))
    embeddings_val /= np.sqrt(np.sum(embeddings_val ** 2, axis=1, keepdims=True))

    # We only get the weights we need
    index2center = OrderedDict()
    for i in range(num_samples):
        if labels_val[i] not in index2center:
            index2center[labels_val[i]] = weights[labels_val[i], :]

    weights_new = []
    weights_index = []
    for index in index2center:
        weights_index.append(index)
        weights_new.append(index2center[index])
    weights_new = np.stack(weights_new, axis=0)
    num_weights = len(weights_index)

    # tSNE
    combined = np.concatenate([weights_new, embeddings_val], axis=0)
    Y = run_bh_tsne(combined, no_dims=2, initial_dims=50)
    Y_weights = Y[:num_weights, :]
    Y_embeddings = Y[num_weights:, :]
    plt.figure(1)
    plt.scatter(Y_embeddings[:, 0], Y_embeddings[:, 1], c=labels_val)
    plt.scatter(Y_weights[:, 0], Y_weights[:, 1], marker="x")
    plt.savefig(args.embedding_pic)

