# import argparse
# import numpy as np
# import os
# import sys
# from misc.utils import Params
# from dataset.kaldi_io import open_or_fd, read_mat_ark, read_vec_int, write_mat
# from six.moves import range
#
# # TODO: We don't need alignment to extract the phone embedding. Delete the ali parameter.
# # TODO: We may need to re-write this script
#
# parser = argparse.ArgumentParser()
# parser.add_argument("-g", "--gpu", type=int, default=-1, help="The GPU id. GPU disabled if -1.")
# parser.add_argument("-m", "--min-chunk-size", type=int, default=10,
#                     help="The minimum length of the segments. Any segment shorted than this value will be ignored.")
# parser.add_argument("-s", "--chunk-size", type=int, default=10000,
#                     help="The length of the segments used to extract the embeddings. "
#                          "Segments longer than this value will be splited before extraction. "
#                          "Then the splited embeddings will be averaged to get the final embedding. "
#                          "L2 normalizaion will be applied before the averaging if specified.")
# parser.add_argument("-n", "--normalize", action="store_true",
#                     help="Normalize the embedding before averaging and output.")
# parser.add_argument("--node", type=str, help="The node to output the embeddings.")
# parser.add_argument("model_dir", type=str, help="The model directory.")
# parser.add_argument("rspecifier", type=str, help="Kaldi feature rspecifier (or ark file).")
# parser.add_argument("ali_rspecifier", type=str, help="Kaldi ali rspecifier.")
# parser.add_argument("wspecifier", type=str, help="Kaldi output wspecifier (or ark file).")
#
# args = parser.parse_args()
#
# if args.gpu == -1:
#     # Disable GPU
#     os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#
# # In the GPU situation, it is difficult to know how to specify the GPU id.
# # If the program is launched locally, you can set CUDA_VISIBLE_DEVICES to the id.
# # However, if SGE is used, we cannot simply set CUDA_VISIBLE_DEVICES.
# # So it is better to specify the GPU id outside the program.
# # Give an arbitrary number (except for -1) to --gpu can enable it. Leave it blank if you want to disable gpu.
#
# import tensorflow as tf
#
#
# def read_ali(fd):
#     line = fd.readline()
#     key = None
#     ali = None
#     if line:
#         key, rxfile = line.decode().split(' ')
#         ali = read_vec_int(rxfile)
#     return key, ali
#
#
# if __name__ == '__main__':
#     tf.reset_default_graph()
#     tf.logging.set_verbosity(tf.logging.INFO)
#
#     nnet_dir = os.path.join(args.model_dir, "nnet")
#
#     config_json = os.path.join(args.model_dir, "nnet/config.json")
#     if not os.path.isfile(config_json):
#         sys.exit("Cannot find params.json in %s" % config_json)
#     params = Params(config_json)
#
#     if args.normalize:
#         tf.logging.info("Normalize the embedding to L2=1.")
#     tf.logging.info("Extract embedding from %s." % args.node)
#
#     with open(os.path.join(nnet_dir, "feature_dim"), "r") as f:
#         dim = int(f.readline().strip())
#     with open(os.path.join(nnet_dir, "num_speakers"), "r") as f:
#         num_total_speakers = int(f.readline().strip())
#     with open(os.path.join(nnet_dir, "num_phones"), "r") as f:
#         num_total_phones = int(f.readline().strip())
#
#     from model.multitask_v1.base_v1 import BaseMT
#
#     trainer = BaseMT(params, args.model_dir, dim, num_total_speakers, num_total_phones, single_cpu=True)
#     trainer.build("predict")
#     tf.logging.info("Extract embeddings (or outputs) from node %s" % args.node)
#     assert args.node in trainer.endpoints, "The node %s is not in the endpoints" % args.node
#
#     if args.rspecifier.rsplit(".", 1)[1] == "scp":
#         # The rspecifier cannot be scp
#         sys.exit("The rspecifier must be ark or input pipe.")
#
#     if args.ali_rspecifier.rsplit(".", 1)[1] != "scp":
#         sys.exit("The ali-rspecifier is expected to be scp file.")
#
#     num_err = 0
#     num_done = 0
#     # Preload the first alignment.
#     fp_ali = open_or_fd(args.ali_rspecifier)
#     ali_key, ali_value = read_ali(fp_ali)
#
#     fp_out = open_or_fd(args.wspecifier, "wb")
#     for index, (key, feature) in enumerate(read_mat_ark(args.rspecifier)):
#         # if feature.shape[0] < args.min_chunk_size:
#         #     tf.logging.info(
#         #         "[INFO] Key %s length too short, %d < %d, skip." % (key, feature.shape[0], args.min_chunk_size))
#         #     continue
#
#         # The alignments are assumed to be less than the features (due to decoding failure).
#         if ali_key != key:
#             tf.logging.warn("Cannot find the ali for %s." % key)
#             num_err += 1
#             continue
#
#         if feature.shape[0] > args.chunk_size:
#             feature_array = []
#             ali_array = []
#             feature_length = []
#             num_chunks = int(np.ceil(float(feature.shape[0]) / args.chunk_size))
#             tf.logging.info("[INFO] Key %s length %d > %d, split to %d segments." % (
#                             key, feature.shape[0], args.chunk_size, num_chunks))
#             for i in range(num_chunks):
#                 start = i * args.chunk_size
#                 this_chunk_size = args.chunk_size if feature.shape[0] - start > args.chunk_size else feature.shape[
#                                                                                                          0] - start
#                 feature_length.append(this_chunk_size)
#                 feature_array.append(feature[start:start + this_chunk_size])
#                 ali_array.append(ali_value[start:start + this_chunk_size])
#
#             # Except for the last feature, the length of other features should be the same (=chunk_size)
#             embeddings = trainer.predict_phone(args.node,
#                                          np.array(feature_array[:-1], dtype=np.float32),
#                                          np.array(ali_array[:-1], dtype=np.int32),
#                                          feature_length[:-1])
#             embedding_last = trainer.predict_phone(args.node, feature_array[-1], ali_array[-1], [feature_length[-1]])
#
#             feature_length = np.expand_dims(np.array(feature_length), axis=1)
#             if len(embeddings.shape) == 3:
#                 embeddings = np.reshape(embeddings, [embeddings.shape[0] * embeddings.shape[1], embeddings.shape[2]])
#                 embedding = np.concatenate([embeddings, embedding_last], axis=0)
#             else:
#                 sys.exit("It is speaker-related task. Use extract_mt.py instead.")
#         else:
#             tf.logging.info("[INFO] Key %s length %d." % (key, feature.shape[0]))
#             embedding = trainer.predict(args.node, feature, ali_value, [feature.shape[0]])
#             if len(embedding.shape) == 1:
#                 sys.exit("It is speaker-related task. Use extract_mt.py instead.")
#
#         if args.normalize:
#             embedding /= np.sqrt(np.sum(np.square(embedding), axis=-1, keepdims=True))
#
#         # Write mat or vec
#         assert len(embedding.shape) == 2
#         write_mat(fp_out, embedding, key=key)
#
#         num_done += 1
#
#         # Read the next alignment.
#         ali_key, ali_value = read_ali(fp_ali)
#         if ali_key is None:
#             # This is the last alignment.
#             break
#
#     fp_out.close()
#     fp_ali.close()
#     trainer.close()
#     tf.logging.info("Extract %d embeddings, %d errors" % (num_done, num_err))
