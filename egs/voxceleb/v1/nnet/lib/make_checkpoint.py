import argparse
import os
import sys
from misc.utils import get_checkpoint
from misc.utils import Params
import tensorflow as tf

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint", type=str, default='-1',
                        help="The checkpoint to load. The default is to load the BEST checkpoint (according to valid_loss).")
    parser.add_argument("model_dir", type=str, help="The model directory.")
    args = parser.parse_args()
    checkpoint = get_checkpoint(os.path.join(args.model_dir, "nnet"), args.checkpoint)
    print("Set the checkpoint to %s" % checkpoint)
