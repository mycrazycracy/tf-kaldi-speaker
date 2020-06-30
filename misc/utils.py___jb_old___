import json
from sklearn import metrics
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import numpy as np
import tensorflow as tf
from distutils.dir_util import copy_tree
import os
import sys
import shutil
from six.moves import range

class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        self.update(json_path)

    def save(self, json_path):
        """Saves parameters to json file"""
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__


class ParamsPlain():
    """Class that saves hyperparameters manually.
    This is used to debug the code since we don't have the json file to feed the parameters.

    Example:
    ```
    params = ParamsPlain()
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self):
        pass

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__


def save_codes_and_config(cont, model, config):
    """Save the codes and configuration file.

    During the training, we may modify the codes. It will be problematic when we try to extract embeddings using the old
    model and the new code. So we save the codes when we train the model and use the saved codes to extract embeddings.

    Args:
        cont: bool, continue training.
        model: the entire model directory (including nnet/).
        config: the config file.
    :return: A structure params.
    """
    if cont:
        # If we want to continue the model training, we need to check the existence of the checkpoint.
        if not os.path.isdir(os.path.join(model, "nnet")) or not os.path.isdir(os.path.join(model, "codes")):
            sys.exit("To continue training the model, nnet and codes must be existed in %s." % model)
        # Simply load the configuration from the saved model.
        tf.logging.info("Continue training from %s." % model)
        params = Params(os.path.join(model, "nnet/config.json"))
    else:
        # Save the codes in the model directory so that it is more convenient to extract the embeddings.
        # The codes would be changed when we extract the embeddings, making the network loading impossible.
        # When we want to extract the embeddings, we should use the code in `model/codes/...`
        if os.path.isdir(os.path.join(model, "nnet")):
            # Backup the codes and configuration in .backup. Keep the model unchanged.
            tf.logging.info("Save backup to %s" % os.path.join(model, ".backup"))
            if os.path.isdir(os.path.join(model, ".backup")):
                tf.logging.warn("The dir %s exisits. Delete it and continue." % os.path.join(model, ".backup"))
                shutil.rmtree(os.path.join(model, ".backup"))
            os.makedirs(os.path.join(model, ".backup"))
            if os.path.exists(os.path.join(model, "codes")):
                shutil.move(os.path.join(model, "codes"), os.path.join(model, ".backup/"))
            if os.path.exists(os.path.join(model, "nnet")):
                shutil.move(os.path.join(model, "nnet"), os.path.join(model, ".backup/"))

        # `model/codes` is used to save the codes and `model/nnet` is used to save the model and configuration
        if os.path.isdir(os.path.join(model, "codes")):
            shutil.rmtree(os.path.join(model, "codes"))
        os.makedirs(os.path.join(model, "codes"))

        # We need to set the home directory of the tf-kaldi-speaker (TF_KALDI_ROOT).
        if not os.environ.get('TF_KALDI_ROOT'):
            tf.logging.error("TF_KALDI_ROOT should be set before training. Refer to path.sh to set the value manually. ")
            quit()
        copy_tree(os.path.join(os.environ['TF_KALDI_ROOT'], "dataset"), os.path.join(model, "codes/dataset/"))
        copy_tree(os.path.join(os.environ['TF_KALDI_ROOT'], "model"), os.path.join(model, "codes/model/"))
        copy_tree(os.path.join(os.environ['TF_KALDI_ROOT'], "misc"), os.path.join(model, "codes/misc/"))
        if not os.path.isdir(os.path.join(model, "nnet")):
            os.makedirs(os.path.join(model, "nnet"))
        shutil.copyfile(config, os.path.join(model, "nnet", "config.json"))
        tf.logging.info("Train the model from scratch.")
        params = Params(config)
    return params


def get_pretrain_model(pretrain_model, target_model, checkpoint='-1'):
    """Get the pre-trained model and copy to the target model as the initial version.

        Note: After the copy, the checkpoint becomes 0.
    Args:
        pretrain_model: The pre-trained model directory.
        target_model: The target model directory.
        checkpoint: The checkpoint in the pre-trained model directory. If None, set to the BEST one. Also support "last"
    """
    if not os.path.isfile(os.path.join(pretrain_model, "checkpoint")):
        sys.exit("[ERROR] Cannot find checkpoint in %s." % pretrain_model)
    ckpt = tf.train.get_checkpoint_state(pretrain_model)

    model_checkpoint_path = ckpt.model_checkpoint_path
    all_model_checkpoint_paths = ckpt.all_model_checkpoint_paths

    if not ckpt or not model_checkpoint_path:
        sys.exit("[ERROR] Cannot read checkpoint %s." % os.path.join(pretrain_model, "checkpoint"))

    steps = [int(c.rsplit('-', 1)[1]) for c in all_model_checkpoint_paths]
    steps = sorted(steps)

    if checkpoint == "last":
        tf.logging.info("Load the last saved model.")
        checkpoint = steps[-1]
    else:
        checkpoint = int(checkpoint)
        if checkpoint == -1:
            tf.logging.info("Load the best model according to valid_loss")
            min_epoch = -1
            min_loss = 1e10
            with open(os.path.join(pretrain_model, "valid_loss")) as f:
                for line in f.readlines():
                    epoch, loss, eer = line.split(" ")
                    epoch = int(epoch)
                    loss = float(loss)
                    if loss < min_loss:
                        min_loss = loss
                        min_epoch = epoch
                # Add 1 to min_epoch since epoch is 0-based
                config_json = os.path.join(pretrain_model, "config.json")
                params = Params(config_json)
                checkpoint = (min_epoch + 1) * params.num_steps_per_epoch
    assert checkpoint in steps, "The checkpoint %d not in the model directory" % checkpoint

    pretrain_model_checkpoint_path = model_checkpoint_path.rsplit("-", 1)[0] + "-" + str(checkpoint)
    tf.logging.info("Copy the pre-trained model %s as the fine-tuned initialization" % pretrain_model_checkpoint_path)

    import glob
    for filename in glob.glob(pretrain_model_checkpoint_path + "*"):
        bas = os.path.basename(filename).split("-", 1)[0]
        ext = os.path.basename(filename).rsplit(".", 1)[1]
        shutil.copyfile(filename, os.path.join(target_model, bas + "-0." + ext))

    with open(os.path.join(target_model, "checkpoint"), "w") as f:
        f.write("model_checkpoint_path: \"%s\"\n" % os.path.join(target_model, os.path.basename(model_checkpoint_path).rsplit("-", 1)[0] + "-0"))
        f.write("all_model_checkpoint_paths: \"%s\"\n" % os.path.join(target_model, os.path.basename(model_checkpoint_path).rsplit("-", 1)[0] + "-0"))
    return


class ValidLoss():
    """Class that save the valid loss history"""
    def __init__(self):
        self.min_loss = 1e16
        self.min_loss_epoch = -1


def load_lr(filename):
    """Load learning rate from a saved file"""
    learning_rate_array = []
    with open(filename, "r") as f:
        for line in f.readlines():
            _, lr = line.strip().split(" ")
            learning_rate_array.append(float(lr))
    return learning_rate_array


def load_valid_loss(filename):
    """Load valid loss from a saved file"""
    min_loss = ValidLoss()
    with open(filename, "r") as f:
        for line in f.readlines():
            epoch, loss = line.strip().split(" ")[:2]
            epoch = int(epoch)
            loss = float(loss)
            if loss < min_loss.min_loss:
                min_loss.min_loss = loss
                min_loss.min_loss_epoch = epoch
    return min_loss


def get_checkpoint(model, checkpoint='-1'):
    """Set the checkpoint in the model directory and return the name of the checkpoint
    Note: This function will modify `checkpoint` in the model directory.

    Args:
        model: The model directory.
        checkpoint: The checkpoint id. If None, set to the BEST one. Also support "last"
    :return: The name of the checkpoint.
    """
    if not os.path.isfile(os.path.join(model, "checkpoint")):
        sys.exit("[ERROR] Cannot find checkpoint in %s." % model)
    ckpt = tf.train.get_checkpoint_state(model)

    model_checkpoint_path = ckpt.model_checkpoint_path
    all_model_checkpoint_paths = ckpt.all_model_checkpoint_paths

    if not ckpt or not model_checkpoint_path:
        sys.exit("[ERROR] Cannot read checkpoint %s." % os.path.join(model, "checkpoint"))

    steps = [int(c.rsplit('-', 1)[1]) for c in all_model_checkpoint_paths]
    steps = sorted(steps)
    if checkpoint == "last":
        tf.logging.info("Load the last saved model.")
        checkpoint = steps[-1]
    else:
        checkpoint = int(checkpoint)
        if checkpoint == -1:
            tf.logging.info("Load the best model according to valid_loss")
            min_epoch = -1
            min_loss = 1e10
            with open(os.path.join(model, "valid_loss")) as f:
                for line in f.readlines():
                    epoch, loss, eer = line.split(" ")
                    epoch = int(epoch)
                    loss = float(loss)
                    if loss < min_loss:
                        min_loss = loss
                        min_epoch = epoch
                # Add 1 to min_epoch since epoch is 0-based
                config_json = os.path.join(model, "config.json")
                params = Params(config_json)
                checkpoint = (min_epoch + 1) * params.num_steps_per_epoch
    tf.logging.info("The checkpoint is %d" % checkpoint)
    assert checkpoint in steps, "The checkpoint %d not in the model directory" % checkpoint

    model_checkpoint_path = model_checkpoint_path.rsplit("-", 1)[0] + "-" + str(checkpoint)
    model_checkpoint_path = os.path.join(model, os.path.basename(model_checkpoint_path))

    with open(os.path.join(model, "checkpoint"), "w") as f:
        f.write("model_checkpoint_path: \"%s\"\n" % model_checkpoint_path)
        for checkpoint in all_model_checkpoint_paths:
            checkpoint_new = os.path.join(model, os.path.basename(checkpoint))
            f.write("all_model_checkpoint_paths: \"%s\"\n" % checkpoint_new)
    return model_checkpoint_path


def compute_cos_pairwise_eer(embeddings, labels, max_num_embeddings=1000):
    """Compute pairwise EER using cosine similarity.
    The EER is estimated by interp1d and brentq, so it is not the exact value and may be a little different each time.

    Args:
        embeddings: The embeddings.
        labels: The class labels.
        max_num_embeddings: The max number of embeddings to compute the EER.
    :return: The pairwise EER.
    """
    embeddings /= np.sqrt(np.sum(embeddings ** 2, axis=1, keepdims=True) + 1e-12)
    num_embeddings = embeddings.shape[0]
    if num_embeddings > max_num_embeddings:
        # Downsample the embeddings and labels
        step = num_embeddings / max_num_embeddings
        embeddings = embeddings[range(0, num_embeddings, step), :]
        labels = labels[range(0, num_embeddings, step)]
        num_embeddings = embeddings.shape[0]

    score_mat = np.dot(embeddings, np.transpose(embeddings))
    scores = np.zeros((num_embeddings * (num_embeddings - 1) / 2))
    keys = np.zeros((num_embeddings * (num_embeddings - 1) / 2))
    index = 0
    for i in range(num_embeddings - 1):
        for j in range(i + 1, num_embeddings):
            scores[index] = score_mat[i, j]
            keys[index] = 1 if labels[i] == labels[j] else 0
            index += 1

    fpr, tpr, thresholds = metrics.roc_curve(keys, scores, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    # thresh = interp1d(fpr, thresholds)(eer)

    with open("test.txt", "w") as f:
        for i in range(num_embeddings):
            if keys[i] == 1:
                f.write("%f target" % scores[i])
            else:
                f.write("%f nontarget" % scores[i])
    return eer


def substring_in_list(s, varlist):
    """Check whether part of the string s appears in the list.

    Args:
        s: A string
        varlist: A list. Some elements may be the sub-string of s.
    :return: Bool. Is a element in the varlist is the substring of s?
    """
    if varlist is None:
        return False
    is_sub = False
    for v in varlist:
        if v in s:
            is_sub = True
            break
    return is_sub


def activation_summaries(endpoints):
    """Create a summary for activations given the endpoints.

    Args:
        endpoints: The endpoints from the model.
    :return: A tf summary.
    """
    sum = []
    with tf.name_scope('summaries'):
        for act in endpoints.values():
            tensor_name = act.op.name
            sum.append(tf.summary.histogram(tensor_name + '/activations', act))
            # sum.append(tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(act)))
    return tf.summary.merge(sum)


def remove_params_prefix(params, prefix):
    new_params = ParamsPlain()
    prefix += '_'
    l = len(prefix)
    for key in params.dict:
        new_key = key
        if key[:l] == prefix:
            new_key = key[l:]
        new_params.dict[new_key] = params.dict[key]
    return new_params


def add_dict_prefix(d, prefix):
    new_d = {}
    for key in d:
        new_key = prefix + "_" + key
        new_d[new_key] = d[key]
    return new_d
