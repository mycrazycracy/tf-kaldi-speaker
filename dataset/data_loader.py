import tensorflow as tf
import os
import random
import numpy as np
from multiprocessing import Process, Queue, Event
from dataset.kaldi_io import FeatureReader
from six.moves import range
import time

class DataOutOfRange(Exception):
    pass


def get_speaker_info(data, spklist):
    """Get speaker information from the data directory.

    This function will be used in KaldiDataReader and KaldiDataQueue. So make it a normal function rather than a class
    method would be fine.

    Args:
        data: The kaldi data directory.
        spklist: The spklist file gives the index of each speaker.
    :return:
        spk2features: A dict. The key is the speaker id and the value is the segments belonging to this speaker.
        features2spk: A dict. The key is the segment and the value is the corresponding speaker id.
        spk2index: A dict from speaker NAME to speaker ID. This is useful to get the number of speakers. Because
                   sometimes, the speakers are not all included in the data directory (like in the valid set).

        segment format: "utt_name filename:offset"
    """
    assert (os.path.isdir(data) and os.path.isfile(spklist))
    spk2index = {}
    with open(spklist, "r") as f:
        for line in f.readlines():
            spk, index = line.strip().split(" ")
            spk2index[spk] = int(index)

    utt2spk = {}
    with open(os.path.join(data, "spk2utt"), "r") as f:
        for line in f.readlines():
            spk, utts = line.strip().split(" ", 1)
            for utt in utts.split(" "):
                utt2spk[utt] = spk2index[spk]

    spk2features = {}
    features2spk = {}
    with open(os.path.join(data, "feats.scp"), "r") as f:
        for line in f.readlines():
            (key, rxfile) = line.decode().split(' ')
            spk = utt2spk[key]
            if spk not in spk2features:
                spk2features[spk] = []
            spk2features[spk].append(key + ' ' + rxfile)
            features2spk[key + ' ' + rxfile] = spk
    return spk2features, features2spk, spk2index


def get_aux_speaker_info(data, aux_data, spklist):
    """Get speaker information and auxiliary features from the data directory.
    This function is similar to the above one, while it also loads auxiliary features.

    Args:
        data: The kaldi data directory.
        aux_data: A dict contains arbitrary number of auxiliary auxiliary data directories.
        spklist: The spklist file gives the index of each speaker.
    :return:
        spk2features: A dict. The key is the speaker id and the value is the segments
                      and auxiliary features belonging to this speaker.
                      spk2features[spk] is a list, each element a dict.
                      The normal feature is in spk2features[spk][n]["features"]
        features2spk: A dict. The key is the segment and the value is the corresponding speaker id.
        spk2index: A dict from speaker NAME to speaker ID. This is useful to get the number of speakers. Because
                   sometimes, the speakers are not all included in the data directory (like in the valid set).
    """
    assert (os.path.isdir(data) and os.path.isfile(spklist))
    spk2index = {}
    with open(spklist, "r") as f:
        for line in f.readlines():
            spk, index = line.strip().split(" ")
            spk2index[spk] = int(index)

    utt2spk = {}
    with open(os.path.join(data, "spk2utt"), "r") as f:
        for line in f.readlines():
            spk, utts = line.strip().split(" ", 1)
            for utt in utts.split(" "):
                utt2spk[utt] = spk2index[spk]

    # Load auxiliary features.
    aux_utt2features = {}
    for name in aux_data:
        with open(os.path.join(aux_data[name], "feats.scp"), "r") as f:
            for line in f.readlines():
                (key, rxfile) = line.decode().split(' ')
                if key not in aux_utt2features:
                    aux_utt2features[key] = {}
                aux_utt2features[key][name] = key + ' ' + rxfile

    spk2features = {}
    features2spk = {}
    with open(os.path.join(data, "feats.scp"), "r") as f:
        for line in f.readlines():
            (key, rxfile) = line.decode().split(' ')
            spk = utt2spk[key]
            if spk not in spk2features:
                spk2features[spk] = []
            features2spk[key + ' ' + rxfile] = spk
            aux_utt2features[key]["features"] = key + ' ' + rxfile
            spk2features[spk].append(aux_utt2features[key])
    return spk2features, features2spk, spk2index


class KaldiDataRandomReader(object):
    """Used to read data from a kaldi data directory."""

    def __init__(self, data_dir, spklist, num_parallel=1, num_speakers=None, num_segments=None, min_len=None, max_len=None, shuffle=True):
        """ Create a data_reader from a given directory.

        Args:
            data_dir: The kaldi data directory.
            spklist: The spklist tells the relation between speaker and index.
            num_parallel: The number of threads to read features.
            num_speakers: The number of speakers per batch.
            num_segments: The number of semgents per speaker.
              batch_size = num_speakers * num_segments
              When num_segments = 1, the batch is randomly chosen from n speakers,
              which is used for softmax-like loss function. While we can sample multiple segments for each speaker,
              which is used for triplet-loss or GE2E loss.
            min_len: The minimum length of the sampled sequence.
            max_len: The maximum length of the sampled sequence.
            shuffle: Load the feature from the 0-th frame or a random frame.
        """
        self.data = data_dir
        self.num_speakers = num_speakers
        self.num_segments = num_segments
        self.min_len = min_len
        self.max_len = max_len
        self.shuffle = shuffle

        # We process the data directory and fetch speaker information
        self.dim = FeatureReader(data_dir).get_dim()
        self.spk2features, self.features2spk, spk2index = get_speaker_info(data_dir, spklist)
        self.speakers = list(self.spk2features.keys())
        self.num_total_speakers = len(list(spk2index.keys()))
        self.num_parallel_datasets = num_parallel
        if self.num_parallel_datasets != 1:
            raise NotImplementedError("When num_parallel_datasets != 1, we got some strange problem with the dataset. Waiting for fix.")

    def set_batch(self, num_speakers, num_segments):
        """Set the batch-related parameters

        Args:
            num_speakers: The number of speakers per batch.
            num_segments: The number of semgents per speaker.
        """
        self.num_speakers = num_speakers
        self.num_segments = num_segments

    def set_length(self, min_len, max_len):
        """Set the length of the sequence

        Args:
            min_len: The minimum length of the sampled sequence.
            max_len: The maximum length of the sampled sequence.
        """
        self.min_len = min_len
        self.max_len = max_len

    def batch_random(self):
        """Randomly load features to form a batch

        This function is used in the load routine to feed data to the dataset object
        It can also be used as a generator to get data directly.
        """
        feature_reader = FeatureReader(self.data)
        speakers = self.speakers
        if self.num_total_speakers < self.num_speakers:
            print(
                "[Warning] The number of available speakers are less than the required speaker. Some speakers will be duplicated.")
            speakers = self.speakers * (int(self.num_speakers / self.num_total_speakers) + 1)

        while True:
            batch_length = random.randint(self.min_len, self.max_len)
            batch_speakers = random.sample(speakers, self.num_speakers)
            features = np.zeros((self.num_speakers * self.num_segments, batch_length, feature_reader.dim),
                                dtype=np.float32)
            labels = np.zeros((self.num_speakers * self.num_segments), dtype=np.int32)

            for i, speaker in enumerate(batch_speakers):
                labels[i * self.num_segments:(i + 1) * self.num_segments] = speaker
                feature_list = self.spk2features[speaker]
                if len(feature_list) < self.num_segments:
                    feature_list *= (int(self.num_segments / len(feature_list)) + 1)
                # Now the length of the list must be greater than the sample size.
                speaker_features = random.sample(feature_list, self.num_segments)
                for j, feat in enumerate(speaker_features):
                    features[i * self.num_segments + j, :, :], _ = feature_reader.read(feat, batch_length, shuffle=self.shuffle)
            yield (features, labels)

    def load_dataset(self):
        """ Load data from Kaldi features and return tf.dataset.
        The function is useful for training, since it randomly loads features and labels from N speakers,
          with K segments per speaker.
        The batch is sampled randomly, so there is no need to do shuffle again.

        :return: A nested tensor (features, labels)
        """
        batch_size = self.num_speakers * self.num_segments
        if self.num_parallel_datasets == 1:
            # Single thread loading
            dataset = tf.data.Dataset.from_generator(self.batch_random, (tf.float32, tf.int32),
                                                     (tf.TensorShape([batch_size, None, self.dim]),
                                                      tf.TensorShape([batch_size])))
        else:
            # Multiple threads loading
            # It is very strange that the following code doesn't work properly.
            # I guess the reason may be the py_func influence the performance of parallel_interleave.
            dataset = tf.data.Dataset.range(self.num_parallel_datasets).apply(
                tf.contrib.data.parallel_interleave(
                    lambda x: tf.data.Dataset.from_generator(self.batch_random, (tf.float32, tf.int32),
                                                             (tf.TensorShape([batch_size, None, self.dim]),
                                                              tf.TensorShape([batch_size]))),
                    cycle_length=self.num_parallel_datasets,
                    sloppy=False))
        dataset = dataset.prefetch(1)
        return dataset.make_one_shot_iterator().get_next()


def batch_random(stop_event,
                 queue,
                 data,
                 spk2features,
                 num_total_speakers,
                 num_speakers=10,
                 num_segments=10,
                 min_len=200,
                 max_len=400,
                 shuffle=True,
                 seed=0):
    """Load features and fill a queue. Used in KaldiDataRandomQueue

    Args:
        stop_event: An event to tell the process to stop.
        queue: A queue to put the data.
        data: The kaldi data directory.
        spk2features: A dict from speaker index to the segments.
        num_total_speakers: The total number of speakers.
        num_speakers: The number of speakers in the batch.
        num_segments: The number of segments per speaker.
        min_len: The minimum length of the features.
        max_len: The maximum length of the features.
        shuffle: Load the feature from the 0-th frame or a random frame.
        seed: The value used to generate the random seed.
    """
    # TODO: If you use numpy.random in the sub-process, it is better to use:
    # local_state = np.random.RandomState(seed)
    # print local_state.uniform(0, 1, 5)
    #
    # The re-seed is necessary if numpy.random is used
    # You can use os.urandom to generate the `random` seed.
    rd = random.Random(os.urandom(4))
    rd.jumpahead(seed)

    feature_reader = FeatureReader(data)
    speakers = list(spk2features.keys())
    if num_total_speakers < num_speakers:
        print(
            "[Warning] The number of available speakers are less than the required speaker. Some speakers will be duplicated.")
        speakers = speakers * (int(num_speakers / num_total_speakers) + 1)
    # Now we have enough speakers
    while not stop_event.is_set():
        batch_speakers = rd.sample(speakers, num_speakers)
        batch_length = rd.randint(min_len, max_len)
        features = np.zeros((num_speakers * num_segments, batch_length, feature_reader.dim), dtype=np.float32)
        labels = np.zeros((num_speakers * num_segments), dtype=np.int32)
        for i, speaker in enumerate(batch_speakers):
            # The length may be larger than the utterance length. A check should be applied first.
            feature_list = []
            spk = speaker
            while len(feature_list) == 0:
                feature_list = []
                for feat in spk2features[spk]:
                    if feature_reader.utt2num_frames[feat.split(' ')[0]] > batch_length:
                        feature_list.append(feat)
                if len(feature_list) == 0:
                    # The speaker is not appropriate for this batch. Resample the speaker
                    spk = rd.choice(list(set(speakers) - set(batch_speakers)))
                    batch_speakers[i] = spk

            labels[i * num_segments:(i + 1) * num_segments] = spk
            # If the number is not enough
            if len(feature_list) < num_segments:
                feature_list *= (int(num_segments / len(feature_list)) + 1)
            # Now the length of the list must be greater than the sample size.
            speaker_features = rd.sample(feature_list, num_segments)
            for j, feat in enumerate(speaker_features):
                features[i * num_segments + j, :, :], _ = feature_reader.read_segment(feat, batch_length, shuffle=shuffle)
        queue.put((features, labels))

    time.sleep(3)
    while not queue.empty():
        try:
            queue.get(block=False)
        except:
            pass
    print("The process {} is about to exit.".format(os.getpid()))
    return


class KaldiDataRandomQueue(object):
    """A queue to read features from Kaldi data directory."""

    def __init__(self, data_dir, spklist, num_parallel=1, max_qsize=10, num_speakers=None, num_segments=None, min_len=None, max_len=None, shuffle=True):
        """ Create a queue from a given directory.

        This is basically similar with KaldiDataRead. The difference is that KaldiDataReader uses tf.data to load
        features and KaldiDataQueue uses multiprocessing to load features which seems to be a better choice since
        the multiprocessing significantly speed up the loading in my case. If you can make parallel_interleave works,
        it is definitely more convenient to use KaldiDataReader because it's more simple.

        Args:
            data_dir: The kaldi data directory.
            spklist: The spklist tells the mapping from the speaker name to the speaker id.
            num_parallel: The number of threads to read features.
            max_qsize: The capacity of the queue
            num_speakers: The number of speakers per batch.
            num_segments: The number of semgents per speaker.
              batch_size = num_speakers * num_segments
              When num_segments = 1, the batch is randomly chosen from n speakers,
              which is used for softmax-like loss function. While we can sample multiple segments for each speaker,
              which is used for triplet-loss or GE2E loss.
            min_len: The minimum length of the sampled sequence.
            max_len: The maximum length of the sampled sequence.
            shuffle: Loading data from the 0-th frame or a random frame.
        """
        self.data = data_dir
        self.num_speakers = num_speakers
        self.num_segments = num_segments
        self.min_len = min_len
        self.max_len = max_len
        self.num_parallel_datasets = num_parallel
        self.shuffle = shuffle

        # We process the data directory and fetch speaker information.
        self.spk2features, self.features2spk, spk2index = get_speaker_info(data_dir, spklist)

        # The number of speakers should be
        self.num_total_speakers = len(list(spk2index.keys()))

        # The Queue is thread-safe and used to save the features.
        self.queue = Queue(max_qsize)
        self.stop_event = Event()

        # And the prcesses are saved
        self.processes = []

    def set_batch(self, num_speakers, num_segments):
        """Set the batch-related parameters

        Args:
            num_speakers: The number of speakers per batch.
            num_segments: The number of semgents per speaker.
        """
        self.num_speakers = num_speakers
        self.num_segments = num_segments

    def set_length(self, min_len, max_len):
        """Set the length of the sequence

        Args:
            min_len: The minimum length of the sampled sequence.
            max_len: The maximum length of the sampled sequence.
        """
        self.min_len = min_len
        self.max_len = max_len

    def start(self):
        """Start processes to load features
        """
        self.processes = [Process(target=batch_random, args=(self.stop_event,
                                                             self.queue,
                                                             self.data,
                                                             self.spk2features,
                                                             self.num_total_speakers,
                                                             self.num_speakers,
                                                             self.num_segments,
                                                             self.min_len,
                                                             self.max_len,
                                                             self.shuffle,
                                                             i))
                          for i in range(self.num_parallel_datasets)]
        for process in self.processes:
            process.daemon = True
            process.start()

    def fetch(self):
        """Fetch data from the queue"""
        return self.queue.get()

    def stop(self):
        """Stop the threads

        After stop, the processes are terminated and the queue may become unavailable.
        """
        self.stop_event.set()
        print("Clean the data queue that subprocesses can detect the stop event...")
        while not self.queue.empty():
            # Clear the queue content before join the threads. They may wait for putting the data to the queue.
            self.queue.get()
        time.sleep(3)
        for process in self.processes:
            # TODO: fix the join problem
            process.terminate()
            # process.join()


def batch_sequence(stop_event,
                   queue,
                   data,
                   feature_list,
                   features2spk,
                   batch_size=128,
                   min_len=200,
                   max_len=400,
                   shuffle=True,
                   seed=0):
    """Load features and fill a queue. Used in KaldiDataSeqQueue.

    Args:
        stop_event: An event indicating the reading is finished.
        queue: A queue to put the data.
        data: The kaldi data directory.
        feature_list: A list shows which features the process should read.
        features2spk: A dict map features to speaker index.
        batch_size: The batch_size
        min_len: The minimum length of the features.
        max_len: The maximum length of the features.
        shuffle: Load the feature from the 0-th frame or a random frame.
        seed: The number is used to generate a random seed
    """
    # Read the comment in batch_random
    rd = random.Random(os.urandom(4))
    rd.jumpahead(seed)

    feature_reader = FeatureReader(data)
    num_batches = int(len(feature_list) / batch_size)
    for i in range(num_batches):
        batch_length = rd.randint(min_len, max_len)

        # In some cases, the minimum length of the utterances is smaller than the batch length.
        # Use the smallest length as the real batch length.
        for j in range(batch_size):
            if feature_reader.utt2num_frames[feature_list[i * batch_size + j].split(' ')[0]] < batch_length:
                batch_length = feature_reader.utt2num_frames[feature_list[i * batch_size + j].split(' ')[0]]

        features = np.zeros((batch_size, batch_length, feature_reader.dim), dtype=np.float32)
        labels = np.zeros((batch_size), dtype=np.int32)
        for j in range(batch_size):
            features[j, :, :], _ = feature_reader.read_segment(feature_list[i * batch_size + j], batch_length, shuffle=shuffle)
            labels[j] = features2spk[feature_list[i * batch_size + j]]
        queue.put((features, labels))
    stop_event.set()
    print("The process {} is about to exit.".format(os.getpid()))
    return


class KaldiDataSeqQueue(object):
    """A queue to read features from Kaldi data directory."""

    def __init__(self, data_dir, spklist, num_parallel=1, max_qsize=10, batch_size=128, min_len=None, max_len=None, shuffle=True):
        """ Create a queue from a given directory.

        Unlike KaldiDataRandomQueue, KaldiDataSeqQueue load data in sequence which means each segment appears once
        in one epoch. This is usually used for validation (using softmax-like loss or EER).

        Args:
            data_dir: The kaldi data directory.
            spklist: The spklist tells the mapping from the speaker name to the speaker id.
            num_parallel: The number of threads to read features.
            max_qsize: The capacity of the queue.
            batch_size: The batch size.
            min_len: The minimum length of the sampled sequence.
            max_len: The maximum length of the sampled sequence.
            shuffle: Shuffle the load sequence and loading data from a random frame.
        """
        self.data = data_dir
        self.batch_size = batch_size
        self.min_len = min_len
        self.max_len = max_len
        self.num_parallel_datasets = num_parallel
        self.shuffle = shuffle

        # We process the data directory and fetch speaker information.
        self.spk2features, self.features2spk, spk2index = get_speaker_info(data_dir, spklist)
        self.num_total_speakers = len(list(spk2index.keys()))

        # Arrange features in sequence
        self.feature_list = []
        self.sub_feature_list = []
        for spk in self.spk2features:
            self.feature_list += self.spk2features[spk]

        if shuffle:
            random.shuffle(self.feature_list)
        # Split the features to N sub-list. The lists are used in each process.
        num_sub_features = len(self.feature_list) / num_parallel
        for i in range(num_parallel):
            if i == num_parallel - 1:
                self.sub_feature_list.append(self.feature_list[i * num_sub_features:])
            else:
                self.sub_feature_list.append(self.feature_list[i * num_sub_features:(i + 1) * num_sub_features])

        # The Queue is thread-safe and used to save the features.
        self.queue = Queue(max_qsize)

        # The events will be set once the processes finish its job
        self.stop_event = [Event() for _ in range(num_parallel)]

        # And the prcesses are saved
        self.processes = []

    def set_batch(self, batch_size):
        """Set the batch size
        """
        self.batch_size = batch_size

    def set_length(self, min_len, max_len):
        """Set the length of the sequence

        Args:
            min_len: The minimum length of the sampled sequence.
            max_len: The maximum length of the sampled sequence.
        """
        self.min_len = min_len
        self.max_len = max_len

    def start(self):
        """Start processes to load features
        """
        self.processes = [Process(target=batch_sequence, args=(self.stop_event[i],
                                                               self.queue,
                                                               self.data,
                                                               self.sub_feature_list[i],
                                                               self.features2spk,
                                                               self.batch_size,
                                                               self.min_len,
                                                               self.max_len,
                                                               self.shuffle,
                                                               i))
                          for i in range(self.num_parallel_datasets)]
        for process in self.processes:
            process.daemon = True
            process.start()

    def fetch(self):
        """Fetch data from the queue"""
        if self.queue.empty():
            all_finish = [self.stop_event[i].is_set() for i in range(self.num_parallel_datasets)]
            if all(all_finish):
                # If the queue is empty and all processes are finished, we got nothing to read.
                for process in self.processes:
                    # TODO: fix the join problem
                    process.terminate()
                raise DataOutOfRange

        return self.queue.get()

    def stop(self):
        """Stop the threads"""
        for process in self.processes:
            # TODO: fix the join problem
            process.terminate()
            # process.join()


def multi_batch_random(stop_event,
                       queue,
                       data,
                       aux_data,
                       spk2features,
                       num_total_speakers,
                       num_speakers=10,
                       num_segments=10,
                       min_len=200,
                       max_len=400,
                       shuffle=True,
                       seed=0):
    """Load features and auxiliary features, fill a queue. Used in KaldiMultiDataRandomQueue

    Args:
        stop_event: An event to tell the process to stop.
        queue: A queue to put the data.
        data: The kaldi data directory.
        aux_data: A dict. The auxiliary data directories.
        spk2features: A dict from speaker index to the segments.
        num_total_speakers: The total number of speakers.
        num_speakers: The number of speakers in the batch.
        num_segments: The number of segments per speaker.
        min_len: The minimum length of the features.
        max_len: The maximum length of the features.
        shuffle: Load the feature from the 0-th frame or a random frame.
        seed: The value used to generate the random seed.
    """
    # TODO: If you use numpy.random in the sub-process, it is better to use:
    # local_state = np.random.RandomState(seed)
    # print local_state.uniform(0, 1, 5)
    #
    # The re-seed is necessary if numpy.random is used
    # You can use os.urandom to generate the `random` seed.
    rd = random.Random(os.urandom(4))
    rd.jumpahead(seed)

    feature_reader = {}
    feature_reader["features"] = FeatureReader(data)
    for name in aux_data:
        feature_reader[name] = FeatureReader(aux_data[name])

    speakers = list(spk2features.keys())
    if num_total_speakers < num_speakers:
        print(
            "[Warning] The number of available speakers are less than the required speaker. Some speakers will be duplicated.")
        speakers = speakers * (int(num_speakers / num_total_speakers) + 1)
    while not stop_event.is_set():
        batch_speakers = rd.sample(speakers, num_speakers)
        batch_length = rd.randint(min_len, max_len)
        features = {}
        for name in feature_reader:
            features[name] = np.zeros((num_speakers * num_segments, batch_length, feature_reader[name].dim),
                                      dtype=np.float32)
        labels = np.zeros((num_speakers * num_segments), dtype=np.int32)

        for i, speaker in enumerate(batch_speakers):
            # The length may be larger than the utterance length. A check should be applied first.
            feature_list = []
            spk = speaker
            while len(feature_list) == 0:
                feature_list = []
                for feat in spk2features[spk]:
                    if feature_reader["features"].utt2num_frames[feat["features"].split(' ')[0]] > batch_length:
                        feature_list.append(feat)
                if len(feature_list) == 0:
                    # The speaker is not appropriate for this batch. Resample the speaker
                    spk = rd.choice(list(set(speakers) - set(batch_speakers)))
                    batch_speakers[i] = spk

            labels[i * num_segments:(i + 1) * num_segments] = spk
            if len(feature_list) < num_segments:
                feature_list *= (int(num_segments / len(feature_list)) + 1)
            # Now the length of the list must be greater than the sample size.
            speaker_features = rd.sample(feature_list, num_segments)

            for j, feat in enumerate(speaker_features):
                # Load features first.
                features["features"][i * num_segments + j, :, :], start_pos = feature_reader["features"].read_segment(feat["features"],
                                                                                                              batch_length,
                                                                                                              shuffle=shuffle)
                for name in feature_reader:
                    if name == "features":
                        continue
                    features[name][i * num_segments + j, :, :], _ = feature_reader[name].read_segment(feat[name], batch_length, start=start_pos)

        queue.put((features, labels))

    time.sleep(3)
    while not queue.empty():
        try:
            queue.get(block=False)
        except:
            pass
    print("The process {} is about to exit.".format(os.getpid()))
    return


class KaldiMultiDataRandomQueue(KaldiDataRandomQueue):
    """A queue to read features from Kaldi data directory (with auxiliary features)."""
    def __init__(self, data_dir, aux_data, spklist, num_parallel=1, max_qsize=10, num_speakers=None, num_segments=None,
                 min_len=None, max_len=None, shuffle=True):
        """Create a queue from a feature directory and some auxiliary feature directories.
        """
        super(KaldiMultiDataRandomQueue, self).__init__(data_dir, spklist, num_parallel, max_qsize, num_speakers,
                                                        num_segments, min_len, max_len, shuffle)
        self.aux_data = {}
        for dirname in os.listdir(aux_data):
            if dirname[0] == ".":
                continue
            if os.path.isdir(os.path.join(aux_data, dirname)):
                self.aux_data[dirname] = os.path.join(aux_data, dirname)

        # Preload the information. We should build the map from the utterance to the auxiliary feature.
        self.spk2features, self.features2spk, spk2index = get_aux_speaker_info(data_dir, self.aux_data, spklist)

    def start(self):
        """Start processes to load features
        """
        self.processes = [Process(target=multi_batch_random, args=(self.stop_event,
                                                                   self.queue,
                                                                   self.data,
                                                                   self.aux_data,
                                                                   self.spk2features,
                                                                   self.num_total_speakers,
                                                                   self.num_speakers,
                                                                   self.num_segments,
                                                                   self.min_len,
                                                                   self.max_len,
                                                                   self.shuffle,
                                                                   i))
                          for i in range(self.num_parallel_datasets)]
        for process in self.processes:
            process.daemon = True
            process.start()


def multi_batch_sequence(stop_event,
                         queue,
                         data,
                         aux_data,
                         feature_list,
                         features2spk,
                         batch_size=128,
                         min_len=200,
                         max_len=400,
                         shuffle=True,
                         seed=0):
    """Load features, auxiliary features, fill a queue. Used in KaldiMultiDataSeqQueue.

    Args:
        stop_event: An event indicating the reading is finished.
        queue: A queue to put the data.
        data: The kaldi data directory.
        aux_data: A dict. The auxiliary data directories.
        feature_list: A dict. A list shows which features the process should read (containing auxiliary features).
        features2spk: A dict map features to speaker index.
        batch_size: The batch_size
        min_len: The minimum length of the features.
        max_len: The maximum length of the features.
        shuffle: Load the feature from the 0-th frame or a random frame.
        seed: The number is used to generate a random seed
    """
    # Read the comment in batch_random
    rd = random.Random(os.urandom(4))
    rd.jumpahead(seed)

    feature_reader = {}
    feature_reader["features"] = FeatureReader(data)
    for name in aux_data:
        feature_reader[name] = FeatureReader(aux_data[name])

    num_batches = int(len(feature_list) / batch_size)
    for i in range(num_batches):
        batch_length = rd.randint(min_len, max_len)
        for j in range(batch_size):
            if feature_reader["features"].utt2num_frames[feature_list[i * batch_size + j]["features"].split(' ')[0]] < batch_length:
                batch_length = feature_reader["features"].utt2num_frames[feature_list[i * batch_size + j]["features"].split(' ')[0]]

        features = {}
        for name in feature_reader:
            features[name] = np.zeros((batch_size, batch_length, feature_reader[name].dim), dtype=np.float32)
        labels = np.zeros((batch_size), dtype=np.int32)
        for j in range(batch_size):
            # Load the features first
            features["features"][j, :, :], start_pos = feature_reader["features"].read_segment(feature_list[i * batch_size + j]["features"],
                                                                                       batch_length,
                                                                                       shuffle=shuffle)
            for name in feature_reader:
                if name == "features":
                    continue
                features[name][j, :, :], _ = feature_reader[name].read_segment(feature_list[i * batch_size + j][name], batch_length, start=start_pos)
            labels[j] = features2spk[feature_list[i * batch_size + j]["features"]]
        queue.put((features, labels))
    stop_event.set()
    print("The process {} is about to exit.".format(os.getpid()))
    return


class KaldiMultiDataSeqQueue(KaldiDataSeqQueue):
    """A queue to read features from Kaldi data directory (with auxiliary features)."""

    def __init__(self, data_dir, aux_data, spklist, num_parallel=1, max_qsize=10, batch_size=128,
                 min_len=None, max_len=None, shuffle=True):
        """Create a queue from a feature directory and some auxiliary feature directories.
        """
        super(KaldiMultiDataSeqQueue, self).__init__(data_dir, spklist, num_parallel, max_qsize, batch_size,
                                                     min_len, max_len, shuffle)
        self.aux_data = {}
        for dirname in os.listdir(aux_data):
            # Skip hidden directories (e.g. .backup/)
            if dirname[0] == ".":
                continue
            if os.path.isdir(os.path.join(aux_data, dirname)):
                self.aux_data[dirname] = os.path.join(aux_data, dirname)

        # Preload the information. We should build the map from the utterance to the auxiliary feature.
        self.spk2features, self.features2spk, spk2index = get_aux_speaker_info(data_dir, self.aux_data, spklist)

        # Re-arrange the feature list since now we have auxiliary features.
        self.feature_list = []
        self.sub_feature_list = []
        for spk in self.spk2features:
            self.feature_list += self.spk2features[spk]
        if shuffle:
            random.shuffle(self.feature_list)
        num_sub_features = len(self.feature_list) / num_parallel
        for i in range(num_parallel):
            if i == num_parallel - 1:
                self.sub_feature_list.append(self.feature_list[i * num_sub_features:])
            else:
                self.sub_feature_list.append(self.feature_list[i * num_sub_features:(i + 1) * num_sub_features])

    def start(self):
        """Start processes to load features
        """
        self.processes = [Process(target=multi_batch_sequence, args=(self.stop_event[i],
                                                                     self.queue,
                                                                     self.data,
                                                                     self.aux_data,
                                                                     self.sub_feature_list[i],
                                                                     self.features2spk,
                                                                     self.batch_size,
                                                                     self.min_len,
                                                                     self.max_len,
                                                                     self.shuffle,
                                                                     i))
                          for i in range(self.num_parallel_datasets)]
        for process in self.processes:
            process.daemon = True
            process.start()


if __name__ == "__main__":
    data_simple = "/home/heliang05/liuyi/voxceleb/data/voxceleb_train_combined_no_sil/softmax_valid"
    spklist_simple = "/home/heliang05/liuyi/voxceleb/data/voxceleb_train_combined_no_sil/train/spklist"
    data = "/home/heliang05/liuyi/voxceleb/data/voxceleb_train_combined_no_sil/end2end_valid"
    spklist = "/home/heliang05/liuyi/voxceleb/data/voxceleb_train_combined_no_sil/end2end_valid/spklist"
    aux_data_dir = "/home/heliang05/liuyi/voxceleb/data/voxceleb_train_combined_no_sil/test_aux_data"
    num_loads = 10
    import time
    import pdb

    num_speakers = 16
    num_segments = 1
    min_len = 200
    max_len = 400
    batch_size = num_speakers * num_segments
    shuffle = False
    spk2features, features2spk, spk2index = get_speaker_info(data, spklist)
    num_total_speakers = len(list(spk2index.keys()))

    rd = random.Random(os.urandom(4))
    feature_reader = FeatureReader(data)
    speakers = list(spk2features.keys())
    if num_total_speakers < num_speakers:
        print(
            "[Warning] The number of available speakers are less than the required speaker. Some speakers will be duplicated.")
        speakers = speakers * (int(num_speakers / num_total_speakers) + 1)

    # Single input and single output.

    batch_speakers = rd.sample(speakers, num_speakers)
    batch_length = rd.randint(min_len, max_len)
    features = np.zeros((num_speakers * num_segments, batch_length, feature_reader.dim), dtype=np.float32)
    features_comp = np.zeros((num_speakers * num_segments, batch_length, feature_reader.dim), dtype=np.float32)
    labels = np.zeros((num_speakers * num_segments), dtype=np.int32)
    for i, speaker in enumerate(batch_speakers):
        # The length may be larger than the utterance length. A check should be applied first.
        feature_list = []
        spk = speaker
        while len(feature_list) == 0:
            feature_list = []
            for feat in spk2features[spk]:
                if feature_reader.utt2num_frames[feat.split(' ')[0]] > batch_length:
                    feature_list.append(feat)
            if len(feature_list) == 0:
                # The speaker is not appropriate for this batch. Resample the speaker
                spk = rd.choice(list(set(speakers) - set(batch_speakers)))
                batch_speakers[i] = spk

        labels[i * num_segments:(i + 1) * num_segments] = spk
        # If the number is not enough
        if len(feature_list) < num_segments:
            feature_list *= (int(num_segments / len(feature_list)) + 1)
        # Now the length of the list must be greater than the sample size.
        speaker_features = rd.sample(feature_list, num_segments)
        for j, feat in enumerate(speaker_features):
            features[i * num_segments + j, :, :], start_pos = feature_reader.read_segment(feat, batch_length, shuffle=shuffle)
            features_comp[i * num_segments + j, :, :], _ = feature_reader.read(feat, batch_length, start=start_pos)

    pdb.set_trace()
    assert np.allclose(features, features_comp)
    print(labels)

    feature_list = []
    for spk in spk2features:
        feature_list += spk2features[spk]
    num_batches = len(feature_list) / batch_size
    for i in range(10):
        batch_length = rd.randint(min_len, max_len)

        # In some cases, the minimum length of the utterances is smaller than the batch length.
        # Use the smallest length as the real batch length.
        for j in range(batch_size):
            if feature_reader.utt2num_frames[feature_list[i * batch_size + j].split(' ')[0]] < batch_length:
                batch_length = feature_reader.utt2num_frames[feature_list[i * batch_size + j].split(' ')[0]]

        features = np.zeros((batch_size, batch_length, feature_reader.dim), dtype=np.float32)
        features_comp = np.zeros((batch_size, batch_length, feature_reader.dim), dtype=np.float32)
        labels = np.zeros((batch_size), dtype=np.int32)
        for j in range(batch_size):
            features[j, :, :], start_pos = feature_reader.read_segment(feature_list[i * batch_size + j], batch_length, shuffle=shuffle)
            features_comp[j, :, :], _ = feature_reader.read(feature_list[i * batch_size + j], batch_length, start=start_pos)
            labels[j] = features2spk[feature_list[i * batch_size + j]]

    pdb.set_trace()
    assert np.allclose(features, features_comp)
    print(labels)

    # Using KaldiDataQueue (multiprocessing)
    # Although this will introduce CPU-GPU transfer overhead, it seems to be much faster.
    data_loader = KaldiDataRandomQueue(data_simple, spklist_simple, num_parallel=8, max_qsize=10, num_speakers=64, num_segments=1, min_len=2000, max_len=2000, shuffle=True)
    with tf.Session() as sess:
        ts = time.time()
        features = tf.placeholder(tf.float32, shape=[None, None, None])
        labels = tf.placeholder(tf.int32, shape=[None])
        features += 1
        data_loader.start()
        for _ in range(num_loads):
            features_val, labels_val = data_loader.fetch()
            features_test, labels_test = sess.run([features, labels], feed_dict={features: features_val,
                                                                                 labels: labels_val})
        te = time.time() - ts
        data_loader.stop()
        print("Time: %.4f s" % te)
        pdb.set_trace()
        print(labels_test)

    data_loader = KaldiDataSeqQueue(data, spklist, num_parallel=8, max_qsize=10, batch_size=64, min_len=200, max_len=400, shuffle=True)
    with tf.Session() as sess:
        features = tf.placeholder(tf.float32, shape=[None, None, None])
        labels = tf.placeholder(tf.int32, shape=[None])
        features += 1
        data_loader.start()
        index = 1
        while index < 10:
            try:
                features_val, labels_val = data_loader.fetch()
                features_test, labels_test = sess.run([features, labels], feed_dict={features: features_val,
                                                                                     labels: labels_val})
                index += 1
            except DataOutOfRange:
                break
        data_loader.stop()
        pdb.set_trace()
        print(labels_test)

    # Multiple input

    aux_data = {}
    for dirname in os.listdir(aux_data_dir):
        if os.path.isdir(os.path.join(aux_data_dir, dirname)):
            aux_data[dirname] = os.path.join(aux_data_dir, dirname)
    spk2features, features2spk, spk2index = get_aux_speaker_info(data, aux_data, spklist)

    feature_reader = {}
    feature_reader["features"] = FeatureReader(data)
    for name in aux_data:
        feature_reader[name] = FeatureReader(aux_data[name])

    speakers = list(spk2features.keys())
    if num_total_speakers < num_speakers:
        print(
            "[Warning] The number of available speakers are less than the required speaker. Some speakers will be duplicated.")
        speakers = speakers * (int(num_speakers / num_total_speakers) + 1)

    batch_speakers = rd.sample(speakers, num_speakers)
    batch_length = rd.randint(min_len, max_len)
    features = {}
    for name in feature_reader:
        features[name] = np.zeros((num_speakers * num_segments, batch_length, feature_reader[name].dim),
                                  dtype=np.float32)
    features_comp = {}
    for name in feature_reader:
        features_comp[name] = np.zeros((num_speakers * num_segments, batch_length, feature_reader[name].dim),
                                  dtype=np.float32)
    labels = np.zeros((num_speakers * num_segments), dtype=np.int32)

    for i, speaker in enumerate(batch_speakers):
        # The length may be larger than the utterance length. A check should be applied first.
        feature_list = []
        spk = speaker
        while len(feature_list) == 0:
            feature_list = []
            for feat in spk2features[spk]:
                if feature_reader["features"].utt2num_frames[feat["features"].split(' ')[0]] > batch_length:
                    feature_list.append(feat)
            if len(feature_list) == 0:
                # The speaker is not appropriate for this batch. Resample the speaker
                spk = rd.choice(list(set(speakers) - set(batch_speakers)))
                batch_speakers[i] = spk

        labels[i * num_segments:(i + 1) * num_segments] = spk
        if len(feature_list) < num_segments:
            feature_list *= (int(num_segments / len(feature_list)) + 1)
        # Now the length of the list must be greater than the sample size.
        speaker_features = rd.sample(feature_list, num_segments)

        for j, feat in enumerate(speaker_features):
            # Load features first.
            features["features"][i * num_segments + j, :, :], start_pos = feature_reader["features"].read_segment(
                feat["features"],
                batch_length,
                shuffle=shuffle)
            for name in feature_reader:
                if name == "features":
                    continue
                features[name][i * num_segments + j, :, :], _ = feature_reader[name].read_segment(feat[name], batch_length,
                                                                                              start=start_pos)

            features_comp["features"][i * num_segments + j, :, :], _ = feature_reader["features"].read(
                feat["features"],
                batch_length,
                start=start_pos)
            for name in feature_reader:
                if name == "features":
                    continue
                features_comp[name][i * num_segments + j, :, :], _ = feature_reader[name].read(feat[name], batch_length,
                                                                                          start=start_pos)

    # Test the consistency of the featuers (the starting points)
    for name in feature_reader:
        assert np.allclose(features[name], features_comp[name])
        assert np.allclose(features[name], features["features"])
    print(labels)

    feature_list = []
    for spk in spk2features:
        feature_list += spk2features[spk]
    num_batches = int(len(feature_list) / batch_size)
    pdb.set_trace()
    for i in range(10):
        batch_length = rd.randint(min_len, max_len)
        for j in range(batch_size):
            if feature_reader["features"].utt2num_frames[feature_list[i * batch_size + j]["features"].split(' ')[0]] < batch_length:
                batch_length = feature_reader["features"].utt2num_frames[feature_list[i * batch_size + j]["features"].split(' ')[0]]

        features = {}
        for name in feature_reader:
            features[name] = np.zeros((batch_size, batch_length, feature_reader[name].dim), dtype=np.float32)
        features_comp = {}
        for name in feature_reader:
            features_comp[name] = np.zeros((batch_size, batch_length, feature_reader[name].dim), dtype=np.float32)
        labels = np.zeros((batch_size), dtype=np.int32)
        for j in range(batch_size):
            # Load the features first
            features["features"][j, :, :], start_pos = feature_reader["features"].read_segment(
                feature_list[i * batch_size + j]["features"],
                batch_length,
                shuffle=shuffle)
            for name in feature_reader:
                if name == "features":
                    continue
                features[name][j, :, :], _ = feature_reader[name].read_segment(feature_list[i * batch_size + j][name],
                                                                       batch_length, start=start_pos)
            features_comp["features"][j, :, :], _ = feature_reader["features"].read(
                feature_list[i * batch_size + j]["features"],
                batch_length,
                start=start_pos)
            for name in feature_reader:
                if name == "features":
                    continue
                features_comp[name][j, :, :], _ = feature_reader[name].read(feature_list[i * batch_size + j][name],
                                                                       batch_length, start=start_pos)
            labels[j] = features2spk[feature_list[i * batch_size + j]["features"]]

    pdb.set_trace()
    for name in feature_reader:
        assert np.allclose(features[name], features_comp[name])
        assert np.allclose(features[name], features["features"])
    print(labels)

    data_loader = KaldiMultiDataRandomQueue(data, aux_data_dir, spklist, num_parallel=10)
    data_loader.set_batch(64, 1)
    data_loader.set_length(200, 400)
    data_loader.start()
    for _ in range(num_loads):
        features_val, labels_val = data_loader.fetch()
    data_loader.stop()
    pdb.set_trace()
    print(labels_val)
    for name in features_val:
        assert np.allclose(features_val[name], features_val["features"])

    data_loader = KaldiMultiDataSeqQueue(data, aux_data_dir, spklist)
    data_loader.set_batch(64)
    data_loader.set_length(200, 400)
    data_loader.start()
    for _ in range(num_loads):
        features_val, labels_val = data_loader.fetch()
    data_loader.stop()
    pdb.set_trace()
    print(labels_val)
    for name in features_val:
        assert np.allclose(features_val[name], features_val["features"])
