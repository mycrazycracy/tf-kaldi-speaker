# In this class, speakers and phones are selected according to their numbers of samples in the dataset, i.e.,
# the more sample the speaker has, the higher probability it can be selected.

import tensorflow as tf
import os
import subprocess
import random
import numpy as np
from multiprocessing import Process, Queue, Event
from dataset.kaldi_io import FeatureReaderV2
from six.moves import range
from dataset.data_loader import get_speaker_info, DataOutOfRange
import time


def sample_with_probability(rd, candidates, num_selects, regions):
    """Sample speakers with their frames.
    The more #frames, the higher probability to be selected.

    Args:
        rd: random generator
        candidates: the list
        num_selects: selected number
        regions: how to pick the candidates
    :return: the selected candidates
    """
    selected = []
    num_candidates = len(candidates)
    while len(selected) < num_selects:
        r = rd.uniform(0, regions[-1])
        for k in range(num_candidates):
            if regions[k] >= r:
                if candidates[k] not in selected:
                    selected.append(candidates[k])
                break
    return selected


def batch_random_v2(stop_event,
                    queue,
                    data_dir,
                    ali_dir,
                    spk2features,
                    spk2num_frames,
                    utt2num_frames,
                    left_context,
                    right_context,
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
        data_dir: The kaldi data directory.
        ali_dir: The kaldi ali directory.
        spk2features: A dict from speaker index to the segments.
        spk2num_frames: #frames per speaker
        utt2num_frames: #frames per utt
        num_speakers: The number of speakers in the batch.
        num_segments: The number of segments per speaker.
        min_len: The minimum length of the features.
        max_len: The maximum length of the features.
        shuffle: Load the feature from the 0-th frame or a random frame.
        seed: The value used to generate the random seed.
    """
    rd = random.Random(os.urandom(4))
    rd.jumpahead(seed)

    feature_reader = FeatureReaderV2(data_dir, ali_dir, left_context, right_context)
    speakers = list(spk2features.keys())

    total_num_frames = np.sum(spk2num_frames.values())
    spk_sample_region = []
    current_region = 0
    for spk in speakers:
        current_region += spk2num_frames[spk]
        spk_sample_region.append(current_region)
    assert total_num_frames == current_region

    spk2utt_sample_region = {}
    for spk in speakers:
        spk2utt_sample_region[spk] = []
        current_region = 0
        for utt in spk2features[spk]:
            current_region += utt2num_frames[utt]
            spk2utt_sample_region[spk].append(current_region)

    while not stop_event.is_set():
        batch_speakers = sample_with_probability(rd, speakers, num_speakers, spk_sample_region)
        batch_length = rd.randint(min_len, max_len)
        # The feature should be expanded
        features = np.zeros((num_speakers * num_segments, batch_length + left_context + right_context, feature_reader.dim), dtype=np.float32)
        # The other variables still have the original length
        vad = np.zeros((num_speakers * num_segments, batch_length), dtype=np.float32)
        ali = np.zeros((num_speakers * num_segments, batch_length), dtype=np.int32)
        labels = np.zeros((num_speakers * num_segments), dtype=np.int32)
        valid_length = np.zeros((num_speakers * num_segments), dtype=np.int32)
        # valid_pos is [start, end) position that the forward will use the `true` features, rather than the expansion
        # This is used for logging
        valid_pos = np.zeros((num_speakers * num_segments, 2), dtype=np.int32)
        # resample indicates whether we should sample the segment to find a phonetic training example
        # or just use the beginning of the segment.
        # Useful when training a network.
        resample = np.zeros((num_speakers * num_segments), dtype=np.int32)

        for i, speaker in enumerate(batch_speakers):
            spk = speaker
            # The speaker should have enough number of speakers
            assert spk2features[spk] > num_segments, "Speaker %s does not have enough segments to sample" % spk
            labels[i * num_segments:(i + 1) * num_segments] = spk

            batch_segments = sample_with_probability(rd, spk2features[spk], num_segments, spk2utt_sample_region[spk])
            for j, utt in enumerate(batch_segments):
                utt_feat, utt_vad, utt_ali, utt_start = feature_reader.read_segment(utt, batch_length, shuffle=shuffle)
                # utt_length is the valid length before feature expansion.
                # The valid length excludes the expanded features.
                utt_length = utt_feat.shape[0] - left_context - right_context
                features[i * num_segments + j, :utt_feat.shape[0], :] = utt_feat
                # Expand the feature by the last frame if the length is not long enough.
                # (Though this is not necessary...)
                if utt_length < batch_length:
                    features[i * num_segments + j, utt_feat.shape[0]:, :] = features[i * num_segments + j, utt_feat.shape[0]-1, :]
                vad[i * num_segments + j, :utt_length] = utt_vad
                ali[i * num_segments + j, :utt_length] = utt_ali
                valid_length[i * num_segments + j] = utt_length
                resample[i * num_segments + j] = 1 if utt_start + utt_length == utt2num_frames[utt] else 0
                valid_pos[i * num_segments + j, 0] = left_context - utt_start if left_context > utt_start else 0
                valid_pos[i * num_segments + j, 1] = utt2num_frames[utt] - utt_start - right_context \
                    if utt_start + utt_length > utt2num_frames[utt] - right_context else utt_length
        queue.put((features, vad, ali, valid_length, labels, resample, valid_pos))

    time.sleep(3)
    while not queue.empty():
        try:
            queue.get(block=False)
        except:
            pass
    print("The process {} is about to exit.".format(os.getpid()))
    return


class KaldiDataRandomQueueV2(object):
    """A queue to read features from Kaldi data directory."""
    def __init__(self, data_dir,
                 ali_dir,
                 spklist,
                 num_parallel=1,
                 max_qsize=10,
                 left_context=None,
                 right_context=None,
                 num_speakers=None,
                 num_segments=None,
                 min_len=None,
                 max_len=None,
                 shuffle=True):
        """ Create a queue from a given directory.
        This is used to load features, vad and alignment to train our new model.
        As the same as KaldiDataRandomQueue though we add loading with probability (rather than a uniform distribution).

        Args:
            data_dir: The kaldi data directory.
            ali_dir: The kaldi ali directory.
            spklist: The spklist tells the mapping from the speaker name to the speaker id.
            num_parallel: The number of threads to read features.
            max_qsize: The capacity of the queue
            left_context: The left context to expand the feature
            right_context: The right context to expand the feature
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
        self.data_dir = data_dir
        self.ali_dir = ali_dir
        self.num_speakers = num_speakers
        self.num_segments = num_segments
        self.min_len = min_len
        self.max_len = max_len
        self.num_parallel_datasets = num_parallel
        self.shuffle = shuffle
        self.left_context = left_context
        self.right_context = right_context

        # We process the data directory and fetch speaker information.
        self.spk2features, self.features2spk, spk2index = get_speaker_info(data_dir, spklist)

        # Restore the relation of utt and spk. That's annoying...
        self.spk2features_new = {}
        for spk in self.spk2features:
            self.spk2features_new[spk] = []
            for utt in self.spk2features[spk]:
                self.spk2features_new[spk].append(utt.split(" ")[0])
        self.spk2features = self.spk2features_new

        self.features2spk_new = {}
        for utt in self.features2spk:
            self.features2spk_new[utt.split(" ")[0]] = self.features2spk[utt]
        self.features2spk = self.features2spk_new

        # We also load #frames for each speaker and #frames for each utt
        self.utt2num_frames = {}
        with open(os.path.join(data_dir, "utt2num_frames"), 'r') as f:
            for line in f.readlines():
                utt, n = line.strip().split(" ")
                self.utt2num_frames[utt] = int(n)

        self.spk2num_frames = {}
        for spk in self.spk2features:
            n = 0
            for utt in self.spk2features[spk]:
                n += self.utt2num_frames[utt]
            self.spk2num_frames[spk] = n

        # The number of speakers should be
        self.num_total_speakers = len(list(spk2index.keys()))

        # The number of phones
        self.num_total_phones = int(subprocess.check_output("tree-info %s | grep num-pdfs | awk '{print $2}'" %
                                                            os.path.join(self.ali_dir, "tree"), shell=True))

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
        self.processes = [Process(target=batch_random_v2, args=(self.stop_event,
                                                                self.queue,
                                                                self.data_dir,
                                                                self.ali_dir,
                                                                self.spk2features,
                                                                self.spk2num_frames,
                                                                self.utt2num_frames,
                                                                self.left_context,
                                                                self.right_context,
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


def batch_sequence_v2(stop_event,
                      queue,
                      data_dir,
                      ali_dir,
                      feature_list,
                      features2spk,
                      left_context,
                      right_context,
                      batch_size=128,
                      min_len=200,
                      max_len=400,
                      shuffle=True,
                      seed=0):
    """Load features and fill a queue. Used in KaldiDataSeqQueue.

    Args:
        stop_event: An event indicating the reading is finished.
        queue: A queue to put the data.
        data_dir: The kaldi data directory.
        ali_dir: The kaldi ali directory.
        feature_list: A list shows which features the process should read.
        features2spk: A dict map features to speaker index.
        left_context: The left context.
        right_context: The right context.
        batch_size: The batch_size
        min_len: The minimum length of the features.
        max_len: The maximum length of the features.
        shuffle: Load the feature from the 0-th frame or a random frame.
        seed: The number is used to generate a random seed
    """
    # Read the comment in batch_random
    rd = random.Random(os.urandom(4))
    rd.jumpahead(seed)

    feature_reader = FeatureReaderV2(data_dir, ali_dir, left_context, right_context)
    num_batches = int(len(feature_list) / batch_size)
    for i in range(num_batches):
        batch_length = rd.randint(min_len, max_len)
        # In some cases, the minimum length of the utterances is smaller than the batch length.
        # Use the smallest length as the real batch length.
        for j in range(batch_size):
            if feature_reader.utt2num_frames[feature_list[i * batch_size + j].split(' ')[0]] < batch_length:
                batch_length = feature_reader.utt2num_frames[feature_list[i * batch_size + j].split(' ')[0]]

        # Feature expansion is applied.
        features = np.zeros((batch_size, batch_length + left_context + right_context, feature_reader.dim), dtype=np.float32)
        vad = np.zeros((batch_size, batch_length), dtype=np.float32)
        ali = np.zeros((batch_size, batch_length), dtype=np.int32)
        labels = np.zeros((batch_size), dtype=np.int32)
        valid_length = np.zeros((batch_size), dtype=np.int32)
        resample = np.zeros((batch_size), dtype=np.int32)
        valid_pos = np.zeros((batch_size, 2), dtype=np.int32)
        for j in range(batch_size):
            utt_feat, utt_vad, utt_ali, utt_start = feature_reader.read_segment(feature_list[i * batch_size + j], batch_length, shuffle=shuffle)
            labels[j] = features2spk[feature_list[i * batch_size + j]]
            utt_length = utt_feat.shape[0] - left_context - right_context
            features[j, :utt_feat.shape[0], :] = utt_feat
            # Expand the feature by the last frame if the length is not long enough.
            if utt_length < batch_length:
                features[j, utt_feat.shape[0]:, :] = features[j, utt_feat.shape[0] - 1, :]
            vad[j, :utt_length] = utt_vad
            ali[j, :utt_length] = utt_ali
            valid_length[j] = utt_length
            resample[j] = 1 if utt_start + utt_length == feature_reader.utt2num_frames[feature_list[i * batch_size + j]] else 0
            valid_pos[j, 0] = left_context - utt_start if left_context > utt_start else 0
            valid_pos[j, 1] = feature_reader.utt2num_frames[feature_list[i * batch_size + j]] - utt_start - right_context \
                if utt_start + utt_length > feature_reader.utt2num_frames[feature_list[i * batch_size + j]] - right_context else utt_length

        queue.put((features, vad, ali, valid_length, labels, resample, valid_pos))
    stop_event.set()
    print("The process {} is about to exit.".format(os.getpid()))
    return


class KaldiDataSeqQueueV2(object):
    """A queue to read features from Kaldi data directory."""

    def __init__(self, data_dir,
                 ali_dir,
                 spklist,
                 num_parallel=1,
                 max_qsize=10,
                 left_context=None,
                 right_context=None,
                 batch_size=128,
                 min_len=None,
                 max_len=None,
                 shuffle=True):
        """ Create a queue from a given directory.

        Load features sequentially.

        Args:
            data_dir: The kaldi data directory.
            ali_dir: The kaldi ali directory.
            spklist: The spklist tells the mapping from the speaker name to the speaker id.
            num_parallel: The number of threads to read features.
            max_qsize: The capacity of the queue.
            batch_size: The batch size.
            min_len: The minimum length of the sampled sequence.
            max_len: The maximum length of the sampled sequence.
            shuffle: Shuffle the load sequence and loading data from a random frame.
        """
        self.data_dir = data_dir
        self.ali_dir = ali_dir
        self.batch_size = batch_size
        self.min_len = min_len
        self.max_len = max_len
        self.num_parallel_datasets = num_parallel
        self.shuffle = shuffle
        self.left_context = left_context
        self.right_context = right_context

        # We process the data directory and fetch speaker information.
        self.spk2features, self.features2spk, spk2index = get_speaker_info(data_dir, spklist)
        self.num_total_speakers = len(list(spk2index.keys()))
        self.num_total_phones = int(
            subprocess.check_output("tree-info %s | grep num-pdfs | awk '{print $2}'" % os.path.join(self.ali_dir, "tree"), shell=True))

        # Restore the relation of utt and spk. That's annoying...
        self.spk2features_new = {}
        for spk in self.spk2features:
            self.spk2features_new[spk] = []
            for utt in self.spk2features[spk]:
                self.spk2features_new[spk].append(utt.split(" ")[0])
        self.spk2features = self.spk2features_new

        self.features2spk_new = {}
        for utt in self.features2spk:
            self.features2spk_new[utt.split(" ")[0]] = self.features2spk[utt]
        self.features2spk = self.features2spk_new

        # Arrange features in sequence
        self.feature_list = list(self.features2spk.keys())
        if shuffle:
            random.shuffle(self.feature_list)

        self.sub_feature_list = []
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
        # And the processes are saved
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
        self.processes = [Process(target=batch_sequence_v2, args=(self.stop_event[i],
                                                                  self.queue,
                                                                  self.data_dir,
                                                                  self.ali_dir,
                                                                  self.sub_feature_list[i],
                                                                  self.features2spk,
                                                                  self.left_context,
                                                                  self.right_context,
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


if __name__ == "__main__":
    data_dir = "/home/heliang05/liuyi/fisher.full/data/train_background_hires_multitask/train"
    ali_dir = "/home/heliang05/liuyi/fisher.full/exp/tri5a_ali_3k"
    spklist = "/home/heliang05/liuyi/fisher.full/data/train_background_hires_multitask/train/spklist"
    num_speakers = 64
    num_segments = 1
    min_len = 100
    max_len = 300
    num_parallel_datasets = 1
    shuffle = True
    left_context = 20
    right_context = 20
    seed = 1

    batch_size = num_speakers * num_segments
    import pdb

    spk2features, features2spk, spk2index = get_speaker_info(data_dir, spklist)
    spk2features_new = {}
    for spk in spk2features:
        spk2features_new[spk] = []
        for utt in spk2features[spk]:
            spk2features_new[spk].append(utt.split(" ")[0])
    spk2features = spk2features_new
    features2spk_new = {}
    for utt in features2spk:
        features2spk_new[utt.split(" ")[0]] = features2spk[utt]
    features2spk = features2spk_new
    utt2num_frames = {}
    with open(os.path.join(data_dir, "utt2num_frames"), 'r') as f:
        for line in f.readlines():
            utt, n = line.strip().split(" ")
            utt2num_frames[utt] = int(n)
    spk2num_frames = {}
    for spk in spk2features:
        n = 0
        for utt in spk2features[spk]:
            n += utt2num_frames[utt]
        spk2num_frames[spk] = n
    num_total_speakers = len(list(spk2index.keys()))
    num_total_phones = int(subprocess.check_output("tree-info %s | grep num-pdfs | awk '{print $2}'" %
                                                        os.path.join(ali_dir, "tree"), shell=True))

    rd = random.Random(os.urandom(4))
    rd.jumpahead(seed)

    feature_reader = FeatureReaderV2(data_dir, ali_dir, left_context, right_context)
    speakers = list(spk2features.keys())

    total_num_frames = np.sum(spk2num_frames.values())
    spk_sample_region = []
    current_region = 0
    for spk in speakers:
        current_region += spk2num_frames[spk]
        spk_sample_region.append(current_region)
    assert total_num_frames == current_region

    spk2utt_sample_region = {}
    for spk in speakers:
        spk2utt_sample_region[spk] = []
        current_region = 0
        for utt in spk2features[spk]:
            current_region += utt2num_frames[utt]
            spk2utt_sample_region[spk].append(current_region)

    batch_speakers = sample_with_probability(rd, speakers, num_speakers, spk_sample_region)
    batch_length = rd.randint(min_len, max_len)
    # The feature should be expanded
    features = np.zeros((num_speakers * num_segments, batch_length + left_context + right_context, feature_reader.dim), dtype=np.float32)
    # The other variables still have the original length
    vad = np.zeros((num_speakers * num_segments, batch_length), dtype=np.float32)
    ali = np.zeros((num_speakers * num_segments, batch_length), dtype=np.int32)
    labels = np.zeros((num_speakers * num_segments), dtype=np.int32)
    valid_length = np.zeros((num_speakers * num_segments), dtype=np.int32)
    valid_pos = np.zeros((num_speakers * num_segments, 2), dtype=np.int32)
    # resample indicates whether we should sample the segment to find a phonetic training example
    # or just use the beginning of the segment.
    # Useful when training a network.
    resample = np.zeros((num_speakers * num_segments), dtype=np.int32)

    for i, speaker in enumerate(batch_speakers):
        spk = speaker
        # The speaker should have enough number of speakers
        assert spk2features[spk] > num_segments, "Speaker %s does not have enough segments to sample" % spk
        labels[i * num_segments:(i + 1) * num_segments] = spk

        batch_segments = sample_with_probability(rd, spk2features[spk], num_segments, spk2utt_sample_region[spk])
        for j, utt in enumerate(batch_segments):
            utt_feat, utt_vad, utt_ali, utt_start = feature_reader.read_segment(utt, batch_length, shuffle=shuffle)
            # utt_length is the valid length before feature expansion.
            # The valid length excludes the expanded features.
            utt_length = utt_feat.shape[0] - left_context - right_context
            features[i * num_segments + j, :utt_feat.shape[0], :] = utt_feat
            # Expand the feature by the last frame if the length is not long enough.
            # (Though this is not necessary...)
            if utt_length < batch_length:
                features[i * num_segments + j, utt_feat.shape[0]:, :] = features[i * num_segments + j, utt_feat.shape[0]-1, :]
            vad[i * num_segments + j, :utt_length] = utt_vad
            ali[i * num_segments + j, :utt_length] = utt_ali
            valid_length[i * num_segments + j] = utt_length
            resample[i * num_segments + j] = 1 if utt_start + utt_length == utt2num_frames[utt] else 0
            valid_pos[i * num_segments + j, 0] = left_context - utt_start if left_context > utt_start else 0
            valid_pos[i * num_segments + j, 1] = utt2num_frames[utt] - utt_start - right_context \
                if utt_start + utt_length > utt2num_frames[utt] - right_context else utt_length

    pdb.set_trace()

    # We process the data directory and fetch speaker information.
    spk2features, features2spk, spk2index = get_speaker_info(data_dir, spklist)
    num_total_speakers = len(list(spk2index.keys()))
    num_total_phones = int(
        subprocess.check_output("tree-info %s | grep num-pdfs | awk '{print $2}'" % os.path.join(ali_dir, "tree"),
                                shell=True))

    # Restore the relation of utt and spk. That's annoying...
    spk2features_new = {}
    for spk in spk2features:
        spk2features_new[spk] = []
        for utt in spk2features[spk]:
            spk2features_new[spk].append(utt.split(" ")[0])
    spk2features = spk2features_new

    features2spk_new = {}
    for utt in features2spk:
        features2spk_new[utt.split(" ")[0]] = features2spk[utt]
    features2spk = features2spk_new

    # Arrange features in sequence
    feature_list = list(features2spk.keys())
    if shuffle:
        random.shuffle(feature_list)

    sub_feature_list = []
    num_sub_features = len(feature_list) / 1
    for i in range(1):
        if i == 1 - 1:
            sub_feature_list.append(feature_list[i * num_sub_features:])
        else:
            sub_feature_list.append(feature_list[i * num_sub_features:(i + 1) * num_sub_features])

    rd = random.Random(os.urandom(4))
    rd.jumpahead(seed)

    feature_reader = FeatureReaderV2(data_dir, ali_dir, left_context, right_context)
    num_batches = int(len(feature_list) / batch_size)

    batch_length = rd.randint(min_len, max_len)
    # In some cases, the minimum length of the utterances is smaller than the batch length.
    # Use the smallest length as the real batch length.
    for j in range(batch_size):
        if feature_reader.utt2num_frames[feature_list[i * batch_size + j].split(' ')[0]] < batch_length:
            batch_length = feature_reader.utt2num_frames[feature_list[i * batch_size + j].split(' ')[0]]

    # Feature expansion is applied.
    features = np.zeros((batch_size, batch_length + left_context + right_context, feature_reader.dim),
                        dtype=np.float32)
    vad = np.zeros((batch_size, batch_length), dtype=np.float32)
    ali = np.zeros((batch_size, batch_length), dtype=np.int32)
    labels = np.zeros((batch_size), dtype=np.int32)
    valid_length = np.zeros((batch_size), dtype=np.int32)
    resample = np.zeros((batch_size), dtype=np.int32)
    valid_pos = np.zeros((batch_size, 2), dtype=np.int32)
    i = 0
    for j in range(batch_size):
        utt_feat, utt_vad, utt_ali, utt_start = feature_reader.read_segment(feature_list[i * batch_size + j],
                                                                            batch_length, shuffle=shuffle)
        labels[j] = features2spk[feature_list[i * batch_size + j]]
        utt_length = utt_feat.shape[0] - left_context - right_context
        features[j, :utt_feat.shape[0], :] = utt_feat
        # Expand the feature by the last frame if the length is not long enough.
        if utt_length < batch_length:
            features[j, utt_feat.shape[0]:, :] = features[j, utt_feat.shape[0] - 1, :]
        vad[j, :utt_length] = utt_vad
        ali[j, :utt_length] = utt_ali
        valid_length[j] = utt_length
        resample[j] = 1 if utt_start + utt_length == feature_reader.utt2num_frames[
            feature_list[i * batch_size + j]] else 0
        valid_pos[j, 0] = left_context - utt_start if left_context > utt_start else 0
        valid_pos[j, 1] = feature_reader.utt2num_frames[feature_list[i * batch_size + j]] - utt_start - right_context \
            if utt_start + utt_length > feature_reader.utt2num_frames[
            feature_list[i * batch_size + j]] - right_context else utt_length

    pdb.set_trace()

    feat_reader = KaldiDataRandomQueueV2(data_dir, ali_dir, spklist,
                                         left_context=left_context,
                                         right_context=right_context,
                                         num_speakers=num_speakers,
                                         num_segments=num_segments,
                                         min_len=min_len,
                                         max_len=max_len,
                                         shuffle=True)
    feat_reader.start()
    test = feat_reader.fetch()
    feat_reader.stop()
    pdb.set_trace()
    print(test)

    feat_reader = KaldiDataSeqQueueV2(data_dir, ali_dir, spklist,
                                      left_context=left_context,
                                      right_context=right_context,
                                      batch_size=batch_size,
                                      min_len=min_len,
                                      max_len=max_len,
                                      shuffle=shuffle)
    feat_reader.start()
    test = feat_reader.fetch()
    feat_reader.stop()
    pdb.set_trace()
    print(test)
