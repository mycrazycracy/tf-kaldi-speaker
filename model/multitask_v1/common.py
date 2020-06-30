import numpy as np


def make_phone_masks(length, resample, num_frames_per_utt):
    """Randomly select frames for each utterance.

    Args:
        length: The length of each utterance.
        resample: If 0, return the beginning frame; otherwise random select a frame.
                  resample is designed to try to make every frame has the same probability to be sampled.
        num_frames_per_utt: #frames selected. if -1, then select all frames
    :return: a mat with [n_selected_frames, 2], each row is the index of the selected frame
    """
    n_utts = length.shape[0]

    # This sampling strategy will make the sampling probability of each frame the same
    if num_frames_per_utt == -1:
        mat = []
        for i in range(n_utts):
            for j in range(length[i]):
                mat.append([i, j])
        mat = np.array(mat, dtype=np.int32)
    else:
        # # Uniform sampling
        # mat = np.zeros((length.shape[0] * num_frames_per_utt, 2), dtype=np.int32)
        # assert num_frames_per_utt > 0, "The num of frames should be greater than 0 (or -1)"
        # for i in range(n_utts):
        #     mat[i * num_frames_per_utt:(i+1) * num_frames_per_utt, 0] = i
        #     if resample[i] == 1:
        #         # Resample the last segment
        #         tmp = []
        #         for _ in range(num_frames_per_utt):
        #             while True:
        #                 a = np.random.randint(0, length[i], dtype=np.int32)
        #                 if a not in tmp:
        #                     tmp.append(a)
        #                     break
        #         mat[i * num_frames_per_utt:(i + 1) * num_frames_per_utt, 1] = tmp
        #     else:
        #         mat[i * num_frames_per_utt:(i + 1) * num_frames_per_utt, 1] = np.arange(num_frames_per_utt, dtype=np.int32)

        # Totally random sampling (the central frames will get higher sampling probabilities)
        mat = np.zeros((length.shape[0] * num_frames_per_utt, 2), dtype=np.int32)
        for i in range(n_utts):
            mat[i * num_frames_per_utt:(i + 1) * num_frames_per_utt, 0] = i
            # Resample the last segment
            tmp = []
            for _ in range(num_frames_per_utt):
                while True:
                    a = np.random.randint(0, length[i], dtype=np.int32)
                    if a not in tmp:
                        tmp.append(a)
                        break
            mat[i * num_frames_per_utt:(i + 1) * num_frames_per_utt, 1] = tmp

    return mat
