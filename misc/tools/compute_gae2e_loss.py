import sys
import numpy as np
from sklearn import metrics
from scipy.optimize import brentq
from scipy.interpolate import interp1d

margin = 0.8
target_margin = 1.0

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('usage : %s weights embeddings' % sys.argv[0])
        quit()

    weights = sys.argv[1]
    embeddings = sys.argv[2]

    weights = np.loadtxt(weights)
    embeddings = np.loadtxt(embeddings)

    labels = np.array(embeddings[:, 0], dtype=np.int32)
    embeddings = embeddings[:, 1:]

    weights_norm = weights / np.sqrt(np.sum(weights ** 2, axis=1, keepdims=True))
    embeddings_norm = embeddings / np.sqrt(np.sum(embeddings ** 2, axis=1, keepdims=True))

    # Compute loss (hardest negative)
    loss_hardest = 0
    n_hard = 0
    loss = 0
    loss_center = 0
    n_center = 0
    loss_between = 0
    n_between = 0
    loss_new = 0
    n_new = 0
    loss_center_new = 0
    n_center_new = 0
    score_mat = []
    for i in range(embeddings_norm.shape[0]):
        dist = np.sum((embeddings_norm[i, :] - weights_norm) ** 2, axis=1)
        score_mat.append(dist)
        min_neg = 1e8
        for j in range(weights_norm.shape[0]):
            if labels[i] == j:
                continue
            if dist[j] < min_neg:
                min_neg = dist[j]
            loss += dist[labels[i]] - dist[j]
            if dist[labels[i]] - dist[j] + margin > 0:
                if dist[labels[i]] > target_margin:
                    loss_new += dist[labels[i]] - dist[j] + margin
                    n_new += 1
            loss_between -= dist[j]
            n_between += 1
        if dist[labels[i]] > target_margin:
            loss_hardest += dist[labels[i]] - min_neg
        n_hard += 1
        # print("%f %f %f" % (dist[labels[i]], min_neg, dist[labels[i]] - min_neg))
        if dist[labels[i]] > target_margin:
            loss_center += dist[labels[i]]
        n_center += 1
    loss_hardest /= n_hard
    loss /= embeddings_norm.shape[0] * (weights_norm.shape[0] - 1)
    loss_center /= n_center
    loss_between /= embeddings_norm.shape[0] * (weights_norm.shape[0] - 1)
    loss_new /= n_new

    score_mat = np.array(score_mat)
    print("Loss (hardest): %f" % loss_hardest)
    print("Loss: %f" % loss)
    print("Loss center: %f" % loss_center)
    print("Loss between: %f" % loss_between)
    print("Loss new 1: %f" % (loss_new + loss_center))
    print("Loss new 2: %f" % (loss_center + loss_between))

    # Compute EER (with weights)
    score = np.dot(embeddings_norm, np.transpose(weights_norm))
    keys = []

    for i in range(embeddings_norm.shape[0]):
        for j in range(weights_norm.shape[0]):
            if labels[i] == j:
                keys.append(1)
            else:
                keys.append(0)
    score = np.reshape(score, [score.shape[0] * score.shape[-1]])
    fpr, tpr, thresholds = metrics.roc_curve(keys, score, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

    print("EER with weights: %f%%" % (eer * 100))

    # Output score distribution
    fp_target = open("score.target", "w")
    fp_nontarget = open("score.nontarget", "w")

    for i in range(score.shape[0]):
        if keys[i] == 1:
            fp_target.write("%f\n" % score[i])
        else:
            fp_nontarget.write("%f\n" % score[i])

    fp_target.close()
    fp_nontarget.close()

    # Compute EER (with embeddings)
    score = np.dot(embeddings_norm, np.transpose(embeddings_norm))
    keys = []

    for i in range(embeddings_norm.shape[0]):
        for j in range(embeddings_norm.shape[0]):
            if labels[i] == labels[j]:
                keys.append(1)
            else:
                keys.append(0)
    score = np.reshape(score, [score.shape[0] * score.shape[-1]])
    fpr, tpr, thresholds = metrics.roc_curve(keys, score, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

    print("EER with embeddings: %f%%" % (eer * 100))
