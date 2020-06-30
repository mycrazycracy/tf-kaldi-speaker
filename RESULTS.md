## VoxCeleb

The pipeline is the basically the same with Kaldi egs/voxceleb recipe expect for the network training.
The Ring loss and MHE need carefully tuning to achieve good performance.

Note: The official training set is VoxCeleb 2 dev and test on VoxCeleb 1.
During the development, I use Kaldi as the baseline. I train the network on VoxCeleb 1 dev & VoxCeleb 2 instead.
This results in different performance in the table.

| Network | Pooling | Loss | Training set | EER(%) | minDCF08 | minDCF10 |
| ------- | ------- | ---- | ------------ | :------: | :--------: | :--------: |
| Thin ResNet-34 [1] | TAP | Softmax | VoxCeleb2 dev | 10.48 | - | - |
| Thin ResNet-34 [1] | GhostVLAD | Softmax | VoxCeleb2 dev | 3.22 | - | - |
| Kaldi [2] | Stat | Softmax | VoxCeleb2 + VoxCeleb1 dev | 3.10 | 0.0169 | 0.4977 |
| TDNN (ours) | Stat | Softmax | VoxCeleb2 + VoxCeleb1 dev | 2.34 | 0.0122 | 0.3754 |
| TDNN (ours) | Stat | ASoftmax (m=1) | VoxCeleb2 + VoxCeleb1 dev | 2.62 | 0.0131 | 0.4146 |
| TDNN (ours) | Stat | ASoftmax (m=2) | VoxCeleb2 + VoxCeleb1 dev | 2.18 | 0.0119 | 0.3791 |
| TDNN (ours) | Stat | ASoftmax (m=4) | VoxCeleb2 + VoxCeleb1 dev | 2.15 | 0.0113 | 0.3108 |
| TDNN (ours) | Stat | ArcSoftmax (m=0.20) | VoxCeleb2 + VoxCeleb1 dev | 2.14 | 0.0119 | 0.3610 |
| TDNN (ours) | Stat | ArcSoftmax (m=0.25) | VoxCeleb2 + VoxCeleb1 dev | 2.03 | 0.0120 | 0.4010 |
| TDNN (ours) | Stat | ArcSoftmax (m=0.30) | VoxCeleb2 + VoxCeleb1 dev | 2.12 | 0.0115 | 0.3138 |
| TDNN (ours) | Stat | ArcSoftmax (m=0.35) | VoxCeleb2 + VoxCeleb1 dev | 2.23 | 0.0123 | 0.3622 |
| TDNN (ours) | Stat | AMSoftmax (m=0.15) | VoxCeleb2 + VoxCeleb1 dev | 2.13 | 0.0113 | 0.3707 |
| TDNN (ours) | Stat | AMSoftmax (m=0.20) | VoxCeleb2 + VoxCeleb1 dev | 2.04 | 0.0111 | 0.2922 |
| TDNN (ours) | Stat | AMSoftmax (m=0.25) | VoxCeleb2 + VoxCeleb1 dev | 2.15 | 0.0119 | 0.3559 |
| TDNN (ours) | Stat | AMSoftmax (m=0.30) | VoxCeleb2 + VoxCeleb1 dev | 2.18 | 0.0115 | 0.3152 |
| TDNN (ours) | Stat | AMSoftmax (m=0.20) + Ring Loss ($\lambda=0.01$) | VoxCeleb2 + VoxCeleb1 dev | 2.07 | 0.0107 | 0.2687 |
| TDNN (ours) | Stat | AMSoftmax (m=0.20) + MHE ($\lambda=0.01$) | VoxCeleb2 + VoxCeleb1 dev | 2.00 | 0.0106 | 0.2487 |


## NIST SRE

The pipeline is the same with Kaldi egs/sre10. For both SRE10 and SRE16, the results are reported in the pooled trials.

| Network | Pooling | Loss | Training set | SRE10 EER(%) | minDCF08 | minDCF10 | SRE16 EER(%) | minDCF08 | minDCF10 |
| ------- | ------- | ---- | ------------ | :------: | :--------: | :--------: | :------: | :--------: | :--------: |
| Kaldi | Stat | Softmax | SRE04-SRE08 + SWBD | 1.68 | 0.0095 | 0.3764 | 8.95 | 0.0384 | 0.8671 |
| TF/L2/LReLU/Att [3] | Att | Softmax | SRE04-08,12 + Mixer6 + Fisher + SWBD + VoxCeleb1&2 | - | - | - | 7.06 | - | - |
| TDNN (ours) | Stat | Softmax | SRE04-SRE08 + SWBD | 1.49 | 0.0084 | 0.3672 | 7.72 | 0.0330 | 0.8301 |
| TDNN (ours) | Stat | ASoftmax (m=1) | SRE04-SRE08 + SWBD | 1.35 | 0.0075 | 0.2976 | 7.82 | 0.0327 | 0.7867 |
| TDNN (ours) | Stat | ASoftmax (m=2) | SRE04-SRE08 + SWBD | 1.12 | 0.0065 | 0.2939 | 7.45 | 0.0314 | 0.7906 |
| TDNN (ours) | Stat | ASoftmax (m=4) | SRE04-SRE08 + SWBD | 1.03 | 0.0061 | 0.3072 | 7.46 | 0.0317 | 0.8067 |
| TDNN (ours) | Stat | ArcSoftmax (m=0.10) | SRE04-SRE08 + SWBD | 1.12 | 0.0061 | 0.2804 | 7.47 | 0.0309 | 0.7787 |
| TDNN (ours) | Stat | ArcSoftmax (m=0.15) | SRE04-SRE08 + SWBD | 1.20 | 0.0070 | 0.2989 | 7.44 | 0.0312 | 0.7997 |
| TDNN (ours) | Stat | ArcSoftmax (m=0.20) | SRE04-SRE08 + SWBD | 1.25 | 0.0072 | 0.3373 | 7.49 | 0.0317 | 0.7960 |
| TDNN (ours) | Stat | AMSoftmax (m=0.10) | SRE04-SRE08 + SWBD | 1.29 | 0.0068 | 0.2916 | 7.57 | 0.0315 | 0.7893 |
| TDNN (ours) | Stat | AMSoftmax (m=0.15) | SRE04-SRE08 + SWBD | 1.00 | 0.0060 | 0.2731 | 7.28 | 0.0306 | 0.7748 |
| TDNN (ours) | Stat | AMSoftmax (m=0.20) | SRE04-SRE08 + SWBD | 1.18 | 0.0066 | 0.3069 | 7.42 | 0.0309 | 0.8150 |
| TDNN (ours) | Stat | AMSoftmax (m=0.25) | SRE04-SRE08 + SWBD | 1.26 | 0.0076 | 0.3117 | 7.60 | 0.0317 | 0.7885 |


[1] Xie, W., Nagrani, A., Chung, J. S. & Zisserman, A., Utterance-level Aggregation For Speaker Recognition In The Wild. arXiv preprint arXiv:1902.10107 (2019)

[2] Kaldi, egs/voxceleb/v2

[3] Zeinali, H., Burget, L., Rohdin, J., Stafylakis, T. & Cernocky, J., How to Improve Your Speaker Embeddings Extractor in Generic Toolkits. arXiv preprint arXiv:1811.02066 (2018).