Due to the copyright issue, I can only publish the single GPU version which was developed before Jan. 2019. Some implementations should also be improved, e.g., the GPU memory allocation. The library can still be used as a framework for speaker verification. Multi-GPU and other approaches could be added with less efforts.  

**Note** When you extract the speaker embedding using extract.sh, make sure that your TensorFlow is compiled WITHOUT MKL. As I know, some versions of TF installed by anaconda are compiled with MKL. It will use multiple threads when TF is running on CPUs. This is harmful if you run multiple processes (say 40). The threads conflict will make the extraction extreamly slow. For me, I use pip to install TF 1.12, and that works. 

----

# Overview

The **tf-kaldi-speaker** implements a neural network based speaker verification system
using [Kaldi](https://github.com/kaldi-asr/kaldi) and [TensorFlow](https://github.com/tensorflow/tensorflow).

The main idea is that Kaldi can be used to do the pre- and post-processings
while TF is a better choice to build the neural network.
Compared with Kaldi nnet3, the modification of the network (e.g. adding attention, using different loss functions) using TF costs less.
Adding other features to support text-dependent speaker verification is also possible.

The purpose of the project is to make researches on neural network based speaker verification easier.
I also try to reproduce some results in my papers.


# Requirement

* Python: 2.7 (Update to 3.6/3.7 should be easy.)

* Kaldi: >5.5

    Since Kaldi is only used to do the pre- and post-processing, most version >5.2 works.
    Though I'm not 100% sure, I believe Kaldi with x-vector support (e.g. egs/sre16/v2) is enough.
    But if you want to run egs/voxceleb, make sure your Kaldi also contains this examples.

* Tensorflow: >1.4.0

    I write the code using TF 1.4.0 at the very beginning. Then I updated to v1.12.0.
    The future version will support TF >1.12 but I will try to make the API compatible with lower versions.
    Due to the API changes (e.g. keep_dims to keepdims in some functions), some may experience incorrect parameters.
    In that case, simply check the parameters may fix these problems.


# Methodology

The general pipeline of our framework is:

* For training:
1. Kaldi: Data preparation --> feature extraction --> training example generateion (CMVN + VAD + ...)
2. TF: Network training (training examples + nnet config)

* For test:
1. Kaldi: Data preparation --> feature extraction
2. TF: Embedding extraction
3. Kaldi: Backend classifier (Cosine/PLDA) --> performance evaluation

* Evaluate the performance:
    * MATLAB is used to compute the EER, minDCF08, minDCF10, minDCF12.
    * If you do not have MATLAB, Kaldi also provides scripts to compute the EER and minDCFs. The minDCF08 from Kaldi is 10x larger than DETware due to the computation method.

In our framework, the speaker embedding can be trained and extracted using different network architectures.
Again, the backend classifier is integrated using Kaldi.

# Features

* Entire pipeline of neural network based speaker verification.
* Both training from scratch and fine-tuning a pre-trained model are supported.
* Examples including VoxCeleb and SRE. Refer to Fisher to customized the dataset.
* Standard x-vector architecture (with minor modification).
* Angular softmax, additive margin softmax, additive angular margin softmax, triplet loss and other loss functions.
* Self attention

# Usage
 * The demos for SRE and VoxCeleb are included in egs/{sre,voxceleb}. Follow `run.sh` to go through the code.
 * The neural networks are configured using JSON files which are included in nnet_conf and the usage of the parameters is exhibited in the demos.

# Performance & Speed

* Performance

    I've test the code on three datasets and the results are better than the standard Kaldi recipe. (Of course, you can achieve better performance using Kaldi by carefully tuning the parameters.)

    See [RESULTS](./RESULTS.md) for details.

* Speed

    Since it only support single gpu, the speed is not very fast but acceptable in medium-scale datasets.
    For VoxCeleb, the training takes about 2.5 days using Nvidia P100 and it takes ~4 days for SRE.
    

# Pretrained models

* VoxCeleb

	Training data: VoxCeleb1 dev set and VoxCeleb2
	
	[Google Drive](https://drive.google.com/open?id=1ELcqFifG8bqeMqAu3BDef8S206rTvieD) and 
	
	[BaiduYunDisk](https://pan.baidu.com/s/1zByC_zwY9YM5bhkJzH4gNQ) (extraction code: xwu6)

* NIST SRE

	Training data: NIST SRE04-08, SWBD
	
	Only the models trained with large margin softmax are released at this moment.
	
	[Google Drive](https://drive.google.com/open?id=1yWE3yLiSsCSz-EPdUAstyozHWcgcThLQ) and
	
	[BaiduYunDisk](https://pan.baidu.com/s/1MVz7haYgozQ8ViJlII_YFw) (extraction code: rt9p)


# Pros and cons

* Advantages
    1. Performance: The performance of our code is shown to perform better than Kaldi.
    2. Storage: There is no need to generate a *packed* egs as Kaldi when training the network. The training will load the data on the fly.
    3. Flexibility: Changing the network architecture and loss function is pretty easy.

* Disadvantages
    1. Training speed: Due to the copyright issue, although the multi-gpu version is implemented, only the single GPU is supported in the public version. 
    2. Since no packed egs are generated. Multiple CPUs must be used to load the data during training.
    This is a overhead when the utterances are very long. You have to assign enough CPUs to make the loading speech fast enough to match the GPU processing speed.

# Other discussions

* In this code, I provide two possible methods to tune the learning rate when SGD is used: using validation set and using fixed file.
The first method works well but it may take longer to train the network.

* More complicated network architectures could be implemented (similar to the TDNN in model/tdnn.py). Deeper network is worth trying since we have enough training data. That would result in better performance.

# License

**Apache License, Version 2.0 (Refer to [LICENCE](./LICENSE))**

# Acknowledgements

The computational resources are initially provided by Prof. Mark Gales in Cambridge University Engineering Department (CUED). After my visiting to Cambridge, the resources are mainly supported by Dr. Liang He in Tsinghua University Electronic Engineering Department (THUEE).


# Last ...

* Unfortunately, the code is developed under Windows. The file property cannot be maintained properly.
After downloading the code, simply run:
    ```
    find ./ -name "*.sh" | awk '{print "chmod +x "$1}' | sh
    ```
    to add the 'x' property to the .sh files.

* For cluster setup, please refer to [Kaldi](http://kaldi-asr.org/doc/queue.html) for help.
    In my case, the program is run locally.
    Modify cmd.sh and path.sh just according to standard Kaldi setup.

* Contact:

    Website: <http://yiliu.org.cn>

    E-mail: liu-yi15 (at) tsinghua {dot} org {dot}cn


# Related papers

```
@inproceedings{liu2019speaker,
   author={Yi Liu and Liang He and Jia Liu},
   Title = {Large Margin Softmax Loss for Speaker Verification},
   BookTitle = {Proc. INTERSPEECH},
   Year = {2019}
}
```
