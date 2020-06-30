#!/bin/bash


. ./cmd.sh
. ./path.sh
set -e

train_nj=16
nnet_nj=40

# The kaldi fisher egs directory
# kaldi_fisher=/home/liuyi/kaldi-master/egs/fisher
# We do not need real fisher egs here.
kaldi_fisher=/home/heliang05/liuyi/software/kaldi_gpu/egs/sre16

root=/home/heliang05/liuyi/fisher.full
data=$root/data
exp=$root/exp
mfccdir=$root/mfcc
vaddir=$root/mfcc

export trials=$data/test/trials

stage=3

if [ $stage -le -1 ]; then
    # link the directories
    rm -fr utils steps sid conf local nnet
    ln -s $kaldi_fisher/v2/utils ./
    ln -s $kaldi_fisher/v2/steps ./
    ln -s $kaldi_fisher/v2/sid ./
    ln -s $kaldi_fisher/v2/local ./
    ln -s ../../voxceleb/v1/nnet ./
    exit 1
fi


if [ $stage -le 0 ]; then
  local/nnet3/xvector/prepare_feats_for_egs_new.sh --nj $train_nj --cmd "$train_cmd" \
    $data/train_background_hires $data/train_background_hires_nosil $exp/train_background_hires_nosil
  utils/fix_data_dir.sh $data/train_background_hires_nosil
fi

if [ $stage -le 1 ]; then
  # have look at the length and num utts
  cp $data/train_background_hires_nosil/utt2num_frames ./
  awk '{print $1, NF-1}' $data/train_background_hires_nosil/spk2utt > ./spk2num
  mkdir -p $data/train_background_hires_nosil.bak
  cp -r $data/train_background_hires_nosil/* $data/train_background_hires_nosil.bak

  # remove speakers with too little data
  min_len=150
  mv $data/train_background_hires_nosil/utt2num_frames $data/train_background_hires_nosil/utt2num_frames.bak
  awk -v min_len=${min_len} '$2 > min_len {print $1, $2}' $data/train_background_hires_nosil/utt2num_frames.bak > $data/train_background_hires_nosil/utt2num_frames
  utils/filter_scp.pl $data/train_background_hires_nosil/utt2num_frames $data/train_background_hires_nosil/utt2spk > $data/train_background_hires_nosil/utt2spk.new
  mv $data/train_background_hires_nosil/utt2spk.new $data/train_background_hires_nosil/utt2spk
  utils/fix_data_dir.sh $data/train_background_hires_nosil

  min_num_utts=5
  awk '{print $1, NF-1}' $data/train_background_hires_nosil/spk2utt > $data/train_background_hires_nosil/spk2num
  awk -v min_num_utts=${min_num_utts} '$2 >= min_num_utts {print $1, $2}' $data/train_background_hires_nosil/spk2num | utils/filter_scp.pl - $data/train_background_hires_nosil/spk2utt > $data/train_background_hires_nosil/spk2utt.new
  mv $data/train_background_hires_nosil/spk2utt.new $data/train_background_hires_nosil/spk2utt
  utils/spk2utt_to_utt2spk.pl $data/train_background_hires_nosil/spk2utt > $data/train_background_hires_nosil/utt2spk
  utils/filter_scp.pl $data/train_background_hires_nosil/utt2spk $data/train_background_hires_nosil/utt2num_frames > $data/train_background_hires_nosil/utt2num_frames.new
  mv $data/train_background_hires_nosil/utt2num_frames.new $data/train_background_hires_nosil/utt2num_frames
  utils/fix_data_dir.sh $data/train_background_hires_nosil
fi

if [ $stage -le 2 ]; then
  make_train_valid.sh $data
  exit 1
fi


if [ $stage -le 3 ]; then
#  # Training a softmax network
#  nnetdir=$exp/xvector_nnet_tdnn_softmax_1e-2
#  nnet/run_train_nnet.sh --cmd "$cuda_cmd" --env tf_gpu --continue-training false nnet_conf/tdnn_softmax_1e-2.json \
#    $data/train_background_hires_nosil/train $data/train_background_hires_nosil/train/spklist \
#    $data/train_background_hires_nosil/valid $data/train_background_hires_nosil/train/spklist \
#    $nnetdir

#  # ASoftmax
#  nnetdir=$exp/xvector_nnet_tdnn_asoftmax_m1_1e-2
#  nnet/run_train_nnet.sh --cmd "$cuda_cmd" --env tf_gpu --continue-training false nnet_conf/tdnn_asoftmax_m1_1e-2.json \
#    $data/train_background_hires_nosil/train $data/train_background_hires_nosil/train/spklist \
#    $data/train_background_hires_nosil/valid $data/train_background_hires_nosil/train/spklist \
#    $nnetdir

#  nnetdir=$exp/xvector_nnet_tdnn_asoftmax_m2_linear_bn_1e-2
#  nnet/run_train_nnet.sh --cmd "$cuda_cmd" --env tf_gpu --continue-training false nnet_conf/tdnn_asoftmax_m2_linear_bn_1e-2.json \
#    $data/train_background_hires_nosil/train $data/train_background_hires_nosil/train/spklist \
#    $data/train_background_hires_nosil/valid $data/train_background_hires_nosil/train/spklist \
#    $nnetdir

#  nnetdir=$exp/xvector_nnet_tdnn_asoftmax_m4_linear_bn_1e-2
#  nnet/run_train_nnet.sh --cmd "$cuda_cmd" --env tf_gpu --continue-training false nnet_conf/tdnn_asoftmax_m4_linear_bn_1e-2.json \
#    $data/train_background_hires_nosil/train $data/train_background_hires_nosil/train/spklist \
#    $data/train_background_hires_nosil/valid $data/train_background_hires_nosil/train/spklist \
#    $nnetdir

#  # AMSoftmax
#  nnetdir=$exp/xvector_nnet_tdnn_amsoftmax_m0.10_linear_bn_1e-2
#  nnet/run_train_nnet.sh --cmd "$cuda_cmd" --env tf_gpu --continue-training false nnet_conf/tdnn_amsoftmax_m0.10_linear_bn_1e-2.json \
#    $data/train_background_hires_nosil/train $data/train_background_hires_nosil/train/spklist \
#    $data/train_background_hires_nosil/valid $data/train_background_hires_nosil/train/spklist \
#    $nnetdir

#  nnetdir=$exp/xvector_nnet_tdnn_amsoftmax_m0.15_linear_bn_1e-2
#  nnet/run_train_nnet.sh --cmd "$cuda_cmd" --env tf_gpu --continue-training false nnet_conf/tdnn_amsoftmax_m0.15_linear_bn_1e-2.json \
#    $data/train_background_hires_nosil/train $data/train_background_hires_nosil/train/spklist \
#    $data/train_background_hires_nosil/valid $data/train_background_hires_nosil/train/spklist \
#    $nnetdir

#  nnetdir=$exp/xvector_nnet_tdnn_amsoftmax_m0.20_linear_bn_1e-2
#  nnet/run_train_nnet.sh --cmd "$cuda_cmd" --env tf_gpu --continue-training false nnet_conf/tdnn_amsoftmax_m0.20_linear_bn_1e-2.json \
#    $data/train_background_hires_nosil/train $data/train_background_hires_nosil/train/spklist \
#    $data/train_background_hires_nosil/valid $data/train_background_hires_nosil/train/spklist \
#    $nnetdir

#  nnetdir=$exp/xvector_nnet_tdnn_amsoftmax_m0.25_linear_bn_1e-2
#  nnet/run_train_nnet.sh --cmd "$cuda_cmd" --env tf_gpu --continue-training false nnet_conf/tdnn_amsoftmax_m0.25_linear_bn_1e-2.json \
#    $data/train_background_hires_nosil/train $data/train_background_hires_nosil/train/spklist \
#    $data/train_background_hires_nosil/valid $data/train_background_hires_nosil/train/spklist \
#    $nnetdir

#  nnetdir=$exp/xvector_nnet_tdnn_amsoftmax_m0.30_linear_bn_1e-2
#  nnet/run_train_nnet.sh --cmd "$cuda_cmd" --env tf_gpu --continue-training false nnet_conf/tdnn_amsoftmax_m0.30_linear_bn_1e-2.json \
#    $data/train_background_hires_nosil/train $data/train_background_hires_nosil/train/spklist \
#    $data/train_background_hires_nosil/valid $data/train_background_hires_nosil/train/spklist \
#    $nnetdir

#  nnetdir=$exp/xvector_nnet_tdnn_amsoftmax_m0.35_linear_bn_1e-2
#  nnet/run_train_nnet.sh --cmd "$cuda_cmd" --env tf_gpu --continue-training false nnet_conf/tdnn_amsoftmax_m0.35_linear_bn_1e-2.json \
#    $data/train_background_hires_nosil/train $data/train_background_hires_nosil/train/spklist \
#    $data/train_background_hires_nosil/valid $data/train_background_hires_nosil/train/spklist \
#    $nnetdir

#  nnetdir=$exp/xvector_nnet_tdnn_amsoftmax_m0.45_linear_bn_1e-2
#  nnet/run_train_nnet.sh --cmd "$cuda_cmd" --env tf_gpu --continue-training false nnet_conf/tdnn_amsoftmax_m0.45_linear_bn_1e-2.json \
#    $data/train_background_hires_nosil/train $data/train_background_hires_nosil/train/spklist \
#    $data/train_background_hires_nosil/valid $data/train_background_hires_nosil/train/spklist \
#    $nnetdir

#  # ArcSoftmax
#  nnetdir=$exp/xvector_nnet_tdnn_arcsoftmax_m0.10_linear_bn_1e-2
#  nnet/run_train_nnet.sh --cmd "$cuda_cmd" --env tf_gpu --continue-training false nnet_conf/tdnn_arcsoftmax_m0.10_linear_bn_1e-2.json \
#    $data/train_background_hires_nosil/train $data/train_background_hires_nosil/train/spklist \
#    $data/train_background_hires_nosil/valid $data/train_background_hires_nosil/train/spklist \
#    $nnetdir

#  nnetdir=$exp/xvector_nnet_tdnn_arcsoftmax_m0.15_linear_bn_1e-2
#  nnet/run_train_nnet.sh --cmd "$cuda_cmd" --env tf_gpu --continue-training false nnet_conf/tdnn_arcsoftmax_m0.15_linear_bn_1e-2.json \
#    $data/train_background_hires_nosil/train $data/train_background_hires_nosil/train/spklist \
#    $data/train_background_hires_nosil/valid $data/train_background_hires_nosil/train/spklist \
#    $nnetdir

#  nnetdir=$exp/xvector_nnet_tdnn_arcsoftmax_m0.20_linear_bn_1e-2
#  nnet/run_train_nnet.sh --cmd "$cuda_cmd" --env tf_gpu --continue-training false nnet_conf/tdnn_arcsoftmax_m0.20_linear_bn_1e-2.json \
#    $data/train_background_hires_nosil/train $data/train_background_hires_nosil/train/spklist \
#    $data/train_background_hires_nosil/valid $data/train_background_hires_nosil/train/spklist \
#    $nnetdir

#  nnetdir=$exp/xvector_nnet_tdnn_arcsoftmax_m0.25_linear_bn_1e-2
#  nnet/run_train_nnet.sh --cmd "$cuda_cmd" --env tf_gpu --continue-training false nnet_conf/tdnn_arcsoftmax_m0.25_linear_bn_1e-2.json \
#    $data/train_background_hires_nosil/train $data/train_background_hires_nosil/train/spklist \
#    $data/train_background_hires_nosil/valid $data/train_background_hires_nosil/train/spklist \
#    $nnetdir

#  nnetdir=$exp/xvector_nnet_tdnn_arcsoftmax_m0.30_linear_bn_1e-2
#  nnet/run_train_nnet.sh --cmd "$cuda_cmd" --env tf_gpu --continue-training false nnet_conf/tdnn_arcsoftmax_m0.30_linear_bn_1e-2.json \
#    $data/train_background_hires_nosil/train $data/train_background_hires_nosil/train/spklist \
#    $data/train_background_hires_nosil/valid $data/train_background_hires_nosil/train/spklist \
#    $nnetdir

#  nnetdir=$exp/xvector_nnet_tdnn_arcsoftmax_m0.35_linear_bn_1e-2
#  nnet/run_train_nnet.sh --cmd "$cuda_cmd" --env tf_gpu --continue-training false nnet_conf/tdnn_arcsoftmax_m0.35_linear_bn_1e-2.json \
#    $data/train_background_hires_nosil/train $data/train_background_hires_nosil/train/spklist \
#    $data/train_background_hires_nosil/valid $data/train_background_hires_nosil/train/spklist \
#    $nnetdir

#  nnetdir=$exp/xvector_nnet_tdnn_arcsoftmax_m0.40_linear_bn_1e-2
#  nnet/run_train_nnet.sh --cmd "$cuda_cmd" --env tf_gpu --continue-training false nnet_conf/tdnn_arcsoftmax_m0.40_linear_bn_1e-2.json \
#    $data/train_background_hires_nosil/train $data/train_background_hires_nosil/train/spklist \
#    $data/train_background_hires_nosil/valid $data/train_background_hires_nosil/train/spklist \
#    $nnetdir

#  # Attention
#  nnetdir=$exp/xvector_nnet_tdnn_softmax_tdnn4_att.2
#  nnet/run_train_nnet.sh --cmd "$cuda_cmd" --env tf_gpu --continue-training false nnet_conf/tdnn_softmax_tdnn4_att.json \
#    $data/train_background_hires_nosil/train $data/train_background_hires_nosil/train/spklist \
#    $data/train_background_hires_nosil/valid $data/train_background_hires_nosil/train/spklist \
#    $nnetdir

#  nnetdir=$exp/xvector_nnet_tdnn_softmax_tdnn4_att_2.2
#  nnet/run_train_nnet.sh --cmd "$cuda_cmd" --env tf_gpu --continue-training false nnet_conf/tdnn_softmax_tdnn4_att_2.json \
#    $data/train_background_hires_nosil/train $data/train_background_hires_nosil/train/spklist \
#    $data/train_background_hires_nosil/valid $data/train_background_hires_nosil/train/spklist \
#    $nnetdir

#  nnetdir=$exp/xvector_nnet_tdnn_softmax_tdnn4_att_3.2
#  nnet/run_train_nnet.sh --cmd "$cuda_cmd" --env tf_gpu --continue-training false nnet_conf/tdnn_softmax_tdnn4_att_3.json \
#    $data/train_background_hires_nosil/train $data/train_background_hires_nosil/train/spklist \
#    $data/train_background_hires_nosil/valid $data/train_background_hires_nosil/train/spklist \
#    $nnetdir

  nnetdir=$exp/xvector_nnet_tdnn_softmax_tdnn4_att_4.2
  nnet/run_train_nnet.sh --cmd "$cuda_cmd" --env tf_gpu --continue-training false nnet_conf/tdnn_softmax_tdnn4_att_4.json \
    $data/train_background_hires_nosil/train $data/train_background_hires_nosil/train/spklist \
    $data/train_background_hires_nosil/valid $data/train_background_hires_nosil/train/spklist \
    $nnetdir

  exit 1
fi


nnetdir=$exp/xvector_nnet_tdnn_softmax_tdnn4_att_lr2
checkpoint='last'

if [ $stage -le 4 ]; then
  cp $data/train_background-ivector/vad.scp $data/train_background-ivector_hires
  nnet/run_extract_embeddings.sh --cmd "$train_cmd" --nj $nnet_nj --use-gpu false --checkpoint $checkpoint --stage 0 \
    --chunk-size 10000 --normalize false --node "tdnn6_dense" \
    $nnetdir $data/train_background-ivector_hires $nnetdir/xvectors_background-ivector_hires

  cp $data/enroll/vad.scp $data/enroll_hires
  nnet/run_extract_embeddings.sh --cmd "$train_cmd" --nj $nnet_nj --use-gpu false --checkpoint $checkpoint --stage 0 \
    --chunk-size 10000 --normalize false --node "tdnn6_dense" \
    $nnetdir $data/enroll_hires $nnetdir/xvectors_enroll_hires

  cp $data/test/vad.scp $data/test_hires
  nnet/run_extract_embeddings.sh --cmd "$train_cmd" --nj $nnet_nj --use-gpu false --checkpoint $checkpoint --stage 0 \
    --chunk-size 10000 --normalize false --node "tdnn6_dense" \
    $nnetdir $data/test_hires $nnetdir/xvectors_test_hires
fi

if [ $stage -le 5 ]; then
  lda_dim=150

  $train_cmd $nnetdir/xvectors_background-ivector_hires/log/compute_mean.log \
    ivector-mean scp:$nnetdir/xvectors_background-ivector_hires/xvector.scp $nnetdir/xvectors_background-ivector_hires/mean.vec || exit 1;

  $train_cmd $nnetdir/xvectors_background-ivector_hires/log/lda.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
    "ark:ivector-subtract-global-mean scp:$nnetdir/xvectors_background-ivector_hires/xvector.scp ark:- |" \
    ark:$data/train_background-ivector_hires/utt2spk $nnetdir/xvectors_background-ivector_hires/transform.mat || exit 1;

  #  Train the PLDA model.
  $train_cmd $nnetdir/xvectors_background-ivector_hires/log/plda_lda${lda_dim}.log \
    ivector-compute-plda ark:$data/train_background-ivector_hires/spk2utt \
    "ark:ivector-subtract-global-mean scp:$nnetdir/xvectors_background-ivector_hires/xvector.scp ark:- | transform-vec $nnetdir/xvectors_background-ivector_hires/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    $nnetdir/xvectors_background-ivector_hires/plda_lda${lda_dim} || exit 1;

  $train_cmd $nnetdir/xvector_scores_hires/log/test.log \
    ivector-plda-scoring --normalize-length=true \
    --num-utts=ark:$nnetdir/xvectors_enroll_hires/num_utts.ark \
    "ivector-copy-plda --smoothing=0.0 $nnetdir/xvectors_background-ivector_hires/plda_lda${lda_dim} - |" \
    "ark:ivector-mean ark:$data/enroll_hires/spk2utt scp:$nnetdir/xvectors_enroll_hires/xvector.scp ark:- | ivector-subtract-global-mean $nnetdir/xvectors_background-ivector_hires/mean.vec ark:- ark:- | transform-vec $nnetdir/xvectors_background-ivector_hires/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean $nnetdir/xvectors_background-ivector_hires/mean.vec scp:$nnetdir/xvectors_test_hires/xvector.scp ark:- | transform-vec $nnetdir/xvectors_background-ivector_hires/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$trials' | cut -d\  --fields=1,2 |" $nnetdir/xvector_scores_hires/test || exit 1;

  # LDA + Cosine scoring
  $train_cmd $nnetdir/xvector_scores_hires/log/test_lda_cos.log \
    ivector-compute-dot-products "cat '$trials' | cut -d\  --fields=1,2 |" \
    "ark:ivector-mean ark:$data/enroll_hires/spk2utt scp:$nnetdir/xvectors_enroll_hires/xvector.scp ark:- | ivector-subtract-global-mean $nnetdir/xvectors_background-ivector_hires/mean.vec ark:- ark:- | transform-vec $nnetdir/xvectors_background-ivector_hires/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean $nnetdir/xvectors_background-ivector_hires/mean.vec scp:$nnetdir/xvectors_test_hires/xvector.scp ark:- | transform-vec $nnetdir/xvectors_background-ivector_hires/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    $nnetdir/xvector_scores_hires/test_lda_cos

  eval_plda.sh $nnetdir
  exit 1
fi


nnetdir=$exp/xvector_nnet_tdnn_asoftmax_m1_1e-2
checkpoint='last'

if [ $stage -le 6 ]; then
  cp $data/enroll/vad.scp $data/enroll_hires
  nnet/run_extract_embeddings.sh --cmd "$train_cmd" --nj $nnet_nj --use-gpu false --checkpoint $checkpoint --stage 0 \
    --chunk-size 10000 --normalize false --node "output" \
    $nnetdir $data/enroll_hires $nnetdir/xvectors_enroll_hires

  cp $data/test/vad.scp $data/test_hires
  nnet/run_extract_embeddings.sh --cmd "$train_cmd" --nj $nnet_nj --use-gpu false --checkpoint $checkpoint --stage 0 \
    --chunk-size 10000 --normalize false --node "output" \
    $nnetdir $data/test_hires $nnetdir/xvectors_test_hires
fi

if [ $stage -le 7 ]; then
  # Cosine scoring
  $train_cmd $nnetdir/xvector_scores_hires/log/test_cos.log \
    ivector-compute-dot-products "cat '$trials' | cut -d\  --fields=1,2 |" \
    "ark:ivector-mean ark:$data/enroll_hires/spk2utt scp:$nnetdir/xvectors_enroll_hires/xvector.scp ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-normalize-length scp:$nnetdir/xvectors_test_hires/xvector.scp ark:- |" \
    $nnetdir/xvector_scores_hires/test_cos

  eval_cos.sh $nnetdir
  exit 1
fi


