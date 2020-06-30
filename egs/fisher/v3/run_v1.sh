#!/bin/bash
# The first version of multi-task learning.
# Only share the lower layers between the speaker and phonetic networks.

. ./cmd.sh
. ./path.sh
set -e

train_nj=40
nnet_nj=40
decode_nj=80

# The kaldi fisher egs directory
# kaldi_fisher=/home/liuyi/kaldi-master/egs/fisher
# We do not need real fisher egs here.
kaldi_fisher=/home/heliang05/liuyi/software/kaldi_gpu/egs/sre16

root=/home/heliang05/liuyi/fisher.full
data=$root/data
exp=$root/exp
mfccdir=$root/mfcc
vaddir=$root/mfcc

gmmdir=$exp/tri5a_3k
alidir=$exp/tri5a_ali_3k
transmdl=$gmmdir/trans

export trials=$data/test/trials

stage=6

if [ $stage -le -2 ]; then
  # Make graph for ASR decoding
  LM_fg=$data/local/lm_4gram/4gram-mincount/lm_unpruned.gz
  utils/build_const_arpa_lm.sh $LM_fg $data/lang_test $data/lang_test_fg
  utils/mkgraph.sh $data/lang_test $gmmdir $gmmdir/graph

  # Create transition model using the alignments
  # We need an additional file from Kaldi:
  #   src/bin/train-transitions-prior.cc
  mkdir -p $transmdl
  train-transitions-prior $gmmdir/final.mdl "ark:gunzip -c $gmmdir/ali.*.gz|" $transmdl/final.trans_mdl $transmdl/prior.vec
  exit 1
fi

if [ $stage -le -1 ]; then
  # link the directories
  rm -fr utils steps sid conf local nnet scripts
  ln -s $kaldi_fisher/v2/utils ./
  ln -s $kaldi_fisher/v2/steps ./
  ln -s $kaldi_fisher/v2/sid ./
  ln -s $kaldi_fisher/v2/local ./
  ln -s ../../voxceleb/v1/nnet ./
  ln -s $TF_KALDI_ROOT/scripts ./
  exit 1
fi

if [ $stage -le 0 ]; then
  # In this case, do wcmvn and remain the silence frames.
  # train_background_hires_multitask should be the same with train_background_hires_filtered_wcmvn
  scripts/prepare_feats_for_multitask_egs.sh --nj $train_nj --cmd "$train_cmd" \
    $data/train_background_hires $data/train_background_hires_multitask $exp/train_background_hires_multitask
  utils/fix_data_dir.sh $data/train_background_hires_multitask
fi


if [ $stage -le 1 ]; then
  script/prepare_pdf_for_multitask_egs.sh $data/train_background_hires_multitask $alidir

  # We need to filter the data so that each utterance has the alignment.
  utils/filter_scp.pl $alidir/pdf.scp $data/train_background_hires_multitask/utt2spk > $data/train_background_hires_multitask/utt2spk.new
  mv $data/train_background_hires_multitask/utt2spk.new $data/train_background_hires_multitask/utt2spk
  utils/fix_data_dir.sh $data/train_background_hires_multitask
fi


if [ $stage -le 2 ]; then
  mkdir -p $data/train_background_hires_multitask.bak
  cp -r $data/train_background_hires_multitask/* $data/train_background_hires_multitask.bak

  # remove speakers with too little data
  min_len=150
  mv $data/train_background_hires_multitask/utt2num_frames $data/train_background_hires_multitask/utt2num_frames.bak
  awk -v min_len=${min_len} '$2 > min_len {print $1, $2}' $data/train_background_hires_multitask/utt2num_frames.bak > $data/train_background_hires_multitask/utt2num_frames
  utils/filter_scp.pl $data/train_background_hires_multitask/utt2num_frames $data/train_background_hires_multitask/utt2spk > $data/train_background_hires_multitask/utt2spk.new
  mv $data/train_background_hires_multitask/utt2spk.new $data/train_background_hires_multitask/utt2spk
  utils/fix_data_dir.sh $data/train_background_hires_multitask

  min_num_utts=5
  awk '{print $1, NF-1}' $data/train_background_hires_multitask/spk2utt > $data/train_background_hires_multitask/spk2num
  awk -v min_num_utts=${min_num_utts} '$2 >= min_num_utts {print $1, $2}' $data/train_background_hires_multitask/spk2num | utils/filter_scp.pl - $data/train_background_hires_multitask/spk2utt > $data/train_background_hires_multitask/spk2utt.new
  mv $data/train_background_hires_multitask/spk2utt.new $data/train_background_hires_multitask/spk2utt
  utils/spk2utt_to_utt2spk.pl $data/train_background_hires_multitask/spk2utt > $data/train_background_hires_multitask/utt2spk
  utils/filter_scp.pl $data/train_background_hires_multitask/utt2spk $data/train_background_hires_multitask/utt2num_frames > $data/train_background_hires_multitask/utt2num_frames.new
  mv $data/train_background_hires_multitask/utt2num_frames.new $data/train_background_hires_multitask/utt2num_frames
  utils/fix_data_dir.sh $data/train_background_hires_multitask
fi

if [ $stage -le 3 ]; then
  make_train_valid.sh $data
  exit 1
fi

if [ $stage -le 4 ]; then
#  nnetdir=$exp/tuning_multitask/xvector_nnet_tdnn_softmax_1e-2
#  nnet/run_train_nnet.sh --cmd "$cuda_cmd" --env tf_gpu --continue-training false nnet_conf/tdnn_softmax_1e-2.json \
#    $data/train_background_hires_multitask/train $data/train_background_hires_multitask/train/spklist \
#    $data/train_background_hires_multitask/valid $data/train_background_hires_multitask/train/spklist \
#    $nnetdir

  nnetdir=$exp/tuning_multitask/xvector_nnet_tdnn_softmax_1e-2_subset_2
  nnet/run_train_nnet.sh --cmd "$cuda_cmd" --env tf_gpu --continue-training false nnet_conf/tdnn_softmax_1e-2.json \
    $data/train_background_hires_multitask_subset/train $data/train_background_hires_multitask_subset/train/spklist \
    $data/train_background_hires_multitask_subset/valid $data/train_background_hires_multitask_subset/train/spklist \
    $nnetdir

  exit 1
fi

if [ $stage -le 5 ]; then
  nnetdir=$exp/tuning_multitask/xvector_nnet_tdnn_softmax_1e-2
  checkpoint='last'

  nnet/run_extract_embeddings_no_vad.sh --cmd "$train_cmd" --nj $nnet_nj --use-gpu false --checkpoint $checkpoint --stage 0 \
    --chunk-size 10000 --normalize false --node "output" \
    $nnetdir $data/enroll_hires $nnetdir/xvectors_enroll_hires

  nnet/run_extract_embeddings_no_vad.sh --cmd "$train_cmd" --nj $nnet_nj --use-gpu false --checkpoint $checkpoint --stage 0 \
    --chunk-size 10000 --normalize false --node "output" \
    $nnetdir $data/test_hires $nnetdir/xvectors_test_hires

  # Cosine scoring
  $train_cmd $nnetdir/xvector_scores_hires/log/test_cos.log \
    ivector-compute-dot-products "cat '$trials' | cut -d\  --fields=1,2 |" \
    "ark:ivector-mean ark:$data/enroll_hires/spk2utt scp:$nnetdir/xvectors_enroll_hires/xvector.scp ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-normalize-length scp:$nnetdir/xvectors_test_hires/xvector.scp ark:- |" \
    $nnetdir/xvector_scores_hires/test_cos

  eval_cos.sh $nnetdir
  exit 1
fi

if [ $stage -le 6 ]; then
  # Train a baseline x-vector using new data (without VAD)
#  nnetdir=$exp/tuning_multitask/xvector_mt_tdnn_softmax_1e-2_subset
#  nnet/run_train_mt_nnet.sh --cmd "$cuda_cmd" --env tf_gpu --continue-training false nnet_conf/mt_softmax.json \
#    $data/train_background_hires_multitask_subset/train $alidir $data/train_background_hires_multitask_subset/train/spklist \
#    $data/train_background_hires_multitask_subset/valid $alidir $data/train_background_hires_multitask_subset/train/spklist \
#    $nnetdir

#  # Train a baseline ASR using new data (re-implement ASR with Kaldi)
#  nnetdir=$exp/tuning_multitask/xvector_mt_tdnn_softmax_1e-2_subset_6
#  nnet/run_train_mt_nnet.sh --cmd "$cuda_cmd" --env tf_gpu --continue-training false nnet_conf/mt_softmax_6.json \
#    $data/train_background_hires_multitask_subset/train $alidir $data/train_background_hires_multitask_subset/train/spklist \
#    $data/train_background_hires_multitask_subset/valid $alidir $data/train_background_hires_multitask_subset/train/spklist \
#    $nnetdir

#  nnetdir=$exp/tuning_multitask/xvector_mt_tdnn_softmax_1e-2_subset_7
#  nnet/run_train_mt_nnet.sh --cmd "$cuda_cmd" --env tf_gpu --continue-training false nnet_conf/mt_softmax_7.json \
#    $data/train_background_hires_multitask_subset/train $alidir $data/train_background_hires_multitask_subset/train/spklist \
#    $data/train_background_hires_multitask_subset/valid $alidir $data/train_background_hires_multitask_subset/train/spklist \
#    $nnetdir

#  nnetdir=$exp/tuning_multitask/xvector_mt_tdnn_softmax_1e-2_subset_6.2
#  nnet/run_train_mt_nnet.sh --cmd "$cuda_cmd" --env tf_gpu --continue-training false nnet_conf/mt_softmax_6.json \
#    $data/train_background_hires_multitask_subset/train $alidir $data/train_background_hires_multitask_subset/train/spklist \
#    $data/train_background_hires_multitask_subset/valid $alidir $data/train_background_hires_multitask_subset/train/spklist \
#    $nnetdir

#  nnetdir=$exp/tuning_multitask/xvector_mt_tdnn_softmax_1e-2_subset_7.2
#  nnet/run_train_mt_nnet.sh --cmd "$cuda_cmd" --env tf_gpu --continue-training false nnet_conf/mt_softmax_7.json \
#    $data/train_background_hires_multitask_subset/train $alidir $data/train_background_hires_multitask_subset/train/spklist \
#    $data/train_background_hires_multitask_subset/valid $alidir $data/train_background_hires_multitask_subset/train/spklist \
#    $nnetdir

  nnetdir=$exp/tuning_multitask/xvector_mt_tdnn_softmax_1e-2_subset_8
  nnet/run_train_mt_nnet.sh --cmd "$cuda_cmd" --env tf_gpu --continue-training false nnet_conf/mt_softmax_8.json \
    $data/train_background_hires_multitask_subset/train $alidir $data/train_background_hires_multitask_subset/train/spklist \
    $data/train_background_hires_multitask_subset/valid $alidir $data/train_background_hires_multitask_subset/train/spklist \
    $nnetdir

#  nnetdir=$exp/tuning_multitask/xvector_mt_tdnn_softmax_1e-2_subset_8.2
#  nnet/run_train_mt_nnet.sh --cmd "$cuda_cmd" --env tf_gpu --continue-training false nnet_conf/mt_softmax_8.2.json \
#    $data/train_background_hires_multitask_subset/train $alidir $data/train_background_hires_multitask_subset/train/spklist \
#    $data/train_background_hires_multitask_subset/valid $alidir $data/train_background_hires_multitask_subset/train/spklist \
#    $nnetdir

#  nnetdir=$exp/tuning_multitask/xvector_mt_tdnn_softmax_1e-2_subset_8.3
#  nnet/run_train_mt_nnet.sh --cmd "$cuda_cmd" --env tf_gpu --continue-training false nnet_conf/mt_softmax_8.3.json \
#    $data/train_background_hires_multitask_subset/train $alidir $data/train_background_hires_multitask_subset/train/spklist \
#    $data/train_background_hires_multitask_subset/valid $alidir $data/train_background_hires_multitask_subset/train/spklist \
#    $nnetdir



  exit 1
fi

if [ $stage -le 7 ]; then
  scripts/prepare_pdf_for_multitask_egs.sh ${alidir}_train_background-ivector
  scripts/prepare_pdf_for_multitask_egs.sh ${alidir}_enroll
  scripts/prepare_pdf_for_multitask_egs.sh ${alidir}_test
fi

nnetdir=$exp/tuning_multitask/xvector_mt_tdnn_softmax_1e-2_subset_6
checkpoint='last'

if [ $stage -le 8 ]; then
  # Extract speaker embeddings
  nnet/run_extract_mt_embeddings.sh --cmd "$train_cmd" --nj $nnet_nj --use-gpu false --checkpoint $checkpoint --stage 0 \
    --chunk-size 10000 --normalize false --node "zs_mu_relu" \
    $nnetdir $data/enroll_hires ${alidir}_enroll $nnetdir/xvectors_enroll_hires

  nnet/run_extract_mt_embeddings.sh --cmd "$train_cmd" --nj $nnet_nj --use-gpu false --checkpoint $checkpoint --stage 0 \
    --chunk-size 10000 --normalize false --node "zs_mu_relu" \
    $nnetdir $data/test_hires ${alidir}_test $nnetdir/xvectors_test_hires
fi

if [ $stage -le 9 ]; then
  # Speaker scoring
  # Be careful if some utterances do not exist due to the alignment failure.
  $train_cmd $nnetdir/xvector_scores_hires/log/test_cos.log \
    ivector-compute-dot-products "cat '$trials' | cut -d\  --fields=1,2 |" \
    "ark:ivector-mean ark:$data/enroll_hires/spk2utt scp:$nnetdir/xvectors_enroll_hires/xvector.scp ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-normalize-length scp:$nnetdir/xvectors_test_hires/xvector.scp ark:- |" \
    $nnetdir/xvector_scores_hires/test_cos

  eval_cos.sh $nnetdir
  exit 1
fi

if [ $stage -le 10 ]; then
  # Extract phonetic embeddings or posteriors.
  nnet/run_extract_mt_phone_embeddings.sh --cmd "$train_cmd" --nj $nnet_nj --use-gpu false --checkpoint $checkpoint --stage 0 \
    --chunk-size 10000 --normalize false --node "zp_mu_relu" \
    $nnetdir $data/enroll_hires ${alidir}_enroll $nnetdir/xvectors_enroll_hires

  nnet/run_extract_mt_phone_embeddings.sh --cmd "$train_cmd" --nj $nnet_nj --use-gpu false --checkpoint $checkpoint --stage 0 \
    --chunk-size 10000 --normalize false --node "zp_mu_relu" \
    $nnetdir $data/test_hires ${alidir}_test $nnetdir/xvectors_test_hires
  exit 1
fi

if [ $stage -le 11 ]; then
  nnet/run_decode.sh --cmd "$train_cmd" --nj $decode_nj --use-gpu false --checkpoint $checkpoint --stage 0 \
    $gmmdir/graph $transmdl $nnetdir $data/test_hires $nnetdir/decode_test_hires

  # Rescore
  scripts/lmrescore_const_arpa.sh --stage 0 \
    --cmd "$train_cmd" ${data}/lang_test $data/lang_test_fg \
    ${data}/test_hires ${nnetdir}/decode_test_hires{,_fg} || exit 1
fi

