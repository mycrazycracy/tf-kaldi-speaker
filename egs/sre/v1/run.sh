#!/bin/bash
# Copyright      2017   David Snyder
#                2017   Johns Hopkins University (Author: Daniel Garcia-Romero)
#                2017   Johns Hopkins University (Author: Daniel Povey)
#                2019   Yi Liu. Modified to support network training using TensorFlow
# Apache 2.0.
#
#
# See README.txt for more info on data required.
# Results (mostly EERs) are inline in comments below.
#
# This example demonstrates a "bare bones" NIST SRE 2016 recipe using xvectors.
# In the future, we will add score-normalization and a more effective form of
# PLDA domain adaptation.
#
# Pretrained models are available for this recipe.
# See http://kaldi-asr.org/models.html and
# https://david-ryan-snyder.github.io/2017/10/04/model_sre16_v2.html
# for details.

. ./cmd.sh
. ./path.sh
set -e

fea_nj=32
nnet_nj=80

data_root=/home/heliang05/liuyi/sre.full/sre16_kaldi_list/
root=/home/heliang05/liuyi/sre.full/
data=$root/data
exp=$root/exp
mfccdir=$root/mfcc
vaddir=$root/mfcc

gender=pool

export sre10_dir=$data_root/sre10_eval/
export sre10_train_c5_ext=$sre10_dir/coreext_c5/enroll/$gender/
export sre10_trials_c5_ext=$sre10_dir/coreext_c5/test/$gender/
export sre10_train_10s=$sre10_dir/10sec/enroll/$gender/
export sre10_trials_10s=$sre10_dir/10sec/test/$gender/

export sre16_trials=/home/heliang05/liuyi/sre.full/sre16_eval_test/trials
export sre16_trials_tgl=/home/heliang05/liuyi/sre.full/sre16_eval_test/trials_tgl
export sre16_trials_yue=/home/heliang05/liuyi/sre.full/sre16_eval_test/trials_yue

rirs_noises=/mnt/lv10/person/liuyi/ly_database/RIRS_NOISES/
musan=/mnt/lv10/person/liuyi/ly_database/musan/

# The kaldi sre egs directory
kaldi_sre=/home/heliang05/liuyi/software/kaldi_gpu/egs/sre16

stage=-1

if [ $stage -le -1 ]; then
    # link the directories
    rm -fr utils steps sid conf local nnet
    ln -s $kaldi_sre/v2/utils ./
    ln -s $kaldi_sre/v2/steps ./
    ln -s $kaldi_sre/v2/sid ./
    ln -s $kaldi_sre/v2/conf ./
    ln -s $kaldi_sre/v2/local ./
    ln -s ../../voxceleb/v1/nnet ./
    exit 1
fi

## Data preparation
if [ $stage -le 0 ]; then
#   # Prepare Mixer6
#   local/make_mx6.sh /mnt/lv10/person/liuyi/ly_database $data

  # combine all sre data (04-08) and Mixer6
  utils/combine_data.sh $data/sre \
      $data_root/sre2004 $data_root/sre2005_train $data_root/sre2005_test \
      $data_root/sre2006_train $data_root/sre2006_test $data_root/sre08
  utils/validate_data_dir.sh --no-text --no-feats $data/sre
  utils/fix_data_dir.sh $data/sre

  # combine all swbd data.
  utils/combine_data.sh $data/swbd \
      $data_root/swbd2_phase1_train $data_root/swbd2_phase2_train $data_root/swbd2_phase3_train \
      $data_root/swbd_cellular1_train $data_root/swbd_cellular2_train
  utils/validate_data_dir.sh --no-text --no-feats $data/swbd
  utils/fix_data_dir.sh $data/swbd

  # prepare unlabeled Cantonese and Tagalog development data.
  # prepare sre16 evaluation data.
  rm -rf $data/sre16_major && cp -r $data_root/sre16_major $data/sre16_major
  rm -rf $data/sre16_minor && cp -r $data_root/sre16_minor $data/sre16_minor
  rm -rf $data/sre16_eval_enroll && cp -r $data_root/sre16_eval_enroll $data/sre16_eval_enroll
  rm -rf $data/sre16_eval_test && cp -r $data_root/sre16_eval_test $data/sre16_eval_test

  # prepare sre10 evaluation data.
  rm -rf $data/sre10_enroll_coreext_c5_$gender && cp -r $sre10_train_c5_ext $data/sre10_enroll_coreext_c5_$gender
  rm -rf $data/sre10_test_coreext_c5_$gender && cp -r $sre10_trials_c5_ext $data/sre10_test_coreext_c5_$gender
  rm -rf $data/sre10_enroll_10s_$gender && cp -r $sre10_train_10s $data/sre10_enroll_10s_$gender
  rm -rf $data/sre10_test_10s_$gender && cp -r $sre10_trials_10s $data/sre10_test_10s_$gender
fi

if [ $stage -le 1 ]; then
  # Make filterbanks and compute the energy-based VAD for each dataset
  for name in sre swbd sre10_enroll_coreext_c5_$gender sre10_test_coreext_c5_$gender sre10_enroll_10s_$gender sre10_test_10s_$gender; do
    steps/make_mfcc.sh --write-utt2num-frames true --mfcc-config conf/mfcc.conf --nj $fea_nj --cmd "$train_cmd" \
      $data/$name $exp/make_mfcc $mfccdir
    utils/fix_data_dir.sh $data/$name
    sid/compute_vad_decision.sh --nj $fea_nj --cmd "$train_cmd" \
      $data/$name $exp/make_vad $vaddir
    utils/fix_data_dir.sh $data/$name
  done

  for name in sre16_major sre16_minor sre16_eval_enroll sre16_eval_test; do
    steps/make_mfcc.sh --write-utt2num-frames true --mfcc-config conf/mfcc.conf --nj $fea_nj --cmd "$train_cmd" \
      $data/$name $exp/make_mfcc $mfccdir
    utils/fix_data_dir.sh $data/$name
    sid/compute_vad_decision.sh --nj $fea_nj --cmd "$train_cmd" \
      $data/$name $exp/make_vad $vaddir
    utils/fix_data_dir.sh $data/$name
  done

  utils/combine_data.sh --extra-files "utt2num_frames" $data/swbd_sre $data/swbd $data/sre
  utils/fix_data_dir.sh $data/swbd_sre
fi

if [ $stage -le 2 ]; then
  frame_shift=0.01
  awk -v frame_shift=$frame_shift '{print $1, $2*frame_shift;}' $data/swbd_sre/utt2num_frames > $data/swbd_sre/reco2dur

  # Make a version with reverberated speech
  rvb_opts=()
  rvb_opts+=(--rir-set-parameters "0.5, $rirs_noises/simulated_rirs/smallroom/rir_list")
  rvb_opts+=(--rir-set-parameters "0.5, $rirs_noises/simulated_rirs/mediumroom/rir_list")

  # Make a reverberated version of the SWBD+SRE list.  Note that we don't add any
  # additive noise here.
  python steps/data/reverberate_data_dir.py \
    "${rvb_opts[@]}" \
    --speech-rvb-probability 1 \
    --pointsource-noise-addition-probability 0 \
    --isotropic-noise-addition-probability 0 \
    --num-replications 1 \
    --source-sampling-rate 8000 \
    $data/swbd_sre $data/swbd_sre_reverb
  cp $data/swbd_sre/vad.scp $data/swbd_sre_reverb/
  utils/copy_data_dir.sh --utt-suffix "-reverb" $data/swbd_sre_reverb $data/swbd_sre_reverb.new
  rm -rf $data/swbd_sre_reverb
  mv $data/swbd_sre_reverb.new $data/swbd_sre_reverb

  # Prepare the MUSAN corpus, which consists of music, speech, and noise
  # suitable for augmentation.
  local/make_musan.sh $musan $data

  # Get the duration of the MUSAN recordings.  This will be used by the
  # script augment_data_dir.py.
  for name in speech noise music; do
    utils/data/get_utt2dur.sh $data/musan_${name}
    mv $data/musan_${name}/utt2dur $data/musan_${name}/reco2dur
  done

  # Augment with musan_noise
  python steps/data/augment_data_dir_new.py --utt-suffix "noise" --fg-interval 1 --fg-snrs "15:10:5:0" --fg-noise-dir "$data/musan_noise" $data/swbd_sre $data/swbd_sre_noise
  # Augment with musan_music
  python steps/data/augment_data_dir_new.py --utt-suffix "music" --bg-snrs "15:10:8:5" --num-bg-noises "1" --bg-noise-dir "$data/musan_music" $data/swbd_sre $data/swbd_sre_music
  # Augment with musan_speech
  python steps/data/augment_data_dir_new.py --utt-suffix "babble" --bg-snrs "20:17:15:13" --num-bg-noises "3:4:5:6:7" --bg-noise-dir "$data/musan_speech" $data/swbd_sre $data/swbd_sre_babble

  # Combine reverb, noise, music, and babble into one directory.
  utils/combine_data.sh $data/swbd_sre_aug $data/swbd_sre_reverb $data/swbd_sre_noise $data/swbd_sre_music $data/swbd_sre_babble

  utils/subset_data_dir.sh $data/swbd_sre_aug 128000 $data/swbd_sre_aug_128k
  utils/filter_scp.pl $data/swbd_sre_aug_128k/utt2spk $data/swbd_sre_aug/utt2uniq > $data/swbd_sre_aug_128k/utt2uniq
  utils/fix_data_dir.sh $data/swbd_sre_aug_128k

  # Make filterbanks for the augmented data.  Note that we do not compute a new
  # vad.scp file here.  Instead, we use the vad.scp from the clean version of
  # the list.
  steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj $fea_nj --cmd "$train_cmd" \
    $data/swbd_sre_aug_128k $exp/make_mfcc $mfccdir

  # Combine the clean and augmented SWBD+SRE list.  This is now roughly
  # double the size of the original clean list.
  utils/combine_data.sh $data/swbd_sre_combined $data/swbd_sre_aug_128k $data/swbd_sre
  utils/fix_data_dir.sh $data/swbd_sre_combined

  # Filter out the clean + augmented portion of the SRE list.  This will be used to
  # train the PLDA model later in the script.
  # Just use SRE data for PLDA training
  utils/copy_data_dir.sh $data/swbd_sre_combined $data/sre_combined
  utils/filter_scp.pl $data/sre/spk2utt $data/swbd_sre_combined/spk2utt | utils/spk2utt_to_utt2spk.pl > $data/sre_combined/utt2spk
  utils/fix_data_dir.sh $data/sre_combined
fi

if [ $stage -le 3 ]; then
  local/nnet3/xvector/prepare_feats_for_egs.sh --nj $fea_nj --cmd "$train_cmd" \
    $data/swbd_sre_combined $data/swbd_sre_combined_nosil $exp/swbd_sre_combined_nosil
  utils/fix_data_dir.sh $data/swbd_sre_combined_nosil
fi

if [ $stage -le 4 ]; then
  # Now, we need to remove features that are too short after removing silence
  # frames.  We want at least 5s (500 frames) per utterance.
  min_len=500
  mv $data/swbd_sre_combined_nosil/utt2num_frames $data/swbd_sre_combined_nosil/utt2num_frames.bak
  awk -v min_len=${min_len} '$2 > min_len {print $1, $2}' $data/swbd_sre_combined_nosil/utt2num_frames.bak > $data/swbd_sre_combined_nosil/utt2num_frames
  utils/filter_scp.pl $data/swbd_sre_combined_nosil/utt2num_frames $data/swbd_sre_combined_nosil/utt2spk > $data/swbd_sre_combined_nosil/utt2spk.new
  mv $data/swbd_sre_combined_nosil/utt2spk.new $data/swbd_sre_combined_nosil/utt2spk
  utils/fix_data_dir.sh $data/swbd_sre_combined_nosil

  # We also want several utterances per speaker. Now we'll throw out speakers
  # with fewer than 8 utterances.
  min_num_utts=8
  awk '{print $1, NF-1}' $data/swbd_sre_combined_nosil/spk2utt > $data/swbd_sre_combined_nosil/spk2num
  awk -v min_num_utts=${min_num_utts} '$2 >= min_num_utts {print $1, $2}' $data/swbd_sre_combined_nosil/spk2num | utils/filter_scp.pl - $data/swbd_sre_combined_nosil/spk2utt > $data/swbd_sre_combined_nosil/spk2utt.new
  mv $data/swbd_sre_combined_nosil/spk2utt.new $data/swbd_sre_combined_nosil/spk2utt
  utils/spk2utt_to_utt2spk.pl $data/swbd_sre_combined_nosil/spk2utt > $data/swbd_sre_combined_nosil/utt2spk

  utils/filter_scp.pl $data/swbd_sre_combined_nosil/utt2spk $data/swbd_sre_combined_nosil/utt2num_frames > $data/swbd_sre_combined_nosil/utt2num_frames.new
  mv $data/swbd_sre_combined_nosil/utt2num_frames.new $data/swbd_sre_combined_nosil/utt2num_frames

  # Now we're ready to create training examples.
  utils/fix_data_dir.sh $data/swbd_sre_combined_nosil
fi

if [ $stage -le 5 ]; then
  # Split the validation set
  mkdir -p $data/swbd_sre_combined_nosil/valid $data/swbd_sre_combined_nosil/train

  num_heldout_spk=128
  num_heldout_utts_per_spk=5

  # The augmented data is similar with the not-augmented one. If an augmented version is in the valid set, it should not appear in the training data.
  # We first remove the augmented data and only sample from the original version
  sed 's/-noise//' $data/swbd_sre_combined_nosil/utt2spk | sed 's/-music//' | sed 's/-babble//' | sed 's/-reverb//' |\
    paste -d ' ' $data/swbd_sre_combined_nosil/utt2spk - | cut -d ' ' -f 1,3 > $data/swbd_sre_combined_nosil/utt2uniq
  utils/utt2spk_to_spk2utt.pl $data/swbd_sre_combined_nosil/utt2uniq > $data/swbd_sre_combined_nosil/uniq2utt
  cat $data/swbd_sre_combined_nosil/utt2spk | utils/apply_map.pl -f 1 $data/swbd_sre_combined_nosil/utt2uniq |\
    sort | uniq > $data/swbd_sre_combined_nosil/utt2spk.uniq
  utils/utt2spk_to_spk2utt.pl $data/swbd_sre_combined_nosil/utt2spk.uniq > $data/swbd_sre_combined_nosil/spk2utt.uniq
  python utils/sample_validset_spk2utt.py $num_heldout_spk $num_heldout_utts_per_spk $data/swbd_sre_combined_nosil/spk2utt.uniq > $data/swbd_sre_combined_nosil/valid/spk2utt.uniq

  # Then we find all the data that is augmented from the original version.
  cat $data/swbd_sre_combined_nosil/valid/spk2utt.uniq | utils/apply_map.pl -f 2- $data/swbd_sre_combined_nosil/uniq2utt > $data/swbd_sre_combined_nosil/valid/spk2utt
  utils/spk2utt_to_utt2spk.pl $data/swbd_sre_combined_nosil/valid/spk2utt > $data/swbd_sre_combined_nosil/valid/utt2spk
  cp $data/swbd_sre_combined_nosil/feats.scp $data/swbd_sre_combined_nosil/valid
  utils/filter_scp.pl $data/swbd_sre_combined_nosil/valid/utt2spk $data/swbd_sre_combined_nosil/utt2num_frames > $data/swbd_sre_combined_nosil/valid/utt2num_frames
  utils/fix_data_dir.sh $data/swbd_sre_combined_nosil/valid

  utils/filter_scp.pl --exclude $data/swbd_sre_combined_nosil/valid/utt2spk $data/swbd_sre_combined_nosil/utt2spk > $data/swbd_sre_combined_nosil/train/utt2spk
  utils/utt2spk_to_spk2utt.pl $data/swbd_sre_combined_nosil/train/utt2spk > $data/swbd_sre_combined_nosil/train/spk2utt
  cp $data/swbd_sre_combined_nosil/feats.scp $data/swbd_sre_combined_nosil/train
  utils/filter_scp.pl $data/swbd_sre_combined_nosil/train/utt2spk $data/swbd_sre_combined_nosil/utt2num_frames > $data/swbd_sre_combined_nosil/train/utt2num_frames
  utils/fix_data_dir.sh $data/swbd_sre_combined_nosil/train

  # In the training, we need an additional file `spklist` to map the speakers to the indices.
  awk -v id=0 '{print $1, id++}' $data/swbd_sre_combined_nosil/train/spk2utt > $data/swbd_sre_combined_nosil/train/spklist
fi


if [ $stage -le 6 ]; then
#  # Training a softmax network
#  nnetdir=$exp/xvector_nnet_tdnn_softmax_1e-2
#  nnet/run_train_nnet.sh --cmd "$cuda_cmd" --env tf_gpu --continue-training false nnet_conf/tdnn_softmax_1e-2.json \
#    $data/swbd_sre_combined_nosil/train $data/swbd_sre_combined_nosil/train/spklist \
#    $data/swbd_sre_combined_nosil/valid $data/swbd_sre_combined_nosil/train/spklist \
#    $nnetdir
#
#
#  # Train asoftmax network
#  nnetdir=$exp/xvector_nnet_tdnn_asoftmax_m1_linear_bn
#  nnet/run_train_nnet.sh --cmd "$cuda_cmd" --env tf_gpu --continue-training false nnet_conf/tdnn_asoftmax_m1_linear_bn.json \
#    $data/swbd_sre_combined_nosil/train $data/swbd_sre_combined_nosil/train/spklist \
#    $data/swbd_sre_combined_nosil/valid $data/swbd_sre_combined_nosil/train/spklist \
#    $nnetdir
#
#  nnetdir=$exp/xvector_nnet_tdnn_asoftmax_m2_linear_bn_1e-2
#  nnet/run_train_nnet.sh --cmd "$cuda_cmd" --env tf_gpu --continue-training false nnet_conf/tdnn_asoftmax_m2_linear_bn_1e-2.json \
#    $data/swbd_sre_combined_nosil/train $data/swbd_sre_combined_nosil/train/spklist \
#    $data/swbd_sre_combined_nosil/valid $data/swbd_sre_combined_nosil/train/spklist \
#    $nnetdir
#
#  nnetdir=$exp/xvector_nnet_tdnn_asoftmax_m4_linear_bn_1e-2
#  nnet/run_train_nnet.sh --cmd "$cuda_cmd" --env tf_gpu --continue-training false nnet_conf/tdnn_asoftmax_m4_linear_bn_1e-2.json \
#    $data/swbd_sre_combined_nosil/train $data/swbd_sre_combined_nosil/train/spklist \
#    $data/swbd_sre_combined_nosil/valid $data/swbd_sre_combined_nosil/train/spklist \
#    $nnetdir
#
#
#  # AMSoftmax
#  nnetdir=$exp/xvector_nnet_tdnn_amsoftmax_m0.10_linear_bn_1e-2
#  nnet/run_train_nnet.sh --cmd "$cuda_cmd" --env tf_gpu --continue-training false nnet_conf/tdnn_amsoftmax_m0.10_linear_bn_1e-2.json \
#    $data/swbd_sre_combined_nosil/train $data/swbd_sre_combined_nosil/train/spklist \
#    $data/swbd_sre_combined_nosil/valid $data/swbd_sre_combined_nosil/train/spklist \
#    $nnetdir
#
#  nnetdir=$exp/xvector_nnet_tdnn_amsoftmax_m0.15_linear_bn_1e-2
#  nnet/run_train_nnet.sh --cmd "$cuda_cmd" --env tf_gpu --continue-training false nnet_conf/tdnn_amsoftmax_m0.15_linear_bn_1e-2.json \
#    $data/swbd_sre_combined_nosil/train $data/swbd_sre_combined_nosil/train/spklist \
#    $data/swbd_sre_combined_nosil/valid $data/swbd_sre_combined_nosil/train/spklist \
#    $nnetdir
#
#  nnetdir=$exp/xvector_nnet_tdnn_amsoftmax_m0.20_linear_bn_1e-2
#  nnet/run_train_nnet.sh --cmd "$cuda_cmd" --env tf_gpu --continue-training false nnet_conf/tdnn_amsoftmax_m0.20_linear_bn_1e-2.json \
#    $data/swbd_sre_combined_nosil/train $data/swbd_sre_combined_nosil/train/spklist \
#    $data/swbd_sre_combined_nosil/valid $data/swbd_sre_combined_nosil/train/spklist \
#    $nnetdir
#
#  nnetdir=$exp/xvector_nnet_tdnn_amsoftmax_m0.25_linear_bn_1e-2
#  nnet/run_train_nnet.sh --cmd "$cuda_cmd" --env tf_gpu --continue-training false nnet_conf/tdnn_amsoftmax_m0.25_linear_bn_1e-2.json \
#    $data/swbd_sre_combined_nosil/train $data/swbd_sre_combined_nosil/train/spklist \
#    $data/swbd_sre_combined_nosil/valid $data/swbd_sre_combined_nosil/train/spklist \
#    $nnetdir
#
#  nnetdir=$exp/xvector_nnet_tdnn_amsoftmax_m0.30_linear_bn_1e-2
#  nnet/run_train_nnet.sh --cmd "$cuda_cmd" --env tf_gpu --continue-training false nnet_conf/tdnn_amsoftmax_m0.30_linear_bn_1e-2.json \
#    $data/swbd_sre_combined_nosil/train $data/swbd_sre_combined_nosil/train/spklist \
#    $data/swbd_sre_combined_nosil/valid $data/swbd_sre_combined_nosil/train/spklist \
#    $nnetdir
#
#  nnetdir=$exp/xvector_nnet_tdnn_amsoftmax_m0.35_linear_bn_1e-2
#  nnet/run_train_nnet.sh --cmd "$cuda_cmd" --env tf_gpu --continue-training false nnet_conf/tdnn_amsoftmax_m0.35_linear_bn_1e-2.json \
#    $data/swbd_sre_combined_nosil/train $data/swbd_sre_combined_nosil/train/spklist \
#    $data/swbd_sre_combined_nosil/valid $data/swbd_sre_combined_nosil/train/spklist \
#    $nnetdir


#  # ArcSoftmax
#  nnetdir=$exp/xvector_nnet_tdnn_arcsoftmax_m0.10_linear_bn_1e-2
#  nnet/run_train_nnet.sh --cmd "$cuda_cmd" --env tf_gpu --continue-training false nnet_conf/tdnn_arcsoftmax_m0.10_linear_bn_1e-2.json \
#    $data/swbd_sre_combined_nosil/train $data/swbd_sre_combined_nosil/train/spklist \
#    $data/swbd_sre_combined_nosil/valid $data/swbd_sre_combined_nosil/train/spklist \
#    $nnetdir

#  nnetdir=$exp/xvector_nnet_tdnn_arcsoftmax_m0.15_linear_bn_1e-2
#  nnet/run_train_nnet.sh --cmd "$cuda_cmd" --env tf_gpu --continue-training false nnet_conf/tdnn_arcsoftmax_m0.15_linear_bn_1e-2.json \
#    $data/swbd_sre_combined_nosil/train $data/swbd_sre_combined_nosil/train/spklist \
#    $data/swbd_sre_combined_nosil/valid $data/swbd_sre_combined_nosil/train/spklist \
#    $nnetdir

#  nnetdir=$exp/xvector_nnet_tdnn_arcsoftmax_m0.20_linear_bn_1e-2
#  nnet/run_train_nnet.sh --cmd "$cuda_cmd" --env tf_gpu --continue-training false nnet_conf/tdnn_arcsoftmax_m0.20_linear_bn_1e-2.json \
#    $data/swbd_sre_combined_nosil/train $data/swbd_sre_combined_nosil/train/spklist \
#    $data/swbd_sre_combined_nosil/valid $data/swbd_sre_combined_nosil/train/spklist \
#    $nnetdir
#
#  nnetdir=$exp/xvector_nnet_tdnn_arcsoftmax_m0.25_linear_bn_1e-2
#  nnet/run_train_nnet.sh --cmd "$cuda_cmd" --env tf_gpu --continue-training false nnet_conf/tdnn_arcsoftmax_m0.25_linear_bn_1e-2.json \
#    $data/swbd_sre_combined_nosil/train $data/swbd_sre_combined_nosil/train/spklist \
#    $data/swbd_sre_combined_nosil/valid $data/swbd_sre_combined_nosil/train/spklist \
#    $nnetdir

#  nnetdir=$exp/xvector_nnet_tdnn_arcsoftmax_m0.30_linear_bn_1e-2
#  nnet/run_train_nnet.sh --cmd "$cuda_cmd" --env tf_gpu --continue-training false nnet_conf/tdnn_arcsoftmax_m0.30_linear_bn_1e-2.json \
#    $data/swbd_sre_combined_nosil/train $data/swbd_sre_combined_nosil/train/spklist \
#    $data/swbd_sre_combined_nosil/valid $data/swbd_sre_combined_nosil/train/spklist \
#    $nnetdir

  nnetdir=$exp/xvector_nnet_tdnn_arcsoftmax_m0.35_linear_bn_1e-2
  nnet/run_train_nnet.sh --cmd "$cuda_cmd" --env tf_gpu --continue-training false nnet_conf/tdnn_arcsoftmax_m0.35_linear_bn_1e-2.json \
    $data/swbd_sre_combined_nosil/train $data/swbd_sre_combined_nosil/train/spklist \
    $data/swbd_sre_combined_nosil/valid $data/swbd_sre_combined_nosil/train/spklist \
    $nnetdir

exit 1
fi


nnetdir=$exp/xvector_nnet_tdnn_softmax_1e-2
checkpoint='last'

if [ $stage -le 7 ];then
  nnet/run_extract_embeddings.sh --cmd "$train_cmd" --nj $nnet_nj --use-gpu false --checkpoint $checkpoint --stage 0 \
    --chunk-size 10000 --normalize false --node "tdnn6_dense" \
    $nnetdir $data/sre_combined $nnetdir/xvectors_sre_combined

  nnet/run_extract_embeddings.sh --cmd "$train_cmd" --nj $nnet_nj --use-gpu false --checkpoint $checkpoint --stage 0 \
    --chunk-size 10000 --normalize false --node "tdnn6_dense" \
    $nnetdir $data/sre10_enroll_coreext_c5_$gender $nnetdir/xvectors_sre10_enroll_coreext_c5_$gender

  nnet/run_extract_embeddings.sh --cmd "$train_cmd" --nj $nnet_nj --use-gpu false --checkpoint $checkpoint --stage 0 \
    --chunk-size 10000 --normalize false --node "tdnn6_dense" \
    $nnetdir $data/sre10_test_coreext_c5_$gender $nnetdir/xvectors_sre10_test_coreext_c5_$gender

  nnet/run_extract_embeddings.sh --cmd "$train_cmd" --nj $nnet_nj --use-gpu false --checkpoint $checkpoint --stage 0 \
    --chunk-size 10000 --normalize false --node "tdnn6_dense" \
    $nnetdir $data/sre10_enroll_10s_$gender $nnetdir/xvectors_sre10_enroll_10s_$gender

  nnet/run_extract_embeddings.sh --cmd "$train_cmd" --nj $nnet_nj --use-gpu false --checkpoint $checkpoint --stage 0 \
    --chunk-size 10000 --normalize false --node "tdnn6_dense" \
    $nnetdir $data/sre10_test_10s_$gender $nnetdir/xvectors_sre10_test_10s_$gender
fi

# PLDA scoring on NIST SRE 2010
if [ $stage -le 8 ]; then
  ### !!! Re-train LDA and PLDA models when it is referred again
  lda_dim=150

  $train_cmd $nnetdir/xvectors_sre_combined/log/compute_mean.log \
    ivector-mean scp:$nnetdir/xvectors_sre_combined/xvector.scp \
    $nnetdir/xvectors_sre_combined/mean.vec || exit 1;

  # This script uses LDA to decrease the dimensionality prior to PLDA.
  $train_cmd $nnetdir/xvectors_sre_combined/log/lda.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
    "ark:ivector-subtract-global-mean scp:$nnetdir/xvectors_sre_combined/xvector.scp ark:- |" \
    ark:$data/sre_combined/utt2spk $nnetdir/xvectors_sre_combined/transform.mat || exit 1;

  $train_cmd $nnetdir/xvectors_sre_combined/log/plda_lda${lda_dim}.log \
    ivector-compute-plda ark:$data/sre_combined/spk2utt \
    "ark:ivector-subtract-global-mean scp:$nnetdir/xvectors_sre_combined/xvector.scp ark:- | transform-vec $nnetdir/xvectors_sre_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
    $nnetdir/xvectors_sre_combined/plda_lda${lda_dim} || exit 1;

  # Coreext C5
  $train_cmd $nnetdir/xvector_scores/log/sre10_coreext_c5_$gender.log \
    ivector-plda-scoring --normalize-length=true \
    --num-utts=ark:$nnetdir/xvectors_sre10_enroll_coreext_c5_$gender/num_utts.ark \
    "ivector-copy-plda --smoothing=0.0 $nnetdir/xvectors_sre_combined/plda_lda${lda_dim} - |" \
    "ark:ivector-mean ark:$data/sre10_enroll_coreext_c5_$gender/spk2utt scp:$nnetdir/xvectors_sre10_enroll_coreext_c5_$gender/xvector.scp ark:- | ivector-subtract-global-mean $nnetdir/xvectors_sre_combined/mean.vec ark:- ark:- | transform-vec $nnetdir/xvectors_sre_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean $nnetdir/xvectors_sre_combined/mean.vec scp:$nnetdir/xvectors_sre10_test_coreext_c5_$gender/xvector.scp ark:- | transform-vec $nnetdir/xvectors_sre_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$data/sre10_test_coreext_c5_$gender/trials' | cut -d\  --fields=1,2 |" $nnetdir/xvector_scores/sre10_coreext_c5_scores_$gender || exit 1;

  # 10s-10s
  $train_cmd $nnetdir/xvector_scores/log/sre10_10s_$gender.log \
    ivector-plda-scoring --normalize-length=true \
    --num-utts=ark:$nnetdir/xvectors_sre10_enroll_10s_$gender/num_utts.ark \
    "ivector-copy-plda --smoothing=0.0 $nnetdir/xvectors_sre_combined/plda_lda${lda_dim} - |" \
    "ark:ivector-mean ark:$data/sre10_enroll_10s_$gender/spk2utt scp:$nnetdir/xvectors_sre10_enroll_10s_$gender/xvector.scp ark:- | ivector-subtract-global-mean $nnetdir/xvectors_sre_combined/mean.vec ark:- ark:- | transform-vec $nnetdir/xvectors_sre_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean $nnetdir/xvectors_sre_combined/mean.vec scp:$nnetdir/xvectors_sre10_test_10s_$gender/xvector.scp ark:- | transform-vec $nnetdir/xvectors_sre_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$data/sre10_test_10s_$gender/trials' | cut -d\  --fields=1,2 |" $nnetdir/xvector_scores/sre10_10s_scores_$gender || exit 1;

  eval_plda_sre10.sh $gender $data $nnetdir
fi

if [ $stage -le 9 ]; then
  nnet/run_extract_embeddings.sh --cmd "$train_cmd" --nj $nnet_nj --use-gpu false --checkpoint $checkpoint --stage 0 \
    --chunk-size 10000 --normalize false --node "tdnn6_dense" \
    $nnetdir $data/sre16_major $nnetdir/xvectors_sre16_major

  nnet/run_extract_embeddings.sh --cmd "$train_cmd" --nj $nnet_nj --use-gpu false --checkpoint $checkpoint --stage 0 \
    --chunk-size 10000 --normalize false --node "tdnn6_dense" \
    $nnetdir $data/sre16_eval_enroll $nnetdir/xvectors_sre16_eval_enroll

  nnet/run_extract_embeddings.sh --cmd "$train_cmd" --nj $nnet_nj --use-gpu false --checkpoint $checkpoint --stage 0 \
    --chunk-size 10000 --normalize false --node "tdnn6_dense" \
    $nnetdir $data/sre16_eval_test $nnetdir/xvectors_sre16_eval_test
fi

# PLDA scoring on SRE 2016
if [ $stage -le 10 ]; then
  ### !!! Re-train LDA and PLDA models when it is referred again
  lda_dim=150

  $train_cmd $nnetdir/xvectors_sre16_major/log/compute_mean.log \
    ivector-mean scp:$nnetdir/xvectors_sre16_major/xvector.scp \
    $nnetdir/xvectors_sre16_major/mean.vec || exit 1;

  # This script uses LDA to decrease the dimensionality prior to PLDA.
  $train_cmd $nnetdir/xvectors_sre_combined/log/lda.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
    "ark:ivector-subtract-global-mean scp:$nnetdir/xvectors_sre_combined/xvector.scp ark:- |" \
    ark:$data/sre_combined/utt2spk $nnetdir/xvectors_sre_combined/transform.mat || exit 1;

  # Train an out-of-domain PLDA model.
  $train_cmd $nnetdir/xvectors_sre_combined/log/plda_lda${lda_dim}.log \
    ivector-compute-plda ark:$data/sre_combined/spk2utt \
    "ark:ivector-subtract-global-mean scp:$nnetdir/xvectors_sre_combined/xvector.scp ark:- | transform-vec $nnetdir/xvectors_sre_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
    $nnetdir/xvectors_sre_combined/plda_lda${lda_dim} || exit 1;

  # Here we adapt the out-of-domain PLDA model to SRE16 major, a pile
  # of unlabeled in-domain data.  In the future, we will include a clustering
  # based approach for domain adaptation, which tends to work better.
  $train_cmd $nnetdir/xvectors_sre16_major/log/plda_lda${lda_dim}_sre16_adapt.log \
    ivector-adapt-plda --within-covar-scale=0.75 --between-covar-scale=0.25 \
    $nnetdir/xvectors_sre_combined/plda_lda${lda_dim} \
    "ark:ivector-subtract-global-mean scp:$nnetdir/xvectors_sre16_major/xvector.scp ark:- | transform-vec $nnetdir/xvectors_sre_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    $nnetdir/xvectors_sre16_major/plda_lda${lda_dim}_sre16_adapt || exit 1;

  # Get results using the out-of-domain PLDA model.
  $train_cmd $nnetdir/xvector_scores/log/sre16_eval.log \
    ivector-plda-scoring --normalize-length=true \
    --num-utts=ark:$nnetdir/xvectors_sre16_eval_enroll/num_utts.ark \
    "ivector-copy-plda --smoothing=0.0 $nnetdir/xvectors_sre_combined/plda_lda${lda_dim} - |" \
    "ark:ivector-mean ark:$data/sre16_eval_enroll/spk2utt scp:$nnetdir/xvectors_sre16_eval_enroll/xvector.scp ark:- | ivector-subtract-global-mean $nnetdir/xvectors_sre16_major/mean.vec ark:- ark:- | transform-vec $nnetdir/xvectors_sre_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean $nnetdir/xvectors_sre16_major/mean.vec scp:$nnetdir/xvectors_sre16_eval_test/xvector.scp ark:- | transform-vec $nnetdir/xvectors_sre_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$sre16_trials' | cut -d\  --fields=1,2 |" $nnetdir/xvector_scores/sre16_eval_scores || exit 1;

  # Get results using the adapted PLDA model.
  $train_cmd $nnetdir/xvector_scores/log/sre16_eval_scoring_adapt.log \
    ivector-plda-scoring --normalize-length=true \
    --num-utts=ark:$nnetdir/xvectors_sre16_eval_enroll/num_utts.ark \
    "ivector-copy-plda --smoothing=0.0 $nnetdir/xvectors_sre16_major/plda_lda${lda_dim}_sre16_adapt - |" \
    "ark:ivector-mean ark:$data/sre16_eval_enroll/spk2utt scp:$nnetdir/xvectors_sre16_eval_enroll/xvector.scp ark:- | ivector-subtract-global-mean $nnetdir/xvectors_sre16_major/mean.vec ark:- ark:- | transform-vec $nnetdir/xvectors_sre_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean $nnetdir/xvectors_sre16_major/mean.vec scp:$nnetdir/xvectors_sre16_eval_test/xvector.scp ark:- | transform-vec $nnetdir/xvectors_sre_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$sre16_trials' | cut -d\  --fields=1,2 |" $nnetdir/xvector_scores/sre16_eval_scores_adapt || exit 1;

  eval_plda_sre16.sh $gender $data $nnetdir
  exit 1
fi

nnetdir=$exp/xvector_nnet_tdnn_softmax_1e-2
checkpoint='last'

if [ $stage -le 11 ];then
  nnet/run_extract_embeddings.sh --cmd "$train_cmd" --nj $nnet_nj --use-gpu false --checkpoint $checkpoint --stage 0 \
    --chunk-size 10000 --normalize false --node "output" \
    $nnetdir $data/sre10_enroll_coreext_c5_$gender $nnetdir/xvectors_sre10_enroll_coreext_c5_$gender

  nnet/run_extract_embeddings.sh --cmd "$train_cmd" --nj $nnet_nj --use-gpu false --checkpoint $checkpoint --stage 0 \
    --chunk-size 10000 --normalize false --node "output" \
    $nnetdir $data/sre10_test_coreext_c5_$gender $nnetdir/xvectors_sre10_test_coreext_c5_$gender

  nnet/run_extract_embeddings.sh --cmd "$train_cmd" --nj $nnet_nj --use-gpu false --checkpoint $checkpoint --stage 0 \
    --chunk-size 10000 --normalize false --node "output" \
    $nnetdir $data/sre10_enroll_10s_$gender $nnetdir/xvectors_sre10_enroll_10s_$gender

  nnet/run_extract_embeddings.sh --cmd "$train_cmd" --nj $nnet_nj --use-gpu false --checkpoint $checkpoint --stage 0 \
    --chunk-size 10000 --normalize false --node "output" \
    $nnetdir $data/sre10_test_10s_$gender $nnetdir/xvectors_sre10_test_10s_$gender

  nnet/run_extract_embeddings.sh --cmd "$train_cmd" --nj $nnet_nj --use-gpu false --checkpoint $checkpoint --stage 0 \
    --chunk-size 10000 --normalize false --node "output" \
    $nnetdir $data/sre16_eval_enroll $nnetdir/xvectors_sre16_eval_enroll

  nnet/run_extract_embeddings.sh --cmd "$train_cmd" --nj $nnet_nj --use-gpu false --checkpoint $checkpoint --stage 0 \
    --chunk-size 10000 --normalize false --node "output" \
    $nnetdir $data/sre16_eval_test $nnetdir/xvectors_sre16_eval_test
fi

if [ $stage -le 12 ]; then
  mkdir -p $nnetdir/xvector_scores

  # Coreext C5
  cat $data/sre10_test_coreext_c5_$gender/trials | awk '{print $1, $2}' | \
    ivector-compute-dot-products - \
      "ark:ivector-mean ark:$data/sre10_enroll_coreext_c5_$gender/spk2utt scp:$nnetdir/xvectors_sre10_enroll_coreext_c5_$gender/xvector.scp ark:- | ivector-normalize-length ark:- ark:- |" \
      "ark:ivector-normalize-length scp:$nnetdir/xvectors_sre10_test_coreext_c5_$gender/xvector.scp ark:- |" \
      $nnetdir/xvector_scores/sre10_coreext_c5_scores_$gender.cos

  # 10s-10s
  cat $data/sre10_test_10s_$gender/trials | awk '{print $1, $2}' | \
    ivector-compute-dot-products - \
      "ark:ivector-mean ark:$data/sre10_enroll_10s_$gender/spk2utt scp:$nnetdir/xvectors_sre10_enroll_10s_$gender/xvector.scp ark:- | ivector-normalize-length ark:- ark:- |" \
      "ark:ivector-normalize-length scp:$nnetdir/xvectors_sre10_test_10s_$gender/xvector.scp ark:- |" \
      $nnetdir/xvector_scores/sre10_10s_scores_$gender.cos

  # SRE2016
  cat $sre16_trials | awk '{print $1, $2}' | \
    ivector-compute-dot-products - \
      "ark:ivector-mean ark:$data/sre16_eval_enroll/spk2utt scp:$nnetdir/xvectors_sre16_eval_enroll/xvector.scp ark:- | ivector-normalize-length ark:- ark:- |" \
      "ark:ivector-normalize-length scp:$nnetdir/xvectors_sre16_eval_test/xvector.scp ark:- |" \
      $nnetdir/xvector_scores/sre16_eval_scores.cos

  eval_cos.sh $gender $data $nnetdir
  exit 1
fi

