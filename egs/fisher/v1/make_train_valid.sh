#!/bin/bash

data=$1

# Split the validation set
mkdir -p $data/train_background_hires_nosil/valid $data/train_background_hires_nosil/train

cut -d ' ' -f 1,2 $data/train_background_hires_nosil/segments > $data/train_background_hires_nosil/segment2utt
utils/utt2spk_to_spk2utt.pl $data/train_background_hires_nosil/segment2utt > $data/train_background_hires_nosil/utt2segment
awk '{print $1" "(NF-1)}' $data/train_background_hires_nosil/utt2segment > $data/train_background_hires_nosil/utt2num_segments
awk '{print $1" "(NF-1)}' $data/train_background_hires_nosil/spk2utt > $data/train_background_hires_nosil/spk2num_utts
awk '{if($2<10) print $0}' $data/train_background_hires_nosil/utt2num_segments > $data/train_background_hires_nosil/valid/utt2num_segments

utils/filter_scp.pl $data/train_background_hires_nosil/valid/utt2num_segments $data/train_background_hires_nosil/utt2segment > $data/train_background_hires_nosil/valid/utt2segment
utils/spk2utt_to_utt2spk.pl $data/train_background_hires_nosil/valid/utt2segment > $data/train_background_hires_nosil/valid/segment2utt
utils/filter_scp.pl $data/train_background_hires_nosil/valid/segment2utt $data/train_background_hires_nosil/utt2spk > $data/train_background_hires_nosil/valid/utt2spk
utils/utt2spk_to_spk2utt.pl $data/train_background_hires_nosil/valid/utt2spk > $data/train_background_hires_nosil/valid/spk2utt

awk '{print $1" "(NF-1)}' $data/train_background_hires_nosil/valid/spk2utt > $data/train_background_hires_nosil/valid/spk2num_utts
sort -k 2 -n $data/train_background_hires_nosil/valid/spk2num_utts | tail -n 350 > $data/train_background_hires_nosil/valid/spk2num_utts.new
mv $data/train_background_hires_nosil/valid/spk2num_utts.new $data/train_background_hires_nosil/valid/spk2num_utts

python -c "
import sys
spk2num = {}
with open(sys.argv[1], 'r') as f:
  for line in f.readlines():
    spk, num = line.strip().split(' ')
    spk2num[spk] = int(num)
fp_out = open(sys.argv[3], 'w')
with open(sys.argv[2], 'r') as f:
  for line in f.readlines():
    spk, num = line.strip().split(' ')
    if int(num) == spk2num[spk]:
      continue
    else:
      fp_out.write(line)
fp_out.close()
" $data/train_background_hires_nosil/spk2num_utts $data/train_background_hires_nosil/valid/spk2num_utts $data/train_background_hires_nosil/valid/spk2num_utts.new
mv $data/train_background_hires_nosil/valid/spk2num_utts.new $data/train_background_hires_nosil/valid/spk2num_utts

utils/filter_scp.pl $data/train_background_hires_nosil/valid/spk2num_utts $data/train_background_hires_nosil/valid/spk2utt > $data/train_background_hires_nosil/valid/spk2utt.new
mv $data/train_background_hires_nosil/valid/spk2utt.new $data/train_background_hires_nosil/valid/spk2utt
utils/spk2utt_to_utt2spk.pl $data/train_background_hires_nosil/valid/spk2utt > $data/train_background_hires_nosil/valid/utt2spk
cp $data/train_background_hires_nosil/feats.scp $data/train_background_hires_nosil/valid
utils/filter_scp.pl $data/train_background_hires_nosil/valid/utt2spk $data/train_background_hires_nosil/utt2num_frames > $data/train_background_hires_nosil/valid/utt2num_frames
utils/fix_data_dir.sh $data/train_background_hires_nosil/valid
rm $data/train_background_hires_nosil/valid/utt2num_segments $data/train_background_hires_nosil/valid/utt2segment $data/train_background_hires_nosil/valid/segment2utt

utils/filter_scp.pl --exclude $data/train_background_hires_nosil/valid/utt2spk $data/train_background_hires_nosil/utt2spk > $data/train_background_hires_nosil/train/utt2spk
utils/utt2spk_to_spk2utt.pl $data/train_background_hires_nosil/train/utt2spk > $data/train_background_hires_nosil/train/spk2utt
cp $data/train_background_hires_nosil/feats.scp $data/train_background_hires_nosil/train
utils/filter_scp.pl $data/train_background_hires_nosil/train/utt2spk $data/train_background_hires_nosil/utt2num_frames > $data/train_background_hires_nosil/train/utt2num_frames
utils/fix_data_dir.sh $data/train_background_hires_nosil/train
awk '{print $1" "(NF-1)}' $data/train_background_hires_nosil/train/spk2utt > $data/train_background_hires_nosil/train/spk2num_utts

#  num_heldout_spk=256
#  num_heldout_utts_per_spk=16
#
#  # The augmented data is similar with the not-augmented one. If an augmented version is in the valid set, it should not appear in the training data.
#  # We first remove the augmented data and only sample from the original version
#  sed 's/-noise//' $data/train_background_hires_nosil/utt2spk | sed 's/-music//' | sed 's/-babble//' | sed 's/-reverb//' |\
#    paste -d ' ' $data/train_background_hires_nosil/utt2spk - | cut -d ' ' -f 1,3 > $data/train_background_hires_nosil/utt2uniq
#  utils/utt2spk_to_spk2utt.pl $data/train_background_hires_nosil/utt2uniq > $data/train_background_hires_nosil/uniq2utt
#  cat $data/train_background_hires_nosil/utt2spk | utils/apply_map.pl -f 1 $data/train_background_hires_nosil/utt2uniq |\
#    sort | uniq > $data/train_background_hires_nosil/utt2spk.uniq
#  utils/utt2spk_to_spk2utt.pl $data/train_background_hires_nosil/utt2spk.uniq > $data/train_background_hires_nosil/spk2utt.uniq
#  python utils/sample_validset_spk2utt.py $num_heldout_spk $num_heldout_utts_per_spk $data/train_background_hires_nosil/spk2utt.uniq > $data/train_background_hires_nosil/valid/spk2utt.uniq
#
#  # Then we find all the data that is augmented from the original version.
#  cat $data/train_background_hires_nosil/valid/spk2utt.uniq | utils/apply_map.pl -f 2- $data/train_background_hires_nosil/uniq2utt > $data/train_background_hires_nosil/valid/spk2utt
#  utils/spk2utt_to_utt2spk.pl $data/train_background_hires_nosil/valid/spk2utt > $data/train_background_hires_nosil/valid/utt2spk
#  cp $data/train_background_hires_nosil/feats.scp $data/train_background_hires_nosil/valid
#  utils/filter_scp.pl $data/train_background_hires_nosil/valid/utt2spk $data/train_background_hires_nosil/utt2num_frames > $data/train_background_hires_nosil/valid/utt2num_frames
#  utils/fix_data_dir.sh $data/train_background_hires_nosil/valid
#
#  utils/filter_scp.pl --exclude $data/train_background_hires_nosil/valid/utt2spk $data/train_background_hires_nosil/utt2spk > $data/train_background_hires_nosil/train/utt2spk
#  utils/utt2spk_to_spk2utt.pl $data/train_background_hires_nosil/train/utt2spk > $data/train_background_hires_nosil/train/spk2utt
#  cp $data/train_background_hires_nosil/feats.scp $data/train_background_hires_nosil/train
#  utils/filter_scp.pl $data/train_background_hires_nosil/train/utt2spk $data/train_background_hires_nosil/utt2num_frames > $data/train_background_hires_nosil/train/utt2num_frames
#  utils/fix_data_dir.sh $data/train_background_hires_nosil/train

# In the training, we need an additional file `spklist` to map the speakers to the indices.
awk -v id=0 '{print $1, id++}' $data/train_background_hires_nosil/train/spk2utt > $data/train_background_hires_nosil/train/spklist