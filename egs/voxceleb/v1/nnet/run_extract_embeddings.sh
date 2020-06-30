#!/bin/bash

nj=32
use_gpu=false
cmd="run.pl"
min_chunk_size=25
chunk_size=10000
stage=0
normalize=false
checkpoint=-1
env=tf_cpu
node="output"

echo "$0 $@"

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
  echo "Usage: $0 [options] <nnet-dir> <data> <embeddings-dir>"
  echo "Options:"
  echo "  --use-gpu <false>"
  echo "  --nj <32>"
  echo "  --min-chunk-size <25>"
  echo "  --chunk-size <10000>"
  echo "  --normalize <false>"
  echo "  --checkpoint <-1>"
  echo "  --node <output>"
  echo ""
  exit 100
fi

nnetdir=$1
data=$2
dir=$3

for f in $nnetdir/nnet/checkpoint $data/feats.scp $data/vad.scp; do
  [ ! -f $f ] && echo "No such file $f" && exit 1;
done

mkdir -p $dir/log

utils/split_data.sh $data $nj
echo "$0: extracting embeddings for $data"
sdata=$data/split$nj/JOB

feat="ark:apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 scp:${sdata}/feats.scp ark:- | select-voiced-frames ark:- scp,s,cs:${sdata}/vad.scp ark:- |"

# I use conda to load TF (in cpu case), so some preparations are applied before python. So a wrapper make things more flexible.
# If no conda is used, simply set "--use-env false"
if [ $stage -le 0 ]; then
  echo "$0: extracting xvectors from nnet"
  echo "$0: embedding from node $node"

  # Set the checkpoint.
  source $TF_ENV/$env/bin/activate
  export PYTHONPATH=$TF_KALDI_ROOT:$PYTHONPATH
  python nnet/lib/make_checkpoint.py --checkpoint $checkpoint "$nnetdir"
  deactivate

  if $use_gpu; then
    echo "Using CPU to do inference is a better choice."
    exit 1
#    $cmd JOB=1:$nj ${dir}/log/extract.JOB.log \
#      nnet/wrap/extract_wrapper.sh --gpuid JOB --env $env --min-chunk-size $min_chunk_size --chunk-size $chunk_size --normalize $normalize \
#        "$nnetdir" "$feat" "ark:| copy-vector ark:- ark,scp:${dir}/xvector.JOB.ark,${dir}/xvector.JOB.scp"
  else
    $cmd JOB=1:$nj ${dir}/log/extract.JOB.log \
      nnet/wrap/extract_wrapper.sh --gpuid -1 --env $env --min-chunk-size $min_chunk_size --chunk-size $chunk_size \
        --normalize $normalize --node $node \
        "$nnetdir" "$feat" "ark:| copy-vector ark:- ark,scp:${dir}/xvector.JOB.ark,${dir}/xvector.JOB.scp"
  fi
fi

if [ $stage -le 1 ]; then
  echo "$0: combining xvectors across jobs"
  for j in $(seq $nj); do cat $dir/xvector.$j.scp; done >$dir/xvector.scp || exit 1;
fi

if [ $stage -le 2 ]; then
  # Average the utterance-level xvectors to get speaker-level xvectors
  echo "$0: computing mean of xvectors for each speaker"
  if $normalize; then
    echo "$0:   Normalize xvectors before computing the mean."
    $cmd $dir/log/speaker_mean.log \
      ivector-normalize-length --scaleup=false scp:$dir/xvector.scp ark:- \| \
      ivector-mean ark:$data/spk2utt ark:- ark:- ark,t:$dir/num_utts.ark \| \
      ivector-normalize-length --scaleup=false ark:- ark,scp:$dir/spk_xvector.ark,$dir/spk_xvector.scp || exit 1
  else
    $cmd $dir/log/speaker_mean.log \
      ivector-mean ark:$data/spk2utt scp:$dir/xvector.scp \
        ark,scp:$dir/spk_xvector.ark,$dir/spk_xvector.scp ark,t:$dir/num_utts.ark || exit 1;
  fi
fi

if [ $stage -le 3 ]; then
  if $normalize; then
    # Normalize the output embeddings
    cp $dir/xvector.scp $dir/xvector_before_norm.scp
    $cmd $dir/log/length_norm.log \
      ivector-normalize-length --scaleup=false scp:$dir/xvector_before_norm.scp ark,scp:$dir/xvector.ark,$dir/xvector.scp
  fi
fi

exit 0