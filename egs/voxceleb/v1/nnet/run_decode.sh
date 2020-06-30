#!/bin/bash

nj=32
use_gpu=false
cmd="run.pl"
chunk_size=10000
stage=0
checkpoint=-1
env=tf_cpu

# Decoding related
acwt=0.1
beam=15.0
max_active=7000
min_active=200
lattice_beam=8.0
minimize=false
post_decode_acwt=1.0
skip_diagnostics=false
skip_scoring=false
scoring_opts=

echo "$0 $@"

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 5 ]; then
  echo "Usage: $0 [options] <graph-dir> <transition-model> <nnet-dir> <data> <decode-dir>"
  echo "Options:"
  echo "  --use-gpu <false>"
  echo "  --nj <32>"
  echo "  --chunk-size <10000>"
  echo "  --checkpoint <-1>"
  echo ""
  exit 100
fi

graphdir=$1
transdir=$2
nnetdir=$3
data=$4
dir=$5

for f in $graphdir/HCLG.fst $nnetdir/nnet/checkpoint $data/feats.scp; do
  [ ! -f $f ] && echo "No such file $f" && exit 1;
done

mkdir -p $dir/log
echo $nj > $dir/num_jobs
utils/split_data.sh $data $nj
sdata=$data/split$nj/JOB

feat="ark:apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 scp:${sdata}/feats.scp ark:- |"

if $use_gpu; then
  env=tf_cpu
else
  env=tf_gpu
fi

rm -f $dir/lat.*.gz

if [ "$post_decode_acwt" == 1.0 ]; then
  lat_wspecifier="ark:|gzip -c >$dir/lat.JOB.gz"
else
  lat_wspecifier="ark:|lattice-scale --acoustic-scale=$post_decode_acwt ark:- ark:- | gzip -c >$dir/lat.JOB.gz"
fi

if [ $stage -le 0 ]; then
  if $use_gpu; then
    echo "Using CPU to do inference is a better choice."
    exit 1
  else
    source $TF_ENV/$env/bin/activate
    export PYTHONPATH=$TF_KALDI_ROOT:$PYTHONPATH
    python nnet/lib/make_checkpoint.py --checkpoint $checkpoint "$nnetdir"

    export PYTHONPATH=`pwd`/../../:$PYTHONPATH

    # It is pretty cool if we can feed the log-likelihood directly into the lattice decoder.
    $cmd JOB=1:$nj ${dir}/log/decode.JOB.log \
    python nnet/lib/compute_loglike.py --gpu -1 --chunk-size $chunk_size \
      $transdir/prior.vec \
      "$nnetdir" \
      "$feat" \
      "ark:| latgen-faster-mapped --minimize=$minimize --min-active=$min_active --max-active=$max_active --beam=$beam \
               --lattice-beam=$lattice_beam --acoustic-scale=$acwt --allow-partial=true --word-symbol-table=$graphdir/words.txt \
               $transdir/final.trans_mdl $graphdir/HCLG.fst ark:- \"$lat_wspecifier\""

#    python nnet/lib/compute_loglike.py --gpu -1 --chunk-size $chunk_size \
#      $transdir/prior.vec \
#      "$nnetdir" \
#      "ark:apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 scp:$data/split1/1/feats.scp ark:- |" \
#      "ark:| latgen-faster-mapped --minimize=$minimize --min-active=$min_active --max-active=$max_active --beam=$beam \
#               --lattice-beam=$lattice_beam --acoustic-scale=$acwt --allow-partial=true --word-symbol-table=$graphdir/words.txt \
#               $transdir/final.trans_mdl $graphdir/HCLG.fst ark:- \"$lat_wspecifier\""
#     exit 1

    deactivate
  fi
fi

if [ $stage -le 1 ]; then
  if ! $skip_diagnostics ; then
    scripts/diagnostic/analyze_lats.sh --cmd "$cmd" $transdir $graphdir $dir
  fi
fi

if [ $stage -le 2 ]; then
  if ! $skip_scoring ; then
    [ ! -x scripts/diagnostic/score.sh ] && \
      echo "Not scoring because local/score.sh does not exist or not executable." && exit 1;
    echo "score best paths"
    scripts/diagnostic/score.sh $scoring_opts --cmd "$cmd" $data $graphdir $dir
    echo "score confidence and timing with sclite"
  fi
fi

exit 0
