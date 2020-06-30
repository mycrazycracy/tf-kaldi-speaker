#!/bin/bash

cmd="run.pl"
continue_training=false
env=tf_gpu
num_gpus=1

echo "$0 $@"

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 8 ]; then
  echo "Usage: $0 [options] <config> <train-dir> <train-ali-dir> <train-spklist> <valid-dir> <valid-ali-dir> <valid-spklist> <nnet>"
  echo "Options:"
  echo "  --continue-training <false>"
  echo "  --env <tf_gpu>"
  echo "  --num-gpus <n_gpus>"
  exit 100
fi

config=$1
train=$2
train_ali_dir=$3
train_spklist=$4
valid=$5
valid_ali_dir=$6
valid_spklist=$7
nnetdir=$8

# add the library to the python path.
export PYTHONPATH=$TF_KALDI_ROOT:$PYTHONPATH

mkdir -p $nnetdir/log

if [ $continue_training == 'true' ]; then
  cmdopts="-c"
fi

# Get available GPUs before we can train the network.
num_total_gpus=`nvidia-smi -L | wc -l`
num_gpus_assigned=0
while [ $num_gpus_assigned -ne $num_gpus ]; do
  num_gpus_assigned=0
  for i in `seq 0 $[$num_total_gpus-1]`; do
    # going over all GPUs and check if it is idle, and add to the list if yes
    if nvidia-smi -i $i | grep "No running processes found" >/dev/null; then
      num_gpus_assigned=$[$num_gpus_assigned+1]
    fi
    # once we have enough GPUs, break out of the loop
    [ $num_gpus_assigned -eq $num_gpus ] && break
  done
  [ $num_gpus_assigned -eq $num_gpus ] && break
  sleep 300
done

if [ -d $nnetdir/log ] && [ `ls $nnetdir/log | wc -l` -ge 1 ]; then
  mkdir -p $nnetdir/.backup/log
  cp $nnetdir/log/* $nnetdir/.backup/log
fi

# Activate the gpu virtualenv
# The tensorflow is installed using pip (virtualenv). Modify the code if you activate TF by other ways.
# Limit the GPU number to what we want.
source $TF_ENV/$env/bin/activate
$cmd $nnetdir/log/train_nnet.log utils/parallel/limit_num_gpus.sh --num-gpus $num_gpus \
    python nnet/lib/train_mt.py $cmdopts --config $config $train $train_ali_dir $train_spklist $valid $valid_ali_dir $valid_spklist $nnetdir
deactivate

exit 0
