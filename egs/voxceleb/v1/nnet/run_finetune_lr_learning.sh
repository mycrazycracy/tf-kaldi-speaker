#!/bin/bash

cmd="run.pl"
env=tf_gpu
num_gpus=1
checkpoint=-1
tune_period=100

echo "$0 $@"

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 7 ]; then
  echo "Usage: $0 [options] <config> <train-dir> <train-spklist> <valid-dir> <valid-spklist> <pretrained-nnet> <nnet>"
  echo "Options:"
  echo "  --tune-period <100>"
  echo "  --checkpoint <-1>"
  echo "  --env <tf_gpu>"
  echo "  --num-gpus <n_gpus>"
  exit 100
fi

config=$1
train=$2
train_spklist=$3
valid=$4
valid_spklist=$5
pretrain_nnetdir=$6
nnetdir=$7

# add the library to the python path.
export PYTHONPATH=$TF_KALDI_ROOT:$PYTHONPATH

mkdir -p $nnetdir/log

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

source $TF_ENV/$env/bin/activate
$cmd $nnetdir/log/finetune_lr_learning.log utils/parallel/limit_num_gpus.sh --num-gpus $num_gpus \
  python nnet/lib/finetune_lr_learning.py --tune_period $tune_period --checkpoint $checkpoint --config $config $train $train_spklist $valid $valid_spklist $pretrain_nnetdir $nnetdir
deactivate

exit 0