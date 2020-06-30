# The virtualenv path
export TF_ENV=/home/heliang05/liuyi/venv

export TF_KALDI_ROOT=/home/heliang05/liuyi/base/tf-kaldi-speaker
export KALDI_ROOT=/home/heliang05/liuyi/software/kaldi_gpu
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/sph2pipe_v2.5:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C