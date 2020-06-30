#!/bin/bash

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 1 ]; then
  echo "Usage: $0 <ali-dir>"
  echo "e.g.: $0 data/train exp/tri5a_ali"
  exit 1;
fi

dir=$1

for f in $dir/ali.1.gz $dir/final.mdl ; do
  [ ! -f $f ] && echo "$0: No such file $f" && exit 1;
done

num_ali_jobs=$(cat $dir/num_jobs) || exit 1;
for id in $(seq $num_ali_jobs); do gunzip -c $dir/ali.$id.gz; done | \
  ali-to-pdf $dir/final.mdl ark:- ark,scp:$dir/pdf.ark,$dir/pdf.scp || exit 1;

# TODO: pdf to phones? pdf to phone classes? pdf to ali? We may need to get other types of alignments.

exit 0