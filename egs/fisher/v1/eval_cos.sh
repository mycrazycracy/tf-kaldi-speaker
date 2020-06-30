#!/bin/bash

nnetdir=$1

eer=$(paste $trials $nnetdir/xvector_scores_hires/test_cos | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
echo "EER: ${eer}%"

paste $trials $nnetdir/xvector_scores_hires/test_cos | awk '{print $6, $3}' > $nnetdir/xvector_scores_hires/test_cos.new
grep ' target$' $nnetdir/xvector_scores_hires/test_cos.new | cut -d ' ' -f 1 > $nnetdir/xvector_scores_hires/test_cos.target
grep ' nontarget$' $nnetdir/xvector_scores_hires/test_cos.new | cut -d ' ' -f 1 > $nnetdir/xvector_scores_hires/test_cos.nontarget
comm=`echo "addpath('../../../misc/DETware_v2.1'); Get_DCF('$nnetdir/xvector_scores_hires/test_cos.target', '$nnetdir/xvector_scores_hires/test_cos.nontarget', '$nnetdir/xvector_scores_hires/test_cos.result')"`
echo "$comm"| matlab -nodesktop -noFigureWindows > /dev/null
tail -n 1 $nnetdir/xvector_scores_hires/test_cos.result