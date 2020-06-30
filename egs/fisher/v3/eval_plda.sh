#!/bin/bash

nnetdir=$1

eer=$(paste $trials $nnetdir/xvector_scores_hires/test | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
echo "EER: ${eer}%"
paste $trials $nnetdir/xvector_scores_hires/test | awk '{print $6, $3}' > $nnetdir/xvector_scores_hires/test.new
grep ' target$' $nnetdir/xvector_scores_hires/test.new | cut -d ' ' -f 1 > $nnetdir/xvector_scores_hires/test.target
grep ' nontarget$' $nnetdir/xvector_scores_hires/test.new | cut -d ' ' -f 1 > $nnetdir/xvector_scores_hires/test.nontarget
comm=`echo "addpath('../../../misc/DETware_v2.1'); Get_DCF('$nnetdir/xvector_scores_hires/test.target', '$nnetdir/xvector_scores_hires/test.nontarget', '$nnetdir/xvector_scores_hires/test_lda_plda.result')"`
echo "$comm"| matlab -nodesktop -noFigureWindows > /dev/null
tail -n 1 $nnetdir/xvector_scores_hires/test_lda_plda.result

eer=$(paste $trials $nnetdir/xvector_scores_hires/test_lda_cos | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
echo "EER: ${eer}%"
paste $trials $nnetdir/xvector_scores_hires/test_lda_cos | awk '{print $6, $3}' > $nnetdir/xvector_scores_hires/test_lda_cos.new
grep ' target$' $nnetdir/xvector_scores_hires/test_lda_cos.new | cut -d ' ' -f 1 > $nnetdir/xvector_scores_hires/test_lda_cos.target
grep ' nontarget$' $nnetdir/xvector_scores_hires/test_lda_cos.new | cut -d ' ' -f 1 > $nnetdir/xvector_scores_hires/test_lda_cos.nontarget
comm=`echo "addpath('../../../misc/DETware_v2.1'); Get_DCF('$nnetdir/xvector_scores_hires/test_lda_cos.target', '$nnetdir/xvector_scores_hires/test_lda_cos.nontarget', '$nnetdir/xvector_scores_hires/test_lda_cos.result')"`
echo "$comm"| matlab -nodesktop -noFigureWindows > /dev/null
tail -n 1 $nnetdir/xvector_scores_hires/test_lda_cos.result
