#!/bin/bash

gender=$1
data=$2
nnetdir=$3

utils/filter_scp.pl $sre16_trials_tgl $nnetdir/xvector_scores/sre16_eval_scores > $nnetdir/xvector_scores/sre16_eval_tgl_scores
utils/filter_scp.pl $sre16_trials_yue $nnetdir/xvector_scores/sre16_eval_scores > $nnetdir/xvector_scores/sre16_eval_yue_scores
pooled_eer=$(paste $sre16_trials $nnetdir/xvector_scores/sre16_eval_scores | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
tgl_eer=$(paste $sre16_trials_tgl $nnetdir/xvector_scores/sre16_eval_tgl_scores | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
yue_eer=$(paste $sre16_trials_yue $nnetdir/xvector_scores/sre16_eval_yue_scores | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
echo "Using Out-of-Domain PLDA, EER: Pooled ${pooled_eer}%, Tagalog ${tgl_eer}%, Cantonese ${yue_eer}%"

paste $sre16_trials $nnetdir/xvector_scores/sre16_eval_scores | awk '{print $6, $3}' > $nnetdir/xvector_scores/sre16_eval_scores.new
grep ' target$' $nnetdir/xvector_scores/sre16_eval_scores.new | cut -d ' ' -f 1 > $nnetdir/xvector_scores/sre16_eval_scores.target
grep ' nontarget$' $nnetdir/xvector_scores/sre16_eval_scores.new | cut -d ' ' -f 1 > $nnetdir/xvector_scores/sre16_eval_scores.nontarget
comm=`echo "addpath('../../../misc/DETware_v2.1'); Get_DCF('$nnetdir/xvector_scores/sre16_eval_scores.target', '$nnetdir/xvector_scores/sre16_eval_scores.nontarget', '$nnetdir/xvector_scores/sre16_eval_scores.result')"`
echo "$comm"| matlab -nodesktop -noFigureWindows > /dev/null
# rm -f $nnetdir/xvector_scores/sre16_eval_scores.new $nnetdir/xvector_scores/sre16_eval_scores.target $nnetdir/xvector_scores/sre16_eval_scores.nontarget
tail -n 1 $nnetdir/xvector_scores/sre16_eval_scores.result

paste $sre16_trials_tgl $nnetdir/xvector_scores/sre16_eval_tgl_scores | awk '{print $6, $3}' > $nnetdir/xvector_scores/sre16_eval_tgl_scores.new
grep ' target$' $nnetdir/xvector_scores/sre16_eval_tgl_scores.new | cut -d ' ' -f 1 > $nnetdir/xvector_scores/sre16_eval_tgl_scores.target
grep ' nontarget$' $nnetdir/xvector_scores/sre16_eval_tgl_scores.new | cut -d ' ' -f 1 > $nnetdir/xvector_scores/sre16_eval_tgl_scores.nontarget
comm=`echo "addpath('../../../misc/DETware_v2.1'); Get_DCF('$nnetdir/xvector_scores/sre16_eval_tgl_scores.target', '$nnetdir/xvector_scores/sre16_eval_tgl_scores.nontarget', '$nnetdir/xvector_scores/sre16_eval_tgl_scores.result')"`
echo "$comm"| matlab -nodesktop -noFigureWindows > /dev/null
# rm -f $nnetdir/xvector_scores/sre16_eval_tgl_scores.new $nnetdir/xvector_scores/sre16_eval_tgl_scores.target $nnetdir/xvector_scores/sre16_eval_tgl_scores.nontarget
tail -n 1 $nnetdir/xvector_scores/sre16_eval_tgl_scores.result

paste $sre16_trials_yue $nnetdir/xvector_scores/sre16_eval_yue_scores | awk '{print $6, $3}' > $nnetdir/xvector_scores/sre16_eval_yue_scores.new
grep ' target$' $nnetdir/xvector_scores/sre16_eval_yue_scores.new | cut -d ' ' -f 1 > $nnetdir/xvector_scores/sre16_eval_yue_scores.target
grep ' nontarget$' $nnetdir/xvector_scores/sre16_eval_yue_scores.new | cut -d ' ' -f 1 > $nnetdir/xvector_scores/sre16_eval_yue_scores.nontarget
comm=`echo "addpath('../../../misc/DETware_v2.1'); Get_DCF('$nnetdir/xvector_scores/sre16_eval_yue_scores.target', '$nnetdir/xvector_scores/sre16_eval_yue_scores.nontarget', '$nnetdir/xvector_scores/sre16_eval_yue_scores.result')"`
echo "$comm"| matlab -nodesktop -noFigureWindows > /dev/null
# rm -f $nnetdir/xvector_scores/sre16_eval_yue_scores.new $nnetdir/xvector_scores/sre16_eval_yue_scores.target $nnetdir/xvector_scores/sre16_eval_yue_scores.nontarget
tail -n 1 $nnetdir/xvector_scores/sre16_eval_yue_scores.result


utils/filter_scp.pl $sre16_trials_tgl $nnetdir/xvector_scores/sre16_eval_scores_adapt > $nnetdir/xvector_scores/sre16_eval_tgl_scores_adapt
utils/filter_scp.pl $sre16_trials_yue $nnetdir/xvector_scores/sre16_eval_scores_adapt > $nnetdir/xvector_scores/sre16_eval_yue_scores_adapt
pooled_eer=$(paste $sre16_trials $nnetdir/xvector_scores/sre16_eval_scores_adapt | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
tgl_eer=$(paste $sre16_trials_tgl $nnetdir/xvector_scores/sre16_eval_tgl_scores_adapt | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
yue_eer=$(paste $sre16_trials_yue $nnetdir/xvector_scores/sre16_eval_yue_scores_adapt | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
echo "Using Adapted PLDA, EER: Pooled ${pooled_eer}%, Tagalog ${tgl_eer}%, Cantonese ${yue_eer}%"

paste $sre16_trials $nnetdir/xvector_scores/sre16_eval_scores_adapt | awk '{print $6, $3}' > $nnetdir/xvector_scores/sre16_eval_scores_adapt.new
grep ' target$' $nnetdir/xvector_scores/sre16_eval_scores_adapt.new | cut -d ' ' -f 1 > $nnetdir/xvector_scores/sre16_eval_scores_adapt.target
grep ' nontarget$' $nnetdir/xvector_scores/sre16_eval_scores_adapt.new | cut -d ' ' -f 1 > $nnetdir/xvector_scores/sre16_eval_scores_adapt.nontarget
comm=`echo "addpath('../../../misc/DETware_v2.1'); Get_DCF('$nnetdir/xvector_scores/sre16_eval_scores_adapt.target', '$nnetdir/xvector_scores/sre16_eval_scores_adapt.nontarget', '$nnetdir/xvector_scores/sre16_eval_scores_adapt.result')"`
echo "$comm"| matlab -nodesktop -noFigureWindows > /dev/null
# rm -f $nnetdir/xvector_scores/sre16_eval_scores_adapt.new $nnetdir/xvector_scores/sre16_eval_scores_adapt.target $nnetdir/xvector_scores/sre16_eval_scores_adapt.nontarget
tail -n 1 $nnetdir/xvector_scores/sre16_eval_scores_adapt.result

paste $sre16_trials_tgl $nnetdir/xvector_scores/sre16_eval_tgl_scores_adapt | awk '{print $6, $3}' > $nnetdir/xvector_scores/sre16_eval_tgl_scores_adapt.new
grep ' target$' $nnetdir/xvector_scores/sre16_eval_tgl_scores_adapt.new | cut -d ' ' -f 1 > $nnetdir/xvector_scores/sre16_eval_tgl_scores_adapt.target
grep ' nontarget$' $nnetdir/xvector_scores/sre16_eval_tgl_scores_adapt.new | cut -d ' ' -f 1 > $nnetdir/xvector_scores/sre16_eval_tgl_scores_adapt.nontarget
comm=`echo "addpath('../../../misc/DETware_v2.1'); Get_DCF('$nnetdir/xvector_scores/sre16_eval_tgl_scores_adapt.target', '$nnetdir/xvector_scores/sre16_eval_tgl_scores_adapt.nontarget', '$nnetdir/xvector_scores/sre16_eval_tgl_scores_adapt.result')"`
echo "$comm"| matlab -nodesktop -noFigureWindows > /dev/null
# rm -f $nnetdir/xvector_scores/sre16_eval_tgl_scores_adapt.new $nnetdir/xvector_scores/sre16_eval_tgl_scores_adapt.target $nnetdir/xvector_scores/sre16_eval_tgl_scores_adapt.nontarget
tail -n 1 $nnetdir/xvector_scores/sre16_eval_tgl_scores_adapt.result

paste $sre16_trials_yue $nnetdir/xvector_scores/sre16_eval_yue_scores_adapt | awk '{print $6, $3}' > $nnetdir/xvector_scores/sre16_eval_yue_scores_adapt.new
grep ' target$' $nnetdir/xvector_scores/sre16_eval_yue_scores_adapt.new | cut -d ' ' -f 1 > $nnetdir/xvector_scores/sre16_eval_yue_scores_adapt.target
grep ' nontarget$' $nnetdir/xvector_scores/sre16_eval_yue_scores_adapt.new | cut -d ' ' -f 1 > $nnetdir/xvector_scores/sre16_eval_yue_scores_adapt.nontarget
comm=`echo "addpath('../../../misc/DETware_v2.1'); Get_DCF('$nnetdir/xvector_scores/sre16_eval_yue_scores_adapt.target', '$nnetdir/xvector_scores/sre16_eval_yue_scores_adapt.nontarget', '$nnetdir/xvector_scores/sre16_eval_yue_scores_adapt.result')"`
echo "$comm"| matlab -nodesktop -noFigureWindows > /dev/null
# rm -f $nnetdir/xvector_scores/sre16_eval_yue_scores_adapt.new $nnetdir/xvector_scores/sre16_eval_yue_scores_adapt.target $nnetdir/xvector_scores/sre16_eval_yue_scores_adapt.nontarget
tail -n 1 $nnetdir/xvector_scores/sre16_eval_yue_scores_adapt.result
