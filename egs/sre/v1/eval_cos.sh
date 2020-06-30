#!/bin/bash

gender=$1
data=$2
nnetdir=$3

utils/filter_scp.pl $data/sre10_test_coreext_c5_$gender/trials_male $nnetdir/xvector_scores/sre10_coreext_c5_scores_${gender}.cos > $nnetdir/xvector_scores/sre10_coreext_c5_scores_male.cos
utils/filter_scp.pl $data/sre10_test_coreext_c5_$gender/trials_female $nnetdir/xvector_scores/sre10_coreext_c5_scores_${gender}.cos > $nnetdir/xvector_scores/sre10_coreext_c5_scores_female.cos
pooled_eer=$(paste $data/sre10_test_coreext_c5_$gender/trials $nnetdir/xvector_scores/sre10_coreext_c5_scores_${gender}.cos | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
male_eer=$(paste $data/sre10_test_coreext_c5_$gender/trials_male $nnetdir/xvector_scores/sre10_coreext_c5_scores_male.cos | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
female_eer=$(paste $data/sre10_test_coreext_c5_$gender/trials_female $nnetdir/xvector_scores/sre10_coreext_c5_scores_female.cos | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
echo "EER: Pooled ${pooled_eer}%, Male ${male_eer}%, Female ${female_eer}%"

paste $data/sre10_test_coreext_c5_$gender/trials $nnetdir/xvector_scores/sre10_coreext_c5_scores_${gender}.cos | awk '{print $6, $3}' > $nnetdir/xvector_scores/sre10_coreext_c5_scores_${gender}.cos.new
grep ' target$' $nnetdir/xvector_scores/sre10_coreext_c5_scores_${gender}.cos.new | cut -d ' ' -f 1 > $nnetdir/xvector_scores/sre10_coreext_c5_scores_${gender}.cos.target
grep ' nontarget$' $nnetdir/xvector_scores/sre10_coreext_c5_scores_${gender}.cos.new | cut -d ' ' -f 1 > $nnetdir/xvector_scores/sre10_coreext_c5_scores_${gender}.cos.nontarget
comm=`echo "addpath('../../../misc/DETware_v2.1'); Get_DCF('$nnetdir/xvector_scores/sre10_coreext_c5_scores_${gender}.cos.target', '$nnetdir/xvector_scores/sre10_coreext_c5_scores_${gender}.cos.nontarget', '$nnetdir/xvector_scores/sre10_coreext_c5_scores_${gender}.cos.result')"`
echo "$comm"| matlab -nodesktop -noFigureWindows > /dev/null
tail -n 1 $nnetdir/xvector_scores/sre10_coreext_c5_scores_${gender}.cos.result

paste $data/sre10_test_coreext_c5_$gender/trials_male $nnetdir/xvector_scores/sre10_coreext_c5_scores_male.cos | awk '{print $6, $3}' > $nnetdir/xvector_scores/sre10_coreext_c5_scores_male.cos.new
grep ' target$' $nnetdir/xvector_scores/sre10_coreext_c5_scores_male.cos.new | cut -d ' ' -f 1 > $nnetdir/xvector_scores/sre10_coreext_c5_scores_male.cos.target
grep ' nontarget$' $nnetdir/xvector_scores/sre10_coreext_c5_scores_male.cos.new | cut -d ' ' -f 1 > $nnetdir/xvector_scores/sre10_coreext_c5_scores_male.cos.nontarget
comm=`echo "addpath('../../../misc/DETware_v2.1'); Get_DCF('$nnetdir/xvector_scores/sre10_coreext_c5_scores_male.cos.target', '$nnetdir/xvector_scores/sre10_coreext_c5_scores_male.cos.nontarget', '$nnetdir/xvector_scores/sre10_coreext_c5_scores_male.cos.result')"`
echo "$comm"| matlab -nodesktop -noFigureWindows > /dev/null
tail -n 1 $nnetdir/xvector_scores/sre10_coreext_c5_scores_male.cos.result

paste $data/sre10_test_coreext_c5_$gender/trials_female $nnetdir/xvector_scores/sre10_coreext_c5_scores_female.cos | awk '{print $6, $3}' > $nnetdir/xvector_scores/sre10_coreext_c5_scores_female.cos.new
grep ' target$' $nnetdir/xvector_scores/sre10_coreext_c5_scores_female.cos.new | cut -d ' ' -f 1 > $nnetdir/xvector_scores/sre10_coreext_c5_scores_female.cos.target
grep ' nontarget$' $nnetdir/xvector_scores/sre10_coreext_c5_scores_female.cos.new | cut -d ' ' -f 1 > $nnetdir/xvector_scores/sre10_coreext_c5_scores_female.cos.nontarget
comm=`echo "addpath('../../../misc/DETware_v2.1'); Get_DCF('$nnetdir/xvector_scores/sre10_coreext_c5_scores_female.cos.target', '$nnetdir/xvector_scores/sre10_coreext_c5_scores_female.cos.nontarget', '$nnetdir/xvector_scores/sre10_coreext_c5_scores_female.cos.result')"`
echo "$comm"| matlab -nodesktop -noFigureWindows > /dev/null
tail -n 1 $nnetdir/xvector_scores/sre10_coreext_c5_scores_female.cos.result


utils/filter_scp.pl $data/sre10_test_10s_$gender/trials_male $nnetdir/xvector_scores/sre10_10s_scores_${gender}.cos > $nnetdir/xvector_scores/sre10_10s_scores_male.cos
utils/filter_scp.pl $data/sre10_test_10s_$gender/trials_female $nnetdir/xvector_scores/sre10_10s_scores_${gender}.cos > $nnetdir/xvector_scores/sre10_10s_scores_female.cos
pooled_eer=$(paste $data/sre10_test_10s_$gender/trials $nnetdir/xvector_scores/sre10_10s_scores_${gender}.cos | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
male_eer=$(paste $data/sre10_test_10s_$gender/trials_male $nnetdir/xvector_scores/sre10_10s_scores_male.cos | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
female_eer=$(paste $data/sre10_test_10s_$gender/trials_female $nnetdir/xvector_scores/sre10_10s_scores_female.cos | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
echo "EER: Pooled ${pooled_eer}%, Male ${male_eer}%, Female ${female_eer}%"

paste $data/sre10_test_10s_$gender/trials $nnetdir/xvector_scores/sre10_10s_scores_${gender}.cos | awk '{print $6, $3}' > $nnetdir/xvector_scores/sre10_10s_scores_${gender}.cos.new
grep ' target$' $nnetdir/xvector_scores/sre10_10s_scores_${gender}.cos.new | cut -d ' ' -f 1 > $nnetdir/xvector_scores/sre10_10s_scores_${gender}.cos.target
grep ' nontarget$' $nnetdir/xvector_scores/sre10_10s_scores_${gender}.cos.new | cut -d ' ' -f 1 > $nnetdir/xvector_scores/sre10_10s_scores_${gender}.cos.nontarget
comm=`echo "addpath('../../../misc/DETware_v2.1'); Get_DCF('$nnetdir/xvector_scores/sre10_10s_scores_${gender}.cos.target', '$nnetdir/xvector_scores/sre10_10s_scores_${gender}.cos.nontarget', '$nnetdir/xvector_scores/sre10_10s_scores_${gender}.cos.result')"`
echo "$comm"| matlab -nodesktop -noFigureWindows > /dev/null
tail -n 1 $nnetdir/xvector_scores/sre10_10s_scores_${gender}.cos.result

paste $data/sre10_test_10s_$gender/trials_male $nnetdir/xvector_scores/sre10_10s_scores_male.cos | awk '{print $6, $3}' > $nnetdir/xvector_scores/sre10_10s_scores_male.cos.new
grep ' target$' $nnetdir/xvector_scores/sre10_10s_scores_male.cos.new | cut -d ' ' -f 1 > $nnetdir/xvector_scores/sre10_10s_scores_male.cos.target
grep ' nontarget$' $nnetdir/xvector_scores/sre10_10s_scores_male.cos.new | cut -d ' ' -f 1 > $nnetdir/xvector_scores/sre10_10s_scores_male.cos.nontarget
comm=`echo "addpath('../../../misc/DETware_v2.1'); Get_DCF('$nnetdir/xvector_scores/sre10_10s_scores_male.cos.target', '$nnetdir/xvector_scores/sre10_10s_scores_male.cos.nontarget', '$nnetdir/xvector_scores/sre10_10s_scores_male.cos.result')"`
echo "$comm"| matlab -nodesktop -noFigureWindows > /dev/null
tail -n 1 $nnetdir/xvector_scores/sre10_10s_scores_male.cos.result

paste $data/sre10_test_10s_$gender/trials_female $nnetdir/xvector_scores/sre10_10s_scores_female.cos | awk '{print $6, $3}' > $nnetdir/xvector_scores/sre10_10s_scores_female.cos.new
grep ' target$' $nnetdir/xvector_scores/sre10_10s_scores_female.cos.new | cut -d ' ' -f 1 > $nnetdir/xvector_scores/sre10_10s_scores_female.cos.target
grep ' nontarget$' $nnetdir/xvector_scores/sre10_10s_scores_female.cos.new | cut -d ' ' -f 1 > $nnetdir/xvector_scores/sre10_10s_scores_female.cos.nontarget
comm=`echo "addpath('../../../misc/DETware_v2.1'); Get_DCF('$nnetdir/xvector_scores/sre10_10s_scores_female.cos.target', '$nnetdir/xvector_scores/sre10_10s_scores_female.cos.nontarget', '$nnetdir/xvector_scores/sre10_10s_scores_female.cos.result')"`
echo "$comm"| matlab -nodesktop -noFigureWindows > /dev/null
tail -n 1 $nnetdir/xvector_scores/sre10_10s_scores_female.cos.result


utils/filter_scp.pl $sre16_trials_tgl $nnetdir/xvector_scores/sre16_eval_scores.cos > $nnetdir/xvector_scores/sre16_eval_tgl_scores.cos
utils/filter_scp.pl $sre16_trials_yue $nnetdir/xvector_scores/sre16_eval_scores.cos > $nnetdir/xvector_scores/sre16_eval_yue_scores.cos
pooled_eer=$(paste $sre16_trials $nnetdir/xvector_scores/sre16_eval_scores.cos | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
tgl_eer=$(paste $sre16_trials_tgl $nnetdir/xvector_scores/sre16_eval_tgl_scores.cos | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
yue_eer=$(paste $sre16_trials_yue $nnetdir/xvector_scores/sre16_eval_yue_scores.cos | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
echo "EER: Pooled ${pooled_eer}%, Tagalog ${tgl_eer}%, Cantonese ${yue_eer}%"

paste $sre16_trials $nnetdir/xvector_scores/sre16_eval_scores.cos | awk '{print $6, $3}' > $nnetdir/xvector_scores/sre16_eval_scores.cos.new
grep ' target$' $nnetdir/xvector_scores/sre16_eval_scores.cos.new | cut -d ' ' -f 1 > $nnetdir/xvector_scores/sre16_eval_scores.cos.target
grep ' nontarget$' $nnetdir/xvector_scores/sre16_eval_scores.cos.new | cut -d ' ' -f 1 > $nnetdir/xvector_scores/sre16_eval_scores.cos.nontarget
comm=`echo "addpath('../../../misc/DETware_v2.1'); Get_DCF('$nnetdir/xvector_scores/sre16_eval_scores.cos.target', '$nnetdir/xvector_scores/sre16_eval_scores.cos.nontarget', '$nnetdir/xvector_scores/sre16_eval_scores.cos.result')"`
echo "$comm"| matlab -nodesktop -noFigureWindows > /dev/null
tail -n 1 $nnetdir/xvector_scores/sre16_eval_scores.cos.result

paste $sre16_trials_tgl $nnetdir/xvector_scores/sre16_eval_tgl_scores.cos | awk '{print $6, $3}' > $nnetdir/xvector_scores/sre16_eval_tgl_scores.cos.new
grep ' target$' $nnetdir/xvector_scores/sre16_eval_tgl_scores.cos.new | cut -d ' ' -f 1 > $nnetdir/xvector_scores/sre16_eval_tgl_scores.cos.target
grep ' nontarget$' $nnetdir/xvector_scores/sre16_eval_tgl_scores.cos.new | cut -d ' ' -f 1 > $nnetdir/xvector_scores/sre16_eval_tgl_scores.cos.nontarget
comm=`echo "addpath('../../../misc/DETware_v2.1'); Get_DCF('$nnetdir/xvector_scores/sre16_eval_tgl_scores.cos.target', '$nnetdir/xvector_scores/sre16_eval_tgl_scores.cos.nontarget', '$nnetdir/xvector_scores/sre16_eval_tgl_scores.cos.result')"`
echo "$comm"| matlab -nodesktop -noFigureWindows > /dev/null
tail -n 1 $nnetdir/xvector_scores/sre16_eval_tgl_scores.cos.result

paste $sre16_trials_yue $nnetdir/xvector_scores/sre16_eval_yue_scores.cos | awk '{print $6, $3}' > $nnetdir/xvector_scores/sre16_eval_yue_scores.cos.new
grep ' target$' $nnetdir/xvector_scores/sre16_eval_yue_scores.cos.new | cut -d ' ' -f 1 > $nnetdir/xvector_scores/sre16_eval_yue_scores.cos.target
grep ' nontarget$' $nnetdir/xvector_scores/sre16_eval_yue_scores.cos.new | cut -d ' ' -f 1 > $nnetdir/xvector_scores/sre16_eval_yue_scores.cos.nontarget
comm=`echo "addpath('../../../misc/DETware_v2.1'); Get_DCF('$nnetdir/xvector_scores/sre16_eval_yue_scores.cos.target', '$nnetdir/xvector_scores/sre16_eval_yue_scores.cos.nontarget', '$nnetdir/xvector_scores/sre16_eval_yue_scores.cos.result')"`
echo "$comm"| matlab -nodesktop -noFigureWindows > /dev/null
tail -n 1 $nnetdir/xvector_scores/sre16_eval_yue_scores.cos.result