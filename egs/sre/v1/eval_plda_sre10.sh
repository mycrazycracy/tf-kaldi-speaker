#!/bin/bash

gender=$1
data=$2
nnetdir=$3

cp $sre10_trials_c5_ext/../male/trials $data/sre10_test_coreext_c5_$gender/trials_male
cp $sre10_trials_c5_ext/../female/trials $data/sre10_test_coreext_c5_$gender/trials_female
utils/filter_scp.pl $data/sre10_test_coreext_c5_$gender/trials_male $nnetdir/xvector_scores/sre10_coreext_c5_scores_$gender > $nnetdir/xvector_scores/sre10_coreext_c5_scores_male
utils/filter_scp.pl $data/sre10_test_coreext_c5_$gender/trials_female $nnetdir/xvector_scores/sre10_coreext_c5_scores_$gender > $nnetdir/xvector_scores/sre10_coreext_c5_scores_female
pooled_eer=$(paste $data/sre10_test_coreext_c5_$gender/trials $nnetdir/xvector_scores/sre10_coreext_c5_scores_$gender | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
male_eer=$(paste $data/sre10_test_coreext_c5_$gender/trials_male $nnetdir/xvector_scores/sre10_coreext_c5_scores_male | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
female_eer=$(paste $data/sre10_test_coreext_c5_$gender/trials_female $nnetdir/xvector_scores/sre10_coreext_c5_scores_female | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
echo "EER: Pooled ${pooled_eer}%, Male ${male_eer}%, Female ${female_eer}%"

paste $data/sre10_test_coreext_c5_$gender/trials $nnetdir/xvector_scores/sre10_coreext_c5_scores_$gender | awk '{print $6, $3}' > $nnetdir/xvector_scores/sre10_coreext_c5_scores_${gender}.new
grep ' target$' $nnetdir/xvector_scores/sre10_coreext_c5_scores_${gender}.new | cut -d ' ' -f 1 > $nnetdir/xvector_scores/sre10_coreext_c5_scores_${gender}.target
grep ' nontarget$' $nnetdir/xvector_scores/sre10_coreext_c5_scores_${gender}.new | cut -d ' ' -f 1 > $nnetdir/xvector_scores/sre10_coreext_c5_scores_${gender}.nontarget
comm=`echo "addpath('../../../misc/DETware_v2.1'); Get_DCF('$nnetdir/xvector_scores/sre10_coreext_c5_scores_${gender}.target', '$nnetdir/xvector_scores/sre10_coreext_c5_scores_${gender}.nontarget', '$nnetdir/xvector_scores/sre10_coreext_c5_scores_${gender}.result')"`
echo "$comm"| matlab -nodesktop -noFigureWindows > /dev/null
# rm -f $nnetdir/xvector_scores/sre10_coreext_c5_scores_${gender}.new $nnetdir/xvector_scores/sre10_coreext_c5_scores_${gender}.target $nnetdir/xvector_scores/sre10_coreext_c5_scores_${gender}.nontarget
tail -n 1 $nnetdir/xvector_scores/sre10_coreext_c5_scores_${gender}.result

paste $data/sre10_test_coreext_c5_$gender/trials_male $nnetdir/xvector_scores/sre10_coreext_c5_scores_male | awk '{print $6, $3}' > $nnetdir/xvector_scores/sre10_coreext_c5_scores_male.new
grep ' target$' $nnetdir/xvector_scores/sre10_coreext_c5_scores_male.new | cut -d ' ' -f 1 > $nnetdir/xvector_scores/sre10_coreext_c5_scores_male.target
grep ' nontarget$' $nnetdir/xvector_scores/sre10_coreext_c5_scores_male.new | cut -d ' ' -f 1 > $nnetdir/xvector_scores/sre10_coreext_c5_scores_male.nontarget
comm=`echo "addpath('../../../misc/DETware_v2.1'); Get_DCF('$nnetdir/xvector_scores/sre10_coreext_c5_scores_male.target', '$nnetdir/xvector_scores/sre10_coreext_c5_scores_male.nontarget', '$nnetdir/xvector_scores/sre10_coreext_c5_scores_male.result')"`
echo "$comm"| matlab -nodesktop -noFigureWindows > /dev/null
# rm -f $nnetdir/xvector_scores/sre10_coreext_c5_scores_male.new $nnetdir/xvector_scores/sre10_coreext_c5_scores_male.target $nnetdir/xvector_scores/sre10_coreext_c5_scores_male.nontarget
tail -n 1 $nnetdir/xvector_scores/sre10_coreext_c5_scores_male.result

paste $data/sre10_test_coreext_c5_$gender/trials_female $nnetdir/xvector_scores/sre10_coreext_c5_scores_female | awk '{print $6, $3}' > $nnetdir/xvector_scores/sre10_coreext_c5_scores_female.new
grep ' target$' $nnetdir/xvector_scores/sre10_coreext_c5_scores_female.new | cut -d ' ' -f 1 > $nnetdir/xvector_scores/sre10_coreext_c5_scores_female.target
grep ' nontarget$' $nnetdir/xvector_scores/sre10_coreext_c5_scores_female.new | cut -d ' ' -f 1 > $nnetdir/xvector_scores/sre10_coreext_c5_scores_female.nontarget
comm=`echo "addpath('../../../misc/DETware_v2.1'); Get_DCF('$nnetdir/xvector_scores/sre10_coreext_c5_scores_female.target', '$nnetdir/xvector_scores/sre10_coreext_c5_scores_female.nontarget', '$nnetdir/xvector_scores/sre10_coreext_c5_scores_female.result')"`
echo "$comm"| matlab -nodesktop -noFigureWindows > /dev/null
# rm -f $nnetdir/xvector_scores/sre10_coreext_c5_scores_female.new $nnetdir/xvector_scores/sre10_coreext_c5_scores_female.target $nnetdir/xvector_scores/sre10_coreext_c5_scores_female.nontarget
tail -n 1 $nnetdir/xvector_scores/sre10_coreext_c5_scores_female.result


cp $sre10_trials_10s/../male/trials $data/sre10_test_10s_$gender/trials_male
cp $sre10_trials_10s/../female/trials $data/sre10_test_10s_$gender/trials_female
utils/filter_scp.pl $data/sre10_test_10s_$gender/trials_male $nnetdir/xvector_scores/sre10_10s_scores_$gender > $nnetdir/xvector_scores/sre10_10s_scores_male
utils/filter_scp.pl $data/sre10_test_10s_$gender/trials_female $nnetdir/xvector_scores/sre10_10s_scores_$gender > $nnetdir/xvector_scores/sre10_10s_scores_female
pooled_eer=$(paste $data/sre10_test_10s_$gender/trials $nnetdir/xvector_scores/sre10_10s_scores_$gender | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
male_eer=$(paste $data/sre10_test_10s_$gender/trials_male $nnetdir/xvector_scores/sre10_10s_scores_male | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
female_eer=$(paste $data/sre10_test_10s_$gender/trials_female $nnetdir/xvector_scores/sre10_10s_scores_female | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
echo "EER: Pooled ${pooled_eer}%, Male ${male_eer}%, Female ${female_eer}%"

paste $data/sre10_test_10s_$gender/trials $nnetdir/xvector_scores/sre10_10s_scores_$gender | awk '{print $6, $3}' > $nnetdir/xvector_scores/sre10_10s_scores_${gender}.new
grep ' target$' $nnetdir/xvector_scores/sre10_10s_scores_${gender}.new | cut -d ' ' -f 1 > $nnetdir/xvector_scores/sre10_10s_scores_${gender}.target
grep ' nontarget$' $nnetdir/xvector_scores/sre10_10s_scores_${gender}.new | cut -d ' ' -f 1 > $nnetdir/xvector_scores/sre10_10s_scores_${gender}.nontarget
comm=`echo "addpath('../../../misc/DETware_v2.1'); Get_DCF('$nnetdir/xvector_scores/sre10_10s_scores_${gender}.target', '$nnetdir/xvector_scores/sre10_10s_scores_${gender}.nontarget', '$nnetdir/xvector_scores/sre10_10s_scores_${gender}.result')"`
echo "$comm"| matlab -nodesktop -noFigureWindows > /dev/null
# rm -f $nnetdir/xvector_scores/sre10_10s_scores_${gender}.new $nnetdir/xvector_scores/sre10_10s_scores_${gender}.target $nnetdir/xvector_scores/sre10_10s_scores_${gender}.nontarget
tail -n 1 $nnetdir/xvector_scores/sre10_10s_scores_${gender}.result

paste $data/sre10_test_10s_$gender/trials_male $nnetdir/xvector_scores/sre10_10s_scores_male | awk '{print $6, $3}' > $nnetdir/xvector_scores/sre10_10s_scores_male.new
grep ' target$' $nnetdir/xvector_scores/sre10_10s_scores_male.new | cut -d ' ' -f 1 > $nnetdir/xvector_scores/sre10_10s_scores_male.target
grep ' nontarget$' $nnetdir/xvector_scores/sre10_10s_scores_male.new | cut -d ' ' -f 1 > $nnetdir/xvector_scores/sre10_10s_scores_male.nontarget
comm=`echo "addpath('../../../misc/DETware_v2.1'); Get_DCF('$nnetdir/xvector_scores/sre10_10s_scores_male.target', '$nnetdir/xvector_scores/sre10_10s_scores_male.nontarget', '$nnetdir/xvector_scores/sre10_10s_scores_male.result')"`
echo "$comm"| matlab -nodesktop -noFigureWindows > /dev/null
# rm -f $nnetdir/xvector_scores/sre10_10s_scores_male.new $nnetdir/xvector_scores/sre10_10s_scores_male.target $nnetdir/xvector_scores/sre10_10s_scores_male.nontarget
tail -n 1 $nnetdir/xvector_scores/sre10_10s_scores_male.result

paste $data/sre10_test_10s_$gender/trials_female $nnetdir/xvector_scores/sre10_10s_scores_female | awk '{print $6, $3}' > $nnetdir/xvector_scores/sre10_10s_scores_female.new
grep ' target$' $nnetdir/xvector_scores/sre10_10s_scores_female.new | cut -d ' ' -f 1 > $nnetdir/xvector_scores/sre10_10s_scores_female.target
grep ' nontarget$' $nnetdir/xvector_scores/sre10_10s_scores_female.new | cut -d ' ' -f 1 > $nnetdir/xvector_scores/sre10_10s_scores_female.nontarget
comm=`echo "addpath('../../../misc/DETware_v2.1'); Get_DCF('$nnetdir/xvector_scores/sre10_10s_scores_female.target', '$nnetdir/xvector_scores/sre10_10s_scores_female.nontarget', '$nnetdir/xvector_scores/sre10_10s_scores_female.result')"`
echo "$comm"| matlab -nodesktop -noFigureWindows > /dev/null
# rm -f $nnetdir/xvector_scores/sre10_10s_scores_female.new $nnetdir/xvector_scores/sre10_10s_scores_female.target $nnetdir/xvector_scores/sre10_10s_scores_female.nontarget
tail -n 1 $nnetdir/xvector_scores/sre10_10s_scores_female.result

