function [eer, dcf08, dcf10, dcf12] = Get_DCF(target, imposter, output)

tar = load(target);
non = load(imposter);

lim = [0.0001 0.95];
Set_DET_limits(lim(1), lim(2), lim(1), lim(2));

% EER
[Pmiss, Pfa, eer] = Compute_DET(tar, non);

% DCF08 for DCF12
Set_DCF(1, 1, 0.01);
[DCF_opt, Popt_miss, Popt_fa] = Min_DCF(Pmiss, Pfa);
dcf08 = DCF_opt * 100;

% DCF10 for DCF12
Set_DCF(1, 1, 0.001);
[DCF_opt, Popt_miss, Popt_fa] = Min_DCF(Pmiss, Pfa);
Plot_DET(Popt_miss, max(Popt_fa, lim(1)), 'ro', 2);
dcf10 = DCF_opt * 1000;

% DCF12
dcf12 = (dcf08 + dcf10) / 2;

% DCF08
Set_DCF(10, 1, 0.01);
[DCF_opt, Popt_miss, Popt_fa] = Min_DCF(Pmiss, Pfa);
dcf08 = DCF_opt;

% DCF10
Set_DCF(1, 1, 0.001);
[DCF_opt, Popt_miss, Popt_fa] = Min_DCF(Pmiss, Pfa);
dcf10 = DCF_opt * 1000;

fid = fopen(output, 'a');
fprintf(fid, 'eer: %5.4f%%; mindcf08: %5.4f%%; mindcf10: %5.4f%%; mindcf12: %5.4f%%\n', eer*100, dcf08, dcf10, dcf12);
fclose(fid);