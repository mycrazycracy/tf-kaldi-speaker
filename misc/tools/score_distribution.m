
tar = load('score.target.amsoftmax');
nontar = load('score.nontarget.amsoftmax');

[n_tar, c_tar] = hist(tar, 30);
n_tar = n_tar / sum(n_tar);
[n_nontar, c_nontar] = hist(nontar, 30);
n_nontar = n_nontar / sum(n_nontar);

plot(c_tar, n_tar, 'r--');
hold on;
plot(c_nontar, n_nontar, 'b--');