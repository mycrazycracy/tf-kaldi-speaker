clear

step = 1:1000000;

lambda_min = 10;
lambda_base = 1000;
gamma = 0.00001;
lambda_power = 5;

lambda = max(lambda_min, lambda_base * (1 + gamma * step).^(-lambda_power));
fa = 1.0 ./ (1.0 + lambda);
figure
plot(step, lambda);
xlim([0 800000])
ylim([0 100])
figure();
plot(step, fa);