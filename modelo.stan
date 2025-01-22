
data {
    int<lower=1> N;
    vector[N] x;
    vector[N] y;
}
parameters {
    real alpha;
    real beta;
    real<lower=0> sigma;
}
model {
    alpha ~ normal(0, 1000);
    beta ~ normal(0, 1000);
    sigma ~ cauchy(0, 5);
    y ~ normal(alpha + beta * x, sigma);
}
