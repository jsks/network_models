data {
  int N;
  int D;
  int K;
  int n_actors;
  array[N, 2] int<lower=1, upper=n_actors> actor_id;
  matrix[N, K] X;
  array[N] int<lower=0, upper=1> y;
}

parameters {
  vector[K] beta;
  vector[D] mu;
  matrix[D, n_actors] Z_raw;
}

transformed parameters {
  matrix[D, n_actors] Z;
  for (i in 1:D)
    Z[i] = mu[i] + Z_raw[i];

  vector[N] log_odds;
  for (i in 1:N)
    log_odds[i] = X[i, ] * beta - distance(Z[, actor_id[i, 1]], Z[, actor_id[i, 2]]);
}

model {
  beta ~ normal(0, 5);

  mu ~ std_normal();
  for (i in 1:D)
    Z_raw[, i] ~ std_normal();

  y ~ bernoulli_logit(log_odds);
}

generated quantities {
  array[N] int y_hat;
  for (i in 1:N)
    y_hat[i] = bernoulli_logit_rng(log_odds[i]);
}
