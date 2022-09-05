data {
  int N;        // Num of obs.
  int D;        // Num of latent dimensions
  int K;        // Num of regressors, including intercept
  int n_actors; // Num of unique actors

  // Actor IDs per dyad
  array[N, 2] int<lower=1, upper=n_actors> actor_id;

  // Covariate design matrix
  matrix[N, K] X;

  array[N] int<lower=0, upper=1> y;
}

parameters {
  vector[K] beta;
  real<lower=0> sigma;
  matrix[D, n_actors] Z_raw;
}

transformed parameters {
  matrix[D, n_actors] Z = sigma * Z_raw;
  vector[N] d;
  for (i in 1:N)
    d[i] = distance(Z[, actor_id[i, 1]], Z[, actor_id[i, 2]]);

  vector[N] nu = X * beta - d;
}

model {
  beta ~ normal(0, 5);
  sigma ~ exponential(1);
  to_vector(Z_raw) ~ std_normal();

  y ~ bernoulli_logit(nu);
}

generated quantities {
  array[N] int y_hat;
  for (i in 1:N)
    y_hat[i] = bernoulli_logit_rng(nu[i]);
}
