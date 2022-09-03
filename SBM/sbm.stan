data {
  int N;             // Num of observations
  int K;             // Num of latent classes
  int n_actors;      // Total num of unique actors

  // Sender and receiver IDs for each directed dyad
  array[N, 2] int<lower=1, upper=n_actors> actor_id;

  // Dyadic trade outcome
  array[N] real y;
}

parameters {
  real<lower=0, upper=pi()/2> sigma_unif;
  matrix[K, K] mu;
  array[n_actors] simplex[K] theta;
}

transformed parameters {
  // sigma ~ Half-Cauchy(0, 1)
  real sigma = tan(sigma_unif);

  // Cache the log transformation of mixture probabilities
  matrix[n_actors, K] ltheta;
  for (i in 1:n_actors) {
    for (j in 1:K)
      ltheta[i, j] = log(theta[i, j]);
  }
}

model {
  for (i in 1:n_actors)
    target += dirichlet_lpdf(theta[i, ] | rep_vector(1, K));

  for (i in 1:K)
    target += normal_lpdf(mu[i, ] | 0, 5);

  for (n in 1:N) {
    matrix[K, K] acc;
    for (i in 1:K) {
      for (j in 1:K) {
        acc[i, j] = ltheta[actor_id[n, 1], i] + ltheta[actor_id[n, 2], j] +
          normal_lpdf(y[n] | mu[i, j], sigma);
      }
    }

    target += log_sum_exp(acc);
  }
}
