data {
  int n_dyads;                       // Total num unique undirected dyads
  int n_actors;                      // Total num of actors
  int K_dyad;                        // Num of dyadic level predictors
  int K_actor;                       // Num of actor level predicts

  matrix[n_dyads, K_dyad] X_dyad;    // Dyadic level predictors
  matrix[n_actors, K_actor] X_actor; // Actor level predictors

  // Actor IDs comprising each undirected dyad
  array[n_dyads, 2] int<lower=1, upper=n_actors> actor_id;

  array[n_dyads] vector[2] y;        // Two directed outcomes per dyad
}

parameters {
  real alpha;                        // Intercept
  vector[K_dyad] lambda;             // Reg. coefficients for dyadic predictors

  // Reg. coefficients for actor predictors. beta[1:K_actor] are the
  // sender coefficients, and beta[(K_actor+1):(2*K_actor)] are the
  // receiver coefficients.
  vector[2 * K_actor] beta;

  cholesky_factor_corr[2] L_dyad;
  real<lower=0, upper=pi()/2> sigma_unif;

  cholesky_factor_corr[2] L_actor;
  vector<lower=0, upper=pi()/2>[2] tau_unif;
  matrix[2, n_actors] gamma_raw;
}

transformed parameters {
  // Transformed priors
  // sigma ~ Half-Cauchy(0, 1)
  real<lower=0> sigma = tan(sigma_unif);

  // Cholesky factor for dyadic level covariance matrix
  matrix[2, 2] L_Omega = diag_pre_multiply(rep_vector(sigma, 2), L_dyad);

  // tau ~ Half-Cauchy(0, 1)
  vector<lower=0>[2] tau = tan(tau_unif);

  // gamma[actor, ] ~ multi_normal(0, CovMat)
  matrix[n_actors, 2] gamma = (diag_pre_multiply(tau, L_actor) * gamma_raw)';

  // Mean vector for likelihood
  array[n_dyads] vector[2] mu;
  {
    vector[n_dyads] nu = alpha + X_dyad * lambda;
    for (i in 1:n_dyads) {
      // Actor1 as sender and Actor2 as receiver
      mu[i, 1] = nu[i] + gamma[actor_id[i, 1], 1] + gamma[actor_id[i, 2], 2] +
        X_actor[actor_id[i, 1], ] * beta[:K_actor] +
        X_actor[actor_id[i, 2], ] * beta[(K_actor+1):];

      // Actor2 as sender and Actor1 as receiver
      mu[i, 2] = nu[i] + gamma[actor_id[i, 2], 1] + gamma[actor_id[i, 1], 2] +
        X_actor[actor_id[i, 2], ] * beta[:K_actor] +
        X_actor[actor_id[i, 1], ] * beta[(K_actor+1):];
    }
  }
}

model {
  // Priors
  target += normal_lpdf(alpha | 0, 5);
  target += normal_lpdf(lambda | 0, 2.5);
  target += normal_lpdf(beta | 0, 2.5);

  target += lkj_corr_cholesky_lpdf(L_dyad | 2);

  target += lkj_corr_cholesky_lpdf(L_actor | 2);
  target += std_normal_lpdf(to_vector(gamma_raw));

  // Likelihood
  for (i in 1:n_dyads)
    target += multi_normal_cholesky_lpdf(y[i, ] | mu[i, ], L_Omega);
}

generated quantities {
  corr_matrix[2] ActorCorrMat = multiply_lower_tri_self_transpose(L_actor);
  corr_matrix[2] DyadCorrMat = multiply_lower_tri_self_transpose(L_dyad);

  array[n_dyads] vector[2] y_hat;
  for (i in 1:n_dyads)
    y_hat[i, ] = multi_normal_cholesky_rng(mu[i, ], L_Omega);
}
