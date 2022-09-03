data {
  int n_dyads;                       // Total num unique undirected dyads
  int n_actors;                      // Total num of actors
  int K_dyad;                        // Num of dyadic level predictors
  int K_actor;                       // Num of actor level predicts
  int D;                             // Dimensions for multi. latent terms

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

  //cholesky_factor_corr[D] L_multi;
  //matrix[2, D] kappa;
  matrix[n_actors, 2*D] theta;
  //array[D] matrix[n_actors, 2] theta;
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

  //array[D] matrix[n_actors, 2] theta;
  //for (i in 1:D)
    // theta[i, actor, ] ~ multi_normal(0, CovMat)
    //theta[i, ] = (diag_pre_multiply(kappa[, i], L_multi) * theta_raw[i, ])';

  matrix[n_actors, n_actors] multi = theta[, :D] * theta[, (D+1):]';

  // Mean vector for likelihood
  array[n_dyads] vector[2] mu;
  {
    vector[n_dyads] nu = alpha + X_dyad * lambda;
    for (i in 1:n_dyads) {
      // Actor1 as sender and Actor2 as receiver
      mu[i, 1] = nu[i] + gamma[actor_id[i, 1], 1] + gamma[actor_id[i, 2], 2] +
        X_actor[actor_id[i, 1], ] * beta[:K_actor] +
        X_actor[actor_id[i, 2], ] * beta[(K_actor+1):] +
        multi[actor_id[i, 1], actor_id[i, 2]];

      // Actor2 as sender and Actor1 as receiver
      mu[i, 2] = nu[i] + gamma[actor_id[i, 2], 1] + gamma[actor_id[i, 1], 2] +
        X_actor[actor_id[i, 2], ] * beta[:K_actor] +
        X_actor[actor_id[i, 1], ] * beta[(K_actor+1):] +
        multi[actor_id[i, 2], actor_id[i, 1]];
    }
  }
}

model {
  // Priors
  alpha ~ normal(0, 5);
  lambda ~ normal(0, 2.5);
  beta ~ normal(0, 2.5);

  L_dyad ~ lkj_corr_cholesky(2);

  L_actor ~ lkj_corr_cholesky(2);
  to_vector(gamma_raw) ~ std_normal();

  //L_multi ~ lkj_corr_cholesky(2);
  //to_vector(kappa) ~ student_t(4, 0, 1);
  to_vector(theta) ~ std_normal();

  // Likelihood
  for (i in 1:n_dyads)
    y[i, ] ~ multi_normal_cholesky(mu[i, ], L_Omega);
}

generated quantities {
  corr_matrix[2] ActorCorrMat = multiply_lower_tri_self_transpose(L_actor);
  corr_matrix[2] DyadCorrMat = multiply_lower_tri_self_transpose(L_dyad);
  //corr_matrix[2] MultiCorrMat = multiply_lower_tri_self_transpose(L_multi);

  array[n_dyads] vector[2] y_hat;
  for (i in 1:n_dyads)
    y_hat[i, ] = multi_normal_cholesky_rng(mu[i, ], L_Omega);
}
