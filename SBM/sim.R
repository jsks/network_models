#!/usr/bin/env Rscript
#
# Simulation script for a stochastic block model using a gaussian
# mixture.
###

library(bayesplot)
library(blockmodels)
library(cmdstanr)
library(dplyr)
library(tidyr)

# Function for cluster assignment comparison that is invariant to
# label switching
compare_clusters <- function(x, y) {
    a <- factor(x, unique(x))
    b <- factor(y, unique(y))
    as.integer(a) == as.integer(b)
}

###
# Simulate dataset
set.seed(1682)

K <- 3
n_actors <- 10
sigma <- 1

Z <- sample(1:K, 10, replace = T)
mu <- matrix(rnorm(K*K, 0, 5), K, K)

dyads <- expand.grid(1:n_actors, 1:n_actors) |>
    filter(Var1 != Var2) |>
    mutate(y = NA)

dyads$y <- sapply(1:nrow(dyads), function(i) {
    actor1 <- dyads$Var1[i]
    actor2 <- dyads$Var2[i]
    rnorm(1, mu[Z[actor1], Z[actor2]], sigma)
})

data <- list(N = nrow(dyads),
             K = K,
             n_actors = n_actors,
             actor_id = select(dyads, Var1, Var2) |> data.matrix(),
             y = dyads$y)
str(data)

###
# Estimate model - note: due to label switching run only one chain
mod <- cmdstan_model("./sbm.stan")
fit <- mod$sample(data, chains = 1)

Z_hat <- sapply(1:n_actors, function(i) {
    theta <- fit$draws(sprintf("theta[%d,%d]", i, 1:K), format = "draws_matrix")
    apply(theta, 1, which.max)
})

# Proportion of posterior draws that recover all of the true cluster
# assignments, taking into account label switching
apply(Z_hat, 1, \(v) all(compare_clusters(Z, v))) |> mean()

###
# Estimation with variational EM using blockmodels
adjm <- pivot_wider(dyads, id_cols = Var1, names_from = Var2, values_from = y) |>
    arrange(Var1) |>
    select(-Var1) |>
    data.matrix()

diag(adjm) <- 0

sbm <- BM_gaussian("SBM", adjm)
sbm$estimate()

sbm_Z_hat <- apply(sbm$memberships[[3]]$Z, 1, which.max)

compare_clusters(Z, sbm_Z_hat) |> mean()
