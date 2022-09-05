#!/usr/bin/env Rscript

library(bayesplot)
library(dplyr)
library(cmdstanr)
library(MASS, exclude = "select")
library(tidyr)

options(mc.cores = parallel::detectCores() - 1)

###
# Simulate binary undirected network
set.seed(8745)

D <- 2
n_actors <- 20
dyads <- combn(1:n_actors, 2) |>
    t() |>
    as.data.frame()

alpha <- 1
mu <- rnorm(D)
sigma <- 1
Z <- mvrnorm(n_actors, mu, diag(sigma, D, D))

distances <- dist(Z) |> as.matrix()
dyads$p <- apply(dyads, 1, function(actors) {
    plogis(alpha - distances[actors[1], actors[2]])
})

dyads$y <- rbinom(nrow(dyads), 1, dyads$p)
table(dyads$y)

mod <- cmdstan_model("./ldm.stan")
data <- list(N = nrow(dyads),
             D = D,
             K = 1,
             X = model.matrix(~ 1, data = dyads),
             n_actors = n_actors,
             actor_id = select(dyads, V1, V2) |> data.matrix(),
             y = dyads$y)
str(data)

fit <- mod$sample(data, chains = 1)
fit$cmdstan_diagnose()

###
# Adjacency matrix - estimate with latentnet
library(latentnet)

full.df <- dyads |> select(V1 = V2, V2 = V1, y, p) |> bind_rows(dyads)

adjm <- arrange(full.df, V1) |>
    pivot_wider(id_cols = V1, names_from = V2, values_from = y) |>
    select(`1`, everything(), -V1) |>
    as.matrix()
diag(adjm) <- 0

net <- as.network(adjm)
m1 <- ergmm(net ~ euclidean(d = D))

Z_hat <- m1$sample$Z[,,1]
colnames(Z_hat) <- 1:ncol(Z_hat)
mcmc_recover_intervals(Z_hat, Z[, 1])
