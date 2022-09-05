#!/usr/bin/env Rscript
#
# Creates a simulated dataset and checks that we can recover the true
# parameters with Stan.
###

library(bayesplot)
library(cmdstanr)
library(dplyr)
library(extraDistr)
library(MASS, exclude = "select")

options(mc.cores = parallel::detectCores() - 1)

###
# Simulate a directed dyadic dataset
set.seed(6602)

n_actors <- 10
n_dyads <- choose(n_actors, 2)

# List of undirected dyads
dyads <- combn(1:n_actors, 2) |>
    t() |>
    data.frame() |>
    select(Actor1 = X1, Actor2 = X2)

# Dyadic level predictors
X_dyad <- data.frame(X = rnorm(n_dyads))

# Actor level predictors
X_actor <- data.frame(X = rnorm(n_actors))

# Actor level correlation matrix
Omega_actor <- matrix(c(1, 0.8, 0.8, 1), 2, 2)

# Actor level standard deviation
tau <- rexp(2, 10)

# Actor level covariance matrix
Sigma_actor <- diag(tau, 2, 2) %*% Omega_actor %*% diag(tau, 2, 2)

# Actor random effects
gamma <- mvrnorm(n_actors, c(0, 0), Sigma_actor)

# Dyadic level correlation matrix
rho <- 0.6
Omega_dyad <- matrix(c(1, rho, rho, 1), 2, 2)

# Dyadic level variance
sigma <- rexp(1)

# Dyadic level covariance matrix
Sigma_dyad <- sigma * Omega_dyad

# Correlated error terms
epsilon <- mvrnorm(n_dyads, c(0, 0), Sigma_dyad)

# Multiplicative latent effects
D <- 2

mlatent <- rnorm(n_actors * 2 * D) |> matrix(n_actors, 2 * D)
uv <- tcrossprod(mlatent[, 1:D], mlatent[, (D+1):ncol(mlatent)])

# Column one is multiplicative terms for i->j dyad, and column two for j->i
multi <- select(dyads, Actor1, Actor2) |>
    apply(1, \(actors) c(uv[actors[1], actors[2]], uv[actors[2], actors[1]])) |>
    t()

# Regression Parameters
alpha <- 0.5
lambda <- 4
beta <- c(-2, 1.5)

# Dyadic level regression terms
nu <- alpha + lambda * X_dyad[, 1]

mu <- matrix(NA, n_dyads, 2)

# Simulate mean for i->j relation w/ Actor1 as sender and Actor2 as receiver
mu[, 1] <- with(dyads, nu + beta[1] * X_actor[Actor1, ] + beta[2] * X_actor[Actor2, ])

# Simulate mean for j->i relation w/ Actor2 as sender and Actor1 as receiver
mu[, 2] <- with(dyads, nu + beta[1] * X_actor[Actor2, ] + beta[2] * X_actor[Actor1, ])

# Add in random effects and error terms. gamma[, 1] captures sender
# heterogeneity and gamma[, 2] receiver heterogeneity.
y <- matrix(NA, n_dyads, 2)
y[, 1] <- with(dyads, mu[, 1] + gamma[Actor1, 1] + gamma[Actor2, 2] +
                      multi[, 1] + epsilon[, 1])
y[, 2] <- with(dyads, mu[, 2] + gamma[Actor2, 1] + gamma[Actor1, 2] +
                      multi[, 2] + epsilon[, 2])

###
# Fit simulated data
mod <- cmdstan_model("ame.stan")
data <- list(n_dyads = n_dyads,
             n_actors = n_actors,
             K_dyad = 1,
             K_actor = 1,
             D = 2,
             X_dyad = data.matrix(X_dyad),
             X_actor = data.matrix(X_actor),
             actor_id = data.matrix(dyads),
             y = y)
str(data)

fit <- mod$sample(data = data, chains = 4)

###
# Check if we recover the true parameter values
theta <- fit$draws(c("alpha", "beta", "lambda", "sigma", "tau", "DyadCorrMat[2,1]"),
                   format = "draws_df") |>
    rename(rho = `DyadCorrMat[2,1]`) |>
    mutate(sigma = sigma^2)

mcmc_recover_intervals(theta, c(alpha, beta, lambda, sigma, tau, rho))

###
# Latent interaction term per directed dyad
uv_hat <- fit$draws("uv", format = "draws_matrix")

mcmc_recover_intervals(uv_hat, as.vector(uv))

###
# Plot posterior predictions for y
y_hat <- fit$draws("y_hat", format = "draws_matrix")
ppc_dens_overlay(c(y[, 1], y[, 2]), y_hat[1:100, ])
