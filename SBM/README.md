
Stochastic Block Model
---

Given a directed relationship, $y_{ij} \in \mathbb{R}$, from actor $i$
to actor $j$, the stochastic block model with $K$ latent groups is
specified as follows:

$$
\begin{align*}
y_{ij}|Z_i, Z_j & \sim \text{Normal}(\mu_{z_i, z_j}, \sigma^2) \\
Z_i & \sim \text{Categorical}(\boldsymbol{\pi_i}) \\
\boldsymbol{\pi_i} & \sim \text{Dirichlet}(\alpha)
\end{align*}
$$

where $Z_i = h$ if actor $i$ belongs to latent class $h$ and
$\boldsymbol{\pi_i}$ is the actor specific mixture probabilities,
$\boldsymbol{\pi} = (\pi_1, \ldots, \pi_K)$. Thus, $y$ is modeled as a
mixture of gaussians where the location parameter, 
$\boldsymbol{\mu}\in \mathbb{R}^{K \times K}$, is indexed by the 
latent classification of both actors.

Since Stan doesn't support discrete priors on model parameters, the
latent classifications are marginalized out to form the log
likelihood.

$$
\begin{align*}
f(y_{ij} | \pi_i, \pi_j, \mu_{z_i, z_j}, \sigma^2) =
  \sum_{h=1}^K \sum_{z=1}^K \pi_{i,h} \pi_{j,z} f(y_{ij} | \mu_{h,z}, \sigma^2) \\
\log{f(y_{ij} | \pi_i, \pi_j, \mu_{z_i, z_j}, \sigma^2)} =
  \sum_{h=1}^K \sum_{z=1}^K \log{\pi_{i,h}} +  \log{\pi_{j,z}} +
    \log{f(y_{ij} | \mu_{h,z}, \sigma^2)}
\end{align*}
$$
