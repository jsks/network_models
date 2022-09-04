Social Relations Model
---

Let the directed relationship, $y_{ij} \in \mathbb{R}$, between a
sender, $i$, and receiver, $j$, be a function of a set of dyadic-level
and actor-level regressors and additive effects with the following
specification:

$$
\begin{align*}
y_{ij} & = \alpha +  X_{ij}' \beta + \gamma_i + \upsilon_j + \epsilon_{ij} \\
(\gamma_i, \upsilon_i) & \sim \text{MVN}(0, \Sigma_{\gamma\upsilon}) \\
(\epsilon_{ij}, \epsilon_{ji}) & \sim \text{MVN}(0, \Sigma_{\epsilon}) \\
\Sigma_{\gamma\upsilon} & =
\begin{bmatrix}
\sigma_{\gamma}^2 & \sigma_{\gamma\upsilon} \\
\sigma_{\gamma\upsilon} & \sigma_{\upsilon}^2
\end{bmatrix} \\;
\Sigma_{\epsilon} = \sigma_{\epsilon}^2
\begin{bmatrix}
1 & \rho \\
\rho & 1
\end{bmatrix}
\end{align*}
$$

where $\gamma_i$ captures sender heterogeneity, $\upsilon_j$ receiver
heterogeneity, and $\rho$ the within-dyad correlation between error
terms.
