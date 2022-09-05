Latent Distance Model
---

Given a symmetric adjacency matrix, let $y_{ij} \in \\{ 0, 1 \\}$ denote whether a tie exists between actor $i$ and $j$. Then, 

$$
\begin{align*}
y_{ij} & \sim \text{Bernoulli}(\pi_{ij}) \\
\pi_{ij} & = \text{logit}^{-1}\left( X_{ij}' \beta - \|\| Z_i - Z_j \|\| \right) \\
Z_i & \sim \text{MVN}(0, \sigma^2 I)
\end{align*}
$$

**Note**: The stan implementation is still a work in progress and does not work as is yet.
