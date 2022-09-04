Additive and Multiplicative Effects Model
---

The AME model extends the social relations model by adding
multiplicative latent terms to account for third-order
dependencies. Returning to the [SRM specification](../SRM/README.md)
where $y_{ij} \in \mathbb{R}$ is the directed relationship between
sender $i$ and receiver $j$, then

$$y_{ij} = \alpha +  X_{ij}' \beta + \gamma_i + \upsilon_j + u_{i}' v_{j} + \epsilon_{ij}$$

where $u_i$ is a $K$-vector of sender-specific latent variables and
$v_j$ a $K$-vector of receiver-specific latent variables.
