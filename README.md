Network Models in Stan
---

The following models are implemented in this repository:

- AME (additive and multiplicative effects model)
- LDM (latent distance model)
- SBM (stochastic block model)
- SRM (social relations model)


### Library Dependencies

The R packages used in this repository can be found with `grep`.

```bash
$ grep --include '*.R' -hr '^library' . | sed -r 's/^library\(([^,]+).*\)/\1/' | sort | uniq
```
