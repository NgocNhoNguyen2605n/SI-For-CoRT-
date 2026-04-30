# Adaptive CoRT-SI
This repository now focuses on selective inference for Adaptive CoRT in high-dimensional regression.

The main entry points are exported from `cort_si.CORT_SI` and from the package root:

```python
from cort_si import SI, SI_randj
```

`SI(...)` returns a list of `(feature_index, selective_p_value)` pairs for the selected target features.
`SI_randj(...)` returns one selective p-value for a randomly chosen selected target feature.


## Requirements & Installation
This package has the following requirements:
- **[numpy](https://numpy.org/doc/stable/)**
- **[mpmath](https://mpmath.org/)** 
- **[skglm](https://contrib.scikit-learn.org/skglm/)**
- **[scipy](https://docs.scipy.org/doc/)** 
- **[matplotlib](https://matplotlib.org/)** 
- **[statsmodels](https://www.statsmodels.org/stable/index.html)**


This package can be installed using pip:
```bash
pip install cort_si
```

We recommend to install or update anaconda to the latest version and use Python 3 (We used Python 3.11.4).

## Adaptive CoRT-SI quick start

```python
import numpy as np
from cort_si import SI, gen_data

np.random.seed(0)
XS_list, YS_list, X0, Y0, _, SigmaS_list, Sigma0, _ = gen_data.generate_data(
  p=5, s=2, nS=6, nT=7, true_beta=1.0, num_info_aux=1, num_uninfo_aux=1, gamma=0.05)

p_values = SI(
  X0, Y0, XS_list, YS_list,
  lambda_sel=0.05, lambda0=0.05, lambdak_list=[0.05] * len(XS_list),
  SigmaS_list=SigmaS_list, Sigma0=Sigma0, T=3, z_min=-5, z_max=5)

print(p_values)
```

## Example
The `examples/` directory contains one minimal runnable example for the Adaptive CoRT-SI pipeline:

- `adaptive_cort_si.py`


## Status
This codebase contains only the Adaptive CoRT-SI implementation and supporting utilities.