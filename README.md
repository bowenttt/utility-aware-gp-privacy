GP privacy helper

This small script demonstrates how to:

- generate n random 2D points
- compute an RBF covariance matrix K_xx
- build A = K_xx + v * I
- create a low-rank PSD matrix M_S via L @ L.T (rank < n)
- create a full-rank PSD matrix M_T via R @ R.T

Run:

```bash
python3 gp_privacy.py
```

Adjust parameters (n, lengthscale, noise_variance, low_rank) in the `__main__` block.

**This project was developed under the guidance of Professor Rui Tuo at Texas A&M University.

**link: https://engineering.tamu.edu/industrial/profiles/tuo-rio.html
