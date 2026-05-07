import numpy as np


# ---------------------------------------------------------------------
# Legacy toy generator used in the first CoRT-SI debugging version.
# Kept for reference only.
# ---------------------------------------------------------------------
# def generate_coef(p, s, true_beta=0.25, num_info_aux=3, num_uninfo_aux=2, gamma=0.01):
#     K = num_info_aux + num_uninfo_aux
#     beta_0 = np.concatenate([np.full(s, true_beta), np.zeros(p - s)])
#     noisy_prefix = min(p, 50)
#     informative_prefix = min(p, 25)
#
#     Beta_S = np.tile(beta_0, (K, 1)).T
#     if s >= 0:
#         Beta_S[0, :] -= 2 * true_beta
#         for m in range(K):
#             if m < num_uninfo_aux:
#                 Beta_S[:noisy_prefix, m] += np.random.normal(0, true_beta * gamma * 10, noisy_prefix)
#             else:
#                 Beta_S[:informative_prefix, m] += np.random.normal(0, true_beta * gamma, informative_prefix)
#     return Beta_S, beta_0
#
#
# def generate_data(p, s, nS, nT, true_beta=0.25, num_info_aux=3, num_uninfo_aux=2, gamma=0.01):
#     K = num_info_aux + num_uninfo_aux
#
#     Beta_S, beta_0 = generate_coef(p, s, true_beta, num_info_aux, num_uninfo_aux, gamma)
#     Beta = np.column_stack([Beta_S[:, i] for i in range(K)] + [beta_0])
#
#     X_list = []
#     Y_list = []
#     true_Y_list = []
#
#     cov = np.eye(p)
#     N_vec = [nS] * K + [nT]
#
#     for k in range(K + 1):
#         Xk = np.random.multivariate_normal(mean=np.zeros(p), cov=cov, size=N_vec[k])
#         true_Yk = Xk @ Beta[:, k]
#         noise = np.random.normal(0, 1, N_vec[k])
#         Yk = true_Yk + noise
#         X_list.append(Xk)
#         Y_list.append(Yk)
#         true_Y_list.append(true_Yk)
#
#     XS_list = X_list[:-1]
#     YS_list = Y_list[:-1]
#     X0 = X_list[-1]
#     Y0 = Y_list[-1]
#     true_Y = np.concatenate(true_Y_list)
#     SigmaS_list = [np.eye(nS) for _ in range(K)]
#     Sigma0 = np.eye(nT)
#
#     return XS_list, YS_list, X0, Y0, true_Y, SigmaS_list, Sigma0, beta_0


def toeplitz_cov(p, rho=0.0):
    if not 0.0 <= rho < 1.0:
        raise ValueError("rho must be in [0.0, 1.0).")

    idx = np.arange(p)
    return rho ** np.abs(idx[:, None] - idx[None, :])


def _resolve_num_sources(K, num_info_aux=None, num_uninfo_aux=None):
    if num_info_aux is None and num_uninfo_aux is None:
        return K

    return (num_info_aux or 0) + (num_uninfo_aux or 0)


def generate_data(
    p=500, nS=200, nT=75, K=5, h=10, rho=0.0, sigma_noise=1.0,
    source_shift_sd=0.3, covariate_shift=False, seed=None, s=None, true_beta=None,
    num_info_aux=None, num_uninfo_aux=None, gamma=None,
):
    if p <= 0:
        raise ValueError("p must be positive.")
    if nS <= 0 or nT <= 0:
        raise ValueError("nS and nT must be positive.")

    del s, gamma

    K = _resolve_num_sources(K, num_info_aux=num_info_aux, num_uninfo_aux=num_uninfo_aux)
    if K < 0:
        raise ValueError("K must be non-negative.")

    rng = np.random if seed is None else np.random.RandomState(seed)

    base_signal = np.array([0.25, 0.25, 0.25, 0.3, 0.3], dtype=float)
    beta_0 = np.zeros(p, dtype=float)
    q = min(p, len(base_signal))
    beta_0[:q] = base_signal[:q]

    if true_beta is not None:
        beta_0[:q] *= true_beta

    Sigma_x0 = toeplitz_cov(p, rho=rho)

    X0 = rng.multivariate_normal(mean=np.zeros(p), cov=Sigma_x0, size=nT)
    true_Y0 = X0 @ beta_0
    Y0 = true_Y0 + rng.normal(0.0, sigma_noise, size=nT)

    XS_list = []
    YS_list = []
    true_Y_list = []
    SigmaS_list = []

    for k in range(K):
        Sigma_xk = Sigma_x0.copy()
        if covariate_shift:
            eps = rng.normal(0.0, source_shift_sd, size=p)
            Sigma_xk = Sigma_xk + np.outer(eps, eps)

        Xk = rng.multivariate_normal(mean=np.zeros(p), cov=Sigma_xk, size=nS)
        if k < num_info_aux:
            R = rng.choice([-1.0, 1.0], size=p)
            beta_k = beta_0 + (h / p) * R
        else:
            beta_k = np.zeros(p)
            
            idx_shift = np.arange(q, 2 * q)
            idx_random = rng.choice(np.arange(2 * q, p), size=q, replace=False)
            active_indices = np.concatenate([idx_shift, idx_random])
            
            mask = np.zeros(p, dtype=bool)
            mask[active_indices] = True
            
            beta_k[mask] = 0.5 + (2 * h / p) * rng.choice([-1, 1], size=len(active_indices))
            beta_k[~mask] = 2 * h * rng.choice([-1, 1], size=np.sum(~mask))

        true_Yk = Xk @ beta_k
        Yk = true_Yk + rng.normal(0.0, sigma_noise, size=nS)

        XS_list.append(Xk)
        YS_list.append(Yk)
        true_Y_list.append(true_Yk)
        SigmaS_list.append((sigma_noise ** 2) * np.eye(nS))

    true_Y = np.concatenate(true_Y_list + [true_Y0])
    Sigma0 = (sigma_noise ** 2) * np.eye(nT)

    return XS_list, YS_list, X0, Y0, true_Y, SigmaS_list, Sigma0, beta_0


    