import numpy as np

def generate_coef(p, s, true_beta=0.25, num_info_aux=3, num_uninfo_aux=2, gamma=0.01):
    K = num_info_aux + num_uninfo_aux
    beta_0 = np.concatenate([np.full(s, true_beta), np.zeros(p - s)])
    noisy_prefix = min(p, 50)
    informative_prefix = min(p, 25)

    Beta_S = np.tile(beta_0, (K, 1)).T
    if s >= 0:
        Beta_S[0, :] -= 2 * true_beta
        for m in range(K):
            if m < num_uninfo_aux:
                Beta_S[:noisy_prefix, m] += np.random.normal(0, true_beta * gamma * 10, noisy_prefix)
            else:
                Beta_S[:informative_prefix, m] += np.random.normal(0, true_beta * gamma, informative_prefix)
    return Beta_S, beta_0


def generate_data(p, s, nS, nT, true_beta=0.25, num_info_aux=3, num_uninfo_aux=2, gamma=0.01):
    K = num_info_aux + num_uninfo_aux

    Beta_S, beta_0 = generate_coef(p, s, true_beta, num_info_aux, num_uninfo_aux, gamma)
    Beta = np.column_stack([Beta_S[:, i] for i in range(K)] + [beta_0])

    X_list = []
    Y_list = []
    true_Y_list = []

    cov = np.eye(p)
    N_vec = [nS] * K + [nT]

    for k in range(K+1):
        Xk = np.random.multivariate_normal(mean=np.zeros(p), cov=cov, size=N_vec[k])
        true_Yk = Xk @ Beta[:, k]
        noise = np.random.normal(0, 1, N_vec[k])
        Yk = true_Yk + noise
        X_list.append(Xk)
        Y_list.append(Yk)
        true_Y_list.append(true_Yk)
    
    XS_list = X_list[:-1]
    YS_list = Y_list[:-1]
    X0 = X_list[-1]
    Y0 = Y_list[-1]
    true_Y = np.concatenate(true_Y_list)
    SigmaS_list = [np.eye(nS) for _ in range(K)]
    Sigma0 = np.eye(nT)

    return XS_list, YS_list, X0, Y0, true_Y, SigmaS_list, Sigma0, beta_0