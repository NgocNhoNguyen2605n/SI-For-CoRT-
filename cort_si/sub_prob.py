import numpy as np
from numpy.linalg import pinv

try:
    from . import utils
    from . import transfer_learning_hdr
except ImportError:
    import utils
    import transfer_learning_hdr


def lasso_state_interval(X_train, a_train, b_train, active_set, sign_vec, lambda_sel, n_samples):
    a_train = np.asarray(a_train, dtype=float).reshape(-1, 1)
    b_train = np.asarray(b_train, dtype=float).reshape(-1, 1)
    active_set = list(active_set)
    inactive_set = [idx for idx in range(X_train.shape[1]) if idx not in active_set]
    sign_vec = np.asarray(sign_vec, dtype=float).reshape(-1, 1)

    c_full = np.zeros((X_train.shape[1], 1))
    d_full = np.zeros((X_train.shape[1], 1))
    psi0 = np.array([])
    gamma0 = np.array([])
    psi1 = np.array([])
    gamma1 = np.array([])

    if active_set:
        X_active = X_train[:, active_set]
        gram_inv = pinv(X_active.T @ X_active)
        X_active_plus = gram_inv @ X_active.T
        embed_active = np.eye(X_train.shape[1])[:, active_set]

        c_active = gram_inv @ (X_active.T @ a_train - (n_samples * lambda_sel) * sign_vec)
        d_active = gram_inv @ (X_active.T @ b_train)
        c_full = embed_active @ c_active
        d_full = embed_active @ d_active

        psi0 = (-sign_vec * (X_active_plus @ b_train)).ravel()
        gamma0 = (sign_vec * (X_active_plus @ a_train - (n_samples * lambda_sel) * (gram_inv @ sign_vec))).ravel()
        projection = np.eye(X_train.shape[0]) - X_active @ X_active_plus
    else:
        X_active = np.zeros((X_train.shape[0], 0))
        gram_inv = np.zeros((0, 0))
        projection = np.eye(X_train.shape[0])

    if inactive_set:
        X_inactive = X_train[:, inactive_set]
        inactive_term = X_inactive.T @ projection
        term_b = inactive_term @ b_train
        psi1 = np.concatenate([
            (term_b / (n_samples * lambda_sel)).ravel(),
            (-term_b / (n_samples * lambda_sel)).ravel(),
        ])

        if active_set:
            coupling = X_inactive.T @ X_active @ gram_inv @ sign_vec
        else:
            coupling = np.zeros((len(inactive_set), 1))

        term_a = inactive_term @ a_train
        gamma1 = np.concatenate([
            (np.ones_like(term_a) - coupling - (term_a / (n_samples * lambda_sel))).ravel(),
            (np.ones_like(term_a) + coupling + (term_a / (n_samples * lambda_sel))).ravel(),
        ])

    psi = np.concatenate((psi0, psi1))
    gamma = np.concatenate((gamma0, gamma1))
    interval = utils.solve_linear_inequalities_1d(psi, gamma)
    return psi, gamma, c_full, d_full, interval


def target_fold_state_interval(X0_train, a0_train, b0_train, active_set, sign_vec, lambda_sel, n_train):
    return lasso_state_interval(X0_train, a0_train, b0_train, active_set, sign_vec, lambda_sel, n_train)


def augmented_fold_state_interval(X_aug_train, a_aug_train, b_aug_train, active_set, sign_vec, lambda_sel, n_aug):
    return lasso_state_interval(X_aug_train, a_aug_train, b_aug_train, active_set, sign_vec, lambda_sel, n_aug)


def validation_quadratic(X0_val, a0_val, b0_val, c0, d0, ck, dk, n_val):
    a0_val = np.asarray(a0_val, dtype=float).reshape(-1, 1)
    b0_val = np.asarray(b0_val, dtype=float).reshape(-1, 1)
    c0 = np.asarray(c0, dtype=float).reshape(-1, 1)
    d0 = np.asarray(d0, dtype=float).reshape(-1, 1)
    ck = np.asarray(ck, dtype=float).reshape(-1, 1)
    dk = np.asarray(dk, dtype=float).reshape(-1, 1)

    residual0_a = a0_val - X0_val @ c0
    residual0_b = b0_val - X0_val @ d0
    residualk_a = a0_val - X0_val @ ck
    residualk_b = b0_val - X0_val @ dk

    A0 = 0.5 * float(residual0_b.T @ residual0_b) / n_val
    B0 = float(residual0_a.T @ residual0_b) / n_val
    C0 = 0.5 * float(residual0_a.T @ residual0_a) / n_val

    Ak = 0.5 * float(residualk_b.T @ residualk_b) / n_val
    Bk = float(residualk_a.T @ residualk_b) / n_val
    Ck = 0.5 * float(residualk_a.T @ residualk_a) / n_val

    return Ak - A0, Bk - B0, Ck - C0


def kkt_interval(X_tilde, a_adapt, b_adapt, theta_hat, w_tilde, p, tol=1e-10):
    a_adapt = np.asarray(a_adapt, dtype=float).reshape(-1, 1)
    b_adapt = np.asarray(b_adapt, dtype=float).reshape(-1, 1)
    theta_hat = np.asarray(theta_hat, dtype=float).ravel()
    w_tilde = np.asarray(w_tilde, dtype=float).reshape(-1, 1)

    active_set = [idx for idx, value in enumerate(theta_hat) if abs(value) > tol]
    inactive_set = [idx for idx in range(X_tilde.shape[1]) if idx not in active_set]
    sign_vec = np.sign(theta_hat[active_set]).reshape(-1, 1) if active_set else np.zeros((0, 1))

    psi0 = np.array([])
    gamma0 = np.array([])
    psi1 = np.array([])
    gamma1 = np.array([])

    if active_set:
        X_active = X_tilde[:, active_set]
        gram_inv = pinv(X_active.T @ X_active)
        X_active_plus = gram_inv @ X_active.T
        weighted_sign = w_tilde[active_set] * sign_vec

        psi0 = (-sign_vec * (X_active_plus @ b_adapt)).ravel()
        gamma0 = (sign_vec * (X_active_plus @ a_adapt) - sign_vec * (gram_inv @ weighted_sign)).ravel()
        projection = np.eye(X_tilde.shape[0]) - X_active @ X_active_plus
    else:
        X_active = np.zeros((X_tilde.shape[0], 0))
        gram_inv = np.zeros((0, 0))
        weighted_sign = np.zeros((0, 1))
        projection = np.eye(X_tilde.shape[0])

    if inactive_set:
        X_inactive = X_tilde[:, inactive_set]
        inactive_term = X_inactive.T @ projection
        term_b = inactive_term @ b_adapt
        psi1 = np.concatenate([term_b.ravel(), -term_b.ravel()])

        if active_set:
            coupling = X_inactive.T @ X_active @ gram_inv @ weighted_sign
        else:
            coupling = np.zeros((len(inactive_set), 1))

        term_a = inactive_term @ a_adapt
        gamma1 = np.concatenate([
            (w_tilde[inactive_set] - coupling - term_a).ravel(),
            (w_tilde[inactive_set] + coupling + term_a).ravel(),
        ])

    psi = np.concatenate((psi0, psi1))
    gamma = np.concatenate((gamma0, gamma1))
    interval = utils.solve_linear_inequalities_1d(psi, gamma)

    target_start = X_tilde.shape[1] - p
    target_active = [idx - target_start for idx in active_set if idx >= target_start]
    return interval, target_active


def fold_win_region(source_idx, fold_idx, X0, Y0, XS_list, YS_list, a, b, folds, lambda_sel, z_min, z_max, eps=1e-5, tol=1e-10):
    ns_list = [ys.shape[0] for ys in YS_list]
    source_a_blocks, target_a = utils.split_stacked_response(a, ns_list, Y0.shape[0])
    source_b_blocks, target_b = utils.split_stacked_response(b, ns_list, Y0.shape[0])

    fold_indices = np.asarray(folds[fold_idx], dtype=int)
    train_indices = utils.complement_fold_indices(Y0.shape[0], fold_indices)
    X0_train = X0[train_indices]
    X0_val = X0[fold_indices]

    a0_train = target_a[train_indices]
    b0_train = target_b[train_indices]
    a0_val = target_a[fold_indices]
    b0_val = target_b[fold_indices]

    X_source = XS_list[source_idx]
    a_source = source_a_blocks[source_idx]
    b_source = source_b_blocks[source_idx]
    X_aug_train = np.vstack([X_source, X0_train])
    a_aug_train = np.concatenate([a_source, a0_train])
    b_aug_train = np.concatenate([b_source, b0_train])

    intervals = []
    z = z_min
    while z < z_max:
        y0_train = a0_train + (b0_train * z)
        beta_target = transfer_learning_hdr.solve_lasso(X0_train, y0_train, lambda_sel)
        _, active_target, sign_target, _ = utils.construct_betaM_M_SM_Mc(beta_target)
        _, _, c0, d0, target_interval = target_fold_state_interval(
            X0_train,
            a0_train,
            b0_train,
            active_target,
            np.asarray(sign_target).ravel(),
            lambda_sel,
            len(train_indices),
        )

        y_aug_train = a_aug_train + (b_aug_train * z)
        beta_aug = transfer_learning_hdr.solve_lasso(X_aug_train, y_aug_train, lambda_sel)
        _, active_aug, sign_aug, _ = utils.construct_betaM_M_SM_Mc(beta_aug)
        _, _, ck, dk, aug_interval = augmented_fold_state_interval(
            X_aug_train,
            a_aug_train,
            b_aug_train,
            active_aug,
            np.asarray(sign_aug).ravel(),
            lambda_sel,
            X_aug_train.shape[0],
        )

        linear_region = utils.intersect_interval_unions(target_interval, aug_interval, tol=tol)
        linear_region = utils.clip_interval_union(linear_region, z, z_max, tol=tol)
        if not linear_region:
            z += eps
            continue

        Atilde, Btilde, Ctilde = validation_quadratic(X0_val, a0_val, b0_val, c0, d0, ck, dk, len(fold_indices))
        quad_region = utils.solve_quadratic_leq(Atilde, Btilde, Ctilde)
        local_region = utils.intersect_interval_unions(linear_region, quad_region, tol=tol)
        local_region = utils.clip_interval_union(local_region, z_min, z_max, tol=tol)
        intervals = utils.union_interval_unions(intervals, local_region, tol=tol)

        right_endpoint = linear_region[-1][1]
        if not np.isfinite(right_endpoint):
            break
        if right_endpoint <= z + tol:
            z += eps
        else:
            z = right_endpoint + eps

    return utils.clip_interval_union(intervals, z_min, z_max, tol=tol)


def source_selection_region(X0, Y0, XS_list, YS_list, a, b, folds, I_obs, lambda_sel, z_min, z_max, eps=1e-5, tol=1e-10):
    total_region = [(z_min, z_max)]
    majority = (len(folds) + 1) // 2
    selected_sources = set(I_obs)

    for source_idx in range(len(XS_list)):
        win_regions = [
            fold_win_region(
                source_idx,
                fold_idx,
                X0,
                Y0,
                XS_list,
                YS_list,
                a,
                b,
                folds,
                lambda_sel,
                z_min,
                z_max,
                eps=eps,
                tol=tol,
            )
            for fold_idx in range(len(folds))
        ]

        mode = "selected" if source_idx in selected_sources else "discarded"
        source_region = utils.count_region_from_fold_wins(win_regions, majority, mode, z_min=z_min, z_max=z_max, tol=tol)
        total_region = utils.intersect_interval_unions(total_region, source_region, tol=tol)
        if not total_region:
            return []

    return total_region


def model_selection_region(X0, Y0, XS_list, YS_list, a, b, I_obs, M_obs, lambda0, lambdak_list, z_min, z_max, eps=1e-5, tol=1e-10):
    ns_list = [ys.shape[0] for ys in YS_list]
    source_a_blocks, target_a = utils.split_stacked_response(a, ns_list, Y0.shape[0])
    source_b_blocks, target_b = utils.split_stacked_response(b, ns_list, Y0.shape[0])

    selected_source_set = list(I_obs)
    source_a_selected = [source_a_blocks[idx] for idx in selected_source_set]
    source_b_selected = [source_b_blocks[idx] for idx in selected_source_set]

    a_adapt = np.concatenate([
        np.asarray(source_a_selected[idx]).reshape(-1) / np.sqrt(source_a_selected[idx].shape[0])
        for idx in range(len(source_a_selected))
    ] + [target_a / np.sqrt(Y0.shape[0])])
    b_adapt = np.concatenate([
        np.asarray(source_b_selected[idx]).reshape(-1) / np.sqrt(source_b_selected[idx].shape[0])
        for idx in range(len(source_b_selected))
    ] + [target_b / np.sqrt(Y0.shape[0])])

    intervals = []
    z = z_min
    while z < z_max:
        Y0_z = target_a + (target_b * z)
        YS_z = [source_a_blocks[idx] + (source_b_blocks[idx] * z) for idx in range(len(XS_list))]
        theta_hat, beta0_hat, X_tilde, w_tilde = transfer_learning_hdr.solve_cort_model(
            X0,
            Y0_z,
            XS_list,
            YS_z,
            selected_source_set,
            lambda0,
            lambdak_list,
        )

        local_interval, target_active = kkt_interval(
            X_tilde,
            a_adapt,
            b_adapt,
            theta_hat,
            w_tilde,
            p=X0.shape[1],
            tol=tol,
        )

        local_interval = utils.clip_interval_union(local_interval, z, z_max, tol=tol)
        if not local_interval:
            z += eps
            continue

        if list(target_active) == list(M_obs):
            intervals = utils.union_interval_unions(intervals, local_interval, tol=tol)

        right_endpoint = local_interval[-1][1]
        if not np.isfinite(right_endpoint):
            break
        if right_endpoint <= z + tol:
            z += eps
        else:
            z = right_endpoint + eps

    return utils.clip_interval_union(intervals, z_min, z_max, tol=tol)
