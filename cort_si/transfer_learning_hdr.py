from skglm import WeightedLasso, Lasso
import numpy as np
import warnings
try:
    from . import utils
except ImportError:  # pragma: no cover - fallback for direct script execution
    import utils
warnings.filterwarnings("ignore")


def solve_lasso(X, y, lam, fit_intercept=False, tol=1e-10, max_iter=10_000):
    model = Lasso(alpha=lam, fit_intercept=fit_intercept, tol=tol, max_iter=max_iter)
    model.fit(X, np.asarray(y).ravel())
    return model.coef_


def solve_cort_model(X0, Y0, XS_list, YS_list, source_set, lambda0, lambdak_list, tol=1e-10):
    X_tilde = utils.construct_X_tilde(XS_list, X0, source_set)
    Y_tilde = utils.construct_Y_tilde(YS_list, Y0, source_set)
    w_tilde = utils.construct_w_tilde(X0.shape[1], lambda0, lambdak_list, source_set)

    weighted_lasso = WeightedLasso(
        alpha=1.0 / X_tilde.shape[0],
        fit_intercept=False,
        tol=tol,
        weights=w_tilde,
    )
    weighted_lasso.fit(X_tilde, Y_tilde)
    theta_hat = weighted_lasso.coef_
    beta0_hat = theta_hat[-X0.shape[1]:]
    return theta_hat, beta0_hat, X_tilde, w_tilde


def adaptive_source_selection(X0, Y0, XS_list, YS_list, folds, lambda_sel):
    n0 = Y0.shape[0]
    if folds is None:
        folds = utils.construct_folds(n0, T=5, shuffle=False)

    if len(folds) % 2 == 0:
        raise ValueError("The number of folds T must be odd for majority voting")

    selected_sources = []
    majority = (len(folds) + 1) // 2

    for source_idx, (Xk, Yk) in enumerate(zip(XS_list, YS_list)):
        source_votes = []

        for fold_indices in folds:
            train_indices = utils.complement_fold_indices(n0, fold_indices)

            X0_train = X0[train_indices]
            Y0_train = Y0[train_indices]
            X0_valid = X0[fold_indices]
            Y0_valid = Y0[fold_indices]

            beta_target = solve_lasso(X0_train, Y0_train, lambda_sel)
            X_aug_train = np.vstack([Xk, X0_train])
            Y_aug_train = np.concatenate([Yk, Y0_train])
            beta_aug = solve_lasso(X_aug_train, Y_aug_train, lambda_sel)

            loss_target = 0.5 * np.mean((Y0_valid - X0_valid @ beta_target) ** 2)
            loss_aug = 0.5 * np.mean((Y0_valid - X0_valid @ beta_aug) ** 2)
            source_votes.append(loss_aug <= loss_target)

        if int(np.sum(source_votes)) >= majority:
            selected_sources.append(source_idx)

    return selected_sources