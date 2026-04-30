import random

try:
    from . import sub_prob
    from . import transfer_learning_hdr
    from . import utils
except ImportError:  # pragma: no cover - fallback for direct script execution
    import sub_prob
    import transfer_learning_hdr
    import utils


def SI(X0, Y0, XS_list, YS_list, lambda_sel, lambda0, lambdak_list, SigmaS_list, Sigma0, folds=None, T=5, z_min=-20, z_max=20):
    if len(XS_list) != len(YS_list):
        raise ValueError("XS_list and YS_list must have the same length")
    if len(lambdak_list) != len(XS_list):
        raise ValueError("lambdak_list must have the same length as XS_list")

    if folds is None:
        folds = utils.construct_folds(X0.shape[0], T=T, shuffle=False)
    if len(folds) % 2 == 0:
        raise ValueError("The number of folds T must be odd")

    I_obs = transfer_learning_hdr.adaptive_source_selection(X0, Y0, XS_list, YS_list, folds, lambda_sel)
    _, beta0_hat, _, _ = transfer_learning_hdr.solve_cort_model(X0, Y0, XS_list, YS_list, I_obs, lambda0, lambdak_list)
    M_obs = [idx for idx, value in enumerate(beta0_hat) if abs(value) > 1e-10]

    if not M_obs:
        return None

    Y_obs = utils.construct_Y(YS_list, Y0)
    Sigma = utils.construct_Sigma(SigmaS_list, Sigma0)
    X0M = X0[:, M_obs]
    p_values = []
    for j in M_obs:
        etaj, etajTy = utils.construct_test_statistic(j, X0M, Y_obs, M_obs, Y0.shape[0], Y_obs.shape[0])
        a, b = utils.calculate_a_b(etaj, Y_obs, Sigma, Y_obs.shape[0])

        intervals_source = sub_prob.source_selection_region(X0, Y0, XS_list, YS_list, a, b, folds, I_obs, lambda_sel, z_min, z_max)
        intervals_model = sub_prob.model_selection_region(
            X0, Y0, XS_list, YS_list, a, b,
            I_obs, M_obs, lambda0, lambdak_list, z_min, z_max,
        )
        intervals = utils.intersect_interval_unions(intervals_source, intervals_model)
        p_sel = None
        if intervals:
            p_sel = utils.calculate_TN_p_value(intervals, etaj, etajTy, Sigma, 0.0)
        p_values.append((j, p_sel))

    return p_values


def SI_randj(X0, Y0, XS_list, YS_list, lambda_sel, lambda0, lambdak_list, SigmaS_list, Sigma0, folds=None, T=5, z_min=-20, z_max=20):
    if len(XS_list) != len(YS_list):
        raise ValueError("XS_list and YS_list must have the same length")
    if len(lambdak_list) != len(XS_list):
        raise ValueError("lambdak_list must have the same length as XS_list")

    if folds is None:
        folds = utils.construct_folds(X0.shape[0], T=T, shuffle=False)
    if len(folds) % 2 == 0:
        raise ValueError("The number of folds T must be odd")

    I_obs = transfer_learning_hdr.adaptive_source_selection(X0, Y0, XS_list, YS_list, folds, lambda_sel)
    _, beta0_hat, _, _ = transfer_learning_hdr.solve_cort_model(X0, Y0, XS_list, YS_list, I_obs, lambda0, lambdak_list)
    M_obs = [idx for idx, value in enumerate(beta0_hat) if abs(value) > 1e-10]

    if not M_obs:
        return None

    Y_obs = utils.construct_Y(YS_list, Y0)
    Sigma = utils.construct_Sigma(SigmaS_list, Sigma0)
    X0M = X0[:, M_obs]
    j = random.choice(M_obs)

    etaj, etajTy = utils.construct_test_statistic(j, X0M, Y_obs, M_obs, Y0.shape[0], Y_obs.shape[0])
    a, b = utils.calculate_a_b(etaj, Y_obs, Sigma, Y_obs.shape[0])

    intervals_source = sub_prob.source_selection_region(X0, Y0, XS_list, YS_list, a, b, folds, I_obs, lambda_sel, z_min, z_max)
    intervals_model = sub_prob.model_selection_region(
        X0, Y0, XS_list, YS_list, a, b,
        I_obs, M_obs, lambda0, lambdak_list, z_min, z_max,
    )
    intervals = utils.intersect_interval_unions(intervals_source, intervals_model)
    if not intervals:
        return None

    p_sel = utils.calculate_TN_p_value(intervals, etaj, etajTy, Sigma, 0.0)
    return p_sel