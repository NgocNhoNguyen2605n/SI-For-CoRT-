import numpy as np
from scipy.linalg import block_diag
from mpmath import mp
mp.dps = 500


def construct_Sigma(SigmaS_list, Sigma0):
    Sigma = block_diag(*SigmaS_list, Sigma0)
    return Sigma


def construct_betaM_M_SM_Mc(beta_hat):
    M = []
    betaM = []
    SM = []
    Mc = []

    for i, val in enumerate(beta_hat):
        if val != 0.0:
            M.append(i)
            SM.append(np.sign(val))
            betaM.append(val)
        else:
            Mc.append(i)

    SM = np.array(SM).reshape(-1,1)

    return betaM, M, SM, Mc

#_____________________________________________________________________________
def construct_test_statistic(j, X0M, Y, M, nT, N):
    ej = np.zeros(len(M))

    for i, ac in enumerate(M):
        if ac == j:
            ej[i] = 1
            break

    inv = np.linalg.pinv(X0M.T@X0M)
    X0M_inv = X0M @ inv

    _X = np.zeros((N, len(M)))
    _X[N - nT :, :] = X0M_inv
    etaj = _X @ ej
    etajTY = float(etaj @ Y)

    return etaj.reshape(-1, 1), etajTY


def calculate_a_b(etaj, Y, Sigma, N):
    e1 = etaj.T @ Sigma @ etaj
    b = (Sigma @ etaj)/e1

    e2 = np.eye(N) - b @ etaj.T
    a = e2 @ Y

    return a.reshape(-1, 1), b.reshape(-1, 1)


def merge_intervals(intervals, tol=1e-2):
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = []
    for interval in intervals:
        if not merged or interval[0] - merged[-1][1] > tol:
            merged.append(interval)
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], interval[1]))
    return merged


def pivot(intervals, etajTy, etaj, Sigma, tn_mu=0):
    if len(intervals) == 0: return None #
    intervals = merge_intervals(intervals, tol=1e-2)

    etaj = etaj.ravel()
    stdev = np.sqrt(etaj @ (Sigma @ etaj))

    numerator = mp.mpf('0')
    denominator = mp.mpf('0')

    for (left, right) in intervals:
        cdf_left= mp.ncdf((left- tn_mu)/ stdev)
        cdf_right= mp.ncdf((right- tn_mu)/ stdev)
        piece = cdf_right- cdf_left
        denominator += piece

        if etajTy >= right:
            numerator += piece
        elif left <= etajTy < right:
            numerator += mp.ncdf((etajTy - tn_mu)/ stdev) - cdf_left

    if denominator == 0:
        return None
    return float(numerator/ denominator)


def calculate_TN_p_value(intervals, etaj, etajTy, Sigma, tn_mu=0.0):
    cdf = pivot(intervals, etajTy, etaj, Sigma, tn_mu)
    return 2.0 * min(cdf, 1.0 - cdf)


def construct_Y(YS_list, Y0):
    return np.concatenate([np.asarray(y).ravel() for y in YS_list] + [np.asarray(Y0).ravel()])


def construct_Y_tilde(YS_list, Y0, source_set):
    blocks = [(np.asarray(YS_list[source_idx]).ravel() / np.sqrt(YS_list[source_idx].shape[0])) for source_idx in source_set]
    blocks.append(np.asarray(Y0).ravel() / np.sqrt(Y0.shape[0]))
    return np.concatenate(blocks)


def construct_X_tilde(XS_list, X0, source_set):
    p = X0.shape[1]
    num_sources = len(source_set)
    rows = []

    for block_idx, source_idx in enumerate(source_set):
        Xk = XS_list[source_idx]
        nk = Xk.shape[0]
        Xk_scaled = Xk / np.sqrt(nk)
        row = [Xk_scaled if idx == block_idx else np.zeros((nk, p)) for idx in range(num_sources)]
        row.append(Xk_scaled)
        rows.append(np.hstack(row))

    n0 = X0.shape[0]
    row0 = [np.zeros((n0, p)) for _ in range(num_sources)]
    row0.append(X0 / np.sqrt(n0))
    rows.append(np.hstack(row0))
    return np.vstack(rows)


def construct_w_tilde(p, lambda0, lambdak_list, source_set):
    weights = [np.full(p, lambdak_list[source_idx]) for source_idx in source_set]
    weights.append(np.full(p, lambda0))
    return np.concatenate(weights)


def construct_folds(n0, T=5, shuffle=False, random_state=0):
    indices = np.arange(n0)
    if shuffle:
        rng = np.random.default_rng(random_state)
        indices = rng.permutation(indices)
    return [np.array(split, dtype=int) for split in np.array_split(indices, T)]


def construct_block_slices(ns_list, n0):
    slices = []
    start = 0
    for nk in ns_list:
        stop = start + nk
        slices.append(slice(start, stop))
        start = stop
    slices.append(slice(start, start + n0))
    return slices


def split_stacked_response(y, ns_list, n0):
    y = np.asarray(y)
    block_slices = construct_block_slices(ns_list, n0)
    source_blocks = [np.asarray(y[block_slices[idx]]).reshape(-1) for idx in range(len(ns_list))]
    target_block = np.asarray(y[block_slices[-1]]).reshape(-1)
    return source_blocks, target_block


def complement_fold_indices(n0, fold_indices):
    mask = np.ones(n0, dtype=bool)
    mask[np.asarray(fold_indices, dtype=int)] = False
    return np.flatnonzero(mask)


def solve_linear_inequalities_1d(psi, gamma, tol=1e-12):
    psi = np.asarray(psi, dtype=float).ravel()
    gamma = np.asarray(gamma, dtype=float).ravel()

    left = -np.inf
    right = np.inf

    for coeff, bound in zip(psi, gamma):
        if abs(coeff) <= tol:
            if bound < -tol:
                return []
            continue

        value = bound / coeff
        if coeff > 0:
            right = min(right, value)
        else:
            left = max(left, value)

    if right + tol < left:
        return []
    return [(left, right)]


def solve_quadratic_leq(A, B, C, tol=1e-12):
    A = float(A)
    B = float(B)
    C = float(C)

    if abs(A) <= tol:
        return solve_linear_inequalities_1d([B], [-C], tol=tol)

    discriminant = (B * B) - (4.0 * A * C)
    if discriminant < -tol:
        return [(-np.inf, np.inf)] if A < 0 else []

    if abs(discriminant) <= tol:
        root = -B / (2.0 * A)
        return [(-np.inf, np.inf)] if A < 0 else [(root, root)]

    sqrt_discriminant = np.sqrt(discriminant)
    root1 = (-B - sqrt_discriminant) / (2.0 * A)
    root2 = (-B + sqrt_discriminant) / (2.0 * A)
    left, right = min(root1, root2), max(root1, root2)

    if A > 0:
        return [(left, right)]
    return [(-np.inf, left), (right, np.inf)]


def intersect_interval_unions(intervals_a, intervals_b, tol=1e-8):
    if not intervals_a or not intervals_b:
        return []

    intervals_a = merge_intervals(list(intervals_a), tol=tol)
    intervals_b = merge_intervals(list(intervals_b), tol=tol)

    result = []
    idx_a = 0
    idx_b = 0

    while idx_a < len(intervals_a) and idx_b < len(intervals_b):
        left = max(intervals_a[idx_a][0], intervals_b[idx_b][0])
        right = min(intervals_a[idx_a][1], intervals_b[idx_b][1])
        if left <= right + tol:
            result.append((left, right))

        if intervals_a[idx_a][1] < intervals_b[idx_b][1] - tol:
            idx_a += 1
        else:
            idx_b += 1

    return merge_intervals(result, tol=tol)


def union_interval_unions(intervals_a, intervals_b, tol=1e-8):
    if not intervals_a:
        return merge_intervals(list(intervals_b), tol=tol) if intervals_b else []
    if not intervals_b:
        return merge_intervals(list(intervals_a), tol=tol)
    return merge_intervals(list(intervals_a) + list(intervals_b), tol=tol)


def clip_interval_union(intervals, left, right, tol=1e-8):
    return intersect_interval_unions(intervals, [(left, right)], tol=tol)


def point_in_interval_union(value, intervals, tol=1e-10):
    for left, right in intervals:
        if left - tol <= value <= right + tol:
            return True
    return False


def count_region_from_fold_wins(win_regions, m, mode, z_min=None, z_max=None, tol=1e-8):
    if mode not in {"selected", "discarded"}:
        raise ValueError("mode must be 'selected' or 'discarded'")

    endpoints = []
    if z_min is not None and z_max is not None:
        endpoints.extend([float(z_min), float(z_max)])

    for intervals in win_regions:
        for left, right in intervals:
            if np.isfinite(left):
                endpoints.append(float(left))
            if np.isfinite(right):
                endpoints.append(float(right))

    if not endpoints:
        return [] if mode == "selected" else [(-np.inf, np.inf)]

    endpoints = sorted(set(endpoints))
    if len(endpoints) == 1:
        return [(endpoints[0], endpoints[0])] if mode == "discarded" else []

    kept = []
    for left, right in zip(endpoints[:-1], endpoints[1:]):
        if right < left + tol:
            continue
        midpoint = 0.5 * (left + right)
        wins = sum(point_in_interval_union(midpoint, intervals, tol=tol) for intervals in win_regions)

        if mode == "selected" and wins >= m:
            kept.append((left, right))
        if mode == "discarded" and wins <= m - 1:
            kept.append((left, right))

    return merge_intervals(kept, tol=tol)
