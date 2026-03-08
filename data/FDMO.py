
# FDMO helpers
def _project_to_simplex(v: np.ndarray) -> np.ndarray:
    v = v.ravel()
    n = v.size
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1.0) / float(rho + 1)
    return np.maximum(v - theta, 0.0)

def _fdmo_obj_and_grad(n_vals: np.ndarray, w: np.ndarray, lam: float):
    v = w * n_vals
    m = n_vals.size
    sum_v = v.sum()
    diff_term = m * (v**2).sum() - sum_v**2
    grad_diff = 2.0 * m * v * n_vals - 2.0 * sum_v * n_vals

    smooth = 0.0
    grad_smooth = np.zeros_like(w)
    if m >= 2:
        dif = np.diff(w)
        smooth = np.sum(dif**2)
        grad_smooth[0] = 2.0 * (w[0] - w[1])
        grad_smooth[1:-1] = 2.0 * (2*w[1:-1] - w[:-2] - w[2:])
        grad_smooth[-1] = 2.0 * (w[-1] - w[-2])

    return diff_term - lam * smooth, grad_diff - lam * grad_smooth

def fdmo_optimize(n_vals: np.ndarray, lam: float = 0.1,
                  max_iter: int = 400, lr: float = 0.2, tol: float = 1e-7) -> np.ndarray:
    n_vals = np.asarray(n_vals, dtype=float)
    if n_vals.size == 0:
        return np.array([])
    if np.allclose(n_vals, n_vals[0]):
        return np.ones_like(n_vals) / n_vals.size

    w = np.ones_like(n_vals) / n_vals.size
    last_obj = -np.inf
    for _ in range(max_iter):
        obj, grad = _fdmo_obj_and_grad(n_vals, w, lam)
        w = _project_to_simplex(w + lr * grad)
        if abs(obj - last_obj) < tol:
            break
        lr *= 1.02 if obj > last_obj else 0.5
        lr = float(np.clip(lr, 1e-4, 1.0))
        last_obj = obj
    return w