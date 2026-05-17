"""
gp_privacy.py

Generates:
- n random 2D points
- RBF covariance matrix K_xx
- A = K_xx + v * I (observation noise variance v)
- M_S: low-rank PSD matrix via L @ L.T with rank < n
- M_T: full-rank PSD matrix via R @ R.T

Simple example usage included under __main__.
"""

import numpy as np
try:
    import cvxpy as cp
except Exception:
    cp = None

#N个坐标点
def generate_random_points(n, low=0.0, high=1.0, seed=None):
    rng = np.random.default_rng(seed)
    return rng.uniform(low=low, high=high, size=(n, 2))

#点之间距离关系
def rbf_kernel(X, Y=None, lengthscale=1.0, variance=1.0):
    X = np.asarray(X)
    if Y is None:
        Y = X
    else:
        Y = np.asarray(Y)
    # 计算 X 和 Y 之间的成对平方距离
    sq_dist = np.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=-1)
    K = variance * np.exp(-0.5 * sq_dist / (lengthscale ** 2))
    return K

#生成矩阵A = K + v * I
def construct_A(K, noise_variance):
    n = K.shape[0]
    return K + noise_variance * np.eye(n)



def solve_for_W_cvxpy(A, M_S, M_T, epsilon_prime, reg=1e-6, small=1e-7, solver=cp.SCS if cp is not None else None):
    """Form and solve the CVXPY problem described by the user.

    Minimizes: trace(M_S_clean @ W) + reg * ||W||_F
    s.t. W >> 0, inv(A) - W >> 0, trace(M_T_clean @ W) >= epsilon_prime

    Returns: (W_opt, prob) where W_opt is the numpy array of the solution (or None).
    """
    if cp is None:
        raise ImportError("cvxpy is not installed. Install with: pip install cvxpy")

    n = A.shape[0]
    # Clean matrices to avoid tiny negative eigenvalues
    M_S_clean = M_S + small * np.eye(n)
    M_T_clean = M_T + small * np.eye(n)

    # Compute inverse of A (user expected inv_A = np.linalg.inv(A))
    inv_A = np.linalg.inv(A)
    inv_A_clean = inv_A + small * np.eye(n)

    W = cp.Variable((n, n), symmetric=True)

    # Enforce a stricter lower bound on W to avoid near-singularity
    w_min_bound = 1e-4
    constraints = [
        W >> (w_min_bound * np.eye(n)),
        cp.Constant(inv_A_clean) - W >> 0,
        cp.trace(cp.Constant(M_T_clean) @ W) >= float(epsilon_prime),
    ]

    objective = cp.Minimize(cp.trace(cp.Constant(M_S_clean) @ W) + float(reg) * cp.norm(W, 'fro'))
    prob = cp.Problem(objective, constraints)

    # Prefer a high-precision interior-point solver (CVXOPT), fall back to CLARABEL, then to provided solver
    solved = False
    if cp is not None:
        preferred = []
        if hasattr(cp, 'CVXOPT'):
            preferred.append(cp.CVXOPT)
        if hasattr(cp, 'CLARABEL'):
            preferred.append(cp.CLARABEL)

        for s in preferred:
            try:
                prob.solve(solver=s)
                solved = True
                break
            except Exception:
                continue

    if not solved:
        try:
            if solver is not None:
                prob.solve(solver=solver)
            else:
                prob.solve()
        except Exception:
            prob.solve()

    print("CVX status:", prob.status)
    try:
        print("Optimal objective value:", prob.value)
    except Exception:
        print("Objective unavailable")

    W_opt = None
    try:
        W_opt = W.value
    except Exception:
        W_opt = None

    return W_opt, prob


if __name__ == "__main__":
    # Example parameters
    n = 50
    seed = 42
    lengthscale = 0.2
    variance = 1.0
    noise_variance = 1e-2

    # 1) Generate n random 2D points (Observation points X)
    X = generate_random_points(n, seed=seed)

    # 2) Compute RBF covariance K_xx
    K_xx = rbf_kernel(X, lengthscale=lengthscale, variance=variance)

    # 3) Construct A = K_xx + v * I
    A = construct_A(K_xx, noise_variance)

    # ---------------------------------------------------------
    # NEW CODE: 4) Spatial Region Selection & Matrix Generation
    # ---------------------------------------------------------
    rng = np.random.default_rng(seed + 1)
    
    # 4.1 Define Sensitive Region S ([0.4, 0.6]x[0.4, 0.6], 10 points)
    S_points = rng.uniform(low=0.4, high=0.6, size=(10, 2))
    
    # 4.2 Define Task Region T (Global 10x10 grid, 100 points)
    grid_x, grid_y = np.mgrid[0:1:10j, 0:1:10j]
    T_points = np.vstack((grid_x.flatten(), grid_y.flatten())).T

    # 4.3 Compute Cross-Covariance Matrices
    K_XS = rbf_kernel(X, S_points, lengthscale=lengthscale, variance=variance)
    K_SX = K_XS.T
    
    K_XT = rbf_kernel(X, T_points, lengthscale=lengthscale, variance=variance)
    K_TX = K_XT.T

    # 4.4 Deterministically Generate M_S and M_T
    M_S = K_XS @ K_SX
    M_T = K_XT @ K_TX
    
    # ---------------------------------------------------------

    # Quick checks / prints
    print("X shape:", X.shape)
    print("K_xx shape:", K_xx.shape)
    print("A shape:", A.shape)
    print("S_points shape:", S_points.shape)
    print("T_points shape:", T_points.shape)
    print("M_S shape:", M_S.shape)
    print("M_T shape:", M_T.shape)

    # Show numeric properties
    eigs_M_S = np.linalg.eigvalsh(M_S)
    eigs_M_T = np.linalg.eigvalsh(M_T)
    eigs_A = np.linalg.eigvalsh(A)

    # 这里的 rank 会非常漂亮地显示为 10，完美印证你的理论！
    print("rank(M_S) (numerical):", np.linalg.matrix_rank(M_S))
    print("min eigen M_S:", eigs_M_S[0])
    print("min eigen M_T:", eigs_M_T[0])
    print("min eigen A:", eigs_A[0])

    # Sanity: M_S should be positive semidefinite, M_T positive definite (typically)
    assert eigs_M_S.min() >= -1e-10
    assert eigs_M_T.min() >= -1e-10
    assert eigs_A.min() >= -1e-10
    # --------------- CVXPY solver call -----------------
    epsilon_prime = 10.0
    print(f"Solving CVXPY problem with epsilon_prime={epsilon_prime}...")
    try:
        W_opt, prob = solve_for_W_cvxpy(A, M_S, M_T, epsilon_prime)
    except ImportError as e:
        print(str(e))
        W_opt = None
        prob = None

    if W_opt is None:
        print("No solution W returned. Exiting.")
        exit(1)
    else:
        # Compute clean matrices same as solver's small additive
        small = 1e-7
        M_S_clean = M_S + small * np.eye(n)
        M_T_clean = M_T + small * np.eye(n)

        # Compute final noise matrix C_opt = inv(W_opt) - A
        # Use pseudo-inverse to avoid inversion blow-up from near-singular W_opt
        inv_W = np.linalg.pinv(W_opt)
        C_opt = inv_W - A

        # Clean up numeric asymmetry
        C_opt = (C_opt + C_opt.T) / 2

        # Project C_opt onto PSD cone to remove tiny negative eigenvalues
        evals, evecs = np.linalg.eigh(C_opt)
        evals[evals < 0] = 0.0
        C_opt = evecs @ np.diag(evals) @ evecs.T

        eigs_C = np.linalg.eigvalsh(C_opt)
        min_eig_C = eigs_C[0]
        print("min eigenvalue of C_opt after PSD projection:", min_eig_C)

        # Print solver status and the optimal privacy leakage (objective: trace(M_S_clean @ W))
        if prob is not None:
            print("Solver status:", prob.status)
        privacy_leakage = float(np.trace(M_S_clean @ W_opt))
        utility_value = float(np.trace(M_T_clean @ W_opt))
        print("Optimal privacy leakage (trace(M_S @ W_opt)):", privacy_leakage)
        print("Achieved utility (trace(M_T @ W_opt)):", utility_value)

    # Task3 Graph
    import matplotlib.pyplot as plt
    from scipy.stats import multivariate_normal
    import matplotlib.patches as patches

    # ---------------------------------------------------------
    # 1. 构造真实的观测数据 Y (假设是一个 2D 空间函数)
    # ---------------------------------------------------------
    # True function: sin(4*pi*x) + cos(4*pi*y)
    def true_function(X_pts):
        return np.sin(4 * np.pi * X_pts[:, 0]) + np.cos(4 * np.pi * X_pts[:, 1])

    Y_clean = true_function(X) + np.random.normal(0, np.sqrt(noise_variance), n)

    # ---------------------------------------------------------
    # 2. 采样隐私保护噪声并生成 Private Y
    # ---------------------------------------------------------
    # 从 C_opt 中采样噪声 eta ~ N(0, C_opt)
    eta = np.random.multivariate_normal(mean=np.zeros(n), cov=C_opt)
    Y_private = Y_clean + eta

    # ---------------------------------------------------------
    # 3. 在整个地图上生成密集的测试点 X_star (画图用)
    # ---------------------------------------------------------
    grid_res = 50
    x_lin = np.linspace(0, 1, grid_res)
    y_lin = np.linspace(0, 1, grid_res)
    xx, yy = np.meshgrid(x_lin, y_lin)
    X_star = np.vstack([xx.ravel(), yy.ravel()]).T

    # 计算核矩阵
    K_star_X = rbf_kernel(X_star, X, lengthscale=lengthscale, variance=variance)
    K_star_star_diag = np.full(X_star.shape[0], variance) # 只需要对角线元素画方差

    # ---------------------------------------------------------
    # 4. 手动计算高斯过程后验 (GPR Posterior)
    # ---------------------------------------------------------
    # A. Non-Private (Baseline) GPR: 使用原始的 A_inv
    A_inv = np.linalg.inv(A)
    mu_nonpriv = K_star_X @ A_inv @ Y_clean
    var_nonpriv = K_star_star_diag - np.einsum('ij,ji->i', K_star_X @ A_inv, K_star_X.T)

    # B. Private GPR: 使用我们优化出来的 W_opt (即 (A + C_opt)^-1)
    mu_priv = K_star_X @ W_opt @ Y_private
    var_priv = K_star_star_diag - np.einsum('ij,ji->i', K_star_X @ W_opt, K_star_X.T)

    # 转换为 2D 网格用于画图
    mu_nonpriv_2d = mu_nonpriv.reshape(grid_res, grid_res)
    var_nonpriv_2d = var_nonpriv.reshape(grid_res, grid_res)
    mu_priv_2d = mu_priv.reshape(grid_res, grid_res)
    var_priv_2d = var_priv.reshape(grid_res, grid_res)

    # ---------------------------------------------------------
    # 5. 绘制对比热力图 (Predictive Mean & Variance)
    # ---------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    cmap_mean = 'viridis'
    cmap_var = 'magma'

    # 定义画敏感区框框的辅助函数
    def draw_sensitive_box(ax):
        rect = patches.Rectangle((0.4, 0.4), 0.2, 0.2, linewidth=2, edgecolor='red', facecolor='none', linestyle='--')
        ax.add_patch(rect)
        ax.scatter(X[:, 0], X[:, 1], c='white', edgecolors='black', s=15, alpha=0.6, label='Sensors')

    # 图 1: Non-Private Mean
    ax = axes[0, 0]
    im = ax.contourf(xx, yy, mu_nonpriv_2d, levels=30, cmap=cmap_mean)
    draw_sensitive_box(ax)
    ax.set_title("Non-Private: Predictive Mean")
    fig.colorbar(im, ax=ax)

    # 图 2: Private Mean
    ax = axes[0, 1]
    im = ax.contourf(xx, yy, mu_priv_2d, levels=30, cmap=cmap_mean)
    draw_sensitive_box(ax)
    ax.set_title("Private: Predictive Mean")
    fig.colorbar(im, ax=ax)

    # 图 3: Non-Private Variance (Confidence)
    ax = axes[1, 0]
    im = ax.contourf(xx, yy, var_nonpriv_2d, levels=30, cmap=cmap_var, vmin=0)
    draw_sensitive_box(ax)
    ax.set_title("Non-Private: Predictive Variance")
    fig.colorbar(im, ax=ax)

    # 图 4: Private Variance (Confidence)
    ax = axes[1, 1]
    # 使用相同的 vmin, vmax 才能看出方差的剧烈增加
    im = ax.contourf(xx, yy, var_priv_2d, levels=30, cmap=cmap_var, vmin=0, vmax=np.max(var_priv_2d))
    draw_sensitive_box(ax)
    ax.set_title("Private: Predictive Variance (Privacy Achieved!)")
    fig.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig("gpr_privacy_comparison.png", dpi=300)
    plt.show()
    print("Done.")
