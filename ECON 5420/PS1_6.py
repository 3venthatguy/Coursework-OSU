import numpy as np

def sim_data(S, N, beta, rho12, rho13, rho23, seed):
    if seed is not None:
        np.random.seed(seed)

    # Mean Vectors
    mu = np.zeros(3)

    # Covariance Matrix
    sigma = np.array([
        [1, rho12, rho13],
        [rho12, 1, rho23],
        [rho13, rho23, 1]
    ])

    results = []

    for s in range(S):
        # Simulate the data
        samples = np.random.multivariate_normal(mu, sigma, N)
        x1, x2, u1 = samples[:, 0], samples[:, 1], samples[:, 2]

        if len(beta) != 3:
            raise ValueError("Coefficient vector b must have exactly three elements.")
        Y = beta[0] + beta[1]*x1 + beta[2]*x2 + u1
        X = np.column_stack((np.ones(N), x1, x2))

        results.append((Y, X))

    return results

# def OLS_est(X, Y):
#     beta_hat = np.linalg.inv(X.T @ X) @ (X.T @ Y)
#     return beta_hat

# c
simulation = sim_data(100, 100, [0, 1, 0], 0, 0, 0, None)

