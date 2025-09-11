import numpy as np
import matplotlib.pyplot as plt

''' A '''
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

''' B '''
def OLS_est(X, Y):
    beta_hat = np.linalg.inv(X.T @ X) @ (X.T @ Y)
    return beta_hat

''' C '''
simulation = sim_data(100, 100, [0, 1, 0], 0, 0, 0, None)

''' D '''
# # Run OLS estimation on each simulation
# coefficients = []
# for Y, X in simulation:
#     beta_hat = OLS_est(X, Y)
#     coefficients.append(beta_hat)

# coefficients = np.array(coefficients)

# # Create histograms for each coefficient
# plt.figure(num="Figure 1")
# fig1, axes1 = plt.subplots(1, 3, figsize=(15, 5))

# # Histograms
# for i in range(3):
#     axes1[i].hist(coefficients[:, i], bins=20, alpha=0.7, edgecolor='black')
#     axes1[i].set_title(f'Histogram of Coefficient β{i}')
#     axes1[i].set_xlabel('Coefficient Value')
#     axes1[i].set_ylabel('Frequency')

# plt.tight_layout()
# plt.show()

# simulation2 = sim_data(100, 100, [0, 1, 1], 0.4, 0, 0, None)

# # Run OLS estimation on simulation2
# coefficients2 = []
# for Y, X in simulation2:
#     beta_hat = OLS_est(X, Y)
#     coefficients2.append(beta_hat)

# coefficients2 = np.array(coefficients2)

# plt.figure(num="Figure 2")
# fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))

# # Histograms
# for i in range(3):
#     axes2[i].hist(coefficients2[:, i], bins=20, alpha=0.7, edgecolor='black')
#     axes2[i].set_title(f'Histogram of Coefficient β{i}')
#     axes2[i].set_xlabel('Coefficient Value')
#     axes2[i].set_ylabel('Frequency')

# plt.tight_layout()
# plt.show()

# Create simulations with 3 different rho12 values
rho12_values = [0.0, 0.4, 0.9]
simulation_results = []

for rho12 in rho12_values:
    sim = sim_data(100, 100, [0, 1, 0], rho12, 0, 0, None)
    simulation_results.append(sim)

# Run OLS estimation on each simulation for different rho12 values
all_coefficients = []

for sim in simulation_results:
    coefficients = []
    for Y, X in sim:
        beta_hat = OLS_est(X, Y)
        coefficients.append(beta_hat)
    all_coefficients.append(np.array(coefficients))

# Create histograms for each rho12 scenario
fig, axes = plt.subplots(3, 3, figsize=(15, 12))

for row, (rho12, coeffs) in enumerate(zip(rho12_values, all_coefficients)):
    for col in range(3):
        axes[row, col].hist(coeffs[:, col], bins=20, alpha=0.7, edgecolor='black')
        axes[row, col].set_title(f'ρ₁₂ = {rho12}: Coefficient β{col}')
        axes[row, col].set_xlabel('Coefficient Value')
        axes[row, col].set_ylabel('Frequency')

plt.tight_layout()
plt.show()