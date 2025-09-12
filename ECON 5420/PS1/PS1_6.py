import numpy as np
import matplotlib.pyplot as plt

'''Part A'''
def run_simulation(S, N, beta, rho12, rho13, rho23):
    # Mean Vectors
    mu = np.zeros(3)

    # Covariance Matrix
    sigma = np.array([
        [1, rho12, rho13],
        [rho12, 1, rho23],
        [rho13, rho23, 1]
    ])

    results = []

    # Simulate the data over S number of simulations
    for s in range(S):
        samples = np.random.multivariate_normal(mu, sigma, N)
        x1, x2, u1 = samples[:, 0], samples[:, 1], samples[:, 2]

        Y = beta[0] + beta[1]*x1 + beta[2]*x2 + u1
        X = np.column_stack((np.ones(N), x1, x2))

        results.append((Y, X))

    return results

#----------------------------------------------------------------------------------------------

''' Part B '''
def OLS_est(X, Y):
    beta_hat = np.linalg.inv(X.T @ X) @ (X.T @ Y)
    return beta_hat

#----------------------------------------------------------------------------------------------

''' Part C '''
def plot_histogram(simulation, details, true_beta):
    coefficients = []
    for Y, X in simulation:
        beta_hat = OLS_est(X, Y)
        coefficients.append(beta_hat)
    
    # Convert list to numpy array
    coefficients = np.array(coefficients)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Calculate means for reporting
    means = np.mean(coefficients, axis=0)

    # Histograms
    for i in range(3):
        axes[i].hist(coefficients[:, i], bins=20, alpha=0.7, edgecolor='black')
        axes[i].axvline(means[i], color='red', linestyle='--', 
                       label=f'Sample Mean: {means[i]:.3f}')
        axes[i].axvline(true_beta[i], color='green', linestyle='-', 
                       label=f'True Value: {true_beta[i]}')
        axes[i].set_title(f'Histogram of Coefficient β{i}')
        axes[i].set_xlabel('Coefficient Value')
        axes[i].set_ylabel('Frequency')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.suptitle(details)  # Fixed: suptitle, not subtitle
    plt.tight_layout()
    plt.show()
    
    # Report means
    print(f"\n{details}")
    for i in range(3):
        print(f"Mean β̂_{i}: {means[i]:.4f} (True: {true_beta[i]})")
    print("-" * 40)

simulation_c = run_simulation(100, 100, [0, 1, 0], 0, 0, 0)
plot_histogram(simulation_c, 'Part C, rho12 = 0', [0, 1, 0])

#----------------------------------------------------------------------------------------------

''' Part D '''
simulation_d1 = run_simulation(100, 100, [0, 1, 0], 0.4, 0, 0)
plot_histogram(simulation_d1, 'Part D, rho12 = 0.4', [0, 1, 0])

simulation_d2 = run_simulation(100, 100, [0, 1, 0], 0.9, 0, 0)
plot_histogram(simulation_d2, 'Part D, rho12 = 0.9', [0, 1, 0])

# Across all three rho12 values, a consistent result was that rho12=0 estimated a sample mean
# closest to the true mean while rho12=0.4 estimated a sample mean furthest from the true mean.
# All the plotted histogram data across the various rho12 values follow normal trend.

#----------------------------------------------------------------------------------------------

''' Part E '''
simulation_e1 = run_simulation(100, 10000, [0, 1, 0], 0, 0, 0)
plot_histogram(simulation_e1, 'Part E, rho12 = 0, N = 10,000', [0, 1, 0])

simulation_e2 = run_simulation(100, 10000, [0, 1, 0], 0.4, 0, 0)
plot_histogram(simulation_e2, 'Part E, rho12 = 0.4, N = 10,000', [0, 1, 0])

simulation_e3 = run_simulation(100, 10000, [0, 1, 0], 0.9, 0, 0)
plot_histogram(simulation_e3, 'Part E, rho12 = 0.9, N = 10,000', [0, 1, 0])

# It appears the the simulations using a larger number of observations tend to produce histograms
# with a slightly more normal trend end less shock biases in distribution.

#----------------------------------------------------------------------------------------------

''' Part F '''
simulation_f = run_simulation(100, 10000, [0, 1, 0], 0.3, 0.3, 0)
plot_histogram(simulation_f, 'Part F, rho12 = 0.3, rho13 = 0.3', [0, 1, 0])

# The sample coefficients for b1 and b2 are very far off from the true values.
# This can be explained by that rho12 = rho13 = 0.3 means that x1 is correlated with the error term
# and thus violates the exogeneity assumption. Increasing N will now solve the problem as it is
# a fundamental bias. In fact, increasing N would often make the bias more visible.

#----------------------------------------------------------------------------------------------

''' Part G '''
def run_uniform_error_simulation(S, N, beta, rho12, rho13, rho23):
    # Mean Vectors
    mu = np.zeros(3)

    # Covariance Matrix
    sigma = np.array([
        [1, rho12, rho13],
        [rho12, 1, rho23],
        [rho13, rho23, 1]
    ])

    results = []

    # Simulate the data over S number of simulations
    for s in range(S):
        samples = np.random.multivariate_normal(mu, sigma, N)
        x1, x2 = samples[:, 0], samples[:, 1]
        v = np.random.uniform(-2, 2, N)

        Y = beta[0] + beta[1]*x1 + beta[2]*x2 + v
        X = np.column_stack((np.ones(N), x1, x2))

        results.append((Y, X))

    return results

#----------------------------------------------------------------------------------------------

''' Part H '''
simulation_h1 = run_uniform_error_simulation(100, 100, [0, 1, 0], 0, 0, 0)
plot_histogram(simulation_h1, 'Part H, N = 100', [0, 1, 0])

simulation_h2 = run_uniform_error_simulation(100, 10000, [0, 1, 0], 0, 0, 0)
plot_histogram(simulation_h2, 'Part H, N = 10,000', [0, 1, 0])

# The histograms here are relatively similar to the histograms in parts c & d.

#----------------------------------------------------------------------------------------------

''' Part I '''
def run_new_model_simulation(S, N, beta, rho12, rho13, rho23):
    # Mean Vectors
    mu = np.zeros(3)

    # Covariance Matrix
    sigma = np.array([
        [1, rho12, rho13],
        [rho12, 1, rho23],
        [rho13, rho23, 1]
    ])

    results = []

    # Simulate the data over S number of simulations
    for s in range(S):
        samples = np.random.multivariate_normal(mu, sigma, N)
        x1, x2, u1 = samples[:, 0], samples[:, 1], samples[:, 2]

        Y = beta[0] + beta[1]*x1 + beta[2]*(x1**2) + beta[3]*x2 + u1

        results.append((Y, x1, x2))

    return results

#----------------------------------------------------------------------------------------------

''' Part J '''
def compare_models(S, N, beta, rho12, rho13, rho23):
    """
    Compare Model A (misspecified) vs Model B (correctly specified)
    """
    # Generate data from true DGP
    simulation_data = run_new_model_simulation(S, N, beta, rho12, rho13, rho23)
    
    results_A = []  # Model A: yi = α₀ + α₁xi,1 + α₂xi,2 + ui
    results_B = []  # Model B: yi = β₀ + β₁xi,1 + β₂xi,1² + β₃xi,2 + ui
    
    for Y, x1, x2 in simulation_data:
        # Model A (Misspecified): Omits x1² term
        X_A = np.column_stack([np.ones(N), x1, x2])
        alpha_hat = OLS_est(X_A, Y)
        results_A.append(alpha_hat)
        
        # Model B (Correctly specified): Includes x1² term  
        X_B = np.column_stack([np.ones(N), x1, x1**2, x2])
        beta_hat = OLS_est(X_B, Y)
        results_B.append(beta_hat)
    
    # Convert to arrays
    results_A = np.array(results_A)
    results_B = np.array(results_B)
    
    # Plot Model A results
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    model_A_labels = ['α̂₀', 'α̂₁', 'α̂₂']
    means_A = np.mean(results_A, axis=0)
    
    for i in range(3):
        axes[i].hist(results_A[:, i], bins=20, alpha=0.7, edgecolor='black', color='lightcoral')
        axes[i].axvline(means_A[i], color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {means_A[i]:.3f}')
        axes[i].set_title(f'Distribution of {model_A_labels[i]}')
        axes[i].set_xlabel('Coefficient Value')
        axes[i].set_ylabel('Frequency')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.suptitle('Model A: Misspecified (yi = α₀ + α₁xi,1 + α₂xi,2 + ui)', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # Plot Model B results
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    model_B_labels = ['β̂₀', 'β̂₁', 'β̂₂', 'β̂₃']
    means_B = np.mean(results_B, axis=0)
    true_beta = beta
    
    for i in range(4):
        axes[i].hist(results_B[:, i], bins=20, alpha=0.7, edgecolor='black', color='lightblue')
        axes[i].axvline(means_B[i], color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {means_B[i]:.3f}')
        axes[i].axvline(true_beta[i], color='green', linestyle='-', linewidth=2,
                       label=f'True: {true_beta[i]}')
        axes[i].set_title(f'Distribution of {model_B_labels[i]}')
        axes[i].set_xlabel('Coefficient Value')
        axes[i].set_ylabel('Frequency')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

        plt.suptitle('Model B: Correctly Specified (yi = β₀ + β₁xi,1 + β₂xi,1² + β₃xi,2 + ui)', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    return results_A, results_B

results_A, results_B = compare_models(100, 1000, [0, 1, 1, 1], 0.4, 0, 0)

# Model A suffers from omitted variable bias due to excluding x1^2 and thus won't be able to
# caputure the true estimates as Model B does.