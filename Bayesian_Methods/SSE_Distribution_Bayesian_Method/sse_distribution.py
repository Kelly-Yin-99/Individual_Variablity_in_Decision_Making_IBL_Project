import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cholesky
from scipy.stats import multivariate_normal

# True parameters
tau1_true = 0.02  # seconds
tau2_true = 0.5  # seconds
c1_true = 0.2
c2_true = 0.7
f_true = 2  # Hz
c_osc_true = 0.1
T_test = 1000  # ms
M_test = 500  # samples
max_lag = 600  # ms
bin_size = 1
n_repetitions_test = 500  # reduced repetitions for testing
epsilon_initial = 0.1  # initial threshold
n_samples_per_iteration = 500  # number of accepted samples per iteration
max_iterations = 12500  # maximum SMC iterations

# Priors for Bayesian fitting
prior_ranges = [(0.001, 0.03), (0.05, 1), (0, 1), (0, 1), (0, 4), (0, 1)]  # tau1, tau2, c1, c2, f, c_osc

# Ground truth autocorrelation
def ground_truth_ac(t, tau1, tau2, c1, c2, f, c_osc):
    exp_term = c1 * np.exp(-t / (tau1 * 1000)) + c2 * np.exp(-t / (tau2 * 1000))
    osc_term = c_osc * np.cos(2 * np.pi * f * t / 1000)
    return exp_term + osc_term

# Generate covariance matrix
def construct_covariance_matrix(N, tau1, tau2, c1, c2, f, c_osc):
    t_lags = np.arange(N) * bin_size
    ac_values = ground_truth_ac(t_lags, tau1, tau2, c1, c2, f, c_osc)
    covariance_matrix = np.empty((N, N))
    for i in range(N):
        covariance_matrix[i, :] = ac_values[np.abs(np.arange(N) - i)]
    return covariance_matrix

# Generate samples using Cholesky decomposition
def generate_samples(N, M, tau1, tau2, c1, c2, f, c_osc):
    covariance_matrix = construct_covariance_matrix(N, tau1, tau2, c1, c2, f, c_osc)
    L = cholesky(covariance_matrix, lower=True)
    samples = []
    for _ in range(M):
        z = np.random.normal(size=N)
        sample = L @ z
        samples.append(sample)
    return np.array(samples)

# Optimized sample autocorrelation calculation
def optimized_sample_autocorrelation(samples, max_lag):
    M, N = samples.shape
    global_mean = np.mean(samples)
    global_variance = (1 / (M * N)) * np.sum((samples - global_mean) ** 2)
    ac = np.zeros(max_lag + 1)
    for j in range(max_lag + 1):
        shifted_samples = samples[:, j:] if j > 0 else samples
        lagged_samples = samples[:, :N - j] if j > 0 else samples
        ac[j] = np.mean((lagged_samples - global_mean) * (shifted_samples - global_mean)) / global_variance
    return ac
from scipy.stats import multivariate_normal
import numpy as np
def uniform_prior(value, prior_range):
    low, high = prior_range
    return 1 / (high - low) if low <= value <= high else 0
def gaussian_kernel(theta, mean, cov):
    """
    Computes the Gaussian kernel for the given theta with specified mean and covariance.
    """
    return multivariate_normal.pdf(theta, mean=mean, cov=cov)

def proposal_distribution(theta, accepted_params, weights, cov):
    """
    Computes the mixture of Gaussians proposal distribution probability for theta.
    """
    proposal_prob = 0
    for r in range(len(accepted_params)):
        proposal_prob += weights[r] * gaussian_kernel(theta, accepted_params[r], cov)
    return proposal_prob

def abc_smc(mean_ac, time_lags, prior_ranges, epsilon, n_samples, max_iterations):
    """
    ABC-SMC framework with robust weight handling, using mean AC instead of ground truth AC.
    """
    all_accepted_params = []
    all_accepted_distances = []
    global_best_params = None
    global_min_distance = float('inf')  # Initialize to a very large value

    for iteration in range(max_iterations):
        print('Iteration {}'.format(iteration))
        # Use a proposal distribution after the first iteration
        proposal_dist = None if iteration == 0 else multivariate_normal(mean_params, cov_params)

        accepted_samples = []
        trial_count = 0

        # Collect accepted samples
        while len(accepted_samples) < n_samples:
            trial_count += 1

            # Sample parameters from the proposal distribution or prior
            theta = proposal_dist.rvs() if proposal_dist else [np.random.uniform(low, high) for low, high in prior_ranges]

            # Ensure parameter constraints (e.g., c1 + c2 <= 1)
            if sum(theta[2:4]) > 1:  # c1 + c2 > 1 is invalid
                continue
            if theta[0] <= 0 or theta[1] <= 0:
                continue

            # Evaluate the sample
            tau1, tau2, c1, c2, f, c_osc = theta
            model_ac = ground_truth_ac(time_lags, tau1, tau2, c1, c2, f, c_osc)  # Predicted AC

            # Compute distance between mean AC and model AC
            distance = np.mean((mean_ac - model_ac) ** 2)

            # Accept the sample if the distance is below the threshold (epsilon)
            if distance < epsilon:
                accepted_samples.append((theta, distance))

                # Update the global best parameters if this distance is the smallest so far
                if distance < global_min_distance:
                    global_best_params = theta
                    global_min_distance = distance

        # Extract accepted parameters and distances
        accepted_params, accepted_distances = zip(*accepted_samples)
        accepted_params = np.array(accepted_params)
        accepted_distances = np.array(accepted_distances)

        # Ensure sufficient samples for covariance calculation
        if len(accepted_params) < len(prior_ranges):
            print("Not enough accepted samples to compute covariance. Exiting.")
            break

        # Calculate prior probabilities
        prior_probs = np.array([
            np.prod([uniform_prior(theta[i], prior_ranges[i]) for i in range(len(theta))])
            for theta in accepted_params
        ])

        # Calculate weights robustly
        if iteration == 0:
            weights = prior_probs / (np.sum(prior_probs))  # Add small constant to avoid division by zero
        else:
            proposal_probs = np.array([
                proposal_dist.pdf(theta) for theta in accepted_params
            ])
            weights = prior_probs / (proposal_probs)
            weights /= np.sum(weights)  # Normalize weights

        # Update mean and covariance of the proposal distribution
        mean_params = np.average(accepted_params, axis=0, weights=weights)
        cov_params = 2 * np.cov(accepted_params.T, aweights=weights) + 1e-6*np.eye(len(mean_params))

        # Validate covariance matrix
        if np.any(np.isnan(cov_params)) or np.any(np.isinf(cov_params)):
            print("Invalid covariance matrix encountered. Exiting iteration.")
            break

        # Update epsilon using the 25th percentile of distances
        epsilon = np.percentile(accepted_distances, 25)

        # Store results
        all_accepted_params.append(accepted_params)
        all_accepted_distances.append(accepted_distances)

        # Convergence check
        #if len(accepted_params) / trial_count < 0.004:
            #print("Convergence achieved with acceptance rate below 0.004.")
            #break
        if epsilon <= 0.0001:
            print("convergence achieved")
            break

    # Print global best parameters
    print(f"Global best parameters: {global_best_params}, achieved distance: {global_min_distance}")

    # Return global best parameters
    return all_accepted_params, all_accepted_distances, global_best_params

# Main process

# Initialize lists for storing results
best_tau1_list, best_tau2_list = [], []
sse_list = []  # Store the SSE for the best-fitting parameters

# Main loop for repetitions
for rep in range(n_repetitions_test):
    print(f"Repetition {rep + 1}/{n_repetitions_test}")
    samples = generate_samples(T_test // bin_size, M_test, tau1_true, tau2_true, c1_true, c2_true, f_true, c_osc_true)
    mean_ac = optimized_sample_autocorrelation(samples, max_lag)
    time_lags = np.arange(max_lag + 1) * bin_size

    _, _, best_params = abc_smc(mean_ac, time_lags, prior_ranges, epsilon_initial, n_samples_per_iteration, max_iterations)
    best_tau1_list.append(best_params[0])
    best_tau2_list.append(best_params[1])

    # Compute SSE for the best-fitting parameters
    tau1, tau2, c1, c2, f, c_osc = best_params
    model_ac = ground_truth_ac(time_lags, tau1, tau2, c1, c2, f, c_osc)
    sse = np.sum((mean_ac - model_ac) ** 2)  # Sum of squared errors
    print(sse)
    sse_list.append(sse)

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Best τ1 and τ2 histogram
axes[0].hist(np.array(best_tau1_list) * 1000, bins=30, range=(0, 50), alpha=0.7, label='Tau1 Fits', color='blue')
axes[0].axvline(tau1_true * 1000, color='blue', linestyle='--', label='Ground True τ1')
axes[0].hist(np.array(best_tau2_list) * 1000, bins=30, range=(0, 600), alpha=0.7, label='Tau2 Fits', color='orange')
axes[0].axvline(tau2_true * 1000, color='orange', linestyle='--', label='Ground True τ2')
axes[0].set_xlabel("Tau (ms)")
axes[0].set_ylabel("Frequency")
axes[0].set_title("500 Repetitions: Bayesian-Global Mean Algorithm Fitting Method")
axes[0].legend()
# Plot 2: SSE distribution
axes[1].hist(sse_list, bins=30, alpha=0.7, color='green')
axes[1].set_xlabel("SSE")
axes[1].set_ylabel("Frequency")
axes[1].set_title("500 Repetitions: SSE Distribution")

# Show the plots
plt.tight_layout()
plt.show()

print("Completed 500 repetitions and plotting successfully!")
