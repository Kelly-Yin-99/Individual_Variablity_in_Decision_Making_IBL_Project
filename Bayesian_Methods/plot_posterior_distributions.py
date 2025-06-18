import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
epsilon_initial = 2  # initial threshold
n_samples_per_iteration = 500  # number of accepted samples per iteration

# Priors for Bayesian fitting
prior_ranges = [[0.001, 0.03], [0.03, 2], (0, 1), (0, 1), (0, 40), (0, 1)]  # tau1, tau2, c1, c2, f, c_osc

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



def optimized_sample_autocorrelation(samples, max_lag):
    M, N = samples.shape  # M: number of samples, N: length of each sample
    acs = np.zeros((M, max_lag + 1))  # Return matrix of individual sample autocorrelations

    global_mean = np.mean(samples)  # Global mean across all samples
    global_variance = (1 / (M * N)) * np.sum((samples - global_mean) ** 2)  # Global variance

    for j in range(max_lag + 1):
        # Compute lag-specific global means (mean_1 and mean_2)
        mean_1 = np.mean([np.mean(samples[m, :N - j]) for m in range(M)])
        mean_2 = np.mean([np.mean(samples[m, j:]) for m in range(M)])

        # Compute autocorrelation for each sample
        for m in range(M):
            numerator = np.mean((samples[m, :N - j] - mean_1) * (samples[m, j:] - mean_2))
            acs[m, j] = numerator / global_variance

    return acs


# Uniform prior
def uniform_prior(value, prior_range):
    low, high = prior_range
    return 1 / (high - low) if low <= value <= high else 0


def abc_smc(sample_ac_matrix, time_lags, prior_ranges, epsilon, n_samples):
    all_mean_params = []  # To store mean_params at each iteration
    all_cov_params = []  # To store cov_params at each iteration
    iteration = 0

    while True:
        print(f"Iteration {iteration + 1}: Starting with epsilon {epsilon}")

        proposal_dist = None if iteration == 0 else multivariate_normal(mean_params, cov_params)

        accepted_samples = []
        trial_count = 0

        while len(accepted_samples) < n_samples:
            trial_count += 1
            if trial_count > 1666667:
                print(f"Trial count exceeded 1666667 in iteration {iteration + 1}. Convergence achieved.")
                break
            theta = proposal_dist.rvs() if proposal_dist else [np.random.uniform(low, high) for low, high in prior_ranges]
            if theta[0] <= 0 or theta[1] <= 0 or theta[2] <= 0 or theta[3] <= 0 or theta[4] <= 0:
                continue
            if sum(theta[2:4]) > 1:  # Ensure c1 + c2 <= 1
                continue

            tau1, tau2, c1, c2, f, c_osc = theta
            model_ac = ground_truth_ac(time_lags, tau1, tau2, c1, c2, f, c_osc)

            # 样本级距离计算
            distances = np.mean((sample_ac_matrix - model_ac) ** 2, axis=1)
            mean_distance = np.mean(distances)

            if mean_distance < epsilon:
                accepted_samples.append((theta, mean_distance))

        accepted_params, accepted_distances = zip(*accepted_samples)
        accepted_params = np.array(accepted_params)
        accepted_distances = np.array(accepted_distances)

        # Calculate prior probabilities
        prior_probs = np.array([
            np.prod([uniform_prior(theta[i], prior_ranges[i]) for i in range(len(theta))])
            for theta in accepted_params
        ])

        # Weights calculation
        if iteration == 0:
            weights = prior_probs / np.sum(prior_probs)
        else:
            proposal_probs = np.array([proposal_dist.pdf(theta) for theta in accepted_params])
            if np.sum(weights) == 0 or np.isnan(np.sum(weights)):
                print("Warning: Weights sum to zero or NaN. Skipping iteration.")
                continue  # Skip this iteration
            weights /= np.sum(weights)

            weights = prior_probs / proposal_probs
            weights /= np.sum(weights)

        # Update proposal distribution parameters
        mean_params = np.average(accepted_params, axis=0, weights=weights)
        cov_params = 4*np.cov(accepted_params.T, aweights=weights)

        all_mean_params.append(mean_params)
        all_cov_params.append(cov_params)

        # Update epsilon
        epsilon = np.percentile(accepted_distances, 25)

        # Check convergence
        acceptance_rate = len(accepted_params) / trial_count
        print(f"Iteration {iteration + 1}: Acceptance rate = {acceptance_rate:.4f}")
        if acceptance_rate < 0.003:
            print("Convergence achieved.")
            break

        iteration += 1

    return all_mean_params, all_cov_params


import time  # Import the time module

def abc_smc(sample_ac_matrix, time_lags, prior_ranges, epsilon, n_samples):
    all_mean_params = []  # To store mean_params at each iteration
    all_cov_params = []  # To store cov_params at each iteration
    iteration = 0

    while True:
        start_time = time.time()  # Start timing the iteration
        print(f"Iteration {iteration + 1}: Starting with epsilon {epsilon}")

        proposal_dist = None if iteration == 0 else multivariate_normal(mean_params, cov_params)

        accepted_samples = []
        trial_count = 0

        while len(accepted_samples) < n_samples:
            trial_count += 1

            # Check if time exceeds 5 minutes (300 seconds)
            elapsed_time = time.time() - start_time
            #print(f"Iteration {iteration + 1}: Elapsed time = {elapsed_time:.4f}")
            if elapsed_time > 180:
                print(f"Iteration {iteration + 1} exceeded 3 minutes. Convergence achieved.")
                return all_mean_params, all_cov_params  # Break out and return results

            theta = proposal_dist.rvs() if proposal_dist else [np.random.uniform(low, high) for low, high in prior_ranges]
            if theta[0] <= 0 or theta[1] <= 0 or theta[2] <= 0 or theta[3] <= 0 or theta[4] <= 0:
                continue
            if sum(theta[2:4]) > 1:  # Ensure c1 + c2 <= 1
                continue

            tau1, tau2, c1, c2, f, c_osc = theta
            model_ac = ground_truth_ac(time_lags, tau1, tau2, c1, c2, f, c_osc)

            # 样本级距离计算
            distances = np.mean((sample_ac_matrix - model_ac) ** 2, axis=1)
            mean_distance = np.mean(distances)

            if mean_distance < epsilon:
                accepted_samples.append((theta, mean_distance))

        accepted_params, accepted_distances = zip(*accepted_samples)
        accepted_params = np.array(accepted_params)
        accepted_distances = np.array(accepted_distances)

        # Calculate prior probabilities
        prior_probs = np.array([
            np.prod([uniform_prior(theta[i], prior_ranges[i]) for i in range(len(theta))])
            for theta in accepted_params
        ])

        # Weights calculation
        if iteration == 0:
            weights = prior_probs / np.sum(prior_probs)
        else:
            proposal_probs = np.array([proposal_dist.pdf(theta) for theta in accepted_params])
            if np.sum(weights) == 0 or np.isnan(np.sum(weights)):
                print("Warning: Weights sum to zero or NaN. Skipping iteration.")
                continue  # Skip this iteration
            weights /= np.sum(weights)

            weights = prior_probs / proposal_probs
            weights /= np.sum(weights)

        # Update proposal distribution parameters
        mean_params = np.average(accepted_params, axis=0, weights=weights)
        cov_params = 2*np.cov(accepted_params.T, aweights=weights)

        all_mean_params.append(mean_params)
        all_cov_params.append(cov_params)

        # Update epsilon
        epsilon = np.percentile(accepted_distances, 25)

        # Check convergence
        acceptance_rate = len(accepted_params) / trial_count
        print(f"Iteration {iteration + 1}: Acceptance rate = {acceptance_rate:.4f}")
        if acceptance_rate < 0.003:
            print("Convergence achieved.")
            break

        iteration += 1

    return all_mean_params, all_cov_params


def plot_posterior_distributions(mean_params, cov_params, labels, color, title):
    theta_dim = len(mean_params)
    fig, axes = plt.subplots(theta_dim, theta_dim, figsize=(15, 15))

    # Sample from the posterior distribution
    sampled_posterior = np.random.multivariate_normal(mean_params, cov_params, size=5000)

    for i in range(theta_dim):
        for j in range(theta_dim):
            if i == j:
                sns.kdeplot(sampled_posterior[:, i], ax=axes[i, j], fill=True, color=color)
                axes[i, j].set_title(labels[i])
                axes[i, j].tick_params(axis='both', which='major', labelsize=8)
                # Format tick labels to three decimal places
                axes[i, j].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.3f}"))
            elif i < j:
                sns.kdeplot(
                    x=sampled_posterior[:, j],
                    y=sampled_posterior[:, i],
                    ax=axes[i, j],
                    fill=True,
                    color=color,
                )
                axes[i, j].tick_params(axis='both', which='major', labelsize=8)
                # Format tick labels to three decimal places
                axes[i, j].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.3f}"))
                axes[i, j].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.3f}"))
            else:
                axes[i, j].axis('off')

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()



# Generate data
samples = generate_samples(T_test // bin_size, M_test, tau1_true, tau2_true, c1_true, c2_true, f_true, c_osc_true)
mean_ac = optimized_sample_autocorrelation(samples, max_lag)
time_lags = np.arange(max_lag + 1) * bin_size

# Run ABC-SMC
all_mean_params, all_cov_params = abc_smc(mean_ac, time_lags, prior_ranges, epsilon_initial, n_samples_per_iteration)

# Extract data for plotting
first_iteration_mean_params = all_mean_params[0]
first_iteration_cov_params = all_cov_params[0]

middle_index = len(all_mean_params) // 2
middle_iteration_mean_params = all_mean_params[middle_index]
middle_iteration_cov_params = all_cov_params[middle_index]

last_iteration_mean_params = all_mean_params[-1]
last_iteration_cov_params = all_cov_params[-1]

# Plot distributions
labels = ["tau1", "tau2", "c1", "c2", "f", "c_osc"]
plot_posterior_distributions(
    first_iteration_mean_params,
    first_iteration_cov_params,
    labels,
    "yellow",
    "Posterior Distribution for First Iteration"
)

plot_posterior_distributions(
    middle_iteration_mean_params,
    middle_iteration_cov_params,
    labels,
    "pink",
    "Posterior Distribution for Middle Iteration"
)

plot_posterior_distributions(
    last_iteration_mean_params,
    last_iteration_cov_params,
    labels,
    "blue",
    "Posterior Distribution for Last Iteration"
)
