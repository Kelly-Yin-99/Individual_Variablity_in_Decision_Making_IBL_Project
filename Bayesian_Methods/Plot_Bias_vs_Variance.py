import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import time

# Parameters
tau1 = 0.02  # first timescale in seconds
tau2 = 0.1  # second timescale in seconds
f = 2  # frequency of oscillation (Hz)
c1 = 0.2  # weight of the first timescale
c2 = 0.7 # weight of the second timescale
c_osc = 0.1  # weight of the oscillatory component
T = 4000  # Duration in ms
max_lag = 600  # Maximum lag in ms
bin_size = 1
M = 50  # Number of independent samples per process
num_workers = 8  # Number of parallel workers


# Generate OU process for a single timescale
def generate_ou_process(N, tau):
    dt = bin_size / 1000
    x = np.zeros(N)
    noise = np.random.normal(0, 1, N)
    for i in range(1, N):
        x[i] = x[i - 1] - (x[i - 1] / tau) * dt + np.sqrt(2 / tau * dt) * noise[i]
    return x


# Generate combined OU process with oscillation
def generate_combined_process(N):
    t = np.arange(N) * bin_size / 1000
    A1 = generate_ou_process(N, tau1)
    A2 = generate_ou_process(N, tau2)
    oscillatory_component = np.sqrt(2 * c_osc) * np.sin(2 * np.pi * f * t)
    return np.sqrt(c1) * A1 + np.sqrt(c2) * A2 + oscillatory_component


# Compute sample autocorrelation with global mean
def sample_autocorrelation_with_global_mean(samples, max_lag):
    M, N = samples.shape  # M: number of samples, N: length of each sample
    ac = np.zeros(max_lag + 1)  # Store mean autocorrelation across all samples
    global_mean = np.mean(samples)  # Compute global mean across all samples and time points

    # Compute global variance
    global_variance = (1 / (M * N)) * np.sum((samples - global_mean) ** 2)

    for j in range(max_lag + 1):
        mean_1 = np.mean([np.mean(samples[m, :N - j]) for m in range(M)])
        mean_2 = np.mean([np.mean(samples[m, j:]) for m in range(M)])
        ac_j = np.mean([np.mean((samples[m, :N - j] - mean_1) * (samples[m, j:] - mean_2)) / global_variance
                        for m in range(M)])
        ac[j] = ac_j

    return ac


# Ground truth autocorrelation function (updated with correct parameters)
def ground_truth_ac(t, tau1, tau2, c1, c2, c_osc, f):
    exponential_term = c1 * np.exp(-t / (tau1 * 1000))+ c2 * np.exp(-t / (tau2 * 1000))
    oscillatory_term = c_osc * np.cos(2 * np.pi * f * t / 1000)
    return exponential_term + oscillatory_term


# Worker function to run the simulation for a specific repetition
def run_single_process(n, M, N, max_lag):
    mean_acs = []
    for _ in range(n):
        # Generate samples for each repetition
        samples = np.array([generate_combined_process(N) for _ in range(M)])

        # Calculate mean autocorrelation for the repetition
        mean_ac = sample_autocorrelation_with_global_mean(samples, max_lag)
        mean_acs.append(mean_ac)

    return np.array(mean_acs)


# Main execution for all values of n
if __name__ == "__main__":
    N = int(T / bin_size)
    t_lags = np.arange(max_lag + 1) * bin_size  # Time lags in ms

    # Different values of n to test
    n_values = [10, 100, 200, 400]
    bias_all_n = {}  # To store bias for each n
    variance_all_n = {}  # To store variance per lag for each n
    average_variance_list = []  # To store average variance for each n

    start_time = time.time()

    for n in n_values:
        print(f"Processing n={n}...")

        # Run the simulation for the current value of n (repeating the entire process n times)
        mean_acs = run_single_process(n, M, N, max_lag)

        # Calculate AC_bar for the current n (mean of autocorrelations from all repetitions)
        ac_bar = np.mean(mean_acs, axis=0)

        # Ground truth autocorrelation (using c1, c2, c_osc as specified)
        ac_gt = ground_truth_ac(t_lags, tau1, tau2, c1, c2, c_osc, f)  # Ground truth AC

        # Bias for each lag
        bias = ac_gt - ac_bar
        bias_all_n[n] = bias  # Store bias for this n

        # Variance for each lag
        variance_per_lag = np.mean((mean_acs - ac_bar) ** 2, axis=0)  # Variance at each lag
        variance_all_n[n] = variance_per_lag  # Store variance curve for this n

        # Average variance across all lags
        average_variance = np.mean(variance_per_lag)
        average_variance_list.append(average_variance)  # Store scalar average variance for this n

    # Plot Bias vs. Lag for different n
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    for n, bias in bias_all_n.items():
        plt.plot(t_lags, bias, label=f"n={n}")
    plt.xlabel('Lag (ms)')
    plt.ylabel('Bias')
    plt.title('Bias vs. Lag for Different n T=1000, M=50, double+osc, OU')
    plt.legend()

    # Plot Variance vs. Lag for different n
    plt.subplot(1, 2, 2)
    for n, variance in variance_all_n.items():
        plt.plot(t_lags, variance, label=f"n={n}")
    plt.xlabel('Lag (ms)')
    plt.ylabel('Variance')
    plt.title('Variance vs. Lag for Different n')
    plt.legend()

    plt.tight_layout()
    plt.show()
