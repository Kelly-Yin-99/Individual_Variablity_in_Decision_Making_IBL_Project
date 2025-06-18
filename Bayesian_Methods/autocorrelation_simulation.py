import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor


tau1 = 0.06  # first timescale in seconds
tau2 = 0.1    # second timescale in seconds
f = 2         # frequency of oscillation (Hz)
c1 = 0.8     # weight of the first timescale
c2 = 0.5      # weight of the second timescale
c_osc = 0.2   # weight of the oscillatory component
T_values = [1000, 2000, 4000, 8000]  # different time durations in ms
max_lag = 600  # Maximum lag in ms
bin_size = 1
M = 1000  # Number of independent samples
num_workers = 8  #


# Generate OU process for a each timescale
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
    A1 = generate_ou_process(N, tau1)  # OU process with tau1
    #A2 = generate_ou_process(N, tau2)  # OU process with tau2
    oscillatory_component = np.sqrt(2 * c_osc) * np.sin(2 * np.pi * f * t)
    return np.sqrt(c1) * A1  + oscillatory_component


def sample_autocorrelation_with_global_mean(samples, max_lag):
    M, N = samples.shape  # M: number of samples, N: length of each sample
    ac = np.zeros(max_lag + 1)  #
    global_mean = np.mean(samples)


    global_variance = (1 / (M * N)) * np.sum((samples - global_mean) ** 2)

    for j in range(max_lag + 1):
        mean_1 = np.mean([np.mean(samples[m, :N - j]) for m in range(M)])
        mean_2 = np.mean([np.mean(samples[m, j:]) for m in range(M)])
        ac_j = np.mean([
            np.mean((samples[m, :N - j] - mean_1) * (samples[m, j:] - mean_2)) / global_variance
            for m in range(M)
        ])
        ac[j] = ac_j
    return ac


def ground_truth_ac(t):
    # Double timescale exponential decay
    exponential_term = c1 * np.exp(-t /(tau1*1000))
    oscillatory_term = c_osc * np.cos(2 * np.pi * f * t / 1000)  # Convert ms to seconds
    return exponential_term + oscillatory_term


if __name__ == "__main__":
    fig, axes = plt.subplots(2, 1, figsize=(12, 12))
    colors = ['blue', 'orange', 'green', 'red']

    # Time lags for plotting
    t_lags = np.arange(max_lag + 1) * bin_size

    # Plot ground truth autocorrelation
    ground_truth = ground_truth_ac(t_lags)
    axes[0].plot(t_lags, ground_truth, 'k--', label='Ground Truth')

    # Simulate for different T values
    for i, T in enumerate(T_values):
        N = int(T / bin_size)  # Number of time bins
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            samples = np.array(list(executor.map(generate_combined_process, [N] * M)))

        # Calculate autocorrelation using global mean adjustment
        mean_ac = sample_autocorrelation_with_global_mean(samples, max_lag)

        # Plot mean autocorrelation
        axes[0].plot(t_lags, mean_ac, label=f"Mean AC, T={T} ms", color=colors[i])

    # Plot stochastic processes for all T values
    for i, T in enumerate(T_values):
        N = int(T / bin_size)
        example_process = generate_combined_process(N)
        time_points = np.arange(N) * bin_size  # Time points in ms
        axes[1].plot(time_points, example_process, label=f"T={T} ms", color=colors[i])

    # Plot settings for stochastic processes
    axes[1].set_xlabel("Time (ms)")
    axes[1].set_ylabel("Value")
    axes[1].set_title("Stochastic Processes for Different T Intervals")
    axes[1].legend()

    # Plot settings for autocorrelation plot
    axes[0].set_xlabel("Lag (ms)")
    axes[0].set_ylabel("Autocorrelation")
    axes[0].set_title("Autocorrelation Comparison: Ground Truth vs Simulated")
    axes[0].legend()

    plt.tight_layout()
    plt.show()
