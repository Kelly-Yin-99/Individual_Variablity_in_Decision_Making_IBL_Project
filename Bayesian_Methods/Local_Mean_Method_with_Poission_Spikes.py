import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

# Parameters
tau1 = 0.02  # first timescale in seconds
tau2 = 0.1  # second timescale in seconds
f = 2  # frequency of oscillation (Hz)
c1 = 0.3  # weight of the first timescale
c2 = 0.5  # weight of the second timescale
c_osc = 0.2  # weight of the oscillatory component
T_values = [1000, 2000, 4000, 8000]  # different time durations in ms
max_lag = 600  # Maximum lag in ms
bin_size = 1
M = 500  # Number of independent samples
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


# Calculate autocorrelation using global mean
def sample_autocorrelation_with_global_mean(samples, max_lag):
    M, N = samples.shape  # M: number of samples, N: length of each sample
    ac = np.zeros(max_lag + 1)  # Store mean autocorrelation across all samples

    for j in range(max_lag + 1):
        local_ac = []  # Store autocorrelations for each sample
        for m in range(M):
            local_mean_1 = np.mean(samples[m, :N - j])  # Local mean for the first segment
            local_mean_2 = np.mean(samples[m, j:])  # Local mean for the second segment
            local_variance = np.var(samples[m])  # Variance based on local mean

            # Calculate autocorrelation for current lag
            ac_value = np.mean(
                (samples[m, :N - j] - local_mean_1) * (samples[m, j:] - local_mean_2)
            ) / local_variance
            local_ac.append(ac_value)
        ac[j] = np.mean(local_ac)  # Average autocorrelation across all samples
    return ac



# Ground truth autocorrelation function
def ground_truth_ac(t, tau1, tau2, a, b, f):
    c1 = a
    c2 = b
    c_osc = 1 - a - b
    exponential_term = c1 * np.exp(-t / (tau1 * 1000)) + c2 * np.exp(-t / (tau2 * 1000))
    oscillatory_term = c_osc * np.cos(2 * np.pi * f * t / 1000)
    return exponential_term + oscillatory_term


# Convert samples to Poisson spike trains
def transform_signal(A, sigma, mu, shift_value=0):
    """
    Transform signal A(t') using a shift before applying the max function.
    Parameters:
        - A: Input signal (array-like).
        - sigma: Scaling factor (σ').
        - mu: Offset (μ').
        - shift_value: Amount to shift the signal before applying max (default: 0).
    Returns:
        - A_trans: Transformed signal.
    """
    # Step 1: Apply scaling and offset
    A_trans = A + mu

    # Step 2: Apply shift
    # A_trans += shift_value

    # Step 3: Apply ReLU-like transformation
    A_trans = np.maximum(A_trans, 0)

    return A_trans


def generate_poisson_spikes(samples, dt=1):
    """
    Convert rate-based samples to Poisson spike trains.
    Parameters:
        - samples: Rate-based signal (e.g., output of OU process).
        - dt: Time bin size in seconds (default is 1 ms = 0.001 s).
    Returns:
        - spikes: Poisson spike trains with the same shape as samples.
    """
    # Ensure rates are non-negative
    rates = np.maximum(samples, 0)  # Clamp negative rates to 0
    spikes = np.random.poisson(rates * dt)  # Generate spikes
    return spikes


if __name__ == "__main__":
    fig, axes = plt.subplots(3, 1, figsize=(12, 16))
    colors = ['blue', 'orange', 'green', 'red']

    t_lags = np.arange(max_lag + 1) * bin_size

    # Ground truth for autocorrelation
    ground_truth = ground_truth_ac(t_lags, tau1, tau2, c1, c2, f)
    log_ground_truth = np.log(ground_truth[:101])  # Log(AC) for the first 100 ms

    for i, T in enumerate(T_values):
        N = int(T / bin_size)
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Generate combined OU process
            samples = np.array(list(executor.map(generate_combined_process, [N] * M)))

        # Convert to Poisson spike trains
        spike_samples = generate_poisson_spikes(samples)

        # Compute autocorrelation for spikes
        mean_ac = sample_autocorrelation_with_global_mean(spike_samples, max_lag)

        # Plot mean autocorrelation for all T
        axes[0].plot(t_lags, mean_ac, label=f"Mean AC, T={T} ms", color=colors[i])

        # Example stochastic process
        example_process = spike_samples[0]
        time_points = np.arange(N) * bin_size
        axes[1].plot(time_points, example_process, label=f"T={T} ms", color=colors[i])

        # Log(AC) for first 100 ms lags
        valid = (t_lags <= 100) & (mean_ac > 0)
        axes[2].plot(
            t_lags[valid],
            np.log(mean_ac[valid]),
            label=f"Log(AC), T={T} ms",
            color=colors[i],
        )

    # Finalize plots
    axes[0].set_title("local mean algorithm with Poisson spikes, tau1 =0.02s, tau2 = 0.1s, f = 2 Hz")
    axes[0].set_xlabel("Lag (ms)")
    axes[0].set_ylabel("Autocorrelation")
    axes[0].legend()
    """
    # Add ground truth Log(AC) to the third plot
    axes[2].plot(
        t_lags[:101],
        log_ground_truth,
        'k--',
        label="Log Ground Truth",
        linewidth=2,
    )
    """

    axes[1].set_title("Example Poisson Spike Trains")
    axes[1].set_xlabel("Time (ms)")
    axes[1].set_ylabel("Spike Count")
    axes[1].legend()

    axes[2].set_title("Logarithm of Autocorrelation (First 100 ms)")
    axes[2].set_xlabel("Lag (ms)")
    axes[2].set_ylabel("Log(AC)")
    axes[2].legend()

    plt.tight_layout()
    plt.show()
