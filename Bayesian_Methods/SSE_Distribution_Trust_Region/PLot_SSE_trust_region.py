import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from concurrent.futures import ProcessPoolExecutor
from scipy.linalg import cholesky
import time

# Parameters
tau1 = 0.02  # first timescale in seconds
tau2 = 0.5  # second timescale in seconds
f = 2  # frequency of oscillation (Hz)
c1 = 0.2  # weight of the first timescale
c2 = 0.7     # weight of the second timescale
c_osc = 0.1  # weight of the oscillatory component
T = 1000  # Duration in ms
phi = 1
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


def construct_covariance_matrix(N):
    t_lags = np.arange(N) * bin_size
    ac_values = ground_truth_ac(t_lags,tau1,tau2,c1,c2,c_osc,f)  # Autocorrelation values for lags
    covariance_matrix = np.empty((N, N))
    for i in range(N):
        covariance_matrix[i, :] = ac_values[np.abs(np.arange(N) - i)]
    return covariance_matrix


# Generate samples using Cholesky decomposition with progress printing
def generate_samples(N):
    covariance_matrix = construct_covariance_matrix(N)
    L = cholesky(covariance_matrix, lower=True)
    samples = []
    for i in range(M):
        z = np.random.normal(size=N)  # Generate independent Gaussian noise
        sample = L @ z  # Transform to correlated sample
        samples.append(sample)
        #if (i + 1) % 50 == 0:
            #print(f"Generated {i + 1}/{M} samples")
    return np.array(samples)

# Autocorrelation calculation with global mean adjustment
def sample_autocorrelation_with_global_mean(samples, max_lag):
    M, N = samples.shape
    ac = np.zeros(max_lag + 1)
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
        #print(ac[j])
    return ac

def ground_truth_ac(t,tau1,tau2,c1,c2,c_osc,f):
    exponential_term = c1 * np.exp(-t / (tau1 * 1000)) + c2 * np.exp(-t / (tau2 * 1000))
    oscillatory_term = c_osc * np.cos(2 * np.pi * f * t / 1000)  # Convert ms to seconds
    return exponential_term + oscillatory_term



# Fit autocorrelation with random search
def fit_ac_random_search(t, ac, processing_index, num_initial_guesses=200):
    best_fit = None
    best_sse = np.inf

    for _ in range(num_initial_guesses):
        p0 = [
            np.random.uniform(0.001, 0.03),
            np.random.uniform(0.01, 1),
            np.random.uniform(0.01, 1),
            np.random.uniform(0.01, 1),
            np.random.uniform(0.01, 1),
            np.random.uniform(0.01, 2 * np.pi),
            np.random.uniform(0.01, 2 * np.pi),
        ]

        try:
            popt, _ = curve_fit(
                lambda t, tau1, tau2, a, b,c,f,p: ground_truth_ac(t, tau1,tau2,a,b,c,f,p),
                t,
                ac,
                p0=p0,
                bounds=([0.001, 0.03, 0, 0, 0,0], [0.03, np.inf, np.inf, np.inf,np.inf, np.inf]),
                maxfev=500000
            )
            fitted_ac = ground_truth_ac(t, *popt)
            sse = np.sum((fitted_ac - ac) ** 2)
            if sse < best_sse:
                best_sse = sse
                best_fit = popt
        except RuntimeError:
            continue

    print(f"Processed {processing_index} times - SSE: {best_sse:.4f}- tau1:{best_fit[0]:.4f} - tau2:{best_fit[1]:.4f}")
    return best_fit, best_sse

def fit_ac_random_search(t, ac, processing_index, num_initial_guesses=200):
    best_fit = None
    best_sse = np.inf

    for _ in range(num_initial_guesses):
        p0 = [
            np.random.uniform(0.001, 0.03),  # tau1 within bounds
            np.random.uniform(0.03, 1),     # tau2 within bounds
            np.random.uniform(0, 1),        # a within bounds
            np.random.uniform(0, 1),        # b within bounds
            np.random.uniform(0, 1),        # c within bounds
            np.random.uniform(0, 2 * np.pi),  # f within bounds
        ]

        try:
            popt, _ = curve_fit(
                lambda t, tau1, tau2, a, b, c, f: ground_truth_ac(t, tau1, tau2, a, b, c, f),
                t,
                ac,
                p0=p0,
                bounds=([0.001, 0.03, 0, 0, 0, 0], [0.03, np.inf, np.inf, np.inf, np.inf, np.inf]),
                maxfev=500000
            )
            fitted_ac = ground_truth_ac(t, *popt)
            sse = np.sum((fitted_ac - ac) ** 2)

            # Skip invalid fits based on conditions

            if sse < best_sse:
                best_sse = sse
                best_fit = popt
        except RuntimeError:
            continue

    # Print results of the best fit
    if best_fit is not None:
        print(f"Processed {processing_index} times - SSE: {best_sse:.4f} - tau1: {best_fit[0]:.4f} - tau2: {best_fit[1]:.4f}")
    else:
        print(f"Processed {processing_index} times - No valid fit found.")

    return best_fit, best_sse
def fit_ac_random_search(t, ac, processing_index, num_initial_guesses=200):
    best_fit = None
    best_sse = np.inf

    for _ in range(num_initial_guesses):
        p0 = [
            np.random.uniform(0.001, 0.03),  # tau1 within bounds
            np.random.uniform(0.03, 1),     # tau2 within bounds
            np.random.uniform(0, 1),        # a within bounds
            np.random.uniform(0, 1),        # b within bounds
            np.random.uniform(0, 1),        # c within bounds
            np.random.uniform(0, 2 * np.pi),  # f within bounds

        ]

        try:
            popt, _ = curve_fit(
                lambda t, tau1, tau2, a, b, c, f: ground_truth_ac(t, tau1, tau2, a, b, c, f),
                t,
                ac,
                p0=p0,
                bounds=([0.001, 0.03, 0, 0, 0, 0], [0.03, np.inf, np.inf, np.inf, np.inf, np.inf]),
                maxfev=500000
            )
            fitted_ac = ground_truth_ac(t, *popt)
            sse = np.sum((fitted_ac - ac) ** 2)

            # Update the best fit and best SSE
            if sse < best_sse:
                best_sse = sse
                best_fit = popt

        except RuntimeError:
            continue

    # Apply filtering AFTER the loop
    if best_fit is not None:
        if best_sse > 0.1 or best_fit[3] < 0.01 or best_fit[3] > 1000:  # c2 is best_fit[3]
            print(f"Final fit discarded: SSE={best_sse:.4f}, c2={best_fit[3]:.4f}")
            best_fit = None
            best_sse = np.inf

    # Print results of the best fit
    if best_fit is not None:
        print(f"Processed {processing_index} times - SSE: {best_sse:.4f} - tau1: {best_fit[0]:.4f} - tau2: {best_fit[1]:.4f}")
    else:
        print(f"Processed {processing_index} times - No valid fit found.")

    return best_fit, best_sse

"""
def fit_ac_random_search(t, ac, processing_index, num_initial_guesses=200):
    best_fit = None
    best_sse = np.inf

    for _ in range(num_initial_guesses):
        p0 = [
            np.random.uniform(0.001, 0.1),
            np.random.uniform(0.1, 1),
            np.random.uniform(0.01, 1),
            np.random.uniform(0.01, 1),
            np.random.uniform(0.01, 1),
            np.random.uniform(0.01, 2 * np.pi),
            # np.random.uniform(0.01, 1),
        ]

        try:
            popt, _ = curve_fit(
                lambda t, tau1, tau2, a, b, c, f: ground_truth_ac(t, tau1, tau2, a, b, c, f),
                t,
                ac,
                p0=p0,
                bounds=([0.001, 0.03, 0, 0, 0, 0], [0.03, np.inf, np.inf, np.inf, np.inf, np.inf]),
                maxfev=500000
            )
            fitted_ac = ground_truth_ac(t, *popt)
            sse = np.sum((fitted_ac - ac) ** 2)

            # Skip invalid fits based on conditions
            if sse > 0.1 or popt[3] < 0.01 or popt[3] > 1000:  # c2 is popt[3]
                print(f"Skipping due to invalid fit: SSE={sse:.4f}, c2={popt[3]:.4f}")
                continue

            if sse < best_sse:
                best_sse = sse
                best_fit = popt
        except RuntimeError:
            continue

    # Print results of the best fit
    if best_fit is not None:
        print(f"Processed {processing_index} times - SSE: {best_sse:.4f} - tau1: {best_fit[0]:.4f} - tau2: {best_fit[1]:.4f}")
    else:
        print(f"Processed {processing_index} times - No valid fit found.")

    return best_fit, best_sse

"""
# Parallelized fitting
def parallel_fit(t_lags, mean_ac, processing_index=1):
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                fit_ac_random_search,
                t_lags,
                mean_ac,
                processing_index
            )
        ]
        results = [future.result() for future in futures]
    return results[0]


#
if __name__ == "__main__":
    # Number of time points
    N = int(T / bin_size)
    t_lags = np.arange(max_lag + 1) * bin_size  # Time lags in ms

    all_tau1_fits = []
    all_tau2_fits = []
    all_sse_values = []

    print("Starting 500 repetitions of the fitting process...")
    for i in range(500):  # Repeat the whole process 500 times
        # Generate samples
        samples = generate_samples(N)

        # Calculate mean autocorrelation
        mean_ac = sample_autocorrelation_with_global_mean(samples, max_lag)
        #print(mean_ac)
        # Fit autocorrelation using random search
        popt, sse = fit_ac_random_search(t_lags, mean_ac, i + 1)

        # Collect results if fitting was successful
        if popt is not None:
            all_tau1_fits.append(popt[0] * 1000)  # Convert tau1 to ms
            all_tau2_fits.append(popt[1] * 1000)  # Convert tau2 to ms
            all_sse_values.append(sse)

    # Plotting
    print("Plotting results...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # First plot: Tau1 and Tau2 distributions
    axes[0].hist(all_tau1_fits, bins=30, alpha=0.7, label="Tau1 Fits", color="blue")
    axes[0].hist(all_tau2_fits, bins=30, alpha=0.7, label="Tau2 Fits", color="orange")
    axes[0].axvline(tau1 * 1000, color="blue", linestyle="--", label="Ground Truth Tau1")
    axes[0].axvline(tau2 * 1000, color="orange", linestyle="--", label="Ground Truth Tau2")
    axes[0].set_xlabel("Tau (ms)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("500 Repetitions: Trust Region Global Mean Algorithm Fitting Method")
    axes[0].legend()

    # Second plot: SSE distribution
    axes[1].hist(all_sse_values, bins=30, alpha=0.7, label="SSE Distribution", color="green")
    axes[1].set_xlabel("SSE")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("500 Repetitions: SSE Distribution")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    print("Completed 500 repetitions and plotting successfully!")
