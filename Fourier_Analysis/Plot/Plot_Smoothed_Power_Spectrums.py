

import json
import matplotlib.pyplot as plt
import os
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np



def smooth_sharp_jumps(freqs, log_power, delta=0.3, mean_window=10):
    """
    Smooth log power while preserving oscillations:
    - Handles isolated spikes
    - Handles clustered and flat spikes
    - Handles final spike at the end

    Parameters:
    - freqs: list of frequency values
    - log_power: list of log10 power values
    - delta: threshold for detecting a spike
    - mean_window: how many points before to average for cluster/flat spikes

    Returns:
    - freqs, smoothed_log_power
    """
    log_power = np.array(log_power)
    smoothed = log_power.copy()
    n = len(smoothed)

    for i in range(1, n - 1):
        prev = smoothed[i - 1]
        curr = smoothed[i]
        nxt = smoothed[i + 1]

        is_isolated_spike = (curr > prev + delta and curr > nxt + delta)
        is_cluster_leading = (curr > prev + delta and nxt > curr - delta)
        is_flat_jump = (curr > prev + delta and abs(curr - nxt) < delta)

        if is_isolated_spike:
            smoothed[i] = np.mean([prev, nxt])
        elif is_cluster_leading or is_flat_jump:
            start = max(0, i - mean_window)
            smoothed[i] = np.mean(smoothed[start:i])

    # Handle last point
    if n >= 2 and smoothed[-1] > smoothed[-2] + delta:
        smoothed[-1] = smoothed[-2]

    return freqs, smoothed.tolist()





with open("/Users/naokihiratani/Documents/power_spectrum.json", "r") as f:
    all_results = json.load(f)

region = "MOs"
region_data = all_results.get(region, [])

if not region_data:
    raise ValueError(f"No data found for region {region}")

bin_sizes = list(region_data[0]["bin_sizes"].keys())
output_dir = "/Users/Documents/smooth_power_spectrum"
os.makedirs(output_dir, exist_ok=True)

for bin_size in bin_sizes:
    pdf_path = os.path.join(output_dir, f"log_power_spectrum_{region}_{bin_size}.pdf")
    with PdfPages(pdf_path) as pdf:
        n_sessions = len(region_data)
        n_cols = 3
        n_rows = (n_sessions + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
        fig.suptitle(f"{region} - {bin_size} log power spectrum", fontsize=16)

        for idx, session in enumerate(region_data):
            row, col = divmod(idx, n_cols)
            ax = axes[row][col]

            session_id = session['session_id']
            task_spec = session["bin_sizes"][bin_size]["task"]["power_spectrum"]
            sp_spec = session["bin_sizes"][bin_size]["SP"]["power_spectrum"]

            task_4_12 = None
            sp_4_12 = None

            if task_spec:
                # task_freqs = sorted([float(f.replace("Hz", "")) for f in task_spec])

                task_freqs = sorted([float(f.replace("Hz", "")) for f in task_spec])
                task_freqs = [f for f in task_freqs if 1 <= f <= 49]
                task_power_raw = [task_spec[f"{f:.1f}Hz"] for f in task_freqs if task_spec[f"{f:.1f}Hz"] > 0]

                task_freqs = [f for f in task_freqs if task_spec[f"{f:.1f}Hz"] > 0]

                task_log_power = np.log10(task_power_raw)
                _, smoothed_log_power = smooth_sharp_jumps(task_freqs, task_log_power, delta=0.2)
                smoothed_power = 10 ** np.array(smoothed_log_power)

                ax.plot(task_freqs, smoothed_log_power, label="Task", color="lightblue", marker='o', markersize=2)

                total = np.sum(smoothed_power)
                band = np.sum([p for f, p in zip(task_freqs, smoothed_power) if 4 <= f <= 12])
                task_4_12 = 100 * band / total if total > 0 else 0

            if sp_spec:
                sp_freqs = sorted([float(f.replace("Hz", "")) for f in sp_spec])
                sp_freqs = [f for f in sp_freqs if 1 <= f <= 49]
                sp_power_raw = [sp_spec[f"{f:.1f}Hz"] for f in sp_freqs if sp_spec[f"{f:.1f}Hz"] > 0]

                sp_log_power = np.log10(sp_power_raw)
                _, smoothed_log_power = smooth_sharp_jumps(sp_freqs, sp_log_power, delta=0.2)
                smoothed_power = 10 ** np.array(smoothed_log_power)

                ax.plot(sp_freqs, smoothed_log_power, label="SP", color="grey", marker='o', markersize=2)

                total = np.sum(smoothed_power)
                band = np.sum([p for f, p in zip(sp_freqs, smoothed_power) if 4 <= f <= 12])
                sp_4_12 = 100 * band / total if total > 0 else 0

            txt_lines = []
            if task_4_12 is not None:
                txt_lines.append(f"T 4–12Hz: {task_4_12:.2f}%")
            if sp_4_12 is not None:
                txt_lines.append(f"S 4–12Hz: {sp_4_12:.2f}%")

            ax.text(0.98, 0.95, '\n'.join(txt_lines),
                    transform=ax.transAxes,
                    fontsize=9,
                    ha='right', va='top',
                    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'))

            ax.set_title(session_id[:8])
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("log10 Power")
            ax.legend()

        for i in range(n_sessions, n_rows * n_cols):
            row, col = divmod(i, n_cols)
            fig.delaxes(axes[row][col])

        plt.tight_layout(rect=[0, 0, 1, 0.98])
        pdf.savefig(fig)
        plt.close(fig)

    print(f"Saved PDF: {pdf_path}")
