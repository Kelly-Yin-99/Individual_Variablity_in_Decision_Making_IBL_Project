import json
import matplotlib.pyplot as plt
import os
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

#  Load the local JSON file 
with open("/Users/Documents/power_spectrum.json", "r") as f:
    all_results = json.load(f)

region = ("MOs")
region_data = all_results.get(region, [])

if not region_data:
    raise ValueError(f"No data found for region {region}")

# Extract all bin sizes from the first session
bin_sizes = list(region_data[0]["bin_sizes"].keys())
output_dir = "/Users/naokihiratani/Documents/power_spectrum_iti"
os.makedirs(output_dir, exist_ok=True)
pdf = PdfPages(os.path.join(output_dir, "power_spectrum.pdf"))


for bin_size in bin_sizes:
    # Create one PDF per bin size
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
            task_rel = session["bin_sizes"][bin_size]["task"]["relative_power"]
            sp_rel = session["bin_sizes"][bin_size]["SP"]["relative_power"]

            if task_spec:
                task_freqs = sorted([float(f.replace("Hz", "")) for f in task_spec])
                task_freqs = [f for f in task_freqs if 0 <= f <= 49 and task_spec[f"{f:.1f}Hz"] > 0]
                task_power = [np.log10(task_spec[f"{f:.1f}Hz"]) for f in task_freqs]
                ax.plot(task_freqs, task_power, label="Task", color="lightblue", marker='o', markersize=2)

            if sp_spec:
                sp_freqs = sorted([float(f.replace("Hz", "")) for f in sp_spec])
                sp_freqs = [f for f in sp_freqs if 0 <= f <= 49 and sp_spec[f"{f:.1f}Hz"] > 0]
                sp_power = [np.log10(sp_spec[f"{f:.1f}Hz"]) for f in sp_freqs]
                ax.plot(sp_freqs, sp_power, label="SP", color="grey", marker='o', markersize=2)

            # Annotate relative power
            txt_lines = []
            if task_rel:
                task_4_12 = 100 * task_rel.get('4-12Hz', 0)
                txt_lines.append(f"T 4–12Hz: {task_4_12:.2f}%")
            if sp_rel:
                sp_4_12 = 100 * sp_rel.get('4-12Hz', 0)
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

        # Remove empty subplots
        for i in range(n_sessions, n_rows * n_cols):
            row, col = divmod(i, n_cols)
            fig.delaxes(axes[row][col])

        plt.tight_layout(rect=[0, 0, 1, 0.98])
        pdf.savefig(fig)
        plt.close(fig)

    print(f"Saved PDF: {pdf_path}")

