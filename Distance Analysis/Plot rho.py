import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np
from scipy.stats import pearsonr
import os
from matplotlib.backends.backend_pdf import PdfPages

# === Load data ===
impulsivity_df = pd.read_csv("/Users/naokihiratani/Documents/HMM_RIS/summary_impulsivity_by_session_1.csv")
with open("/Users/naokihiratani/Documents/HMM_RIS/rho_results_all_regions_1.json", "r") as f:
    rho_data = json.load(f)

# === Metric Descriptions ===
metric_descriptions = {
    'rho_rt_method1': "Trajectory Length / Straight Line Distance",
    'rho_rt_method2': "Projected Trajectory Length / Straight Line Distance",
    'rho_rt_LR': "Trajectory Path Length on LR Plane",
    'trajectory_deviation': "Max Deviation From LR Axis"
}
all_metrics = list(metric_descriptions.keys())

# === Group results by bin size
results_by_bin = {}

for region, entries in rho_data.items():
    for entry in entries:
        session_id = entry['session_id']
        pid = entry['pid']
        for bin_key, res in entry['results_by_bin_size'].items():
            if bin_key not in results_by_bin:
                results_by_bin[bin_key] = []
            row = {'region': region, 'pid': pid, 'session_id': session_id}
            for metric in all_metrics:
                values = res.get(metric, {})
                metric_vals = []
                for v in values.values():
                    try:
                        fv = float(v)
                        if not np.isnan(fv):
                            metric_vals.append(fv)
                    except (TypeError, ValueError):
                        continue
                row[metric] = np.mean(metric_vals) if metric_vals else np.nan
            results_by_bin[bin_key].append(row)

# === Plotting ===
save_dir = "/Users/naokihiratani/Documents/HMM_RIS/metric_vs_impulsivity_plots_by_bin"
os.makedirs(save_dir, exist_ok=True)

for bin_key, metric_data in results_by_bin.items():
    df = pd.DataFrame(metric_data)
    merged_df = df.merge(impulsivity_df, on=['pid', 'session_id', 'region'], how='left')
    if merged_df.empty:
        continue

    with PdfPages(os.path.join(save_dir, f"metrics_vs_anticipatory_bin_{bin_key}.pdf")) as pdf:
        for metric in all_metrics:
            plot_df = merged_df.dropna(subset=[metric, 'impulsivity'])
            if plot_df.empty:
                continue

            g = sns.FacetGrid(plot_df, col="region", col_wrap=5, height=3.5, sharex=False, sharey=False)
            g.map_dataframe(sns.scatterplot, x=metric, y='impulsivity', s=20, alpha=0.6)

            for ax, region in zip(g.axes.flat, g.col_names):
                sub = plot_df[plot_df['region'] == region]
                x = sub[metric]
                y = sub['impulsivity']
                mask = ~(x.isna() | y.isna())
                if mask.sum() >= 2:
                    r, p = pearsonr(x[mask], y[mask])
                    label_color = 'red' if p < 0.05 else 'black'
                    region_label = f"{region}{'*' if p < 0.05 else ''}"
                    ax.set_title(region_label, color=label_color, fontsize=9)
                    ax.text(0.05, 0.95, f"r={r:.2f}\np={p:.2g}", transform=ax.transAxes,
                            fontsize=7, verticalalignment='top', horizontalalignment='left',
                            color=label_color)
                ax.set_xlabel(metric, fontsize=8)
                ax.set_ylabel("Anticipatory Index", fontsize=8)

            g.fig.subplots_adjust(top=0.85)
            g.fig.suptitle(f"{metric_descriptions[metric]}\n(bin size = {bin_key}s)", fontsize=12, color='red')
            plt.tight_layout(rect=[0, 0, 1, 0.99])
            pdf.savefig(g.fig)
            plt.close()
