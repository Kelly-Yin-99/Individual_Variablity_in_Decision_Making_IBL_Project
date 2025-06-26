
 import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np
from scipy.stats import pearsonr
import os
from matplotlib.backends.backend_pdf import PdfPages


impulsivity_df = pd.read_csv("/Users/naokihiratani/Documents/HMM_RIS/summary_impulsivity_by_session_1.csv")
with open("/Users/naokihiratani/Documents/HMM_RIS/rho_results_all_regions.json", "r") as f:
    rho_data = json.load(f)


cortical_group_map = {
    'prefrontal': ['FRP', 'ACAd', 'ACAv', 'PL', 'ILA', 'ORBl', 'ORBm', 'ORBvl'],
    'lateral': ['AId', 'AIv', 'AIp', 'GU', 'VISC', 'TEa', 'PERI', 'ECT'],
    'somato-motor': ['SSs', 'SSp', 'MOs', 'MOp'],
    'visual': ['VISal', 'VISli', 'VIpl', 'VISpor', 'VISrl', 'VISp', 'VISl', 'VISpl'],
    'medial': ['VISam', 'VISpm', 'RSPagl', 'RSPd', 'RSPv', 'VISa'],
    'auditory': ['AUDd', 'AUDpo', 'AUDp', 'AUDv'],
}
region_to_group = {region: group for group, regions in cortical_group_map.items() for region in regions}

# === Metric Descriptions ===
metric_descriptions = {
    'rho_rt_method1': "Trajectory Length / Straight Line Distance",
    'rho_rt_method2': "Projected Trajectory Length / Straight Line Distance",
    'rho_rt_LR': "Trajectory Path Length on LR Plane",
    'trajectory_deviation_left': "Left Trial Deviation (Normalized by Neuron Count)",
    'trajectory_deviation_right': "Right Trial Deviation (Normalized by Neuron Count)"
}
core_metrics = ['rho_rt_method1', 'rho_rt_method2', 'rho_rt_LR']
deviation_metrics = ['trajectory_deviation_left', 'trajectory_deviation_right']
all_metrics = core_metrics + deviation_metrics

results_by_bin = {}
filtered_out_rows = []


for region, entries in rho_data.items():
    for entry in entries:
        session_id = entry['session_id']
        pid = entry['pid']
        n_neurons = entry.get('neuron count', 1)
        for bin_key, res in entry['results_by_bin_size'].items():
            if bin_key not in results_by_bin:
                results_by_bin[bin_key] = []
            row = {'region': region, 'pid': pid, 'session_id': session_id}
            # rho metrics
            for metric in core_metrics:
                values = res.get(metric, {})
                metric_vals = []
                for rt, v in values.items():
                    try:
                        rt_val = float(rt)
                        v_val = float(v)
                        if 0.1 <= rt_val <= 0.6 and not np.isnan(v_val) and abs(v_val) < 500:
                            metric_vals.append(v_val)
                        elif abs(v_val) >= 500:
                            filtered_out_rows.append({
                                'session_id': session_id,
                                'pid': pid,
                                'region': region,
                                'metric': metric,
                                'rt_bin': rt,
                                'neurons': n_neurons,
                                'value': v_val
                            })
                    except (ValueError, TypeError):
                        continue
                row[metric] = np.mean(metric_vals) if metric_vals else np.nan
            # deviation metrics
            for side in ['left', 'right']:
                side_key = f'trajectory_deviation_{side}'
                values = res.get('trajectory_deviation', {}).get(side, {})
                dev_vals = []
                for rt, v in values.items():
                    try:
                        rt_val = float(rt)
                        v_val = float(v) / n_neurons
                        if 0.1 <= rt_val <= 0.6 and not np.isnan(v_val) and abs(v_val) < 500:
                            dev_vals.append(v_val)
                    except (ValueError, TypeError):
                        continue
                row[side_key] = np.mean(dev_vals) if dev_vals else np.nan
            # lr_vals = []
            # values = res.get('rho_rt_LR', {})
            # for rt, v in values.items():
            #     try:
            #         rt_val = float(rt)
            #         v_val = float(v)
            #         if 0.1 <= rt_val <= 0.6 and v_val < 500:
            #             lr_vals.append(v_val)
            #     except:
            #         continue
            # row['rho_rt_LR'] = np.mean(lr_vals) if lr_vals else np.nan
            results_by_bin[bin_key].append(row)


filtered_df = pd.DataFrame(filtered_out_rows)
filtered_df.to_csv("/Users/naokihiratani/Documents/HMM_RIS/filtered_out_large_metrics.csv", index=False)

#
save_dir = "/Users/naokihiratani/Documents/HMM_RIS/metric_vs_impulsivity_plots_by_bin_grouped_filtered_with_deviation"
os.makedirs(save_dir, exist_ok=True)

for bin_key, metric_data in results_by_bin.items():
    df = pd.DataFrame(metric_data)
    df['cortical_group'] = df['region'].map(region_to_group)
    merged_df = df.merge(impulsivity_df, on=['pid', 'session_id', 'region'], how='left')
    merged_df = merged_df.dropna(subset=['cortical_group'])

    if merged_df.empty:
        continue

    with PdfPages(os.path.join(save_dir, f"metrics_vs_anticipatory_bin_{bin_key}.pdf")) as pdf:
        for metric in all_metrics:
            plot_df = merged_df.dropna(subset=[metric, 'impulsivity'])
            if plot_df.empty:
                continue

            g = sns.FacetGrid(plot_df, col="cortical_group", col_wrap=3, height=3.5, sharex=False, sharey=False)
            g.map_dataframe(sns.scatterplot, x=metric, y='impulsivity', hue='region', s=20, alpha=0.7)

            for ax, group in zip(g.axes.flat, g.col_names):
                sub = plot_df[plot_df['cortical_group'] == group]
                x = sub[metric]
                y = sub['impulsivity']
                mask = ~(x.isna() | y.isna())

                sns.scatterplot(data=sub, x=metric, y='impulsivity', hue='region', ax=ax,
                                palette='tab10', s=20, alpha=0.7, legend=False)

                if mask.sum() >= 2:
                    r, p = pearsonr(x[mask], y[mask])
                    p_bonf = min(p * 6, 1.0)
                    label_color = 'red' if p_bonf < 0.05 else 'black'
                    group_label = f"{group}{'*' if p_bonf < 0.05 else ''}"
                    ax.set_title(group_label, color=label_color, fontsize=9)
                    ax.text(0.05, 0.95, f"r={r:.2f}\np={p_bonf:.2g}", transform=ax.transAxes,
                            fontsize=7, verticalalignment='top', horizontalalignment='left',
                            color=label_color)

                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    ax.legend(handles, labels, title='Region', fontsize=6, title_fontsize=7,
                              loc='lower left', frameon=False, handletextpad=0.1, borderpad=0.1)

                ax.set_xlabel(metric, fontsize=8)
                ax.set_ylabel("Anticipatory Index", fontsize=8)

            g.add_legend(title="Region", bbox_to_anchor=(1.05, 0.5), loc='center left')
            g.fig.subplots_adjust(top=0.85, right=0.85)
            g.fig.suptitle(
                f"{metric_descriptions[metric]}\n(RT from 0.1 to 0.6, bin size = {bin_key}s, after Bonferroni correction)",
                fontsize=12, color='red')
            plt.tight_layout(rect=[0, 0, 0.85, 0.99])
            pdf.savefig(g.fig)
            plt.close()
