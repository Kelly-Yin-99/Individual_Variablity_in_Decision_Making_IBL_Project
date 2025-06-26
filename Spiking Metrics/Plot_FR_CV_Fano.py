import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from matplotlib.backends.backend_pdf import PdfPages
import os
import json
import math


impulsivity_df = pd.read_csv("/Users/naokihiratani/Documents/HMM_RIS/summary_impulsivity_by_session_1.csv")
with open("/Users/naokihiratani/Documents/HMM_RIS/spiking_metrics_1.json", "r") as f:
    spiking_data = json.load(f)


cortical_group_map = {
    'prefrontal': ['FRP', 'ACAd', 'ACAv', 'PL', 'ILA', 'ORBl', 'ORBm', 'ORBvl'],
    'lateral': ['AId', 'AIv', 'AIp', 'GU', 'VISC', 'TEa', 'PERI', 'ECT'],
    'somato-motor': ['SSs', 'SSp', 'MOs', 'MOp'],
    'visual': ['VISal', 'VISli', 'VIpl', 'VISpor', 'VISrl', 'VISp', 'VISl', 'VISpl'],
    'medial': ['VISam', 'VISpm', 'RSPagl', 'RSPd', 'RSPv', 'VISa'],
    'auditory': ['AUDd', 'AUDpo', 'AUDp', 'AUDv'],
}
region_to_group = {region: group for group, regions in cortical_group_map.items() for region in regions}


top_results = []
target_bin = "200ms"
for region, entries in spiking_data.items():
    for entry in entries:
        if target_bin not in entry.get('metrics_by_bin', {}):
            continue

        region_name = entry['region']
        if region_name not in region_to_group:
            continue  #

        session_id = entry['session_id']
        pid = entry['pid']
        region_name = entry['region']
        subject = entry['subject']
        metrics = entry['metrics_by_bin'][target_bin]
        n_neurons = entry.get('n_neurons', np.nan)

        for period in ['SP', 'task']:
            for metric in ['mean_fr', 'median_fr', 'mean_fano', 'median_fano',
                           'mean_cv', 'median_cv', 'sd_fr', 'sd_fano', 'sd_cv']:
                value = metrics.get(period, {}).get(metric, np.nan)
                top_results.append({
                    'session_id': session_id,
                    'pid': pid,
                    'region': region_name,
                    'subject': subject,
                    'n_neurons': n_neurons,
                    'cortical_group': region_to_group.get(region_name, 'Other'),
                    'metric': metric,
                    'period': period,
                    'value': value
                })


df_top = pd.DataFrame(top_results)


top_5_records = []
for group in df_top['cortical_group'].unique():
    sub_df = df_top[df_top['cortical_group'] == group]
    for (metric, period), metric_df in sub_df.groupby(['metric', 'period']):
        top_5 = metric_df.sort_values(by='value', ascending=False).head(5)
        top_5_records.append(top_5)

final_df = pd.concat(top_5_records, ignore_index=True)
final_df_sorted = final_df.sort_values(by=['cortical_group', 'metric', 'period', 'value'], ascending=[True, True, True, False])

output_csv_path = "/Users/naokihiratani/Documents/HMM_RIS/top_5_metrics.csv"
final_df_sorted.to_csv(output_csv_path, index=False)



metrics_by_bin = {}
for region, entries in spiking_data.items():
    for entry in entries:
        region_name = entry['region']
        if region_name not in region_to_group:
            continue  #

        session_id = entry['session_id']
        pid = entry['pid']
        region_name = entry['region']
        subject = entry['subject']
        metrics_by_bin_session = entry['metrics_by_bin']
        for bin_key, metric_dict in metrics_by_bin_session.items():
            if bin_key not in metrics_by_bin:
                metrics_by_bin[bin_key] = []
            row = {
                'session_id': session_id,
                'pid': pid,
                'region': region_name,
                'subject': subject,
                'cortical_group': region_to_group.get(region_name,'Other'),
            }
            for period in ['SP','task']:
                for metric in ['mean_fr', 'median_fr', 'mean_fano', 'median_fano', 'mean_cv', 'median_cv', 'sd_fr', 'sd_fano', 'sd_cv']:
                    value = metric_dict.get(period, {}).get(metric, np.nan)
                    row[f"{metric}_{period}"] = value
            metrics_by_bin[bin_key].append(row)

# Output directory
save_dir = "/Users/naokihiratani/Documents/HMM_RIS/spiking_vs_impulsivity_combined_plot_shapes_sd_norm_1"
os.makedirs(save_dir, exist_ok=True)

import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

# Plotting
for bin_key, data in metrics_by_bin.items():
    df = pd.DataFrame(data)
    df = df.merge(impulsivity_df, on=['pid', 'session_id', 'region'], how='left')
    df = df.dropna(subset=['cortical_group'])

    if df.empty:
        continue

    # Normalize metrics by region mean
    for metric in ['mean_fr', 'median_fr', 'mean_fano', 'median_fano', 'mean_cv', 'median_cv',
                   'sd_fr', 'sd_fano', 'sd_cv']:
        for period in ['SP', 'task']:
            col = f"{metric}_{period}"
            if col in df.columns:
                df[col + "_norm"] = df.groupby('region')[col].transform(lambda x: x / x.mean() if x.mean() else np.nan)

    pdf_path = os.path.join(save_dir, f"spiking_vs_impulsivity_{bin_key}.pdf")
    with PdfPages(pdf_path) as pdf:
        for metric in ['mean_fr', 'median_fr', 'mean_fano', 'median_fano', 'mean_cv', 'median_cv',
                       'sd_fr', 'sd_fano', 'sd_cv']:
            groups = ['auditory', 'lateral', 'medial', 'prefrontal', 'somato-motor', 'visual']
            fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10), squeeze=False)
            fig.suptitle(f"{metric} (normalized) vs Anticipatory Index\nBin size: {bin_key}", fontsize=16)

            for idx, group in enumerate(groups):
                row, col = divmod(idx, 3)
                ax = axes[row][col]
                sub_df = df[df['cortical_group'] == group]

                task_col = f"{metric}_task_norm"
                sp_col = f"{metric}_SP_norm"

                task_df = sub_df.dropna(subset=[task_col, 'impulsivity']).copy()
                sp_df = sub_df.dropna(subset=[sp_col, 'impulsivity']).copy()

                task_df['metric_value'] = task_df[task_col]
                task_df['source'] = 'task'
                sp_df['metric_value'] = sp_df[sp_col]
                sp_df['source'] = 'SP'

                combined_df = pd.concat([task_df, sp_df])

                if not combined_df.empty:
                    sns.scatterplot(
                        data=combined_df,
                        x='metric_value', y='impulsivity',
                        hue='source', style='source',
                        markers={'task': 'o', 'SP': 'X'},
                        palette={'task': '#1f77b4', 'SP': '#ff7f0e'},
                        ax=ax, s=40, alpha=0.7
                    )

                    for source_label, sub in combined_df.groupby('source'):
                        if sub.shape[0] >= 2:
                            r, p = pearsonr(sub['metric_value'], sub['impulsivity'])
                            p_bonf = min(p * 6, 1.0)
                            y_offset = 0.95 if source_label == 'task' else 0.80
                            ax.text(
                                0.05, y_offset,
                                f"{source_label}\nr={r:.2f}\np={p_bonf:.2g}",
                                transform=ax.transAxes,
                                fontsize=8,
                                verticalalignment='top',
                                horizontalalignment='left',
                                color='red' if p_bonf < 0.05 else 'black'
                            )

                ax.set_title(group, fontsize=10)
                ax.set_xlabel(metric + " (normalized)", fontsize=9)
                ax.set_ylabel("Anticipatory Index", fontsize=9)

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            pdf.savefig(fig)
            plt.close()

