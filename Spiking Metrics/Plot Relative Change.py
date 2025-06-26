import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np
from scipy.stats import pearsonr
import os
from matplotlib.backends.backend_pdf import PdfPages

impulsivity_df = pd.read_csv("/Users/naokihiratani/Documents/HMM_RIS/summary_impulsivity_by_session.csv")
with open("/Users/naokihiratani/Documents/HMM_RIS/RelChange_all_regions.json", "r") as f:
    relchange_data = json.load(f)

# Cortical Region Grouping Map 
cortical_group_map = {
    'prefrontal': ['FRP', 'ACAd', 'ACAv', 'PL', 'ILA', 'ORBl', 'ORBm', 'ORBvl'],
    'lateral': ['AId', 'AIv', 'AIp', 'GU', 'VISC', 'TEa', 'PERI', 'ECT'],
    'somato-motor': ['SSs', 'SSp', 'MOs', 'MOp'],
    'visual': ['VISal', 'VISli', 'VIpl', 'VISpor', 'VISrl', 'VISp', 'VISl', 'VISpl'],
    'medial': ['VISam', 'VISpm', 'RSPagl', 'RSPd', 'RSPv', 'VISa'],
    'auditory': ['AUDd', 'AUDpo', 'AUDp', 'AUDv'],
}

region_to_group = {region: group for group, regions in cortical_group_map.items() for region in regions}

# 
relchange_records = []
for region, entries in relchange_data.items():
    for entry in entries:
        pid = entry['pid']
        session_id = entry['session_id']
        for stat_type in ['median', 'mean']:
            for suffix in ["200ms", "300ms", "400ms"]:
                key = f"{stat_type}_relative_change_{suffix}"
                if key in entry and isinstance(entry[key], (float, int)) and not np.isnan(entry[key]):
                    relchange_records.append({
                        "region": region,
                        "session_id": session_id,
                        "pid": pid,
                        "rel_change": entry[key],
                        "bin_label": suffix,
                        "stat_type": stat_type
                    })

relchange_df = pd.DataFrame(relchange_records)
relchange_df["cortical_group"] = relchange_df["region"].map(region_to_group)

#Merge with behavioral impulsivity 
merged_relchange_df = relchange_df.merge(impulsivity_df, on=["pid", "session_id", "region"], how="left")
merged_relchange_df = merged_relchange_df.dropna(subset=["cortical_group", "impulsivity", "rel_change"])

#
save_dir = "/Users/naokihiratani/Documents/HMM_RIS/metric_vs_impulsivity_plots"
os.makedirs(save_dir, exist_ok=True)

for stat_type in ['median', 'mean']:
    pdf_path = os.path.join(save_dir, f"{stat_type}_relative_change_vs_impulsivity.pdf")
    with PdfPages(pdf_path) as pdf:
        for bin_label in ["200ms", "300ms", "400ms"]:
            df = merged_relchange_df[
                (merged_relchange_df["bin_label"] == bin_label) &
                (merged_relchange_df["stat_type"] == stat_type)
            ]
            if df.empty:
                continue

            g = sns.FacetGrid(df, col="cortical_group", col_wrap=3, height=3.5, sharex=False, sharey=False)
            g.map_dataframe(sns.scatterplot, x='rel_change', y='impulsivity', hue='region', s=20, alpha=0.7)

            for ax, group in zip(g.axes.flat, g.col_names):
                sub = df[df['cortical_group'] == group]
                x = sub['rel_change']
                y = sub['impulsivity']
                mask = ~(x.isna() | y.isna())

                sns.scatterplot(data=sub, x='rel_change', y='impulsivity', hue='region', ax=ax,
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

                ax.set_xlabel(f"{stat_type.capitalize()} Relative Change ({bin_label})", fontsize=8)
                ax.set_ylabel("Anticipatory Index", fontsize=8)

            g.add_legend(title="Region", bbox_to_anchor=(1.05, 0.5), loc='center left')
            g.fig.subplots_adjust(top=0.85, right=0.85)
            g.fig.suptitle(f"{stat_type.capitalize()} Relative Change in Spontaneous Firing Rate — Bin Size {bin_label}\n(Bonferroni corrected p-values)",
                           fontsize=12, color='red')
            plt.tight_layout(rect=[0, 0, 0.85, 0.99])
            pdf.savefig(g.fig)
            plt.close()
