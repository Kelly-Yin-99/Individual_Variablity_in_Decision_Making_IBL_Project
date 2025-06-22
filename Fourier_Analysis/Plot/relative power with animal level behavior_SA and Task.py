import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import pearsonr
from one.api import ONE
import os

# Initialize ONE
one = ONE(base_url='https://openalyx.internationalbrainlab.org')

with open("/Users/naokihiratani/Documents/HMM_RIS/power_spectrum.json", "r") as f:
    all_results = json.load(f)

impulsivity_df = pd.read_csv("/Users/naokihiratani/Documents/summary_impulsivity_by_session_1.csv")

cortical_group_map = {
    'prefrontal': ['FRP', 'ACAd', 'ACAv', 'PL', 'ILA', 'ORBl', 'ORBm', 'ORBvl'],
    'lateral': ['AId', 'AIv', 'AIp', 'GU', 'VISC', 'TEa', 'PERI', 'ECT'],
    'somato-motor': ['SSs', 'SSp', 'MOs', 'MOp'],
    'visual': ['VISal', 'VISli', 'VIpl', 'VISpor', 'VISrl', 'VISp', 'VISl', 'VISpl'],
    'medial': ['VISam', 'VISpm', 'RSPagl', 'RSPd', 'RSPv', 'VISa'],
    'auditory': ['AUDd', 'AUDpo', 'AUDp', 'AUDv'],
    # 'CA': ['CA1'],
    # 'SUB': ['SUB', 'PRE', 'POST'],
}
region_to_group = {region: group for group, regions in cortical_group_map.items() for region in regions}

def smooth_sharp_jumps(freqs, log_power, delta=0.3, mean_window=4):
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

    if n >= 2 and smoothed[-1] > smoothed[-2] + delta:
        smoothed[-1] = smoothed[-2]

    return freqs, smoothed.tolist()

# Extract relative 4–12 Hz power
records = []
for region, region_data in all_results.items():
    for session in region_data:
        session_id = session['session_id']
        pid = session['pid']
        region_name = session['region']
        group = region_to_group.get(region_name, None)
        if group is None:
            continue

        for period in ['task', 'SP']:
            spec = session["bin_sizes"]["10ms"][period]["power_spectrum"]
            if not spec:
                continue
            freqs = sorted([float(f.replace("Hz", "")) for f in spec if 1 <= float(f.replace("Hz", "")) <= 49])

            power = [spec[f"{f:.1f}Hz"] for f in freqs if spec[f"{f:.1f}Hz"] > 0]
            if not power:
                continue

            log_power = np.log10(power)
            _, smoothed_log_power = smooth_sharp_jumps(freqs, log_power, delta=0.2)
            smoothed_power = 10 ** np.array(smoothed_log_power)

            total_power = np.sum(smoothed_power)
            band_power = np.sum([p for f, p in zip(freqs, smoothed_power) if 4 <= f <= 8])
            rel_power = band_power / total_power if total_power > 0 else np.nan

            records.append({
                "session_id": session_id,
                "pid": pid,
                "region": region_name,
                "cortical_group": group,
                "period": period,
                "relative_power_4_12": rel_power
            })


df = pd.DataFrame(records)
df = df.merge(impulsivity_df, on=['pid', 'session_id', 'region'], how='left')
df = df.dropna(subset=['cortical_group'])

from concurrent.futures import ThreadPoolExecutor

def fetch_subject(sess_id):
    try:
        info = one.alyx.rest('sessions', 'read', id=sess_id)
        return sess_id, info['subject']
    except Exception as e:
        print(f"Failed to get subject for session {sess_id}: {e}")
        return sess_id, None

session_ids = df['session_id'].unique()

with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(fetch_subject, session_ids))

session_to_subject = dict(results)
df['subject'] = df['session_id'].map(session_to_subject)
print(1)
# Normalize by region
df['rel_power_norm'] = df.groupby(['region', 'period'])['relative_power_4_12'].transform(lambda x: x / x.mean())


pdf_path = "/Users/naokihiratani/Documents/Relative_Power_Task_SA_1-49hz4-8.pdf"
groups = ['auditory', 'lateral', 'medial', 'prefrontal', 'somato-motor', 'visual']
n_groups = len(groups)
n_cols = 3
n_rows = (n_groups + n_cols - 1) // n_cols

with PdfPages(pdf_path) as pdf:
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(6 * n_cols, 4 * n_rows), squeeze=False)
    fig.suptitle("Normalized 4–12 Hz Relative Power During SA and Task vs. Anticipatory Index (Bonferroni-corrected)", fontsize=16)
    for i in range(n_groups, n_rows * n_cols):
        row, col = divmod(i, n_cols)
        fig.delaxes(axes[row][col])

    for idx, group in enumerate(groups):
        row, col = divmod(idx, 3)
        ax = axes[row][col]
        sub_df = df[df['cortical_group'] == group]
        task_df = sub_df[sub_df['period'] == 'task'].dropna(subset=['rel_power_norm', 'impulsivity']).copy()
        sp_df = sub_df[sub_df['period'] == 'SP'].dropna(subset=['rel_power_norm', 'impulsivity']).copy()

        task_df['source'] = 'task'
        sp_df['source'] = 'SP'
        combined_df = pd.concat([task_df, sp_df])

        if not combined_df.empty:
            sns.scatterplot(
                data=combined_df, x='rel_power_norm', y='impulsivity', hue='source', style='source',
                markers={'task': 'o', 'SP': 'X'}, palette={'task': '#1f77b4', 'SP': '#ff7f0e'}, ax=ax, s=40, alpha=0.7)

            for source_label, sub in combined_df.groupby('source'):
                if sub.shape[0] >= 2:
                    r, p = pearsonr(sub['rel_power_norm'], sub['impulsivity'])
                    p_bonf = min(p * 6, 1.0)
                    y_offset = 0.95 if source_label == 'task' else 0.80
                    ax.text(0.05, y_offset,
                            f"{source_label}\nr={r:.2f}\np={p_bonf:.3g}",
                            transform=ax.transAxes, fontsize=8,
                            verticalalignment='top', horizontalalignment='left',
                            color='red' if p_bonf < 0.05 else 'black')

        ax.set_title(group, fontsize=10)
        ax.set_xlabel("Relative Power", fontsize=9)
        ax.set_ylabel("Anticipatory Index", fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig)
    plt.close()

df['rel_power_region_norm'] = df.groupby(['region', 'period'])['relative_power_4_12'].transform(lambda x: x / x.mean())

# 再按 subject 聚合
subject_df = df.groupby(['subject', 'period']).agg({
    'rel_power_region_norm': 'mean',
    'impulsivity': 'mean'
}).reset_index()


pdf_path_subject = "/Users/naokihiratani/Documents/relative_power_subjectAvg_1-49hz4-8.pdf"
with PdfPages(pdf_path_subject) as pdf:
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(data=subject_df, x='rel_power_region_norm', y='impulsivity', hue='period', style='period',
                    markers={'task': 'o', 'SP': 'X'}, s=50, alpha=0.8, ax=ax)

    for period_label, sub in subject_df.groupby('period'):
        if sub.shape[0] >= 2:
            r, p = pearsonr(sub['rel_power_region_norm'], sub['impulsivity'])
            ax.text(0.05, 0.9 if period_label == 'task' else 0.75,
                    f"{period_label}\nr={r:.2f}\np={p:.3g}",
                    transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', horizontalalignment='left',
                    color='red' if p < 0.05 else 'black')

    ax.set_title("Region-Normalized Relative Power vs Anticipatory Index\n(Subject Averaged)")
    ax.set_xlabel("Relative Power (Region-Normalized, Subject Average)")
    ax.set_ylabel("Anticipatory Index")
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()
