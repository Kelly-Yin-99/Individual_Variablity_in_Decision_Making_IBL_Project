import json
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import pearsonr
from one.api import ONE
import os
one = ONE(base_url='https://openalyx.internationalbrainlab.org')

# Load ITI-only data
with open("/Users/naokihiratani/Documents/power_specturm_iti_0.5.json", "r") as f:
    all_results = json.load(f)

impulsivity_df = pd.read_csv("/Users/naokihiratani/Documents/summary_impulsivity_by_session_1.csv")

# Define cortical region groups
cortical_group_map = {
    'prefrontal': ['FRP', 'ACAd', 'ACAv', 'PL', 'ILA', 'ORBl', 'ORBm', 'ORBvl'],
    'lateral': ['AId', 'AIv', 'AIp', 'GU', 'VISC', 'TEa', 'PERI', 'ECT'],
    'somato-motor': ['SSs', 'SSp', 'MOs', 'MOp'],
    'visual': ['VISal', 'VISli', 'VIpl', 'VISpor', 'VISrl', 'VISp', 'VISl', 'VISpl'],
    'medial': ['VISam', 'VISpm', 'RSPagl', 'RSPd', 'RSPv', 'VISa'],
    'auditory': ['AUDd', 'AUDpo', 'AUDp', 'AUDv'],
    # 'CA': ['CA1'],
    # 'SUB':['SUB','PRE','POST'],



}
region_to_group = {region: group for group, regions in cortical_group_map.items() for region in regions}

def smooth_sharp_jumps(freqs, log_power, delta=0.25, mean_window=4):
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



records = []
for region, region_data in all_results.items():
    for session in region_data:
        session_id = session['session_id']
        subject = session.get('subject', None)
        pid = session['pid']
        region_name = session['region']
        group = region_to_group.get(region_name, None)
        if group is None:
            continue
        bin_data = session["bin_sizes"].get("10ms", {})
        iti_spec = bin_data.get("ITI", {}).get("power_spectrum", None)
        if not iti_spec:
            continue
        binned_power = {}
        for f_str, p in iti_spec.items():
            f = float(f_str.replace("Hz", ""))
            if 2 <= f <= 49 and p > 0:
                bin_center = int(f)
                binned_power.setdefault(bin_center, []).append(p)
        freqs = sorted(binned_power)
        power = [np.mean(binned_power[f]) for f in freqs]
        if not power:
            continue
        log_power = np.log10(power)
        smoothed_log_power = pd.Series(log_power).rolling(window=3, center=True, min_periods=1).mean()
        smoothed_power = 10 ** np.array(smoothed_log_power)
        total_power = np.sum(smoothed_power)
        band_power = np.sum([p for f, p in zip(freqs, smoothed_power) if 4 <= f <= 12])
        rel_power = band_power / total_power if total_power > 0 else np.nan
        records.append({
            "session_id": session_id,
            "pid": pid,
            "region": region_name,
            "subject": subject,
            "cortical_group": group,
            "period": "ITI",
            "relative_power_4_12": rel_power
        })

df = pd.DataFrame(records)
df = df.merge(impulsivity_df, on=['pid', 'session_id', 'region'], how='left')
df = df.dropna(subset=['cortical_group'])

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

def plot_pdf(df, norm_mode, pdf):
    groups = ['auditory', 'lateral', 'medial', 'prefrontal', 'somato-motor', 'visual']
    title_suffix = "Bonferroni-corrected" if norm_mode == 'region' else "Uncorrected"
    fig, axes = plt.subplots(nrows=(len(groups)+2)//3, ncols=3, figsize=(18, 12), squeeze=False)
    fig.suptitle(f"4–12 Hz Relative Power vs. Impulsivity\n(Normalized by {norm_mode.title()} — {title_suffix})", fontsize=16)

    for i in range(len(groups), axes.shape[0] * 3):
        fig.delaxes(axes[i//3][i%3])

    for idx, group in enumerate(groups):
        ax = axes[idx//3][idx%3]
        sub_df = df[df['cortical_group'] == group].dropna(subset=['rel_power_norm', 'impulsivity']).copy()
        if sub_df.empty:
            continue
        sns.scatterplot(
            data=sub_df, x='rel_power_norm', y='impulsivity',
            ax=ax, color='grey', s=40, alpha=0.7
        )
        if sub_df.shape[0] >= 2:
            r, p = pearsonr(sub_df['rel_power_norm'], sub_df['impulsivity'])
            p_val = min(p * 6, 1.0) if norm_mode == 'region' else p
            ax.text(0.05, 0.95, f"r={r:.2f}\np={p_val:.2g}",
                    transform=ax.transAxes, fontsize=8,
                    verticalalignment='top', horizontalalignment='left',
                    color='red' if p_val < 0.05 else 'black')
        ax.set_title(group, fontsize=10)
        ax.set_xlabel("Relative Power", fontsize=9)
        ax.set_ylabel("Anticipatory Index", fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig)
    plt.close()

save_path = "/Users/naokihiratani/Documents/"

# Region-normalized
df['rel_power_norm'] = df.groupby(['region', 'period'])['relative_power_4_12'].transform(lambda x: x / x.mean())
with PdfPages(os.path.join(save_path, "relative_power_iti_region_norm4-12.pdf")) as pdf:
    plot_pdf(df, norm_mode='region', pdf=pdf)

# Subject-averaged figure using region-normalized data 
subject_df = df.groupby(['subject', 'period'])[['rel_power_norm', 'impulsivity']].mean().reset_index()
with PdfPages(os.path.join(save_path, "relative_power_iti_subject_avg4-12.pdf")) as pdf:
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(data=subject_df, x='rel_power_norm', y='impulsivity', hue='period', style='period',
                    markers={'ITI': 'X'}, s=50, alpha=0.8, ax=ax)

    for period_label, sub in subject_df.groupby('period'):
        if sub.shape[0] >= 2:
            r, p = pearsonr(sub['rel_power_norm'], sub['impulsivity'])
            ax.text(0.05, 0.9,
                    f"{period_label}\nr={r:.2f}\np={p:.3g}",
                    transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', horizontalalignment='left',
                    color='red' if p < 0.05 else 'black')

    ax.set_title("Relative Power vs Anticipatory Index (Region-Normalized, Subject-Averaged)")
    ax.set_xlabel("Relative Power")
    ax.set_ylabel("Anticipatory Index")
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()
