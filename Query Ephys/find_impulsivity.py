import os
import pandas as pd


# Define acronym match priority: longest/specific names come first
ACRONYM_PREFIXES = [
    # VIS-related
    'VISam', 'VISal', 'VISpm', 'VISpor', 'VISli', 'VISrl', 'VISpl',
    'VISa', 'VISp', 'VISl',

    # AUD-related
    'AUDpo', 'AUDp', 'AUDv', 'AUDd',

    # RSP
    'RSPagl', 'RSPd', 'RSPv',

    # Other brain areas
    'FRP', 'ACAd', 'ACAv', 'PL', 'ILA', 'ORBl', 'ORBm', 'ORBvl',
    'AId', 'AIv', 'AIp', 'GU', 'VISC', 'TEa', 'PERI', 'ECT',
    'SSs', 'SSp', 'MOs', 'MOp','SCop', 'SCsg', 'SCzo','ICc', 'ICd', 'ICe'
    'CA1','CA2', 'CA3','SUB','PRE','POST'
]


behavior_dir = "/storage1/fs1/hiratani/Active/shared/ibl_space/ONE/openalyx.internationalbrainlab.org/HMM_Results/behavior_by_region"
summary_rows = []

for region in ACRONYM_PREFIXES:
    behavior_path = os.path.join(behavior_dir, f"{region}_behavior.pkl")
    if not os.path.exists(behavior_path):
        print(f"Skipping {region} (no behavior file)")
        continue

    df = pd.read_pickle(behavior_path)
    if df.empty:
        print(f"{region} behavior file is empty.")
        continue

    # ➤ 计算反应时间和 impulsive 标记
    df['reaction_time'] = df['first_movement_onset_times'] - df['stimOn_times']
    df['is_impulsive'] = df['reaction_time'] < 0.08
    df['is_very_slow'] = df['reaction_time'] > 1.25
    df['is_rewarded'] = df['rewardVolume'] > 0

    # ➤ 聚合行为指标（每个 session_id 级别）
    behavior_summary = df.groupby('session_id').agg(
        num_trials=('reaction_time', 'size'),
        num_impulsive=('is_impulsive', 'sum'),
        num_very_slow=('is_very_slow', 'sum'),
        sex=('sex', 'first'),
        subject=('subject', 'first'),
        age_days=('age', 'first')
    ).reset_index()

    behavior_summary['impulsivity'] = (
        behavior_summary['num_impulsive'] - behavior_summary['num_very_slow']
    ) / behavior_summary['num_trials']

    # ➤ 提取 session_id - pid 对应关系（去重）
    pid_mapping = df[['session_id', 'pid']].drop_duplicates()

    # ➤ 合并行为指标和 pid 信息（left merge 保留每个 pid）
    merged = pid_mapping.merge(behavior_summary, on='session_id', how='left')
    merged['region'] = region

    summary_rows.append(merged)

# === 拼接所有区域结果 ===
summary_df = pd.concat(summary_rows, ignore_index=True)

# === 保存为 CSV ===
output_path = "/storage1/fs1/hiratani/Active/shared/ibl_space/ONE/openalyx.internationalbrainlab.org/HMM_Results/summary_impulsivity_by_session_1.csv"
summary_df.to_csv(output_path, index=False)
print(f"Saved summary to {output_path}")
