import numpy as np
import pandas as pd
from one.api import ONE
from iblatlas.atlas import AllenAtlas
from brainbox.io.one import SpikeSortingLoader, SessionLoader
import os
from pathlib import Path
atlas = AllenAtlas()
ONE.setup(base_url='https://openalyx.internationalbrainlab.org', silent=True,
          cache_dir=Path('/storage1/fs1/hiratani/Active/shared/ibl_space/ONE/openalyx.internationalbrainlab.org'))
one = ONE(base_url='https://openalyx.internationalbrainlab.org', password=os.getenv('ALYX_PASSWORD'))
from datetime import datetime
import pickle
import json
from joblib import Parallel, delayed
from collections import defaultdict


ACRONYM_PREFIXES = [
    # VIS-related
    'VISam', 'VISal', 'VISpm', 'VISpl', 'VISpor', 'VISli', 'VISrl',
    'VISa', 'VISp', 'VISl',

    # AUD-related
    'AUDpo', 'AUDp', 'AUDv', 'AUDd',

    # RSP
    'RSPagl', 'RSPd', 'RSPv',

    # Other brain areas
    'FRP', 'ACAd', 'ACAv', 'PL', 'ILA', 'ORBl', 'ORBm', 'ORBvl',
    'AId', 'AIv', 'AIp', 'GU', 'VISC', 'TEa', 'PERI', 'ECT',
    'SSs', 'SSp', 'MOs', 'MOp'
]


ACRONYM_RULES = [
    (prefix, lambda a, p=prefix: a.startswith(p)) for prefix in ACRONYM_PREFIXES
]


def classify_acronym(acronym):
    for region, rule in ACRONYM_RULES:
        if rule(acronym):
            return region
    return None


def compute_trajectory(pid, one, region_of_interest, behavior_df, atlas=AllenAtlas(), bin_sizes=[0.1, 0.2]):

    session_id, _ = one.pid2eid(pid)
    session_id = str(session_id)

    session_loader = SessionLoader(eid=session_id, one=one)
    session_loader.load_trials()
    trials_data = session_loader.trials
    if trials_data is None or 'stimOn_times' not in trials_data:
        return None
    if len(trials_data) < 400:
        print(f"Session skipped: only {len(trials_data)} trials (< 400).")
        return None

    trials_data = trials_data.iloc[:-40].copy()

    session_data = one.alyx.rest('sessions', 'read', id=session_id)
    subject_nickname = session_data['subject']
    subject_data = one.alyx.rest('subjects', 'list', nickname=subject_nickname)[0]
    start_time_date = datetime.strptime(session_data['start_time'][:10], '%Y-%m-%d')
    birth_date = datetime.strptime(subject_data['birth_date'], '%Y-%m-%d')
    age_days = (start_time_date - birth_date).days
    sex = subject_data['sex']

    contrast_mask = (
        (trials_data['contrastLeft'].abs() < 0.25) |
        (trials_data['contrastRight'].abs() < 0.25) |
        (trials_data['contrastLeft'].isna()) |
        (trials_data['contrastRight'].isna())
    )
    trials_data = trials_data[contrast_mask].reset_index(drop=True)
    filtered_trial_times = trials_data['stimOn_times']

    sl = SpikeSortingLoader(pid=pid, one=one, atlas=atlas)
    spikes, clusters, channels = sl.load_spike_sorting()
    clusters = sl.merge_clusters(spikes, clusters, channels)

    good_cluster_ids = clusters['cluster_id'][clusters['label'] >= 0]
    combined_acronyms = {region: [] for region, _ in ACRONYM_RULES}
    all_regions = {}

    for cluster_id, acronym in zip(clusters['cluster_id'], clusters['acronym']):
        region = classify_acronym(acronym)
        if region:
            combined_acronyms[region].append(acronym)
        else:
            all_regions.setdefault(acronym, []).append(cluster_id)

    for combined_acronym, acronyms in combined_acronyms.items():
        if acronyms:
            combined_cluster_ids = []
            for acronym in acronyms:
                combined_cluster_ids.extend(clusters['cluster_id'][(clusters['acronym'] == acronym) & (
                    np.isin(clusters['cluster_id'], good_cluster_ids))])
            if combined_cluster_ids:
                all_regions[combined_acronym] = list(set(combined_cluster_ids))

    if region_of_interest not in all_regions:
        print(f"Region {region_of_interest} not found in session {session_id}.")
        return None

    spikes_times = spikes['times']
    spikes_clusters = spikes['clusters']
    task_start = trials_data['stimOn_times'].min() - 2
    task_end = trials_data['response_times'].max() + 2
    mask = (spikes['times'] >= task_start) & (spikes['times'] <= task_end)
    spikes_in_task = spikes['clusters'][mask]
    cluster_ids = all_regions[region_of_interest]
    firing_rates = np.array([np.sum(spikes_in_task == cid) / (task_end - task_start) for cid in cluster_ids])
    valid_cluster_ids = np.array(cluster_ids)[firing_rates >= 1]

    if len(valid_cluster_ids) < 10:
        print(f"Not enough neurons with firing rate >= 0.1 Hz in region {region_of_interest} for session {session_id}.")
        return None

    total_spikes_task = np.sum(np.isin(spikes_in_task, valid_cluster_ids))
    if total_spikes_task <= 400000:
        print(f"Not enough spikes in region {region_of_interest} for session {session_id}.")
        return None



    valid_trials_df = behavior_df[
        (behavior_df['pid'] == pid) &
        (behavior_df['session_id'] == session_id) &
        (behavior_df['stimOn_times'].isin(filtered_trial_times))
    ]
    if valid_trials_df.empty:
        return None


    t_hold_offset = -0.5

    results_by_bin = {}
    total_zero_trials = 0
    last_bin_zero_trials = 0

    for bin_size in bin_sizes:
        overlap = bin_size / 2
        rho_method1, rho_method2 = [], []
        rho2_LR_list = []
        segment_count_by_rt = defaultdict(list)
        num_trials_used = 0
        all_segments_by_rt = {'left': defaultdict(list), 'right': defaultdict(list)}
        stored_dX_info = []

        for _, trial in valid_trials_df.iterrows():
            stim_on = trial['stimOn_times']
            move_on = trial['first_movement_onset_times']
            if np.isnan(stim_on) or np.isnan(move_on):
                continue

            rt_aligned = move_on - stim_on
            rt_val = round(np.floor(rt_aligned / bin_size + 0.5) * bin_size, 2)
            if rt_val > 1:
                continue

            rt_bin_end = np.floor((rt_aligned + bin_size / 2) / bin_size) * bin_size + bin_size / 2
            t_hold_bin_start = np.floor((t_hold_offset + bin_size / 2) / bin_size) * bin_size - bin_size / 2

            bin_starts = []
            current = t_hold_bin_start
            while current + bin_size <= rt_bin_end + 1e-8:
                bin_starts.append(current)
                current += overlap


            final_bin_start = rt_bin_end - bin_size
            if len(bin_starts) == 0 or abs(bin_starts[-1] - final_bin_start) > 1e-8:
                bin_starts.append(final_bin_start)

            segment = np.zeros((len(valid_cluster_ids), len(bin_starts)))

            for j, start in enumerate(bin_starts):
                window_start = stim_on + start
                is_last_bin = j == len(bin_starts) - 1

                if is_last_bin:
                    window_end = stim_on + rt_bin_end
                    width = window_end - window_start
                    if width <= 0:
                        print(
                            f" Bin skipped: move_on ({move_on:.3f}) < window_start ({window_start:.3f}), bin_start = {start:.3f}, rt_val = {rt_val:.3f}")
                        continue
                else:
                    window_end = window_start + bin_size
                    width = bin_size

                if width < 1e-4:
                    width += 1e-8

                mask = (spikes_times >= window_start) & (spikes_times < window_end)
                spike_clusters_window = spikes_clusters[mask]
                counts = np.array([np.sum(spike_clusters_window == cid) for cid in valid_cluster_ids]) / width
                if counts.shape[0] != len(valid_cluster_ids):
                    print(
                        f" Skipped due to count dimension mismatch: got {counts.shape[0]}, expected {len(valid_cluster_ids)}")
                    continue
                segment[:, j] = counts

            if segment.shape[1] < 2:
                continue
            if np.all(segment == 0):
                total_zero_trials += 1
                continue

            if np.all(segment[:, -1] == 0):
                last_bin_zero_trials += 1
                continue

            dX = np.diff(segment, axis=1)
            r_start = segment[:, 0]
            r_end = segment[:, -1]
            denom = np.linalg.norm(r_end - r_start)
            if denom == 0:
                continue

            rho1 = np.sum(np.linalg.norm(dX, axis=0)) / denom
            v_unit = (r_end - r_start) / denom
            rho2 = np.sum(np.abs(np.dot(dX.T, v_unit))) / denom

            rho_method1.append((rt_val, rho1))
            rho_method2.append((rt_val, rho2))
            segment_count_by_rt[rt_val].append(dX.shape[1])

            direction = 'left' if trial['first_movement_directions'] == -1 else 'right'

            all_segments_by_rt[direction][rt_val].append(segment)
            stored_dX_info.append((dX, denom, rt_val, direction))

            num_trials_used += 1


        def compute_mean_trajectory_deviation(all_segments_by_rt, valid_cluster_ids):


            deviation_by_rt = {'left': {}, 'right': {}}

            for direction in ['left', 'right']:
                for rt_val, segments in all_segments_by_rt[direction].items():
                    segments = [s for s in segments if s.shape[0] == len(valid_cluster_ids)]
                    if len(segments) < 2:
                        continue
                    min_len = min(seg.shape[1] for seg in segments)
                    segments_trimmed = [seg[:, :min_len] for seg in segments]
                    stack = np.stack(segments_trimmed, axis=0)  # shape (N, d, T)
                    avg_traj = np.mean(stack, axis=0)
                    deviations = [np.linalg.norm(seg - avg_traj, ord='fro') for seg in segments_trimmed]
                    deviation_by_rt[direction][rt_val] = float(np.mean(deviations))

            return deviation_by_rt

        def average_final_bin(direction):
            means = []
            for rt, segs in all_segments_by_rt[direction].items():
                for s in segs:
                    means.append(s[:, -1])
            return np.mean(means, axis=0)


        V_left_global = average_final_bin('left')
        V_right_global = average_final_bin('right')

        v_LR = V_right_global - V_left_global
        v_LR_unit = v_LR / np.linalg.norm(v_LR) if np.linalg.norm(v_LR) > 0 else np.zeros_like(v_LR)

        for dX, denom, rt_val, direction in stored_dX_info:
            if denom == 0 or np.all(v_LR_unit == 0):
                rho2_LR_list.append((rt_val, np.nan))
            else:
                rho2_LR = np.sum(np.abs(np.dot(dX.T, v_LR_unit))) / denom
                rho2_LR_list.append((rt_val, rho2_LR))


        def aggregate_by_rt(pairs):
            df = pd.DataFrame(pairs, columns=['rt', 'rho'])
            return df.groupby('rt')['rho'].mean().to_dict()

        deviation_by_rt = compute_mean_trajectory_deviation(all_segments_by_rt, valid_cluster_ids)

        results_by_bin[str(bin_size)] = {
            'rho_rt_method1': aggregate_by_rt(rho_method1),
            'rho_rt_method2': aggregate_by_rt(rho_method2),
            'rho_rt_LR': aggregate_by_rt(rho2_LR_list),
            'trajectory_deviation': deviation_by_rt
            }


    return {
        'pid': pid,
        'region': region_of_interest,
        'session_id': session_id,
        'subject': subject_nickname,
        'sex': sex,
        'age_days': age_days,
        'trials_used': num_trials_used,
        'neuron count': len(valid_cluster_ids),
        'results_by_bin_size': results_by_bin
    }



ephys_path = "/storage1/fs1/hiratani/Active/shared/ibl_space/ONE/openalyx.internationalbrainlab.org/HMM_Results/ephys_session_info.pkl"
with open(ephys_path, "rb") as f:
    ephys_info = pickle.load(f)

successful_pids_all_regions = ephys_info["successful_pids_ephys"]
behavior_dir = "/storage1/fs1/hiratani/Active/shared/ibl_space/ONE/openalyx.internationalbrainlab.org/HMM_Results/behavior_by_region"


n_jobs = 20  #


def process_one_region(region, pid_list):
    print(f"\n Processing region: {region} ({len(pid_list)} PIDs)")
    behavior_path = os.path.join(behavior_dir, f"{region}_behavior.pkl")
    if not os.path.exists(behavior_path):
        print(f" Behavior file not found for {region}, skipping.")
        return region, []

    behavior_df = pd.read_pickle(behavior_path)
    if behavior_df.empty:
        print(f" Empty behavior_df for {region}, skipping.")
        return region, []

    region_results = []
    for pid in pid_list:
        try:
            result = compute_trajectory(pid, one, region, behavior_df)
            if result:
                region_results.append(result)
        except Exception as e:
            print(f" PID {pid} in {region} failed: {e}")
    return region, region_results


#
region_results_list = Parallel(n_jobs=20)(
    delayed(process_one_region)(region, pid_list)
    for region, pid_list in successful_pids_all_regions.items()
)

#
all_results = {region: results for region, results in region_results_list if results}


#
output_path = "/storage1/fs1/hiratani/Active/shared/ibl_space/ONE/openalyx.internationalbrainlab.org/HMM_Results/rho_results_all_regions_1.json"
with open(output_path, "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\n Saved all results to {output_path}")


