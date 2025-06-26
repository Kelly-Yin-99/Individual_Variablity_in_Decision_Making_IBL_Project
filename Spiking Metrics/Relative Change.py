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
from sklearn.decomposition import PCA
from scipy.stats import entropy

# Define acronym match priority: longest/specific names come first
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

# Build matching rule list using simple startswith
ACRONYM_RULES = [
    (prefix, lambda a, p=prefix: a.startswith(p)) for prefix in ACRONYM_PREFIXES
]

# Classify function
def classify_acronym(acronym):
    for region, rule in ACRONYM_RULES:
        if rule(acronym):
            return region
    return None

def compute_trajectory_metrics(pid, one, region_of_interest):
    session_id, _ = one.pid2eid(pid)
    session_id = str(session_id)

    passive_times = one.load_dataset(session_id, '*passivePeriods*', collection='alf')
    if 'spontaneousActivity' not in passive_times:
        return None
    SP_times = passive_times['spontaneousActivity']

    session_loader = SessionLoader(eid=session_id, one=one)
    session_loader.load_trials()
    trials_data = session_loader.trials
    if trials_data is None or 'stimOn_times' not in trials_data:
        return None

    if len(trials_data) < 400:
        print(f"Session skipped: only {len(trials_data)} trials (< 400).")
        return None

    session_data = one.alyx.rest('sessions', 'read', id=session_id)
    subject_nickname = session_data['subject']
    subject_data = one.alyx.rest('subjects', 'list', nickname=subject_nickname)[0]
    start_time_date = datetime.strptime(session_data['start_time'][:10], '%Y-%m-%d')
    birth_date = datetime.strptime(subject_data['birth_date'], '%Y-%m-%d')
    age_days = (start_time_date - birth_date).days
    sex = subject_data['sex']

    sl = SpikeSortingLoader(pid=pid, one=one, atlas=atlas)
    spikes, clusters, channels = sl.load_spike_sorting()
    clusters = sl.merge_clusters(spikes, clusters, channels)

    good_cluster_ids = clusters['cluster_id'][clusters['label'] >= 0]
    all_regions = {}
    for cid, acr in zip(clusters['cluster_id'], clusters['acronym']):
        if cid not in good_cluster_ids:
            continue
        region = classify_acronym(acr)
        if region is not None:
            all_regions.setdefault(region, []).append(cid)
    if region_of_interest not in all_regions:
        print(f"Region {region_of_interest} not found in session {session_id}.")
        return None

    cluster_ids = all_regions[region_of_interest]
    mask = (spikes['times'] >= SP_times[0]) & (spikes['times'] <= SP_times[1])
    spikes_in_SP = spikes['clusters'][mask]
    firing_rates = np.array([np.sum(spikes_in_SP == cid) / (SP_times[1] - SP_times[0]) for cid in cluster_ids])
    valid_cluster_ids = np.array(cluster_ids)[firing_rates >= 0.1]

    print(f"Valid clusters (>= 0.1 Hz): {len(valid_cluster_ids)}")
    if len(valid_cluster_ids) < 10:
        print(f"Not enough neurons with firing rate >= 0.1 Hz in region {region_of_interest} for session {session_id}.")
        return None

    #num_neurons = len(valid_cluster_ids)

    result = {
        'pid': pid,
        'region': region_of_interest,
        'session_id': session_id,
        'neuron count': len(valid_cluster_ids),
        'age_days' : age_days,
        'sex' : sex
    }

    # 在这里循环不同 bin sizes
    bin_sizes = [0.2, 0.3, 0.4]

    for bin_size in bin_sizes:
        bins = np.arange(SP_times[0], SP_times[1] + bin_size, bin_size)
        spike_matrix = np.zeros((len(valid_cluster_ids), len(bins) - 1))

        for i, cluster_id in enumerate(valid_cluster_ids):
            times = spikes['times'][(spikes['clusters'] == cluster_id) &
                                    (spikes['times'] >= SP_times[0]) & (spikes['times'] <= SP_times[1])]
            counts, _ = np.histogram(times, bins)
            spike_matrix[i, :] = counts

        X_fr = spike_matrix / bin_size

        diffs = np.linalg.norm(np.diff(X_fr, axis=1), axis=0)
        norms = np.linalg.norm(X_fr[:, :-1], axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            p_t = np.where(norms > 1e-10, diffs / norms, np.nan)
        median_change = np.nanmedian(p_t)
        mean_change = np.nanmean(p_t)

        suffix = f"{int(bin_size * 1000)}ms"
        result[f'median_relative_change_{suffix}'] = median_change
        result[f'mean_relative_change_{suffix}'] = mean_change

    return result

# 
ephys_path = "/storage1/fs1/hiratani/Active/shared/ibl_space/ONE/openalyx.internationalbrainlab.org/HMM_Results/ephys_session_info.pkl"
with open(ephys_path, "rb") as f:
    ephys_info = pickle.load(f)

successful_pids_all_regions = ephys_info["successful_pids_ephys"]


def process_one_region(region, pid_list):
    print(f"\n📍 Processing region: {region} ({len(pid_list)} PIDs)")

    region_results = []
    for pid in pid_list:
        try:
            result = compute_trajectory_metrics(pid, one, region)
            if result:
                region_results.append(result)
        except Exception as e:
            print(f" PID {pid} in {region} failed: {e}")
    return region, region_results


# 
region_results_list = Parallel(n_jobs=10)(  
    delayed(process_one_region)(region, pid_list)
    for region, pid_list in successful_pids_all_regions.items()
)


# 
all_results = {region: results for region, results in region_results_list if results}


# 
output_path = "/storage1/fs1/hiratani/Active/shared/ibl_space/ONE/openalyx.internationalbrainlab.org/HMM_Results/RelChange_all_regions.json"
with open(output_path, "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\n Saved all results to {output_path}")


