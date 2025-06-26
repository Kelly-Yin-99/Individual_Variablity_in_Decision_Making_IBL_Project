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


def compute_spiking_metrics(pid, one, region_of_interest, atlas=AllenAtlas(), bin_sizes=[0.1, 0.2, 0.4, 0.6, 0.8, 1.0]):

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
    valid_cluster_ids = np.array(cluster_ids)[firing_rates >= 0.1]

    if len(valid_cluster_ids) < 10:
        print(f"Not enough neurons with firing rate >= 0.1 Hz in region {region_of_interest} for session {session_id}.")
        return None

    def compute_metrics_in_window(start, end, bin_size=0.1):
        duration = end - start
        n_bins = int(np.floor(duration / bin_size))
        if n_bins < 2:
            return {
                'mean_fr': np.nan, 'median_fr': np.nan,
                'mean_fano': np.nan, 'median_fano': np.nan,
                'mean_cv': np.nan, 'median_cv': np.nan,
                'sd_fr':np.nan, 'sd_fano':np.nan, 'sd_cv':np.nan
            }

        bin_edges = np.linspace(start, start + n_bins * bin_size, n_bins + 1)

        all_fano = []
        all_fr = []
        all_cv = []

        for cid in valid_cluster_ids:
            times = spikes_times[spikes_clusters == cid]

            #  Spike count per bin
            spike_counts, _ = np.histogram(times, bins=bin_edges)
            mean_count = np.mean(spike_counts)
            if mean_count > 0:
                fano = np.var(spike_counts) / mean_count
                fr = mean_count / bin_size
            else:
                fano = np.nan
                fr = 0.0
            all_fano.append(fano)
            all_fr.append(fr)

            #  ISI CV
            isi = np.diff(times[(times >= start) & (times < end)])
            if len(isi) > 1:
                all_cv.append(np.std(isi) / np.mean(isi))
            else:
                all_cv.append(np.nan)

        frs = np.array(all_fr)
        fanos = np.array(all_fano)
        cvs = np.array(all_cv)

        pr = (np.sum(frs > 0) ** 2) / np.sum(frs ** 2) if np.sum(frs ** 2) > 0 else np.nan

        return {
            'mean_fr': float(np.nanmean(frs)),
            'median_fr': float(np.nanmedian(frs)),
            'mean_fano': float(np.nanmean(fanos)),
            'median_fano': float(np.nanmedian(fanos)),
            'mean_cv': float(np.nanmean(cvs)),
            'median_cv': float(np.nanmedian(cvs)),
            'sd_fr': float(np.nanstd(frs)),
            'sd_fano': float(np.nanstd(fanos)),
            'sd_cv': float(np.nanstd(cvs)),
        }

    try:
        passive_times = one.load_dataset(session_id, '*passivePeriods*')
        if 'spontaneousActivity' not in passive_times:
            print(f"No SP in session {session_id}")
            sp_start, sp_end = None, None
        else:
            sp_start = passive_times['spontaneousActivity'][0]
            sp_end = passive_times['spontaneousActivity'][1]
            if sp_end - sp_start < min(bin_sizes):
                print(f" SP too short in session {session_id}")
                sp_start, sp_end = None, None
    except Exception as e:
        print(f"Error loading SP for session {session_id}: {e}")
        sp_start = sp_end = None

   
    session_loader = SessionLoader(eid=session_id, one=one)
    session_loader.load_trials()
    trials = session_loader.trials
    task_start = trials['stimOn_times'].min() - 2
    task_end = trials['response_times'].max() + 2

    results = {}
    for bin_size in bin_sizes:
        key = f"{int(bin_size * 1000)}ms"
        results[key] = {
            'SP': {
                'mean_fr': np.nan, 'median_fr': np.nan,
                'mean_fano': np.nan, 'median_fano': np.nan,
                'mean_cv': np.nan, 'median_cv': np.nan,
                'sd_fr': np.nan,
                'sd_fano': np.nan, 'sd_cv': np.nan
            },
            'task': {
                'mean_fr': np.nan, 'median_fr': np.nan,
                'mean_fano': np.nan, 'median_fano': np.nan,
                'mean_cv': np.nan, 'median_cv': np.nan,
                'sd_fr': np.nan,
                'sd_fano': np.nan, 'sd_cv': np.nan
            }
        }

        # SP
        if sp_start is not None and sp_end - sp_start > bin_size * 2:
            results[key]['SP'] = compute_metrics_in_window(sp_start, sp_end, bin_size)

        # Task
        if task_end - task_start > bin_size * 2:
            results[key]['task'] = compute_metrics_in_window(task_start, task_end, bin_size)

    session_data = one.alyx.rest('sessions', 'read', id=session_id)
    subject_nickname = session_data['subject']
    subject_data = one.alyx.rest('subjects', 'list', nickname=subject_nickname)[0]
    start_time_date = datetime.strptime(session_data['start_time'][:10], '%Y-%m-%d')
    birth_date = datetime.strptime(subject_data['birth_date'], '%Y-%m-%d')
    age_days = (start_time_date - birth_date).days
    sex = subject_data['sex']

    return {
        'pid': pid,
        'region': region_of_interest,
        'session_id': session_id,
        'subject': subject_nickname,
        'sex': sex,
        'age_days': age_days,
        'n_neurons': len(valid_cluster_ids),
        'metrics_by_bin': results
    }



ephys_path = "/storage1/fs1/hiratani/Active/shared/ibl_space/ONE/openalyx.internationalbrainlab.org/HMM_Results/ephys_session_info_1.pkl"
with open(ephys_path, "rb") as f:
    ephys_info = pickle.load(f)

successful_pids_all_regions = ephys_info["successful_pids_ephys"]

n_jobs = 20  


def process_one_region(region, pid_list):
    print(f"\n Processing region: {region} ({len(pid_list)} PIDs)")


    region_results = []
    for pid in pid_list:
        try:
            result = compute_spiking_metrics(pid, one, region)
            if result:
                region_results.append(result)
        except Exception as e:
            print(f" PID {pid} in {region} failed: {e}")
    return region, region_results



region_results_list = Parallel(n_jobs=20)(
    delayed(process_one_region)(region, pid_list)
    for region, pid_list in successful_pids_all_regions.items()
)


all_results = {region: results for region, results in region_results_list if results}

output_path = "/storage1/fs1/hiratani/Active/shared/ibl_space/ONE/openalyx.internationalbrainlab.org/HMM_Results/spiking_metrics_1.json"
with open(output_path, "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\n Saved all results to {output_path}")


