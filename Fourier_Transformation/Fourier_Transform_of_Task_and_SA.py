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
from collections import defaultdict
from scipy.fft import rfft, rfftfreq
from collections import defaultdict


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
    'SSs', 'SSp', 'MOs', 'MOp','SCop', 'SCsg', 'SCzo','ICc', 'ICd', 'ICe',
    'CA1', 'CA2', 'CA3', 'SUB','PRE','POST'
]

# Build matching rule list using simple startswith
ACRONYM_RULES = [
    (prefix, lambda a, p=prefix: a.startswith(p)) for prefix in ACRONYM_PREFIXES
]

def classify_acronym(acronym):
    for region, rule in ACRONYM_RULES:
        if rule(acronym):
            return region
    return None

def extract_power_and_relative_power(signal, sampling_rate, max_freq=50, bin_width=0.1):
    n = len(signal)
    freqs = fftfreq(n, d=1 / sampling_rate)
    fft_vals = fft(signal)
    power = np.abs(fft_vals) ** 2

    # Full range for output (0 to max_freq)
    full_mask = (freqs >= 0) & (freqs <= max_freq)
    freqs_full = freqs[full_mask]
    power_full = power[full_mask]

    # Bin frequencies to nearest bin_width (e.g., 0.1 Hz)
    rounded_freqs = np.round(freqs_full / bin_width) * bin_width
    power_binned = {}
    for f, p in zip(rounded_freqs, power_full):
        f_key = f"{f:.1f}Hz"
        power_binned[f_key] = power_binned.get(f_key, 0.0) + float(p)

    # Relative power (only use 1 to 49 Hz for normalization and 4–12 Hz for numerator)
    rel_mask = (freqs >= 1) & (freqs <= 49)
    freqs_rel = freqs[rel_mask]
    power_rel = power[rel_mask]

    total_power = np.sum(power_rel)
    band_power = np.sum([p for f, p in zip(freqs_rel, power_rel) if 4 <= f <= 12])
    rel_power = {'4-12Hz': float(band_power / total_power) if total_power > 0 else np.nan}

    return power_binned, rel_power

def compute_population_rate(spikes_times, spikes_clusters, valid_cluster_ids, start, end, bin_size):
    time_bins = np.arange(start, end, bin_size)
    counts_per_bin = np.zeros(len(time_bins) - 1)
    for cid in valid_cluster_ids:
        spike_times = spikes_times[spikes_clusters == cid]
        counts, _ = np.histogram(spike_times, bins=time_bins)
        counts_per_bin += counts
    avg_firing_rate = counts_per_bin / len(valid_cluster_ids)/bin_size
    return avg_firing_rate


def compute_spiking_metrics(pid, one, region_of_interest, atlas=AllenAtlas(), bin_sizes=[0.01]):
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
    cluster_ids = all_regions[region_of_interest]

    task_start = trials_data['stimOn_times'].min() - 2
    task_end = trials_data['response_times'].max() + 2
    task_mask = (spikes_times >= task_start) & (spikes_times <= task_end)
    spikes_in_task = spikes_clusters[task_mask]
    firing_rates_task = np.array([np.sum(spikes_in_task == cid) / (task_end - task_start) for cid in cluster_ids])
    valid_cluster_ids_task = np.array(cluster_ids)[firing_rates_task >= 1.0]
    total_spikes_task = np.sum(np.isin(spikes_in_task, valid_cluster_ids_task))
    task_pass = len(valid_cluster_ids_task) >= 10 and total_spikes_task >= 400000

    try:
        passive_times = one.load_dataset(session_id, '*passivePeriods*')
        if 'spontaneousActivity' in passive_times:
            sp_start, sp_end = passive_times['spontaneousActivity']
        else:
            sp_start = sp_end = None
    except Exception:
        sp_start = sp_end = None

    sp_pass = False
    valid_cluster_ids_sp = []
    total_spikes_sp = 0
    if sp_start is not None and sp_end - sp_start > 0:
        sp_mask = (spikes_times >= sp_start) & (spikes_times <= sp_end)
        spikes_in_sp = spikes_clusters[sp_mask]
        firing_rates_sp = np.array([np.sum(spikes_in_sp == cid) / (sp_end - sp_start) for cid in cluster_ids])
        valid_cluster_ids_sp = np.array(cluster_ids)[firing_rates_sp >= 1.0]
        total_spikes_sp = np.sum(np.isin(spikes_in_sp, valid_cluster_ids_sp))
        sp_pass = len(valid_cluster_ids_sp) >= 10 and total_spikes_sp >= 100000

    if not task_pass and not sp_pass:
        print(f"Skipped session {session_id}: neither task nor SP passed QC.")
        return None

    results = {
        'session_id': session_id,
        'pid': pid,
        'region': region_of_interest,
        'neuron_count_task': len(valid_cluster_ids_task),
        'neuron_count_sp': len(valid_cluster_ids_sp),
        'bin_sizes': {},
    }

    for bin_size in bin_sizes:
        key = f"{int(bin_size * 1000)}ms"
        sampling_rate = int(1 / bin_size)
        results['bin_sizes'][key] = {'task': {}, 'SP': {}}

        # Task
        if task_pass:
            task_rate = compute_population_rate(spikes_times, spikes_clusters, valid_cluster_ids_task, task_start,
                                                task_end, bin_size)
            power_spec_task, rel_power_task = extract_power_and_relative_power(task_rate, sampling_rate, max_freq=49)
            results['bin_sizes'][key]['task']['power_spectrum'] = power_spec_task
            results['bin_sizes'][key]['task']['relative_power'] = rel_power_task
        else:
            results['bin_sizes'][key]['task']['power_spectrum'] = None
            results['bin_sizes'][key]['task']['relative_power'] = None

        # SP
        if sp_pass:
            sp_rate = compute_population_rate(spikes_times, spikes_clusters, valid_cluster_ids_sp, sp_start, sp_end,
                                              bin_size)
            power_spec_sp, rel_power_sp = extract_power_and_relative_power(sp_rate, sampling_rate, max_freq=49)
            results['bin_sizes'][key]['SP']['power_spectrum'] = power_spec_sp
            results['bin_sizes'][key]['SP']['relative_power'] = rel_power_sp
        else:
            results['bin_sizes'][key]['SP']['power_spectrum'] = None
            results['bin_sizes'][key]['SP']['relative_power'] = None

    return results


ephys_path = "/storage1/fs1/hiratani/Active/shared/ibl_space/ONE/openalyx.internationalbrainlab.org/HMM_Results/ephys_session_info.pkl"
with open(ephys_path, "rb") as f:
    ephys_info = pickle.load(f)

successful_pids_all_regions = ephys_info["successful_pids_ephys"]


n_jobs = 20  #


def process_one_region(region, pid_list):
    print(f"\n Processing region: {region} ({len(pid_list)} PIDs)")
    # behavior_path = os.path.join(behavior_dir, f"{region}_behavior.pkl")
    # if not os.path.exists(behavior_path):
    #     print(f"⚠ Behavior file not found for {region}, skipping.")
    #     return region, []
    #
    # behavior_df = pd.read_pickle(behavior_path)
    # if behavior_df.empty:
    #     print(f"⚠ Empty behavior_df for {region}, skipping.")
    #     return region, []

    region_results = []
    for pid in pid_list:
        try:
            result = compute_spiking_metrics(pid, one, region)
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


# Save all-region results to JSON
output_path = "/storage1/fs1/hiratani/Active/shared/ibl_space/ONE/openalyx.internationalbrainlab.org/HMM_Results/power_spectrum.json"
with open(output_path, "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\n Saved all results to {output_path}")


