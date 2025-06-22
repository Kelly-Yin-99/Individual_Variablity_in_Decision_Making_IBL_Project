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

def compute_spiking_metrics_iti(pid, one, region_of_interest, atlas=AllenAtlas(), bin_sizes=[0.01]):
    session_id, _ = one.pid2eid(pid)
    session_id = str(session_id)

    session_loader = SessionLoader(eid=session_id, one=one)
    session_loader.load_trials()
    trials_data = session_loader.trials

    if trials_data is None or 'stimOn_times' not in trials_data:
        return None

    if len(trials_data) < 400:
        print(f"Session {session_id} skipped: only {len(trials_data)} trials (< 400) before trimming.")
        return None

    # Remove the last 40 trials
    trials_data = trials_data.iloc[:-40].copy()

    # Load spike data
    sl = SpikeSortingLoader(pid=pid, one=one, atlas=atlas)
    spikes, clusters, channels = sl.load_spike_sorting()
    clusters = sl.merge_clusters(spikes, clusters, channels)
    spikes_times = spikes['times']
    spikes_clusters = spikes['clusters']

    # Build ITI intervals based on reward condition
    stim_ons = trials_data['stimOn_times'].values
    feedbacks = trials_data['feedback_times'].values
    rewards = trials_data['rewardVolume'].values

    iti_intervals = []
    for i in range(len(stim_ons) - 1):
        if np.isnan(feedbacks[i]) or np.isnan(stim_ons[i + 1]):
            continue
        iti_start = feedbacks[i] + (1 if rewards[i] > 0 else 2)
        iti_end = stim_ons[i + 1] - 0.5
        if iti_end > iti_start and (iti_end - iti_start) >= 0.5:
            iti_intervals.append((iti_start, iti_end))

    if not iti_intervals:
        print(f"Session {session_id} skipped: no valid ITI intervals.")
        return None

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

    cluster_ids = all_regions[region_of_interest]

    cluster_spike_counts = {cid: 0 for cid in cluster_ids}
    total_iti_duration = 0
    for start, end in iti_intervals:
        duration = end - start
        total_iti_duration += duration
        spike_mask = (spikes_times >= start) & (spikes_times <= end)
        spikes_in_iti = spikes_clusters[spike_mask]
        for cid in cluster_ids:
            cluster_spike_counts[cid] += np.sum(spikes_in_iti == cid)

    valid_cluster_ids = [cid for cid, count in cluster_spike_counts.items()
                         if (count / total_iti_duration) >= 1.0]

    if len(valid_cluster_ids) < 10:
        print(f"Session {session_id} skipped: less than 10 valid clusters during ITI.")
        return None

    # Compute spike count threshold and collect per-ITI segments in parallel
    total_spikes = 0
    iti_segments_by_bin = {f"{int(bin_size * 1000)}ms": [] for bin_size in bin_sizes}
    durations = []

    def process_iti_interval(start, end):
        duration = end - start
        spike_mask = (spikes_times >= start) & (spikes_times <= end)
        spike_count = np.sum(np.isin(spikes_clusters[spike_mask], valid_cluster_ids))
        per_bin_data = []
        for bin_size in bin_sizes:
            fr = compute_population_rate(spikes_times, spikes_clusters, valid_cluster_ids, start, end, bin_size)
            key = f"{int(bin_size * 1000)}ms"
            per_bin_data.append((key, fr, duration))
        return duration, spike_count, per_bin_data

    # Run per-ITI processing in parallel
    iti_results = Parallel(n_jobs=20)(
        delayed(process_iti_interval)(start, end) for start, end in iti_intervals
    )

    for duration, spike_count, segment_data in iti_results:
        total_spikes += spike_count
        durations.append(duration)
        for key, fr, dur in segment_data:
            iti_segments_by_bin[key].append((fr, dur))

    if total_spikes < 200000:
        print(f"Session {session_id} skipped: total spikes in ITI = {total_spikes}")
        return None

    # Compute power spectrum for each bin size
    results = {
        'session_id': session_id,
        'pid': pid,
        'region': region_of_interest,
        'neuron_count_iti': len(valid_cluster_ids),
        'bin_sizes': {}
    }

    for bin_size in bin_sizes:
        key = f"{int(bin_size * 1000)}ms"
        segments = iti_segments_by_bin[key]
        sampling_rate = int(1 / bin_size)

        weighted_power = defaultdict(float)
        total_weight = sum([d for _, d in segments])
        rel_power_vals = []

        for rate, duration in segments:
            p_spec, rel_power = extract_power_and_relative_power(rate, sampling_rate)
            for f, val in p_spec.items():
                weighted_power[f] += val * duration
            rel_power_vals.append((rel_power['4-12Hz'], duration))

        if total_weight > 0:
            for f in weighted_power:
                weighted_power[f] /= total_weight
            rel_4_12 = sum(r * d for r, d in rel_power_vals) / total_weight
        else:
            weighted_power = {}
            rel_4_12 = np.nan

        results['bin_sizes'][key] = {
            'ITI': {
                'power_spectrum': dict(weighted_power),
                'relative_power': {'4-12Hz': rel_4_12}
            }
        }

    return results



def process_one_region(region, pid_list):
    print(f"\n Processing region: {region} ({len(pid_list)} PIDs)")
  

    region_results = []
    for pid in pid_list:
        try:
            result = compute_spiking_metrics_iti(pid, one, region)
            if result:
                region_results.append(result)
        except Exception as e:
            print(f" PID {pid} in {region} failed: {e}")
    return region, region_results





### Load ephys info and define paths, parallel processing all regions,  and save all-region results to json file

ephys_path = "/storage1/fs1/hiratani/Active/shared/ibl_space/ONE/openalyx.internationalbrainlab.org/ephys_session_info.pkl"
with open(ephys_path, "rb") as f:
    ephys_info = pickle.load(f)

successful_pids_all_regions = ephys_info["successful_pids_ephys"]

region_results_list = Parallel(n_jobs=20)(  
    delayed(process_one_region)(region, pid_list)
    for region, pid_list in successful_pids_all_regions.items()
)


all_results = {region: results for region, results in region_results_list if results}
output_path = "/storage1/fs1/hiratani/Active/shared/ibl_space/ONE/openalyx.internationalbrainlab.org/power_specturm_iti.json"
with open(output_path, "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\n Saved all results to {output_path}")




