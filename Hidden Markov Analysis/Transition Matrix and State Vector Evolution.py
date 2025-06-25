import ssm
from one.api import ONE

import traceback
from brainbox.io.one import SessionLoader
import matplotlib.pyplot as plt
from brainbox.io.one import SpikeSortingLoader
from iblatlas.atlas import AllenAtlas

atlas = AllenAtlas()

ONE.setup(base_url='https://openalyx.internationalbrainlab.org', silent=True)


# one = ONE(password='international')


def compute_HMM_wrapper(args):
    pid, region_of_interest = args
    ONE.setup(base_url='https://openalyx.internationalbrainlab.org', silent=True)
    one = ONE(password='international')
    result = process_task((pid, region_of_interest), one)
    return result

def compute_task_state_vectors(trials_data, spikes, valid_cluster_ids):
    num_neurons = len(valid_cluster_ids)
    cluster_id_to_index = {cid: i for i, cid in enumerate(valid_cluster_ids)}

    def get_spike_counts(start, end):
        if np.isnan(start) or np.isnan(end):
            return None  # NaN，
        mask = (spikes['times'] >= start) & (spikes['times'] < end)
        times = spikes['times'][mask]
        clusters = spikes['clusters'][mask]
        counts = np.zeros(num_neurons)
        for t, c in zip(times, clusters):
            if c in cluster_id_to_index:
                counts[cluster_id_to_index[c]] += 1
        duration = end - start
        return counts / duration

    vectors = {key: [] for key in ['Hold', 'CL', 'CR', 'Reward', 'Miss', 'Null']}

    task_start = trials_data['stimOn_times'].min() - 2
    task_end = trials_data['feedback_times'].max() + 2

    for _, trial in trials_data.iterrows():
        stim_on = trial['stimOn_times']
        feedback = trial['feedback_times']
        reward = trial['rewardVolume']
        choice = trial['choice']
        fb_duration = feedback - stim_on

        # Hold
        hold = get_spike_counts(stim_on - 0.55, stim_on)
        if hold is not None:
            vectors['Hold'].append(hold)

        # Correct Left: choice == -1, reward > 0, feedback - stimOn <= 1
        # Correct Right: choice == 1, reward > 0, feedback - stimOn <= 1
        if reward > 0 and fb_duration <= 1:
            if choice == -1:
                cl = get_spike_counts(stim_on, feedback)
                if cl is not None:
                    vectors['CL'].append(cl)
            elif choice == 1:
                cr = get_spike_counts(stim_on, feedback)
                if cr is not None:
                    vectors['CR'].append(cr)

        # Reward
        if reward > 0:
            rw = get_spike_counts(feedback, feedback + 1)
            if rw is not None:
                vectors['Reward'].append(rw)

        # Miss
        if reward == 0:
            miss = get_spike_counts(feedback, feedback + 1)
            if miss is not None:
                vectors['Miss'].append(miss)

    # Null
    null = get_spike_counts(task_start, task_end)
    if null is not None:
        vectors['Null'].append(null)

    return {k: np.mean(v, axis=0) if v else np.zeros(num_neurons) for k, v in vectors.items()}

# V_Hold : = 1/N_total sum_t Vt[-0.55 <= Tt <= 0]
# V_CL : = 1/N_CL sum_{correct left trials, feedback time <=1} Vt[0 <= Tt <= T[‘feedback_times'] ]
# V_CR : = 1/N_CR sum_{correct right trials, feedback time <=1} Vt[0 <= Tt <= T[‘feedback_times'] ]
# V_Reward = 1/N_rewarded_trials sum_{correct trials} Vt[ trials_data['feedback_times'] <= Tt <= trials_data['feedback_times'] +1]
# V_Miss = 1/N_incorrect_trials sum_{incorrect trials} Vt[ trials_data['feedback_times'] <= Tt <= trials_data['feedback_times'] +1]
# V_Null = 1/N_total sum_t Vt[task period]



def initialize_hmm_with_task_vectors(K, num_neurons, state_vectors):
    hmm = ssm.HMM(K, num_neurons, observations="poisson")

    # Ensure the 6 known task-based states
    ordered_states = ['Hold', 'CL', 'CR', 'Reward', 'Miss', 'Null']
    init_rates = []
    for state in ordered_states:
        if state in state_vectors:
            init_rates.append(state_vectors[state])
        else:
            init_rates.append(np.random.rand(num_neurons))  # fallback for missing

    init_rates = np.clip(np.stack(init_rates), 1e-3, None)  # avoid zeros
    hmm.observations.params = init_rates.copy()

    return hmm,ordered_states


def compute_HMM(pid, one, region_of_interest, atlas=AllenAtlas()):
    try:
        # Load passive times dataset
        session_id, pname = one.pid2eid(pid)
        session_id = str(session_id)
        passive_times = one.load_dataset(session_id, '*passivePeriods*', collection='alf')
        if 'spontaneousActivity' not in passive_times:
            print(f"Spontaneous activity periods not found in session {session_id}.")
            return None

        SP_times = passive_times['spontaneousActivity']

        session_loader = SessionLoader(eid=session_id, one=one)
        session_loader.load_trials()
        trials_data = session_loader.trials

        if trials_data is None or 'stimOn_times' not in trials_data:
            print(f"No trials data found for session {session_id}.")
            return None

        # Load spike sorting data
        # pid = one.alyx.rest('insertions', 'list', session=session_id)[1]['id']
        sl = SpikeSortingLoader(pid=pid, one=one, atlas=atlas)
        spikes, clusters, channels = sl.load_spike_sorting()
        clusters = sl.merge_clusters(spikes, clusters, channels)

        good_cluster_ids = clusters['cluster_id'][clusters['label'] >= 0]
        combined_acronyms = {
            'SS': [], 'AUD': [], 'VIS': [],
            'MOp': [], 'MOs': [], 'ACA': [], 'PL': [], 'ORB': [],
            'VISa': [], 'RSP': [],
            'DG': [], 'CA': [], 'SUB': [], 'ENT': [],
            'SCs': [], 'SCm': [], 'IC': [], 'AI': []
        }

        # Filter clusters by region and good quality
        all_regions = {}
        for cluster_id, acronym in zip(clusters['cluster_id'], clusters['acronym']):
            if acronym.startswith('SSs') or acronym.startswith('SSp'):
                combined_acronyms['SS'].append(acronym)
            elif acronym.startswith('AUD'):
                combined_acronyms['AUD'].append(acronym)
            elif acronym.startswith('VISp') or acronym.startswith('VISl') or acronym.startswith(
                    'VISli') or acronym.startswith('VISam'):
                combined_acronyms['VIS'].append(acronym)
            elif acronym.startswith('AIp') or acronym.startswith('AIv') or acronym.startswith(
                    'AId'):
                combined_acronyms['AI'].append(acronym)
            elif acronym.startswith('MOp'):
                combined_acronyms['MOp'].append(acronym)
            elif acronym.startswith('MOs'):
                combined_acronyms['MOs'].append(acronym)
            elif acronym.startswith('ACA'):
                combined_acronyms['ACA'].append(acronym)
            elif acronym.startswith('PL'):
                combined_acronyms['PL'].append(acronym)
            elif acronym.startswith('ORB'):
                combined_acronyms['ORB'].append(acronym)
            elif acronym.startswith('RSPv') or acronym.startswith('RSPd'):
                combined_acronyms['RSP'].append(acronym)
            elif acronym.startswith('VISa') or acronym.startswith('VISrl'):
                combined_acronyms['VISa'].append(acronym)
            elif acronym.startswith('CA1') or acronym.startswith('CA2') or acronym.startswith('CA3'):
                combined_acronyms['CA'].append(acronym)
            elif acronym.startswith('DG'):
                combined_acronyms['DG'].append(acronym)
            elif acronym.startswith('SUB') or acronym.startswith('PRE') or acronym.startswith('POST'):
                combined_acronyms['SUB'].append(acronym)
            elif acronym.startswith('ENT'):
                combined_acronyms['ENT'].append(acronym)
            elif acronym.startswith('SCop') or acronym.startswith('SCsg') or acronym.startswith('SCzo'):
                combined_acronyms['SCs'].append(acronym)
            elif acronym.startswith('SCi') or acronym.startswith('SCd'):
                combined_acronyms['SCm'].append(acronym)
            elif acronym.startswith('IC'):
                combined_acronyms['IC'].append(acronym)
            else:
                if acronym not in all_regions:
                    all_regions[acronym] = []
                all_regions[acronym].append(cluster_id)

        for combined_acronym, acronyms in combined_acronyms.items():
            if acronyms:
                combined_cluster_ids = []
                for acronym in acronyms:
                    combined_cluster_ids.extend(clusters['cluster_id'][(clusters['acronym'] == acronym) & (
                        np.isin(clusters['cluster_id'], good_cluster_ids))])
                if combined_cluster_ids:
                    all_regions[combined_acronym] = list(set(combined_cluster_ids))

        # Check if region exists in all_regions
        if region_of_interest not in all_regions:
            print(f"Region {region_of_interest} not found in session {session_id}.")
            return None

        # Extract and validate cluster IDs
        cluster_ids = all_regions[region_of_interest]
        mask = (spikes['times'] >= SP_times[0]) & (spikes['times'] <= SP_times[1])
        spikes_in_SP = spikes['clusters'][mask]
        firing_rates = np.array([np.sum(spikes_in_SP == cid) / (SP_times[1] - SP_times[0]) for cid in cluster_ids])
        valid_cluster_ids = np.array(cluster_ids)[firing_rates >= 0.1]
        print(f"Valid clusters (>= 0.1 Hz): {len(valid_cluster_ids)}")

        if len(valid_cluster_ids) < 10:
            print(
                f"Not enough neurons with firing rate >= 10 Hz in region {region_of_interest} for session {session_id}.")
            return None

        # Preselect spikes during task intervals (ITI)

        cluster_id_to_index = {cluster_id: i for i, cluster_id in enumerate(valid_cluster_ids)}

        SP_times = SP_times.copy()
        SP_idx = np.where((spikes['times'] >= SP_times[0]) & (spikes['times'] <= SP_times[1]))[0]
        region_spikes_SP_idx = np.isin(spikes['clusters'][SP_idx], valid_cluster_ids)
        region_spike_times_SP = spikes['times'][SP_idx][region_spikes_SP_idx]

        bin_size = 0.2
        bins_SP = np.arange(SP_times[0], SP_times[1] + bin_size, bin_size)
        spike_matrix_SP = np.zeros((len(valid_cluster_ids), len(bins_SP) - 1))

        for i, cluster_id in enumerate(valid_cluster_ids):
            neuron_spike_times_SP = spikes['times'][(spikes['clusters'] == cluster_id) &
                                                    (spikes['times'] >= SP_times[0]) & (spikes['times'] <= SP_times[1])]
            spike_counts, _ = np.histogram(neuron_spike_times_SP, bins_SP)
            spike_matrix_SP[i, :] = spike_counts

        # print(np.sum(spike_matrix_SP))
        spike_matrix_SP = spike_matrix_SP.astype(np.int32)
        # print(np.sum(spike_matrix_SP))
        # Set maximum number of HMM states
        max_states = min(len(valid_cluster_ids), 100)
        # hidden_states_range = range(2, max_states + 1)
        hidden_states_range = range(2, 6)
        num_folds = 5
        num_time_bins = spike_matrix_SP.shape[1]
        num_workers = 5

        task_vectors = compute_task_state_vectors(trials_data, spikes, valid_cluster_ids)
        hmm, state_labels = initialize_hmm_with_task_vectors(6, len(valid_cluster_ids), task_vectors)
        # Freeze emissions if desired
        #hmm.observations.params_constrained = lambda: (hmm.observations.params,)
        #hmm.observations.m_step = lambda *args, **kwargs: None

        # Fit full spike matrix
        hmm.fit(spike_matrix_SP.T, method="em", num_restarts=10, num_iters=200,
                                 return_log_likelihoods=True)

        transition_matrix = hmm.transitions.transition_matrix
        state_probabilities = hmm.expected_states(spike_matrix_SP.T)[0]
        state_evolution = state_probabilities.T
        state_occurrences = state_probabilities.sum(axis=0)
        state_representations = hmm.observations.params

        eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
        # Compute stationary distribution
        stationary_distribution = np.real(eigenvectors[:, np.isclose(eigenvalues, 1)].flatten())
        stationary_distribution /= stationary_distribution.sum()  # Normalize to sum to 1

        participation_ratio = (np.sum(stationary_distribution) ** 2) / np.sum(stationary_distribution ** 2)
        print(f"Participation Ratio: {participation_ratio}")
        # Compute expected duration (D_i) for each state

        task_start_time = trials_data['stimOn_times'].min() - 1
        task_end_time = trials_data['response_times'].max() + 2
        task_idx = np.where((spikes['times'] >= task_start_time) & (spikes['times'] <= task_end_time))[0]
        region_spikes_idx = np.isin(spikes['clusters'][task_idx], valid_cluster_ids)

        if not np.any(region_spikes_idx):
            print(f"No spikes in region {region_of_interest} during task period for session {session_id}.")
            return None

        region_spike_times = spikes['times'][task_idx][region_spikes_idx]
        region_spike_clusters = spikes['clusters'][task_idx][region_spikes_idx]

        rL, rR = compute_rL_rR(trials_data, valid_cluster_ids, region_spike_times, region_spike_clusters)
        print("successfully computed rL and rR")
        mean_firing_rate = np.sum(spike_matrix_SP) / (SP_times[1] - SP_times[0])
        best_K = 6
        return (
            np.sum(spike_matrix_SP),  # Total spike count
            len(valid_cluster_ids),  # Number of neurons
            participation_ratio,
            transition_matrix,
            stationary_distribution,
            state_representations,
            rL,
            rR,
            mean_firing_rate,
            state_evolution,
            state_occurrences,
            state_labels,
            best_K
        )
    except Exception as e:
        print(f"Error processing session {'session_id'}: {e}")
        traceback.print_exc()

        return None


def compute_rL_rR(valid_trials, cluster_ids, region_spike_times, region_spike_clusters):
    num_neurons = len(cluster_ids)
    rL_total = np.zeros(num_neurons)
    rR_total = np.zeros(num_neurons)

    for _, trial in valid_trials.iterrows():
        trial_start_time = trial['stimOn_times']
        trial_end_time = trial['response_times'] + 1 if pd.isna(trial['stimOff_times']) else trial['stimOff_times']
        feedback_time = trial_end_time - trial_start_time
        if feedback_time <= 0:
            continue

        mask = (region_spike_times >= trial_start_time) & (region_spike_times <= trial_end_time)
        trial_spike_times = region_spike_times[mask]
        trial_spike_clusters = region_spike_clusters[mask]

        if len(trial_spike_times) == 0:
            continue

        rL = np.zeros(num_neurons)
        rR = np.zeros(num_neurons)

        for neuron_idx, neuron_id in enumerate(cluster_ids):
            neuron_mask = trial_spike_clusters == neuron_id
            spike_count = np.sum(neuron_mask)
            if trial['choice'] == -1:
                rL[neuron_idx] += spike_count / feedback_time
            elif trial['choice'] == 1:
                rR[neuron_idx] += spike_count / feedback_time

        rL_total += rL
        rR_total += rR

    return rL_total, rR_total


from sklearn.metrics.pairwise import cosine_similarity

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd


def process_task(task, one):
    pid, region = task
    print(f"🛠 process_task received task: {task}")  # Debug print

    results = compute_HMM(pid, one, region)
    base_output_dir = "HMM_Results1"

    print(f"🔍 process_task: compute_HMM returned: {results}")  # Debug print

    if results is None:
        print(f"⚠️ process_task: No valid results for pid {pid}, region {region}")
        return None

    (
        total_spike_count_sp, num_neurons, participation_ratio,
        transition_matrix, stationary_distribution, state_representations,rL, rR,
        mean_firing_rate, state_evolution, state_occurrences, state_labels, best_K) = results

    dij_df = compute_state_distances(state_representations, pid, mean_firing_rate)

    # Convert dij_df into matrices matching the transition matrix shape
    dij_matrix = np.zeros((best_K, best_K))  # Initialize square matrix (KxK)
    sij_matrix = np.zeros((best_K, best_K))

    for _, row in dij_df.iterrows():
        i, j = int(row['state_i']), int(row['state_j'])  # State indices
        dij_matrix[i, j] = row['dij']
        sij_matrix[i, j] = row['sij']

    # Compute d_bar and s_bar using the correct summation formula
    d_bar = np.sum(stationary_distribution[:, None] * transition_matrix * dij_matrix)
    s_bar = np.sum(stationary_distribution[:, None] * transition_matrix * sij_matrix)

    # Compute distances to rL and rR
    state_distances = []
    for i, ri in enumerate(state_representations):
        d_euclid_L = np.linalg.norm(ri - rL) / (np.sqrt(num_neurons) * mean_firing_rate)
        d_euclid_R = np.linalg.norm(ri - rR) / (np.sqrt(num_neurons) * mean_firing_rate)
        cos_L = cosine_similarity([ri], [rL])[0][0]
        cos_R = cosine_similarity([ri], [rR])[0][0]
        state_distances.append({
            'session_id': pid, 'state_id': i,
            'euclidean_distance_L': d_euclid_L,
            'euclidean_distance_R': d_euclid_R,
            'cos_distance_L': cos_L, 'cos_distance_R': cos_R
        })

    ri_trial_df = pd.DataFrame(state_distances)

    results_dict = {
        'pid': pid, 'region': region,
        'neuron_count': num_neurons,
        'total_spike_counts_sp': total_spike_count_sp,
        'number_of_hidden_states': best_K,
        'participation_ratio': participation_ratio,
        'd_bar': d_bar,
        's_bar': s_bar,
        'dij_df': dij_df,
        'ri_trial_df': ri_trial_df,
        'transition_matrix': transition_matrix,
        'state_evolution': state_evolution,  # shape: (K x T)
        'time_bins': np.arange(state_evolution.shape[1]),
        'state_occurrences': state_occurrences,
        'state_labels': state_labels,
    }

    save_results_and_plots(results_dict, base_output_dir)

    return results_dict


def save_results_and_plots(result, base_output_dir):
    """Save plots and results to region-specific directories."""

    region_name = f"{result['region']}_plot_HMM"
    region_dir = os.path.join(base_output_dir, region_name)
    os.makedirs(region_dir, exist_ok=True)

    pid = result['pid']
    transition_matrix = result['transition_matrix']
    state_labels = result.get('state_labels', [f"State {i}" for i in range(result['number_of_hidden_states'])])

    best_K = result['number_of_hidden_states']
    state_evolution = result['state_evolution']
    time_bins = result['time_bins']
    T = state_evolution.shape[1]
    start_idx = T // 3
    end_idx = min(start_idx + 200, T)

    state_evolution = state_evolution[:, start_idx:end_idx]
    time_bins = time_bins[start_idx:end_idx]

    # Plot  State Vector Evolution
    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(state_evolution.shape[0]):
        label = state_labels[i] if i < len(state_labels) else f"State {i}"
        ax.plot(time_bins, state_evolution[i, :], label=label)
    ax.set_title(f"State Vector Evolution for {pid}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Probability")
    ax.legend()
    ax.grid()
    save_plot(fig, region_dir, f"{pid}_state_vector_evolution.png")

    # Plot  State Occurrences
    state_occurrences = result['state_occurrences']

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(range(best_K), state_occurrences)
    ax.set_title(f"State Occurrences for {pid}")
    ax.set_xlabel("State")
    ax.set_ylabel("Expected Time Bins in State")
    ax.set_xticks(range(best_K))
    ax.set_xticklabels(state_labels[:best_K])
    save_plot(fig, region_dir, f"{pid}_state_occurrences.png")

    # Plot  Transition Matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(transition_matrix, cmap='coolwarm', aspect='auto')
    fig.colorbar(cax, label="Transition Probability")
    ax.set_title(f"Transition Matrix for {pid}")
    ax.set_xlabel("To State")
    ax.set_ylabel("From State")
    ax.set_xticks(range(best_K))
    ax.set_yticks(range(best_K))
    ax.set_xticklabels(state_labels[:best_K])
    ax.set_yticklabels(state_labels[:best_K])
    save_plot(fig, region_dir, f"{pid}_transition_matrix.png")


def save_plot(fig, outdir, filename):
    os.makedirs(outdir, exist_ok=True)
    filepath = os.path.join(outdir, filename)
    fig.savefig(filepath, dpi=300)
    plt.close(fig)


def compute_cos_rho_self(transition_matrix, stationary_distribution, state_representations):
    """
    Computes the cos_rho_self using the formula: sum_i stationary_distribution_i * sum_j T_ij * cos(r_i, r_j)
    """
    cos_sim = cosine_similarity(state_representations)
    return np.sum(stationary_distribution[:, None] * transition_matrix * cos_sim)


def compute_state_distances(state_representations, session_id, mean_firing_rate):
    num_states, num_neurons = state_representations.shape
    distance_records = []
    cos_sim_matrix = cosine_similarity(state_representations)

    for i in range(num_states):
        for j in range(i + 1, num_states):  # Only store unique dij values
            l1_norm = np.sum(np.abs(state_representations[i] - state_representations[j]))
            dij = (l1_norm / np.sqrt(num_neurons)) / mean_firing_rate
            sij = cos_sim_matrix[i, j]
            distance_records.append(
                {'session_id': session_id, 'state_i': i, 'state_j': j, 'dij': dij, 'sij': sij})  # Fix

    dij_df = pd.DataFrame(distance_records)
    return dij_df


import os
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed


def plot_HMM_parallel(regions_of_interest, successful_pids, base_output_dir="HMM_Results1", num_workers=4):
    tasks = [(pid, region) for region in regions_of_interest for pid in successful_pids]
    print(f"Processing {len(tasks)} tasks in parallel with {num_workers} workers...")

    # Initialize DataFrames to store results for each region
    region_dfs = {region: pd.DataFrame(columns=[
        'pid', 'region', 'neuron_count',
        'total_spike_counts_sp', 'number_of_hidden_states',
        'participation_ratio', 'd_bar', 's_bar'  # Added d_bar and s_bar
    ]) for region in regions_of_interest}

    # Initialize DataFrames for dij and ri_trial values per region
    region_dij_dfs = {region: pd.DataFrame(columns=['session_id', 'state_i', 'state_j', 'dij', 'sij']) for region in
                      regions_of_interest}
    region_ri_trial_dfs = {region: pd.DataFrame(columns=[
        'pid', 'state_id', 'euclidean_distance_L', 'euclidean_distance_R',
        'cos_distance_L', 'cos_distance_R'
    ]) for region in regions_of_interest}

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_task = {executor.submit(compute_HMM_wrapper, task): task for task in tasks}

        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                result = future.result()
                print(f" Task {task} returned: {result}")  # Debug print

                if result is None:
                    print(f"️ Skipping task {task}: Result is None")
                    continue

                region = result['region']

                # **Separate summary data from dij and ri_trial data**
                summary_data = {k: v for k, v in result.items() if
                                k not in ['dij_df', 'ri_trial_df']}  # Exclude dataframes
                df_entry = pd.DataFrame([summary_data])

                # Append to summary DataFrame
                region_dfs[region] = pd.concat([region_dfs[region], df_entry], ignore_index=True)

                # Append `dij` values separately
                if 'dij_df' in result and isinstance(result['dij_df'], pd.DataFrame):
                    region_dij_dfs[region] = pd.concat([region_dij_dfs[region], result['dij_df']], ignore_index=True)

                # Append `ri_trial_df` separately
                if 'ri_trial_df' in result and isinstance(result['ri_trial_df'], pd.DataFrame):
                    region_ri_trial_dfs[region] = pd.concat([region_ri_trial_dfs[region], result['ri_trial_df']],
                                                            ignore_index=True)

            except Exception as e:
                print(f"Error processing task {task}: {e}")

    # **Save summary results**
    os.makedirs(base_output_dir, exist_ok=True)
    for region, df in region_dfs.items():
        print(f"📁 Saving summary for region {region}: {df.shape[0]} rows")

        if df.empty:
            print(f"Warning: No summary data to save for region {region}!")

        region_csv_path = os.path.join(base_output_dir, f"summary_{region}.csv")
        df.to_csv(region_csv_path, index=False)
        print(f"Summary for region '{region}' saved to {region_csv_path}")

    # **Save dij data**
    for region, df in region_dij_dfs.items():
        print(f"Saving dij data for region {region}: {df.shape[0]} rows")

        if df.empty:
            print(f" Warning: No dij data to save for region {region}!")

        dij_csv_path = os.path.join(base_output_dir, f"dij_{region}.csv")
        df.to_csv(dij_csv_path, index=False)
        print(f" Dij data for region '{region}' saved to {dij_csv_path}")

    # **Save ri_trial data**
    for region, df in region_ri_trial_dfs.items():
        print(f"Saving ri_trial data for region {region}: {df.shape[0]} rows")

        if df.empty:
            print(f" Warning: No ri_trial data to save for region {region}!")

        ri_trial_csv_path = os.path.join(base_output_dir, f"ri_trial_{region}.csv")
        df.to_csv(ri_trial_csv_path, index=False)
        print(f" ri_trial data for region '{region}' saved to {ri_trial_csv_path}")

    return region_dfs, region_dij_dfs, region_ri_trial_dfs


if __name__ == "__main__":
    ONE.setup(base_url='https://openalyx.internationalbrainlab.org', silent=True)
    # one = ONE(password='international')

    successful_pids_AI = ['08ed0b3c-9f94-4c1f-8522-3d42a642a6b0',
 '41ea689a-7e05-463b-915a-90e63b8c0572',
 '31d8dfb1-71fd-4c53-9229-7cd48bee07e4',
 '8d661567-49f3-4547-997d-a345c0ffe2dd',
 '49a86b2e-3db4-42f2-8da8-7ebb7e482c70',
 'ecd07b7e-6450-4e31-bef1-f195129eb3d3',
 '2c401cbd-b95c-4ebe-8a83-92b55ab49542',
 'e5bd461c-a713-4b3c-b165-a132a711e59d',
 '310c60b6-d68f-4018-a86a-3668ce296837',
 '7b05cccc-44f6-4491-a0ea-e38d6e95513d',
 'f8249eab-d4c2-4fb0-a258-7f4b99fa19e4',
 'da17779f-7728-4c51-8b54-b88764c0908d',
 'b1455ca8-3999-4eb8-9e1e-5c8d0d2e5719',
 'd23dec01-b35e-4950-9afa-72a1b3e74148',
 'b98f6b89-3de4-4295-be1c-59e465de1e32',
 '98bde8a6-6b6a-4e53-96b1-9078c3974a97',
 'eb7e9f3f-b79d-4cdd-bc24-b13a4008c1b5']
    regions_of_interest_AI = ['AI']
    HMM_ACA = plot_HMM_parallel(regions_of_interest_AI, successful_pids_AI, base_output_dir="HMM_Results1",
                                num_workers=4)


