
from pathlib import Path
import numpy as np
import pandas as pd
import ssm
from one.api import ONE
from sklearn.model_selection import KFold
from iblatlas.atlas import AllenAtlas
import traceback
from brainbox.io.one import SessionLoader
from scipy.optimize import curve_fit
from brainbox.io.one import SpikeSortingLoader
from iblatlas.atlas import AllenAtlas
atlas = AllenAtlas()
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed


ONE.setup(base_url='https://openalyx.internationalbrainlab.org', silent=True,
          cache_dir=Path('/storage1/fs1/hiratani/Active/shared/ibl_space/ONE/openalyx.internationalbrainlab.org'))
one = ONE(base_url='https://openalyx.internationalbrainlab.org', password=os.getenv('ALYX_PASSWORD'))


def compute_HMM_wrapper(args):
    pid, region_of_interest = args
    ONE.setup(base_url='https://openalyx.internationalbrainlab.org', silent=True,
              cache_dir=Path('/storage1/fs1/hiratani/Active/shared/ibl_space/ONE/openalyx.internationalbrainlab.org'))
    one = ONE(base_url='https://openalyx.internationalbrainlab.org', password=os.getenv('ALYX_PASSWORD'))
    result = process_task((pid,region_of_interest))
    return result

def train_fold_wrapper(args):
    fold_idx, train_indices, test_indices, spike_matrix_SP, valid_cluster_ids, hidden_states_range = args
    return train_fold(fold_idx, train_indices, test_indices, spike_matrix_SP, valid_cluster_ids, hidden_states_range)


def train_fold(fold_idx, train_indices, test_indices, spike_matrix_SP, valid_cluster_ids, hidden_states_range):
    print(f"Processing Fold {fold_idx + 1}...", flush=True)

    train_data = spike_matrix_SP[:, train_indices]
    test_data = spike_matrix_SP[:, test_indices]

    log_likelihood_result = {}

    for K in hidden_states_range:
    
        hmm = ssm.HMM(K, len(valid_cluster_ids), observations="poisson")
        hmm.fit(train_data.T, method="em", num_restarts=10, num_iters=100)
        log_likelihood_test = hmm.log_likelihood(test_data.T)
        log_likelihood_result[K] = log_likelihood_test

    return log_likelihood_result



def compute_HMM(pid, one, region_of_interest, result_df, atlas=AllenAtlas()):
    try:
        session_id, pname = one.pid2eid(pid)
        session_id = str(session_id)
        
        passive_times = one.load_dataset(session_id, '*passivePeriods*')
        if 'spontaneousActivity' not in passive_times:
            print(f"Spontaneous activity periods not found in session {session_id}.")
            return None
        
        SP_times = passive_times['spontaneousActivity']
        
        session_loader = SessionLoader(eid=session_id, one=one)
        session_loader.load_trials()
        trials_data = session_loader.trials

        if trials_data is None:
            print(f"No trials data found for session {session_id}.")
            return None

        sl = SpikeSortingLoader(pid=pid, one=one, atlas=atlas)
        spikes, clusters, channels = sl.load_spike_sorting()

        if not clusters or not spikes:
                print(f"No clusters found for session {session_id}, pid {pid}")
                return None
            
        clusters = sl.merge_clusters(spikes, clusters, channels)
        good_cluster_ids = clusters['cluster_id'][clusters['label'] >= 0]
            
        combined_acronyms = {
                    'SS': [],'AUD': [],
                    'VIS':[],'VISC':[],'VISa':[],'VISp': [],'VISa': [],'VISal': [],'VISl': [],
                    'VISli': [],'VISlla':[],'VISm':[],'VISmma':[],'VISmmp':[],'VISp':[],'VISpl':[],'VISpm':[],'VISpor':[],
                    'VISrl':[],'VISrll':[],
                    'MOp': [],'MOs': [],'ACA': [],'PL': [],'ORB': [],
                    'RSP': [],'CA1': [],'DG': [],'SUB': [],'ENT': [],'SCs': [],'SCm': [], 'IC': [],
                    'LP':[],'PO':[]
        }
            
        all_regions = {}
            
        for cluster_id, acronym in zip(clusters['cluster_id'], clusters['acronym']):
                
                if acronym.startswith('SSs') or acronym.startswith('SSp'):
                    combined_acronyms['SS'].append(acronym)
                elif acronym.startswith('AUD'):
                    combined_acronyms['AUD'].append(acronym)
                elif acronym.startswith('VIS') and not acronym.startswith('VISa') and not acronym.startswith('VISC')and not acronym.startswith('VISl')and not acronym.startswith('VISm')and not acronym.startswith('VISp')and not acronym.startswith('VISr'):
                    combined_acronyms['VIS'].append(acronym)
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
                elif acronym.startswith('CA1'):
                    combined_acronyms['CA1'].append(acronym)
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
                elif acronym.startswith('LP') and len(acronym) == 2:
                    combined_acronyms['LP'].append(acronym)
                elif acronym.startswith('PO') and len(acronym) == 2:
                    combined_acronyms['PO'].append(acronym)
                elif acronym.startswith('VISa') and not acronym.startswith('VISal'):
                    combined_acronyms['VISa'].append(acronym)
                elif acronym.startswith('VISal'):
                    combined_acronyms['VISal'].append(acronym)
                elif acronym.startswith('VISl') and not acronym.startswith('VISli')and not acronym.startswith('VISlla'):
                    combined_acronyms['VISl'].append(acronym)
                elif acronym.startswith('VISli'):
                    combined_acronyms['VISli'].append(acronym)
                elif acronym.startswith('VISlla'):
                    combined_acronyms['VISlla'].append(acronym)
                elif acronym.startswith('VISm') and not acronym.startswith('VISmma') and not acronym.startswith('VISmmp'):
                    combined_acronyms['VISm'].append(acronym)
                elif acronym.startswith('VISmma'):
                    combined_acronyms['VISmma'].append(acronym)
                elif acronym.startswith('VISmmp'):
                    combined_acronyms['VISmmp'].append(acronym)
                elif acronym.startswith('VISp') and not acronym.startswith('VISpm') and not acronym.startswith('VISpl') and not acronym.startswith('VISpo'):
                    combined_acronyms['VISp'].append(acronym)
                elif acronym.startswith('VISpm'):
                    combined_acronyms['VISpm'].append(acronym)
                elif acronym.startswith('VISpl'):
                    combined_acronyms['VISpl'].append(acronym)
                elif acronym.startswith('VISpor'):
                    combined_acronyms['VISpor'].append(acronym)
                elif acronym.startswith('VISrl') and not acronym.startswith('VISrll'):
                    combined_acronyms['VISrl'].append(acronym)
                elif acronym.startswith('VISrll'):
                    combined_acronyms['VISrll'].append(acronym)
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

        if region_of_interest not in all_regions:
            print(f"Region {region_of_interest} not found in session {session_id}, pid {pid}.")
            return None
        
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

        total_spike_counts = np.sum(spike_matrix_SP, axis=1)  
        duration_SP = SP_times[1] - SP_times[0] 
        neuron_mean_firing_rates = total_spike_counts / duration_SP
        spike_matrix_SP = spike_matrix_SP.astype(np.int32)
   
        max_states = min(len(valid_cluster_ids), 100)
        hidden_states_range = range(2, max_states + 1)

        num_folds =5
        num_time_bins = spike_matrix_SP.shape[1]
        num_workers = 5

        
        kf = KFold(n_splits=num_folds, shuffle=False)
        folds = [
            (
                fold_idx,
                train_indices,
                test_indices,
                spike_matrix_SP,
                valid_cluster_ids,
                hidden_states_range
            )
            for fold_idx, (train_indices, test_indices) in enumerate(kf.split(spike_matrix_SP.T))
        ]

    
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            fold_results = list(executor.map(train_fold_wrapper, folds))

       
        avg_log_likelihoods = {K: 0 for K in hidden_states_range}
        for fold_result in fold_results:
            for K, log_likelihood in fold_result.items():
                avg_log_likelihoods[K] += log_likelihood
        avg_log_likelihoods = {K: ll / num_folds for K, ll in avg_log_likelihoods.items()}
        best_K = max(avg_log_likelihoods, key=avg_log_likelihoods.get)
        final_hmm = ssm.HMM(best_K, len(valid_cluster_ids), observations="poisson")
        final_hmm.fit(spike_matrix_SP.T, method="em", num_restarts=10, num_iters=200)

        transition_matrix = final_hmm.transitions.transition_matrix

        if final_hmm is None:
            print(f"HMM fitting failed for session {session_id}, pid {pid}, region {region_of_interest}.")
            return None

        print(f"Best K based on highest log-likelihood: {best_K}")

        state_probabilities = final_hmm.expected_states(spike_matrix_SP.T)[0]
        state_evolution = state_probabilities.T  
        state_occurrences = state_probabilities.sum(axis=0)

        eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
        stationary_distribution = np.real(eigenvectors[:, np.isclose(eigenvalues, 1)].flatten())
        stationary_distribution /= stationary_distribution.sum()
        
        participation_ratio = (np.sum(stationary_distribution) ** 2) / np.sum(stationary_distribution ** 2)
        #print(f"Participation Ratio: {participation_ratio}")

        absorbing_states = np.isclose(np.diag(transition_matrix), 1)
        num_absorbing_states = np.sum(absorbing_states)  
        eps = 1e-8  
        expected_durations = np.where(
            absorbing_states, 
            np.inf,  
            1 / np.maximum(eps, 1 - np.diag(transition_matrix))  
        )

    
        finite_mask = np.isfinite(expected_durations)
        if np.sum(finite_mask) == 0: 
            rho_self = np.nan 
        else:
            rho_self = np.sum(stationary_distribution[finite_mask] * expected_durations[finite_mask])
    
        print(f"HMM Stickiness (ρ_self): {rho_self}")
        #print(f"Number of absorbing states: {num_absorbing_states}") 

        
        log_state_representations = final_hmm.observations.params
        state_representations = np.exp(log_state_representations)
        cos_rho_self = compute_cos_rho_self(transition_matrix, stationary_distribution, state_representations)
        print(f"cos rho-self : {cos_rho_self}")

        task_start_time = trials_data['stimOn_times'].min() - 1
        task_end_time = trials_data['response_times'].max() + 2
        task_idx = np.where((spikes['times'] >= task_start_time) & (spikes['times'] <= task_end_time))[0]
        region_spikes_idx = np.isin(spikes['clusters'][task_idx], valid_cluster_ids)

        if not np.any(region_spikes_idx):
            print(f"No spikes in region {region_of_interest} during task period for session {session_id}.")
            return None

        region_spike_times = spikes['times'][task_idx][region_spikes_idx]
        region_spike_clusters = spikes['clusters'][task_idx][region_spikes_idx]

        valid_trials = result_df[(result_df['reaction_time'] >= 0.1) &
                                 (result_df['feedback_time'] <= 1) &
                                 (result_df['session_id'] == session_id)]

        if valid_trials.empty:
            print(f"No valid typical trials for session {session_id}.")
            return None

        rL, rR = compute_rL_rR(valid_trials, valid_cluster_ids, region_spike_times, region_spike_clusters, neuron_mean_firing_rates)
        print("successfully computed rL and rR")
        mean_firing_rate = np.sum(spike_matrix_SP)/(SP_times[1]-SP_times[0])

        diagonal_components = np.diag(transition_matrix)
        
        middle_time = (SP_times[0] + SP_times[1]) / 2
        time_window_start = middle_time - 10  
        time_window_end = middle_time + 10   

        spike_times_window = {}
        for cluster_id in valid_cluster_ids:
 
            spike_times = spikes['times'][(spikes['clusters'] == cluster_id) &
                                  (spikes['times'] >= time_window_start) & 
                                  (spikes['times'] <= time_window_end)]
            spike_times_window[cluster_id] = spike_times

        state_probabilities_all = final_hmm.filter(spike_matrix_SP.T)  
        most_likely_states_all = np.argmax(state_probabilities_all, axis=1)

        bins_all = np.arange(SP_times[0], SP_times[1] + bin_size, bin_size)

        within_window = (bins_all[:-1] >= time_window_start) & (bins_all[:-1] <= time_window_end)
        most_likely_states_window = most_likely_states_all[within_window]


        bins_window = np.arange(time_window_start, time_window_end + bin_size, bin_size)

        additional_data = {
            'session_id': session_id,
            'pid':pid,
            'region': region_of_interest,
            'diagonal_components': diagonal_components,
            'bins_window':bins_window,
            'most_likely_states_window': most_likely_states_window,
            'time_window': (time_window_start, time_window_end),
            'spike_times_window': spike_times_window,
            'state_representations':state_representations,
            'valid_cluster_ids':valid_cluster_ids,
            'number_of_hidden_states':best_K, 
            'participation_ratio':participation_ratio,
            'rho_self':rho_self,
            'num_absorbing_states':num_absorbing_states,
            'cos_rho_self':cos_rho_self,
            'transition_matrix':transition_matrix,
            'stationary_distribution':stationary_distribution,
            'state_representations':state_representations,
            'rL':rL,
            'rR':rR,
            'mean_firing_rate_across_neuron': mean_firing_rate,
            'neuron_mean_firing_rates':neuron_mean_firing_rates
            
        }
        
        return (
            session_id,
            np.sum(spike_matrix_SP),  
            best_K,  
            len(valid_cluster_ids),  
            participation_ratio,
            rho_self,
            num_absorbing_states,
            cos_rho_self,
            transition_matrix,
            stationary_distribution,
            state_representations,
            hidden_states_range,
            rL,
            rR,
            mean_firing_rate,
            additional_data,
            neuron_mean_firing_rates,
            pid
        )
    except Exception as e:
        print(f"Error processing session {session_id}: {e}")
        return None


def compute_rL_rR(valid_trials, cluster_ids, region_spike_times, region_spike_clusters,neuron_mean_firing_rates):
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
            if trial['choice'] == -1 and trial['rewardVolume'] > 0:
                rL[neuron_idx] += (spike_count / feedback_time)/neuron_mean_firing_rates[neuron_idx]
            elif trial['choice'] == 1 and trial['rewardVolume'] > 0:
                rR[neuron_idx] += (spike_count / feedback_time)/neuron_mean_firing_rates[neuron_idx]

        rL_total += rL
        rR_total += rR

        #print(f"Number of negative values in rL_total: {np.sum(rL_total < 0)}")
        #print(f"Number of negative values in rR_total: {np.sum(rR_total < 0)}")

    return rL_total, rR_total


from sklearn.metrics.pairwise import cosine_similarity


def compute_cos_rho_self(transition_matrix, stationary_distribution, state_representations):
    """
    Computes the cos_rho_self using the formula: sum_i stationary_distribution_i * sum_j T_ij * cos(r_i, r_j)
    """
    cos_sim = cosine_similarity(state_representations)
    return np.sum(stationary_distribution[:, None] * transition_matrix * cos_sim)

def compute_state_distances(state_representations, session_id, pid, mean_firing_rate):
    num_states, num_neurons = state_representations.shape
    distance_records = []
    cos_sim_matrix = cosine_similarity(state_representations)

    for i in range(num_states):
        for j in range(num_states):  # Only store unique dij values
            l1_norm = np.sum(np.abs(state_representations[i] - state_representations[j]))
            dij = (l1_norm / np.sqrt(num_neurons)) / mean_firing_rate
            sij = cos_sim_matrix[i, j]
            distance_records.append({'session_id': session_id, 'pid': pid, 'state_i': i, 'state_j': j, 'sij': sij, 'dij': dij})  # Fix

    dij_df = pd.DataFrame(distance_records)
    return dij_df



def process_task(task):
    pid, region = task
    print(f"🛠 process_task received task: {task}")  # Debug print

    results = compute_HMM(pid, one, region, result_df)
    base_output_dir = "HMM_Results1"

    if results is None:
        print(f"process_task: No valid results for pid {pid}, region {region}")
        return None

    (
        session_id, total_spike_count_sp, best_K, num_neurons, participation_ratio, rho_self, num_absorbing_states, cos_rho_self,
        transition_matrix, stationary_distribution, state_representations, hidden_states_range, rL, rR, mean_firing_rate, additional_data, neuron_mean_firing_rates, pid
    ) = results


    neuron_mean_firing_rates = np.array(neuron_mean_firing_rates)
    if neuron_mean_firing_rates.shape[0] != state_representations.shape[1]:
        raise ValueError(f"Mismatch: neuron_mean_firing_rates ({neuron_mean_firing_rates.shape}) vs state_representations ({state_representations.shape})")

    # **计算三种不同的 Z-score 归一化**
    state_representations_1 = state_representations / neuron_mean_firing_rates[None, :]
    state_representations_2 = (state_representations - neuron_mean_firing_rates[None, :]) / np.std(state_representations, axis=0, keepdims=True)
    state_representations_3 = state_representations - neuron_mean_firing_rates[None, :]

    # 
    dij_df   = compute_state_distances(state_representations, session_id, pid,  mean_firing_rate)
    dij_df_1 = compute_state_distances(state_representations_1, session_id, pid, mean_firing_rate)
    dij_df_2 = compute_state_distances(state_representations_2, session_id, pid, mean_firing_rate)
    dij_df_3 = compute_state_distances(state_representations_3, session_id, pid, mean_firing_rate)

    # 
    sij_matrix = np.zeros((best_K, best_K))
    sij_matrix_1 = np.zeros((best_K, best_K))
    sij_matrix_2 = np.zeros((best_K, best_K))
    sij_matrix_3 = np.zeros((best_K, best_K))

    dij_matrix = np.zeros((best_K, best_K))
    dij_matrix_1 = np.zeros((best_K, best_K))
    dij_matrix_2 = np.zeros((best_K, best_K))
    dij_matrix_3 = np.zeros((best_K, best_K))

    sij_list, sij_list_1, sij_list_2, sij_list_3 = [], [], [], []
    dij_list, dij_list_1, dij_list_2, dij_list_3 = [], [], [], []
    
    for _, row in dij_df.iterrows():
        i, j = int(row['state_i']), int(row['state_j'])
        sij_matrix[i, j] = row['sij']
        dij_matrix[i, j] = row['dij']
        sij_list.append({'session_id': session_id, 'pid': pid, 'state_i': i, 'state_j': j, 'sij': row['sij'], 'dij': row['dij'], 'normalization': 'none'})

    for _, row in dij_df_1.iterrows():
        i, j = int(row['state_i']), int(row['state_j'])
        sij_matrix_1[i, j] = row['sij']
        dij_matrix_1[i, j] = row['dij']
        sij_list_1.append({'session_id': session_id, 'pid': pid, 'state_i': i, 'state_j': j, 'sij': row['sij'], 'dij': row['dij'], 'normalization': 'Z1'})

    for _, row in dij_df_2.iterrows():
        i, j = int(row['state_i']), int(row['state_j'])
        sij_matrix_2[i, j] = row['sij']
        dij_matrix_2[i, j] = row['dij']
        sij_list_2.append({'session_id': session_id, 'pid': pid, 'state_i': i, 'state_j': j, 'sij': row['sij'], 'dij': row['dij'], 'normalization': 'Z2'})
        
    for _, row in dij_df_3.iterrows():
        i, j = int(row['state_i']), int(row['state_j'])
        sij_matrix_3[i, j] = row['sij']
        dij_matrix_3[i, j] = row['dij']
        sij_list_3.append({'session_id': session_id, 'pid': pid, 'state_i': i, 'state_j': j, 'sij': row['sij'], 'dij': row['dij'], 'normalization': 'Z3'})

    sij_df = pd.DataFrame(sij_list)    
    sij_df_1 = pd.DataFrame(sij_list_1)
    sij_df_2 = pd.DataFrame(sij_list_2)
    sij_df_3 = pd.DataFrame(sij_list_3)

    # **计算三种不同归一化下的 s_bar**
    s_bar = np.sum(stationary_distribution[:, None] * transition_matrix * sij_matrix)
    s_bar_1 = np.sum(stationary_distribution[:, None] * transition_matrix * sij_matrix_1)
    s_bar_2 = np.sum(stationary_distribution[:, None] * transition_matrix * sij_matrix_2)
    s_bar_3 = np.sum(stationary_distribution[:, None] * transition_matrix * sij_matrix_3)

    dij_dff = pd.DataFrame(dij_list)    
    dij_dff_1 = pd.DataFrame(dij_list_1)
    dij_dff_2 = pd.DataFrame(dij_list_2)
    dij_dff_3 = pd.DataFrame(dij_list_3)

    # **计算三种不同归一化下的 s_bar**
    d_bar = np.sum(stationary_distribution[:, None] * transition_matrix * dij_matrix)
    d_bar_1 = np.sum(stationary_distribution[:, None] * transition_matrix * dij_matrix_1)
    d_bar_2 = np.sum(stationary_distribution[:, None] * transition_matrix * dij_matrix_2)
    d_bar_3 = np.sum(stationary_distribution[:, None] * transition_matrix * dij_matrix_3)

    state_distances = []
    for i, ri in enumerate(state_representations):
        d_euclid_L = np.linalg.norm(ri - rL) / (np.sqrt(num_neurons) * mean_firing_rate)
        d_euclid_R = np.linalg.norm(ri - rR) / (np.sqrt(num_neurons) * mean_firing_rate)
        cos_L = cosine_similarity([ri], [rL])[0][0]
        cos_R = cosine_similarity([ri], [rR])[0][0]

        #print(f"Number of negative values in ri: {np.sum(ri < 0)}")
        
        state_distances.append({
            'session_id': session_id, 'state_id': i,
            'euclidean_distance_L': d_euclid_L,
            'euclidean_distance_R': d_euclid_R,
            'cos_distance_L': cos_L, 'cos_distance_R': cos_R
        })

    ri_trial_df = pd.DataFrame(state_distances)

    return {
        'session_id': session_id, 
        'pid':pid,
        'region': region,
        'neuron_count': num_neurons,
        'total_spike_counts_sp': total_spike_count_sp,
        'number_of_hidden_states': best_K,
        'participation_ratio': participation_ratio,
        'rho_self': rho_self,
        'cos_rho_self': cos_rho_self,
        'num_absorbing_states': num_absorbing_states,
        'd_bar' : d_bar,
        'd_bar_1': d_bar_1,
        'd_bar_2': d_bar_2,
        'd_bar_3': d_bar_3,
        'dij_df': dij_df,  
        'dij_df_1': dij_df_1,
        'dij_df_2': dij_df_2,
        'dij_df_3': dij_df_3,
        's_bar' : s_bar,
        's_bar_1': s_bar_1,
        's_bar_2': s_bar_2,
        's_bar_3': s_bar_3,
        'sij_df': sij_df,  
        'sij_df_1': sij_df_1,
        'sij_df_2': sij_df_2,
        'sij_df_3': sij_df_3,
        'ri_trial_df': ri_trial_df,
        'additional_data': additional_data
    }




def plot_HMM_parallel(regions_of_interest, successful_pids, result_df, 
                      base_output_dir="/storage1/fs1/hiratani/Active/shared/ibl_space/ONE/openalyx.internationalbrainlab.org/HMM_Results", 
                      num_workers=4):
    
    tasks = [(pid, region) for region in regions_of_interest for pid in successful_pids]
    print(f"Processing {len(tasks)} tasks in parallel with {num_workers} workers...")

    region_dfs = {region: pd.DataFrame(columns=[
        'session_id', 'pid', 'region', 'neuron_count', 'total_spike_counts_sp',
        'number_of_hidden_states', 'participation_ratio', 'rho_self', 
        'cos_rho_self', 'num_absorbing_states', 'd_bar', 'd_bar_1', 'd_bar_2', 'd_bar_3', 's_bar', 's_bar_1', 's_bar_2', 's_bar_3'
    ]) for region in regions_of_interest}
    
    region_sij_dfs   = {region: pd.DataFrame(columns=['session_id', 'pid', 'state_i', 'state_j', 'sij', 'dij', 'normalization']) for region in regions_of_interest}
    region_sij_dfs_1 = {region: pd.DataFrame(columns=['session_id', 'pid', 'state_i', 'state_j', 'sij', 'dij', 'normalization']) for region in regions_of_interest}
    region_sij_dfs_2 = {region: pd.DataFrame(columns=['session_id', 'pid', 'state_i', 'state_j', 'sij','dij', 'normalization']) for region in regions_of_interest}
    region_sij_dfs_3 = {region: pd.DataFrame(columns=['session_id', 'pid', 'state_i', 'state_j', 'sij', 'dij', 'normalization']) for region in regions_of_interest}
    
    region_ri_trial_dfs = {region: pd.DataFrame(columns=[
        'session_id', 'state_id', 'euclidean_distance_L', 'euclidean_distance_R',
        'cos_distance_L', 'cos_distance_R'
    ]) for region in regions_of_interest}

    all_additional_data = {}
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_task = {executor.submit(process_task, task): task for task in tasks}

        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                result = future.result()
                #print(f" Task {task} returned")  # Debug print

                if result is None:
                    #print(f"Skipping task {task}: Result is None")
                    continue

                session_id = result['session_id']
                region = result['region']
                additional_data = result.get('additional_data', None)
                if session_id not in all_additional_data:
                    all_additional_data[session_id] = {}
                all_additional_data[session_id][region] = additional_data

                summary_data = {k: v for k, v in result.items() if k not in ['dij_df', 'dij_df_1', 'dij_df_2', 'dij_df_3','sij_df', 'sij_df_1', 'sij_df_2', 'sij_df_3', 'ri_trial_df', 'additional_data']}  
                df_entry = pd.DataFrame([summary_data])
                region_dfs[region] = pd.concat([region_dfs[region], df_entry], ignore_index=True)
                
                if 'sij_df' in result and isinstance(result['sij_df'], pd.DataFrame):
                    region_sij_dfs[region] = pd.concat([region_sij_dfs[region], result['sij_df']], ignore_index=True)

                if 'sij_df_1' in result and isinstance(result['sij_df_1'], pd.DataFrame):
                    region_sij_dfs_1[region] = pd.concat([region_sij_dfs_1[region], result['sij_df_1']], ignore_index=True)

                if 'sij_df_2' in result and isinstance(result['sij_df_2'], pd.DataFrame):
                    region_sij_dfs_2[region] = pd.concat([region_sij_dfs_2[region], result['sij_df_2']], ignore_index=True)
                    
                if 'sij_df_3' in result and isinstance(result['sij_df_3'], pd.DataFrame):
                    region_sij_dfs_3[region] = pd.concat([region_sij_dfs_3[region], result['sij_df_3']], ignore_index=True)

                if 'ri_trial_df' in result and isinstance(result['ri_trial_df'], pd.DataFrame):
                    region_ri_trial_dfs[region] = pd.concat([region_ri_trial_dfs[region], result['ri_trial_df']], ignore_index=True)

            except Exception as e:
                print(f"❌ Error processing task {task}: {e}")

    region_name = regions_of_interest[0]
    output_dir = os.path.join(base_output_dir)
    os.makedirs(output_dir, exist_ok=True)
    combined_output_file = os.path.join(output_dir, f"{region_name}_additional_data_ephys.pkl")

    with open(combined_output_file, 'wb') as f:
        pickle.dump(all_additional_data, f)

    print(f"All session additional data saved to {combined_output_file}")

    combined_data = {
        'summary_data': region_dfs,
        'sij_df'  : region_sij_dfs,
        'sij_df_1': region_sij_dfs_1,
        'sij_df_2': region_sij_dfs_2,
        'sij_df_3': region_sij_dfs_3,
        'ri_trial_df': region_ri_trial_dfs,
        'additional_data': all_additional_data
    }

    combined_output_file_final = os.path.join(output_dir, f"{region_name}_HMM_analysis_data_ephys.pkl")

    with open(combined_output_file_final, 'wb') as f:
        pickle.dump(combined_data, f)

    print(f"All data (summary, sij, sij_1, sij_2, sij_3, ri_trial, additional) saved to {combined_output_file_final}")

    for region, df in region_dfs.items():
         print(f"Region: {region}, Unique PIDs: {df['pid'].nunique()}")


    return combined_data




if __name__ == "__main__":

    ONE.setup(base_url='https://openalyx.internationalbrainlab.org', silent=True)
    one = ONE(base_url='https://openalyx.internationalbrainlab.org', password=os.getenv('ALYX_PASSWORD'))

    
    file_path = '/storage1/fs1/hiratani/Active/shared/ibl_space/ONE/openalyx.internationalbrainlab.org/result_df.csv'
    result_df = pd.read_csv(file_path)
    
    successful_pids_CA1 = ['00c425fd-ec3e-4cd2-b8af-c0bc0c4bdd44',
 '02cc03e4-8015-4050-bb42-6c832091febb',
 '0393f34c-a2bd-4c01-99c9-f6b4ec6e786d',
 '071f02e7-752a-4094-af79-8dd764e9d85d',
 '07c79864-6fff-4e72-9fce-1c982e3457f9',
 '0851db85-2889-4070-ac18-a40e8ebd96ba',
 '099c0519-640b-4eb7-867c-998dc337579d',
 '0aafb6f1-6c10-4886-8f03-543988e02d9e',
 '0b877123-0902-4432-b789-c4c6cc681df4',
 '0eb65305-bb95-4bf1-a154-1b810c0cff25',
 '0f306cdc-878a-4ea0-9e91-b97736731637',
 '12c0f3a9-518e-4dac-9651-5d95c2b4730f',
 '143dd7cf-6a47-47a1-906d-927ad7fe9117',
 '16799c7a-e395-435d-a4c4-a678007e1550',
 '1841cf1f-725d-499e-ab8e-7f6fc8c308b6',
 '188fe7d5-fd1c-494f-88bc-672d77b9779e',
 '18d316bf-d322-4c5c-814e-a58147f7bf5f',
 '19e0bf1e-018e-40cb-9acd-af8f0f66bd4a',
 '1a60a6e1-da99-4d4e-a734-39b1d4544fad',
 '1e104bf4-7a24-4624-a5b2-c2c8289c0de7',
 '220bca21-4cf8-43f1-a213-71645899c571',
 '27bac116-ea57-4512-ad35-714a62d259cd',
 '2aea57ac-d3d0-4a09-b6fa-0aa9d58d1e11',
 '3530526e-e690-4222-878f-0f9f401fa759',
 '36362f75-96d8-4ed4-a728-5e72284d0995',
 '38124fca-a0ac-4b58-8e8b-84a2357850e6',
 '39180bcb-13e5-46f9-89e1-8ea2cba22105',
 '3d3d5a5e-df26-43ee-80b6-2d72d85668a5',
 '3fded122-619c-4e65-aadd-d5420978d167',
 '4364cabe-a27c-4bd5-b398-1d068aca47cf',
 '44fc463b-405f-4048-9242-ac018b9f50f7',
 '461ecb39-ab58-4a6c-acd7-22da5b4b4b22',
 '4762e8ed-4d94-4fd7-9522-e927f5ffca74',
 '485b50c8-71e1-4654-9a07-64395c15f5ed',
 '4b93a168-0f3b-4124-88fa-a57046ca70e1',
 '4c04120d-523a-4795-ba8f-49dbb8d9f63a',
 '4f10fbb7-57be-427b-bb1d-a19bd31dc27b',
 '50f1512d-dd41-4a0c-b3ab-b0564f0424d7',
 '5135e93f-2f1f-4301-9532-b5ad62548c49',
 '5246af08-0730-40f7-83de-29b5d62b9b6d',
 '531423f6-d36d-472b-8234-c8f7b8293f79',
 '532bb33e-f22f-454a-ba1f-561d1578b250',
 '53b3a7c6-ae42-49da-b69c-69e55a43a427',
 '55001cb4-e10f-4657-9760-1228b590a69c',
 '5544da75-8d34-47cb-9092-1f91e713e79c',
 '56f2a378-78d2-4132-b3c8-8c1ba82be598',
 '57656bee-e32e-4848-b924-0f6f18cfdfb1',
 '5810514e-2a86-4a34-b7bd-1e4b0b601295',
 '63435e73-6a72-4007-b0da-03e6473e6870',
 '63517fd4-ece1-49eb-9259-371dc30b1dd6',
 '63a32e5c-f63a-450d-85cb-140947b67eaf',
 '642a3373-6fee-4f2a-b8d6-38dd42ada763',
 '64f4afa4-ec5f-4ad6-ab28-415a9a8098ed',
 '6506252f-1141-4e11-8816-4db01b71dda0',
 '69f42a9c-095d-4a25-bca8-61a9869871d3',
 '6b6af675-e1ef-43a6-b408-95cfc71fe2cc',
 '6bd246d2-b2c2-4a88-ac44-1a3e0facbaee',
 '6be21156-33b0-4f70-9a0f-65b3e3cd6d4a',
 '6d24683c-da42-4610-baf0-7ceee7014394',
 '6d9b6393-6729-4a15-ad08-c6838842a074',
 '6fc4d73c-2071-43ec-a756-c6c6d8322c8b',
 '70da415f-444d-4148-ade7-a1f58a16fcf8',
 '76de0e1a-30aa-4713-9fe5-25ad2dff653f',
 '7be00744-2e27-4062-8f56-3969e24e9990',
 '7beb9419-113d-4e47-938c-68ab2657031e',
 '80f6ffdd-f692-450f-ab19-cd6d45bfd73e',
 '8169d556-f994-4735-b4c8-f7c85ddc39b0',
 '834fca72-0b69-44e4-b77e-95a61290b50d',
 '84fd7fa3-6c2d-4233-b265-46a427d3d68d',
 '8ab470cc-de5a-4d65-92bd-9a25fbe6d07d',
 '8abf098f-d4f6-4957-9c0a-f53685db74cc',
 '8d59da25-3a9c-44be-8b1a-e27cdd39ca34',
 '9338f8bb-e097-46d8-81b4-7f37beb4d308',
 '94761f31-0dfc-4f67-87ef-6d1ac7f95144',
 '954b1983-e603-4611-ba15-dc2db6f988ab',
 '95fd67e6-cbff-4356-80c7-5a03b1bf6b8a',
 '9657af01-50bd-4120-8303-416ad9e24a51',
 'a63a7248-9393-478b-9478-64f421ef5eb8',
 'a6fe3779-2b77-4b66-a625-a6078720e412',
 'ac839451-05bc-493e-b167-558b2b195baa',
 'ad714133-1e03-4d3a-8427-33fc483daf1a',
 'afe87fbb-3a17-461f-b333-e22903f1d70d',
 'b73ef9cf-462c-4de3-ae61-5907790ebd0e',
 'b78b3c42-eee5-47c6-9717-743b78c0b721',
 'b8df4cc3-e973-4267-8cf8-a55cea77e1ac',
 'b976e816-bc24-42e3-adf4-2801f1a52657',
 'be9a2119-a4d7-4e62-82a1-b1a6623ecc99',
 'c0c3c95d-43c3-4e30-9ce7-0519d0473911',
 'c17772a9-21b5-49df-ab31-3017addea12e',
 'c2363000-27a6-461e-940b-15f681496ed8',
 'c4f6665f-8be5-476b-a6e8-d81eeae9279d',
 'c6e294f7-5421-4697-8618-8ccc9b0269f6',
 'c9fb5e2e-bd92-41d8-8b7e-394005860a1e',
 'cc701ed3-84b6-4b8e-8390-2e3f54414b19',
 'ccb501d1-a4fa-41c6-819e-54aaf74d439d',
 'ce397420-3cd2-4a55-8fd1-5e28321981f4',
 'd004f105-9886-4b83-a59a-f9173131a383',
 'd151b391-e6e4-4daf-bd9c-191f4ad837b4',
 'dab512bd-a02d-4c1f-8dbc-9155a163efc0',
 'db2100c3-64ac-463e-97a1-20ce8266cd5f',
 'dd619e10-5df1-4c79-bd62-cc00937b5d36',
 'dd75e810-a399-4364-9d4a-517312cf3010',
 'dec6ad00-01c8-4bfd-bcdc-b37c5e0cdb0e',
 'df7d2ca7-068b-4df2-9782-8f6bb7d67f18',
 'e31b4e39-e350-47a9-aca4-72496d99ff2a',
 'e42e948c-3154-45cb-bf52-408b7cda0f2f',
 'e45a00b1-14a0-4f5e-9ea5-9f76d042b11c',
 'e55266c7-eb05-47bb-b263-1cc08dc3c00c',
 'e9cf749b-85dc-4b59-834b-325cec608c48',
 'eb99c2c8-e614-4240-931b-169bed23e9f5',
 'ed469fda-4519-456f-a671-cf901222b4f8',
 'ee3345e6-540d-4cea-9e4a-7f1b2fb9a4e4',
 'f03b61b4-6b13-479d-940f-d1608eb275cc',
 'f26a6ab1-7e37-4f8d-bb50-295c056e1062',
 'f2a098e7-a67e-4125-92d8-36fc6b606c45',
 'f2ee886d-5b9c-4d06-a9be-ee7ae8381114',
 'f336f6a4-f693-4b88-b12c-c5cf0785b061',
 'f68d9f26-ac40-4c67-9cbf-9ad1851292f7',
 'f84f36c9-88f8-4d80-ba34-7f7a2b254ece',
 'f93bfce4-e814-4ae3-9cdf-59f4dcdedf51',
 'febb430e-2d50-4f83-87a0-b5ffbb9a4943']
    regions_of_interest_CA1 = ['CA1']
    HMM_CA1 = plot_HMM_parallel(regions_of_interest_CA1, successful_pids_CA1,result_df,base_output_dir="/storage1/fs1/hiratani/Active/shared/ibl_space/ONE/openalyx.internationalbrainlab.org/HMM_Results", num_workers=4)
    
