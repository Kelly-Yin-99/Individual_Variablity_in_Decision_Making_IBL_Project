
def compute_spiking_metrics_SA_Task(pid, one, region_of_interest, atlas=AllenAtlas(), bin_sizes=[0.01]):
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

