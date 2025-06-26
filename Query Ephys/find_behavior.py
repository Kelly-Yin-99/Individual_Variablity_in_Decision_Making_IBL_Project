import numpy as np
import pandas as pd
from one.api import ONE
from iblatlas.atlas import AllenAtlas
from brainbox.io.one import SpikeSortingLoader, SessionLoader
from sklearn.decomposition import PCA

import pickle
from sklearn.metrics.pairwise import cosine_similarity
import os
from pathlib import Path
atlas = AllenAtlas()
ONE.setup(base_url='https://openalyx.internationalbrainlab.org', silent=True,
          cache_dir=Path('/storage1/fs1/hiratani/Active/shared/ibl_space/ONE/openalyx.internationalbrainlab.org'))
one = ONE(base_url='https://openalyx.internationalbrainlab.org', password=os.getenv('ALYX_PASSWORD'))


from brainbox.io.one import SessionLoader
from datetime import datetime

def load_wheel_data(pid):
    try:
        session_id, pname = one.pid2eid(pid)
        eid = str(session_id)
        wheel_data = one.load_object(eid, 'wheel', collection='alf')
        wheel_position = wheel_data['position']
        wheel_timestamps = wheel_data['timestamps']
        return wheel_position, wheel_timestamps
    except Exception as e:
        print(f"No wheel data for {eid}: {e}")
        return None, None


def calc_wheel_velocity(position, timestamps):
    wheel_velocity = [0.0]
    for widx in range(len(position) - 1):
        time_diff = timestamps[widx + 1] - timestamps[widx]
        if time_diff != 0:
            velocity = (position[widx + 1] - position[widx]) / time_diff
        else:
            velocity = 0.0
        wheel_velocity.append(velocity)
    return wheel_velocity


def calc_trialwise_wheel(position, timestamps, velocity, stimOn_times, feedback_times):
    stimOn_pre_duration = 0.3  # [s]
    total_trial_count = len(stimOn_times)

    trial_position = [[] for _ in range(total_trial_count)]
    trial_timestamps = [[] for _ in range(total_trial_count)]
    trial_velocity = [[] for _ in range(total_trial_count)]

    tridx = 0
    for tsidx in range(len(timestamps)):
        timestamp = timestamps[tsidx]
        while tridx < total_trial_count - 1 and timestamp > stimOn_times[tridx + 1] - stimOn_pre_duration:
            tridx += 1

        if stimOn_times[tridx] - stimOn_pre_duration <= timestamp < feedback_times[tridx]:
            trial_position[tridx].append(position[tsidx])
            trial_timestamps[tridx].append(timestamps[tsidx])
            trial_velocity[tridx].append(velocity[tsidx])

    return trial_position, trial_timestamps, trial_velocity


def calc_movement_onset_times(trial_timestamps, trial_velocity, stimOn_times):
    speed_threshold = 0.5
    duration_threshold = 0.05  # [s]

    movement_onset_times = []
    first_movement_onset_times = np.zeros(len(trial_timestamps))
    last_movement_onset_times = np.zeros(len(trial_timestamps))
    movement_onset_counts = np.zeros(len(trial_timestamps))

    for tridx in range(len(trial_timestamps)):
        movement_onset_times.append([])
        cm_dur = 0.0  # continuous stationary duration
        for tpidx in range(len(trial_timestamps[tridx])):
            t = trial_timestamps[tridx][tpidx]
            if tpidx == 0:
                tprev = stimOn_times[tridx] - 0.3
            cm_dur += (t - tprev)
            if abs(trial_velocity[tridx][tpidx]) > speed_threshold:
                if cm_dur > duration_threshold:
                    movement_onset_times[tridx].append(t)
                cm_dur = 0.0
            tprev = t
        movement_onset_counts[tridx] = len(movement_onset_times[tridx])
        if len(movement_onset_times[tridx]) == 0:
            first_movement_onset_times[tridx] = np.NaN
            last_movement_onset_times[tridx] = np.NaN
        else:
            first_movement_onset_times[tridx] = movement_onset_times[tridx][0]
            last_movement_onset_times[tridx] = movement_onset_times[tridx][-1]

    return movement_onset_times, first_movement_onset_times, last_movement_onset_times



def calc_movement_onset_times(trial_timestamps, trial_velocity, stimOn_times):
    speed_threshold = 0.5
    duration_threshold = 0.05  # [s]

    movement_onset_times = []
    movement_directions = []
    first_movement_onset_times = np.zeros(len(trial_timestamps))
    last_movement_onset_times = np.zeros(len(trial_timestamps))
    first_movement_directions = np.zeros(len(trial_timestamps))
    last_movement_directions = np.zeros(len(trial_timestamps))
    movement_onset_counts = np.zeros(len(trial_timestamps))

    for tridx in range(len(trial_timestamps)):
        movement_onset_times.append([])
        movement_directions.append([])
        cm_dur = 0.0
        for tpidx in range(len(trial_timestamps[tridx])):
            t = trial_timestamps[tridx][tpidx]
            if tpidx == 0:
                tprev = stimOn_times[tridx] - 0.3
            cm_dur += (t - tprev)
            if abs(trial_velocity[tridx][tpidx]) > speed_threshold:
                if cm_dur > duration_threshold:
                    movement_onset_times[tridx].append(t)
                    movement_directions[tridx].append(np.sign(trial_velocity[tridx][tpidx]))
                cm_dur = 0.0
            tprev = t

        movement_onset_counts[tridx] = len(movement_onset_times[tridx])
        if len(movement_onset_times[tridx]) == 0:
            first_movement_onset_times[tridx] = np.NaN
            last_movement_onset_times[tridx] = np.NaN
            first_movement_directions[tridx] = 0
            last_movement_directions[tridx] = 0
        else:
            first_movement_onset_times[tridx] = movement_onset_times[tridx][0]
            last_movement_onset_times[tridx] = movement_onset_times[tridx][-1]
            first_movement_directions[tridx] = movement_directions[tridx][0]
            last_movement_directions[tridx] = movement_directions[tridx][-1]

    return movement_onset_times, first_movement_onset_times, last_movement_onset_times,first_movement_directions, last_movement_directions



def process_trials_sessions(results_df_VISp):
    all_trials_data = []
    # results_df_VISp = pd.DataFrame(results_df_VISp)

    for single_pid in results_df_VISp:
        try:
            session_id, pname = one.pid2eid(single_pid)
            session_id = str(session_id)
            sl = SessionLoader(eid=session_id, one=one)
            sl.load_trials()
            trials_data = sl.trials

            if len(trials_data) < 400:
                print(f"Session skipped: only {len(trials_data)} trials (< 400).")
                continue

            if len(trials_data) > 40:
                trials_data = trials_data.iloc[:-40].copy()

            # Fetch and calculate subject info
            session_data = one.alyx.rest('sessions', 'read', id=session_id)
            subject_nickname = session_data['subject']
            subject_data = one.alyx.rest('subjects', 'list', nickname=subject_nickname)[0]

            # Calculate age in days
            start_time_date = datetime.strptime(session_data['start_time'][:10], '%Y-%m-%d')
            birth_date = datetime.strptime(subject_data['birth_date'], '%Y-%m-%d')
            age_days = (start_time_date - birth_date).days

            # Insert subject info directly into the DataFrame
            trials_data['sex'] = [subject_data['sex']] * len(trials_data)
            trials_data['age'] = [age_days] * len(trials_data)
            trials_data['subject'] = [subject_nickname] * len(trials_data)  # Add subject name to each trial

            wheel_position, wheel_timestamps = load_wheel_data(single_pid)
            if wheel_position is None or wheel_timestamps is None:
                continue  # Skips further processing if wheel data is unavailable

            wheel_velocity = calc_wheel_velocity(wheel_position, wheel_timestamps)
            trial_position, trial_timestamps, trial_velocity = calc_trialwise_wheel(
                wheel_position, wheel_timestamps, wheel_velocity,
                trials_data['stimOn_times'], trials_data['feedback_times']
            )
            movement_onset_times, first_movement_onset_times, last_movement_onset_times,first_movement_directions, last_movement_directions = calc_movement_onset_times(
                trial_timestamps, trial_velocity, trials_data['stimOn_times']
            )


            trials_data['wheel_position'] = trial_position
            trials_data['wheel_timestamps'] = trial_timestamps
            trials_data['wheel_velocity'] = trial_velocity
            trials_data['first_movement_directions'] = first_movement_directions
            trials_data['last_movement_directions'] = last_movement_directions
            trials_data['movement_onset_times'] = movement_onset_times
            trials_data['first_movement_onset_times'] = first_movement_onset_times
            trials_data['last_movement_onset_times'] = last_movement_onset_times

            trials_df = pd.DataFrame(trials_data)
            trials_df['pid'] = single_pid
            trials_df['session_id'] = session_id

            all_trials_data.append(trials_df)

        except Exception as e:
            print(f"Failed to load trials for {single_pid}: {e}")

    combined_trials_df = pd.concat(all_trials_data, ignore_index=True)
    combined_trials_df.dropna(subset=['first_movement_onset_times', 'stimOn_times'])

    return combined_trials_df


# === 1. 载入 ephys_session_info.pkl ===
with open("/storage1/fs1/hiratani/Active/shared/ibl_space/ONE/openalyx.internationalbrainlab.org/HMM_Results/ephys_session_info_1.pkl", "rb") as f:
    ephys = pickle.load(f)

# === 2. 遍历每个脑区，提取行为数据 ===
save_dir = "/storage1/fs1/hiratani/Active/shared/ibl_space/ONE/openalyx.internationalbrainlab.org/HMM_Results/behavior_by_region"
os.makedirs(save_dir, exist_ok=True)

successful_pids_by_region = ephys["successful_pids_ephys"]

for region, pid_list in successful_pids_by_region.items():
    print(f" Processing region: {region}, {len(pid_list)} PIDs")
    try:
        region_behavior_df = process_trials_sessions(list(pid_list))
        output_path = os.path.join(save_dir, f"{region}_behavior.pkl")
        region_behavior_df.to_pickle(output_path)
        print(f"✅ Saved {region} behavior to {output_path}")
    except Exception as e:
        print(f" Failed to process {region}: {e}")
