import time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.stats as scist
from one.api import ONE
from brainbox.io.one import SpikeSortingLoader
import traceback
from brainbox.io.one import SessionLoader
from iblatlas.atlas import AllenAtlas
ba = AllenAtlas()
import os
import pickle


ONE.setup(base_url='https://openalyx.internationalbrainlab.org', silent=True,
          cache_dir=Path('/storage1/fs1/hiratani/Active/shared/ibl_space/ONE/openalyx.internationalbrainlab.org'))
one = ONE(password='international')


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
    'SSs', 'SSp', 'MOs', 'MOp','SCop', 'SCsg', 'SCzo','ICc', 'ICd', 'ICe',
    'CA1','CA2', 'CA3','SUB','PRE','POST'
]

# Construct rules from prefix list
ACRONYM_RULES = [
    (prefix, lambda a, p=prefix: a.startswith(p)) for prefix in ACRONYM_PREFIXES
]

# Classifier
def classify_acronym(acronym):
    for region, rule in ACRONYM_RULES:
        if rule(acronym):
            return region
    return None





def process_single_session(session_id, cluster_qc):
    try:
        insertions = one.alyx.rest('insertions', 'list', session=session_id)
        if not insertions:
            print(f"Insertions not found for session {session_id}.")
            return None, {}
        #passive_times = one.load_dataset(session_id, '*passivePeriods*', collection='alf')
        #if passive_times is None:
            #print(f"Dataset '*passivePeriods*' not found for session {session_id}.")
            #return None, {}
        #if 'spontaneousActivity' not in passive_times:
            #print(f"Spontaneous activity periods not found for session {session_id}.")
            #return None, {}

        region_cluster_counts = {}
        successful_pids = []
        session_acronyms = []
        region_to_pids = {} 

        for insertion in insertions:
            pid = insertion['id']
            sl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
            spikes, clusters, channels = sl.load_spike_sorting()
            if clusters is None:
                print(f"No clusters found for session {session_id}, pid {pid}")
                continue
    
            clusters = sl.merge_clusters(spikes, clusters, channels)
            if clusters is None:
                print(f"Failed to merge clusters for session {session_id}, pid {pid}")
                continue
           
            good_cluster_ids = clusters['cluster_id'][clusters['label'] >= cluster_qc]
            if good_cluster_ids is None:
                print(f"No good cluster IDs found for session {session_id}, pid {pid}")
                continue
            unique_acronyms = np.unique(clusters['acronym'])
            session_acronyms.extend(unique_acronyms)

            combined_acronyms = {region: [] for region, _ in ACRONYM_RULES}


            for acronym in unique_acronyms:
                region = classify_acronym(acronym)
                if region:
                    combined_acronyms[region].append(acronym)
                else:
                    region_cluster_counts[acronym] = clusters['cluster_id'][
                        (clusters['acronym'] == acronym) & 
                        (np.isin(clusters['cluster_id'], good_cluster_ids))
                    ]

            for combined_acronym, acronyms in combined_acronyms.items():
                combined_cluster_ids = []
                for acronym in acronyms:
                    combined_cluster_ids.extend(clusters['cluster_id'][
                        (clusters['acronym'] == acronym) & 
                        (np.isin(clusters['cluster_id'], good_cluster_ids))
                    ])
                if combined_cluster_ids:
                    region_cluster_counts[combined_acronym] = combined_cluster_ids
                    if combined_acronym not in region_to_pids:  
                        region_to_pids[combined_acronym] = set()
                    region_to_pids[combined_acronym].add(pid)  


        session_results = {
            'session_id': session_id,
            'regions': {},
            'pids': successful_pids
        }

        for region, cluster_ids in region_cluster_counts.items():
            num_good_clusters = len(cluster_ids)
            total_clusters_in_region = len(clusters['cluster_id'][clusters['acronym'] == region])
            percentage_good_clusters = (num_good_clusters / total_clusters_in_region) * 100 if total_clusters_in_region > 0 else 0
            session_results['regions'][region] = {
                'num_good_clusters': num_good_clusters,
                'percentage_good_clusters': percentage_good_clusters
            }

        return session_results, region_to_pids

    except Exception as e:
        print(f"Error processing session {session_id}: {e}")
        return None, {}



def process_all_sessions_rep(session_ids, cluster_qc):
    processed_sessions = []
    unique_acronyms_set = set()
    successful_session_ids = {}
    successful_pids = {}

    start_time = time.time()
    for idx, session_id in enumerate(session_ids):
        result, region_to_pids = process_single_session(session_id, cluster_qc)  
        if result:
            processed_sessions.append(result)

            for region, pids_in_region in region_to_pids.items():
                if region not in successful_session_ids:
                    successful_session_ids[region] = set()
                    successful_pids[region] = set()
                
                successful_session_ids[region].add(session_id)
                successful_pids[region].update(pids_in_region) 

            unique_acronyms_set.update(region_to_pids.keys())

        elapsed_time = time.time() - start_time
        avg_time_per_session = elapsed_time / (idx + 1)
        remaining_sessions = len(session_ids) - (idx + 1)
        estimated_remaining_time = avg_time_per_session * remaining_sessions
        print(f"Processed {idx + 1}/{len(session_ids)} sessions. Estimated remaining time: {estimated_remaining_time // 60:.0f} minutes {estimated_remaining_time % 60:.0f} seconds")

    total_elapsed_time = time.time() - start_time
    print(f"Total elapsed time: {total_elapsed_time // 60:.0f} minutes {total_elapsed_time % 60:.0f} seconds")

    return processed_sessions, unique_acronyms_set, successful_session_ids, successful_pids



def print_results(processed_sessions, unique_acronyms_set, successful_session_ids, successful_pids):
    for result in processed_sessions:
        print(f"\nSession ID: {result['session_id']}")
        for region, data in result['regions'].items():
            print(f"Region: {region}, Number of Good Clusters: {data['num_good_clusters']}, Percentage of Good Clusters: {data['percentage_good_clusters']:.2f}%")
        print(f"PIDs: {result['pids']}")

    print(f"\nUnique acronyms from all processed sessions: {sorted(unique_acronyms_set)}")
    for region, session_ids in successful_session_ids.items():
        print(f"Region: {region}, Successful Session IDs: {sorted(session_ids)}")
        print(f"Region: {region}, Successful PIDs: {sorted(successful_pids[region])}")




ephys_session_id = ['3a3ea015-b5f4-4e8b-b189-9364d1fc7435',
 'd85c454e-8737-4cba-b6ad-b2339429d99b',
 'de905562-31c6-4c31-9ece-3ee87b97eab4',
 'e6594a5b-552c-421a-b376-1a1baa9dc4fd',
 '4e560423-5caf-4cda-8511-d1ab4cd2bf7d',
 'caa5dddc-9290-4e27-9f5e-575ba3598614',
 '642c97ea-fe89-4ec9-8629-5e492ea4019d',
 'c958919c-2e75-435d-845d-5b62190b520e',
 'f56194bc-8215-4ae8-bc6a-89781ad8e050',
 '29a6def1-fc5c-4eea-ac48-47e9b053dcb5',
 '28338153-4113-485b-835b-91cb96d984f2',
 '5c0c560e-9e1f-45e9-b66e-e4ee7855be84',
 'f4eb56a4-8bf8-4bbc-a8f3-6e6535134bad',
 '2d9bfc10-59fb-424a-b699-7c42f86c7871',
 '7cc74598-9c1b-436b-84fa-0bf89f31adf6',
 '3bcb81b4-d9ca-4fc9-a1cd-353a966239ca',
 '5cbb9a87-bd6b-4c5d-82cf-ecf88c5d23b7',
 'bb8d9451-fdbd-4f46-b52e-9290e8f84d2e',
 '8c025071-c4f3-426c-9aed-f149e8f75b7b',
 'a2ec6341-c55f-48a0-a23b-0ef2f5b1d71e',
 'ae8787b1-4229-4d56-b0c2-566b61a25b77',
 '0f77ca5d-73c2-45bd-aa4c-4c5ed275dbde',
 '195443eb-08e9-4a18-a7e1-d105b2ce1429',
 '6c6b0d06-6039-4525-a74b-58cfaa1d3a60',
 '169c9a39-cb63-4b77-93e2-10e076d4c472',
 '80653a5b-c7aa-479d-9ae0-c92f296fface',
 'dc962048-89bb-4e6a-96a9-b062a2be1426',
 '8a1cf4ef-06e3-4c72-9bc7-e1baa189841b',
 '0841d188-8ef2-4f20-9828-76a94d5343a4',
 'fe83be91-8b8b-44d2-ab32-fdda1992fcfb',
 '1d4a7bd6-296a-48b9-b20e-bd0ac80750a5',
 'd035c5ba-d51e-49a9-a94b-23531a598ec3',
 '3513e7f2-d2e6-4411-8055-54dac50458f6',
 '687017d4-c9fc-458f-a7d5-0979fe1a7470',
 '258b4a8b-28e3-4c18-9f86-1ea2bc0dc806',
 'e0928e11-2b86-4387-a203-80c77fab5d52',
 'c7e4e6ad-280f-432f-ac85-9be299890d6e',
 'd62a64f4-fdc6-448b-8f2a-53ed08d645a7',
 '196a2adf-ff83-49b2-823a-33f990049c2e',
 '78b4fff5-c5ec-44d9-b5f9-d59493063f00',
 '1ca83b26-30fc-4350-a616-c38b7d00d240',
 '27ef44c0-acb2-4220-b776-477d0d5abd35',
 'f115196e-8dfe-4d2a-8af3-8206d93c1729',
 'a2701b93-d8e1-47e9-a819-f1063046f3e7',
 '446f4724-1690-49f9-819a-2bd8e2ea88ce',
 'c8d46ee6-eb68-4535-8756-7c9aa32f10e4',
 '8928f98a-b411-497e-aa4b-aa752434686d',
 '3f859b5c-e73a-4044-b49e-34bb81e96715',
 'f819d499-8bf7-4da0-a431-15377a8319d5',
 '752456f3-9f47-4fbf-bd44-9d131c0f41aa',
 '1b9e349e-93f2-41cc-a4b5-b212d7ddc8df',
 '6b0b5d24-bcda-4053-a59c-beaa1fe03b8f',
 'd901aff5-2250-467a-b4a1-0cb9729df9e2',
 '5b49aca6-a6f4-4075-931a-617ad64c219c',
 'eacc49a9-f3a1-49f1-b87f-0972f90ee837',
 '3e6a97d3-3991-49e2-b346-6948cb4580fb',
 '1425bd6f-c625-4f6a-b237-dc5bcfc42c87',
 '63c70ae8-4dfb-418b-b21b-f0b1e5fba6c9',
 '7f5df7eb-cf36-4589-a20a-14b535441142',
 'b887df2c-bb9c-44c9-a9c0-538da87b2cab',
 '9b528ad0-4599-4a55-9148-96cc1d93fb24',
 'a4747ac8-6a75-444f-b99b-696fff0243fd',
 'f4ffb731-8349-4fe4-806e-0232a84e52dd',
 '23c75e0b-05d8-452e-8efb-a3687ab94079',
 '94fc8ca2-1241-4735-988c-63ff48e78174',
 '91bac580-76ed-41ab-ac07-89051f8d7f6e',
 'f180b2b2-5a6a-4fea-9550-4fb0c4376666',
 '0c828385-6dd6-4842-a702-c5075f5f5e81',
 'e6043c7d-8f6e-4b66-8309-2ec0abac0f79',
 'd04feec7-d0b7-4f35-af89-0232dd975bf0',
 'ffef0311-8ffa-49e3-a857-b3adf6d86e12',
 'e45481fa-be22-4365-972c-e7404ed8ab5a',
 'e2b845a1-e313-4a08-bc61-a5f662ed295e',
 'f99ac31f-171b-4208-a55d-5644c0ad51c3',
 '32d27583-56aa-4510-bc03-669036edad20',
 '90e524a2-aa63-47ce-b5b8-1b1941a1223a',
 '09156021-9a1d-4e1d-ae59-48cbde3c5d42',
 '9a6e127b-bb07-4be2-92e2-53dd858c2762',
 '64977c74-9c04-437a-9ea1-50386c4996db',
 '6f36868f-5cc1-450c-82fa-6b9829ce0cfe',
 '5455a21c-1be7-4cae-ae8e-8853a8d5f55e',
 '111c1762-7908-47e0-9f40-2f2ee55b6505',
 'bda2faf5-9563-4940-a80f-ce444259e47b',
 '8d316998-28c3-4265-b029-e2ca82375b2f',
 'fa8ad50d-76f2-45fa-a52f-08fe3d942345',
 '2bdf206a-820f-402f-920a-9e86cd5388a4',
 '3537d970-f515-4786-853f-23de525e110f',
 '19b44992-d527-4a12-8bda-aa11379cb08c',
 'ebc9392c-1ecb-4b4b-a545-4e3d70d23611',
 '8a3a0197-b40a-449f-be55-c00b23253bbf',
 '58c4bf97-ec3b-45b4-9db4-d5d9515d5b00',
 '19e66dc9-bf9f-430b-9d6a-acfa85de6fb7',
 '30af8629-7b96-45b7-8778-374720ddbc5e',
 'cea755db-4eee-4138-bdd6-fc23a572f5a1',
 '1b61b7f2-a599-4e40-abd6-3e758d2c9e25',
 'd71e565d-4ddb-42df-849e-f99cfdeced52',
 'bb099402-fb31-4cfd-824e-1c97530a0875',
 'f27e6cd6-cdd3-4524-b8e3-8146046e2a7d',
 '68775ca0-b056-48d5-b6ae-a4c2a76ae48f',
 '8b1f4024-3d96-4ee7-95f9-8a1dfd4ce4ef',
 '10fac7a1-919d-4ca7-83c4-4675bd8d9416',
 '03d9a098-07bf-4765-88b7-85f8d8f620cc',
 '36dee8f1-a58f-4fca-bf10-9edd6baf07ef',
 'd57df551-6dcb-4242-9c72-b806cff5613a',
 '5bcafa14-71cb-42fa-8265-ce5cda1b89e0',
 '08102cfc-a040-4bcf-b63c-faa0f4914a6f',
 '1b20f589-6177-4e5c-9090-9a609a16944b',
 '824cf03d-4012-4ab1-b499-c83a92c5589e',
 '7502ae93-7437-4bcd-9e14-d73b51193656',
 '537677fe-1e24-4755-948c-fa4a4e8ecce5',
 '51e53aff-1d5d-4182-a684-aba783d50ae5',
 '1f095590-6669-46c9-986b-ccaf0620c5e9',
 '3d59aa1a-b4ba-47fe-b9cf-741b5fdb0c7b',
 '93ad879a-aa42-4150-83e1-38773c9785e4',
 'a405053a-eb13-4aa4-850c-5a337e5dc7fd',
 'c51f34d8-42f6-4c9c-bb5b-669fd9c42cd9',
 'fc43390d-457e-463a-9fd4-b94a0a8b48f5',
 '64e3fb86-928c-4079-865c-b364205b502e',
 '288bfbf3-3700-4abe-b6e4-130b5c541e61',
 '6434f2f5-6bce-42b8-8563-d93d493613a2',
 '72982282-e493-45ee-87ce-aa45cb3a3ec1',
 '0802ced5-33a3-405e-8336-b65ebc5cb07c',
 '15948667-747b-4702-9d53-354ac70e9119',
 'd32876dd-8303-4720-8e7e-20678dc2fd71',
 'f140a2ec-fd49-4814-994a-fe3476f14e66',
 'd55a4fbe-c40c-4c06-ad0b-e4defd6197f3',
 '028e302c-ff7b-421a-a44f-0df99b5d48fa',
 '64365e73-47f5-4404-b01a-c5f4ce865c98',
 '58b1e920-cfc8-467e-b28b-7654a55d0977',
 '86b6ba67-c1db-4333-add0-f8105ea6e363',
 '69a0e953-a643-4f0e-bb26-dc65af3ea7d7',
 '821f1883-27f3-411d-afd3-fb8241bbc39a',
 'af55d16f-0e31-4073-bdb5-26da54914aa2',
 '72028382-a869-4745-bacf-cb8789e16953',
 'e9fc0a2d-c69d-44d1-9fa3-314782387cae',
 '0cc486c3-8c7b-494d-aa04-b70e2690bcba',
 'd2832a38-27f6-452d-91d6-af72d794136c',
 '7cec9792-b8f9-4878-be7e-f08103dc0323',
 'a4a74102-2af5-45dc-9e41-ef7f5aed88be',
 '88224abb-5746-431f-9c17-17d7ef806e6a',
 '72cb5550-43b4-4ef0-add5-e4adfdfb5e02',
 '3555ce3a-ccce-4b7e-8eb1-c2571913599e',
 '91a3353a-2da1-420d-8c7c-fad2fedfdd18',
 'ac7d3064-7f09-48a3-88d2-e86a4eb86461',
 '7082d8ff-255a-47d7-a839-bf093483ec30',
 'ee212778-3903-4f5b-ac4b-a72f22debf03',
 '4503697e-af44-47d9-898d-4924be990240',
 'fa1f26a1-eb49-4b24-917e-19f02a18ac61',
 '0a018f12-ee06-4b11-97aa-bbbff5448e9f',
 '35ed605c-1a1a-47b1-86ff-2b56144f55af',
 '6ed57216-498d-48a6-b48b-a243a34710ea',
 '872ce8ff-9fb3-485c-be00-bc5479e0095b',
 'b22f694e-4a34-4142-ab9d-2556c3487086',
 '30c4e2ab-dffc-499d-aae4-e51d6b3218c2',
 '6f6d2c8e-28be-49f4-ae4d-06be2d3148c1',
 'd0c91c3c-8cbb-4929-8657-31f18bffc294',
 'f359281f-6941-4bfd-90d4-940be22ed3c3',
 '71855308-7e54-41d7-a7a4-b042e78e3b4f',
 '8ca740c5-e7fe-430a-aa10-e74e9c3cbbe8',
 '5157810e-0fff-4bcf-b19d-32d4e39c7dfc',
 '0b7ee1b6-42db-46cd-a465-08f531366187',
 '113c5b6c-940e-4b21-b462-789b4c2be0e5',
 '71e03be6-b497-4991-a121-9416dcc1a6e7',
 'ab8a0899-a59f-42e4-8807-95b14056104b',
 'f5591ac5-311d-4fa8-9bad-029d7be9c491',
 '1a507308-c63a-4e02-8f32-3239a07dc578',
 '21d21fc3-4201-4edc-802a-c67b61952548',
 'dc21e80d-97d7-44ca-a729-a8e3f9b14305',
 '09394481-8dd2-4d5c-9327-f2753ede92d7',
 'aec5d3cc-4bb2-4349-80a9-0395b76f04e2',
 '8c2f7f4d-7346-42a4-a715-4d37a5208535',
 'a9138924-4395-4981-83d1-530f6ff7c8fc',
 'ff96bfe1-d925-4553-94b5-bf8297adf259',
 '83d85891-bd75-4557-91b4-1cbb5f8bfc9d',
 'f88d4dd4-ccd7-400e-9035-fa00be3bcfa8',
 '7af49c00-63dd-4fed-b2e0-1b3bd945b20b',
 '3f6e25ae-c007-4dc3-aa77-450fd5705046',
 '2c44a360-5a56-4971-8009-f469fb59de98',
 '9b5a1754-ac99-4d53-97d3-35c2f6638507',
 '73918ae1-e4fd-4c18-b132-00cb555b1ad2',
 'f304211a-81b1-446f-a435-25e589fe3a5a',
 '4aa1d525-5c7d-4c50-a147-ec53a9014812',
 'd0ea3148-948d-4817-94f8-dcaf2342bbbe',
 '952870e5-f2a7-4518-9e6d-71585460f6fe',
 'c728f6fd-58e2-448d-aefb-a72c637b604c',
 '91e04f86-89df-4dec-a8f8-fa915c9a5f1a',
 'cf063e05-a087-4093-b4e7-03427018a56d',
 '25d1920e-a2af-4b6c-9f2e-fc6c65576544',
 '1735d2be-b388-411a-896a-60b01eaa1cfe',
 '9545aa05-3945-4054-a5c3-a259f7209d61',
 '0cad7ea8-8e6c-4ad1-a5c5-53fbb2df1a63',
 '781b35fd-e1f0-4d14-b2bb-95b7263082bb',
 'd3a2b25e-46d3-4f0b-ade6-4e32255f4c35',
 '5285c561-80da-4563-8694-739da92e5dd0',
 'f1db6257-85ef-4385-b415-2d078ec75df2',
 'a6fe44a8-07ab-49b8-81f9-e18575aa85cc',
 'e5c75b62-6871-4135-b3d0-f6464c2d90c0',
 '3638d102-e8b6-4230-8742-e548cd87a949',
 'd9f0c293-df4c-410a-846d-842e47c6b502',
 '9468fa93-21ae-4984-955c-e8402e280c83',
 '571d3ffe-54a5-473d-a265-5dc373eb7efc',
 '4333b9f7-a4e7-4b83-a457-7122ae080034',
 'aa3432cd-62bd-40bc-bc1c-a12d53bcbdcf',
 'dfbe628d-365b-461c-a07f-8b9911ba83aa',
 '9e9c6fc0-4769-4d83-9ea4-b59a1230510e',
 '69c9a415-f7fa-4208-887b-1417c1479b48',
 '0ac8d013-b91e-4732-bc7b-a1164ff3e445',
 '8a039e2b-637e-45ed-8da6-0641924626f0',
 'c4432264-e1ae-446f-8a07-6280abade813',
 'd7e60cc3-6020-429e-a654-636c6cc677ea',
 '360eac0c-7d2d-4cc1-9dcf-79fc7afc56e7',
 '81a1dca0-cc90-47c5-afe3-c277319c47c8',
 '875c1e5c-f7ec-45ac-ab82-ecfe7276a707',
 '07dc4b76-5b93-4a03-82a0-b3d9cc73f412',
 'dac3a4c1-b666-4de0-87e8-8c514483cacf',
 '493170a6-fd94-4ee4-884f-cc018c17eeb9',
 '8c552ddc-813e-4035-81cc-3971b57efe65',
 '032452e9-1886-449d-9c13-0f192572e19f',
 '75b6b132-d998-4fba-8482-961418ac957d',
 '239cdbb1-68e2-4eb0-91d8-ae5ae4001c7a',
 '63f3dbc1-1a5f-44e5-98dd-ce25cd2b7871',
 '5d6aa933-4b00-4e99-ae2d-5003657592e9',
 '09b2c4d1-058d-4c84-9fd4-97530f85baf6',
 'de588204-8fd6-4ce3-92da-7a6d1dcae238',
 'f84045b0-ce09-4ace-9d11-5ea491620707',
 '56bc129c-6265-407a-a208-cc16d20a6c01',
 '7cb81727-2097-4b52-b480-c89867b5b34c',
 '5ec72172-3901-4771-8777-6e9490ca51fc',
 '22e04698-b974-4805-b241-3b547dbf37bf',
 '5b609f9b-75cb-43d3-9c39-b5b4b7a0db67',
 '5569f363-0934-464e-9a5b-77c8e67791a1',
 '1928bf72-2002-46a6-8930-728420402e01',
 '746d1902-fa59-4cab-b0aa-013be36060d5',
 '77e6dc6a-66ed-433c-b1a2-778c914f523c',
 'cde63527-7f5a-4cc3-8ac2-215d82e7da26',
 '7416f387-b302-4ca3-8daf-03b585a1b7ec',
 '7691eeb3-715b-4571-8fda-6bb57aab8253',
 'fc14c0d6-51cf-48ba-b326-56ed5a9420c3',
 '45ef6691-7b80-4a43-bd1a-85fc00851ae8',
 'e8b4fda3-7fe4-4706-8ec2-91036cfee6bd',
 '54238fd6-d2d0-4408-b1a9-d19d24fd29ce',
 '6bb5da8f-6858-4fdd-96d9-c34b3b841593',
 '671c7ea7-6726-4fbe-adeb-f89c2c8e489b',
 'eebacd5a-7dcd-4ba6-9dff-ec2a4d2f19e0',
 '6c6983ef-7383-4989-9183-32b1a300d17a',
 '91796ceb-e314-4859-9a1f-092f85cc846a',
 '2e22c1fc-eec6-4856-85a0-7dba8668f646',
 '56b57c38-2699-4091-90a8-aba35103155e',
 'e26c6001-defe-42a9-9ded-368e3f03ac61',
 '948fd27b-507b-41b3-bdf8-f9f5f0af8e0b',
 'e012d3e3-fdbc-4661-9ffa-5fa284e4e706',
 '26aa51ff-968c-42e4-85c8-8ff47d19254d',
 '9fe512b8-92a8-4642-83b6-01158ab66c3c',
 'a52f5a1b-7f45-4f2c-89a9-fb199d2a0d63',
 '9eec761e-9762-4897-b308-a3a08c311e69',
 'f7335a49-4a98-46d2-a8ce-d041d2eac1d6',
 '4a45c8ba-db6f-4f11-9403-56e06a33dfa4',
 'd6c86d3c-3980-4f28-b24b-1f9f8c73f0a7',
 '7b26ce84-07f9-43d1-957f-bc72aeb730a3',
 '4ecb5d24-f5cc-402c-be28-9d0f7cb14b3a',
 '754b74d5-7a06-4004-ae0c-72a10b6ed2e6',
 'c6db3304-c906-400c-aa0f-45dd3945b2ea',
 '88d24c31-52e4-49cc-9f32-6adbeb9eba87',
 'c7bd79c9-c47e-4ea5-aea3-74dda991b48e',
 '6fb1e12c-883b-46d1-a745-473cde3232c8',
 '695a6073-eae0-49e0-bb0f-e9e57a9275b9',
 'fece187f-b47f-4870-a1d6-619afe942a7d',
 '6f09ba7e-e3ce-44b0-932b-c003fb44fb89',
 '993c7024-0abc-4028-ad30-d397ad55b084',
 'f3ce3197-d534-4618-bf81-b687555d1883',
 '251ece37-7798-477c-8a06-2845d4aa270c',
 '5522ac4b-0e41-4c53-836a-aaa17e82b9eb',
 'aa20388b-9ea3-4506-92f1-3c2be84b85db',
 'fcd49e34-f07b-441c-b2ac-cb8c462ec5ac',
 'c3d9b6fb-7fa9-4413-a364-92a54df0fc5d',
 '1ec23a70-b94b-4e9c-a0df-8c2151da3761',
 'd23a44ef-1402-4ed7-97f5-47e9a7a504d9',
 '5339812f-8b91-40ba-9d8f-a559563cc46b',
 'e349a2e7-50a3-47ca-bc45-20d1899854ec',
 '768a371d-7e88-47f8-bf21-4a6a6570dd6e',
 '064a7252-8e10-4ad6-b3fd-7a88a2db5463',
 'b01df337-2d31-4bcc-a1fe-7112afd50c50',
 '3332414c-d20a-404e-bdc9-9984a6940aca',
 '15763234-d21e-491f-a01b-1238eb96d389',
 '25f77e81-c1af-46ab-8686-73ac3d67c4a7',
 'a7763417-e0d6-4f2a-aa55-e382fd9b5fb8',
 'e12fbe11-8a6b-4bf6-a955-e5f6cec31ef1',
 '4ef13091-1bc8-4f32-9619-107bdf48540c',
 '037d75ca-c90a-43f2-aca6-e86611916779',
 'cae5cd75-55e5-4277-8db3-cf4d6c5ff918',
 'b69b86be-af7d-4ecf-8cbf-0cd356afa1bd',
 'ebe2efe3-e8a1-451a-8947-76ef42427cc9',
 'edd22318-216c-44ff-bc24-49ce8be78374',
 '626126d5-eecf-4e9b-900e-ec29a17ece07',
 '71e55bfe-5a3a-4cba-bdc7-f085140d798e',
 '49e0ab27-827a-4c91-bcaa-97eea27a1b8d',
 '81a78eac-9d36-4f90-a73a-7eb3ad7f770b',
 '5adab0b7-dfd0-467d-b09d-43cb7ca5d59c',
 'e56541a5-a6d5-4750-b1fe-f6b5257bfe7c',
 '6527e2f1-8b2b-4b9b-a9dd-2a0206603ad8',
 '7f6b86f9-879a-4ea2-8531-294a221af5d0',
 '5d01d14e-aced-4465-8f8e-9a1c674f62ec',
 '8c33abef-3d3e-4d42-9f27-445e9def08f9',
 'c557324b-b95d-414c-888f-6ee1329a2329',
 '61e11a11-ab65-48fb-ae08-3cb80662e5d6',
 'c7248e09-8c0d-40f2-9eb4-700a8973d8c8',
 '280ee768-f7b8-4c6c-9ea0-48ca75d6b6f3',
 'ff48aa1d-ef30-4903-ac34-8c41b738c1b9',
 '03063955-2523-47bd-ae57-f7489dd40f15',
 '1e45d992-c356-40e1-9be1-a506d944896f',
 '90c61c38-b9fd-4cc3-9795-29160d2f8e55',
 '4d8c7767-981c-4347-8e5e-5d5fffe38534',
 'b10ed1ba-2099-42c4-bee3-d053eb594f09',
 '6274dda8-3a59-4aa1-95f8-a8a549c46a26',
 '41872d7f-75cb-4445-bb1a-132b354c44f0',
 'fe1fd79f-b051-411f-a0a9-2530a02cc78d',
 '934dd7a4-fbdc-459c-8830-04fe9033bc28',
 '549caacc-3bd7-40f1-913d-e94141816547',
 '20c112a1-8a42-44e0-a4cd-e5b932f7bda9',
 '65f5c9b4-4440-48b9-b914-c593a5184a18',
 'dcceebe5-4589-44df-a1c1-9fa33e779727',
 'f8041c1e-5ef4-4ae6-afec-ed82d7a74dc1',
 'd832d9f7-c96a-4f63-8921-516ba4a7b61f',
 '4ddb8a95-788b-48d0-8a0a-66c7c796da96',
 'b39752db-abdb-47ab-ae78-e8608bbf50ed',
 'ee8b36de-779f-4dea-901f-e0141c95722b',
 'f9860a11-24d3-452e-ab95-39e199f20a93',
 'bd456d8f-d36e-434a-8051-ff3997253802',
 'b658bc7d-07cd-4203-8a25-7b16b549851b',
 '862ade13-53cd-4221-a3fa-dda8643641f2',
 '7622da34-51b6-4661-98ae-a57d40806008',
 '4720c98a-a305-4fba-affb-bbfa00a724a4',
 '66d98e6e-bcd9-4e78-8fbb-636f7e808b29',
 'f25642c6-27a5-4a97-9ea0-06652db79fbd',
 '28741f91-c837-4147-939e-918d38d849f2',
 'fb70ebf7-8175-42b0-9b7a-7c6e8612226e',
 'f312aaec-3b6f-44b3-86b4-3a0c119c0438',
 '8207abc6-6b23-4762-92b4-82e05bed5143',
 'b81e3e11-9a60-4114-b894-09f85074d9c3',
 '90e74228-fd1a-482f-bd56-05dbad132861',
 '6a601cc5-7b79-4c75-b0e8-552246532f82',
 'a82800ce-f4e3-4464-9b80-4c3d6fade333',
 'a66f1593-dafd-4982-9b66-f9554b6c86b5',
 'd855576e-5b34-41bf-8e3b-2bea0cae1380',
 '41431f53-69fd-4e3b-80ce-ea62e03bf9c7',
 '1eac875c-feaa-4a30-b148-059b954b11d8',
 '8db36de1-8f17-4446-b527-b5d91909b45a',
 'da188f2c-553c-4e04-879b-c9ea2d1b9a93',
 '03cf52f6-fba6-4743-a42e-dd1ac3072343',
 '5b44c40f-80f4-44fb-abfb-c7f19e27a6ca',
 '7be8fec4-406b-4e74-8548-d2885dcc3d5e',
 '6364ff7f-6471-415a-ab9e-632a12052690',
 '56d38157-bb5a-4561-ab5c-3df05a5d6e28',
 'e535fb62-e245-4a48-b119-88ce62a6fe67',
 'f10efe41-0dc0-44d0-8f26-5ff68dca23e9',
 '1191f865-b10a-45c8-9c48-24a980fd9402',
 '765ba913-eb0e-4f7d-be7e-4964abe8b27b',
 '6f8bbc01-51ae-4260-a90c-4830723b35b2',
 '2e6e179c-fccc-4e8f-9448-ce5b6858a183',
 '6668c4a0-70a4-4012-a7da-709660971d7a',
 '37e96d0b-5b4b-4c6e-9b29-7edbdc94bbd0',
 'd16a9a8d-5f42-4b49-ba58-1746f807fcc1',
 '9a629642-3a9c-42ed-b70a-532db0e86199',
 'e5c772cd-9c92-47ab-9525-d618b66a9b5d',
 'dda5fc59-f09a-4256-9fb5-66c67667a466',
 '1120797f-c2b0-4c09-b6ea-2555d69cb7ee',
 '2f63c555-eb74-4d8d-ada5-5c3ecf3b46be',
 'a19c7a3a-7261-42ce-95d5-1f4ca46007ed',
 '57b5ae8f-d446-4161-b439-b191c5e3e77b',
 'f64128ad-e201-4cbc-a839-85565804a89b',
 'f4a4143d-d378-48a3-aed2-fa7958648c24',
 '413a6825-2144-4a50-b3fc-cf38ddd6fd1a',
 '9ac2be3b-6e0b-4f49-b8bf-82344d9f5e67',
 'ee13c19e-2790-4418-97ca-48f02e8013bb',
 '30e5937e-e86a-47e6-93ae-d2ae3877ff8e',
 '720a3fe6-5dfc-4a23-84f0-2f0b08e10ec2',
 '849c9acb-8223-4e09-8cb1-95004b452baf',
 '0f25376f-2b78-4ddc-8c39-b6cdbe7bf5b9',
 'd2f5a130-b981-4546-8858-c94ae1da75ff',
 '158d5d35-a2ab-4a76-87b0-51048c5d5283',
 'f8bf5591-8d75-40e8-961c-30bc1ee50d0f',
 '3c851386-e92d-4533-8d55-89a46f0e7384',
 '3dd347df-f14e-40d5-9ff2-9c49f84d2157',
 '55d53b83-4245-4c91-847d-b59dded1c5f6',
 '7939711b-8b4d-4251-b698-b97c1eaa846e',
 '46794e05-3f6a-4d35-afb3-9165091a5a74',
 'db4df448-e449-4a6f-a0e7-288711e7a75a',
 'fa704052-147e-46f6-b190-a65b837e605e',
 'dfd8e7df-dc51-4589-b6ca-7baccfeb94b4',
 '034e726f-b35f-41e0-8d6c-a22cc32391fb',
 '56956777-dca5-468c-87cb-78150432cc57',
 '4b00df29-3769-43be-bb40-128b1cba6d35',
 '266a0360-ea0a-4580-8f6a-fe5bad9ed17c',
 '83e77b4b-dfa0-4af9-968b-7ea0c7a0c7e4',
 '5386aba9-9b97-4557-abcd-abc2da66b863',
 'dd0faa76-4f49-428c-9507-6de7382a5d9e',
 'edcf0051-57b3-47ce-afa3-443c62cd8aae',
 '6713a4a7-faed-4df2-acab-ee4e63326f8d',
 '85dc2ebd-8aaf-46b0-9284-a197aee8b16f',
 'fb7b21c9-b50e-4145-9254-a91a50d656ca',
 '3663d82b-f197-4e8b-b299-7b803a155b84',
 '01864d6f-31e8-49c9-aadd-2e5021ea0ee7',
 '0cbeae00-e229-4b7d-bdcc-1b0569d7e0c3',
 'd1442e39-68de-41d0-9449-35e5cfe5a94f',
 '38d95489-2e82-412a-8c1a-c5377b5f1555',
 'da926936-9383-463a-8722-fd89e50b6941',
 'e6adaabd-2bd8-4956-9c4d-53cf02d1c0e7',
 '57fd2325-67f4-4d45-9907-29e77d3043d7',
 'bc9ea019-b560-4435-ab53-780d9276f15c',
 'a71175be-d1fd-47a3-aa93-b830ea3634a1',
 '79de526f-aed6-4106-8c26-5dfdfa50ce86',
 'cf43dbb1-6992-40ec-a5f9-e8e838d0f643',
 '572a95d1-39ca-42e1-8424-5c9ffcb2df87',
 '741979ce-3f10-443a-8526-2275620c8473',
 'd42bb88e-add2-414d-a60a-a3efd66acd2a',
 'c7b0e1a3-4d4d-4a76-9339-e73d0ed5425b',
 'ab583ab8-08bd-4ebc-a0b8-5d40af551068',
 'ecb5520d-1358-434c-95ec-93687ecd1396',
 'd9bcf951-067e-41c0-93a2-14818adf88fe',
 '202128f9-02af-4c6c-b6ce-25740e6ba8cd',
 '74bae29c-f614-4abe-8066-c4d83d7da143',
 '1c213d82-32c3-49f7-92ca-06e28907e1b4',
 '810b1e07-009e-4ebe-930a-915e4cd8ece4',
 '36280321-555b-446d-9b7d-c2e17991e090',
 '115d264b-1939-4b8e-9d17-4ed8dfc4fadd',
 'eef82e27-c20e-48da-b4b7-c443031649e3',
 '7bee9f09-a238-42cf-b499-f51f765c6ded',
 'f8d5c8b0-b931-4151-b86c-c471e2e80e5d',
 'c8e60637-de79-4334-8daf-d35f18070c29',
 '097afc11-4214-4879-bd7a-643a4d16396e',
 'ee40aece-cffd-4edb-a4b6-155f158c666a',
 '2199306e-488a-40ab-93cb-2d2264775578',
 '0deb75fb-9088-42d9-b744-012fb8fc4afb',
 '097ba865-f424-49a3-96fb-863506fac3e0',
 '12dc8b34-b18e-4cdd-90a9-da134a9be79c',
 'e49d8ee7-24b9-416a-9d04-9be33b655f40',
 '02fbb6da-3034-47d6-a61b-7d06c796a830',
 '3ce452b3-57b4-40c9-885d-1b814036e936',
 '465c44bd-2e67-4112-977b-36e1ac7e3f8c',
 '931a70ae-90ee-448e-bedb-9d41f3eda647',
 'ff4187b5-4176-4e39-8894-53a24b7cf36b',
 '9f1b915b-d437-4426-8dcc-1124538069e8',
 '1538493d-226a-46f7-b428-59ce5f43f0f9',
 'b03fbc44-3d8e-4a6c-8a50-5ea3498568e0',
 '1c27fd32-e872-4284-b9a5-7079453f4cbc',
 'f3ff65f1-7d59-4abe-b94e-b0478ab5e921',
 '994df46a-6e5f-472d-96dd-0d86e76a8107',
 '3d6f6788-0b99-410f-9703-c43ca3e42a21',
 'bb6a5aae-2431-401d-8f6a-9fdd6de655a9',
 '193fe7a8-4eb5-4f3e-815a-0c45864ddd77',
 '629f25be-1b05-44d0-bcac-e8c40701d5f4',
 '510b1a50-825d-44ce-86f6-9678f5396e02',
 '032ffcdf-7692-40b3-b9ff-8def1fc18b2e',
 '90d1e82c-c96f-496c-ad4e-ee3f02067f25',
 'a9272cce-6914-4b45-a05f-9e925b4c472a',
 'a8a8af78-16de-4841-ab07-fde4b5281a03',
 '3d5996a0-13bc-47ac-baae-e551f106bddc',
 '259927fd-7563-4b03-bc5d-17b4d0fa7a55',
 '2d5f6d81-38c4-4bdc-ac3c-302ea4d5f46e',
 'c9fec76e-7a20-4da4-93ad-04510a89473b',
 'b182b754-3c3e-4942-8144-6ee790926b58',
 '4fa70097-8101-4f10-b585-db39429c5ed0',
 'cb2ad999-a6cb-42ff-bf71-1774c57e5308',
 'b52182e7-39f6-4914-9717-136db589706e',
 'd33baf74-263c-4b37-a0d0-b79dcb80a764',
 '4364a246-f8d7-4ce7-ba23-a098104b96e4',
 '34d20aff-10e5-4a07-8b08-64051a1dc6ac',
 '89f0d6ff-69f4-45bc-b89e-72868abb042a',
 'ea624fbc-2003-452f-be27-7159d0f95abb',
 '5b266a92-49d9-4376-a9f8-46a25b1c9655',
 'ee45f168-6f0e-4535-8824-19e8367bf274',
 '3e7ae7c0-fe8b-487c-9354-036236fa1010',
 'f52282d7-bdc0-4a85-b059-adb37808661c',
 'ff2e3f6c-a338-4c59-829d-0b225055b2df',
 'f2b3b10c-8adf-4ba1-9264-9ed9e1d04e0d',
 'e9b57a5a-b06d-476d-ad20-7ec42a16f5f5',
 'ee5c418c-e7fa-431d-8796-b2033e000b75',
 '2ffd3ed5-477e-4153-9af7-7fdad3c6946b',
 '46459362-8b7d-44c3-a01e-08853b8acf97',
 'a74cc3b6-76ea-46d9-9801-a17c86ae485a',
 '08e25a6c-4223-462a-a856-a3a587146668',
 'd839491f-55d8-4cbe-a298-7839208ba12b',
 'd2918f52-8280-43c0-924b-029b2317e62c',
 'c99d53e6-c317-4c53-99ba-070b26673ac4',
 '4b7fbad4-f6de-43b4-9b15-c7c7ef44db4b',
 '8e5adb43-efaf-4d79-9652-0b480bbc5c2c',
 'fff7d745-bbce-4756-a690-3431e2f3d108',
 '7cffad38-0f22-4546-92b5-fd6d2e8b2be9',
 '4153bd83-2168-4bd4-a15c-f7e82f3f73fb',
 'c6d5cea7-e1c4-48e1-8898-78e039fabf2b',
 '53738f95-bd08-4d9d-9133-483fdb19e8da',
 '614e1937-4b24-4ad3-9055-c8253d089919',
 'aad23144-0e52-4eac-80c5-c4ee2decb198',
 'a3df91c8-52a6-4afa-957b-3479a7d0897c',
 '21e16736-fd59-44c7-b938-9b1333d25da8',
 '15f742e1-1043-45c9-9504-f1e8a53c1744',
 'dd87e278-999d-478b-8cbd-b5bf92b84763',
 'd6d829f9-a5b9-4ea5-a916-c7d2aadeccba',
 '16c3667b-e0ea-43fb-9ad4-8dcd1e6c40e1',
 'a92c4b1d-46bd-457e-a1f4-414265f0e2d4',
 'b9c205c3-feac-485b-a89d-afc96d9cb280',
 'cc45c568-c3b9-4f74-836e-c87762e898c8',
 'c660af59-5803-4846-b36e-ab61afebe081',
 '75c4f32e-1dfc-4375-96bd-41100099753d',
 'a08d3dcc-b8d1-4c22-834f-2a619c399bdf',
 'a4000c2f-fa75-4b3e-8f06-a7cf599b87ad',
 'e5fae088-ed96-4d9b-82f9-dfd13c259d52',
 '6cf2a88a-515b-4f7f-89a2-7d53eab9b5f4',
 'e1931de1-cf7b-49af-af33-2ade15e8abe7',
 '266c32c3-4f75-4d44-9337-ef12f2980ecc',
 '7a887357-850a-4378-bd2a-b5bc8bdd3aac',
 '2d92ad3b-422c-445f-b2ca-4d74692dd7e5',
 '9102a080-f884-4c0a-b7a9-b306eb47201b',
 'ee76b915-f649-4156-827a-ab661b761207',
 'ebe090af-5922-4fcd-8fc6-17b8ba7bad6d',
 '16693458-0801-4d35-a3f1-9115c7e5acfd',
 '920316c5-c471-4196-8db9-0d52cbe55830',
 'dd4da095-4a99-4bf3-9727-f735077dba66',
 '49368f16-de69-4647-9a7a-761e94517821',
 '46b0d871-23d3-4630-8a6b-c79f99b2958c',
 '5139ce2c-7d52-44bf-8129-692d61dd6403',
 'b985b86f-e0e1-4d63-afaf-448b91cb4d74',
 '1211f4af-d3e4-4c4e-9d0b-75a0bc2bf1f0',
 '4458289c-1935-4209-84c3-2e6a80eb45d5',
 'f3f406bd-e138-44c2-8a02-7f11bf8ce87a',
 '63b83ddf-b7ea-40db-b1e2-93c2a769b6e5',
 '713cf757-688f-4fc1-a2f6-2f997c9915c0',
 'f6f947b8-c123-4e27-8933-f624a8c3e8cc',
 '8c2e6449-57f0-4632-9f18-66e6ca90c522',
 '4330cd7d-a513-4385-86ea-ca1a6cc04e1d',
 '2328b53b-0568-4f76-b5a3-d0a6698741a7',
 '31087c1d-e5b0-4a01-baf0-b26ddf03f3ca',
 '1b966923-de4a-4afd-8ed3-5f6842d9ec29',
 'e410ecc5-a808-4fa7-88ca-d5594ebbc76d',
 'ebce500b-c530-47de-8cb1-963c552703ea',
 '5ae68c54-2897-4d3a-8120-426150704385',
 '15b69921-d471-4ded-8814-2adad954bcd8',
 '6899a67d-2e53-4215-a52a-c7021b5da5d4',
 'ca4ecb4c-4b60-4723-9b9e-2c54a6290a53',
 '6ab9d98c-b1e9-4574-b8fe-b9eec88097e0',
 '239dd3c9-35f3-4462-95ee-91b822a22e6b',
 'b196a2ad-511b-4e90-ac99-b5a29ad25c22',
 '1b715600-0cbc-442c-bd00-5b0ac2865de1',
 '5dcee0eb-b34d-4652-acc3-d10afc6eae68',
 'c7bf2d49-4937-4597-b307-9f39cb1c7b16',
 'ebce500b-c530-47de-8cb1-963c552703ea',
 '5ae68c54-2897-4d3a-8120-426150704385',
 '15b69921-d471-4ded-8814-2adad954bcd8',
 '6899a67d-2e53-4215-a52a-c7021b5da5d4',
 'ca4ecb4c-4b60-4723-9b9e-2c54a6290a53',
 '6ab9d98c-b1e9-4574-b8fe-b9eec88097e0',
 '239dd3c9-35f3-4462-95ee-91b822a22e6b',
 'b196a2ad-511b-4e90-ac99-b5a29ad25c22',
 '1b715600-0cbc-442c-bd00-5b0ac2865de1',
 '5dcee0eb-b34d-4652-acc3-d10afc6eae68',
 'c7bf2d49-4937-4597-b307-9f39cb1c7b16',
 '3a3ea015-b5f4-4e8b-b189-9364d1fc7435',
 'd85c454e-8737-4cba-b6ad-b2339429d99b']
 
 
processed_sessions_ephys, unique_acronyms_set_ephys, successful_session_ids_ephys,successful_pids_ephys = process_all_sessions_rep(ephys_session_id,0)


base_output_dir="/storage1/fs1/hiratani/Active/shared/ibl_space/ONE/openalyx.internationalbrainlab.org/HMM_Results"

#  dictionary to store session information
rep_session_info = {
    "processed_sessions_ephys": processed_sessions_ephys,
    "unique_acronyms_set_ephys": unique_acronyms_set_ephys,
    "successful_session_ids_ephys": successful_session_ids_ephys,
    "successful_pids_ephys": successful_pids_ephys
}


os.makedirs(base_output_dir, exist_ok=True)


pkl_file_path = os.path.join(base_output_dir, "ephys_session_info_1.pkl")

# Save dictionary to a pickle file
with open(pkl_file_path, "wb") as f:
    pickle.dump(rep_session_info, f)

print(f" ephys_session_info saved to {pkl_file_path}")

