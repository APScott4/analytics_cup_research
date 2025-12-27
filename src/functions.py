import pandas as pd
import numpy as np
from collections import Counter
from unravel.utils import GraphDataset
import requests
import polars as pl
import json
from os.path import exists
from kloppy import skillcorner
from unravel.soccer import KloppyPolarsDataset, SoccerGraphConverter


def classify_play_out_from_back_with_trace(phase_row, phase_data, event_data):
    """
    Determines if a play out from the back was successful or failed,
    using both phase and event-level information.
    Returns both the success label and the involved phases.
    """
    current_index = phase_row['index']
    team_id = phase_row['team_in_possession_id']

    # Only consider future phases from the SAME team
    future_phases = phase_data[phase_data['index'] >= current_index]
    future_phases = future_phases[future_phases['team_in_possession_id'] == team_id].reset_index(drop=True)

    involved_phases = []

    for _, phase in future_phases.iterrows():
        involved_phases.append(phase['index'])

        # --- 1️⃣ Check event-level success conditions ---
        phase_events = event_data[
            (event_data['frame_start'] >= phase['frame_start']) &
            (event_data['frame_end'] <= phase['frame_end']) &
            (event_data['team_id'] == team_id)
        ]

        if not phase_events.empty:
            if (
                (phase_events['player_in_possession_third_start'].eq('middle_third').any()) or
                (phase_events['player_in_possession_third_end'].eq('middle_third').any())
            ):
                return pd.Series({'play_out_success': 'success', 'involved_phases': involved_phases})

        # --- 2️⃣ Phase-level fail conditions ---
        if (phase['team_possession_loss_in_phase'] == True) or \
           (phase['team_in_possession_phase_type'] in ['chaotic', 'direct']):
            return pd.Series({'play_out_success': 'fail', 'involved_phases': involved_phases})

        # --- 3️⃣ Phase-level success conditions ---
        if (phase['third_end'] in ['middle_third', 'attacking_third']) or \
           (phase['team_possession_lead_to_shot'] == True) or \
           (phase['team_possession_lead_to_goal'] == True):
            return pd.Series({'play_out_success': 'success', 'involved_phases': involved_phases})

    # If none met
    return pd.Series({'play_out_success': 'unknown', 'involved_phases': involved_phases})




# df is your original DataFrame with play data
# phase_data is the same DataFrame (or another one) that has frame_start and frame_end for each phase index



def filtered_join(filtered_phase, phase_data):
    rows = []
    seen_frames = set()
    filtered_phase["play_out_success"] = filtered_phase["play_out_success"].map({"success": 1, "fail": 0})


    for _, row in filtered_phase.iterrows():
        involved = row.involved_phases  # list of phase indices
        if not involved:
            continue

        # Get first and last phase indices
        first_phase = involved[0]
        last_phase = involved[-1]
        
        # Look up frame_start and frame_end from phase_data
        frame_start = phase_data.loc[phase_data.index == first_phase, "frame_start"].values[0]
        frame_end = phase_data.loc[phase_data.index == last_phase, "frame_end"].values[0]

        # Expand all frames in that range
        frames = np.arange(frame_start, frame_end + 1)
        for frame in frames:
            if frame not in seen_frames:
                seen_frames.add(frame)
                rows.append({
                    "game_id": row.match_id,
                    "period_id": row.period,
                    "frame_id": frame,
                    "label": row.play_out_success
                })
    return rows

# Make final DataFrame

def remove_nan_graphs(graph_dataset):
    cleaned_graphs = []
    for g in graph_dataset.graphs:
        # Keep only labels that are not 'NaN'
        mask = g.y != 'NaN'
        if np.any(mask):  # make sure there's at least one valid label
            g.y = g.y[mask]
            cleaned_graphs.append(g)

    # Return a new GraphDataset with cleaned graphs
    return GraphDataset(graphs=cleaned_graphs)

def label_distribution(graphs):
    all_labels = np.concatenate([g.y for g in graphs])
    return Counter(all_labels)

def graphs_to_raw(graph_list):
    raw = {
        "x": [],
        "a": [],
        "e": [],
        "y": []
    }

    for g in graph_list:
        raw["x"].append(g.x)
        raw["a"].append(g.a)
        raw["e"].append(g.e)
        raw["y"].append(g.y)

    return raw

def process_matches(match_ids, pickle_folder="pickles"):
    """
    Downloads and processes SkillCorner match data, filters play-out-from-back phases,
    and saves the resulting graph dataset as compressed pickles.

    Parameters
    ----------
    match_ids : list[str]
        List of match IDs to process
    pickle_folder : str
        Folder to store pickle files
    """
    
    compressed_pickle_file_path = f"{pickle_folder}/{{match_id}}.pickle.gz"
    
    for match_id in match_ids:
        print(f"Processing match {match_id}...")
        
        # Load match metadata
        meta_url = f'https://raw.githubusercontent.com/SkillCorner/opendata/master/data/matches/{match_id}/{match_id}_match.json'
        meta_data = requests.get(meta_url).json()
        
        with open(f"{match_id}_match.json", "w") as f:
            json.dump(meta_data, f)
        
        match_pickle_file_path = compressed_pickle_file_path.format(match_id=match_id)
        
        if exists(match_pickle_file_path):
            print(f"Match {match_id} already processed, skipping.")
            continue
        
        # Load phase and event data
        phase_data = pd.read_csv(f"https://raw.githubusercontent.com/SkillCorner/opendata/master/data/matches/{match_id}/{match_id}_phases_of_play.csv")
        event_data = pd.read_csv(f"https://raw.githubusercontent.com/SkillCorner/opendata/master/data/matches/{match_id}/{match_id}_dynamic_events.csv")
        
        # Filter relevant events
        filtered_events = event_data[
            ((event_data['game_interruption_before'].notna()) &
             (event_data['third_start'] == 'defensive_third') &
             ((event_data['team_in_possession_phase_type'].isin(['build_up', 'create'])))) |
            ((event_data['hand_pass'].notna()) &
             (event_data['player_position']=='GK') &
             (event_data['first_player_possession_in_team_possession']==True))
        ]
        
        # Map events to phases
        def find_phase(frame):
            match = phase_data[(phase_data['frame_start'] <= frame) & (phase_data['frame_end'] >= frame)]
            return match['index'].iloc[0] if not match.empty else None

        filtered_events = filtered_events.copy()
        filtered_events['phase_id'] = filtered_events['frame_end'].apply(find_phase)
        phase_ids = filtered_events['phase_id'].dropna().unique()
        
        filtered_phase = phase_data.loc[phase_data['index'].isin(phase_ids)]
        filtered_phase = filtered_phase[filtered_phase['team_in_possession_phase_type'].isin(['build_up', 'create'])]
        
        # Classify phases
        filtered_phase[['play_out_success', 'involved_phases']] = filtered_phase.apply(
            classify_play_out_from_back_with_trace,
            axis=1,
            phase_data=phase_data,
            event_data=event_data
        )
        
        rows = filtered_join(filtered_phase, phase_data)
        
        # Load SkillCorner dataset
        dataset = skillcorner.load_open_data(match_id=match_id, coordinates="skillcorner", only_alive=False)
        kloppy_polars_dataset = KloppyPolarsDataset(dataset, ball_carrier_threshold=25.0)
        kloppy_polars_dataset.add_graph_ids(by=["game_id", "period_id"])
        
        # Merge labels
        polar_labels = pl.DataFrame(rows).with_columns(pl.col("game_id").cast(pl.Utf8))
        kloppy_polars_dataset.data = kloppy_polars_dataset.data.join(
            polar_labels.select(["game_id", "period_id", "frame_id", "label"]),
            on=["game_id", "period_id", "frame_id"],
            how="inner"
        )
        
        # Convert to graph
        converter = SoccerGraphConverter(
            dataset=kloppy_polars_dataset,
            self_loop_ball=True,
            adjacency_matrix_connect_type="ball",
            adjacency_matrix_type="split_by_team",
            label_type="binary",
            defending_team_node_value=0.1,
            non_potential_receiver_node_value=0.1,
            random_seed=False,
            pad=False,
            verbose=False
        )
        
        print(f"Saving match {match_id} to {match_pickle_file_path}")
        converter.to_pickle(file_path=match_pickle_file_path)


