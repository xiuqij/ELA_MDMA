#%%
import os
import pandas as pd
import numpy as np
import joblib
import pyarrow.parquet as pq
import gc
import psutil
import os
from numpy.lib.stride_tricks import sliding_window_view

#%%
'''
workflow
1. parquet files -> subset into pairs -> add calculated features
2. run model prediction
3. threshold, make events
'''


def log_memory(tag=""):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024**3)
    print(f"[MEM] {tag}: {mem:.2f} GB")
### HELPERS - PREPARE INPUT DATAFRAME ###
def load_pair_from_parquet(pq_path, pair):
    '''Updated function for parquet loading (read only needed columns per pair)'''
    mouse_dict = {'B':'Blue','G':'Green','R':'Red','Y':'Yellow'}
    mice = [mouse_dict[i] for i in pair]

    # full schema
    parquet_file = pq.ParquetFile(pq_path)
    all_cols = parquet_file.schema.names

    # each mouse = 24 columns
    cols_per_mouse = len(all_cols) // 4
    mice_order = ['Red','Blue','Green','Yellow']

    selected_cols = []
    for i, m in enumerate(mice_order):
        if m in mice:
            start = i * cols_per_mouse
            end = (i + 1) * cols_per_mouse
            selected_cols.extend(all_cols[start:end])

    table = parquet_file.read(columns=selected_cols)
    df = table.to_pandas()

    # reduce memory
    df = df.astype('float32')

    return df

# HELPERS - CALCULATING FEATURES
def euclidean_distance(x1, x2, y1, y2, px_per_mm):
    '''Compute Euclidean distance in millimeters between two body-parts.'''
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) / px_per_mm

def movement(x, y, px_per_mm):
    dx = x.diff().fillna(0)
    dy = y.diff().fillna(0)
    return np.sqrt(dx**2 + dy**2) / px_per_mm

def get_bp(df, animal, bp):
    """Helper to extract x,y columns"""
    return df[f"{bp}_{animal}_x"], df[f"{bp}_{animal}_y"]

def compute_features(df, PX_PER_MM=0.59):
    '''Get all features needed (17 cols)'''
    # rename cols (no copy)
    ANIMALS = ["1", "2"]
    BODYPARTS = ["Ear_left","Ear_right","Nose","Center","Lat_left","Lat_right","Tail_base","Tail_end"]
    COORDS = ["x", "y", "p"]

    expected_cols = [
        f"{bp}_{animal}_{coord}"
        for animal in ANIMALS
        for bp in BODYPARTS
        for coord in COORDS
    ]

    if len(expected_cols) != len(df.columns):
        raise ValueError(f"Column mismatch! Expected {len(expected_cols)}, got {len(df.columns)}")

    df.columns = expected_cols

    # extract bps needed
    M1_nose_x, M1_nose_y = get_bp(df, "1", "Nose")
    M2_nose_x, M2_nose_y = get_bp(df, "2", "Nose")

    M1_cent_x, M1_cent_y = get_bp(df, "1", "Center")
    M2_cent_x, M2_cent_y = get_bp(df, "2", "Center")

    M2_latL_x, M2_latL_y = get_bp(df, "2", "Lat_left")
    M2_latR_x, M2_latR_y = get_bp(df, "2", "Lat_right")

    M1_latL_x, M1_latL_y = get_bp(df, "1", "Lat_left")
    M1_latR_x, M1_latR_y = get_bp(df, "1", "Lat_right")

    # calculate
    # --- distances ---
    centroid_dist = euclidean_distance(M1_cent_x, M2_cent_x, M1_cent_y, M2_cent_y, PX_PER_MM)

    nose_dist = euclidean_distance(M1_nose_x, M2_nose_x, M1_nose_y, M2_nose_y, PX_PER_MM)

    m1_to_m2_L = euclidean_distance(M1_nose_x, M2_latL_x, M1_nose_y, M2_latL_y, PX_PER_MM)
    m1_to_m2_R = euclidean_distance(M1_nose_x, M2_latR_x, M1_nose_y, M2_latR_y, PX_PER_MM)

    m2_to_m1_L = euclidean_distance(M2_nose_x, M1_latL_x, M2_nose_y, M1_latL_y, PX_PER_MM)
    m2_to_m1_R = euclidean_distance(M2_nose_x, M1_latR_x, M2_nose_y, M1_latR_y, PX_PER_MM)

    # --- movement ---
    move1 = movement(M1_cent_x, M1_cent_y, PX_PER_MM)
    move2 = movement(M2_cent_x, M2_cent_y, PX_PER_MM)

    # --- chase features ---
    rel_speed = np.abs(move1 - move2)

    nose1_body2 = np.minimum(m1_to_m2_L, m1_to_m2_R)
    nose2_body1 = np.minimum(m2_to_m1_L, m2_to_m1_R)

    # --- pursuit alignment ---
    dx1 = M1_cent_x.diff().fillna(0)
    dy1 = M1_cent_y.diff().fillna(0)
    dx2 = M2_cent_x.diff().fillna(0)
    dy2 = M2_cent_y.diff().fillna(0)

    rx12 = M2_cent_x - M1_cent_x
    ry12 = M2_cent_y - M1_cent_y

    rx21 = -rx12
    ry21 = -ry12

    def safe_cos(dx, dy, rx, ry):
        return (dx * rx + dy * ry) / (np.sqrt(dx**2 + dy**2) * np.sqrt(rx**2 + ry**2) + 1e-6)

    pursuit_1 = safe_cos(dx1, dy1, rx12, ry12)
    pursuit_2 = safe_cos(dx2, dy2, rx21, ry21)

    # --- FINAL FEATURE DF (17 columns) ---
    df_feats = pd.DataFrame({
        "Centroid_distance": centroid_dist,
        "Nose_to_nose_distance": nose_dist,
        "M1_Nose_to_M2_lat_left": m1_to_m2_L,
        "M1_Nose_to_M2_lat_right": m1_to_m2_R,
        "M2_Nose_to_M1_lat_left": m2_to_m1_L,
        "M2_Nose_to_M1_lat_right": m2_to_m1_R,
        "Movement_mouse_1_centroid": move1,
        "Movement_mouse_2_centroid": move2,
        "pursuit_cos_1": pursuit_1,
        "pursuit_cos_2": pursuit_2,
        "rel_speed": rel_speed,
        "nose1_to_body2": nose1_body2,
        "nose2_to_body1": nose2_body1,
        "Center_1_x": M1_cent_x,
        "Center_1_y": M1_cent_y,
        "Center_2_x": M2_cent_x,
        "Center_2_y": M2_cent_y
    })

    del df
    return df_feats


### HELPERS - RUN PREDICTIONS
def frames_to_events(pred_labels, frames):
    """
    Convert a 0/1 frame label vector into a list of (start_frame, end_frame) events.
    """
    events = []
    in_event = False
    start_frame = None

    for i, lab in enumerate(pred_labels):
        if lab == 1 and not in_event:
            in_event = True
            start_frame = frames[i]
        elif lab == 0 and in_event:
            in_event = False
            end_frame = frames[i - 1]
            events.append((start_frame, end_frame))

    if in_event:
        events.append((start_frame, frames[len(pred_labels) - 1]))

    return events

def merge_close_events(events, max_gap):
    """
    events: list of (start, end) tuples, sorted by start.
    max_gap: maximum allowed gap (in frames) between events to merge.

    Returns a new list of merged (start, end) events.
    """
    if not events:
        return []

    events = sorted(events, key=lambda x: x[0])
    merged = []
    cur_start, cur_end = events[0]

    for s, e in events[1:]:
        # if next event starts before or within max_gap after current end
        if s <= cur_end + max_gap:
            # extend current event
            cur_end = max(cur_end, e)
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = s, e

    merged.append((cur_start, cur_end))
    return merged

def add_chase_direction_role(df):
    df = df.copy()

    # relative vector 1->2
    rx = df["Center_2_x"] - df["Center_1_x"]
    ry = df["Center_2_y"] - df["Center_1_y"]

    # velocities (frame-to-frame centroid motion)
    v1x = df["Center_1_x"].diff().fillna(0)
    v1y = df["Center_1_y"].diff().fillna(0)
    v2x = df["Center_2_x"].diff().fillna(0)
    v2y = df["Center_2_y"].diff().fillna(0)

    def safe_cos(vx, vy, rx, ry):
        num = vx * rx + vy * ry
        denom = np.sqrt(vx**2 + vy**2) * np.sqrt(rx**2 + ry**2) + 1e-6
        return num / denom

    # how much each mouse moves toward the other
    df["cos_1_toward_2"] = safe_cos(v1x, v1y, rx, ry)
    df["cos_2_toward_1"] = safe_cos(v2x, v2y, -rx, -ry)

    return df

def load_model(model_path):
    return joblib.load(model_path)

def sliding_window_prediction(model, df_feats, pair, video,
                              WINDOW_SIZE=20,
                              WINDOW_STEP=5,
                              PROB_THR=0.9,
                              MAX_GAP_FRAMES=3,
                              MARGIN=0.1):

    """Run prediction on one pair feature dataframe (17 columns expected)."""

    # set columns in the output dataframe 
    COLUMNS = ["video","pair","start","end","duration",
               "chaser","chased","box","date","timestamp"]

    n_frames = len(df_feats)
    frames = df_feats.index.to_numpy()
    # Skip if too short
    if n_frames < WINDOW_SIZE:
        return pd.DataFrame(columns=COLUMNS)
    
    # ---- direction features (keep as pandas for later indexing) ----
    df_dir = add_chase_direction_role(df_feats.copy())    
    
    # ---- convert to numpy early ----
    X = df_feats.to_numpy(dtype=np.float32)

    # ---- number of windows ----
    n_windows = (n_frames - WINDOW_SIZE) // WINDOW_STEP + 1
   
    # ---- create rolling windows  ----
    shape = ( n_windows,WINDOW_SIZE,X.shape[1] )
    strides = (X.strides[0]*WINDOW_STEP,X.strides[0],X.strides[1])
    windows = np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)
    #windows = sliding_window_view(X, (WINDOW_SIZE, X.shape[1]))[::WINDOW_STEP, 0]
    
    # ---- mean per window ----
    X_win = windows.mean(axis=1)

    # ---- predict in batch ----
    probs = model.predict_proba(X_win)[:,1]
    preds = probs >= PROB_THR

    # ---- reconstruct frame labels ----
    pred_frame_labels = np.zeros(n_frames, dtype=np.uint8)

    for i, label in enumerate(preds):
        if label:
            s = i * WINDOW_STEP
            e = s + WINDOW_SIZE
            pred_frame_labels[s:e] = 1

    # ---- events ----
    events = frames_to_events(pred_frame_labels, frames)
    events = merge_close_events(events, MAX_GAP_FRAMES)

    if not events:
        return pd.DataFrame(columns=COLUMNS)

    # ---- metadata parsing (safe) ----
    parts = video.split('_')
    date = parts[0] if len(parts) > 0 else "NA"
    timestamp = parts[1] if len(parts) > 1 else "NA"
    box = parts[3] if len(parts) > 3 else "NA"

    mouse_dict = {'B':'Blue','G':'Green','R':'Red','Y':'Yellow'}
    m1_color, m2_color = [mouse_dict[i] for i in pair]

    # ---- assign roles ----
    pred_events = []

    for (ps, pe) in events:
        # use iloc for safety
        seg = df_dir.iloc[ps:pe+1]

        mean_cos_1 = seg["cos_1_toward_2"].mean()
        mean_cos_2 = seg["cos_2_toward_1"].mean()

        if mean_cos_1 - mean_cos_2 > MARGIN:
            chaser_color = m1_color
            chased_color = m2_color
        elif mean_cos_2 - mean_cos_1 > MARGIN:
            chaser_color = m2_color
            chased_color = m1_color
        else:
            chaser_color = "undetermined"
            chased_color = "undetermined"

        pred_events.append({
            "video": video,
            "pair": pair,
            "start": int(ps),
            "end": int(pe),
            "duration": int(pe - ps + 1),
            "chaser": chaser_color.lower(),
            "chased": chased_color.lower(),
            "box": box,
            "date": date,
            "timestamp": timestamp
        })

    return pd.DataFrame(pred_events, columns=COLUMNS)

def process_one_file(model, pq_path, video, output_path, first_write,
                     WINDOW_SIZE=20, WINDOW_STEP=5, PROB_THR=0.9,
                     MAX_GAP_FRAMES=3,MARGIN=0.1):
    pairs = ['BG','BY','RB','RG','RY','GY']

    for pair in pairs:
        try:
            # ---- load minimal data ----
            df = load_pair_from_parquet(pq_path, pair)
            #log_memory(f"{video} - after load")
            # ---- compute features ----
            df_feats = compute_features(df)
            #log_memory(f"{video} - after features")
            # free raw df early 
            del df

            if df_feats is None or df_feats.empty:
                continue

            # ---- prediction ----
            df_events = sliding_window_prediction(
                model=model,
                df_feats=df_feats,
                pair=pair,
                video=video,
                WINDOW_SIZE=WINDOW_SIZE,
                WINDOW_STEP=WINDOW_STEP,
                PROB_THR=PROB_THR,
                MAX_GAP_FRAMES=MAX_GAP_FRAMES,
                MARGIN=MARGIN
            )
            #log_memory(f"{video} - after prediction")
            # ---- write immediately ----
            if df_events is not None and not df_events.empty:
                df_events.to_csv(
                    output_path,
                    mode='w' if first_write else 'a',
                    header=first_write,
                    index=False
                )
                first_write = False

            # ---- cleanup ----
            del df_feats, df_events
            gc.collect()
            #log_memory(f"{video} - after cleanup")

        except Exception as e:
            print(f"[WARNING] {video} pair {pair} failed: {e}")
            continue

    return first_write

def process_one_file_batch(model, pq_path, video,
                          WINDOW_SIZE=20, WINDOW_STEP=5,
                          PROB_THR=0.9,
                          MAX_GAP_FRAMES=3,
                          MARGIN=0.1):

    pairs = ['BG','BY','RB','RG','RY','GY']
    all_events = []

    for pair in pairs:
        try:
            df = load_pair_from_parquet(pq_path, pair)
            df_feats = compute_features(df)
            del df

            if df_feats is None or df_feats.empty:
                continue

            df_events = sliding_window_prediction(
                model=model,
                df_feats=df_feats,
                pair=pair,
                video=video,
                WINDOW_SIZE=WINDOW_SIZE,
                WINDOW_STEP=WINDOW_STEP,
                PROB_THR=PROB_THR,
                MAX_GAP_FRAMES=MAX_GAP_FRAMES,
                MARGIN=MARGIN
            )

            if df_events is not None and not df_events.empty:
                all_events.append(df_events)

            del df_feats, df_events
            gc.collect()

        except Exception as e:
            print(f"[WARNING] {video} pair {pair} failed: {e}")

    if len(all_events) > 0:
        return pd.concat(all_events, ignore_index=True)

    return pd.DataFrame()

def sliding_window_prediction_slow(model, df_feats, pair, video,
                              WINDOW_SIZE=20,
                              WINDOW_STEP=5,
                              PROB_THR=0.9,
                              MAX_GAP_FRAMES=3,
                              MARGIN=0.1):

    """Run prediction on one pair feature dataframe (17 columns expected)."""

    # set columns in the output dataframe 
    COLUMNS = ["video","pair","start","end","duration",
               "chaser","chased","box","date","timestamp"]

    n_frames = len(df_feats)
    frames = df_feats.index.to_numpy()
    # Skip if too short
    if n_frames < WINDOW_SIZE:
        return pd.DataFrame(columns=COLUMNS)
    
    # ---- direction features (keep as pandas for later indexing) ----
    df_dir = add_chase_direction_role(df_feats.copy())    
    
    # ---- convert to numpy early ----
    X = df_feats.to_numpy(dtype=np.float32)

    # ---- number of windows ----
    starts = list(range(0, n_frames - WINDOW_SIZE + 1, WINDOW_STEP))
   
    # ---- create rolling windows (fast alternative but be careful) ----
    windows = np.array([X[s:s+WINDOW_SIZE] for s in starts])
    # ---- mean per window ----
    X_win = windows.mean(axis=1)

    # ---- predict in batch ----
    probs = model.predict_proba(X_win)[:,1]
    preds = probs >= PROB_THR

    # ---- reconstruct frame labels ----
    pred_frame_labels = np.zeros(n_frames, dtype=np.uint8)

    for i, label in enumerate(preds):
        if label:
            s = i * WINDOW_STEP
            e = s + WINDOW_SIZE
            pred_frame_labels[s:e] = 1

    # ---- events ----
    events = frames_to_events(pred_frame_labels, frames)
    events = merge_close_events(events, MAX_GAP_FRAMES)

    if not events:
        return pd.DataFrame(columns=COLUMNS)

    # ---- metadata parsing (safe) ----
    parts = video.split('_')
    date = parts[0] if len(parts) > 0 else "NA"
    timestamp = parts[1] if len(parts) > 1 else "NA"
    box = parts[3] if len(parts) > 3 else "NA"

    mouse_dict = {'B':'Blue','G':'Green','R':'Red','Y':'Yellow'}
    m1_color, m2_color = [mouse_dict[i] for i in pair]

    # ---- assign roles ----
    pred_events = []

    for (ps, pe) in events:
        # use iloc for safety
        seg = df_dir.iloc[ps:pe+1]

        mean_cos_1 = seg["cos_1_toward_2"].mean()
        mean_cos_2 = seg["cos_2_toward_1"].mean()

        if mean_cos_1 - mean_cos_2 > MARGIN:
            chaser_color = m1_color
            chased_color = m2_color
        elif mean_cos_2 - mean_cos_1 > MARGIN:
            chaser_color = m2_color
            chased_color = m1_color
        else:
            chaser_color = "undetermined"
            chased_color = "undetermined"

        pred_events.append({
            "video": video,
            "pair": pair,
            "start": int(ps),
            "end": int(pe),
            "duration": int(pe - ps + 1),
            "chaser": chaser_color.lower(),
            "chased": chased_color.lower(),
            "box": box,
            "date": date,
            "timestamp": timestamp
        })

    return pd.DataFrame(pred_events, columns=COLUMNS)