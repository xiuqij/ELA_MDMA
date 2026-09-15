#%%
'''
Fine-resolution (15min/30min) companion to regroup_timebins.py / chase_stats.py /
locomotion_stats.py / nest_social.py / ROI_social.py.

Purely additive: writes new {name}_15min.csv / {name}_30min.csv files alongside the
existing {name}_{n}h.csv files, in the same folders. Does not import from or modify
any of those scripts, and does not touch utils_stats.py's existing functions - only
the new functions added there (split_events_by_interval_with_frame,
add_fine_time_labels, add_timebin_labels_fine, convert_to_resolution,
format_resolution) are used.

Same data scope as the existing pipeline: both batches, all 4 exps, both time_points.
'''
import os
import pandas as pd
import numpy as np
import utils_stats as utils

#%%
# Define paths - same scope as regroup_timebins.py
main_path = "/Volumes/labs/Lopez Laboratory - NEURO/Xiuqi/ELA_MDMA"
batches = ['April','July']
exps = {'April':['male_P35','female_P42'],'July':['female_P35','male_P42']}
time_points = ['baseline','MDMA']
resolutions = [0.25, 0.5]
fps = 25

def apply_day1_correction(df, b, exp, t):
    '''April male_P35 MDMA recording started late -> day off by one. Matches the
    correction applied inline for ROIs/motionless/speeding in regroup_timebins.py
    (nest there only gets this via a one-time post-hoc CSV patch; applied inline here
    for all behaviors since this pipeline is generated fresh each run).'''
    if b == 'April' and exp == 'male_P35' and t == 'MDMA':
        df = df.copy()
        df['day'] = df['day'] + 1
    return df

def regroup_by_timebin_fine(df_fine, resolution, group_base=['day','phase','box','mouse'], nest=False):
    '''Same aggregation as utils.regroup_by_timebin, but starting from event-level rows
    carrying the fractional ZT_time label (from convert_to_resolution) and binning via
    add_timebin_labels_fine - utils.regroup_by_timebin instead calls add_timebin_labels
    (keyed on the integer ZT_hour) internally, which can't distinguish sub-hour bins.'''
    df = utils.add_timebin_labels_fine(df_fine.copy(), resolution)
    group_cols = group_base + ['time_bin','time_window']
    group_df = (
        df.groupby(group_cols, observed=True)['duration_f']
        .agg(duration_f='sum', count='size')
        .reset_index()
    )
    group_df['duration'] = group_df['duration_f'] / fps
    group_df['mean_duration'] = np.where(group_df['count'] > 0, group_df['duration'] / group_df['count'], 0)
    if nest:
        unit_total_s = resolution * 60 * 60
        group_df['outside_nest_duration'] = unit_total_s - group_df['duration']
    return group_df

def normalize_by_nest_fine(df, nest_df, group_cols=['day','phase','box','mouse','time_bin','time_window']):
    '''Same idea as utils.normalize_by_nest, but keyed to regroup_by_timebin_fine's
    actual output column names (duration/count/mean_duration). The existing
    nest_{n}h.csv files on disk predate a utils_stats.py refactor and still use an
    older total_time/avg_time/outside_total_time naming that utils.normalize_by_nest's
    rename dict targets; this fine-resolution pipeline is generated fresh with current
    code, so its own nest_15min.csv/nest_30min.csv use the current naming instead.'''
    nest_df = nest_df.rename(columns={'duration': 'nest_duration', 'count': 'nest_count', 'mean_duration': 'nest_mean_duration'})
    nest_cols = group_cols + ['nest_duration', 'outside_nest_duration', 'nest_count', 'nest_mean_duration']
    df = df.merge(nest_df[nest_cols], how='outer', on=group_cols)
    df['duration_fraction'] = df['duration'] / df['outside_nest_duration']
    df['event_rate'] = df['count'] / df['outside_nest_duration']
    return df

#%%
# nest
keep_cols = ['day','ZT_hour','ZT_time','phase','box','mouse','duration_f']
base_cols = ['day','phase','box','mouse']
for b in batches:
    print(b)
    path = os.path.join(main_path,f'{b}_2026')
    for exp in exps[b]:
        print(exp)
        for t in time_points:
            print(t)
            df_4h = pd.read_csv(os.path.join(path,'nest',exp,t,'nest_events.csv'))
            df_fine = utils.convert_to_resolution(df_4h, interval_minutes=15, keep_cols=keep_cols)
            df_fine = apply_day1_correction(df_fine, b, exp, t)
            for res in resolutions:
                df_res = regroup_by_timebin_fine(df_fine, resolution=res, group_base=base_cols, nest=True)
                df_res.to_csv(os.path.join(path,'nest',exp,t,f'nest_{utils.format_resolution(res)}.csv'),index=False)

#%%
# ROIs
keep_cols = ['day','ZT_hour','ZT_time','phase','box','mouse','duration_f']
base_cols = ['day','phase','box','mouse']
rois = ['s_wall','ramp1','ramp2','non_wall','woodstick','feeder_prox', 'feeder_dist','water_prox','water_dist']
for b in batches:
    print(b)
    path = os.path.join(main_path,f'{b}_2026')
    for exp in exps[b]:
        print(exp)
        for t in time_points:
            print(t)
            for roi in rois:
                df_4h = pd.read_csv(os.path.join(path,'ROIs',exp,t,f'{roi}_events.csv'))
                df_fine = utils.convert_to_resolution(df_4h, interval_minutes=15, keep_cols=keep_cols)
                df_fine = apply_day1_correction(df_fine, b, exp, t)
                for res in resolutions:
                    df_res = regroup_by_timebin_fine(df_fine, resolution=res, group_base=base_cols, nest=False)
                    nest_df = pd.read_csv(os.path.join(path,'nest',exp,t,f'nest_{utils.format_resolution(res)}.csv'))
                    df_res = normalize_by_nest_fine(df_res, nest_df=nest_df)
                    df_res = df_res.rename(columns = {'duration':f'{roi}_duration',
                                                      'count':f'{roi}_count',
                                                      'mean_duration':f'{roi}_mean_duration',
                                                      'duration_fraction':f'{roi}_duration_fraction',
                                                      'event_rate':f'{roi}_event_rate'})
                    df_res.to_csv(os.path.join(path,'ROIs',exp,t,f'{roi}_{utils.format_resolution(res)}.csv'),index=False)

#%%
# motionless
keep_cols = ['day','ZT_hour','ZT_time','phase','box','mouse','duration_f']
base_cols = ['day','phase','box','mouse']
for b in batches:
    print(b)
    path = os.path.join(main_path,f'{b}_2026')
    for exp in exps[b]:
        print(exp)
        for t in time_points:
            print(t)
            df_4h = pd.read_csv(os.path.join(path, 'immobility',exp,t,'motionless_events.csv'))
            df_4h["date"] = df_4h["video"].str[:10]
            df_4h["timestamp"] = df_4h["video"].str[11:19]
            df_fine = utils.convert_to_resolution(df_4h, interval_minutes=15, keep_cols=keep_cols)
            df_fine = apply_day1_correction(df_fine, b, exp, t)
            for res in resolutions:
                df_res = regroup_by_timebin_fine(df_fine, resolution=res, group_base=base_cols, nest=False)
                nest_df = pd.read_csv(os.path.join(path,'nest',exp,t,f'nest_{utils.format_resolution(res)}.csv'))
                df_res = normalize_by_nest_fine(df_res, nest_df=nest_df)
                df_res = df_res.rename(columns = {'duration':'motionless_duration','count':'motionless_count',
                                                'mean_duration':'motionless_mean_duration',
                                                'duration_fraction':'motionless_duration_fraction',
                                                'event_rate':'motionless_event_rate'})
                df_res.to_csv(os.path.join(path,'immobility',exp,t,f'motionless_{utils.format_resolution(res)}.csv'),index=False)

#%%
# speeding
keep_cols = ['day','ZT_hour','ZT_time','phase','box','mouse','duration_f']
base_cols = ['day','phase','box','mouse']
for b in batches:
    print(b)
    path = os.path.join(main_path,f'{b}_2026')
    for exp in exps[b]:
        print(exp)
        for t in time_points:
            print(t)
            df_4h = pd.read_csv(os.path.join(path, 'immobility',exp,t,'speeding_events.csv'))
            df_4h["date"] = df_4h["video"].str[:10]
            df_4h["timestamp"] = df_4h["video"].str[11:19]
            df_fine = utils.convert_to_resolution(df_4h, interval_minutes=15, keep_cols=keep_cols)
            df_fine = apply_day1_correction(df_fine, b, exp, t)
            for res in resolutions:
                df_res = regroup_by_timebin_fine(df_fine, resolution=res, group_base=base_cols, nest=False)
                nest_df = pd.read_csv(os.path.join(path,'nest',exp,t,f'nest_{utils.format_resolution(res)}.csv'))
                df_res = normalize_by_nest_fine(df_res, nest_df=nest_df)
                df_res = df_res.rename(columns = {'duration':'speeding_duration','count':'speeding_count',
                                                'mean_duration':'speeding_mean_duration',
                                                'duration_fraction':'speeding_duration_fraction',
                                                'event_rate':'speeding_event_rate'})
                df_res.to_csv(os.path.join(path,'immobility',exp,t,f'speeding_{utils.format_resolution(res)}.csv'),index=False)

#%%
# chase
def chase_regroup_by_timebin_fine(df_fine, resolution, group_base=['day','phase','box','mouse']):
    df = utils.add_timebin_labels_fine(df_fine.copy(), resolution)
    group_cols = group_base + ['time_bin','time_window']
    group_df = (
        df.groupby(group_cols,observed=True)
        .agg(
            frames_chasing = ('frames_chasing','sum'),
            frames_chased = ('frames_chased','sum'),
            chasing_count = ('count_chasing','sum'),
            chased_count = ('count_chased','sum')
        )
        .reset_index()
    )
    group_df['chasing_duration'] = group_df['frames_chasing'] / fps
    group_df['chased_duration'] = group_df['frames_chased'] / fps
    group_df['chasing_mean_duration'] = np.where(group_df['chasing_count'] > 0, group_df['chasing_duration'] / group_df['chasing_count'],0)
    group_df['chased_mean_duration'] = np.where(group_df['chased_count'] > 0, group_df['chased_duration'] / group_df['chased_count'],0)
    keep_cols = ['day','phase','box','mouse','time_bin','time_window','chasing_duration','chasing_count','chasing_mean_duration','chased_duration','chased_count','chased_mean_duration']
    return group_df[keep_cols]

def chase_normalize_by_nest_fine(df, nest_df, group_cols=['day','phase','box','mouse','time_bin','time_window']):
    nest_df = nest_df.rename(columns={'duration': 'nest_duration', 'count': 'nest_count', 'mean_duration': 'nest_mean_duration'})
    nest_cols = group_cols + ['nest_duration','outside_nest_duration','nest_count','nest_mean_duration']
    df = df.merge(nest_df[nest_cols],how='outer',on=group_cols)
    df['chasing_duration_fraction'] = df['chasing_duration'] / df['outside_nest_duration']
    df['chasing_event_rate'] = df['chasing_count'] / df['outside_nest_duration']
    df['chased_duration_fraction'] = df['chased_duration'] / df['outside_nest_duration']
    df['chased_event_rate'] = df['chased_count'] / df['outside_nest_duration']
    df['chasing_duration_ratio'] = df['chasing_duration'] / (df['chasing_duration']+df['chased_duration'])
    df['chased_duration_ratio'] = df['chased_duration'] / (df['chasing_duration']+df['chased_duration'])
    df['chasing_event_ratio'] = df['chasing_count'] / (df['chasing_count']+df['chased_count'])
    df['chased_event_ratio'] = df['chased_count'] / (df['chasing_count']+df['chased_count'])
    return df

cols_1h = ['day','ZT_hour','ZT_time','phase','box','mouse','frames_chasing','frames_chased','count_chasing','count_chased']
for b in batches:
    print(b)
    path_chase = os.path.join(main_path,f'{b}_2026','chase')
    for exp in exps[b]:
        print(exp)
        for t in time_points:
            print(t)
            df = pd.read_csv(os.path.join(path_chase,exp,t,'chase_events_by_mouse.csv'))
            df = utils.compute_event_times(df)
            # skipped the fine-interval separation since chasing events are usually short
            df = utils.add_fine_time_labels(df)
            df = utils.add_day_order(df)
            df = apply_day1_correction(df, b, exp, t)
            df_fine = df[cols_1h]
            for res in resolutions:
                df_res = chase_regroup_by_timebin_fine(df_fine, resolution=res)
                nest_df = pd.read_csv(os.path.join(main_path,f'{b}_2026','nest',exp,t,f'nest_{utils.format_resolution(res)}.csv'))
                df_res = chase_normalize_by_nest_fine(df_res, nest_df=nest_df)
                df_res.to_csv(os.path.join(path_chase,exp,t,f'chase_{utils.format_resolution(res)}.csv'),index=False)

#%%
# locomotion
def locomotion_regroup_by_timebin_fine(df_fine, resolution, group_base=['day','phase','box','mouse']):
    df = utils.add_timebin_labels_fine(df_fine.copy(), resolution)
    group_cols = group_base + ['time_bin','time_window']
    df['valid_locomotion'] = df['speed'].notna()
    df['abs_angular_velocity'] = df['angular_velocity'].abs()
    df['abs_acceleration'] = df['acceleration'].abs()
    group_df = (
        df.groupby(group_cols, observed=True)
        .agg(
            total_distance=('distance', 'sum'),
            valid_locomotion_duration=('valid_locomotion', 'sum'),
            mean_speed=('speed', 'mean'),
            median_speed=('speed', 'median'),
            mean_abs_angular_velocity=('abs_angular_velocity', 'mean'),
            mean_abs_acceleration=('abs_acceleration', 'mean')
        )
        .reset_index()
    )
    keep_cols = ['day','phase','box','mouse','time_bin','time_window','total_distance','valid_locomotion_duration','mean_speed','median_speed','mean_abs_angular_velocity','mean_abs_acceleration']
    return group_df[keep_cols]

def locomotion_normalize_by_nest_fine(df, nest_df, group_cols=['day','phase','box','mouse','time_bin','time_window']):
    nest_df = nest_df.rename(columns={'duration': 'nest_duration', 'count': 'nest_count', 'mean_duration': 'nest_mean_duration'})
    nest_cols = group_cols + ['nest_duration','outside_nest_duration','nest_count','nest_mean_duration']
    df = df.merge(nest_df[nest_cols],how='outer',on=group_cols)
    df['valid_locomotion_duration_fraction'] = df['valid_locomotion_duration'] / df['outside_nest_duration']
    return df

def compute_event_time_locomotion(df):
    df["recording_start"] = pd.to_datetime(
        df["date"].astype(str) + " " +
        df["timestamp"].astype(str).str.replace("-", ":")
    )
    df["event_start"] = (
        df["recording_start"] +
        pd.to_timedelta(df["time_in_seconds"], unit="s")
    )
    return df

cols_1h = ['day','ZT_hour','ZT_time','phase','box','mouse','distance','speed','angular_velocity','acceleration']
for b in batches:
    print(b)
    path = os.path.join(main_path,f'{b}_2026','locomotion')
    for exp in exps[b]:
        print(exp)
        for t in time_points:
            print(t)
            df = pd.read_csv(os.path.join(path,exp,t,'locomotion.csv'))
            df = compute_event_time_locomotion(df)
            # skipped the fine-interval separation - already 1-row-per-second
            df = utils.add_fine_time_labels(df)
            df = utils.add_day_order(df)
            df = apply_day1_correction(df, b, exp, t)
            df_fine = df[cols_1h]
            for res in resolutions:
                df_res = locomotion_regroup_by_timebin_fine(df_fine, resolution=res)
                nest_df = pd.read_csv(os.path.join(main_path,f'{b}_2026','nest',exp,t,f'nest_{utils.format_resolution(res)}.csv'))
                df_res = locomotion_normalize_by_nest_fine(df_res, nest_df=nest_df)
                df_res.to_csv(os.path.join(path,exp,t,f'locomotion_{utils.format_resolution(res)}.csv'),index=False)

#%%
# nest_social / ROI_social - deeper rewrite: occupancy is built directly at the finest
# (15-min) window instead of per whole ZT_hour, then re-binned into 15min/30min via the
# same floor-division formula the existing local regroup_timebin() functions use.
BASE_RESOLUTION = 0.25  # 15 min - finest grain the occupancy timelines are built at

def compute_sociability_weighted_fine(df_4h):
    df_4h = utils.compute_event_times(df_4h, fps=fps, start_col='start_frame', end_col='end_frame')
    df = utils.split_events_by_interval_with_frame(df_4h, interval_minutes=15, fps=fps)
    df = utils.add_fine_time_labels(df)
    df = utils.add_timebin_labels_fine(df, resolution=BASE_RESOLUTION)

    results = []
    group_cols = ["ZT_day", "box", "time_bin", "time_window"]

    for group_keys, group in df.groupby(group_cols):

        mice = sorted(group["mouse"].unique())
        if len(mice) < 4:
            print("WARNING:", group_keys, "mice:", mice)

        frame_min = int(group["start_frame"].min())
        frame_max = int(group["end_frame"].max())

        occupancy = {}
        for mouse in mice:
            bouts = group[group["mouse"] == mouse]
            occ = pd.Series(False, index=range(frame_min, frame_max+1))
            for _, row in bouts.iterrows():
                occ.loc[int(row.start_frame):int(row.end_frame)] = True
            occupancy[mouse] = occ

        for mouse in mice:
            focal = occupancy[mouse]
            nest_frames = int(focal.sum())

            if nest_frames == 0:
                results.append({
                    "ZT_day": group_keys[0], "box": group_keys[1],
                    "time_bin": group_keys[2], "time_window": group_keys[3],
                    "mouse": mouse,
                    "nest_frames": 0, "weighted_sum": 0.0, "alone_sum": 0
                })
                continue

            others = [occupancy[m] for m in mice if m != mouse]
            if others:
                others_sum = sum(others)
                coocc = others_sum[focal]
            else:
                coocc = pd.Series(0, index=focal.index)

            weighted_sum = (coocc / 3).sum()
            alone_sum = int((coocc == 0).sum())

            results.append({
                "ZT_day": group_keys[0], "box": group_keys[1],
                "time_bin": group_keys[2], "time_window": group_keys[3],
                "mouse": mouse,
                "nest_frames": nest_frames, "weighted_sum": weighted_sum, "alone_sum": alone_sum
            })

    out = pd.DataFrame(results)
    out["ZT_hour"] = out["time_bin"] - BASE_RESOLUTION  # left edge, for regroup_timebin below
    return out

def regroup_timebin_social(df, resolution):
    '''Same formula as nest_social.py/ROI_social.py's local regroup_timebin, unchanged -
    copied here so this script stays independent of those files. Operates on ZT_hour
    (the finest window's LEFT edge, set above) so it reproduces the base 15-min bins
    exactly at resolution=0.25 and correctly re-bins to 0.5 by summation.'''
    group_base = ["ZT_day", "box", "mouse"]
    df_res = df.copy()

    df_res['out_time_bin'] = (
        ((df_res['ZT_hour'] // resolution) + 1) * resolution
    )
    df_res['out_time_window'] = (
        (df_res['out_time_bin'] - resolution).round(2).astype(str)
        + "-"
        + (df_res['out_time_bin']).round(2).astype(str)
    )

    group_cols = group_base + ['out_time_bin','out_time_window']
    agg = (
        df_res.groupby(group_cols, as_index=False).agg({
            "nest_frames":"sum", "weighted_sum": "sum", "alone_sum": "sum"
        })
    )
    agg = agg.rename(columns={'out_time_bin':'time_bin','out_time_window':'time_window'})

    mask = agg["nest_frames"] > 0
    agg["weighted_co_occupancy"] = 0.0
    agg["alone_fraction"] = 0.0
    agg.loc[mask,"weighted_co_occupancy"] = agg.loc[mask, "weighted_sum"]/agg.loc[mask, "nest_frames"]
    agg.loc[mask,"alone_fraction"] = agg.loc[mask, "alone_sum"]/agg.loc[mask, "nest_frames"]

    agg = agg.sort_values(group_base + ['time_bin'])
    return agg

def add_labels_fine(df):
    df = df.copy()
    df["ZT_day"] = pd.to_datetime(df["ZT_day"])
    first_day = df["ZT_day"].min()
    df["day"] = ((df["ZT_day"] - first_day).dt.days
    + 1
    - (df["time_bin"] <= 12).astype(int)
    )
    df["phase"] = (df["time_bin"] > 12).map( {True: "active", False: "inactive"} )
    return df

def merge_nest_fine(df, nest_df, group_cols=['day','phase','box','mouse','time_bin','time_window']):
    nest_df = nest_df.rename(columns={'duration': 'nest_duration', 'count': 'nest_count', 'mean_duration': 'nest_mean_duration'})
    nest_cols = group_cols + ['nest_duration','outside_nest_duration','nest_count','nest_mean_duration']
    df = df.merge(nest_df[nest_cols],how='outer',on=group_cols)
    return df

for b in batches:
    print(b)
    nest_path = os.path.join(main_path,f'{b}_2026','nest')
    output_path = os.path.join(main_path,f'{b}_2026','nest_social')
    for exp in exps[b]:
        print(exp)
        for t in time_points:
            print(t)
            nest_df_4h = pd.read_csv(os.path.join(nest_path,exp,t,'nest_events.csv'))
            nest_social_base = compute_sociability_weighted_fine(nest_df_4h)
            os.makedirs(os.path.join(output_path,exp,t),exist_ok=True)
            for res in resolutions:
                df_res = regroup_timebin_social(nest_social_base,resolution=res)
                df_res = add_labels_fine(df_res)
                df_res = df_res.drop(columns=['ZT_day'])
                nest_res_df = pd.read_csv(os.path.join(nest_path,exp,t,f'nest_{utils.format_resolution(res)}.csv'))
                df_res = merge_nest_fine(df_res,nest_df=nest_res_df)
                df_res.to_csv(os.path.join(output_path,exp,t,f'nest_social_{utils.format_resolution(res)}.csv'),index=False)

#%%
def load_roi_group(folder, roi_files):
    dfs = []
    for roi_file in roi_files:
        path = os.path.join(folder,roi_file)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        dfs.append(df)
    if len(dfs) == 0:
        return None
    return pd.concat(dfs, ignore_index=True)

def compute_roi_sociability_fine(df_4h, roi_name):
    df_4h = utils.compute_event_times(df_4h, fps=fps, start_col='start_frame', end_col='end_frame')
    df = utils.split_events_by_interval_with_frame(df_4h, interval_minutes=15, fps=fps)
    df = utils.add_fine_time_labels(df)
    df = utils.add_timebin_labels_fine(df, resolution=BASE_RESOLUTION)

    results = []
    group_cols = ["ZT_day", "box", "time_bin", "time_window"]

    for group_keys, group in df.groupby(group_cols):
        if group.empty:
            frame_min,frame_max = 0,1
        else:
            frame_min = int(group["start_frame"].min())
            frame_max = int(group["end_frame"].max())
        mice = sorted(group["mouse"].dropna().unique())

        occupancy = {}
        for mouse in mice:
            bouts = group[group["mouse"] == mouse]
            occ = pd.Series(False, index=range(frame_min, frame_max + 1))
            for _, row in bouts.iterrows():
                occ.loc[int(row["start_frame"]):int(row["end_frame"])] = True
            occupancy[mouse] = occ

        for mouse in mice:
            focal = occupancy[mouse]
            roi_frames = int(focal.sum())

            result = {
                "ZT_day": group_keys[0], "box": group_keys[1],
                "time_bin": group_keys[2], "time_window": group_keys[3],
                "mouse": mouse,
                f"{roi_name}_frames": roi_frames,
                f"{roi_name}_together_frames": 0,
                f"{roi_name}_alone_frames": 0
            }

            if roi_frames == 0:
                results.append(result)
                continue

            others = [occupancy[m] for m in mice if m != mouse]
            if others:
                others_sum = sum(others)
                coocc = others_sum[focal]
                together_frames = int((coocc >= 1).sum())
                alone_frames = int((coocc == 0).sum())
            else:
                together_frames = 0
                alone_frames = roi_frames

            result.update({
                f"{roi_name}_together_frames": together_frames,
                f"{roi_name}_alone_frames": alone_frames
            })
            results.append(result)

    out = pd.DataFrame(results)
    out["ZT_hour"] = out["time_bin"] - BASE_RESOLUTION
    return out

def regroup_timebin_roi_social(df, resolution):
    group_base = ["ZT_day", "box", "mouse"]
    df_res = df.copy()

    df_res['out_time_bin'] = (((df_res['ZT_hour'] // resolution) + 1) * resolution)
    df_res['out_time_window'] = (
        (df_res['out_time_bin'] - resolution).round(2).astype(str)
        + "-" + (df_res['out_time_bin']).round(2).astype(str)
    )

    frame_cols = [c for c in df_res.columns if c.endswith("_frames") and not c.endswith("_together_frames") and not c.endswith("_alone_frames")]
    together_cols = [c for c in df_res.columns if c.endswith("_together_frames")]
    alone_cols = [c for c in df_res.columns if c.endswith("_alone_frames")]
    agg_dict = {c: "sum" for c in frame_cols + together_cols + alone_cols}

    group_cols = group_base + ['out_time_bin','out_time_window']
    agg = df_res.groupby(group_cols, as_index=False).agg(agg_dict)
    agg = agg.rename(columns={'out_time_bin':'time_bin','out_time_window':'time_window'})

    for roi_name in [c.replace("_frames","") for c in frame_cols]:
        frame_col, together_col, alone_col = f'{roi_name}_frames', f'{roi_name}_together_frames', f'{roi_name}_alone_frames'
        together_fraction_col, alone_fraction_col = f'{roi_name}_together_fraction', f'{roi_name}_alone_fraction'
        agg[together_fraction_col] = 0.0
        agg[alone_fraction_col] = 0.0
        mask = agg[frame_col] > 0
        agg.loc[mask, together_fraction_col] = agg.loc[mask, together_col] / agg.loc[mask, frame_col]
        agg.loc[mask, alone_fraction_col] = agg.loc[mask, alone_col] / agg.loc[mask, frame_col]

    agg = agg.sort_values(group_base + ["time_bin"])
    return agg

ROI_GROUPS = {
    "feeding": ["feeder_prox_events.csv", "feeder_dist_events.csv"],
    "drinking": ["water_prox_events.csv", "water_dist_events.csv"],
    "ramps": ["ramp1_events.csv", "ramp2_events.csv"],
    "s_wall": ["s_wall_events.csv"]
}

for b in batches:
    print(b)
    ROI_path = os.path.join(main_path,f'{b}_2026','ROIs')
    nest_path = os.path.join(main_path,f'{b}_2026','nest')
    output_path = os.path.join(main_path,f'{b}_2026','ROI_social')
    for exp in exps[b]:
        print(exp)
        for t in time_points:
            print(t)
            roi_folder = os.path.join(ROI_path,exp,t)
            output_folder = os.path.join(output_path,exp,t)
            os.makedirs(output_folder,exist_ok=True)
            roi_results = []
            for roi_name, roi_filenames in ROI_GROUPS.items():
                df_roi = load_roi_group(roi_folder,roi_filenames)
                if df_roi is None:
                    continue
                social = compute_roi_sociability_fine(df_4h=df_roi,roi_name=roi_name)
                roi_results.append(social)
            if len(roi_results) == 0:
                print("No ROI data found")
                continue

            df_social = roi_results[0]
            for df in roi_results[1:]:
                df_social = pd.merge(df_social,df,on=["ZT_day","box","time_bin","time_window","mouse","ZT_hour"],how="outer")

            for res in resolutions:
                df_res = regroup_timebin_roi_social(df_social,resolution=res)
                df_res = add_labels_fine(df_res)
                df_res = df_res.drop(columns=['ZT_day'])
                nest_res_df = pd.read_csv(os.path.join(nest_path,exp,t,f'nest_{utils.format_resolution(res)}.csv'))
                df_res = merge_nest_fine(df_res,nest_df=nest_res_df)
                df_res.to_csv(os.path.join(output_folder,f"ROI_social_{utils.format_resolution(res)}.csv"),index=False)

# %%
