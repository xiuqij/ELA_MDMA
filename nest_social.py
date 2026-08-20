#%%
import pandas as pd
import os
import time
import utils_stats as utils

#%% helper functions (modified from serenas scripts)
# info cols: video, SB, mouse, time_point, date, ZT_hour
# merge other conditions (background, treatment) later. 

def compute_sociability_weighted(df_4h):
    '''use nest_events_4h as input
    '''
    
    df_4h  = utils.compute_event_times(df_4h, fps=25,start_col='start_frame',end_col='end_frame')
    df = utils.split_events_by_hour_with_frame(df_4h,fps=25)
    df = utils.add_time_labels(df)
    #cols (nest_events_1h):'video','mouse','box','duration_f','event_start', 'event_end', 'date','CT_hour','ZT_day','ZT_hour','phase','start_frame','end_frame'
    
    results = []
    group_cols = ["ZT_day", "box", "ZT_hour"]

    for group_keys, group in df.groupby(group_cols):

        mice = sorted(group["mouse"].unique())
        if len(mice) < 4:
            print("WARNING:", group_keys, "mice:", mice)

        frame_min = int(group["start_frame"].min())
        frame_max = int(group["end_frame"].max())

        occupancy = {}

        # --------------------------------------------------
        # Build occupancy timelines
        # --------------------------------------------------
        for mouse in mice:

            bouts = group[group["mouse"] == mouse]

            occ = pd.Series(False, index=range(frame_min, frame_max+1)) #note the change here

            for _, row in bouts.iterrows():
                occ.loc[int(row.start_frame):int(row.end_frame)] = True

            occupancy[mouse] = occ

        # --------------------------------------------------
        # Compute metrics per mouse
        # --------------------------------------------------
        for mouse in mice:

            focal = occupancy[mouse]
            nest_frames = int(focal.sum())

            if nest_frames == 0:

                results.append({
                    "ZT_day": group_keys[0],
                    "box": group_keys[1],
                    "ZT_hour": group_keys[2],
                    "mouse": mouse,

                    # SAFE DEFAULTS
                    "nest_frames": 0,
                    "weighted_sum": 0.0,
                    "alone_sum": 0
                })

                continue

            # ------------------------------------------
            # Other mice occupancy
            # ------------------------------------------
            others = [occupancy[m] for m in mice if m != mouse]
            if others:
                others_sum = sum(others)
                coocc = others_sum[focal]
            else:
                # no other mice available in this group
                coocc = pd.Series(0, index=focal.index)
            # ------------------------------------------
            # RAW (IMPORTANT CHANGE)
            # ------------------------------------------

            weighted_sum = (coocc / 3).sum()
            alone_sum = int((coocc == 0).sum())

            results.append({
                "ZT_day": group_keys[0],
                "box": group_keys[1],
                "ZT_hour": group_keys[2],
                "mouse": mouse,

                # denominators + numerators. Needed this way to then aggregate correctly by timebin
                "nest_frames": nest_frames,
                "weighted_sum": weighted_sum,
                "alone_sum": alone_sum
            })

    return pd.DataFrame(results)

def regroup_timebin(df,resolution):
    '''re-group the output into different resolution (1,2,3,4,6,12-hour timebins).'''
    group_base = ["ZT_day", "box", "mouse"]
    df_res = df.copy()

    # create numeric timebin for sorting; e.g., 6,12,18,24 for 6h resolution
    df_res['time_bin'] = (
        ((df_res['ZT_hour'] // resolution) + 1) * resolution
    )

    # create readable time window for plotting
    df_res['time_window'] = (
        (df_res['time_bin'] - resolution).astype(int).astype(str)
        + "-"
        + (df_res['time_bin']).astype(int).astype(str)
    )

    # aggregate raw numerators and denominators
    group_cols = group_base + ['time_bin','time_window']

    agg = (
        df_res.groupby(group_cols, as_index= False).agg({
            "nest_frames":"sum",
            "weighted_sum": "sum",
            "alone_sum": "sum"
        })
    )

    # calculate final metrics
    mask = agg["nest_frames"] > 0
    agg["weighted_co_occupancy"] = 0.0
    agg["alone_fraction"] = 0.0

    agg.loc[mask,"weighted_co_occupancy"] = (
        agg.loc[mask, "weighted_sum"]/agg.loc[mask, "nest_frames"]
    )

    agg.loc[mask,"alone_fraction"] = (
        agg.loc[mask, "alone_sum"]/agg.loc[mask, "nest_frames"]
    )

    # sort
    agg = agg.sort_values(
        group_base + ['time_bin']
    )

    return agg
# paths
nest_path = r'L:\Lopez Laboratory - NEURO\Xiuqi\ELA_MDMA\July_2026\nest'
output_path = r'L:\Lopez Laboratory - NEURO\Xiuqi\ELA_MDMA\July_2026\nest_social'
exps = ['female_P35','male_P42']
time_points = ['baseline','MDMA']

for exp in exps:
    print(exp)
    for t in time_points:
        print(t)
        nest_csv_path = os.path.join(nest_path,exp,t,'nest_events.csv') #4h result
        nest_df_4h = pd.read_csv(nest_csv_path)
        # 1-hour data
        nest_social_df = compute_sociability_weighted(nest_df_4h)
        print("saving...")
        os.makedirs(os.path.join(output_path,exp,t),exist_ok= True)
        output_csv_path = os.path.join(output_path, exp, t, 'nest_social.csv')
        nest_social_df.to_csv(output_csv_path, index=False)
        print("saved raw data.")

        # rebin into different resolutions
        resolutions = [1,2,3,4,6,12]
        for res in resolutions:
            df_res = regroup_timebin(nest_social_df,resolution=res)
            df_res.to_csv(os.path.join(output_path,exp,t,f'nest_social_{res}h.csv'),index=False)
            print(f"saved {res}h data.")
