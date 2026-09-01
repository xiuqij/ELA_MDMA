#%%
import pandas as pd
import os
import time
import utils_stats as utils

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
        print(f"No ROI files found in {folder} for {roi_files}.")
        return None
    return pd.concat(dfs, ignore_index=True)

def compute_roi_sociability(df_4h, roi_name):
    """
    Calculate ROI co-occupancy and alone time per mouse per hour.

    Input:
        df = ROI events already split into hourly segments and labelled
             with date, box, ZT_hour, video, mouse, start_frame, end_frame

    Output:
        One row per date × box × ZT_hour × video × mouse.
    """
    # pre processing
    df_4h  = utils.compute_event_times(df_4h, fps=25,start_col='start_frame',end_col='end_frame')
    df = utils.split_events_by_hour_with_frame(df_4h,fps=25)
    df = utils.add_time_labels(df)

    results = []
    group_cols = ["ZT_day", "box", "ZT_hour"]

    for group_keys, group in df.groupby(group_cols):
        if group.empty:
            frame_min,frame_max = 0,1
        else:
            frame_min = int(group["start_frame"].min())
            frame_max = int(group["end_frame"].max())
        mice = sorted(group["mouse"].dropna().unique())

        # Build occupancy timelines
        occupancy = {}
        for mouse in mice:
            bouts = group[group["mouse"] == mouse]
            # +1 because end_frame should be included
            occ = pd.Series(
                False,
                index=range(frame_min, frame_max + 1)
            )
            for _, row in bouts.iterrows():
                start = int(row["start_frame"])
                end = int(row["end_frame"])

                occ.loc[start:end] = True

            occupancy[mouse] = occ


        # Calculate metrics per mouse
        for mouse in mice:
            focal = occupancy[mouse]
            roi_frames = int(focal.sum())

            result = {
                "ZT_day": group_keys[0],
                "box": group_keys[1],
                "ZT_hour": group_keys[2],
                "mouse": mouse,

                f"{roi_name}_frames": roi_frames,
                f"{roi_name}_together_frames": 0,
                f"{roi_name}_alone_frames": 0
            }

            if roi_frames == 0:
                results.append(result)
                continue

            # Other mice occupancy
            others = [
                occupancy[m]
                for m in mice
                if m != mouse
            ]

            if others:
                others_sum = sum(others)
                # Number of other mice present at each focal frame
                coocc = others_sum[focal]

                together_frames = int(
                    (coocc >= 1).sum()
                )

                alone_frames = int(
                    (coocc == 0).sum()
                )

            else:
                # No other mice available
                together_frames = 0
                alone_frames = roi_frames

            result.update({
                f"{roi_name}_together_frames": together_frames,
                f"{roi_name}_alone_frames": alone_frames
            })

            results.append(result)

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

    # Identify ROI metric columns
    frame_cols = [
        col for col in df_res.columns 
        if col.endswith("_frames")
        and not col.endswith("_together_frames")
        and not col.endswith("_alone_frames")
        ]

    together_cols = [
        col for col in df_res.columns
        if col.endswith("_together_frames")
        ]

    alone_cols = [
        col for col in df_res.columns
        if col.endswith("_alone_frames")
        ]

    agg_dict = {}

    for col in frame_cols:
        agg_dict[col] = "sum"

    for col in together_cols:
        agg_dict[col] = "sum"

    for col in alone_cols:
        agg_dict[col] = "sum"

    # Aggregate raw values
    group_cols = group_base + ['time_bin','time_window']

    agg = (
        df_res.groupby(group_cols, as_index= False).agg(agg_dict)
    )

    # calculate fractions
    for roi_name in [
        col.replace("_frames","")
        for col in frame_cols
    ]:
        frame_col = f'{roi_name}_frames'
        together_col = f'{roi_name}_together_frames'
        alone_col = f'{roi_name}_alone_frames'

        together_fraction_col = (
            f'{roi_name}_together_fraction'
        )
        alone_fraction_col = (
            f'{roi_name}_alone_fraction'
        )

        agg[together_fraction_col] = 0.0
        agg[alone_fraction_col] = 0.0

        mask = agg[frame_col] > 0

        agg.loc[mask, together_fraction_col] = (
            agg.loc[mask, together_col] / agg.loc[mask, frame_col]
            )

        agg.loc[mask, alone_fraction_col] = (
            agg.loc[mask, alone_col] / agg.loc[mask, frame_col]
            )

    # sort
    agg = agg.sort_values(
        group_base + ["time_bin"]
    )
    return agg

# Paths
ROI_path = r"L:\Lopez Laboratory - NEURO\Xiuqi\ELA_MDMA\July_2026\ROIs" 
output_path = r"L:\Lopez Laboratory - NEURO\Xiuqi\ELA_MDMA\July_2026\ROI_social"

exps = ['female_P35','male_P42']
time_points =['baseline','MDMA']

ROI_GROUPS = {
    "feeding": ["feeder_prox_events.csv", "feeder_dist_events.csv"],
    "drinking": ["water_prox_events.csv", "water_dist_events.csv"],
    "ramps": ["ramp1_events.csv", "ramp2_events.csv"],
    "s_wall": ["s_wall_events.csv"]
}

for exp in exps:
    print(exp)
    for t in time_points:
        print(t)
        # I/O paths
        roi_folder = os.path.join(ROI_path,exp,t)
        output_folder = os.path.join(output_path,exp,t)
        os.makedirs(output_folder,exist_ok=True)
        roi_results = []

        # process each ROI
        for roi_name, roi_filenames in ROI_GROUPS.items():
            print(f'ROI: {roi_name}')
            df_roi = load_roi_group(roi_folder,roi_filenames)
            if df_roi is None:
                continue
            social = compute_roi_sociability(df_4h=df_roi,roi_name=roi_name)
            roi_results.append(social)
        if len(roi_results) == 0:
            print("No ROI data found")
            continue

        # merge ROIs
        df_social = roi_results[0]
        for df in roi_results[1:]:
            df_social = pd.merge(
                df_social,df,
                on=["ZT_day","box","ZT_hour","mouse"],
                how="outer"
                )
        # save raw data
        output_file = os.path.join(output_folder,"ROI_social.csv")
        df_social.to_csv(output_file,index=False)
        print(f"Saved raw output to {output_file}.")

        for res in [1,2,3,4,6,12]:
            df_res = regroup_timebin(df_social,resolution=res)
            df_res.to_csv(os.path.join(output_folder,f"ROI_social_{res}h.csv"),index=False)
            print(f"Saved {res}h data.")
