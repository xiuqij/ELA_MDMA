#%%
import os
import pandas as pd
import numpy as np

#%%
# cols in hour df:mouse_color,SB,ZT_day,s_wall_event_count,s_wall_total_time_s,s_wall_avg_time_s,total_time_outside_s,nest_entries,nest_avg_time_s,s_wall_norm_total_time_s,s_wall_norm_event_count,time_point,ramp1_event_count,ramp1_total_time_s,ramp1_avg_time_s,ramp1_norm_total_time_s,ramp1_norm_event_count,ramp2_event_count,ramp2_total_time_s,ramp2_avg_time_s,ramp2_norm_total_time_s,ramp2_norm_event_count,non_wall_event_count,non_wall_total_time_s,non_wall_avg_time_s,non_wall_norm_total_time_s,non_wall_norm_event_count,woodstick_event_count,woodstick_total_time_s,woodstick_avg_time_s,woodstick_norm_total_time_s,woodstick_norm_event_count,feeder_prox_event_count,feeder_prox_total_time_s,feeder_prox_avg_time_s,feeder_prox_norm_total_time_s,feeder_prox_norm_event_count,feeder_dist_event_count,feeder_dist_total_time_s,feeder_dist_avg_time_s,feeder_dist_norm_total_time_s,feeder_dist_norm_event_count,water_prox_event_count,water_prox_total_time_s,water_prox_avg_time_s,water_prox_norm_total_time_s,water_prox_norm_event_count,water_dist_event_count,water_dist_total_time_s,water_dist_avg_time_s,water_dist_norm_total_time_s,water_dist_norm_event_count,mouse_ID,background,treatment,batch

# break ZT hour windows

#%%
def aggregate_time_window(df, window_size):
    """
    Aggregate an hour-level dataframe into larger time windows.

    Parameters
    ----------
    df : pd.DataFrame
        Hour-level dataframe.
    window_size : int
        Size of time window in hours (e.g. 2, 4, 6).

    Returns
    -------
    pd.DataFrame
    """

    df = df.copy()

    # create bin index
    df["time_bin"] = (df["ZT_hour"] // window_size) * window_size

    # columns identifying each mouse/day
    group_cols = [
        "mouse_color",
        "SB",
        "ZT_day",
        "phase",
        "mouse_ID",
        "background",
        "treatment",
        "batch",
        "time_bin"
    ]

    # numeric columns to aggregate
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    # remove grouping columns
    numeric_cols = [c for c in numeric_cols if c not in group_cols]

    agg_dict = {}

    for col in numeric_cols:

        # average variables
        if "avg" in col.lower():
            agg_dict[col] = "mean"

        # counts and durations
        elif (
            "count" in col.lower()
            or "time_s" in col.lower()
            or "entries" in col.lower()
        ):
            agg_dict[col] = "sum"

        else:
            agg_dict[col] = "mean"

    out = (
        df.groupby(group_cols)
          .agg(agg_dict)
          .reset_index()
    )

    # useful label
    out["time_window"] = (
        out["time_bin"].astype(str)
        + "-"
        + (out["time_bin"] + window_size - 1).astype(str)
    )

    return out

#%%
hour_df = pd.read_csv("/Volumes/labs/Lopez Laboratory - NEURO/Xiuqi/ELA_MDMA/April_2026/rois_female_P42_hour.csv")

df_2h = aggregate_time_window(hour_df, 2)
df_4h = aggregate_time_window(hour_df, 4)
df_6h = aggregate_time_window(hour_df, 6)
 
#%%
df_2h.to_csv("/Volumes/labs/Lopez Laboratory - NEURO/Xiuqi/ELA_MDMA/April_2026/rois_female_P42_2h.csv",index=False)
df_4h.to_csv("/Volumes/labs/Lopez Laboratory - NEURO/Xiuqi/ELA_MDMA/April_2026/rois_female_P42_4h.csv",index=False)
df_6h.to_csv("/Volumes/labs/Lopez Laboratory - NEURO/Xiuqi/ELA_MDMA/April_2026/rois_female_P42_6h.csv",index=False)