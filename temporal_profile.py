#%%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%
def plot_baseline_progression(
    df,
    feature,
    group_col="background",
    active_start=12,
    active_end=24,
):
    """
    Plot the progression of a feature during the active phase
    for the first 3 baseline days.

    Parameters
    ----------
    df : DataFrame
        2-hour summary dataframe.
    feature : str
        Column to plot.
    group_col : str
        Column defining groups (e.g. 'background' or 'condition').
    """

    df = df.copy()

    # sort dates
    baseline_days = (
        np.sort(df["ZT_day"].unique())[:3]
    )

    df = df[df["ZT_day"].isin(baseline_days)]

    # keep active phase only
    df = df[
        (df["time_bin"] >= active_start) &
        (df["time_bin"] < active_end)
    ]

    # assign Day 1-3
    day_map = {
        d: f"Day {i+1}"
        for i, d in enumerate(baseline_days)
    }
    df["baseline_day"] = df["ZT_day"].map(day_map)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15,4),
        sharey=True
    )

    for ax, day in zip(axes, ["Day 1","Day 2","Day 3"]):

        day_df = df[df["baseline_day"] == day]

        summary = (
            day_df
            .groupby([group_col, "time_bin"])[feature]
            .agg(["mean","sem"])
            .reset_index()
        )

        for group in summary[group_col].unique():

            tmp = summary[summary[group_col] == group]

            ax.errorbar(
                tmp["time_bin"],
                tmp["mean"],
                yerr=tmp["sem"],
                marker="o",
                capsize=3,
                label=group,
            )

        ax.set_title(day)
        ax.set_xlabel("ZT hour")
        ax.set_xticks([12,14,16,18,20,22])

    axes[0].set_ylabel(feature)
    axes[-1].legend(title=group_col)

    plt.tight_layout()
    plt.show()


#%%
exp = 'male_P35'
df = pd.read_csv(f"/Volumes/labs/Lopez Laboratory - NEURO/Xiuqi/ELA_MDMA/April_2026/rois_{exp}_4h.csv")

#%%
plot_baseline_progression(df, feature = 'total_time_outside_s')
# %%
cols = [
       's_wall_event_count', 's_wall_total_time_s', 's_wall_avg_time_s',
       'total_time_outside_s', 'nest_entries', 'nest_avg_time_s',
       's_wall_norm_total_time_s', 's_wall_norm_event_count',
       'ramp1_event_count', 'ramp1_total_time_s', 'ramp1_avg_time_s',
       'ramp1_norm_total_time_s', 'ramp1_norm_event_count',
       'ramp2_event_count', 'ramp2_total_time_s', 'ramp2_avg_time_s',
       'ramp2_norm_total_time_s', 'ramp2_norm_event_count',
       'non_wall_event_count', 'non_wall_total_time_s', 'non_wall_avg_time_s',
       'non_wall_norm_total_time_s', 'non_wall_norm_event_count',
       'woodstick_event_count', 'woodstick_total_time_s',
       'woodstick_avg_time_s', 'woodstick_norm_total_time_s',
       'woodstick_norm_event_count', 'feeder_prox_event_count',
       'feeder_prox_total_time_s', 'feeder_prox_avg_time_s',
       'feeder_prox_norm_total_time_s', 'feeder_prox_norm_event_count',
       'feeder_dist_event_count', 'feeder_dist_total_time_s',
       'feeder_dist_avg_time_s', 'feeder_dist_norm_total_time_s',
       'feeder_dist_norm_event_count', 'water_prox_event_count',
       'water_prox_total_time_s', 'water_prox_avg_time_s',
       'water_prox_norm_total_time_s', 'water_prox_norm_event_count',
       'water_dist_event_count', 'water_dist_total_time_s',
       'water_dist_avg_time_s', 'water_dist_norm_total_time_s',
       'water_dist_norm_event_count']
for col in cols:
    plot_baseline_progression(df, feature = col)
# %%

