import pandas as pd
import numpy as np

def compute_event_times(df, fps=25,start_col = 'start_frame', end_col = 'end_frame'): 
    '''compute the actual event start and end in date time format '''
    # recording start time 
    df["recording_start"] = pd.to_datetime(
        df["date"].astype(str) + " " + df["timestamp"].str.replace("-", ":")
        ) 
    
    # event timestamps 
    df["event_start"] = df["recording_start"] + pd.to_timedelta(
        df[start_col] / fps, unit="s"
        ) 
    
    df["event_end"] = df["recording_start"] + pd.to_timedelta(
        df[end_col] / fps, unit="s"
        ) 
    
    return df 
def split_events_by_hour(df, fps=25):
    """
    Only necessary for behaviors that can have long duration, e.g., nest and ROIs
    Split events according to REAL clock-hour boundaries.

    Example:
    18:59:58 → 19:00:05
    becomes:
        18:59:58 → 19:00:00
        19:00:00 → 19:00:05
    """

    rows = []

    for r in df.itertuples():

        current_start = r.event_start
        final_end = r.event_end

        while current_start < final_end:

            # next real clock hour
            next_hour = (
                current_start
                .replace(minute=0, second=0, microsecond=0)
                + pd.Timedelta(hours=1)
            )

            segment_end = min(final_end, next_hour)

            duration_s = (segment_end - current_start).total_seconds()
            duration_f = duration_s * fps

            rows.append({
                "mouse": r.mouse,
                "video": r.video,
                "date": r.date,
                "box": r.box,

                "event_start": current_start,
                "event_end": segment_end,

                "duration_f": duration_f
            })

            current_start = segment_end

    return pd.DataFrame(rows)

def split_events_by_hour_with_frame(df,fps=25):
    
    rows = []

    for r in df.itertuples():

        current_start = r.event_start
        final_end = r.event_end

        # Original frame boundaries
        current_start_frame = r.start_frame
        final_end_frame = r.end_frame

        while current_start < final_end:

            next_hour = (
                current_start.replace(
                    minute=0,
                    second=0,
                    microsecond=0
                )
                + pd.Timedelta(hours=1)
            )

            segment_end = min(final_end, next_hour)

            # Number of frames from current timestamp to segment end
            frames_to_boundary = round(
                (segment_end - current_start).total_seconds() * fps
            )

            segment_start_frame = current_start_frame
            segment_end_frame = min(
                current_start_frame + frames_to_boundary,
                final_end_frame
            )

            rows.append({
                "mouse": r.mouse,
                "video": r.video,
                "date": r.date,
                "box": r.box,

                "event_start": current_start,
                "event_end": segment_end,

                "start_frame": segment_start_frame,
                "end_frame": segment_end_frame,

                "duration_f": segment_end_frame - segment_start_frame
            })

            # Move forward
            current_start_frame = segment_end_frame
            current_start = segment_end

    return pd.DataFrame(rows)

def add_time_labels(df): 
    '''ZT0=lights on, ZT12=lights off'''
    #df = df.copy() 
    df["CT_hour"] = df["event_start"].dt.hour 
    
    # active vs inactive 
    df["phase"] = ((df["CT_hour"] >= 19) | (df["CT_hour"] < 7)).map( {True: "active", False: "inactive"} ) 
    
    # day starting at 07:00 
    df["ZT_day"] = (df["event_start"] - pd.Timedelta(hours=7)).dt.date 
    
    # ZT hour (useful for plotting) 
    df["ZT_hour"] = (df["CT_hour"] - 7) % 24 

    return df

def add_day_order(df):
    '''for df of individual experiments
    e.g., date: 2026-04-04 -> day: 1'''
    first_day = df["date"].min()
    first_day = pd.to_datetime(first_day)
    # hard code start time
    day1_start = first_day.normalize() + pd.Timedelta(hours=19)
    # assign day
    df["day"] = (
    (df["event_start"] - day1_start).dt.total_seconds() // (24 * 60 * 60)
    ).astype(int) + 1

    return df

def convert_to_1h (df_4h,fps=25,start_col = 'start_frame', end_col = 'end_frame',keep_cols = ['video','mouse','box','duration_f','event_start', 'event_end', 'date','day','CT_hour','ZT_day','ZT_hour','phase']):
    df_4h = compute_event_times(df_4h, fps=fps,start_col=start_col,end_col=end_col)
    df_1h = split_events_by_hour_with_frame(df_4h, fps=fps)
    df_1h = add_time_labels(df_1h)
    df_1h = add_day_order(df_1h)
    return df_1h[keep_cols]

def add_timebin_labels(df, resolution):
    '''
    resolution 1,2,3,4,6,12 (hr)
    time_bin: numeric timebin (=the last hour of the window) for sorting
    time_window: readable time windows (start-end) for plotting
    '''
    df['time_bin'] = (
        ((df['ZT_hour'] // resolution) + 1) * resolution
    )

    df['time_window'] = (
        (df['time_bin'] - resolution).astype(int).astype(str)
        + '-'
        + (df['time_bin']).astype(int).astype(str)
    )
    return df

def regroup_by_timebin(df_1h,resolution,fps=25,group_base = ['mouse','box'],nest=False):
    '''regroup and compute sums and event counts. 
    '''

    df = df_1h.copy()

    df = add_timebin_labels(df,resolution)
    group_cols = group_base + ['time_bin','time_window']
    group_df = (
        df.groupby(group_cols,observed=True)['duration_f']
        .agg(
            duration_f = 'sum',
            count = 'size'
        )
        .reset_index()
    )
    group_df['duration'] = group_df['duration_f'] / fps
    group_df['mean_duration'] = np.where(group_df['count'] > 0, group_df['duration'] / group_df['count'],0)
    
    if nest:
        unit_total_s = resolution * 60 *60
        group_df['outside_nest_duration'] = unit_total_s - group_df['duration']

    return group_df

def normalize_by_nest(df,nest_df,group_cols=['day','phase','box','mouse','time_bin','time_window']):
    # duration, count, mean_duation, outside_nest_duration, duration_fraction, event_rate
    nest_df = nest_df.rename(columns = {'total_time':'nest_duration','count':'nest_count','avg_time':'nest_mean_duration','outside_total_time':'outside_nest_duration'})
    #nest_df = nest_df.rename(columns = {'duration':'nest_duration','count':'nest_count','mean_duration':'nest_mean_duration'})
    nest_cols = group_cols + ['nest_duration','outside_nest_duration','nest_count','nest_mean_duration']
    df = df.merge(nest_df[nest_cols],how='outer',on=group_cols)
    df['duration_fraction'] = df['duration'] / df['outside_nest_duration']
    df['event_rate'] = df['count'] / df['outside_nest_duration']
    return df

def get_summary_stats(raw_df, 
                      fps = 25, 
                      group_by = 'hour',
                      nest=False,nest_df= None, 
                      base_cols = ['mouse','box']):
    '''Group by mouse per hour/phase/day, and calculate the sum of duration and events.
     For nest, the total time outside is calculated; 
      for other behaviors, normalized time and event count are calculated if the correct nest_df is provided. '''
    if group_by not in ['hour','phase','day']:
        raise ValueError(f"'group_by' must be 'hour', 'phase' or 'day', but got {group_by}")
    
    if group_by=='hour':
        groupby_columns = base_cols + ['ZT_day','phase','ZT_hour','CT_hour']
        unit_total_s = 60 * 60
    elif group_by=='phase':
        groupby_columns = base_cols + ['ZT_day','phase']
        unit_total_s = 60 * 60 * 12
    elif group_by=='day':
        groupby_columns = base_cols + ['ZT_day']
        unit_total_s = 60 * 60 * 24
    
    #raw_df['total_time_s'] = raw_df['duration_f'] / fps
    stats_df = raw_df.groupby(groupby_columns)['duration_f'].sum().reset_index()
    stats_df['event_count'] = raw_df.groupby(groupby_columns).size().reset_index(name = 'event_count')['event_count']
    stats_df['total_time_s'] = stats_df['duration_f'] / fps
    stats_df['avg_time_s'] = stats_df['total_time_s'] / stats_df['event_count']
    if nest == True:
        stats_df['total_time_outside_s'] = unit_total_s - stats_df['total_time_s']
        stats_df['nest_entries'] = stats_df['event_count']
        stats_df['nest_avg_time_s'] = stats_df['avg_time_s']
    if (nest == False) and (nest_df is not None):
        nest_cols = groupby_columns.copy()
        nest_cols.extend(['total_time_outside_s','nest_entries','nest_avg_time_s'])
        stats_df['ZT_day'] = pd.to_datetime(stats_df['ZT_day']).dt.date
        nest_df['ZT_day'] = pd.to_datetime(nest_df['ZT_day']).dt.date

        stats_df = stats_df.merge(nest_df[nest_cols],how='outer',on=groupby_columns)
        stats_df['norm_total_time_s'] = stats_df['total_time_s']/stats_df['total_time_outside_s']
        stats_df['norm_event_count'] = stats_df['event_count']/stats_df['total_time_outside_s']
    stats_df = stats_df.drop(columns = 'duration_f')
    return stats_df