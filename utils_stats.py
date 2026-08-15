import pandas as pd

def compute_event_times(df, fps=25,start_col = 'start', end_col = 'end'): 
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

def convert_to_1h (df_4h,fps=25,start_col = 'start_frame', end_col = 'end_frame'):
    cols = ['video','mouse','box','duration_f','event_start', 'event_end', 'date','CT_hour','ZT_day','ZT_hour','phase']
    df_4h = compute_event_times(df_4h, fps=fps,start_col=start_col,end_col=end_col)
    df_1h = split_events_by_hour(df_4h, fps=fps)
    df_1h = add_time_labels(df_1h)
    return df_1h[cols]

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