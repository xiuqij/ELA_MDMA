#%%
import os
import pandas as pd

#%%
feature_metadata = {

    # =========================================================
    # NEST
    # =========================================================

    'nest_duration': {
        'family': 'nest',
        'type': 'continuous',
        'units': 's',
        'denominator': 'time_window',
        'model': 'LME',
        'transform': 'none',
        'notes': 'Total time spent in nest during the time window'
    },

    'outside_nest_duration': {
        'family': 'nest',
        'type': 'continuous',
        'units': 's',
        'denominator': 'time_window',
        'model': 'LME',
        'transform': 'none',
        'notes': 'Total time spent outside nest during the time window; important exposure variable for ROI/event features'
    },

    'nest_count': {
        'family': 'nest',
        'type': 'count',
        'units': 'events',
        'denominator': 'time_window',
        'model': 'negative_binomial',
        'transform': 'none',
        'notes': 'Number of nest-entry/occupancy events'
    },

    'nest_mean_duration': {
        'family': 'nest',
        'type': 'positive_continuous',
        'units': 's/event',
        'denominator': 'nest_count',
        'model': 'gamma_or_log_LME',
        'transform': 'log_if_needed',
        'notes': 'Mean duration per nest event'
    }
}


    # =========================================================
    # ROI
    # =========================================================

    # Each ROI has the same structure:
    #
    # count
    # duration
    # mean_duration
    # duration_fraction
    # event_rate
    #
    # duration_fraction = duration / outside_nest_duration
    # event_rate        = count / outside_nest_duration

roi_behaviors = [
    's_wall',
    'ramp1',
    'ramp2',
    'non_wall',
    'woodstick',
    'feeder_prox',
    'feeder_dist',
    'water_prox',
    'water_dist'
]

roi_metadata = {}

for behavior in roi_behaviors:

    roi_metadata[f'{behavior}_count'] = {
        'family': 'roi',
        'type': 'count',
        'units': 'events',
        'denominator': 'outside_nest_duration',
        'model': 'negative_binomial',
        'transform': 'none',
        'notes': f'Number of {behavior} events'
    }

    roi_metadata[f'{behavior}_duration'] = {
        'family': 'roi',
        'type': 'positive_continuous',
        'units': 's',
        'denominator': 'outside_nest_duration',
        'model': 'gamma_or_log_LME',
        'transform': 'log_if_needed',
        'notes': f'Total time spent at {behavior}'
    }

    roi_metadata[f'{behavior}_mean_duration'] = {
        'family': 'roi',
        'type': 'positive_continuous',
        'units': 's/event',
        'denominator': f'{behavior}_count',
        'model': 'gamma_or_log_LME',
        'transform': 'log_if_needed',
        'notes': f'Mean duration per {behavior} event'
    }

    roi_metadata[f'{behavior}_duration_fraction'] = {
        'family': 'roi',
        'type': 'proportion',
        'units': 'fraction',
        'denominator': 'outside_nest_duration',
        'model': 'beta_or_fraction_model',
        'transform': 'none',
        'notes': f'Fraction of outside-nest time spent at {behavior}'
    }

    roi_metadata[f'{behavior}_event_rate'] = {
        'family': 'roi',
        'type': 'rate',
        'units': 'events/s',
        'denominator': 'outside_nest_duration',
        'model': 'negative_binomial_with_offset',
        'transform': 'none',
        'notes': f'{behavior} events per unit outside-nest exposure'
    }
feature_metadata.update(roi_metadata)

#%%
chase_metadata = {

    'chasing_duration': {
        'family': 'chase',
        'type': 'positive_continuous',
        'units': 's',
        'denominator': 'time_window',
        'model': 'gamma_or_log_LME',
        'transform': 'log_if_needed',
        'notes': 'Total duration spent chasing'
    },

    'chasing_count': {
        'family': 'chase',
        'type': 'count',
        'units': 'events',
        'denominator': 'time_window',
        'model': 'negative_binomial',
        'transform': 'none',
        'notes': 'Number of chasing events'
    },

    'chasing_mean_duration': {
        'family': 'chase',
        'type': 'positive_continuous',
        'units': 's/event',
        'denominator': 'chasing_count',
        'model': 'gamma_or_log_LME',
        'transform': 'log_if_needed',
        'notes': 'Mean duration per chasing event'
    },

    'chased_duration': {
        'family': 'chase',
        'type': 'positive_continuous',
        'units': 's',
        'denominator': 'time_window',
        'model': 'gamma_or_log_LME',
        'transform': 'log_if_needed',
        'notes': 'Total duration spent being chased'
    },

    'chased_count': {
        'family': 'chase',
        'type': 'count',
        'units': 'events',
        'denominator': 'time_window',
        'model': 'negative_binomial',
        'transform': 'none',
        'notes': 'Number of times being chased'
    },

    'chased_mean_duration': {
        'family': 'chase',
        'type': 'positive_continuous',
        'units': 's/event',
        'denominator': 'chased_count',
        'model': 'gamma_or_log_LME',
        'transform': 'log_if_needed',
        'notes': 'Mean duration per chased event'
    },

    'chasing_duration_fraction': {
        'family': 'chase',
        'type': 'proportion',
        'units': 'fraction',
        'denominator': 'time_window',
        'model': 'beta_or_fraction_model',
        'transform': 'none',
        'notes': 'Fraction of time spent chasing'
    },

    'chasing_event_rate': {
        'family': 'chase',
        'type': 'rate',
        'units': 'events/s',
        'denominator': 'time_window',
        'model': 'negative_binomial_with_offset',
        'transform': 'none',
        'notes': 'Chasing events per unit time'
    },

    'chased_duration_fraction': {
        'family': 'chase',
        'type': 'proportion',
        'units': 'fraction',
        'denominator': 'time_window',
        'model': 'beta_or_fraction_model',
        'transform': 'none',
        'notes': 'Fraction of time spent being chased'
    },

    'chased_event_rate': {
        'family': 'chase',
        'type': 'rate',
        'units': 'events/s',
        'denominator': 'time_window',
        'model': 'negative_binomial_with_offset',
        'transform': 'none',
        'notes': 'Chased events per unit time'
    },

    'chasing_duration_ratio': {
        'family': 'chase',
        'type': 'ratio',
        'units': 'ratio',
        'denominator': 'chased_duration',
        'model': 'LME_if_stable',
        'transform': 'log_if_needed',
        'notes': 'Chasing duration relative to chased duration'
    },

    'chased_duration_ratio': {
        'family': 'chase',
        'type': 'ratio',
        'units': 'ratio',
        'denominator': 'chasing_duration',
        'model': 'LME_if_stable',
        'transform': 'log_if_needed',
        'notes': 'Chased duration relative to chasing duration'
    },

    'chasing_event_ratio': {
        'family': 'chase',
        'type': 'ratio',
        'units': 'ratio',
        'denominator': 'chased_count',
        'model': 'LME_if_stable',
        'transform': 'log_if_needed',
        'notes': 'Chasing event count relative to chased event count'
    },

    'chased_event_ratio': {
        'family': 'chase',
        'type': 'ratio',
        'units': 'ratio',
        'denominator': 'chasing_count',
        'model': 'LME_if_stable',
        'transform': 'log_if_needed',
        'notes': 'Chased event count relative to chasing event count'
    }
}

feature_metadata.update(chase_metadata)

#%%
hierarchy_metadata = {

    'normDS': {
        'family': 'hierarchy',
        'type': 'continuous',
        'units': 'normalized_score',
        'denominator': None,
        'model': 'LME',
        'transform': 'none',
        'notes': 'Continuous hierarchy score underlying rank; calculated per mouse and phase'
    },

    'rank': {
        'family': 'hierarchy',
        'type': 'ordinal',
        'units': 'alpha_beta_gamma_delta',
        'denominator': None,
        'model': 'ordinal_mixed_model',
        'transform': 'none',
        'notes': 'Ordinal social hierarchy rank; alpha > beta > gamma > delta'
    }
}

feature_metadata.update(hierarchy_metadata)

#%%
locomotion_metadata = {

    'total_distance': {
        'family': 'locomotion',
        'type': 'continuous',
        'units': 'm',
        'denominator': None,
        'model': 'LME',
        'transform': 'none',
        'notes': 'Total distance traveled during valid locomotion observations'
    },

    'valid_locomotion_duration': {
        'family': 'locomotion',
        'type': 'continuous',
        'units': 's',
        'denominator': 'time_window',
        'model': 'LME_or_beta',
        'transform': 'none',
        'notes': 'Duration for which locomotion was valid/measurable; equivalent to valid outside-nest locomotion observation time'
    },

    'mean_speed': {
        'family': 'locomotion',
        'type': 'continuous',
        'units': 'm/s',
        'denominator': 'valid_locomotion_duration',
        'model': 'LME',
        'transform': 'none',
        'notes': 'Mean speed during valid locomotion observations'
    },

    'median_speed': {
        'family': 'locomotion',
        'type': 'continuous',
        'units': 'm/s',
        'denominator': 'valid_locomotion_duration',
        'model': 'LME',
        'transform': 'none',
        'notes': 'Median speed during valid locomotion observations'
    },

    'mean_abs_angular_velocity': {
        'family': 'locomotion',
        'type': 'positive_continuous',
        'units': 'rad/s',
        'denominator': 'valid_locomotion_duration',
        'model': 'LME_or_gamma',
        'transform': 'log_if_needed',
        'notes': 'Mean absolute angular velocity; measures turning intensity independent of turning direction'
    },

    'mean_abs_acceleration': {
        'family': 'locomotion',
        'type': 'positive_continuous',
        'units': 'm/s2',
        'denominator': 'valid_locomotion_duration',
        'model': 'LME_or_gamma',
        'transform': 'log_if_needed',
        'notes': 'Mean absolute acceleration during valid locomotion observations'
    },

    'valid_locomotion_duration_fraction': {
        'family': 'locomotion',
        'type': 'proportion',
        'units': 'fraction',
        'denominator': 'time_window',
        'model': 'beta_or_fraction_model',
        'transform': 'none',
        'notes': 'Fraction of the time window with valid locomotion observations'
    }
}

feature_metadata.update(locomotion_metadata)
#%%
motionless_metadata = {

    'motionless_count': {
        'family': 'motionless',
        'type': 'count',
        'units': 'events',
        'denominator': 'time_window',
        'model': 'negative_binomial',
        'transform': 'none',
        'notes': 'Number of motionless bouts'
    },

    'motionless_duration': {
        'family': 'motionless',
        'type': 'positive_continuous',
        'units': 's',
        'denominator': 'time_window',
        'model': 'gamma_or_log_LME',
        'transform': 'log_if_needed',
        'notes': 'Total motionless duration'
    },

    'motionless_mean_duration': {
        'family': 'motionless',
        'type': 'positive_continuous',
        'units': 's/event',
        'denominator': 'motionless_count',
        'model': 'gamma_or_log_LME',
        'transform': 'log_if_needed',
        'notes': 'Mean duration per motionless bout'
    },

    'motionless_duration_fraction': {
        'family': 'motionless',
        'type': 'proportion',
        'units': 'fraction',
        'denominator': 'time_window',
        'model': 'beta_or_fraction_model',
        'transform': 'none',
        'notes': 'Fraction of time spent motionless'
    },

    'motionless_event_rate': {
        'family': 'motionless',
        'type': 'rate',
        'units': 'events/s',
        'denominator': 'time_window',
        'model': 'negative_binomial_with_offset',
        'transform': 'none',
        'notes': 'Motionless events per unit time'
    }
}

feature_metadata.update(motionless_metadata)

#%%
speeding_metadata = {

    'speeding_count': {
        'family': 'speeding',
        'type': 'count',
        'units': 'events',
        'denominator': 'time_window',
        'model': 'negative_binomial',
        'transform': 'none',
        'notes': 'Number of speeding bouts'
    },

    'speeding_duration': {
        'family': 'speeding',
        'type': 'positive_continuous',
        'units': 's',
        'denominator': 'time_window',
        'model': 'gamma_or_log_LME',
        'transform': 'log_if_needed',
        'notes': 'Total speeding duration'
    },

    'speeding_mean_duration': {
        'family': 'speeding',
        'type': 'positive_continuous',
        'units': 's/event',
        'denominator': 'speeding_count',
        'model': 'gamma_or_log_LME',
        'transform': 'log_if_needed',
        'notes': 'Mean duration per speeding bout'
    },

    'speeding_duration_fraction': {
        'family': 'speeding',
        'type': 'proportion',
        'units': 'fraction',
        'denominator': 'time_window',
        'model': 'beta_or_fraction_model',
        'transform': 'none',
        'notes': 'Fraction of time spent speeding'
    },

    'speeding_event_rate': {
        'family': 'speeding',
        'type': 'rate',
        'units': 'events/s',
        'denominator': 'time_window',
        'model': 'negative_binomial_with_offset',
        'transform': 'none',
        'notes': 'Speeding events per unit time'
    }
}

feature_metadata.update(speeding_metadata)
#%%
social_metadata = {

    'nest_frames': {
        'family': 'social',
        'type': 'continuous',
        'units': 'frames',
        'denominator': None,
        'model': 'LME_or_count_like',
        'transform': 'none',
        'notes': 'Frames associated with nest/social occupancy measure'
    },

    'weighted_sum': {
        'family': 'social',
        'type': 'continuous',
        'units': 'index',
        'denominator': None,
        'model': 'LME_if_residuals_acceptable',
        'transform': 'none',
        'notes': 'Weighted social interaction/occupancy index'
    },

    'alone_sum': {
        'family': 'social',
        'type': 'continuous',
        'units': 'frames_or_index',
        'denominator': None,
        'model': 'LME_if_residuals_acceptable',
        'transform': 'none',
        'notes': 'Amount of time/frames classified as alone'
    },

    'weighted_co_occupancy': {
        'family': 'social',
        'type': 'continuous_or_proportion',
        'units': 'index',
        'denominator': None,
        'model': 'inspect_distribution',
        'transform': 'none',
        'notes': 'Weighted measure of co-occupancy'
    },

    'alone_fraction': {
        'family': 'social',
        'type': 'proportion',
        'units': 'fraction',
        'denominator': 'relevant_social_observation',
        'model': 'beta_or_fraction_model',
        'transform': 'none',
        'notes': 'Fraction of relevant observation time classified as alone'
    },

    'feeding_frames': {
        'family': 'social',
        'type': 'continuous',
        'units': 'frames',
        'denominator': None,
        'model': 'LME_or_count_like',
        'transform': 'none',
        'notes': 'Total feeder-related frames'
    },

    'drinking_frames': {
        'family': 'social',
        'type': 'continuous',
        'units': 'frames',
        'denominator': None,
        'model': 'LME_or_count_like',
        'transform': 'none',
        'notes': 'Total water-related frames'
    },

    'ramps_frames': {
        'family': 'social',
        'type': 'continuous',
        'units': 'frames',
        'denominator': None,
        'model': 'LME_or_count_like',
        'transform': 'none',
        'notes': 'Total ramp-related frames'
    },

    's_wall_frames': {
        'family': 'social',
        'type': 'continuous',
        'units': 'frames',
        'denominator': None,
        'model': 'LME_or_count_like',
        'transform': 'none',
        'notes': 'Total side-wall-related frames'
    }
}

social_rois = ['feeding', 'drinking', 'ramps', 's_wall']

for roi in social_rois:

    social_metadata[f'{roi}_together_frames'] = {
        'family': 'social',
        'type': 'continuous',
        'units': 'frames',
        'denominator': f'{roi}_frames',
        'model': 'binomial_or_fraction_model',
        'transform': 'none',
        'notes': f'Frames with multiple mice together at {roi}'
    }

    social_metadata[f'{roi}_alone_frames'] = {
        'family': 'social',
        'type': 'continuous',
        'units': 'frames',
        'denominator': f'{roi}_frames',
        'model': 'binomial_or_fraction_model',
        'transform': 'none',
        'notes': f'Frames with mouse alone at {roi}'
    }

    social_metadata[f'{roi}_together_fraction'] = {
        'family': 'social',
        'type': 'proportion',
        'units': 'fraction',
        'denominator': f'{roi}_frames',
        'model': 'binomial_or_fraction_model',
        'transform': 'none',
        'notes': f'Fraction of {roi} frames spent together'
    }

    social_metadata[f'{roi}_alone_fraction'] = {
        'family': 'social',
        'type': 'proportion',
        'units': 'fraction',
        'denominator': f'{roi}_frames',
        'model': 'binomial_or_fraction_model',
        'transform': 'none',
        'notes': f'Fraction of {roi} frames spent alone'
    }

feature_metadata.update(social_metadata)

#%%
metadata_df = (
    pd.DataFrame.from_dict(feature_metadata, orient='index')
    .reset_index()
    .rename(columns={'index': 'feature'})
)

metadata_df.head()

#%%
metadata_df.to_csv("feature_metadata.csv", index=False)
# %%
