library(ggplot2)
library(dplyr)
library(tidyverse)
library(reshape2)

############# Helper functions ##################
get_filtered_table_path <- function(sex, res){
  QC_path <- "/Users/xiuqi.ji/Library/CloudStorage/OneDrive-KarolinskaInstitutet/MDMA_ELA/social_box_2026/github_repo/behavior_dataset/final/QC_table"
  return(file.path(QC_path,paste0(sex,"_",res,"h_filtered.csv")))
}

subset_data <- function(df,exp = NULL, time_point = NULL, phase = NULL, day = NULL, condition = NULL){
  out <- df
  if (!is.null(exp)) {
    out <- out[out$exp == exp, , drop = FALSE]
  }
  
  if (!is.null(time_point)) {
    out <- out[out$time_point == time_point, , drop = FALSE]
  }
  
  if (!is.null(phase)) {
    out <- out[out$phase == phase, , drop = FALSE]
  }
  
  if (!is.null(day)) {
    out <- out[out$day == day, , drop = FALSE]
  }
  
  if (!is.null(condition)) {
    out <- out[out$condition %in% condition, , drop = FALSE]
  }
  return(out)
}

flag_statictical_outlier <- function(df, col) {
  
  median_value <- median(df[[col]], na.rm = TRUE)
  mad_value <- median(abs(df[[col]] - median_value), na.rm = TRUE)
  
  qc_col <- paste0("qc_", col, "_statistical_outlier")
  
  if (mad_value == 0) {
    df[[qc_col]] <- FALSE
    return(df)
  }
  
  robust_z <- 0.6745 * (df[[col]] - median_value) / mad_value
  
  df[[qc_col]] <- abs(robust_z) > 3.5
  
  df
}
#############  ##################

############# constants ##################
## feature groups 
info_cols <- c('day', 'phase', 'box', 'box_ID', 'mouse', 'time_bin', 'time_window','time_point', 'exp', 'sex', 'age', 'mouse_ID', 'background', 'treatment', 'condition')
qc_cols <- c('qc_exclude', 'qc_speed_outlier', 'qc_exclude_timebin')
nest_cols <- c('nest_duration', 'outside_nest_duration', 'nest_count', 'nest_mean_duration',)
roi_cols <-c('s_wall_count', 's_wall_duration', 's_wall_mean_duration', 's_wall_duration_fraction', 's_wall_event_rate', 
            'ramp1_count', 'ramp1_duration', 'ramp1_mean_duration', 'ramp1_duration_fraction', 'ramp1_event_rate', 
            'ramp2_count', 'ramp2_duration', 'ramp2_mean_duration', 'ramp2_duration_fraction', 'ramp2_event_rate', 
            'non_wall_count', 'non_wall_duration', 'non_wall_mean_duration', 'non_wall_duration_fraction', 'non_wall_event_rate', 
            'woodstick_count', 'woodstick_duration', 'woodstick_mean_duration', 'woodstick_duration_fraction', 'woodstick_event_rate', 
            'feeder_prox_count', 'feeder_prox_duration', 'feeder_prox_mean_duration', 'feeder_prox_duration_fraction', 'feeder_prox_event_rate', 
            'feeder_dist_count', 'feeder_dist_duration', 'feeder_dist_mean_duration', 'feeder_dist_duration_fraction', 'feeder_dist_event_rate', 
            'water_prox_count', 'water_prox_duration', 'water_prox_mean_duration', 'water_prox_duration_fraction', 'water_prox_event_rate', 
            'water_dist_count', 'water_dist_duration', 'water_dist_mean_duration', 'water_dist_duration_fraction', 'water_dist_event_rate')
chase_cols <- c('chasing_duration', 'chasing_count', 'chasing_mean_duration', 'chased_duration', 'chased_count', 'chased_mean_duration', 
              'chasing_duration_fraction', 'chasing_event_rate', 'chased_duration_fraction', 'chased_event_rate',
              'chasing_duration_ratio', 'chased_duration_ratio', 'chasing_event_ratio', 'chased_event_ratio')
hierarchy_cols <- c('normDS', 'rank')
locomotion_cols <- c('total_distance', 'valid_locomotion_duration', 'mean_speed', 'median_speed', 'mean_abs_angular_velocity', 'mean_abs_acceleration', 'valid_locomotion_duration_fraction')
motionless_cols <- c('motionless_count', 'motionless_duration', 'motionless_mean_duration', 'motionless_duration_fraction', 'motionless_event_rate')
speeding_cols <- c('speeding_count', 'speeding_duration', 'speeding_mean_duration', 'speeding_duration_fraction', 'speeding_event_rate')
social_cols <- c('nest_frames', 'weighted_sum', 'alone_sum', 'weighted_co_occupancy', 'alone_fraction', 'feeding_frames', 'drinking_frames', 'ramps_frames', 's_wall_frames', 'feeding_together_frames', 'drinking_together_frames', 'ramps_together_frames', 's_wall_together_frames', 'feeding_alone_frames', 'drinking_alone_frames', 'ramps_alone_frames', 's_wall_alone_frames', 'feeding_together_fraction', 'feeding_alone_fraction', 'drinking_together_fraction', 'drinking_alone_fraction', 'ramps_together_fraction', 'ramps_alone_fraction', 's_wall_together_fraction', 's_wall_alone_fraction')

#############  ##################

############# descriptive stats ##################
## READ DATA
data <- read.csv(get_filtered_table_path('male',12))
baseline_active <- subset_data(data,time_point = 'baseline',phase = 'active')
mdma_active <- subset_data(data,time_point = 'MDMA',phase = 'active')
baseline_active_ELA <- subset_data(baseline_active,condition = c('ELA_saline','ELA_MDMA'))
summary(baseline_active)
ggplot(
  baseline_active,
  aes(
    x = factor(day),
    y = valid_locomotion_duration,
    colour = background,
    group = mouse_ID
  )
) +
  geom_line(alpha = .3) +
  geom_point() +
  stat_summary(
    aes(group = background),
    fun = mean,
    geom = "line",
    linewidth = 1
  ) +
  stat_summary(
    aes(group = background),
    fun = mean,
    geom = "point",
    size = 3
  ) +
  facet_wrap(~ sex) +
  theme_bw()
#############  ##################

############# Quick PCA ##################
male_12h <- read.csv(get_filtered_table_path('male',12))
female_12h <- read.csv(get_filtered_table_path('female',12))

pca_df <- subset_data(female_12h,time_point = 'baseline',phase = 'active')
pca_features <- c('nest_duration',#'total_distance',
                  'mean_speed', 'mean_abs_angular_velocity','motionless_mean_duration','speeding_duration_fraction',
                  's_wall_duration_fraction', 'ramp1_duration_fraction','ramp2_duration_fraction', 'feeder_prox_duration_fraction','feeder_dist_duration_fraction',
                  'water_prox_duration_fraction','water_dist_duration_fraction','chasing_duration_fraction', 'chased_duration_fraction', 
                  'weighted_co_occupancy', 'feeding_together_fraction','drinking_together_fraction','ramps_together_fraction','s_wall_together_fraction')
# check first!
pca_df <- pca_df %>% 
  mutate(
    across(all_of(pca_features), ~replace_na(.x,0))
  )

feature_mat <- pca_df %>%
  select(all_of(pca_features))

####check correlation first
cor_mat <- cor(
  feature_mat,
  use = "pairwise.complete.obs"
)
corrplot::corrplot(cor_mat)
# be careful for pairs with |r| > 0.8

### scaling
X <- scale(feature_mat)
pca <- prcomp(
  X,
  center = TRUE,
  scale. = TRUE
)
# make a dataframe from result
scores <- as.data.frame(pca$x) %>%
  bind_cols(
    pca_df %>%
      select(mouse_ID, sex, background, exp,day)
  )
# plot 
ggplot(scores, aes(PC1, PC2, colour = background, shape = sex)) +
  geom_point(size = 3) +
  theme_bw()

ggplot(scores, aes(PC1, PC2, colour = background)) +
  geom_point(size = 3) +
  facet_wrap(~ day) +
  theme_bw()
# check explained variance
summary(pca)
# check loadings
loadings <- as.data.frame(pca$rotation)

loadings %>%
  select(PC1, PC2) %>%
  arrange(desc(abs(PC1)))
#############  ##################