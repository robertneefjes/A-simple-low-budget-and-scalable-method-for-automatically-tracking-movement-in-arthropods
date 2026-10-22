#--------------------------------------
# 0. Clean environment & load packages
#--------------------------------------
rm(list = ls(all.names = TRUE))

library(tidyverse)
library(lubridate)
library(purrr)
library(survival)
library(broom)
library(ggsurvfit)
library(readxl)

#--------------------------------------
# 1. Read data
#--------------------------------------
Movement_log_30min_tube <- read_excel("Scriptie_data/movement_log_10min_tubes.xlsx")

testassay <- read_excel("~/Documents/testassay.xlsx")

Movement_log_30min_tube <- Movement_log_30min_tube %>%
  filter(!if_any(ends_with("_score"), ~ .x > 10000))

Movement_log_30min_tube <- Movement_log_30min_tube %>%
  mutate(across(
    ends_with("_score"),
    ~ {
      m <- max(.x, na.rm = TRUE)
      if (m == 0 | is.infinite(m)) 0 else .x / m
    }
  ))


Movement_log_30min_tube <- Movement_log_30min_tube %>%
  mutate(
    timestamp_posix = ymd_hms(timestamp_iso),
    minutes_since_start = as.numeric(timestamp_posix - min(timestamp_posix), units = "mins")
  )
#--------------------------------------
# 2. Remove movement rows around manual measurements
#    (4 min before, 1 min after each manual time)
#--------------------------------------

# Parse timestamps in the manual dataset
testassay <- testassay %>%
  mutate(
    timestamp_manual = dmy_hms(timestamp_iso)  # "29-4-2026T14:07:00"
  )

# Create the time windows to remove: [t - 4 min, t + 1 min]
intervals <- testassay %>%
  distinct(timestamp_manual) %>%                 # just in case there are duplicates
  transmute(
    start_rm = timestamp_manual - minutes(4),
    end_rm   = timestamp_manual + minutes(1)
  )

# Drop all movement rows whose timestamp falls in any of these windows
Movement_log_30min_tube <- Movement_log_30min_tube %>%
  mutate(
    to_remove = purrr::map_lgl(
      timestamp_posix,
      ~ any(.x >= intervals$start_rm & .x <= intervals$end_rm)
    )
  ) %>%
  filter(!to_remove) %>%
  dplyr::select(-to_remove)
Movement_log_30min_tube <- Movement_log_30min_tube %>%
  filter(
    !(
      # 07:00:00 up to (but not including) 07:10:00
      (hour(timestamp_posix) == 7  & minute(timestamp_posix) < 10) |
        # 20:00:00 up to (but not including) 20:10:00
        (hour(timestamp_posix) == 20 & minute(timestamp_posix) < 10)
    )
  )
Movement_long <- Movement_log_30min_tube %>%
  mutate(
    timestamp_posix = ymd_hms(timestamp_iso),
    minutes_since_start = as.numeric(timestamp_posix - min(timestamp_posix), units = "mins")
  ) %>%
  pivot_longer(
    cols = ends_with("_score"),
    names_to = "score_type",
    values_to = "score"
  )
#--------------------------------------
# 3.3.2 Relationship manual vs automated activity
# Assumes you already have:
#   - Movement_long: from your pivot_longer()
#   - testassay: manual observations
#
# KEY:
#   testassay$location  <->  Movement_long$score_type ("1_score", "2_score", ...)
#--------------------------------------

library(tidyverse)
library(lubridate)
library(MASS)    # polr()
library(broom)   # tidy(), glance()

#--------------------------------------
# 1. Prepare manual dataset (testassay)
#--------------------------------------
# Expected columns in testassay (adapt names if different):
#   - timestamp_iso : character timestamp of manual observation
#   - location      : numeric or integer (1,2,3,...)   <<< CHECK NAME >>>
#   - species       : species ID                       <<< CHECK NAME >>>
#   - manual_act    : manual activity score 0,1,2      <<< CHECK NAME >>>

testassay_clean <- testassay %>%
  mutate(
    timestamp_manual = dmy_hms(timestamp_iso),
    location         = as.integer(location),          # ensure numeric
    species          = factor(species),
    activity_amount   = factor(
      activity_amount,                                     # manual activity column
      levels = c(0, 1, 2),
      ordered = TRUE
    )
  )

#--------------------------------------
# 2. Prepare automated dataset (Movement_long)
#--------------------------------------
# Movement_long from your code has:
#   - timestamp_posix : POSIXct timestamp
#   - score_type      : e.g. "1_score", "2_score", ...
#   - score           : automated activity score (0–1)
# We derive a numeric location from score_type.

Movement_long_clean <- Movement_long %>%
  mutate(
    timestamp_auto = timestamp_posix,
    # extract the number part before "_score"; adjust if naming differs
    location       = score_type %>%
      str_replace("_score", "") %>%
      as.integer()
  )

# If you have multiple types of scores and only want a subset, filter here.
# For example, if score_type is "1_score", "2_score", etc. and all are relevant,
# you do NOT need to filter. (Leave as is.)

#--------------------------------------
# 3. Pair manual and automated data by location + time interval
#--------------------------------------
# Choose the interval length to match your automated measurement resolution.
# Your file is "movement_log_10min_tubes.xlsx" so we use 10-minute bins.

testassay_interval <- testassay_clean %>%
  mutate(interval = floor_date(timestamp_manual, unit = "10 minutes"))

Movement_interval <- Movement_long_clean %>%
  mutate(interval = floor_date(timestamp_auto, unit = "10 minutes"))

# Join by location and interval
paired_data <- testassay_interval %>%
  left_join(
    Movement_interval %>%
      dplyr::select(location, interval, score),
    by = c("location", "interval")
  )

# Keep only rows with both manual and automated data
paired_data <- paired_data %>%
  filter(!is.na(score), !is.na(activity_amount))

# Quick checks
paired_data %>% count(species)
summary(paired_data$score)
summary(paired_data$activity_amount)

#--------------------------------------
# 4. Spearman rank correlation per species
#--------------------------------------
# Manual: 0,1,2 (ordinal)  -> convert to numeric 0,1,2 for correlation
# Automated: score (0–1, continuous)

spearman_results <- paired_data %>%
  group_by(species) %>%
  summarise(
    n       = n(),
    rho     = suppressWarnings(
      cor(as.numeric(activity_amount), score,
          method = "spearman", use = "complete.obs")
    ),
    p_value = suppressWarnings(
      cor.test(as.numeric(activity_amount), score,
               method = "spearman")$p.value
    ),
    .groups = "drop"
  ) %>%
  arrange(p_value)

spearman_results

# Optional: visualization of relationship per species
ggplot(paired_data,
       aes(x = score,
           y = as.numeric(activity_amount),
           color = species)) +
  geom_jitter(width = 0, height = 0.05, alpha = 0.6) +
  labs(
    x = "Automated activity score (0–1)",
    y = "Manual activity (0–2, numeric)",
    color = "Species"
  ) +
  theme_minimal()

#--------------------------------------
# 5. Ordinal logistic regression
#--------------------------------------
# Response: activity_amount (0 < 1 < 2)
# Predictor: score (continuous 0–1)
# A: Global model (all species, species as factor)
# B: Separate model per species

# A) Global model
model_global <- polr(
  activity_amount ~ score + species,
  data = paired_data,
  Hess = TRUE
)

summary(model_global)

model_global_tidy <- tidy(model_global, conf.int = TRUE, exponentiate = TRUE)
model_global_tidy   # odds ratios + 95% CI

# B) Per-species models
models_by_species <- paired_data %>%
  group_by(species) %>%
  nest() %>%
  mutate(
    model = map(data, ~ polr(activity_amount ~ score, data = .x, Hess = TRUE)),
    coef  = map(model, ~ tidy(.x, conf.int = TRUE, exponentiate = TRUE)),
    fit   = map(model, glance)
  )

# Coefficients (odds ratios) per species
models_by_species %>%
  dplyr::select(species, coef) %>%
  unnest(coef)

# Model fit summaries per species (incl. pseudo R², etc.)
models_by_species %>%
  dplyr::select(species, fit) %>%
  unnest(fit)

# Create a grid for predictions for each species
newdat <- expand.grid(
  score   = seq(0, 1, by = 0.01),
  species = levels(paired_data$species)
)

newdat$pred <- predict(model_global, newdata = newdat, type = "probs")

# Convert to long format for plotting
pred_long <- newdat %>%
  bind_cols(as.data.frame(newdat$pred)) %>%
  pivot_longer(cols = c(`0`, `1`, `2`),
               names_to = "activity_cat",
               values_to = "prob")

ggplot(pred_long,
       aes(x = score, y = prob, color = activity_cat)) +
  geom_line(size = 1) +
  facet_wrap(~ species) +
  labs(
    x = "Automated activity score",
    y = "Predicted probability",
    color = "Manual activity\ncategory"
  ) +
  theme_minimal()
