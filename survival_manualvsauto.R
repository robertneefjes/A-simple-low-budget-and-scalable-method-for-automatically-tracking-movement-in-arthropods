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
library(survminer)

#--------------------------------------
# 1. Read manual measurement times (testassay)
#--------------------------------------
testassay <- read_excel("~/Documents/testassay.xlsx")
# Columns: timestamp_iso, tube_petri, location, activity_amount,
#          species, salt, agar

# Parse timestamps in the manual dataset, assuming e.g. "29-4-2026T14:07:00"
testassay <- testassay %>%
  mutate(
    timestamp_manual = timestamp_iso %>%
      str_replace("T", " ") %>%
      dmy_hms()
  )

# Create removal windows: [t - 4 min, t + 1 min]
intervals <- testassay %>%
  distinct(timestamp_manual) %>%
  transmute(
    start_rm = timestamp_manual - minutes(4),
    end_rm   = timestamp_manual + minutes(1)
  )

#--------------------------------------
# 2. Helper: function to find last run of ≥3 scores > 0.02
#--------------------------------------
find_last_run_end <- function(x, threshold = 0.02, min_len = 3) {
  high <- x > threshold
  r    <- rle(high)
  idx  <- which(r$values & r$lengths >= min_len)
  if (length(idx) == 0L) return(NA_integer_)
  last_idx <- tail(idx, 1)
  end_pos  <- sum(r$lengths[seq_len(last_idx)])
  end_pos
}

#--------------------------------------
# 3. Helper: process a single movement file
#    - Clean, scale, remove intervals, remove 07:00–07:10 and 20:00–20:10
#    - Compute *_alive columns (1 until last 3× >0.02 run, 0 after)
#    - Add `setup` label
#--------------------------------------
process_movement_file <- function(file_path, setup_label) {
  dat <- read_excel(file_path)
  # Expected columns: timestamp_iso, 1_score, 2_score, ..., 24_score
  
  # Remove extreme scores and rescale to [0,1] per ROI
  dat <- dat %>%
    filter(!if_any(ends_with("_score"), ~ .x > 10000)) %>%
    mutate(
      across(
        ends_with("_score"),
        ~ {
          m <- max(.x, na.rm = TRUE)
          if (is.na(m) || m == 0 || is.infinite(m)) 0 else .x / m
        }
      )
    )
  
  # Add POSIX time & minutes_since_start (within this file)
  dat <- dat %>%
    mutate(
      timestamp_posix     = ymd_hms(timestamp_iso),
      minutes_since_start = as.numeric(timestamp_posix - min(timestamp_posix),
                                       units = "mins")
    )
  
  # Remove rows around manual measurement times
  dat <- dat %>%
    mutate(
      to_remove = map_lgl(
        timestamp_posix,
        ~ any(.x >= intervals$start_rm & .x <= intervals$end_rm)
      )
    ) %>%
    filter(!to_remove) %>%
    dplyr::select(-to_remove)
  
  # Remove 07:00–07:10 and 20:00–20:10 (clock time)
  dat <- dat %>%
    filter(
      !(
        (hour(timestamp_posix) == 7  & minute(timestamp_posix) < 10) |
          (hour(timestamp_posix) == 20 & minute(timestamp_posix) < 10)
      )
    )
  
  # Define ROI columns
  roi_cols <- names(dat) %>%
    str_subset("_score$")
  
  # Find last run row per ROI
  last_run_row <- purrr::map_int(
    dat[roi_cols],
    find_last_run_end
  )
  
  # Create *_alive columns (1 until last run row, 0 after)
  dat <- dat %>%
    mutate(row_id = dplyr::row_number()) %>%
    mutate(
      across(
        all_of(roi_cols),
        ~ {
          cut_row <- last_run_row[[cur_column()]]
          if (is.na(cut_row)) {
            # never had 3 consecutive > 0.02: always 0
            as.integer(0L)
          } else {
            # alive (1) up to cut_row, dead (0) after
            as.integer(row_id <= cut_row)
          }
        },
        .names = "{.col}_alive"
      )
    ) %>%
    # drop original movement columns, keep *_alive + time + metadata
    dplyr::select(-dplyr::all_of(roi_cols)) %>%
    mutate(setup = setup_label)
  
  return(dat)
}

#--------------------------------------
# 4. Process both movement files and merge
#--------------------------------------
movement_log_filter_petri <- process_movement_file(
  "Scriptie_data/movement_log_10sec_petri.xlsx",
  setup_label = "petri"
)

movement_log_filter_tube <- process_movement_file(
  "Scriptie_data/movement_log_10sec_tubes.xlsx",
  setup_label = "tube"
)

movement_log_filter_all <- bind_rows(
  movement_log_filter_petri,
  movement_log_filter_tube
)

#--------------------------------------
# 5. Movement data: long format with species assignment
#--------------------------------------
# Species mapping (movement):
# - PETRI:
#     polyctena: 1,4,5,8,9,12,13,16,17,20,21,24
#     diaperinus: all others
#
# - TUBE:
#     polyctena: 1,2,7,8,9,10,15,16,17,18,23,24
#     diaperinus: all others

alive_long_movement <- movement_log_filter_all %>%
  pivot_longer(
    cols      = ends_with("_alive"),
    names_to  = "roi",
    values_to = "alive"
  ) %>%
  mutate(
    roi     = str_remove(roi, "_alive$"),        # e.g., "1_score"
    roi_num = readr::parse_number(roi)           # e.g., 1
  ) %>%
  mutate(
    species = case_when(
      # PETRI mapping
      setup == "petri" &
        roi_num %in% c(1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24)
      ~ "polyctena",
      setup == "petri" ~ "diaperinus",
      
      # TUBE mapping
      setup == "tube" &
        roi_num %in% c(1, 2, 7, 8, 9, 10, 15, 16, 17, 18, 23, 24)
      ~ "polyctena",
      setup == "tube" ~ "diaperinus",
      
      TRUE ~ NA_character_
    )
  )

#--------------------------------------
# 6. Movement data: survival dataset per ROI and setup
#--------------------------------------
ant_surv_movement <- alive_long_movement %>%
  arrange(setup, roi, minutes_since_start) %>%
  group_by(setup, roi, species) %>%   # each (setup, roi) is one individual
  summarise(
    time = if (any(alive == 1, na.rm = TRUE)) {
      # last time with alive == 1
      max(minutes_since_start[alive == 1], na.rm = TRUE)
    } else {
      # never alive (all 0): set time to 0
      0
    },
    status = as.integer(
      any(alive == 0, na.rm = TRUE) & any(alive == 1, na.rm = TRUE)
    ),
    .groups = "drop"
  ) %>%
  mutate(
    source = "movement",
    id     = paste(setup, roi, source, sep = "_")
  )

#--------------------------------------
# 7. Manual testassay data: survival dataset (with censoring)
#    - You keep measuring after death.
#    - Death = first time activity_amount == 0 AND stays 0 afterwards.
#    - If never satisfies that, censored at last observation time.
#--------------------------------------
manual_surv_raw <- testassay %>%
  arrange(tube_petri, location, timestamp_manual) %>%
  group_by(tube_petri, location, species, salt, agar) %>%
  mutate(
    minutes_since_start = as.numeric(
      timestamp_manual - first(timestamp_manual),
      units = "mins"
    ),
    nonzero       = activity_amount != 0,
    nonzero_after = rev(cummax(rev(nonzero))),  # TRUE if any later non-zero
    death_row     = activity_amount == 0 & !nonzero_after
  )

manual_surv <- manual_surv_raw %>%
  summarise(
    time = case_when(
      any(death_row, na.rm = TRUE) ~
        min(minutes_since_start[death_row], na.rm = TRUE),  # first such 0
      TRUE ~
        max(minutes_since_start, na.rm = TRUE)              # censored: last obs
    ),
    status = as.integer(any(death_row, na.rm = TRUE)),      # 1 = dead, 0 = censored
    .groups = "drop"
  ) %>%
  rename(
    setup = tube_petri,
    roi   = location
  ) %>%
  mutate(
    roi    = as.character(roi),
    source = "manual",
    id     = paste(setup, roi, source, sep = "_")
  )
# species in manual_surv is already "polyctena"/"diaperinus"

#--------------------------------------
# 8. Combine movement + manual survival datasets
#--------------------------------------
surv_all <- bind_rows(
  ant_surv_movement,
  manual_surv
)

# species names are already harmonised to "polyctena"/"diaperinus"

#--------------------------------------
# 9. Pairwise log-rank tests (Bonferroni-adjusted)
#    Across all combinations of species × source
#--------------------------------------
surv_all <- surv_all %>%
  mutate(group = interaction(species, source, drop = TRUE))
# e.g. "polyctena.movement", "polyctena.manual", etc.

pw_logrank <- pairwise_survdiff(
  Surv(time, status) ~ group,
  data            = surv_all,
  p.adjust.method = "bonferroni"
)

pw_logrank  # shows pairwise comparisons and adjusted p-values

#--------------------------------------
# Plot only polyctena (manual vs movement), with p-value
#--------------------------------------
surv_poly <- surv_all %>%
  filter(species == "diaperinus")

# Log-rank test (manual vs movement for polyctena only)
surv_diff_poly <- survdiff(Surv(time, status) ~ source, data = surv_poly)
pval_poly <- 1 - pchisq(surv_diff_poly$chisq,
                        df = length(surv_diff_poly$n) - 1)

# Nicely formatted label
pval_label <- paste0("Log-rank p = ", signif(pval_poly, 3))

# Position for the p-value text (near top-right)
max_time <- max(surv_poly$time, na.rm = TRUE)

fit_poly_source <- survfit(Surv(time, status) ~ source, data = surv_poly)

ggsurvfit(fit_poly_source, linewidth = 1.4) +
  coord_cartesian(ylim = c(0, 1)) +
  scale_x_continuous(
    expand = expansion(mult = c(0.01, 0.01))
  ) +
  scale_color_manual(
    name   = "",
    values = c(
      "source=movement" = "steelblue4",
      "source=manual"   = "firebrick3"
    ),
    labels = c(
      "source=movement" = "Automatic movement measurements",
      "source=manual"   = "Manual movement observations"
    )
  ) +
  labs(
    x     = "Time since start (min)",
    y     = "Survival probability",
    title = "30 minutes time interval"
  ) +
  # Add p-value text
  annotate(
    "text",
    x      = max_time * 0.99,  # a bit from the right edge
    y      = 0.05,             # near bottom; change if you prefer
    label  = pval_label,
    hjust  = 1,
    vjust  = 0,
    size   = 6
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title   = element_text(hjust = 0.5, face = "bold", size = 18),
    axis.title.x = element_text(face  = "bold", size = 12),
    axis.title.y = element_text(face  = "bold", size = 12),
    axis.text.x  = element_text(size  = 12),
    axis.text.y  = element_text(size  = 12),
    legend.title = element_text(size  = 13),
    legend.text  = element_text(size  = 12),
    legend.position = "none"
  ) +
  guides(
    color = guide_legend(order = 1)
  )
