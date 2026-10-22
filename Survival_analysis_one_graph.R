library(dplyr)
library(survival)
library(ggsurvfit)
library(survminer)

library(dplyr)
library(survival)
library(ggsurvfit)
library(survminer)

df <- dat_long

# Filter species
ant_desiccation <- df %>%
  filter(plant == "E. cheiranthoides")

# Convert repeated observations into one row per ant
survival_data <- ant_desiccation %>%
  group_by(plant_nmbr, species, plant) %>%
  arrange(day) %>%
  summarise(
    time = ifelse(
      any(value == 0),
      min(day[value == 0]),
      max(day)
    ),
    event = ifelse(any(value == 0), 1, 0),
    .groups = "drop"
  )

# Look at the resulting survival data
print(survival_data)

# Kaplan-Meier model
fit_desicc <- survfit2(
  Surv(time, event) ~ species,
  data = survival_data
)

# Plot
ggsurvfit(fit_desicc, linewidth = 1) +
  labs(
    title = expression(
      "Survival curve of " *
        italic("P. napi") *
        " vs " *
        italic("P. rapae") *
        " on " *
        italic("E. cheiranthoides")
    ),
    x = "Time (days)",
    y = "Survival probability",
    color = "plant"
  ) +
  add_pvalue() +
  scale_y_continuous(limits = c(0, 1)) +
  add_confidence_interval() +
  guides(
    fill = "none"
  ) +
  theme(
    legend.text = element_text(face = "italic")
  )


library(dplyr)
library(survival)
library(ggsurvfit)
library(survminer)
library(ggplot2)

df <- dat_long

# Convert repeated observations into one row per caterpillar
survival_data <- df %>%
  group_by(plant_nmbr, species, plant) %>%
  arrange(day) %>%
  summarise(
    time = ifelse(
      any(value == 0),
      min(day[value == 0]),
      max(day)
    ),
    event = ifelse(any(value == 0), 1, 0),
    .groups = "drop"
  ) %>%
  mutate(
    group = interaction(species, plant, sep = " - ")
  )

# Inspect the survival data
print(survival_data)

# Kaplan-Meier model: all species-plant combinations
fit_all <- survfit2(
  Surv(time, event) ~ group,
  data = survival_data
)

# Plot all four survival curves
ggsurvfit(fit_all, linewidth = 1) +
  labs(
    title = "Survival on different host plants",
    x = "Time (days)",
    y = "Survival probability",
    color = "Species and host plant"
  ) +
  scale_y_continuous(limits = c(0, 1)) +
  scale_color_manual(
    values = c(
      "#0072B2",
      "#D55E00",
      "#009E73",
      "#CC79A7"
    ),
    labels = c(
      expression(italic("P. napi") ~ "on" ~ italic("B. oleracea")),
      expression(italic("P. rapae") ~ "on" ~ italic("B. oleracea")),
      expression(italic("P. napi") ~ "on" ~ italic("E. cheiranthoides")),
      expression(italic("P. rapae") ~ "on" ~ italic("E. cheiranthoides"))
    )
  ) +
  guides(fill = "none") +
  theme_classic() +
  theme(
    legend.text = element_text(face = "plain")
  ) +
  scale_x_continuous(
    breaks = 0:11,
    limits = c(0, 11)
  )


# Pairwise log-rank tests between all groups
pairwise_survdiff(
  Surv(time, event) ~ group,
  data = survival_data,
  p.adjust.method = "none"
)
