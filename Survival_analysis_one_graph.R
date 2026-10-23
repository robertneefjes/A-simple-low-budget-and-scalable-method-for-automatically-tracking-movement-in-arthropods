library(dplyr)
library(survival)
library(ggsurvfit)
library(survminer)

df <- assay2 # <- insert dataset here

  ant_surv <- df %>%   # <- use your long data frame name here
    arrange(source_sheet, position, minutes) %>%      # <- What columns define a unique indiviual
    group_by(source_sheet, position, species, treatment, nest, place, assay, climate) %>%              #add here what data you need
    summarise(
      time = if (any(score == 0)) min(minutes[score == 0]) else max(minutes),
      status = as.integer(any(score == 0)),  # 1 = died, 0 = censored
      .groups = "drop"
    )
  
  
  
#Filtering the data#
  
  ant_desiccation <- ant_surv %>%
    filter(assay == "assay2")
  

  fit_desicc <- survfit2(Surv(time, status) ~ treatment, data = ant_desiccation)     #what to group by
  
  ggsurvfit(fit_desicc, linewidth = 1) +
    labs(
      title = "Kaplan Meier survival curve of 384 ants comparing control vs desiccation",
      x = "Time since strart (min)",
      y = "Survival probability"
    ) +
    add_pvalue() +
    scale_y_continuous(limits = c(0, 1)) +
    scale_x_continuous(
      breaks = seq(0, 5000, by = 1000)
    ) +
    coord_cartesian(xlim = c(0, 3000)) # <- range of the survival plot#


# Use the same ant_desiccation data as before
# ant_desiccation has: time, status, climate, ...

# Global log-rank test (already implicit in add_pvalue())
survdiff(Surv(time, status) ~ treatment, data = ant_desiccation)                   #what to group by

# Pairwise log-rank tests between climates
pairwise_results <- pairwise_survdiff(
  Surv(time, status) ~ treatment,
  data = ant_desiccation,
  p.adjust.method = "bonferroni"   # or "BH", "holm", etc.
)

pairwise_results
