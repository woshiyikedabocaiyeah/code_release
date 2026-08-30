# ============================================================================
# Effect-size add-on for the ANCOVA / interaction analyses
# Drop these chunks into Untitled.Rmd (after ex_all and the log transforms
# are created). Requires: car, effectsize, emmeans.
#
# Reports, for every DV that already has a group main effect + Tukey post-hoc:
#   (1) partial eta-squared (with 90% CI) for the omnibus group effect
#   (2) Cohen's d (with 95% CI) for each pairwise group contrast
#
# Group -> manuscript label map:
#   Categorization  = Concept Verification
#   VoE             = Plausibility Assessment
#   Sensorimotor    = Affordance Recognition
# ============================================================================

library(car)
library(effectsize)
library(emmeans)

# Path repair after reorganisation: normally source()d into
# Untitled.Rmd, which defines folder_path; fall back to the
# original data directory when run on its own.
if (!exists("folder_path")) {
  folder_path <- "."
}

# Type III SS requires orthogonal contrasts to be meaningful
options(contrasts = c("contr.sum", "contr.poly"))

namemap <- c("Categorization" = "Concept Verification",
             "VoE"            = "Plausibility Assessment",
             "Sensorimotor"   = "Affordance Recognition")

# ---- (1) Partial eta-squared for the group main effect --------------------
# mtype = "ancova"      -> DV ~ group + covariate           (used for ACC)
# mtype = "interaction" -> DV ~ group * covariate           (used for log RT/IES)
eta_group <- function(dv, mtype, cov, label) {
  form <- if (mtype == "ancova") paste0(dv, " ~ group + ", cov)
          else                   paste0(dv, " ~ group * ", cov)
  m  <- lm(as.formula(form), data = ex_all)
  aov3 <- car::Anova(m, type = "III")
  e  <- eta_squared(aov3, partial = TRUE, alternative = "two.sided")
  r  <- e[e$Parameter == "group", ]
  data.frame(metric       = label,
             F            = round(aov3["group", "F value"], 2),
             df1          = aov3["group", "Df"],
             df_res       = df.residual(m),
             eta2_partial = round(r$Eta2_partial, 3),
             CI_low       = round(r$CI_low, 3),
             CI_high      = round(r$CI_high, 3))
}

eta_table <- rbind(
  eta_group("mean_ACC",         "ancova",      "mean_T_encode", "ACC"),
  eta_group("log_RT_onset",     "interaction", "log_T_encode",  "RTonset"),
  eta_group("log_RT_critical",  "interaction", "log_T_encode",  "RTcritical"),
  eta_group("log_IES_onset",    "interaction", "log_T_encode",  "IESonset"),
  eta_group("log_IES_critical", "interaction", "log_T_encode",  "IEScritical")
)
cat("=== Partial eta-squared for group main effect (90% CI) ===\n")
print(eta_table, row.names = FALSE)
write.csv(eta_table, file.path(folder_path, "effect_sizes_main_effects.csv"),
          row.names = FALSE)

# ---- (2) Cohen's d for each pairwise group contrast -----------------------
# d is standardised by the model residual SD (sigma), matching the
# covariate-adjusted model used for the Tukey comparisons.
cohend_pairs <- function(dv, cov, label) {
  m   <- lm(as.formula(paste0(dv, " ~ group + ", cov)), data = ex_all)
  emm <- emmeans(m, "group")
  d   <- as.data.frame(eff_size(emm, sigma = sigma(m), edf = df.residual(m)))
  d$metric   <- label
  d$contrast <- sapply(strsplit(d$contrast, " - "),
                       function(z) paste(namemap[z], collapse = " - "))
  d[, c("metric", "contrast", "effect.size", "lower.CL", "upper.CL")]
}

d_table <- do.call(rbind, list(
  cohend_pairs("mean_ACC",         "mean_T_encode", "ACC"),
  cohend_pairs("log_RT_onset",     "log_T_encode",  "RTonset"),
  cohend_pairs("log_RT_critical",  "log_T_encode",  "RTcritical"),
  cohend_pairs("log_IES_onset",    "log_T_encode",  "IESonset"),
  cohend_pairs("log_IES_critical", "log_T_encode",  "IEScritical")
))
names(d_table) <- c("metric", "contrast", "cohens_d", "CI_low", "CI_high")
d_table[, c("cohens_d", "CI_low", "CI_high")] <-
  round(d_table[, c("cohens_d", "CI_low", "CI_high")], 3)
cat("\n=== Cohen's d for pairwise group contrasts (95% CI) ===\n")
print(d_table, row.names = FALSE)
write.csv(d_table, file.path(folder_path, "effect_sizes_pairwise_cohensd.csv"),
          row.names = FALSE)
