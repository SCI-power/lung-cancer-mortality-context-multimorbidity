options(survey.lonely.psu = "adjust")
suppressPackageStartupMessages(library(survey))

args_all <- commandArgs(trailingOnly = FALSE)
file_arg <- args_all[grepl("^--file=", args_all)]
this_file <- if (length(file_arg)) normalizePath(sub("^--file=", "", file_arg[1]), winslash = "/", mustWork = TRUE) else normalizePath(".", winslash = "/", mustWork = FALSE)
project <- normalizePath(file.path(dirname(this_file), ".."), winslash = "/", mustWork = FALSE)
processed <- file.path(project, "external_public_data", "processed")
out_dir <- file.path(project, "outputs")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

as_num <- function(x) suppressWarnings(as.numeric(x))

nhis <- read.csv(file.path(processed, "nhis_2023_2024_analysis_ready.csv"), stringsAsFactors = FALSE, check.names = FALSE)
for (v in c("age", "sex", "race", "hispanic_origin", "region", "weight", "stratum", "psu",
            "lung_cancer_history", "any_cancer_history", "copd", "diabetes", "hypertension",
            "coronary_heart_disease", "depression", "ever_smoked", "current_smoking")) {
  nhis[[v]] <- as_num(nhis[[v]])
}

nhis$weight_adj <- nhis$weight / 2
nhis$stratum_design <- interaction(nhis$year, nhis$stratum, drop = TRUE)
nhis$psu_design <- interaction(nhis$year, nhis$stratum, nhis$psu, drop = TRUE)

nhis$cancer_comparator_group <- NA_character_
nhis$cancer_comparator_group[nhis$any_cancer_history == 0 & nhis$lung_cancer_history == 0] <- "No cancer history"
nhis$cancer_comparator_group[nhis$any_cancer_history == 1 & nhis$lung_cancer_history == 0] <- "Other cancer history"
nhis$cancer_comparator_group[nhis$lung_cancer_history == 1] <- "Lung cancer history"
nhis$cancer_comparator_group <- factor(
  nhis$cancer_comparator_group,
  levels = c("No cancer history", "Other cancer history", "Lung cancer history")
)

nhis$age_group <- cut(
  nhis$age,
  breaks = c(17, 44, 54, 64, 74, Inf),
  labels = c("18-44", "45-54", "55-64", "65-74", "75+"),
  right = TRUE
)
nhis$sex_f <- factor(ifelse(nhis$sex == 1, "Male", ifelse(nhis$sex == 2, "Female", "Unknown")))
nhis$race_f <- factor(ifelse(is.na(nhis$race), "Unknown", paste0("Race recode ", nhis$race)))
nhis$hispanic_f <- factor(ifelse(is.na(nhis$hispanic_origin), "Unknown", paste0("Hispanic-origin recode ", nhis$hispanic_origin)))
nhis$region_f <- factor(ifelse(is.na(nhis$region), "Unknown", paste0("Region ", nhis$region)))
nhis$smoking_status <- "Unknown"
nhis$smoking_status[nhis$ever_smoked == 0] <- "Never"
nhis$smoking_status[nhis$ever_smoked == 1 & nhis$current_smoking == 0] <- "Former"
nhis$smoking_status[nhis$ever_smoked == 1 & nhis$current_smoking == 1] <- "Current"
nhis$smoking_status <- factor(nhis$smoking_status, levels = c("Never", "Former", "Current", "Unknown"))

analysis <- nhis[
  !is.na(nhis$cancer_comparator_group) &
    !is.na(nhis$age_group) &
    !is.na(nhis$weight_adj) &
    nhis$weight_adj > 0,
]

design <- svydesign(
  ids = ~psu_design,
  strata = ~stratum_design,
  weights = ~weight_adj,
  data = analysis,
  nest = TRUE
)

metrics <- data.frame(
  metric_var = c("copd", "coronary_heart_disease", "diabetes", "hypertension", "depression", "ever_smoked", "current_smoking"),
  metric_label = c("COPD", "Coronary heart disease", "Diabetes", "Hypertension", "Depression", "Ever smoked", "Current smoking"),
  model_class = c(rep("disease", 5), rep("smoking", 2)),
  stringsAsFactors = FALSE
)

quality_flag <- function(n, events, estimate, se) {
  if (is.na(estimate) || is.na(se)) return("not_estimable")
  if (is.na(events) || n < 30 || events < 10) return("suppress")
  rse <- ifelse(abs(estimate) > 0, abs(se / estimate), NA_real_)
  if (!is.na(rse) && rse > 0.5) return("suppress")
  if (n < 50 || (!is.na(rse) && rse > 0.3)) return("unstable")
  "stable"
}

estimate_prev <- function(group_label, metric_var, metric_label) {
  idx <- analysis$cancer_comparator_group == group_label & !is.na(analysis[[metric_var]])
  n <- sum(idx, na.rm = TRUE)
  events <- sum(analysis[[metric_var]][idx] == 1, na.rm = TRUE)
  sub_design <- subset(design, cancer_comparator_group == group_label)
  estimate <- se <- ci_low <- ci_high <- NA_real_
  status <- "ok"
  tryCatch({
    est <- svyciprop(as.formula(paste0("~I(", metric_var, " == 1)")), sub_design, method = "beta", na.rm = TRUE)
    estimate <- as.numeric(coef(est)[1])
    se <- as.numeric(SE(est)[1])
    ci <- suppressWarnings(confint(est))
    ci_low <- as.numeric(ci[1, 1])
    ci_high <- as.numeric(ci[1, 2])
  }, error = function(e) {
    status <<- paste0("error: ", conditionMessage(e))
  })
  data.frame(
    dataset = "NHIS",
    cycle = "2023-2024 pooled",
    group = group_label,
    metric = metric_label,
    metric_var = metric_var,
    unweighted_n = n,
    unweighted_events = events,
    estimate = estimate,
    se = se,
    ci_low = ci_low,
    ci_high = ci_high,
    prevalence_pct = estimate * 100,
    ci_low_pct = ci_low * 100,
    ci_high_pct = ci_high * 100,
    rse = ifelse(!is.na(estimate) && estimate != 0, abs(se / estimate), NA_real_),
    reliability_flag = quality_flag(n, events, estimate, se),
    method_status = status,
    stringsAsFactors = FALSE
  )
}

prevalence_rows <- list()
k <- 1
for (g in levels(analysis$cancer_comparator_group)) {
  for (i in seq_len(nrow(metrics))) {
    prevalence_rows[[k]] <- estimate_prev(g, metrics$metric_var[i], metrics$metric_label[i])
    k <- k + 1
  }
}
prevalence <- do.call(rbind, prevalence_rows)

count_events <- function(df, group_label, metric_var) {
  idx <- df$cancer_comparator_group == group_label & !is.na(df[[metric_var]])
  c(n = sum(idx, na.rm = TRUE), events = sum(df[[metric_var]][idx] == 1, na.rm = TRUE))
}

fit_pr <- function(metric_var, metric_label, model_class, reference_group) {
  sub <- analysis[analysis$cancer_comparator_group %in% c(reference_group, "Lung cancer history") & !is.na(analysis[[metric_var]]), ]
  sub$lung_history_exposure <- ifelse(sub$cancer_comparator_group == "Lung cancer history", 1, 0)
  sub$lung_history_exposure <- factor(sub$lung_history_exposure, levels = c(0, 1), labels = c("Reference", "Lung cancer history"))
  sub_design <- svydesign(
    ids = ~psu_design,
    strata = ~stratum_design,
    weights = ~weight_adj,
    data = sub,
    nest = TRUE
  )
  covars <- c("lung_history_exposure", "age_group", "sex_f", "race_f", "hispanic_f", "region_f")
  model_name <- "demographic-adjusted"
  if (model_class == "disease") {
    covars <- c(covars, "smoking_status")
    model_name <- "demographic-and-smoking-adjusted"
  }
  formula <- as.formula(paste(metric_var, "~", paste(covars, collapse = " + ")))

  pr <- lcl <- ucl <- pval <- NA_real_
  status <- "ok"
  n_ref <- events_ref <- n_lung <- events_lung <- NA_integer_
  counts_ref <- count_events(sub, reference_group, metric_var)
  counts_lung <- count_events(sub, "Lung cancer history", metric_var)
  n_ref <- as.integer(counts_ref[["n"]])
  events_ref <- as.integer(counts_ref[["events"]])
  n_lung <- as.integer(counts_lung[["n"]])
  events_lung <- as.integer(counts_lung[["events"]])

  tryCatch({
    mod <- svyglm(formula, design = sub_design, family = quasipoisson(link = "log"))
    coef_name <- grep("^lung_history_exposure", names(coef(mod)), value = TRUE)[1]
    if (is.na(coef_name)) {
      stop("lung_history_exposure coefficient not found")
    }
    b <- coef(mod)[coef_name]
    v <- vcov(mod)[coef_name, coef_name]
    se <- sqrt(v)
    pr <- exp(b)
    lcl <- exp(b - 1.96 * se)
    ucl <- exp(b + 1.96 * se)
    pval <- 2 * pnorm(-abs(b / se))
  }, error = function(e) {
    status <<- paste0("error: ", conditionMessage(e))
  })

  data.frame(
    dataset = "NHIS",
    cycle = "2023-2024 pooled",
    metric = metric_label,
    metric_var = metric_var,
    reference_group = reference_group,
    comparison = paste("Lung cancer history vs", reference_group),
    model = model_name,
    adjusted_pr = pr,
    ci_low = lcl,
    ci_high = ucl,
    p_value = pval,
    lung_unweighted_n = n_lung,
    lung_unweighted_events = events_lung,
    reference_unweighted_n = n_ref,
    reference_unweighted_events = events_ref,
    model_status = status,
    stringsAsFactors = FALSE
  )
}

pr_rows <- list()
k <- 1
for (i in seq_len(nrow(metrics))) {
  for (ref in c("No cancer history", "Other cancer history")) {
    pr_rows[[k]] <- fit_pr(metrics$metric_var[i], metrics$metric_label[i], metrics$model_class[i], ref)
    k <- k + 1
  }
}
adjusted_pr <- do.call(rbind, pr_rows)

prevalence_wide <- reshape(
  prevalence[, c("group", "metric", "prevalence_pct", "ci_low_pct", "ci_high_pct", "unweighted_n", "unweighted_events", "reliability_flag")],
  idvar = "metric",
  timevar = "group",
  direction = "wide"
)

write.csv(prevalence, file.path(out_dir, "P1_nhis_lung_vs_comparator_weighted_prevalence.csv"), row.names = FALSE)
write.csv(prevalence_wide, file.path(out_dir, "P1_nhis_lung_vs_comparator_weighted_prevalence_wide.csv"), row.names = FALSE)
write.csv(adjusted_pr, file.path(out_dir, "P1_nhis_lung_vs_comparator_adjusted_pr.csv"), row.names = FALSE)

qc <- data.frame(
  group = levels(analysis$cancer_comparator_group),
  unweighted_n = as.integer(table(analysis$cancer_comparator_group)[levels(analysis$cancer_comparator_group)]),
  weighted_population = as.numeric(svyby(~I(!is.na(age)), ~cancer_comparator_group, design, svytotal, na.rm = TRUE)[["I(!is.na(age))TRUE"]]),
  stringsAsFactors = FALSE
)
write.csv(qc, file.path(out_dir, "P1_nhis_lung_vs_comparator_qc.csv"), row.names = FALSE)

cat("NHIS comparator analysis complete\n")
cat(nrow(prevalence), "prevalence estimates written\n")
cat(nrow(adjusted_pr), "adjusted PR estimates written\n")

