options(survey.lonely.psu = "adjust")
suppressPackageStartupMessages(library(survey))

args_all <- commandArgs(trailingOnly = FALSE)
file_arg <- args_all[grepl("^--file=", args_all)]
this_file <- if (length(file_arg)) normalizePath(sub("^--file=", "", file_arg[1]), winslash = "/", mustWork = TRUE) else normalizePath(".", winslash = "/", mustWork = FALSE)
project <- normalizePath(file.path(dirname(this_file), ".."), winslash = "/", mustWork = FALSE)
processed <- file.path(project, "external_public_data", "processed")
out_dir <- file.path(project, "outputs")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

read_processed <- function(name) {
  read.csv(file.path(processed, name), stringsAsFactors = FALSE, check.names = FALSE)
}

as_num <- function(x) suppressWarnings(as.numeric(x))

prep_binary <- function(df, vars) {
  for (v in vars) {
    if (v %in% names(df)) df[[v]] <- as_num(df[[v]])
  }
  df
}

domain_count <- function(df, domain_var, metric_var, type) {
  idx <- df[[domain_var]] == 1 & !is.na(df[[metric_var]])
  n <- sum(idx, na.rm = TRUE)
  events <- NA_integer_
  if (type == "proportion") {
    events <- sum(df[[metric_var]][idx] == 1, na.rm = TRUE)
  }
  c(n = n, events = events)
}

quality_flag <- function(n, events, estimate, se, type) {
  if (is.na(estimate) || is.na(se)) return("not_estimable")
  if (type == "proportion") {
    if (is.na(events) || n < 30 || events < 10) return("suppress")
  } else {
    if (n < 30) return("suppress")
  }
  rse <- ifelse(abs(estimate) > 0, abs(se / estimate), NA_real_)
  if (!is.na(rse) && rse > 0.5) return("suppress")
  if (n < 50 || (!is.na(rse) && rse > 0.3)) return("unstable")
  "stable"
}

estimate_metric <- function(design, df, dataset, cycle, domain_var, domain_label, metric_var, metric_label, type, weight_type) {
  counts <- domain_count(df, domain_var, metric_var, type)
  n <- as.integer(counts[["n"]])
  events <- as.integer(counts[["events"]])
  sub_design <- subset(design, get(domain_var) == 1)
  estimate <- se <- ci_low <- ci_high <- NA_real_
  method_status <- "ok"

  result <- tryCatch({
    if (type == "proportion") {
      est <- svyciprop(as.formula(paste0("~", metric_var)), sub_design, method = "beta", na.rm = TRUE)
      estimate <- as.numeric(coef(est)[1])
      se <- as.numeric(SE(est)[1])
      ci <- suppressWarnings(confint(est))
      ci_low <- as.numeric(ci[1, 1])
      ci_high <- as.numeric(ci[1, 2])
    } else {
      est <- svymean(as.formula(paste0("~", metric_var)), sub_design, na.rm = TRUE)
      estimate <- as.numeric(coef(est)[1])
      se <- as.numeric(SE(est)[1])
      ci_low <- estimate - 1.96 * se
      ci_high <- estimate + 1.96 * se
    }
    NULL
  }, error = function(e) {
    method_status <<- paste0("error: ", conditionMessage(e))
    NULL
  })

  flag <- quality_flag(n, events, estimate, se, type)
  data.frame(
    dataset = dataset,
    cycle = cycle,
    domain = domain_label,
    domain_var = domain_var,
    metric = metric_label,
    metric_var = metric_var,
    type = type,
    weight_type = weight_type,
    unweighted_n = n,
    unweighted_events = events,
    estimate = estimate,
    se = se,
    ci_low = ci_low,
    ci_high = ci_high,
    estimate_pct = ifelse(type == "proportion", estimate * 100, estimate),
    ci_low_pct = ifelse(type == "proportion", ci_low * 100, ci_low),
    ci_high_pct = ifelse(type == "proportion", ci_high * 100, ci_high),
    rse = ifelse(!is.na(estimate) && estimate != 0, abs(se / estimate), NA_real_),
    design_df = tryCatch(degf(sub_design), error = function(e) NA_real_),
    reliability_flag = flag,
    method_status = method_status,
    stringsAsFactors = FALSE
  )
}

run_metrics <- function(design, df, dataset, cycle, domains, metrics, weight_type) {
  rows <- list()
  i <- 1
  for (d in seq_len(nrow(domains))) {
    for (m in seq_len(nrow(metrics))) {
      if (!(metrics$metric_var[m] %in% names(df))) next
      rows[[i]] <- estimate_metric(
        design, df, dataset, cycle,
        domains$domain_var[d], domains$domain_label[d],
        metrics$metric_var[m], metrics$metric_label[m],
        metrics$type[m], weight_type
      )
      i <- i + 1
    }
  }
  do.call(rbind, rows)
}

nhis <- read_processed("nhis_2023_2024_analysis_ready.csv")
nhis$age <- as_num(nhis$age)
nhis$weight_adj <- as_num(nhis$weight) / 2
nhis$stratum_design <- interaction(nhis$year, nhis$stratum, drop = TRUE)
nhis$psu_design <- interaction(nhis$year, nhis$stratum, nhis$psu, drop = TRUE)
nhis <- prep_binary(nhis, c("lung_cancer_history", "any_cancer_history", "copd", "diabetes", "diabetes_medication", "hypertension", "coronary_heart_disease", "depression", "ever_smoked", "current_smoking"))
nhis$all_adults <- ifelse(!is.na(nhis$age) & nhis$age >= 18, 1, 0)
nhis$any_cancer_domain <- ifelse(nhis$all_adults == 1 & nhis$any_cancer_history == 1, 1, 0)
nhis$lung_cancer_domain <- ifelse(nhis$all_adults == 1 & nhis$lung_cancer_history == 1, 1, 0)
nhis$no_lung_cancer_domain <- ifelse(nhis$all_adults == 1 & nhis$lung_cancer_history == 0, 1, 0)
nhis_design <- svydesign(ids = ~psu_design, strata = ~stratum_design, weights = ~weight_adj, data = nhis, nest = TRUE)

nhis_domains <- data.frame(
  domain_var = c("all_adults", "any_cancer_domain", "lung_cancer_domain", "no_lung_cancer_domain"),
  domain_label = c("All adults", "Any cancer history", "Lung cancer history", "No lung cancer history"),
  stringsAsFactors = FALSE
)
nhis_metrics <- data.frame(
  metric_var = c("copd", "diabetes", "diabetes_medication", "hypertension", "coronary_heart_disease", "depression", "ever_smoked", "current_smoking"),
  metric_label = c("COPD", "Diabetes", "Diabetes medication", "Hypertension", "Coronary heart disease", "Depression", "Ever smoked", "Current smoking"),
  type = rep("proportion", 8),
  stringsAsFactors = FALSE
)
nhis_results <- run_metrics(nhis_design, nhis, "NHIS", "2023-2024 pooled", nhis_domains, nhis_metrics, "WTFA_A / 2")

meps <- read_processed("meps_2023_analysis_ready.csv")
meps$age <- as_num(meps$age)
meps$weight_adj <- as_num(meps$weight)
meps$stratum_design <- as.factor(meps$stratum)
meps$psu_design <- interaction(meps$stratum, meps$psu, drop = TRUE)
meps <- prep_binary(meps, c("lung_cancer_history", "any_cancer_history", "asthma", "emphysema", "diabetes", "hypertension", "coronary_heart_disease", "condition_lung_cancer_icd", "condition_copd_icd", "condition_diabetes_icd", "condition_cvd_icd", "condition_ckd_icd", "condition_mental_icd"))
meps$all_adults <- ifelse(!is.na(meps$age) & meps$age >= 18, 1, 0)
meps$any_cancer_domain <- ifelse(meps$all_adults == 1 & meps$any_cancer_history == 1, 1, 0)
meps$lung_cancer_domain <- ifelse(meps$all_adults == 1 & meps$lung_cancer_history == 1, 1, 0)
meps$no_lung_cancer_domain <- ifelse(meps$all_adults == 1 & meps$lung_cancer_history == 0, 1, 0)
meps_design <- svydesign(ids = ~psu_design, strata = ~stratum_design, weights = ~weight_adj, data = meps, nest = TRUE)

meps_domains <- data.frame(
  domain_var = c("all_adults", "any_cancer_domain", "lung_cancer_domain", "no_lung_cancer_domain"),
  domain_label = c("All adults", "Any cancer history", "Lung cancer history", "No lung cancer history"),
  stringsAsFactors = FALSE
)
meps_metrics <- data.frame(
  metric_var = c("condition_copd_icd", "diabetes", "condition_diabetes_icd", "hypertension", "coronary_heart_disease", "condition_cvd_icd", "condition_ckd_icd", "condition_mental_icd", "total_expenditure", "office_visits", "outpatient_visits", "er_visits", "inpatient_discharges", "rx_fills"),
  metric_label = c("COPD ICD-10 condition cluster", "Diabetes", "Diabetes ICD-10 condition cluster", "Hypertension", "Coronary heart disease", "Cardiovascular ICD-10 condition cluster", "CKD ICD-10 condition cluster", "Mental health ICD-10 condition cluster", "Total annual expenditure", "Office-based visits", "Outpatient visits", "Emergency room visits", "Inpatient discharges", "Prescription fills"),
  type = c(rep("proportion", 8), rep("mean", 6)),
  stringsAsFactors = FALSE
)
meps_results <- run_metrics(meps_design, meps, "MEPS", "2023", meps_domains, meps_metrics, "PERWT23F")

nhanes <- read_processed("nhanes_2017_2023_analysis_ready.csv")
nhanes$age <- as_num(nhanes$age)
nhanes <- prep_binary(nhanes, c("any_cancer_history", "asthma", "copd_or_emphysema", "heart_failure", "coronary_heart_disease", "angina", "myocardial_infarction", "stroke", "hypertension", "diabetes", "kidney_disease", "ever_smoked", "current_smoking"))
nhanes$all_adults20 <- ifelse(!is.na(nhanes$age) & nhanes$age >= 20, 1, 0)
nhanes$any_cancer_domain <- ifelse(nhanes$all_adults20 == 1 & nhanes$any_cancer_history == 1, 1, 0)

nhanes_domains <- data.frame(
  domain_var = c("all_adults20", "any_cancer_domain"),
  domain_label = c("All adults 20+", "Any cancer history"),
  stringsAsFactors = FALSE
)
nhanes_questionnaire_metrics <- data.frame(
  metric_var = c("copd_or_emphysema", "diabetes", "hypertension", "coronary_heart_disease", "heart_failure", "myocardial_infarction", "stroke", "kidney_disease", "ever_smoked", "current_smoking", "phq9_score"),
  metric_label = c("COPD/emphysema/chronic bronchitis", "Diabetes", "Hypertension", "Coronary heart disease", "Heart failure", "Myocardial infarction", "Stroke", "Kidney disease", "Ever smoked", "Current smoking", "PHQ-9 score"),
  type = c(rep("proportion", 10), "mean"),
  stringsAsFactors = FALSE
)
nhanes_exam_metrics <- data.frame(
  metric_var = c("bmi", "hba1c", "serum_creatinine", "bicarbonate"),
  metric_label = c("BMI", "HbA1c", "Serum creatinine", "Bicarbonate"),
  type = rep("mean", 4),
  stringsAsFactors = FALSE
)

nhanes_results_list <- list()
k <- 1
for (cy in unique(nhanes$cycle)) {
  tmp <- nhanes[nhanes$cycle == cy, ]
  tmp$stratum_design <- as.factor(tmp$stratum)
  tmp$psu_design <- interaction(tmp$stratum, tmp$psu, drop = TRUE)
  q_design <- svydesign(ids = ~psu_design, strata = ~stratum_design, weights = ~interview_weight, data = tmp, nest = TRUE)
  nhanes_results_list[[k]] <- run_metrics(q_design, tmp, "NHANES", cy, nhanes_domains, nhanes_questionnaire_metrics, "WTINT2YR")
  k <- k + 1
  e_design <- svydesign(ids = ~psu_design, strata = ~stratum_design, weights = ~mec_weight, data = tmp, nest = TRUE)
  nhanes_results_list[[k]] <- run_metrics(e_design, tmp, "NHANES", cy, nhanes_domains, nhanes_exam_metrics, "WTMEC2YR")
  k <- k + 1
}
nhanes_results <- do.call(rbind, nhanes_results_list)

all_results <- rbind(nhis_results, meps_results, nhanes_results)
all_results$estimate_display <- ifelse(all_results$type == "proportion", all_results$estimate_pct, all_results$estimate)
all_results$ci_low_display <- ifelse(all_results$type == "proportion", all_results$ci_low_pct, all_results$ci_low)
all_results$ci_high_display <- ifelse(all_results$type == "proportion", all_results$ci_high_pct, all_results$ci_high)
all_results$display_unit <- ifelse(all_results$type == "proportion", "percent", "mean")

write.csv(all_results, file.path(out_dir, "P1_external_triangulation_weighted_estimates.csv"), row.names = FALSE)

qc <- data.frame(
  source = c("NHIS", "MEPS", "NHANES"),
  rows = c(nrow(nhis), nrow(meps), nrow(nhanes)),
  domains = c(
    paste(names(table(nhis$lung_cancer_domain)), table(nhis$lung_cancer_domain), collapse = "; "),
    paste(names(table(meps$lung_cancer_domain)), table(meps$lung_cancer_domain), collapse = "; "),
    paste(names(table(nhanes$any_cancer_domain)), table(nhanes$any_cancer_domain), collapse = "; ")
  ),
  stringsAsFactors = FALSE
)
write.csv(qc, file.path(out_dir, "P1_external_triangulation_survey_qc.csv"), row.names = FALSE)

cat("External survey triangulation complete\n")
cat(nrow(all_results), "estimates written\n")

