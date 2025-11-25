# ==============================================================================
# Title:        Enhanced GWR by Time Points with CSV Results Export
# Description:  Runs GWR for each time point and exports detailed results
# ==============================================================================

# ==============================================================================
# SECTION 1: USER CONFIGURATION
# ==============================================================================
cat("Step 1: Loading configuration settings...\n")

CONFIG <- list(
  # --- File Paths ---
  input_csv_path = "./calib_diffdisp/20251118_GWR_Input_MLCW_InSAR_Layer_All.csv",
  output_directory = "GWR_Output_Layer_All",
  
  # --- Model Parameters ---
  formula_string = "Layer_All ~ DIFFDISP",
  
  # --- GWR Settings ---
  kernel_method = "gaussian",
  adaptive_bandwidth = TRUE,
  bandwidth_approach = "CV",
  distance_metric = 2,
  
  # --- Data Column Names ---
  time_field_name = "monthly",
  x_coord_name = "X_TWD97",
  y_coord_name = "Y_TWD97",
  
  # --- Metadata Preservation ---
  preserve_columns = c("STATION", "Layer_All", "DIFFDISP"),
  
  # --- Output Settings ---
  save_individual_models = TRUE,
  save_individual_csv = TRUE,        # NEW: Export CSV for each time point
  save_combined_csv = TRUE,          # NEW: Export single CSV with all time points
  save_summary_table = TRUE,
  verbose_output = TRUE
)

# ==============================================================================
# SECTION 2: SCRIPT INITIALIZATION
# ==============================================================================
cat("Step 2: Initializing environment and loading libraries...\n")

suppressPackageStartupMessages({
  library("GWmodel")
  library("sf")
})

if (!dir.exists(CONFIG$output_directory)) {
  dir.create(CONFIG$output_directory, recursive = TRUE)
  cat("   - Created output directory:", CONFIG$output_directory, "\n")
}

# ==============================================================================
# SECTION 3: DATA LOADING AND PREPARATION
# ==============================================================================
cat("Step 3: Loading and preparing data...\n")

input_data <- read.csv(CONFIG$input_csv_path, stringsAsFactors = FALSE)
cat("   - Successfully loaded dataset with dimensions:", dim(input_data)[1], "rows,", dim(input_data)[2], "columns.\n")

# Validate preserve_columns
available_preserve <- intersect(CONFIG$preserve_columns, names(input_data))
if (length(available_preserve) < length(CONFIG$preserve_columns)) {
  missing <- setdiff(CONFIG$preserve_columns, names(input_data))
  warning("   - Columns not found: ", paste(missing, collapse = ", "))
}
CONFIG$preserve_columns <- available_preserve
cat("   - Preserving columns:", paste(CONFIG$preserve_columns, collapse = ", "), "\n")

# Check required columns
required_cols <- c(CONFIG$time_field_name, CONFIG$x_coord_name, CONFIG$y_coord_name)
missing_cols <- required_cols[!required_cols %in% names(input_data)]
if (length(missing_cols) > 0) {
  stop("Missing required columns: ", paste(missing_cols, collapse = ", "))
}

# Get unique time points
unique_times <- sort(unique(input_data[[CONFIG$time_field_name]]))
cat("   - Found", length(unique_times), "unique time points\n")

# Prepare model formula
model_formula <- as.formula(CONFIG$formula_string)
dependent_var <- all.vars(model_formula)[1]
predictor_vars <- all.vars(model_formula)[-1]

# ==============================================================================
# SECTION 4: GWR ANALYSIS FOR EACH TIME POINT
# ==============================================================================
cat("\nStep 4: Running GWR analysis for each time point...\n")

# Enhanced results summary with both GWR and OLS metrics
results_summary <- data.frame(
  time_point = numeric(),
  n_observations = numeric(),
  
  # GWR-specific metrics
  gwr_optimal_bandwidth = numeric(),
  gwr_AICc = numeric(),
  gwr_R_squared = numeric(),
  gwr_adjusted_R_squared = numeric(),
  gwr_RSS = numeric(),
  
  # OLS (Global Linear Regression) metrics  
  ols_AIC = numeric(),
  ols_R_squared = numeric(),
  ols_adjusted_R_squared = numeric(),
  ols_RSS = numeric(),
  ols_F_statistic = numeric(),
  ols_F_pvalue = numeric(),
  
  # Model comparison metrics
  R_squared_improvement = numeric(),  # GWR R² - OLS R²
  model_file = character(),
  
  stringsAsFactors = FALSE
)

# Storage for combined results
all_time_results <- list()

# Main analysis loop
for (i in seq_along(unique_times)) {
  current_time <- unique_times[i]
  
  cat(sprintf("\n--- Processing Time Point %d/%d: %s ---\n", 
              i, length(unique_times), current_time))
  
  # Subset data
  time_subset <- input_data[input_data[[CONFIG$time_field_name]] == current_time, ]
  
  if (nrow(time_subset) < 10) {
    cat("   WARNING: Only", nrow(time_subset), "observations. Skipping.\n")
    next
  }
  
  cat("   - Subset contains", nrow(time_subset), "observations\n")
  
  # Preserve original metadata
  metadata_subset <- time_subset[, CONFIG$preserve_columns, drop = FALSE]
  
  # Create spatial data frame
  coords <- as.matrix(time_subset[, c(CONFIG$x_coord_name, CONFIG$y_coord_name)])
  spatial_data <- SpatialPointsDataFrame(
    coords = coords,
    data = time_subset,
    proj4string = CRS("EPSG:3826")
  )
  
  tryCatch({
    # Find optimal bandwidth
    cat("   - Finding optimal bandwidth...\n")
    bandwidth_start <- Sys.time()
    
    optimal_bw <- bw.gwr(
      formula = model_formula,
      data = spatial_data,
      approach = CONFIG$bandwidth_approach,
      kernel = CONFIG$kernel_method,
      adaptive = CONFIG$adaptive_bandwidth,
      p = CONFIG$distance_metric
    )
    
    bandwidth_time <- difftime(Sys.time(), bandwidth_start)
    cat("   - Optimal bandwidth found:", optimal_bw, 
        if(CONFIG$adaptive_bandwidth) "(neighbors)" else "(distance units)", 
        "in", format(bandwidth_time), "\n")
    
    # Run GWR model
    cat("   - Running GWR model...\n")
    model_start <- Sys.time()
    
    gwr_model <- gwr.basic(
      formula = model_formula,
      data = spatial_data,
      bw = optimal_bw,
      kernel = CONFIG$kernel_method,
      adaptive = CONFIG$adaptive_bandwidth,
      p = CONFIG$distance_metric
    )
    
    model_time <- difftime(Sys.time(), model_start)
    cat("   - Model completed in", format(model_time), "\n")
    
    # Extract diagnostics
    diagnostics <- gwr_model$GW.diagnostic
    if (is.null(diagnostics)) {
      cat("   WARNING: No diagnostic information available\n")
      AICc_val <- R2_val <- adj_R2_val <- RSS_val <- NA
    } else {
      AICc_val <- diagnostics$AICc
      R2_val <- diagnostics$gw.R2
      adj_R2_val <- diagnostics$gwR2.adj
      RSS_val <- diagnostics$RSS.gw
    }

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # Extract global model metrics from existing GWR object
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

    global_model <- gwr_model$lm
    global_summary <- summary(global_model)
    
    ols_R2 <- global_summary$r.squared
    ols_adj_R2 <- global_summary$adj.r.squared
    ols_AIC_val <- AIC(global_model)
    ols_RSS_val <- sum(global_model$residuals^2)
    ols_F_stat <- global_summary$fstatistic[1]
    ols_F_pval <- pf(global_summary$fstatistic[1], 
                     global_summary$fstatistic[2], 
                     global_summary$fstatistic[3], 
                     lower.tail = FALSE)
    
    # Calculate R² improvement
    R2_improvement <- if(!is.na(R2_val) && !is.na(ols_R2)) {
      R2_val - ols_R2
    } else {
      NA
    }
    
    # =================================================================
    # NEW: Extract and process results for CSV export
    # =================================================================
    cat("   - Extracting results for CSV export...\n")
    
    # Extract coefficients from SDF
    if (inherits(gwr_model$SDF, "sf")) {
      coefficient_data <- sf::st_drop_geometry(gwr_model$SDF)
    } else {
      coefficient_data <- as.data.frame(gwr_model$SDF)
    }
    
    # Prefix metadata columns with "input_"
    metadata_prefixed <- metadata_subset
    names(metadata_prefixed) <- paste0("input_", names(metadata_prefixed))
    
    # Combine: metadata + coordinates + coefficients
    result_data <- cbind(
      metadata_prefixed,
      time_subset[, c(CONFIG$x_coord_name, CONFIG$y_coord_name)],
      coefficient_data
    )
    
    # Add time value
    result_data$Time_value <- current_time
    
    # Calculate predicted values
    # predicted = Intercept + coef1*var1 + coef2*var2 + ...
    result_data$predicted_value <- result_data$Intercept
    
    for (pred_var in predictor_vars) {
      coef_col <- pred_var
      input_col <- paste0("input_", pred_var)
      
      if (coef_col %in% names(result_data) && input_col %in% names(result_data)) {
        result_data$predicted_value <- result_data$predicted_value + 
          (result_data[[coef_col]] * result_data[[input_col]])
      }
    }
    
    # Calculate prediction errors
    response_input_col <- paste0("input_", dependent_var)
    if (response_input_col %in% names(result_data)) {
      result_data$prediction_error <- result_data[[response_input_col]] - result_data$predicted_value
      result_data$absolute_error <- abs(result_data$prediction_error)
    }
    
    # Add bandwidth info
    result_data$bandwidth_used <- optimal_bw
    
    # Save individual CSV if requested
    if (CONFIG$save_individual_csv) {
      bw_str <- if(CONFIG$adaptive_bandwidth) {
        sprintf("bw%d", round(optimal_bw))
      } else {
        sprintf("bw%.0f", optimal_bw)
      }
      
      csv_filename <- sprintf("GWR_%s_%s_time%s_%s_results.csv",
                             dependent_var,
                             CONFIG$kernel_method,
                             current_time,
                             bw_str)
      
      csv_filepath <- file.path(CONFIG$output_directory, csv_filename)
      write.csv(result_data, csv_filepath, row.names = FALSE)
      cat("   - Results CSV saved:", csv_filename, "\n")
    }
    
    # Store for combined CSV
    all_time_results[[as.character(current_time)]] <- result_data
    
    # =================================================================
    # End of CSV export section
    # =================================================================
    
    # Generate output filename for model
    bw_str <- if(CONFIG$adaptive_bandwidth) {
      sprintf("bw%d", round(optimal_bw))
    } else {
      sprintf("bw%.0f", optimal_bw)
    }
    
    model_filename <- sprintf("GWR_%s_%s_time%s_%s.rds",
                             dependent_var,
                             CONFIG$kernel_method,
                             current_time,
                             bw_str)
    
    # Save model if requested
    if (CONFIG$save_individual_models) {
      model_filepath <- file.path(CONFIG$output_directory, model_filename)
      saveRDS(gwr_model, model_filepath)
      cat("   - Model saved to:", model_filename, "\n")
    }
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # Add to results summary with both GWR and OLS metrics
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    results_summary <- rbind(results_summary, data.frame(
      time_point = current_time,
      n_observations = nrow(time_subset),
      
      # GWR metrics
      gwr_optimal_bandwidth = optimal_bw,
      gwr_AICc = AICc_val,
      gwr_R_squared = R2_val,
      gwr_adjusted_R_squared = adj_R2_val,
      gwr_RSS = RSS_val,
      
      # OLS metrics
      ols_AIC = ols_AIC_val,
      ols_R_squared = ols_R2,
      ols_adjusted_R_squared = ols_adj_R2,
      ols_RSS = ols_RSS_val,
      ols_F_statistic = ols_F_stat,
      ols_F_pvalue = ols_F_pval,
      
      # Comparison
      R_squared_improvement = R2_improvement,
      model_file = model_filename,
      
      stringsAsFactors = FALSE
    ))
    
    # Print summary statistics
    if (CONFIG$verbose_output && !is.null(diagnostics)) {
      cat("   - Model diagnostics:\n")
      cat("     * AICc:", round(AICc_val, 3), "\n")
      cat("     * R-squared:", round(R2_val, 3), "\n")
      cat("     * Adjusted R-squared:", round(adj_R2_val, 3), "\n")
    }
    
  }, error = function(e) {
    cat("   ERROR in analysis for time", current_time, ":", e$message, "\n")
  })
}

# ==============================================================================
# SECTION 5: SAVE SUMMARY RESULTS
# ==============================================================================
cat("\nStep 5: Saving summary results...\n")

if (nrow(results_summary) > 0) {
  # Save summary table
  if (CONFIG$save_summary_table) {
    summary_filename <- sprintf("GWR_%s_%s_summary.csv", 
                               dependent_var, CONFIG$kernel_method)
    summary_filepath <- file.path(CONFIG$output_directory, summary_filename)
    write.csv(results_summary, summary_filepath, row.names = FALSE)
    cat("   - Summary table saved to:", summary_filename, "\n")
  }
  
  # NEW: Save combined CSV with all time points
  if (CONFIG$save_combined_csv && length(all_time_results) > 0) {
    cat("\nStep 5b: Creating combined CSV with all time points...\n")
    
    combined_data <- do.call(rbind, all_time_results)
    
    combined_filename <- sprintf("GWR_%s_%s_combined_results.csv",
                                dependent_var, CONFIG$kernel_method)
    combined_filepath <- file.path(CONFIG$output_directory, combined_filename)
    write.csv(combined_data, combined_filepath, row.names = FALSE)
    
    cat("   - Combined CSV saved:", combined_filename, "\n")
    cat("   - Total rows:", nrow(combined_data), "\n")
    cat("   - Total columns:", ncol(combined_data), "\n")
  }
  
  # # # # # # # # # # # # # # # # # # # # # # # # # # 
  # Print final summary
  # # # # # # # # # # # # # # # # # # # # # # # # # # 
  cat("\n--- ANALYSIS COMPLETE ---\n")
  cat("Successfully processed", nrow(results_summary), "time points\n")
  cat("Average bandwidth:", round(mean(results_summary$gwr_optimal_bandwidth, na.rm = TRUE), 2), "\n")
  cat("Bandwidth range:", round(min(results_summary$gwr_optimal_bandwidth, na.rm = TRUE), 2), 
      "to", round(max(results_summary$gwr_optimal_bandwidth, na.rm = TRUE), 2), "\n")
  
  if (!all(is.na(results_summary$gwr_R_squared))) {
    cat("Average GWR R-squared:", round(mean(results_summary$gwr_R_squared, na.rm = TRUE), 3), "\n")
    cat("Average OLS R-squared:", round(mean(results_summary$ols_R_squared, na.rm = TRUE), 3), "\n")
    cat("Average R² improvement:", round(mean(results_summary$R_squared_improvement, na.rm = TRUE), 3), "\n")
  }
  
} else {
  cat("   WARNING: No models were successfully completed.\n")
}

cat("\nAll results saved in directory:", CONFIG$output_directory, "\n")
cat("--- Script finished successfully ---\n")