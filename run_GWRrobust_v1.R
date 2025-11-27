# ==============================================================================
# Title:        Robust GWR by Time Points with Enhanced Diagnostics
# Description:  Runs Robust GWR for each time point with comprehensive outputs
# Author:       Enhanced GWR Analysis Pipeline
# Date:         2024-11-27
# ==============================================================================

# ==============================================================================
# SECTION 1: USER CONFIGURATION
# ==============================================================================
cat("Step 1: Loading configuration settings...\n")

CONFIG <- list(
  # --- File Paths ---
  input_csv_path = "./calib_diffdisp/20251124_GWR_Input_MLCW_InSAR_Layer_4.csv",
  output_directory = "RobustGWR_Output_Layer_4",
  
  # --- Model Parameters ---
  formula_string = "Layer_4 ~ DIFFDISP",
  
  # --- GWR Settings ---
  kernel_method = "bisquare",        # bisquare, tricube, gaussian, exponential
  adaptive_bandwidth = TRUE,
  bandwidth_approach = "AICc",
  distance_metric = 2,
  
  # --- Robust GWR Specific Settings ---
  robust_settings = list(
    max_iterations = 200,              # Maximum iterations for robust convergence
    convergence_threshold = 1.0e-10,   # Convergence criterion (delta)
    weight_cutoff_1 = 2,              # First cutoff for automatic weighting
    weight_cutoff_2 = 3,              # Second cutoff for automatic weighting
    export_outlier_reports = TRUE,    # Generate detailed outlier analysis
    export_weights = TRUE,            # Export observation weights
    robust_method = "automatic"       # "automatic" or "filtered"
  ),
  
  # --- Parallel Processing Settings ---
  parallel_settings = list(
    use_parallel = TRUE,              # Enable parallel processing
    n_cores = NULL,                   # NULL = auto-detect cores - 1
    parallel_method = "mclapply"      # Parallel method for bandwidth optimization
  ),
  
  # --- Data Column Names ---
  time_field_name = "monthly",
  x_coord_name = "X_TWD97",
  y_coord_name = "Y_TWD97",
  
  # --- Metadata Preservation ---
  preserve_columns = c("STATION", "Layer_4", "DIFFDISP"),
  
  # --- Output Settings ---
  save_individual_models = TRUE,
  save_individual_csv = TRUE,
  save_combined_csv = TRUE,
  save_summary_table = TRUE,
  verbose_output = TRUE,
  
  # --- Weights Export Settings ---
  weights_export = list(
    enabled = TRUE,
    folder_name = "weights",          # Base weights folder
    timestamp_format = "%Y%m%d_%H%M%S",
    include_outlier_maps = TRUE,
    weight_threshold = 0.1            # Threshold for identifying outliers
  )
)

# ==============================================================================
# SECTION 2: SCRIPT INITIALIZATION
# ==============================================================================
cat("Step 2: Initializing environment and loading libraries...\n")

suppressPackageStartupMessages({
  library("GWmodel")
  library("sf")
  library("parallel")
  library("doParallel")
})

# Setup parallel processing
if (CONFIG$parallel_settings$use_parallel) {
  if (is.null(CONFIG$parallel_settings$n_cores)) {
    n_cores <- max(1, detectCores() - 1)
  } else {
    n_cores <- CONFIG$parallel_settings$n_cores
  }
  cat("   - Parallel processing enabled with", n_cores, "cores\n")
  
  # Register parallel backend
  cl <- makeCluster(n_cores)
  registerDoParallel(cl)
} else {
  n_cores <- 1
  cat("   - Sequential processing mode\n")
}

# Create output directory structure
if (!dir.exists(CONFIG$output_directory)) {
  dir.create(CONFIG$output_directory, recursive = TRUE)
  cat("   - Created output directory:", CONFIG$output_directory, "\n")
}

# Create weights directory if weights export is enabled
if (CONFIG$robust_settings$export_weights && CONFIG$weights_export$enabled) {
  weights_base_dir <- file.path(CONFIG$output_directory, CONFIG$weights_export$folder_name)
  if (!dir.exists(weights_base_dir)) {
    dir.create(weights_base_dir, recursive = TRUE)
    cat("   - Created weights directory:", weights_base_dir, "\n")
  }
  
  # Create single timestamped folder for this entire run
  run_timestamp <- format(Sys.time(), CONFIG$weights_export$timestamp_format)
  layer_name <- all.vars(as.formula(CONFIG$formula_string))[1]
  weights_folder_name <- paste0("weights_", layer_name, "_", run_timestamp)
  weights_folder_path <- file.path(weights_base_dir, weights_folder_name)
  dir.create(weights_folder_path, recursive = TRUE)
  cat("   - Created run-specific weights folder:", weights_folder_name, "\n")
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
# unique_times <- c(1,2,3)
cat("   - Found", length(unique_times), "unique time points\n")

# Prepare model formula
model_formula <- as.formula(CONFIG$formula_string)
dependent_var <- all.vars(model_formula)[1]
predictor_vars <- all.vars(model_formula)[-1]

# ==============================================================================
# SECTION 4: ROBUST GWR ANALYSIS FOR EACH TIME POINT
# ==============================================================================
cat("\nStep 4: Running Robust GWR analysis for each time point...\n")

# Enhanced results summary
results_summary <- data.frame(
  time_point = numeric(),
  n_observations = numeric(),
  
  # Robust GWR-specific metrics
  robust_gwr_optimal_bandwidth = numeric(),
  n_outliers_identified = numeric(),
  outlier_percentage = numeric(),
  min_observation_weight = numeric(),
  mean_observation_weight = numeric(),
  
  # Model performance metrics
  gwr_AICc = numeric(),
  gwr_R_squared = numeric(),
  gwr_adjusted_R_squared = numeric(),
  gwr_RSS = numeric(),
  
  # OLS comparison metrics  
  ols_AIC = numeric(),
  ols_R_squared = numeric(),
  ols_F_pvalue = numeric(),
  
  # Model comparison
  R_squared_improvement = numeric(),
  AICc_improvement = numeric(),
  model_file = character(),
  weights_folder = character(),
  
  stringsAsFactors = FALSE
)

# Storage for combined results
all_time_results <- list()

# Helper function for progress tracking
progress_bar <- function(current, total, width = 50) {
  percent <- current / total
  filled <- round(width * percent)
  bar <- paste0(rep("=", filled), collapse = "")
  spaces <- paste0(rep(" ", width - filled), collapse = "")
  cat(sprintf("\r[%s%s] %d%% (%d/%d)", bar, spaces, round(percent * 100), current, total))
  flush.console()
}

# Main analysis loop with progress tracking
cat("Processing", length(unique_times), "time points with Robust GWR...\n")

for (i in seq_along(unique_times)) {
  current_time <- unique_times[i]
  
  # Update progress
  progress_bar(i, length(unique_times))
  
  cat(sprintf("\n\n--- Processing Time Point %d/%d: %s ---\n", 
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
    # Find optimal bandwidth (with potential parallel processing)
    cat("   - Finding optimal bandwidth for robust GWR...\n")
    bandwidth_start <- Sys.time()
    
    optimal_bw <- bw.gwr(
      formula = model_formula,
      data = spatial_data,
      approach = CONFIG$bandwidth_approach,
      kernel = CONFIG$kernel_method,
      adaptive = CONFIG$adaptive_bandwidth,
      p = CONFIG$distance_metric,
      parallel.method = if(CONFIG$parallel_settings$use_parallel) CONFIG$parallel_settings$parallel_method else FALSE,
      parallel.arg = if(CONFIG$parallel_settings$use_parallel) list(cl = cl) else NULL
    )
    
    bandwidth_time <- difftime(Sys.time(), bandwidth_start)
    cat("   - Optimal bandwidth found:", optimal_bw, 
        if(CONFIG$adaptive_bandwidth) "(neighbors)" else "(distance units)", 
        "in", format(bandwidth_time), "\n")
    
    # Run Robust GWR model
    cat("   - Running Robust GWR model...\n")
    model_start <- Sys.time()
    
    robust_gwr_model <- gwr.robust(
      formula = model_formula,
      data = spatial_data,
      bw = optimal_bw,
      kernel = CONFIG$kernel_method,
      adaptive = CONFIG$adaptive_bandwidth,
      p = CONFIG$distance_metric,
      maxiter = CONFIG$robust_settings$max_iterations,
      delta = CONFIG$robust_settings$convergence_threshold,
      cut1 = CONFIG$robust_settings$weight_cutoff_1,
      cut2 = CONFIG$robust_settings$weight_cutoff_2,
      filtered = (CONFIG$robust_settings$robust_method == "filtered"),
      parallel.method = if(CONFIG$parallel_settings$use_parallel) CONFIG$parallel_settings$parallel_method else FALSE,
      parallel.arg = if(CONFIG$parallel_settings$use_parallel) list(cl = cl) else NULL
    )
    
    model_time <- difftime(Sys.time(), model_start)
    cat("   - Robust GWR completed in", format(model_time), "\n")
    
    # Extract diagnostics
    diagnostics <- robust_gwr_model$GW.diagnostic
    if (is.null(diagnostics)) {
      cat("   WARNING: No diagnostic information available\n")
      AICc_val <- R2_val <- adj_R2_val <- RSS_val <- NA
    } else {
      AICc_val <- diagnostics$AICc
      R2_val <- diagnostics$gw.R2
      adj_R2_val <- diagnostics$gwR2.adj
      RSS_val <- diagnostics$RSS.gw
    }
    
    # Extract global model metrics from existing GWR object
    global_model <- robust_gwr_model$lm
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
    
    # Calculate AICc improvement
    AICc_improvement <- if(!is.na(AICc_val) && !is.na(ols_AIC_val)) {
      ols_AIC_val - AICc_val  # Positive = robust GWR is better
    } else {
      NA
    }

    # Calculate R² improvement  
    R2_improvement <- if(!is.na(R2_val) && !is.na(ols_R2)) {
      R2_val - ols_R2
    } else {
      NA
    }
    
    # ==============================================================================
    # SECTION 5: WEIGHTS AND OUTLIER ANALYSIS
    # ==============================================================================
    
    n_outliers <- 0
    outlier_percentage <- 0
    min_weight <- 1.0
    mean_weight <- 1.0
    
    if (CONFIG$robust_settings$export_weights && CONFIG$weights_export$enabled) {
      cat("   - Exporting observation weights and outlier analysis...\n")
      
      # Extract observation weights (if available)
      if (!is.null(robust_gwr_model$SDF)) {
        # Get the SDF data
        if (inherits(robust_gwr_model$SDF, "sf")) {
          sdf_data <- sf::st_drop_geometry(robust_gwr_model$SDF)
          sdf_coords <- st_coordinates(robust_gwr_model$SDF)
        } else {
          sdf_data <- as.data.frame(robust_gwr_model$SDF)
          sdf_coords <- coordinates(robust_gwr_model$SDF)
        }
        
        # Extract actual weights from robust GWR output
        actual_weights <- if("E_weigts" %in% names(sdf_data)) sdf_data$E_weigts else rep(1.0, nrow(sdf_data))
        
        # Calculate robust-specific metrics
        min_weight <- min(actual_weights, na.rm = TRUE)
        mean_weight <- mean(actual_weights, na.rm = TRUE)
        n_outliers <- sum(actual_weights < CONFIG$weights_export$weight_threshold, na.rm = TRUE)
        outlier_percentage <- round(100 * n_outliers / length(actual_weights), 2)
        
        # Create weights dataframe
        weights_data <- data.frame(
          time_point = current_time,
          x_coord = sdf_coords[, 1],
          y_coord = sdf_coords[, 2],
          residual = sdf_data$residual,
          std_residual = sdf_data$Stud_residual,
          observation_weight = actual_weights,
          is_outlier = actual_weights < CONFIG$weights_export$weight_threshold,
          stringsAsFactors = FALSE
        )
        
        # Add original metadata
        weights_data <- cbind(weights_data, metadata_subset)
        
        # Save weights data
        weights_filename <- paste0("observation_weights_time", current_time, ".csv")
        weights_filepath <- file.path(weights_folder_path, weights_filename)
        write.csv(weights_data, weights_filepath, row.names = FALSE)
        
        # Generate outlier report if requested
        if (CONFIG$robust_settings$export_outlier_reports && n_outliers > 0) {
          outliers_data <- weights_data[weights_data$is_outlier, ]
          
          # Save outlier summary
          outlier_filename <- paste0("outlier_report_time", current_time, ".txt")
          outlier_filepath <- file.path(weights_folder_path, outlier_filename)
          
          writeLines(c(
            paste("Robust GWR Outlier Analysis Report"),
            paste("Time Point:", current_time),
            paste("Total Observations:", nrow(weights_data)),
            paste("Outliers Identified:", n_outliers),
            paste("Outlier Percentage:", outlier_percentage, "%"),
            paste("Weight Threshold:", CONFIG$weights_export$weight_threshold),
            paste(""),
            paste("Weight Statistics:"),
            paste("  Minimum Weight:", round(min_weight, 4)),
            paste("  Mean Weight:", round(mean_weight, 4)),
            paste("  Weight Range:", round(max(actual_weights) - min_weight, 4)),
            paste(""),
            paste("Outlier Statistics:"),
            paste("  Mean |Std. Residual|:", round(mean(abs(outliers_data$std_residual), na.rm = TRUE), 4)),
            paste("  Max |Std. Residual|:", round(max(abs(outliers_data$std_residual), na.rm = TRUE), 4))
          ), outlier_filepath)
          
          # Save detailed outlier data
          outlier_detail_filename <- paste0("outlier_details_time", current_time, ".csv")
          outlier_detail_filepath <- file.path(weights_folder_path, outlier_detail_filename)
          write.csv(outliers_data, outlier_detail_filepath, row.names = FALSE)
        }
        
        cat(sprintf("   - Weights: min=%.3f, mean=%.3f, outliers=%d (%.1f%%)\n", 
                   min_weight, mean_weight, n_outliers, outlier_percentage))
      }
    }
    
    # ==============================================================================
    # SECTION 6: CSV EXPORT PROCESSING
    # ==============================================================================
    
    cat("   - Extracting results for CSV export...\n")
    
    # Extract coefficients from SDF
    if (inherits(robust_gwr_model$SDF, "sf")) {
      coefficient_data <- sf::st_drop_geometry(robust_gwr_model$SDF)
    } else {
      coefficient_data <- as.data.frame(robust_gwr_model$SDF)
    }
    
    # Prefix metadata columns with "input_"
    metadata_prefixed <- metadata_subset
    names(metadata_prefixed) <- paste0("input_", names(metadata_prefixed))
    
    # Remove problematic columns from coefficient_data
    cols_to_remove <- c("CV_Score")
    coefficient_data_clean <- coefficient_data[, !names(coefficient_data) %in% cols_to_remove, drop = FALSE]
    
    # Combine: metadata + coefficients
    result_data <- cbind(
      metadata_prefixed,
      coefficient_data_clean
    )
    
    # Add time and bandwidth info (remove useless robust metrics)
    result_data$Time_value <- current_time
    result_data$bandwidth_used <- optimal_bw
    
    # Save individual CSV if requested
    if (CONFIG$save_individual_csv) {
      bw_str <- if(CONFIG$adaptive_bandwidth) {
        sprintf("bw%d", round(optimal_bw))
      } else {
        sprintf("bw%.0f", optimal_bw)
      }
      
      csv_filename <- sprintf("RobustGWR_%s_time%03d_%s_%s_results.csv",
                             dependent_var,
                             current_time,
                             CONFIG$kernel_method,
                             bw_str)
      
      csv_filepath <- file.path(CONFIG$output_directory, csv_filename)
      write.csv(result_data, csv_filepath, row.names = FALSE)
      cat("   - Results CSV saved:", csv_filename, "\n")
    }
    
    # Store for combined CSV
    all_time_results[[as.character(current_time)]] <- result_data
    
    # Generate output filename for model
    bw_str <- if(CONFIG$adaptive_bandwidth) {
      sprintf("bw%d", round(optimal_bw))
    } else {
      sprintf("bw%.0f", optimal_bw)
    }
    
    model_filename <- sprintf("RobustGWR_%s_time%03d_%s_%s.rds",
                             dependent_var,
                             current_time,
                             CONFIG$kernel_method,
                             bw_str)
    
    # Save model if requested
    if (CONFIG$save_individual_models) {
      model_filepath <- file.path(CONFIG$output_directory, model_filename)
      saveRDS(robust_gwr_model, model_filepath)
      cat("   - Model saved to:", model_filename, "\n")
    }
    
    # Add to results summary
    new_row <- data.frame(
      time_point = current_time,
      n_observations = nrow(time_subset),
      
      robust_gwr_optimal_bandwidth = optimal_bw,
      n_outliers_identified = n_outliers,
      outlier_percentage = outlier_percentage,
      min_observation_weight = min_weight,
      mean_observation_weight = mean_weight,
      
      gwr_AICc = AICc_val,
      gwr_R_squared = R2_val,
      gwr_adjusted_R_squared = adj_R2_val,
      gwr_RSS = RSS_val,
      
      ols_AIC = ols_AIC_val,
      ols_R_squared = ols_R2,
      ols_F_pvalue = ols_F_pval,
      
      R_squared_improvement = R2_improvement,
      AICc_improvement = AICc_improvement,
      model_file = model_filename,
      weights_folder = weights_folder_name,
      
      stringsAsFactors = FALSE
    )
    
    results_summary <- rbind(results_summary, new_row)
    
    cat("   - Time point", current_time, "completed successfully\n")
    
  }, error = function(e) {
    cat("   ERROR processing time point", current_time, ":", e$message, "\n")
  })
}

# Complete progress bar
cat("\n\nRobust GWR analysis completed!\n")

# ==============================================================================
# SECTION 7: EXPORT COMBINED RESULTS
# ==============================================================================

if (CONFIG$save_combined_csv && length(all_time_results) > 0) {
  cat("\nStep 5: Exporting combined results...\n")
  
  combined_results <- do.call(rbind, all_time_results)
  
  combined_filename <- sprintf("RobustGWR_%s_%s_AllTimes_Combined.csv",
                              dependent_var,
                              CONFIG$kernel_method)
  
  combined_filepath <- file.path(CONFIG$output_directory, combined_filename)
  write.csv(combined_results, combined_filepath, row.names = FALSE)
  cat("   - Combined results CSV saved:", combined_filename, 
      "(", nrow(combined_results), "rows )\n")
}

# Save results summary
if (CONFIG$save_summary_table && nrow(results_summary) > 0) {
  summary_filename <- sprintf("RobustGWR_%s_%s_Summary.csv",
                             dependent_var,
                             CONFIG$kernel_method)
  
  summary_filepath <- file.path(CONFIG$output_directory, summary_filename)
  write.csv(results_summary, summary_filepath, row.names = FALSE)
  cat("   - Model summary saved:", summary_filename, "\n")
}

# ==============================================================================
# SECTION 8: FINAL SUMMARY AND CLEANUP
# ==============================================================================

cat("\n" , rep("=", 80), "\n")
cat("ROBUST GWR ANALYSIS COMPLETE\n")
cat(rep("=", 80), "\n")

if (nrow(results_summary) > 0) {
  cat("Processed time points:", nrow(results_summary), "\n")
  cat("Average R² improvement:", round(mean(results_summary$R_squared_improvement, na.rm = TRUE), 4), "\n")
  cat("Average outliers per time point:", round(mean(results_summary$n_outliers_identified, na.rm = TRUE), 1), "\n")
  # cat("Robust convergence rate:", round(100 * sum(results_summary$robust_converged, na.rm = TRUE) / nrow(results_summary), 1), "%\n")
}

cat("Output directory:", CONFIG$output_directory, "\n")

if (CONFIG$robust_settings$export_weights && CONFIG$weights_export$enabled) {
  weights_dir <- file.path(CONFIG$output_directory, CONFIG$weights_export$folder_name)
  cat("Weights directory:", weights_dir, "\n")
}

# Cleanup parallel resources
if (CONFIG$parallel_settings$use_parallel) {
  stopCluster(cl)
  registerDoSEQ()
  cat("Parallel cluster stopped.\n")
}

cat("Analysis completed at:", as.character(Sys.time()), "\n")
cat(rep("=", 80), "\n")