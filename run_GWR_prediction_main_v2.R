# ==============================================================================
# Title:        GWR Prediction by Time Points (Flexible Parameters)
# Description:  Runs GWR predictions for each time point using time-specific
#               optimal parameters (bandwidth, kernel, adaptive)
#               read from a summary file.
# Author:       Generated for time-specific GWR prediction
# Date:         2025-01-02
# Modified:     2025-10-31
# ==============================================================================

# ==============================================================================
# SECTION 1: USER CONFIGURATION
# ==============================================================================
cat("Step 1: Loading configuration settings...\n")

CONFIG <- list(
  # --- File Paths ---
  calibration_data_path = "./calib_diffdisp/20251016_GTWR_InputData_DiffDisp_Layer_4.csv",
  prediction_data_path = "diffdisp_gridpnt.feather",
  
  # Path to the summary CSV file containing optimal parameters for each time point
  # This file MUST contain columns for time, bandwidth, kernel, and adaptive setting.
  parameters_summary_path = "GWR_AllKernel_Layer_4_model_info.csv",
  
  output_directory = "GWR_Prediction_Output",
  
  # --- Model Parameters ---
  formula_string = "Layer_4 ~ DIFFDISP",

  # --- GWR Parameter Column Names (from parameters_summary_path) ---
  # Specify the column name in your CSV that contains the bandwidth
  bandwidth_column_name = "optimal_bandwidth", 
  # Specify the column name that contains the kernel (e.g., "bisquare", "gaussian")
  kernel_column_name = "optimal_kernel",
  # Specify the column name that contains the adaptive setting (must be TRUE/FALSE)
  adaptive_column_name = "is_adaptive",
  
  # --- Data Column Names ---
  time_field_name = "monthly", # This must match the time column in parameters_summary_path
  x_coord_name = "X_TWD97",
  y_coord_name = "Y_TWD97",

  # --- Column Selection ---
  # Columns from prediction_data_path to keep in the final output CSVs
  columns_to_preserve = c("monthly", "datetime_str", "X_TWD97", "Y_TWD97", "DIFFDISP"),
  
  # --- Time Point Selection (optional) ---
  time_selection = list(
    use_selection = FALSE,    # Set to TRUE to process specific time points
    start_index = 1,          # Starting position in time point list
    end_index = 5             # Ending position in time point list
  ),
  
  # --- Output Settings ---
  save_individual_predictions = TRUE,
  save_summary_table = TRUE,
  verbose_output = TRUE
)

# ==============================================================================
# SECTION 2: SCRIPT INITIALIZATION
# ==============================================================================
cat("Step 2: Initializing environment and loading libraries...\n")

suppressPackageStartupMessages({
  library("GWmodel")
  library("sp")
  library("arrow")  # For read_feather()
  library("sf")     # For handling sf objects
})

# ==============================================================================
# SECTION 3: DATA LOADING AND PREPARATION
# ==============================================================================
cat("Step 3: Loading and preparing data...\n")

# Load calibration data
if (!file.exists(CONFIG$calibration_data_path)) {
  stop("Calibration data file not found: ", CONFIG$calibration_data_path)
}
cal_data <- read.csv(CONFIG$calibration_data_path)
cat("   - Loaded calibration dataset with:", nrow(cal_data), "rows\n")

# Load prediction data
if (!file.exists(CONFIG$prediction_data_path)) {
  stop("Prediction data file not found: ", CONFIG$prediction_data_path)
}
pred_data_full <- read_feather(CONFIG$prediction_data_path)
pred_data_full <- as.data.frame(pred_data_full)

# Select only specified columns
if (!is.null(CONFIG$columns_to_preserve)) {
  missing_cols <- setdiff(CONFIG$columns_to_preserve, colnames(pred_data_full))
  if (length(missing_cols) > 0) {
    stop("Columns not found in prediction data: ", paste(missing_cols, collapse = ", "))
  }
  pred_data <- pred_data_full[, CONFIG$columns_to_preserve, drop = FALSE]
  cat("   - Selected", length(CONFIG$columns_to_preserve), "columns:", 
      paste(CONFIG$columns_to_preserve, collapse = ", "), "\n")
} else {
  pred_data <- pred_data_full
}
cat("   - Loaded prediction dataset with:", nrow(pred_data), "total observations\n")

# Clean prediction data
pred_data <- pred_data[complete.cases(pred_data[, c(CONFIG$x_coord_name, CONFIG$y_coord_name)]), ]

# Load parameters summary
if (!file.exists(CONFIG$parameters_summary_path)) {
  stop("Parameters summary file not found: ", CONFIG$parameters_summary_path)
}
parameters_summary <- read.csv(CONFIG$parameters_summary_path)
cat("   - Loaded GWR parameters summary with", nrow(parameters_summary), "time points\n")

# Check if all required parameter columns exist
required_cols <- c(CONFIG$time_field_name, 
                   CONFIG$bandwidth_column_name, 
                   CONFIG$kernel_column_name, 
                   CONFIG$adaptive_column_name)
missing_param_cols <- setdiff(required_cols, colnames(parameters_summary))
if (length(missing_param_cols) > 0) {
  stop("Columns not found in parameters summary file: ", paste(missing_param_cols, collapse = ", "))
}

# Get unique time points and apply selection if needed
all_time_points <- sort(unique(cal_data[[CONFIG$time_field_name]]))
cat("   - Found", length(all_time_points), "total time points:", paste(all_time_points, collapse = ", "), "\n")

if (CONFIG$time_selection$use_selection) {
  start_idx <- CONFIG$time_selection$start_index
  end_idx <- CONFIG$time_selection$end_index
  
  if (start_idx < 1 || start_idx > length(all_time_points)) {
    stop("start_index out of range. Must be between 1 and ", length(all_time_points))
  }
  if (end_idx < start_idx || end_idx > length(all_time_points)) {
    stop("end_index invalid. Must be between ", start_idx, " and ", length(all_time_points))
  }
  
  selected_time_points <- all_time_points[start_idx:end_idx]
  cat("\n=== TIME POINT SELECTION ACTIVE ===\n")
  cat("   - SELECTED TIME STAMPS:", paste(selected_time_points, collapse = ", "), "\n")
  cat("===================================\n")
} else {
  selected_time_points <- all_time_points
  cat("   - PROCESSING ALL TIME POINTS\n")
}

# Prepare model formula
model_formula <- as.formula(CONFIG$formula_string)
dependent_var <- all.vars(model_formula)[1]

# Create subdirectory for this dependent variable
CONFIG$output_directory <- file.path(CONFIG$output_directory, dependent_var)
if (!dir.exists(CONFIG$output_directory)) {
  dir.create(CONFIG$output_directory, recursive = TRUE)
  cat("   - Created variable-specific output directory:", CONFIG$output_directory, "\n")
}

# ==============================================================================
# SECTION 4: GWR PREDICTION FOR EACH TIME POINT
# ==============================================================================
cat("\nStep 4: Running GWR predictions for each time point...\n")

# Initialize results storage
prediction_summary <- data.frame(
  time_point = numeric(),
  n_calibration_obs = numeric(),
  n_prediction_locations = numeric(),
  bandwidth_used = numeric(),
  kernel_used = character(),
  adaptive_used = logical(),
  prediction_file = character(),
  stringsAsFactors = FALSE
)

# Main prediction loop
for (i in seq_along(selected_time_points)) {
  current_time <- selected_time_points[i]
  
  cat(sprintf("\n--- Processing Time Point %d/%d: %s ---\n", 
              i, length(selected_time_points), current_time))
  
  # --- Get time-specific GWR parameters ---
  param_row <- parameters_summary[parameters_summary[[CONFIG$time_field_name]] == current_time, ]
  if (nrow(param_row) == 0) {
    cat("   WARNING: No GWR parameters found for time", current_time, ". Skipping.\n")
    next
  }
  
  # Extract parameters from the row
  current_bw <- param_row[[CONFIG$bandwidth_column_name]][1]
  current_kernel <- as.character(param_row[[CONFIG$kernel_column_name]][1])
  current_adaptive <- as.logical(param_row[[CONFIG$adaptive_column_name]][1])
  
  cat(sprintf("   - Using Parameters: BW = %s, Kernel = %s, Adaptive = %s\n",
              current_bw, current_kernel, current_adaptive))
  
  # Subset calibration data for current time point
  time_cal_data <- cal_data[cal_data[[CONFIG$time_field_name]] == current_time, ]
  
  if (nrow(time_cal_data) < 10) {
    cat("   WARNING: Only", nrow(time_cal_data), "calibration observations for time", current_time, ". Skipping.\n")
    next
  }
  
  # Subset prediction data for current time point
  time_pred_data <- pred_data[pred_data[[CONFIG$time_field_name]] == current_time, ]
  
  if (nrow(time_pred_data) == 0) {
    cat("   WARNING: No prediction locations for time", current_time, ". Skipping.\n")
    next
  }
  
  cat("   - Using", nrow(time_cal_data), "calibration observations\n")
  cat("   - Predicting for", nrow(time_pred_data), "locations\n")
  
  # Create spatial data objects
  cal_coords <- as.matrix(time_cal_data[, c(CONFIG$x_coord_name, CONFIG$y_coord_name)])
  cal_spatial_data <- SpatialPointsDataFrame(
    coords = cal_coords,
    data = time_cal_data,
    proj4string = CRS("EPSG:3826")
  )
  
  pred_coords_time <- as.matrix(time_pred_data[, c(CONFIG$x_coord_name, CONFIG$y_coord_name)])
  pred_spatial_data_time <- SpatialPointsDataFrame(
    coords = pred_coords_time,
    data = time_pred_data,
    proj4string = CRS("EPSG:3826")
  )
  
  # Calculate distance matrix
  cat("   - Calculating distance matrix...\n")
  distance_matrix <- gw.dist(
    dp.locat = coordinates(cal_spatial_data),
    rp.locat = coordinates(pred_spatial_data_time),
    p = 2,
    theta = 0,
    longlat = FALSE
  )
  
  # Run GWR prediction
  cat("   - Running GWR prediction...\n")
  prediction_start <- Sys.time()
  
  tryCatch({
    gwr_predictions <- gwr.predict(
      formula = model_formula,
      data = cal_spatial_data,
      predictdata = pred_spatial_data_time,
      bw = current_bw,
      kernel = current_kernel,
      adaptive = current_adaptive,
      dMat1 = distance_matrix
      # Note: dMat2 is not needed for gwr.predict, only for full calibration
    )
    
    prediction_time <- difftime(Sys.time(), prediction_start)
    cat("   - Prediction completed in", format(prediction_time), "\n")
    
    # Generate output filename
    prediction_filename <- sprintf("GWR_prediction_%s_%s_bw%d_time%s.rds",
                                  dependent_var,
                                  current_kernel,
                                  round(current_bw),
                                  current_time)
    
    # Save individual prediction if requested
    if (CONFIG$save_individual_predictions) {
      prediction_filepath <- file.path(CONFIG$output_directory, prediction_filename)
      saveRDS(gwr_predictions, prediction_filepath)
      cat("   - Prediction RDS saved to:", prediction_filename, "\n")
      
      # Also save CSV with results
      prediction_results_df <- if(inherits(gwr_predictions$SDF, "sf")) {
        sf::st_drop_geometry(gwr_predictions$SDF)
      } else {
        as.data.frame(gwr_predictions$SDF)
      }

      # Merge with original prediction data to preserve all columns
      original_cols <- time_pred_data[, setdiff(CONFIG$columns_to_preserve, 
                                                c(CONFIG$x_coord_name, CONFIG$y_coord_name)), 
                                      drop = FALSE]
      
      # Combine coordinates, original data, and prediction results
      output_df <- cbind(as.data.frame(pred_coords_time), 
                         original_cols,
                         prediction_results_df)

      csv_filename <- sprintf("GWR_prediction_%s_%s_bw%d_time%s_results.csv",
                             dependent_var,
                             current_kernel,
                             round(current_bw),
                             current_time)
      csv_filepath <- file.path(CONFIG$output_directory, csv_filename)
      write.csv(output_df, csv_filepath, row.names = FALSE)
      cat("   - Prediction CSV saved to:", csv_filename, "\n")
    }
    
    # Add to summary
    prediction_summary <- rbind(prediction_summary, data.frame(
      time_point = current_time,
      n_calibration_obs = nrow(time_cal_data),
      n_prediction_locations = nrow(time_pred_data),
      bandwidth_used = current_bw,
      kernel_used = current_kernel,
      adaptive_used = current_adaptive,
      prediction_file = prediction_filename,
      stringsAsFactors = FALSE
    ))
    
    if (CONFIG$verbose_output) {
      cat("   - Prediction complete for", nrow(time_pred_data), "locations\n")
    }
    
  }, error = function(e) {
    cat("   ERROR in prediction for time", current_time, ":", e$message, "\n")
  })
}

# ==============================================================================
# SECTION 5: SAVE SUMMARY RESULTS
# ==============================================================================
cat("\nStep 5: Saving summary results...\n")

if (nrow(prediction_summary) > 0) {
  # Save prediction summary
  if (CONFIG$save_summary_table) {
    summary_filename <- sprintf("GWR_prediction_%s_summary.csv",
                               dependent_var)
    summary_filepath <- file.path(CONFIG$output_directory, summary_filename)
    write.csv(prediction_summary, summary_filepath, row.names = FALSE)
    cat("   - Prediction summary saved to:", summary_filename, "\n")
  }
  
  # Print final summary
  cat("\n--- PREDICTION ANALYSIS COMPLETE ---\n")
  if (CONFIG$time_selection$use_selection) {
    cat("Processed time points:", paste(selected_time_points, collapse = ", "), "\n")
  }
  cat("Successfully completed predictions for", nrow(prediction_summary), "time points\n")
  cat("Average bandwidth used:", round(mean(prediction_summary$bandwidth_used), 2), "\n")
  cat("Kernels used:", paste(unique(prediction_summary$kernel_used), collapse = ", "), "\n")
  cat("Adaptive settings used:", paste(unique(prediction_summary$adaptive_used), collapse = ", "), "\n")
  cat("Prediction locations per time point:", unique(prediction_summary$n_prediction_locations), "\n")
  
} else {
  cat("   WARNING: No predictions were successfully completed.\n")
}

cat("\nAll prediction results saved in directory:", CONFIG$output_directory, "\n")
cat("--- Script finished successfully ---\n")