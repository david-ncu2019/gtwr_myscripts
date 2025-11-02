# ==============================================================================
# Title:        GTWR Leave-One-Station-Out Validation
# Description:  Validates GTWR model by iteratively removing each station from
#               calibration data while predicting at all original locations
# Based on:     run_GTWR_main.R and GTWR_ExtractValues_to_RegPoints.R
# ==============================================================================

# ==============================================================================
# SECTION 1: USER CONFIGURATION
# ==============================================================================
cat("Step 1: Loading configuration settings for GTWR validation...\n")

CONFIG <- list(
  # --- File Paths ---
  input_csv_path = "20250921_GTWR_InputData_MLCW_InSAR_All_Layer.csv",
  output_directory = "GTWR_Validation/All_Layer",
  
  # --- Model Parameters ---
  formula_string = "All_Layer ~ CUMDISP",
  
  # --- GTWR Settings ---
  st_bw = 17,                    # Pre-determined bandwidth
  kernel_method = "bisquare",
  adaptive = TRUE,
  lambda_param = 0.001,
  ksi_param_degrees = 0,
  temporal_units = "months",
  
  # --- Data Column Names ---
  station_id_field = "STATION",   # Column identifying stations
  time_field_name = "monthly",
  x_coord_name = "X_TWD97",
  y_coord_name = "Y_TWD97",
  
  # --- Validation Settings ---
  save_individual_results = TRUE,
  combine_results = TRUE
)

# ==============================================================================
# SECTION 2: SCRIPT INITIALIZATION
# ==============================================================================
cat("Step 2: Initializing environment...\n")

suppressPackageStartupMessages({
  library("GWmodel")
  library("sf")
  library("sp")
})

# Create output directory
if (!dir.exists(CONFIG$output_directory)) {
  dir.create(CONFIG$output_directory, recursive = TRUE)
  cat("   - Created output directory:", CONFIG$output_directory, "\n")
}

# ==============================================================================
# SECTION 3: HELPER FUNCTIONS
# ==============================================================================

create_spatial_dataframe <- function(data, coord_cols, station_name = "") {
  coords <- as.matrix(data[, c(coord_cols$x, coord_cols$y)])
  
  sp_data <- SpatialPointsDataFrame(
    coords = coords,
    data = data,
    proj4string = CRS("EPSG:3826")
  )
  
  if (station_name != "") {
    cat("   - Created spatial dataframe for station", station_name, 
        "with", nrow(sp_data), "points\n")
  }
  
  return(sp_data)
}

extract_predictor_names <- function(formula) {
  # Extract predictor variable names from formula
  terms_obj <- terms(formula)
  predictor_names <- attr(terms_obj, "term.labels")
  return(predictor_names)
}

extract_response_name <- function(formula) {
  # Extract dependent variable name from formula
  terms_obj <- terms(formula)
  response_name <- as.character(attr(terms_obj, "variables"))[2]
  return(response_name)
}

save_station_results <- function(gtwr_results, station_id, output_dir, original_data, predictor_names, response_name) {
  # Extract results
  if (inherits(gtwr_results$SDF, "SpatialPointsDataFrame")) {
    results_df <- as.data.frame(gtwr_results$SDF)
    coords <- coordinates(gtwr_results$SDF)
    results_df$X_coord <- coords[, 1]
    results_df$Y_coord <- coords[, 2]
  } else {
    results_df <- st_drop_geometry(gtwr_results$SDF)
    coords <- st_coordinates(gtwr_results$SDF)
    results_df$X_coord <- coords[, 1]
    results_df$Y_coord <- coords[, 2]
  }
  
  # Create PointKey using round() to avoid integer overflow
  results_df$PointKey <- paste0("X", 
                               round(results_df$X_coord * 1000),
                               "Y", 
                               round(results_df$Y_coord * 1000),
                               "T", 
                               sprintf("%03d", results_df$time_stamp))
  
  # Rename coefficient columns to avoid conflicts
  if ("Intercept" %in% names(results_df)) {
    names(results_df)[names(results_df) == "Intercept"] <- "Intercept_coeff"
  }
  
  for (pred_name in predictor_names) {
    if (pred_name %in% names(results_df)) {
      names(results_df)[names(results_df) == pred_name] <- paste0(pred_name, "_coeff")
    }
  }
  
  # Map back original values using PointKey
  if ("PointKey" %in% names(original_data)) {
    # Create lookup table including response variable and station
    lookup_cols <- c("PointKey", predictor_names, response_name, "STATION")
    available_cols <- intersect(lookup_cols, names(original_data))
    lookup_table <- original_data[, available_cols, drop = FALSE]
    lookup_table <- lookup_table[!duplicated(lookup_table$PointKey), ]
    
    # Map predictor values
    for (pred_name in predictor_names) {
      if (pred_name %in% names(lookup_table)) {
        results_df[[pred_name]] <- lookup_table[[pred_name]][match(results_df$PointKey, lookup_table$PointKey)]
      }
    }
    
    # Map response variable (observed values)
    if (response_name %in% names(lookup_table)) {
      results_df[[paste0("Observed_", response_name)]] <- lookup_table[[response_name]][match(results_df$PointKey, lookup_table$PointKey)]
    }
    
    # Map station information
    if ("STATION" %in% names(lookup_table)) {
      results_df[["STATION"]] <- lookup_table[["STATION"]][match(results_df$PointKey, lookup_table$PointKey)]
    }
    
    # Calculate predicted values for each predictor
    for (pred_name in predictor_names) {
      coeff_col <- paste0(pred_name, "_coeff")
      if (coeff_col %in% names(results_df) && pred_name %in% names(results_df)) {
        if ("Intercept_coeff" %in% names(results_df)) {
          results_df[[paste0("Predicted_", response_name)]] <- 
            results_df[[pred_name]] * results_df[[coeff_col]] + results_df[["Intercept_coeff"]]
        } else {
          results_df[[paste0("Predicted_", response_name)]] <- 
            results_df[[pred_name]] * results_df[[coeff_col]]
        }
      }
    }
    
    cat("   - Mapped", length(predictor_names), "predictors, response variable, and station info using PointKey\n")
  }
  
  # Add validation metadata
  results_df$excluded_station <- station_id
  results_df$validation_type <- "leave_one_station_out"
  
  # Save individual station results
  station_file <- file.path(output_dir, paste0("station_", station_id, "_results.csv"))
  write.csv(results_df, station_file, row.names = FALSE)
  
  cat("   - Saved results for station", station_id, "to", basename(station_file), "\n")
  
  return(results_df)
}

# ==============================================================================
# SECTION 4: DATA LOADING AND PREPARATION
# ==============================================================================
cat("\nStep 3: Loading and preparing data...\n")

# Load full dataset
input_data <- read.csv(CONFIG$input_csv_path)
cat("   - Loaded dataset:", nrow(input_data), "rows,", ncol(input_data), "columns\n")

# Validate required columns
required_cols <- c(CONFIG$station_id_field, CONFIG$time_field_name, 
                  CONFIG$x_coord_name, CONFIG$y_coord_name)
missing_cols <- setdiff(required_cols, names(input_data))
if (length(missing_cols) > 0) {
  stop("Missing required columns: ", paste(missing_cols, collapse = ", "))
}

# Create PointKey for easier result mapping
cat("   - Creating PointKey for result mapping...\n")
input_data$PointKey <- paste0("X", 
                             round(input_data[[CONFIG$x_coord_name]] * 1000),
                             "Y", 
                             round(input_data[[CONFIG$y_coord_name]] * 1000),
                             "T", 
                             sprintf("%03d", input_data[[CONFIG$time_field_name]]))

cat("   - Generated", length(unique(input_data$PointKey)), "unique PointKeys\n")

# Get unique stations
all_stations <- unique(input_data[[CONFIG$station_id_field]])
n_stations <- length(all_stations)
cat("   - Found", n_stations, "unique stations:", paste(all_stations, collapse = ", "), "\n")

# Create model formula and extract variable names
model_formula <- as.formula(CONFIG$formula_string)
predictor_names <- extract_predictor_names(model_formula)
response_name <- extract_response_name(model_formula)
ksi_param_radians <- CONFIG$ksi_param_degrees * (pi / 180)

cat("   - Model formula:", deparse(model_formula), "\n")
cat("   - Response variable:", response_name, "\n")
cat("   - Predictor variables:", paste(predictor_names, collapse = ", "), "\n")

# Create full regression points (used for all validations)
coords <- as.matrix(input_data[, c(CONFIG$x_coord_name, CONFIG$y_coord_name)])
regression_sp <- create_spatial_dataframe(input_data, 
                                         list(x = CONFIG$x_coord_name, y = CONFIG$y_coord_name))
reg_tv <- input_data[[CONFIG$time_field_name]]

cat("   - Regression points prepared:", nrow(regression_sp), "locations\n")

# ==============================================================================
# SECTION 5: VALIDATION LOOP
# ==============================================================================
cat("\nStep 4: Running leave-one-station-out validation...\n")
cat("========================================================\n")

# Storage for combined results
all_validation_results <- list()
validation_summary <- data.frame(
  station_id = character(),
  n_calibration_points = integer(),
  n_predictions = integer(),
  computation_time_sec = numeric(),
  stringsAsFactors = FALSE
)

# Main validation loop
total_start_time <- Sys.time()

for (i in seq_along(all_stations)) {
  current_station <- all_stations[i]
  
  cat("\n--- Validation", i, "of", n_stations, ": Excluding station", current_station, "---\n")
  
  iteration_start_time <- Sys.time()
  
  # Create calibration data (exclude current station)
  calibration_data <- input_data[input_data[[CONFIG$station_id_field]] != current_station, ]
  
  cat("   - Calibration points:", nrow(calibration_data), 
      "(excluded", sum(input_data[[CONFIG$station_id_field]] == current_station), "points)\n")
  
  # Create spatial calibration object
  calibration_sp <- create_spatial_dataframe(calibration_data, 
                                           list(x = CONFIG$x_coord_name, y = CONFIG$y_coord_name))
  obs_tv <- calibration_data[[CONFIG$time_field_name]]
  
  # Fit GTWR model
  cat("   - Fitting GTWR model...\n")
  
  gtwr_model <- gtwr(
    formula = model_formula,
    data = calibration_sp,
    regression.points = regression_sp,  # Predict at ALL original locations
    obs.tv = obs_tv,
    reg.tv = reg_tv,
    st.bw = CONFIG$st_bw,
    kernel = CONFIG$kernel_method,
    adaptive = CONFIG$adaptive,
    p = 2,
    theta = 0,
    longlat = FALSE,
    lamda = CONFIG$lambda_param,
    t.units = CONFIG$temporal_units,
    ksi = ksi_param_radians
  )
  
  iteration_end_time <- Sys.time()
  computation_time <- as.numeric(difftime(iteration_end_time, iteration_start_time, units = "secs"))
  
  cat("   - Model fitted successfully in", round(computation_time, 2), "seconds\n")
  
  # Save individual results if requested
  if (CONFIG$save_individual_results) {
    station_results <- save_station_results(gtwr_model, current_station, CONFIG$output_directory, 
                                           input_data, predictor_names, response_name)
    if (CONFIG$combine_results) {
      all_validation_results[[as.character(current_station)]] <- station_results
    }
  }
  
  # Update summary
  validation_summary <- rbind(validation_summary, data.frame(
    station_id = current_station,
    n_calibration_points = nrow(calibration_data),
    n_predictions = nrow(regression_sp),
    computation_time_sec = computation_time,
    stringsAsFactors = FALSE
  ))
  
  # Clean up memory
  rm(gtwr_model, calibration_data, calibration_sp, obs_tv)
  gc()
}

total_end_time <- Sys.time()
# FIXED: Use proper quotes and simplified calculation
total_time <- difftime(total_end_time, total_start_time, units = "mins")

# ==============================================================================
# SECTION 6: COMBINE AND SAVE RESULTS
# ==============================================================================
cat("\nStep 5: Finalizing validation results...\n")

# Save validation summary
summary_file <- file.path(CONFIG$output_directory, "validation_summary.csv")
write.csv(validation_summary, summary_file, row.names = FALSE)
cat("   - Saved validation summary to", basename(summary_file), "\n")

# Combine all results if requested
if (CONFIG$combine_results && length(all_validation_results) > 0) {
  cat("   - Combining all validation results...\n")
  
  combined_results <- do.call(rbind, all_validation_results)
  combined_file <- file.path(CONFIG$output_directory, "combined_validation_results.csv")
  write.csv(combined_results, combined_file, row.names = FALSE)
  
  cat("   - Saved combined results to", basename(combined_file), "\n")
  cat("   - Combined dataset:", nrow(combined_results), "predictions across", 
      length(all_validation_results), "validation runs\n")
}

# ==============================================================================
# SECTION 7: VALIDATION SUMMARY
# ==============================================================================
# FIXED: Simplified the separator line creation
separator_line <- paste(rep("=", 60), collapse = "")
cat("\n", separator_line, "\n", sep = "")
cat("GTWR LEAVE-ONE-STATION-OUT VALIDATION COMPLETE\n")
cat(separator_line, "\n")

cat("Total stations validated:", n_stations, "\n")
cat("Total computation time:", round(as.numeric(total_time), 2), "minutes\n")
cat("Average time per station:", round(mean(validation_summary$computation_time_sec), 2), "seconds\n")
cat("Total predictions generated:", sum(validation_summary$n_predictions), "\n")

cat("\nValidation approach:\n")
cat("- Method: Leave-one-station-out cross-validation\n")
# FIXED: Simplified conditional statement in cat()
bw_type <- ifelse(CONFIG$adaptive, "adaptive", "fixed")
cat("- Bandwidth:", CONFIG$st_bw, "(", bw_type, ")\n")
cat("- Kernel:", CONFIG$kernel_method, "\n")
cat("- Lambda:", CONFIG$lambda_param, "\n")

cat("\nOutput files saved in:", CONFIG$output_directory, "\n")
if (CONFIG$save_individual_results) {
  cat("- Individual station results: station_[ID]_results.csv\n")
}
if (CONFIG$combine_results) {
  cat("- Combined results: combined_validation_results.csv\n")
}
cat("- Validation summary: validation_summary.csv\n")

cat("\nNext steps for analysis:\n")
cat("1. Calculate prediction accuracy metrics (RMSE, MAE, R-squared)\n")
cat("2. Analyze spatial patterns in prediction errors\n")
cat("3. Compare predictions at excluded stations vs. observations\n")
cat("4. Assess model stability across different station exclusions\n")

cat(separator_line, "\n")