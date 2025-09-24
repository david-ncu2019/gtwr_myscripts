#!/usr/bin/env Rscript
# calculate_dMat1_main.R
# Main execution script for calculating dMat1 matrices

# Load functions
source("calc_dMat1.R")

# =============================================================================
# CONFIGURATION PARAMETERS
# =============================================================================

# File paths
calib_file <- "20250504_GTWR_InputData_MLCW_InSAR_Layer_4.csv"
# pred_dir <- "test_feather"
pred_dir <- "regpoints_Chunk100_addCUMDISP"
output_base_dir <- "2__Prepare_dMat1"
layer_name <- "Layer_4"

# =============================================================================
# NOOOOOOTEEEEE: MODIFY THESE VALUES TO RUN IN CHUNKS
# =============================================================================
start_idx = 526
end_idx = 600


# =============================================================================

# Column configuration
column_config <- list(
  # Calibration data columns
  calib_x_col = "X_TWD97",
  calib_y_col = "Y_TWD97", 
  calib_time_col = "monthly",
  
  # Prediction data columns  
  pred_x_col = "X_TWD97",
  pred_y_col = "Y_TWD97",
  pred_time_col = "monthly"
)

# GTWR parameters
params <- list(
  lamda = 0.005,           # Spatial-temporal balance (0-1)
  ksi = 0,                # Space-time interaction parameter
  p = 2,                  # Minkowski distance power (2 = Euclidean)
  theta = 0,              # Coordinate rotation angle (radians)
  longlat = FALSE,        # Use great circle distances
  t.units = "months"        # Temporal distance units
)

# Processing options
processing_options <- list(
  progress_interval = 20,  # Show progress every N files
  verbose = TRUE          # Print detailed messages
)

# =============================================================================
# MAIN EXECUTION
# =============================================================================

cat("=== GTWR dMat1 Calculation for", layer_name, "===\n")

# Create output directory
output_dir <- file.path(output_base_dir, paste0(layer_name, "_dMat1_output"))
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# Load calibration data
cat("Loading calibration data...\n")
calib_data <- load_calibration_data(
  calib_file, 
  column_config$calib_x_col,
  column_config$calib_y_col,
  column_config$calib_time_col
)

cat("Calibration points:", calib_data$n_points, "\n")
cat("Time range:", paste(calib_data$time_range, collapse = " - "), "\n")

# Get feather files
feather_files <- list.files(pred_dir, pattern = "*.feather", full.names = TRUE)

feather_files <- feather_files[start_idx:min(end_idx, length(feather_files))]

n_files <- length(feather_files)
cat("Found", n_files, "feather files to process\n")

# Validate first file columns
if (n_files > 0) {
  validate_prediction_file(
    feather_files[1],
    column_config$pred_x_col,
    column_config$pred_y_col, 
    column_config$pred_time_col
  )
  cat("Column validation passed\n")
}

# Process files
cat("\nStarting processing...\n")
start_time <- Sys.time()
output_files <- character(n_files)

for (i in seq_along(feather_files)) {
  
  if (processing_options$verbose) {
    cat(sprintf("[%d/%d] Processing %s...", 
                i, n_files, basename(feather_files[i])))
  }
  
  # Process single file
  output_files[i] <- process_single_feather(
    calib_coords = calib_data$coords,
    calib_times = calib_data$times,
    feather_file = feather_files[i],
    output_dir = output_dir,
    layer_name = layer_name,
    params = params,
    column_config = column_config
  )
  
  if (processing_options$verbose) {
    cat(" Done\n")
  }
  
  # Progress update
  if (i %% processing_options$progress_interval == 0) {
    elapsed <- as.numeric(Sys.time() - start_time, units = "mins")
    avg_time <- elapsed / i
    remaining <- (n_files - i) * avg_time
    
    cat(sprintf("Progress: %d%% | Elapsed: %.1f min | Est. remaining: %.1f min\n", 
                round(100 * i / n_files), elapsed, remaining))
  }
}

# Final summary
total_time <- as.numeric(Sys.time() - start_time, units = "mins")

# Save processing summary
summary_file <- save_processing_summary(
  output_files, layer_name, total_time, params
)

# Print final results
cat("\n=== PROCESSING COMPLETED ===\n")
cat("Layer:", layer_name, "\n")
cat("Files processed:", n_files, "\n")
cat("Total time:", round(total_time, 1), "minutes\n")
cat("Average time per file:", round(total_time / n_files, 2), "minutes\n")
cat("Output directory:", output_dir, "\n")
cat("Summary file:", summary_file, "\n")
# 
# # Display output structure
# cat("\nOutput files created:\n")
# for (i in 1:min(5, length(output_files))) {
#   cat("  ", basename(output_files[i]), "\n")
# }
# if (length(output_files) > 5) {
#   cat("  ... and", length(output_files) - 5, "more files\n")
# }
# 
# cat("\nParameters used:\n")
# cat("  lamda:", params$lamda, "\n")
# cat("  ksi:", params$ksi, "\n")
# cat("  p:", params$p, "\n")
# cat("  t.units:", params$t.units, "\n")