#!/usr/bin/env Rscript
# gtwr_single_run.R
# Single run GTWR distance matrix calculation between two CSV files

# Load functions from your calc_dMat1.R file
source("calc_dMat1.R")

# =============================================================================
# CONFIGURATION
# =============================================================================

# Input files
calib_file <- "calibration_points.csv"
regression_file <- "regression_points_2.csv"
output_file <- "test_dMat1.h5"

# Column configuration
column_config <- list(
  # Calibration data columns
  calib_x_col = "X_TWD97",
  calib_y_col = "Y_TWD97", 
  calib_time_col = "monthly",
  
  # Regression data columns
  pred_x_col = "X_TWD97",
  pred_y_col = "Y_TWD97",
  pred_time_col = "monthly"
)

# GTWR parameters
params <- list(
  lamda = 0.006,      # Spatial-temporal balance (0-1)
  ksi = 0,            # Space-time interaction parameter
  p = 2,              # Minkowski distance power (2 = Euclidean)
  theta = 0,          # Coordinate rotation angle (radians)
  longlat = FALSE,    # Use great circle distances
  t.units = "auto"  # Temporal distance units
)

# =============================================================================
# ADDITIONAL FUNCTIONS FOR CSV INPUT
# =============================================================================

calculate_csv_dMat1 <- function(calib_coords, calib_times, regression_file,
                                pred_x_col, pred_y_col, pred_time_col,
                                lamda = 0.05, ksi = 0, p = 2, theta = 0,
                                longlat = FALSE, t.units = "auto") {
  
  # Load regression data from CSV (adapting your feather function)
  pred_data <- read.csv(regression_file)
  pred_coords <- as.matrix(pred_data[, c(pred_x_col, pred_y_col)])
  pred_times <- pred_data[[pred_time_col]]
  
  # Calculate distance matrix using GWmodel
  dMat1 <- st.dist(
    dp.locat = calib_coords,
    rp.locat = pred_coords,
    obs.tv = calib_times,
    reg.tv = pred_times,
    p = p,
    theta = theta,
    longlat = longlat,
    lamda = lamda,
    t.units = t.units,
    ksi = ksi
  )
  
  return(dMat1)
}



# =============================================================================
# MAIN EXECUTION
# =============================================================================

main <- function() {
  cat("=== GTWR Single Run Distance Matrix Calculation ===\n\n")
  
  start_time <- Sys.time()
  
  # Step 1: Load calibration data using your existing function
  cat("Step 1: Loading calibration data\n")
  calib_data <- load_calibration_data(
    calib_file, 
    column_config$calib_x_col,
    column_config$calib_y_col,
    column_config$calib_time_col
  )
  
  cat("Calibration points:", calib_data$n_points, "\n")
  cat("Time range:", paste(calib_data$time_range, collapse = " - "), "\n")
  
  # Step 2: Calculate distance matrix
  cat("\nStep 2: Calculating GTWR distance matrix\n")
  dMat1 <- calculate_csv_dMat1(
    calib_coords = calib_data$coords,
    calib_times = calib_data$times,
    regression_file = regression_file,
    pred_x_col = column_config$pred_x_col,
    pred_y_col = column_config$pred_y_col,
    pred_time_col = column_config$pred_time_col,
    lamda = params$lamda,
    ksi = params$ksi,
    p = params$p,
    theta = params$theta,
    longlat = params$longlat,
    t.units = params$t.units
  )
  
  cat("Distance matrix dimensions:", dim(dMat1)[1], "x", dim(dMat1)[2], "\n")
  
  # Step 3: Save results using your existing function
  cat("\nStep 3: Saving results\n")
  file_id <- tools::file_path_sans_ext(basename(regression_file))
  layer_name <- "single_run"
  
  save_dMat1(
    dMat1 = dMat1,
    output_file = output_file,
    file_id = file_id,
    layer_name = layer_name,
    params = params
  )
  
  # Summary
  total_time <- as.numeric(Sys.time() - start_time, units = "mins")
  cat("\n=== SUMMARY ===\n")
  cat("Total processing time:", round(total_time, 2), "minutes\n")
  cat("Distance matrix size:", paste(dim(dMat1), collapse = " x "), "\n")
  cat("Output file:", output_file, "\n")
  cat("\nProcessing completed successfully!\n")
  
  return(invisible(output_file))
}

# =============================================================================
# RUN
# =============================================================================

# Execute main function
if (!interactive()) {
  main()
}