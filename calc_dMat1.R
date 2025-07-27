# gtwr_functions.R
# Core functions for GTWR dMat1 calculation

library(GWmodel)
library(arrow)
library(rhdf5)

#' Load and validate calibration data
#' 
#' @param calib_file Path to calibration CSV file
#' @param x_col X coordinate column name
#' @param y_col Y coordinate column name  
#' @param time_col Time column name
#' @return List with coords matrix and times vector
load_calibration_data <- function(calib_file, x_col, y_col, time_col) {
  
  # Load data
  calib_data <- read.csv(calib_file)
  
  # Validate columns
  required_cols <- c(x_col, y_col, time_col)
  missing_cols <- required_cols[!required_cols %in% names(calib_data)]
  if (length(missing_cols) > 0) {
    stop("Missing columns in calibration data: ", paste(missing_cols, collapse = ", "))
  }
  
  # Extract coordinates and times
  coords <- as.matrix(calib_data[, c(x_col, y_col)])
  times <- calib_data[[time_col]]
  
  list(
    coords = coords,
    times = times,
    n_points = nrow(coords),
    time_range = range(times)
  )
}

#' Validate prediction file columns
#' 
#' @param feather_file Path to feather file
#' @param x_col X coordinate column name
#' @param y_col Y coordinate column name
#' @param time_col Time column name
validate_prediction_file <- function(feather_file, x_col, y_col, time_col) {
  
  # Read first few rows to check columns
  sample_data <- read_feather(feather_file)
  
  required_cols <- c(x_col, y_col, time_col)
  missing_cols <- required_cols[!required_cols %in% names(sample_data)]
  if (length(missing_cols) > 0) {
    stop("Missing columns in ", basename(feather_file), ": ", 
         paste(missing_cols, collapse = ", "))
  }
  
  return(TRUE)
}

#' Calculate dMat1 for single feather file
#' 
#' @param calib_coords Calibration coordinates matrix
#' @param calib_times Calibration times vector
#' @param feather_file Path to prediction feather file
#' @param pred_x_col X coordinate column name in prediction data
#' @param pred_y_col Y coordinate column name in prediction data
#' @param pred_time_col Time column name in prediction data
#' @param lamda Spatial-temporal balance parameter
#' @param ksi Space-time interaction parameter
#' @param p Minkowski distance power
#' @param theta Coordinate rotation angle
#' @param longlat Use great circle distances
#' @param t.units Temporal distance units
#' @return Distance matrix
calculate_single_dMat1 <- function(calib_coords, calib_times, feather_file,
                                  pred_x_col, pred_y_col, pred_time_col,
                                  lamda = 0.05, ksi = 0, p = 2, theta = 0,
                                  longlat = FALSE, t.units = "auto") {
  
  # Load prediction data
  pred_data <- read_feather(feather_file)
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

#' Save dMat1 with metadata
#' 
#' @param dMat1 Distance matrix
#' @param output_file Output file path
#' @param file_id Feather file identifier
#' @param layer_name Layer name
#' @param params List of parameters used
save_dMat1 <- function(dMat1, output_file, file_id, layer_name, params) {
  
  # Ensure output directory exists
  dir.create(dirname(output_file), recursive = TRUE, showWarnings = FALSE)
  
  # Remove file if it exists (to avoid conflicts)
  if (file.exists(output_file)) {
    file.remove(output_file)
  }
  
  # Create HDF5 file and save distance matrix
  h5createFile(output_file)
  h5write(dMat1, output_file, "dMat1")
  
  # Create metadata group
  h5createGroup(output_file, "metadata")
  
  # Save metadata
  h5write(file_id, output_file, "metadata/file_id")
  h5write(layer_name, output_file, "metadata/layer_name")
  h5write(as.integer(dim(dMat1)[1]), output_file, "metadata/n_calibration")
  h5write(as.integer(dim(dMat1)[2]), output_file, "metadata/n_prediction")
  h5write(params$lamda, output_file, "metadata/lamda")
  h5write(params$ksi, output_file, "metadata/ksi")
  h5write(as.integer(params$p), output_file, "metadata/p")
  h5write(params$theta, output_file, "metadata/theta")
  h5write(params$t.units, output_file, "metadata/t_units")
  h5write(as.character(Sys.time()), output_file, "metadata/created_time")
  
  # Close HDF5 file
  H5close()
  
  return(output_file)
}

#' Process single feather file (main processing function)
#' 
#' @param calib_coords Calibration coordinates matrix
#' @param calib_times Calibration times vector  
#' @param feather_file Path to feather file
#' @param output_dir Output directory
#' @param layer_name Layer identifier
#' @param params List of parameters
#' @param column_config List of column names
#' @return Output file path
process_single_feather <- function(calib_coords, calib_times, feather_file,
                                  output_dir, layer_name, params, column_config) {
  
  # Extract file identifier
  file_id <- tools::file_path_sans_ext(basename(feather_file))
  
  # Calculate distance matrix
  dMat1 <- calculate_single_dMat1(
    calib_coords = calib_coords,
    calib_times = calib_times,
    feather_file = feather_file,
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
  
  # Save result
  output_file <- file.path(output_dir, paste0(layer_name, "_", file_id, "_dMat1.h5"))
  save_dMat1(dMat1, output_file, file_id, layer_name, params)

  # Clean up memory
  rm(dMat1)
  gc()
  
  return(output_file)
}

#' Create processing summary
#' 
#' @param output_files Vector of output file paths
#' @param layer_name Layer identifier
#' @param processing_time Total processing time
#' @param params Parameters used
save_processing_summary <- function(output_files, layer_name, processing_time, params) {
  
  summary_file <- file.path(dirname(output_files[1]), paste0(layer_name, "_summary.h5"))
  
  # Remove file if exists
  if (file.exists(summary_file)) {
    file.remove(summary_file)
  }
  
  # Create HDF5 file
  h5createFile(summary_file)
  h5createGroup(summary_file, "parameters")
  
  # Write data
  h5write(as.integer(length(output_files)), summary_file, "n_files_processed")
  h5write(processing_time, summary_file, "total_time_minutes")
  h5write(processing_time / length(output_files), summary_file, "avg_time_per_file")
  h5write(params$lamda, summary_file, "parameters/lamda")
  h5write(params$ksi, summary_file, "parameters/ksi")
  h5write(as.character(Sys.time()), summary_file, "completed_time")
  
  # Close HDF5
  H5close()
  
  return(summary_file)
}