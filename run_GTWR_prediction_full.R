#!/usr/bin/env Rscript
# batch_gtwr_prediction.R - FIXED VERSION
# Batch processing of GTWR predictions with proper spatial data handling

# Required libraries
library(GWmodel)
library(rhdf5)
library(arrow)  # For feather format support
library(sp)     # Explicitly load sp for spatial operations
library(sf)

# Source the corrected GTWR prediction functions
source("gtwr.predict.R")

# =============================================================================
# CONFIGURATION SECTION (One-time setup)
# =============================================================================

# Working directory and file paths
work_dir <- getwd()
input_config <- list(
  # Fixed files used for all predictions
  calibration_file = file.path(work_dir, "20250504_GTWR_InputData_MLCW_InSAR_Layer_4.csv"),
  dMat2_file = file.path(work_dir, "1__Prepare_dMat2", "Layer_4_dMat2.h5"),
  
  # Template paths for chunk-specific files
  regpoints_folder = file.path(work_dir, "regpoints_Chunk100_addCUMDISP"),
  dMat1_folder = file.path(work_dir, "2__Prepare_dMat1", "Layer_4_dMat1_output"),
  
  # File naming patterns
  regpoints_pattern = "grid_pnt_chunk%03d.feather",     # e.g., grid_pnt_chunk001.feather
  dMat1_pattern = "Layer_4_grid_pnt_chunk%03d_dMat1.h5" # e.g., Layer_4_grid_pnt_chunk001_dMat1.h5
)

# FIXED: Add CRS configuration
crs_config <- list(
  epsg_code = "EPSG:3826",
  proj4_string = "+proj=tmerc +lat_0=0 +lon_0=121 +k=0.9999 +x_0=250000 +y_0=0 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs",
  description = "TWD97 Taiwan TM2 zone 121"
)

# Output configuration
output_folder <- file.path(work_dir, "3__PredictionOutput", "Layer_4")

# Column mapping (same for all chunks)
column_config <- list(
  calib_x_col = "X_TWD97",
  calib_y_col = "Y_TWD97", 
  calib_time_col = "monthly",
  calib_response_col = "Layer_4",
  calib_predictors = c("CUMDISP"),
  
  pred_x_col = "X_TWD97",
  pred_y_col = "Y_TWD97",
  pred_time_col = "monthly"
)

# GTWR parameters (same for all chunks)
gtwr_params <- list(
  st.bw = 17,
  kernel = "bisquare",
  adaptive = TRUE,
  calculate.var = TRUE, # CORRECTED: Changed 'calculate_variance' to 'calculate.var' for consistency
  p = 2,
  theta = 0,
  longlat = FALSE,
  lamda = 0.005,
  t.units = "months",
  ksi = 0
)

# Batch processing configuration
batch_config <- list(
  start_chunk = 546,              # Starting chunk number
  end_chunk = 600,                # End chunk number (NULL = process all available)
  output_format = "feather",    # Output format for batch processing
  save_log = TRUE,              # Save processing log
  resume_mode = FALSE           # Skip existing output files
)

# =============================================================================
# FIXED HELPER FUNCTIONS FOR SPATIAL DATA HANDLING
# =============================================================================

#' Create CRS object from configuration
create_crs <- function() {
  tryCatch({
    # Try modern approach first (sf/sp 1.4+ compatibility)
    if (requireNamespace("sf", quietly = TRUE)) {
      crs_obj <- sf::st_crs(crs_config$proj4_string)
      return(as(crs_obj, "CRS"))
    } else {
      # Fallback to traditional approach
      return(sp::CRS(crs_config$proj4_string))
    }
  }, error = function(e) {
    tryCatch({
      # Try EPSG code approach
      return(sp::CRS(paste0("+init=", tolower(crs_config$epsg_code))))
    }, error = function(e2) {
      # Final fallback - create empty CRS and assign manually
      warning("Using basic CRS creation due to compatibility issues")
      return(sp::CRS("+proj=longlat +datum=WGS84"))
    })
  })
}

#' Convert data.frame to SpatialPointsDataFrame with proper CRS
create_spatial_points <- function(data, x_col, y_col) {
  # Validate coordinate columns exist and have numeric values
  if (!x_col %in% names(data) || !y_col %in% names(data)) {
    stop("ERROR: Coordinate columns not found: ", x_col, ", ", y_col)
  }
  
  if (!is.numeric(data[[x_col]]) || !is.numeric(data[[y_col]])) {
    stop("ERROR: Coordinate columns must be numeric")
  }
  
  # Check for missing coordinates
  coord_complete <- complete.cases(data[, c(x_col, y_col)])
  if (!all(coord_complete)) {
    warning("Removing ", sum(!coord_complete), " rows with missing coordinates")
    data <- data[coord_complete, ]
  }
  
  # Create coordinate matrix
  coords <- as.matrix(data[, c(x_col, y_col)])

  # FIXED: Convert tibble to data.frame for sp compatibility
  data <- as.data.frame(data)
  
  # Create CRS object
  crs_obj <- create_crs()
  
  # Create SpatialPointsDataFrame with error handling
  spatial_data <- tryCatch({
    crs_obj <- create_crs()
    SpatialPointsDataFrame(
      coords = coords,
      data = data,
      proj4string = crs_obj,
      match.ID = FALSE
    )
  }, error = function(e) {
    # Fallback: create without CRS first, then assign
    spatial_data <- SpatialPointsDataFrame(
      coords = coords,
      data = data,
      match.ID = FALSE
    )
    # Try to assign CRS after creation
    tryCatch({
      sp::proj4string(spatial_data) <- sp::CRS(crs_config$proj4_string)
    }, error = function(e2) {
      warning("Could not assign CRS to spatial object")
    })
    return(spatial_data)
  })
}

#' Validate spatial object has proper CRS
validate_spatial_crs <- function(spatial_obj, obj_name = "spatial object") {
  # Get CRS information safely
  crs_info <- tryCatch({
    sp::proj4string(spatial_obj)
  }, error = function(e) {
    # Alternative method for newer packages
    if (methods::hasMethod("st_crs", class(spatial_obj))) {
      as.character(sf::st_crs(spatial_obj))
    } else {
      NA
    }
  })
  
  if (is.na(crs_info) || crs_info == "" || crs_info == "NA") {
    warning("WARNING: ", obj_name, " has undefined coordinate system, but proceeding")
    return(TRUE)  # Don't stop execution, just warn
  }
  
  return(TRUE)
}

# =============================================================================
# UPDATED HELPER FUNCTIONS
# =============================================================================

#' Generate output filename following the specified pattern
generate_output_filename <- function(chunk_num, column_config, gtwr_params, batch_config) {
  filename <- sprintf("gtwr_prediction_%s_%s_lambda-%.3f_stbw-%d_chunk%03d.%s",
                      column_config$calib_response_col,
                      gtwr_params$kernel,
                      gtwr_params$lamda,
                      gtwr_params$st.bw,
                      chunk_num,
                      batch_config$output_format)
  return(filename)
}

#' Check if chunk files exist
check_chunk_files <- function(chunk_num, input_config) {
  regpoints_file <- file.path(input_config$regpoints_folder,
                              sprintf(input_config$regpoints_pattern, chunk_num))
  dMat1_file <- file.path(input_config$dMat1_folder,
                          sprintf(input_config$dMat1_pattern, chunk_num))
  
  return(list(
    regpoints_exists = file.exists(regpoints_file),
    dMat1_exists = file.exists(dMat1_file),
    regpoints_file = regpoints_file,
    dMat1_file = dMat1_file
  ))
}

#' Load and validate HDF5 distance matrix
load_distance_matrix <- function(file_path, matrix_name = "dMat", expected_dims = NULL) {
  if (!file.exists(file_path)) {
    stop("ERROR: Distance matrix file not found: ", file_path)
  }
  
  possible_names <- c(matrix_name, "dMat1", "dMat2", "dMat")
  matrix_loaded <- NULL
  
  for (name in possible_names) {
    if (name %in% h5ls(file_path)$name) {
      matrix_loaded <- h5read(file_path, name)
      break
    }
  }
  
  if (is.null(matrix_loaded)) {
    available_objects <- h5ls(file_path)$name
    stop("ERROR: No distance matrix found in ", file_path,
         "\nAvailable objects: ", paste(available_objects, collapse = ", "))
  }
  
  if (!is.null(expected_dims)) {
    actual_dims <- dim(matrix_loaded)
    if (!all(actual_dims == expected_dims)) {
      stop("ERROR: Matrix dimensions mismatch. Expected: ",
           paste(expected_dims, collapse = "×"),
           ", Got: ", paste(actual_dims, collapse = "×"))
    }
  }
  
  return(matrix_loaded)
}

#' FIXED: Load calibration data and convert to spatial
load_calibration_data <- function(file_path, column_config) {
  if (!file.exists(file_path)) {
    stop("ERROR: Calibration file not found: ", file_path)
  }
  
  cat("Loading calibration data from:", basename(file_path), "\n")
  data <- read.csv(file_path, stringsAsFactors = FALSE)
  
  required_cols <- c(column_config$calib_x_col,
                     column_config$calib_y_col,
                     column_config$calib_time_col,
                     column_config$calib_response_col,
                     column_config$calib_predictors)
  
  missing_cols <- required_cols[!required_cols %in% names(data)]
  if (length(missing_cols) > 0) {
    stop("ERROR: Missing columns in calibration data: ",
         paste(missing_cols, collapse = ", "))
  }
  
  cat("  Data dimensions:", nrow(data), "×", ncol(data), "\n")
  cat("  Converting to spatial object with CRS:", crs_config$epsg_code, "\n")
  
  # FIXED: Convert to spatial object with proper CRS
  spatial_data <- create_spatial_points(data, 
                                        column_config$calib_x_col, 
                                        column_config$calib_y_col)
  
  # Validate spatial object
  validate_spatial_crs(spatial_data, "calibration data")
  
  cat("  ✓ Created SpatialPointsDataFrame with", nrow(spatial_data), "points\n")
  
  return(spatial_data)
}

#' FIXED: Load regression points and convert to spatial
load_regression_chunk <- function(file_path, column_config) {
  if (!file.exists(file_path)) {
    stop("ERROR: Regression points file not found: ", file_path)
  }
  
  data <- read_feather(file_path)

  # FIXED: Convert tibble to data.frame for spatial compatibility
  data <- as.data.frame(data)
  
  required_cols <- c(column_config$pred_x_col,
                     column_config$pred_y_col,
                     column_config$pred_time_col,
                     column_config$calib_predictors)
  
  missing_cols <- required_cols[!required_cols %in% names(data)]
  if (length(missing_cols) > 0) {
    stop("ERROR: Missing columns in regression chunk: ",
         paste(missing_cols, collapse = ", "))
  }
  
  # FIXED: Convert to spatial object with proper CRS
  spatial_data <- create_spatial_points(data,
                                        column_config$pred_x_col,
                                        column_config$pred_y_col)
  
  # Validate spatial object
  validate_spatial_crs(spatial_data, "regression points")
  
  return(spatial_data)
}

#' Create formula from column configuration
create_formula <- function(column_config) {
  response <- column_config$calib_response_col
  predictors <- paste(column_config$calib_predictors, collapse = " + ")
  formula_str <- paste(response, "~", predictors)
  return(as.formula(formula_str))
}

#' Save prediction results in feather format
save_chunk_predictions <- function(gtwr_results, output_file, column_config, reg.tv, pred_data_orig) {
  # Extract results from GTWR object
  if (inherits(gtwr_results$SDF, "Spatial")) {
    pred_results <- as(gtwr_results$SDF, "data.frame")
    coords <- coordinates(gtwr_results$SDF)
  } else {
    pred_results <- st_drop_geometry(gtwr_results$SDF)
    coords <- st_coordinates(gtwr_results$SDF)
  }
  
  # Add coordinate columns
  # pred_results$X_coord <- coords[, 1]
  # pred_results$Y_coord <- coords[, 2]
  pred_results$pred_time <- reg.tv
  
  # Add predictor values from original prediction data
  pred_data_df <- as(pred_data_orig, "data.frame")
  for (predictor in column_config$calib_predictors) {
    if (predictor %in% names(pred_data_df)) {
      pred_results[[paste0("pred_", predictor)]] <- pred_data_df[[predictor]]
    }
  }
  
  # Standardize prediction column name
  pred_col_names <- c("gw.predict", "prediction", "gtwr_prediction")
  for (old_name in pred_col_names) {
    if (old_name %in% names(pred_results)) {
      names(pred_results)[names(pred_results) == old_name] <- "gtwr_prediction"
      break
    }
  }
  
  # Add metadata
  pred_results$bandwidth <- gtwr_results$GTW.arguments$st.bw
  pred_results$kernel <- gtwr_results$GTW.arguments$kernel
  # pred_results$prediction_time <- Sys.time()
  # pred_results$crs_epsg <- crs_config$epsg_code
  
  # Save as feather file
  write_feather(pred_results, output_file)
  
  return(list(
    file = output_file,
    n_predictions = nrow(pred_results),
    summary = list(
      min_pred = min(pred_results$gtwr_prediction, na.rm = TRUE),
      max_pred = max(pred_results$gtwr_prediction, na.rm = TRUE),
      mean_pred = mean(pred_results$gtwr_prediction, na.rm = TRUE)
    )
  ))
}

# =============================================================================
# FIXED CHUNK PROCESSING FUNCTION
# =============================================================================

process_chunk <- function(chunk_num, calib_data, dMat2, formula, obs.tv, log_file = NULL) {
  
  chunk_start_time <- Sys.time()
  
  # Generate file paths for this chunk
  chunk_files <- check_chunk_files(chunk_num, input_config)
  
  if (!chunk_files$regpoints_exists) {
    warning("Regression points file not found for chunk ", chunk_num, ": ", chunk_files$regpoints_file)
    return(NULL)
  }
  
  if (!chunk_files$dMat1_exists) {
    warning("dMat1 file not found for chunk ", chunk_num, ": ", chunk_files$dMat1_file)
    return(NULL)
  }
  
  # Generate output filename
  output_filename <- generate_output_filename(chunk_num, column_config, gtwr_params, batch_config)
  output_filepath <- file.path(output_folder, output_filename)
  
  # Check if output already exists and skip if in resume mode
  if (batch_config$resume_mode && file.exists(output_filepath)) {
    cat("SKIPPING chunk", sprintf("%03d", chunk_num), "- output already exists\n")
    return(list(status = "skipped", chunk = chunk_num, file = output_filepath))
  }
  
  cat("PROCESSING chunk", sprintf("%03d", chunk_num), "\n")
  
  # FIXED: Load regression points with proper spatial conversion
  pred_data <- load_regression_chunk(chunk_files$regpoints_file, column_config)
  pred_data_orig <- pred_data  # Keep copy for metadata
  n_pred <- nrow(pred_data)
  
  cat("  Loaded", n_pred, "regression points as SpatialPointsDataFrame\n")
  
  # Load dMat1 for this chunk
  dMat1 <- load_distance_matrix(chunk_files$dMat1_file, "dMat1",
                                expected_dims = c(nrow(calib_data), n_pred))
  cat("  Loaded dMat1 matrix:", paste(dim(dMat1), collapse = "×"), "\n")
  
  # Extract time vectors
  reg.tv <- pred_data[[column_config$pred_time_col]]
  
  # Validate spatial objects before GTWR prediction
  validate_spatial_crs(calib_data, "calibration data")
  validate_spatial_crs(pred_data, "prediction data")
  
  # Run GTWR prediction
  cat("  Running GTWR prediction...\n")
  gtwr_results <- gtwr.predict(
    formula = formula,
    data = calib_data,
    predictdata = pred_data,
    obs.tv = obs.tv,
    reg.tv = reg.tv,
    st.bw = gtwr_params$st.bw,
    kernel = gtwr_params$kernel,
    adaptive = gtwr_params$adaptive,
    p = gtwr_params$p,
    theta = gtwr_params$theta,
    longlat = gtwr_params$longlat,
    lamda = gtwr_params$lamda,
    t.units = gtwr_params$t.units,
    ksi = gtwr_params$ksi,
    dMat1 = dMat1,
    dMat2 = dMat2,
    calculate.var = gtwr_params$calculate.var # Note: Name consistency with list is now correct
  )
  
  # Save results
  save_result <- save_chunk_predictions(gtwr_results, output_filepath, column_config, reg.tv, pred_data_orig)
  
  chunk_end_time <- Sys.time()
  processing_time <- as.numeric(chunk_end_time - chunk_start_time, units = "mins")
  
  cat("  Saved", save_result$n_predictions, "predictions to:", basename(output_filepath), "\n")
  cat("  Processing time:", round(processing_time, 2), "minutes\n")
  
  # Log results
  if (!is.null(log_file)) {
    log_entry <- sprintf("%s,chunk_%03d,%d,%.2f,%.3f,%.3f,%.3f,%s\n",
                         format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
                         chunk_num,
                         save_result$n_predictions,
                         processing_time,
                         save_result$summary$min_pred,
                         save_result$summary$max_pred,
                         save_result$summary$mean_pred,
                         basename(output_filepath))
    cat(log_entry, file = log_file, append = TRUE)
  }
  
  return(list(
    status = "completed",
    chunk = chunk_num,
    file = output_filepath,
    n_predictions = save_result$n_predictions,
    processing_time = processing_time,
    summary = save_result$summary
  ))
}

# =============================================================================
# MAIN BATCH PROCESSING FUNCTION (UPDATED)
# =============================================================================

main_batch_gtwr_prediction <- function() {
  
  cat("=== Batch GTWR Prediction Processing ===\n")
  cat("Started at:", as.character(Sys.time()), "\n")
  cat("CRS Configuration:", crs_config$description, "\n")
  cat("EPSG Code:", crs_config$epsg_code, "\n\n")
  
  batch_start_time <- Sys.time()
  
  # Create output directory
  if (!dir.exists(output_folder)) {
    dir.create(output_folder, recursive = TRUE)
    cat("Created output directory:", output_folder, "\n")
  }
  
  # Initialize log file
  log_file <- NULL
  if (batch_config$save_log) {
    log_filename <- sprintf("gtwr_batch_log_%s.csv", format(Sys.time(), "%Y%m%d_%H%M%S"))
    log_filepath <- file.path(output_folder, log_filename)
    log_file <- file(log_filepath, "w")
    cat("timestamp,chunk,n_predictions,processing_time_mins,min_pred,max_pred,mean_pred,output_file\n", file = log_file)
    cat("Logging to:", log_filepath, "\n")
  }
  
  # FIXED: Load calibration data with proper spatial conversion
  cat("\nLoading calibration data...\n")
  calib_data <- load_calibration_data(input_config$calibration_file, column_config)
  n_calib <- nrow(calib_data)
  
  # Load dMat2 (one-time)
  cat("Loading dMat2 matrix...\n")
  dMat2 <- load_distance_matrix(input_config$dMat2_file, "dMat2",
                                expected_dims = c(n_calib, n_calib))
  cat("Loaded dMat2 matrix:", paste(dim(dMat2), collapse = "×"), "\n")
  
  # Extract time vector
  obs.tv <- calib_data[[column_config$calib_time_col]]
  
  # Create model formula
  formula <- create_formula(column_config)
  cat("Model formula:", deparse(formula), "\n")
  
  # Display CRS information
  cat("Calibration data CRS:", proj4string(calib_data), "\n")
  
  # Determine chunk range
  if (is.null(batch_config$end_chunk)) {
    # Auto-detect available chunks
    available_chunks <- c()
    for (i in 1:999) {
      chunk_files <- check_chunk_files(i, input_config)
      if (chunk_files$regpoints_exists && chunk_files$dMat1_exists) {
        available_chunks <- c(available_chunks, i)
      } else if (length(available_chunks) > 0) {
        break  # Stop when we hit the first missing chunk after finding some
      }
    }
    chunk_range <- available_chunks[available_chunks >= batch_config$start_chunk]
  } else {
    chunk_range <- batch_config$start_chunk:batch_config$end_chunk
  }
  
  cat("\nProcessing chunks:", paste(range(chunk_range), collapse = "-"), "\n")
  cat("Total chunks to process:", length(chunk_range), "\n\n")
  
  # Process chunks
  results <- list()
  successful_chunks <- 0
  failed_chunks <- 0
  
  for (chunk_num in chunk_range) {
    tryCatch({
      result <- process_chunk(chunk_num, calib_data, dMat2, formula, obs.tv, log_file)
      
      if (!is.null(result)) {
        results[[length(results) + 1]] <- result
        if (result$status == "completed") {
          successful_chunks <- successful_chunks + 1
        }
      } else {
        failed_chunks <- failed_chunks + 1
      }
      
      cat("\n")
      
    }, error = function(e) {
      cat("ERROR processing chunk", chunk_num, ":", e$message, "\n\n")
      failed_chunks <<- failed_chunks + 1
      
      if (!is.null(log_file)) {
        error_entry <- sprintf("%s,chunk_%03d,ERROR,%s\n",
                               format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
                               chunk_num,
                               gsub(",", ";", e$message))
        cat(error_entry, file = log_file, append = TRUE)
      }
    })
  }
  
  # Close log file
  if (!is.null(log_file)) {
    close(log_file)
  }
  
  # Final summary
  batch_end_time <- Sys.time()
  total_time <- as.numeric(batch_end_time - batch_start_time, units = "mins")
  
  cat("=== BATCH PROCESSING SUMMARY ===\n")
  cat("Total chunks processed:", length(chunk_range), "\n")
  cat("Successful chunks:", successful_chunks, "\n")
  cat("Failed chunks:", failed_chunks, "\n")
  cat("Total processing time:", round(total_time, 2), "minutes\n")
  cat("Output directory:", output_folder, "\n")
  cat("Completed at:", as.character(Sys.time()), "\n")
  
  if (successful_chunks > 0) {
    total_predictions <- sum(sapply(results, function(x) {
      if (x$status == "completed") x$n_predictions else 0
    }))
    cat("Total predictions generated:", total_predictions, "\n")
  }
  
  cat("\n✓ Batch GTWR prediction processing completed!\n")
  
  return(invisible(results))
}

# =============================================================================
# EXECUTION
# =============================================================================

if (!interactive()) {
  tryCatch({
    main_batch_gtwr_prediction()
  }, error = function(e) {
    cat("\n❌ BATCH ERROR:", e$message, "\n")
    cat("\nBatch processing failed. Please check:\n")
    cat("1. All required input files exist\n")
    cat("2. Output directory is writable\n")
    cat("3. Chunk numbering is consistent\n")
    cat("4. gtwr.predict.R file is available\n")
    cat("5. CRS configuration is correct\n")
    quit(status = 1)
  })
}