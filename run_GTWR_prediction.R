#!/usr/bin/env Rscript
# gtwr_prediction_workflow.R (CORRECTED VERSION)
# Complete GTWR prediction workflow using corrected algorithm

# Required libraries
library(GWmodel)
library(rhdf5)

# Source the CORRECTED GTWR prediction functions
source("gtwr.predict.R")

# =============================================================================
# CONFIGURATION SECTION
# =============================================================================

# Input file paths - MODIFY THESE PATHS
input_config <- list(
  # Pre-computed distance matrices (MANDATORY)
  dMat1_file = "test_dMat1_regpnt2.h5",  # Calibration to prediction distances
  dMat2_file = "test_dMat2.h5",  # Calibration to calibration distances
  
  # Data files
  calibration_file = "calibration_points.csv",     # Training data
  prediction_file = "regression_points_2.csv",       # Prediction locations
  
  # Output
  output_file = "gtwr_regpoints2_Prediction_Layer_1_new.csv"
)

# Column mapping configuration
column_config <- list(
  # Calibration data columns
  calib_x_col = "X_TWD97",
  calib_y_col = "Y_TWD97", 
  calib_time_col = "monthly",
  calib_response_col = "Layer_1",  # Dependent variable
  calib_predictors = c("CUMDISP"),  # Independent variables
  
  # Prediction data columns  
  pred_x_col = "X_TWD97",
  pred_y_col = "Y_TWD97",
  pred_time_col = "monthly"
)

# GTWR model parameters (ENHANCED)
gtwr_params <- list(
  st.bw = 23,                    # Spatiotemporal bandwidth
  kernel = "bisquare",           # Kernel function
  adaptive = TRUE,              # Fixed bandwidth
  calculate_variance = TRUE,     # Calculate prediction uncertainty
  
  # Additional GTWR parameters for corrected function
  p = 2,                         # Minkowski distance power
  theta = 0,                     # Coordinate rotation angle
  longlat = FALSE,               # Great circle distances
  lamda = 0.006,                  # Spatial-temporal weighting parameter
  t.units = "months",              # Time units
  ksi = 0                        # Spatial-temporal interaction parameter
)

# Validation configuration
validation_config <- list(
  enable_coefficient_validation = TRUE,
  reference_file = "gtwr_with_regpoints2_Layer_1_coefficients.csv"
)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

#' Load and validate HDF5 distance matrix
load_distance_matrix <- function(file_path, matrix_name = "dMat", expected_dims = NULL) {
  
  if (!file.exists(file_path)) {
    stop("ERROR: Distance matrix file not found: ", file_path)
  }
  
  cat("Loading distance matrix from:", file_path, "\n")
  
  # Try different possible matrix names in HDF5
  possible_names <- c(matrix_name, "dMat1", "dMat2", "dMat")
  matrix_loaded <- NULL
  
  for (name in possible_names) {
    if (name %in% h5ls(file_path)$name) {
      matrix_loaded <- h5read(file_path, name)
      cat("Found matrix:", name, "\n")
      break
    }
  }
  
  if (is.null(matrix_loaded)) {
    available_objects <- h5ls(file_path)$name
    stop("ERROR: No distance matrix found in ", file_path, 
         "\nAvailable objects: ", paste(available_objects, collapse = ", "))
  }
  
  # Validate dimensions if provided
  if (!is.null(expected_dims)) {
    actual_dims <- dim(matrix_loaded)
    if (!all(actual_dims == expected_dims)) {
      stop("ERROR: Matrix dimensions mismatch. Expected: ", 
           paste(expected_dims, collapse = "×"), 
           ", Got: ", paste(actual_dims, collapse = "×"))
    }
  }
  
  cat("Matrix dimensions:", paste(dim(matrix_loaded), collapse = "×"), "\n")
  cat("Distance range: [", round(min(matrix_loaded), 3), ", ", 
      round(max(matrix_loaded), 3), "]\n")
  
  return(matrix_loaded)
}

#' Load and validate calibration data
load_calibration_data <- function(file_path, column_config) {
  
  if (!file.exists(file_path)) {
    stop("ERROR: Calibration file not found: ", file_path)
  }
  
  cat("Loading calibration data from:", file_path, "\n")
  data <- read.csv(file_path, stringsAsFactors = FALSE)
  
  # Check required columns
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
  
  cat("Calibration points:", nrow(data), "\n")
  cat("Time range:", paste(range(data[[column_config$calib_time_col]]), collapse = " - "), "\n")
  
  return(data)
}

#' Load and validate prediction data
load_prediction_data <- function(file_path, column_config) {
  
  if (!file.exists(file_path)) {
    stop("ERROR: Prediction file not found: ", file_path)
  }
  
  cat("Loading prediction data from:", file_path, "\n")
  data <- read.csv(file_path, stringsAsFactors = FALSE)
  
  # Check required columns (predictors + coordinates + time)
  required_cols <- c(column_config$pred_x_col,
                     column_config$pred_y_col, 
                     column_config$pred_time_col,
                     column_config$calib_predictors)  # Must have same predictors
  
  missing_cols <- required_cols[!required_cols %in% names(data)]
  if (length(missing_cols) > 0) {
    stop("ERROR: Missing columns in prediction data: ", 
         paste(missing_cols, collapse = ", "))
  }
  
  cat("Prediction points:", nrow(data), "\n")
  cat("Time range:", paste(range(data[[column_config$pred_time_col]]), collapse = " - "), "\n")
  
  return(data)
}

#' Create formula from column configuration
create_formula <- function(column_config) {
  response <- column_config$calib_response_col
  predictors <- paste(column_config$calib_predictors, collapse = " + ")
  formula_str <- paste(response, "~", predictors)
  return(as.formula(formula_str))
}

#' Validate coefficient accuracy against reference
validate_coefficients <- function(gtwr_results, validation_config) {
  
  if (!validation_config$enable_coefficient_validation) {
    return(invisible(NULL))
  }
  
  if (!file.exists(validation_config$reference_file)) {
    cat("WARNING: Reference coefficient file not found, skipping validation\n")
    return(invisible(NULL))
  }
  
  cat("\n=== COEFFICIENT VALIDATION ===\n")
  cat("Comparing against reference:", validation_config$reference_file, "\n")
  
  # Load reference coefficients from gtwr with regression points
  ref_coeffs <- read.csv(validation_config$reference_file)
  
  # Extract prediction coefficients
  if (inherits(gtwr_results$SDF, "Spatial")) {
    pred_coeffs <- as.data.frame(gtwr_results$SDF)
  } else {
    pred_coeffs <- st_drop_geometry(gtwr_results$SDF)
  }
  
  # Match by coordinates and time
  merged_data <- merge(ref_coeffs, pred_coeffs, 
                      by.x = c(calib_x_col, calib_y_col, calib_time_col), 
                      by.y = c(pred_x_col, pred_y_col, "pred_time"))
  
  if (nrow(merged_data) == 0) {
    cat("WARNING: No matching points found for validation\n")
    return(invisible(NULL))
  }
  
  cat("Matched", nrow(merged_data), "points for validation\n")
  
  # Compare coefficients
  comparisons <- list(
    "Intercept" = c("Intercept", "Intercept_coef"),
    "indep_1" = c("indep_1", "indep_1_coef"),
    "indep_2" = c("indep_2", "indep_2_coef")
  )
  
  cat("\nCoefficient Accuracy Analysis:\n")
  cat("Coefficient    | Correlation | Max Diff   | Mean Diff  | RMSE\n")
  cat("---------------|-------------|------------|------------|----------\n")
  
  validation_results <- list()
  
  for (coef_name in names(comparisons)) {
    ref_col <- comparisons[[coef_name]][1]
    pred_col <- comparisons[[coef_name]][2]
    
    if (ref_col %in% names(merged_data) && pred_col %in% names(merged_data)) {
      ref_vals <- merged_data[[ref_col]]
      pred_vals <- merged_data[[pred_col]]
      
      # Calculate accuracy metrics
      correlation <- cor(ref_vals, pred_vals)
      differences <- abs(ref_vals - pred_vals)
      max_diff <- max(differences)
      mean_diff <- mean(differences)
      rmse <- sqrt(mean((ref_vals - pred_vals)^2))
      
      cat(sprintf("%-14s | %10.4f | %10.4f | %10.4f | %8.4f\n", 
                  coef_name, correlation, max_diff, mean_diff, rmse))
      
      validation_results[[coef_name]] <- list(
        correlation = correlation,
        max_diff = max_diff,
        mean_diff = mean_diff,
        rmse = rmse
      )
    }
  }
  
  # Overall assessment
  correlations <- sapply(validation_results, function(x) x$correlation)
  avg_correlation <- mean(correlations)
  
  cat("\nValidation Summary:\n")
  cat("Average correlation:", round(avg_correlation, 4), "\n")
  
  if (avg_correlation > 0.99) {
    cat("✓ EXCELLENT: Coefficients match reference with high accuracy\n")
  } else if (avg_correlation > 0.95) {
    cat("✓ GOOD: Coefficients show strong agreement with reference\n")
  } else if (avg_correlation > 0.8) {
    cat("⚠ MODERATE: Some differences detected, review algorithm\n")
  } else {
    cat("❌ POOR: Significant differences detected, algorithm correction needed\n")
  }
  
  return(validation_results)
}

#' Save prediction results with comprehensive metadata
save_predictions <- function(gtwr_results, output_file, column_config, reg.tv, pred_data_orig) {
  
  cat("Saving predictions to:", output_file, "\n")
  
  # Extract results from GTWR object
  if (inherits(gtwr_results$SDF, "Spatial")) {
    pred_results <- as(gtwr_results$SDF, "data.frame")
    coords <- coordinates(gtwr_results$SDF)
  } else {
    pred_results <- st_drop_geometry(gtwr_results$SDF)
    coords <- st_coordinates(gtwr_results$SDF)
  }
  
  # Add coordinate columns
  pred_results$X_coord <- coords[, 1]
  pred_results$Y_coord <- coords[, 2]
  
  # Add time values for prediction points
  pred_results$pred_time <- reg.tv
  
  # Add predictor values from original prediction data
  for (predictor in column_config$calib_predictors) {
    if (predictor %in% names(pred_data_orig)) {
      pred_results[[paste0("pred_", predictor)]] <- pred_data_orig[[predictor]]
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
  
  # Add comprehensive metadata
  pred_results$bandwidth <- gtwr_results$GTW.arguments$st.bw
  pred_results$kernel <- gtwr_results$GTW.arguments$kernel
  pred_results$prediction_time <- Sys.time()
  
  write.csv(pred_results, output_file, row.names = FALSE)
  
  cat("Saved", nrow(pred_results), "predictions\n")
  
  # Print summary statistics
  if ("gtwr_prediction" %in% names(pred_results)) {
    cat("\nPrediction Summary:\n")
    cat("Min:", round(min(pred_results$gtwr_prediction, na.rm = TRUE), 3), "\n")
    cat("Max:", round(max(pred_results$gtwr_prediction, na.rm = TRUE), 3), "\n")
    cat("Mean:", round(mean(pred_results$gtwr_prediction, na.rm = TRUE), 3), "\n")
    
    if ("predict.var" %in% names(pred_results)) {
      cat("\nUncertainty Summary:\n")
      cat("Mean variance:", round(mean(pred_results$predict.var, na.rm = TRUE), 3), "\n")
      cat("Mean std error:", round(mean(sqrt(pred_results$predict.var), na.rm = TRUE), 3), "\n")
    }
  }
  
  return(output_file)
}

# =============================================================================
# MAIN PREDICTION WORKFLOW
# =============================================================================

main_gtwr_prediction <- function() {
  
  cat("=== GTWR Prediction Workflow (CORRECTED) ===\n")
  cat("Started at:", as.character(Sys.time()), "\n\n")
  
  start_time <- Sys.time()
  
  # Step 1: Validate critical inputs
  cat("Step 1: Validating distance matrix files (MANDATORY)\n")
  
  if (!file.exists(input_config$dMat1_file)) {
    stop("CRITICAL ERROR: dMat1 file missing: ", input_config$dMat1_file, 
         "\nThis program requires pre-computed distance matrices!")
  }
  
  if (!file.exists(input_config$dMat2_file)) {
    stop("CRITICAL ERROR: dMat2 file missing: ", input_config$dMat2_file,
         "\nThis program requires pre-computed distance matrices!")
  }
  
  cat("✓ Distance matrix files found\n")
  
  # Step 2: Load calibration data
  cat("\nStep 2: Loading calibration data\n")
  calib_data <- load_calibration_data(input_config$calibration_file, column_config)
  n_calib <- nrow(calib_data)
  
  # Step 3: Load prediction data  
  cat("\nStep 3: Loading prediction data\n")
  pred_data <- load_prediction_data(input_config$prediction_file, column_config)
  pred_data_orig <- pred_data  # Keep original before converting to spatial
  n_pred <- nrow(pred_data)
  
  # Step 4: Load distance matrices with dimension validation
  cat("\nStep 4: Loading pre-computed distance matrices\n")
  
  # Load dMat1 (calibration to prediction)
  dMat1 <- load_distance_matrix(input_config$dMat1_file, "dMat1", 
                                expected_dims = c(n_calib, n_pred))
  
  # Load dMat2 (calibration to calibration) - only if calculating variance
  dMat2 <- NULL
  if (gtwr_params$calculate_variance) {
    dMat2 <- load_distance_matrix(input_config$dMat2_file, "dMat2",
                                  expected_dims = c(n_calib, n_calib))
  }
  
  cat("✓ Distance matrices loaded successfully\n")
  
  # Step 5: Prepare spatial data objects
  cat("\nStep 5: Preparing spatial data objects\n")
  
  # Extract time vectors BEFORE converting to spatial
  obs.tv <- calib_data[[column_config$calib_time_col]]
  reg.tv <- pred_data[[column_config$pred_time_col]]
  
  # Convert to SpatialPointsDataFrame (required by GWmodel functions)
  coordinates(calib_data) <- c(column_config$calib_x_col, column_config$calib_y_col)
  coordinates(pred_data) <- c(column_config$pred_x_col, column_config$pred_y_col)
  
  # Step 6: Create model formula
  formula <- create_formula(column_config)
  cat("Model formula:", deparse(formula), "\n")
  
  # Step 7: Run CORRECTED GTWR prediction
  cat("\nStep 6: Running CORRECTED GTWR prediction\n")
  cat("Using corrected algorithm that matches gtwr.R coefficients\n")
  cat("Bandwidth:", gtwr_params$st.bw, "\n")
  cat("Kernel:", gtwr_params$kernel, "\n")
  cat("Lambda:", gtwr_params$lamda, "\n")
  cat("Calculate variance:", gtwr_params$calculate_variance, "\n")
  
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
    calculate.var = gtwr_params$calculate_variance
  )
  
  cat("✓ GTWR prediction completed\n")
  
  # Step 8: Validate coefficients against reference
  validation_results <- validate_coefficients(gtwr_results, validation_config)
  
  # Step 9: Save results
  cat("\nStep 7: Saving prediction results\n")
  output_file <- save_predictions(gtwr_results, input_config$output_file, column_config, reg.tv, pred_data_orig)
  
  # Step 10: Final summary
  total_time <- as.numeric(Sys.time() - start_time, units = "mins")
  
  cat("\n=== PREDICTION SUMMARY ===\n")
  cat("Calibration points:", n_calib, "\n")
  cat("Prediction points:", n_pred, "\n")
  cat("Total processing time:", round(total_time, 2), "minutes\n")
  cat("Output file:", output_file, "\n")
  cat("Algorithm version: CORRECTED (matches gtwr.R)\n")
  cat("Completed at:", as.character(Sys.time()), "\n")
  
  cat("\n✓ GTWR prediction workflow completed successfully!\n")
  
  return(invisible(gtwr_results))
}

# =============================================================================
# EXECUTION
# =============================================================================

# Run the workflow (only if script is called directly)
if (!interactive()) {
  tryCatch({
    main_gtwr_prediction()
  }, error = function(e) {
    cat("\n❌ ERROR:", e$message, "\n")
    cat("\nWorkflow failed. Please check:\n")
    cat("1. Distance matrix files (dMat1 and dMat2) exist\n")
    cat("2. CSV data files exist with correct column names\n") 
    cat("3. Matrix dimensions match data dimensions\n")
    cat("4. gtwr.predict.R file is available\n")
    quit(status = 1)
  })
}