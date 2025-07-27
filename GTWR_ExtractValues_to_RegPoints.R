#!/usr/bin/env Rscript
# =============================================================================
# GTWR Analysis Tool - Simple Coordinates Version
# =============================================================================
# This script performs GTWR analysis treating coordinates as simple numeric values

# =============================================================================
# 1. USER CONFIGURATION SECTION
# =============================================================================

# File Paths
CALIBRATION_FILE <- "calibration_points.csv"
REGRESSION_POINTS_FILE <- "regression_points_2.csv"
SAVED_MODEL_FILE <- "gtwr_Layer_1_kernel-bisquare_lambda-0d006_bw-23.rds"  # Set to NULL if no saved model

# Coordinate Columns
COORDINATE_COLUMNS <- list(
  x = "X_TWD97",
  y = "Y_TWD97"
)

# Model Configuration
TARGET_VARIABLE <- "Layer_1"
PREDICTOR_VARIABLES <- c("CUMDISP")
TIME_VARIABLE <- "monthly"

# GTWR Parameters
GTWR_PARAMS <- list(
  st_bw = 23,
  kernel = "bisquare",
  adaptive = TRUE,
  lamda = 0.006,
  t_units = "months",
  ksi = 0,
  p = 2,
  theta = 0,
  longlat = FALSE
)

# Output Options
OUTPUT_RESULTS <- TRUE
SAVE_MODEL <- TRUE
OUTPUT_MODEL_FILE <- "gtwr_with_regpoints2_Layer_1.rds"

# =============================================================================
# 2. LOAD LIBRARIES
# =============================================================================

required_packages <- c("GWmodel", "sf", "sp")

for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    install.packages(pkg)
    library(pkg, character.only = TRUE)
  }
}

# =============================================================================
# 3. HELPER FUNCTIONS
# =============================================================================

validate_file <- function(file_path, file_description) {
  if (is.null(file_path) || !file.exists(file_path)) {
    stop(paste("Error:", file_description, "not found at:", file_path))
  }
  cat("✓", file_description, "found\n")
}

load_and_validate_csv <- function(file_path, required_columns, file_description) {
  cat("Loading", file_description, "...\n")
  data <- read.csv(file_path, stringsAsFactors = FALSE)
  cat("  Dimensions:", nrow(data), "×", ncol(data), "\n")
  
  missing_cols <- setdiff(required_columns, names(data))
  if (length(missing_cols) > 0) {
    stop(paste("Missing columns:", paste(missing_cols, collapse = ", ")))
  }
  
  cat("  ✓ Required columns present\n\n")
  return(data)
}

create_spatial_dataframe <- function(data, coord_cols) {
  coords <- data[, c(coord_cols$x, coord_cols$y)]
  
  if (any(is.na(coords))) {
    warning("Some coordinates are missing")
  }
  
  # Create SpatialPointsDataFrame without projection
  sp_data <- SpatialPointsDataFrame(
    coords = coords,
    data = data
  )
  
  return(sp_data)
}

# =============================================================================
# 4. DATA LOADING
# =============================================================================

cat("=== GTWR Analysis Started ===\n\n")
cat("Step 1: Loading data\n")
cat("-------------------\n")

calibration_required_cols <- c(COORDINATE_COLUMNS$x, COORDINATE_COLUMNS$y, 
                             TARGET_VARIABLE, PREDICTOR_VARIABLES, TIME_VARIABLE)
regression_required_cols <- c(COORDINATE_COLUMNS$x, COORDINATE_COLUMNS$y, TIME_VARIABLE)

validate_file(CALIBRATION_FILE, "Calibration file")
validate_file(REGRESSION_POINTS_FILE, "Regression points file")

calibration_data <- load_and_validate_csv(CALIBRATION_FILE, calibration_required_cols, "calibration data")
regression_data <- load_and_validate_csv(REGRESSION_POINTS_FILE, regression_required_cols, "regression points")

# =============================================================================
# 5. CREATE SPATIAL OBJECTS
# =============================================================================

cat("Step 2: Creating spatial objects\n")
cat("-------------------------------\n")

calibration_sp <- create_spatial_dataframe(calibration_data, COORDINATE_COLUMNS)
regression_sp <- create_spatial_dataframe(regression_data, COORDINATE_COLUMNS)

cat("✓ Calibration points:", nrow(calibration_sp), "\n")
cat("✓ Regression points:", nrow(regression_sp), "\n\n")

# =============================================================================
# 6. PREPARE TIME VARIABLES
# =============================================================================

cat("Step 3: Preparing time variables\n")
cat("-------------------------------\n")

obs_tv <- calibration_sp[[TIME_VARIABLE]]
reg_tv <- regression_sp[[TIME_VARIABLE]]

cat("✓ Time variables extracted\n")
cat("  Observation range:", min(obs_tv, na.rm = TRUE), "to", max(obs_tv, na.rm = TRUE), "\n")
cat("  Regression range:", min(reg_tv, na.rm = TRUE), "to", max(reg_tv, na.rm = TRUE), "\n\n")

# =============================================================================
# 7. LOAD EXISTING MODEL (OPTIONAL)
# =============================================================================

if (!is.null(SAVED_MODEL_FILE) && file.exists(SAVED_MODEL_FILE)) {
  cat("Step 4: Loading existing model\n")
  cat("-----------------------------\n")
  existing_model <- readRDS(SAVED_MODEL_FILE)
  cat("✓ Model loaded\n")
  if (OUTPUT_RESULTS) print(existing_model)
  cat("\n")
}

# =============================================================================
# 8. BUILD AND RUN MODEL
# =============================================================================

cat("Step 5: Running GTWR analysis\n")
cat("-----------------------------\n")

formula_string <- paste(TARGET_VARIABLE, "~", paste(PREDICTOR_VARIABLES, collapse = " + "))
model_formula <- as.formula(formula_string)

cat("Formula:", formula_string, "\n")
cat("Bandwidth:", GTWR_PARAMS$st_bw, "(", ifelse(GTWR_PARAMS$adaptive, "adaptive", "fixed"), ")\n")
cat("Kernel:", GTWR_PARAMS$kernel, "\n\n")

cat("Fitting model...\n")
start_time <- Sys.time()

gtwr_model <- gtwr(
  formula = model_formula,
  data = calibration_sp,
  regression.points = regression_sp,
  obs.tv = obs_tv,
  reg.tv = reg_tv,
  st.bw = GTWR_PARAMS$st_bw,
  kernel = GTWR_PARAMS$kernel,
  adaptive = GTWR_PARAMS$adaptive,
  p = GTWR_PARAMS$p,
  theta = GTWR_PARAMS$theta,
  longlat = GTWR_PARAMS$longlat,
  lamda = GTWR_PARAMS$lamda,
  t.units = GTWR_PARAMS$t_units,
  ksi = GTWR_PARAMS$ksi
)

end_time <- Sys.time()
elapsed_time <- end_time - start_time

cat("✓ Model fitted successfully!\n")
cat("  Time:", round(elapsed_time, 2), attr(elapsed_time, "units"), "\n\n")

# =============================================================================
# 9. RESULTS AND SAVE
# =============================================================================

if (OUTPUT_RESULTS) {
  cat("Step 6: Results\n")
  cat("=============\n")
  print(gtwr_model)
  cat("\n")
}

if (SAVE_MODEL) {
  cat("Step 7: Saving\n")
  cat("------------\n")
  saveRDS(gtwr_model, OUTPUT_MODEL_FILE)
  cat("✓ Model saved:", OUTPUT_MODEL_FILE, "\n")
  
  if (inherits(gtwr_model$SDF, "SpatialPointsDataFrame")) {
    coeff_file <- gsub("\\.rds$", "_coefficients.csv", OUTPUT_MODEL_FILE)
    write.csv(as.data.frame(gtwr_model$SDF), coeff_file, row.names = FALSE)
    cat("✓ Coefficients saved:", coeff_file, "\n")
  }
}

cat("\n=== Analysis Complete ===\n")
cat("Calibration points:", nrow(calibration_sp), "\n")
cat("Regression points:", nrow(regression_sp), "\n")
cat("Computation time:", round(elapsed_time, 2), attr(elapsed_time, "units"), "\n")

# Check if diagnostic information is available (only when regression points = calibration points)
if (exists("gtwr_model") && !is.null(gtwr_model$GTW.diagnostic) && is.list(gtwr_model$GTW.diagnostic)) {
  cat("R-squared:", round(gtwr_model$GTW.diagnostic$gw.R2, 4), "\n")
  cat("AICc:", round(gtwr_model$GTW.diagnostic$AICc, 2), "\n")
} else {
  cat("Note: Diagnostic information not available when using separate regression points\n")
}