# ==============================================================================
# Title:        Geographically and Temporally Weighted Regression (GTWR) Analysis
# Description:  A user-friendly script to load data, find an optimal
#               spatio-temporal bandwidth, and run a GTWR model. All settings
#               are centralized in the CONFIG section for easy control.
# Author:       Refactored by Gemini for clarity and usability
# Date:         2025-06-11
# ==============================================================================


# ==============================================================================
# SECTION 1: USER CONFIGURATION
# ------------------------------------------------------------------------------
# All parameters for the analysis should be set in this section.
# ==============================================================================
cat("Step 1: Loading configuration settings...\n")

CONFIG <- list(
  # --- File Paths ---
  input_csv_path = "calibration_points.csv", # Corrected from your log
  output_directory = ".", # Corrected from your log

  # --- Model Parameters ---
  formula_string = "Layer_1 ~ CUMDISP", # Corrected from your log

  lambda_param = 0.006,
  ksi_param_degrees = 0,

  # --- GTWR Settings ---
  # MODIFIED: Renamed from 'kernel_function' to 'kernel_method' to avoid name clash
  kernel_method = "bisquare",
  
  bandwidth_method = "AICc",
  
  # Temporal units. "months" is a common setting for monthly data.
  temporal_units = "months",

  # --- Data Column Names ---
  time_field_name = "monthly",
  x_coord_name = "X_TWD97",
  y_coord_name = "Y_TWD97"
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

input_data <- read.csv(CONFIG$input_csv_path)
cat("   - Successfully loaded dataset with dimensions:", dim(input_data)[1], "rows,", dim(input_data)[2], "columns.\n")
cat("   - Column names:", paste(names(input_data), collapse=", "), "\n")

coords <- as.matrix(input_data[, c(CONFIG$x_coord_name, CONFIG$y_coord_name)])
time_values <- input_data[[CONFIG$time_field_name]]

cat("   - Converting data to spatial format (SpatialPointsDataFrame)...\n")
spatial_input_data <- SpatialPointsDataFrame(
  coords = coords,
  data = input_data,
  proj4string = CRS("EPSG:3826")
)

# ==============================================================================
# SECTION 4: OPTIMAL BANDWIDTH SELECTION
# ==============================================================================
cat("\nStep 4: Finding the optimal spatio-temporal bandwidth. This may take a while...\n")

bandwidth_start_time <- Sys.time()
model_formula <- as.formula(CONFIG$formula_string)
ksi_param_radians <- CONFIG$ksi_param_degrees * (pi / 180)

optimal_bandwidth <- bw.gtwr(
  formula = model_formula,
  data = spatial_input_data,
  obs.tv = time_values,
  approach = CONFIG$bandwidth_method,
  # MODIFIED: Using the new, safe variable name here
  kernel = CONFIG$kernel_method,
  adaptive = TRUE,
  lamda = CONFIG$lambda_param,
  ksi = ksi_param_radians,
  t.units = CONFIG$temporal_units
)

bandwidth_end_time <- Sys.time()
elapsed_time <- difftime(bandwidth_end_time, bandwidth_start_time)

cat("   - Bandwidth optimization complete.\n")
cat("   - Time elapsed:", format(elapsed_time), "\n")
cat("   - Optimal adaptive bandwidth found:", optimal_bandwidth, "neighbors\n")


# ==============================================================================
# SECTION 5: GTWR MODEL CALIBRATION
# ==============================================================================
cat("\nStep 5: Calibrating final GTWR model with the optimal bandwidth...\n")

gtwr_model <- gtwr(
  formula = model_formula,
  data = spatial_input_data,
  obs.tv = time_values,
  st.bw = optimal_bandwidth,
  # MODIFIED: Using the new, safe variable name here as well
  kernel = CONFIG$kernel_method,
  adaptive = TRUE,
  lamda = CONFIG$lambda_param,
  ksi = ksi_param_radians,
  t.units = CONFIG$temporal_units
)

cat("   - Model calibration complete.\n")

# ==============================================================================
# SECTION 6: RESULTS AND OUTPUT
# ==============================================================================
cat("\nStep 6: Displaying and saving results...\n")

cat("\n--- GWR Model Summary ---\n")
print(gtwr_model)

base_filename <- sprintf("gtwr_%s_kernel-%s_lambda-%s_bw-%d",
                         all.vars(model_formula)[1],
                         # MODIFIED: Using the new variable name for the filename
                         CONFIG$kernel_method,
                         gsub("\\.", "d", as.character(CONFIG$lambda_param)),
                         optimal_bandwidth)

rds_filepath <- file.path(CONFIG$output_directory, paste0(base_filename, ".rds"))
saveRDS(gtwr_model, rds_filepath)
cat("\n   - Full model object saved to:", rds_filepath, "\n")

coefficient_data <- if(inherits(gtwr_model$SDF, "sf")) {
  sf::st_drop_geometry(gtwr_model$SDF)
} else {
  as.data.frame(gtwr_model$SDF)
}

csv_filepath <- file.path(CONFIG$output_directory, paste0(base_filename, "_coefficients.csv"))
write.csv(coefficient_data, csv_filepath, row.names = FALSE)
cat("   - Model coefficients and diagnostics saved to:", csv_filepath, "\n")

cat("\n--- Script finished successfully ---\n")
