# ==============================================================================
# GTWR Parameter Optimization Script
# Modify parameters below, then run the entire script
# ==============================================================================

# ==============================================================================
# USER INPUT SECTION - MODIFY THESE PARAMETERS
# ==============================================================================

# --- Input Data Configuration ---
INPUT_FILE <- "20251002_GTWR_InputData_MLCW_InSAR_Layer_4.csv"
TARGET_VARIABLE <- "Layer_4"
PREDICTORS <- c("CUMDISP")
TIME_COLUMN <- "monthly"
X_COORD <- "X_TWD97"
Y_COORD <- "Y_TWD97"
CRS_EPSG <- 3826

# --- Parameter Testing Ranges ---
# LAMBDA_MIN <- 0.001
# LAMBDA_MAX <- 0.009
# LAMBDA_STEP <- 0.001

# LAMBDA_MIN <- 0.01
# LAMBDA_MAX <- 0.1
# LAMBDA_STEP <- 0.01

LAMBDA_MIN <- 0.2
LAMBDA_MAX <- 1
LAMBDA_STEP <- 0.1

KSI_MIN_DEGREES <- 15
KSI_MAX_DEGREES <- 90
KSI_STEP_DEGREES <- 15

# --- GTWR Model Settings ---
KERNEL <- "bisquare"
ADAPTIVE_BANDWIDTH <- TRUE
BANDWIDTH_APPROACH <- "CV" # "CV"  # or "AIC"
TIME_UNITS <- "months"

# --- Processing Options ---
MAX_MODELS <- 50           # Limit total models to prevent excessive runtime
VERBOSE_OUTPUT <- TRUE     # Show progress messages
SAVE_ALL_MODELS <- TRUE    # Save individual model files

# --- Output Settings ---
OUTPUT_DIR <- sprintf("gtwr_model_%s_%s", BANDWIDTH_APPROACH, TARGET_VARIABLE)
OUTPUT_PREFIX <- "gtwr_analysis"

# ==============================================================================
# PROGRAM EXECUTION - DO NOT MODIFY BELOW THIS LINE
# ==============================================================================

# Load required libraries
library(GWmodel)
library(sf)
library(sp)

# Start timing
script_start_time <- Sys.time()

# Setup output directory
if (!dir.exists(OUTPUT_DIR)) {
  dir.create(OUTPUT_DIR, recursive = TRUE)
}

# Load and prepare data
tryCatch({
  input_data <- read.csv(INPUT_FILE)
  required_cols <- c(X_COORD, Y_COORD, TIME_COLUMN, TARGET_VARIABLE, PREDICTORS)
  missing_cols <- setdiff(required_cols, names(input_data))
  if (length(missing_cols) > 0) {
    stop("Missing required columns: ", paste(missing_cols, collapse = ", "))
  }
  
  coords <- as.matrix(input_data[, c(X_COORD, Y_COORD)])
  spatial_data <- SpatialPointsDataFrame(
    coords = coords,
    data = input_data,
    proj4string = CRS(paste0("EPSG:", CRS_EPSG))
  )
  time_values <- spatial_data[[TIME_COLUMN]]
  
}, error = function(e) {
  stop("Error loading data: ", e$message)
})

# Create formula
formula_str <- paste(TARGET_VARIABLE, "~", paste(PREDICTORS, collapse = " + "))
formula_obj <- as.formula(formula_str)

# Generate parameter combinations
lambda_seq <- seq(from = LAMBDA_MIN, to = LAMBDA_MAX, by = LAMBDA_STEP)

# Handle special case where ksi step is 0 (single value)
if (KSI_STEP_DEGREES == 0 || KSI_MIN_DEGREES == KSI_MAX_DEGREES) {
  ksi_deg_seq <- KSI_MIN_DEGREES
} else {
  ksi_deg_seq <- seq(from = KSI_MIN_DEGREES, to = KSI_MAX_DEGREES, by = KSI_STEP_DEGREES)
}

ksi_rad_seq <- ksi_deg_seq * (pi / 180)

param_grid <- expand.grid(
  lambda = lambda_seq,
  ksi_deg = ksi_deg_seq,
  ksi_rad = ksi_rad_seq,
  stringsAsFactors = FALSE
)

if (nrow(param_grid) > MAX_MODELS) {
  param_grid <- param_grid[1:MAX_MODELS, ]
}

# Brief startup message
cat("GTWR Optimization: ", nrow(param_grid), " models to test\n")

# Add these lines:
cat("Input file:", INPUT_FILE, "\n")
cat("Target variable:", TARGET_VARIABLE, "\n")
cat("Formula:", formula_str, "\n")
cat("Lambda range:", LAMBDA_MIN, "to", LAMBDA_MAX, "step", LAMBDA_STEP, "\n")
cat("Ksi range:", KSI_MIN_DEGREES, "to", KSI_MAX_DEGREES, "degrees\n")
cat("Kernel:", KERNEL, "| Bandwidth:", BANDWIDTH_APPROACH, "\n")
cat("Starting models...\n\n")

# Initialize results storage (existing line)
results_list <- list()
# Main optimization loop
for (i in 1:nrow(param_grid)) {
  
  current_params <- param_grid[i, ]
  model_start_time <- Sys.time()
  
  tryCatch({
    # Bandwidth optimization
    optimal_bw <- bw.gtwr(
      formula = formula_obj,
      data = spatial_data,
      obs.tv = time_values,
      approach = BANDWIDTH_APPROACH,
      kernel = KERNEL,
      adaptive = ADAPTIVE_BANDWIDTH,
      lamda = current_params$lambda,
      ksi = current_params$ksi_rad,
      t.units = TIME_UNITS,
      verbose = TRUE
    )
    
    if (is.null(optimal_bw) || is.na(optimal_bw)) {
      next
    }

    # # Calculate CV score for the optimal bandwidth
    # cv_score <- gtwr.cv(
    #   bw = optimal_bw,
    #   X = model.matrix(formula_obj, spatial_data),
    #   Y = spatial_data[[TARGET_VARIABLE]],
    #   kernel = KERNEL,
    #   adaptive = ADAPTIVE_BANDWIDTH,
    #   dp.locat = coordinates(spatial_data),
    #   obs.tv = time_values,
    #   lamda = current_params$lambda,
    #   ksi = current_params$ksi_rad,
    #   t.units = TIME_UNITS,
    #   verbose = FALSE
    # )
    
    # Model calibration
    gtwr_model <- suppressWarnings({
      gtwr(
        formula = formula_obj,
        data = spatial_data,
        obs.tv = time_values,
        st.bw = optimal_bw,
        kernel = KERNEL,
        adaptive = ADAPTIVE_BANDWIDTH,
        lamda = current_params$lambda,
        ksi = current_params$ksi_rad,
        t.units = TIME_UNITS
      )
    })
    
    # Extract diagnostics
    diagnostics <- gtwr_model$GTW.diagnostic
    if (is.null(diagnostics) || is.null(diagnostics$AICc)) {
      next
    }
    
    # Save model with specific naming convention
    model_filename <- sprintf("%s_%s_%s_lambda%s_ksi%s.rds",
                              KERNEL, TARGET_VARIABLE, BANDWIDTH_APPROACH,
                              gsub("\\.", "d", as.character(current_params$lambda)),
                              gsub("\\.", "d", as.character(current_params$ksi_deg)))
    model_filepath <- file.path(OUTPUT_DIR, model_filename)
    saveRDS(gtwr_model, model_filepath)
    
    # Store results
    total_time <- as.numeric(difftime(Sys.time(), model_start_time, units = "secs"))
    
    result <- data.frame(
      Model_ID = i,
      Lambda = current_params$lambda,
      Ksi_Degrees = current_params$ksi_deg,
      Bandwidth = optimal_bw,
      # CV_Score = cv_score,  # Add this line
      AICc = diagnostics$AICc,
      AIC = diagnostics$AIC,
      BIC = ifelse(is.null(diagnostics$BIC), NA, diagnostics$BIC),
      R_Squared = diagnostics$gw.R2,
      Adj_R_Squared = diagnostics$gwR2.adj,
      RSS = diagnostics$RSS.gw,
      ENP = diagnostics$enp,
      EDF = diagnostics$edf,
      Total_Time_Sec = total_time,
      Model_File = model_filename,
      stringsAsFactors = FALSE
    )
    
    results_list[[length(results_list) + 1]] <- result
    
    # Minimal completion message
    cat(sprintf("[%d/%d] λ=%.3f ψ=%d° BW=%d AICc=%.1f (%.1fs)\n", 
                i, nrow(param_grid), current_params$lambda, current_params$ksi_deg, 
                optimal_bw, diagnostics$AICc, total_time))
    
  }, error = function(e) {
    cat(sprintf("[%d/%d] λ=%.3f ψ=%d° FAILED\n", 
                i, nrow(param_grid), current_params$lambda, current_params$ksi_deg))
  })
}

# Save comprehensive results
if (length(results_list) > 0) {
  
  results_df <- do.call(rbind, results_list)
  rownames(results_df) <- NULL
  
  total_time <- difftime(Sys.time(), script_start_time, units = "mins")
  best_aicc <- results_df[which.min(results_df$AICc), ]
  best_r2 <- results_df[which.max(results_df$R_Squared), ]
  
  # Save results with timestamp
  timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
  results_file <- file.path(OUTPUT_DIR, sprintf("results_%s.csv", timestamp))
  summary_file <- file.path(OUTPUT_DIR, sprintf("summary_%s.txt", timestamp))
  
  write.csv(results_df, results_file, row.names = FALSE)
  
  # Create summary
  summary_text <- c(
    "GTWR Parameter Optimization Summary",
    paste("Completed:", Sys.time()),
    paste("Runtime:", round(total_time, 2), "minutes"),
    paste("Models tested:", nrow(results_df)),
    "",
    "Best AICc Model:",
    sprintf("  Lambda: %.3f, Ksi: %d°, AICc: %.2f, R²: %.3f", 
            best_aicc$Lambda, best_aicc$Ksi_Degrees, best_aicc$AICc, best_aicc$R_Squared),
    sprintf("  File: %s", best_aicc$Model_File),
    "",
    "Best R² Model:",
    sprintf("  Lambda: %.3f, Ksi: %d°, AICc: %.2f, R²: %.3f", 
            best_r2$Lambda, best_r2$Ksi_Degrees, best_r2$AICc, best_r2$R_Squared),
    sprintf("  File: %s", best_r2$Model_File)
  )
  
  writeLines(summary_text, summary_file)
  
  # Final summary
  cat("\nCompleted:", nrow(results_df), "models,", round(total_time, 2), "min\n")
  cat("Best AICc:", round(best_aicc$AICc, 2), "(λ", best_aicc$Lambda, "ψ", best_aicc$Ksi_Degrees, "°)\n")
  cat("Results:", basename(results_file), "\n")
  
} else {
  cat("\nNo successful models generated\n")
}