# ==============================================================================
# GTWR Cross-Validation Results Visualization
# ==============================================================================

# Load required libraries
suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(gridExtra)
  library(viridis)
  library(corrplot)
})

# ==============================================================================
# CONFIGURATION
# ==============================================================================

CONFIG <- list(
  # Input file from validation script
  validation_results_file = "GTWR_Validation/All_Layer/combined_validation_results.csv",
  
  # Output settings
  output_dir = "GTWR_Validation/All_Layer/validation_figs",
  save_plots = TRUE,
  plot_width = 14,
  plot_height = 10,
  
  # Variable names (adjust based on your data)
  response_var = "All_Layer",
  predictor_var = "CUMDISP"
)

# ==============================================================================
# DATA LOADING AND PREPROCESSING
# ==============================================================================

cat("Loading validation results...\n")
validation_data <- read.csv(CONFIG$validation_results_file, stringsAsFactors = FALSE)

# Create output directory
if (CONFIG$save_plots && !dir.exists(CONFIG$output_dir)) {
  dir.create(CONFIG$output_dir, recursive = TRUE)
}

# Calculate derived metrics
validation_data <- validation_data %>%
  mutate(
    observed = get(paste0("Observed_", CONFIG$response_var)),
    predicted = get(paste0("Predicted_", CONFIG$response_var)),
    residual = observed - predicted,
    abs_error = abs(residual),
    sq_error = residual^2
  )

cat("Loaded", nrow(validation_data), "validation predictions\n")
cat("Stations:", length(unique(validation_data$excluded_station)), "\n")
cat("Time periods:", length(unique(validation_data$time_stamp)), "\n")

# ==============================================================================
# CALCULATE PERFORMANCE METRICS
# ==============================================================================

# Overall metrics
overall_metrics <- validation_data %>%
  summarise(
    RMSE = sqrt(mean(sq_error, na.rm = TRUE)),
    MAE = mean(abs_error, na.rm = TRUE),
    R2 = cor(observed, predicted, use = "complete.obs")^2,
    bias = mean(residual, na.rm = TRUE),
    n_predictions = n()
  )

# Station-level metrics
station_metrics <- validation_data %>%
  group_by(excluded_station) %>%
  summarise(
    RMSE = sqrt(mean(sq_error, na.rm = TRUE)),
    MAE = mean(abs_error, na.rm = TRUE),
    R2 = cor(observed, predicted, use = "complete.obs")^2,
    bias = mean(residual, na.rm = TRUE),
    n_predictions = n(),
    .groups = 'drop'
  )

# Temporal metrics
temporal_metrics <- validation_data %>%
  group_by(time_stamp) %>%
  summarise(
    RMSE = sqrt(mean(sq_error, na.rm = TRUE)),
    MAE = mean(abs_error, na.rm = TRUE),
    R2 = cor(observed, predicted, use = "complete.obs")^2,
    n_predictions = n(),
    .groups = 'drop'
  )

# Print summary
cat("\n=== VALIDATION SUMMARY ===\n")
cat("Overall RMSE:", round(overall_metrics$RMSE, 3), "\n")
cat("Overall MAE:", round(overall_metrics$MAE, 3), "\n")
cat("Overall R²:", round(overall_metrics$R2, 3), "\n")
cat("Overall Bias:", round(overall_metrics$bias, 3), "\n")

# ==============================================================================
# EXPORT DATA TABLES FOR EXTERNAL ANALYSIS
# ==============================================================================

if (CONFIG$save_plots) {
  # Export overall metrics
  write.csv(overall_metrics, file.path(CONFIG$output_dir, "overall_metrics_table.csv"), row.names = FALSE)
  
  # Export station metrics
  write.csv(station_metrics, file.path(CONFIG$output_dir, "station_metrics_table.csv"), row.names = FALSE)
  
  # Export temporal metrics  
  write.csv(temporal_metrics, file.path(CONFIG$output_dir, "temporal_metrics_table.csv"), row.names = FALSE)
  
  # Export spatial data (for spatial error mapping)
  spatial_data <- validation_data %>%
    select(X_coord, Y_coord, abs_error, residual, STATION, excluded_station, time_stamp)
  write.csv(spatial_data, file.path(CONFIG$output_dir, "spatial_error_data.csv"), row.names = FALSE)
  
  # Export observed vs predicted pairs
  prediction_pairs <- validation_data %>%
    select(observed, predicted, residual, abs_error, STATION, excluded_station, time_stamp)
  write.csv(prediction_pairs, file.path(CONFIG$output_dir, "prediction_pairs_table.csv"), row.names = FALSE)
  
  cat("Exported 5 data tables for external analysis\n")
}

# ==============================================================================
# VISUALIZATION FUNCTIONS
# ==============================================================================

# Custom theme for clear, readable plots
custom_theme <- theme_minimal() +
  theme(
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = "grey90"),
    panel.grid.major = element_line(color = "grey95", size = 0.5),
    panel.grid.minor = element_line(color = "grey97", size = 0.3),
    plot.title = element_text(size = 18, face = "bold", margin = margin(b = 20)),
    plot.subtitle = element_text(size = 14, margin = margin(b = 15)),
    axis.title = element_text(size = 14, face = "bold"),
    axis.text = element_text(size = 12),
    legend.title = element_text(size = 14, face = "bold"),
    legend.text = element_text(size = 12),
    strip.text = element_text(size = 12, face = "bold"),
    legend.background = element_rect(fill = "white", color = "grey80"),
    strip.background = element_rect(fill = "grey95", color = "grey80")
  )

save_plot <- function(plot_obj, filename, width = CONFIG$plot_width, height = CONFIG$plot_height) {
  if (CONFIG$save_plots) {
    filepath <- file.path(CONFIG$output_dir, filename)
    ggsave(filepath, plot_obj, width = width, height = height, dpi = 300, bg = "white")
    cat("Saved:", filename, "\n")
  }
  return(plot_obj)
}

# ==============================================================================
# PLOT 1: OBSERVED VS PREDICTED
# ==============================================================================

p1 <- ggplot(validation_data, aes(x = observed, y = predicted)) +
  geom_point(alpha = 0.7, color = "steelblue", size = 2) +
  geom_abline(slope = 1, intercept = 0, color = "red", linewidth = 1.5) +
  geom_smooth(method = "lm", se = TRUE, color = "darkblue", linewidth = 1.2) +
  labs(
    title = "GTWR Cross-Validation: Observed vs Predicted",
    subtitle = paste("R² =", round(overall_metrics$R2, 3), 
                     "| RMSE =", round(overall_metrics$RMSE, 3),
                     "| n =", overall_metrics$n_predictions),
    x = paste("Observed", CONFIG$response_var),
    y = paste("Predicted", CONFIG$response_var)
  ) +
  custom_theme

save_plot(p1, "01_observed_vs_predicted.png")

# ==============================================================================
# PLOT 2: STATION-LEVEL PERFORMANCE
# ==============================================================================

p2a <- ggplot(station_metrics, aes(x = reorder(excluded_station, RMSE), y = RMSE)) +
  geom_col(fill = "coral", color = "black", linewidth = 0.3) +
  coord_flip() +
  labs(
    title = "Prediction Accuracy by Excluded Station",
    x = "Excluded Station",
    y = "RMSE"
  ) +
  custom_theme

p2b <- ggplot(station_metrics, aes(x = reorder(excluded_station, R2), y = R2)) +
  geom_col(fill = "lightblue", color = "black", linewidth = 0.3) +
  coord_flip() +
  labs(
    title = "R² by Excluded Station",
    x = "Excluded Station",
    y = "R²"
  ) +
  custom_theme

p2_combined <- grid.arrange(p2a, p2b, ncol = 2)
save_plot(p2_combined, "02_station_performance.png", width = 18)

# ==============================================================================
# PLOT 3: SPATIAL ERROR PATTERNS
# ==============================================================================

p3 <- ggplot(validation_data, aes(x = X_coord, y = Y_coord, color = abs_error)) +
  geom_point(size = 2.5, alpha = 0.8) +
  scale_color_gradientn(name = "Absolute\nError", colors = c("blue", "cyan", "yellow", "red"),
    limits = quantile(validation_data$abs_error, c(0.05, 0.95), na.rm = TRUE)) +
  labs(
    title = "Spatial Distribution of Prediction Errors",
    x = "X Coordinate",
    y = "Y Coordinate"
  ) +
  custom_theme

save_plot(p3, "03_spatial_errors.png")

# ==============================================================================
# PLOT 4: TEMPORAL PATTERNS
# ==============================================================================

p4a <- ggplot(temporal_metrics, aes(x = time_stamp, y = RMSE)) +
  geom_line(color = "red", linewidth = 1.5) +
  geom_point(color = "darkred", size = 3) +
  labs(
    title = "Model Accuracy Over Time",
    x = "Time Period",
    y = "RMSE"
  ) +
  custom_theme

p4b <- ggplot(temporal_metrics, aes(x = time_stamp, y = R2)) +
  geom_line(color = "blue", linewidth = 1.5) +
  geom_point(color = "darkblue", size = 3) +
  labs(
    title = "R² Over Time",
    x = "Time Period",
    y = "R²"
  ) +
  custom_theme

p4_combined <- grid.arrange(p4a, p4b, nrow = 2)
save_plot(p4_combined, "04_temporal_patterns.png", height = 12)

# ==============================================================================
# PLOT 5: RESIDUAL ANALYSIS
# ==============================================================================

p5a <- ggplot(validation_data, aes(x = predicted, y = residual)) +
  geom_point(alpha = 0.7, color = "purple", size = 2) +
  geom_hline(yintercept = 0, color = "red", linewidth = 1.5) +
  geom_smooth(se = TRUE, color = "darkgreen", linewidth = 1.2) +
  labs(
    title = "Residual Plot",
    x = paste("Predicted", CONFIG$response_var),
    y = "Residual (Observed - Predicted)"
  ) +
  custom_theme

p5b <- ggplot(validation_data, aes(x = residual)) +
  geom_histogram(bins = 30, fill = "lightgreen", alpha = 0.8, color = "black", linewidth = 0.3) +
  geom_vline(xintercept = 0, color = "red", linewidth = 1.5) +
  labs(
    title = "Distribution of Residuals",
    x = "Residual",
    y = "Frequency"
  ) +
  custom_theme

p5_combined <- grid.arrange(p5a, p5b, ncol = 2)
save_plot(p5_combined, "05_residual_analysis.png", width = 18)

# ==============================================================================
# PLOT 6: ERROR BY STATION (DETAILED)
# ==============================================================================

p6 <- validation_data %>%
  filter(STATION == excluded_station) %>%  # Only predictions at excluded stations
  ggplot(aes(x = time_stamp, y = abs_error, color = excluded_station)) +
  geom_line(alpha = 0.8, linewidth = 1.2) +
  geom_point(size = 2) +
  facet_wrap(~excluded_station, scales = "free_y") +
  labs(
    title = "Prediction Errors at Excluded Stations Over Time",
    x = "Time Period",
    y = "Absolute Error"
  ) +
  custom_theme +
  theme(legend.position = "none")

save_plot(p6, "06_station_time_errors.png", width = 18, height = 14)

# ==============================================================================
# SUMMARY REPORT
# ==============================================================================

# Create summary table
summary_report <- data.frame(
  Metric = c("Overall RMSE", "Overall MAE", "Overall R²", "Overall Bias",
             "Best Station RMSE", "Worst Station RMSE", 
             "Station RMSE Range", "Temporal Stability (CV of RMSE)"),
  Value = c(
    round(overall_metrics$RMSE, 3),
    round(overall_metrics$MAE, 3),
    round(overall_metrics$R2, 3),
    round(overall_metrics$bias, 3),
    round(min(station_metrics$RMSE, na.rm = TRUE), 3),
    round(max(station_metrics$RMSE, na.rm = TRUE), 3),
    round(max(station_metrics$RMSE, na.rm = TRUE) - min(station_metrics$RMSE, na.rm = TRUE), 3),
    round(sd(temporal_metrics$RMSE, na.rm = TRUE) / mean(temporal_metrics$RMSE, na.rm = TRUE), 3)
  )
)

# Save summary
if (CONFIG$save_plots) {
  write.csv(summary_report, file.path(CONFIG$output_dir, "validation_summary.csv"), row.names = FALSE)
  write.csv(station_metrics, file.path(CONFIG$output_dir, "station_metrics.csv"), row.names = FALSE)
  write.csv(temporal_metrics, file.path(CONFIG$output_dir, "temporal_metrics.csv"), row.names = FALSE)
}

print(summary_report)

cat("\n=== VISUALIZATION COMPLETE ===\n")
cat("Generated", length(list.files(CONFIG$output_dir, pattern = ".png")), "plots\n")
cat("Output directory:", CONFIG$output_dir, "\n")