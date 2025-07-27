# Combine all GTWR validation station results into single file
# Configurable for different directories

library(dplyr)

# CONFIGURATION - Set your target folder here
target_folder <- "/mnt/hgfs/1000_SCRIPTS/003_Project002/20250222_GTWR001/4_GTWR/15_TestRun_115/GTWR_Validation/Layer_1"

# Change to target directory
setwd(target_folder)
cat("Working in:", getwd(), "\n")

# Get all station result files
station_files <- list.files(pattern = "^station_.*_results\\.csv$", full.names = TRUE)

cat("Found", length(station_files), "station result files\n")

# Read and combine all files
combined_results <- data.frame()

for (file in station_files) {
  station_data <- read.csv(file, stringsAsFactors = FALSE)
  combined_results <- rbind(combined_results, station_data)
  cat("Added", basename(file), "-", nrow(station_data), "rows\n")
}

# Save combined results
output_file <- "combined_validation_results.csv"
write.csv(combined_results, output_file, row.names = FALSE)

cat("\nCombined", nrow(combined_results), "total predictions\n")
cat("Saved to:", file.path(getwd(), output_file), "\n")
cat("Stations processed:", length(unique(combined_results$excluded_station)), "\n")