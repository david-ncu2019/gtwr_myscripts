#' GTWR Spatiotemporal Distance Matrix Calculator
#' 
#' Computes calibration-to-calibration distance matrix using GWmodel's st.dist function.
#' Provides flexible column mapping for coordinate and temporal data.

library(GWmodel)
library(rhdf5)

#' Calculate dMat2 spatiotemporal distance matrix
#' 
#' @param data Data frame with calibration observations
#' @param x_col X coordinate column name
#' @param y_col Y coordinate column name  
#' @param time_col Time values column name
#' @param lamda Spatial-temporal balance parameter (0-1)
#' @param p Minkowski distance power (default: 2)
#' @param theta Coordinate rotation angle in radians (default: 0)
#' @param longlat Use great circle distances (default: FALSE)
#' @param t.units Temporal distance units (default: "auto")
#' @param ksi Space-time interaction parameter (default: 0)
#' 
#' @return Square spatiotemporal distance matrix

calculate_dMat2 <- function(data, 
                           x_col = "x_coord", 
                           y_col = "y_coord",
                           time_col = "time_month",
                           lamda = 0.05,
                           p = 2,
                           theta = 0,
                           longlat = FALSE,
                           t.units = "auto",
                           ksi = 0) {
  
  # Validate required columns
  required_cols <- c(x_col, y_col, time_col)
  missing_cols <- required_cols[!required_cols %in% names(data)]
  if (length(missing_cols) > 0) {
    stop("Missing columns: ", paste(missing_cols, collapse = ", "))
  }
  
  # Extract spatial and temporal components
  dp.locat <- as.matrix(data[, c(x_col, y_col)])
  obs.tv <- data[[time_col]]
  n_obs <- nrow(data)
  
  cat("Computing dMat2 for", n_obs, "observations using GWmodel::st.dist\n")
  cat("Parameters: lamda =", lamda, ", t.units =", t.units, "\n")
  
  # Calculate spatiotemporal distance matrix using GWmodel
  dMat2 <- st.dist(dp.locat = dp.locat,
                   obs.tv = obs.tv,
                   p = p,
                   theta = theta,
                   longlat = longlat,
                   lamda = lamda,
                   t.units = t.units,
                   ksi = ksi)
  
  # Report matrix characteristics
  cat("Matrix dimensions:", nrow(dMat2), "x", ncol(dMat2), "\n")
  cat("Memory usage:", format(object.size(dMat2), units = "MB"), "\n")
  cat("Distance range: [", round(min(dMat2), 3), ",", round(max(dMat2), 3), "]\n")
  
  return(dMat2)
}

#' Save distance matrix in cross-platform format
#' 
#' @param dMat2 Distance matrix to save
#' @param filename Output filename (supports .h5, .csv)

save_dMat2 <- function(dMat2, filename = "dMat2_calibration.h5") {
  
  if (grepl("\\.h5$", filename) && requireNamespace("rhdf5", quietly = TRUE)) {
    rhdf5::h5write(dMat2, filename, "dMat2")
    cat("Saved to", filename, "(HDF5 format)\n")
  } else {
    csv_filename <- gsub("\\.h5$", ".csv", filename)
    write.csv(dMat2, csv_filename, row.names = FALSE)
    cat("Saved to", csv_filename, "(CSV format)\n")
  }
}