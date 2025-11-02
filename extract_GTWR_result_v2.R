# Title: Enhanced GTWR Model Results Compiler with CV Score Support
# Description: This script reads all .rds files from a specified folder,
#              extracts key input parameters and output metrics from each
#              GTWR model object (including CV scores if available), and 
#              compiles them into a single summary CSV file.
# Author: [Your Name]
# Date: 2025-06-11
# Enhancement: Added CV score extraction and improved error handling

# --- 1. SETUP: Define the path and create dynamic output filename ---

# Change this path to the folder containing your .rds files.
results_folder_path <- "gtwr_model_All_Layer"

# Dynamically create the output filename based on the folder path
folder_name <- basename(results_folder_path)
output_csv_filename <- paste0(folder_name, "_summary.csv")


# --- 2. FILE DISCOVERY: Find all .rds files in the target folder ---

rds_files <- list.files(path = results_folder_path, pattern = "\\.rds$", full.names = TRUE)

# Check if any files were found
if (length(rds_files) == 0) {
  stop("No .rds files were found in the specified folder: ", results_folder_path)
}

cat("Found", length(rds_files), "model result files to process.\n\n")


# --- 3. HELPER FUNCTION: Safe extraction with fallback values ---

# This function safely extracts values from nested lists, returning NA if not found
safe_extract <- function(obj, path, default = NA) {
  tryCatch({
    # Navigate through the nested structure
    result <- obj
    for (element in path) {
      if (is.null(result[[element]])) {
        return(default)
      }
      result <- result[[element]]
    }
    return(result)
  }, error = function(e) {
    return(default)
  })
}


# --- 4. DATA EXTRACTION: Loop through files and extract information ---

results_list <- list()

for (file_path in rds_files) {
  
  cat("Processing:", basename(file_path), "\n")
  
  # Use tryCatch to handle potential errors
  model_data <- tryCatch({
    readRDS(file_path)
  }, error = function(e) {
    warning("Could not read file: ", file_path, " - Error: ", e$message)
    return(NULL)
  })
  
  # If the file was read successfully, extract the data
  if (!is.null(model_data)) {
    
    # Print structure for debugging (optional - comment out for cleaner output)
    # cat("Model structure for", basename(file_path), ":\n")
    # str(model_data, max.level = 2)
    
    # Access the arguments and diagnostic lists
    args <- model_data$GTW.arguments
    diag <- model_data$GTW.diagnostic
    
    # Extract CV score using multiple possible locations
    cv_score <- NA
    
    # Method 1: Check if CV score is in diagnostics
    if (!is.null(diag$CV) || !is.null(diag$cv.score) || !is.null(diag$CV.score)) {
      cv_score <- coalesce(diag$CV, diag$cv.score, diag$CV.score)
    }
    
    # Method 2: Check if there's a separate CV results component
    if (is.na(cv_score) && !is.null(model_data$CV)) {
      if (is.numeric(model_data$CV)) {
        # If CV is a single number (total CV score)
        cv_score <- model_data$CV
      } else if (is.vector(model_data$CV)) {
        # If CV is a vector of individual CV values, sum them
        cv_score <- sum(model_data$CV^2, na.rm = TRUE)
      }
    }
    
    # Method 3: Check if CV scores are in the SDF (Spatial Data Frame)
    if (is.na(cv_score) && !is.null(model_data$SDF)) {
      # Check if there's a CV_Score column in the spatial data frame
      if ("CV_Score" %in% names(model_data$SDF@data)) {
        cv_scores <- model_data$SDF@data$CV_Score
        cv_score <- sum(cv_scores^2, na.rm = TRUE)
      }
    }
    
    # Method 4: Check alternative CV naming conventions
    if (is.na(cv_score)) {
      cv_score <- safe_extract(model_data, c("cv_result"))
      if (is.na(cv_score)) {
        cv_score <- safe_extract(model_data, c("cross_validation"))
      }
    }
    
    # Create summary for this file
    file_summary <- list(
      Lambda = safe_extract(args, "lamda", NA),
      Ksi_Degrees = safe_extract(args, "ksi", NA) * (180 / pi),
      Bandwidth = safe_extract(args, "st.bw", NA),
      Kernel = safe_extract(args, "kernel", "unknown"),
      Adaptive = safe_extract(args, "adaptive", NA),
      AICc = safe_extract(diag, "AICc", NA),
      AIC = safe_extract(diag, "AIC", NA),
      BIC = safe_extract(diag, "BIC", NA),
      R_Squared = safe_extract(diag, "gw.R2", NA),
      Adj_R_Squared = safe_extract(diag, "gwR2.adj", NA),
      RSS = safe_extract(diag, "RSS.gw", NA),
      ENP = safe_extract(diag, "enp", NA),
      EDF = safe_extract(diag, "edf", NA),
      CV_Score = cv_score,
      Source_File = basename(file_path)
    )
    
    # Add to results list
    results_list[[length(results_list) + 1]] <- file_summary
    
    # Print extracted CV score for verification
    if (!is.na(cv_score)) {
      cat("  - CV Score found:", cv_score, "\n")
    } else {
      cat("  - No CV Score found\n")
    }
  }
}


# --- 5. HELPER FUNCTION: coalesce (returns first non-NULL value) ---

coalesce <- function(...) {
  args <- list(...)
  for (arg in args) {
    if (!is.null(arg) && !is.na(arg)) {
      return(arg)
    }
  }
  return(NA)
}


# --- 6. CONSOLIDATION AND OUTPUT ---

cat("\nAll files processed. Consolidating results...\n")

if (length(results_list) > 0) {
  
  # Convert to data frame
  summary_df <- do.call(rbind, lapply(results_list, as.data.frame))
  
  # Sort by AICc (or CV_Score if AICc not available)
  if (all(is.na(summary_df$AICc)) && any(!is.na(summary_df$CV_Score))) {
    cat("Sorting by CV Score (AICc not available)...\n")
    summary_df <- summary_df[order(summary_df$CV_Score), ]
  } else {
    summary_df <- summary_df[order(summary_df$AICc), ]
  }
  
  # Save to CSV
  output_path <- file.path(results_folder_path, output_csv_filename)
  write.csv(summary_df, file = output_path, row.names = FALSE)
  
  # Display summary
  cat("\n--- Summary of All Model Results ---\n")
  print(summary_df)
  
  # Report CV score availability
  cv_available <- sum(!is.na(summary_df$CV_Score))
  cat("\nCV Scores found in", cv_available, "out of", nrow(summary_df), "models.\n")
  
  cat("\nResults saved to:", output_path, "\n")
  
} else {
  cat("\nNo valid model data could be extracted from the .rds files.\n")
}


# --- 7. OPTIONAL: Create a detailed model inspection function ---

inspect_model_structure <- function(file_path) {
  cat("\n=== Detailed inspection of:", basename(file_path), "===\n")
  model <- readRDS(file_path)
  
  cat("Top-level components:\n")
  print(names(model))
  
  if (!is.null(model$GTW.diagnostic)) {
    cat("\nGTW.diagnostic components:\n")
    print(names(model$GTW.diagnostic))
  }
  
  if (!is.null(model$SDF)) {
    cat("\nSDF data columns:\n")
    if (inherits(model$SDF, "Spatial")) {
      print(names(model$SDF@data))
    } else {
      print(names(model$SDF))
    }
  }
  
  # Look for any component containing "cv" or "CV"
  all_names <- unlist(rapply(model, function(x) names(x), how = "list"))
  cv_related <- all_names[grepl("cv|CV", all_names, ignore.case = TRUE)]
  if (length(cv_related) > 0) {
    cat("\nPossible CV-related components found:\n")
    print(unique(cv_related))
  }
}

# Uncomment the line below to inspect the first model file in detail:
# inspect_model_structure(rds_files[1])