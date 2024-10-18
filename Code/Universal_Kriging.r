library(gstat)
library(dplyr)
library(tidyr)
library(sp)
library(sf)
library(automap)
library(spdep)
library(scales)

options(warn = -1) # Don't print warnings

# Set working directory
setwd()

# Function to import and preprocess data
import_data <- function(file_name, year, month) {
  read.csv(file_name, fileEncoding = "cp949") %>%
    filter(year == !!year, month == !!month) %>%
    distinct(id, year, month, .keep_all = TRUE)
}

# Function to set coordinates for spatial data
set_spatial_coordinates <- function(data) {
  coordinates(data) <- ~center_x + center_y + mean_altitude
  return(data)
}

# Function to perform variogram analysis and kriging
perform_geostatistics <- function(data, base_data, output_prefix, year, month) {
  variogram_model <- autofitVariogram(chl ~ center_x + center_y + mean_altitude, data)
  png_filename <- sprintf("", output_prefix, year, month)
  png(filename = png_filename)
  plot(variogram_model)
  dev.off()

  kriging_results <- autoKrige(chl ~ center_x + center_y + mean_altitude, data, base_data)
  base_data$ok_chl <- kriging_results$krige_output$var1.pred
  return(base_data)
}

# Function to export results
export_results <- function(data, year, month) {
  csv_filename <- sprintf("", year, month)
  write.csv(data, csv_filename, fileEncoding='cp949', row.names=FALSE)
}

# Main analysis function
run_analysis <- function(year, month) {
  print(paste("Processing year:", year, "month:", month))

  base_data <- import_data("finalDf.csv", year, month)
  chl_data <- import_data("completeDf.csv", year, month)

  base_data <- set_spatial_coordinates(base_data)
  chl_data <- set_spatial_coordinates(chl_data)

  print(paste("Variogram and Kriging year:", year, "month:", month))
  processed_data <- perform_geostatistics(chl_data, base_data, "variogram", year, month)

  print(paste("Exporting year:", year, "month:", month))
  export_results(processed_data, year, month)
}

# Run analysis for specified range
years <- 2006:2024
months <- 1:12
for (year in years) {
  for (month in months) {
    run_analysis(year, month)
  }
}
