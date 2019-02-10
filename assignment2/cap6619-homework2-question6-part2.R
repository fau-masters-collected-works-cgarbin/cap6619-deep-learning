# CAP-6619 Deep Learning Fall 2018
# Homework 2, question 6 part 2 (three classes)
# Christian Garbin
#
# This is the R code used for the notebook
# Used to debug it, then copied to the notebook
#
# This code follows the tidyverse styleguide http://style.tidyverse.org/

# Sources for this code:
# http://www.learnbymarketing.com/tutorials/neural-networks-in-r-tutorial/
# https://www.r-bloggers.com/multilabel-classification-with-neuralnet-package/
# https://stackoverflow.com/questions/20813039/multinomial-classification-using-neuralnet-package

# Environment setup, packages, constants

# Always start with a clean environment to avoid subtle bugs
rm(list = ls())

# To get repeatable results with random numbers
set.seed(1234)

setwd("~/fau/cap6619/assignments/assignment2")

# install.packages("pixmap")
library(pixmap)
library(gdata)

# Name of the class label column in the data frames before hot-encoding
CLASS_LABEL <- "class_label"

# Values for class labels before hot-encoding (the values of the CLASS_LABEL column)
CLASS_A_LABEL <- 1
CLASS_B_LABEL <- 2
CLASS_C_LABEL <- 3

# Names of the class labels columns after they are hot-encoded
# Can be any string - doesn't affect the code, just helps understand the dataset
CLASS_NAMES <- c("Left", "Right", "Straight")

# Helper functions ---------------------------------------------------------------------------------

#' Read image files matching pattern, add class label and return them as grey-scale data frame,
#' with the given class as the last element in the vectors.
#'
#' @param path_name Directory where the images are, relative to current directory.
#' @param file_pattern A pattern to identify the image files to load.
#' @param label The label to be used for these images.
#'
#' @return A data frame with one row for each image and one column for each pixel (converted to
#'         grey scale) and the given label as the last (rightmost) column.
load_images <- function(path_name, file_pattern, label) {
  # Get all file names matching the pattern
  files <- list.files(path = path_name, pattern = file_pattern, all.files = T, full.names = T)

  # Read all images
  images <- lapply(files, read.pnm, cellres = 1)

  # Debug: show the first image
  cat("Image for class", label, "\n\n")
  plot(images[[1]])

  # Transform image to grey scale vectors
  image_matrix <- images[[1]]@grey
  image_vector <- unmatrix(image_matrix, byrow = T)
  for (ii in 2:length(images)) {
    i.matrix <- images[[ii]]@grey
    i.vector <- unmatrix(i.matrix, byrow = T)
    image_vector <- rbind(image_vector, i.vector)
  }

  # Change to a data frame
  image_frame <- data.frame(image_vector)

  # Add class label (variable name must match constant for class label column name)
  number_of_images <- nrow(image_frame)
  class_label <- rep(label, number_of_images)
  image_frame <- cbind(image_frame, class_label)

  return(image_frame)
}

#' Hot-encode the column that has a class label: changes it from one multiclass column to one
#' column of each label, hot encoded.
#'
#' The data set is modified in place, so there is no need to return it.
#'
#' @param data_set The dataframe with the column to be hot-encoded. That column must be the last
#'   (rightmost) column in the dataframe.
hot_encode_label <- function(data_set) {
  # Hot-encode each class (will have "1" for that class, "0" for other classes)
  class_a <- ifelse(data_set[CLASS_LABEL] == CLASS_A_LABEL, 1, 0)
  class_b <- ifelse(data_set[CLASS_LABEL] == CLASS_B_LABEL, 1, 0)
  class_c <- ifelse(data_set[CLASS_LABEL] == CLASS_C_LABEL, 1, 0)

  # Add hot-encoded labels to the end (right) of the data set and name the new columns
  data_set <- cbind(data_set, class_a, class_b, class_c)
  names(data_set)[(length(names(data_set)) - 2):length(names(data_set))] <- CLASS_NAMES

  # Remove the current label - we no longer need it and it would affect training if we leave it here
  data_set <- data_set[, !(names(data_set) == CLASS_LABEL)]

  # Debug code: show last few columns of some rows to check if the labels are there
  # Warning: also modifies the data set - uncomment only when inspecting data
  # num_columns <- length(names(data_set))
  # num_rows <- nrow(data_set)
  # data_set[c(1, num_rows / 2, num_rows), (num_columns - 4):num_columns]
}

#' Creates a neural net with the given layers, trains and tests it.
#'
#' @param training The trainig data frame, with the class label in the last column.
#' @param test The test data frame, with the class label in the last column.
#' @param layers A vector representing the network configuration, to be used for the \code{hidden}
#'   parameter of \code{neuralnet}.
#' @param rep The number of repetitions to be used during training, to be used for the \code{rep}
#'   parameter of \code{neuralnet}.
#'
#' @return Two sets of predicted class using the test data and the \code{compute} function, one set
#'   has the first repetition, the other has the best repetition (the one with the smallest error).
#'   The predicted class is returned as a continuous value, not categorical values.
#'
#' @section IMPORTANT
#' "Best" in this context is "smallest error for the *traninig data*". It doesn't mean the best
#' (smallest) error on the test data. To properly find the best repetition we need a more
#' sophisticated technique, e.g. cross-validation with folds. The code as is simply prevents us from
#' blindly picking the first repetition in all cases, improving the odds we are picking a better
#' one (compared to the first), but could have side effects (e.g. choose a model that overfits).
classify_faces <- function(training, test, layers, rep) {
  # install.packages("neuralnet")
  library(neuralnet)

  # To get repeatable results with random numbers
  set.seed(1234)

  # Create the formula : "label1+label2+... ~ feature1+feature2+..."
  features <- names(training[!names(training) %in% CLASS_NAMES])
  myform <- as.formula(paste(paste(CLASS_NAMES, collapse = " + "),
    paste(features, collapse = " + "),
    sep = " ~ "
  ))

  classifier <- neuralnet(myform, training,
    hidden = layers, rep = rep, linear.output = FALSE,
    threshold = 0.1
  )

  # Last column that has a feature (to exclude class labels)
  last_feature_index <- length(test) - length(CLASS_NAMES)

  predicted_first <- compute(classifier, test[, 1:last_feature_index])
  predicted_best <- compute(classifier, test[, 1:last_feature_index],
    rep = which.min(classifier$result.matrix["error", ])
  )

  return(list(first = predicted_first$net.result, best = predicted_best$net.result))
}

#' Show the results from the classification (confusion matrix and accuracy).
#'
#' @param title A string to identify the results being shown.
#' @param test The test data frame, with the class label in the last column.
#' @param classification The results from the classification (as returned by \code{compute}).
show_results <- function(title, test, classification) {
  # Last column that has a feature
  last_feature_index <- length(test) - length(CLASS_NAMES)

  # Label in the test data
  # Since the labels are hot-encoded into the different columns, this will return the number of
  # the column that has "1" set, i.e. the original label).
  label_columns <- test[, (last_feature_index + 1):length(test)]
  actual_class <- max.col(label_columns)

  # The predicted class is the one with the highest probability
  predicted_class <- max.col(classification)

  t <- table(predicted_class, actual_class)
  accuracy <- (sum(diag(t))) / sum(t)

  # Debug code
  # Show classification, predicted value and actual value side by side
  # predicted_vs_actual <- cbind(classification, predicted_class, actual_class)
  # print(predicted_vs_actual)

  cat(title, "\n\n")
  print(t)
  cat("Accuracy: ", accuracy, "\n\n")
}

#' Randomize a set and split it into a training and a test set.
#'
#' @param set The dataframe to be randomized
#' @param training_fraction How much of the dataframe should be used as training data
#'
#' @return A list with two dataframes, one for training, one for test.
split_data_set <- function(set, training_fraction) {
  # To get repeatable results with random numbers
  set.seed(1234)

  train_index <- sample(nrow(set), nrow(set) * training_fraction)
  training_set <- set[train_index, ]
  test_set <- set[-train_index, ]
  return(list(training = training_set, test = test_set))
}

# Test code ----------------------------------------------------------------------------------------
# Use a simple data set to make it easier to inspect and debug the code
test <- function() {
  test_data <- data.frame(F1 = integer(), F2 = integer(), F3 = integer(), class_label = integer())
  test_data[nrow(test_data) + 1, ] <- list(1, 0, 0, CLASS_A_LABEL)
  test_data[nrow(test_data) + 1, ] <- list(1, 0, 0, CLASS_A_LABEL)
  test_data[nrow(test_data) + 1, ] <- list(0, 1, 0, CLASS_B_LABEL)
  test_data[nrow(test_data) + 1, ] <- list(0, 1, 0, CLASS_B_LABEL)
  test_data[nrow(test_data) + 1, ] <- list(0, 0, 1, CLASS_C_LABEL)
  test_data[nrow(test_data) + 1, ] <- list(0, 0, 1, CLASS_C_LABEL)
  test_data <- hot_encode_label(test_data)
  test_data <- split_data_set(test_data, 0.5)
  training_set <- test_data$training
  test_set <- test_data$test
  results <- classify_faces(training_set, test_set, c(3), 10)
  show_results("Test data, first", test_set, results$first)
  show_results("Test data, best", test_set, results$best)
}
#test()

# Load and pre-process data (hot-encode the classes) -----------------------------------------------

class_a_images <- load_images("faces/left", ".*\\.pgm", CLASS_A_LABEL)
class_b_images <- load_images("faces/right", ".*\\.pgm", CLASS_B_LABEL)
class_c_images <- load_images("faces/straight", ".*\\.pgm", CLASS_C_LABEL)
all_images <- rbind(class_a_images, class_b_images, class_c_images)
all_images <- hot_encode_label(all_images)

# Train and test the neural network ----------------------------------------------------------------
all_images <- split_data_set(all_images, 0.6)
training_set <- all_images$training
test_set <- all_images$test

results <- classify_faces(training_set, test_set, c(9), 100)
show_results("Multiclass - one hidden layer, nine neurons, first", test_set, results$first)
show_results("Multiclass - one hidden layer, nine neurons, best", test_set, results$best)

# Other network configurations ---------------------------------------------------------------------
# Not requested in the homework, just for comparison

results <- classify_faces(training_set, test_set, c(3), 100)
show_results("Multiclass - one hidden layer, three neurons, first", test_set, results$first)
show_results("Multiclass - one hidden layer, three neurons, best", test_set, results$best)

results <- classify_faces(training_set, test_set, c(5), 100)
show_results("Multiclass - one hidden layer, five neurons, first", test_set, results$first)
show_results("Multiclass - one hidden layer, five neurons, best", test_set, results$best)

results <- classify_faces(training_set, test_set, c(7), 100)
show_results("Multiclass - one hidden layer, seven neurons, first", test_set, results$first)
show_results("Multiclass - one hidden layer, seven neurons, best", test_set, results$best)


results <- classify_faces(training_set, test_set, c(11), 100)
show_results("Multiclass - one hidden layer, eleven neurons, first", test_set, results$first)
show_results("Multiclass - one hidden layer, eleven neurons, best", test_set, results$best)

