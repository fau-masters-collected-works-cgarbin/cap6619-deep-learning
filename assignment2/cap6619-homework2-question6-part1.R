# CAP-6619 Deep Learning Fall 2018
# Homework 2, question 6 part 1 (two classes)
# Christian Garbin
#
# This is the R code used for the notebook
# Used to debug it, then copied to the notebook
#
# This code follows the tidyverse styleguide http://style.tidyverse.org/

# Environment setup, packages, constants

# Always start with a clean environment to avoid subtle bugs
rm(list = ls())

# To get repeatable results with random numbers
set.seed(1234)

setwd("~/fau/cap6619/assignments/assignment2")

# install.packages("pixmap")
library(pixmap)
library(gdata)

# Values for class labels
CLASS_A_LABEL <- 1
CLASS_B_LABEL <- -1

# What the class label column is called in the data frames
# This MUST match the name of the class label variable added to the data frame
CLASS_LABEL <- "class_label"

# Helper functions

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

  class_label <- paste(CLASS_LABEL, "~ ", sep = " ")
  myform <- as.formula(paste(class_label, paste(names(training[!names(training) %in%
    CLASS_LABEL]), collapse = " + ")))

  classifier <- neuralnet(myform, training,
    hidden = layers, rep = rep, linear.output = FALSE,
    threshold = 0.1
  )

  class_index <- length(test)

  predicted_first <- compute(classifier, test[, -class_index])
  predicted_best <- compute(classifier, test[, -class_index],
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
  # Changes from continuous to categorical values
  predicted <- ifelse(classification > 0.5, CLASS_A_LABEL, CLASS_B_LABEL)

  # Extract the actual labels (assumes label is the last column in the test set)
  class_index <- length(test)
  actual <- test[, class_index]

  # Show confusion matrix and accuracy
  # As a reminder: TN <- t[1, 1], FP <- t[1, 2], FN <- t[2, 1], TP <- t[2, 2]
  t <- table(predicted, actual)
  accuracy <- (sum(diag(t))) / sum(t)

  # Debug code
  # Show classification, predicted value and actual value side by side
  # predicted_vs_actual <- cbind(classification, predicted, actual)
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

# Reproduce class example --------------------------------------------------------------------------
# This section reproduces the example we saw in class (but note that results will not be exactly
# the same because of differences in random values - the class examples didn't set a seed).
class_example <- function() {
  # Load data
  class_a_images <- load_images("faces/fromclass/left", ".*\\.pgm", CLASS_A_LABEL)
  class_b_images <- load_images("faces/fromclass/right", ".*\\.pgm", CLASS_B_LABEL)
  all_images <- split_data_set(rbind(class_a_images, class_b_images), 0.6)
  training_set <- all_images$training
  test_set <- all_images$test

  results <- classify_faces(training_set, test_set, c(2), 100)
  show_results("Class example - one hidden layer, two neurons, first", test_set, results$first)
  show_results("Class example - one hidden layer, two neurons, best", test_set, results$best)

  results <- classify_faces(training_set, test_set, c(4, 3), 1000)
  show_results("Class example - two hidden layers, 4/3 neurons each, first", test_set, results$first)
  show_results("Class example - two hidden layers, 4/3 neurons each, best", test_set, results$best)
}
# class_example()

# Homework assignment ------------------------------------------------------------------------------
# This section is the homework assignment

# Please download at least 100 face images from CMU website (you can also download images Canvas or
# from other sites) to build a two class classification tasks (50 images for each class). You must
# specify faces in the positive and negative class (e.g., faces turning left are positive, and faces
# turning right are negative). Please explain your classification task and show one example of the
# face in each class (using R) [0.5 pt].
#
# Please randomly select 40% of images as training samples to train neural networks with one hidden
# layer but different number of hidden nodes (3, 5, 7, 9, 11 hidden nodes, respectively). Please
# show the classification accuracies of the neural networks on the remaining 60% of images (which
# are not selected as training samples). Please report R code (e.g., using R notebook), and also use
# a table to summarize the classification accuracy of the neural networks with respect to different
# number of hidden nodes [1 pt].

# Load data
class_a_images <- load_images("faces/straight", ".*\\.pgm", CLASS_A_LABEL)
class_b_images <- load_images("faces/up", ".*\\.pgm", CLASS_B_LABEL)
all_images <- split_data_set(rbind(class_a_images, class_b_images), 0.4)
training_set <- all_images$training
test_set <- all_images$test

result <- classify_faces(training_set, test_set, c(3), 100)
show_results("Homework - three neurons in hidden layer - first", test_set, result$first)
show_results("Homework - three neurons in hidden layer - best", test_set, result$best)

result <- classify_faces(training_set, test_set, c(5), 100)
show_results("Homework - five neurons in hidden layer - first", test_set, result$first)
show_results("Homework - five neurons in hidden layer - best", test_set, result$best)

result <- classify_faces(training_set, test_set, c(7), 100)
show_results("Homework - seven neurons in hidden layer - first", test_set, result$first)
show_results("Homework - seven neurons in hidden layer - best", test_set, result$best)

result <- classify_faces(training_set, test_set, c(9), 100)
show_results("Homework - nine neurons in hidden layer - first", test_set, result$first)
show_results("Homework - nine neurons in hidden layer - best", test_set, result$best)

result <- classify_faces(training_set, test_set, c(11), 100)
show_results("Homework - eleven neurons in hidden layer - first", test_set, result$first)
show_results("Homework - eleven neurons in hidden layer - best", test_set, result$best)

# Homework assignment, different data set ----------------------------------------------------------

# Even with eleven neurons, the accuracy of this tasks is not very high. A possible explanation is
# that the classes are relatively close together, therefore harder to separate with the network
# configuration we used.

# As an experiment, I tried the same network configuration on two classes that are easier to
# separate: looking left and looking right (not the set used in class - used the same people
# selected for the straight/up set above).

# The results below show the classes and the classification results. It has a higher accuracy, as
# hypothesized, but because the code is not using cross-validation to select the best network, the
# results are at most preliminary at this point.

# Load data
class_a_images <- load_images("faces/left", ".*\\.pgm", CLASS_A_LABEL)
class_b_images <- load_images("faces/right", ".*\\.pgm", CLASS_B_LABEL)
all_images <- split_data_set(rbind(class_a_images, class_b_images), 0.4)
training_set <- all_images$training
test_set <- all_images$test

result <- classify_faces(training_set, test_set, c(3), 100)
show_results("Homework extra - three neurons in hidden layer - first", test_set, result$first)
show_results("Homework extra - three neurons in hidden layer - best", test_set, result$best)

result <- classify_faces(training_set, test_set, c(5), 100)
show_results("Homework extra - five neurons in hidden layer - first", test_set, result$first)
show_results("Homework extra - five neurons in hidden layer - best", test_set, result$best)

result <- classify_faces(training_set, test_set, c(7), 100)
show_results("Homework extra - seven neurons in hidden layer - first", test_set, result$first)
show_results("Homework extra - seven neurons in hidden layer - best", test_set, result$best)

result <- classify_faces(training_set, test_set, c(9), 100)
show_results("Homework extra - nine neurons in hidden layer - first", test_set, result$first)
show_results("Homework extra - nine neurons in hidden layer - best", test_set, result$best)

result <- classify_faces(training_set, test_set, c(11), 100)
show_results("Homework extra - eleven neurons in hidden layer - first", test_set, result$first)
show_results("Homework extra - eleven neurons in hidden layer - best", test_set, result$best)
