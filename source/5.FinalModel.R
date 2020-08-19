####################################################################
# Course project - ML (MIRI)
## Authors:
# - Marti Cardoso 
# - Meysam Zamani

# Part 5: Final model and prediction of the test set
# June 2019
####################################################################

rm(list = ls())

# Set environment
setwd(".")

library(e1071)
set.seed (104)

# In model selection, we found that the best model was: 
#  - SVM with RBF kernel, C = 0.1 and gamma = 2^-2
#  - Using D2

# First we load the dataset
load("dataset/bank-processed-cat-train-test.Rdata")
 # Standardize the age variable
col.mean <- mean(dataset.cat.train$age)
col.sd <- sd(dataset.cat.train$age)
dataset.cat.train$age <- (dataset.cat.train$age-col.mean)/col.sd
dataset.cat.test$age <- (dataset.cat.test$age-col.mean)/col.sd

# Create the SVM model
C = 0.1
gamma = 2^-2
class.weights <- 1-table(dataset.cat.train$y)/nrow(dataset.cat.train) #Give more weights to YES
model <- svm(y~., dataset.cat.train, type="C-classification", cost=C, class.weights=class.weights, kernel="radial", gamma=gamma, scale = FALSE)


# Predict test
test.pred <- predict (model, dataset.cat.test)

(ct <- table(Truth=dataset.cat.test$y, Pred=test.pred))


# Compute some statistics

(acc <- round(100*(sum(diag(ct))/sum(ct)),2))

(prop.table(ct,1))

(prop.table(ct,2))
