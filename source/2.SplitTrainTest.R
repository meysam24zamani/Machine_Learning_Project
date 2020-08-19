####################################################################
# Course project - ML (MIRI)
## Authors:
# - Marti Cardoso 
# - Meysam Zamani

# Part 2: Split dataset into train and test
# June 2019
####################################################################

rm(list = ls())

# Set environment
setwd(".")

# Load preprocessed data
load("dataset/bank-processed.Rdata")
load("dataset/bank-processed-cat.Rdata")

# First, we split the 'dataset', and secondly, we apply the same split to 'dataset.cat' 

set.seed (1)
# Shuffle the data
shuffle <- function(data){
  data[sample(nrow(data)),]
}
dataset <- shuffle(dataset)

# Split data into train and test: stratified split (2/3 learning, 1/3 test)
dataset.y.Yes <- dataset[dataset$y == 'yes',]
dataset.y.No <- dataset[dataset$y == 'no',]

dataset.y.Yes.train.idx <- 1:floor(2/3*nrow(dataset.y.Yes)) # Take the first rows, no need to 'sample' because the data has been shuffled before
dataset.y.No.train.idx <- 1:floor(2/3*nrow(dataset.y.No))

#Join both cases
dataset.train <- rbind(dataset.y.Yes[dataset.y.Yes.train.idx,],
                       dataset.y.No[dataset.y.No.train.idx,])
dataset.test <- rbind(dataset.y.Yes[-dataset.y.Yes.train.idx,],
                      dataset.y.No[-dataset.y.No.train.idx,])

####################
## Standarization ##
####################

# We standardize the data using the training set, and then apply the same transformation to the test set
# (if not, the test set would have some influence on the training standardization)
num.vars <- which(unlist(lapply(names(dataset.train), function(col) is.numeric(dataset.train[,col]))))

for(i in num.vars){
  tmp <- scale(dataset.train[,i])
  col.mean <- mean(dataset.train[,i])
  col.sd <- sd(dataset.train[,i])
  dataset.train[,i] <- (dataset.train[,i]-col.mean)/col.sd
  dataset.test[,i] <- (dataset.test[,i]-col.mean)/col.sd
}

# Save data
save(dataset.train, dataset.test, file = "dataset/bank-processed-train-test.Rdata")

# Apply same split (exact) to dataset.cat and save it
dataset.cat.train <- dataset.cat[row.names(dataset.train),]
dataset.cat.test <- dataset.cat[row.names(dataset.test),]
save(dataset.cat.train, dataset.cat.test, file = "dataset/bank-processed-cat-train-test.Rdata")
