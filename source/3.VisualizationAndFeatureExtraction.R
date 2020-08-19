####################################################################
# Course project - ML (MIRI)
## Authors:
# - Marti Cardoso 
# - Meysam Zamani

# Part 3: Feature extraction and visualization
# In this script we do the visualization of our dataset and 
# saves the results of the PCA and MCA as a new dataset (feature extraction)
# June 2019
####################################################################

rm(list = ls())

# Set environment
setwd(".")

library(FactoMineR)
library(factoextra)

par(mfrow=c(1,1))

# we are going to apply two visualization approaches:

#################################
## Approach 1 (Creation of D3) ##
#################################
# Apply PCA to the continuous variables and MCA to the categorical ones
# Select the number of significant dimensions of each one.
# Then, making the concatenation of both and applying again PCA 
# And again selecting the significant dimensions (these projections are the new dataset D3)

# Load preprocessed data
load("dataset/bank-processed-train-test.Rdata")

#### PCA #####

# 1. Apply PCA to continuous variables #
dataset.train.numerical = dataset.train[,-c(2:8,10,12,18)]
pca1.result = prcomp(dataset.train.numerical, scale=TRUE)

# PCA plots
fviz_eig(pca1.result, main="Scree plot - First PCA - D1")
fviz_pca_var(pca1.result, col.var = "contrib",  gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"), repel = TRUE )
ggplot(data.frame(pca1.result$x[,1:2],y=dataset.train$y), aes(x=PC1, y=PC2, group=y, color=y)) +  
  geom_point(size = 0.1) +theme_minimal()

# Looking at the screeplot, we decided to take 6 significant dimensions
pca1.nd = 6

#### MCA #####

# Categorical variables
dataset.train.cat = dataset.train[,c(2:8,10,12)]

# Max number of dimension
ncols <- ncol(dataset.train.cat)
nmod = 0
for (j in 1:ncols) {nmod = nmod + length(levels(dataset.train.cat[,j]))}
mca.max.nd = nmod - ncols

mca.res = MCA(dataset.train.cat,ncp=mca.max.nd)
fviz_screeplot(mca.res,ncp=mca.max.nd)

# Selection of significant dimensions
i <- 1
while (mca.res$eig[i,1] > mean(mca.res$eig[,1])) i <- i+1
(mca1.nd <- i-1)

# Plots
fviz_mca_var(mca.res, choice="mca.cor")
fviz_mca_var(mca.res)
ggplot(data.frame(mca.res$ind$coord[,1:2],y=dataset.train$y), aes(x=Dim.1, y=Dim.2, group=y, color=y)) +  
  geom_point(size = 0.1) +theme_minimal()


#### PCA again on PCA1+MCA #####

join.dataset = data.frame(pca1.result$x[,1:pca1.nd], mca.res$ind$coord[,1:mca1.nd])

join.pca = prcomp(join.dataset)

fviz_eig(join.pca,ncp=ncol(join.dataset),main="Scree plot - PCA over (PCA+MCA) - D1")

join.pca.nd= 6

d3.pcamca.train = data.frame(join.pca$x[,1:join.pca.nd])
d3.pcamca.train$y = dataset.train$y

ggplot(d3.pcamca.train, aes(x=PC1, y=PC2, group=y, color=y)) +  geom_point(size = 0.1) +theme_minimal()

### Apply projections to Test dataset ###

#First PCA (to numerical)
test.pca1.coord = predict(pca1.result, dataset.test[,-c(2:8,10,12,18)])

# Apply MCA to categorical
#Function needed for computing projections (MCA)
fixMCADatasetForPrediction = function(data){
  niveau <- unlist(lapply(data,levels))
  for (i in 1:ncol(data)){
    if (sum(niveau %in% levels(data[, i])) != nlevels(data[, i]))
      levels(data[, i]) = paste(attributes(data)$names[i], levels(data[, i]), sep = "_")
  }
  data
}

test.mca1.coord = predict(mca.res, fixMCADatasetForPrediction(dataset.test[,c(2:8,10,12)]))$coord

# Join PCA+MCA and apply PCA2 projection
test.join.dataframe = data.frame(test.pca1.coord[,1:pca1.nd],test.mca1.coord[,1:mca1.nd])
tmp = predict(join.pca, test.join.dataframe)
d3.pcamca.test = data.frame(tmp[,1:join.pca.nd])
d3.pcamca.test$y = dataset.test$y

save(d3.pcamca.train, d3.pcamca.test, file = "dataset/D3.PCAMCA.dataset.Rdata")

#################################
## Approach 2 (Creation of D4) ##
#################################
# Apply MCA to the dataset, discretizing all continuous variables

#Load preprocessed data
load("dataset/bank-processed-cat-train-test.Rdata")

# All variables are categorical, except age, so we discretize this variable
dataset.cat.train$age <- cut(dataset.cat.train$age, c(20,30,40,50,60,80), labels=c("<30","31<40","41<50","51<60",">61"),right = FALSE, ordered_result=TRUE)
plot(dataset.cat.train$age, main="nr.age")
dataset.cat.test$age <- cut(dataset.cat.test$age, c(20,30,40,50,60,80), labels=c("<30","31<40","41<50","51<60",">61"),right = FALSE, ordered_result=TRUE)


# Find the maximum number of dimension
ncols <- ncol(dataset.cat.train)
nmod = 0
for (j in 1:ncols) {nmod = nmod + length(levels(dataset.cat.train[,j]))}
mca.max.nd = nmod - ncols

#Perform MCA
mca.res = MCA(dataset.cat.train[,-18],ncp=mca.max.nd)

#Dimensions to keep
fviz_screeplot(mca.res,ncp=mca.max.nd)
i <- 1
while (mca.res$eig[i,1] > mean(mca.res$eig[,1])) i <- i+1
(mca.nd <- i-1)

#Plot MCA
fviz_mca_var(mca.res, choice="mca.cor")
fviz_mca_var(mca.res)
ggplot(data.frame(mca.res$ind$coord[,1:2],y=dataset.cat.train$y), aes(x=Dim.1, y=Dim.2, group=y, color=y)) +  
  geom_point(size = 0.1) +theme_minimal()

# Create new dataframe (train)
d4.mca.train = data.frame(mca.res$ind$coord[,1:mca.nd])
d4.mca.train$y = dataset.cat.train$y

### Apply projections to Test dataset ###

# Create new dataframe (test)
d4.mca.test = predict(mca.res, fixMCADatasetForPrediction(dataset.cat.test[,-18]))$coord[,1:mca.nd]
d4.mca.test = data.frame(d4.mca.test)
d4.mca.test$y = dataset.cat.test$y

# Save data
save(d4.mca.train, d4.mca.test, file = "dataset/D4.MCA.dataset.Rdata")
