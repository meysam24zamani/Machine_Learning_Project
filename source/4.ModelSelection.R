####################################################################
# Course project - ML (MIRI)
## Authors:
# - Marti Cardoso 
# - Meysam Zamani

# Part 4: Model Selection
#         We try several ML methods in order to select the best model 
# June 2019
####################################################################
# Warning: this script takes hours to be executed.
rm(list = ls())

# Set environment
setwd(".")

library(MASS)
library(class)
library(e1071)
library(TunePareto)
library(glmnet)
library(class)
library(nnet)
library(ggplot2)
library(e1071)
library(rpart)
library(randomForest)
library(scales)

#Load preprocessed data
load("dataset/bank-processed-train-test.Rdata")
load("dataset/bank-processed-cat-train-test.Rdata")
dataset.cat.train$age <- scale(dataset.cat.train$age)
load("dataset/D3.PCAMCA.dataset.Rdata")
load("dataset/D4.MCA.dataset.Rdata")
set.seed (104)

# First, we load some useful function for the model selection task
source('modelSelectionUtils.R')

####################################################################
# Logistic Regression
####################################################################

# Function that runs 10-fold CV using logistic regression 
run.logisticRegression <- function (dataset,P=0.5)
{
  createModelAndPredict <- function(train, newdata){
    weights <- compute.weights(train$y)
    glm.model <- glm (y~., train, weights=weights,family = binomial) 
    preds <- predict(glm.model, newdata, type="response")
    return(probabilityToFactor(preds,P))
  }
  
  run.k.fold.CV(createModelAndPredict,dataset, performance.metric=c("accuracy","F1"))
}

(logReg.d1 = run.logisticRegression(dataset.train))
(logReg.d2 = run.logisticRegression(dataset.cat.train))
(logReg.d3 = run.logisticRegression(d3.pcamca.train))
(logReg.d4 = run.logisticRegression(d4.mca.train))

####################################################################
# Ridge Regression and Lasso (logistic)
####################################################################

#Function that runs 10-fold-cv using glmnet
#Alpha = 1 -> Lasso
#Alpha = 0 -> Ridge
run.glmnet <- function (dataset, lambda, alpha = 1, P = 0.5)
{
  createModelAndPredict <- function(train, newdata){
    weights = compute.weights(train$y)
    #Create dummy variables for categorical
    x <- model.matrix(y~., train)[,-1]
    
    model <- glmnet(x, train$y, alpha = alpha, weights = weights, family = "binomial", lambda=lambda)
    
    x.test <- model.matrix(y ~., newdata)[,-1]
    preds <- predict (model, newx=x.test, type="response")
    return(probabilityToFactor(preds,P))
  }
  
  run.k.fold.CV(createModelAndPredict,dataset, performance.metric=c("accuracy","F1"))
}

# Function that tries several lambdas
run.glmnet.find.best.lambda <- function (dataset, lambda.v, alpha = 1, P = 0.5)
{
  results = list()
  for(i in 1:(length(lambda.v))){
    print(paste("lambda ", lambda.v[i]))
    results[[i]] <- run.glmnet(dataset, lambda=lambda.v[i], alpha= alpha, P=P)
  }
  z <- list(lambda=lambda.v)
  z$lambda.F1 <- unlist(lapply(results,function(t) t$F1.mean))
  z$lambda.F1.sd <- unlist(lapply(results,function(t) t$F1.sd))
  z$lambda.accuracy <- unlist(lapply(results,function(t) t$accuracy.mean))
  z$lambda.accuracy.sd <- unlist(lapply(results,function(t) t$accuracy.sd))
  
  max.lambda.id <- which.max(z$lambda.F1)[1]
  z$max.lambda=lambda.v[max.lambda.id]
  z$max.F1 = z$lambda.F1[max.lambda.id]
  z$max.F1.sd = z$lambda.F1.sd[max.lambda.id]
  z
}

lambda.max <- 100
lambda.min <- 1e-4
n.lambdas <- 50
lambda.v <- exp(seq(log(lambda.min),log(lambda.max),length=n.lambdas))

d1.Lasso = run.glmnet.find.best.lambda(dataset.train,lambda.v,alpha=1)
d1.ridge = run.glmnet.find.best.lambda(dataset.train,lambda.v,alpha=0)

d2.Lasso = run.glmnet.find.best.lambda(dataset.cat.train,lambda.v,alpha=1)
d2.ridge = run.glmnet.find.best.lambda(dataset.cat.train,lambda.v,alpha=0)

d3.Lasso = run.glmnet.find.best.lambda(d3.pcamca.train,lambda.v,alpha=1)
d3.ridge = run.glmnet.find.best.lambda(d3.pcamca.train,lambda.v,alpha=0)

d4.Lasso = run.glmnet.find.best.lambda(d4.mca.train,lambda.v,alpha=1)
d4.ridge = run.glmnet.find.best.lambda(d4.mca.train,lambda.v,alpha=0)

#Plot results
df <- data.frame(lambda=c(lambda.v,lambda.v,lambda.v,lambda.v,lambda.v,lambda.v,lambda.v,lambda.v),
                 F1=c(d1.Lasso$lambda.F1, d1.ridge$lambda.F1, 
                      d2.Lasso$lambda.F1, d2.ridge$lambda.F1,
                      d3.Lasso$lambda.F1, d3.ridge$lambda.F1,
                      d4.Lasso$lambda.F1, d4.ridge$lambda.F1), 
                 sd=c(d1.Lasso$lambda.F1.sd, d1.ridge$lambda.F1.sd, 
                      d2.Lasso$lambda.F1.sd, d2.ridge$lambda.F1.sd,
                      d3.Lasso$lambda.F1.sd, d3.ridge$lambda.F1.sd,
                      d4.Lasso$lambda.F1.sd, d4.ridge$lambda.F1.sd),
                 Dataset=c(rep('D1 Lasso',n.lambdas),rep('D1 Ridge',n.lambdas),
                           rep('D2 Lasso',n.lambdas),rep('D2 Ridge',n.lambdas),
                           rep('D3 Lasso',n.lambdas),rep('D3 Ridge',n.lambdas),
                           rep('D4 Lasso',n.lambdas),rep('D4 Ridge',n.lambdas)))

ggplot(df, aes(x=lambda, y=F1, group=Dataset, color=Dataset)) + 
  scale_y_continuous(name = "F1") +
  geom_line() + theme_minimal() + ggtitle('Optimization of lambda - Lasso and ridge reg.') + 
  coord_cartesian(ylim=c(0.65, 0.75)) + 
  scale_x_log10(breaks = trans_breaks("log10", function(x) 10^x),
                labels = trans_format("log10", math_format(10^.x)))

ggplot(df, aes(x=lambda, y=sd, group=Dataset, color=Dataset)) + 
  scale_y_continuous(name = "sd(F1)",limits = c(0, 0.028)) + scale_x_continuous(trans='log2')+
  geom_line() + theme_minimal()


####################################################################
# LDA
####################################################################

# 10-fold CV using LDA 
run.lda <- function (dataset)
{
  createModelAndPredict <- function(train, newdata){
    lda.model <- lda(y~., train,prior=c(1,1)/2) 
    test.pred <- predict (lda.model, newdata)$class
    return(test.pred)
  }
  run.k.fold.CV(createModelAndPredict, dataset, performance.metric=c("accuracy","F1"))
}

(d1.lda = run.lda(dataset.train))
(d2.lda = run.lda(dataset.cat.train))
(d3.lda = run.lda(d3.pcamca.train))
(d4.lda = run.lda(d4.mca.train))


####################################################################
# Naïve Bayes
####################################################################

# 10-fold CV using Naive Bayes 
run.NaiveBayes <- function (dataset, laplace=0)
{
  createModelAndPredict <- function(train, newdata){
    model <- naiveBayes(y ~ ., data = train, laplace=laplace)
    test.pred <- predict (model, newdata)
    return(test.pred)
  }
  
  run.k.fold.CV(createModelAndPredict, dataset, performance.metric=c("accuracy","F1"))
}

(d1.naive = run.NaiveBayes(dataset.train))
(d2.naive = run.NaiveBayes(dataset.cat.train))
(d3.naive = run.NaiveBayes(d3.pcamca.train))
(d4.naive = run.NaiveBayes(d4.mca.train))

# Try several laplace
ls = c(0:10)
d1.naive.l.F1 = numeric(length(ls))
d2.naive.l.F1 = numeric(length(ls))
d3.naive.l.F1 = numeric(length(ls))
d4.naive.l.F1 = numeric(length(ls))
for(i in 1:length(ls)){
  print(paste("Laplace: ",ls[i]))
  d1.naive.l.F1[i] = run.NaiveBayes(dataset.train,laplace=ls[i])$F1.mean
  d2.naive.l.F1[i] = run.NaiveBayes(dataset.cat.train,laplace=ls[i])$F1.mean
  d3.naive.l.F1[i] = run.NaiveBayes(d3.pcamca.train,laplace=ls[i])$F1.mean
  d4.naive.l.F1[i] = run.NaiveBayes(d4.mca.train,laplace=ls[i])$F1.mean
}

# Plot results
df.res <- data.frame(laplace=c(ls,ls,ls,ls),
                     F1=c(d1.naive.l.F1, d2.naive.l.F1, d3.naive.l.F1, d4.naive.l.F1), 
                     Dataset=c(rep('D1',length(ls)),rep('D2',length(ls)),
                               rep('D3',length(ls)),rep('D4',length(ls))))
ggplot(df.res, aes(x=as.factor(laplace), y=F1, group=Dataset, color=Dataset)) + 
  labs(x = "laplace smoothing", y = "F1") +
  geom_line() + theme_minimal() + ggtitle('Laplace smoothing optimization  - Naive Bayes') + 
  coord_cartesian(ylim=c(0.58, 0.73)) 

####################################################################
# Multilayer Perceptrons
####################################################################

# 10-fold CV using MLP 
run.MLP <- function (dataset, nneurons, decay=0)
{
  createModelAndPredict <- function(train, newdata){
    weights = compute.weights(train$y)
    model <- nnet(y ~., data = train, weights = weights, size=nneurons, 
                  maxit=100, decay=decay, MaxNWts=10000, trace=FALSE)
    test.pred <- predict (model, newdata)
    return(probabilityToFactor(test.pred))
  }
  
  run.k.fold.CV(createModelAndPredict,dataset, performance.metric=c("accuracy","F1"))
}

optimize.decay <- function(dataset, nneurons, decays=c(0,10^seq(-3,0,by=0.1))){
  results <- list()
  for (i in 1:(length(decays)))
  { 
    print(paste("Decay: ",decays[i]))
    results[[i]] <- run.MLP(dataset,nneurons, decays[i])
  }
  z = list(decays = decays, results=results)
  z$F1 <- unlist(lapply(results,function(t) t$F1.mean))
  z$F1.sd <- unlist(lapply(results,function(t) t$F1.sd))
  z$accuracy <- unlist(lapply(results,function(t) t$accuracy.mean))
  max.idx <- which.max(z$F1)[1]
  z$max.decay <- z$decays[max.idx]
  z$max.F1 <- z$F1[max.idx]
  z
}

# We fix a large number of hidden units in one hidden layer, and explore different regularization values
nneurons <- 30
decays <- c(0,10^seq(-3,5,by=1))

(d1.mlp <- optimize.decay(dataset.train,    nneurons, decays))
(d2.mlp <- optimize.decay(dataset.cat.train,nneurons, decays))
(d3.mlp <- optimize.decay(d3.pcamca.train,  nneurons, decays))
(d4.mlp <- optimize.decay(d4.mca.train,     nneurons, decays))

df <- data.frame(decays=rep(decays,4),
                 F1=c(d1.mlp$F1, d2.mlp$F1, d3.mlp$F1, d4.mlp$F1), 
                 Dataset=c(rep('D1',length(decays)),rep('D2',length(decays)), rep('D3',length(decays)),rep('D4',length(decays))))

ggplot(df, aes(x=decays, y=F1, group=Dataset, color=Dataset)) + 
  labs(x = "Decay", y = "F1") + ggtitle('Decay opt. - MLP') + coord_cartesian(ylim=c(0.65, 0.75)) + 
  geom_line() + theme_minimal()  + scale_x_log10(breaks = trans_breaks("log10", function(x) 10^x),
                                                 labels = trans_format("log10", math_format(10^.x)))
####################################################################
# SVM
####################################################################

# 10-fold CV using SVM 
run.SVM <- function (dataset, C=1, which.kernel="linear", gamma=0.5)
{
  createModelAndPredict <- function(train, newdata){
    class.weights <- 1-table(train$y)/nrow(train) #Give more weights to YES
    switch(which.kernel,
           linear={model <- svm(y~., train, type="C-classification", cost=C, class.weights=class.weights, kernel="linear", scale = FALSE)},
           poly.2={model <- svm(y~., train, type="C-classification", cost=C, class.weights=class.weights, kernel="polynomial", degree=2, coef0=1, scale = FALSE)},
           poly.3={model <- svm(y~., train, type="C-classification", cost=C, class.weights=class.weights, kernel="polynomial", degree=3, coef0=1, scale = FALSE)},
           RBF=   {model <- svm(y~., train, type="C-classification", cost=C, class.weights=class.weights, kernel="radial", gamma=gamma, scale = FALSE)},
           stop("Enter one of 'linear', 'poly.2', 'poly.3', 'radial'"))
    test.pred <- predict (model, newdata)
    return(test.pred)
  }
  
  run.k.fold.CV(createModelAndPredict, dataset, k=10, performance.metric=c("accuracy","F1"))
}

#Run several C values
optimize.C <- function (dataset, Cs = 10^seq(-2,3), which.kernel="linear", gamma=0.5)
{
  results <- list()
  
  for(i in 1:(length(Cs))){
    print(paste("C ", Cs[i]))
    results[[i]] <- run.SVM(dataset,C=Cs[i], which.kernel=which.kernel, gamma=gamma)
  }
  
  z = list(Cs = Cs, results=results)
  
  z$F1 <- unlist(lapply(results,function(t) t$F1.mean))
  z$F1.sd <- unlist(lapply(results,function(t) t$F1.sd))
  z$accuracy <- unlist(lapply(results,function(t) t$accuracy.mean))
  
  max.C.idx <- which.max(z$F1)[1]
  z$max.C <-Cs[max.C.idx]
  z$max.F1 <- z$F1[max.C.idx]
  z$max.F1.sd <- z$F1.sd[max.C.idx]
  z
}

# Linear kernel #
Cs <- 10^seq(-3,1)
d1.svm.lin <- optimize.C(dataset.train, Cs, which.kernel="linear")
d2.svm.lin <- optimize.C(dataset.cat.train, Cs, which.kernel="linear")
d3.svm.lin <- optimize.C(d3.pcamca.train  , Cs, which.kernel="linear")
d4.svm.lin <- optimize.C(d4.mca.train     , Cs, which.kernel="linear")

#Plot results
df.res.lin <- data.frame(k=c(d1.svm.lin$Cs,d2.svm.lin$Cs,d3.svm.lin$Cs,d4.svm.lin$Cs),
                 F1=c(d1.svm.lin$F1,  d2.svm.lin$F1, d3.svm.lin$F1, d4.svm.lin$F1), 
                 group=c(rep('D1 (lineal)',length(d1.svm.lin$Cs)),rep('D2 (lineal)',length(d2.svm.lin$Cs)),rep('D3 (lineal)',length(d3.svm.lin$Cs)),rep('D4 (lineal)',length(d4.svm.lin$Cs))))

ggplot(df.res.lin, aes(x=k, y=F1, group=group, color=group)) + 
  labs(x = "C", y ="F1") +  coord_cartesian(ylim=c(0.65, 0.75)) +
  geom_line() + theme_minimal()+ ggtitle('Optimization of C - SVM (lineal)') + 
  scale_x_log10(breaks = trans_breaks("log10", function(x) 10^x),labels = trans_format("log10", math_format(10^.x)))


# Polinomial 2 #

d1.svm.poly2 <- optimize.C(dataset.train,     Cs, which.kernel="poly.2")
d2.svm.poly2 <- optimize.C(dataset.cat.train, Cs, which.kernel="poly.2")
d3.svm.poly2 <- optimize.C(d3.pcamca.train,   Cs, which.kernel="poly.2")
d4.svm.poly2 <- optimize.C(d4.mca.train,      Cs, which.kernel="poly.2")

#Plot results
df.res.poly2 <- data.frame(k=c(d1.svm.poly2$Cs, d2.svm.poly2$Cs, d3.svm.poly2$Cs, d4.svm.poly2$Cs),
                         F1=c(d1.svm.poly2$F1,  d2.svm.poly2$F1, d3.svm.poly2$F1, d4.svm.poly2$F1), 
                         group=c(rep('D1 (poly2)',length(d1.svm.poly2$Cs)),rep('D2 (poly2)',length(d2.svm.poly2$Cs)),rep('D3 (poly2)',length(d3.svm.poly2$Cs)),rep('D4 (poly2)',length(d4.svm.poly2$Cs))))

ggplot(df.res.poly2, aes(x=k, y=F1, group=group, color=group)) + 
  labs(x = "C", y ="F1") +  coord_cartesian(ylim=c(0.65, 0.75)) +
  geom_line() + theme_minimal()+ ggtitle('Optimization of C - SVM (Poly2)') + 
  scale_x_log10(breaks = trans_breaks("log10", function(x) 10^x),labels = trans_format("log10", math_format(10^.x)))


# Polinomial 3 #

d1.svm.poly3 <- optimize.C(dataset.train,     Cs, which.kernel="poly.3")
d2.svm.poly3 <- optimize.C(dataset.cat.train, Cs, which.kernel="poly.3")
d3.svm.poly3 <- optimize.C(d3.pcamca.train,   Cs, which.kernel="poly.3")
d4.svm.poly3 <- optimize.C(d4.mca.train,      Cs, which.kernel="poly.3")

#Plot results
df.res.poly3 <- data.frame(k=c(d1.svm.poly3$Cs,d2.svm.poly3$Cs,d3.svm.poly3$Cs,d4.svm.poly3$Cs),
                           F1=c(d1.svm.poly3$F1,  d2.svm.poly3$F1, d3.svm.poly3$F1, d4.svm.poly3$F1), 
                           group=c(rep('D1 (poly3)',length(d1.svm.poly3$Cs)),rep('D2 (poly3)',length(d2.svm.poly3$Cs)),rep('D3 (poly3)',length(d3.svm.poly3$Cs)),rep('D4 (poly3)',length(d4.svm.poly3$Cs))))

ggplot(df.res.poly3, aes(x=k, y=F1, group=group, color=group)) + 
  labs(x = "C", y ="F1") +  coord_cartesian(ylim=c(0.65, 0.75)) +
  geom_line() + theme_minimal()+ ggtitle('Optimization of C - SVM (Poly3)') + 
  scale_x_log10(breaks = trans_breaks("log10", function(x) 10^x),labels = trans_format("log10", math_format(10^.x)))


# RBF #

gammas <- 2^seq(-3,2)
d1.svm.RBF.F1 <- matrix(0,length(Cs),length(gammas))
d2.svm.RBF.F1 <- matrix(0,length(Cs),length(gammas))
d3.svm.RBF.F1 <- matrix(0,length(Cs),length(gammas))
d4.svm.RBF.F1 <- matrix(0,length(Cs),length(gammas))
for (i in gammas) #Grid search: gamma and C
{
  print(paste("gamma ", gammas[i]))
  d1.svm.RBF.F1[,i] <- optimize.C(dataset.train,    Cs, which.kernel="RBF", gamma= gammas[i])$F1
  d2.svm.RBF.F1[,i] <- optimize.C(dataset.cat.train,Cs, which.kernel="RBF", gamma=gammas[i])$F1
  d3.svm.RBF.F1[,i] <- optimize.C(d3.pcamca.train,  Cs, which.kernel="RBF", gamma=gammas[i])$F1
  d4.svm.RBF.F1[,i] <- optimize.C(d4.mca.train,     Cs, which.kernel="RBF", gamma=gammas[i])$F1
}

# Plot results
df.res.svm = data.frame()
for(i in 1:(length(Cs))){
  for(j in 1:(length(gammas))){
    df.tmp <- data.frame(C=rep(Cs[i],4),
                         gamma=rep(gammas[j],4),
                         F1=c(d1.svm.RBF.F1[i,j],  d2.svm.RBF.F1[i,j], d3.svm.RBF.F1[i,j], d4.svm.RBF.F1[i,j]), 
                         group=c('D1 (RBF)','D2 (RBF)','D3 (RBF)','D4 (RBF)'))
    df.res.svm <- rbind(df.res.svm,df.tmp)
  }
}
df.res.svm$C <- as.factor(df.res.svm$C)
df.res.svm$gamma <- as.factor(df.res.svm$gamma)

ggplot(data = df.res.svm[df.res.svm$group=='D1 (RBF)',], aes(x=C, y=gamma, fill=F1)) +
  scale_fill_gradient2(limit = c(0,1))+ geom_tile() +ggtitle('Grid search (gamma and C) for D1 - SVM (RBF)')
ggplot(data = df.res.svm[df.res.svm$group=='D2 (RBF)',], aes(x=C, y=gamma, fill=F1)) + 
  scale_fill_gradient2(limit = c(0,1))+ geom_tile() +ggtitle('Grid search (gamma and C) for D2- SVM (RBF)')
ggplot(data = df.res.svm[df.res.svm$group=='D3 (RBF)',], aes(x=C, y=gamma, fill=F1)) + 
  scale_fill_gradient2(limit = c(0,1))+ geom_tile() +ggtitle('Grid search (gamma and C) for D3- SVM (RBF)')
ggplot(data = df.res.svm[df.res.svm$group=='D4 (RBF)',], aes(x=C, y=gamma, fill=F1)) + 
  scale_fill_gradient2(limit = c(0,1))+ geom_tile() +ggtitle('Grid search (gamma and C) for D4 - SVM (RBF)')

####################################################################
# Decision tree
####################################################################

# 10-fold CV using decision tree 
run.tree <- function (dataset)
{
  createModelAndPredict <- function(train, newdata){
    weights = compute.weights(train$y)
    model <- rpart(y ~ ., weights= weights, data=train)
    test.pred <- predict (model, newdata, type="class")
    return(test.pred)
  }
  run.k.fold.CV(createModelAndPredict, dataset, performance.metric=c("accuracy","F1"))
}

(d1.tree = run.tree(dataset.train))
(d2.tree = run.tree(dataset.cat.train))
(d3.tree = run.tree(d3.pcamca.train))
(d4.tree = run.tree(d4.mca.train))

####################################################################
# Random forest
####################################################################

# 10-fold CV using random forest
run.randomForest <- function (dataset, ntree=100)
{
  createModelAndPredict <- function(train, newdata){
    class.sampsize <- min(table(train$y))
    model <- randomForest(y ~ ., data=train, ntree=ntree, proximity=FALSE, 
                          sampsize=c(yes=class.sampsize, no=class.sampsize), strata=train$y)
    test.pred <- predict (model, newdata, type="class")
    return(test.pred)
  }
  run.k.fold.CV(createModelAndPredict, dataset, performance.metric=c("accuracy","F1"))
}

#Run severl ntrees
optimize.ntrees <- function(dataset, ntrees=round(10^seq(1,2,by=0.2))){
  results <- list()
  for (i in 1:(length(ntrees)))
  { 
    print(paste("ntrees: ",ntrees[i]))
    results[[i]] <- run.randomForest(dataset,ntrees[i])
  }
  z = list(ntrees = ntrees)
  z$F1 <- unlist(lapply(results,function(t) t$F1.mean))
  z$accuracy <- unlist(lapply(results,function(t) t$accuracy.mean))
  max.idx <- which.max(z$F1)[1]
  z$max.ntrees <- z$ntrees[max.idx]
  z$max.F1 <- z$F1[max.idx]
  z
}

ntrees= round(10^seq(1,3,by=0.2))
(d1.randomForest = optimize.ntrees(dataset.train, ntrees))
(d2.randomForest = optimize.ntrees(dataset.cat.train, ntrees))
(d3.randomForest = optimize.ntrees(d3.pcamca.train, ntrees))
(d4.randomForest = optimize.ntrees(d4.mca.train, ntrees))


#Plot results
df <- data.frame(ntree=rep(ntrees,4),
                 F1=c(d1.randomForest$F1, d2.randomForest$F1, d3.randomForest$F1, d4.randomForest$F1), 
                 Dataset=c(rep('D1',length(ntrees)),rep('D2',length(ntrees)), rep('D3',length(ntrees)),rep('D4',length(ntrees))))

ggplot(df, aes(x=ntree, y=F1, group=Dataset, color=Dataset)) + 
  labs(x = "Number of trees", y = "F1") + ggtitle('Number of trees - Random forest') + 
  geom_line() + theme_minimal()  + scale_x_log10(breaks = trans_breaks("log10", function(x) 10^x),
                                                 labels = trans_format("log10", math_format(10^.x)))

