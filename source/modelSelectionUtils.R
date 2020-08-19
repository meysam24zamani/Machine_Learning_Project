#############################################
# Some functions usefull for the prediction #
#############################################

# Function that computes the accuracy given prediction and real values
accuracy <- function (pred, real)
{
  ct <- table(Truth=real, Pred=pred)
  round(100*(sum(diag(ct))/sum(ct)),2)
}

# harmonic mean
harm <- function (a,b) { 2/(1/a+1/b) }

# Function that computes the F1 Score (performance measure): the harmonic mean of precision and recall
F1 <- function (pred, real)
{
  ct <- table(Truth=real, Pred=pred)
  harm (prop.table(ct,1)[2,2], prop.table(ct,2)[2,2])
}

# Function that runs a k-fold-CV using:
# - The generateModelAndPredict function creates the model and predicts in each fold, 
# - The performance.metric computes the goodness of the current fold: accuracy or F1
run.k.fold.CV <- function(generateModelAndPredict, dataset, performance.metric = c("accuracy","F1"), k = 10, trace = TRUE){
  set.seed(1234)
  CV.folds <- generateCVRuns (dataset$y, ntimes=1, nfold=k, stratified=TRUE)
  
  z = list() #Output
  if("F1" %in% performance.metric) 
    z$F1 = numeric(k)
  if("accuracy" %in% performance.metric)
    z$accuracy = numeric(k)
  
  if(trace)
    cat('Starting k-fold CV\n')
  for (j in 1:k)
  {
    if(trace)
      print(paste(' # Fold ',j,"/",k))
    va <- unlist(CV.folds[[1]][[j]])
    pred.va <- generateModelAndPredict(dataset[-va,], dataset[va,])
    # Accuracy and F1
    if("F1" %in% performance.metric) 
      z$F1[j] = F1(pred.va, dataset[va,]$y)
    if("accuracy" %in% performance.metric)
      z$accuracy[j] = accuracy(pred.va, dataset[va,]$y)
  }
  
  if("F1" %in% performance.metric) {
    z$F1.mean = mean(z$F1)
    z$F1.sd = sd(z$F1)
  }
  if("accuracy" %in% performance.metric){
    z$accuracy.mean = mean(z$accuracy)
    z$accuracy.sd = sd(z$accuracy)
  }
  z
}

# Function that computes the weights of each class
compute.weights = function(y){
  priors = table(y)/length(y)
  weights = numeric(length(y))
  weights[y=="yes"] = 1/priors["yes"]
  weights[y=="no"] = 1/priors["no"]
  weights
}

# Function that given a vector of probabilities, returns a vector of factors "no"/"yes"
probabilityToFactor <- function(v, P=.5){
  result = factor(levels = c("no","yes")) 
  result[v<P] <- "no"
  result[v>=P] <- "yes"
  result
}
