# --------------------------------------#
# --------------------------------------#
# Project : Portfolio optimization and ML
# Authors : A.Gomez - A.Poizat
# --------------------------------------#
# --------------------------------------#

library(tidyverse)
library(glmnet)
library(neuralnet)
library(glasso) 
library(cvCovEst)
library(plot.matrix)
library(Rsolnp) 
library(ggplot2)
rm(list=ls())

f.eq <- function(w,mu,sigma, target.mu){
  return(c(sum(w)-1,t(w)%*%mu))
}

f.target <- function(w, mu, sigma, target.mu){
  return(t(w)%*%mu)
}

f.to.min <- function(w,mu,sigma,target.mu){
  return(t(w)%*%sigma%*%w)
}

setwd("C:/Users/poiza/Desktop/Cours/Advanced")
#setwd("")
data <- read.csv('2009_logstock.csv',sep = ",")
data2 <- read.csv('2009_cov.csv',sep = ",")
data2 <- data2[1658:1764,]

#Dropping dates
data <- data[,-1]

# ----------------------- #
# 1. Data ----
# ----------------------- #

#Splitting data into training (2009-2015) and testing (2016-2017)
train <- data[1:83,]
test <- data[84:nrow(data),]
train2 <- data2[1:83,]
test2 <- data2[84:nrow(data2),]

#Histograms of the log stock returns
op <- par(mfrow = c(2,2))
for(j in 1:ncol(data)){
  hist(data[,j],xlab=paste('Log return of ',colnames(data)[j]), main = paste('Returns histogram of', colnames(data)[j]))
}
par(op)

#Correlation structure

library(plotly)
library(quantmod)
library(reshape2)
library(ggplot2)

#Correlation between the covariates
data_all <- data.frame(data,data2[,2:10])
corrs <- cor(data_all)
cormat <- round(corrs,2)

melted_cormat <- melt(cormat)
head(melted_cormat)

ggplot(data = melted_cormat, aes(x=Var1, y=Var2, fill=value)) + 
  geom_tile(color="white")+scale_fill_gradient2(low="blue", high="red")

# ----------------------- #
# 2. Modelling ----
# ----------------------- #

BICs.glm <- c()
BICs.cven <- c()
for (i in 1:ncol(train)){
  data_full <- cbind(train[2:(nrow(train)),i], train2[1:(nrow(train2)-1),2:10], train[1:(nrow(train)-1),-i])
  stock_name <- colnames(data)[i]
  
  # SimpleGLM
  
  y <- data_full[,1]
  x <- data_full[,-1]
  dat.glmnet <- data.frame(y,x)
  
  lm.fit <- glm(formula = 'y~.', family = gaussian, data = dat.glmnet)
  
  # CV-EN 
  
  cven.fit <-cv.glmnet(as.matrix(x),y, alpha = 0.1, nfolds = 3)
  plot(cven.fit, xlab=stock_name)
  best.lambda<-cven.fit$lambda.min
  coef.cven<-predict(cven.fit,type="coefficients",s=best.lambda)[2:(ncol(x)+1)]
  
  x.selected <- x[,coef.cven != 0]
  
  dat.glmnet.posten <- data.frame(y,x.selected)
  
  posten.fit <- glm('y~.', family = gaussian, data = dat.glmnet.posten)
  summary(posten.fit)
  BICs.glm <- append(BICs.glm, BIC(lm.fit))
  BICs.cven <- append(BICs.cven, BIC(posten.fit))
  
  fitted <- posten.fit$fitted.values
  
  plot(y,main="Fitted posterior EN",xlab=stock_name, type = 'l')
  lines(fitted, col = 'red')
  
  plot(cumprod(1+y),main="Cumulative product of fitted posterior EN" ,xlab=stock_name,ylab="Cumulative product", type = 'l',ylim = c(0,12))
  lines(cumprod(1+fitted), col = 'red')
  
  # SmallNN and NN
  
  colnames(data_full)[1] <- 'y'
  mins <- apply(data_full, 2, min)
  maxs <- apply(data_full, 2, max)
  
  data.final <- as.data.frame(scale(data_full, center = mins, scale = maxs - mins))
  
  mod_nn_small <- neuralnet('y~.', hidden = c(3,3), rep = 1, stepmax = 1e+8, data = data.final, linear.output = T)
  mod_nn <- neuralnet('y~.', hidden = c(5,10,5), rep = 1, stepmax = 1e+8, data = data.final, linear.output = T)
  
  b<-predict(mod_nn,data.final)
  b_small<-predict(mod_nn_small,data.final)
  
  b.rescaled<-b*(max(data_full$y)-min(data_full$y))+min(data_full$y)
  b_small.rescaled<-b_small*(max(data_full$y)-min(data_full$y))+min(data_full$y)
  
  plot(cumprod(1+data_full$y),main="Cumulative product of NN" ,xlab=stock_name,ylab="Cumulative product",type = 'l', col = 'black')
  lines(cumprod(1+b.rescaled), col = 'red')
  plot(cumprod(1+data_full$y),main="Cumulative product of NN (small)",xlab=stock_name,ylab="Cumulative product",type = 'l', col = 'black')
  lines(cumprod(1+b_small.rescaled), col = 'red')
  
  plot(cumprod(1+data_full$y),main="Cumulative products of both NN",xlab=stock_name,ylab="Cumulative product",type = 'l', col = 'black')
  lines(cumprod(1+b.rescaled), col = 'red')
  lines(cumprod(1+b_small.rescaled), col = 'blue')
  
  #Saving models
  name_nn <- paste("mod", i, sep="")
  name_nn_small <- paste("mod_small", i, sep="")
  name_glm <- paste("modlm", i,sep="")
  name_reg <- paste("moden", i, sep="")
  assign(name_nn, mod_nn)
  assign(name_nn_small, mod_nn_small)
  assign(name_glm, lm.fit)
  assign(name_reg, posten.fit)
  
  #Performance of the models on the test set
  test_full <- cbind(test[2:(nrow(test)),i], test2[1:(nrow(test2)-1),2:10], test[1:(nrow(test)-1),-i])
  
  #SimpleGLM
  predictions1 <- predict(lm.fit, newdata=test_full[,-1])
  plot(test_full[,1],main="Predictions for SimpleGLM model",xlab=paste(stock_name,"2015-2017"),ylab="Log returns", type='l')
  lines(predictions1, col="red")
  
  #CV-EN
  predictions2 <- predict(posten.fit, newdata=test_full[,-1])
  plot(test_full[,1],main="Predictions for CV-EN model",xlab=paste(stock_name,"2015-2017"),ylab="Log returns", type='l')
  lines(predictions2, col="red")
  
  #SmallNN
  mins <- apply(test_full, 2, min)
  maxs <- apply(test_full, 2, max)
  test.final <- as.data.frame(scale(test_full, center = mins, scale = maxs - mins))
  
  pred <- predict(mod_nn_small, newdata=test.final[,-1])
  predictions3 <- pred*(max(test_full[,1])-min(test_full[,1]))+min(test_full[,1])
  plot(test_full[,1],main="Predictions for small NN model",xlab=paste(stock_name,"2015-2017"), ylab="Log returns",type='l')
  lines(predictions3, col="red")
  
  #NN
  pred <- predict(mod_nn, newdata=test.final[,-1])
  predictions4 <- pred*(max(test_full[,1])-min(test_full[,1]))+min(test_full[,1])
  plot(test_full[,1],main="Predictions for NN model",xlab=paste(stock_name,"2015-2017"),ylab="Log returns", type='l')
  lines(predictions4, col="red")
  
  #Comparison of the models
  plot(test_full[,1],main="Comparison of the models",xlab=paste(stock_name,"2015-2017"),ylab="Log returns", type='l')
  lines(predictions1, col="blue")
  lines(predictions2, col="cyan")
  lines(predictions3, col="green")
  lines(predictions4, col="red")
  
  
  print(paste("Models obtained for",stock_name))
}
BICs.glm
BICs.cven


# ----------------------- #
# 1. Portfolio optimization ----
# ----------------------- #

cov.mat <- cov(test)
cov.mat.lin.shrinkage <- linearShrinkLWEst(dat = test)

op <- par(mfrow = c(1,2))
plot(abs(cov2cor(cov.mat)),main = 'Base Var-Cov matrix')
plot(abs(cov2cor(cov.mat.lin.shrinkage)), main = 'Var-Cov matrix with Linear Shrinkage')

rhos <- seq(from = .001,to = 0.02, length.out = 10)
g.lasso.obj <- vector('list',length = length(rhos))

op <- par(mfrow = c(2,2))
plot(cov2cor(cov.mat), main = 'Base Var-Cov matrix')
BICs  <- numeric(length = length(rhos))
AICs <- numeric(length = length(rhos))
for(j in 1:length(rhos)){
  g.lasso.obj[[j]] <- glasso(cov.mat, rho = rhos[j], nobs = nrow(test))
  Loglik.j <- g.lasso.obj[[j]]$loglik - sum(apply(abs(g.lasso.obj[[j]]$w), 2, sum)) #Loglik gives the penalized Log, the penalty being the sum of the absolute values of all elements in the resulting var-cov matrix
  DF <- nrow(which(g.lasso.obj[[j]]$w != 0, arr.ind = T)) #DF is approximated by the number of elements different from zero in the matrix
  BICs[j] <- -2*Loglik.j + log(nrow(test)) * DF #Compute the BC = -2*LL + log(n) * DF
  AICs[j] <- -2*Loglik.j + 2*DF
  if(j %in% c(1,length(rhos)/2,length(rhos))){
    plot(abs(cov2cor(g.lasso.obj[[j]]$w)), main = paste('G. Lasso, rho =',rhos[j] )) 
  }
}
par(op)

par(mfrow=c(1,1))
plot(BICs, type = 'l') 
#Optimality at 2nd value of rho
abline(v = 2, col = 'red', lty = 2)

plot(AICs, type = 'l')

preds <- data.frame()
weights <- data.frame()
weights_LS <- data.frame()
weights_GL <- data.frame()

#Choose rolling window size and model type
# "modlm" : Simple glm / "moden" : Elastic-Net / "mod" : NN / "mod_small" : NN (small)

#Tests mentioned in the report 
model_type = "modlm"
index_wind <- 1
window_size <- index_wind + 1

#For each test, save weights, 

# model_type = "moden"
# index_wind <- 1
# window_size <- index_wind + 1

# model_type = "mod"
# index_wind <- 1
# window_size <- index_wind + 1

# model_type = "mod_small"
# index_wind <- 1
# window_size <- index_wind + 1

# model_type = "modlm"
# index_wind <- 7
# window_size <- index_wind + 1

# model_type = "moden"
# index_wind <- 7
# window_size <- index_wind + 1

# model_type = "mod"
# index_wind <- 7
# window_size <- index_wind + 1

# model_type = "mod_small"
# index_wind <- 7
# window_size <- index_wind + 1


for (k in 1:(nrow(test)-window_size-1)){
  
  pred_t1 <- c()
  for (i in 1:ncol(data)){
    data_full <- cbind(test[2:(nrow(test)),i], test2[1:(nrow(test2)-1),2:10], test[1:(nrow(test)-1),-i])
    #Scaling the data for predictions in a NN
    mins <- apply(data_full, 2, min)
    maxs <- apply(data_full, 2, max)
    data.final <- as.data.frame(scale(data_full, center = mins, scale = maxs - mins))
    
    #Retrieving the model of the current stock
    namenn <- paste(model_type,i, sep="")
    model <- get(namenn)
  
    #Obtaining predictions to build mu
    if(model_type=="mod" || model_type=="mod_small"){  
      b <- predict(model, newdata=data.final[k:k+window_size,-1])
      pred <- b*(max(data_full[,1])-min(data_full[,1]))+min(data_full[,1])
      pred_t1 <- append(pred_t1, pred)
    }else{
      pred <- predict(model, newdata=data_full[k:k+window_size,-1])
      pred_t1 <- append(pred_t1, pred)
    }
  }
  preds <- rbind(preds, pred_t1)
  
  par0 <- rep(1/ncol(test), ncol(test))
  mu <- as.numeric(pred_t1)
  target.mu <- seq(from = min(mu), to = max(mu), length = 4)
  
  #Based on the empirical var-cov matrix
  
  Sharpe.optim <- 0
  for(j in 1:length(target.mu)){
    base.optim <- solnp(pars = par0, fun = f.to.min, eqfun = f.eq, eqB = c(0,0), LB = rep(0, length(par0)), mu = mu, sigma = cov.mat, target.mu = target.mu[j])
    Sharpe <- target.mu[j]/f.to.min(base.optim$pars, mu = mu, sigma = cov.mat, target.mu = target.mu[j])
    if(Sharpe > Sharpe.optim){
      optim.weight <- base.optim$pars
    }
  }
  
  #Based on Linear Shrinked var-cov
  Sharpe.optim <- 0
  for(j in 1:length(target.mu)){
    base.optim <- solnp(pars = par0, fun = f.to.min, eqfun = f.eq, eqB = c(0,0), LB = rep(0, length(par0)), mu = mu, sigma = cov.mat.lin.shrinkage, target.mu = target.mu[j])
    Sharpe <- target.mu[j]/f.to.min(base.optim$pars, mu = mu, sigma = cov.mat.lin.shrinkage, target.mu = target.mu[j])
    if(Sharpe > Sharpe.optim){
      optim.weight.LS <- base.optim$pars
    }
  }
  
  #Base on Graphical Lasso with rho[5]
  Sharpe.optim <- 0
  for(j in 1:length(target.mu)){
    base.optim <- solnp(pars = par0, fun = f.to.min, eqfun = f.eq, eqB = c(0,0), LB = rep(0, length(par0)), mu = mu, sigma = g.lasso.obj[[3]]$w, target.mu = target.mu[j])
    Sharpe <- target.mu[j]/f.to.min(base.optim$pars, mu = mu, sigma = g.lasso.obj[[5]]$w, target.mu = target.mu[j])
    if(Sharpe > Sharpe.optim){
      optim.weight.GLasso <- base.optim$pars
    }
  }
  
  weights <- rbind(weights, optim.weight)
  weights_LS <- rbind(weights, optim.weight.LS)
  weights_GL <- rbind(weights, optim.weight.GLasso)
  
}

#Depending on the chosen model and window size, save the weights
w_glm_1 <- weights
#w_cven_1 <- weights
#w_nn_1 <- weights
#w_snn_1 <- weights

#w_glm_7 <- weights
#w_cven_7 <- weights
#w_nn_7 <- weights
#w_snn_7 <- weights

# write.csv(w_glm_1, "C:/Users/poiza/Desktop/Cours/Advanced/weights_ptf1.csv", row.names=FALSE)
# write.csv(w_cven_1, "C:/Users/poiza/Desktop/Cours/Advanced/weights_ptf2.csv", row.names=FALSE)
# write.csv(w_nn_1, "C:/Users/poiza/Desktop/Cours/Advanced/weights_ptf3.csv", row.names=FALSE)
# write.csv(w_snn_1, "C:/Users/poiza/Desktop/Cours/Advanced/weights_ptf4.csv", row.names=FALSE)
# write.csv(w_glm_1, "C:/Users/poiza/Desktop/Cours/Advanced/weights_ptf5.csv", row.names=FALSE)
# write.csv(w_cven_7, "C:/Users/poiza/Desktop/Cours/Advanced/weights_ptf6.csv", row.names=FALSE)
# write.csv(w_nn_7, "C:/Users/poiza/Desktop/Cours/Advanced/weights_ptf7.csv", row.names=FALSE)
# write.csv(w_snn_7, "C:/Users/poiza/Desktop/Cours/Advanced/weights_ptf8.csv", row.names=FALSE)


#Select weights
w <- w_glm_1


#Performance on out-of-sample
capital <- 1000
revenues <- c()
revenues <- append(revenues, capital)
profit <- capital
for(j in 0:(nrow(test)-3-window_size)){
  testing <- test[(window_size+1+j),]
  ret.ptf <- apply(w[j+1,]*profit*testing,1,sum)
  profit <- profit + ret.ptf
  revenues <- append(revenues, profit)
}
revenues[15]
#revenues[21]
par(mfrow=c(1,1))
plot(revenues,type='l', xlab="Months (2016-2017)",main="Evolution of the invested capital")




