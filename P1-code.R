
setwd("C:\\Users\\xiangjux\\Dropbox\\2020-ML\\P1")
library(caTools)
require(ggplot2)
require(reshape2)
library(tree)
library(h2o)
library(gbm)
library(caret)
range = (1:10)/10
#a=as.numeric(Sys.time())
#set.seed(a)
set.seed(11111)
###########
### https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009
###########

wine = read.table("winequality-red.csv",head=T,sep=",")
wine$Outcome=factor(ifelse(wine$quality<=6,0,1))
wine$quality = NULL

###########
### https://www.kaggle.com/uciml/pima-indians-diabetes-database
###########

diabetes = read.table("diabetes.csv",sep=",",head=T)
diabetes$Outcome = factor(diabetes$Outcome)


###########
##checking datasets, and split train, test parts
##generating 10 training sets, from 10% to 100%
##no missing records were found, therefore no missing value problem
##all predictors were numeric, continuous, no categorical, no qualitive ones
###########
summary(wine)
summary(diabetes)

melt.wine=melt(wine)
ggplot(data = melt.wine, aes(x = value)) + stat_density() + facet_wrap(~variable, scales = "free")
boxplot(wine)

melt.diabetes=melt(diabetes)
ggplot(data = melt.diabetes, aes(x = value)) + stat_density() + facet_wrap(~variable, scales = "free")
boxplot(diabetes)

split1 = sample.split(wine$Outcome, 0.7)
train_wine = wine[split1,]
test_wine  = wine[!split1,]

split2 = sample.split(diabetes$Outcome, 0.7)
train_diabetes = diabetes[split2,]
test_diabetes  = diabetes[!split2,]

test_wine[-12] = scale(test_wine[-12])
test_diabetes[-9] = scale(test_diabetes[-9])

for( rate in range){

  temp1 = paste("train_wine.",rate,sep="")
  temp2 = sample(1:dim(train_wine)[1],rate*dim(train_wine)[1])
  temp3 = train_wine[temp2,]
  temp3[-12] = scale(temp3[-12])
  assign(temp1,temp3 )

  temp1 = paste("train_diabetes.",rate,sep="")
  temp2 = sample(1:dim(train_diabetes)[1],rate*dim(train_diabetes)[1])
  temp3 = train_diabetes[temp2,]
  temp3[-9] = scale(temp3[-9])
  assign(temp1,temp3 )  
}

######################################
###tree classification
###train dataset for complexity curve
###learning curve
######################################


test = train_wine.1
tree.test = tree(Outcome~.-Outcome, test)
summary(tree.test)
cv.test = cv.tree(tree.test,FUN=prune.misclass)
cv.test
plot(cv.test$size,cv.test$dev/nrow(test),type="b", xlab = "Tree Size", ylab = "CV Error Rate",main="Red Wine")

test = train_diabetes.1
tree.test = tree(Outcome~.-Outcome, test)
summary(tree.test)
cv.test = cv.tree(tree.test,FUN=prune.misclass)
cv.test
plot(cv.test$size,cv.test$dev/nrow(test),type="b", xlab = "Tree Size", ylab = "CV Error Rate",main="Diabetes")

##wine the best size is 6
##diabetes the best size is 5
result1=NULL
result2=NULL
for( rate in range){
  
  i1=get(paste("train_wine.",rate,sep=""))
  tree.i1 = tree(Outcome~.-Outcome,i1)
  cv.tree.i1 = cv.tree(tree.i1,FUN=prune.misclass)
  num1 = cv.tree.i1$size[which.min(cv.tree.i1$dev)]
  prune.i1 = prune.misclass(tree.i1,best = num1)
  train.i1.pred = predict(prune.i1,i1,type="class")
  train.wine.error   = sum(train.i1.pred != i1$Outcome)/length(i1$Outcome)
  test.wine.pred = predict(prune.i1,test_wine,type="class")
  test.wine.error = sum(test.wine.pred != test_wine$Outcome)/length(test_wine$Outcome)

  j1=get(paste("train_diabetes.",rate,sep=""))
  tree.j1 = tree(Outcome~.-Outcome,j1)
  cv.tree.j1 = cv.tree(tree.j1,FUN=prune.misclass)
  num2 = cv.tree.j1$size[which.min(cv.tree.j1$dev)]
  prune.j1 = prune.misclass(tree.j1,best = num2)
  train.j1.pred = predict(prune.j1,j1,type="class")
  train.diabetes.error   = sum(train.j1.pred != j1$Outcome)/length(j1$Outcome)
  test.diabetes.pred = predict(prune.j1,test_diabetes,type="class")
  test.diabetes.error = sum(test.diabetes.pred != test_diabetes$Outcome)/length(test_diabetes$Outcome)
  
  result1 = rbind(result1,c(train.wine.error,test.wine.error))
  result2 = rbind(result2,c(train.diabetes.error,test.diabetes.error))
}

plot(range,result1[,1],type="b",col="red",ylim=c(0,0.5),main="Red Wine Data",xlab="Proportion of Training Data",ylab="Error Rate")
lines(range,result1[,2],col="blue",type="b")
legend(0.8, 0.4, legend=c("Training Data", "Testing Data"),
       col=c("red", "blue"), lty=1:1, cex=1)

plot(range,result2[,1],type="b",col="red",ylim=c(0,0.5),main="Diabetes Data",xlab="Proportion of Training Data",ylab="Error Rate")
lines(range,result2[,2],col="blue",type="b")
legend(0.8, 0.4, legend=c("Training Data", "Testing Data"),
       col=c("red", "blue"), lty=1:1, cex=1)

######################################
###neural network
###train dataset for complexity curve
###learning curve
######################################

hyper_params = list(hidden = list(rep(20,2),rep(20,3),rep(20,4),rep(20,5),rep(20,6),rep(20,7),rep(20,8),rep(20,9)))
  
test = train_wine.1
h2o.init(nthreads = -1)
model = h2o.grid(algorithm = "deeplearning",
                         y = 'Outcome',
                         training_frame = as.h2o(test),
                         activation = 'Rectifier',
                         hyper_params = hyper_params,
                         epochs = 100,
                         grid_id="model",
                         train_samples_per_iteration = -2)

grid <- h2o.getGrid("model", sort_by = "err", decreasing = FALSE)
find = function (x){nchar(gsub("[^0-9.-]", "", x))/2}
x1 = grid@summary_table$hidden
x2 = unlist(lapply(x1,find))
x3 = cbind(x2,grid@summary_table$err)
x3 = x3[order(x3[,1]),]
best = x2[1]
plot(x3[,1],x3[,2],type="b",xlab="Layer Number", ylab="Error Rate", main="Red Wine")

h2o.shutdown(F)

test = train_diabetes.1
h2o.init(nthreads = -1)
model = h2o.grid(algorithm = "deeplearning",
                 y = 'Outcome',
                 training_frame = as.h2o(test),
                 activation = 'Rectifier',
                 hyper_params = hyper_params,
                 epochs = 100,
                 grid_id="model",
                 train_samples_per_iteration = -2)

grid <- h2o.getGrid("model", sort_by = "err", decreasing = FALSE)
find = function (x){nchar(gsub("[^0-9.-]", "", x))/2}
x1 = grid@summary_table$hidden
x2 = unlist(lapply(x1,find))
x3 = cbind(x2,grid@summary_table$err)
x3 = x3[order(x3[,1]),]
best = x2[1]
plot(x3[,1],x3[,2],type="b",xlab="Layer Number", ylab="Error Rate", main="Diabetes")

h2o.shutdown(F)

##wine is 7
##diabetes is 6

hyper_params = list(hidden = list(rep(20,2),rep(20,3),rep(20,4),rep(20,5),rep(20,6),rep(20,7),rep(20,8),rep(20,9)))
result1=NULL
result2=NULL

rate = range[10] ##h2o server cannot be looped for connection, has to be manually set rate value from 0.1 to 1 !!!!
  
  i1=get(paste("train_wine.",rate,sep=""))
  h2o.init(nthreads =-1)
  #Sys.sleep(10)
  model = h2o.grid(algorithm = "deeplearning",
                   y = 'Outcome',
                   training_frame = as.h2o(i1),
                   activation = 'Rectifier',
                   hyper_params = hyper_params,
                   epochs = 100,
                   grid_id="model",
                   train_samples_per_iteration = -2)
  
  #Sys.sleep(10)
  grid <- h2o.getGrid("model", sort_by = "err", decreasing = FALSE)
  find = function (x){nchar(gsub("[^0-9.-]", "", x))/2}
  x1 = grid@summary_table$hidden
  x2 = unlist(lapply(x1,find))
  x3 = cbind(x2,grid@summary_table$err)
  x3 = x3[order(x3[,1]),]
  best1 = x3[1]
 
  h2o.shutdown(F)
  
  h2o.init(nthreads =-1)
  #Sys.sleep(10)
  model = h2o.deeplearning(y = 'Outcome',
                           training_frame = as.h2o(i1),
                           activation = 'Rectifier',
                           hidden = rep(20,best1),
                           epochs = 100,
                           train_samples_per_iteration = -2)
  #Sys.sleep(10)
  i1_error = model@model$training_metrics@metrics$cm$table$Error[3]
  y_pred = h2o.predict(model, newdata = as.h2o(test_wine[-12]))
  y_pred = as.data.frame(y_pred)
  y_pred = as.numeric(y_pred$p1>0.5)
  test_wine_error = sum(y_pred != test_wine$Outcome)/length(test_wine$Outcome)

  h2o.shutdown(F)
###
  
  j1=get(paste("train_diabetes.",rate,sep=""))
  h2o.init(nthreads =-1)
  #Sys.sleep(10)
  model = h2o.grid(algorithm = "deeplearning",
                   y = 'Outcome',
                   training_frame = as.h2o(j1),
                   activation = 'Rectifier',
                   hyper_params = hyper_params,
                   epochs = 100,
                   grid_id="model",
                   train_samples_per_iteration = -2)
  
  #Sys.sleep(10)
  grid <- h2o.getGrid("model", sort_by = "err", decreasing = FALSE)
  find = function (x){nchar(gsub("[^0-9.-]", "", x))/2}
  x1 = grid@summary_table$hidden
  x2 = unlist(lapply(x1,find))
  x3 = cbind(x2,grid@summary_table$err)
  x3 = x3[order(x3[,1]),]
  best2 = x3[1]

  h2o.shutdown(F)
  
  h2o.init(nthreads =-1)
  #Sys.sleep(10)
  model = h2o.deeplearning(y = 'Outcome',
                           training_frame = as.h2o(j1),
                           activation = 'Rectifier',
                           hidden = rep(20,best2),
                           epochs = 100,
                           train_samples_per_iteration = -2)
  j1_error = model@model$training_metrics@metrics$cm$table$Error[3]
  y_pred = h2o.predict(model, newdata = as.h2o(test_diabetes[-9]))
  y_pred = as.data.frame(y_pred)
  y_pred = as.numeric(y_pred$p1>0.5)
  test_diabetes_error = sum(y_pred != test_diabetes$Outcome)/length(test_diabetes$Outcome)
  
  h2o.shutdown(F)  
  
  result1 = rbind(result1,c(i1_error,test_wine_error))
  result2 = rbind(result2,c(j1_error,test_diabetes_error))

  
#########

plot(range,result1[,1],type="b",col="red",ylim=c(0,0.5),main="Red Wine Data",xlab="Proportion of Training Data",ylab="Error Rate")
lines(range,result1[,2],col="blue",type="b")
legend(0.8, 0.4, legend=c("Training Data", "Testing Data"),
       col=c("red", "blue"), lty=1:1, cex=1)

plot(range,result2[,1],type="b",col="red",ylim=c(0,0.5),main="Diabetes Data",xlab="Proportion of Training Data",ylab="Error Rate")
lines(range,result2[,2],col="blue",type="b")
legend(0.8, 0.4, legend=c("Training Data", "Testing Data"),
       col=c("red", "blue"), lty=1:1, cex=1)

  
######################################
###boosting
###train dataset for complexity curve
###learning curve
######################################  
  
test = train_wine.1
test$Outcome=as.numeric(as.character(test$Outcome))
set.seed(11111)
cv.test = gbm(formula = Outcome ~ .,
distribution = "bernoulli",
data = test,
n.trees = 2000,
shrinkage = .1,
n.minobsinnode = 100, 
cv.folds = 10,
n.cores = 1)
gbm.perf(cv.test)

test = train_diabetes.1
test$Outcome=as.numeric(as.character(test$Outcome))
set.seed(11111)
cv.test = gbm(formula = Outcome ~ .,
              distribution = "bernoulli",
              data = test,
              n.trees = 2000,
              shrinkage = .1,
              n.minobsinnode = 100, 
              cv.folds = 10,
              n.cores = 1)
gbm.perf(cv.test)

##########
##########

result1=NULL
result2=NULL
set.seed(44)
for( rate in range){
  
  i1=get(paste("train_wine.",rate,sep=""))
  i1$Outcome = as.numeric(as.character(i1$Outcome))
  tmp = rate*100
  cv.test = gbm(formula = Outcome ~ .,
                distribution = "bernoulli",
                data = i1,
                n.trees = 2000,
                shrinkage = .1,
                n.minobsinnode = tmp, 
                cv.folds = 10,
                n.cores = 1)
  best1 = gbm.perf(cv.test)
  
  y.pred = predict(cv.test,newdata = i1,n.trees = best1,type="response")
  y.pred.2 = as.numeric(y.pred>0.5)
  i1.train.error = sum(i1$Outcome != y.pred.2)/length(i1$Outcome)
  
  i2=test_wine
  i2$Outcome = as.numeric(as.character(i2$Outcome))
  y.pred = predict(cv.test,newdata = i2,n.trees = best1,type="response")
  y.pred.2 = as.numeric(y.pred>0.5)
  test.wine.error = sum(i2$Outcome != y.pred.2)/length(i2$Outcome)
  
 ###
  
  j1=get(paste("train_diabetes.",rate,sep=""))
  j1$Outcome = as.numeric(as.character(j1$Outcome))
  tmp = rate*100
  cv.test = gbm(formula = Outcome ~ .,
                distribution = "bernoulli",
                data = j1,
                n.trees = 2000,
                shrinkage = .1,
                n.minobsinnode = tmp, 
                cv.folds = 10,
                n.cores = 1)
  best2 = gbm.perf(cv.test)
  
  y.pred = predict(cv.test,newdata = j1,n.trees = best2,type="response")
  y.pred.2 = as.numeric(y.pred>0.5)
  j1.train.error = sum(j1$Outcome != y.pred.2)/length(j1$Outcome)
  
  j2=test_diabetes
  j2$Outcome = as.numeric(as.character(j2$Outcome))
  y.pred = predict(cv.test,newdata = j2,n.trees = best2,type="response")
  y.pred.2 = as.numeric(y.pred>0.5)
  test.diabetes.error = sum(j2$Outcome != y.pred.2)/length(j2$Outcome)

  result1 = rbind(result1,c(i1.train.error,test.wine.error))
  result2 = rbind(result2,c(j1.train.error,test.diabetes.error))
  
}  
  
plot(range,result1[,1],type="b",col="red",ylim=c(0,0.5),main="Red Wine Data",xlab="Proportion of Training Data",ylab="Error Rate")
lines(range,result1[,2],col="blue",type="b")
legend(0.8, 0.4, legend=c("Training Data", "Testing Data"),
       col=c("red", "blue"), lty=1:1, cex=1)

plot(range,result2[,1],type="b",col="red",ylim=c(0,0.5),main="Diabetes Data",xlab="Proportion of Training Data",ylab="Error Rate")
lines(range,result2[,2],col="blue",type="b")
legend(0.8, 0.4, legend=c("Training Data", "Testing Data"),
       col=c("red", "blue"), lty=1:1, cex=1)

######################################
###k-nearest
###train dataset for complexity curve
###learning curve
######################################  
  
library(caret)
library(e1071)
trControl <- trainControl(method  = "cv",
                          number  = 10)
test = train_wine.1
set.seed(4321)
fit <- train(Outcome ~ .,
             method     = "knn",
             tuneGrid   = expand.grid(k = 2:20),
             trControl  = trControl,
             metric     = "Accuracy",
             data       = test)  
plot(fit$results$k,1-fit$results$Accuracy,type="b",main="Red Wine",xlab="K value",ylab="Error Rate")
  
test = train_diabetes.1
set.seed(4321)
fit <- train(Outcome ~ .,
             method     = "knn",
             tuneGrid   = expand.grid(k = 2:20),
             trControl  = trControl,
             metric     = "Accuracy",
             data       = test)  
plot(fit$results$k,1-fit$results$Accuracy,type="b",main="Diabetes",xlab="K value",ylab="Error Rate")

####
####
result1=NULL
result2=NULL
set.seed(54235)
for( rate in range){
  
  i1=get(paste("train_wine.",rate,sep=""))
  fit <- train(Outcome ~ .,
               method     = "knn",
               tuneGrid   = expand.grid(k = 5:30),
               trControl  = trControl,
               metric     = "Accuracy",
               data       = i1)  
  result = fit$result
  i1.error = 1-max(result$Accuracy)
  test.wine.error = sum(test_wine$Outcome != predict(fit,test_wine))/length(test_wine$Outcome)
  
  
  j1=get(paste("train_diabetes.",rate,sep=""))
  fit <- train(Outcome ~ .,
               method     = "knn",
               tuneGrid   = expand.grid(k = 5:30),
               trControl  = trControl,
               metric     = "Accuracy",
               data       = j1)  
  result = fit$result
  j1.error = 1-max(result$Accuracy)
  test.diabetes.error = sum(test_diabetes$Outcome != predict(fit,test_diabetes))/length(test_diabetes$Outcome)

  result1 = rbind(result1,c(i1.error,test.wine.error))
  result2 = rbind(result2,c(j1.error,test.diabetes.error))
}

plot(range,result1[,1],type="b",col="red",ylim=c(0,0.5),main="Red Wine Data",xlab="Proportion of Training Data",ylab="Error Rate")
lines(range,result1[,2],col="blue",type="b")
legend(0.8, 0.4, legend=c("Training Data", "Testing Data"),
       col=c("red", "blue"), lty=1:1, cex=1)

plot(range,result2[,1],type="b",col="red",ylim=c(0,0.5),main="Diabetes Data",xlab="Proportion of Training Data",ylab="Error Rate")
lines(range,result2[,2],col="blue",type="b")
legend(0.8, 0.4, legend=c("Training Data", "Testing Data"),
       col=c("red", "blue"), lty=1:1, cex=1)

######################################
###SVM
###train dataset for complexity curve
###learning curve
###################################### 

test = train_wine.1
cv = tune(svm, Outcome~., data=test,kernel="radial",ranges=list(cost=c(0.5,1,10,100,1000),gamma=c(0.1,0.5,1,2,3,4,5)))
temp = cv$performances
x1=temp[temp$cost==0.5,]
x2=temp[temp$cost==1,]
x3=temp[temp$cost==10,]
x4=temp[temp$cost==100,]
x5=temp[temp$cost==1000,]
plot(x1$gamma,x1$error,type="b",col="red",ylim=c(0,0.3),xlab="gamma",ylab="Error Rate",main="Red Wine")
lines(x2$gamma,x2$error,type="b",col="blue")
lines(x3$gamma,x3$error,type="b",col="purple")
lines(x4$gamma,x4$error,type="b",col="black")  
lines(x5$gamma,x5$error,type="b",col="green")  
legend(1, 0.25, legend=c("cost=0.5", "cost=1","cost=10","cost=100","cost=1000"),
col=c("red", "blue","purple","black","green"), lty=1:1, cex=1)

test = train_diabetes.1
cv = tune(svm, Outcome~., data=test,kernel="radial",ranges=list(cost=c(0.5,1,10,100,1000),gamma=c(0.1,0.5,1,2,3,4,5)))
temp = cv$performances
x1=temp[temp$cost==0.5,]
x2=temp[temp$cost==1,]
x3=temp[temp$cost==10,]
x4=temp[temp$cost==100,]
x5=temp[temp$cost==1000,]
plot(x1$gamma,x1$error,type="b",col="red",ylim=c(0,0.5),xlab="gamma",ylab="Error Rate",main="Diabetes")
lines(x2$gamma,x2$error,type="b",col="blue")
lines(x3$gamma,x3$error,type="b",col="purple")
lines(x4$gamma,x4$error,type="b",col="black")  
lines(x5$gamma,x5$error,type="b",col="green")  
legend(1, 0.1, legend=c("cost=0.5", "cost=1","cost=10","cost=100","cost=1000"),
       col=c("red", "blue","purple","black","green"), lty=1:1, cex=1)  

####
####
result1=NULL
result2=NULL
set.seed(5435)
for( rate in range){
  
  i1=get(paste("train_wine.",rate,sep=""))
  cv = tune(svm, Outcome~., data=i1,kernel="radial",ranges=list(cost=c(0.5,1,10,100,1000),gamma=c(0.1,0.5,1,2,3,4,5)))
  i1.error = cv$best.performance
  test.wine.error = sum(test_wine$Outcome != predict(cv$best.model,newdata=test_wine))/length(test_wine$Outcome)

  j1=get(paste("train_diabetes.",rate,sep=""))
  cv = tune(svm, Outcome~., data=j1,kernel="radial",ranges=list(cost=c(0.5,1,10,100,1000),gamma=c(0.1,0.5,1,2,3,4,5)))
  j1.error = cv$best.performance
  test.diabetes.error = sum(test_diabetes$Outcome != predict(cv$best.model,newdata=test_diabetes))/length(test_diabetes$Outcome)

  result1 = rbind(result1,c(i1.error,test.wine.error))
  result2 = rbind(result2,c(j1.error,test.diabetes.error))

}

plot(range,result1[,1],type="b",col="red",ylim=c(0,0.5),main="Red Wine Data",xlab="Proportion of Training Data",ylab="Error Rate")
lines(range,result1[,2],col="blue",type="b")
legend(0.8, 0.4, legend=c("Training Data", "Testing Data"),
       col=c("red", "blue"), lty=1:1, cex=1)

plot(range,result2[,1],type="b",col="red",ylim=c(0,0.5),main="Diabetes Data",xlab="Proportion of Training Data",ylab="Error Rate")
lines(range,result2[,2],col="blue",type="b")
legend(0.8, 0.4, legend=c("Training Data", "Testing Data"),
       col=c("red", "blue"), lty=1:1, cex=1)


####################
##summary and discussion
##all results for 5 models were compared.
####################

result = read.table("models-result.csv",sep=",")
names(result)=c("training","testing")
result$index=1:nrow(result)
tree.wine = result[result$index<=10,]
tree.diabetes  = result[result$index>10 & result$index<=20,]
ann.wine       = result[result$index>20 & result$index<=30,]
ann.diabetes   = result[result$index>30 & result$index<=40,]
boost.wine       = result[result$index>40 & result$index<=50,]
boost.diabetes   = result[result$index>50 & result$index<=60,]
knn.wine       = result[result$index>60 & result$index<=70,]
knn.diabetes   = result[result$index>70 & result$index<=80,]
svm.wine       = result[result$index>80 & result$index<=90,]
svm.diabetes   = result[result$index>90 & result$index<=100,]


rate=(1:10)/10
plot(rate,tree.wine$training,type="b",cex=1,ylim=c(0.02,0.18),lwd=2,xlab="Proportion of Training Data",
     main="Red Wine",ylab="Error Rate",cex.lab=1.2)
lines(rate,tree.wine$testing,type="b",cex=1,lwd=2,pch = 3)
lines(rate,ann.wine$training,type="b",cex=1,lwd=2,col="red")
lines(rate,ann.wine$testing ,type="b",cex=1,lwd=2,col="red",pch = 3)
lines(rate,boost.wine$training,type="b",cex=1,lwd=2,col="blue")
lines(rate,boost.wine$testing ,type="b",cex=1,lwd=2,col="blue",pch = 3)
lines(rate,knn.wine$training,type="b",cex=1,lwd=2,col="green")
lines(rate,knn.wine$testing ,type="b",cex=1,lwd=2,col="green",pch = 3)
lines(rate,svm.wine$training,type="b",cex=1,lwd=2,col="purple")
lines(rate,svm.wine$testing ,type="b",cex=1,lwd=2,col="purple",pch = 3)
abline(h=0.1, col="blue",lty=3)
legend(0.8, 0.065, legend=c("Training", "Testing","Decision Tree","Neural Network","Boosting","KNN","SVM"),
       col=c("black", "black","black","red","blue","green","purple"), lty=c(0,0,1,1,1,1,1), 
       pch =c("o","+",rep("-",5)), cex=1, lwd=2)


plot(rate,tree.diabetes$training,type="b",cex=1,ylim=c(0,0.5),lwd=2,xlab="Proportion of Training Data",
     main="Diabetes",ylab="Error Rate",cex.lab=1.2)
lines(rate,tree.diabetes$testing,type="b",cex=1,lwd=2,pch = 3)
lines(rate,ann.diabetes$training,type="b",cex=1,lwd=2,col="red")
lines(rate,ann.diabetes$testing ,type="b",cex=1,lwd=2,col="red",pch = 3)
lines(rate,boost.diabetes$training,type="b",cex=1,lwd=2,col="blue")
lines(rate,boost.diabetes$testing ,type="b",cex=1,lwd=2,col="blue",pch = 3)
lines(rate,knn.diabetes$training,type="b",cex=1,lwd=2,col="green")
lines(rate,knn.diabetes$testing ,type="b",cex=1,lwd=2,col="green",pch = 3)
lines(rate,svm.diabetes$training,type="b",cex=1,lwd=2,col="purple")
lines(rate,svm.diabetes$testing ,type="b",cex=1,lwd=2,col="purple",pch = 3)
abline(h=0.2, col="blue",lty=3)
legend(0.8, 0.1, legend=c("Training", "Testing","Decision Tree","Neural Network","Boosting","KNN","SVM"),
       col=c("black", "black","black","red","blue","green","purple"), lty=c(0,0,1,1,1,1,1), 
       pch =c("o","+",rep("-",5)), cex=1, lwd=2)











