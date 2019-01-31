library(dslabs)
library(dplyr)
library(ggplot2)
data("mnist_27")

mnist_27$train %>% ggplot(aes(x_1, x_2, color = y)) + geom_point()
ggsave("figs/01_plot.png")

fit <- glm(y ~ x_1 + x_2, data=mnist_27$train, family="binomial")
p_hat <- predict(fit, newdata = mnist_27$test) 
y_hat <- factor(ifelse(p_hat > 0.5, 7, 2)) 

library(caret) 
confusionMatrix(data = y_hat, reference = mnist_27$test$y) 

mnist_27$true_p %>% ggplot(aes(x_1, x_2, z=p, fill=p)) +  geom_raster() + 
  scale_fill_gradientn(colors=c("#F8766D","white","#00BFC4")) +  
  stat_contour(breaks=c(0.5),color="black")
ggsave("figs/02_plot.png")

p_hat <- predict(fit, newdata = mnist_27$true_p) 
mnist_27$true_p %>% mutate(p_hat = p_hat) %>% 
  ggplot(aes(x_1, x_2,  z=p_hat, fill=p_hat)) + geom_raster() +   
  scale_fill_gradientn(colors=c("#F8766D","white","#00BFC4")) +   
  stat_contour(breaks=c(0.5),color="black") 
ggsave("figs/03_plot.png")

mnist_27$true_p %>% mutate(p_hat = p_hat) %>% ggplot() +   
  stat_contour(aes(x_1, x_2, z=p_hat), breaks=c(0.5), color="black") +    
  geom_point(mapping = aes(x_1, x_2, color=y), data = mnist_27$test) 
ggsave("figs/04_plot.png")

# Classification trees
# classification by most no of category in the node.
# splitting index will be gini index and entropy than rmse. They look for the purity in nodes

library(rpart)
train_rpart <- train(y~., method="rpart", tuneGrid=data.frame(cp=seq(0.0, 0.1, len=25)),
                     data=mnist_27$train)
plot(train_rpart)
ggsave("figs/05_plot.png")

plot(train_rpart$finalModel, margin = 0.1)
text(train_rpart$finalModel, cex = 0.75)
ggsave("figs/06_plot.png")

confusionMatrix(train_rpart)
# OR
confusionMatrix(predict(train_rpart, mnist_27$test), mnist_27$test$y)$overall["Accuracy"] # 0.82
# variance is high here in tree, intrepretable, decision making esy, visulaisation easy for small ones


###########################################################################################
# MNIST prediction with KNN
# Load the library
library(caret)
library(tidyr)
library(dplyr)
library(purrr)
library(gam)
library(ggplot2)
library(gridExtra)
library(lattice)
library(dslabs)


# Load the data
data("mnist_27") 

# Plot the data for digits 2 and 7
mnist_27$test%>% ggplot(aes(x_1, x_2, color = y)) + geom_point()
ggsave("figs/20_plot.png")

# Create the train set
x <- as.matrix(mnist_27$train[,2:3])
y <- mnist_27$train$y

# Modeling
knn_fit <- knn3(y ~ ., data = mnist_27$train, k = 5)
# Prediction
y_hat_knn <- predict(knn_fit, mnist_27$test, type= "class")
# Confusion Marix
confusionMatrix(data= y_hat_knn, reference = mnist_27$test$y)$overall["Accuracy"] 
# Accuracy 0.815 

# Compare prediction on test to prediction on train set
y_knn <- predict(knn_fit, mnist_27$train, type= "class")
confusionMatrix(data= y_knn, reference = mnist_27$train$y)$overall["Accuracy"] 
# Accuracy 0.8825

# Plot true conditional probabilities for digits 2 and 7 to KNN-5 estimate
plot_cond_prob <- function(p_hat=NULL){
  tmp <- mnist_27$true_p
  if(!is.null(p_hat)){
    tmp <- mutate(tmp, p=p_hat)
  }
  tmp %>% ggplot(aes(x_1, x_2, z=p, fill=p)) +
    geom_raster(show.legend = FALSE) +
    scale_fill_gradientn(colors=c("#F8766D","white","#00BFC4")) +
    stat_contour(breaks=c(0.5),color="black")
}

p1 <- plot_cond_prob() + ggtitle("True conditional probability")
p2 <- plot_cond_prob(predict(knn_fit, mnist_27$true_p)[,2]) +
  ggtitle("kNN-5 estimate")

grid.arrange(p1, p2, nrow=1)
ggsave("figs/21_plot.png")

# train has more saccuracy due to over training
# Model with knn1 on train
knn_fit_1 <- knn3(y ~ ., data = mnist_27$train, k=1)
y_hat_knn_1 <- predict(knn_fit_1, mnist_27$train, type= "class")
confusionMatrix(data= y_hat_knn_1, reference = mnist_27$train$y)$overall["Accuracy"] #Accuracy 0.99625 
# Model with knn1 on test
y_hat_knn_1 <- predict(knn_fit_1, mnist_27$test, type= "class")
confusionMatrix(data= y_hat_knn_1, reference = mnist_27$test$y)$overall["Accuracy"] # Accuracy 0.735
# Plot knn1 prediction for conditional probabilities of 2and 7 digits on train
p1 <- mnist_27$true_p %>% 
  mutate(knn = predict(knn_fit_1, newdata = .)[,2]) %>%
  ggplot() +
  geom_point(data = mnist_27$train, aes(x_1, x_2, color= y),
             pch=21, show.legend = FALSE) +
  scale_fill_gradientn(colors=c("#F8766D","white","#00BFC4")) +
  stat_contour(aes(x_1, x_2, z = knn), breaks=c(0.5), color="black") +
  ggtitle("Train set")
# Plot knn1 prediction for conditional probabilities of 2 and 7 digits on test 
p2 <- mnist_27$true_p %>% 
  mutate(knn = predict(knn_fit_1, newdata = .)[,2]) %>%
  ggplot() +
  geom_point(data = mnist_27$test, aes(x_1, x_2, color= y), 
             pch=21, show.legend = FALSE) +
  scale_fill_gradientn(colors=c("#F8766D","white","#00BFC4")) +
  stat_contour(aes(x_1, x_2, z = knn), breaks=c(0.5), color="black") +
  ggtitle("Test set")
grid.arrange(p1, p2, nrow=1)
ggsave("figs/22_plot.png")
# Compare the accuracy for all odd k between 3 and 251 on train and test with plot
ks <- seq(3, 251, 2)

accuracy <- map_df(ks, function(k){
  fit <- knn3(y ~. , data = mnist_27$train, k=k)
  y_hat <- predict(fit, mnist_27$train, type="class")
  train_error <- confusionMatrix(data= y_hat, reference = mnist_27$train$y)$overall["Accuracy"]
  
  y_hat <- predict(fit, mnist_27$test, type="class")
  test_error <- confusionMatrix(data= y_hat, reference = mnist_27$test$y)$overall["Accuracy"]
  list(train= train_error, test= test_error)
})

accuracy %>% mutate(k = ks) %>%
  gather(set, accuracy, -k) %>%
  mutate(set = factor(set, levels = c("train", "test"))) %>%
  ggplot(aes(k, accuracy, color = set)) + 
  geom_line() +
  geom_point() 

ggsave("figs/23_plot.png")

# Try for a much larger k of 401
knn_fit_401 <- knn3(y ~ ., data = mnist_27$train, k = 401) 
y_hat_knn_401 <- predict(knn_fit_401, mnist_27$test, type = "class") 
confusionMatrix(data=y_hat_knn_401, reference=mnist_27$test$y)$overall["Accuracy"] 

# Compare the predictions for logistic regression and knn with k=401 with plot
fit <- glm(y ~ x_1 + x_2, data=mnist_27$train, family="binomial")
p1 <- plot_cond_prob(predict(fit, mnist_27$true_p)) +
  ggtitle("Logistic regression")
p2 <- plot_cond_prob(predict(knn_fit_401, mnist_27$true_p)[,2]) +
  ggtitle("kNN-401")

grid.arrange(p1, p2, nrow=1)
ggsave("figs/24_plot.png")

# Prediction on train data using k value giving maximum accuracy on test data, k=41
knn_fit_41 <- knn3(y ~ ., data = mnist_27$train, k=41)
y_hat_knn_41 <- predict(knn_fit_41, mnist_27$train, type= "class")
confusionMatrix(data= y_hat_knn_41, reference = mnist_27$train$y)$overall["Accuracy"] #Accuracy 0.8475 
# Prediction on test data using k value giving maximum accuracy on test data, k=41
y_hat_knn_41 <- predict(knn_fit_41, mnist_27$test, type= "class")
confusionMatrix(data= y_hat_knn_41, reference = mnist_27$test$y)$overall["Accuracy"] # Accuracy 0.86

# Plot true conditional probabilty and knn-41 estimate when prediction done on the 
# true probabilities
# Create a function called plot_cond_prob

plot_cond_prob <- function(p_hat=NULL){
  tmp <- mnist_27$true_p 
  if(!is.null(p_hat)){ 
    tmp <- mutate(tmp, p=p_hat)
  } 
  tmp %>% ggplot(aes(x_1, x_2, z=p, fill=p)) + 
    geom_raster(show.legend = FALSE) + 
    scale_fill_gradientn(colors=c("#F8766D","white","#00BFC4")) + 
    stat_contour(breaks=c(0.5),color="black")
}

p1 <- plot_cond_prob() + ggtitle("True conditional probability") 

knn_fit <- knn3(y ~ ., data = mnist_27$train, k = 41) 
p2 <- plot_cond_prob(predict(knn_fit, newdata = mnist_27$true_p)[,2]) +   
  ggtitle("kNN-41 estimate") 

grid.arrange(p1, p2, nrow=1)
ggsave("figs/25_plot.png")
###################
library(caret)
library(dplyr)
library(ggplot2)
library(dslabs)
# Caret Package
data("mnist_27")

# tuning parameter
getModelInfo("knn")
modelLookup("knn") # shows what all parameter needs to be optimised
ggplot(train_knn_cv, highlight=TRUE) # plot the parameter values

predict.train()

train_knn <- train(y ~ ., method = "knn", data = mnist_27$train)
ggplot(train_knn, highlight = TRUE)
ggsave("figs/26_plot.png")
train_knn$bestTune # best value for accuracy
train_knn$finalModel # gives best model

# Cross validation with tuneGrid
train_knn_cv <- train(y~., method="knn", data= mnist_27$train, 
                      tuneGrid=data.frame(k=seq(9,71,2)))
ggplot(train_knn_cv, highlight=TRUE) # plot the parameter values
ggsave("figs/27_plot.png")
train_knn_cv$bestTune # best value for accuracy k=41
train_knn_cv$finalModel # best on train set
# 41-nearest neighbor model
# Training set outcome distribution:
#  
# 2   7 
#379 421
confusionMatrix(predict(train_knn_cv, mnist_27$test,type="raw"), mnist_27$test$y)$overall["Accuracy"] 
# 0.86
# 
# crossvalidation with trainControl and tuneGrid
control <- trainControl(method = "cv", number = 10, p = .9) 
train_knn_cv <- train(y ~ ., method = "knn",                     
                      data = mnist_27$train,                    
                      tuneGrid = data.frame(k = seq(9, 71, 2)),                    
                      trControl = control) 
ggplot(train_knn_cv, highlight = TRUE)
ggsave("figs/28_plot.png")

train_knn_cv$bestTune # best value for accuracy k=41
train_knn_cv$finalModel

# Plot the accuracy with
train_knn_cv$results %>%    
  ggplot(aes(x = k, y = Accuracy)) +   geom_line() +   geom_point() +   
  geom_errorbar(aes(x = k,
                    ymin = Accuracy - AccuracySD,                      
                    ymax = Accuracy + AccuracySD))
ggsave("figs/29_plot.png")

Pred_knn_cv <- predict(train_knn_cv, mnist_27$test,type="raw")
cm <- confusionMatrix(Pred_knn_cv, mnist_27$test$y)
cm

cm$overall["Accuracy"] 

########
# Get a model to predict a smoother conditional probabilities
modelLookup("gamLoess") 
grid <- expand.grid(span = seq(0.15, 0.65, len = 10), degree = 1)

train_loess <- train(y ~ .,                     
                     method = "gamLoess",                     
                     tuneGrid=grid,                    
                     data = mnist_27$train) 


ggplot(train_loess, highlight = TRUE)
ggsave("figs/30_plot.png")
confusionMatrix(data =predict(train_loess, mnist_27$test),                  
                reference = mnist_27$test$y)$overall["Accuracy"]# 0.85

# Create a function called plot_cond_prob

plot_cond_prob <- function(p_hat=NULL){
  tmp <- mnist_27$true_p 
  if(!is.null(p_hat)){ 
    tmp <- mutate(tmp, p=p_hat)
  } 
  tmp %>% ggplot(aes(x_1, x_2, z=p, fill=p)) + 
    geom_raster(show.legend = FALSE) + 
    scale_fill_gradientn(colors=c("#F8766D","white","#00BFC4")) + 
    stat_contour(breaks=c(0.5),color="black")
}
plot_cond_prob(predict(train_loess, mnist_27$true_p, type = "prob")[,2])
ggsave("figs/31_plot.png")
####

###############################################################################################

library(Rborist)
library(dslabs)

# Load the data
mnist <- read_mnist()
names(mnist)
dim(mnist$train$images) 
class(mnist$train$labels) 
table(mnist$train$labels) 

# Subsetting the data for train
set.seed(123) 
index <- sample(nrow(mnist$train$images), 10000) 
x <- mnist$train$images[index,] 
y <- factor(mnist$train$labels[index]) 

# Subsetting the data for test
index <- sample(nrow(mnist$train$images), 1000) 
x_test <- mnist$train$images[index,] 
y_test <- factor(mnist$train$labels[index])

# Explore the data
library(matrixStats)
sds <- colSds(x) 
qplot(sds, bins = 256, color = I("black"))
ggsave("figs/07_plot.png")

library(caret) 
# Identification of near zero variance predictors with nearZeroVar function in caret package
nzv <- nearZeroVar(x) # removes columns with near zero variance
image(matrix(1:784 %in% nzv, 28, 28)) # shows removed or near zero variance columns
ggsave("figs/08_plot.png")

# Get the col index for analysis
col_index <- setdiff(1:ncol(x), nzv) 
length(col_index) 

# Get the columns named in train and test for analysis
colnames(x) <- 1:ncol(mnist$train$images) 
colnames(x_test) <- colnames(mnist$train$images)

# Tuning the parameters with crossvalidation
control <- trainControl(method="cv", number = 5, p = 0.8) 
grid <- expand.grid(minNode = c(1) , predFixed = c(10, 15, 35))
train_rf <-  train(x[ , col_index],                     
                   y,                     
                   method = "Rborist",                     
                   nTree = 50,                    
                   trControl = control,                    
                   tuneGrid = grid,                    
                   nSamp = 5000) 
ggplot(train_rf) 
ggsave("figs/09_plot.png")
# Get the best tuning parameter
train_rf$bestTune 

# Modeling 
fit_rf <- Rborist(x[ ,col_index], y,                    
                  nTree = 1000,                   
                  minNode = train_rf$bestTune$minNode,                   
                  predFixed = train_rf$bestTune$predFixed)
# Prediction
pred <- predict(fit_rf, x_test[ ,col_index])

# Getting predicted values in the same levels of y
y_hat_rf <- factor(levels(y)[pred$yPred]) 

# Confusion matrix
cm <- confusionMatrix(y_hat_rf, y_test) 
cm
cm$overall["Accuracy"] 

###################################################################################

library(randomForest) 
rf <- randomForest(x, y,  ntree = 50)
names(rf)
imp <- importance(rf)
image(matrix(imp, 28, 28))
ggsave("figs/10_plot.png")
##########################################################################
# KNN



n <- 1000 
b <- 2 
index <- sample(nrow(x), n) 
control <- trainControl(method = "cv", number = b, p = .9) 
train_knn <- train(x[index ,col_index], y[index],
                   method = "knn",                     
                   tuneGrid = data.frame(k = c(3,5,7)),                    
                   trControl = control)
ggplot(train_knn)
ggsave("figs/37plot.png")
fit_knn<- knn3(x[ ,col_index], y,  k = 5)
y_hat_knn <- predict(fit_knn,                          
                     x_test[, col_index],                          
                     type="class") 
cm1 <- confusionMatrix(y_hat_knn, factor(y_test)) 
cm1$overall["Accuracy"] 

###################################################################################

#Ensemble

p_rf <- predict(fit_rf, x_test[,col_index])$census   
p_rf<- p_rf / rowSums(p_rf)   
p_knn  <- predict(fit_knn, x_test[,col_index], data=mnist$test) 
p <- (p_rf + p_knn)/2 
y_pred <- factor(apply(p, 1, which.max)-1) 
cm3 <-confusionMatrix(y_pred, y_test)
cm3
cm3$overall["Accuracy"] 

###########################################################################################

# QDA
# qda is a GM assuming conditional probabilities of predictors to be multivariate normal
# so here in mnist with 2 n 7 dataset there will be 2 avg, 2 sd, a correlation for 2s and 7s
data("mnist_27")

params <- mnist_27$train %>% group_by(y) %>% 
  summarize(avg_1 = mean(x_1), avg_2= mean(x_2), sd_1= sd(x_1), sd_2= sd(x_2), 
            r= cor(x_1, x_2))
params

mnist_27$train %>% mutate(y = factor(y)) %>%   
  ggplot(aes(x_1, x_2, fill = y, color=y)) +   
  geom_point(show.legend = FALSE) +    
  stat_ellipse(type="norm", lwd = 1.5)
ggsave("figs/11_plot.png")

# fitting the model
library(caret)
train_qda <- train(y ~. , method= "qda", data= mnist_27$train)
y_hat <- predict(train_qda, mnist_27$test)
confusionMatrix(data= y_hat, reference= mnist_27$test$y)$overall["Accuracy"] #0.82

# K * (2p + p * (p-1) / 2) number of parameters will have to be estimated for qda.
# quadratic function should hold in qda.
# multivariate distribution of normality should hold to do qda

# when predictors are large we assume the correlations are same for all the classes.
# so code will be ....of LDA as the same sd, correlation conditiond are forced in the data.
# so flexibility os less
params_1 <- params %>% mutate(sd_1=mean(sd_1), sd_2=mean(sd_2), r=mean(r))
params_1
train_lda <- train(y ~. , method= "lda", data= mnist_27$train)
y_hat <- predict(train_lda, mnist_27$test)
confusionMatrix(data= y_hat, reference= mnist_27$test$y)$overall["Accuracy"] # 0.75

###############################
# case study with three classes

mnist <- read_mnist()
set.seed(3456)
index_127 <- sample(which(mnist$train$labels %in% c(1,2,7)), 2000)

y <- mnist$train$labels[index_127]
x <- mnist$train$images[index_127]
index_train <- createDataPartition(y, p=0.8, list=FALSE)
## get the quardrants
row_column <- expand.grid(row=1:28, col=1:28)
# temporary object to help figure out the quardrants
upper_left_ind <- which(row_column$col <= 14 & row_column$row <= 14)

lower_right_ind <- which(row_column$col > 14 & row_column$row > 14)

x <- x > 200

# binarize the values. Above 200 is ink, below is no ink
x <- cbind(rowSums(x[upper_left_ind])/rowSums(x),# proportion of pixes in upper rt quardrant
           rowSums(x[, lower_right_ind])/rowSums(x)) 

train_set <- data.frame(y= factor(y[index_train]),
                        x_1 = x[index_train, 1],
                        x_2 = x[index_train,2])
test_set <- data.frame(y =  factor(y[index_train]),
                       x_1 = x[index_train,1],
                       x_2 = x[index_train, 2])
# modeling
train_qda <- train(y~., method="qda", data= train_set)

predict(train_qda, test_set, type= "prob") %>% head() # we get prob for each class, and decision will 
# be chose the class of higher prob.
# or for class use the code:
predict(train_qda, test_set)

confusionMatrix(predict(train_qda, test_set), test_set$y) # acc=0.72
########## for lda

train_lda <- train(y~., method ="lda", data=train_set)
confusionMatrix(predict(train_lda, test_set), test_set$y)$overall["Accuracy"] # acc= 0.664
# bundaries need to be linear so accuracy is bad

## KNN
train_knn <- train(y~., method = "knn", tuneGrid = data.frame(k=seq(15, 51, 2)),
                   data =  train_set)

confusionMatrix(predict(train_knn, test_set), test_set$y)$overall["Accuracy"] 0.769
# Accuracy is not good on glm, esp lda, is that due to lack of fit. plot show that the datapoints are 
# not normally distributed.  esp datapoints of 1 are not normally distributed.


############################################################################################
library(dslabs)
library(dplyr)
library(ggplot2)
if(!exists("mnist")) mnist <- read_mnist()

col_means <- colMeans(mnist$test$images) 
pca <- prcomp(mnist$train$images)

plot(pca$sdev)
ggsave("figs/12_plot.png")

summary(pca)$importance[,1:5] %>% knitr::kable()

data.frame(PC1 = pca$x[,1], PC2 = pca$x[,2],            
           label=factor(mnist$train$label)) %>%   
  sample_n(2000) %>%    
  ggplot(aes(PC1, PC2, fill=label))+   
  geom_point(cex=3, pch=21)
ggsave("figs/13_plot.png")

tmp <- lapply( c(1:4,781:784), function(i){    
  expand.grid(Row=1:28, Column=1:28) %>%       
    mutate(id=i, label=paste0("PC",i),               
           value = pca$rotation[,i]) }) 
tmp <- Reduce(rbind, tmp) 

library(RColorBrewer)
tmp %>% filter(id<5) %>%   
  ggplot(aes(Row, Column, fill=value)) +   
  geom_raster() +   
  scale_y_reverse() +   
  scale_fill_gradientn(colors = brewer.pal(9, "RdBu")) +   
  facet_wrap(~label, nrow = 1)

tmp %>% filter(id>5) %>%   
  ggplot(aes(Row, Column, fill=value)) +   
  geom_raster() +   
  scale_y_reverse() +   
  scale_fill_gradientn(colors = brewer.pal(9, "RdBu")) +   
  facet_wrap(~label, nrow = 1)

library(caret) 
K <- 36 
x_train <- pca$x[,1:K] 
y <- factor(mnist$train$labels) 

x_test <- sweep(mnist$test$images, 2, col_means) %*% pca$rotation 
x_test <- x_test[,1:K]

library(kernlab)
svm.linear <- ksvm(y~x_train, scale =FALSE, kernel="vanilladot")
predict <- predict(svm.linear, x_test)
confusionMatrix(predict, factor(mnist$test$labels))

plot(predict)
ggsave("figs/14_plot.jpeg")
