
###################################
## R을 이용한 머신러닝: 지도학습 ##
## (곽기영, 도서출판 청람)       ## 
###################################

################
## 제14장 GBM ##
################

###############
## 14.4 사례 ##
###############

## 보스턴 주택가격

library(MASS)
str(Boston)

library(caret)
set.seed(123)
index <- createDataPartition(Boston$medv, p=0.7, list=FALSE)
Boston.train <- Boston[index,]
Boston.test <- Boston[-index,]
dim(Boston.train)
dim(Boston.test)

remotes::install_github("gbm-developers/gbm3")

library(gbm3)
set.seed(123)
gbm.fit1 <- gbm(medv ~ ., distribution="gaussian", data=Boston.train, 
                n.trees=1000, shrinkage=0.01, bag.fraction=1,
                interaction.depth=1, n.minobsinnode=10, cv.folds=5)
class(gbm.fit1)

sqrt(min(gbm.fit1$cv_error))
which.min(gbm.fit1$cv_error)

# [그림 14-19]
windows(width=7.0, height=5.5)
gbm.perf(gbm.fit1, method="cv")

set.seed(123)
gbm.fit2 <- gbm(medv ~ ., data=Boston.train, distribution="gaussian",
                n.trees=500, shrinkage=0.1, bag.fraction=1,
                interaction.depth=3, n.minobsinnode=10, cv.folds=5)

sqrt(min(gbm.fit2$cv_error))
which.min(gbm.fit2$cv_error)

# [그림 14-20]
windows(width=7.0, height=5.5)
gbm.perf(gbm.fit2, method="cv")

hyper.grid <- expand.grid(
  shrinkage=c(0.01, 0.1, 0.3),
  bag.fraction=c(0.5, 0.75, 1), 
  interaction.depth=c(1, 3, 5),
  n.minobsinnode=c(5, 10, 15),
  optimal.trees=0,               
  min.RMSE=0
  )
hyper.grid
dim(hyper.grid)

for (i in 1:nrow(hyper.grid)) {
  set.seed(123)
  gbm.models <- gbm(
    medv ~ ., data=Boston.train, distribution="gaussian",
    n.trees=500,
    shrinkage=hyper.grid$shrinkage[i],
    bag.fraction=hyper.grid$bag.fraction[i],
    interaction.depth=hyper.grid$interaction.depth[i],
    n.minobsinnode=hyper.grid$n.minobsinnode[i],
    cv.folds=5
  )
  hyper.grid$optimal.trees[i] <- which.min(gbm.models$cv_error)
  hyper.grid$min.RMSE[i] <- sqrt(min(gbm.models$cv_error))
}

library(dplyr)
slice_min(hyper.grid, order_by=min.RMSE, n=5)

library(foreach)
library(parallel)
library(doParallel)
cl <- makeCluster(detectCores())
registerDoParallel(cl)
hyper.grid <- expand.grid(
  shrinkage=c(0.01, 0.1, 0.3),
  bag.fraction=c(0.5, 0.75, 1), 
  interaction.depth=c(1, 3, 5),
  n.minobsinnode=c(5, 10, 15)
  )
gbm.rmse <- foreach(
  i=1:nrow(hyper.grid), 
  .packages="gbm3", 
  .combine=rbind
) %dopar% {
  set.seed(123)
  gbm.models <- gbm(
    medv ~ ., data=Boston.train, distribution="gaussian",
    n.trees=500,
    shrinkage=hyper.grid$shrinkage[i],
    bag.fraction=hyper.grid$bag.fraction[i],
    interaction.depth=hyper.grid$interaction.depth[i],
    n.minobsinnode=hyper.grid$n.minobsinnode[i],
    cv.folds=5
  )
  optimal.trees <- which.min(gbm.models$cv_error)
  min.RMSE <- sqrt(min(gbm.models$cv_error))
  data.frame(optimal.trees=optimal.trees, min.RMSE=min.RMSE)
}
gbm.rmse
hyper.grid <- cbind(hyper.grid, gbm.rmse)
slice_min(hyper.grid, order_by=min.RMSE, n=5)
stopImplicitCluster()

set.seed(123)
gbm.fit3 <- gbm(medv ~ ., data=Boston.train, distribution="gaussian",
                n.trees=196, shrinkage=0.1, bag.fraction=0.75,
                interaction.depth=5, n.minobsinnode=10)

sqrt(min(gbm.fit3$train.error))

gbm.pred <- predict(gbm.fit3, newdata=Boston.test, n.trees=gbm.fit3$params$num_trees)
head(gbm.pred)

sqrt(mean((Boston.test$medv - gbm.pred)^2))
postResample(gbm.pred, Boston.test$medv)

# [그림 14-21]
windows(width=7.0, height=5.5)
summary(gbm.fit3, cBars=10, las=1)

predVIP <- function(object, newdata) {
  results <- predict(object, newdata=newdata, n.trees=object$params$num_trees)
  return(results)
}
predVIP(object=gbm.fit3, newdata=Boston.train) |>
  head()

# [그림 14-22]
windows(width=7.0, height=5.5)
library(vip)
set.seed(123)
imp <- vip(gbm.fit3, method="permute", train=Boston.train, target="medv",
           metric="rmse", nsim=5, pred_wrapper=predVIP,
           geom="col", num_features=10, 
           aesthetics=list(color="brown", fill="salmon"))
imp
imp$data

#vip(gbm.fit3, method="model")

# [그림 14-23]
windows(width=7.0, height=5.5)
plot(gbm.fit3, var_index="crim", col="royalblue", lwd=1.5)

predPDP <- function(object, newdata) {
  results <- mean(predict(object, newdata=newdata, n.trees=object$params$num_trees))
  return(results)
}
predPDP(object=gbm.fit3, newdata=Boston.train)

# [그림 14-24]
library(pdp)
p1 <- partial(gbm.fit3, pred.var="rm", pred.fun=predPDP, train=Boston.train) |> 
  ggplot(aes(x=rm, y=yhat)) +
  geom_line(color="royalblue", linewidth=1) +
  geom_rug(aes(x=rm, y=NULL), data=Boston.train, sides="b") +
  scale_y_continuous(name="yhat ($1,000)")
p2 <- partial(gbm.fit3, pred.var="lstat", pred.fun=predPDP, train=Boston.train) |> 
  ggplot(aes(x=lstat, y=yhat)) +
  geom_line(color="royalblue", linewidth=1) +
  geom_rug(aes(x=lstat, y=NULL), data=Boston.train, sides="b") +
  scale_y_continuous(name="yhat ($1,000)")
windows(width=7.0, height=3.5)
library(patchwork)
p1 + p2

## 당뇨병

library(mlbench)
data(PimaIndiansDiabetes2)
str(PimaIndiansDiabetes2)
levels(PimaIndiansDiabetes2$diabetes)

pima <- PimaIndiansDiabetes2
pima$diabetes <- as.numeric(pima$diabetes)
pima$diabetes <- pima$diabetes - 1
head(pima)

library(caret)
set.seed(123)
index <- createDataPartition(pima$diabetes, p=0.7, list=FALSE)
pima.train <- pima[index,]
pima.test <- pima[-index,]
dim(pima.train)
dim(pima.test)

library(gbm3)
set.seed(123)
gbm.fit <- gbm(diabetes ~ ., data=pima.train, distribution="bernoulli",
               n.trees=200, shrinkage=0.1, interaction.depth=1, 
               n.minobsinnode=10, cv.folds=10)

min(gbm.fit$cv_error)
which.min(gbm.fit$cv_error)

# [그림 14-25]
windows(width=7.0, height=5.5)
gbm.perf(gbm.fit, method="cv")

gbm.pred <- predict(gbm.fit, newdata=pima.test, 
                    n.trees=gbm.fit$params$num_trees, type="response")
head(gbm.pred)

pred.label <- factor(ifelse(gbm.pred >= 0.5, 1, 0), levels=c(0, 1), labels=c("neg", "pos"))
head(pred.label)
actual.label <- factor(pima.test$diabetes, levels=c(0, 1), labels=c("neg", "pos"))
head(actual.label)

table(actual.label, pred.label, dnn=c("Actual", "Predicted"))
mean(actual.label==pred.label)
postResample(pred.label, actual.label)

predVIP <- function(object, newdata) {
  results <- predict(object, newdata=newdata, 
                     n.trees=object$params$num_trees, type="response")
  results <- factor(ifelse(results >= 0.5, 1, 0), 
                    levels=c(0, 1), labels=c("neg", "pos"))
  return(results)
}
predVIP(object=gbm.fit, newdata=pima.train) |> 
  head()

# [그림 14-26]
windows(width=7.0, height=5.5)
library(vip)
library(dplyr)
set.seed(123)
imp <- vip(gbm.fit, method="permute", 
           train=mutate(pima.train, 
                        diabetes=factor(diabetes, levels=c(0, 1), labels=c("neg", "pos"))), 
           target="diabetes", metric="accuracy", nsim=5, pred_wrapper=predVIP,
           geom="col", aesthetics=list(color="midnightblue", fill="cornflowerblue"))
imp
imp$data

predPDP <- function(object, newdata) {
  results <- predict(object, newdata=newdata, 
                     n.trees=object$params$num_trees, type="response")
  results <- mean(results)
  return(results)
}
predPDP(object=gbm.fit, newdata=pima.train)

# [그림 14-27]
library(pdp)
p1 <- partial(gbm.fit, pred.var="glucose", pred.fun=predPDP, train=pima.train) |>
  ggplot(aes(x=glucose, y=yhat)) +
  geom_line(color="midnightblue", linewidth=1) +
  ylim(c(0, 1))
p2 <- partial(gbm.fit, pred.var="mass", pred.fun=predPDP, train=pima.train) |> 
  ggplot(aes(x=mass, y=yhat)) +
  geom_line(color="midnightblue", linewidth=1) +
  ylim(c(0, 1))
windows(width=7.0, height=3.5)
library(patchwork)
p1 + p2

library(mlbench)
data(PimaIndiansDiabetes2)
pima <- PimaIndiansDiabetes2
library(caret)
set.seed(123)
index <- createDataPartition(pima$diabetes, p=0.7, list=FALSE)
pima.train <- pima[index,]
pima.test <- pima[-index,]

modelLookup("gbm")

hyper.grid <- expand.grid(
  n.trees=200,
  shrinkage=c(0.01, 0.1, 0.3),
  interaction.depth=c(1, 3, 5),
  n.minobsinnode=c(5, 10, 15)
  )
hyper.grid
dim(hyper.grid)

set.seed(123)
caret.cv <- train(diabetes ~ ., data=pima.train, 
                  method="gbm", distribution="bernoulli", na.action=na.pass,
                  trControl=trainControl(method="cv", number=10),
                  tuneGrid=hyper.grid, verbose=FALSE)
caret.cv

caret.cv$bestTune

caret.pred <- predict(caret.cv, newdata=pima.test, na.action=na.pass, type="raw")
head(caret.pred)

table(pima.test$diabetes, caret.pred, dnn=c("Actual", "Predicted"))
mean(pima.test$diabetes==caret.pred)
postResample(caret.pred, pima.test$diabetes)
