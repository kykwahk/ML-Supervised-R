
###################################
## R을 이용한 머신러닝: 지도학습 ##
## (곽기영, 도서출판 청람)       ## 
###################################

##########################
## 제9장 서포트벡터머신 ##
##########################

##############
## 9.5 사례 ##
##############

## 당뇨병

library(mlbench)
data(PimaIndiansDiabetes)
str(PimaIndiansDiabetes)
levels(PimaIndiansDiabetes$diabetes)

library(caret)
set.seed(123)
index <- createDataPartition(y=PimaIndiansDiabetes$diabetes, p=0.7, list=FALSE)
diabetes.train <- PimaIndiansDiabetes[index,]
diabetes.test <- PimaIndiansDiabetes[-index,]
dim(diabetes.train)
dim(diabetes.test)

library(e1071)
svm.fit <- svm(diabetes ~ ., data=diabetes.train, scale=TRUE, kernel="radial")
class(svm.fit)

summary(svm.fit)

svm.pred <- predict(svm.fit, newdata=diabetes.test)
head(svm.pred)
table(diabetes.test$diabetes, svm.pred, dnn=c("Actual", "Predicted"))
mean(diabetes.test$diabetes==svm.pred)

prop.table(table(PimaIndiansDiabetes$diabetes))
svm.fit2 <- svm(diabetes ~ ., data=diabetes.train, scale=TRUE, kernel="radial",
                class.weights=c("neg"=1, "pos"=2))

svm.pred2 <- predict(svm.fit2, newdata=diabetes.test)
table(diabetes.test$diabetes, svm.pred2, dnn=c("Actual", "Predicted"))
mean(diabetes.test$diabetes==svm.pred2)

set.seed(123)
svm.fit <- svm(diabetes ~ ., data=diabetes.train, scale=TRUE, kernel="radial",
               probability=TRUE)

svm.pred <- predict(svm.fit, newdata=diabetes.test, probability=TRUE)
str(svm.pred)
attr(svm.pred, "probabilities")[1:6,]

set.seed(123)
svm.tune <- tune.svm(diabetes ~ ., data=diabetes.train, scale=TRUE, kernel="radial",
                     gamma=2^(-5:5), cost=2^(-5:5))
summary(svm.tune)
svm.tune$best.parameters

svm.fit <- svm(diabetes ~ ., data=diabetes.train, scale=TRUE, kernel="radial",
               gamma=svm.tune$best.parameters$gamma, 
               cost=svm.tune$best.parameters$cost)

svm.pred <- predict(svm.fit, newdata=diabetes.test)
table(diabetes.test$diabetes, svm.pred, dnn=c("Actual", "Predicted"))
mean(diabetes.test$diabetes==svm.pred)

svm.pred <- predict(svm.tune$best.model, newdata=diabetes.test)
mean(diabetes.test$diabetes==svm.pred)

modelLookup("svmRadialSigma")

set.seed(123)
caret.cv <- train(diabetes ~ ., data=diabetes.train, method="svmRadialSigma", 
                  preProcess=c("center", "scale"), 
                  trControl=trainControl(method="cv", number=10),
                  tuneLength=5)
caret.cv

caret.cv$bestTune
getTrainPerf(caret.cv)

# [그림 9-14]
windows(width=7.0, height=5.5)
ggplot(caret.cv) +
  theme_bw()

set.seed(123)
svm.fit <- svm(diabetes ~ ., data=diabetes.train, 
               scale=TRUE, kernel="radial", probability=TRUE,
               gamma=caret.cv$bestTune$sigma, cost=caret.cv$bestTune$C)
svm.pred <- predict(svm.fit, newdata=diabetes.test)
table(diabetes.test$diabetes, svm.pred, dnn=c("Actual", "Predicted"))
mean(diabetes.test$diabetes==svm.pred)

caret.pred <- predict(caret.cv, newdata=diabetes.test)
mean(diabetes.test$diabetes==caret.pred)

predVIP <- function(object, newdata) {
  results <- predict(object, newdata=newdata)
  return(results)
}
predVIP(object=svm.fit, newdata=diabetes.train) |> 
  head()

# [그림 9-15]
windows(width=7.0, height=5.5)
library(vip)
set.seed(123)
imp <- vip(svm.fit, method="permute", train=diabetes.train, target="diabetes", 
           metric="accuracy", nsim=5, pred_wrapper=predVIP,
           geom="boxplot", aesthetics=list(color="orange", fill="orange"))
imp
imp$data

predPDP <- function(object, newdata) {
  pred <- predict(object, newdata=newdata, probability=TRUE)
  results <- mean(attr(pred, "probabilities")[,"pos"])
  return(results)
}
predPDP(object=svm.fit, newdata=diabetes.train)

# [그림 9-16]
library(pdp)
p1 <- partial(svm.fit, pred.var="glucose", pred.fun=predPDP, train=diabetes.train) |>
  ggplot(aes(x=glucose, y=yhat)) +
  geom_line(color="midnightblue", linewidth=1) +
  ylim(c(0, 1))
p2 <- partial(svm.fit, pred.var="insulin", pred.fun=predPDP, train=diabetes.train) |> 
  ggplot(aes(x=insulin, y=yhat)) +
  geom_line(color="midnightblue", linewidth=1) +
  ylim(c(0, 1))
windows(width=7.0, height=3.5)
library(patchwork)
p1 + p2

## 보스턴 주택가격

library(MASS)
str(Boston)

library(caret)
set.seed(123)
index <- createDataPartition(y=Boston$medv, p=0.7, list=FALSE)
Boston.train <- Boston[index,]
Boston.test <- Boston[-index,]
dim(Boston.train)
dim(Boston.test)

library(e1071)
svm.fit <- svm(medv ~ ., data=Boston.train, scale=TRUE, kernel="radial")
summary(svm.fit)

svm.pred <- predict(svm.fit, newdata=Boston.test)
head(svm.pred)
postResample(pred=svm.pred, obs=Boston.test$medv)

set.seed(123)
svm.tune <- tune.svm(medv ~ ., data=Boston.train, scale=TRUE, kernel="radial",
                     gamma=10^(-3:3), cost=2^(-5:5))
summary(svm.tune)
svm.tune$best.parameters

set.seed(123)
svm.fit <- svm(medv ~ ., data=Boston.train, scale=TRUE, kernel="radial",
               gamma=svm.tune$best.parameters$gamma, 
               cost=svm.tune$best.parameters$cost)
svm.pred <- predict(svm.fit, newdata=Boston.test)
postResample(pred=svm.pred, obs=Boston.test$medv)

set.seed(123)
caret.cv <- train(medv ~ ., data=Boston.train, method="svmRadialSigma",
                  preProcess=c("center", "scale"), 
                  trControl=trainControl(method="cv", number=10),
                  tuneLength=5)

caret.cv$bestTune
getTrainPerf(caret.cv)

caret.pred <- predict(caret.cv, newdata=Boston.test)
postResample(pred=caret.pred, obs=Boston.test$medv)

predVIP <- function(object, newdata) {
  results <- predict(object, newdata=newdata)
  return(results)
}
predVIP(object=svm.fit, newdata=Boston.train) |> 
  head()

# [그림 9-17]
windows(width=7.0, height=5.5)
library(vip)
set.seed(123)
imp <- vip(svm.fit, method="permute", train=Boston.train, target="medv", 
           metric="rmse", nsim=5, pred_wrapper=predVIP,
           geom="violin", aesthetics=list(color="olivedrab", fill="olivedrab"))
imp
imp$data

predPDP <- function(object, newdata) {
  results <- mean(predict(object, newdata=newdata))
  return(results)
}
predPDP(object=svm.fit, newdata=Boston.train)

# [그림 9-18]
library(pdp)
p1 <- partial(svm.fit, pred.var="rm", pred.fun=predPDP, train=Boston.train) |> 
  ggplot(aes(x=rm, y=yhat)) +
  geom_line(color="saddlebrown", linewidth=1) +
  scale_y_continuous(limits=c(0, 40), name="yhat ($1,000)")
p2 <- partial(svm.fit, pred.var="crim", pred.fun=predPDP, train=Boston.train) |> 
  ggplot(aes(x=crim, y=yhat)) +
  geom_line(color="saddlebrown", linewidth=1) +
  scale_y_continuous(limits=c(0, 40), name="yhat ($1,000)")
windows(width=7.0, height=3.5)
library(patchwork)
p1 + p2
