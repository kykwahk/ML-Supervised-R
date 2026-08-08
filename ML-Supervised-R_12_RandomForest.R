
###################################
## R을 이용한 머신러닝: 지도학습 ##
## (곽기영, 도서출판 청람)       ## 
###################################

#########################
## 제12장 랜덤포레스트 ##
#########################

###############
## 12.3 사례 ##
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

library(randomForest)
set.seed(123)
rf.fit <- randomForest(medv ~ ., data=Boston.train, importance=TRUE)
class(rf.fit)
rf.fit

rf.fit$ntree
rf.fit$mse
sqrt(rf.fit$mse)
rf.fit$mse[rf.fit$ntree]
sqrt(rf.fit$mse)[rf.fit$ntree]

which.min(rf.fit$mse)
sqrt(rf.fit$mse)[which.min(rf.fit$mse)]

# [그림 12-2]
windows(width=7.0, height=4.5)
library(ggplot2)
ggplot(data.frame(ntree=1:rf.fit$ntree, rmse=sqrt(rf.fit$mse)), aes(x=ntree, y=rmse)) +
  geom_line(color="royalblue", linewidth=1) +
  geom_vline(xintercept=which.min(sqrt(rf.fit$mse)), color="red", linetype="dashed") +
  labs(x="Number of Trees", y="RMSE")+
  theme_classic()

rf.pred <- predict(rf.fit, newdata=Boston.test, type="response")
head(rf.pred)
sqrt(mean((Boston.test$medv - rf.pred)^2))
postResample(rf.pred, Boston.test$medv)

set.seed(123)
rf.fit <- randomForest(medv ~ ., data=Boston.train, importance=TRUE,
                       xtest=Boston.test[setdiff(names(Boston.test), "medv")],
                       ytest=Boston.test$medv)
rf.fit

sqrt(rf.fit$mse)[rf.fit$ntree]
sqrt(rf.fit$test$mse)[rf.fit$ntree]

# [그림 12-3]
windows(width=7.0, height=5.5)
library(tidyr)
rmse <- data.frame(ntree=1:rf.fit$ntree, 
                   OOB=sqrt(rf.fit$mse), Test=sqrt(rf.fit$test$mse)) |> 
  pivot_longer(cols=c(OOB, Test), names_to="type", values_to="RMSE")
head(rmse)
ggplot(rmse, aes(x=ntree, y=RMSE, color=type)) +
  geom_line(linewidth=1) +
  labs(x="Number of Trees", y="RMSE") +
  theme_classic() +
  theme(legend.position="bottom",
        legend.title=element_blank())

# [그림 12-4]
windows(width=7.0, height=5.5)
varImpPlot(rf.fit, pch=21, color="dimgray", bg="magenta", pt.cex=1.2, main="")
importance(rf.fit)

# [그림 12-5]
windows(width=7.0, height=5.5)
set.seed(123)
rf.tune <-tuneRF(x=Boston.train[setdiff(names(Boston.train), "medv")],
                 y=Boston.train$medv, ntreeTry=500, 
                 mtryStart=2, stepFactor=1.5, improve=0.01)
rf.tune

set.seed(123)
rf.tune <- tuneRF(x=Boston.train[setdiff(names(Boston.train), "medv")],
                  y=Boston.train$medv, ntreeTry=500, doBest=TRUE,
                  mtryStart=2, stepFactor=1.5, improve=0.01, trace=FALSE)
rf.tune

rf.pred <- predict(rf.tune, newdata=Boston.test, type="response")
head(rf.pred)
sqrt(mean((Boston.test$medv - rf.pred)^2))
postResample(rf.pred, Boston.test$medv)

library(ranger)
set.seed(123)
ranger.fit <- ranger(medv ~., data=Boston.train, num.trees=500, 
                     importance="permutation", seed=123)
class(ranger.fit)
ranger.fit

sqrt(ranger.fit$prediction.error)

ranger.pred <- predict(ranger.fit, data=Boston.test, type="response")
names(ranger.pred)
head(ranger.pred$predictions)
sqrt(mean((Boston.test$medv - ranger.pred$predictions)^2))
postResample(ranger.pred$predictions, Boston.test$medv)

ranger.fit$variable.importance

# [그림 12-6]
windows(width=7.0, height=5.5)
ggplot(data.frame(var=reorder(names(ranger.fit$variable.importance), 
                              ranger.fit$variable.importance), 
                  imp=ranger.fit$variable.importance), 
       aes(x=var, y=imp)) +
  geom_col(color="thistle4", fill="thistle") +
  coord_flip() +
  labs(x="Variable", y="Importance")

hyper.grid <- expand.grid(
  mtry=seq(2, 13, by=2),
  min.node.size=c(1, 3, 5, 10),
  replace=c(TRUE, FALSE),
  sample.fraction=c(0.5, 0.632, 0.8),
  RMSE=NA
  )
dim(hyper.grid)
hyper.grid

for (i in 1:nrow(hyper.grid)) {
  ranger.models <- ranger(
    formula=medv ~ ., data=Boston.train,
    num.trees=500,
    mtry=hyper.grid$mtry[i],
    min.node.size=hyper.grid$min.node.size[i],
    replace=hyper.grid$replace[i],
    sample.fraction=hyper.grid$sample.fraction[i],
    num.threads=0,
    seed=123
  )
  hyper.grid$RMSE[i] <- sqrt(ranger.models$prediction.error)
}
library(dplyr)
slice_min(hyper.grid, order_by=RMSE, n=10)

ranger.tune <- ranger(medv ~., data=Boston.train, 
                      num.trees=500, mtry=6, min.node.size=1,
                      replace=FALSE, sample.fraction=0.8, 
                      importance="permutation", seed=123)
ranger.tune
sqrt(ranger.tune$prediction.error)

ranger.pred <- predict(ranger.tune, data=Boston.test, type="response")
head(ranger.pred$predictions)
postResample(ranger.pred$predictions, Boston.test$medv)

# [그림 12-7]
windows(width=7.0, height=5.5)
library(vip)
vip(ranger.tune, num_features=10, method="model", geom="point", 
    aesthetics=list(color="royalblue4", fill="orangered", size=4, pch=21))

# [그림 12-8]
library(pdp)
p1 <- partial(ranger.tune, pred.var="rm", plot=FALSE) |>
  ggplot(aes(x=rm, y=yhat)) +
  geom_line(color="deeppink4", linewidth=1) +
  geom_rug(aes(x=rm, y=NULL), data=Boston.train, sides="b") +
  scale_y_continuous(name="yhat ($1,000)") 
p2 <- partial(ranger.tune, pred.var="lstat", plot=FALSE) |>
  ggplot(aes(x=lstat, y=yhat)) +
  geom_line(color="deeppink4", linewidth=1) +
  geom_rug(aes(x=lstat, y=NULL), data=Boston.train, sides="b") +
  scale_y_continuous(name="yhat ($1,000)") 
windows(width=7.0, height=3.5)
library(patchwork)
p1 + p2

## 신용평가

library(modeldata)
str(credit_data)
levels(credit_data$Status)

library(caret)
set.seed(123)
index <- createDataPartition(credit_data$Status, p=0.7, list=FALSE)
credit.train <- credit_data[index,]
credit.test <- credit_data[-index,]
dim(credit.train)
dim(credit.test)
table(credit.train$Status)
table(credit.test$Status)

library(randomForest)
set.seed(123)
rf.fit <- randomForest(Status ~ ., data=credit.train,
                       na.action=na.roughfix, importance=TRUE)
rf.fit

rf.fit$ntree
rf.fit$err.rate[rf.fit$ntree]

rf.pred <- predict(rf.fit, newdata=credit.test, type="prob")
head(rf.pred)
rf.pred <- predict(rf.fit, newdata=credit.test, type="response")
head(rf.pred)
table(credit.test$Status, rf.pred, dnn=c("Actual", "Predicted"))
mean(credit.test$Status==rf.pred, na.rm=TRUE)
postResample(rf.pred, credit.test$Status)

rf.pred <- predict(rf.fit, newdata=credit.test, predict.all=TRUE)
str(rf.pred)

rf.pred$individual[1,]

table(rf.pred$individual[1, ])
na.omit(rf.pred$aggregate)[1]

apply(rf.pred$individual[1:10, ], 1, table)
na.omit(rf.pred$aggregate)[1:10]

# [그림 12-9]
windows(width=7.0, height=5.5)
varImpPlot(rf.fit, pch=21, color="black", bg="firebrick", pt.cex=1.2, main="")
importance(rf.fit)

# [그림 12-10]
library(pdp)
p1 <- partial(rf.fit, pred.var="Seniority", prob=TRUE) |> 
  ggplot(aes(x=Seniority, y=yhat)) +
  geom_line(color="midnightblue", linewidth=1) +
  ylim(c(0, 1))
p2 <- partial(rf.fit, pred.var="Job", prob=TRUE) |> 
  ggplot(aes(x=Job, y=yhat)) +
  geom_boxplot(color="midnightblue") +
  ylim(c(0, 1))
windows(width=7.0, height=3.5)
library(patchwork)
p1 + p2

modelLookup("ranger")

hyper.grid <- expand.grid(
  mtry=seq(2, 10, by=1),
  min.node.size=seq(2, 10, by=2),
  splitrule="gini")
dim(hyper.grid)
hyper.grid

library(parallel)
library(doParallel)
cl <- makeCluster(detectCores())
registerDoParallel(cl)
set.seed(123)
caret.cv <- train(Status ~ ., data=credit.train, method="ranger",
                  trControl=trainControl(method="cv", number=10),
                  tuneGrid=hyper.grid,
                  na.action=na.roughfix)
stopImplicitCluster()
registerDoSEQ() 

caret.cv
caret.cv$bestTune

caret.pred <- predict(caret.cv, newdata=credit.test)
head(caret.pred)
table(na.omit(credit.test)$Status, caret.pred, dnn=c("Actual", "Predicted"))
mean(na.omit(credit.test)$Status==caret.pred)
postResample(caret.pred, na.omit(credit.test)$Status)

library(partykit)
set.seed(123)
cforest.fit <- cforest(Status ~ ., data=credit.train)
cforest.pred <- predict(cforest.fit, newdata=credit.test, type="response")
head(cforest.pred)
table(credit.test$Status, cforest.pred, dnn=c("Actual", "Predicted"))
mean(credit.test$Status==cforest.pred)
postResample(cforest.pred, credit.test$Status)
