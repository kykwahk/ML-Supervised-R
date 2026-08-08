
###################################
## R을 이용한 머신러닝: 지도학습 ##
## (곽기영, 도서출판 청람)       ## 
###################################

#################
## 제11장 배깅 ##
#################

###############
## 11.4 사례 ##
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

library(ipred)
set.seed(123)
bag.fit <- bagging(medv ~ ., data=Boston.train, coob=TRUE,
                   nbagg=100, control=list(minsplit=2, cp=0))
class(bag.fit)
bag.fit

bag.fit$err

bag.pred <- predict(bag.fit, newdata=Boston.test)
head(bag.pred)
sqrt(mean((Boston.test$medv - bag.pred)^2))
postResample(bag.pred, Boston.test$medv)

ntree <- c(10:200)
rmse <- vector(mode="numeric", length=length(ntree))
for (i in seq_along(ntree)) {
  set.seed(123)
  bag.models <- bagging(medv ~ ., data=Boston.train, coob=TRUE, 
                        nbagg=ntree[i], control=list(minsplit=2, cp=0))
  rmse[i] <- bag.models$err
}
rmse
min(rmse)
which.min(rmse) + 9

# [그림 11-2]
library(ggplot2)
windows(width=7.0, height=4.5)
ggplot(data.frame(ntree=ntree, rmse=rmse), aes(x=ntree, y=rmse)) +
  geom_line(color="royalblue", linewidth=1) +
  geom_vline(xintercept=which.min(rmse)+9, color="red", linetype="dashed") +
  labs(x="Number of Trees", y="RMSE")+
  theme_classic()

set.seed(123)
caret.cv <- train(medv ~ ., data=Boston.train, method="treebag",
                  trControl=trainControl(method="cv", number=10),
                  nbagg=100, control=list(minsplit=2, cp=0))

caret.cv

varImp(caret.cv)
windows(width=7.0, height=5.5)
plot(varImp(caret.cv), 10)

# [그림 11-3]
library(vip)
windows(width=7.0, height=5.5)
vip(caret.cv, num_features=10, method="model", geom="point",
    aesthetics=list(color="brown", fill="coral", size=4, pch=21))

# [그림 11-4]
library(pdp)
p1 <- partial(caret.cv, pred.var="crim", plot=FALSE) |>
  ggplot(aes(x=crim, y=yhat)) +
  geom_line(color="turquoise3", linewidth=1) +
  geom_rug(aes(x=crim, y=NULL), data=Boston.train, sides="b") +
  scale_y_continuous(name="yhat ($1,000)") 
p2 <- partial(caret.cv, pred.var="rm", plot=FALSE) |>
  ggplot(aes(x=rm, y=yhat)) +
  geom_line(color="turquoise3", linewidth=1) +
  geom_rug(aes(x=rm, y=NULL), data=Boston.train, sides="b") +
  scale_y_continuous(name="yhat ($1,000)") 
windows(width=7.0, height=3.5)
library(patchwork)
p1 + p2

library(foreach)
library(parallel)
library(doParallel)
cl <- makeCluster(detectCores())
registerDoParallel(cl)

detectCores(logical=FALSE)
detectCores(logical=TRUE)

trees.pred <- foreach(
  icount(1000), 
  .packages="rpart", 
  .combine=cbind
  ) %dopar% {
    index <- sample(nrow(Boston.train), replace=TRUE)
    Boston.train.boot <- Boston.train[index,]
    tree <- rpart(
      medv ~ ., 
      control=list(minsplit=2, cp=0),
      data=Boston.train.boot
      )
    predict(tree, newdata=Boston.test)
  }
str(trees.pred)
trees.pred[1:5, 1:10]

sqrt(mean((Boston.test$medv - trees.pred)^2))

trees.rmse <- foreach(
  i=10:200, 
  .packages="ipred", 
  .combine=c
) %dopar% {
  set.seed(123)
  bag.models <- bagging(medv ~ ., data=Boston.train, coob=TRUE, 
                        nbagg=i, control=list(minsplit=2, cp=0))
  bag.models$err
}
trees.rmse
min(trees.rmse)
which.min(trees.rmse) + 9

stopImplicitCluster()
registerDoSEQ() 

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

library(ipred)
set.seed(123)
bag.fit <- bagging(Status ~ ., data=credit.train, coob=TRUE, 
                   nbagg=100, control=list(minsplit=2, cp=0))
bag.fit
bag.fit$err

bag.pred <- predict(bag.fit, newdata=credit.test, type="prob")
head(bag.pred)
bag.pred <- predict(bag.fit, newdata=credit.test, type="class")
head(bag.pred)
table(credit.test$Status, bag.pred, dnn=c("Actual", "Predicted"))
mean(credit.test$Status==bag.pred)
postResample(bag.pred, credit.test$Status)

set.seed(123)
caret.cv <- train(Status ~ ., data=credit.train, method="treebag",
                  trControl=trainControl(method="cv", number=10),
                  nbagg=100, control=list(minsplit=2, cp=0), na.action=na.pass)

library(naniar)
miss_var_summary(credit_data)

caret.cv

# [그림 11-5]
windows(width=7.0, height=5.5)
library(vip)
vip(caret.cv, num_features=10, method="model", geom="point",
    aesthetics=list(color="slateblue4", fill="palegreen4", size=4, pch=21))

# [그림 11-6]
library(pdp)
p1 <- partial(caret.cv, pred.var="Seniority", prob=TRUE) |> 
  ggplot(aes(x=Seniority, y=yhat)) +
  geom_line(color="palevioletred4", linewidth=1) +
  ylim(c(0, 1))
p2 <- partial(caret.cv, pred.var="Job", prob=TRUE) |> 
  ggplot(aes(x=Job, y=yhat)) +
  geom_boxplot(color="palevioletred4") +
  ylim(c(0, 1))
windows(width=7.0, height=3.5)
library(patchwork)
p1 + p2
