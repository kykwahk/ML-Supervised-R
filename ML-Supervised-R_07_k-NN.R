
###################################
## R을 이용한 머신러닝: 지도학습 ##
## (곽기영, 도서출판 청람)       ## 
###################################

################
## 제7장 k-NN ##
################

##############
## 7.5 사례 ##
##############

## 당뇨병

library(mlbench)
data(PimaIndiansDiabetes2)
str(PimaIndiansDiabetes2)
levels(PimaIndiansDiabetes2$diabetes)

# [그림 7-5]
library(ggplot2)
p1 <- ggplot(data=PimaIndiansDiabetes2, aes(x=glucose, y=insulin, fill=diabetes)) +
  geom_point(pch=21, color="black", size=1.5) +
  scale_fill_discrete(name="Diabetes") +
  theme_bw() 
p2 <- ggplot(data=PimaIndiansDiabetes2, aes(x=glucose, y=triceps, fill=diabetes)) +
  geom_point(pch=21, color="black", size=1.5) +
  scale_fill_discrete(name="Diabetes") +
  theme_bw() 
p3 <- ggplot(data=PimaIndiansDiabetes2, aes(x=insulin, y=age, fill=diabetes)) +
  geom_point(pch=21, color="black", size=1.5) +
  scale_fill_discrete(name="Diabetes") +
  theme_bw() 
p4 <- ggplot(data=PimaIndiansDiabetes2, aes(x=insulin, y=mass, fill=diabetes)) +
  geom_point(pch=21, color="black", size=1.5) +
  scale_fill_discrete(name="Diabetes") +
  theme_bw() 
windows(width=7.0, height=5.5)
library(patchwork)
p1 + p2 + p3 + p4 +
  plot_layout(guides="collect") & theme(legend.position="bottom")

library(caret)
set.seed(123)
index <- createDataPartition(PimaIndiansDiabetes2$diabetes, p=0.8, list=FALSE)
pima.train <- PimaIndiansDiabetes2[index,]
pima.test <- PimaIndiansDiabetes2[-index,]
dim(pima.train)
dim(pima.test)

library(recipes)
recipe.step <- recipe(diabetes ~ ., data=pima.train) |> 
  step_impute_knn(all_predictors(), neighbors=5) |> 
  step_center(all_numeric(), -all_outcomes()) |> 
  step_scale(all_numeric(), -all_outcomes())

diabetes.train <- recipe.step |>  
  prep(training=pima.train) |> 
  bake(new_data=pima.train)
diabetes.test <- recipe.step |>
  prep(training=pima.train) |> 
  bake(new_data=pima.test)

library(class)
set.seed(123)
knn.fit1 <- knn(train=diabetes.train[,c("glucose", "insulin")], 
                test=diabetes.test[,c("glucose", "insulin")], 
                cl=diabetes.train$diabetes, k=1)
set.seed(123)
knn.fit5 <- knn(train=diabetes.train[,c("glucose", "insulin")], 
                test=diabetes.test[,c("glucose", "insulin")],
                cl=diabetes.train$diabetes, k=5)
data.frame(diabetes.test=diabetes.test$diabetes, knn.fit1, knn.fit5)
 
# [그림 7-6]
p1 <- ggplot(data=diabetes.train[,c("glucose", "insulin", "diabetes")], 
             aes(x=glucose, y=insulin, color=diabetes)) +
  geom_point(pch=21, size=2) +
  geom_point(data=diabetes.test[,c("glucose", "insulin", "diabetes")],
             aes(x=glucose, y=insulin, fill=knn.fit1), pch=21, color="black", size=2) +
  labs(title="1-nearest neighbor", color="Diabetes", fill="Diabetes") +
  theme_bw() 
p2 <- ggplot(data=diabetes.train[,c("glucose", "insulin", "diabetes")], 
             aes(x=glucose, y=insulin, color=diabetes)) +
  geom_point(pch=21, size=2) +
  geom_point(data=diabetes.test[,c("glucose", "insulin", "diabetes")],
             aes(x=glucose, y=insulin, fill=knn.fit5), pch=21, color="black", size=2) +
  labs(title="5-nearest neighbor", color="Diabetes", fill="Diabetes") +
  theme_bw() 
windows(width=7.0, height=5.5)
library(patchwork)
p1 + p2 + 
  plot_layout(guides="collect") & theme(legend.position="bottom")

table(diabetes.test$diabetes, knn.fit1, dnn=c("Actual", "Predicted"))
mean(diabetes.test$diabetes==knn.fit1)
table(diabetes.test$diabetes, knn.fit5, dnn=c("Actual", "Predicted"))
mean(diabetes.test$diabetes==knn.fit5)

set.seed(123)
knn.cv <- numeric(10)
for (k in 1:10) {
  knn.pred <- knn.cv(train=diabetes.train[c(1:8)], cl=diabetes.train$diabetes, k)
  knn.cv[k] <- round(mean(diabetes.train$diabetes==knn.pred), 3)
}
knn.cv
range(knn.cv)
which.max(knn.cv)
knn.cv[which.max(knn.cv)]

set.seed(123)
knn.fit <- knn(train=diabetes.train[c(1:8)], test=diabetes.test[c(1:8)], 
               cl=diabetes.train$diabetes, k=which.max(knn.cv))
table(diabetes.test$diabetes, knn.fit, dnn=c("Actual", "Predicted"))
mean(diabetes.test$diabetes==knn.fit)

knnRuns <- function(fraction, k, run) {
  trial.sum <- numeric(k)
  trial.n <- numeric(k)
  for (i in 1:run) {
    index <- createDataPartition(PimaIndiansDiabetes2$diabetes, p=fraction, list=FALSE)
    pima.train <- PimaIndiansDiabetes2[index,]
    pima.test <- PimaIndiansDiabetes2[-index,]
    recipe.step <- recipe(diabetes ~ ., data=pima.train) |> 
      step_impute_knn(all_predictors(), neighbors=5) |> 
      step_center(all_numeric(), -all_outcomes()) |> 
      step_scale(all_numeric(), -all_outcomes())
    diabetes.train <- recipe.step |>  
      prep(training=pima.train) |> 
      bake(new_data=pima.train)
    diabetes.test <- recipe.step |>
      prep(training=pima.train) |> 
      bake(new_data=pima.test)
    test.size <- nrow(diabetes.test)
    for (j in 1:k) {
      knn.fit <- knn(train=diabetes.train[c(1:8)], test=diabetes.test[c(1:8)], 
                     cl=diabetes.train$diabetes, k=j)
      trial.sum[j] <- trial.sum[j] + sum(diabetes.test$diabetes==knn.fit)
      trial.n[j] <- trial.n[j] + test.size
    }
  }
  return(data.frame(k=1:k, accuracy=trial.sum/trial.n))
}

set.seed(123)
knn.cv <- knnRuns(fraction=0.8, k=50, run=100)
knn.cv
summary(knn.cv$accuracy)
which.max(knn.cv$accuracy)
slice(knn.cv, which.max(knn.cv$accuracy))

# [그림 7-7]
windows(width=7.0, height=4.0)
ggplot(knn.cv, aes(x=k, y=accuracy)) +
  geom_line(color="dimgray", lwd=1) +
  geom_point(shape=21, color="black", fill="red", size=2, stroke=1) +
  labs(title="Accuracy for Diabetes Prediction with Varying k(100 Samples)",
       x="k", y="Accuracy") +
  scale_x_continuous(breaks=seq(from=0, to=50, by=5)) +
  theme_bw()

set.seed(123)
knn.fit <- knn(train=diabetes.train[c(1:8)], test=diabetes.test[c(1:8)], 
               cl=diabetes.train$diabetes, k=which.max(knn.cv$accuracy))
table(diabetes.test$diabetes, knn.fit, dnn=c("Actual", "Predicted"))
mean(diabetes.test$diabetes==knn.fit)

## 보스턴 주택가격

library(MASS)
str(Boston)

library(caret)
set.seed(123)
index <- createDataPartition(y=Boston$medv, p=0.8, list=FALSE)
Boston.train <- Boston[index,]
Boston.test <- Boston[-index,]
dim(Boston.train)
dim(Boston.test)

prePro <- preProcess(Boston.train[c(1:13)], method=c("center", "scale"))
class(prePro)
prePro
  
Boston.train <- data.frame(predict(prePro, newdata=Boston.train[c(1:13)]), 
                           medv=Boston.train$medv)
Boston.test <- data.frame(predict(prePro, newdata=Boston.test[c(1:13)]), 
                          medv=Boston.test$medv)

library(FNN)
set.seed(123)
knnreg.fit <- knn.reg(train=Boston.train[c(1:13)], test=Boston.test[c(1:13)], 
                      y=Boston.train$medv, k=5)
class(knnreg.fit)

knnreg.fit$pred
postResample(pred=knnreg.fit$pred, obs=Boston.test$medv)

set.seed(123)
index <- createDataPartition(y=Boston$medv, p=0.8, list=FALSE)
Boston.train <- Boston[index,]
Boston.test <- Boston[-index,]
dim(Boston.train)
dim(Boston.test)

modelLookup("knn")

set.seed(123)
caret.cv <- train(medv ~ ., data=Boston.train, method="knn",
                  preProcess=c("center", "scale"),
                  trControl=trainControl(method="cv", number=10))
caret.cv

caret.pred <- predict(caret.cv, newdata=Boston.test, type="raw")
caret.pred
postResample(pred=caret.pred, obs=Boston.test$medv)

set.seed(123)
caret.cv <- train(medv ~ ., data=Boston.train, method="knn",
                  preProcess=c("center", "scale"),
                  trControl=trainControl(method="cv", number=10),
                  tuneGrid=expand.grid(k=seq(from=1, to=20, by=1)))
caret.cv
caret.cv$bestTune

# [그림 7-8]
windows(width=7.0, height=4.0)
ggplot(data=caret.cv) +
  geom_line(color="dimgray", lwd=1) +
  geom_point(shape=21, color="blue", fill="deeppink", size=2, stroke=1) +
  labs(title="Performance for Housing Price Prediction with Varying k",
       x="k", y="RMSE") +
  scale_x_continuous(breaks=seq(from=1, to=20, by=1)) +
  theme_bw()

caret.pred <- predict(caret.cv, newdata=Boston.test, type="raw")
caret.pred
postResample(pred=caret.pred, obs=Boston.test$medv)
