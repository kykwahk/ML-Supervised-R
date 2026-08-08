
###################################
## R을 이용한 머신러닝: 지도학습 ##
## (곽기영, 도서출판 청람)       ## 
###################################

###################
## 제16장 스태킹 ##
###################

###############
## 16.3 사례 ##
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

x.train <- Boston.train[setdiff(names(Boston.train), "medv")]
y.train <- Boston.train$medv
x.test <- Boston.test[setdiff(names(Boston.test), "medv")]
y.test <- Boston.test$medv

library(SuperLearner)
listWrappers()

SL.glmnet

set.seed(123)
sl.fit <- SuperLearner(Y=y.train, X=x.train, family=gaussian(), 
                       SL.library="SL.glmnet")
class(sl.fit)

sl.fit

sl.fit$cvRisk
sl.fit$coef
coef(sl.fit)

set.seed(123)
sl.fit <- SuperLearner(Y=y.train, X=x.train, family=gaussian(), 
                       SL.library=c("SL.mean", "SL.glmnet", "SL.rpart"))
sl.fit

sl.fit$Z

sl.pred <- predict(sl.fit, newdata=x.test, onlySL=TRUE)

str(sl.pred)
head(sl.pred$pred)
head(sl.pred$library.predict)

mean((y.test - sl.pred$pred[,1])^2)
cor(y.test, sl.pred$pred[,1])

# [그림 16-3]
windows(width=7.0, height=5.5)
library(ggplot2)
ggplot(data.frame(pred=sl.pred$pred[,1], actual=y.test), aes(x=pred, y=actual)) +
  geom_abline(intercept=0, slope=1, color="cornflowerblue", linewidth=1) +
  geom_point(pch=21, color="black", bg="salmon", size=2) + 
  labs(x="Predicted Value", y="Actual Value") +
  theme_classic()

set.seed(123)
sl.cv <- CV.SuperLearner(Y=y.train, X=x.train, family=gaussian(), 
                         cvControl=list(V=10), innerCvControl=list(list(V=5)),
                         SL.library=c("SL.mean", "SL.glmnet", "SL.rpart"))
class(sl.cv)

summary(sl.cv)

sl.cv$whichDiscreteSL
table(simplify2array(sl.cv$whichDiscreteSL))

# [그림 16-4]
windows(width=7.0, height=5.5)
plot(sl.cv) +
  theme_bw()

SL.glmnet

SL.glmnet.tune <- function(...) {
  SL.glmnet(..., alpha=0.5)
}

set.seed(123)
sl.cv <- CV.SuperLearner(Y=y.train, X=x.train, family=gaussian(), 
                         cvControl=list(V=10), innerCvControl=list(list(V=5)),
                         SL.library=c("SL.mean", "SL.glmnet", "SL.rpart", "SL.glmnet.tune"))
summary(sl.cv)

learners <- create.Learner("SL.glmnet", params=list(alpha=0.5))
learners

learners$names
SL.glmnet_1

set.seed(123)
sl.cv <- CV.SuperLearner(Y=y.train, X=x.train, family=gaussian(), 
                         cvControl=list(V=10), innerCvControl=list(list(V=5)),
                         SL.library=c("SL.mean", "SL.glmnet", "SL.rpart", learners$names))
summary(sl.cv)

SL.rpart

learners <- create.Learner("SL.rpart", tune=list(minsplit=c(10, 15), maxdepth=c(20, 30)))
learners$grid
learners$names
SL.rpart_1
SL.rpart_2
SL.rpart_3
SL.rpart_4

set.seed(123)
sl.cv <- CV.SuperLearner(Y=y.train, X=x.train, family=gaussian(), 
                         cvControl=list(V=10), innerCvControl=list(list(V=5)),
                         SL.library=c("SL.mean", "SL.glmnet", "SL.rpart", learners$names))
summary(sl.cv)

set.seed(123)
sl.fit <- SuperLearner(Y=y.train, X=x.train, family=gaussian(), 
                       SL.library=c("SL.mean", "SL.glmnet", "SL.rpart", learners$names))
sl.fit

sl.pred <- predict(sl.fit, newdata=x.test, onlySL=TRUE)
head(sl.pred$pred)
mean((y.test - sl.pred$pred[,1])^2)
cor(y.test, sl.pred$pred[,1])

library(parallel)
detectCores()
cl <- makeCluster(4)
cl

clusterEvalQ(cl, library(SuperLearner))

clusterExport(cl, learners$names)

clusterSetRNGStream(cl, 123)

sl.cv <- CV.SuperLearner(Y=y.train, X=x.train, family=gaussian(), parallel=cl,
                         cvControl=list(V=10), innerCvControl=list(list(V=5)),
                         SL.library=c("SL.mean", "SL.glmnet", "SL.rpart", learners$names))
summary(sl.cv)

stopCluster(cl)

library(parallel)
cl <- makeCluster(4)
clusterEvalQ(cl, library(SuperLearner))
clusterExport(cl, learners$names)
clusterSetRNGStream(cl, 123)
sl.fit <- snowSuperLearner(Y=y.train, X=x.train, family=gaussian(), cluster=cl,
                           SL.library=c("SL.mean", "SL.glmnet", "SL.rpart", learners$names))
sl.fit
stopCluster(cl)

listWrappers()

screen.corP

set.seed(123)
sl.cv <- CV.SuperLearner(Y=y.train, X=x.train, family=gaussian(), 
                         cvControl=list(V=10), innerCvControl=list(list(V=5)),
                         SL.library=list("SL.mean", "SL.glmnet", c("SL.glmnet", "screen.corP")))
summary(sl.cv)

## 당뇨병

library(mlbench)
data(PimaIndiansDiabetes2)
str(PimaIndiansDiabetes2)
levels(PimaIndiansDiabetes2$diabetes)

library(caret)
set.seed(123)
index <- createDataPartition(PimaIndiansDiabetes2$diabetes, p=0.7, list=FALSE)
pimad.train <- PimaIndiansDiabetes2[index,]
pimad.test <- PimaIndiansDiabetes2[-index,]
dim(pimad.train)
dim(pimad.test)

library(recipes)
recipe.step <- recipe(diabetes ~ ., data=pimad.train) |> 
  step_impute_knn(all_predictors(), neighbors=5) 

pima.train <- recipe.step |>  
  prep(training=pimad.train) |> 
  bake(new_data=pimad.train)
pima.test <- recipe.step |>
  prep(training=pimad.train) |> 
  bake(new_data=pimad.test)

modelLookup()["model"]

library(caretEnsemble)
mycontrol <- trainControl(method="boot", number=25, savePredictions="final",
                          classProbs=TRUE, index=createResample(pima.train$diabetes, 25))
mymethod <- c("rpart", "knn")
set.seed(123)
caretlist.fit <- caretList(diabetes ~ ., data=pima.train, trControl=mycontrol, 
                           methodList=mymethod)
class(caretlist.fit)

summary(resamples(caretlist.fit))

mytune <- list(rf1=caretModelSpec(method="rf", tuneGrid=data.frame(mtry=3)),
               rf2=caretModelSpec(method="rf", tuneGrid=data.frame(mtry=5), preProcess="pca"))
set.seed(123)
caretlist.tune <- caretList(diabetes ~ ., data=pima.train, trControl=mycontrol, 
                            methodList=mymethod, tuneList=mytune)
summary(resamples(caretlist.tune))

modelCor(resamples(caretlist.fit))

# [그림 16-5]
windows(width=7.0, height=5.5)
splom(resamples(caretlist.fit))

mycontrol <- trainControl(method="boot", number=10, savePredictions="final",
                          classProbs=TRUE)
set.seed(123)
caretstack.fit <- caretStack(caretlist.fit, method="glm", metric="Accuracy", 
                             trControl=mycontrol)
class(caretstack.fit)
caretstack.fit

caretstack.pred <- predict(caretstack.fit, newdata=pima.test)
head(caretstack.pred)
caretstack.pred <- predict(caretstack.fit, newdata=pima.test, return_class_only=TRUE)
head(caretstack.pred)

table(pima.test$diabetes, caretstack.pred, dnn=c("Actual", "Predicted"))
mean(pima.test$diabetes==caretstack.pred)
postResample(caretstack.pred, pima.test$diabetes)
