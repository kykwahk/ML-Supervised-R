
###################################
## R을 이용한 머신러닝: 지도학습 ##
## (곽기영, 도서출판 청람)       ## 
###################################

#####################
## 제13장 AdaBoost ##
#####################

###############
## 13.4 사례 ##
###############

## 펭귄

library(modeldata)
str(penguins)
levels(penguins$species)

library(caret)
peng <- as.data.frame(penguins)
set.seed(123)
index <- createDataPartition(y=peng$species, p=0.85, list=FALSE)
peng.train <- peng[index,] 
peng.test <- peng[-index,]
dim(peng.train)
dim(peng.test)

library(adabag)
set.seed(123)
adaboost.fit <- boosting(species ~ ., data=peng.train, boos=TRUE, mfinal=50, 
                         control=rpart.control(maxdepth=1))
class(adaboost.fit)

adaboost.fit$trees

adaboost.fit$weights

adaboost.fit$votes
adaboost.fit$prob
adaboost.fit$class

adaboost.fit$votes[1,]/sum(adaboost.fit$votes[1,])

adaboost.fit$importance
importanceplot(adaboost.fit)

# [그림 13-14]
windows(width=7.0, height=5.5)
library(ggplot2)
ggplot(data.frame(var=reorder(names(adaboost.fit$importance), adaboost.fit$importance,
                              decreasing=TRUE), 
                  imp=adaboost.fit$importance), 
       aes(x=var, y=imp)) +
  geom_col(color="thistle4", fill="lightblue") +
  labs(x="Variable", y="Importance")

table(adaboost.fit$class, peng.train$species, dnn=c("Predicted", "Observed"))
1 - mean(adaboost.fit$class==peng.train$species)

adaboost.pred <- predict(adaboost.fit, newdata=peng.test)

head(adaboost.pred$prob)
head(adaboost.pred$class)

adaboost.pred$confusion
adaboost.pred$error

errorevol.train <- errorevol(adaboost.fit,  newdata=peng.train)
errorevol.train
plot(errorevol.train)
errorevol.test <- errorevol(adaboost.fit,  newdata=peng.test)
errorevol.test
plot(errorevol.test)

# [그림 13-15]
windows(width=7.0, height=5.5)
library(tidyr)
errorevol <- data.frame(niter=1:length(adaboost.fit$trees), 
                        train=errorevol.train$error, test=errorevol.test$error) |>
  pivot_longer(cols=c(train, test), names_to="type", values_to="error")
ggplot(errorevol, aes(x=niter, y=error, color=type)) +
  geom_line(linewidth=1) +
  scale_x_continuous(breaks=seq(from=0, to=50, by=5)) +
  labs(x="Number of Trees", y="Error")+
  theme_classic() +
  theme(legend.position="bottom",
        legend.title=element_blank())

adaboost.pruned <- predict(adaboost.fit, newdata=peng.test, newmfinal=25)
adaboost.pruned$confusion
adaboost.pruned$error

set.seed(123)
adaboost.cv <- boosting.cv(species ~ ., data=peng.train, v=10, boos=TRUE, mfinal=50, 
                           control=rpart.control(maxdepth=1))

adaboost.cv$confusion
adaboost.cv$error

modelLookup("AdaBoost.M1")

hyper.grid <- expand.grid(
  mfinal=seq(from=10, to=100, by=10),
  maxdepth=c(1, 2),
  coeflearn="Breiman"
  )
hyper.grid

set.seed(123)
caret.cv <- train(species ~ ., data=peng.train, method="AdaBoost.M1",
                  trControl=trainControl(method="cv", number=10),
                  tuneGrid=hyper.grid, na.action=na.pass)
caret.cv
caret.cv$bestTune

getTrainPerf(caret.cv)

caret.pred <- predict(caret.cv, newdata=peng.test, type="raw", na.action=na.pass)
head(caret.pred)
table(caret.pred, peng.test$species, dnn=c("Predicted", "Observed"))
1 - mean(caret.pred==peng.test$species)
