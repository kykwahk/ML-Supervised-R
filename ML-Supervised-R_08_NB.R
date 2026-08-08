
###################################
## R을 이용한 머신러닝: 지도학습 ##
## (곽기영, 도서출판 청람)       ## 
###################################

########################
## 제8장 나이브베이즈 ##
########################

##############
## 8.4 사례 ##
##############

## 정당

library(mlbench)
data(HouseVotes84)
str(HouseVotes84)
levels(HouseVotes84$V1)
levels(HouseVotes84$Class)

# [그림 8-8]
windows(width=8.0, height=5.5)
library(tidyr)
library(stringr)
library(ggplot2)
ggplot(data=pivot_longer(HouseVotes84, cols=!Class, 
                         names_to="Variable", values_to="Value"),
       aes(x=Class, fill=Value)) +
  facet_wrap(vars(factor(Variable, str_sort(unique(Variable), numeric=TRUE))), 
             scales="free_y") +
  geom_bar(position="fill") +
  scale_fill_manual(values=c("indianred", "green3"), 
                    name="Voting", labels=c("No", "Yes", "Missing")) +
  labs(x="", y="Proportion") +
  theme_bw() +
  theme(legend.position="bottom")

library(rsample)
set.seed(123)
split <- initial_split(data=HouseVotes84, prop=0.7, strata="Class")
HouseVotes84.train <- training(split)
HouseVotes84.test <- testing(split)
dim(HouseVotes84.train)
dim(HouseVotes84.test)
table(HouseVotes84.train$Class)
table(HouseVotes84.test$Class)
prop.table(table(HouseVotes84.train$Class))
prop.table(table(HouseVotes84.test$Class))

sum(is.na(HouseVotes84))
head(HouseVotes84)
sum(is.na(HouseVotes84.train))
sum(is.na(HouseVotes84.test))

library(recipes)
recipe.step <- recipe(Class ~ ., data=HouseVotes84.train)|> 
  step_impute_mode(all_predictors()) 
recipe.step

votes.train <- recipe.step|>  
  prep(training=HouseVotes84.train)|> 
  bake(new_data=HouseVotes84.train)
votes.test <- recipe.step|>  
  prep(training=HouseVotes84.train)|> 
  bake(new_data=HouseVotes84.test)

anyNA(votes.train)
anyNA(votes.test)

library(e1071)
nb.fit <- naiveBayes(Class ~ ., data=votes.train)
class(nb.fit)

nb.fit

nb.pred <- predict(nb.fit, newdata=votes.test[-1], type="class")
nb.pred <- predict(nb.fit, newdata=votes.test[-1])
head(nb.pred)

table(votes.test$Class, nb.pred, dnn=c("Actual", "Predicted"))
mean(votes.test$Class==nb.pred)

nb.pred <- predict(nb.fit, newdata=votes.test[-1], type="raw")
head(nb.pred)

pred <- factor(nb.pred[,"republican"] >= 0.5, levels=c(FALSE, TRUE),
               labels=c("democrat", "republican"))
head(pred)
table(votes.test$Class, pred, dnn=c("Actual", "Predicted"))
mean(votes.test$Class==pred)

nbRuns <- function(fraction, run) {
  results <- numeric()
  for (i in 1:run) {
    split <- initial_split(data=HouseVotes84, prop=0.7, strata="Class")
    HouseVotes84.train <- training(split)
    HouseVotes84.test <- testing(split)
    recipe.step <- recipe(Class ~ ., data=HouseVotes84.train)|> 
      step_impute_mode(all_predictors()) 
    votes.train <- recipe.step|>  
      prep(training=HouseVotes84.train)|> 
      bake(new_data=HouseVotes84.train)
    votes.test <- recipe.step|>  
      prep(training=HouseVotes84.train)|> 
      bake(new_data=HouseVotes84.test)
    nb.fit <- naiveBayes(Class ~ ., data=votes.train)
    nb.pred <- predict(nb.fit, newdata=votes.test[-1])
    results[i] <- mean(votes.test$Class==nb.pred)
  }
  return(results)
}

nb.cv <- nbRuns(fraction=0.7, run=100)
nb.cv
summary(nb.cv)

# [그림 8-9]
windows(width=7.0, height=4.0)
ggplot(data.frame(acc=nb.cv), aes(x="", y=acc)) +
  geom_boxplot(fill="aliceblue", color="darkslategray", width=0.3) +
  geom_point(position="jitter", pch=21, color="blue", fill="cornflowerblue") +
  labs(title="Accuracy for Party Prediction with 100 Samples",
       y="Accuracy") +
  coord_flip() +
  theme(axis.title.y=element_blank(),
        axis.text.y=element_blank(),
        axis.ticks=element_blank())

## 이직

library(modeldata)
str(attrition)

library(dplyr)
attrition <- attrition|>
  mutate(JobLevel=factor(JobLevel), StockOptionLevel=factor(StockOptionLevel))

library(caret)
set.seed(123)
index <- createDataPartition(y=attrition$Attrition, p=0.7, list=FALSE)
attrition.train <- attrition[index,]
attrition.test <- attrition[-index,]
dim(attrition.train)
dim(attrition.test)
table(attrition.train$Attrition)
table(attrition.test$Attrition)
prop.table(table(attrition.train$Attrition))
prop.table(table(attrition.test$Attrition))

library(e1071)
nb.fit <- naiveBayes(Attrition ~ ., data=attrition.train)
nb.fit

nb.pred <- predict(nb.fit, newdata=attrition.test[-2])
head(nb.pred)
table(attrition.test$Attrition, nb.pred, dnn=c("Actual", "Predicted"))
mean(attrition.test$Attrition==nb.pred)

# [그림 8-10]
windows(width=7.0, height=5.5)
library(tidyr)
library(ggplot2)
attrition.train|> 
  pivot_longer(cols=where(is.numeric), names_to="variable", values_to="value")|> 
  filter(variable %in% c("Age", "HourlyRate", "MonthlyIncome", "YearsInCurrentRole"))|> 
  ggplot(aes(x=value, fill=variable)) + 
  geom_density(show.legend=FALSE, outline.type="full") + 
  facet_wrap(vars(variable), scales="free") +
  theme_bw()

library(klaR)
nb.fit <- NaiveBayes(Attrition ~ ., data=attrition.train, usekernel=TRUE)
class(nb.fit)

nb.pred <- predict(nb.fit, newdata=attrition.test[-2])
names(nb.pred)
head(nb.pred$class)
head(nb.pred$posterior)

table(attrition.test$Attrition, nb.pred$class, dnn=c("Actual", "Predicted"))
mean(attrition.test$Attrition==nb.pred$class)

modelLookup("nb")

x <- attrition.train[setdiff(names(attrition.train), "Attrition")]
y <- attrition.train$Attrition
set.seed(123)
caret.cv <- train(x=x, y=y, method="nb",
                  trControl=trainControl(method="cv", number=10),
                  tuneGrid=expand.grid(fL=0:5,
                                       usekernel=c(TRUE, FALSE),
                                       adjust=seq(from=1, to=5, by=1)))

caret.cv$bestTune
caret.cv$results|> 
  slice_max(order_by=Accuracy, n=5, with_ties=FALSE)

caret.pred <- predict(caret.cv, newdata=attrition.test, type="raw")
head(caret.pred)
table(attrition.test$Attrition, caret.pred, dnn=c("Actual", "Predicted"))
mean(attrition.test$Attrition==caret.pred)

set.seed(123)
caret.cv <- train(x=x, y=y, method="nb",
                  preProc=c("BoxCox", "center", "scale", "pca"),
                  trControl=trainControl(method="cv", number=10),
                  tuneGrid=expand.grid(fL=0:5,
                                       usekernel=c(TRUE, FALSE),
                                       adjust=seq(from=1, to=5, by=1)))
caret.cv$bestTune
getTrainPerf(caret.cv)
caret.pred <- predict(caret.cv, newdata=attrition.test, type="raw")
postResample(pred=caret.pred, obs=attrition.test$Attrition)

# [그림 8-11]
windows(width=7.0, height=5.5)
ggplot(caret.cv) +
  theme_bw() +
  theme(legend.position="top")
