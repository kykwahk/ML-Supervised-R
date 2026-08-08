
###################################
## R을 이용한 머신러닝: 지도학습 ##
## (곽기영, 도서출판 청람)       ## 
###################################

########################
## 제3장 로지스틱회귀 ##
########################

##############
## 3.4 사례 ##
##############

## 신용평가

library(modeldata)
str(credit_data)

levels(credit_data$Status)

library(naniar)
miss_var_summary(credit_data, add_cumsum=TRUE)

library(caret)
nearZeroVar(credit_data, saveMetrics=TRUE)

set.seed(123)
index <- createDataPartition(y=credit_data$Status, p=0.7, list=FALSE)
credit_data.train <- credit_data[index,]
credit_data.test <- credit_data[-index,]
dim(credit_data.train)
dim(credit_data.test)

library(recipes)
recipe.step <- recipe(Status ~ ., data=credit_data.train) |>
  step_relevel(Status, ref_level="good") |> 
  step_impute_knn(all_predictors(), neighbors=5) |>
  step_nzv(all_predictors()) |> 
  step_dummy(all_nominal(), -all_outcomes())

credit.train <- recipe.step |>  
  prep(training=credit_data.train) |> 
  bake(new_data=credit_data.train)
credit.test <- recipe.step |>  
  prep(training=credit_data.train) |> 
  bake(new_data=credit_data.test)

levels(credit.train$Status)
levels(credit.test$Status)
anyNA(credit.train)
anyNA(credit.test)

table(credit.train$Status)
prop.table(table(credit.train$Status))
table(credit.test$Status)
prop.table(table(credit.test$Status))

glm.fit <- glm(Status ~ ., data=credit.train, family=binomial(link="logit"))
class(glm.fit)
summary(glm.fit)

deviance(glm.fit)
glm.fit$deviance 

names(glm.fit)

glm.fit$null.deviance

AIC(glm.fit)
glm.fit$aic

exp(coef(glm.fit))

z <- coef(glm.fit)[1] + 
  (as.matrix(credit.test[setdiff(names(credit.test), "Status")]) %*% coef(glm.fit)[-1])
p <- 1/(1+exp(-z))
head(p)

glm.pred <- predict(glm.fit, newdata=credit.test, type="response")
head(glm.pred)

pred <- factor(glm.pred >= 0.5, levels=c(FALSE, TRUE), labels=c("good", "bad"))
head(pred)
table(pred)

table(credit.test$Status, pred, dnn=c("Actual", "Predicted"))
mean(credit.test$Status==pred)

postResample(pred=pred, obs=credit.test$Status)

library(yardstick)
metrics(data=data.frame(pred=pred, obs=credit.test$Status), 
        truth=obs, estimate=pred)

set.seed(123)
caret.cv <- train(Status ~ ., data=credit.train, method="glm", family="binomial",
                  trControl=trainControl(method="cv", number=10))
caret.cv

predict(caret.cv, newdata=credit.test, type="prob")

caret.pred <- predict(caret.cv, newdata=credit.test, type="raw")
head(caret.pred)
table(credit.test$Status, caret.pred, dnn=c("Actual", "Predicted"))
mean(credit.test$Status==caret.pred)

?predict.train
?predict.glm
?predict.lm
methods(predict)

# [그림 3-9]
windows(width=7.0, height=5.5)
library(vip)
imp <- vip(glm.fit, method="model", geom="col", num_features=20,
           aesthetics=list(color="cornflowerblue", fill="cornflowerblue"))
imp
imp$data

# [그림 3-10]
library(pdp)
p1 <- partial(glm.fit, pred.var="Seniority", prob=TRUE) |> 
  ggplot(aes(x=Seniority, y=yhat)) +
  geom_line(color="chocolate", linewidth=1) +
  ylim(c(0, 1))
p2 <- partial(glm.fit, pred.var="Job_partime", prob=TRUE) |> 
  mutate(Job_partime=factor(Job_partime)) |> 
  ggplot(aes(x=Job_partime, y=yhat)) +
  geom_boxplot(color="chocolate") +
  ylim(c(0, 1))
windows(width=7.0, height=3.5)
library(patchwork)
p1 + p2

## 정치성향

library(EffectStars)
data(PID)
str(PID)
head(PID)
levels(PID$PID)

library(rsample)
set.seed(123)
split <- initial_split(data=PID, prop=0.7, strata="PID")
PID.train <- training(split)
PID.test <- testing(split)
dim(PID.train)
dim(PID.test)

library(VGAM)
vglm.fit <- vglm(PID ~ ., family=multinomial(), data=PID.train)
class(vglm.fit)
summary(vglm.fit)

exp(coef(vglm.fit))

vglm.pred <- predict(vglm.fit, newdata=PID.test, type="response")
head(vglm.pred)

pred <- colnames(vglm.pred)[max.col(vglm.pred)]
head(pred)

table(PID.test$PID, pred, dnn=c("Actual", "Predicted"))
mean(PID.test$PID==pred)

testdata <- data.frame(Education=c("low", "high"),
                       TVnews=mean(PID.train$TVnews),
                       Income=mean(PID.train$Income),
                       Age=mean(PID.train$Age),
                       Population=mean(PID.train$Population))
testdata

vglm.pred <- predict(vglm.fit, newdata=testdata, type="response")
cbind(testdata, vglm.pred)

testdata <- data.frame(Education=rep("low", 5),
                       TVnews=mean(PID.train$TVnews),
                       Income=seq(20, 100, 20),
                       Age=mean(PID.train$Age),
                       Population=mean(PID.train$Population))
testdata
vglm.pred <- predict(vglm.fit, newdata=testdata, type="response")
cbind(testdata, vglm.pred)

vglmRuns <- function(fraction, run) {
  results <- numeric()
  for (i in 1:run) {
    split <- initial_split(data=PID, prop=fraction, strata="PID")
    PID.train <- training(split)
    PID.test <- testing(split)
    vglm.fit <- vglm(PID ~ ., family=multinomial(), data=PID.train)
    vglm.pred <- predict(vglm.fit, newdata=PID.test, type="response")
    pred <- colnames(vglm.pred)[max.col(vglm.pred)]
    results[i] <- mean(PID.test$PID==pred)
  }
  return(results)
}
vglm.cv <- vglmRuns(fraction=0.7, run=100)
vglm.cv
summary(vglm.cv)

# [그림 3-11]
windows(width=7.0, height=4.0)
library(ggplot2)
ggplot(data.frame(acc=vglm.cv), aes(x="", y=acc)) +
  geom_violin(fill="mistyrose", color="salmon", width=0.3) +
  geom_point(position="jitter", pch=21, color="black", fill="red") +
  labs(title="Accuracy for Party Identification Prediction with 100 Samples",
       y="Accuracy") +
  coord_flip() +
  theme(axis.title.y=element_blank(),
        axis.text.y=element_blank(),
        axis.ticks=element_blank())

library(caret)
set.seed(123)
caret.cv <- train(PID ~ ., data=PID.train, method="multinom", 
                  trControl=trainControl(method="cv", number=10))
caret.cv
caret.cv$resample
summary(caret.cv$resample$Accuracy)
caret.cv$finalModel

getTrainPerf(caret.cv)

caret.pred <- predict(caret.cv, newdata=PID.test, type="raw")
head(caret.pred)
table(PID.test$PID, caret.pred, dnn=c("Actual", "Predicted"))
mean(PID.test$PID==caret.pred)
