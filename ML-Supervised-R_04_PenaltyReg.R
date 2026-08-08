
###################################
## R을 이용한 머신러닝: 지도학습 ##
## (곽기영, 도서출판 청람)       ## 
###################################

######################
## 제4장 페널티회귀 ##
######################

##############
## 4.5 사례 ##
##############

## 에임스 주택가격

library(modeldata)
str(ames)

library(rsample)
set.seed(123)
split <- initial_split(data=ames, prop=0.7, strata="Sale_Price")
ames.train <- training(split)
ames.test <- testing(split)
dim(ames.train)
dim(ames.test)

library(recipes)
recipe.step <- recipe(Sale_Price ~ ., data=ames.train) |> 
  step_dummy(all_nominal(), -all_outcomes())

homeprice.train <- recipe.step |>  
  prep(training=ames.train) |> 
  bake(new_data=ames.train)
homeprice.test <- recipe.step |>  
  prep(training=ames.train) |> 
  bake(new_data=ames.test)

x <- model.matrix(Sale_Price ~ ., homeprice.train)[,-1]
y <- homeprice.train$Sale_Price

library(glmnet)
glmnet.fit <- glmnet(x=x, y=y, family="gaussian", alpha=0)
class(glmnet.fit)

# [그림 4-7]
windows(width=7.0, height=5.5)
plot(glmnet.fit, xvar="lambda", sign.lambda=1)
plot(glmnet.fit, xvar="lambda")

?plot.glmnet

glmnet.fit$lambda

coef(glmnet.fit)[1:5, 96:100]

coef(glmnet.fit)[c("Gr_Liv_Area", "Overall_Cond_Excellent"), 100]
coef(glmnet.fit)[c("Gr_Liv_Area", "Overall_Cond_Excellent"), 1]

set.seed(123)
ridge.cv <- cv.glmnet(x=x, y=y, family="gaussian", alpha=0)
class(ridge.cv)
set.seed(123)
lasso.cv <- cv.glmnet(x=x, y=y, family="gaussian", alpha=1)
class(lasso.cv)

# [그림 4-8]
windows(width=5.5, height=7.0)
par(mfrow=c(2, 1))
plot(ridge.cv, main="Ridge Regression\n\n", sign.lambda=1)
plot(lasso.cv, main="Lasso Regression\n\n", sign.lambda=1)

ridge.cv$nzero
lasso.cv$nzero

min(ridge.cv$cvm)
ridge.cv$lambda.min

ridge.cv$cvm[ridge.cv$lambda==ridge.cv$lambda.1se]
ridge.cv$lambda.1se

min(lasso.cv$cvm)
lasso.cv$lambda.min
lasso.cv$cvm[lasso.cv$lambda==lasso.cv$lambda.1se]
lasso.cv$lambda.1se

lasso.cv$nzero[which(lasso.cv$lambda==lasso.cv$lambda.min)]
lasso.cv$nzero[which(lasso.cv$lambda==lasso.cv$lambda.1se)]

# [그림 4-9]
windows(width=5.5, height=7.0)
par(mfrow=c(2, 1))
ridge.fit <- glmnet(x=x, y=y, family="gaussian", alpha=0)
lasso.fit <- glmnet(x=x, y=y, family="gaussian", alpha=1)
plot(ridge.fit, xvar="lambda", main="Ridge Regression\n\n", sign.lambda=1)
abline(v=log(ridge.cv$lambda.min), col="red", lty="dashed")
abline(v=log(ridge.cv$lambda.1se), col="blue", lty="dashed")
plot(lasso.fit, xvar="lambda", main="Lasso Regression\n\n", sign.lambda=1)
abline(v=log(lasso.cv$lambda.min), col="red", lty="dashed")
abline(v=log(lasso.cv$lambda.1se), col="blue", lty="dashed")

x.test <- model.matrix(Sale_Price ~ ., homeprice.test)[,-1]

ridge.fit <- glmnet(x, y, family="gaussian", alpha=0, lambda=ridge.cv$lambda.min)
ridge.pred <- predict(ridge.fit, newx=x.test, type="response")
head(ridge.pred)

library(caret)
postResample(pred=ridge.pred, obs=homeprice.test$Sale_Price)

lasso.fit <- glmnet(x, y, family="gaussian", alpha=1, lambda=lasso.cv$lambda.min)
lasso.pred <- predict(lasso.fit, newx=x.test, type="response")
head(lasso.pred)
postResample(pred=lasso.pred, obs=homeprice.test$Sale_Price)

ridge.fit$df
lasso.fit$df

modelLookup("glmnet")
set.seed(123)
caret.cv <- train(x=x, y=y, method="glmnet",
                  preProcess=c("zv", "center", "scale"),
                  trControl=trainControl(method="cv", number=10),
                  tuneLength=10)
caret.cv

caret.cv$bestTune

caret.pred <- predict(caret.cv, newdata=homeprice.test, type="raw")
head(caret.pred)
postResample(pred=caret.pred, obs=homeprice.test$Sale_Price)

lm.fit <- lm(Sale_Price ~ ., data=homeprice.train)
lm.pred <- predict(lm.fit, newdata=homeprice.test)
head(lm.pred)
postResample(pred=lm.pred, obs=homeprice.test$Sale_Price)

predVIP <- function(object, newdata) {
  results <- as.vector(predict(object, newx=newdata, type="response"))
  return(results)
}
predVIP(object=lasso.fit, newdata=x) |> 
  head()

# [그림 4-10]
windows(width=7.0, height=5.5)
library(vip)
set.seed(123)
imp <- vip(lasso.fit, method="permute", train=x, target=y, 
           metric="rmse", nsim=5, pred_wrapper=predVIP,
           geom="point", num_features=20, 
           aesthetics=list(color="maroon", fill="maroon", size=2))
imp
imp$data

list_metrics()

predPDP <- function(object, newdata) {
  results <- mean(predict(object, newx=newdata, type="response"))
  return(results)
}
predPDP(object=lasso.fit, newdata=x)

library(pdp)
partial(lasso.fit, pred.var="Gr_Liv_Area", pred.fun=predPDP, train=x)
partial(lasso.fit, pred.var="Lot_Shape_Irregular", pred.fun=predPDP, train=x)

# [그림 4-11]
library(scales)
p1 <- partial(lasso.fit, pred.var="Gr_Liv_Area", pred.fun=predPDP, train=x) |> 
  ggplot(aes(x=Gr_Liv_Area, y=yhat)) +
  geom_line(color="tomato", linewidth=1) +
  scale_y_continuous(limits=c(0, 300000), labels=label_dollar())
p2 <- partial(lasso.fit, pred.var="Lot_Shape_Irregular", pred.fun=predPDP, train=x) |> 
  mutate(Lot_Shape_Irregular=factor(Lot_Shape_Irregular)) |> 
  ggplot(aes(x=Lot_Shape_Irregular, y=yhat)) +
  geom_boxplot(color="tomato") +
  scale_y_continuous(limits=c(0, 300000), labels=label_dollar())
windows(width=7.0, height=3.5)
library(patchwork)
p1 + p2

# [그림 4-12]
windows(width=7.0, height=5.5)
imp <- vip(caret.cv, method="model", geom="point", num_features=20, 
           aesthetics=list(color="darkslateblue", fill="darkslateblue", size=2))
imp
imp$data

# [그림 4-13]
p1 <- partial(caret.cv, pred.var="Year_Built") |> 
  ggplot(aes(x=Year_Built, y=yhat)) +
  geom_line(color="royalblue", linewidth=1) +
  scale_y_continuous(limits=c(0, 300000), labels=label_dollar())
p2 <- partial(caret.cv, pred.var="Roof_Matl_WdShngl") |> 
  mutate(Roof_Matl_WdShngl=factor(Roof_Matl_WdShngl)) |> 
  ggplot(aes(x=Roof_Matl_WdShngl, y=yhat)) +
  geom_point(pch=21, color="blue", bg="royalblue", size=3, stroke=1) +
  scale_y_continuous(limits=c(0, 300000), labels=label_dollar())
windows(width=7.0, height=3.5)
library(patchwork)
p1 + p2

## 이직

library(modeldata)
str(attrition)
levels(attrition$Attrition)

library(dplyr)
attrition <- attrition |>
  mutate(JobLevel=factor(JobLevel), StockOptionLevel=factor(StockOptionLevel))

library(caret)
set.seed(123)
index <- createDataPartition(y=attrition$Attrition, p=0.7, list=FALSE)
attrition.train <- attrition[index,]
attrition.test <- attrition[-index,]
dim(attrition.train)
dim(attrition.test)
table(attrition.train$Attrition)
prop.table(table(attrition.train$Attrition))
table(attrition.test$Attrition)
prop.table(table(attrition.test$Attrition))

library(recipes)
recipe.step <- recipe(Attrition ~ ., data=attrition.train) |>
  step_dummy(all_nominal(), -all_outcomes())
attr.train <- recipe.step |>  
  prep(training=attrition.train) |> 
  bake(new_data=attrition.train)
attr.test <- recipe.step |>  
  prep(training=attrition.train) |> 
  bake(new_data=attrition.test)

x <- model.matrix(Attrition ~ ., attr.train)[,-1]
y <- attr.train$Attrition

library(glmnet)
set.seed(123)
glmnet.cv <- cv.glmnet(x=x, y=y, family="binomial", alpha=1)
glmnet.cv$lambda.min

lasso.fit <- glmnet(x, y, family="binomial", alpha=1, lambda=glmnet.cv$lambda.min)
coef(lasso.fit)
lasso.fit$df

x.test <- model.matrix(Attrition ~ ., attr.test)[,-1]
predict(lasso.fit, newx=x.test, type="class")
lasso.pred <- predict(lasso.fit, newx=x.test, type="response")
head(lasso.pred)
pred <- factor(ifelse(lasso.pred >= 0.5, "Yes", "No"))
table(attr.test$Attrition, pred, dnn=c("Actual", "Predicted"))
mean(attr.test$Attrition==pred)

set.seed(123)
caret.cv <- train(x=x, y=y, method="glmnet",
                  preProcess=c("zv", "center", "scale"),
                  trControl=trainControl(method="cv", number=10),
                  tuneLength=10)
caret.cv$bestTune

predict(caret.cv, newdata=attr.test, type="prob")
caret.pred <- predict(caret.cv, newdata=attr.test, type="raw")
head(caret.pred)
table(attr.test$Attrition, caret.pred, dnn=c("Actual", "Predicted"))
mean(attr.test$Attrition==caret.pred)

glm.fit <- glm(Attrition ~ ., data=attr.train, family=binomial(link="logit"))
glm.pred <- predict(glm.fit, newdata=attr.test, type="response")
head(glm.pred)
pred <- factor(ifelse(glm.pred >= 0.5, "Yes", "No"))
table(attr.test$Attrition, pred, dnn=c("Actual", "Predicted"))
mean(attr.test$Attrition==pred)

predVIP <- function(object, newdata) {
  results <- as.factor(predict(object, newx=newdata, type="class"))
  return(results)
}
predVIP(object=lasso.fit, newdata=x) |> 
  head()

# [그림 4-14]
windows(width=7.0, height=5.5)
library(vip)
set.seed(123)
imp <- vip(lasso.fit, method="permute", train=x, target=y, 
           metric="accuracy", nsim=5, pred_wrapper=predVIP,
           geom="point", num_features=20, 
           aesthetics=list(color="darkcyan", fill="darkcyan", size=2))
imp
imp$data

predPDP <- function(object, newdata) {
  results <- mean(predict(object, newx=newdata, type="response"))
  return(results)
}
predPDP(object=lasso.fit, newdata=x) |> 
  head()

library(pdp)
partial(lasso.fit, pred.var="NumCompaniesWorked", pred.fun=predPDP, train=x)
partial(lasso.fit, pred.var="OverTime_Yes", pred.fun=predPDP, train=x)

# [그림 4-15]
p1 <- partial(lasso.fit, pred.var="NumCompaniesWorked", pred.fun=predPDP, train=x) |> 
  ggplot(aes(x=NumCompaniesWorked, y=yhat)) +
  geom_line(color="seagreen", linewidth=1) +
  ylim(c(0, 1))
p2 <- partial(lasso.fit, pred.var="OverTime_Yes", pred.fun=predPDP, train=x) |> 
  mutate(OverTime_Yes=factor(OverTime_Yes)) |> 
  ggplot(aes(x=OverTime_Yes, y=yhat)) +
  geom_boxplot(color="seagreen") +
  ylim(c(0, 1))
windows(width=7.0, height=3.5)
library(patchwork)
p1 + p2

# [그림 4-16]
windows(width=7.0, height=5.5)
imp <- vip(caret.cv, method="model", geom="point", num_features=20, 
           aesthetics=list(color="deeppink3", fill="deeppink3", size=2))
imp
imp$data

predict(caret.cv, newdata=x, type="prob")

# [그림 4-17]
p1 <- partial(caret.cv, pred.var="Age", prob=TRUE, which.class=2) |> 
  ggplot(aes(x=Age, y=yhat)) +
  geom_line(color="darkviolet", linewidth=1) +
  ylim(c(0, 1))
p2 <- partial(caret.cv, pred.var="BusinessTravel_Travel_Frequently", prob=TRUE, which.class=2) |> 
  mutate(BusinessTravel_Travel_Frequently=factor(BusinessTravel_Travel_Frequently)) |> 
  ggplot(aes(x=BusinessTravel_Travel_Frequently, y=yhat)) +
  geom_point(pch=21, color="darkred", bg="darkviolet", size=3, stroke=1) +
  ylim(c(0, 1))
windows(width=7.0, height=3.5)
library(patchwork)
p1 + p2
