
###################################
## R을 이용한 머신러닝: 지도학습 ##
## (곽기영, 도서출판 청람)       ## 
###################################

####################
## 제2장 선형회귀 ##
####################

#######################
## 2.2 회귀모델 유형 ##
#######################

## 단순회귀

library(car)
str(Prestige)

lm.fit <- lm(income ~ education, data=Prestige)
class(lm.fit)
lm.fit

# [그림 2-5]
windows(width=7.0, height=5.5)
library(ggplot2)
library(scales)
ggplot(Prestige, aes(x=education, y=income)) +
  geom_abline(slope=coef(lm.fit)[["education"]], 
              intercept=coef(lm.fit)[["(Intercept)"]],
              col="cornflowerblue", lwd=1) +
  geom_point(pch=21, col="black", bg="coral", size=2.5) +
  scale_y_continuous(breaks=seq(0, 30000, 5000), limits=c(500, 30000), labels=label_comma()) +
  labs(title="Linear Regression Line", x="Education (years)", y="Income (dollars)") +
  theme_minimal() +
  theme(plot.title=element_text(face="bold"),
        axis.title=element_text(face="bold"),
        axis.text=element_text(size=8.5, face="bold"),
        axis.line=element_line(color="gray"),
        panel.grid.minor=element_blank())

summary(lm.fit)

lm.fit$residuals
sigma(lm.fit)

confint(lm.fit, level=0.95)

Prestige.new <- data.frame(education=c(7, 11, 15))
predict(lm.fit, newdata=Prestige.new)

predict(lm.fit, newdata=Prestige.new, interval="confidence")

mean(Prestige$education)
lm(income ~ education, data=Prestige, subset=(education > mean(education)))
lm(income ~ education, data=Prestige, subset=(education <= mean(education)))

## 다항회귀

# [그림 2-6]
windows(width=7.0, height=5.5)
ggplot(Prestige, aes(x=education, y=income)) +
  geom_smooth(method="loess", se=FALSE, color="purple", lwd=1) +
  geom_point(pch=21, col="black", bg="coral", size=2.5) +
  scale_y_continuous(breaks=seq(0, 30000, 5000), limits=c(500, 30000), labels=label_comma()) +
  labs(title="Curved Line", x="Education (years)", y="Income (dollars)") +
  theme_minimal() +
  theme(plot.title=element_text(face="bold"),
        axis.title=element_text(face="bold"),
        axis.text=element_text(size=8.5, face="bold"),
        axis.line=element_line(color="gray"),
        panel.grid.minor=element_blank())

lm.fit <- lm(income ~ education + I(education^2), data=Prestige)

summary(lm.fit)

# [그림 2-7]
windows(width=7.0, height=5.5)
library(dplyr)
ggplot(Prestige, aes(x=education, y=income)) +
  geom_line(mapping=aes(x=x, y=y), 
            data=arrange(data.frame(x=Prestige$education, y=fitted(lm.fit)), x), 
            col="olivedrab", lwd=1) +
  geom_point(pch=21, col="black", bg="coral", size=2.5) +
  scale_y_continuous(breaks=seq(0, 30000, 5000), limits=c(500, 30000), labels=label_comma()) +
  labs(title="Polynomial Regression Line", x="Education (years)", y="Income (dollars)") +
  theme_minimal() +
  theme(plot.title=element_text(face="bold"),
        axis.title=element_text(face="bold"),
        axis.text=element_text(size=8.5, face="bold"),
        axis.line=element_line(color="gray"),
        panel.grid.minor=element_blank())

Prestige.new <- data.frame(education=c(7, 11, 15))
predict(lm.fit, newdata=Prestige.new)

## 다중회귀

lm.fit <- lm(income ~ education + women, data=Prestige)
summary(lm.fit)

library(QuantPsyc)
lm.beta(lm.fit)
lm(scale(income) ~ scale(education) + scale(women), data=Prestige)

library(caret)
varImp(lm.fit)

Prestige.new <- data.frame(education=c(7, 11, 15), women=c(5, 15, 25))
predict(lm.fit, newdata=Prestige.new)

##############################
## 2.3 회귀모델 가정과 진단 ##
##############################

lm.fit <- lm(income ~ education + women, data=Prestige)

# [그림 2-10]
windows(width=7.0, height=7.0)
par(mfrow=c(2, 2))
plot(lm.fit)

library(car)
vif(lm.fit)

##################
## 2.4 더미변수 ##
##################

levels(Prestige$type)
lm.fit <- lm(income ~ education + type, data=Prestige)
lm.fit

contrasts(Prestige$type)

Prestige$type <- relevel(Prestige$type, ref="wc")
levels(Prestige$type)
contrasts(Prestige$type)
lm.refit <- lm(income ~ education + type, data=Prestige)
lm.refit

Prestige.new <- data.frame(education=c(7, 11, 15), type=c("bc", "prof", "wc"))
predict(lm.fit, newdata=Prestige.new)
predict(lm.refit, newdata=Prestige.new)

##################
## 2.5 상호작용 ##
##################

data(Prestige)
lm.fit <- lm(income ~ education + women + education:women, data=Prestige)
lm.fit

lm.fit <- lm(income ~ education*women, data=Prestige)
lm.fit

quantile(Prestige$women)

# [그림 2-13]
windows(width=7.0, height=5.5)
library(ggplot2)
library(scales)
library(sjPlot)
plot_model(lm.fit, type="int", mdrt.values="quart", legend.title="Women %", show.data=TRUE) +
  scale_y_continuous(breaks=seq(0, 30000, 5000), limits=c(500, 30000), labels=label_comma()) +
  labs(title="Predicted Values of Income",
       x="Education (years)", y="Income (dollars)") +
  theme_minimal() +
  theme(plot.title=element_text(face="bold"),
        axis.title=element_text(face="bold"),
        axis.text=element_text(size=8.5, face="bold"),
        axis.line=element_line(color="gray"),
        panel.grid.minor=element_blank())

Prestige.new <- data.frame(education=c(7, 11, 15), women=c(5, 15, 25))
predict(lm.fit, newdata=Prestige.new)

######################
## 2.6 모델비교평가 ##
######################

data(Prestige)
library(caret)
set.seed(123)
lm.fit1 <- train(income ~ education, data=Prestige, method="lm",
                 trControl=trainControl(method="cv", number=10))
class(lm.fit1)
lm.fit1

set.seed(123)
lm.fit2 <- train(income ~ education + I(education^2), data=Prestige, method="lm",
                 trControl=trainControl(method="cv", number=10))
set.seed(123)
lm.fit3 <- train(income ~ education + type, data=Prestige, method="lm",
                 trControl=trainControl(method="cv", number=10), na.action=na.pass)
set.seed(123)
lm.fit4 <- train(income ~ education + women, data=Prestige, method="lm",
                 trControl=trainControl(method="cv", number=10))
set.seed(123)
lm.fit5 <- train(income ~ education + women + education:women, data=Prestige, method="lm",
                 trControl=trainControl(method="cv", number=10))
set.seed(123)
lm.fit6 <- train(income ~ education + women + education:women + prestige + type, 
                 data=Prestige, method="lm", trControl=trainControl(method="cv", number=10), 
                 na.action=na.pass)

summary(resamples(list(model1=lm.fit1, model2=lm.fit2, model3=lm.fit3, 
                       model4=lm.fit4, model5=lm.fit5, model6=lm.fit6)))

##############
## 2.7 사례 ##
##############

## 새크라멘토 주택가격

library(modeldata)
str(Sacramento)

library(rsample)
set.seed(123)
split <- initial_split(data=Sacramento[-c(1, 2)], prop=0.7)
Sacramento.train <- training(split)
Sacramento.test <- testing(split)
dim(Sacramento.train)
dim(Sacramento.test)

library(recipes)
recipe.step <- recipe(price ~ ., data=Sacramento.train) |> 
  step_dummy(all_nominal(), -all_outcomes())
recipe.step

homeprice.train <- recipe.step |>      
  prep(training=Sacramento.train) |> 
  bake(new_data=Sacramento.train)
homeprice.test <- recipe.step |>  
  prep(training=Sacramento.train) |> 
  bake(new_data=Sacramento.test)

lm.fit <- lm(price ~ ., data=homeprice.train)
summary(lm.fit)

lm.pred <- predict(lm.fit, newdata=homeprice.test)
lm.pred

library(caret)
options(scipen=999)
postResample(pred=lm.pred, obs=homeprice.test$price)
options(scipen=0)

RMSE(pred=lm.pred, obs=homeprice.test$price)
R2(pred=lm.pred, obs=homeprice.test$price)
MAE(pred=lm.pred, obs=homeprice.test$price)

library(yardstick)
metrics(data=data.frame(pred=lm.pred, obs=homeprice.test$price), 
        truth=obs, estimate=pred)

set.seed(123)
caret.cv <- train(price ~ ., data=homeprice.train, method="lm",
                  trControl=trainControl(method="cv", number=10))
class(caret.cv)
caret.cv
caret.cv$resample
summary(caret.cv$resample$RMSE)
caret.cv$finalModel

getTrainPerf(caret.cv)

caret.pred <- predict(caret.cv, newdata=homeprice.test, type="raw")
head(caret.pred)
options(scipen=999)
postResample(pred=lm.pred, obs=homeprice.test$price)
options(scipen=0)

varImp(lm.fit)

# [그림 2-14]
windows(width=7.0, height=5.5)
library(vip)
imp <- vip(lm.fit, method="model", geom="col", 
           aesthetics=list(color="salmon", fill="salmon"))
imp
imp$data

library(pdp)
partial(lm.fit, pred.var="sqft", plot=FALSE)
partial(lm.fit, pred.var="type_Residential", plot=FALSE)

# [그림 2-16]
library(ggplot2)
library(scales)
p1 <- partial(lm.fit, pred.var="sqft", plot=FALSE) |> 
  ggplot(aes(x=sqft, y=yhat)) +
  geom_line(color="cornflowerblue", linewidth=1) +
  geom_rug(aes(x=sqft, y=NULL), data=homeprice.train, sides="b") +
  scale_y_continuous(labels=label_dollar()) 
p2 <- partial(lm.fit, pred.var="type_Residential", plot=FALSE) |> 
  mutate(type_Residential=factor(type_Residential)) |> 
  ggplot(aes(x=type_Residential, y=yhat)) +
  geom_boxplot(color="cornflowerblue") +
  scale_y_continuous(labels=label_dollar())
windows(width=7.0, height=3.5)
library(patchwork)
p1 + p2
