
###################################
## R을 이용한 머신러닝: 지도학습 ##
## (곽기영, 도서출판 청람)       ## 
###################################

##################################
## 제5장 다변량적응회귀스플라인 ##
##################################

#######################
## 5.2 회귀계수 추정 ##
#######################

# [그림 5-3]
set.seed(123)  
x <- seq(from=0, to=2*pi, length=300)
y <- sin(x) + rnorm(length(x), sd=0.4)
data <- data.frame(x, y)
head(data)

library(earth)
earth.fit1 <- earth(y ~ x, data=data, nk=3)
earth.fit2 <- earth(y ~ x, data=data, nk=5)
earth.fit3 <- earth(y ~ x, data=data, nk=7)
earth.fit4 <- earth(y ~ x, data=data, nk=9)

library(dplyr)
library(ggplot2)
p1 <- mutate(data, pred=as.vector(earth.fit1$fitted.values)) |>
  ggplot(aes(x=x, y=y)) +
  geom_point(pch=21, color="black", bg="dimgray", size=1, alpha=0.3) +
  geom_line(aes(y=pred), linewidth=1, color="salmon") +
  geom_vline(xintercept=earth.fit1$cuts[,1][-1], color="blue", linetype="dashed") +
  labs(title="(a) One cutpoint") +
  theme_bw()
p2 <- mutate(data, pred=as.vector(earth.fit2$fitted.values)) |>
  ggplot(aes(x=x, y=y)) +
  geom_point(pch=21, color="black", bg="dimgray", size=1, alpha=0.3) +
  geom_line(aes(y=pred), linewidth=1, color="salmon") +
  geom_vline(xintercept=earth.fit2$cuts[,1][-1], color="blue", linetype="dashed") +
  labs(title="(b) Two cutpoints") +
  theme_bw()
p3 <- mutate(data, pred=as.vector(earth.fit3$fitted.values)) |>
  ggplot(aes(x=x, y=y)) +
  geom_point(pch=21, color="black", bg="dimgray", size=1, alpha=0.3) +
  geom_line(aes(y=pred), linewidth=1, color="salmon") +
  geom_vline(xintercept=earth.fit3$cuts[,1][-1], color="blue", linetype="dashed") +
  labs(title="(c) Three cutpoints") +
  theme_bw()
p4 <- mutate(data, pred=as.vector(earth.fit4$fitted.values)) |>
  ggplot(aes(x=x, y=y)) +
  geom_point(pch=21, color="black", bg="dimgray", size=1, alpha=0.3) +
  geom_line(aes(y=pred), linewidth=1, color="salmon") +
  geom_vline(xintercept=earth.fit4$cuts[,1][-1], color="blue", linetype="dashed") +
  labs(title="(d) Four cutpoints") +
  theme_bw()
windows(width=9.0, height=7.0)
library(patchwork)
p1 + p2 + p3 + p4

earth.fit1$cuts

earth.fit1$coefficients

coef(earth.fit1)

data$x[148]
earth.fit1$fitted.values[148]
-0.9612639+0.4082149*(5.19046-data$x[148]) #3.08905766: <5.19046

data$x[299]
earth.fit1$fitted.values[299]
-0.9612639+1.0283506*(data$x[299]-5.19046) #6.26217131: >5.19046

earth.fit2$coefficients

data$x[2]
earth.fit2$fitted.values[2]
3.9800966-0.7492338*(5.19046-data$x[2]) #0.02101400: <5.19046 (<1.40794)

data$x[148]
earth.fit2$fitted.values[148]
3.9800966-0.7492338*(5.19046-data$x[148])-1.3917614*(data$x[148]-1.40794) #3.08905766: >1.40794 & <5.19046

data$x[299]
earth.fit2$fitted.values[299]
3.9800966-1.3917614*(data$x[299]-1.40794)+2.8592808*(data$x[299]-5.19046) #6.26217131: >5.19046 ((>1.40794))

##############
## 5.3 사례 ##
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

library(earth)
earth.fit <- earth(Sale_Price ~ ., data=homeprice.train)
class(earth.fit)

summary(earth.fit)

coef(earth.fit)

# [그림 5-4]
windows(width=7.0, height=5.5)
plot(earth.fit)
plot(earth.fit, which=1)

?plot.earth

earth.fit2 <- earth(Sale_Price ~ ., data=homeprice.train, degree=2)
earth.fit2
coef(earth.fit2)

earth.pred <- predict(earth.fit, newdata=homeprice.test, type="response")
head(earth.pred)

library(caret)
postResample(pred=earth.pred, obs=homeprice.test$Sale_Price)

modelLookup("earth")

hyper.grid <- expand.grid(degree=1:3, nprune=seq(from=5, to=60, by=5))
hyper.grid

set.seed(123)
caret.cv <- train(x=subset(homeprice.train, select=-Sale_Price),
                  y=homeprice.train$Sale_Price,
                  method="earth",
                  trControl=trainControl(method="cv", number=10),
                  tuneGrid=hyper.grid)
caret.cv

caret.cv$bestTune
getTrainPerf(caret.cv)

# [그림 5-5]
windows(width=7.0, height=5.5)
ggplot(caret.cv) +
  theme_bw()

caret.pred <- predict(caret.cv, newdata=homeprice.test, type="raw")
head(caret.pred)
postResample(pred=caret.pred, obs=homeprice.test$Sale_Price)

earth.fit <- earth(Sale_Price ~ ., data=homeprice.train, 
                   degree=caret.cv$bestTune$degree, nprune=caret.cv$bestTune$nprune)
earth.fit
coef(earth.fit)

earth.pred <- predict(earth.fit, newdata=homeprice.test, type="response")
head(earth.pred)
postResample(pred=earth.pred, obs=homeprice.test$Sale_Price)

# [그림 5-6]
windows(width=7.0, height=5.5)
library(vip)
imp <- vip(earth.fit, method="model", geom="point", num_features=20, type="gcv",
           aesthetics=list(color="navy", fill="navy", size=2))
imp
imp$data

vi(earth.fit, method="model", type="gcv") |> 
  print(n=40)

# [그림 5-7]
library(pdp)
library(scales)
library(ggplotify)
p1 <- partial(earth.fit, pred.var="Gr_Liv_Area") |> 
  ggplot(aes(x=Gr_Liv_Area, y=yhat)) +
  geom_line(color="purple4", linewidth=1) +
  scale_y_continuous(limits=c(0, 600000), labels=label_dollar())
p2 <- partial(earth.fit, pred.var="Year_Built") |> 
  ggplot(aes(x=Year_Built, y=yhat)) +
  geom_line(color="purple4", linewidth=1) +
  scale_y_continuous(limits=c(0, 600000), labels=label_dollar())
p3 <- partial(earth.fit, pred.var=c("Gr_Liv_Area", "Year_Built")) |>
  plotPartial() |> 
  as.grob()
windows(width=7.0, height=7.0)
library(patchwork)
(p1 + p2) / p3 +
  plot_layout(heights=c(1.5, 2))

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

library(recipes)
recipe.step <- recipe(Attrition ~ ., data=attrition.train) |>
  step_dummy(all_nominal(), -all_outcomes())
attr.train <- recipe.step |>  
  prep(training=attrition.train) |> 
  bake(new_data=attrition.train)
attr.test <- recipe.step |>  
  prep(training=attrition.train) |> 
  bake(new_data=attrition.test)

library(earth)
earth.fit <- earth(Attrition ~ ., data=attr.train, glm=list(family=binomial))
summary(earth.fit)

earth.pred <- predict(earth.fit, newdata=attr.test, type="class")
head(earth.pred)
table(attr.test$Attrition, factor(earth.pred), dnn=c("Actual", "Predicted"))
mean(attr.test$Attrition==factor(earth.pred))

hyper.grid <- expand.grid(degree=1:3, nprune=seq(from=5, to=60, by=5))
set.seed(123)
caret.cv <- train(x=subset(attr.train, select=-Attrition),
                  y=attr.train$Attrition,
                  method="earth",
                  trControl=trainControl(method="cv", number=10),
                  tuneGrid=hyper.grid)
caret.cv$bestTune
getTrainPerf(caret.cv)

# [그림 5-8]
windows(width=7.0, height=5.5)
ggplot(caret.cv) +
  theme_bw()

caret.pred <- predict(caret.cv, newdata=attr.test, type="raw")
head(caret.pred)
postResample(pred=caret.pred, obs=attr.test$Attrition)

earth.fit <- earth(Attrition ~ ., data=attr.train, glm=list(family=binomial),
                   degree=caret.cv$bestTune$degree, nprune=caret.cv$bestTune$nprune)
earth.fit
coef(earth.fit)

earth.pred <- predict(earth.fit, newdata=attr.test, type="response")
head(earth.pred)
pred <- factor(ifelse(earth.pred >= 0.5, "Yes", "No"))
postResample(pred=pred, obs=attr.test$Attrition)

# [그림 5-9]
windows(width=7.0, height=5.5)
library(vip)
imp <- vip(earth.fit, method="model", geom="point", num_features=10, type="gcv",
           aesthetics=list(color="firebrick", fill="firebrick", size=2))
imp
imp$data

# [그림 5-10]
library(pdp)
p1 <- partial(earth.fit, pred.var="NumCompaniesWorked", prob=TRUE) |> 
  ggplot(aes(x=NumCompaniesWorked, y=yhat)) +
  geom_line(color="darkorange", linewidth=1) +
  ylim(c(0, 1))
p2 <- partial(earth.fit, pred.var="OverTime_Yes", prob=TRUE) |>
  mutate(OverTime_Yes=factor(OverTime_Yes)) |>
  ggplot(aes(x=OverTime_Yes, y=yhat)) +
  geom_boxplot(color="darkorange") +
  ylim(c(0, 1))
windows(width=7.0, height=3.5)
library(patchwork)
p1 + p2
