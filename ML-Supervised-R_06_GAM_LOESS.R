
###################################
## R을 이용한 머신러닝: 지도학습 ##
## (곽기영, 도서출판 청람)       ## 
###################################

###################################
## 제6장 일반가법모델과 로컬회귀 ##
###################################

######################
## 6.1 일반가법모델 ##
######################

## 스플라인

# [그림 6-1]
x <- seq(from=-1, to=1, length=300)
y1 <- x
y2 <- x^2
y3 <- x^3
y4 <- x^4
data <- data.frame(x=x, y1=y1, y2=y2, y3=y3, y4=y4)
library(ggplot2)
p1 <- ggplot(data, aes(x=x, y=y1)) +
  geom_line(linewidth=1, color="salmon") +
  labs(title="(a) Linea model") +
  theme_bw()
p2 <- ggplot(data, aes(x=x, y=y2)) +
  geom_line(linewidth=1, color="salmon") +
  labs(title="(b) Quadratic model") +
  theme_bw()
p3 <- ggplot(data, aes(x=x, y=y3)) +
  geom_line(linewidth=1, color="salmon") +
  labs(title="(c) Cubic model") +
  theme_bw()
p4 <- ggplot(data, aes(x=x, y=y4)) +
  geom_line(linewidth=1, color="salmon") +
  labs(title="(d) Quartic model") +
  theme_bw()
windows(width=9.0, height=7.0)
library(patchwork)
p1 + p2 + p3 + p4

## 기저함수

# [그림 6-3]
set.seed(123)
x <- seq(from=0, to=2*pi, length=300)
y <- sin(x) + rnorm(length(x), sd=0.4)
data <- data.frame(x, y)
library(mgcv)
gam.fit <- gam(y ~ s(x, bs="cr", k=6), knots=list(x=c(1:6)), data=data)
gam.pred <- predict(gam.fit, newdata=data.frame(x=data$x))
model.mat <- predict(gam.fit, type="lpmatrix")
library(dplyr)
library(tidyr)
model.df <- as_tibble(model.mat) |> 
  mutate(x=x) |> 
  pivot_longer(cols=contains("s"), names_to="basis")
windows(width=7.0, height=5.5)
library(ggplot2)
ggplot(data, aes(x=x, y=y)) +
  geom_point(pch=21, color="black", bg="dimgray", size=1, alpha=0.5) +
  geom_line(aes(y=gam.pred), linewidth=1, color="salmon") +
  geom_line(data=model.df, aes(x=x, y=value, color=basis), linetype="dashed", linewidth=0.7) +
  scale_x_continuous(breaks=seq(0, 6, 1)) +
  scale_color_discrete(name="Basis\nfunction") +
  theme_classic()

## 평활도

# [그림 6-4]
library(MASS)
gam.fit1 <- gam(accel ~ s(times, k=4), data=mcycle)
gam.fit2 <- gam(accel ~ s(times, k=11), data=mcycle)
p1 <- mutate(mcycle, pred1=gam.fit1$fitted.values, pred2=gam.fit2$fitted.values) |> 
  pivot_longer(cols=contains("pred")) |> 
  ggplot(aes(x=times, y=accel)) +
  geom_point(pch=21, color="black", bg="dimgray", size=1, alpha=0.5) +
  geom_line(aes(y=value, color=name), linewidth=1) +
  labs(title="Varying number of basis function", x="x", y="y") +
  scale_color_discrete(name="Number of\nbasis function", labels=c("3", "10")) +
  theme_classic() 
gam.fit3 <- gam(accel ~ s(times, k=11), sp=0.1, data=mcycle)
gam.fit4 <- gam(accel ~ s(times, k=11), sp=0.001, data=mcycle)
p2 <- mutate(mcycle, pred3=gam.fit3$fitted.values, pred4=gam.fit4$fitted.values) |> 
  pivot_longer(cols=contains("pred")) |> 
  ggplot(aes(x=times, y=accel)) +
  geom_point(pch=21, color="black", bg="dimgray", size=1, alpha=0.5) +
  geom_line(aes(y=value, color=name), linewidth=1) +
  labs(title="Varying smoothing parameter (10 basis functions)", x="x", y="y") +
  scale_color_discrete(name="Smoothing\nparameter", labels=c("0.1", "0.001")) +
  theme_classic() 
windows(width=7.0, height=7.0)
library(patchwork)
p1 / p2

##################
## 6.2 로컬회귀 ##
##################

## 추세선 형태

# [그림 6-7]
library(dplyr)
library(tidyr)
library(ggplot2)
set.seed(123)
x <- seq(from=0, to=2*pi, length=300)
y <- sin(x) + rnorm(length(x), sd=0.4)
data <- data.frame(x, y)
loess.fit1 <- loess(y ~ x, data=data, span=0.1)
loess.fit2 <- loess(y ~ x, data=data, span=0.5)
loess.fit3 <- loess(y ~ x, data=data, span=0.9)
p1 <- as_tibble(data) |> 
  mutate(y1=loess.fit1$fitted, y2=loess.fit2$fitted, y3=loess.fit3$fitted) |> 
  pivot_longer(cols=c(y1, y2, y3), names_to="pred") |> 
  ggplot(aes(x=x, y=y)) +
  geom_point(pch=21, color="black", bg="dimgray", size=1, alpha=0.5) +
  geom_line(aes(x=x, y=value, color=pred), linetype="solid", linewidth=1.0) +
  scale_x_continuous(breaks=seq(0, 6, 1)) +
  scale_color_discrete(name="Smoothing\nspan", labels=c("0.1", "0.5", "0.9")) +
  labs(title="Varying smoothing span (degree=2)", x="x", y="y") +
  theme_classic()
loess.fit4 <- loess(y ~ x, data=data, degree=1)
loess.fit5 <- loess(y ~ x, data=data, degree=2)
p2 <- as_tibble(data) |> 
  mutate(y4=loess.fit4$fitted, y5=loess.fit5$fitted) |> 
  pivot_longer(cols=c(y4, y5), names_to="pred") |> 
  ggplot(aes(x=x, y=y)) +
  geom_point(pch=21, color="black", bg="dimgray", size=1, alpha=0.5) +
  geom_line(aes(x=x, y=value, color=pred), linetype="solid", linewidth=1.0) +
  scale_x_continuous(breaks=seq(0, 6, 1)) +
  scale_color_discrete(name="Degree", labels=c("1", "2", "3")) +
  labs(title="Varying degree (span=0.75)", x="x", y="y") +
  theme_classic()
windows(width=7.0, height=7.0)
library(patchwork)
p1 / p2

##############
## 6.3 사례 ##
##############

## 직업소득: GAM

library(car)
str(Prestige)

library(naniar)
miss_var_summary(Prestige)
prestige <- subset(Prestige, subset=complete.cases(Prestige), select=-census)

# [그림 6-8]
windows(width=9.0, height=9.0)
library(GGally)
lowerFun <- function(data, mapping) {
  ggplot(data=data, mapping=mapping)+ 
    geom_point(pch=21, color="blue", bg="cornflowerblue", alpha=0.8) + 
    geom_smooth(method="gam", formula=y ~ s(x), 
                fill="mistyrose", color="salmon", linewidth=1)
}
ggpairs(prestige, 
        diag=list(continuous=wrap("densityDiag", fill="aliceblue")),
        lower=list(continuous=wrap(lowerFun))) +
  theme_bw()

library(caret)
set.seed(123)
index <- createDataPartition(y=prestige$income, p=0.7, list=FALSE)
prestige.train <- prestige[index,]
prestige.test <- prestige[-index,]
dim(prestige.train)
dim(prestige.test)

lm.fit <- lm(income ~ education + women + prestige + type, data=prestige.train)
summary(lm.fit)

# [그림 6-9]
windows(width=7.0, height=5.5)
plot(lm.fit, which=1)

library(mgcv)
gam.fit <- gam(income ~ s(education) + s(women) + s(prestige) + type, 
               data=prestige.train)
class(gam.fit)

?gam

# [그림 6-10]
windows(width=7.0, height=5.5)
plot.gam(gam.fit, pages=1, scheme=1)
plot.gam(gam.fit, pages=1, all.terms=TRUE, scheme=1)

summary(gam.fit)

# [그림 6-11]
windows(width=7.0, height=5.5)
par(mfrow=c(2, 2))
set.seed(123)
gam.check(gam.fit)

set.seed(123)
k.check(gam.fit)

gam.fit2 <- gam(income ~ s(education) + s(women) + s(prestige, k=20) + type, 
                data=prestige.train)
summary(gam.fit2)

set.seed(123)
k.check(gam.fit2)

AIC(lm.fit, gam.fit)

gam.pred <- predict(gam.fit, newdata=prestige.test, type="response")
head(gam.pred)

postResample(pred=gam.pred, obs=prestige.test$income)

gam.fit3 <- gam(income ~ s(education, by=type) + type, data=prestige.train)
summary(gam.fit3)$s.table

# [그림 6-12]
windows(width=7.0, height=5.5)
plot(gam.fit3, pages=1, all.terms=TRUE, scheme=1)

# [그림 6-13]
windows(width=7.0, height=5.5)
vis.gam(gam.fit3, theta=30, phi=30, n.grid=50, lwd=0.4, color="heat")

modelLookup("gam")

hyper.grid <- expand.grid(select=c(TRUE, FALSE), method=c("GCV.Cp", "REML"))
hyper.grid

set.seed(123)
caret.cv <- train(x=subset(prestige.train, select=-income),
                  y=prestige.train$income,
                  method="gam",
                  trControl=trainControl(method="cv", number=10),
                  tuneGrid=hyper.grid)
caret.cv
caret.cv$bestTune
getTrainPerf(caret.cv)

caret.pred <- predict(caret.cv, newdata=prestige.test, type="raw")
head(caret.pred)
postResample(pred=caret.pred, obs=prestige.test$income)

## 직업소득: LOESS

library(car)
str(Prestige)

library(naniar)
miss_var_summary(Prestige)
prestige <- subset(Prestige, subset=complete.cases(Prestige), select=-census)

library(caret)
set.seed(123)
index.train <- createDataPartition(y=prestige$income, p=0.7, list=FALSE)
prestige.train <- prestige[index.train,]
prestige.test <- prestige[-index.train,]
dim(prestige.train)
dim(prestige.test)

loess.fit <- loess(income ~ education + women + prestige, data=prestige.train)
class(loess.fit)
summary(loess.fit)

loess.pred <- predict(loess.fit, newdata=prestige.test)
head(loess.pred)

postResample(pred=loess.pred, obs=prestige.test$income)

## 당뇨병: GAM

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

library(naniar)
miss_var_summary(PimaIndiansDiabetes2)

library(recipes)
recipe.step <- recipe(diabetes ~ ., data=pimad.train) |> 
  step_impute_knn(all_predictors(), neighbors=5) 

pima.train <- recipe.step |>  
  prep(training=pimad.train) |> 
  bake(new_data=pimad.train)
pima.test <- recipe.step |>
  prep(training=pimad.train) |> 
  bake(new_data=pimad.test)

library(mgcv)
gam.fit <- gam(diabetes ~ s(pregnant) + s(glucose) + s(pressure) + s(triceps) +
                 s(insulin) + s(mass) + s(pedigree) + s(age), 
               data=pima.train, family=binomial(), method="REML")

summary(gam.fit)

# [그림 6-14]
windows(width=7.5, height=7.5)
plot(gam.fit, pages=1, col="blue", scheme=1)

gam.pred <- predict(gam.fit, newdata=pima.test)
head(gam.pred)
plogis(head(gam.pred))

gam.pred <- predict(gam.fit, newdata=pima.test, type="response")
head(gam.pred)

pred <- factor(gam.pred >= 0.5, levels=c(FALSE, TRUE), labels=c("neg", "pos"))
head(pred)

table(pima.test$diabetes, pred, dnn=c("Actual", "Predicted"))
mean(pima.test$diabetes==pred)

postResample(pred=pred, obs=pima.test$diabetes)

hyper.grid <- expand.grid(select=c(TRUE, FALSE), method=c("GCV.Cp", "REML"))
hyper.grid

set.seed(123)
caret.cv <- train(x=subset(pima.train, select=-diabetes),
                  y=pima.train$diabetes,
                  method="gam",
                  trControl=trainControl(method="cv", number=10),
                  tuneGrid=hyper.grid)
caret.cv
caret.cv$bestTune
getTrainPerf(caret.cv)

caret.pred <- predict(caret.cv, newdata=pima.test, type="raw")
head(caret.pred)
postResample(pred=caret.pred, obs=pima.test$diabetes)
