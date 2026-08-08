
###################################
## R을 이용한 머신러닝: 지도학습 ##
## (곽기영, 도서출판 청람)       ## 
###################################

#######################
## 제17장 인공신경망 ##
#######################

###############
## 17.5 사례 ##
###############

## 보스턴 주택가격

library(MASS)
str(Boston)

library(caret)
set.seed(123)
index <- createDataPartition(y=Boston$medv, p=0.7, list=FALSE)
Boston.train <- Boston[index,]
Boston.test <- Boston[-index,]
dim(Boston.train)
dim(Boston.test)

mins.train <- apply(Boston.train, 2, min)
mins.train
maxs.train <- apply(Boston.train, 2, max)
maxs.train
Boston.train <- as.data.frame(scale(x=Boston.train, center=mins.train, 
                                    scale=maxs.train-mins.train))
Boston.test <- as.data.frame(scale(x=Boston.test, center=mins.train, 
                                   scale=maxs.train-mins.train))

head(Boston.train, 3)
head(Boston.test, 3)
apply(Boston.train, 2, range)
apply(Boston.test, 2, range)

set.seed(123)
index <- createDataPartition(y=Boston$medv, p=0.7, list=FALSE)
Boston.train <- Boston[index,]
Boston.test <- Boston[-index,]
prePro <- preProcess(Boston.train, method="range")
prePro$ranges
Boston.train <- predict(prePro, newdata=Boston.train)
Boston.test <- predict(prePro, newdata=Boston.test)

library(neuralnet)
set.seed(123)
nn.fit <- neuralnet(medv ~ ., data=Boston.train, hidden=c(5, 3))
class(nn.fit)

nn.fit$result.matrix
nn.fit$weights

# [그림 17-11]
plot(nn.fit, col.entry="royalblue", col.entry.synapse="royalblue",
     col.hidden="green3", col.hidden.synapse="green3",
     col.out="red", col.out.synapse="red", col.intercept="dimgray")

set.seed(123)
nn.rep <- neuralnet(medv ~ ., data=Boston.train, hidden=1, rep=3,
                    algorithm="backprop", learningrate=0.001)
nn.rep$weights

# [그림 17-12]
plot(nn.rep, rep="best")

nn.pred <- predict(nn.fit, newdata=Boston.test)
head(nn.pred)

cor(Boston.test$medv, nn.pred)

nn.predv <- nn.pred*(maxs.train["medv"]-mins.train["medv"])+mins.train["medv"]
head(nn.predv)

prePro$ranges[2, "medv"]
prePro$ranges[1, "medv"]
nn.predv <- nn.pred*(prePro$ranges[2, "medv"]-prePro$ranges[1, "medv"])+prePro$ranges[1, "medv"]
head(nn.predv)

# [그림 17-13]
windows(width=7.0, height=5.5)
library(ggplot2)
ggplot(data.frame(pred=nn.predv, actual=Boston[-index,]$medv), aes(x=pred, y=actual)) +
  geom_abline(intercept=0, slope=1, color="purple", linewidth=1) +
  geom_point(pch=21, color="black", bg="salmon", size=2) +
  labs(x="Predicted Value", y="Actual Value") +
  theme_classic() 

mean((Boston[-index,]$medv-nn.predv)^2)

lm.fit <- lm(medv ~ ., data=Boston.train)
lm.pred <- predict(lm.fit, newdata=Boston.test)
lm.predv <- lm.pred*(maxs.train["medv"]-mins.train["medv"])+mins.train["medv"]
head(lm.predv)
mean((Boston[-index,]$medv-lm.predv)^2)

nnRuns <- function(fraction, h, run) {
  results <- numeric()
  for (i in 1:run) {
    index <- createDataPartition(y=Boston$medv, p=fraction, list=FALSE)
    Boston.train <- Boston[index,]
    Boston.test <- Boston[-index,]
    prePro <- preProcess(Boston.train, method="range")
    Boston.train <- predict(prePro, newdata=Boston.train)
    Boston.test <- predict(prePro, newdata=Boston.test)
    nn.fit <- neuralnet(medv ~ ., data=Boston.train, hidden=h)
    nn.pred <- predict(nn.fit, newdata=Boston.test)
    nn.predv <- nn.pred*(prePro$ranges[2, "medv"]-prePro$ranges[1, "medv"])+prePro$ranges[1, "medv"]
    results[i] <- mean((Boston[-index,]$medv-nn.predv)^2)
  }
  return(results)
}

nn.cv <- nnRuns(fraction=0.7, h=c(5, 3), run=50)
nn.cv
summary(nn.cv)

# [그림 17-14]
windows(width=7.0, height=4.0)
ggplot(data.frame(mse=nn.cv), aes(x="", y=mse)) +
  geom_boxplot(fill="mistyrose", color="darkslategray", width=0.3) +
  geom_point(position="jitter", pch=21, color="darkred", fill="salmon") +
  labs(title="MSE for Boston Housing Values with 50 Samples",
       y="MSE") +
  coord_flip() +
  theme(axis.title.y=element_blank(),
        axis.text.y=element_blank(),
        axis.ticks=element_blank())

## 펭귄

library(modeldata)
str(penguins)
levels(penguins$species)

library(caret)
set.seed(123)
index <- createDataPartition(y=penguins$species, p=0.7, list=FALSE)
penguins.train <- penguins[index,]
penguins.test <- penguins[-index,]
dim(penguins.train)
dim(penguins.test)

library(naniar)
miss_var_summary(penguins)

library(recipes)
recipe.step <- recipe(species ~ ., data=penguins.train) |>
  step_impute_knn(all_predictors(), neighbors=5) |>
  step_range(all_numeric()) |> 
  step_dummy(all_nominal(), one_hot=TRUE)

peng.train <- recipe.step |>  
  prep(training=penguins.train) |> 
  bake(new_data=penguins.train)
peng.test <- recipe.step |>  
  prep(training=penguins.train) |> 
  bake(new_data=penguins.test)

str(peng.train)
str(peng.test)
anyNA(peng.train)
anyNA(peng.test)

library(neuralnet)
set.seed(123)
nn.fit <- neuralnet(species_Adelie+species_Chinstrap+species_Gentoo ~ ., 
                    data=peng.train, hidden=3, err.fct="ce", 
                    linear.output=FALSE)
nn.fit$result.matrix
nn.fit$weights

# [그림 17-15]
plot(nn.fit, col.entry="royalblue", col.entry.synapse="royalblue",
     col.hidden="green3", col.hidden.synapse="green3",
     col.out="red", col.out.synapse="red", col.intercept="dimgray")

nn.pred <- predict(nn.fit, newdata=peng.test)
head(nn.pred)

nn.predv <- levels(penguins$species)[max.col(nn.pred)]
head(nn.predv)

table(penguins.test$species, nn.predv, dnn=c("Actual", "Predicted"))
mean(penguins.test$species==nn.predv)

nnRuns <- function(fraction, h, run) {
  results <- numeric()
  for (i in 1:run) {
    index <- createDataPartition(y=penguins$species, p=fraction, list=FALSE)
    penguins.train <- penguins[index,]
    penguins.test <- penguins[-index,]
    recipe.step <- recipe(species ~ ., data=penguins.train) |>
      step_impute_knn(all_predictors(), neighbors=5) |>
      step_range(all_numeric_predictors()) |> 
      step_dummy(all_nominal_predictors(), one_hot=TRUE)
    peng.train <- recipe.step |>  
      prep(training=penguins.train) |> 
      bake(new_data=penguins.train)
    peng.test <- recipe.step |>  
      prep(training=penguins.train) |> 
      bake(new_data=penguins.test)
    nn.fit <- neuralnet(species ~ ., 
                        data=peng.train, hidden=h, err.fct="ce", 
                        linear.output=FALSE)
    nn.pred <- predict(nn.fit, newdata=peng.test)
    nn.predv <- levels(penguins$species)[max.col(nn.pred)]
    results[i] <- mean(penguins.test$species==nn.predv)
  }
  return(results)
}

nn.cv <- nnRuns(fraction=0.7, h=3, run=50)
nn.cv
summary(nn.cv)
