
###################################
## R을 이용한 머신러닝: 지도학습 ##
## (곽기영, 도서출판 청람)       ## 
###################################

####################
## 제15장 XGBoost ##
####################

###############
## 15.4 사례 ##
###############

## 에임스 주택가격

library(modeldata)
str(ames)

library(caret)
set.seed(123)
index <- createDataPartition(ames$Sale_Price, p=0.7, list=FALSE)
ames.train <- ames[index,]
ames.test <- ames[-index,]
dim(ames.train)
dim(ames.test)

library(recipes)
recipe.step <- recipe(Sale_Price ~ ., data=ames.train)|> 
  step_integer(all_nominal())
homeprice.train <- recipe.step|>  
  prep(training=ames.train)|> 
  bake(new_data=ames.train)
homeprice.test <- recipe.step|>  
  prep(training=ames.train)|> 
  bake(new_data=ames.test)
dim(homeprice.train)
dim(homeprice.test)

x.train <- as.matrix(homeprice.train[setdiff(names(homeprice.train), "Sale_Price")])
y.train <- homeprice.train$Sale_Price
x.test <- as.matrix(homeprice.test[setdiff(names(homeprice.test), "Sale_Price")])
y.test <- homeprice.test$Sale_Price

library(xgboost)
dtrain <- xgb.DMatrix(data=x.train, label=y.train) 
dtest <- xgb.DMatrix(data=x.test, label=y.test)
class(dtrain)
class(dtest)

set.seed(123)
xgb.fit1 <- xgb.cv(params=list(objective="reg:squarederror"), data=dtrain,  
                   nrounds=1000, nfold=5, verbose=0)
class(xgb.fit1)

xgb.fit1$evaluation_log

summarise(xgb.fit1$evaluation_log, 
          ntrees.train=which.min(train_rmse_mean),
          rmse.train=min(train_rmse_mean),
          ntrees.test=which.min(test_rmse_mean),
          rmse.test=min(test_rmse_mean))

# [그림 15-27]
windows(width=7.0, height=5.5)
library(ggplot2)
library(scales)
ggplot(xgb.fit1$evaluation_log) +
  geom_line(aes(x=iter, y=train_rmse_mean, color="train"), linewidth=1) +
  geom_vline(xintercept=which.min(xgb.fit1$evaluation_log$train_rmse_mean), 
             color="cornflowerblue", linetype="dashed") +
  geom_line(aes(x=iter, y=test_rmse_mean, color="test"), linewidth=1) +
  geom_vline(xintercept=which.min(xgb.fit1$evaluation_log$test_rmse_mean), 
             color="tomato", linetype="dashed") +
  scale_color_manual(name="", breaks=c("train", "test"),
                     values=c("cornflowerblue", "tomato")) +
  scale_y_continuous(labels=label_dollar(prefix="")) +
  labs(x="Iteration", y="RMSE") +
  theme_classic() +
  theme(legend.position="bottom",
        legend.title=element_blank())

set.seed(123)
xgb.fit2 <- xgb.cv(params=list(objective="reg:squarederror"), data=dtrain, 
                   nrounds=1000, nfold=5, verbose=0, early_stopping_rounds=10)
xgb.fit2$evaluation_log
summarise(xgb.fit2$evaluation_log, 
          ntrees.train=which.min(train_rmse_mean),
          rmse.train=min(train_rmse_mean),
          ntrees.test=which.min(test_rmse_mean),
          rmse.test=min(test_rmse_mean))

# [그림 15-28]
windows(width=7.0, height=5.5)
ggplot(xgb.fit2$evaluation_log) +
  geom_line(aes(x=iter, y=train_rmse_mean, color="train"), linewidth=1) +
  geom_vline(xintercept=which.min(xgb.fit2$evaluation_log$train_rmse_mean), 
             color="cornflowerblue", linetype="dashed") +
  geom_line(aes(x=iter, y=test_rmse_mean, color="test"), linewidth=1) +
  geom_vline(xintercept=which.min(xgb.fit2$evaluation_log$test_rmse_mean), 
             color="tomato", linetype="dashed") +
  scale_color_manual(name="", breaks=c("train", "test"),
                     values=c("cornflowerblue", "tomato")) +
  scale_y_continuous(labels=label_dollar(prefix="")) +
  labs(x="Iteration", y="RMSE") +
  theme_classic() +
  theme(legend.position="bottom",
        legend.title=element_blank())

params <- list(objective="reg:squarederror", eta=0.1, max_depth=5, 
               min_child_weight=2, subsample=0.7, colsample_bytree=0.9)
set.seed(123)
xgb.fit3 <- xgb.cv(params=params, data=dtrain,  
                   nrounds=1000, nfold=5, verbose=0, early_stopping_rounds=10)
summarise(xgb.fit3$evaluation_log, 
          ntrees.train=which.min(train_rmse_mean),
          rmse.train=min(train_rmse_mean),
          ntrees.test=which.min(test_rmse_mean),
          rmse.test=min(test_rmse_mean))

hyper.grid <- expand.grid(
  eta=c(0.1, 0.3),
  max_depth=c(3, 6, 9),
  min_child_weight=c(1, 3),
  subsample=c(0.5, 1), 
  colsample_bytree=c(0.5, 1),
  optimal_trees=0,               
  min_RMSE=0
  )
dim(hyper.grid)
hyper.grid

for (i in 1:nrow(hyper.grid)) {
  params <- list(objective="reg:squarederror", 
                 eta=hyper.grid$eta[i], 
                 max_depth=hyper.grid$max_depth[i], 
                 min_child_weight=hyper.grid$min_child_weight[i], 
                 subsample=hyper.grid$subsample[i], 
                 colsample_bytree=hyper.grid$colsample_bytree[i])
  set.seed(123)
  xgb.models <- xgb.cv(params=params, data=dtrain,  
                       nrounds=1000, nfold=5, verbose=0, early_stopping_rounds=10)
  hyper.grid$optimal_trees[i] <- which.min(xgb.models$evaluation_log$test_rmse_mean)
  hyper.grid$min_RMSE[i] <- min(xgb.models$evaluation_log$test_rmse_mean)
}

slice_min(hyper.grid, order_by=min_RMSE, n=5)

library(foreach)
library(parallel)
library(doParallel)
cl <- makeCluster(detectCores())
registerDoParallel(cl)
hyper.grid <- expand.grid(
  eta=c(0.1, 0.3),
  max_depth=c(3, 6, 9),
  min_child_weight=c(1, 3),
  subsample=c(0.5, 1), 
  colsample_bytree=c(0.5, 1)
  )
xgb.rmse <- foreach(
  i=1:nrow(hyper.grid), 
  .packages="xgboost", 
  .combine=rbind
) %dopar% {
  params <- list(objective="reg:squarederror", 
                 eta=hyper.grid$eta[i], 
                 max_depth=hyper.grid$max_depth[i], 
                 min_child_weight=hyper.grid$min_child_weight[i], 
                 subsample=hyper.grid$subsample[i], 
                 colsample_bytree=hyper.grid$colsample_bytree[i])
  dtrain <- xgb.DMatrix(data=x.train, label=y.train) 
  set.seed(123)
  xgb.models <- xgb.cv(params=params, data=dtrain, 
                       nrounds=1000, nfold=5, verbose=0, early_stopping_rounds=10)
  optimal_trees <- which.min(xgb.models$evaluation_log$test_rmse_mean)
  min_RMSE <- min(xgb.models$evaluation_log$test_rmse_mean)
  data.frame(optimal_trees=optimal_trees, min_RMSE=min_RMSE)
}
hyper.grid <- cbind(hyper.grid, xgb.rmse)
slice_min(hyper.grid, order_by=min_RMSE, n=5)
stopImplicitCluster()

set.seed(123)
xgb.fit4 <- xgboost(x=x.train, y=y.train, objective="reg:squarederror", 
                    learning_rate=0.1, max_depth=3, min_child_weight=1, 
                    subsample=1, colsample_bytree=1, nrounds=352, 
                    verbosity=0, monitor_training=TRUE)
class(xgb.fit4)

attributes(xgb.fit4)$evaluation_log

attr(xgb.fit4, "evaluation_log")
summarise(attr(xgb.fit4, "evaluation_log"), 
          ntrees.train=which.min(train_rmse),
          rmse.train=min(train_rmse))

# [그림 15-29]
install.packages("DiagrammeR")
xgb.plot.tree(model=xgb.fit4, tree_idx=1)

# [그림 15-30]
xgb.plot.multi.trees(model=xgb.fit4, features_keep=3)

imp <- xgb.importance(model=xgb.fit4)
imp
colSums(imp[,c("Gain", "Cover", "Frequency")])

# [그림 15-31]
windows(width=7.0, height=5.5)
install.packages("Ckmeans.1d.dp")
xgb.ggplot.importance(imp, top_n=10, measure="Gain")

xgb.plot.importance(xgb.importance(model=xgb.fit4), top_n=10, measure="Gain", xlim=c(0, 0.25))

# [그림 15-32]
windows(width=7.0, height=5.5)
library(vip)
imp <- vip(xgb.fit4, num_features=10, method="model", geom="col",
           aesthetics=list(color="palegreen4", fill="olivedrab4"))
imp
imp$data

predPDP <- function(object, newdata) {
  results <- mean(predict(object, newdata=newdata))
  return(results)
  }
predPDP(object=xgb.fit4, newdata=x.train)

# [그림 15-33]
library(pdp)
library(scales)
p1 <- partial(xgb.fit4, pred.var="Gr_Liv_Area", pred.fun=predPDP, train=x.train) |> 
  ggplot(aes(x=Gr_Liv_Area, y=yhat)) +
  geom_line(color="orangered", linewidth=1) +
  scale_y_continuous(labels=label_dollar())
p2 <- partial(xgb.fit4, pred.var="Overall_Cond", pred.fun=predPDP, train=x.train) |> 
  mutate(Overall_Cond=factor(Overall_Cond, levels=1:length(levels(ames$Overall_Cond)), 
                        labels=levels(ames$Overall_Cond))) |> 
  ggplot(aes(x=Overall_Cond, y=yhat)) +
  geom_boxplot(color="orangered") +
  scale_y_continuous(labels=label_dollar())
windows(width=8.0, height=7.0)
library(patchwork)
p1 / p2

?predict.xgboost
?predict.xgb.Booster

xgb.pred <- predict(xgb.fit4, newdata=x.test)
head(xgb.pred)

sqrt(mean((y.test - xgb.pred)^2))
postResample(xgb.pred, y.test)

xgb.shapley <- predict(xgb.fit4, newdata=x.train, predcontrib=TRUE)
dim(xgb.shapley)
head(xgb.shapley)

head(rowSums(predict(xgb.fit4, newdata=x.train, type="contrib")))
head(predict(xgb.fit4, newdata=x.train))

library(tidyr)
feature.value <- x.train |> 
  as.data.frame() |> 
  pivot_longer(cols=everything(), names_to="feature", values_to="feature_value") |> 
  pull(feature_value)
head(feature.value)

shapley <- xgb.shapley |> 
  as.data.frame() |> 
  select(-"(Intercept)") |>
  pivot_longer(cols=everything(), names_to="feature", values_to="shapley_value") |> 
  mutate(feature_value=feature.value) |> 
  group_by(feature) |> 
  mutate(shapley_imp=mean(abs(shapley_value)))
shapley  

shapley10.feature <- shapley |> 
  ungroup() |> 
  distinct(feature, .keep_all=TRUE) |> 
  slice_max(order_by=shapley_imp, n=10) |> 
  pull(feature) 
shapley10.feature

shapley10 <- filter(shapley, feature %in% shapley10.feature)
shapley10

# [그림 15-34]
library(ggbeeswarm)
p1 <- ggplot(shapley10, aes(x=shapley_value, y=reorder(feature, shapley_imp))) +
  geom_quasirandom(varwidth=TRUE, size=0.4, alpha=0.3, color="tomato") +
  scale_x_continuous(limits=c(-50000, 150000), 
                     breaks=seq(-50000, 100000, 50000), 
                     labels=c("-50000", "0", "50000", "100000")) +
  labs(x="Shapley Value", y=NULL)
p2 <- shapley10 |> 
  select(feature, shapley_imp) |> 
  filter(row_number()==1) |> 
  ggplot(aes(x=shapley_imp, y=reorder(feature, shapley_imp))) +
  geom_col(color="cornflowerblue", fill="cornflowerblue") +
  labs(x="Shapley Importance", y=NULL)
windows(width=7.0, height=5.5)
library(patchwork)
p1 + p2

# [그림 15-35]
windows(width=7.0, height=3.5)
library(scales)
shapley |> 
  filter(feature %in% c("Gr_Liv_Area", "Overall_Cond")) |> 
  ggplot(aes(x=feature_value, y=shapley_value)) +
  geom_point(aes(color=shapley_value)) +
  scale_color_viridis_c(name="Shapley Value", option="C") +
  facet_wrap(~feature, scales="free") +
  scale_y_continuous(labels=comma) +
  labs(x="Feature Value", y="Shapley Value")

library(foreach)
library(parallel)
library(doParallel)
cl <- makeCluster(detectCores())
registerDoParallel(cl)
hyper.grid <- expand.grid(
  eta=0.1, 
  max_depth=3, 
  min_child_weight=1, 
  subsample=1, 
  colsample_bytree=1,
  gamma=seq(0, 10, 2),
  lambda=seq(0, 1, 0.2),
  alpha=seq(0, 1, 0.2)
  )
dim(hyper.grid)
xgb.rmse <- foreach(
  i=1:nrow(hyper.grid), 
  .packages="xgboost", 
  .combine=rbind
) %dopar% {
  params <- list(objective="reg:squarederror",
                 eta=hyper.grid$eta[i], 
                 max_depth=hyper.grid$max_depth[i], 
                 min_child_weight=hyper.grid$min_child_weight[i], 
                 subsample=hyper.grid$subsample[i], 
                 colsample_bytree=hyper.grid$colsample_bytree[i],
                 gamma=hyper.grid$gamma[i],
                 lambda=hyper.grid$lambda[i],
                 alpha=hyper.grid$alpha[i])
  dtrain <- xgb.DMatrix(data=x.train, label=y.train) 
  set.seed(123)
  xgb.models <- xgb.cv(params=params, data=dtrain, 
                       nrounds=1000, nfold=5, verbose=0, early_stopping_rounds=10)
  optimal_trees <- which.min(xgb.models$evaluation_log$test_rmse_mean)
  min_RMSE <- min(xgb.models$evaluation_log$test_rmse_mean)
  data.frame(optimal_trees=optimal_trees, min_RMSE=min_RMSE)
}
hyper.grid <- cbind(hyper.grid, xgb.rmse)
stopImplicitCluster()

slice_min(hyper.grid, order_by=min_RMSE, n=5)

set.seed(123)
xgb.fit5 <- xgboost(x=x.train, y=y.train, objective="reg:squarederror", 
                    learning_rate=0.1, max_depth=3, min_child_weight=1, subsample=1, 
                    colsample_bytree=1, min_split_loss=0, reg_lambda=0, reg_alpha=1, 
                    nrounds=393, verbosity=0, monitor_training=TRUE)
summarise(attributes(xgb.fit5)$evaluation_log, 
          ntrees.train=which.min(train_rmse),
          rmse.train=min(train_rmse))
xgb.pred <- predict(xgb.fit5, newdata=x.test)
sqrt(mean((y.test - xgb.pred)^2))

params <- list(objective="reg:squarederror", eta=0.1, max_depth=3, 
               min_child_weight=1, subsample=1, colsample_bytree=1)
set.seed(123)
xgb.fit6 <- xgb.train(params=params, data=dtrain, nrounds=352,
                      evals=list(train=dtrain, val=dtest), 
                      verbose=1, print_every_n=100)

summarise(attributes(xgb.fit6)$evaluation_log, 
          ntrees.train=which.min(train_rmse),
          rmse.train=min(train_rmse),
          ntrees.val=which.min(val_rmse),
          rmse.val=min(val_rmse))

ptrain <- predict(xgb.fit6, newdata=dtrain, outputmargin=TRUE)
ptest <- predict(xgb.fit6, newdata=dtest, outputmargin=TRUE)

setinfo(dtrain, name="base_margin", ptrain)
setinfo(dtest, name="base_margin", ptest)

set.seed(123)
xgb.fit7 <- xgb.train(params, data=dtrain, nrounds=5, 
                      evals=list(train=dtrain, val=dtest))

## 이직

library(modeldata)
str(attrition)
levels(attrition$Attrition)

library(caret)
set.seed(123)
index <- createDataPartition(attrition$Attrition, p=0.7, list=FALSE)
attrition.train <- attrition[index,]
attrition.test <- attrition[-index,]
dim(attrition.train)
dim(attrition.test)

library(recipes)
recipe.step <- recipe(Attrition ~ ., data=attrition.train) |> 
  step_integer(all_nominal())
attr.train <- recipe.step |>  
  prep(training=attrition.train) |> 
  bake(new_data=attrition.train)
attr.test <- recipe.step |>  
  prep(training=attrition.train) |> 
  bake(new_data=attrition.test)
dim(attr.train)
dim(attr.test)
x.train <- as.matrix(attr.train[setdiff(names(attr.train), "Attrition")])
y.train <- attr.train$Attrition - 1
x.test <- as.matrix(attr.test[setdiff(names(attr.test), "Attrition")])
y.test <- attr.test$Attrition - 1

library(xgboost)
dtrain <- xgb.DMatrix(data=x.train, label=y.train) 
dtest <- xgb.DMatrix(data=x.test, label=y.test)

params <- list(objective="binary:logistic", eval_metric="error",
               eta=0.3, max_depth=6, min_child_weight=1,
               subsample=1, colsample_bytree=1, gamma=0)
set.seed(123)
xgb.fit1 <- xgb.cv(params=params, data=dtrain, 
                   nrounds=1000, nfold=5, verbose=0, early_stopping_rounds=10)

xgb.fit1$evaluation_log
summarise(xgb.fit1$evaluation_log, 
          ntrees.train=which.min(train_error_mean),
          error.train=min(train_error_mean),
          ntrees.test=which.min(test_error_mean),
          error.test=min(test_error_mean))

# [그림 15-36]
windows(width=7.0, height=5.5)
library(ggplot2)
ggplot(xgb.fit1$evaluation_log) +
  geom_line(aes(x=iter, y=train_error_mean, color="train"), linewidth=1) +
  geom_vline(xintercept=which.min(xgb.fit1$evaluation_log$train_error_mean), 
             color="cornflowerblue", linetype="dashed") +
  geom_line(aes(x=iter, y=test_error_mean, color="test"), linewidth=1) +
  geom_vline(xintercept=which.min(xgb.fit1$evaluation_log$test_error_mean), 
             color="tomato", linetype="dashed") +
  scale_color_manual(name="", breaks=c("train", "test"),
                     values=c("cornflowerblue", "tomato")) +
  labs(x="Iteration", y="Error") +
  theme_classic() +
  theme(legend.position="bottom",
        legend.title=element_blank())

set.seed(123)
xgb.fit2 <- xgb.train(params=params, data=dtrain, nrounds=11, 
                      evals=list(train=dtrain, val=dtest), print_every_n=5)

summarise(attributes(xgb.fit2)$evaluation_log, 
          ntrees.train=which.min(train_error),
          error.train=min(train_error),
          ntrees.test=which.min(val_error),
          error.test=min(val_error))

xgb.pred <- predict(xgb.fit2, newdata=dtest, iterationrange=c(1, 9))
head(xgb.pred)
pred <- factor(ifelse(xgb.pred >= 0.5, 1, 0), 
               levels=c(0, 1), labels=c("No", "Yes"))
head(pred)
table(attrition.test$Attrition, pred, dnn=c("Actual", "Predicted"))
mean(attrition.test$Attrition==pred)
postResample(pred, attrition.test$Attrition)

imp <- xgb.importance(model=xgb.fit2)
imp

modelLookup("xgbTree")

hyper.grid <- expand.grid(
  nrounds=1000,
  eta=c(0.1, 0.3),
  max_depth=c(1, 3),
  min_child_weight=c(0, 1),
  subsample=1, 
  colsample_bytree=1,
  gamma=c(0, 1)
  )
hyper.grid
dim(hyper.grid)

set.seed(123)
caret.cv <- train(Attrition ~ ., data=attrition.train, 
                  method="xgbTree", na.action=na.pass, 
                  trControl=trainControl(method="cv", number=5),
                  tuneGrid=hyper.grid, verbose=TRUE)
caret.cv

caret.cv$bestTune

caret.pred <- predict(caret.cv, newdata=attrition.test, na.action=na.pass, type="prob")
head(caret.pred)
caret.pred <- predict(caret.cv, newdata=attrition.test, na.action=na.pass, type="raw")
head(caret.pred)

table(attrition.test$Attrition, caret.pred, dnn=c("Actual", "Predicted"))
mean(attrition.test$Attrition==caret.pred)

library(doParallel)
cl <- makePSOCKcluster(detectCores())
registerDoParallel(cl)
set.seed(123)
caret.cv <- train(Attrition ~ ., data=attrition.train, 
                  method="xgbTree", na.action=na.pass,
                  trControl=trainControl(method="cv", number=5),
                  tuneGrid=hyper.grid, verbose=FALSE)
caret.pred <- predict(caret.cv, newdata=attrition.test, na.action=na.pass, type="raw")
table(attrition.test$Attrition, caret.pred, dnn=c("Actual", "Predicted"))
mean(attrition.test$Attrition==caret.pred)
stopImplicitCluster()

registerDoSEQ()

# [그림 15-37]
windows(width=7.0, height=5.5)
library(vip)
vip(caret.cv, num_features=10, method="model", geom="col",
    aesthetics=list(color="orange4", fill="tan"))

predict(caret.cv, newdata=attrition.train, type="prob")

# [그림 15-38]
library(pdp)
p1 <- partial(caret.cv, pred.var="NumCompaniesWorked", prob=TRUE, which.class=2) |> 
  ggplot(aes(x=NumCompaniesWorked, y=yhat)) +
  geom_line(color="maroon", linewidth=1) +
  ylim(c(0, 1))
p2 <- partial(caret.cv, pred.var="OverTime", prob=TRUE, which.class=2) |> 
  ggplot(aes(x=OverTime, y=yhat)) +
  geom_boxplot(color="maroon") +
  ylim(c(0, 1))
windows(width=7.0, height=3.5)
library(patchwork)
p1 + p2
