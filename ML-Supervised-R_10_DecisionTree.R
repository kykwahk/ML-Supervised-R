
###################################
## R을 이용한 머신러닝: 지도학습 ##
## (곽기영, 도서출판 청람)       ## 
###################################

#########################
## 제10장 의사결정나무 ##
#########################

################################
## 10.4 조건부추론나무와 C5.0 ##
################################

## 조건부추론나무

dv <- c(2,2,5,4,6,3)
iv <- c(1,3,5,4,5,2)

library(gtools)
perms <- permutations(n=6, r=6, v=dv, set=FALSE) 
head(perms)
tail(perms)
dim(perms)

factorial(6)/factorial(6-6)

cors <- apply(perms, 1, function(dv.each) cor(dv.each, iv)) 
cors <- cors[order(cors)]
head(cors)
tail(cors)

cor.test <- cor(dv, iv)
cor.test

mean(abs(cors) >= abs(cor.test))

###############
## 10.5 사례 ##
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

library(rpart)
set.seed(123)
rpart.fit <- rpart(medv ~ ., data=Boston.train, method="anova")
class(rpart.fit)

rpart.fit

summary(rpart.fit)

# [그림 10-26]
windows(width=7.0, height=5.5)
library(rpart.plot)
rpart.plot(rpart.fit)

rpart.pred <- predict(rpart.fit, newdata=Boston.test)
head(rpart.pred)

sqrt(mean((Boston.test$medv - rpart.pred)^2))

postResample(rpart.pred, Boston.test$medv)

rpart.fit$cptable
printcp(rpart.fit)
printcp(rpart.fit, digits=3)

# [그림 10-27]
windows(width=7.0, height=5.5)
plotcp(rpart.fit, col="red")

rpart.pruned <- rpart(medv ~ ., data=Boston.train, method="anova", 
                      control=list(cp=0.01195233))
rpart.pruned
rpart.pruned$cptable

prune(rpart.fit, cp=0.01195233)

rpart.pred <- predict(rpart.pruned, newdata=Boston.test)
head(rpart.pred)
postResample(rpart.pred, Boston.test$medv)

rpart.pruned$variable.importance

# [그림 10-28]
windows(width=7.0, height=5.5)
library(vip)
vip(rpart.pruned, method="model", geom="col",
    aesthetics=list(color="royalblue", fill="cornflowerblue"))

# [그림 10-29]
library(pdp)
library(ggplot2)
p1 <- partial(rpart.pruned, pred.var="rm", plot=FALSE) |>
  ggplot(aes(x=rm, y=yhat)) +
  geom_line(color="salmon", linewidth=1) +
  geom_rug(aes(x=rm, y=NULL), data=Boston.train, sides="b") +
  scale_y_continuous(name="yhat ($1,000)") 
p2 <- partial(rpart.pruned, pred.var="lstat", plot=FALSE) |>
  ggplot(aes(x=lstat, y=yhat)) +
  geom_line(color="salmon", linewidth=1) +
  geom_rug(aes(x=lstat, y=NULL), data=Boston.train, sides="b") +
  scale_y_continuous(name="yhat ($1,000)") 
windows(width=7.0, height=3.5)
library(patchwork)
p1 + p2

modelLookup("rpart")

set.seed(123)
caret.cv <- train(medv ~ ., data=Boston.train, method="rpart",
                  trControl=trainControl(method="cv", number=10),
                  tuneLength=500)

caret.cv$bestTune
caret.cv$finalModel$cptable

caret.pred <- predict(caret.cv, newdata=Boston.test, type="raw")
head(caret.pred)
postResample(pred=caret.pred, obs=Boston.test$medv)

## 신용평가

library(modeldata)
str(credit_data)
levels(credit_data$Status)

library(caret)
set.seed(123)
index <- createDataPartition(credit_data$Status, p=0.7, list=FALSE)
credit.train <- credit_data[index,]
credit.test <- credit_data[-index,]
dim(credit.train)
dim(credit.test)
table(credit.train$Status)
table(credit.test$Status)

library(rpart)
set.seed(123)
rpart.fit <- rpart(Status ~ ., data=credit.train, method="class",
                   parms=list(split="gini"),
                   control=list(cp=0))

rpart.fit

rpart.pred <- predict(rpart.fit, newdata=credit.test, type="prob")
head(rpart.pred)
rpart.pred <- predict(rpart.fit, newdata=credit.test, type="class")
head(rpart.pred)

table(credit.test$Status, rpart.pred, dnn=c("Actual", "Predicted"))
mean(credit.test$Status==rpart.pred)

postResample(rpart.pred, credit.test$Status)

rpart.fit$cptable

# [그림 10-30]
windows(width=12.0, height=5.5)
plotcp(rpart.fit, col="blue")

min.index <- which.min(rpart.fit$cptable[,"xerror"])
rpart.fit$cptabl[min.index, c("xerror", "xstd")]

rpart.pruned <- rpart(Status ~ ., data=credit.train, method="class",
                      control=list(cp=0.01))
rpart.pruned
rpart.pruned$cptable

# [그림 10-31]
windows(width=7.0, height=5.5)
library(rpart.plot)
rpart.plot(rpart.pruned)
cols <- ifelse(rpart.pruned$frame$yval==1, "darkred", "green4")
prp(rpart.pruned, type=2, extra=104, fallen.leaves=TRUE, roundint=FALSE, 
    branch.lty=3, col=cols, border.col=cols, shadow.col="gray",
    split.cex=1.2, split.suffix="?",
    split.box.col="lightgray", split.border.col="darkgray", split.round=0.5)

# [그림 10-32]
windows(width=7.0, height=5.5)
library(rattle)
fancyRpartPlot(rpart.pruned, sub=NULL)

rpart.pred <- predict(rpart.pruned, newdata=credit.test, type="class")
head(rpart.pred)
table(credit.test$Status, rpart.pred, dnn=c("Actual", "Predicted"))
mean(credit.test$Status==rpart.pred)

set.seed(123)
rpart.fit <- rpart(Status ~ ., data=credit.train, method="class",
                   control=list(cp=0, minsplit=30, maxdepth=7))
rpart.fit$cptable

hyper.grid <- expand.grid(
  minsplit=seq(10, 30, 1),
  maxdepth=seq(5, 10, 1)
  )
dim(hyper.grid)
hyper.grid

rpart.models <- list()
for (i in 1:nrow(hyper.grid)) {
  minsplit <- hyper.grid$minsplit[i]
  maxdepth <- hyper.grid$maxdepth[i]
  rpart.models[[i]] <- rpart(Status ~ ., data=credit.train, method="class",
                             control=list(cp=0, minsplit=minsplit, maxdepth=maxdepth))
}
length(rpart.models)

getCP <- function(x) {
  min.index <- which.min(x$cptable[, "xerror"])
  cp <- x$cptable[min.index, "CP"]
}
getMinError <- function(x) {
  min.index <- which.min(x$cptable[, "xerror"])
  xerror <- x$cptable[min.index, "xerror"] 
}

library(dplyr)
hyper.grid |> 
  mutate(cp=sapply(rpart.models, getCP), xerror=sapply(rpart.models, getMinError)) |> 
  slice_min(n=5, order_by=xerror, with_ties=FALSE)

rpart.fit <- rpart(Status ~ ., data=credit.train, method="class",
                   control=list(cp=0.003644647, minsplit=10, maxdepth=9))
rpart.fit$cptable

rpart.pred <- predict(rpart.fit, newdata=credit.test, type="class")
head(rpart.pred)
table(credit.test$Status, rpart.pred, dnn=c("Actual", "Predicted"))
mean(credit.test$Status==rpart.pred)

rpart.fit$variable.importance

# [그림 10-33]
windows(width=7.0, height=5.5)
library(vip)
vip(rpart.fit, method="model", geom="col",
    aesthetics=list(color="palegreen4", fill="olivedrab"))

# [그림 10-34]
library(pdp)
library(ggplot2)
p1 <- partial(rpart.fit, pred.var="Seniority", prob=TRUE) |> 
  ggplot(aes(x=Seniority, y=yhat)) +
  geom_line(color="magenta4", linewidth=1) +
  ylim(c(0, 1))
p2 <- partial(rpart.fit, pred.var="Job", prob=TRUE) |>
  ggplot(aes(x=Job, y=yhat)) +
  geom_boxplot(color="magenta4") +
  ylim(c(0, 1))
windows(width=7.0, height=3.5)
library(patchwork)
p1 + p2

## 유방암

library(mlbench)
data(BreastCancer)
str(BreastCancer)
levels(BreastCancer$Class)

bc <- BreastCancer[-1]
bc <- cbind(lapply(bc[-10], function(x) as.numeric(x)), bc[10])
str(bc)
library(caret)
set.seed(123)
index <- createDataPartition(bc$Class, p=0.7, list=FALSE)
bc.train <- bc[index,]
bc.test <- bc[-index,]
dim(bc.train)
dim(bc.test)
table(bc.train$Class)
table(bc.test$Class)

library(partykit)
ctree.fit <- ctree(Class ~ ., data=bc.train, control=ctree_control(maxsurrogate=3))
class(ctree.fit)
ctree.fit

# [그림 10-35]
windows(width=12.0, height=8.0)
plot(ctree.fit, 
     inner_panel=node_inner(ctree.fit, fill=c("lightgoldenrod", "whitesmoke")),
     terminal_panel=node_barplot(ctree.fit, fill=c("maroon", "green4")))

?plot.party

# [그림 10-36]
windows(width=15.0, height=8.0)
plot(ctree.fit, type="simple",
     inner_panel=node_inner(ctree.fit, fill=c("lightgoldenrod", "whitesmoke")))

predict(ctree.fit, newdata=bc.test, type="node")
ctree.pred <- predict(ctree.fit, newdata=bc.test, type="prob")
head(ctree.pred)
ctree.pred <- predict(ctree.fit, newdata=bc.test, type="response")
head(ctree.pred)

table(bc.test$Class, ctree.pred, dnn=c("Actual", "Predicted"))
mean(bc.test$Class==ctree.pred)

# [그림 10-37]
windows(width=7.0, height=5.5)
library(rpart)
rpart.fit <- rpart(Class ~ ., data=bc.train, method="class")
plot(as.party(rpart.fit),
     inner_panel=node_inner(as.party(rpart.fit), fill=c("darkseagreen", "beige")),
     terminal_panel=node_barplot(as.party(rpart.fit), fill=c("salmon", "cornflowerblue")))

## 고객이탈

library(modeldata)
str(mlc_churn)

churn <- mlc_churn[,c("voice_mail_plan", "total_day_minutes", "total_intl_minutes", 
                      "number_customer_service_calls", "churn")]
str(churn)
levels(churn$churn)

churn.train <- churn[1:3333,]
churn.test <- churn[3334:5000,]
dim(churn.train)
dim(churn.test)
table(churn.train$churn)
table(churn.test$churn)

library(C50)
C50.fit <- C5.0(x=churn.train[-5], y=churn.train$churn)
class(C50.fit)
C50.fit

# [그림 10-38]
windows(width=9.0, height=6.0)
plot(C50.fit)

library(partykit)
plot(as.party(C50.fit), 
     inner_panel=node_inner(as.party(C50.fit), fill=c("lightgoldenrod", "whitesmoke")),
     terminal_panel=node_barplot(as.party(C50.fit), beside=TRUE, fill=c("maroon", "green4")))

summary(C50.fit)

C50.pred <- predict(C50.fit, newdata=churn.test, type="prob")
head(C50.pred)
C50.pred <- predict(C50.fit, newdata=churn.test, type="class")
head(C50.pred)
table(churn.test$churn, C50.pred, dnn=c("Actual", "Predicted"))
mean(churn.test$churn==C50.pred)

C50.rules <- C5.0(churn ~ ., data=churn.train, rules=TRUE)
summary(C50.rules)

C50.boost <- C5.0(churn ~ ., data=churn.train, trials=10)
C50.boost
summary(C50.boost)

C50.boost.pred <- predict(C50.boost, newdata=churn.test, type="class")
head(C50.boost.pred)
table(churn.test$churn, C50.boost.pred, dnn=c("Actual", "Predicted"))
mean(churn.test$churn==C50.boost.pred)

error.costs <- matrix(c(0,1,10,0), 2, 2, 
                      dimnames=list(Predicted=c("no", "yes"), 
                                    Actual=c("no", "yes")))
error.costs

C50.costs <- C5.0(churn ~ ., data=churn.train, costs=error.costs)

C50.costs.pred <- predict(C50.costs, newdata=churn.test, type="class")
head(C50.costs.pred)
table(churn.test$churn, C50.costs.pred, dnn=c("Actual", "Predicted"))
mean(churn.test$churn==C50.costs.pred)
