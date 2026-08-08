
###################################
## R을 이용한 머신러닝: 지도학습 ##
## (곽기영, 도서출판 청람)       ## 
###################################

###################
## 제18장 딥러닝 ##
###################

################
## 18.6 Keras ##
################

# WSL2 설치

wsl --install

wsl --version

sudo apt update && sudo apt upgrade -y
sudo apt install -y \
  build-essential \
  curl \
  git \
  wget \
  ca-certificates \
  software-properties-common

lsb_release -a

nvidia-smi

# Python/TensorFlow 환경 구축

mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
source ~/.bashrc

conda --version

conda create -n r-keras python=3.11
conda activate r-keras

pip install tensorflow[and-cuda]

import tensorflow as tf
print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))

# Keras 3 설치

pip install keras

echo 'export KERAS_BACKEND=tensorflow' >> ~/.bashrc
source ~/.bashrc

# R 설치

sudo apt install r-base

R --version

R
install.packages("reticulate")
install.packages("keras3")

# R과 Python/TensorFlow 연결

library(reticulate)
use_condaenv("r-keras", required=TRUE) 
py_config()

library(tensorflow)
tf$config$list_physical_devices("GPU")

# RStudio Server 설치

sudo apt-get install gdebi-core
wget https://download2.rstudio.org/server/jammy/amd64/rstudio-server-2026.01.2-418-amd64.deb
sudo gdebi rstudio-server-2026.01.2-418-amd64.deb

sudo rstudio-server start

sudo systemctl enable rstudio-server

library(tensorflow)
tf_config() 
tf$constant("Hello TensorFlow!")

tf$config$list_physical_devices("GPU")

getwd()
file.edit("~/.Rprofie")

setwd("/mnt/d/Google Drive/MyWorkspaceR")

###############
## 18.7 사례 ##
###############

## 숫자 이미지

library(keras3)
mnist <- dataset_mnist()
str(mnist)

x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

str(x_train)
str(y_train)

str(x_test)
str(y_test)

dim(x_train)
typeof(x_train)

length(dim(x_train))

batch1 <- x_train[1:128,,]
str(batch1)

batch2 <- x_train[129:256,,]
str(batch2)

model <- keras_model_sequential(input_shape=c(28*28))
model |> 
  layer_dense(units=256, activation="relu") |>
  layer_dense(units=128, activation="relu") |> 
  layer_dense(units=10, activation="softmax")

summary(model)

model <- keras_model_sequential(input_shape=c(28*28))
layer_dense(model, units=256, activation="relu")
layer_dense(model, units=128, activation="relu")
layer_dense(model, units=10, activation="softmax")

model <- keras_model_sequential(input_shape=c(28*28))
model <- model |>  
  layer_dense(units=256, activation="relu") |> 
  layer_dense(units=128, activation="relu") |>  
  layer_dense(units=10, activation="softmax")

model <- keras_model_sequential(input_shape=c(28*28))
model |>  
  layer_dense(units=256) |> 
  layer_activation("relu") |>  
  layer_dense(units=128) |> 
  layer_activation("relu") |> 
  layer_dense(units=10) |>  
  layer_activation("softmax")

ls(pattern="^optimizer_", "package:keras3")
ls(pattern="^loss_", "package:keras3")
ls(pattern="^metric_", "package:keras3")

model |> compile(
  optimizer="rmsprop",
  loss="categorical_crossentropy",
  metrics="accuracy"
  )

x_train <- array_reshape(x_train, dim=c(nrow(x_train), 28*28))
dim(x_train)

x_train <- x_train/255

x_test <- array_reshape(x_test, dim=c(nrow(x_test), 28*28))
x_test <- x_test/255

y_train[1]
y_train <- to_categorical(y_train, num_classes=10)
dim(y_train)
y_train[1,]

y_test <- to_categorical(y_test, num_classes=10)

history <- model |>  fit(
  x_train, y_train,
  epochs=10, 
  batch_size=128,
  validation_split=0.2
  )

str(history)

history$params$steps
history$metrics$accuracy
history$metrics$val_accuracy

# [그림 18-22]
plot(history)
plot(history, metrics="accuracy")

history_df <- as.data.frame(history)
str(history_df)
history_df

metrics <- model |> evaluate(x_test, y_test)
metrics

as.matrix(model(x_test))

pred <- model |> predict(x_test)
pred

sum(pred[1,])

which.max(pred[1,])

pred <- model |>  
  predict(x_test) |>  
  op_argmax(axis=-1) |>  
  as.integer()
pred

table(mnist$test$y, pred-1, dnn=c("Actual", "Predicted"))
mean(mnist$test$y==pred-1)

## 숫자 이미지: 모델튜닝 

library(keras3)
mnist <- dataset_mnist()
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y
x_train <- array_reshape(x_train, c(nrow(x_train), 28*28))/255
x_test <- array_reshape(x_test, c(nrow(x_test), 28*28))/255
y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)

baseline_model <-  keras_model_sequential(input_shape=c(28*28)) |> 
  layer_dense(units=256, activation="relu") |> 
  layer_dense(units=128, activation="relu") |> 
  layer_dense(units=10, activation="softmax")
summary(baseline_model)
baseline_model |> compile(
  optimizer="rmsprop",
  loss="categorical_crossentropy",
  metrics="accuracy"
  )
baseline_history <- baseline_model |> fit(
  x_train, y_train,
  epochs=15, 
  batch_size=32,
  validation_split=0.5
  )

# 모델축소

smaller_model <- keras_model_sequential(input_shape=c(28*28)) |>  
  layer_dense(units=64, activation="relu") |> 
  layer_dense(units=32, activation="relu") |> 
  layer_dense(units=10, activation="softmax")
summary(smaller_model)
smaller_model |> compile(
  optimizer="rmsprop",
  loss="categorical_crossentropy",
  metrics="accuracy"
  )
smaller_history <- smaller_model |> fit(
  x_train, y_train,
  epochs=15, 
  batch_size=32,
  validation_split=0.5
  )

bigger_model <- keras_model_sequential(input_shape=c(28*28)) |>  
  layer_dense(units=512, activation="relu") |> 
  layer_dense(units=256, activation="relu") |> 
  layer_dense(units=10, activation="softmax")
summary(bigger_model)
bigger_model |> compile(
  optimizer="rmsprop",
  loss="categorical_crossentropy",
  metrics="accuracy"
  )
bigger_history <- bigger_model |> fit(
  x_train, y_train,
  epochs=15, 
  batch_size=32,
  validation_split=0.5
  )

compare_metrics <- data.frame(
  baseline_train=baseline_history$metrics$loss,
  baseline_validation=baseline_history$metrics$val_loss,
  smaller_train=smaller_history$metrics$loss,
  smaller_validation=smaller_history$metrics$val_loss,
  bigger_train=bigger_history$metrics$loss,
  bigger_validation=bigger_history$metrics$val_loss
  )
compare_metrics

library(dplyr)
library(tidyr)
library(tibble)
compare_metrics <- compare_metrics |>  
  rownames_to_column(var="epoch") |>  
  mutate(epoch=as.integer(epoch)) |>  
  pivot_longer(names_to="model_data", values_to="value", -epoch) |> 
  separate(col=model_data, into=c("model", "data"), sep="_", remove=FALSE)
compare_metrics

# [그림 18-23]
library(ggplot2)
ggplot(filter(compare_metrics, data=="validation"), 
       aes(x=epoch, y=value, color=model, bg=model)) +
  geom_point(pch=21, color="black") +
  geom_smooth(method="loess", se=FALSE) +
  labs(x="epoch", y="loss for validation data") +
  theme(legend.position="bottom",
        legend.title=element_blank())

# [그림 18-24]
ggplot(filter(compare_metrics, data=="train"), 
       aes(x=epoch, y=value, color=model, bg=model)) +
  geom_point(pch=21, color="black") +
  geom_smooth(method="loess", se=FALSE) +
  labs(x="epoch", y="loss for train data") +
  theme(legend.position="bottom",
        legend.title=element_blank())

# 가중치 제약

l2_model <-  keras_model_sequential(input_shape=c(28*28)) |> 
  layer_dense(units=256, activation="relu", 
              kernel_regularizer=regularizer_l2(l2=0.001)) |> 
  layer_dense(units=128, activation="relu",
              kernel_regularizer=regularizer_l2(l2=0.001)) |>  
  layer_dense(units=10, activation="softmax")
summary(l2_model)

l2_model |> compile(
  optimizer="rmsprop",
  loss="categorical_crossentropy",
  metrics="accuracy"
  )
l2_history <- l2_model |> fit(
  x_train, y_train,
  epochs=15, 
  batch_size=32,
  validation_split=0.5
  )

compare_metrics <- data.frame(
  baseline_train=baseline_history$metrics$loss,
  baseline_validation=baseline_history$metrics$val_loss,
  l2_train=l2_history$metrics$loss,
  l2_validation=l2_history$metrics$val_loss
  )
compare_metrics <- compare_metrics |>  
  rownames_to_column(var="epoch") |> 
  mutate(epoch=as.integer(epoch)) |>  
  pivot_longer(names_to="model_data", values_to="value", -epoch) |> 
  separate(col=model_data, into=c("model", "data"), sep="_", remove=FALSE)

# [그림 18-25]
ggplot(filter(compare_metrics, data=="validation"), 
       aes(x=epoch, y=value, color=model, bg=model)) +
  geom_point(pch=21, color="black") +
  geom_smooth(method="loess", se=FALSE) +
  labs(x="epoch", y="loss for validation data") +
  theme(legend.position="bottom",
        legend.title=element_blank())

# 드롭아웃

dropout_model <- keras_model_sequential(input_shape=c(28*28)) |>  
  layer_dense(units=256, activation="relu") |> 
  layer_dropout(rate=0.5) |> 
  layer_dense(units=128, activation="relu") |> 
  layer_dropout(rate=0.5) |> 
  layer_dense(units=10, activation="softmax")
summary(dropout_model)
dropout_model |> compile(
  optimizer="rmsprop",
  loss="categorical_crossentropy",
  metrics="accuracy"
  )
dropout_history <- dropout_model |> fit(
  x_train, y_train,
  epochs=15, 
  batch_size=32,
  validation_split=0.5
  )

compare_metrics <- data.frame(
  baseline_train=baseline_history$metrics$loss,
  baseline_validation=baseline_history$metrics$val_loss,
  dropout_train=dropout_history$metrics$loss,
  dropout_validation=dropout_history$metrics$val_loss
  )
compare_metrics <- compare_metrics |> 
  rownames_to_column(var="epoch") |> 
  mutate(epoch=as.integer(epoch)) |>  
  pivot_longer(names_to="model_data", values_to="value", -epoch) |> 
  separate(col=model_data, into=c("model", "data"), sep="_", remove=FALSE)

# [그림 18-26]
ggplot(filter(compare_metrics, data=="validation"), 
       aes(x=epoch, y=value, color=model, bg=model)) +
  geom_point(pch=21, color="black") +
  geom_smooth(method="loess", se=FALSE) +
  labs(x="epoch", y="loss for validation data") +
  theme(legend.position="bottom",
        legend.title=element_blank()) 

## 숫자 이미지: 모델저장

library(keras3)
mnist <- dataset_mnist()
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y
x_train <- array_reshape(x_train, c(nrow(x_train), 28*28))/255
x_test <- array_reshape(x_test, c(nrow(x_test), 28*28))/255
y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)
model <- keras_model_sequential(input_shape=c(28*28)) |>  
    layer_dense(units=256, activation="relu") |> 
    layer_dense(units=128, activation="relu") |>  
    layer_dense(units=10, activation="softmax")
model |> compile(
    optimizer="rmsprop",
    loss="categorical_crossentropy",
    metrics="accuracy"
    )

model |> fit(
  x_train, y_train, 
  epochs=5, 
  batch_size=128
  )
pred <- model |> predict(x_test)

# 전체 모델 저장

model |> save_model("my_model.keras")

new_model <- load_model("my_model.keras")
summary(new_model)

new_pred <- new_model |> predict(x_test)

isTRUE(all.equal(pred, new_pred))

new_model |> fit(
  x_train, y_train, 
  epochs=5, 
  batch_size=128
  )

# 아키텍처 저장

config <- get_config(model)
reinitialized_model <- from_config(config)
summary(reinitialized_model)

new_pred <- reinitialized_model |> predict(x_test)
isTRUE(all.equal(pred, new_pred))

# 가중치 저장

weights <- get_weights(model)
set_weights(reinitialized_model, weights)

new_pred <- reinitialized_model |> predict(x_test)
isTRUE(all.equal(pred, new_pred))

config <- get_config(model)
weights <- get_weights(model)
new_model <- from_config(config)
set_weights(new_model, weights)
new_pred <- new_model |> predict(x_test)
isTRUE(all.equal(pred, new_pred))

model |> save_model_weights("my_model.weights.h5")

config <- get_config(model)
reinitialized_model <- from_config(config)
new_model <- load_model_weights(reinitialized_model, "my_model.weights.h5")
new_pred <- new_model |> predict(x_test)
isTRUE(all.equal(pred, new_pred))

## 패션소품 이미지: MLP

library(keras3)
fashion_mnist <- dataset_fashion_mnist()

str(fashion_mnist)

c(x_train, y_train) %<-% fashion_mnist$train
c(x_test, y_test) %<-% fashion_mnist$test

c(c(x_train, y_train), c(x_test, y_test)) %<-% fashion_mnist

dim(x_train)
dim(y_train)
dim(x_test)
dim(y_test)

class_names <- c("T-shirt/top",
                 "Trouser",
                 "Pullover",
                 "Dress",
                 "Coat", 
                 "Sandal",
                 "Shirt",
                 "Sneaker",
                 "Bag",
                 "Ankle boot")

x_train <- x_train/255
x_test <- x_test/255

model <- keras_model_sequential(input_shape=c(28, 28)) |> 
  layer_flatten() |> 
  layer_dense(units=128, activation="relu") |> 
  layer_dense(units=10, activation="softmax")
summary(model)

model |> compile(
  optimizer=optimizer_adam(learning_rate=0.002),
  loss="sparse_categorical_crossentropy",
  metrics="accuracy"
  )

val_index <- 1:10000
x_val <- x_train[val_index,,]
part_x_train <- x_train[-val_index,,]
y_val <- y_train[val_index]
part_y_train <- y_train[-val_index]

history <- model |> fit(
  part_x_train, part_y_train,
  epochs=10, 
  batch_size=128,
  verbose=2,
  validation_data=list(x_val, y_val)
  )

# [그림 18-28]
plot(history)

metrics <- model |> evaluate(x_test, y_test)
metrics

model |> predict(x_test)

pred <- model |>  
  predict(x_test) |>  
  op_argmax(axis=-1) |>  
  as.integer()
pred

pred[1]
class_names[pred[1]]
class_names[fashion_mnist$test$y[1]+1]

table(class_names[fashion_mnist$test$y+1], class_names[pred], 
      dnn=c("Actual", "Predicted"))
mean(fashion_mnist$test$y+1==pred)

# [그림 18-29]
old.par <- par(mfrow=c(5,10), mar=c(1.5,0,1.5,0), xaxs="i", yaxs="i")
pred <- model |>  
  predict(x_test) |>  
  op_argmax(axis=-1) |>  
  as.integer()
set.seed(123)
for (i in sample(dim(x_test)[1], 50)) {
  img <- x_test[i,,]
  img <- t(apply(img, 2, rev))
  y_predicted <- pred[i]
  y_actual <- fashion_mnist$test$y[i]+1
  if (y_predicted==y_actual) {
    color <- "royalblue"
    image(1:28, 1:28, img, col=gray((255:0)/255), xaxt="n", yaxt="n",
          main=class_names[y_predicted],
          col.main=color)
  } else {
    color <- "salmon"
    image(1:28, 1:28, img, col=hcl.colors(12, "Purples", rev=TRUE), 
          xaxt="n", yaxt="n",
          main=class_names[y_predicted],
          col.main=color)
    mtext(paste0("(", class_names[y_actual], ")"), side=1, line=0.3, 
          cex=0.8, col=color, font=2)
  }
}
par(old.par)

# 가용한 모든 GPU 사용하여 분산처리
library(tensorflow)
strategy <- tf$distribute$MirroredStrategy()
strategy$num_replicas_in_sync

strategy <- tf$distribute$MirroredStrategy(devices=list("/gpu:0", "/gpu:1"))

library(keras3)
fashion_mnist <- dataset_fashion_mnist()
c(c(x_train, y_train), c(x_test, y_test)) %<-% fashion_mnist

x_train <- x_train/255
x_test <- x_test/255

val_index <- 1:10000
x_val <- x_train[val_index,,]
part_x_train <- x_train[-val_index,,]
y_val <- y_train[val_index]
part_y_train <- y_train[-val_index]

with(strategy$scope(), {
  model <- keras_model_sequential(input_shape=c(28, 28)) |> 
    layer_flatten() |> 
    layer_dense(units=128, activation="relu") |> 
    layer_dense(units=10, activation="softmax")
  model |> compile(
    optimizer=optimizer_adam(learning_rate=0.002),
    loss="sparse_categorical_crossentropy",
    metrics="accuracy"
  )
})
model |> fit(
  part_x_train, part_y_train,
  epochs=10, 
  batch_size=128,
  verbose=2,
  validation_data=list(x_val, y_val)
)
model |> evaluate(x_test, y_test)

## 패션소품 이미지: CNN

library(keras3)
fashion_mnist <- dataset_fashion_mnist()
c(c(x_train, y_train), c(x_test, y_test)) %<-% fashion_mnist
str(x_train); str(y_train)
str(x_test); str(y_test)

x_train <- array_reshape(x_train, c(60000, 28, 28, 1))/255
x_test <- array_reshape(x_test, c(10000, 28, 28, 1))/255
dim(x_train)
dim(x_test)

val_index <- 1:10000
x_val <- x_train[val_index,,,,drop=FALSE]
part_x_train <- x_train[-val_index,,,,drop=FALSE]
str(part_x_train); str(x_val)
y_val <- y_train[val_index]
part_y_train <- y_train[-val_index]
str(part_y_train); str(y_val)

model <- keras_model_sequential(input_shape=c(28, 28, 1)) |>
  layer_conv_2d(filters=32, kernel_size=c(3, 3), activation="relu") |>
  layer_max_pooling_2d(pool_size=c(2, 2)) |>
  layer_conv_2d(filters=64, kernel_size=c(3, 3), activation="relu") |>
  layer_max_pooling_2d(pool_size=c(2, 2)) |>
  layer_conv_2d(filters=64, kernel_size=c(3, 3), activation="relu")
summary(model)

model |>
  layer_flatten() |>
  layer_dense(units=64, activation="relu") |>
  layer_dense(units=10, activation="softmax")

summary(model)

model |> compile(
  optimizer=optimizer_adam(learning_rate=0.002),
  loss="sparse_categorical_crossentropy",
  metrics="accuracy"
  )
model |> fit(
  part_x_train, part_y_train,
  epochs=10, 
  batch_size=128,
  verbose=2,
  validation_data=list(x_val, y_val)
  )

metrics <- model |> evaluate(x_test, y_test)
metrics

## 보스턴 주택가격

library(keras3)
boston_housing <- dataset_boston_housing()
c(c(x_train, y_train), c(x_test, y_test)) %<-% boston_housing

str(x_train); str(y_train)
str(x_test); str(y_test)

x_train[1,]
apply(x_train, 2, range)

mean <- apply(x_train, 2, mean)
std <- apply(x_train, 2, sd)
x_train <- scale(x_train, center=mean, scale=std)
x_train[1,]

x_test <- scale(x_test, center=mean, scale=std)
x_test[1,]

model <- keras_model_sequential(input_shape=dim(x_train)[2]) |> 
  layer_dense(units=64, activation="relu") |> 
  layer_dense(units=64, activation="relu") |> 
  layer_dense(units=1)
summary(model)

model |> compile(
  optimizer="rmsprop",
  loss="mse",
  metrics="mean_absolute_error"
  )

history <- model |> fit(
  x_train,  y_train,
  epochs=500,
  batch_size=16,
  validation_split=0.2,
  verbose=2
  )

history$metrics$mean_absolute_error[500]
history$metrics$val_mean_absolute_error[500]

model |> evaluate(x_test, y_test)

c(loss, mae) %<-% (model |> evaluate(x_test, y_test))
loss
mae

# [그림 18-31]
plot(history)

# [그림 18-32]
library(ggplot2)
plot(history, metrics="mean_absolute_error", smooth=FALSE) +
  theme(legend.position="bottom",
        legend.title=element_blank()) +
  coord_cartesian(ylim=c(0, 4))

pred <- model |> predict(x_test)
pred
pred[,1][1:5]

cor(y_test, pred[,1])

# [그림 18-33]
ggplot(data.frame(pred=pred[,1], actual=y_test), 
       aes(x=pred, y=actual)) + 
  geom_point(pch=21, color="black", bg="tomato", size=2, stroke=1.2) +
  geom_smooth(method="loess", color="slateblue", linewidth=1.2) +
  labs(x="Predicted Housing Prices ($1,000)", y="Actual Housing Prices ($1,000)") +
  theme_classic()

build_model <- function() {
  model <- keras_model_sequential(input_shape=dim(x_train)[2]) |> 
    layer_dense(units=64, activation="relu") |> 
    layer_dense(units=64, activation="relu") |> 
    layer_dense(units=1)
  model |> compile(
    optimizer="rmsprop",
    loss="mse",
    metrics="mean_absolute_error"
  )
}
model <- build_model()
summary(model)

k <- 5
set.seed(123)
index <- sample(1:nrow(x_train))
fold <- cut(index, breaks=k, labels=FALSE)
fold

num_epochs <- 100
all_metrics <- c()
for (i in 1:k) {
  cat("processing fold #", i, "\n")
  val_index <- which(fold==i)
  x_val <- x_train[val_index,]
  y_val <- y_train[val_index]
  part_x_train <- x_train[-val_index,]
  part_y_train <- y_train[-val_index]
  model <- build_model()
  model |> fit(part_x_train, part_y_train, 
               epochs=num_epochs, batch_size=16, verbose=0)
  metrics <- model |> evaluate(x_val, y_val, verbose=2)
  all_metrics <- c(all_metrics, metrics["mean_absolute_error"])
}

all_metrics
range(all_metrics)
mean(unlist(all_metrics))

num_epochs <- 500
all_mae_history <- c()
for (i in 1:k) {
  cat("processing fold #", i, "\n")
  val_index <- which(fold==i)
  x_val <- x_train[val_index,]
  y_val <- y_train[val_index]
  part_x_train <- x_train[-val_index,]
  part_y_train <- y_train[-val_index]
  model <- build_model()
  history <- model |> fit(
    part_x_train, part_y_train, 
    validation_data=list(x_val, y_val),
    epochs=num_epochs, batch_size=16, verbose=0
    )
  mae_history <- history$metrics$val_mean_absolute_error
  cat("mean_absolute_error:", sprintf("%.4f", mae_history[num_epochs]), "\n")
  all_mae_history <- rbind(all_mae_history, mae_history)
}

dim(all_mae_history)
str(all_mae_history)

average_mae_history <- data.frame(
  epoch=seq(1:ncol(all_mae_history)),
  validation_mae=apply(all_mae_history, 2, mean)
  )
str(average_mae_history)

# [그림 18-34]
ggplot(average_mae_history, aes(x=epoch, y=validation_mae)) + 
  geom_point(pch=21, color="black", bg="turquoise") +
  geom_smooth(method="loess", color="deepskyblue", linewidth=1.2) +
  coord_cartesian(ylim=c(2, 4)) +
  theme_bw()

model <- build_model()
model |> fit(x_train, y_train,
             epochs=150, batch_size=16, validation_split=0, verbose=2)

c(loss, mae) %<-% (model |> evaluate(x_test, y_test))
loss
mae

## 보스턴 주택가격: 콜백

# 단계별 가중치 저장

library(keras3)
boston_housing <- dataset_boston_housing()
c(c(x_train, y_train), c(x_test, y_test)) %<-% boston_housing
mean <- apply(x_train, 2, mean)
std <- apply(x_train, 2, sd)
x_train <- scale(x_train, center=mean, scale=std)
x_test <- scale(x_test, center=mean, scale=std)
build_model <- function() {
  model <- keras_model_sequential(input_shape=dim(x_train)[2]) |> 
    layer_dense(units=64, activation="relu") |> 
    layer_dense(units=64, activation="relu") |> 
    layer_dense(units=1)
  model |> compile(
    optimizer="rmsprop",
    loss="mse",
    metrics="mean_absolute_error"
  )
}

checkpoint_dir <- "checkpoints"
unlink(checkpoint_dir, recursive=TRUE)
dir.create(checkpoint_dir, showWarnings=FALSE)
filepath <- file.path(checkpoint_dir, "{epoch:02d}-{val_loss:.2f}.weights.h5")
save_weights_callback <- callback_model_checkpoint(
  filepath=filepath,
  save_weights_only=TRUE,
  verbose=1
  )

model <- build_model()
model |> fit(
  x_train,  y_train,
  epochs=20, 
  validation_data=list(x_test, y_test),
  callbacks=list(save_weights_callback),
  verbose=2
  )
model |> evaluate(x_test, y_test)

list.files(checkpoint_dir)

new_model <- build_model()
new_model |> evaluate(x_test, y_test)

list.files(checkpoint_dir)[20]
new_model |> load_model_weights(
  file.path(checkpoint_dir, list.files(checkpoint_dir)[20])
  )

new_model |> evaluate(x_test, y_test)

checkpoint_dir <- "checkpoints"
unlink(checkpoint_dir, recursive=TRUE)
dir.create(checkpoint_dir)
filepath <- file.path(checkpoint_dir, "{epoch:02d}-{val_loss:.2f}.weights.h5")
save_weights_callback <- callback_model_checkpoint(
  filepath=filepath,
  save_weights_only=TRUE,
  save_best_only=TRUE,
  verbose=1
  )
model <- build_model()
model |> fit(
  x_train,  y_train,
  epochs=20, 
  validation_data=list(x_test, y_test),
  callbacks=list(save_weights_callback),
  verbose=2
  )
list.files(checkpoint_dir)

# 학습 조기종료

stop_early_callback <- callback_early_stopping(monitor="val_loss", patience=20)

checkpoint_dir <- "checkpoints"
unlink(checkpoint_dir, recursive=TRUE)
dir.create(checkpoint_dir)
filepath <- file.path(checkpoint_dir, "{epoch:02d}-{val_loss:.2f}.weights.h5")
save_weights_callback <- callback_model_checkpoint(
  filepath=filepath,
  save_weights_only=TRUE,
  save_best_only=TRUE,
  verbose=1
  )

model <- build_model()
history <- model |> fit(
  x_train,  y_train,
  epochs=500,
  batch_size=1,
  validation_split=0.2,
  callbacks=list(stop_early_callback, save_weights_callback),
  verbose=2
  )
list.files(checkpoint_dir)

# [그림 18-35]
plot(history, metrics="loss", smooth=FALSE) +
  theme(legend.position="bottom",
        legend.title=element_blank()) +
  coord_cartesian(xlim=c(0, 40), ylim=c(0, 30))

# 학습률 조정

reduce_lr_callback <- callback_reduce_lr_on_plateau(
  monitor="val_loss",
  factor=0.1,
  patience=10
  )

save_weights_callback <- callback_model_checkpoint(
  filepath=filepath,
  save_weights_only=TRUE,
  save_best_only=TRUE,
  verbose=1
  )
stop_early_callback <- callback_early_stopping(monitor="val_loss", patience=20)
checkpoint_dir <- "checkpoints"
unlink(checkpoint_dir, recursive=TRUE)
dir.create(checkpoint_dir)
filepath <- file.path(checkpoint_dir, "{epoch:02d}-{val_loss:.2f}.weights.h5")
model <- build_model()
history <- model |> fit(
  x_train,  y_train,
  epochs=500,
  batch_size=1,
  validation_split=0.2,
  callbacks=list(stop_early_callback, save_weights_callback, reduce_lr_callback),
  verbose=2
  )

list.files(checkpoint_dir)
history$metrics$learning_rate
