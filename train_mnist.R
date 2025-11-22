library(keras3)
library(tensorflow)
library(magrittr)

set.seed(42)

mnist <- dataset_mnist()  
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test  <- mnist$test$x
y_test  <- mnist$test$y

x_train <- array(as.numeric(x_train) / 255, dim = c(dim(x_train)[1], 28, 28, 1))
x_test  <- array(as.numeric(x_test)  / 255, dim = c(dim(x_test)[1],  28, 28, 1))

y_train <- as.integer(y_train)
y_test  <- as.integer(y_test)

model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = "relu",
                input_shape = c(28,28,1)) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 10, activation = "softmax")

model %>% compile(
  optimizer = "adam",
  loss = "sparse_categorical_crossentropy",
  metrics = "accuracy"
)

model %>% summary()

epochs <- 6
batch_size <- 128

history <- model %>% fit(
  x_train, y_train,
  epochs = epochs,
  batch_size = batch_size,
  validation_split = 0.12
)

eval <- model %>% evaluate(x_test, y_test, verbose = 1)
cat(sprintf("Test loss: %.4f   Test accuracy: %.4f\n", eval[[1]], eval[[2]]))

pred_probs <- model %>% predict(x_test)
pred_labels <- apply(pred_probs, 1, which.max) - 1  

model_file <- "mnist_model.keras"

save_model(model, filepath = model_file, overwrite = TRUE)

saveRDS(list(x_test = x_test, y_test = y_test, pred_labels = pred_labels, pred_probs = pred_probs),
        file = "mnist_test_data.rds")

saveRDS(history, file = "mnist_history.rds")

cat("Training complete.\n")
cat("Model saved to:", model_file, "\n")
cat("Test data saved to: mnist_test_data.rds\n")