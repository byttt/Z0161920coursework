rm(list=ls()) # clears all objects in "global environment"
cat("\014") # clears the console area
install.packages("tidyverse")
install.packages("data.table")
install.packages("mlr3verse")
install.packages("rgeos")
library("tidyverse")
library("ggplot2")
library("skimr")
library("DataExplorer")
library("data.table")
library("mlr3verse")
library("xgboost")
library("rgeos")
#Load data
hotels <- readr::read_csv("hotels.csv")

#Initial data
hotels <- hotels %>%
  filter(adr < 4000) %>% 
  mutate(total_nights = stays_in_weekend_nights+stays_in_week_nights)

hotels <- hotels %>%
  select(-reservation_status, -reservation_status_date) %>% 
  mutate(kids = case_when(
    children + babies > 0 ~ "kids",
    TRUE ~ "none"
  ))

hotels <- hotels %>% 
  select(-babies, -children)

hotels <- hotels %>% 
  mutate(parking = case_when(
    required_car_parking_spaces > 0 ~ "parking",
    TRUE ~ "none"
  )) %>% 
  select(-required_car_parking_spaces)

hotels.bycountry <- hotels %>% 
  group_by(country) %>% 
  summarise(total = n(),
            cancellations = sum(is_canceled),
            pct.cancelled = cancellations/total*100)

hotels <- hotels %>% 
  mutate(is_canceled = case_when(
    is_canceled > 0 ~ "YES",
    TRUE ~ "NO"
  ))

hotels.par <- hotels %>%
  select(hotel, is_canceled, kids, meal, customer_type) %>%
  group_by(hotel, is_canceled, kids, meal, customer_type) %>%
  summarize(value = n())

hotels2 <- hotels %>% 
  select(-country, -reserved_room_type, -assigned_room_type, -agent, -company,
         -stays_in_weekend_nights, -stays_in_week_nights)

# Visualize Data
skimr::skim(hotels2)

DataExplorer::plot_bar(hotels2, ncol = 2)
DataExplorer::plot_histogram(hotels2, ncol = 2)
DataExplorer::plot_boxplot(hotels2, by = "is_canceled", ncol = 2)

# fit a logisitic regression model
fit.lr <- glm(as.factor(is_canceled) ~ ., binomial, hotels2)
summary(fit.lr)
pred.lr <- predict(fit.lr, hotels2, type = "response")
ggplot(data.frame(x = pred.lr), aes(x = x)) + geom_histogram()
# Bayes classifier
# confusion matrix
conf.mat <- table(`true cancel` = hotels2$is_canceled, `predict cancel` = pred.lr > 0.5)
conf.mat
conf.mat/rowSums(conf.mat)*100

# LDA
hotel_lda <- MASS::lda(is_canceled ~ ., hotels2)
hotel_pred <- predict(hotel_lda, na.omit(hotels2))
mean(I(hotel_pred$class == na.omit(hotels2)$is_canceled))
table(truth = na.omit(hotels2)$is_canceled, prediction = hotel_pred$class)

# MLR3
set.seed(212) # set seed for reproducibility
# Define factor
hotels2$hotel <- as.factor(hotels2$hotel)
hotels2$arrival_date_month <- as.factor(hotels2$arrival_date_month)
hotels2$meal <- as.factor(hotels2$meal)
hotels2$market_segment <- as.factor(hotels2$market_segment)
hotels2$distribution_channel <- as.factor(hotels2$distribution_channel)
hotels2$deposit_type <- as.factor(hotels2$deposit_type)
hotels2$customer_type <- as.factor(hotels2$customer_type)
hotels2$kids <- as.factor(hotels2$kids)
hotels2$parking <- as.factor(hotels2$parking)
hotels2$is_canceled <- as.factor(hotels2$is_canceled)
#Define task
hotel_task <- TaskClassif$new(id = "HotelCancel",
                              backend = hotels2, # <- NB: no na.omit() this time
                              target = "is_canceled",
                              positive = "YES")
# Cross validation resampling strategy
cv5 <- rsmp("cv", folds = 5)
cv5$instantiate(hotel_task)
# Define a collection of base learners
lrn_baseline <- lrn("classif.featureless", predict_type = "prob")
lrn_cart     <- lrn("classif.rpart", predict_type = "prob")
lrn_xgboost  <- lrn("classif.xgboost", predict_type = "prob")
lrn_log_reg  <- lrn("classif.log_reg", predict_type = "prob")

res_baseline <- resample(hotel_task, lrn_baseline, cv5, store_models = TRUE)
res_cart <- resample(hotel_task, lrn_cart, cv5, store_models = TRUE)
res_baseline$aggregate()
res_cart$aggregate()
#benchmark function
res <- benchmark(data.table(
  task       = list(hotel_task),
  learner    = list(lrn_baseline,
                    lrn_cart),
  resampling = list(cv5)
), store_models = TRUE)

res$aggregate(list(msr("classif.ce"),
                   msr("classif.acc"),
                   msr("classif.auc"),
                   msr("classif.fpr"),
                   msr("classif.fnr")))

lrn_cart_cv <- lrn("classif.rpart", predict_type = "prob", xval = 10)
res_cart_cv <- resample(hotel_task, lrn_cart_cv, cv5, store_models = TRUE)
rpart::plotcp(res_cart_cv$learners[[5]]$model)
lrn_cart_cp <- lrn("classif.rpart", predict_type = "prob", cp = 0.022)
#fit, not use benchmark
res.log <- resample(hotel_task, lrn_log_reg, cv5, store_models = TRUE)
res.cart <- resample(hotel_task, lrn_cart_cp, cv5, store_models = TRUE)

res <- benchmark(data.table(
  task       = list(hotel_task),
  learner    = list(lrn_log_reg,
                    lrn_cart_cp),
  resampling = list(cv5)
), store_models = TRUE)
res$aggregate(list(msr("classif.ce"),
                   msr("classif.acc"),
                   msr("classif.auc"),
                   msr("classif.fpr"),
                   msr("classif.fnr")))

#Dealing with missingness and factors:modelling pipeline
# Create a pipeline which encodes and then fits an XGBoost model
lrn_xgboost <- lrn("classif.xgboost", predict_type = "prob")
pl_xgb <- po("encode") %>>%
  po(lrn_xgboost)
# Handling missingness
# First create a pipeline of just missing fixes we can later use with models
pl_missing <- po("fixfactors") %>>%
  po("removeconstants") %>>%
  po("imputesample", affect_columns = selector_type(c("ordered", "factor"))) %>>%
  po("imputemean")
# Now try with a model that needs no missingness
lrn_log_reg <- lrn("classif.log_reg", predict_type = "prob")
pl_log_reg <- pl_missing %>>%
  po(lrn_log_reg)
# Now fit as normal ... we can just add it to our benchmark set
res <- benchmark(data.table(
  task       = list(hotel_task),
  learner    = list(lrn_baseline,
                    lrn_cart,
                    lrn_cart_cp,
                    pl_xgb,
                    pl_log_reg),
  resampling = list(cv5)
), store_models = TRUE)
res$aggregate(list(msr("classif.ce"),
                   msr("classif.acc"),
                   msr("classif.fpr"),
                   msr("classif.fnr")))
#ROC
autoplot(res.log, type = 'roc')
autoplot(res.cart, type = "roc")

# 10 FOld Cross Validation
cv10 <- rsmp("cv", folds = 10)
cv10$instantiate(hotel_task)
res10 <- benchmark(data.table(
  task       = list(hotel_task),
  learner    = list(lrn_baseline,
                    lrn_cart,
                    lrn_cart_cp,
                    pl_xgb,
                    pl_log_reg),
  resampling = list(cv10)
), store_models = TRUE)

res10$aggregate(list(msr("classif.ce"),
                     msr("classif.acc"),
                     msr("classif.auc"),
                     msr("classif.fpr"),
                     msr("classif.fnr")))

# Deep Learning
library("rsample")
set.seed(212) # by setting the seed we know everyone will see the same results
# First get the training
hotel_split <- initial_split(hotels2)
hotel_train <- training(hotel_split)
# Then further split the training into validate and test
hotel_split2 <- initial_split(testing(hotel_split), 0.5)
hotel_validate <- training(hotel_split2)
hotel_test <- testing(hotel_split2)

library("recipes")
cake <- recipe(is_canceled~ ., data = hotels2) %>%
  step_center(all_numeric()) %>% # center by subtracting the mean from all numeric features
  step_scale(all_numeric()) %>% # scale by dividing by the standard deviation on all numeric features
  step_unknown(all_nominal(), -all_outcomes()) %>% # create a new factor level called "unknown" to account for NAs in factors, except for the outcome (response can't be NA)
  step_dummy(all_nominal(), one_hot = TRUE) %>% # turn all factors into a one-hot coding
  prep(training = hotel_train) # learn all the parameters of preprocessing on the training data

hotel_train_final <- bake(cake, new_data = hotel_train) # apply preprocessing to training data
hotel_validate_final <- bake(cake, new_data = hotel_validate) # apply preprocessing to validation data
hotel_test_final <- bake(cake, new_data = hotel_test) # apply preprocessing to testing data

library("keras")
hotel_train_x <- hotel_train_final %>%
  select(-starts_with("is_canceled")) %>%
  as.matrix()
hotel_train_y <- hotel_train_final %>%
  select(is_canceled_NO) %>%
  as.matrix()

hotel_validate_x <- hotel_validate_final %>%
  select(-starts_with("is_canceled")) %>%
  as.matrix()
hotel_validate_y <- hotel_validate_final %>%
  select(is_canceled_NO) %>%
  as.matrix()

hotel_test_x <- hotel_test_final %>%
  select(-starts_with("is_canceled")) %>%
  as.matrix()
hotel_test_y <- hotel_test_final %>%
  select(is_canceled_NO) %>%
  as.matrix()


deep.net <- keras_model_sequential() %>%
  layer_dense(units = 32, activation = "relu",
              input_shape = c(ncol(hotel_train_x))) %>%
  #layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

#deeper
deep.net <- keras_model_sequential() %>%
  layer_dense(units = 128, activation = "relu",
              input_shape = c(ncol(hotel_train_x))) %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")
deep.net

deep.net %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(),
  metrics = c("accuracy")
)

deep.net %>% fit(
  hotel_train_x, hotel_train_y,
  epochs = 50, batch_size = 32,
  validation_data = list(hotel_validate_x, hotel_validate_y),
)

# To get the probability predictions on the test set:
pred_test_prob <- deep.net %>% predict_proba(hotel_test_x)

# To get the raw classes (assuming 0.5 cutoff):
pred_test_res <- deep.net %>% predict_classes(hotel_test_x)

table(pred_test_res, hotel_test_y)
yardstick::accuracy_vec(as.factor(hotel_test_y),
                        as.factor(pred_test_res))
yardstick::roc_auc_vec(factor(hotel_test_y, levels = c("1","0")),
                       c(pred_test_prob))



