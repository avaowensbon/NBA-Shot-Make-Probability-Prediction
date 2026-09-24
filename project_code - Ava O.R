# Analyst Intern Project
# Name: Ava Owens-Bonheur
# Number: 8182

# Install all necessary packages 
install.packages(c(
  "tidyverse",
  "tidymodels",
  "glmnet",
  "xgboost",
  "ranger",
  "janitor"
))

# Load packages
library(tidyverse)
library(tidymodels)
library(glmnet)
library(xgboost)
library(ranger)
library(janitor)

# ================================================================
# Set random seed to insure it is reproducible
set.seed(123)

# ================================================================
# Load data 
train <- read_csv("training.csv.gz")
test <- read_csv("testing.csv.gz")

# ===============================================================
# Dimensions
dim(train) # 425719 x 19
dim(test) # 213977 x 28

# ================================================================
# Visual/Basic stats of data set
head(train)
head(test)
summary(train)
summary(test)

# Check outcome variable/proportions
table(train$outcome) # False: 230710, True: 195009
prop.table(table(train$outcome)) # False: .5419302, True: .4580698

# Visualization: 
barplot(
  table(train$outcome),
  main = "Shot Outcomes",
  xlab = "Shot Made",
  ylab = "Number of Shots"
) 

# About 54% of shots missed, 46% of shots made

# =================================================================
# Data Audit
# =================================================================

# Making sure R recognizes values
str(train)
str(test)

# Checking for missing values
missing_train <- colSums(is.na(train))
missing_test <- colSums(is.na(test))

missing_train
missing_test
# Pattern of missingness in both data sets are similar! (Mostly contester&distance values)

# Percentages
missing_train_pct <- 100 * missing_train / nrow(train)
missing_test_pct <- 100 * missing_test / nrow(test)

missing_train_pct
missing_test_pct

# Check for duplicates
sum(duplicated(train)) # 0
sum(duplicated(test)) # 0

# Checking for outliers - Train 
range(train$distance, na.rm = TRUE)
range(train$locationx, na.rm = TRUE)
range(train$locationy, na.rm = TRUE)
range(train$shotclock, na.rm = TRUE)
range(train$shooterspeed, na.rm = TRUE)
range(train$dribblesbefore, na.rm = TRUE)
range(train$closestdefdist, na.rm = TRUE)
range(train$closestdefapproach, na.rm = TRUE)

# Checking for outliers - Test
range(test$distance, na.rm = TRUE)
range(test$locationx, na.rm = TRUE)
range(test$locationy, na.rm = TRUE)
range(test$shotclock, na.rm = TRUE)
range(test$shooterspeed, na.rm = TRUE)
range(test$dribblesbefore, na.rm = TRUE)
range(test$closestdefdist, na.rm = TRUE)
# =================================================================
# Explore data
summary(train$distance)
summary(train$locationx)
summary(train$locationy)
summary(train$shotclock)
summary(train$shooterspeed)
summary(train$num_contesters)

# Categorical variable distributions
table(train$month)
table(test$month)

table(train$gamestate)
table(test$gamestate)

table(train$shottype)
table(test$shottype)

table(train$contested)
table(test$contested)

table(train$three)
table(test$three)
summary(train$closestdefapproach)

# Numerical summaries
summary(train[, c(
  "distance",
  "locationx",
  "locationy",
  "shotclock",
  "shooterspeed",
  "dribblesbefore",
  "closestdefdist",
  "num_contesters"
)])

# Numerical variable distributions
hist(train$distance,
     breaks = 40,
     main = "Distribution of Shot Distance",
     xlab = "Distance")

hist(train$shotclock,
     breaks = 25,
     main = "Distribution of Shot Clock",
     xlab = "Seconds Remaining")

hist(train$shooterspeed,
     breaks = 40,
     main = "Distribution of Shooter Speed",
     xlab = "Shooter Speed",
     col = "red")

hist(train$num_contesters,
     breaks = 5,
     main = "Number of Contesters",
     xlab = "Number of Defenders",
     col = "red")

# Does shot distance appear to be related to whether the shot goes in?
boxplot(distance ~ outcome,
        data = train,
        main = "Shot Distance by Outcome",
        xlab = "Shot Made",
        ylab = "Distance")

boxplot(shotclock ~ outcome,
        data = train,
        main = "Shot Clock by Outcome",
        xlab = "Shot Made",
        ylab = "Seconds Remaining")

# Percentage by shot type
prop.table(table(train$shottype, train$outcome), 1)
prop.table(table(train$three, train$outcome), 1)
prop.table(table(train$contested, train$outcome), 1)
prop.table(table(train$gamestate, train$outcome), 1)

# How does shot distance relate to the probability of making a shot?
train$distance_group <- cut(
  train$distance,
  breaks = c(0, 5, 10, 15, 20, 25, 30, 40, 100),
  include.lowest = TRUE
)

distance_make_rate <- aggregate(
  outcome ~ distance_group,
  data = train,
  mean
)

distance_make_rate

plot(
  distance_make_rate$distance_group,
  distance_make_rate$outcome,
  type = "b",
  xlab = "Shot Distance",
  ylab = "Make Probability",
  main = "Shot Make Probability by Distance"
)

# Split training data 
set.seed(123)

data_split <- rsample::initial_split(
  train,
  prop = 0.80,
  strata = outcome
)

train_data <- rsample::training(data_split)
valid_data <- rsample::testing(data_split)

# ============================================================
# Step 8: First prediction model
# ============================================================
# Model 1: Predict shot outcome using distance only

# Can distance alone predict whether a shot is made?
model1 <- glm(
  outcome ~ distance,
  data = train_data,
  family = binomial
)

summary(model1)

# Considering other factors
model2 <- glm(
  outcome ~ distance + locationx + locationy +
    shooterspeed + dribblesbefore + closestdefdist +
    num_contesters + contested + three + shottype,
  family = binomial,
  data = train_data
)

summary(model2)
# Model 2 fits the training data better than Model 1

valid_data$pred_prob2 <- predict(
  model2,
  newdata = valid_data,
  type = "response"
)

# Prediction
valid_data$pred_prob <- predict(
  model1,
  newdata = valid_data,
  type = "response"
)

summary(valid_data$pred_prob2)

valid_data$pred_outcome2 <- valid_data$pred_prob2 >= 0.5

cm2 <- table(
  Actual = valid_data$outcome,
  Predicted = valid_data$pred_outcome2
)

cm2

TN2 <- cm2["FALSE", "FALSE"]
FP2 <- cm2["FALSE", "TRUE"]
FN2 <- cm2["TRUE", "FALSE"]
TP2 <- cm2["TRUE", "TRUE"]

accuracy2 <- (TP2 + TN2) / sum(cm2)

sensitivity2 <- TP2 / (TP2 + FN2)

specificity2 <- TN2 / (TN2 + FP2)

precision2 <- TP2 / (TP2 + FP2)

accuracy2
sensitivity2
specificity2
precision2

hist(
  valid_data$pred_prob2,
  main = "Model 2: Predicted Probability of Making a Shot",
  xlab = "Predicted Probability"
)
# Model 2 has higher accuracy and precision

# Convert predicted probabilities to predicted outcomes
valid_data$pred_outcome <- valid_data$pred_prob >= 0.5

# Confusion matrix
table(
  Actual = valid_data$outcome,
  Predicted = valid_data$pred_outcome
)

# Calculate accuracy
mean(valid_data$pred_outcome == valid_data$outcome)

# Model Performance Metrics
cm <- table(
  Actual = valid_data$outcome,
  Predicted = valid_data$pred_outcome
)

TN <- cm["FALSE", "FALSE"]
FP <- cm["FALSE", "TRUE"]
FN <- cm["TRUE", "FALSE"]
TP <- cm["TRUE", "TRUE"]

accuracy <- (TP + TN) / sum(cm)
sensitivity <- TP / (TP + FN)
specificity <- TN / (TN + FP)
precision <- TP / (TP + FP)

# Summary
accuracy
sensitivity
specificity
precision

summary(model1) # As shot distance increases, 
# the predicted probability of making the shot decreases
# Strong evidence of an association between distance and shot outcome

# Odds ratio for distance
exp(coef(model1)) # 1-unit increase in shot distance, 
# the odds of making the shot are multiplied by about 0.957 
# Or decrease by about 4.3%

summary(valid_data$pred_prob)

hist(
  valid_data$pred_prob,
  main = "Predicted Probability of Making a Shot",
  xlab = "Predicted Probability"
)

# Installing pROC to evaluate models
install.packages("pROC")
library(pROC)

roc1 <- roc(valid_data$outcome, valid_data$pred_prob)
auc1 <- auc(roc1)
auc1

roc2 <- roc(valid_data$outcome, valid_data$pred_prob2)
auc2 <- auc(roc2)
auc2

# Comparison plot 
plot(
  roc1,
  main = "ROC Curves: Model 1 vs Model 2",
  col = "blue"
)

lines(
  roc2,
  col = "red"
)

legend(
  "bottomright",
  legend = c("Model 1", "Model 2"),
  col = c("blue", "red"),
  lty = 1
)

# Log loss for Model 1
p1 <- pmin(pmax(valid_data$pred_prob, 1e-15), 1 - 1e-15)

logloss1 <- -mean(
  ifelse(
    valid_data$outcome == TRUE,
    log(p1),
    log(1 - p1)
  )
)

logloss1

# Log loss for Model 2
p2 <- pmin(pmax(valid_data$pred_prob2, 1e-15), 1 - 1e-15)

logloss2 <- -mean(
  ifelse(
    valid_data$outcome == TRUE,
    log(p2),
    log(1 - p2)
  )
)

logloss2

# XGBoost model
library(xgboost)
names(train_data)

xgb_formula <- outcome ~ distance + locationx + locationy +
  shooterspeed + dribblesbefore + closestdefdist +
  num_contesters + contested + three + shottype

x_train <- model.matrix(xgb_formula, data = train_data)[, -1]
x_valid <- model.matrix(xgb_formula, data = valid_data)[, -1]

y_train <- as.numeric(train_data$outcome == TRUE)
y_valid <- as.numeric(valid_data$outcome == TRUE)

xgb_train <- xgb.DMatrix(
  data = x_train,
  label = y_train
)

xgb_valid <- xgb.DMatrix(
  data = x_valid,
  label = y_valid
)

model3 <- xgb.train(
  data = xgb_train,
  nrounds = 100,
  objective = "binary:logistic",
  eval_metric = "logloss",
  max_depth = 6,
  eta = 0.1,
  subsample = 0.8,
  colsample_bytree = 0.8
)

valid_data$pred_prob3 <- predict(
  model3,
  xgb_valid
)

summary(valid_data$pred_prob3)

importance_matrix <- xgb.importance(
  feature_names = colnames(x_train),
  model = model3
)

xgb.plot.importance(
  importance_matrix = importance_matrix,
  top_n = 10,
  main = "XGBoost Feature Importance"
)

# Calculate AUC
roc3 <- roc(
  valid_data$outcome,
  valid_data$pred_prob3
)

auc3 <- auc(roc3)

auc3

p3 <- pmin(pmax(valid_data$pred_prob3, 1e-15), 1 - 1e-15)

logloss3 <- mean(
  ifelse(
    valid_data$outcome == TRUE,
    -log(p3),
    -log(1 - p3)
  )
)

logloss3

valid_data$pred_outcome3 <- valid_data$pred_prob3 >= 0.5

# Confusion Matrix
cm3 <- table(
  Actual = valid_data$outcome,
  Predicted = valid_data$pred_outcome3
)

cm3

TN3 <- cm3["FALSE", "FALSE"]
FP3 <- cm3["FALSE", "TRUE"]
FN3 <- cm3["TRUE", "FALSE"]
TP3 <- cm3["TRUE", "TRUE"]

accuracy3 <- (TP3 + TN3) / sum(cm3)

sensitivity3 <- TP3 / (TP3 + FN3)

specificity3 <- TN3 / (TN3 + FP3)

precision3 <- TP3 / (TP3 + FP3)

accuracy3
sensitivity3
specificity3
precision3

# Compare Model Performance

model_comparison <- data.frame(
  Model = c("Model 1", "Model 2", "Model 3"),
  AUC = c(auc1, auc2, auc3),
  LogLoss = c(logloss1, logloss2, logloss3),
  Accuracy = c(accuracy, accuracy2, accuracy3),
  Sensitivity = c(sensitivity, sensitivity2, sensitivity3),
  Specificity = c(specificity, specificity2, specificity3),
  Precision = c(precision, precision2, precision3)
)

model_comparison

# View predicted probabilities
summary(test$pred_prob3)

# Create XGBoost test matrix 
x_test <- model.matrix(
  delete.response(terms(xgb_formula)),
  data = test
)[, -1]

# Convert test data to XGBoost matrix
xgb_test <- xgb.DMatrix(data = x_test)

# Predict probabilities
test$pred_prob3 <- predict(model3, xgb_test)

xgb_test <- xgb.DMatrix(data = x_test)

test$pred_prob3 <- predict(model3, xgb_test)

test$pred_outcome3 <- test$pred_prob3 >= 0.5

# View Model 3 test predictions
head(test[, c("pred_prob3", "pred_outcome3")])

# Summary of predicted probabilities
summary(test$pred_prob3)

# Number of predicted makes and misses
table(test$pred_outcome3)

# Save Prediction
write.csv(
  test[, c("pred_prob3", "pred_outcome3")],
  "model3_test_predictions.csv",
  row.names = FALSE
)

head(test[, c("pred_prob3", "pred_outcome3")])
summary(test$pred_prob3)
table(test$pred_outcome3)

# ROC curve for Model 3 
roc3 <- roc(
  valid_data$outcome,
  valid_data$pred_prob3
)

auc3 <- auc(roc3)

auc3

plot(
  roc1,
  col = "blue",
  main = "ROC Curves: Model 1 vs Model 2 vs Model 3"
)

lines(roc2, col = "red")
lines(roc3, col = "green")

legend(
  "bottomright",
  legend = c("Model 1", "Model 2", "Model 3"),
  col = c("blue", "red", "green"),
  lwd = 2
)

# Load submission template
submission <- read.csv("submission.csv")

head(submission)
names(submission)
nrow(submission)

# Fill submission template with Model 3 predictions
submission$make_prob <- test$pred_prob3

# Check the completed submission
head(submission)
summary(submission$make_prob)

# Verify row counts match
nrow(submission)
length(test$pred_prob3)

# Save completed submission file
write.csv(
  submission,
  "submission.csv",
  row.names = FALSE
)

submission_check <- read.csv("submission.csv")

head(submission_check)