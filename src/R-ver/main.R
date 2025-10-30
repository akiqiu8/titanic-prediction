library(dplyr)

cat("=== Loading and preparing training data ===\n")

# Load training data
df <- read.csv("data/train.csv", stringsAsFactors = FALSE)
cat("Train shape before cleaning:", dim(df), "\n")

# Data cleaning and feature engineering
df <- df %>%
  select(-Cabin) %>%
  filter(complete.cases(.)) %>% 
  mutate(
    LogFare = log(Fare + 1),
    Pclass = factor(Pclass),
    Sex = factor(Sex),
    Embarked = factor(Embarked)
  )

cat("Train shape after cleaning:", dim(df), "\n")

# Split 80/20
set.seed(400)
train_index <- sample(seq_len(nrow(df)), 0.8 * nrow(df))
train_data <- df[train_index, ]
test_data  <- df[-train_index, ]

# Fit logistic regression (base R glm)
cat("=== Training logistic regression ===\n")
model <- glm(
  Survived ~ Age + LogFare + SibSp + Parch + Pclass + Sex + Embarked,
  data = train_data,
  family = binomial
)

# Predict on test_data (split from train.csv)
y_pred_prob <- predict(model, newdata = test_data, type = "response")
y_pred <- ifelse(y_pred_prob > 0.5, 1, 0)
acc <- mean(y_pred == test_data$Survived)
cat("Accuracy:", round(acc, 4), "\n")

# Predict on actual test.csv
cat("\n=== Predicting on actual test.csv ===\n")

test <- read.csv("data/test.csv", stringsAsFactors = FALSE)
cat("Loaded test.csv with shape:", dim(test), "\n")

# Clean and feature engineer test data
test <- test %>%
  select(-Cabin) %>%
  mutate(
    Age = ifelse(is.na(Age), mean(df$Age, na.rm = TRUE), Age),
    Fare = ifelse(is.na(Fare), mean(df$Fare, na.rm = TRUE), Fare),
    LogFare = log(Fare + 1),
    Pclass = factor(Pclass, levels = levels(df$Pclass)),
    Sex = factor(Sex, levels = levels(df$Sex)),
    Embarked = factor(Embarked, levels = levels(df$Embarked))
    
  )

# Predict
test_pred_prob <- predict(model, newdata = test, type = "response")
test_pred <- ifelse(test_pred_prob > 0.5, 1, 0)


# Combine PassengerId and predictions into a data frame
pred_df <- data.frame(
  PassengerId = test$PassengerId,
  Survived = test_pred
)

# Save predictions to CSV
write.csv(pred_df, "data/R_test_predictions.csv",row.names = FALSE)
cat("Predictions saved to 'src/data/R_test_predictions.csv'\n")