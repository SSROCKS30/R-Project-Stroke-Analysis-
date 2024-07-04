# Load necessary libraries
library(dplyr)
library(ggplot2)
library(caret)
library(cluster)

# Load the dataset
data <- read.csv("healthcare-dataset-stroke-data.csv")

# View the structure of the dataset
str(data)

# Data Preprocessing

# Handle missing values in 'bmi' column (we'll replace N/A with the median)
data$bmi <- as.numeric(data$bmi)
data$bmi[is.na(data$bmi)] <- median(data$bmi, na.rm = TRUE)

# Convert categorical variables to factors
data$gender <- as.factor(data$gender)
data$ever_married <- as.factor(data$ever_married)
data$work_type <- as.factor(data$work_type)
data$Residence_type <- as.factor(data$Residence_type)
data$smoking_status <- as.factor(data$smoking_status)
data$stroke <- as.factor(data$stroke)

# Split the data into training and testing sets
set.seed(123) # For reproducibility
trainIndex <- createDataPartition(data$stroke, p = .8, 
                                  list = FALSE, 
                                  times = 1)
strokeTrain <- data[trainIndex,]
strokeTest  <- data[-trainIndex,]

# Train a logistic regression model
model <- glm(stroke ~ age + hypertension + heart_disease + ever_married +
               avg_glucose_level + bmi + gender + work_type + 
               Residence_type + smoking_status, 
             family = binomial, data = strokeTrain)

# Summarize the model
summary(model)

# Predict on test data
predictions <- predict(model, strokeTest, type = "response")
predicted_classes <- ifelse(predictions > 0.5, 1, 0)

# Evaluate the model
confusionMatrix(as.factor(predicted_classes), strokeTest$stroke)

# Visualization

# Age distribution by stroke status
ggplot(data, aes(x = age, fill = stroke)) + 
  geom_histogram(binwidth = 5, position = "dodge") +
  labs(title = "Age Distribution by Stroke Status", x = "Age", y = "Count")

# Average glucose level by stroke status
ggplot(data, aes(x = stroke, y = avg_glucose_level, fill = stroke)) + 
  geom_boxplot() +
  labs(title = "Average Glucose Level by Stroke Status", x = "Stroke", y = "Average Glucose Level")

# BMI distribution by stroke status
ggplot(data, aes(x = bmi, fill = stroke)) + 
  geom_histogram(binwidth = 1, position = "dodge") +
  labs(title = "BMI Distribution by Stroke Status", x = "BMI", y = "Count")

# Hypertension by stroke status
ggplot(data, aes(x = factor(hypertension), fill = stroke)) + 
  geom_bar(position = "dodge") +
  labs(title = "Hypertension by Stroke Status", x = "Hypertension", y = "Count")

# Heart disease by stroke status
ggplot(data, aes(x = factor(heart_disease), fill = stroke)) + 
  geom_bar(position = "dodge") +
  labs(title = "Heart Disease by Stroke Status", x = "Heart Disease", y = "Count")

# Smoking status by stroke status
ggplot(data, aes(x = smoking_status, fill = stroke)) + 
  geom_bar(position = "dodge") +
  labs(title = "Smoking Status by Stroke Status", x = "Smoking Status", y = "Count")

# Clustering

# Select relevant features for clustering
clustering_data <- data %>% 
  select(age, hypertension, heart_disease, avg_glucose_level, bmi)

# Scale the data
clustering_data_scaled <- scale(clustering_data)

# Determine the optimal number of clusters using the Elbow method
wss <- (nrow(clustering_data_scaled) - 1) * sum(apply(clustering_data_scaled, 2, var))
for (i in 2:15) wss[i] <- sum(kmeans(clustering_data_scaled, centers = i)$withinss)

# Plot the Elbow method
plot(1:15, wss, type = "b", xlab = "Number of Clusters", ylab = "Within groups sum of squares", main = "Elbow Method for Optimal Clusters")

# Perform K-means clustering with the chosen number of clusters (e.g., 3)
set.seed(123) # For reproducibility
kmeans_result <- kmeans(clustering_data_scaled, centers = 3)

# Add cluster assignments to the original data
data$cluster <- as.factor(kmeans_result$cluster)

# Visualization of clusters

# Clusters by age and average glucose level
ggplot(data, aes(x = age, y = avg_glucose_level, color = cluster)) + 
  geom_point() +
  labs(title = "Clusters by Age and Average Glucose Level", x = "Age", y = "Average Glucose Level")

# Clusters by BMI and average glucose level
ggplot(data, aes(x = bmi, y = avg_glucose_level, color = cluster)) + 
  geom_point() +
  labs(title = "Clusters by BMI and Average Glucose Level", x = "BMI", y = "Average Glucose Level")
