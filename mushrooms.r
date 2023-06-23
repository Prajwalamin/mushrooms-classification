# Setting working directory
setwd("/Users/prajwalamin/Documents/MSc Data Science And Analytics/Sem 2/Statistical Learning/Practicals/Assessed Practical 3")

# DEPENDENCIES

# Installing packages
install.packages("randomForest")
install.packages("glmnet")
install.packages("rattle")
install.packages("MLmetrics")

# Loading packages
library(tidyverse)
library(caret)
library(caTools)
library(rpart)
library(rpart.plot)
library(ggplot2)
library(randomForest)
library(glmnet)
library(rattle)
library(MLmetrics)


# DATA COLLECTION

    # Loading dataset
    mush <- read.csv("mushrooms.csv")

# DATA EXPLORATION
    
    #bargraph
    ggplot(mush, aes(x = Edible, fill = Edible)) +
      geom_bar(stat = "count", width = 0.5) +
      scale_fill_manual(values = c("#018571", "#a6611a"))
    
    #jitterplot 'CapShape' 
    ggplot(mush, aes(x = CapSurface, y = CapShape, col = Edible)) + 
      geom_jitter(alpha = 0.5) +
      scale_color_manual(values = c("#0571b0", "#ca0020"))
    
    #jitterplot that shows important feature 'Odor'
    ggplot(mush, aes(x = Edible, y = Odor, col = Edible)) + 
      geom_jitter(alpha = 0.5) +
      scale_color_manual(values = c("#0571b0", "#ca0020"))


# DATA PRE-PROCESSING

    # Converting output to 0 or 1
    mush$Edible <- ifelse(mush$Edible == 'Edible', 1, 0)
    # Converting the columns to factor
    mush[] <- lapply(mush, as.factor)


# MODEL TRAINING

    # Splitting the data
    set.seed(123)
    splitIndex <- createDataPartition(mush$Edible, p = .7, list = FALSE, times = 1)
    train <- mush[splitIndex,]
    test  <- mush[-splitIndex,]


    # 1. Logistic Regression

        # Finding the best tuning parameters

        # Prepare matrix of predictor variables and response variable
        x_train <- model.matrix(Edible ~ .-1, data = train) 
        y_train <- train$Edible

        # Define grid of alpha values
        alpha_grid <- seq(0, 1, by = 0.1)

        # Create empty vector to store best lambda for each alpha
        best_lambdas <- numeric(length(alpha_grid))

        # Outer loop to cross-validate alpha
        for (i in 1:length(alpha_grid)) {
          alpha <- alpha_grid[i]

          # Perform cross-validation for lambda
          cvfit <- cv.glmnet(x_train, y_train, family = "binomial", alpha = alpha)

          ## Check if lambda.min exists
          if (!is.null(cvfit$lambda.min)) {
            # Store the best lambda for this alpha
            best_lambdas[i] <- cvfit$lambda.min
          } else {
            # Handle case when lambda.min does not exist
            best_lambdas[i] <- NA
          }
        }

        # Find alpha with lowest cross-validation error
        best_alpha <- alpha_grid[which.min(best_lambdas)]

        # Fit final model with best alpha and corresponding best lambda
        best_lambda <- best_lambdas[which.min(best_lambdas)]

        best_alpha
        best_lambda

        # Training the model

        # Refit the model using the best lambda
        logit_model <- glmnet(x_train, y_train, family =  "binomial", alpha = best_alpha,
         lambda = best_lambda)

        # Plotting the parameter values
        plot(cvfit)

        # Predictions

       

        # Logistic Regression Predictions
        logit_pred_probs <- predict(logit_model, newx = x_test, type = "response")
        
        logit_preds <- ifelse(logit_pred_probs > 0.5, 1, 0)
  

    # Evaluation!
        # 1. Accuracy
        log_accuracy <- mean((logit_preds) == y_test)
        log_accuracy

        # 2. Confusion Matrix
        logit_preds <- as.factor(logit_preds)
        con_log <- confusionMatrix(logit_preds, y_test)
        con_log
        f1_score_lr <- 2 * con_log$byClass["Sensitivity"] * con_log$byClass["Pos Pred Value"] / (con_log$byClass["Sensitivity"] + con_log$byClass["Pos Pred Value"])
        f1_score_lr
    


    # 2. Decision Trees

        # Finding the optimal tuning parameter

        # Define a grid of parameters to search over
        cp_grid <- expand.grid(.cp = seq(0.01, 0.5, 0.01))

        # Set up the training control for cross-validation
        ctrl <- trainControl(method = "cv", number = 5)  # 5-fold cross-validation

        # Perform cross-validation tuning of the decision tree model
        tuned_tree <- train(Edible ~ .,
                             data = train,
                             method = "rpart",
                             trControl = ctrl,
                             tuneGrid = cp_grid)

        # Retrieve the optimal value of cp
        optimal_cp <- tuned_tree$bestTune$cp
        optimal_cp

        # Print the tuned model
        print(tuned_tree)

        # Visualizing the tree
        fancyRpartPlot(tuned_tree$finalModel)

        # Training the model with the optimal tuning parameter
        final_tree <- rpart(Edible ~ .,
                            data = train,
                            control = rpart.control(cp = optimal_cp),
                            method = "class")

        fancyRpartPlot(final_tree)
        prp(final_tree, box.palette = "auto")


        # Predictions
        tree_preds <- predict(final_tree, newdata = test, type = "class")

        # Evaluation!

        # 1. Accuracy
        tree_accuracy <- mean((tree_preds) == test$Edible)
        tree_accuracy

        # 2. Confusion Matrix
        con_dt <- confusionMatrix(tree_preds, test$Edible)
        f1_score_dt <- 2 * cf_dt$byClass["Sensitivity"] * cf_dt$byClass["Pos Pred Value"] / (con_dt$byClass["Sensitivity"] + con_dt$byClass["Pos Pred Value"])
        f1_score_dt    

        con_rf

    # 3. Random Forests

        # Find the best tuning parameter

        outcome <- train$Edible
        input <- train[,-1]


        best_mtry <- tuneRF(input,
                            outcome,
                            ntreeTry = 1000,
                            stepFactor = 1.5,
                            improve = 0.01,
                            trace = TRUE,
                            plot = TRUE)

        best_mtry <- best_mtry[1,1]
        best_mtry

        # Training the model with the optimal tuning parameter
        final_rf <- randomForest(Edible ~ .,
                                data = train,
                                ntree = 1000,
                                mtry = best_mtry,
                                importance = TRUE)

        # Predictions
        rf_preds <- predict(final_rf, newdata = test)

        # Evaluation!
            # 1. Accuracy
            rf_accuracy <- mean((rf_preds) == test$Edible)
            rf_accuracy

            # 2. Confusion Matrix
            con_rf <- confusionMatrix(rf_preds, test$Edible)
            f1_score_rf <- 2 * con_rf$byClass["Sensitivity"] * con_rf$byClass["Pos Pred Value"] / (con_rf$byClass["Sensitivity"] + con_rf$byClass["Pos Pred Value"])


# MODEL SELECTION

    # Cross Validation
    
    # Define k (number of folds)
    k <- 10

    set.seed(123)
    # Create k folds
    folds <- createFolds(train$Edible, k = 10)

    # Initialization the necessary variables

    winner <- rep(NA, k)
    accuracy <- rep(NA, 3)
    f1_score <- rep(NA, 3)
    model_names <- c("Logistic Regression", "Decision Tree", "Random Forests")
    f1_scores <- matrix(NA, nrow = k, ncol = 3, dimnames = list(NULL, model_names))

    for (iteration in 1:k){
        # Create train and test out of the folds
        train_cv <- train[folds[[iteration]], ]
        test_cv <- train[-folds[[iteration]], ]

        # Logistic Regression

            # Convert data to the format required by glmnet
                x_traincv <- model.matrix(Edible ~ . - 1, train_cv)
                y_traincv <- train_cv$Edible
                x_testcv <- model.matrix(Edible ~ . - 1, test_cv)
                y_testcv <- test_cv$Edible

            current_model_glmnet <- glmnet(x_traincv, y_traincv,
                                  family =  "binomial",
                                  alpha = best_alpha,
                                  lambda = best_lambda)

            # Make predictions
            predictions_glmnet <- predict(current_model_glmnet,
                                          newx = x_testcv, 
                                          type = "response")
            predictions_glmnet <- ifelse(predictions_glmnet > 0.5, 1, 0)
            predictions_glmnet <- as.factor(predictions_glmnet)

            # Calculate accuracy
            accuracy[1] <- mean((predictions_glmnet) == y_testcv)
            cf_log <- confusionMatrix(predictions_glmnet, y_testcv)
            f1_score[1] <- 2 * cf_log$byClass["Sensitivity"] * cf_log$byClass["Pos Pred Value"] / (cf_log$byClass["Sensitivity"] + cf_log$byClass["Pos Pred Value"])

        # Decision Tree
            current_model_dt <- rpart(Edible ~ .,
                                data = train_cv,
                                control = rpart.control(cp = optimal_cp),
                                method = "class")

            # Make predictions
            predictions_dt <- predict(current_model_dt,
                                      newdata = test_cv,
                                      type = "class")

            # Calculate accuracy
            accuracy[2] <- mean((predictions_dt) == y_testcv)
            cf_dt <- confusionMatrix(predictions_dt, test_cv$Edible)
            f1_score[2] <- 2 * cf_dt$byClass["Sensitivity"] * cf_dt$byClass["Pos Pred Value"] / (cf_dt$byClass["Sensitivity"] + cf_dt$byClass["Pos Pred Value"])

        # Random Forest
            current_model_rf <- randomForest(Edible ~ .,
                                          data = train_cv,
                                          ntree = 1000,
                                          mtry = best_mtry,
                                          importance = TRUE)

            # Make predictions
            predictions_rf <- predict(current_model_rf, 
                                      newdata = test_cv)

            # Calculate accuracy
            accuracy[3] <- mean((predictions_rf) == y_testcv)
            cf_rf <- confusionMatrix(predictions_rf, test_cv$Edible)
            f1_score[3] <- 2 * cf_rf$byClass["Sensitivity"] * cf_rf$byClass["Pos Pred Value"] / (cf_rf$byClass["Sensitivity"] + cf_rf$byClass["Pos Pred Value"])

        # Find the winning model for this iteration
        #winner[iteration] <- model_names[which.max(accuracy)]
        winner[iteration] <- model_names[which.max(f1_score)]


        # Storing the f1 scores for every model
        f1_scores[iteration, ] <- f1_score

    }
    
    f1_scores

     # Comparing the f1 score 
        avg_f1_LR <- mean(f1_scores[, 1])
        avg_f1_DT <- mean(f1_scores[, 2])
        avg_f1_RF <- mean(f1_scores[, 3])

        # Organize the data into a data frame
        avg_f1 <- data.frame(
          Model = c("Logistic Regression", "Decision Tree", "Random Forest"),
          F1_Score = c(avg_f1_LR, avg_f1_DT, avg_f1_RF)
        )
        avg_f1
            
        library(ggplot2)

        # Create the bar plot
        # Create the point plot
        ggplot(avg_f1, aes(x = Model, y = F1_Score, color = Model)) +
        geom_point(size = 5) +
        geom_text(aes(label = round(F1_Score, 4)), vjust = -2)+
        geom_segment(aes(x = Model, xend = Model, y = 0, yend = F1_Score), linetype = "dotted") +
        theme_minimal() +
        labs(y = "Average F1 Score", color = "Model")+
        ylim(0.95, 1)

        
    # Performing paired t-test

        # Define the indices for LR, RF, and DT in the f1_scores matrix
        idx_LR <- 1  # Assuming LR is the first column
        idx_DT <- 2  # Assuming DT is the second column
        idx_RF <- 3  # Assuming RF is the third column

        # Perform the paired t-test between each pair
        ttest_result_LR_RF <- t.test(f1_scores[, idx_LR], f1_scores[, idx_RF], paired = TRUE)$p.value
        ttest_result_LR_DT <- t.test(f1_scores[, idx_LR], f1_scores[, idx_DT], paired = TRUE)$p.value
        ttest_result_RF_DT <- t.test(f1_scores[, idx_RF], f1_scores[, idx_DT], paired = TRUE)$p.value

        # Print the results
        cat("The p-value from the paired t-test between LR and RF is", ttest_result_LR_RF, "\n")
        cat("The p-value from the paired t-test between LR and DT is", ttest_result_LR_DT, "\n")
        cat("The p-value from the paired t-test between RF and DT is", ttest_result_RF_DT, "\n")



       
