---
title: "SuperLearner"
output:
  html_document:
    number_sections: yes
    toc: yes
    toc_float: yes
theme: lumen
number_sections: false
---

# Goal

1. Fit single model to train/test sets and evaluate performance
2. Deploy a cross-validated ensemble of multiple algorithms for simultaneous comparison

# SuperLearner

> SuperLearner is an algorithm that uses cross-validation to estimate the performance of multiple machine learning models, or the same model with different settings. It then creates an optimal weighted average of those models, aka an "ensemble", using the test data performance. This approach has been proven to be asymptotically as accurate as the best possible prediction algorithm that is tested. [Guide to SuperLearner - 1 Background](https://cran.r-project.org/web/packages/SuperLearner/vignettes/Guide-to-SuperLearner.html)

In this manner, an ensemble is a powerful tool because it:

1. Eliminates bias of single algorithm selection for framing a research problem

2. Allows for comparison of multiple algorithms, and/or comparison of the same model(s) but tuned in many different ways

3. Utilizes a second-level algorithm that produces an ideal weighted prediction that is suitable for data of virtually all distributions and uses external cross-validation to prevent overfitting. 

**Further Reading:**

- [Van der Laan, M.J.; Polley, E.C.; Hubbard, A.E. Super Learner. Stat. Appl. Genet. Mol. Biol. 2007, 6, 1–21.](https://www.degruyter.com/document/doi/10.2202/1544-6115.1309/html)

- [Polley, E.C.; van der Laan, M.J. Super Learner in Prediction, UC Berkeley Division of Biostatistics Working Paper Series Paper 266.](https://biostats.bepress.com/ucbbiostat/paper266/)

# Setup

```{r setup, echo = F}
# when you "knit" this file, do you want the resulting PDF to print the code in each chunk (TRUE = yes)?
knitr::opts_chunk$set(echo = TRUE)

################################################################################
# set your working directory
####CHANGE THIS TO THE APPROPRIATE PATH
knitr::opts_knit$set(root.dir = '~/Desktop/Intro_ML_R/')
################################################################################
# note that outside of an Rmd code chunk, use `setwd()` to set the working directory in R
```


## Package installation/loading

Install and/or load the libraries that will be used in this workflow. The `pacman` package management tool will load the necessary libraries in the code chunk below. [Read the documentation here](https://www.rdocumentation.org/packages/pacman/versions/0.5.1).

```{r}
if (!require("pacman")) install.packages("pacman")

pacman::p_load("caret",         # create stratified random split of the dataset
               "ck37r",         # Chris Kennedy's Machine Learning Helper Toolkit
               "ggplot2",       # visualize risk and ROC curve
               "glmnet",        # elastic net algorithm
               "ranger",        # random forest algorithm
               "rpart",         # decision tree algorithm
               "scales",        # for scaling data
               "SuperLearner",  # fit individual algorithms and ensembles
               "ROCR",          # compute AUC-ROC performance
               "xgboost")       # boosted tree algorithm
```

# Import pre-processed dataset

# Wipe your global environment clean

Reload the clean dataset with:

```{r}
load("~/Desktop/Intro_ML_R/data/preprocessed/clean.RData")
str(clean)
```

# Machine learning in practice: How to stay organized

A good way to machine learning worflows organized is by storing the various components in a list. In this example, the list is `heart_class`, for the task of yes/no heart disease classification. 

The first two list components will be the dataset and outcome variable. In the heart dataset, the `target` variable is coded with a 1 if the patient had heart disease and a 0 if they did not.

The list will look like the screenshot below and will contain: 
1. The dataset  
2. Outcome column name  
3. Covariates  
4. Training set row indices  
5. Training set X covariates  
6. Training set Y outcome vector  
7. Test set X covariates  
8. Test set Y outcome vector  
9. Y outcomes for the entire dataset (for cross-validation)  
10. X covariates for the entire dataset (for cross-validation)

![heart_class](~/Desktop/Intro_ML_R/img/heart_class.png)

```{r}
heart_class <- list(
  data = clean, 
  outcome = "target" # Y variable column name
)

# View the list contents
names(heart_class)

# Access part of the list with the dollar sign
head(heart_class$data)
heart_class$outcome
```

## Add covariates

```{r}
# setdiff is a handy base R function to define the covariates as all variables *except* the outcome
heart_class$covariates <- setdiff(names(heart_class$data), heart_class$outcome)

# The covariate names appear in the heart_class list!
names(heart_class)

# Call the covariates with (notice that "target" is excluded)
heart_class$covariates
```

## Add training rows

```{r}
# ?createDataPartition
set.seed(1) 
training_rows <- 
  caret::createDataPartition(heart_class$data[[heart_class$outcome]], 
                             p = 0.70, 
                             list = FALSE)

# assign the row indices
heart_class$train_rows <- training_rows

# We have added the training rows to our list
names(heart_class)
head(heart_class$train_rows)
```

## Split data into training and test sets

```{r}
# use the train_rows indices to add the correct covariates and outcome to the training set
x_train <- heart_class$data[heart_class$train_rows, heart_class$covariates]
y_train <- heart_class$data[heart_class$train_rows, heart_class$outcome]

# note the use of the minus sign here
# this says to use the train_rows indices *not* assigned to the training set in the test set
x_test <- heart_class$data[-heart_class$train_rows, heart_class$covariates]
y_test <- heart_class$data[-heart_class$train_rows, heart_class$outcome]

# Mean and frequencies of training set
mean(heart_class$data[training_rows, "target"])
table(heart_class$data[training_rows, "target"])

# Mean and frequencies of test set
mean(heart_class$data[-training_rows, "target"])
table(heart_class$data[-training_rows, "target"])

# Add these variables to our list for safekeeping
heart_class$x_train <- x_train
heart_class$y_train <- y_train
heart_class$x_test <- x_test
heart_class$y_test <- y_test

# Also save X and Y for the ensemble, since we will use all of the data in the cross-validation process
heart_class$Y <- clean$target
heart_class$X <- clean[,-4] 
names(heart_class)
```

### Save the list `heart_class`

This way, you do not have to repeat all of these steps. You can just `load("data/preprocessed/heart_class.RData")`to load the data along with our preprocessed list!

```{r}
names(heart_class)

save(clean, heart_class, # save the clean dataset as well
     file = "(~/Desktop/Intro_ML_R/data/preprocessed/heart_class.RData")
```

# Again, wipe your global environment clean

```{r}
# Then, load heart_class with the load function!
load("~/Desktop/Intro_ML_R/data/preprocessed/heart_class.RData")
names(heart_class)

# or
# View(heart_class)

# or
# str(heart_class)
```

# Fit single models

It can be good to fit a single model to prototype your machine learning task. 

## View available models

SuperLearner supports a wide variety of useful algorithms:

```{r}
listWrappers()
```

The following sections will fit using three common models: `single decision tree`, `random forest` and `lasso`, one at a time. See the links below for more information:

1. [Single decision tree](https://www.amazon.com/Classification-Regression-Wadsworth-Statistics-Probability/dp/0412048418)

2. [Random forest](https://link.springer.com/article/10.1023/A:1010933404324)

3. [Lasso](https://cran.r-project.org/web/packages/glmnet/vignettes/glmnet.pdf)

## Decision tree

The decision tree algorithm can be useful because it is relatively simple to construct and interpret. It can work with categorical data directly, without the need for one-hot encoding. 

Decision trees work by recursively partitioning, or dividing the feature space into smaller regions that contain less data. Splitting takes place at **nodes** and each directional split is called a **branch**. The top node is called the **root node** and seeks to find the most discriminating split, thus always putting the most optimal partition first. The tree stops partitioning when it fails to meet some hyperparameter condition. 

Tree-based methods work by stratifying or segmenting the predictor space into smaller distinct, non-overlapping regions. In order to make a prediction they use the median or the mode response of the training set observations of the same region or **terminal node**. Decision trees can be applied to both regression and classification problems. 

For a regression tree, the predicted response for an observation is given by the mean response of the training observations that belong to the same **terminal node**. By contrast, for a classification tree, the model predicts that each observation belongs to the most commonly occurring class of training observations in the region to which it belongs. 

Regression decision trees are used to predict a quantitative response, while classification trees, a qualitative one. 

`rpart` R package provides a useful and simple way to grow classification and regression trees. See`?rpart` for additional information and `?rpart.control` for a list of tunings. 

- [A tutorial for a step-by-step decision tree walkthrough](https://www.gormanalysis.com/blog/decision-trees-in-r-using-rpart/) 

- [Constructing a decision tree](https://www.gormanalysis.com/blog/magic-behind-constructing-a-decision-tree/)

- [Stackoverflow thread about decision tree information gain.](https://stackoverflow.com/questions/4553947/decision-tree-on-information-gain)

Keep in mind that single decision trees are prone to overfitting and high variance in results on different training data.

Unhashtag the code below to reproduce the tree figure for the training data.

```{r eval = F}
?rpart
?rpart.control
#install.packages("rpart.plot")
library(rpart.plot)
(dt_example_plot <- rpart(heart_class$y_train ~ .,
                           data = heart_class$x_train,
                           method = "class",
                           parms = list(split = "information")))
rpart.plot(dt_example_plot)
```

![training set decision tree plot](~/Desktop/Intro_ML_R/img/dt_example_plot.png)

### Train the model with SuperLearner

```{r}
set.seed(1)
dt <- SuperLearner(Y = heart_class$y_train, 
                   X = heart_class$x_train, 
                   family = binomial(),
                   SL.library = "SL.rpart")
dt

# Accuracy
(dt_acc <- 1 - dt$cvRisk)
```

### Compute decision tree test set AUC-ROC: 

```{r}
# Predict on test set
dt_predictions = predict(dt, heart_class$x_test, onlySL = TRUE)
str(dt_predictions)
summary(dt_predictions$library.predict)

# Visualize predicted probabilities
qplot(dt_predictions$pred[, 1]) + theme_minimal()

# Compute AUC-ROC and save as a separate object
dt_rocr_score = ROCR::prediction(dt_predictions$pred, heart_class$y_test)
dt_auc_roc = ROCR::performance(dt_rocr_score, measure = "auc", x.measure = "cutoff")@y.values[[1]]

# AUC-ROC
dt_auc_roc
```

Pros & Cons of Decision Trees:

- Trees are simple to explain and interpret visually.
- Trees can be applied to both regression and classification problems.
- Trees can handle qualitative predictors (like categorical variables) directly, without the need for converting to numeric variables.

- Trees do not have the same level of predictive accuracy as other regression or classification approaches.
- Trees are not very robust and small changes in the data can cause large changes in the final estimated tree. 

The main disadvantage of decision trees is that they suffer from high variance. This means that splitting the training data into two parts at random, and fitting a decision tree to both halves, may generate results the are quite different.

## Random forest

The random forest algorithm is an improvement over the single decision tree, and it works by constructing many decision trees at the time of training and averaging performance across the trees. It is therefore in itself an ensemble method. 

This model is called **forest** because it grows multiple decision trees. 

It is called **random** because instead of trying to place the most discriminating split as the root node, it selects a random sample of features to try at each split. This allows for good splits to be followed by bad splits, and vice versa, thus helping decorrelate the average performance estimator for a more statistically robust conclusion. 

It is also **random** because it uses "bootstrapping" with replacement. This is when a new training set is generated by sampling from the main set, uniformly and with replacement: e.g. taking two-thirds of the dataset for each tree (this is called "bootstrap aggregating", or "bagging") and evaluating performance on the remaining one-third of the data (called the "out-of-bag" sample). 


See `?ranger` for hyperparameter tuning options.

```{r eval = F}
?ranger
```

- [Read the ranger paper for explanations and examples.](https://arxiv.org/pdf/1508.04409.pdf). 

![https://en.wikipedia.org/wiki/Random_forest](~/Desktop/Intro_ML_R/img/rf.png)

### Train the model

```{r}
set.seed(1)
rf <- SuperLearner(Y = heart_class$y_train, 
                   X = heart_class$x_train, 
                   family = binomial(),
                   SL.library = "SL.ranger")
rf

# Accuracy
(rf_acc <- 1 - rf$cvRisk)
```

### Test set AUC-ROC:

```{r}
rf_predictions = predict(rf, heart_class$x_test, onlySL = TRUE)
str(rf_predictions)
summary(rf_predictions$library.predict)

# Visualize predicted probabilities
qplot(rf_predictions$pred[, 1]) + theme_minimal()

# Compute AUC-ROC and save as a new object
rf_rocr_score = ROCR::prediction(rf_predictions$pred, heart_class$y_test)
rf_auc_roc = ROCR::performance(rf_rocr_score, measure = "auc", x.measure = "cutoff")@y.values[[1]]

# AUC-ROC 
rf_auc_roc
```

## Lasso

The `Lasso` (Least Absolute Square Shrinkage Operator) is a relatively recent alternative to `Ridge Regression`, which is analogous to `Least Squares Regresson`, where least sum of squares is applied to best fit line and the `Sum of Squared Residuals` (`SSR` or `RSS`) is the measure of error, or deviations of predicted values from actual values. This could result in situations where the `RSS` is lower for the training data but high for test data. This is called "overfitting".  

The advantage of `Ridge Regression` is to reduce overfitting by the addition of a "penalty term" that is introduced as the slope of the least sum of squares line is rotated slightly during training to make the model more "generalized". This is also called "regularization". The idea is that increasing the variance of an overfitted model by tweaking the slope will make it perform slightly worse on the training set but much better on the test set. This is also known as "bias-variance trade-off".

In general, in situations where the relationship between the response and the predictors is close to linear, the `Least Squares Regresson` will have low bias but may have high variance. This means that a small change in the training data can cause a large change in the least squares coefficient estimates. In particular, when the number of variables is almost as large as the number of observations the least squares estimates will be extremely variable. Whereas `Ridge Regression` can still perform well by trading off a small increase in bias for a large decrease in variance.

The "penalizing", or addition of a "penalty term", in essence improves the generalization capability of the model. 

In `Ridge Regression`, the error = RSS + (λ * β^2). The latter is the "penalty term" defined by the β coefficient (or the slope of the line) squared, multiplied by the "regularization" or tuning parameter λ, to be determined separately.   

The `Lasso` is a form of penalized regression and classification that "shrinks" the β coefficient to zero and identifies features that are not related to the outcome variable. This is achieved by taking the absolute value of the β coefficient, rather than squaring it, so the error = RSS + |λ * β|.

To summarize: 

- Least Squares Regression: error = RSS
- Ridge Regression: error = RSS + (λ * β^2)
- Lasso Regression: error = RSS + |λ * β|

`Lasso Regression` can help reduce overfitting, but unlike `Ridge Regression` it can reduce the slope to exactly zero. This can be used for feature selection, where every non-zero value is selected to be used in the model, and variables or paramaters not related to the response variable can be identified and eliminated.

From this, the "one standard error rule" can be applied, which allows selecting a higher λ value (the regularization parameter) within one standard error of the minimum value, and sacrifice a little error for a model that contains less variables but is ostensibly easier to interpret. 

`Lasso` performs best when the number of observations is low and the number of features is high. It heavily relies on parameter λ, which is the controlling factor in shrinkage (where every non-zero value is selected to be used in the model). The larger λ becomes, then the more coefficients are forced to be zero.

View the help files with `?glmnet` and `?cv.glmnet` to learn more.

```{r eval = F}
?glmnet
?cv.glmnet
```

- [Remember to read An Introduction to glmnet](https://cran.r-project.org/web/packages/glmnet/vignettes/glmnet.pdf)

### Lasso - train the model with SuperLearner

```{r}
set.seed(1)
la <- SuperLearner(Y = heart_class$y_train, 
                   X = heart_class$x_train, 
                   family = binomial(),
                   SL.library = "SL.glmnet")
la

# Accuracy
(la_risk <- 1-la$cvRisk)
```

### Lasso test set AUC-ROC:

```{r}
la_predictions = predict(la, heart_class$x_test, onlySL = TRUE)
str(la_predictions)
summary(la_predictions$library.predict)

# Visualize predicted probabilities
qplot(la_predictions$pred[, 1]) + theme_minimal()

# Compute AUC-ROC and save as a new object
la_rocr_score = ROCR::prediction(la_predictions$pred, heart_class$y_test)
la_auc_roc = ROCR::performance(la_rocr_score, measure = "auc", x.measure = "cutoff")@y.values[[1]]

# AUC-ROC 
la_auc_roc
```

### Bonus: cross-validated lasso with `cv.glmnet`

```{r}
set.seed(1)
lasso <- cv.glmnet(x = as.matrix(heart_class$X), 
                   y = heart_class$Y, 
                   family = "binomial", 
                   alpha = 1)
lasso

# View path plot
plot(lasso)

# View coefficients
plot(lasso$glmnet.fit, xvar = "lambda", label = TRUE)

# Show coefficients for 1se model
(coef_min = coef(lasso, s = "lambda.1se"))
```

# Ensemble

In this example, three algorithms have been compared: the decision tree performed relatively poorly compared to the random forest and lasso. So, how do we know which algorithm to choose? Let's fit an ensemble with two more algorithms:

4. [xgboost](https://www.google.com/search?client=firefox-b-1-d&q=short+introduction+to+boosting+freund)

5. [The mean of Y](https://biostats.bepress.com/ucbbiostat/paper266/)

## Summary of algorithm definitions

![https://www.mdpi.com/2227-7080/9/2/23/htm](~/Desktop/Intro_ML_R/img/algorithms.png)

## Train the ensemble of models

```{r}
set.seed(1)
system.time({
cv_sl <- CV.SuperLearner(Y = heart_class$Y,  # heart disease yes/no
                         X = heart_class$X,  # excluding the "target" variable
                         family = binomial(),
                         # For your own research, 5, 10, or 20 are good options
                         cvControl = list(V = 5), 
                         innerCvControl = list(list(V = 5)),
                         verbose = FALSE, 
                         method = "method.NNLS",
                         SL.library = c("SL.rpart",   # decision tree
                                        "SL.ranger",  # random forest
                                        "SL.glmnet",  # lasso
                                        "SL.xgboost", # boosted trees
                                        "SL.mean"))   # Y mean
})
```

## Explore the output

```{r}
# View the function call
cv_sl

# View the risk table for: 
# 1. The individual algorithms
# 2. The discrete winner
# 3. The SuperLearner ensemble
summary(cv_sl)

# View the discrete winner
table(simplify2array(cv_sl$whichDiscreteSL))

# View the AUC-ROC table for: 
# 1. The individual algorithms
# 2. The discrete winner
# 3. The SuperLearner ensemble
ck37r::auc_table(cv_sl)

# Visualize the cross-validated risk
plot.CV.SuperLearner(cv_sl) + theme_bw()

# Print the weights table
cvsl_weights(cv_sl)

# Compute ensemble AUC-ROC 
ck37r::cvsl_auc(cv_sl)

# Plot AUC-ROC curve
ck37r::plot_roc(cv_sl)

# View rough estimates of variable importance
set.seed(1)
var_importance <- ck37r::vim_corr(covariates = heart_class$covariates, 
                                  outcome = heart_class$outcome, 
                                  data = clean, 
                                  bootse = FALSE, 
                                  verbose = TRUE)
var_importance
```

# Challenge

## Customize model hyperparameters and refit the ensemble

See Chapter 9 and Chapter 10 in the [Guide to SuperLearner](https://cran.r-project.org/web/packages/SuperLearner/vignettes/Guide-to-SuperLearner.html) for how to customize model hyperparameters and test the same algorithms with multiple hyperparameter settings. Refit the ensemble with a few new, tuned algorithms and compare to the existing algorithms contained within the `SL.library` parameter, inside of `CV.SuperLearner` above ("SL.rpart", "SL.ranger", "SL.glmnet",  "SL.xgboost", "SL.mean"). 

What changed, and how were you able to optimize performance? 

How did you find out which hyperparameters to tune?

> HINT: type `?rpart`, `?ranger`, etc.

Also, consider the following points for improving performance: 
- Change nominal factors to ordinal type when appropriate  
- Explore other ways to scale numeric variables  
- Include other algorithms  
- Utilize other/more performance metrics  
- [Read more](https://machinelearningmastery.com/machine-learning-performance-improvement-cheat-sheet/)

# Other tutorials

caret: https://topepo.github.io/caret/

tidymodels: https://www.tidymodels.org/

mikropml: http://www.schlosslab.org/mikropml/articles/paper.html

