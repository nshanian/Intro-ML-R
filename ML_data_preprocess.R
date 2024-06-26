---
title: "ML-Data-Preprocessing"
output:
  html_document:
    number_sections: yes
    toc: yes
    toc_float: yes
theme: lumen
number_sections: false
---

# Goal

1. Become familiar with the basic predictive machine learning landscape 
2. Appropriately preprocess the UCI heart disease dataset



# Introduction: What is machine learning? 

Machine learning (or statistical learning) is a combination of statistics and computer science that is often defined as an algorithm or a computer program that can learn on its own, as it is given more data and without human input. It can be thought of as a toolbox for exploring and modeling data with applications in scientific research, such as in making predictions or inferences, or exploring the structure of a dataset. 

Machine learning can be defined as: 

> "a vast set of tools for _understanding data_. These tools can be classified as supervised or unsupervised. Broadly speaking, supervised statistical learning involves building a statistical model for predicting, or estimating, an output based on one or more inputs... With unsupervised statistical learning, there are inputs but no supervising output; nevertheless we can learn relationships and structure from such data." 
> (James et al. 2021, p. 1). 

[James G, Witten D, Hastie T, Tibshirani R. 2021. An Introduction to Statistical Learning: With Applications in R, 2nd edition.](https://www.statlearning.com)

>NOTE: there are other approaches as well, such as: semi-supervised, deep, reinforcement, and targeted/causal. 

## Machine learning terminology

First, it's important to become familiar with the basic vocabulary of relevant terms, to understand how the different parts work independently and together. 

The focus of this workflow is on **supervised learning** for a **classification task** using the [SuperLearner R package](https://cran.r-project.org/web/packages/SuperLearner/index.html). We will follow the [Guide to SuperLearner](https://cran.r-project.org/web/packages/SuperLearner/vignettes/Guide-to-SuperLearner.html). 

1. **Supervised learning:** is a predictive technique that trains a model on known, labeled data. The goal is to understand the relationships between variables in a dataset so that the trained model can be used to make predictions on new data it has not yet seen, and whose labels are unknown. Or, as James et al. (2021, p. 26) state:

> "We wish to fit a model that relates the response to the predictors, with the aim of accurately predicting the response for future observations (prediction) or better understanding the relationship between the response and the predictors (inference)." 

- __Classification:__ is the supervised task when the Y variable is discrete/categorical ($Pr(y=1|x)$). For binary classification, 1 is the "yes", or positive class. 0 is the "no", or negative class. This can also extend to multiclass problems as well. 
  
- __Regression:__ is the supervised task when the Y variable is continuous (i.e., numeric or integer) ($E(y|x)$) 

2. **Supervised syntax**

$Y ~ X1 + X2 + X3 ... + Xn$

Simply put, we want to use the X variables to predict Y. 

3. **X and Y variables:** 

- __X__ is/are the independent variable(s) that we use to do the predicting. These are also referred to as features, covariates, predictors, and explanatory/input variables. 

- __Y__ is the dependent variable and the one we want to predict. It is also referred to as the outcome, target, or response variable. Although predicting the label itself might be convenient, predicting the class probabilities is more efficient. 

- A machine learning function might look like this: 

$y=f(x)+ϵ$

where f is the function to estimate the relationships between __X__ to __Y__. The random error epsilon ϵ is independent of x and averages to zero. We can therefore use $y=f(x)$ to predict __Y__ for new data (the test dataset) and evaluate how well the algorithm learned the target function when trained and then introduced to the new data. 

4. **Data splitting:** Predictions are usually evaluated twice: 

1. First, on the labeled training set, to see how well the model can learn the relationships between the __X__ and __Y__ variables, and then 
2. Then on the test set, to see how well the trained model can generalize to predicting on new data. 

To accomplish this task, a given dataset is subdivided into training and test sets. 

- The **training set:** generally consists of the majority portion of the original dataset (70%, for example) where the model can learn the relationships between the **X** and **Y** variables. 
  - __Hyperparameters:__ are the configurations manually set by the programmer prior to model training through heuristics and trial and error, or a grid search. We do not know the optimal combinations of these options, and must tune the hyperparameters through the model training process to optimize prediction performance. 

- The **test set:** consists of the remaining portion of the dataset (30% in this example) that the trained model will then try to predict without seeing the Y labels. 

> NOTE: A validation set is also sometimes used for hyperparameter tuning/model selection on the training dataset and are to be learned. While the parameters represent the internal configuration of the model, hyperparameters are defined by the user before training begins. 

- **[k-fold cross-validation:](https://en.wikipedia.org/wiki/Cross-validation_(statistics))** is a preferred method for approaching the data splitting process because it repeats the train/test split process "k" number of times and rotates portions of the dataset to ensure that each observation is in the test set at least once. 

![https://en.wikipedia.org/wiki/Cross-validation_(statistics)](~/Desktop/Intro_ML_R/img/cvwiki.png)

5. **Performance metrics:** It is necessary to evaluate model performance on the training and test sets (and validation set, when applicable) through a variety of [confusion matrix derivations](https://en.wikipedia.org/wiki/Confusion_matrix). 

This workflow will focus on two classification metrics: 
1. Risk - as measured by [mean squared error](https://en.wikipedia.org/wiki/Mean_squared_error)
2. [ROC-AUC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve) - Receiver Operator Characteristic - Area Under the Curve; a measure of true positive rate versus true negative rate.

The `ROC-AUC` is a common metric for evaluating the performance of the classification algorithm. “ROC” is a historic acronym that comes from communications theory. It provides a visualization of the relationship between true vs. false positive rates of a given classification algorithm. The figure below shows an example ROC curve for a Linear Discriminant Analysis (LDA) classifier. The overall performance of a classifier, summarized over all possible thresholds, is given by the area under the (ROC) curve (AUC). 

An ideal ROC curve will hug the top left corner, so the larger area under the ROC curve, the AUC, the better the classifier. For this data the AUC is 0.95, which is close to the maximum of 1, and is considered very good. A classifier that performs no better than chance would have an AUC of 0.5 (when evaluated on an independent test set not used in model training). ROC curves are useful for comparing different classifiers, since they take into account all possible thresholds.

![https://en.wikipedia.org/wiki/Confusion_matrix](~/Desktop/Intro_ML_R/img/cmwiki.png)

> Keep in mind that misclassification error is often an inappropriate performance metric, particularly when dealing with a __Y__ variable whose distribution is imbalanced. 

- A model is **underfit** if it cannot learn the relationships between the **X** and **Y** variables on the training set. 

- A model is **overfit** if it adequately learns the relationships between the **X** and **Y** variables and performs well on the training set, but performs poorly on the test set. 

## Example machine learning workflow:

1. Read literature in your field to understand what has been done; annotate what you read
2. Formulate research question(s)
3. Obtain data
4. Preprocess data (often the most time-consuming part of ML workflows!)
- scale variables when necessary
- handle missing values if present (listwise delete, median impute, multiple impute, generalized low rank model impute, etc.)
- if present, convert factor variables to numeric indicators
5. Define x and y variables
6. Split data into train and test sets
7. Train and evaluate performance of a single algorithm as a prototype
8. Examine the trained model's performance on the test set
9. Create and deploy a cross-validated ensemble

> NOTE: You might also prefer to perform the cross-validated ensemble step first, followed by a single train/test split. 

# Goal of this machine learning workflow: Predicting heart disease

This workflow will demonstrate how individual algorithms and a SuperLearner ensemble weighted average can predict heart disease (yes/no) using other health indicators called features as predictors. Learn more about the dataset at the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/heart+disease)

# SuperLearner

> SuperLearner is an algorithm that uses cross-validation to estimate the performance of multiple machine learning models, or the same model with different settings. It then creates an optimal weighted average of those models, aka an "ensemble", using the test data performance. This approach has been proven to be asymptotically as accurate as the best possible prediction algorithm that is tested. [Guide to SuperLearner - 1 Background](https://cran.r-project.org/web/packages/SuperLearner/vignettes/Guide-to-SuperLearner.html)

In this manner, an ensemble is a powerful tool because it:

1. Eliminates bias of single algorithm selection for framing a research problem,

2. Allows for comparison of multiple algorithms, and/or comparison of the same model(s) but tuned in many different ways,

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

### The ck37r package

If the CRAN installation for the ck37r package does not work correctly, install it from GitHub by unhashtagging and running the three lines of code below. 

> If prompted to update packages, select "All" by entering a number 1 in your console and press the Enter key. 

```{r}
# install.packages("remotes")
# remotes::install_github("ck37/ck37r")
library(ck37r)
```

# Import and preprocess the data

Import and preprocess the data in six steps, and create a new variable for each step: 

1. **Import the dataset:** The variable `raw` contains data from the raw .csv file.

2. **Convert categorical variables:** `raw_fac` is used to convert categorical variables to factor type.

3. **Convert factors to numeric indicators:** `raw_df` will be for conversion of factors to numeric indicators.

4. **Identify and remove variables to be scaled:** `removed` consists of the preprocessed data _excluding_ the continuous variables to be scaled.

5. **Scale the continuous variables:** `rescaled` is the variable for the scaled variables.

6. **Produce clean dataset:** `clean` is the final merged dataset; a combination of the variables `removed` and `rescaled`.

7. **Save the clean dataset:** using the `save` function. You can load it with the `load` function so you do not have to repeat these preprocessing steps again. 

## 1. Import the dataset

Read in the raw .csv file and save it as an object named `raw`.

```{r}
raw <- read.csv("~/Desktop/Intro_ML_R/data/raw/heart.csv")
str(raw)
```

## 2. Convert categorical variables

The variables cp, restecg, slope, ca, and thal are variables that have to be converted to nominal categorical type. 

Save this factorized version as an object named `raw_fac`. 

```{r}
raw_fac <- ck37r::categoricals_to_factors(data = raw, 
                                          categoricals = c("cp", "restecg", "slope", "ca", "thal"))
str(raw_fac)
```

## 3. Convert factors to numeric indicators

Most machine learning algorithms require that input variables are numeric and therefore do not handle factor data well (although with some exceptions, such as decision trees). 

Therefore, we want to convert the factor variables to numeric indicator variables (aka one-hot/dummy coding). [See a few examples here](https://datatricks.co.uk/one-hot-encoding-in-r-three-simple-methods).

Name this object `raw_df`. 

```{r}
# Save this as an intermediate object named raw_ind 
raw_ind <- ck37r::factors_to_indicators(data = raw_fac, 
                                        verbose = TRUE)

# Extract the actual data portion from raw_ind and save as raw_df
raw_df <- raw_ind$data
str(raw_df)
```

## 4. Identify and remove variables to be scaled

Identify and remove the variables age, trestbps, chol, thalach, and oldpeak to be scaled.

```{r}
# First, investigate the data to identify variables to scale
summary(raw_df)
summary(raw_df[,c("age", "trestbps", "chol", "thalach", "oldpeak")])

# Save this as an intermediate object named to_scale
to_scale <- raw_df[,c("age", "trestbps", "chol", "thalach", "oldpeak")]
```

> Also see `?ck37r::standardize` for a simple extension

Remove the variables to be scaled and save the remaining variables in a dataframe named `removed`. 

```{r}
removed <- raw_df[,-c(1, # age
                      3, # trestbps
                      4, # chol
                      6, # thalach
                      8  # oldpeak
                      )]
```

## 5. Scale the continuous variables

Rescale these variables to a rang of 0 to 1 in a new dataframe named `rescaled`. 

```{r}
rescaled <- as.data.frame(rescale(as.matrix(to_scale), to = c(0,1), ncol = 1))
summary(rescaled)
```

## 6. Produce clean dataset

Finally, recombine the original data with the scaled variables for the final `clean` dataset. 

```{r}
clean <- cbind(removed,       # cleaned data without the scaled variables
               rescaled       # scaled variables
               )

# The scaled variables are the last five!
str(clean)
```

## 7. Save the `clean` dataset

Save the clean dataset so you do not have to repeat these preprocessing steps.

```{r}
save(clean,                                  # variable(s) to be saved
     file = "~/Desktop/Intro_ML_R/data/preprocessed/clean.RData") # file location and name
```

# Wipe your global environment clean

Reload the clean dataset with:

```{r}
load("~/Desktop/Intro_ML_R/data/preprocessed/clean.RData")
str(clean)
```
