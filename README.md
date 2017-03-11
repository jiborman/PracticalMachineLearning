# Final Project
JiBorman  
3/11/2017  
# Setup

```r
rm(list=ls())
set.seed(12345)
options(stringsAsFactors = F)
library(caret)
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```r
library(randomForest)
```

```
## randomForest 4.6-12
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```r
library(rpart)
```

# Introduction
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, the goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. Participants were asked to perform barbell lifts correctly and incorrectly in 5 different ways, defined as follows: exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E)

The goal of this project is to predict the manner in which participants did the exercise. This is the "classe" variable in the training set. We look at any of the other variables as potential predictors. Below, we create a report describing how the model was built , how cross validation was used, the expected out of sample error is from our model, and why particular choices were made. 

# Loading in Data
## Training Data

```r
if(file.exists('./trainingdata.csv')){
  TrainingData <- read.csv('./trainingdata.csv', na.strings = c('NA', '', '#DIV/0!'))
}else{
  download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", dest='./trainingdata.csv')
  TrainingData <- read.csv('./trainingdata.csv', na.strings = c('NA', '', '#DIV/0!'))
}
```

## Test Data

```r
if(file.exists('./testdata.csv')){
  TestData <- read.csv('./testdata.csv', na.strings = c('NA', '', '#DIV/0!'))
}else{
  download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", dest='./testdata.csv')
  TestData <- read.csv('./testdata.csv', na.strings = c('NA', '', '#DIV/0!'))
}
```

# Data Manipulation
## Training Data
We start by cleaning out predictor variables which are incomplete (any are NA), the indexubg variable and timestamp information. Then we remove predictors with near zero variance.

```r
if(any(apply(is.na(TrainingData), 2, any))){
  TrainingData <- TrainingData[,-which(apply(is.na(TrainingData), 2, any))]
}

nzv <- nearZeroVar(TrainingData, saveMetrics=TRUE)
```
The following variables have near zero variance, so we remove them from our list of potential predictors: new_window

```r
TrainingData <- TrainingData[,-which(nzv$nzv)]
```
We also remove some extraneous columns which have the user name, etc, that should not influence ability.

```r
TrainingData <- TrainingData[,-seq(1,6)]
```
Now we partition the training set into two sets. We will use 60% of TrainingData for training the model and 40% for testing the model. The TestData we downloaded above will be for predictions once models have been selected. 

```r
set.seed(12345)
inTrain <- createDataPartition(TrainingData$classe, p=.6, list=F)
data.train <- TrainingData[inTrain,]
data.test <- TrainingData[-inTrain,]
```

# Model Selection and Cross Validation
We use the data.train dataset to train two models, a classification model and a random forest model. We consider 5-fold cross validation to validate our predictor choices. To test our models we use the data.test dataset. For both models, we produce a confusion matrix to evaluate performance.

```r
set.seed(12345)
control <- trainControl(method='cv', number=5)

set.seed(12345)
modFit_Class <- train(classe~., data=data.train, method='rpart', trControl=control)
prediction_Class <- predict(modFit_Class, data.test)
cm_Class <- confusionMatrix(prediction_Class, data.test$classe)

cm_Class
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2224 1518 1368 1286  782
##          B    0    0    0    0    0
##          C    0    0    0    0    0
##          D    0    0    0    0    0
##          E    8    0    0    0  660
## 
## Overall Statistics
##                                           
##                Accuracy : 0.3676          
##                  95% CI : (0.3569, 0.3784)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.1266          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9964   0.0000   0.0000   0.0000  0.45770
## Specificity            0.1176   1.0000   1.0000   1.0000  0.99875
## Pos Pred Value         0.3098      NaN      NaN      NaN  0.98802
## Neg Pred Value         0.9880   0.8065   0.8256   0.8361  0.89106
## Prevalence             0.2845   0.1935   0.1744   0.1639  0.18379
## Detection Rate         0.2835   0.0000   0.0000   0.0000  0.08412
## Detection Prevalence   0.9149   0.0000   0.0000   0.0000  0.08514
## Balanced Accuracy      0.5570   0.5000   0.5000   0.5000  0.72822
```

```r
set.seed(12345)
modFit_RF <- train(classe~., data=data.train, method='rf', trControl=control, ntree=5)
prediction_RF <- predict(modFit_RF, data.test)
cm_RF <- confusionMatrix(prediction_RF, data.test$classe)

cm_RF
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2217   25    1    1    1
##          B   11 1455   23    6   12
##          C    1   29 1330   27   10
##          D    1    2   12 1247    5
##          E    2    7    2    5 1414
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9767          
##                  95% CI : (0.9731, 0.9799)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2e-16         
##                                           
##                   Kappa : 0.9705          
##  Mcnemar's Test P-Value : 0.02189         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9933   0.9585   0.9722   0.9697   0.9806
## Specificity            0.9950   0.9918   0.9897   0.9970   0.9975
## Pos Pred Value         0.9875   0.9655   0.9520   0.9842   0.9888
## Neg Pred Value         0.9973   0.9901   0.9941   0.9941   0.9956
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2826   0.1854   0.1695   0.1589   0.1802
## Detection Prevalence   0.2861   0.1921   0.1781   0.1615   0.1823
## Balanced Accuracy      0.9941   0.9751   0.9809   0.9833   0.9890
```
The accuracy for the classification model is 0.3675758. The out of sample error is 0.6324242. For our random forest model, we have an accuracy of 0.976676 and out of sample error is 0.023324. Therefore, the random forest model is predicting the classe variable better, and we will use this model to do our final predictions.

# Prediction
Our preferred model is the random forest model because the accuracy is far better than with the classification model. We use it to predict values from the test set given above.

```r
TestData <- TestData[,which(names(TestData)%in%names(TrainingData))]
predict(modFit_RF, newdata = TestData)
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
