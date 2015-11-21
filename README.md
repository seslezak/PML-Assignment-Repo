# PML-Assignment-Repo
###Repository for Practical Machine Learning Assignment, 22 Nov 15

The verbatim details of the assignment do not need to be restated here.  In sum, the project is to write code in R that allows for two _.csv_ files containing data from users of personal fitness devices, such as a FitBit, to be analyzed.  The data are drawn from a study available at the [Human Activity Recognition homepage](http://groupware.les.inf.puc-rio.br/har).

The data in one file -- the `pml-training.csv` -- are to be used to train and evaluate one or more machine learning models for prediction purposes. The trained model is to be used on the second file -- `pml-testing.csv` -- to predict activity quality for 20 individual users.

The prediction involves classifying the individuals into one of five _classes_, each class representing activity quality.  The activity was the Unilateral Dumbbell Biceps Curl, which participants were to execute in five different fashions.  The curl is often done incorrectly, so four of the classes represent these types of errors.  Class A represents the curl done in the correct fashion, according to specification. The other classes represent four incorrect forms.

The classes are:

* **Class A**: according to specification
* **Class B**: throwing elbows out to the front
* **Class C**: lifting dumbbell halfway
* **Class D**: lowering dumbbell halfway
* **Class E**: throwing hips out to the front

Using data from the training set, the model can predict which fashion each of the 20 individual participants in the testing set employed while being monitored by their fitness device. 

###Planning the Machine Learning Model

The steps followed in preparing the model were:

   1. viewing and cleaning the data
   2. choosing the packages
   3. dividing data into two parts (training and validation)
   4. viewing the cleaned data sets
   5. selecting models to train
   6. training models
   7. validating models
   8. running models on test data
   9. assessing accuracy of prediction

#####Viewing and Cleaning Data

The training data set contained 19,622 observations of 160 variables.  The testing data set contained 20 observations of 160 variables. Nearly two-thirds of the observations were missing.  The first seven columns of each file contained superfluous information which had to be removed.  We also had to set a seed to allow for replicability of the experiment.  The following code was used for these purposes:

```
set.seed(3663)  
trainDatRaw <- read.csv("path\\Practical Machine Learning\\pml-training.csv", na.strings = c('NA', '#DIV/0!', ''))  
testDatRaw <- read.csv("path\\Practical Machine Learning\\pml-testing.csv", na.strings = c('NA', '#DIV/0!', ''))  
traindatClean <- trainDatRaw[, -c(1:7)]  
testdatClean <- testDatRaw[, -c(1:7)]  
traindatClean <- traindatClean[, colSums(is.na(traindatClean)) == 0]  
testdatClean <- testdatClean[, colSums(is.na(testdatClean)) == 0]  
```

#####Package Selection

The R packages used in the experiment were selected as the code was written and tested. Packages were chosen for their usefulness in terms of running the models and for their graphics capabilities.  This code was used for choosing packages:

```
library(caret)  
library(rpart)  
library(rpart.plot)  
library(RColorBrewer)  
library(rattle)  
library(e1071)  
library(randomForest)
```

#####Creating the Testing and Validation Data Sets

It was necessary to divide the large _training.csv_ file into two parts, one for training the model and the other for validating it. We decided to reserve 60% of the data for training and 40% for validation.  We also wanted to look at the dimensions of the resulting data sets.  The test set contained 11,776 observations of 53 variables.  The vailidation set contained 7,846 observations of 53 variables. The following code was used for this stage of the experiment:

```
inTrain <- createDataPartition(traindatClean$classe, p = .6, list = FALSE)  
trainDat <- traindatClean[inTrain, ]  
validDat <- traindatClean[-inTrain, ]  
dim(trainDat)  
dim(validDat)  
```

#####Viewing the Cleaned Data Sets

At this point, we wanted to play around with the data a bit to develop some visuals of the data sets.  A number of different graphic representations were explored.  The code below was used to generate the following graphics.  Please note that several of the graphics, though interesting, were deactivated in the code.  These are signified by the hashtags.

```
plot(trainDat$classe, col = 'red', main = 'Five Class Levels in Training Set', xlab = 'Levels', ylab = 'Frequency')  
qplot(accel_forearm_x, total_accel_dumbbell, col = classe, data = trainDat, main = 'Forearm v Dumbell Acceleration #1')  
qplot(accel_forearm_y, total_accel_dumbbell, col = classe, data = trainDat, main = 'Forearm v Dumbell Acceleration #2')  
qplot(accel_forearm_z, total_accel_dumbbell, col = classe, data = trainDat, main = 'Forearm v Dumbell Acceleration #3')  
#qplot(accel_forearm_x, accel_forearm_y, col = classe, data = trainDat, main = 'Forearm Acceleration in Training Set')  
#qplot(accel_forearm_x, accel_dumbbell_x, col = classe, data = trainDat, main = 'Acceleration Pairs in Training Set')  
#qplot(accel_forearm_z, total_accel_forearm, col = classe, data = trainDat)  
#qplot(accel_forearm_x, total_accel_forearm, col = classe, data = trainDat)  
```

The first graph depicts the distribution of the five classes in the training set.

INSERT HISTOGRAM HERE

The relationship between the acceleration of the forearm and the acceleration of the dumbbell was interesting graphically, so those were plotted.  Forearm acceleration was measured along the x-, y-, and z-axes.  These values are plotted against total acceleration of the dumbbell.

INSERT PLOT #1 HERE

INSERT PLOT #2 HERE

INSERT PLOT #3 HERE

####Selecting Models for Training

Initially, the plan was to use three models for this project -- Random Forest, Decision Tree, and GBM. All three were tried, but only the Random Forest model gave good results.  There wasn't time to debug the code for the other two models, though the Decision Tree approach did work.  I could not get the GBM model to function properly before the project deadline.

There was enough time to play around with the Random Forest model.  On the first pass, using five folds and 250 trees, it gave good results but ran slowly. So the model was run using two different settings for the number of folds -- 5 and 10 -- and three different settings for the numbers of trees -- 250, 50, and 25.

The model using 5 folds and 250 trees had an accuracy rate of 0.998.  

```
Random Forest 

11776 samples
   52 predictor
    5 classes: 'A', 'B', 'C', 'D', 'E' 

No pre-processing
Resampling: Cross-Validated (5 fold) 
Summary of sample sizes: 9420, 9421, 9422, 9420, 9421 
Resampling results across tuning parameters:

  mtry  Accuracy   Kappa      Accuracy SD  Kappa SD   
   2    0.9888756  0.9859277  0.001484507  0.001876981
  27    0.9888757  0.9859270  0.002173340  0.002747416
  52    0.9800443  0.9747497  0.005979820  0.007560934
```

To speed things up, a model using 10 fold and 50 trees was tried.  It had an accuracy of 0.9





 
```  
contParams <- trainControl(method = 'cv', 5)  
modelRF <- train(classe ~. , data = trainDat, method = 'rf', trControl = contParams, ntree = 50)#250  
modelRF  
predictRF <- predict(modelRF, validDat)  
confusionMatrix(validDat$classe, predictRF)  
accRF <- postResample(predictRF, validDat$classe)  
accResultsRF <- accRF[1]
outOfSample <- 1 - as.numeric(confusionMatrix(validDat$classe, predictRF)$overall[1])
endResults <- predict(modelRF, testdatClean[, -length(testdatClean)])
endResults
```


The complete code follows without comments:

