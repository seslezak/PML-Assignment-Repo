# PML-Assignment-Repo
Repository for Practical Machine Learning Assignment, 22 Nov 15

The verbatim details of the assignment do not need to be restated here.  In sum, the project is to write code in R that allows for two .csv files containing data from users of personal fitness devices, such as a FitBit, to be analyzed.  The data are drawn from a study available at http://groupware.les.inf.puc-rio.br/har.

The data in one file -- the "pml-training.csv" -- are to be used to train and evaluate one or more machine learning models for prediction purposes. The trained model is to be used on the second file -- "pml-testing.csv" -- to predict activity quality for 20 individual users.

The prediction involves classifying the individuals into one of five "classes," each class representing activity quality.  The activity was the Unilateral Dumbbell Biceps Curl, which participants were to execute in five different fashions.  The curl is often done incorrectly, so four of the classes represent these types of errors.  One class represents the curl done in the correct fashion, according to specification.

The classes are:

Class A -- according to specification
Class B -- throwing elbows out to the front
Class C -- lifting dumbbell halfway
Class D -- lowering dumbbell halfway
Class E -- throwing hips out to the front

From the data in the training set, the model is to predict which fashion each of the 20 individual participants in the testing set employed while being monitored by their fitness device. 






library(caret)
library(rpart)
library(rpart.plot)
library(RColorBrewer)
library(rattle)
library(e1071)
library(randomForest)
set.seed(3663)
trainDatRaw <- read.csv("C:\\Users\\Steven\\Desktop\\R Coursera\\Practical Machine Learning\\pml-training.csv", na.strings = c('NA', '#DIV/0!', ''))
testDatRaw <- read.csv("C:\\Users\\Steven\\Desktop\\R Coursera\\Practical Machine Learning\\pml-testing.csv", na.strings = c('NA', '#DIV/0!', ''))
traindatClean <- trainDatRaw[, -c(1:7)]
testdatClean <- testDatRaw[, -c(1:7)]
traindatClean <- traindatClean[, colSums(is.na(traindatClean)) == 0]
testdatClean <- testdatClean[, colSums(is.na(testdatClean)) == 0]
inTrain <- createDataPartition(traindatClean$classe, p = .6, list = FALSE)
trainDat <- traindatClean[inTrain, ]
validDat <- traindatClean[-inTrain, ]
dim(trainDat)
dim(validDat)
plot(trainDat$classe, col = 'red', main = 'Five Classe Levels', xlab = 'Levels', ylab = 'Frequency')
qplot(accel_forearm_x, accel_forearm_y, col = classe, data = trainDat)
qplot(accel_forearm_z, total_accel_forearm, col = classe, data = trainDat)
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



The complete code follows without comments:

