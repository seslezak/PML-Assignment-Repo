library(caret)
library(rpart)
library(rpart.plot)
library(RColorBrewer)
library(rattle)
library(e1071)
library(randomForest)
set.seed(3663)
trainDatRaw <- read.csv("C:\\Path\\Practical Machine Learning\\pml-training.csv", na.strings = c('NA', '#DIV/0!', ''))
testDatRaw <- read.csv("C:\\Path\\Practical Machine Learning\\pml-testing.csv", na.strings = c('NA', '#DIV/0!', ''))
traindatClean <- trainDatRaw[, -c(1:7)]
testdatClean <- testDatRaw[, -c(1:7)]
traindatClean <- traindatClean[, colSums(is.na(traindatClean)) == 0]
testdatClean <- testdatClean[, colSums(is.na(testdatClean)) == 0]
inTrain <- createDataPartition(traindatClean$classe, p = .6, list = FALSE)
trainDat <- traindatClean[inTrain, ]
validDat <- traindatClean[-inTrain, ]
dim(trainDat)
dim(validDat)
plot(trainDat$classe, col = 'red', main = 'Five Class Levels in Training Set', xlab = 'Levels', ylab = 'Frequency')
qplot(accel_forearm_x, total_accel_dumbbell, col = classe, data = trainDat, main = 'Forearm v Dumbell Acceleration #1')
qplot(accel_forearm_y, total_accel_dumbbell, col = classe, data = trainDat, main = 'Forearm v Dumbell Acceleration #2')
qplot(accel_forearm_z, total_accel_dumbbell, col = classe, data = trainDat, main = 'Forearm v Dumbell Acceleration #3')
#qplot(accel_forearm_x, accel_forearm_y, col = classe, data = trainDat, main = 'Forearm Acceleration in Training Set')
#qplot(accel_forearm_x, accel_dumbbell_x, col = classe, data = trainDat, main = 'Acceleration Pairs in Training Set')
#qplot(accel_forearm_z, total_accel_forearm, col = classe, data = trainDat)
#qplot(accel_forearm_x, total_accel_forearm, col = classe, data = trainDat)
contParams <- trainControl(method = 'cv', 5)
#contParams <- trainControl(method = 'cv', 10)
#modelRF <- train(classe ~. , data = trainDat, method = 'rf', trControl = contParams, ntree = 250)
#modelRF <- train(classe ~. , data = trainDat, method = 'rf', trControl = contParams, ntree = 50)
modelRF <- train(classe ~. , data = trainDat, method = 'rf', trControl = contParams, ntree = 25)
modelRF
predictRF <- predict(modelRF, validDat)
confusionMatrix(validDat$classe, predictRF)
accRF <- postResample(predictRF, validDat$classe)
accResultsRF <- accRF[1]
outOfSample <- 1 - as.numeric(confusionMatrix(validDat$classe, predictRF)$overall[1])
outOfSample
endResults <- predict(modelRF, testdatClean[, -length(testdatClean)])
endResults
