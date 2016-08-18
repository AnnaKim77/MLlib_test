
## iris data 내장
## randomforest 제공 library : caret

library(caret)
data(iris)

# iris dataset 형태 확인
head(iris)
str(iris)

# Training을 위해 8:2로 데이터셋 분할
set.seed(20160818)
inTrain<-createDataPartition(y=iris$Species, p=.8, list=F)
trainSet<-iris[inTrain,]
testSet<-iris[-inTrain,]

# Train a Random Forest model
iris.rf<-train(Species ~. , data=testSet, method="rf", trControl=trainControl(method="cv",number=3))

# Predictions
predictions<-predict(iris.rf, testSet)
confusionMatrix(predictions, testSet$Species)