#------------------------------for 22 param data------------------------------
voice = read.csv("/Users/shaojy11/Documents/2017Spring/259/PSYCH259_Project/data/processedData/voice.csv")
voice[!complete.cases(voice),]
voice = na.omit(voice)
#shuffle and split
set.seed(2)
split <- sample(dim(voice)[1], floor(0.7*dim(voice)[1]))
train <-voice[split,]
test <- voice[-split,]
p = dim(train)[2]

#--------------------------for spectrum data------------------------------
train = read.table("/Users/shaojy11/Documents/2017Spring/259/PSYCH259_Project/data/processedData/voice_CMU_train.txt")
trainlabel = read.table("/Users/shaojy11/Documents/2017Spring/259/PSYCH259_Project/data/processedData/voice_CMU_trainlabel.txt")
test = read.table("/Users/shaojy11/Documents/2017Spring/259/PSYCH259_Project/data/processedData/voice_CMU_test.txt")
testlabel = read.table("/Users/shaojy11/Documents/2017Spring/259/PSYCH259_Project/data/processedData/voice_CMU_testlabel.txt")
train <- as.data.frame(cbind(train, factor(trainlabel$V1)))
test <- as.data.frame(cbind(test, factor(testlabel$V1)))
names(train)[802] = "label"
names(test)[802] = "label"
voice <- rbind(train, test)
p = dim(train)[2]
split <- seq(1, dim(train)[1])

#----------------without pca------------------------------------------------------------------------------------
#random forest
library(randomForest)
library("factoextra")
rf.model <- randomForest(label ~ ., data = train, mtry = 3, ntree = 1000)
table(predict(rf.model), train$label)
rf.trainpred <- predict(rf.model, newdata = train[,1:(p-1)])
rf.trainerror <- 1 - sum(train$label == rf.trainpred) / dim(train)[1]
rf.testpred <- predict(rf.model, newdata = test[,1:(p-1)])
rf.testerror <- 1 - sum(test$label == rf.testpred) / dim(test)[1]

# visualization
# print(rfmodel)
# plot(rfmodel)
# importance(rfmodel)
# varImpPlot(rfmodel)


#svm
library(e1071)
svm.model <- svm(label ~., data = train, kernel="polynomial", scale = FALSE, type = "C")
svm.trainpred <- predict(svm.model, train[,1:(p-1)])
svm.trainerror <- 1 - sum(train$label == svm.trainpred) / length(svm.trainpred)
svm.testpred <- predict(svm.model, test[,1:(p-1)])
svm.testerror <- 1 - sum(test$label == svm.testpred) / length(svm.testpred)


#logistic
library(glmnet)
lg.model = cv.glmnet(as.matrix(train[,1:(p-1)]), train[,p], family = "binomial", alpha = 1)
lg.trainpred <- predict(lg.model, as.matrix(train[,1:(p-1)]), type = "class", s = lg.model$lambda.min)
lg.trainerror <- 1 - sum(lg.trainpred == train$label) / dim(train)[1]
lg.testpred <- predict(lg.model, as.matrix(test[,1:(p-1)]), type = "class", s = lg.model$lambda.min)
lg.testerror <- 1 - sum(lg.testpred == test$label) / dim(test)[1]


#xgboost
library(xgboost)
trainlabel <- as.numeric(train$label == "1")
testlabel <- as.numeric(test$label == "1")
xgb.model <- xgboost(data = as.matrix(train[,1:(p-1)]), label = trainlabel, nrounds = 10, objective = "binary:logistic")
xgb.trainpred <- predict(xgb.model, as.matrix(train[,1:(p-1)]))
xgb.trainlabel <- as.numeric(xgb.trainpred > 0.5)
xgb.trainerror <- 1 - sum(trainlabel == xgb.trainlabel) / dim(train)[1]
xgb.testpred <- predict(xgb.model, as.matrix(test[,1:(p-1)]))
xgb.testlabel <- as.numeric(xgb.testpred > 0.5)
xgb.testerror <- 1 - sum(testlabel == xgb.testlabel) / dim(test)[1]


#naive bayes
bys.model <- naiveBayes(label ~., data = train)
bys.trainpred<-predict(bys.model, newdata = train[,1:(p-1)])
bys.trainerror <- 1 - sum(train$label == bys.trainpred) / dim(train)[1]
bys.testpred <- predict(bys.model, newdata = test[, 1:(p-1)])
bys.testerror <- 1 - sum(test$label == bys.testpred) / dim(test)[1]


# -------------------------------------feature engineering---------------------------------------
library(factoextra)
pca <- prcomp(voice[,1:(p-1)], center=TRUE)
ind <- get_pca_ind(pca)
trainFeature <- as.data.frame(ind$coord[split, 1:10])
testFeature <- as.data.frame(ind$coord[-split, 1:10])
newdata = as.data.frame(cbind(trainFeature, train$label))
eig.val <- get_eigenvalue(pca)
barplot(eig.val[1:10, 2], names.arg=1:10, 
        main = "Variances",
        xlab = "Principal Components",
        ylab = "Percentage of variances",
        col ="steelblue")
# Add connected line segments to the plot
lines(x = 1:10, 
      eig.val[1:10, 2], 
      type="b", pch=19, col = "red")

#svm
library(e1071)
svm.model <- svm(newdata$`train$label` ~., data = newdata, kernel="sigmoid", scale = FALSE)
svm.trainpred <- predict(svm.model, trainFeature)
svm.trainerror <- 1 - sum(train$label == svm.trainpred) / length(svm.trainpred)
svm.testpred <- predict(svm.model, testFeature)
svm.testerror <- 1 - sum(test$label == svm.testpred) / length(svm.testpred)


#rondom forest
library(randomForest)
library("factoextra")
rf.model <- randomForest(newdata$`train$label` ~., data = newdata, mtry = 3, ntree = 100)
rf.trainpred<-predict(rf.model, newdata = trainFeature)
rf.trainerror <- 1 - sum(train$label == rf.trainpred) / dim(train)[1]
rf.testpred <- predict(rf.model, newdata = testFeature)
rf.testerror <- 1 - sum(test$label == rf.testpred) / dim(test)[1]

#naive bayes
bys.model <- naiveBayes(newdata$`train$label` ~., data = newdata)
bys.trainpred<-predict(bys.model, newdata = trainFeature)
bys.trainerror <- 1 - sum(train$label == bys.trainpred) / dim(train)[1]
bys.testpred <- predict(bys.model, newdata = testFeature)
bys.testerror <- 1 - sum(test$label == bys.testpred) / dim(test)[1]

#logistic
library(glmnet)
lg.model = cv.glmnet(as.matrix(trainFeature), train[,802], family = "binomial", alpha = 1)
lg.trainpred <- predict(lg.model, as.matrix(trainFeature), type = "class", s = lg.model$lambda.min)
lg.trainerror <- 1 - sum(lg.trainpred == train$label) / dim(train)[1]
lg.testpred <- predict(lg.model, as.matrix(testFeature), type = "class", s = lg.model$lambda.min)
lg.testerror <- 1 - sum(lg.testpred == test$label) / dim(test)[1]
