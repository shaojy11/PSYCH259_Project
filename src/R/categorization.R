voice = read.csv("/Users/shaojy11/Documents/2017Spring/259/PSYCH259_Project/data/voice.csv")

voice[!complete.cases(voice),]
voice = na.omit(voice)

#shuffle and split
factor = 0.7
voice = voice[sample(nrow(voice), nrow(voice)),]
train = voice[1: (nrow(voice) * factor),]
test = voice[(nrow(voice) * factor + 1) : nrow(voice),]

#random forest
library(randomForest)
# train
rfmodel <- randomForest(label ~ ., data = train, mtry = 3, ntree = 1000)
# table(predict(rfmodel), train$label)
# print(rfmodel)
# plot(rfmodel)
# importance(rfmodel)
# varImpPlot(rfmodel)
#test
pred<-predict(rfmodel,newdata=test)
table(pred, test$label)
plot(margin(rfmodel), test$label)
test.rferr <- 1 - sum(test$label == pred) / dim(test)[1]
oob.rferr <- rfmodel$err.rate[dim(rfmodel$err.rate)[1],1]
