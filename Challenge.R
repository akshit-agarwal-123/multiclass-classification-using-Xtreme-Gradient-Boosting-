library(MASS, quietly=TRUE)
library(caret)
library(ROSE)
library(ROCR)
library(BBmisc)
library(xgboost)
library(magrittr)
library(dplyr)
library(Matrix)
library(ROSE)
library(lattice)
library(ggplot2)
library(caret)

data<-read.csv('train.csv',header = T)
View(data)
dim(data)
names(data)
head(data, 12)
str(data)
set.seed(123)
#data$NSP <- as.factor(data$NSP)

ccs <- as.matrix(data)
View(cor(ccs[,-1],data$NSP))


#data$NSP <- as.factor(data$NSP)



head(data)
data <- data[c(22,(1:21))]
dim(data)

# data$FM <- normalize(data$FM, range=c(0,1))
# data$ALTV <- normalize(data$ALTV, range=c(0,1))
# data$Tendency <- normalize(data$Tendency, range=c(0,1))




# 
# undertrainDF <- ovun.sample(no_show~., data=trainDF, method='both')$data
# table(undertrainDF$no_show)
# 
# undertestDF <- ovun.sample(no_show~., data=testDF, method='both')$data
# table(undertestDF$no_show)
# 

# Partition data

head(data)
ind <- sample(2, nrow(data), replace = T, prob = c(0.8, 0.2))
trainDF <- data[ind==1,]
testDF <- data[ind==2,]

table(trainDF$NSP)
head(trainDF)
# overtrain <- ovun.sample(no_show~., data=train, method='over', N= 136254)$data
# table(overtrain$no_show)

table(testDF$NSP)
head(testDF)
# overtest <- ovun.sample(no_show~., data=test, method='over', N= 34028)$data
# table(overtest$no_show)
head(data)
# View(overtrain)

# Create matrix - One-Hot Encoding for Factor variables
trainm <- sparse.model.matrix(NSP ~ .-1, data = trainDF)
head(trainm)
train_label <- trainDF[,"NSP"]
train_l
train_matrix <- xgb.DMatrix(data = as.matrix(trainm), label = train_label)

testm <- sparse.model.matrix(NSP~.-1, data = testDF)
test_label <- testDF[,"NSP"]
test_matrix <- xgb.DMatrix(data = as.matrix(testm), label = test_label)

# Parameters
nc <- length(unique(train_label))
nc
xgb_params <- list("objective" = "multi:softmax",
                   "eval_metric" = "mlogloss",
                   "num_class" = nc)
watchlist <- list(train = train_matrix, test = test_matrix)


names(trainDF)
# eXtreme Gradient Boosting Model
bst_model <- xgb.train(params = xgb_params,
                       data = train_matrix,
                       nrounds = 300,
                       watchlist = watchlist,
                       eta = 0.001,
                       max.depth = 3,
                       gamma = 0,
                       subsample = 1,
                       colsample_bytree = 1,
                       missing = NA,
                       seed = 333)

# Training & test error plot
e <- data.frame(bst_model$evaluation_log)
plot(e$iter, e$train_mlogloss, col = 'blue')
lines(e$iter, e$test_mlogloss, col = 'red')

#min(e$test_mlogloss)
#e[e$test_mlogloss == 0.625217,]

# Feature importance
imp <- xgb.importance(colnames(train_matrix), model = bst_model)
print(imp)
xgb.plot.importance(imp)

# Prediction & confusion matrix - test data
p <- predict(bst_model, newdata = test_matrix)
pred <- matrix(p, nrow = nc, ncol = length(p)/(nc)) %>%
  t() %>%
  data.frame() %>%
  mutate(label = test_label, max_prob = max.col(., "last")-1)
View(pred)

t <- table(Prediction = pred$max_prob, Actual = pred$label)
print(t)
x  <-table(pred$max_prob,pred$label)
percent_accuracy <- round((x['0','0'] + x['1','1'])/(x['0','0'] + x['0','1'] + x['1','0'] + x['1','1'])*100,2)
cat('Model Accuracy is:',percent_accuracy,'%')
cat('Misclassification Rate is:',100-percent_accuracy,'%')

