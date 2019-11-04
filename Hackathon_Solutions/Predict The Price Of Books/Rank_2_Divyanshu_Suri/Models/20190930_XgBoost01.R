setwd("C:\\Kaggle\\BooksPrice\\Participants_Data")

library(Metrics)
library(readr)
library(Metrics)
library(xgboost)
library(sqldf)
library(openxlsx)
library(dummies)
library(stringr)
library(openxlsx)

train01 <- read.csv("Data_Train01.csv",stringsAsFactors = FALSE, check.names = FALSE)
test01 <- read.csv("Data_Test01.csv",stringsAsFactors = FALSE, check.names = FALSE)
Fnames <- read.csv("FeatureNames.csv",stringsAsFactors = FALSE, check.names = FALSE)

submission <- read.xlsx("C:\\Kaggle\\BooksPrice\\Participants_Data\\Sample_Submission.xlsx",
                        sheet = 1,
                        startRow = 1,
                        colNames = TRUE)

feature.names <- Fnames$x

train01$Price_Log <- log10(train01$Price + 1)
hist(train01$Price_Log)

num_folds <- 5

set.seed(212)
train01$FOLD_NUM <- runif(nrow(train01), min = 0, max = 1)
train01$FOLD_NUM <- cut(train01$FOLD_NUM,breaks=num_folds,labels = FALSE)
table(train01$FOLD_NUM)

for (i in 1:num_folds)
{
  cat("RUNNING FOR ",i," ")
  
  testData <- subset(train01, FOLD_NUM == i)
  trainData <- subset(train01, FOLD_NUM != i)
  
  dtrain <- xgb.DMatrix(data = data.matrix(trainData[,feature.names]),
                        label = trainData[,"Price_Log"])
  
  dvalid <- xgb.DMatrix(data = data.matrix(testData[,feature.names]),
                        label = testData[,"Price_Log"])
  
  param <- list(  objective = "reg:linear",
                  booster = "gbtree",
                  eta = 0.01,
                  max_depth = 6,#5
                  subsample = 0.9,
                  colsample_bytree = 0.6,#0.6
                  min_child_weight = 5#5
  )
  
  watchlist <- list(train = dtrain, val = dvalid)
  set.seed(501+i)
  XGB_mod_1 <- xgb.train(   params = param,
                            data = dtrain,
                            nrounds = 100000,
                            verbose = 1,
                            watchlist = watchlist,
                            eval_metric = "rmse",
                            maximize = FALSE,
                            early.stop.round=200,
                            print.every.n = 25  ) #[2679]	train-rmse:0.212135	val-rmse:0.265194
  
  testData$Price_Log_Pred <- predict(XGB_mod_1, data.matrix(testData[,feature.names]))
  
  cat(rmse(testData$Price_Log_Pred,testData$Price_Log),"\n")
  
  if(i == 1) {
    preds_sub <- predict(XGB_mod_1, data.matrix(test01[,feature.names]))
    cv_scrd_trnData <- testData
  }
  else{
    preds_sub <- preds_sub+predict(XGB_mod_1, data.matrix(test01[,feature.names]))
    cv_scrd_trnData <- rbind.data.frame(cv_scrd_trnData,testData)
  }
}
cat("\nFINAL XGB RMSLE : ",rmse(cv_scrd_trnData$Price_Log_Pred,cv_scrd_trnData$Price_Log),"\n")
cat("\nFINAL XGB 1-RMSLE : ",1-rmse(cv_scrd_trnData$Price_Log_Pred,cv_scrd_trnData$Price_Log),"\n")
#FINAL XGB 1-RMSLE :  0.7439308
#LB : 0.748558

preds_sub <- preds_sub / num_folds
summary(preds_sub)
test01$Price_Log_Pred <- preds_sub
preds_sub <- (10**preds_sub)-1
summary(preds_sub)

write_csv(cv_scrd_trnData, "C:\\Kaggle\\BooksPrice\\CV Scrd Trn Datasets\\20190930_XGB01_DS.csv")
write_csv(test01, "C:\\Kaggle\\BooksPrice\\CV Scrd Tst Datasets\\20190930_XGB01_DS.csv")

submission$Price <- preds_sub
write.xlsx(submission, "C:\\Kaggle\\BooksPrice\\Submissions\\20190930_XGB01_DS.xlsx", colNames = TRUE)