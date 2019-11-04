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
library(tidytext)
library(tidyverse)
library(glue)
library(glmnet)
library(cleanNLP)
library(ggplot2)
library(tokenizers)
library(stringi)
library(methods)
library(readr)
library(dplyr)
library(keras)
library(fastTextR)
library(tidymodels)
library(smodels)
library(tensorflow)
library(fasttextM)
library(xgboost)
library(tm)
library(Metrics)

train01 <- read.csv("C:\\Kaggle\\BooksPrice\\CV Scrd Trn Datasets\\20190930_XGB01_DS.csv",stringsAsFactors = FALSE, check.names = FALSE)
test01 <- read.csv("C:\\Kaggle\\BooksPrice\\CV Scrd Tst Datasets\\20190930_XGB01_DS.csv",stringsAsFactors = FALSE, check.names = FALSE)
Fnames <- read.csv("FeatureNames.csv",stringsAsFactors = FALSE, check.names = FALSE)

submission <- read.xlsx("C:\\Kaggle\\BooksPrice\\Participants_Data\\Sample_Submission.xlsx",
                        sheet = 1,
                        startRow = 1,
                        colNames = TRUE)

feature.names <- Fnames$x

train01$Price_Log <- log10(train01$Price + 1)
hist(train01$Price_Log)

test01$Price_Log <- NA
test01$FOLD_NUM <- NA

CombinedData <- rbind.data.frame(train01,test01)

CombinedData$Synopsis2 <- tolower(gsub("[^[:alnum:] ]", " ", CombinedData$Synopsis))

stopwords_regex = paste(stopwords('en'), collapse = '\\b|\\b')
stopwords_regex = paste0('\\b', stopwords_regex, '\\b')
CombinedData$Synopsis2 = stringr::str_replace_all(CombinedData$Synopsis2, stopwords_regex, '')
CombinedData$Synopsis2 <- gsub('[[:digit:]]+', '', CombinedData$Synopsis2)

token_list_train_test <- tokenize_words(CombinedData$Synopsis2)
token_df_train_test <- term_list_to_df(token_list_train_test)
X_train_test <- term_df_to_matrix(token_df_train_test, min_df = 0.01, max_df = 1,
                                  scale = FALSE)

dim(X_train_test)

X <- X_train_test[1:nrow(train01),]
X_test <- X_train_test[(nrow(train01)+1):nrow(X_train_test),]

dim(X)
dim(X_test)

remove(X_train_test,token_df_train_test,token_list_train_test)

X <- data.frame(as.matrix(X))
X[,(ncol(X)+1)] <- train01$Price_Log_Pred

dim(X)
X <- as(as.matrix(X), "dgTMatrix")
dim(X)

X_test <- data.frame(as.matrix(X_test))
X_test[,(ncol(X_test)+1)] <- test01$Price_Log_Pred

dim(X_test)
X_test <- as(as.matrix(X_test), "dgTMatrix")
dim(X_test)

num_folds <- 5

for (i in 1:num_folds)
{
  cat("RUNNING FOR ",i," ")
  
  testData <- subset(train01, FOLD_NUM == i)
  trainData <- subset(train01, FOLD_NUM != i)
  
  X_train <- X[train01$FOLD_NUM != i,]
  X_val <- X[train01$FOLD_NUM == i,]
  
  dtrain <- xgb.DMatrix(data = as.matrix(X_train),
                        label = trainData[,"Price_Log"])
  
  dvalid <- xgb.DMatrix(data = as.matrix(X_val),
                        label = testData[,"Price_Log"])
  
  param <- list(  objective = "reg:linear",
                  booster = "gbtree",
                  eta = 0.05,
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
                            print.every.n = 25  )
  
  testData$Price_Log_Pred <- predict(XGB_mod_1, as.matrix(X_val))
  
  cat(sqrt(mean((testData$Price_Log-testData$Price_Log_Pred)**2)),"\n")
  
  if(i == 1) {
    preds_sub <- predict(XGB_mod_1, as.matrix(X_test))
    cv_scrd_trnData <- testData
  }
  else {
    preds_sub <- preds_sub+predict(XGB_mod_1, as.matrix(X_test))
    cv_scrd_trnData <- rbind.data.frame(cv_scrd_trnData,testData)
  }
}
cat("\nFINAL XGB RMSLE : ",sqrt(mean((cv_scrd_trnData$Price_Log-cv_scrd_trnData$Price_Log_Pred)**2)),"\n")
cat("\nFINAL XGB 1-RMSLE : ",1-sqrt(mean((cv_scrd_trnData$Price_Log-cv_scrd_trnData$Price_Log_Pred)**2)),"\n")
#FINAL XGB 1-RMSLE :  0.7696596
#LB : 0.774239

preds_sub <- preds_sub / num_folds
summary(preds_sub)
test01$Price_Log_Pred <- preds_sub
preds_sub <- (10**preds_sub)-1
summary(preds_sub)

write_csv(cv_scrd_trnData, "C:\\Kaggle\\BooksPrice\\CV Scrd Trn Datasets\\20190930_XGB02_DS.csv")
write_csv(test01, "C:\\Kaggle\\BooksPrice\\CV Scrd Tst Datasets\\20190930_XGB02_DS.csv")

submission$Price <- preds_sub
write.xlsx(submission, "C:\\Kaggle\\BooksPrice\\Submissions\\20190930_XGB02_DS.xlsx", colNames = TRUE)