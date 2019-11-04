library(Metrics)
library(readr)
library(xgboost)
library(sqldf)
library(openxlsx)

setwd("C:\\Kaggle\\BooksPrice\\CV Scrd Trn Datasets")
file_names = list.files("C:\\Kaggle\\BooksPrice\\CV Scrd Trn Datasets")
file_names

training01 <- read.csv(file_names[1], stringsAsFactors = FALSE, check.names = FALSE)
training01 <- training01[,c("id","Price","Price_Log","FOLD_NUM","Price_Log_Pred_Keras")]
names(training01) <- c("id","Price","Price_Log","FOLD_NUM","Price_Log_Pred1")
training01 <- training01[order(training01$id),]

for(i in 2:length(file_names)) {
  temp_trn <- read.csv(file_names[i], stringsAsFactors = FALSE, check.names = FALSE)
  temp_trn <- temp_trn[order(temp_trn$id),]
  training01[,paste("Price_Log_Pred",i,sep="")] <- temp_trn[,ncol(temp_trn)]
  remove(temp_trn)
}

setwd("C:\\Kaggle\\BooksPrice\\CV Scrd Tst Datasets")

testing01 <- read.csv(file_names[1], stringsAsFactors = FALSE, check.names = FALSE)
testing01 <- testing01[,c("id","Price_Log_Pred_Keras")]
names(testing01) <- c("id","Price_Log_Pred1")
testing01 <- testing01[order(testing01$id),]

for(i in 2:length(file_names)) {
  temp_trn <- read.csv(file_names[i], stringsAsFactors = FALSE, check.names = FALSE)
  temp_trn <- temp_trn[order(temp_trn$id),]
  testing01[,paste("Price_Log_Pred",i,sep="")] <- temp_trn[,ncol(temp_trn)]
  remove(temp_trn)
}

num_folds <- 5

train01 <- training01
test01 <- testing01
feature.names <- names(test01)[2:ncol(test01)]

for (i in 1:num_folds)
{
  cat("RUNNING FOR ",i," ")
  
  testData <- subset(train01, FOLD_NUM == i)
  trainData <- subset(train01, FOLD_NUM != i)
  
  dtrain <- xgb.DMatrix(data = as.matrix(trainData[,feature.names]),
                        label = trainData[,"Price_Log"])
  
  dvalid <- xgb.DMatrix(data = as.matrix(testData[,feature.names]),
                        label = testData[,"Price_Log"])
  
  param <- list(  objective = "reg:linear",
                  booster = "gbtree",
                  eta = 0.001,
                  max_depth = 3,#5
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
  
  testData$Price_Log_Pred <- predict(XGB_mod_1, as.matrix(testData[,feature.names]))
  
  cat(sqrt(mean((testData$Price_Log-testData$Price_Log_Pred)**2)),"\n")
  
  if(i == 1) {
    preds_sub <- predict(XGB_mod_1, as.matrix(test01[,feature.names]))
    cv_scrd_trnData <- testData
  }
  else {
    preds_sub <- preds_sub+predict(XGB_mod_1, as.matrix(test01[,feature.names]))
    cv_scrd_trnData <- rbind.data.frame(cv_scrd_trnData,testData)
  }
}
cat("\nFINAL XGB RMSLE : ",sqrt(mean((cv_scrd_trnData$Price_Log-cv_scrd_trnData$Price_Log_Pred)**2)),"\n")
cat("\nFINAL XGB 1-RMSLE : ",1-sqrt(mean((cv_scrd_trnData$Price_Log-cv_scrd_trnData$Price_Log_Pred)**2)),"\n")
#FINAL XGB 1-RMSLE :  0.7818561
#LB : 0.782622

preds_sub <- preds_sub / num_folds
summary(preds_sub)
test01$Price_Log_Pred <- preds_sub
preds_sub <- (10**preds_sub)-1
summary(preds_sub)

write_csv(cv_scrd_trnData, "C:\\Kaggle\\BooksPrice\\CV Scrd Trn Datasets\\20191003_Stacking01_DS.csv")
write_csv(test01, "C:\\Kaggle\\BooksPrice\\CV Scrd Tst Datasets\\20191003_Stacking01_DS.csv")

submission <- read.xlsx("C:\\Kaggle\\BooksPrice\\Participants_Data\\Sample_Submission.xlsx",
                        sheet = 1,
                        startRow = 1,
                        colNames = TRUE)

submission$Price <- preds_sub
write.xlsx(submission, "C:\\Kaggle\\BooksPrice\\Submissions\\20191003_Stacking01_DS.xlsx", colNames = TRUE)