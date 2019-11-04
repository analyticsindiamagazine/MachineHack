setwd("C:\\Kaggle\\Cars\\Data\\")

library(Metrics)
library(readr)
library(Metrics)
library(xgboost)
library(sqldf)
library(openxlsx)
library(Metrics)
library(readr)
library(Metrics)
library(xgboost)
library(sqldf)
library(plyr)
library(dplyr)
library(caret)
library(nnet)
library(e1071)
library(caretEnsemble)
library(randomForest)
library(parallel)
library(doParallel)
library(FNN)
library(gbm)
library(lightgbm)
library(dummies)

train01 <- read.csv("C:\\Kaggle\\BooksPrice\\Participants_Data\\Data_Train07.csv",stringsAsFactors = FALSE)
test01 <- read.csv("C:\\Kaggle\\BooksPrice\\Participants_Data\\Data_Test07.csv",stringsAsFactors = FALSE)

feature.names <- names(test01)[2:ncol(test01)]
feature.names

train01$Price_Log <- log10(train01$Price)
hist(train01$Price_Log)

num_folds <- 5
train_2 <- train01
folds <- train_2$FOLD_NUM

test02 <- test01

#######################################################################################
############################MODEL 1####################################################
#######################################################################################

param <- list(  objective = "reg:linear",
                booster = "gbtree",
                eta = 0.01,
                max_depth = 3,
                subsample = 0.4,
                colsample_bytree = 0.3,
                min_child_weight = 1
)

for (i in 1:num_folds)
{
  cat("RUNNING FOR ",i," ")
  
  testIndexes <- which(folds==i,arr.ind=TRUE)
  
  testData <- train_2[testIndexes, ]
  trainData <- train_2[-testIndexes, ]
  
  dtrain <- xgb.DMatrix(data = data.matrix(trainData[,feature.names]),
                        label = trainData[,"Price_Log"])
  
  dvalid <- xgb.DMatrix(data = data.matrix(testData[,feature.names]),
                        label = testData[,"Price_Log"])
  
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
                            print.every.n = 25
  )
  #[2072]	train-rmse:0.044950	val-rmse:0.148246
  
  preds <- predict(XGB_mod_1, data.matrix(testData[,feature.names]))
  
  cat(rmse(preds,testData$Price_Log),"\n")
  
  preds2 <- predict(XGB_mod_1, data.matrix(test02[,feature.names]))
  
  testData$pred_XGB_Final <- preds
  
  if (i == 1)
  {
    cv_preds <- preds
    cv_acts <- (testData$Price_Log)
    cv_testData <- testData
    
    preds_sub <- preds2
  }
  else
  {
    cv_preds <- c(cv_preds,preds)
    cv_acts <- c(cv_acts,(testData$Price_Log))
    cv_testData <- rbind(cv_testData,testData)
    
    preds_sub <- preds_sub + preds2
  }
}
cat("\nFINAL RMSE : ",rmse(cv_preds,cv_acts),"\n")
#FINAL RMSE :  0.210163

preds_sub <- preds_sub / num_folds
summary(preds_sub)
test02$pred_XGB_Final <- preds_sub
train_2 <- cv_testData

remove(cv_testData,param,i,testIndexes,testData,preds,preds2,preds_sub,cv_preds,cv_acts)

#######################################################################################
############################MODEL 2####################################################
#######################################################################################

k_vals <- c(30)

for(k1 in k_vals)
{
  for (i in 1:num_folds)
  {
    cat("RUNNING FOR ",i," ")
    
    testIndexes <- which(folds==i,arr.ind=TRUE)
    
    testData <- train_2[testIndexes, ]
    trainData <- train_2[-testIndexes, ]
    
    knn_mod_1 <- knn.reg(train = trainData[,feature.names[!(feature.names %in% c())]],
                         test = testData[,feature.names],
                         y = trainData[,"Price_Log"],
                         k = k1,
                         algorithm = "kd_tree")
    
    preds <- knn_mod_1$pred
    
    cat(rmse(preds,testData$Price_Log),"\n")
    
    preds2 <- knn.reg(train = trainData[,feature.names],test = test02[,feature.names],
                      y = trainData[,"Price_Log"],k = k1,algorithm = "kd_tree")$pred
    
    testData$pred_KNN_Final <- preds
    
    if (i == 1)
    {
      cv_preds <- preds
      cv_acts <- (testData$Price_Log)
      cv_testData <- testData
      
      preds_sub <- preds2
    }
    else
    {
      cv_preds <- c(cv_preds,preds)
      cv_acts <- c(cv_acts,(testData$Price_Log))
      cv_testData <- rbind(cv_testData,testData)
      
      preds_sub <- preds_sub + preds2
    }
  }
  cat("\nFINAL RMSE : ",rmse(cv_preds,cv_acts)," : k = ",k1,"\n")
}

#FINAL RMSE :  0.2141943  : k =  30
preds_sub <- preds_sub / num_folds
summary(preds_sub)
test02$pred_KNN_Final <- preds_sub
train_2 <- cv_testData

remove(cv_testData,param,i,testIndexes,testData,preds,preds2,preds_sub,cv_preds,cv_acts)

#######################################################################################
############################MODEL 3####################################################
#######################################################################################

size_vals <- c(1)
iter_vals <- c(100)

for(s in size_vals)
{
  for(it in iter_vals)
  {
    for (i in 1:num_folds)
    {
      cat("RUNNING FOR ",i," ")
      
      testIndexes <- which(folds==i,arr.ind=TRUE)
      
      testData <- train_2[testIndexes, ]
      trainData <- train_2[-testIndexes, ]
      set.seed(501+i)
      ANN_mod_1 <- pcaNNet(x = trainData[,feature.names],
                           y = trainData[,"Price_Log"],
                           size = s,
                           trace = TRUE,
                           linout = TRUE,
                           maxit = it,
                           skip = TRUE)
      
      preds <- (predict(ANN_mod_1,testData[,feature.names]))
      cat(rmse(preds,testData$Price_Log),"\n") #0.1417051
      
      preds2 <- (predict(ANN_mod_1,test02[,feature.names]))
      
      testData$pred_ANN_Final <- preds
      
      if (i == 1)
      {
        cv_preds <- preds
        cv_acts <- (testData$Price_Log)
        cv_testData <- testData
        
        preds_sub <- preds2
      }
      else
      {
        cv_preds <- c(cv_preds,preds)
        cv_acts <- c(cv_acts,(testData$Price_Log))
        cv_testData <- rbind(cv_testData,testData)
        
        preds_sub <- preds_sub + preds2
      }
    }
    cat("\nFINAL RMSE : ",rmse(cv_preds,cv_acts)," ; size, iter : ",s,", ",it,"\n")
  }
}
cat("\nFINAL RMSE : ",rmse(cv_preds,cv_acts),"\n")

#FINAL RMSE :  0.2113337  ; size, iter :  1 ,  100
preds_sub <- preds_sub / num_folds
summary(preds_sub)
test02$pred_ANN_Final <- preds_sub
train_2 <- cv_testData

remove(cv_testData,param,i,testIndexes,testData,preds,preds2,preds_sub,cv_preds,cv_acts)

#######################################################################################
############################MODEL 4####################################################
#######################################################################################

mtry_vals <- c(9)
ntree_vals <- c(100)
sampsize_vals <- c(0.5)
nodesize_vals <- c(1)

for(i1 in mtry_vals)
{
  for(i2 in ntree_vals)
  {
    for(i3 in sampsize_vals)
    {
      for(i4 in nodesize_vals)
      {
        for (i in 1:num_folds)
        {
          cat("RUNNING FOR ",i," ")
          
          testIndexes <- which(folds==i,arr.ind=TRUE)
          
          testData <- train_2[testIndexes, ]
          trainData <- train_2[-testIndexes, ]
          set.seed(501+i)
          RF_mod_1 <- randomForest( x = trainData[,feature.names],
                                    y = (trainData[,"Price_Log"]),
                                    mtry = i1,
                                    ntree = i2,
                                    sampsize = round(i3*nrow(trainData)),
                                    nodesize = i4,
                                    nPerm = 5,
                                    do.trace = TRUE,
                                    keep.forest = TRUE,
                                    importance = TRUE )
          
          preds <- (predict(RF_mod_1,testData[,feature.names]))
          cat(rmse(preds,testData$Price_Log),"\n")
          
          preds2 <- (predict(RF_mod_1,test02[,feature.names]))
          testData$pred_RF_Final <- preds
          
          if (i == 1)
          {
            cv_preds <- preds
            cv_acts <- (testData$Price_Log)
            cv_testData <- testData
            
            preds_sub <- preds2
          }
          else
          {
            cv_preds <- c(cv_preds,preds)
            cv_acts <- c(cv_acts,(testData$Price_Log))
            cv_testData <- rbind(cv_testData,testData)
            
            preds_sub <- preds_sub + preds2
          }
        }
        cat("\nFINAL RSQUARE : ",rmse(cv_preds,cv_acts)," ; mtry, ntree, sampsize, nodesize : ",i1,", ",i2,", ",i3,", ",i4,"\n")
      }
    }
  }
}

#FINAL RSQUARE :  0.2091497  ; mtry, ntree, sampsize, nodesize :  9 ,  100 ,  0.5 ,  1
preds_sub <- preds_sub / num_folds
summary(preds_sub)
test02$pred_RF_Final <- preds_sub
train_2 <- cv_testData

remove(cv_testData,param,i,testIndexes,testData,preds,preds2,preds_sub,cv_preds,cv_acts)

#######################################################################################
############################MODEL 5####################################################
#######################################################################################

num_trees <- c(500)
depth_vals <- c(3)
minobs <- c(1)
bagf <- c(0.5)

for(i1 in num_trees)
{
  for(i2 in depth_vals)
  {
    for(i3 in minobs)
    {
      for(i4 in bagf)
      {
        for (i in 1:num_folds)
        {
          cat("RUNNING FOR ",i," ")
          
          testIndexes <- which(folds==i,arr.ind=TRUE)
          
          testData <- train_2[testIndexes, ]
          trainData <- train_2[-testIndexes, ]
          set.seed(501+i)
          GBM_mod_1 <- gbm.fit(x = trainData[,feature.names],
                               y = (trainData[,"Price_Log"]),
                               distribution = "gaussian",
                               n.trees = i1,
                               interaction.depth = i2,
                               n.minobsinnode = i3,
                               bag.fraction = i4,
                               shrinkage = 0.05,
                               verbose = TRUE)
          
          best.iter <- gbm.perf(GBM_mod_1,method="OOB")
          # best.iter <- i1
          
          preds <- (predict(GBM_mod_1,testData[,feature.names],best.iter))
          cat(rmse(preds,testData$Price_Log),"\n")
          
          preds2 <- (predict(GBM_mod_1,test02[,feature.names],best.iter))
          
          testData$pred_GBM_Final <- preds
          
          if (i == 1)
          {
            cv_preds <- preds
            cv_acts <- (testData$Price_Log)
            cv_testData <- testData
            
            preds_sub <- preds2
          }
          else
          {
            cv_preds <- c(cv_preds,preds)
            cv_acts <- c(cv_acts,(testData$Price_Log))
            cv_testData <- rbind(cv_testData,testData)
            
            preds_sub <- preds_sub + preds2
          }
        }
        cat("\nFINAL RMSE : ",rmse((cv_preds),(cv_acts))," ; numTrees, IntDepth, MinObs, BagFrac, best.iter : ",i1,", ",i2,", ",i3,", ",i4,", ",best.iter,"\n")
        #remove(cv_preds,cv_acts,cv_ids,preds,GBM_mod_1,testData,trainData,testIndexes,best.iter)
      }
    }
  }
}

#FINAL RMSE :  0.2117489  ; numTrees, IntDepth, MinObs, BagFrac, best.iter :  500 ,  3 ,  1 ,  0.5 ,  87
preds_sub <- preds_sub / num_folds
summary(preds_sub)
test02$pred_GBM_Final <- preds_sub
train_2 <- cv_testData

remove(cv_testData,param,i,testIndexes,testData,preds,preds2,preds_sub,cv_preds,cv_acts)

#######################################################################################
############################MODEL 6####################################################
#######################################################################################

for (i in 1:num_folds)
{
  cat("RUNNING FOR ",i," ")
  
  testIndexes <- which(folds==i,arr.ind=TRUE)
  
  testData <- train_2[testIndexes, ]
  trainData <- train_2[-testIndexes, ]
  LM_mod_1 <- lm(Price_Log ~ .,
                 data = trainData[,c("Price_Log",feature.names)])
  
  print(summary(LM_mod_1))
  
  preds <- (predict(LM_mod_1,testData))
  cat(rmse(preds,testData$Price_Log),"\n")
  
  preds2 <- (predict(LM_mod_1,test02))
  
  testData$pred_LM_Final <- preds
  
  if (i == 1)
  {
    cv_preds <- preds
    cv_acts <- (testData$Price_Log)
    cv_testData <- testData
    
    preds_sub <- preds2
  }
  else
  {
    cv_preds <- c(cv_preds,preds)
    cv_acts <- c(cv_acts,(testData$Price_Log))
    cv_testData <- rbind(cv_testData,testData)
    
    preds_sub <- preds_sub + preds2
  }
}
cat("\nFINAL RMSE : ",rmse((cv_preds),(cv_acts)),"\n")
#FINAL RMSE :  0.208761

preds_sub <- preds_sub / num_folds
summary(preds_sub)
test02$pred_LM_Final <- preds_sub
train_2 <- cv_testData

remove(cv_testData,param,i,testIndexes,testData,preds,preds2,preds_sub,cv_preds,cv_acts)

#######################################################################################
############################MODEL 7####################################################
#######################################################################################

library(brnn)

for (i in 1:num_folds)
{
  cat("RUNNING FOR ",i," ")
  
  testIndexes <- which(folds==i,arr.ind=TRUE)
  
  testData <- train_2[testIndexes, ]
  trainData <- train_2[-testIndexes, ]
  brnn_mod_1 <- brnn(x = as.matrix(trainData[,feature.names]),
                     y = trainData[,"Price_Log"],
                     neurons = 2,
                     verbose = TRUE)
  
  preds <- (predict(brnn_mod_1,testData[,feature.names]))
  cat(rmse(preds,testData$Price_Log),"\n") #0.3120006
  
  preds2 <- (predict(brnn_mod_1,test02[,feature.names]))
  
  testData$pred_BRNN_Final <- preds
  
  if (i == 1)
  {
    cv_preds <- preds
    cv_acts <- (testData$Price_Log)
    cv_testData <- testData
    
    preds_sub <- preds2
  }
  else
  {
    cv_preds <- c(cv_preds,preds)
    cv_acts <- c(cv_acts,(testData$Price_Log))
    cv_testData <- rbind(cv_testData,testData)
    
    preds_sub <- preds_sub + preds2
  }
}
cat("\nFINAL RMSE : ",rmse((cv_preds),(cv_acts)),"\n")
#FINAL RMSE :  0.2087997

preds_sub <- preds_sub / num_folds
summary(preds_sub)
test02$pred_BRNN_Final <- preds_sub
train_2 <- cv_testData

remove(cv_testData,param,i,testIndexes,testData,preds,preds2,preds_sub,cv_preds,cv_acts)

#######################################################################################
############################MODEL 8####################################################
#######################################################################################

library(earth)

for (i in 1:num_folds)
{
  cat("RUNNING FOR ",i," ")
  
  testIndexes <- which(folds==i,arr.ind=TRUE)
  
  testData <- train_2[testIndexes, ]
  trainData <- train_2[-testIndexes, ]
  set.seed(111+i)
  earth_mod_1 <- earth(x = as.matrix(trainData[,feature.names]),
                       y = trainData[,"Price_Log"],
                       degree = 10,
                       nprune = 10,
                       trace = 1)
  
  preds <- (predict(earth_mod_1,testData[,feature.names]))
  cat(rmse(preds,testData$Price_Log),"\n") #0.3131404
  
  preds2 <- (predict(earth_mod_1,test02[,feature.names]))
  
  testData$pred_EARTH_Final <- preds
  
  if (i == 1)
  {
    cv_preds <- preds
    cv_acts <- (testData$Price_Log)
    cv_testData <- testData
    
    preds_sub <- preds2
  }
  else
  {
    cv_preds <- c(cv_preds,preds)
    cv_acts <- c(cv_acts,(testData$Price_Log))
    cv_testData <- rbind(cv_testData,testData)
    
    preds_sub <- preds_sub + preds2
  }
}
cat("\nFINAL RMSE : ",rmse((cv_preds),(cv_acts)),"\n")
#FINAL RMSE :  0.2121694

preds_sub <- preds_sub / num_folds
summary(preds_sub)
test02$pred_EARTH_Final <- preds_sub
train_2 <- cv_testData

remove(cv_testData,param,i,testIndexes,testData,preds,preds2,preds_sub,cv_preds,cv_acts)

#######################################################################################
############################MODEL ENSEMBLE 2###########################################
#######################################################################################

feature.names2 <- c(feature.names,"pred_XGB_Final","pred_KNN_Final","pred_ANN_Final","pred_RF_Final","pred_GBM_Final","pred_LM_Final",
                    "pred_BRNN_Final","pred_EARTH_Final")

param <- list(  objective = "reg:linear",
                booster = "gbtree",
                eta = 0.001,
                max_depth = 2,
                subsample = 0.8,
                colsample_bytree = 0.7,
                min_child_weight = 1
)

for (i in 1:num_folds)
{
  cat("RUNNING FOR ",i," ")
  
  testIndexes <- which(folds==i,arr.ind=TRUE)
  
  testData <- train_2[testIndexes, ]
  trainData <- train_2[-testIndexes, ]
  
  dtrain <- xgb.DMatrix(data = data.matrix(trainData[,feature.names2]),
                        label = trainData[,"Price_Log"])
  
  dvalid <- xgb.DMatrix(data = data.matrix(testData[,feature.names2]),
                        label = testData[,"Price_Log"])
  
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
                            print.every.n = 25
  )
  #[6160]	train-rmse:0.112288	val-rmse:0.118700
  
  preds <- predict(XGB_mod_1, data.matrix(testData[,feature.names2]))
  
  cat(rmse(preds,testData$Price_Log),"\n")
  
  preds2 <- predict(XGB_mod_1, data.matrix(test02[,feature.names2]))
  
  testData$pred_XGB_Stack_Final <- preds
  
  if (i == 1)
  {
    cv_preds <- preds
    cv_acts <- (testData$Price_Log)
    cv_testData <- testData
    
    preds_sub <- preds2
  }
  else
  {
    cv_preds <- c(cv_preds,preds)
    cv_acts <- c(cv_acts,(testData$Price_Log))
    cv_testData <- rbind(cv_testData,testData)
    
    preds_sub <- preds_sub + preds2
  }
}
cat("\nFINAL RMSE : ",rmse(cv_preds,cv_acts),"\n")
#FINAL RMSE :  0.2086342
#LB : 0.209349

preds_sub <- preds_sub / num_folds
summary(preds_sub)
preds2 <- (10**preds_sub)-1
summary(preds2)

tst_file <- data.frame(Price = preds2)
hs <- createStyle(textDecoration = "BOLD", halign = "center")
write.xlsx(tst_file, "C:\\Kaggle\\BooksPrice\\Submissions\\20191018_R_STACK_01_DS.xlsx", colNames = TRUE, headerStyle = hs)