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

train01 <- read.csv("Data_Train_Mod01.csv",stringsAsFactors = FALSE)
test01 <- read.csv("Data_Test_Mod01.csv",stringsAsFactors = FALSE)

train01$NA_ROW_SUM <- rowSums(is.na(train01))
table(train01$NA_ROW_SUM)

test01$NA_ROW_SUM <- rowSums(is.na(test01))-1
table(test01$NA_ROW_SUM)

combined01 <- rbind(train01,test01)

names(combined01)
feature.names <- names(combined01)[!(names(combined01) %in% c("id","Price","New_Price"))]

for (f in feature.names) {
  if(sum(is.na(combined01[[f]])) > 0) {
    cat(f," : ",sum(is.na(combined01[[f]])),"\n")
  }
}

combined01[is.na(combined01$Mileage),]
sum(combined01$Fuel_TypeElectric)
table(test01$Fuel_TypeElectric)
combined01[is.na(combined01$Mileage),"Mileage"] <- -999
sum(is.na(combined01[["Mileage"]]))

table(combined01[is.na(combined01$Engine),"Name"])
EngineImputationTable <- data.frame(Name = c("BMW 5 Series 520d Sedan"
                                             ,"Fiat Punto 1.4 Emotion"
                                             ,"Hyundai i20 new Sportz AT 1.4"
                                             ,"Hyundai Santro Xing XG"
                                             ,"Mahindra TUV 300 P4"
                                             ,"Maruti Swift 1.3 VXI ABS"
                                             ,"Skoda Laura 1.8 TSI Ambition"
                                             ,"Fiat Punto 1.2 Dynamic"
                                             ,"Honda City 1.3 DX"
                                             ,"Hyundai Santro GLS II - Euro II"
                                             ,"Land Rover Range Rover 3.0 D"
                                             ,"Maruti Swift 1.3 LXI"
                                             ,"Maruti Swift 1.3 ZXI"
                                             ,"Toyota Etios Liva Diesel TRD Sportivo"
                                             ,"Fiat Punto 1.3 Emotion"
                                             ,"Honda CR-V AT With Sun Roof"
                                             ,"Hyundai Santro LP zipPlus"
                                             ,"Land Rover Range Rover Sport 2005 2012 Sport"
                                             ,"Maruti Swift 1.3 VXi"
                                             ,"Maruti Wagon R Vx"),
                                    EngineImpute = c(2000
                                                     ,1400
                                                     ,1400
                                                     ,1086
                                                     ,2200
                                                     ,1300
                                                     ,1800
                                                     ,1200
                                                     ,1300
                                                     ,1086
                                                     ,3000
                                                     ,1300
                                                     ,1300
                                                     ,1364
                                                     ,1300
                                                     ,1597
                                                     ,1086
                                                     ,2993
                                                     ,1300
                                                     ,998),
                                    stringsAsFactors = FALSE)
combined01 <- sqldf("select a.*, b.EngineImpute
                     from combined01 as a left join EngineImputationTable as b on a.Name = b.Name")
combined01$Engine <- ifelse(is.na(combined01$Engine),combined01$EngineImpute,combined01$Engine)
combined01$EngineImpute <- NULL
sum(is.na(combined01$Engine))

table(combined01[is.na(combined01$Power),"Name"])
PowerImputationTable <- data.frame(Name = c("Audi A4 3.2 FSI Tiptronic Quattro"
                                            ,"Fiat Petra 1.2 EL"
                                            ,"Fiat Punto 1.4 Emotion"
                                            ,"Ford Endeavour XLT TDCi 4X4"
                                            ,"Ford Fiesta 1.6 SXI ABS Duratec"
                                            ,"Honda CR-V AT With Sun Roof"
                                            ,"Hyundai Santro AT"
                                            ,"Hyundai Santro GLS I - Euro II"
                                            ,"Hyundai Santro GS zipDrive - Euro II"
                                            ,"Hyundai Santro LS zipDrive Euro I"
                                            ,"Hyundai Santro Xing XG AT eRLX Euro III"
                                            ,"Hyundai Santro Xing XL AT eRLX Euro II"
                                            ,"Hyundai Santro Xing XO"
                                            ,"Hyundai Santro Xing XP"
                                            ,"Mahindra Jeep MM 540 DP"
                                            ,"Maruti 1000 AC"
                                            ,"Maruti Esteem LX BSII"
                                            ,"Maruti Swift 1.3 LXI"
                                            ,"Maruti Swift 1.3 ZXI"
                                            ,"Mercedes-Benz E-Class 220 CDI"
                                            ,"Porsche Cayman 2009-2012 S"
                                            ,"Skoda Laura Classic 1.8 TSI"
                                            ,"Toyota Etios Liva Diesel TRD Sportivo"
                                            ,"Volkswagen Jetta 2007-2011 1.9 L TDI"
                                            ,"BMW 5 Series 520d Sedan"
                                            ,"Fiat Punto 1.2 Dynamic"
                                            ,"Fiat Siena 1.2 ELX"
                                            ,"Ford Fiesta 1.4 SXI Duratorq"
                                            ,"Hindustan Motors Contessa 2.0 DSL"
                                            ,"Honda CR-V Sport"
                                            ,"Hyundai Santro DX"
                                            ,"Hyundai Santro GLS II - Euro II"
                                            ,"Hyundai Santro LP - Euro II"
                                            ,"Hyundai Santro LS zipPlus"
                                            ,"Hyundai Santro Xing XG eRLX Euro III"
                                            ,"Hyundai Santro Xing XL AT eRLX Euro III"
                                            ,"Hyundai Santro Xing XO CNG"
                                            ,"Land Rover Range Rover 3.0 D"
                                            ,"Mahindra Jeep MM 550 PE"
                                            ,"Maruti Baleno LXI - BSIII"
                                            ,"Maruti Esteem Vxi"
                                            ,"Maruti Swift 1.3 VXi"
                                            ,"Maruti Swift VDI BSIV W ABS"
                                            ,"Nissan Micra Diesel"
                                            ,"Porsche Cayman 2009-2012 S tiptronic"
                                            ,"Smart Fortwo CDI AT"
                                            ,"Toyota Qualis Fleet A3"
                                            ,"Chevrolet Optra 1.6 Elite"
                                            ,"Fiat Punto 1.3 Emotion"
                                            ,"Ford Endeavour Hurricane LE"
                                            ,"Ford Fiesta 1.4 SXi TDCi"
                                            ,"Honda City 1.3 DX"
                                            ,"Hyundai i20 new Sportz AT 1.4"
                                            ,"Hyundai Santro GLS I - Euro I"
                                            ,"Hyundai Santro GS"
                                            ,"Hyundai Santro LP zipPlus"
                                            ,"Hyundai Santro Xing XG"
                                            ,"Hyundai Santro Xing XL"
                                            ,"Hyundai Santro Xing XL eRLX Euro III"
                                            ,"Hyundai Santro Xing XO eRLX Euro II"
                                            ,"Land Rover Range Rover Sport 2005 2012 Sport"
                                            ,"Mahindra TUV 300 P4"
                                            ,"Maruti Baleno Vxi"
                                            ,"Maruti Estilo LXI"
                                            ,"Maruti Swift 1.3 VXI ABS"
                                            ,"Maruti Wagon R Vx"
                                            ,"Nissan Teana 230jM"
                                            ,"Skoda Laura 1.8 TSI Ambition"
                                            ,"Tata Indica DLS"
                                            ,"Toyota Qualis RS E2"),
                                   PowerImpute = c(253
                                                   ,72
                                                   ,89
                                                   ,197
                                                   ,101
                                                   ,118
                                                   ,63
                                                   ,63
                                                   ,62
                                                   ,62
                                                   ,63
                                                   ,63
                                                   ,62
                                                   ,63
                                                   ,62
                                                   ,60
                                                   ,85
                                                   ,74
                                                   ,82
                                                   ,194
                                                   ,265
                                                   ,158
                                                   ,67
                                                   ,105
                                                   ,190
                                                   ,67
                                                   ,72
                                                   ,68
                                                   ,54
                                                   ,152
                                                   ,63
                                                   ,63
                                                   ,63
                                                   ,63
                                                   ,63
                                                   ,63
                                                   ,63
                                                   ,306
                                                   ,84
                                                   ,94
                                                   ,84
                                                   ,82
                                                   ,74
                                                   ,63
                                                   ,265
                                                   ,45
                                                   ,75
                                                   ,103
                                                   ,90
                                                   ,154
                                                   ,68
                                                   ,100
                                                   ,89
                                                   ,62
                                                   ,62
                                                   ,62
                                                   ,62
                                                   ,62
                                                   ,62
                                                   ,62
                                                   ,252
                                                   ,100
                                                   ,94
                                                   ,67
                                                   ,83
                                                   ,67
                                                   ,170
                                                   ,158
                                                   ,55
                                                   ,75),
                                   stringsAsFactors = FALSE)

combined01 <- sqldf("select a.*, b.PowerImpute
                     from combined01 as a left join PowerImputationTable as b on a.Name = b.Name")
combined01$Power <- ifelse(is.na(combined01$Power),combined01$PowerImpute,combined01$Power)
combined01$PowerImpute <- NULL
sum(is.na(combined01$Power))

table(combined01[is.na(combined01$Seats),"Name"])
SeatsImputationTable <- data.frame(Name = c("BMW 5 Series 520d Sedan"
                                            ,"Fiat Punto 1.4 Emotion"
                                            ,"Honda City 1.3 DX"
                                            ,"Hyundai i20 new Sportz AT 1.4"
                                            ,"Hyundai Santro Xing XG"
                                            ,"Mahindra TUV 300 P4"
                                            ,"Maruti Swift 1.3 VXi"
                                            ,"Maruti Wagon R Vx"
                                            ,"Fiat Punto 1.2 Dynamic"
                                            ,"Ford Endeavour Hurricane LE"
                                            ,"Honda CR-V AT With Sun Roof"
                                            ,"Hyundai Santro GLS II - Euro II"
                                            ,"Land Rover Range Rover 3.0 D"
                                            ,"Maruti Estilo LXI"
                                            ,"Maruti Swift 1.3 VXI ABS"
                                            ,"Skoda Laura 1.8 TSI Ambition"
                                            ,"Fiat Punto 1.3 Emotion"
                                            ,"Ford Figo Diesel"
                                            ,"Honda Jazz 2020 Petrol"
                                            ,"Hyundai Santro LP zipPlus"
                                            ,"Land Rover Range Rover Sport 2005 2012 Sport"
                                            ,"Maruti Swift 1.3 LXI"
                                            ,"Maruti Swift 1.3 ZXI"
                                            ,"Toyota Etios Liva Diesel TRD Sportivo"),
                                   SeatsImpute = c(5
                                                   ,5
                                                   ,5
                                                   ,5
                                                   ,5
                                                   ,9
                                                   ,5
                                                   ,5
                                                   ,5
                                                   ,7
                                                   ,5
                                                   ,5
                                                   ,5
                                                   ,5
                                                   ,5
                                                   ,5
                                                   ,5
                                                   ,5
                                                   ,5
                                                   ,5
                                                   ,5
                                                   ,5
                                                   ,5
                                                   ,5),
                                   stringsAsFactors = FALSE)
combined01 <- sqldf("select a.*, b.SeatsImpute
                     from combined01 as a left join SeatsImputationTable as b on a.Name = b.Name")
combined01$Seats <- ifelse(is.na(combined01$Seats),combined01$SeatsImpute,combined01$Seats)
combined01$SeatsImpute <- NULL
sum(is.na(combined01$Seats))

for (f in feature.names) {
  if(sum(is.na(combined01[[f]])) > 0) {
    cat(f," : ",sum(is.na(combined01[[f]])),"\n")
  }
}

Name_Mapping <- sqldf("select Name, avg(Price) as Avg_Price
                       from combined01
                       group by Name
                       order by Avg_Price desc")

Name_Mapping$Rank <- rank(Name_Mapping$Avg_Price, ties.method = "first")
Name_Mapping[is.na(Name_Mapping$Avg_Price),"Rank"] <- NA

Name_Mapping <- Name_Mapping[order(Name_Mapping$Name),]
Name_Mapping$Rank_Lag <- lag(Name_Mapping$Rank)
Name_Mapping$Rank_Lag2 <- lag(Name_Mapping$Rank,2)
Name_Mapping$Rank_Fin <- ifelse(is.na(Name_Mapping$Rank),Name_Mapping$Rank_Lag,Name_Mapping$Rank)
Name_Mapping$Rank_Fin <- ifelse(is.na(Name_Mapping$Rank_Fin),Name_Mapping$Rank_Lag2,Name_Mapping$Rank_Fin)
summary(Name_Mapping$Rank_Fin)

combined01 <- sqldf("select a.*, b.Rank_Fin
                     from combined01 as a inner join Name_Mapping as b on a.Name = b.Name")

combined01$Name <- combined01$Rank_Fin
combined01$Rank_Fin <- NULL

summary(combined01$Name)

corr_matrix <- cor(combined01, use = "pairwise.complete.obs")

combined01$TrainTestInd <- ifelse(!is.na(combined01$Price),"TRAIN","TEST")
table(combined01$TrainTestInd)

combined02 <- sqldf("select a.TrainTestInd, a.id, a.Name, a.Year, avg(b.Price) as Lag_Price
                     from combined01 as a left join combined01 as b on a.Name = b.Name and a.Year > b.Year
                     group by a.TrainTestInd, a.id, a.Name, a.Year
                     order by a.TrainTestInd, a.id, a.Name, a.Year")

summary(combined01$Power)
hist(combined01$Power)

combined01$Power_Group <- ifelse(combined01$Power <= 50,"01",
                                 ifelse(combined01$Power <= 100,"02",
                                        ifelse(combined01$Power <= 150,"03","04")))

table(combined01$Power_Group,combined01$TrainTestInd)

combined03 <- sqldf("select a.TrainTestInd, a.id, a.Power_Group, a.Year, avg(b.Price) as Lag_Price2
                     from combined01 as a left join combined01 as b on a.Power_Group = b.Power_Group and a.Year > b.Year
                     group by a.TrainTestInd, a.id, a.Power_Group, a.Year
                     order by a.TrainTestInd, a.id, a.Power_Group, a.Year")

combined04 <- sqldf("select a.*, b.Lag_Price, c.Lag_Price2
                     from combined01 as a inner join combined02 as b on a.TrainTestInd = b.TrainTestInd and a.id = b.id
                                          inner join combined03 as c on a.TrainTestInd = c.TrainTestInd and a.id = c.id")

summary(combined04$Lag_Price)
summary(combined04$Lag_Price2)

combined04$Lag_Price_M <- ifelse(is.na(combined04$Lag_Price),1,0)
combined04[is.na(combined04$Lag_Price),"Lag_Price"] <- 0

combined04$Lag_Price2_M <- ifelse(is.na(combined04$Lag_Price2),1,0)
combined04[is.na(combined04$Lag_Price2),"Lag_Price2"] <- 0

summary(combined04$Lag_Price)
summary(combined04$Lag_Price2)

summary(combined04$Lag_Price_M)
summary(combined04$Lag_Price2_M)

feature.names <- c(feature.names,"Lag_Price","Lag_Price2","Lag_Price_M","Lag_Price2_M")

train02 <- combined04[!is.na(combined04$Price),]
test02 <- combined04[is.na(combined04$Price),]

train02 <- train02[order(train02$id),]
test02 <- test02[order(test02$id),]

corr_matrix <- cor(train02[,c(feature.names,"Price")], use = "pairwise.complete.obs")

train02$Price <- log(train02$Price+1)

num_folds <- 5
set.seed(10)
train_2 <- train02[sample(nrow(train02)),]
folds <- cut(seq(1,nrow(train_2)),breaks = num_folds,labels = FALSE)

#######################################################################################
############################MODEL 1####################################################
#######################################################################################

param <- list(  objective = "reg:linear",
                booster = "gbtree",
                eta = 0.01,
                max_depth = 6,#5
                subsample = 0.9,
                colsample_bytree = 0.4,#0.6
                min_child_weight = 10#5
                )

for (i in 1:num_folds)
{
  cat("RUNNING FOR ",i," ")
  
  testIndexes <- which(folds==i,arr.ind=TRUE)
  
  testData <- train_2[testIndexes, ]
  trainData <- train_2[-testIndexes, ]
  
  dtrain <- xgb.DMatrix(data = data.matrix(trainData[,feature.names]),
                        label = trainData[,"Price"])
  
  dvalid <- xgb.DMatrix(data = data.matrix(testData[,feature.names]),
                        label = testData[,"Price"])
  
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
  
  cat(rmse(preds,testData$Price),"\n")
  
  preds2 <- predict(XGB_mod_1, data.matrix(test02[,feature.names]))
  
  testData$pred_XGB_Final <- preds
  
  if (i == 1)
  {
    cv_preds <- preds
    cv_acts <- (testData$Price)
    cv_testData <- testData
    
    preds_sub <- preds2
  }
  else
  {
    cv_preds <- c(cv_preds,preds)
    cv_acts <- c(cv_acts,(testData$Price))
    cv_testData <- rbind(cv_testData,testData)
    
    preds_sub <- preds_sub + preds2
  }
}
cat("\nFINAL RMSE : ",rmse(cv_preds,cv_acts),"\n")
#FINAL RMSE :  0.119727

preds_sub <- preds_sub / num_folds
summary(preds_sub)
test02$pred_XGB_Final <- preds_sub
train_2 <- cv_testData

remove(cv_testData,param,i,testIndexes,testData,preds,preds2,preds_sub,cv_preds,cv_acts)

#######################################################################################
############################MODEL 2####################################################
#######################################################################################

k_vals <- c(5)

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
                         y = trainData[,"Price"],
                         k = k1,
                         algorithm = "kd_tree")
    
    preds <- knn_mod_1$pred
    
    cat(rmse(preds,testData$Price),"\n")
    
    preds2 <- knn.reg(train = trainData[,feature.names],test = test02[,feature.names],
                      y = trainData[,"Price"],k = k1,algorithm = "kd_tree")$pred
    
    testData$pred_KNN_Final <- preds
    
    if (i == 1)
    {
      cv_preds <- preds
      cv_acts <- (testData$Price)
      cv_testData <- testData
      
      preds_sub <- preds2
    }
    else
    {
      cv_preds <- c(cv_preds,preds)
      cv_acts <- c(cv_acts,(testData$Price))
      cv_testData <- rbind(cv_testData,testData)
      
      preds_sub <- preds_sub + preds2
    }
  }
  cat("\nFINAL RMSE : ",rmse(cv_preds,cv_acts)," : k = ",k1,"\n")
}

#FINAL RMSE :  0.3052945  : k =  5
preds_sub <- preds_sub / num_folds
summary(preds_sub)
test02$pred_KNN_Final <- preds_sub
train_2 <- cv_testData

remove(cv_testData,param,i,testIndexes,testData,preds,preds2,preds_sub,cv_preds,cv_acts)

#######################################################################################
############################MODEL 3####################################################
#######################################################################################

size_vals <- c(8)
iter_vals <- c(2000)

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
                           y = trainData[,"Price"],
                           size = s,
                           trace = TRUE,
                           linout = TRUE,
                           maxit = it,
                           skip = TRUE)
      
      preds <- (predict(ANN_mod_1,testData[,feature.names]))
      cat(rmse(preds,testData$Price),"\n") #0.1417051
      
      preds2 <- (predict(ANN_mod_1,test02[,feature.names]))
      
      testData$pred_ANN_Final <- preds
      
      if (i == 1)
      {
        cv_preds <- preds
        cv_acts <- (testData$Price)
        cv_testData <- testData
        
        preds_sub <- preds2
      }
      else
      {
        cv_preds <- c(cv_preds,preds)
        cv_acts <- c(cv_acts,(testData$Price))
        cv_testData <- rbind(cv_testData,testData)
        
        preds_sub <- preds_sub + preds2
      }
    }
    cat("\nFINAL RMSE : ",rmse(cv_preds,cv_acts)," ; size, iter : ",s,", ",it,"\n")
  }
}
cat("\nFINAL RMSE : ",rmse(cv_preds,cv_acts),"\n")

#FINAL RMSE :  0.1826579  ; size, iter :  8 ,  2000
preds_sub <- preds_sub / num_folds
summary(preds_sub)
test02$pred_ANN_Final <- preds_sub
train_2 <- cv_testData

remove(cv_testData,param,i,testIndexes,testData,preds,preds2,preds_sub,cv_preds,cv_acts)

#######################################################################################
############################MODEL 4####################################################
#######################################################################################

mtry_vals <- c(5)
ntree_vals <- c(50)
sampsize_vals <- c(0.8)
nodesize_vals <- c(5)

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
                                    y = (trainData[,"Price"]),
                                    mtry = i1,
                                    ntree = i2,
                                    sampsize = round(i3*nrow(trainData)),
                                    nodesize = i4,
                                    nPerm = 5,
                                    do.trace = TRUE,
                                    keep.forest = TRUE,
                                    importance = TRUE )
          
          preds <- (predict(RF_mod_1,testData[,feature.names]))
          cat(rmse(preds,testData$Price),"\n")
          
          preds2 <- (predict(RF_mod_1,test02[,feature.names]))
          testData$pred_RF_Final <- preds
          
          if (i == 1)
          {
            cv_preds <- preds
            cv_acts <- (testData$Price)
            cv_testData <- testData
            
            preds_sub <- preds2
          }
          else
          {
            cv_preds <- c(cv_preds,preds)
            cv_acts <- c(cv_acts,(testData$Price))
            cv_testData <- rbind(cv_testData,testData)
            
            preds_sub <- preds_sub + preds2
          }
        }
        cat("\nFINAL RSQUARE : ",rmse(cv_preds,cv_acts)," ; mtry, ntree, sampsize, nodesize : ",i1,", ",i2,", ",i3,", ",i4,"\n")
      }
    }
  }
}

#FINAL RSQUARE :  0.1456624  ; mtry, ntree, sampsize, nodesize :  5 ,  50 ,  0.8 ,  5
preds_sub <- preds_sub / num_folds
summary(preds_sub)
test02$pred_RF_Final <- preds_sub
train_2 <- cv_testData

remove(cv_testData,param,i,testIndexes,testData,preds,preds2,preds_sub,cv_preds,cv_acts)

#######################################################################################
############################MODEL 5####################################################
#######################################################################################

num_trees <- c(500)
depth_vals <- c(6)
minobs <- c(5)
bagf <- c(0.9)

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
                               y = (trainData[,"Price"]),
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
          cat(rmse(preds,testData$Price),"\n")
          
          preds2 <- (predict(GBM_mod_1,test02[,feature.names],best.iter))
          
          testData$pred_GBM_Final <- preds
          
          if (i == 1)
          {
            cv_preds <- preds
            cv_acts <- (testData$Price)
            cv_testData <- testData
            
            preds_sub <- preds2
          }
          else
          {
            cv_preds <- c(cv_preds,preds)
            cv_acts <- c(cv_acts,(testData$Price))
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

#FINAL RMSE :  0.1354233  ; numTrees, IntDepth, MinObs, BagFrac, best.iter :  500 ,  6 ,  5 ,  0.9 ,  145
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
  LM_mod_1 <- lm(Price ~ Name+LocationDelhi+LocationMumbai+Fuel_TypeDiesel+TransmissionManual+Mileage+Lag_Price+LocationAhmedabad+LocationHyderabad+LocationPune+Fuel_TypeElectric+Owner_TypeFirst+Engine+Lag_Price2+LocationBangalore+LocationJaipur+Year+Fuel_TypeLPG+Owner_TypeFourth...Above+Power+Lag_Price_M+LocationChennai+LocationKochi+Kilometers_Driven+Fuel_TypePetrol+Owner_TypeSecond+Seats+Lag_Price2_M+LocationCoimbatore+LocationKolkata+Fuel_TypeCNG+TransmissionAutomatic+Owner_TypeThird+NA_ROW_SUM,
                 data = trainData)
  
  print(summary(LM_mod_1))
  
  preds <- (predict(LM_mod_1,testData))
  cat(rmse(preds,testData$Price),"\n")
  
  preds2 <- (predict(LM_mod_1,test02))
  
  testData$pred_LM_Final <- preds
  
  if (i == 1)
  {
    cv_preds <- preds
    cv_acts <- (testData$Price)
    cv_testData <- testData
    
    preds_sub <- preds2
  }
  else
  {
    cv_preds <- c(cv_preds,preds)
    cv_acts <- c(cv_acts,(testData$Price))
    cv_testData <- rbind(cv_testData,testData)
    
    preds_sub <- preds_sub + preds2
  }
}
cat("\nFINAL RMSE : ",rmse((cv_preds),(cv_acts)),"\n")
#FINAL RMSE :  0.19505

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
                     y = trainData[,"Price"],
                     neurons = 8,
                     verbose = TRUE)
  
  preds <- (predict(brnn_mod_1,testData[,feature.names]))
  cat(rmse(preds,testData$Price),"\n") #0.3120006
  
  preds2 <- (predict(brnn_mod_1,test02[,feature.names]))
  
  testData$pred_BRNN_Final <- preds
  
  if (i == 1)
  {
    cv_preds <- preds
    cv_acts <- (testData$Price)
    cv_testData <- testData
    
    preds_sub <- preds2
  }
  else
  {
    cv_preds <- c(cv_preds,preds)
    cv_acts <- c(cv_acts,(testData$Price))
    cv_testData <- rbind(cv_testData,testData)
    
    preds_sub <- preds_sub + preds2
  }
}
cat("\nFINAL RMSE : ",rmse((cv_preds),(cv_acts)),"\n")
#FINAL RMSE :  0.151476

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
                       y = trainData[,"Price"],
                       degree = 10,
                       nprune = 10,
                       trace = 1)
  
  preds <- (predict(earth_mod_1,testData[,feature.names]))
  cat(rmse(preds,testData$Price),"\n") #0.3131404
  
  preds2 <- (predict(earth_mod_1,test02[,feature.names]))
  
  testData$pred_EARTH_Final <- preds
  
  if (i == 1)
  {
    cv_preds <- preds
    cv_acts <- (testData$Price)
    cv_testData <- testData
    
    preds_sub <- preds2
  }
  else
  {
    cv_preds <- c(cv_preds,preds)
    cv_acts <- c(cv_acts,(testData$Price))
    cv_testData <- rbind(cv_testData,testData)
    
    preds_sub <- preds_sub + preds2
  }
}
cat("\nFINAL RMSE : ",rmse((cv_preds),(cv_acts)),"\n")
#FINAL RMSE :  0.1493947

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
                max_depth = 2,#8
                subsample = 0.6,
                colsample_bytree = 0.9,#0.8
                min_child_weight = 10
)

for (i in 1:num_folds)
{
  cat("RUNNING FOR ",i," ")
  
  testIndexes <- which(folds==i,arr.ind=TRUE)
  
  testData <- train_2[testIndexes, ]
  trainData <- train_2[-testIndexes, ]
  
  dtrain <- xgb.DMatrix(data = data.matrix(trainData[,feature.names2]),
                        label = trainData[,"Price"])
  
  dvalid <- xgb.DMatrix(data = data.matrix(testData[,feature.names2]),
                        label = testData[,"Price"])
  
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
  
  cat(rmse(preds,testData$Price),"\n")
  
  preds2 <- predict(XGB_mod_1, data.matrix(test02[,feature.names2]))
  
  testData$pred_XGB_Stack_Final <- preds
  
  if (i == 1)
  {
    cv_preds <- preds
    cv_acts <- (testData$Price)
    cv_testData <- testData
    
    preds_sub <- preds2
  }
  else
  {
    cv_preds <- c(cv_preds,preds)
    cv_acts <- c(cv_acts,(testData$Price))
    cv_testData <- rbind(cv_testData,testData)
    
    preds_sub <- preds_sub + preds2
  }
}
cat("\nFINAL RMSE : ",rmse(cv_preds,cv_acts),"\n")
#FINAL RMSE :  0.1205614
#LB Price : 0.9195

preds_sub <- preds_sub / num_folds
summary(preds_sub)
preds2 <- exp(preds_sub)-1
summary(preds2)

sub <- data.frame(Price = preds2)
write.xlsx(sub, "C:\\Kaggle\\Cars\\Submission\\20190712_StackModel01_TEST_DS.xlsx", colNames = TRUE, rowNames = FALSE)

remove(cv_testData,param,i,testIndexes,testData,preds,preds2,preds_sub,cv_preds,cv_acts,sub)