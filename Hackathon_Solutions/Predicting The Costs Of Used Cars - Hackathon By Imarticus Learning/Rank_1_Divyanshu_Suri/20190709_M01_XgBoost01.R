setwd("C:\\Kaggle\\Cars\\Data\\")

library(Metrics)
library(readr)
library(Metrics)
library(xgboost)
library(sqldf)
library(openxlsx)

train01 <- read.csv("Data_Train_Mod01.csv",stringsAsFactors = FALSE)
test01 <- read.csv("Data_Test_Mod01.csv",stringsAsFactors = FALSE)

combined01 <- rbind(train01,test01)

names(combined01)
feature.names <- names(combined01)[!(names(combined01) %in% c("id","Price"))]

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

for (f in feature.names) {
  if (class(combined01[[f]])=="character") {
    cat("VARIABLE : ",f,"\n")
    levels <- unique(c(combined01[[f]]))
    combined01[[f]] <- as.integer(factor(combined01[[f]], levels=levels))
  }
}

train02 <- combined01[!is.na(combined01$Price),]
test02 <- combined01[is.na(combined01$Price),]

train02 <- train02[order(train02$id),]
test02 <- test02[order(test02$id),]

sum(test02$Name %in% train02$Name)
sum(train02$Name %in% test02$Name)

summary(train02$Price)
hist(train02$Price)
hist(log(train02$Price))
hist(train02$Price**0.1)

train02$Price <- log(train02$Price)

#Building Model

num_folds <- 5
set.seed(501)
train_2 <- train02[sample(nrow(train02)),]
folds <- cut(seq(1,nrow(train_2)),breaks = num_folds,labels = FALSE)

param <- list(  objective = "reg:linear",
                booster = "gbtree",
                eta = 0.05,
                max_depth = 6,#4
                subsample = 0.9,
                colsample_bytree = 0.6,#0.6
                min_child_weight = 5#20
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
  
  preds <- predict(XGB_mod_1, data.matrix(testData[,feature.names]))
  
  cat(rmse(preds,testData$Price),"\n")
  
  test02[,paste("pred_XGB_",i,sep="")] <- predict(XGB_mod_1, data.matrix(test02[,feature.names]))
  
  testData$Predicted_Scores <- preds
  
  if (i == 1)
  {
    cv_preds <- preds
    cv_acts <- (testData$Price)
    cv_testData <- testData
    
    preds_sub <- test02[,paste("pred_XGB_",i,sep="")]
  }
  else
  {
    cv_preds <- c(cv_preds,preds)
    cv_acts <- c(cv_acts,(testData$Price))
    cv_testData <- rbind(cv_testData,testData)
    
    preds_sub <- preds_sub + test02[,paste("pred_XGB_",i,sep="")]
  }
}
cat("\nFINAL RMSE : ",rmse(cv_preds,cv_acts),"\n")

#FINAL RMSE :  0.1829215
#LB SCORE : 0.9356 , 0.0644
1 - 0.9356

preds_sub <- preds_sub / num_folds
summary(preds_sub)
test02$pred_XGB_Final <- exp(preds_sub)
summary(test02$pred_XGB_Final)

write_csv(test02, "C:\\Kaggle\\Cars\\Test_Scored\\20190709_XGB01_TEST_DS.csv")
write_csv(cv_testData, "C:\\Kaggle\\Cars\\CV_Scored\\20190709_XGB01_CVTRAIN_DS.csv")

sub <- data.frame(Price = test02$pred_XGB_Final)
write.xlsx(sub, "C:\\Kaggle\\Cars\\Submission\\20190709_XGB01_TEST_DS.xlsx", colNames = TRUE, rowNames = FALSE)
