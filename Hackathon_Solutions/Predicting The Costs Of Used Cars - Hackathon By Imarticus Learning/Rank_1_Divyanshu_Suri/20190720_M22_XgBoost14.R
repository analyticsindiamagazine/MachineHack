setwd("C:\\Kaggle\\Cars\\Data\\")

library(Metrics)
library(readr)
library(Metrics)
library(xgboost)
library(sqldf)
library(openxlsx)

train01 <- read.csv("Data_Train_Mod01.csv",stringsAsFactors = FALSE)
test01 <- read.csv("Data_Test_Mod01.csv",stringsAsFactors = FALSE)

temp1 <- read.csv("Data_Train.csv",stringsAsFactors = FALSE)
temp2 <- read.csv("Data_Test.csv",stringsAsFactors = FALSE)

train01 <- sqldf("select a.*, b.Owner_Type as Location from train01 as a inner join temp1 as b on a.id = b.id order by a.id")
test01 <- sqldf("select a.*, b.Owner_Type as Location from test01 as a inner join temp2 as b on a.id = b.id order by a.id")

combined01 <- rbind(train01,test01)

NAME_NEW <- (strsplit(toupper(train01$Name), " "))

for(i in 1:nrow(train01)) {
  if(i == 1){
    temp3 <- NAME_NEW[[i]][1]
  }
  else{
    temp3 <- c(temp3,NAME_NEW[[i]][1])
  }
}
table(temp3)

for(i in 1:nrow(train01)) {
  if(i == 1){
    temp4 <- NAME_NEW[[i]][2]
  }
  else{
    temp4 <- c(temp4,NAME_NEW[[i]][2])
  }
}
table(temp4)

CAR_NAME_U1 <- data.frame(table(temp3))
CAR_NAME_U2 <- data.frame(table(temp4))

CAR_NAME_U11 <- subset(CAR_NAME_U1, Freq > 10)
CAR_NAME_U21 <- subset(CAR_NAME_U2, Freq > 10)

for(car_comp in CAR_NAME_U11$temp3) {
  combined01[,paste("CompName_",car_comp,sep="")] <- ifelse(grepl(car_comp, toupper(combined01$Name)), 1, 0)
}

for(car_comp2 in CAR_NAME_U21$temp4) {
  combined01[,paste("CompNameCarName_",car_comp2,sep="")] <- ifelse(grepl(car_comp2, toupper(combined01$Name)), 1, 0)
}

NAME_NEW <- (strsplit(toupper(combined01$Name), " "))
combined01$CarCompName <- sapply(NAME_NEW ,`[`, 1)
table(combined01$CarCompName)
sum(is.na(combined01$CarCompName))

names(combined01)
feature.names <- names(combined01)[!(names(combined01) %in% c("id","Price","New_Price","Name","CarCompName","Location"))]

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

combined03 <- sqldf("select a.TrainTestInd, a.id, a.Power_Group, a.Year, avg(b.Price) as Lag_Price2,
                     min(b.Price) as Lag_Price2_MIN, max(b.Price) as Lag_Price2_MAX
                     from combined01 as a left join combined01 as b on a.Power_Group = b.Power_Group and a.Year > b.Year
                     group by a.TrainTestInd, a.id, a.Power_Group, a.Year
                     order by a.TrainTestInd, a.id, a.Power_Group, a.Year")

summary(combined01$Engine)
hist(combined01$Engine)

combined01$Engine_Group <- ifelse(combined01$Engine <= 1000,"01",
                                  ifelse(combined01$Engine <= 1250,"02",
                                         ifelse(combined01$Engine <= 1600,"03",
                                                ifelse(combined01$Engine <= 2000,"04","05"))))

table(combined01$Engine_Group,combined01$TrainTestInd)
table(combined01$Engine_Group,combined01$Power_Group)

combined03b <- sqldf("select a.TrainTestInd, a.id, a.Engine_Group, a.Year, avg(b.Price) as Lag_Price3,
                      min(b.Price) as Lag_Price3_MIN, max(b.Price) as Lag_Price3_MAX
                      from combined01 as a left join combined01 as b on a.Engine_Group = b.Engine_Group and a.Year > b.Year
                      group by a.TrainTestInd, a.id, a.Engine_Group, a.Year
                      order by a.TrainTestInd, a.id, a.Engine_Group, a.Year")

combined03c <- sqldf("select a.CarCompName, a.Year, avg(a.Price) as Avg_Price
                      from combined01 as a group by a.CarCompName, a.Year")

combined03d <- sqldf("select a.*, b.Avg_Price as Avg_Price_Lag1, c.Avg_Price as Avg_Price_Lag2, d.Avg_Price as Avg_Price_Lag3
                      from combined03c as a left join combined03c as b on a.CarCompName = b.CarCompName and a.Year = b.Year+1
                                            left join combined03c as c on a.CarCompName = c.CarCompName and a.Year = c.Year+2
                                            left join combined03c as d on a.CarCompName = d.CarCompName and a.Year = d.Year+3
                      order by a.CarCompName, a.Year")

combined03d$RateChng1 <- (combined03d$Avg_Price / combined03d$Avg_Price_Lag1) - 1
combined03d$RateChng2 <- (combined03d$Avg_Price_Lag1 / combined03d$Avg_Price_Lag2) - 1
combined03d$RateChng3 <- (combined03d$Avg_Price_Lag2 / combined03d$Avg_Price_Lag3) - 1

combined03e <- sqldf("select a.TrainTestInd, a.id, a.CarCompName, a.Year, avg(b.Price) as Lag_Price3,
                      min(b.Price) as Lag_Price3_MIN, max(b.Price) as Lag_Price3_MAX
                      from combined01 as a left join combined01 as b on a.CarCompName = b.CarCompName and a.Year > b.Year
                      group by a.TrainTestInd, a.id, a.CarCompName, a.Year
                      order by a.TrainTestInd, a.id, a.CarCompName, a.Year")

combined03f <- sqldf("select a.TrainTestInd, a.id, a.Power_Group, a.Year, avg(b.Price) as Lag_Price5
                      from combined01 as a left join combined01 as b on a.Power_Group = b.Power_Group and a.Year = b.Year+1
                      group by a.TrainTestInd, a.id, a.Power_Group, a.Year
                      order by a.TrainTestInd, a.id, a.Power_Group, a.Year")

combined03g <- sqldf("select a.TrainTestInd, a.id, a.CarCompName, a.Year, avg(b.Price) as Lag_Price6
                      from combined01 as a left join combined01 as b on a.CarCompName = b.CarCompName and a.Year = b.Year+1
                      group by a.TrainTestInd, a.id, a.CarCompName, a.Year
                      order by a.TrainTestInd, a.id, a.CarCompName, a.Year")

combined04 <- sqldf("select a.*, b.Lag_Price, c.Lag_Price2, d.Lag_Price3, c.Lag_Price2_MIN, c.Lag_Price2_MAX,
                            d.Lag_Price3_MIN, d.Lag_Price3_MAX, e.RateChng1, e.RateChng2, e.RateChng3,
                            f.Lag_Price3 as Lag_Price4, f.Lag_Price3_MIN as Lag_Price4_MIN, f.Lag_Price3_MAX as Lag_Price4_MAX,
                            g.Lag_Price5, h.Lag_Price6
                     from combined01 as a inner join combined02 as b on a.TrainTestInd = b.TrainTestInd and a.id = b.id
                                          inner join combined03 as c on a.TrainTestInd = c.TrainTestInd and a.id = c.id
                                          inner join combined03b as d on a.TrainTestInd = d.TrainTestInd and a.id = d.id
                                          inner join combined03d as e on a.CarCompName = e.CarCompName and a.Year = e.Year
                                          inner join combined03e as f on a.TrainTestInd = f.TrainTestInd and a.id = f.id
                                          inner join combined03f as g on a.TrainTestInd = g.TrainTestInd and a.id = g.id
                                          inner join combined03g as h on a.TrainTestInd = h.TrainTestInd and a.id = h.id")

summary(combined04$Lag_Price)
summary(combined04$Lag_Price2)
summary(combined04$Lag_Price3)
summary(combined04$Lag_Price2_MIN)
summary(combined04$Lag_Price3_MIN)
summary(combined04$Lag_Price2_MAX)
summary(combined04$Lag_Price3_MAX)
summary(combined04$RateChng1)
summary(combined04$RateChng2)
summary(combined04$RateChng3)
summary(combined04$Lag_Price4)
summary(combined04$Lag_Price5)
summary(combined04$Lag_Price6)
summary(combined04$Lag_Price4_MIN)
summary(combined04$Lag_Price4_MAX)

combined04$Lag_Price4_MIN_BY_MAX <- combined04$Lag_Price4_MIN / combined04$Lag_Price4_MAX
summary(combined04$Lag_Price4_MIN_BY_MAX)

temp_train <- combined04[!is.na(combined04$New_Price),]
temp_test <- combined04[is.na(combined04$New_Price),]

temp_train$New_Price <- log(temp_train$New_Price+1)

feature.names.New_Price <- c("LocationDelhi"
                             ,"LocationMumbai"
                             ,"Fuel_TypeDiesel"
                             ,"TransmissionManual"
                             ,"Year"
                             ,"Fuel_TypeLPG"
                             ,"CompName_DATSUN"
                             ,"CompName_JAGUAR"
                             ,"CompName_MERCEDES-BENZ"
                             ,"CompName_RENAULT"
                             ,"CompName_VOLVO"
                             ,"CompNameCarName_A-STAR"
                             ,"CompNameCarName_ALTO"
                             ,"CompNameCarName_BALENO"
                             ,"CompNameCarName_CELERIO"
                             ,"CompNameCarName_COMPASS"
                             ,"CompNameCarName_CRUZE"
                             ,"CompNameCarName_EECO"
                             ,"CompNameCarName_ERTIGA"
                             ,"CompNameCarName_FORTUNER"
                             ,"CompNameCarName_I10"
                             ,"CompNameCarName_INNOVA"
                             ,"CompNameCarName_LAURA"
                             ,"CompNameCarName_MOBILIO"
                             ,"CompNameCarName_OPTRA"
                             ,"CompNameCarName_Q7"
                             ,"CompNameCarName_SANTA"
                             ,"CompNameCarName_SUPERB"
                             ,"CompNameCarName_VENTO"
                             ,"CompNameCarName_X3"
                             ,"CompNameCarName_XYLO"
                             ,"LocationBangalore"
                             ,"LocationJaipur"
                             ,"LocationAhmedabad"
                             ,"LocationHyderabad"
                             ,"LocationPune"
                             ,"Fuel_TypeElectric"
                             ,"CompName_FIAT"
                             ,"CompName_JEEP"
                             ,"CompName_MINI"
                             ,"CompName_SKODA"
                             ,"CompNameCarName_3"
                             ,"CompNameCarName_A4"
                             ,"CompNameCarName_AMAZE"
                             ,"CompNameCarName_BEAT"
                             ,"CompNameCarName_CIAZ"
                             ,"CompNameCarName_COOPER"
                             ,"CompNameCarName_DUSTER"
                             ,"CompNameCarName_ELANTRA"
                             ,"CompNameCarName_ETIOS"
                             ,"CompNameCarName_GL-CLASS"
                             ,"CompNameCarName_I20"
                             ,"CompNameCarName_JAZZ"
                             ,"CompNameCarName_LINEA"
                             ,"CompNameCarName_NANO"
                             ,"CompNameCarName_PAJERO"
                             ,"CompNameCarName_RAPID"
                             ,"CompNameCarName_SANTRO"
                             ,"CompNameCarName_SWIFT"
                             ,"CompNameCarName_VERNA"
                             ,"CompNameCarName_X5"
                             ,"CompNameCarName_ZEN"
                             ,"Power"
                             ,"CompName_AUDI"
                             ,"CompName_FORD"
                             ,"CompName_LAND"
                             ,"CompName_MITSUBISHI"
                             ,"CompName_TATA"
                             ,"CompNameCarName_5"
                             ,"CompNameCarName_A6"
                             ,"CompNameCarName_AMEO"
                             ,"CompNameCarName_BOLERO"
                             ,"CompNameCarName_CITY"
                             ,"CompNameCarName_COROLLA"
                             ,"CompNameCarName_DZIRE"
                             ,"CompNameCarName_ELITE"
                             ,"CompNameCarName_FABIA"
                             ,"CompNameCarName_GLA"
                             ,"CompNameCarName_IKON"
                             ,"CompNameCarName_JETTA"
                             ,"CompNameCarName_M-CLASS"
                             ,"CompNameCarName_NEW"
                             ,"CompNameCarName_POLO"
                             ,"CompNameCarName_RITZ"
                             ,"CompNameCarName_SCORPIO"
                             ,"CompNameCarName_SX4"
                             ,"CompNameCarName_VITARA"
                             ,"CompNameCarName_XCENT"
                             ,"CompNameCarName_ZEST"
                             ,"LocationChennai"
                             ,"LocationKochi"
                             ,"Kilometers_Driven"
                             ,"Fuel_TypePetrol"
                             ,"Seats"
                             ,"CompName_BMW"
                             ,"CompName_HONDA"
                             ,"CompName_MAHINDRA"
                             ,"CompName_NISSAN"
                             ,"CompName_TOYOTA"
                             ,"CompNameCarName_7"
                             ,"CompNameCarName_ACCENT"
                             ,"CompNameCarName_AVEO"
                             ,"CompNameCarName_BRIO"
                             ,"CompNameCarName_CIVIC"
                             ,"CompNameCarName_CR-V"
                             ,"CompNameCarName_E-CLASS"
                             ,"CompNameCarName_ENDEAVOUR"
                             ,"CompNameCarName_FIESTA"
                             ,"CompNameCarName_GLE"
                             ,"CompNameCarName_INDICA"
                             ,"CompNameCarName_KUV"
                             ,"CompNameCarName_MANZA"
                             ,"CompNameCarName_OCTAVIA"
                             ,"CompNameCarName_Q3"
                             ,"CompNameCarName_ROVER"
                             ,"CompNameCarName_SSANGYONG"
                             ,"CompNameCarName_TERRANO"
                             ,"CompNameCarName_WAGON"
                             ,"CompNameCarName_XF"
                             ,"LocationCoimbatore"
                             ,"LocationKolkata"
                             ,"Fuel_TypeCNG"
                             ,"TransmissionAutomatic"
                             ,"Mileage"
                             ,"Engine"
                             ,"CompName_CHEVROLET"
                             ,"CompName_HYUNDAI"
                             ,"CompName_MARUTI"
                             ,"CompName_PORSCHE"
                             ,"CompName_VOLKSWAGEN"
                             ,"CompNameCarName_800"
                             ,"CompNameCarName_ACCORD"
                             ,"CompNameCarName_B"
                             ,"CompNameCarName_CAMRY"
                             ,"CompNameCarName_CLA"
                             ,"CompNameCarName_CRETA"
                             ,"CompNameCarName_ECOSPORT"
                             ,"CompNameCarName_EON"
                             ,"CompNameCarName_FIGO"
                             ,"CompNameCarName_GRAND"
                             ,"CompNameCarName_INDIGO"
                             ,"CompNameCarName_KWID"
                             ,"CompNameCarName_MICRA"
                             ,"CompNameCarName_OMNI"
                             ,"CompNameCarName_Q5"
                             ,"CompNameCarName_S"
                             ,"CompNameCarName_SUNNY"
                             ,"CompNameCarName_TIAGO"
                             ,"CompNameCarName_X1"
                             ,"CompNameCarName_XUV500")

num_folds <- 5
set.seed(10)
train_2 <- temp_train[sample(nrow(temp_train)),]
folds <- cut(seq(1,nrow(train_2)),breaks = num_folds,labels = FALSE)

param <- list(  objective = "reg:linear",
                booster = "gbtree",
                eta = 0.05,
                max_depth = 4,#5
                subsample = 0.9,
                colsample_bytree = 0.9,#0.6
                min_child_weight = 5#5
)

for (i in 1:num_folds)
{
  cat("RUNNING FOR ",i," ")
  
  testIndexes <- which(folds==i,arr.ind=TRUE)
  
  testData <- train_2[testIndexes, ]
  trainData <- train_2[-testIndexes, ]
  
  dtrain <- xgb.DMatrix(data = data.matrix(trainData[,feature.names.New_Price]),
                        label = trainData[,"New_Price"])
  
  dvalid <- xgb.DMatrix(data = data.matrix(testData[,feature.names.New_Price]),
                        label = testData[,"New_Price"])
  
  watchlist <- list(train = dtrain, val = dvalid)
  set.seed(501+i)
  XGB_mod_1 <- xgb.train(   params = param,
                            data = dtrain,
                            nrounds = 1000000,
                            verbose = 1,
                            watchlist = watchlist,
                            eval_metric = "rmse",
                            maximize = FALSE,
                            early.stop.round=300,
                            print.every.n = 25
  )#[2332]	train-rmse:0.017570	val-rmse:0.116346
  
  preds <- predict(XGB_mod_1, data.matrix(testData[,feature.names.New_Price]))
  
  cat(rmse(preds,testData$New_Price),"\n")
  
  preds2 <- predict(XGB_mod_1, data.matrix(temp_test[,feature.names.New_Price]))
  
  testData$P_New_Price <- preds
  
  if (i == 1)
  {
    cv_preds <- preds
    cv_acts <- (testData$New_Price)
    cv_testData <- testData
    
    preds_sub <- preds2
  }
  else
  {
    cv_preds <- c(cv_preds,preds)
    cv_acts <- c(cv_acts,(testData$New_Price))
    cv_testData <- rbind(cv_testData,testData)
    
    preds_sub <- preds_sub + preds2
  }
}
cat("\nFINAL RMSE : ",rmse(cv_preds,cv_acts),"\n")
#FINAL RMSE :  0.09935011

temp_test$P_New_Price <- preds_sub / 5

summary(cv_testData$P_New_Price)
summary(temp_test$P_New_Price)

combined05 <- rbind.data.frame(cv_testData,temp_test)

################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################

feature.names <- c(feature.names,"Lag_Price","Lag_Price2")
feature.names <- c(feature.names,"Lag_Price2_MIN","Lag_Price2_MAX")

feature.names <- names(combined05)[!(names(combined05) %in% c("id","Price","New_Price","Name",
                                                              "Engine_Group",
                                                              "Power_Group","TrainTestInd","CarCompName",
                                                              "Location",
                                                              "P_New_Price"))]

feature.names <- c(feature.names,"New_Price")
feature.names

combined06 <- combined05
combined06$New_Price <- ifelse(is.na(combined06$New_Price),combined06$P_New_Price,combined06$New_Price)
summary(combined06$New_Price)

train02 <- combined06[!is.na(combined06$Price),]
test02 <- combined06[is.na(combined06$Price),]

train02 <- train02[order(train02$id),]
test02 <- test02[order(test02$id),]

corr_matrix <- cor(train02[,c(feature.names,"Price")], use = "pairwise.complete.obs")

sum(test02$Name %in% train02$Name)
sum(train02$Name %in% test02$Name)

summary(train02$Price)
hist(train02$Price)
hist(log(train02$Price))
hist(train02$Price**0.1)

train02$Price <- log10(train02$Price+1)

write.csv(train02, "C:\\Kaggle\\Cars\\Data\\TrnDataForLGB2.csv", row.names = FALSE)
write.csv(test02, "C:\\Kaggle\\Cars\\Data\\TstDataForLGB2.csv", row.names = FALSE)

#Building Model

num_folds <- 5
set.seed(10)
train_2 <- train02[sample(nrow(train02)),]
folds <- cut(seq(1,nrow(train_2)),breaks = num_folds,labels = FALSE)

param <- list(  objective = "reg:linear",
                booster = "gbtree",
                eta = 0.01,
                max_depth = 12,#5
                subsample = 0.6,
                colsample_bytree = 0.2,#0.6
                min_child_weight = 5#5
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
                            nrounds = 1000000,
                            verbose = 1,
                            watchlist = watchlist,
                            eval_metric = "rmse",
                            maximize = FALSE,
                            early.stop.round=300,
                            print.every.n = 25
  )#[616]	train-rmse:0.013838	val-rmse:0.064744
  
  preds <- predict(XGB_mod_1, data.matrix(testData[,feature.names]))
  preds <- (10**preds)-1
  preds <- log(preds+1)
  testData$Price <- log(((10**testData$Price) - 1)+1)
  
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

#FINAL RMSE :   0.1481822
#LB SCORE : 0.9419

preds_sub <- preds_sub / num_folds
summary(preds_sub)
test02$pred_XGB_Final <- (10**preds_sub)-1
summary(test02$pred_XGB_Final)

write_csv(test02, "C:\\Kaggle\\Cars\\Test_Scored\\20190720_XGB14_TEST_DS.csv")
write_csv(cv_testData, "C:\\Kaggle\\Cars\\CV_Scored\\20190720_XGB14_CVTRAIN_DS.csv")

sub <- data.frame(Price = test02$pred_XGB_Final)
write.xlsx(sub, "C:\\Kaggle\\Cars\\Submission\\20190720_XGB14_TEST_DS.xlsx", colNames = TRUE, rowNames = FALSE)
