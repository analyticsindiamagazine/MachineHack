library(stringr)
library(sqldf)
library(dummies)

train01 <- read.csv("C:\\Kaggle\\Cars\\Data\\Data_Train.csv", stringsAsFactors = FALSE)
test01 <- read.csv("C:\\Kaggle\\Cars\\Data\\Data_Test.csv", stringsAsFactors = FALSE)

col_to_create <- names(train01)[!(names(train01) %in% names(test01))]

test01[,col_to_create] <- NA

combined_data <- rbind.data.frame(train01,test01)
names(combined_data)

Mile <- (strsplit(combined_data$Mileage, " "))
for(i in 1:nrow(combined_data)) {
  if(i == 1){
    temp <- Mile[[i]][2]
  }
  else{
    temp <- c(temp,Mile[[i]][2])
  }
}
table(temp)

Engine <- (strsplit(combined_data$Engine, " "))
for(i in 1:nrow(combined_data)) {
  if(i == 1){
    temp2 <- Engine[[i]][2]
  }
  else{
    temp2 <- c(temp2,Engine[[i]][2])
  }
}
table(temp2)

Power <- (strsplit(combined_data$Power, " "))
for(i in 1:nrow(combined_data)) {
  if(i == 1){
    temp3 <- Power[[i]][2]
  }
  else{
    temp3 <- c(temp3,Power[[i]][2])
  }
}
table(temp3)

NP <- (strsplit(combined_data$New_Price, " "))
for(i in 1:nrow(combined_data)) {
  if(i == 1){
    temp4 <- NP[[i]][2]
  }
  else{
    temp4 <- c(temp4,NP[[i]][2])
  }
}
table(temp4)

# [1] "id"                "Name"              "Location"          "Year"              "Kilometers_Driven" "Fuel_Type"         "Transmission"
# [8] "Owner_Type"        "Mileage"           "Engine"            "Power"             "Seats"             "New_Price"         "Price"

str(combined_data)
combined_data2 <- combined_data

combined_data2$Mileage <- as.numeric(sapply(Mile ,`[`, 1))
summary(combined_data2$Mileage)
combined_data[is.na(combined_data2$Mileage),]

combined_data2$Engine <- as.numeric(sapply(Engine ,`[`, 1))
summary(combined_data2$Engine)
combined_data[is.na(combined_data2$Engine),]

combined_data2$Power <- as.numeric(sapply(Power ,`[`, 1))
summary(combined_data2$Power)
combined_data[is.na(combined_data2$Power),]

combined_data2$New_Price2 <- as.numeric(sapply(NP ,`[`, 1))
summary(combined_data2$New_Price2)
combined_data[is.na(combined_data2$New_Price2),]
combined_data2$New_Price2 <- ifelse(str_detect(combined_data2$New_Price, coll("Cr")) == TRUE,
                                    combined_data2$New_Price2*100,combined_data2$New_Price2)
combined_data2$New_Price <- combined_data2$New_Price2
combined_data2$New_Price2 <- NULL

str(combined_data2)

combined_data2 <- combined_data2[order(combined_data2$Name,
                                       combined_data2$Year,
                                       combined_data2$Transmission,
                                       combined_data2$Engine,
                                       combined_data2$Power,
                                       combined_data2$Seats,
                                       combined_data2$Kilometers_Driven),]

chk01 <- sqldf("select Name, avg(New_Price) as New_Price_Avg
                from combined_data2
                group by Name")

combined_data2 <- sqldf("select a.*, b.New_Price_Avg
                         from combined_data2 as a inner join chk01 as b on a.Name = b.Name")
summary(combined_data2$New_Price)
combined_data2$New_Price <- ifelse(is.na(combined_data2$New_Price), combined_data2$New_Price_Avg,
                                   combined_data2$New_Price)
summary(combined_data2$New_Price)
combined_data2$New_Price_Avg <- NULL

for(fname in names(combined_data2)) {
  cat(fname," : ",sum(is.na(combined_data2[[fname]])),"\n")
}

remove(chk01)

chk01 <- sqldf("select Name, avg(Mileage) as Mileage_Avg, avg(Engine) as Engine_Avg,
                             avg(Power) as Power_Avg, avg(Seats) as Seats_Avg
                from combined_data2
                group by Name")

combined_data2 <- sqldf("select a.*, b.Mileage_Avg, b.Engine_Avg, b.Power_Avg, b.Seats_Avg
                         from combined_data2 as a inner join chk01 as b on a.Name = b.Name")

summary(combined_data2$Mileage)
summary(combined_data2$Engine)
summary(combined_data2$Power)
summary(combined_data2$Seats)

combined_data2$Mileage <- ifelse(is.na(combined_data2$Mileage), combined_data2$Mileage_Avg,
                                 combined_data2$Mileage)
combined_data2$Engine <- ifelse(is.na(combined_data2$Engine), combined_data2$Engine_Avg,
                                combined_data2$Engine)
combined_data2$Power <- ifelse(is.na(combined_data2$Power), combined_data2$Power_Avg,
                               combined_data2$Power)
combined_data2$Seats <- ifelse(is.na(combined_data2$Seats), combined_data2$Seats_Avg,
                               combined_data2$Seats)

summary(combined_data2$Mileage)
summary(combined_data2$Engine)
summary(combined_data2$Power)
summary(combined_data2$Seats)

combined_data2$Mileage_Avg <- NULL
combined_data2$Engine_Avg <- NULL
combined_data2$Power_Avg <- NULL
combined_data2$Seats_Avg <- NULL

remove(chk01,Engine,Mile,NP,Power,col_to_create,fname,i,temp,temp2,temp3,temp4)

for(fname in names(combined_data2)) {
  cat(fname," : ",sum(is.na(combined_data2[[fname]])),"\n")
}

combined_data2 <- combined_data2[order(combined_data2$id),]

for(fname in names(combined_data2)) {
  cat(fname," : Train Mis = ",sum(is.na(combined_data2[!is.na(combined_data2$Price),fname])),
      " : Test Mis = ",sum(is.na(combined_data2[is.na(combined_data2$Price),fname])),"\n")
}

str(combined_data2)

combined_data3 <- dummy.data.frame(combined_data2,
                                   names = c("Location","Fuel_Type","Transmission","Owner_Type"))

str(combined_data3)

train02 <- combined_data3[!is.na(combined_data3$Price),]
test02 <- combined_data3[is.na(combined_data3$Price),]

write.csv(train02,"C:\\Kaggle\\Cars\\Data\\Data_Train_Mod01.csv",row.names = FALSE)
write.csv(test02,"C:\\Kaggle\\Cars\\Data\\Data_Test_Mod01.csv",row.names = FALSE)