setwd("C:\\Kaggle\\BooksPrice\\Participants_Data")

library(Metrics)
library(readr)
library(Metrics)
library(xgboost)
library(sqldf)
library(openxlsx)
library(dummies)
library(stringr)

train01 <- read.csv("Data_Train.csv",stringsAsFactors = FALSE)
test01 <- read.csv("Data_Test.csv",stringsAsFactors = FALSE)

test01$Price <- NA

combo01 <- rbind.data.frame(train01,test01)

summary((as.numeric(sapply(strsplit(combo01$Reviews," "),`[`, 1))))
combo01$ReviewRating <- (as.numeric(sapply(strsplit(combo01$Reviews," "),`[`, 1)))
summary(combo01$ReviewRating)

summary(as.numeric(sapply(strsplit(str_replace_all(combo01$Ratings,",","")," "),`[`, 1)))
combo01$ReviewCount <- (as.numeric(sapply(strsplit(str_replace_all(combo01$Ratings,",","")," "),`[`, 1)))
summary(combo01$ReviewCount)

CreateIndicators <- function(DATAFRAME, VARNAME, FREQ_CUTOFF) {
  a <- data.frame(table(DATAFRAME[,VARNAME]))
  a <- a[order(-a$Freq),]
  a <- subset(a, Freq > FREQ_CUTOFF)
  
  for(val_lev in a$Var1) {
    DATAFRAME[,val_lev] <- ifelse(DATAFRAME[,VARNAME] == val_lev,1,0)
  }
  
  return(DATAFRAME)
}

combo01 <- CreateIndicators(combo01,"BookCategory",5)

table(combo01$Genre)
combo01 <- CreateIndicators(combo01,"Genre",5)

table(combo01$Author)
combo01 <- CreateIndicators(combo01,"Author",5)

temp1 <- strsplit(combo01$Edition," ")
for(i in 1:length(temp1)) {
  a <- temp1[[i]]
  b <- a[length(a)]
  if(i == 1) { YearVals <- b }
  else { YearVals <- c(YearVals, b) }
}
table(YearVals)
YearVals <- as.numeric(YearVals)
summary(YearVals)

combo01$Num_Years <- 2019 - YearVals
summary(combo01$Num_Years)

combo01$Title[is.na(combo01$Num_Years)]
combo01$Author[is.na(combo01$Num_Years)]
combo01$Synopsis[is.na(combo01$Num_Years)]
combo01$id[is.na(combo01$Num_Years)]

NumYearsImputation <- data.frame(Title = c("Long Walk to Freedom: Illustrated Children's edition (Picture Book Edition)"
                                           ,"Alfred's Basic Adult All-in-One Course: Lesson, Theory, Technic (Alfred's Basic Adult Piano Course)"
                                           ,"Fundamentals of Drawing Portraits: A Practical and Inspirational Course"
                                           ,"Cartooning, The Professional Step-by-Step Guide to: Learn to draw cartoons with over 1500 practical illustrations; all you need to know to create ... for digital enhancement and simple animation"
                                           ,"Amma Tell Me About Raksha Bandhan!"
                                           ,"Figure it out for Yourself"
                                           ,"An Introduction to Linguistics: Language, Grammar and Semantics"
                                           ,"Swimming: Swimming Made Easy: Beginner and Expert Strategies for Becoming a Better Swimmer (Swimming Secrets Tips Coaching Training Strategy)"
                                           ,"The Merchant of Venice (Text with Paraphrase) (Ratna Sagar Shakespeare)"
                                           ,"The Human Face of Big Data"
                                           ,"Living Language Dothraki: A Conversational Language Course Based on the Hit Original HBO Series Game of Thrones (Living Language Courses)"
                                           ,"Wise and Otherwise: A salute to Life"
                                           ,"The Armada Legacy (Ben Hope)"
                                           ,"Madhubani Art: Indian Art Series"
                                           ,"Frank Miller's Sin City Volume 2: A Dame to Kill For 3rd Edition"
                                           ,"While the Light Lasts (The Agatha Christie Collection)"
                                           ,"Turning Points : A Journey Through Challanges: A Journey Through Challenges"
                                           ,"Mountaineering: The Freedom of the Hills"
                                           ,"Indian Tibet Tibetan India: The Cultural Legacy of the Western Himalayas"
                                           ,"Pashu"
                                           ,"Sanskrit is Fun: A Sanskrit Coursebook for Beginner - Part - 1"
                                           ,"34 Bubblegums and Candies"
                                           ,"Gandhi: My Life is My Message"
                                           ,"Measurement"
                                           ,"The Spy Chronicles: RAW, ISI and the Illusion of Peace"
                                           ,"Sachin Tendulkar: The Man Cricket Loved Back"
                                           ,"Max Payne 3: The Complete Series"
                                           ,"Pain is Really Strange"
                                           ,"Frank Miller's Sin City Volume 2: A Dame to Kill For 3rd Edition"
                                           ,"Cartooning, The Professional Step-by-Step Guide to: Learn to draw cartoons with over 1500 practical illustrations; all you need to know to create ... for digital enhancement and simple animation"),
                                 id = c(170
                                        ,236
                                        ,583
                                        ,973
                                        ,1234
                                        ,1559
                                        ,1606
                                        ,1632
                                        ,1644
                                        ,1770
                                        ,2102
                                        ,2230
                                        ,2661
                                        ,2780
                                        ,3512
                                        ,3876
                                        ,3961
                                        ,4037
                                        ,4404
                                        ,5118
                                        ,5861
                                        ,98
                                        ,179
                                        ,397
                                        ,580
                                        ,956
                                        ,1180
                                        ,1192
                                        ,1409
                                        ,1542),
                                 Num_Years_Imp = c(2013
                                                   ,1994
                                                   ,2010
                                                   ,2018
                                                   ,2018
                                                   ,2003
                                                   ,2007
                                                   ,2017
                                                   ,2013
                                                   ,2014
                                                   ,2014
                                                   ,2005
                                                   ,2013
                                                   ,2016
                                                   ,2005
                                                   ,1997
                                                   ,2014
                                                   ,2017
                                                   ,2016
                                                   ,2014
                                                   ,2012
                                                   ,2010
                                                   ,2013
                                                   ,2014
                                                   ,2018
                                                   ,2015
                                                   ,2013
                                                   ,2015
                                                   ,2010
                                                   ,2018),
                                 stringsAsFactors = FALSE
                                 )

NumYearsImputation$Num_Years_Imp <- 2019 - NumYearsImputation$Num_Years_Imp

combo02 <- sqldf("select a.*, b.Num_Years_Imp
                  from combo01 as a left join NumYearsImputation as b on a.Title = b.Title and a.id = b.id")

combo02$Num_Years <- ifelse(is.na(combo02$Num_Years),combo02$Num_Years_Imp,combo02$Num_Years)
summary(combo02$Num_Years)

combo02$Num_Years_Imp <- NULL

remove(temp1,NumYearsImputation,a,b,i,YearVals)

EditionIndList <- c("Import"
                    ,"Illustrated"
                    ,"Special Edition"
                    ,"Unabridged"
                    ,"Abridged"
                    ,"Audiobook"
                    ,"Box set"
                    ,"International Edition"
                    ,"Special Edition"
                    ,"Student Edition"
                    ,"Paperback"
                    ,"Board book"
                    ,"Cards"
                    ,"Flexibound"
                    ,"Hardcover"
                    ,"Leather Bound"
                    ,"Library Binding"
                    ,"Loose Leaf"
                    ,"Mass Market Paperback"
                    ,"Perfect Paperback"
                    ,"Plastic Comb"
                    ,"Product Bundle"
                    ,"Sheet music"
                    ,"Spiral-bound"
                    ,"Tankobon Softcover")

for(IndV in EditionIndList) {
  combo02[,IndV] <- as.numeric(str_detect(combo02$Edition,IndV))
  print(IndV)
  print(table(combo02[,IndV]))
}

remove(EditionIndList,IndV,CreateIndicators)

train02 <- subset(combo02,!is.na(Price))
test02 <- subset(combo02,is.na(Price))

FeatureNames <- names(train02)[!(names(train02) %in% names(train01))]

train02$Price_Log <- log(train02$Price+1)

cor_matrix <- cor(train02[,c(FeatureNames,"Price_Log")])
cor_matrix <- data.frame(cor_matrix[,ncol(cor_matrix)])
cor_matrix$VarName <- rownames(cor_matrix)
cor_matrix <- data.frame(cor_matrix[order(-abs(cor_matrix[,1])),])

write.csv(train02,"Data_Train02.csv",row.names = FALSE)
write.csv(test02,"Data_Test02.csv",row.names = FALSE)
write.csv(FeatureNames,"FeatureNames02.csv",row.names = FALSE)