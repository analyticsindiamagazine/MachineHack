setwd("C:\\Kaggle\\Cars\\Submission")

library(readr)
library(sqldf)
library(Metrics)
library(openxlsx)

tst01 <- read.xlsx("20190716_XGB10_TEST_DS.xlsx",
                   sheet = 1,
                   startRow = 1,
                   colNames = TRUE)
tst02 <- read.xlsx("20190716_LGB01_TEST_DS.xlsx",
                   sheet = 1,
                   startRow = 1,
                   colNames = TRUE)
tst03 <- read.xlsx("20190718_XGB12_TEST_DS.xlsx",
                   sheet = 1,
                   startRow = 1,
                   colNames = TRUE)
tst04 <- read.xlsx("20190716_LGB01copy_TEST_DS.xlsx",
                   sheet = 1,
                   startRow = 1,
                   colNames = TRUE)
tst05 <- read.xlsx("20190716_Keras01_TEST_DS.xlsx",
                   sheet = 1,
                   startRow = 1,
                   colNames = TRUE)
tst06 <- read.xlsx("20190720_XGB13_TEST_DS.xlsx",
                   sheet = 1,
                   startRow = 1,
                   colNames = TRUE)

cormat <- cor(data.frame(x1 = (tst01$Price),
                         x2 = (tst02$Price),
                         x3 = (tst03$Price),
                         x4 = (tst04$Price),
                         x5 = (tst05$Price),
                         x6 = (tst06$Price)))

Pred01_2 <- ((tst06$Price)*0.15+(tst02$Price)*0+(tst03$Price)*0.45+(tst04$Price)*0.2+(tst05$Price)*0.2)
#0.9473
# Pred01_2 <- ((tst06$Price)*0.1+(tst02$Price)*0.15+(tst03$Price)*0.35+(tst04$Price)*0.2+(tst05$Price)*0.2)
#0.9473
# Pred01_2 <- ((tst01$Price)*0.1+(tst02$Price)*0.15+(tst03$Price)*0.35+(tst04$Price)*0.2+(tst05$Price)*0.2)
#0.9472
# Pred01_2 <- ((tst01$Price)*0.15+(tst02$Price)*0.2+(tst03$Price)*0.25+(tst04$Price)*0.2+(tst05$Price)*0.2)
#0.9471
summary(Pred01_2)

tst_file <- data.frame(Price = Pred01_2)
write_csv(tst_file, "20190720_ENSEMBLE_11_DS.csv")