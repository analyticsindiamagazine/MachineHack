setwd("C:\\Kaggle\\Cars\\Submission")

library(readr)
library(sqldf)
library(Metrics)
library(openxlsx)

tst01 <- read.xlsx("20190709_XGB01_TEST_DS.xlsx",
                   sheet = 1,
                   startRow = 1,
                   colNames = TRUE)
tst02 <- read.xlsx("20190709_XGB02_TEST_DS.xlsx",
                   sheet = 1,
                   startRow = 1,
                   colNames = TRUE)
tst03 <- read.xlsx("20190716_XGB10_TEST_DS.xlsx",
                   sheet = 1,
                   startRow = 1,
                   colNames = TRUE)
tst04 <- read.xlsx("20190716_LGB01_TEST_DS.xlsx",
                   sheet = 1,
                   startRow = 1,
                   colNames = TRUE)
tst05 <- read.xlsx("20190716_Keras01_TEST_DS.xlsx",
                   sheet = 1,
                   startRow = 1,
                   colNames = TRUE)
tst06 <- read.xlsx("20190716_Keras01copy_TEST_DS.xlsx",
                   sheet = 1,
                   startRow = 1,
                   colNames = TRUE)
tst07 <- read.xlsx("20190718_XGB12_TEST_DS.xlsx",
                   sheet = 1,
                   startRow = 1,
                   colNames = TRUE)
tst08 <- read.xlsx("20190720_XGB13_TEST_DS.xlsx",
                   sheet = 1,
                   startRow = 1,
                   colNames = TRUE)
tst09 <- read.xlsx("20190716_LGB01copy_TEST_DS.xlsx",
                   sheet = 1,
                   startRow = 1,
                   colNames = TRUE)
tst10 <- read.xlsx("20190715_StackModel02_TEST_DS.xlsx",
                   sheet = 1,
                   startRow = 1,
                   colNames = TRUE)
tst11 <- read.xlsx("20190711_XGB03_TEST_DS.xlsx",
                   sheet = 1,
                   startRow = 1,
                   colNames = TRUE)
tst12 <- read.xlsx("20190711_XGB04_TEST_DS.xlsx",
                   sheet = 1,
                   startRow = 1,
                   colNames = TRUE)
tst13 <- read.xlsx("20190715_XGB05_TEST_DS.xlsx",
                   sheet = 1,
                   startRow = 1,
                   colNames = TRUE)
tst14 <- read.xlsx("20190717_RF01_TEST_DS.xlsx",
                   sheet = 1,
                   startRow = 1,
                   colNames = TRUE)
tst15 <- read.xlsx("20190715_XGB06_TEST_DS.xlsx",
                   sheet = 1,
                   startRow = 1,
                   colNames = TRUE)
tst16 <- read.xlsx("20190715_XGB07_TEST_DS.xlsx",
                   sheet = 1,
                   startRow = 1,
                   colNames = TRUE)
tst17 <- read.xlsx("20190715_XGB08_TEST_DS.xlsx",
                   sheet = 1,
                   startRow = 1,
                   colNames = TRUE)
tst18 <- read.xlsx("20190715_XGB09_TEST_DS.xlsx",
                   sheet = 1,
                   startRow = 1,
                   colNames = TRUE)
tst19 <- read.xlsx("20190712_StackModel01_TEST_DS.xlsx",
                   sheet = 1,
                   startRow = 1,
                   colNames = TRUE)
tst20 <- read.xlsx("20190720_ENSEMBLE_11_DS.xlsx",
                   sheet = 1,
                   startRow = 1,
                   colNames = TRUE)
tst21 <- read.xlsx("20190717_XGB11_TEST_DS.xlsx",
                   sheet = 1,
                   startRow = 1,
                   colNames = TRUE)
tst22 <- read.xlsx("20190720_XGB14_TEST_DS.xlsx",
                   sheet = 1,
                   startRow = 1,
                   colNames = TRUE)
tst23 <- read.xlsx("20190721_LASSO01_TEST_DS.xlsx",
                   sheet = 1,
                   startRow = 1,
                   colNames = TRUE)
tst24 <- read.xlsx("20190721_RIDGE01_TEST_DS.xlsx",
                   sheet = 1,
                   startRow = 1,
                   colNames = TRUE)

RMSE_VECTOR <- c(0.0644,
                 0.0681,
                 0.0543,
                 0.0538,
                 0.0625,
                 0.0613,
                 0.0536,
                 0.0549,
                 0.0537,
                 0.0629,
                 0.0641,
                 0.0639,
                 0.0753,
                 0.0685,
                 0.0611,
                 0.0602,
                 0.0563,
                 0.0555,
                 0.0805,
                 0.0527,
                 0.0584,
                 0.0581,
                 0.0708,
                 0.0702)

tst_fin <- data.frame(id = seq(1,nrow(tst01),by=1),
                      Interc = rep(1,nrow(tst01)),
                      Pred01 = log10(tst01$Price+1),
                      Pred02 = log10(tst02$Price+1),
                      Pred03 = log10(tst03$Price+1),
                      Pred04 = log10(tst04$Price+1),
                      Pred05 = log10(tst05$Price+1),
                      Pred06 = log10(tst06$Price+1),
                      Pred07 = log10(tst07$Price+1),
                      Pred08 = log10(tst08$Price+1),
                      Pred09 = log10(tst09$Price+1),
                      Pred10 = log10(tst10$Price+1),
                      Pred11 = log10(tst11$Price+1),
                      Pred12 = log10(tst12$Price+1),
                      Pred13 = log10(tst13$Price+1),
                      Pred14 = log10(tst14$Price+1),
                      Pred15 = log10(tst15$Price+1),
                      Pred16 = log10(tst16$Price+1),
                      Pred17 = log10(tst17$Price+1),
                      Pred18 = log10(tst18$Price+1),
                      Pred19 = log10(tst19$Price+1),
                      Pred20 = log10(tst20$Price+1),
                      Pred21 = log10(tst21$Price+1),
                      Pred22 = log10(tst22$Price+1),
                      Pred23 = log10(tst23$Price+1),
                      Pred24 = log10(tst24$Price+1))

tst_fin2 <- tst_fin[,3:ncol(tst_fin)]

DropVars <- character()
DropVarsIndex <- numeric()

CorMatrix <- cor(tst_fin2)

CorrCutOff <- 0.9999999

NumberVars <- ncol(tst_fin2)

for(i in 1:(NumberVars-1)) {
  for(j in (i+1):NumberVars) {
    if(abs(CorMatrix[i,j]) >= CorrCutOff) {
      if(RMSE_VECTOR[i] < RMSE_VECTOR[j]) {
        if(!(names(tst_fin2)[j] %in% DropVars)) {
          DropVars <- c(DropVars,names(tst_fin2)[j])
          DropVarsIndex <- c(DropVarsIndex,j)
        }
      }
      else {
        if(!(names(tst_fin2)[i] %in% DropVars)) {
          DropVars <- c(DropVars,names(tst_fin2)[i])
          DropVarsIndex <- c(DropVarsIndex,i)
        }
      }
    }
  }
}

VarsShortlisted <- names(tst_fin2)
VarsIndex <- 1:NumberVars

VarsShortlisted <- VarsShortlisted[!(VarsShortlisted %in% DropVars)]
VarsIndex <- VarsIndex[!(VarsIndex %in% DropVarsIndex)]
VarsShortlisted
RMSE_VECTOR[VarsIndex]

tst_fin3 <- tst_fin[,c("id","Interc",VarsShortlisted)]
RMSE_VECTOR2 <- RMSE_VECTOR[VarsIndex]

cor(tst_fin3)

number <- length(RMSE_VECTOR2)
lambda <- 0

total_obs <- nrow(tst_fin)

SummationYiSquare <- total_obs * (0.9174**2)

SummationYi <- (SummationYiSquare + (total_obs*(0.30103**2)) - (total_obs*(0.6434**2))) / (2*0.30103)

x <- as.matrix(tst_fin3[,2:ncol(tst_fin3)])
x_prime_x <- t(x) %*% x
x_prime_x_inv <- solve(x_prime_x + lambda*diag(number+1))

x_prime_y <- SummationYi

for (i in 3:ncol(tst_fin3))
{
  temp_yihat_square <- sum(tst_fin3[,i] * tst_fin3[,i])
  temp_val <- (SummationYiSquare + temp_yihat_square - (total_obs*(RMSE_VECTOR2[i-2]**2))) / 2
  x_prime_y <- c(x_prime_y,temp_val)
}

Beta_Coeff <- x_prime_x_inv %*% x_prime_y
Beta_Coeff
Preds <-  (x %*% Beta_Coeff)
summary(Preds)
summary(tst_fin)
rmse(Preds,tst_fin$Pred01)
summary((10**Preds)-1)

Preds <- (10**Preds) - 1
summary(Preds)

tst_file <- data.frame(Price = Preds)
write_csv(tst_file, "20190721_ENSEMBLE_15_DS.csv")

#0.9493