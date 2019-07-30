library(openxlsx)

df <- read.xlsx("C:\\Kaggle\\Cars\\Data\\Data_Train.xlsx",
                sheet = 1,
                startRow = 1,
                colNames = TRUE)

df_names <- names(df)
df$id <- seq.int(nrow(df))
df <- df[,c("id",df_names)]
head(df)

write.csv(df,"C:\\Kaggle\\Cars\\Data\\Data_Train.csv",row.names = FALSE)
remove(df, df_names)

df <- read.xlsx("C:\\Kaggle\\Cars\\Data\\Data_Test.xlsx",
                sheet = 1,
                startRow = 1,
                colNames = TRUE)

df_names <- names(df)
df$id <- seq.int(nrow(df))
df <- df[,c("id",df_names)]
head(df)

write.csv(df,"C:\\Kaggle\\Cars\\Data\\Data_Test.csv",row.names = FALSE)