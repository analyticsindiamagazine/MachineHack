##### ReadMe for reproducing Neptune MH18 submissions
Run the steps in the same order as described in this file

## Set up
Submission zip file contains the following folders:
a. Data - this contains raw data (Train.csv, Test.csv) and prepared data (MH18_Train2.csv, MH18_Test2.csv)
b. Submissions - this contains 4 files
   submission_01.csv -- corresponds to Model 1
   submission_04.csv -- corresponds to Model 2
   MH18 Output03.csv -- corresponds to Model 3
   submission_05.csv -- average of submissions from Models 1, 2 & 3

##### Model 1: AutoGluon with autostacking
1. Use Google Colab to run notebook 'AutoGluon MH18-Notebook-1'
2. Upload the raw data files to Google Drive in order to access the files from notebook
3. Run the first section to install autogluon -- restart Colab run-time after installation
4. Run the training & scoring cells
   4.1 Train AutoGluon with auto_stack, 5 folds and 5 bagging folds

Output: submission_01.csv -- this model has a private score of 0.931

##### Model 2: Overriding Model 1 outputs with overlapping training data points
1. Test data has some data points that are coming from training data
2. Open 'MH18-Notebook-1' and go to Section 3
3. Identify the rows in test data that are already present in training data (data frame c)
4. In submission_01, look for rows that are in c and if the predicted probability is less than 0.1, force it to zero
   4.1 -- replacing rest of the training values actually made the public score worse, so restrict it to this
   
Output: submission_04.csv -- this model has a private score of 0.932

##### Model 3: AutoGluon with hyperparmaeter tuning + outlier detection outputs as features
1. Open 'MH18-Notebook-1' and go to Section 4 -- here we add some outlier detection methods to add them as features
   1.1 Run LOF, Isolation Forest, kNN and One-Class SVM; Save updated training data as 'MH18_Train2.csv'
   1.2 Replicate these for test data as well; Save updated test data as 'MH18_Test2.csv'
2. Open 'AutoGluon MH18-Notebook-2' in Google Colab
3. Run the first section to install autogluon
4. Run the training & scoring cells
   4.1 Train AutoGluon with auto_stack=False, hyperparameter_tune=True, stopping metric set to 'roc_auc', 5 folds and 5 bagging_folds
   
Output: MH18 Output03.csv -- this model has a private score of 0.918

##### Model 4: Final model
Average the outputs from models 1, 2 & 3

Output: submission_05.csv -- this model has a private score of 0.931