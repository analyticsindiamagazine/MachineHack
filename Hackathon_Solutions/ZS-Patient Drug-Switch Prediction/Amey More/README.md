Please make sure that all the requirement(libraries) in the requirements.txt are installed.
Python 3.6 was used for develpment.

First for Feature Creation run:
1. Run recency_feature_creator.py
2. Run frequency_feature_creator.py

Post this run the data_preprocessing_script.py which outputs final_df.csv which will be used for training

Pass in the directory where the final_df.csv is stored. This can be set in line 15 of Model.py

Model.py is the python script that will generate the final solution. Output of this notebook is
1) final_sub.csv (predictions for the test set)
2) feature_importance.csv (feature importance list)
3)hard_classes.csv (this is the hard classes file)
