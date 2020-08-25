Steps to get the final Solution
-------------------------------

A. Put all the competition data csv files in the same folder as the notebooks.

B. Run the notebooks in the following order:

1. flower_type_pred_v6.ipynb
2. flower_type_pred_v7.ipynb
3. flower_type_pred_v10.ipynb
4. flower_type_blend.ipynb

Final Output Name:  "blend_v6_v7_v10_33_26_41.csv"


Approach:
---------

1.Focus on frequency features, aggregation features and number of unique values of one features, for all values of some other feature, to deal with high dimensionality.

2. Three 15 fold stratified models of LightGBM, XGBoost and Catboost

3. Final model, weighted blend of the three models based on LB and CV scores.