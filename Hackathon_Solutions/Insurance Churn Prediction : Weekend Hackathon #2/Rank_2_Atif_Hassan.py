import csv
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import f1_score
from lightgbm import LGBMClassifier, plot_importance
from xgboost import XGBClassifier, XGBRFClassifier
from vecstack import StackingTransformer
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler
import warnings


#---------------------------------------------------------------------- DEFINE ALL GLOBAL VARIABLES ----------------------------------------------------------------------------------

warnings.filterwarnings("ignore")

#-------------------------------------------------------------------- ALL GLOBAL VARIABLE DEFINITIONS END -----------------------------------------------------------------------------






#--------------------------------------------------------------------------- DEFINE ALL FUNCTIONS -------------------------------------------------------------------------------------

# Function to perform one-hot encoding of a single feature
def ohe(X):
    unique_vals = list(set(X))
    unique_vals.sort()
    X_new = np.zeros((len(X), len(unique_vals)), dtype=np.int8)
    for i in range(len(X)):
        X_new[i,int(X[i])] = 1
    return X_new



# Function to perform k-fold cross-validation
def cv(num_splits, X_train, Y):
    # Define the type of cross-validation
    kf, scores = StratifiedKFold(n_splits=num_splits, random_state=0), list()

    # Perform CV
    for train_index, test_index in kf.split(X_train, Y):
        # Splitting into train and test
        x_train, y_train, x_train1 = X_train[train_index], Y[train_index], X_train1[train_index]
        x_test, y_test, x_test1 = X_train[test_index], Y[test_index], X_train1[test_index]

        # Define base estimators for stacking
        estimators = [('lgbm', LGBMClassifier(random_state=0, n_estimators=520, learning_rate=0.1, num_leaves=31, is_unbalance=True)),
                  ('rf', RandomForestClassifier(random_state=0, max_depth=10, class_weight={0:0.2, 1:0.8}, n_estimators=500, max_features=None, n_jobs=4))]
        # Perform stacking
        stack = StackingTransformer(estimators, regression=False, verbose=2, needs_proba=True, stratified=True, shuffle=True)
        stack = stack.fit(x_train, y_train)

        # Get the stacked features
        S_train = stack.transform(x_train)
        S_test = stack.transform(x_test)
        # Also take the weighted average of the stacked features as another feature
        S_train_av, S_test_av = np.zeros((len(S_train), 2), dtype=np.float32), np.zeros((len(S_test), 2), dtype=np.float32)
        for index, vals in enumerate(S_train):
            S_train_av[index, 0] = (vals[0]*0.7) + (vals[2]*0.3)
            S_train_av[index, 1] = (vals[1]*0.7) + (vals[3]*0.3)
        for index, vals in enumerate(S_test):
            S_test_av[index, 0] = (vals[0]*0.7) + (vals[2]*0.3)
            S_test_av[index, 1] = (vals[1]*0.7) + (vals[3]*0.3)

        # Define the final estimator
        model = XGBClassifier(random_state=0, n_jobs=4, max_depth=4, scale_pos_weight=2.5, n_estimators=200, learning_rate=0.1, gamma=1)
        model.fit(np.concatenate((S_train, S_train_av, x_train1), axis=1), y_train)
        preds4 =  model.predict_proba(np.concatenate((S_test, S_test_av, x_test1), axis=1))

        # Now perform random under-sampling on the data
        rus = RandomUnderSampler(random_state=0, sampling_strategy=0.3)
        x_train, y_train_ = rus.fit_resample(x_train, y_train)

        # Get predictions from models on this majority class under-sampled dataset
        model1 = LGBMClassifier(random_state=0, n_estimators=100, learning_rate=0.1, num_leaves=31, categorical_feature=[8, 9, 10, 11, 12, 13, 14])
        model2 = RandomForestClassifier(random_state=0, max_depth=13, n_estimators=100, max_features=None, n_jobs=4, class_weight={0:0.4, 1:0.6})
        model1.fit(x_train, y_train_), model2.fit(x_train, y_train_)
        preds1, preds2 = model1.predict_proba(x_test), model2.predict_proba(x_test)

        # Get weighted average predictions
        preds3 = list()
        for a, b in zip(preds1, preds2):
            preds3.append([(0.7*a[0]) + (0.3*b[0]), (0.7*a[1]) + (0.3*b[1])])

        # Finally, perform weighted average prediction of stacked ensemble and weighted average ensemble
        preds = list()
        for a, b in zip(preds3, preds4):
            preds.append([(0.5*a[0]) + (0.5*b[0]), (0.5*a[1]) + (0.5*b[1])])
        preds = np.array(preds)
        preds = np.argmax(preds, axis=1)

        # Check out the score
        scores.append(f1_score(y_test, preds))
        print("Score: ", scores[-1])
    print("Average Score: ", sum(scores)/len(scores))



def final_submission(X_train, Y, X_test):
    # Define base estimators for stacking
    estimators = [('lgbm', LGBMClassifier(random_state=0, n_estimators=520, learning_rate=0.1, num_leaves=31, is_unbalance=True)),
              ('rf', RandomForestClassifier(random_state=0, max_depth=10, class_weight={0:0.2, 1:0.8}, n_estimators=500, max_features=None, n_jobs=4))]
    # Perform stacking
    stack = StackingTransformer(estimators, regression=False, verbose=2, needs_proba=True, stratified=True, shuffle=True)
    stack = stack.fit(X_train, Y)

    # Get the stacked features
    S_train = stack.transform(X_train)
    S_test = stack.transform(X_test)
    # Also take the weighted average of the stacked features as another feature
    S_train_av, S_test_av = np.zeros((len(S_train), 2), dtype=np.float32), np.zeros((len(S_test), 2), dtype=np.float32)
    for index, vals in enumerate(S_train):
        S_train_av[index, 0] = (vals[0]*0.7) + (vals[2]*0.3)
        S_train_av[index, 1] = (vals[1]*0.7) + (vals[3]*0.3)
    for index, vals in enumerate(S_test):
        S_test_av[index, 0] = (vals[0]*0.7) + (vals[2]*0.3)
        S_test_av[index, 1] = (vals[1]*0.7) + (vals[3]*0.3)

    # Define the final estimator
    model = XGBClassifier(random_state=0, n_jobs=4, max_depth=4, scale_pos_weight=2.5, n_estimators=200, learning_rate=0.1, gamma=1)
    model.fit(np.concatenate((S_train, S_train_av, X_train1), axis=1), Y)
    preds4 =  model.predict_proba(np.concatenate((S_test, S_test_av, X_test1), axis=1))

    # Now perform random under-sampling on the data
    rus = RandomUnderSampler(random_state=0, sampling_strategy=0.3)
    X_train, Y_ = rus.fit_resample(X_train, Y)

    # Get predictions from models on this majority class under-sampled dataset
    model1 = LGBMClassifier(random_state=0, n_estimators=100, learning_rate=0.1, num_leaves=31, categorical_feature=[8, 9, 10, 11, 12, 13, 14])
    model2 = RandomForestClassifier(random_state=0, max_depth=13, n_estimators=100, max_features=None, n_jobs=4, class_weight={0:0.4, 1:0.6})
    model1.fit(X_train, Y_), model2.fit(X_train, Y_)
    preds1, preds2 = model1.predict_proba(X_test), model2.predict_proba(X_test)

    # Get weighted average predictions
    preds3 = list()
    for a, b in zip(preds1, preds2):
        preds3.append([(0.7*a[0]) + (0.3*b[0]), (0.7*a[1]) + (0.3*b[1])])

    # Finally, perform weighted average prediction of stacked ensemble and weighted average ensemble
    preds = list()
    for a, b in zip(preds3, preds4):
        preds.append([(0.5*a[0]) + (0.5*b[0]), (0.5*a[1]) + (0.5*b[1])])
    preds = np.array(preds)
    preds = np.argmax(preds, axis=1)

    # Make the submission!
    fp = open("submit.csv", "w")
    fp.write("labels\n")
    for pred in preds:
        fp.write(str(pred)+"\n")
    fp.close()

#---------------------------------------------------------------------- ALL FUNCTION DEFINITIONS END ------------------------------------------------------------------------------------






#-------------------------------------------------------------------------------- MAIN CODE ---------------------------------------------------------------------------------------------

# Load the train data
fp = open("Train.csv")
csvreader = csv.reader(fp)
header = next(csvreader)
X_train, Y = list(), list()
for row in tqdm(csvreader):
    X_train.append([float(i) for i in row[:-1]])
    Y.append(int(row[-1]))
X_train, Y = np.array(X_train), np.array(Y)

# Load the test data
fp = open("Test.csv")
csvreader = csv.reader(fp)
header = next(csvreader)
X_test = list()
for row in tqdm(csvreader):
    X_test.append([float(i) for i in row])
X_test = np.array(X_test)

print("Majority Samples: ", len(np.where(Y==0)[0]), "\nMinority Samples: ", len(np.where(Y==1)[0]), "\nRatio of minority samples: ", round((len(np.where(Y==1)[0])/len(Y)) * 100, 2), "(%)")
print("\nTrain data shape: ", X_train.shape, "\nTest Data shape: ", X_test.shape)

# XGBoost performs well with one-hot-encoding while lightgbm can handle categorical data internally
# RandomForest worked better without any one-hot-encoding
X_train1 = np.concatenate((X_train[:,:8], ohe(X_train[:,8]), X_train[:,9:]), axis=1)
X_test1 = np.concatenate((X_test[:,:8], ohe(X_test[:,8]), X_test[:,9:]), axis=1)
print("\nFinal train data shape: ", X_train1.shape, "\nFinal test Data shape: ", X_test1.shape)

# Perform 10-fold CV to get hyper-parameters
#cv(10, X_train, Y)
# Final code
final_submission(X_train, Y, X_test)
