import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import *
from sklearn.metrics import *
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.svm import *
from sklearn.ensemble import *
from sklearn.neighbors import *
from sklearn.naive_bayes import *

class EnsembleModels():
    def __init__(self, X_data, y_data, test_size=0.2, random_state=3107):
        indices = np.random.permutation(X_data.shape[0])
        self.data_train, self.data_test, self.labels_train, self.labels_test = train_test_split(
            X_data,
            y_data,
            test_size=test_size,
            random_state=random_state,
            stratify=y_data)
        self.results = {}

    def run_models(self):
        print('############### Random Forest ##############')
        rf = RandomForestClassifier(n_estimators=100,
                                    criterion='entropy',
                                    random_state=3107)
        rf.fit(self.data_train, self.labels_train)
        y_pred = rf.predict(self.data_test)

        print('Accuracy :', accuracy_score(self.labels_test, y_pred))

        self.results['rf'] = {
            'name': 'random forest',
            'model': rf,
            'accuracy': accuracy_score(self.labels_test, y_pred)
        }
        print(classification_report(self.labels_test, y_pred))

        print('############### Logistic Regression ##############')
        logreg = LogisticRegression()

        logreg.fit(self.data_train, self.labels_train)
        y_pred = logreg.predict(self.data_test)

        print('Accuracy :', accuracy_score(self.labels_test, y_pred))

        self.results['lr'] = {
            'name': 'Logistic Regression',
            'model': logreg,
            'accuracy': accuracy_score(self.labels_test, y_pred)
        }
        print(classification_report(self.labels_test, y_pred))

        print('############### svc ##############')
        svc = SVC(probability=True)
        svc.fit(self.data_train, self.labels_train)
        y_pred = svc.predict(self.data_test)

        print('Accuracy :', accuracy_score(self.labels_test, y_pred))

        self.results['svc'] = {
            'name': 'svc',
            'model': svc,
            'accuracy': accuracy_score(self.labels_test, y_pred)
        }
        print(classification_report(self.labels_test, y_pred))
        
        print('############### Extra Trees Classifier ##############')
        extra_tree = ExtraTreesClassifier(n_estimators=1000)
        extra_tree.fit(self.data_train, self.labels_train)
        y_pred = extra_tree.predict(self.data_test)

        print('Accuracy :', accuracy_score(self.labels_test, y_pred))

        self.results['extra_tree'] = {
            'name': 'extra tree',
            'model': extra_tree,
            'accuracy': accuracy_score(self.labels_test, y_pred)
        }
        print(classification_report(self.labels_test, y_pred))
        
        return self.results
        