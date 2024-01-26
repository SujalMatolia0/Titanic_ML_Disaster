import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns


class AgeImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y =None):
        return self
    def transform(self, X):
        imputer = SimpleImputer(strategy="mean")
        X["Age"] = imputer.fit_transform(X[["Age"]])
        return X
    
class FeatureEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y =None):
        return self
    def transform(self, X):
        encoder = OneHotEncoder()
        matrix  = encoder.fit_transform(X[["Embarked"]]).toarray()
        column_names =  ["C", "S", "Q", "N"]
        
        for i in range(len(matrix.T)):
            X[column_names[i]] = matrix.T[i]
        matrix = encoder.fit_transform(X[["Sex"]]).toarray()    
        
        column_names =  ["Female", "Male"]
        
        for i in range(len(matrix.T)):
            X[column_names[i]] = matrix.T[i]
        
        return X
    
class FeatureDropper(BaseEstimator, TransformerMixin):
    def fit(self, X, y =None):
        return self
    def transform(self, X):
        return X.drop(["Embarked", "Name", "Ticket", "Cabin", "Sex", "N" ], axis = 1, errors = "ignore")

split = StratifiedShuffleSplit(n_splits= 1, test_size= 0.2)
for train_indice, test_indice in split.split(df, df[["Survived", "Pclass", "Sex"]]):
    strat_train_set = df.loc[train_indice]
    strat_test_set = df.loc[test_indice]
    
pipeline = Pipeline([("ageimputer", AgeImputer()),
                     ("featureencoder", FeatureEncoder()),
                     ("featuredropper", FeatureDropper())])

strat_train_set = pipeline.fit_transform(strat_train_set)

X = strat_train_set.drop(['Survived'], axis = 1)
Y = strat_train_set['Survived']

scaler = StandardScaler()
X_data = scaler.fit_transform(X)
y_data = Y.to_numpy()

clf = RandomForestClassifier()
param_grid = [
    {"n_estimators" : [10, 100, 200, 500], "max_depth": [None, 5, 10], "min_samples_split": [2, 3, 4]}
]

grid_search = GridSearchCV(clf, param_grid, cv = 3, scoring=  "accuracy", return_train_score= True)
grid_search.fit(X_data, y_data)

final_clf = grid_search.best_estimator_

strat_test_set = pipeline.fit_transform(strat_test_set)

X_test = strat_test_set.drop(["Survived"], axis = 1)
Y_test = strat_test_set["Survived"]

scaler = StandardScaler()
X_data_test = scaler.fit_transform(X_test)
y_data_test = Y_test.to_numpy()
final_clf.score(X_data_test, y_data_test)
print("The accuracy is : ", final_clf.score(X_data_test, y_data_test))
final_data = pipeline.fit_transform(df)

X_final = final_data.drop(["Survived"], axis  = 1)
Y_final = final_data["Survived"]

scaler = StandardScaler()
X_data_final = scaler.fit_transform(X_final)
y_data_final = Y_final.to_numpy()

prod_clf = RandomForestClassifier()
param_grid = [
    {"n_estimators" : [10, 100, 200, 500], "max_depth": [None, 5, 10], "min_samples_split": [2, 3, 4]}
]

grid_search = GridSearchCV(prod_clf, param_grid, cv = 3, scoring=  "accuracy", return_train_score= True)
grid_search.fit(X_data_final, y_data_final)
prod_final_clf = grid_search.best_estimator_

test_df = pd.read_csv("test.csv")
final_test_data = pipeline.fit_transform(test_df)
final_test_data.info()

X_final_test = final_test_data
X_final_test = X_final_test.ffill()

scaler = StandardScaler()
X_data_final_test = scaler.fit_transform(X_final_test)

predictions = prod_final_clf.predict(X_data_final_test)

final_df = pd.DataFrame(test_df["PassengerId"])
final_df["Survived"] = predictions
final_df.to_csv("gender_submission.csv", index = False)

final_df
##Time to dump it##
import joblib
joblib.dump(prod_final_clf, "predicting_model.joblib")