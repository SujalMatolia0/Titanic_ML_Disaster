import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV
import joblib
from PIL import Image


# 1. Machine Learning Pipeline Classes
class AgeImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        imputer = SimpleImputer(strategy="mean")
        X["Age"] = imputer.fit_transform(X[["Age"]])
        return X


class FeatureEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):

        self.embarked_encoder = OneHotEncoder(
            sparse_output=False, handle_unknown="ignore"
        )
        self.sex_encoder = OneHotEncoder(drop="if_binary", sparse_output=False)

        self.embarked_encoder.fit(X[["Embarked"]].fillna("S"))
        self.sex_encoder.fit(X[["Sex"]].fillna("male"))
        return self

    def transform(self, X):

        X = X.copy()
        X["Embarked"] = X["Embarked"].fillna("S")
        X["Sex"] = X["Sex"].fillna("male")

        embarked_matrix = self.embarked_encoder.transform(X[["Embarked"]])
        sex_matrix = self.sex_encoder.transform(X[["Sex"]])

        embarked_cols = [
            f"Embarked_{cat}" for cat in self.embarked_encoder.categories_[0]
        ]
        sex_col = "Female"

        for i, col in enumerate(embarked_cols):
            X[col] = embarked_matrix[:, i]

        X[sex_col] = sex_matrix[:, 0] if sex_matrix.shape[1] > 0 else 0

        if "PassengerId" not in X.columns:
            X["PassengerId"] = np.arange(len(X))

        return X


class FeatureDropper(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):

        self.feature_names = [
            col
            for col in X.columns
            if col not in ["Embarked", "Name", "Ticket", "Cabin", "Sex"]
        ]
        return self

    def transform(self, X):

        return X[[col for col in self.feature_names if col in X.columns]]


def train_model():

    df = pd.read_csv("train.csv")

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, test_idx in split.split(df, df[["Survived", "Pclass", "Sex"]]):
        strat_train_set = df.loc[train_idx]
        strat_test_set = df.loc[test_idx]

    pipeline = Pipeline(
        [
            ("ageimputer", AgeImputer()),
            ("featureencoder", FeatureEncoder()),
            ("featuredropper", FeatureDropper()),
        ]
    )

    strat_train_set = pipeline.fit_transform(strat_train_set)
    X_train = strat_train_set.drop(["Survived"], axis=1)
    y_train = strat_train_set["Survived"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10],
        "min_samples_split": [2, 4],
    }

    clf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(clf, param_grid, cv=3, scoring="accuracy", n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)

    joblib.dump(grid_search.best_estimator_, "titanic_model.joblib")
    joblib.dump(pipeline, "pipeline.joblib")
    joblib.dump(scaler, "scaler.joblib")
    joblib.dump(X_train.columns.tolist(), "feature_columns.joblib")

    return grid_search.best_estimator_, pipeline, scaler, X_train.columns


def run_app(model, pipeline, scaler, feature_columns):
    st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢")

    st.title("Titanic Survival Prediction")
    st.markdown("Predict whether a passenger would have survived the Titanic disaster")

    with st.form("passenger_details"):
        col1, col2 = st.columns(2)
        with col1:
            pclass = st.selectbox(
                "Passenger Class", [1, 2, 3], format_func=lambda x: f"Class {x}"
            )
        with col2:
            sex = st.radio("Gender", ["Female", "Male"], horizontal=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age", min_value=0, max_value=100, value=30)
        with col2:
            sibsp = st.number_input(
                "Siblings/Spouses", min_value=0, max_value=10, value=0
            )
        with col3:
            parch = st.number_input(
                "Parents/Children", min_value=0, max_value=10, value=0
            )

        fare = st.number_input("Ticket Fare (£)", min_value=0.0, value=32.2, step=1.0)
        embarked = st.selectbox(
            "Embarkation Port",
            ["S", "C", "Q"],
            format_func=lambda x: {
                "S": "Southampton",
                "C": "Cherbourg",
                "Q": "Queenstown",
            }[x],
        )

        submitted = st.form_submit_button("Predict Survival")

    if submitted:

        input_dict = {
            "PassengerId": [999],  # Dummy ID
            "Pclass": [pclass],
            "Name": ["Dummy"],
            "Sex": [sex.lower()],
            "Age": [age],
            "SibSp": [sibsp],
            "Parch": [parch],
            "Ticket": ["Dummy"],
            "Fare": [fare],
            "Cabin": [None],
            "Embarked": [embarked],
        }
        input_df = pd.DataFrame(input_dict)

        try:
            processed_data = pipeline.transform(input_df)

            processed_data = processed_data.reindex(
                columns=feature_columns, fill_value=0
            )

            scaled_data = scaler.transform(processed_data)

            prediction = model.predict(scaled_data)[0]
            proba = model.predict_proba(scaled_data)[0][1]

            st.subheader("Prediction Result")
            if prediction == 1:
                st.success(f"Survived ({proba:.0%} probability)")
            else:
                st.error(f"Did Not Survive ({(1-proba):.0%} probability)")

            st.progress(int(proba * 100))

        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train new model")
    args = parser.parse_args()

    if args.train:
        print("Training new model...")
        model, pipeline, scaler, feature_columns = train_model()
    else:
        try:
            print("Loading existing model...")
            model = joblib.load("titanic_model.joblib")
            pipeline = joblib.load("pipeline.joblib")
            scaler = joblib.load("scaler.joblib")
            feature_columns = joblib.load("feature_columns.joblib")
        except FileNotFoundError:
            st.warning("Model not found. Training new model...")
            model, pipeline, scaler, feature_columns = train_model()

    run_app(model, pipeline, scaler, feature_columns)
