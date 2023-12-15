# ==============================================================================
#  Copyright (c) 2023. Imen Ben Amor
# ==============================================================================
import os
import pickle
import sys

import pandas as pd
from lime import lime_tabular
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from test import *


def save_explanation(explanation, ba, idx):
    path = f'Step3/explainability_results/lime/{ba}/{idx}'
    os.makedirs(path, exist_ok=True)
    with open(f'{path}/explanation.pkl', 'wb') as file:
        pickle.dump(explanation, file)


def load_explanation(ba, idx):
    path = f'Step3/explainability_results/lime/{ba}/{idx}'
    with open(f'{path}/explanation.pkl', 'rb') as file:
        explanation = pickle.load(file)
        return explanation


def prepare_data(ba, mloc_train, floc_train):
    logging.info(f'Number of men in Train={mloc_train}')
    logging.info(f'Number of female in Train={floc_train}')
    ba0 = pd.read_csv(f"data/BA/{ba}_0.csv")
    ba1 = pd.read_csv(f"data/BA/{ba}_1.csv")
    X = pd.concat([ba0, ba1], ignore_index=True)
    S = StandardScaler()
    X_scaled = pd.DataFrame(S.fit_transform(X.iloc[:, 1:]), columns=X.iloc[:, 1:].columns)
    X_scaled.insert(0, 'name', X["name"])
    y = [0] * len(ba0) + [1] * len(ba1)
    return X_scaled, y, ba0, ba1


def lime_tabular_explainer():
    meta_vox2 = pd.read_csv("data/vox2_meta.csv")
    floc_train = meta_vox2[meta_vox2["Set"] == "dev"]["Gender"].to_list().count("f")
    mloc_train = meta_vox2[meta_vox2["Set"] == "dev"]["Gender"].to_list().count("m")

    # BA = [f"BA{i}" for i in range(256)]
    BA = ['BA2', 'BA3', 'BA4', 'BA5', 'BA8', 'BA9', 'BA10']
    for ba in tqdm(BA):
        if os.path.isfile(f"data/BA/{ba}_0.csv"):
            print(f"BEGIN preparing data: {ba}")
            sys.stdout.flush()
            X, y, ba0, ba1 = prepare_data(ba, mloc_train, floc_train)
            input_features = X.columns[1:].to_list()
            X_ = X[input_features]
            print(f"END preparing data: {ba}")
            sys.stdout.flush()

            print(f"BEGIN training model: {ba}")
            sys.stdout.flush()
            X_train, X_test, y_train, y_test = train_test_split(X_, y, test_size=10, random_state=0)
            model = GradientBoostingClassifier()
            model.fit(X=X_train, y=y_train)
            explainer = lime_tabular.LimeTabularExplainer(
                X_train.values,
                feature_names=input_features,
                class_names=[0, 1],
                verbose=True,
                mode='classification'
            )
            y_predict = model.predict(X_test)
            save_explained_data(ba, X, y, y_predict)
            print(f"END training model: {ba}")
            sys.stdout.flush()

            print(f"BEGIN explaining model: {ba}")
            sys.stdout.flush()

            for idx, row in X_test.iterrows():
                print(f"{ba} - {idx}")
                sys.stdout.flush()
                explanation = explainer.explain_instance(row, model.predict_proba, num_features=10)
                save_explanation(explanation, ba, idx)
            print(f"END explaining model: {ba}")
            sys.stdout.flush()


def save_explained_data(ba, X, y, y_pred):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10, random_state=0)
    X_test['y_true'] = y_test
    X_test['y_pred'] = y_pred
    X_test.to_csv(f'Step3/explainability_results/lime/{ba}/explain_data.csv')


if __name__ == "__main__":
    lime_tabular_explainer()
