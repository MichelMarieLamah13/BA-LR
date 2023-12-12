# ==============================================================================
#  Copyright (c) 2023. Imen Ben Amor
# ==============================================================================
import pdb
import pickle
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import GradientBoostingClassifier

from sklearn import tree
from sklearn.model_selection import GridSearchCV
import shap
import os

from attribute_explainer import prepare_data
from build_contributions import *
from plots import *
from test import *
import lime
from lime import lime_tabular
from sklearn.model_selection import train_test_split


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
            input_features = X.columns[:-1].to_list()
            X = X[input_features]
            print(f"END preparing data: {ba}")
            sys.stdout.flush()

            print(f"BEGIN training model: {ba}")
            sys.stdout.flush()
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=2, random_state=0)
            model = GradientBoostingClassifier(
                max_depth=2,
                random_state=0,
                n_estimators=10)
            model.fit(X=X_train, y=y_train)
            explainer = lime_tabular.LimeTabularExplainer(
                X_train.values,
                feature_names=input_features,
                class_names=[0, 1],
                verbose=True,
                mode='classification'
            )
            print(f"END training model: {ba}")
            sys.stdout.flush()

            print(f"BEGIN explaining model: {ba}")
            sys.stdout.flush()
            for idx, row in X_test.iterrows():
                print(f"{ba} - {idx}")
                sys.stdout.flush()
                explanation = explainer.explain_instance(row, model.predict_proba,
                                                         num_features=10)
                save_explanation(explanation, ba, idx)
            print(f"END explaining model: {ba}")
            sys.stdout.flush()


if __name__ == "__main__":
    lime_tabular_explainer()
