# ==============================================================================
#  Copyright (c) 2023. Imen Ben Amor
# ==============================================================================
import pdb
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

import var_env as env
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

env.logging_config("logs/logFile_contribution_BA")


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

    BA = [f"BA{i}" for i in range(256)]
    for ba in BA:
        if os.path.isfile(f"data/BA/{ba}_0.csv"):
            logging.info(f"===================={ba}=========================")
            X, y, ba0, ba1 = prepare_data(ba, mloc_train, floc_train)
            input_features = X.columns[:-1].to_list()
            target_feature = ['ba']
            X = X[input_features]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=2, random_state=0)
            model = GradientBoostingClassifier(
                n_estimators=500,
                validation_fraction=0.2,
                n_iter_no_change=5,
                tol=0.01,
                random_state=0
            )
            model.fit(X=X_train.values, y=y_train)
            logging.info("=======Building explainer=======")
            explainer = lime_tabular.LimeTabularExplainer(
                X_train.values,
                feature_names=input_features,
                class_names=target_feature,
                verbose=True,
                mode='classification'
            )
            indexes = X_test.index.to_list()
            for i in range(len(indexes)):
                idx = indexes[i]
                row = X_test.iloc[i]
                print(f"{ba} - {idx}")
                sys.stdout.flush()
                explanation = explainer.explain_instance(row, model.predict_proba,
                                                         num_features=10)
                save_explanation(explanation, ba, idx)


if __name__ == "__main__":
    lime_tabular_explainer()
