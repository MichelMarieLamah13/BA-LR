# ==============================================================================
#  Copyright (c) 2023. Imen Ben Amor
# ==============================================================================
import pdb
import pickle
import sys

import numpy as np
import pandas as pd
from h2o import H2OFrame
from h2o.estimators.random_forest import H2ORandomForestEstimator
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from torch.utils.data import Dataset, DataLoader

from sklearn import tree
from sklearn.model_selection import GridSearchCV
import shap
import os

from tqdm import tqdm

from attribute_explainer import prepare_data
from build_contributions import *
from plots import *
from test import *
import lime
from lime import lime_tabular
from sklearn.model_selection import train_test_split
from interpret import show
from interpret.data import Marginal
from interpret.glassbox import ExplainableBoostingClassifier, ClassificationTree, DecisionListClassifier
from interpret.perf import RegressionPerf


def save_data(data, path, filename):
    os.makedirs(path, exist_ok=True)
    with open(f'{path}/{filename}.pkl', 'wb') as file:
        pickle.dump(data, file)


def load_data(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
        return data


def process(X_train, y_train, X_test, ba, model, predict_fn, folder):
    print(f"BEGIN {folder}: {ba}")
    sys.stdout.flush()

    model.fit(X_train, y_train)
    path = f'Step3/explainability_results/shap/{ba}/{folder}'
    save_data(model, path, 'model')

    explainer = shap.KernelExplainer(predict_fn, X_test)
    save_data(explainer, path, 'explainer')

    rf_shap_values = explainer.shap_values(X_test)
    save_data(rf_shap_values, path, 'shap_values')

    print(f"END {folder}: {ba}")
    sys.stdout.flush()


def use_shap():
    meta_vox2 = pd.read_csv("data/vox2_meta.csv")
    floc_train = meta_vox2[meta_vox2["Set"] == "dev"]["Gender"].to_list().count("f")
    mloc_train = meta_vox2[meta_vox2["Set"] == "dev"]["Gender"].to_list().count("m")

    # BA = [f"BA{i}" for i in range(256)]
    BA = ['BA2', 'BA3', 'BA4', 'BA5', 'BA8', 'BA9', 'BA10']
    for ba in tqdm(BA):
        if os.path.isfile(f"data/BA/{ba}_0.csv"):
            if os.path.isfile(f"data/BA/{ba}_0.csv"):
                X, y, ba0, ba1 = prepare_data(ba, mloc_train, floc_train)
                input_features = X.columns[:-1].to_list()
                X = X[input_features]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=100, random_state=0)

                # RandomForestClassifier
                model = RandomForestClassifier(
                    max_depth=2,
                    random_state=0,
                    n_jobs=4,
                    n_estimators=10
                )
                process(X_train, y_train, X_test, ba, model, model.predict, 'random_forest')

                # GBM
                model = GradientBoostingClassifier(n_estimators=10, random_state=0)
                process(X_train, y_train, X_test, ba, model, model.predict, 'gradient_boosting')


if __name__ == "__main__":
    use_shap()
