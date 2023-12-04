# ==============================================================================
#  Copyright (c) 2023. Imen Ben Amor
# ==============================================================================
import pdb
import pickle
import sys

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

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
from interpret import show
from interpret.data import Marginal
from interpret.glassbox import ExplainableBoostingClassifier, ClassificationTree, DecisionListClassifier
from interpret.perf import RegressionPerf, ROC

env.logging_config("logs/logFile_contribution_BA")


def save_data(data, path, filename):
    os.makedirs(path, exist_ok=True)
    with open(f'{path}/{filename}.pkl', 'wb') as file:
        pickle.dump(data, file)


def load_data(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
        return data


def use_interpret():
    meta_vox2 = pd.read_csv("data/vox2_meta.csv")
    floc_train = meta_vox2[meta_vox2["Set"] == "dev"]["Gender"].to_list().count("f")
    mloc_train = meta_vox2[meta_vox2["Set"] == "dev"]["Gender"].to_list().count("m")

    BA = [f"BA{i}" for i in range(256)]
    for ba in BA:
        if os.path.isfile(f"data/BA/{ba}_0.csv"):
            if os.path.isfile(f"data/BA/{ba}_0.csv"):
                logging.info(f"===================={ba}=========================")
                X, y, ba0, ba1 = prepare_data(ba, mloc_train, floc_train)
                input_features = X.columns[:-1].to_list()
                X = X[input_features]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=100, random_state=0)

                # Exploring Data
                print(f"BEGIN Explore Data: {ba}")
                sys.stdout.flush()

                marginal = Marginal().explain_data(X_train, y_train, name='Train Data')
                path = f'Step3/explainability_results/interpret/{ba}'
                save_data(marginal, path, 'marginal')

                print(f"END Explore Data: {ba}")
                sys.stdout.flush()

                print(f"BEGIN Training: {ba}")
                sys.stdout.flush()

                ct = ClassificationTree(random_state=0)
                ct.fit(X_train, y_train)

                dlc = DecisionListClassifier(random_state=0)
                dlc.fit(X_train, y_train)

                ebm = ExplainableBoostingClassifier(random_state=0)
                ebm.fit(X_train, y_train)

                print(f"END Training: {ba}")
                sys.stdout.flush()

                # Performance
                print(f"BEGIN Performance: {ba}")
                sys.stdout.flush()

                ct_perf = ROC(ct.predict_proba).explain_perf(X_test, y_test, name='Classification Tree')
                save_data(ct_perf, path, 'ct_perf')

                dlc_perf = ROC(dlc.predict_proba).explain_perf(X_test, y_test,
                                                               name='Decision List Classifier')
                save_data(dlc_perf, path, 'dlc_perf')

                ebm_perf = ROC(ebm.predict_proba).explain_perf(X_test, y_test, name='EBM')
                save_data(ebm_perf, path, 'ebm_perf')

                print(f"END Performance: {ba}")
                sys.stdout.flush()

                # Global Interpretability
                print(f"BEGIN Global interpretability: {ba}")
                sys.stdout.flush()

                ebm_global = ebm.explain_global(name='EBM')
                save_data(ebm_global, path, 'ebm_global')

                print(f"END Global interpretability: {ba}")
                sys.stdout.flush()

                # Local Interpretability
                print(f"BEGIN Local interpretability: {ba}")
                sys.stdout.flush()

                ebm_local = ebm.explain_local(X_test[:5], y_test[:5], name='EBM')
                save_data(ebm_local, path, 'ebm_local')

                print(f"BEGIN Local interpretability: {ba}")
                sys.stdout.flush()


if __name__ == "__main__":
    use_interpret()
