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
from interpret.perf import RegressionPerf

env.logging_config("logs/logFile_contribution_BA")


def save_data(data, path):
    os.makedirs(path, exist_ok=True)
    with open(path, 'wb') as file:
        pickle.dump(data, file)


def load_data(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
        return data


class InterpretDataset(Dataset):
    def __init__(self):
        self.meta_vox2 = pd.read_csv("data/vox2_meta.csv")
        self.floc_train = self.meta_vox2[self.meta_vox2["Set"] == "dev"]["Gender"].to_list().count("f")
        self.mloc_train = self.meta_vox2[self.meta_vox2["Set"] == "dev"]["Gender"].to_list().count("m")

        self.BA = [f"BA{i}" for i in range(256)]

    def __len__(self):
        return len(self.BA)

    def __getitem__(self, idx):
        ba = self.BA[idx]
        if os.path.isfile(f"data/BA/{ba}_0.csv"):
            if os.path.isfile(f"data/BA/{ba}_0.csv"):
                logging.info(f"===================={ba}=========================")
                X, y, ba0, ba1 = prepare_data(ba, self.mloc_train, self.floc_train)
                input_features = X.columns[:-1].to_list()
                X = X[input_features]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

                # Exploring Data
                print("BEGIN Explore Data")
                sys.stdout.flush()

                marginal = Marginal().explain_data(X_train, y_train, name='Train Data')
                path = f'Step3/explainability_results/interpret/{ba}/marginal.pkl'
                save_data(marginal, path)

                print("END Explore Data")
                sys.stdout.flush()

                print("BEGIN Training")
                sys.stdout.flush()

                ct = ClassificationTree(random_state=0)
                ct.fit(X_train, y_train)

                dlc = DecisionListClassifier(random_state=0)
                dlc.fit(X_train, y_train)

                ebm = ExplainableBoostingClassifier(random_state=0)
                ebm.fit(X_train, y_train)

                print("END Training")
                sys.stdout.flush()

                # Performance
                print("BEGIN Performance")
                sys.stdout.flush()

                ct_perf = RegressionPerf(ct.predict).explain_perf(X_test, y_test, name='Classification Tree')
                path = f'Step3/explainability_results/interpret/{ba}/ct_perf.pkl'
                save_data(ct_perf, path)

                dlc_perf = RegressionPerf(dlc.predict).explain_perf(X_test, y_test, name='Decision List Classifier')
                path = f'Step3/explainability_results/interpret/{ba}/dlc_perf.pkl'
                save_data(dlc_perf, path)

                ebm_perf = RegressionPerf(ebm.predict).explain_perf(X_test, y_test, name='EBM')
                path = f'Step3/explainability_results/interpret/{ba}/ebm_perf.pkl'
                save_data(ebm_perf, path)

                print("END Performance")
                sys.stdout.flush()

                # Global Interpretability
                print("BEGIN Global interpretability")
                sys.stdout.flush()

                ebm_global = ebm.explain_global(name='EBM')
                path = f'Step3/explainability_results/interpret/{ba}/ebm_global.pkl'
                save_data(ebm_global, path)

                print("END Global interpretability")
                sys.stdout.flush()

                # Local Interpretability
                print("BEGIN Local interpretability")
                sys.stdout.flush()

                ebm_local = ebm.explain_local(X_test[:5], y_test[:5], name='EBM')
                path = f'Step3/explainability_results/interpret/{ba}/ebm_local.pkl'
                save_data(ebm_local, path)

                print("BEGIN Local interpretability")
                sys.stdout.flush()
        return ba


def use_interpret():
    dataset = InterpretDataset()
    loader = DataLoader(dataset, num_workers=4, batch_size=10)
    for idx, data in enumerate(loader):
        print(f"Batch [{idx:2d}/{len(loader)}]")
        sys.stdout.flush()


if __name__ == "__main__":
    use_interpret()
