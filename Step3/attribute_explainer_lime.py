# ==============================================================================
#  Copyright (c) 2023. Imen Ben Amor
# ==============================================================================
import pdb
import pickle

import numpy as np
import pandas as pd

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


def save_explanation(explanation, ba):
    path = f'Step3/explainability_results/lime/{ba}'
    with open(f'{path}/explanation.pkl', 'wb') as file:
        pickle.dump(explanation, file)


def load_explanation(ba):
    path = f'Step3/explainability_results/lime/{ba}'
    with open(f'{path}/explanation.pkl', 'rb') as file:
        explanation = pickle.load(file)
        return explanation


def lime_tabular_explainer():
    BA = [f"BA{i}" for i in range(256)]
    features_vox1 = pd.read_csv("data/vox1_opensmile.csv.new")
    df_binary = pd.read_csv("data/vec_vox1.txt.new")  # df_binary.csv
    for ba in BA:
        if os.path.isfile(f"data/BA/{ba}_0.csv"):
            logging.info(f"===================={ba}=========================")
            X, y, ba0, ba1 = prepare_data(ba)
            input_features = X.columns[:-1]
            target_feature = ['ba']
            X = X[input_features]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10)
            parameters = {'max_depth': range(3, 15)}
            model = GridSearchCV(tree.DecisionTreeClassifier(), parameters, n_jobs=4)
            model.fit(X=X_train, y=y_train)
            tree_model = model.best_estimator_
            logging.info(model.best_score_, model.best_params_)
            logging.info("=======Test ba model on voxceleb1======")
            test_acc = test_vox1(ba, tree_model, features_vox1, df_binary, meta_vox1, mloc_train, floc_train)
            logging.info("=======Building explainer=======")
            explainer = lime_tabular.LimeTabularExplainer(
                np.array(X_train),
                feature_names=input_features,
                class_names=target_feature,
                verbose=True,
                mode='regression'
            )
            explanation = explainer.explain_instance(X.iloc[0], model.predict_proba,
                                                     num_features=len(X.iloc[:, :-1].columns))


if __name__ == "__main__":
    meta_vox2 = pd.read_csv("data/vox2_meta.csv")
    meta_vox1 = pd.read_csv("data/voxceleb1.csv", sep='\t')
    floc_train = meta_vox2[meta_vox2["Set"] == "dev"]["Gender"].to_list().count("f")
    mloc_train = meta_vox2[meta_vox2["Set"] == "dev"]["Gender"].to_list().count("m")
    env.logging_config("logs/logFile_contribution_BA")

    lime_tabular_explainer()