# ==============================================================================
#  Copyright (c) 2023. Imen Ben Amor
# ==============================================================================
import pdb

import var_env as env
from sklearn import tree
from sklearn.model_selection import GridSearchCV
import shap
import os
from build_contributions import *
from plots import *
from test import *


def prepare_data(ba, mloc_train, floc_train):
    logging.info(f'Number of men in Train={mloc_train}')
    logging.info(f'Number of female in Train={floc_train}')
    ba0 = pd.read_csv(f"data/BA/{ba}_0.csv")
    ba1 = pd.read_csv(f"data/BA/{ba}_1.csv")
    X = pd.concat([ba0, ba1], ignore_index=True)
    S = StandardScaler()
    X_scaled = pd.DataFrame(S.fit_transform(X.iloc[:, 1:]), columns=X.iloc[:, 1:].columns)
    X_scaled["name"] = X["name"]
    y = [0] * len(ba0) + [1] * len(ba1)
    return X_scaled, y, ba0, ba1


# =================================================
def shap_tree_explainer():
    meta_vox2 = pd.read_csv("data/vox2_meta.csv")
    meta_vox1 = pd.read_csv("data/voxceleb1.csv", sep='\t')
    floc_train = meta_vox2[meta_vox2["Set"] == "dev"]["Gender"].to_list().count("f")
    mloc_train = meta_vox2[meta_vox2["Set"] == "dev"]["Gender"].to_list().count("m")

    BA = [f"BA{i}" for i in range(256)]
    features_vox1 = pd.read_csv("data/vox1_opensmile.csv.new")
    df_binary = pd.read_csv("data/vec_vox1.txt.new")  # df_binary.csv
    for ba in BA:
        if os.path.isfile(f"data/BA/{ba}_0.csv"):
            logging.info(f"===================={ba}=========================")
            X, y, ba0, ba1 = prepare_data(ba, mloc_train, floc_train)
            parameters = {'max_depth': range(3, 15)}
            model = GridSearchCV(tree.DecisionTreeClassifier(), parameters, n_jobs=4)
            model.fit(X=X.iloc[:, :-1], y=y)
            tree_model = model.best_estimator_
            logging.info(model.best_score_, model.best_params_)
            logging.info("=======Test ba model on voxceleb1======")
            test_acc = test_vox1(ba, tree_model, features_vox1, df_binary, meta_vox1, mloc_train, floc_train)
            logging.info("=======Building explainer=======")
            X = X.iloc[:, :-1]
            explainer = shap.TreeExplainer(tree_model)
            shap_values = explainer.shap_values(X)
            logging.info("=======End explainer=======")
            df_0 = pd.DataFrame(shap_values[0], columns=X.columns)
            var_groups, mean_shap_group, meanshap_val_var = contributions(df_0, X)
            df_plot, new_shap_group, new_shap_var = build_dataframe(var_groups, mean_shap_group, meanshap_val_var)
            logging.info(f"==Contribution of each member to the family for {ba}==")
            plot_family_bars(df_plot, ba)
            logging.info("Contribution of BAs to each family")
            logging.info(f"==================== End {ba}======================")
            logging.info(f"===================================================")
    logging.info("Finish !")


if __name__ == '__main__':
    shap_tree_explainer()
