## ML 모형 평가

import numpy as np
import pandas as pd
from preprocess import fit_decision_tree, load_data
from sklearn.metrics import confusion_matrix, roc_curve

from sklearn.tree import DecisionTreeClassifier
def load_data(x, y):
    pass 

def fit_decision_tree(x, y):
    tree = DecisionTreeClassifier()
    tree.fit(x, y)
    return tree

# 평가를 위해 변경하지 마세요.
SEED = 42
np.random.seed(SEED)


def get_pred(model, X):
    # 지시사항 1번을 참고하여 코드를 작성하세요.
    y_pred = model.predict(X)
    # y_pred = None

    return y_pred


def get_conf_mat(y, y_pred):
    # 지시사항 2번을 참고하여 코드를 작성하세요.
    conf_mat = None

    return conf_mat


def get_cm_values(conf_mat):
    # 지시사항 3번을 참고하여 코드를 작성하세요.
    None

    return tn, fp, fn, tp


def get_eval_idx(tn, fp, fn, tp):
    # 지시사항 4번을 참고하여 코드를 작성하세요.
    precision = None
    recall = None
    f1_score = None

    return precision, recall, f1_score


def get_f_rate(y, y_pred):
    # 지시사항 5번을 참고하여 코드를 작성하세요.
    None

    return fper, tper


def main():
    X, y = load_data()
    dt_model = fit_decision_tree(X, y)

    y_pred = get_pred(dt_model, X)
    conf_mat = get_conf_mat(y, y_pred)
    tn, fp, fn, tp = get_cm_values(conf_mat)

    prec, recall, f1 = get_eval_idx(tn, fp, fn, tp)
    print(
        "Decision Tree의 precision: {} / recall: {} / f1_score: {}".format(
            prec, recall, f1
        )
    )
    fper, tper = get_f_rate(y, y_pred)
    print("FPR: {} / TPR: {}".format(fper, tper))


if __name__ == "__main__":
    main()
