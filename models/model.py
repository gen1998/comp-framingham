from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import StackingRegressor
from catboost import CatBoostRegressor
from sklearn import metrics
import xgboost as xgb
import lightgbm as lgb
import numpy as np
from abc import abstractmethod
import pandas as pd



class Model:
    @abstractmethod
    def train_and_predict(self, X_train, X_valid, y_train, y_valid, X_test, params):
        raise NotImplementedError


# Validation function
n_folds = 5


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def auc_cv(model, train, y_train, x_test, params):
    target = y_train

    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    score = []
    y_preds = []
    y_valids = np.empty(0)
    y_valid_preds = np.empty(0)

    for train_index, valid_index in kf.split(target):
        x_train, x_valid = (train.iloc[train_index, :], train.iloc[valid_index, :])
        y_train, y_valid = (target.iloc[train_index], target.iloc[valid_index])

        y_pred, y_valid_pred = model.train_and_predict(x_train, y_train, x_valid, y_valid, x_test, params)

        fpr, tpr, thresholds = metrics.roc_curve(y_valid, y_valid_pred)
        auc = metrics.auc(fpr, tpr)

        score.append(auc)
        y_preds.append(y_pred)
        y_valids = np.hstack([y_valids, y_valid])
        y_valid_preds = np.hstack([y_valid_preds, y_valid_pred])

    return np.array(score), y_preds, y_valids, y_valid_preds


class Model_Lasso(Model):
    def train_and_predict(self, x_train, y_train, x_valid, y_valid, x_test, params):
        model = Lasso(alpha=params["alpha"])

        model.fit(x_train, y_train)

        y_valid_pred = model.predict(x_valid)
        y_pred = model.predict(x_test)

        return y_pred, y_valid_pred


class Model_Enet(Model):
    def train_and_predict(self, x_train, y_train, x_valid, y_valid, x_test, params):
        model = ElasticNet(alpha=params["alpha"], l1_ratio=params["l1_ratio"])
        model.fit(x_train, y_train)

        y_valid_pred = model.predict(x_valid)
        y_pred = model.predict(x_test)

        return y_pred, y_valid_pred


class Model_Ridge(Model):
    def train_and_predict(self, x_train, y_train, x_valid, y_valid, x_test, params):
        model = Ridge(alpha=params["alpha"])
        model.fit(x_train, y_train)

        y_valid_pred = model.predict(x_valid)
        y_pred = model.predict(x_test)

        return y_pred, y_valid_pred


class Model_RandomForest(Model):
    def train_and_predict(self, x_train, y_train, x_valid, y_valid, x_test, params):
        model = RandomForestRegressor(**params)
        model.fit(x_train, y_train)

        y_valid_pred = model.predict(x_valid)
        y_pred = model.predict(x_test)

        return y_pred, y_valid_pred


class Model_SVR(Model):
    def train_and_predict(self, x_train, y_train, x_valid, y_valid, x_test, params):
        model = SVR(kernel=params["kernel"],
                    degree=params["degree"],
                    coef0=params["coef0"],
                    C=params["C"],
                    epsilon=params["epsilon"])

        model.fit(x_train, y_train)

        y_valid_pred = model.predict(x_valid)
        y_pred = model.predict(x_test)

        return y_pred, y_valid_pred


class Model_GBR(Model):
    def train_and_predict(self, x_train, y_train, x_valid, y_valid, x_test, params):
        model = GradientBoostingRegressor(**params)
        model.fit(x_train, y_train)

        y_valid_pred = model.predict(x_valid)
        y_pred = model.predict(x_test)

        return y_pred, y_valid_pred


class Model_CatBoost(Model):
    def train_and_predict(self, x_train, y_train, x_valid, y_valid, x_test, params):
        model = CatBoostRegressor(**params)
        model.fit(x_train, y_train)

        y_valid_pred = model.predict(x_valid)
        y_pred = model.predict(x_test)

        return y_pred, y_valid_pred


class Model_Lgb(Model):
    def train_and_predict(self, x_train, y_train, x_valid, y_valid, x_test, params):
        lgb_train = lgb.Dataset(x_train, y_train)
        lgb_eval = lgb.Dataset(x_valid, y_valid, reference=lgb_train)

        model = lgb.train(params,
                          lgb_train,
                          valid_sets=lgb_eval,
                          num_boost_round=5000,
                          early_stopping_rounds=50,
                          verbose_eval=False)

        y_valid_pred = model.predict(x_valid, num_iteration=model.best_iteration)
        y_pred = model.predict(x_test, num_iteration=model.best_iteration)
        print(pd.DataFrame(sorted(zip(model.feature_importance(), x_train.columns)), columns=['Value','Feature']))

        return y_pred, y_valid_pred


class Model_Xgb(Model):
    def train_and_predict(self, x_train, y_train, x_valid, y_valid, x_test, params):
        xgb_train = xgb.DMatrix(x_train, y_train)
        xgb_eval = xgb.DMatrix(x_valid, y_valid)
        xgb_test = xgb.DMatrix(x_test)
        xgb_valid = xgb.DMatrix(x_valid)

        watchlist = [(xgb_eval, 'eval'), (xgb_train, 'train')]

        model = xgb.train(params,
                          xgb_train,
                          num_boost_round=4000,
                          evals=watchlist,
                          early_stopping_rounds=10)

        y_valid_pred = model.predict(xgb_valid)
        y_pred = model.predict(xgb_test)

        return y_pred, y_valid_pred


class Model_Stacking(Model):
    def train_and_predict(self, x_train, y_train, x_valid, y_valid, x_test, params):
        model_list = [GradientBoostingRegressor(**params["GBR"]), RandomForestRegressor(**params["RandomForest"])]
        model_name = ["GBR", "RandomForest"]

        estimators = []
        for model, name in zip(model_list, model_name):
            estimators.append((name, model))
        model = StackingRegressor(estimators=estimators, final_estimator=RandomForestRegressor(**params["RandomForest"]))
        model.fit(x_train, y_train)

        y_valid_pred = model.predict(x_valid)
        y_pred = model.predict(x_test)

        return y_pred, y_valid_pred
