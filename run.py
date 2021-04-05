import pandas as pd
import datetime
import logging
from sklearn.model_selection import KFold
import argparse
import json
import numpy as np

from utils import load_datasets, load_target
from logs.logger import log_best
from models.model import *


parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./configs/default.json')
options = parser.parse_args()
config = json.load(open(options.config))

now = datetime.datetime.now()
logging.basicConfig(
    filename='./logs/log_{0:%Y%m%d%H%M%S}.log'.format(now), level=logging.DEBUG
)
logging.debug('./logs/log_{0:%Y%m%d%H%M%S}.log'.format(now))

feats = config['features']
logging.debug(feats)

target_name = config['target_name']

X_train_all, X_test = load_datasets(feats)
y_train_all = load_target(target_name)

logging.debug(X_train_all.shape)
logging.debug(X_train_all.columns)

# models = [Model_Lasso(), Model_Enet(), Model_RandomForest(), Model_GBR(), Model_SVR(), Model_CatBoost(), Model_Lgb()]
models = [Model_GBR(), Model_RandomForest(), Model_CatBoost(), Model_Xgb(), Model_Lgb()]
# models = [Model_Stacking()]
# model_name = ["Lasso", "Enet", "RandomForest", "GBR", "SVR", "CatBoost", "Lightgbm"]
model_name = ["GBR", "RandomForest", "CatBoost", "Xgboost", "Lightgbm"]
# model_name = ["stacking"]
params = config["base_models"]
y_preds = {}
for model, name in zip(models, model_name):
    if name in params.keys():
        param = params[name]
    else:
        param = []

    score, y_pred = rmsle_cv(model, X_train_all, y_train_all, X_test, param)
    print('\n===CV scores model: {}==='.format(name))
    print(score)
    print("score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
    logging.debug("{}, score: {:.4f} ({:.4f})".format(name, score.mean(), score.std()))
    y_preds[name] = sum(y_pred) / len(y_pred)

# submitファイルの作成
ID_name = config['ID_name']
sub = pd.DataFrame(pd.read_csv('./data/input/test.csv')[ID_name])

y_sub = 0.5*y_preds["CatBoost"] + 0.5*y_preds["Xgboost"]

sub[target_name] = np.exp(y_sub)

sub.to_csv('./data/output/sub_{0:%Y%m%d%H%M%S}_{1}.csv'.format(now, score),index=False)

sub = pd.DataFrame(pd.read_csv('./data/input/test.csv')[ID_name])
y_sub = 0.4*y_preds["Xgboost"] + 0.4*y_preds["CatBoost"] + 0.2*y_preds["Lightgbm"]

sub[target_name] = np.exp(y_sub)

sub.to_csv('./data/output/sub_{0:%Y%m%d%H%M%S}_{1}_2.csv'.format(now, score),index=False)