import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
import optuna
import xgboost as xgb

def load_datasets(feats):
    dfs = [pd.read_feather(f'../features/{f}_train.ftr') for f in feats]
    X_train = pd.concat(dfs, axis=1, sort=False)
    dfs = [pd.read_feather(f'../features/{f}_test.ftr') for f in feats]
    X_test = pd.concat(dfs, axis=1, sort=False)
    return X_train, X_test


def load_target(target_name):
    train = pd.read_csv('../data/input/train.csv')
    y_train = train[target_name]
    return y_train

def objective(trial):
    x_train_, x_test, y_train_, y_test = train_test_split(X_train, Y_train, test_size=0.25, shuffle=True)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train_, y_train_, test_size=0.25, shuffle=True)

    xgb_train = xgb.DMatrix(x_train, y_train)
    xgb_eval = xgb.DMatrix(x_valid, y_valid)
    xgb_test = xgb.DMatrix(x_test)

    params = {
           "random_state" : 42,
           "learning_rate" : trial.suggest_loguniform('learning_rate', 0.01, 0.2),
           "max_depth" : trial.suggest_int('max_depth', 2, 8),
            "objective" : "binary:logistic",
            "colsample_bytree":trial.suggest_uniform("colsample_bytree", 0.01, 1.0)
        }

    watchlist = [(xgb_eval, 'eval'), (xgb_train, 'train')]

    model = xgb.train(
            params, xgb_train,
            num_boost_round=5000,
            evals=watchlist,
            early_stopping_rounds=10,
            verbose_eval = False
        )

    y_pred = model.predict(xgb_test)

    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
    score = metrics.auc(fpr, tpr)

    return score

X_train, X_test = load_datasets(["category", "continuous", "additional"])
Y_train = load_target("TenYearCHD")

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
trial = study.best_trial

print('Best Trial')
print('\tValue: {}'.format(trial.value))
print(' \tParams: ')
for key, value in trial.params.items():
    print('\t\t{}: {}'.format(key, value))