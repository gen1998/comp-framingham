import pandas as pd
import numpy as np

from base import Feature, get_arguments, generate_features
from scipy import stats
from scipy.stats import norm, skew
from scipy.special import boxcox1p
from sklearn.preprocessing import LabelEncoder

Feature.dir = 'features'


def skewed_change(all_data, feats):
    skewed_feats = all_data[feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    skewness = skewed_feats[abs(skewed_feats) > 0.75]
    skewed_features = skewness.index
    lam = 0.15
    for feat in skewed_features:
        all_data[feat] = boxcox1p(all_data[feat], lam)

    return all_data

class Default(Feature):
    def create_features(self):
        self.train = train.drop(['TenYearCHD', "prevalentStroke"], axis=1)
        self.test = test.drop(["prevalentStroke"], axis=1)

class Category(Feature):
    def create_features(self):
        feature_columns = ['male', 'education', 'currentSmoker', 'cigsPerDay', 'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes']
        self.train[feature_columns] = train[feature_columns]
        self.test[feature_columns] = test[feature_columns]

class Continuous(Feature):
    def create_features(self):
        n_train = train.shape[0]
        feature_columns = ['totChol', 'age', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']
        all_data = pd.concat((train[feature_columns], test[feature_columns]), sort=False)
        all_data["sys_diaBP"] = all_data["sysBP"] * all_data["diaBP"]
        all_data["age_2"] = all_data["age"]**2
        all_data["age_sys"] = all_data["age"] * all_data["sysBP"]

        feature_columns.extend(['sys_diaBP', 'age_2', 'age_sys'])
        all_data = skewed_change(all_data, feature_columns)

        self.train[feature_columns] = all_data[feature_columns][:n_train]
        self.test[feature_columns] = all_data[feature_columns][n_train:]

class Additional(Feature):
    def create_features(self):
        self.train["prevalentmind"] = train["prevalentStroke"] + train["prevalentHyp"]
        self.test["prevalentmind"] = test["prevalentStroke"] + test["prevalentHyp"]
        #self.train["prevalentmind_s"] = train["prevalentStroke"] * train["prevalentHyp"]
        #self.test["prevalentmind_s"] = test["prevalentStroke"] * test["prevalentHy

if __name__ == '__main__':
    args = get_arguments("hello")

    train = pd.read_csv('data/input/train.csv')
    test = pd.read_csv('data/input/test.csv')

    generate_features(globals(), args.force)
