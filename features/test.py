import pandas as pd

from base import Feature, get_arguments, generate_features

Feature.dir = ''


class Flrsize(Feature):
    def create_features(self):
        self.train['1-2'] = train['1stFlrSF'] + train['2ndFlrSF'] + 1
        self.test['1-2'] = test['1stFlrSF'] + test['2ndFlrSF'] + 1


if __name__ == '__main__':
    args = get_arguments("やあ")

    train = pd.read_csv('../data/input/train.csv')
    test = pd.read_csv('../data/input/test.csv')

    generate_features(globals(), args.force)
