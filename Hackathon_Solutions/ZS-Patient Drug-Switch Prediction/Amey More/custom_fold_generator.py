from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold, TimeSeriesSplit, train_test_split
from enum import Enum
'''
include an option to stratify by a column and not only target variable
'''

class FoldScheme(Enum):
    '''
    Class to select the type of fold scheme to apply
    '''
    StratifiedKFold = "StratifiedKFold"
    KFold = "KFold"
    GroupKFold = "GroupKFold"
    TimeSeriesSplit = "TimeSeriesSplit"
    train_test_split = "train_test_split"


class CustomFolds(object):
    def __init__(self, validation_scheme, num_folds=5, random_state=100, shuffle=True, test_size = 0.2):
        self.validation_scheme=validation_scheme
        self.random_state=random_state
        self.shuffle=shuffle
        self.num_folds=num_folds
        self.test_size = test_size

    def split(self, X, y=None, group=None, **kwargs): ## the group here will be passed on from the class where this is being called

        if self.validation_scheme is None or isinstance(self.validation_scheme, KFold) or self.validation_scheme == FoldScheme.KFold.name or self.validation_scheme == FoldScheme.KFold:
            folds = KFold(n_splits=self.num_folds, random_state=self.random_state, shuffle=self.shuffle)
            self.indices = [(train_index, test_index) for (train_index, test_index) in folds.split(X)]

        elif isinstance(self.validation_scheme, StratifiedKFold) or self.validation_scheme == FoldScheme.StratifiedKFold.name or self.validation_scheme == FoldScheme.StratifiedKFold:
            if y is None or X.shape[0] != y.shape[0]: raise ValueError("Y should be passed and X and Y should be of same length for StratifiedKFold")
            folds = StratifiedKFold(n_splits=self.num_folds, random_state=self.random_state, shuffle=self.shuffle)
            self.indices = [(train_index, test_index) for (train_index, test_index) in folds.split(X, y)]

        elif isinstance(self.validation_scheme, GroupKFold) or self.validation_scheme == FoldScheme.GroupKFold.name or self.validation_scheme == FoldScheme.GroupKFold:
            folds = GroupKFold(n_splits=self.num_folds)
            self.indices = [(train_index, test_index) for (train_index, test_index) in folds.split(X, y, groups=group)]

        elif isinstance(self.validation_scheme, TimeSeriesSplit) or self.validation_scheme == FoldScheme.TimeSeriesSplit.name or self.validation_scheme == FoldScheme.TimeSeriesSplit:
            folds = TimeSeriesSplit(n_splits=self.num_folds)
            self.indices = [(train_index, test_index) for (train_index, test_index) in folds.split(X)]

        elif self.validation_scheme == FoldScheme.train_test_split.name or self.validation_scheme == FoldScheme.train_test_split:
            # validation_scheme is a simple train test split. testsize is used to determine the size of test samples
            self.indices = [train_test_split(list(range(X.shape[0])), test_size=self.test_size, shuffle=self.shuffle)]

        elif callable(self.validation_scheme):
            # validation_scheme is a callable funtion which will take X and y as params.
            self.indices  = self.validation_scheme(X, y, **kwargs)

        else:
            if not isinstance(self.validation_scheme, list):
                raise ValueError("Validation Schema should be a list of (train_indexes, test_indexes)")
            self.indices=self.validation_scheme
        return self.indices
