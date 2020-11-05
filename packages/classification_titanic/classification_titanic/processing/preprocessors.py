import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

from classification_titanic.processing.errors import InvalidModelInputError


# Add binary variable to indicate missing values
class MissingIndicator(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        # to accommodate sklearn pipeline functionality
        return self

    def transform(self, X):
        # add indicator
        X = X.copy()
        for var in self.variables:
            X[var+'NA'] = np.where(X[var].isnull(), 1, 0)
        return X


# categorical missing value imputer
class CategoricalImputer(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        # we need the fit statement to accommodate the sklearn pipeline
        return self

    def transform(self, X):
        X = X.copy()
        for var in self.variables:
            X[var] = X[var].fillna("Missing")

        return X


# Numerical missing value imputer
class NumericalImputer(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        # persist mode in a dictionary
        self.imputer_dict_ = {}
        
        for var in self.variables:
            self.imputer_dict_[var] = X[var].median()
        return self

    def transform(self, X):

        X = X.copy()

        # TODO: Check the validity of this error
        # Check that the values are non-negative
        """if not (X[self.variables] > 0).all().all():
            vars_ = self.variables[(X[self.variables] > 0).any()]

            raise InvalidModelInputError(
                f"Variables contain negative values: {vars_}"
            )"""

        for var in self.variables:
            X[var] = X[var].fillna(self.imputer_dict_[var])
        return X


# frequent label categorical encoder
class RareLabelCategoricalEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, tol=0.05, variables=None):
        self.tol = tol

        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):

        # persist frequent labels in dictionary
        self.encoder_dict_ = {}
        
        for var in self.variables:
            frequent_values = pd.Series(X[var].value_counts() / np.float(len(X)))

            self.encoder_dict_[var] = list(frequent_values[frequent_values >= self.tol].index)

        return self

    def transform(self, X):
        X = X.copy()
        for var in self.variables:
            X[var] = np.where(X[var].isin(self.encoder_dict_[var]), X[var], "Rare")
        return X


# string to numbers categorical encoder
class CategoricalEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):

        # HINT: persist the dummy variables found in train set
        self.dummies = pd.get_dummies(X[self.variables], drop_first=True).columns
        
        return self

    def transform(self, X):
        # encode labels
        X = X.copy()
        # get dummies
        X = pd.concat([X, pd.get_dummies(X[self.variables], drop_first=True)],
                axis=1)
        # drop original variables
        X.drop(labels=self.variables, axis=1, inplace=True)
        # add missing dummies if any
        missing_columns = set(self.dummies).difference(X.columns)
        for var in missing_columns:
            X[var] = 0

        return X
