from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from classification_titanic.processing import preprocessors as pp
from classification_titanic.processing import features
from classification_titanic.config import config

import logging

_logger = logging.getLogger(__name__)

titanic_pipe = Pipeline(
    # complete with the list of steps from the preprocessors file
    # and the list of variables from the config
    [
        ('missing_indicator',
            pp.MissingIndicator(variables=config.NUMERICAL_VARS)),
        ('categorical_imputer',
            pp.CategoricalImputer(variables=config.CATEGORICAL_VARS)),
        ('numerical_imputer',
            pp.NumericalImputer(variables=config.NUMERICAL_VARS)),
        ('extract_first_letter',
            features.ExtractFirstLetter(variables=config.CABIN)),
        ('rare_label_encoder',
            pp.RareLabelCategoricalEncoder(tol=0.05, variables=config.CATEGORICAL_VARS)),
        ('categorical_encoder',
            pp.CategoricalEncoder(variables=config.CATEGORICAL_VARS)),
        ('scaler', StandardScaler()),
        ('lr_model', LogisticRegression(C=0.0005, random_state=0))
        ]
    )
