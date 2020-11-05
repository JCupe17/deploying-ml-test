import pathlib
import classification_titanic

PACKAGE_ROOT = pathlib.Path(classification_titanic.__file__).resolve().parent
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"
DATASET_DIR = PACKAGE_ROOT / "datasets"

# model
PIPELINE_NAME = "logistic_regression.pkl"

# data
DATA_FILE = "titanic.csv"
TESTING_DATA_FILE = "test.csv"
TRAINING_DATA_FILE = "train.csv"

# Variables
TARGET = "survived"

CATEGORICAL_VARS = ['sex', 'cabin', 'embarked', 'title']

NUMERICAL_VARS = ['age', 'fare', 'sibsp', 'parch']

FEATURES = CATEGORICAL_VARS + NUMERICAL_VARS

NUMERICAL_VARS_WITH_NA = []
CATEGORICAL_VARS_WITH_NA = []

NUMERICAL_NA_NOT_ALLOWED = []
CATEGORICAL_NA_NOT_ALLOWED = []

CABIN = 'cabin'
