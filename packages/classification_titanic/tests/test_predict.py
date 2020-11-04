import numpy as np

from classification_titanic.predict import make_prediction
from classification_titanic.processing.data_management import load_dataset
from classification_titanic.config import config


def test_make_single_prediction():
    # Data
    test_data = load_dataset(file_name=config.TESTING_DATA_FILE)
    single_test_json = test_data[0:1].to_json(orient='records')

    # Predict
    subject = make_prediction(input_data=single_test_json)

    # Test
    assert subject is not None
    assert isinstance(subject.get("predictions")[0], (int, np.integer))
