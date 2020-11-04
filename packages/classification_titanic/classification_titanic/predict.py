import numpy as np
import pandas as pd

from classification_titanic.processing.data_management import load_pipeline
from classification_titanic.config import config


_titanic_pipe = load_pipeline(file_name=config.PIPELINE_NAME)


def make_prediction(*, input_data) -> dict:
    """Make a predicion using the saved model pipeline"""

    data = pd.read_json(input_data)
    prediction = _titanic_pipe.predict(data[config.FEATURES])
    response = {"predictions":prediction}

    return response
