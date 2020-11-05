import numpy as np
import pandas as pd

from classification_titanic.processing.data_management import load_pipeline
from classification_titanic.config import config
from classification_titanic.processing.validation import validate_inputs


_titanic_pipe = load_pipeline(file_name=config.PIPELINE_NAME)


def make_prediction(*, input_data) -> dict:
    """Make a predicion using the saved model pipeline"""

    data = pd.read_json(input_data)
    validated_data = validate_inputs(data)
    prediction = _titanic_pipe.predict(validated_data[config.FEATURES])
    response = {"predictions": prediction}

    return response
