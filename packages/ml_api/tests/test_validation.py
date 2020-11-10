import json

from classification_titanic.config import config
from classification_titanic.processing.data_management import load_dataset


def test_prediction_endpoint_validation_200(flask_test_client):
    # Load the test data from the classification_titanic package
    test_data = load_dataset(file_name=config.TESTING_DATA_FILE)
    post_json = test_data.to_json(orient='records')

    # Predict
    response = flask_test_client.post('/v1/predict/classification',
                                      json=post_json)

    # Test
    assert response.status_code == 200
    response_json = json.loads(response.data)

    # Check of correct number of errors removec if any
    if response_json.get('errors') is None:
        assert len(response_json.get('predictions')) == len(test_data)
    else:
        assert len(response_json.get('predictions')) + \
               len(response_json.get('errors')) == len(test_data)
