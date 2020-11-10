from classification_titanic.config import config as model_config
from classification_titanic.processing.data_management import load_dataset
from classification_titanic import __version__ as _version

import json

from api import __version__ as api_version


def test_health_endpoint_returns_200(flask_test_client):

    response = flask_test_client.get('/health')

    assert response.status_code == 200


def test_version_endpoint_returns_version(flask_test_client):

    response = flask_test_client.get('/version')

    assert response.status_code == 200
    response_json = json.loads(response.data)
    assert response_json['model_version'] == _version
    assert response_json['api_version'] == api_version


def test_prediction_endpoint_returns_prediction(flask_test_client):

    # Load the test data from the classification_titanic package
    # This is important as it makes it harder for the test
    # data versions to get confused by not spreading it
    # across packages
    test_data = load_dataset(file_name=model_config.TESTING_DATA_FILE)
    post_json = test_data[0:1].to_json(orient='records')

    print(f"POST JSON {post_json}")

    response = flask_test_client.post('/v1/predict/classification',
                                      json=post_json)

    assert response.status_code == 200

    response_json = json.loads(response.data)
    if isinstance(response_json['predictions'], list):
        prediction = response_json['predictions'][0]
    else:
        prediction = response_json['predictions']
    response_version = response_json['version']

    assert isinstance(prediction, int)
    assert response_version == _version
