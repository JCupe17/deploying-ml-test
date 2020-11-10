from flask import Blueprint, request, jsonify
from classification_titanic.predict import make_prediction
from classification_titanic import __version__ as _version

from api.config import get_logger
from api.validation import validate_inputs
from api import __version__ as api_version

_logger = get_logger(logger_name=__name__)

prediction_app = Blueprint('prediction_app', __name__)


@prediction_app.route('/health', methods=['GET'])
def health():
    if request.method == 'GET':
        _logger.info('health status OK')
        return 'OK'


@prediction_app.route('/version', methods=['GET'])
def version():
    if request.method == 'GET':
        return jsonify({'model_version': _version,
                        'api_version': api_version})


@prediction_app.route('/v1/predict/classification', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Step 1: Extract POST data from request body as JSON
        json_data = request.get_json()
        _logger.info(f'Inputs: {json_data}')

        print(f"JSON DATA TYPE {type(json_data)}")
        # Step 2: Validate the input using marshmallow schema
        input_data, errors = validate_inputs(input_data=json_data)

        # Step 3: Model Prediction
        print(f"INPUT DATA TYPE {type(input_data)}")
        result = make_prediction(input_data=json_data)
        _logger.info(f'Outputs: {result}')

        # Cast to int because np.array are not recognize by json
        # predictions = int(result.get('predictions')[0])
        # Step 4: Convert numpy ndarray to list
        predictions = result.get("predictions").tolist()
        version = result.get('version')

        # Step 5: Return the response as JSON
        return jsonify({'predictions': predictions,
                        'version': version,
                        'errors': errors})
