from flask import Flask


def create_app() -> Flask:
    """Create a Flask app instance."""

    flask_app = Flask('ml_api')

    # Import blueprints
    from api.controller import prediction_app
    flask_app.register_blueprint(prediction_app)

    return flask_app
