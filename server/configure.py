from flask import Flask
import os

FLASK_NAME = os.environ.get("FLASK_NAME")
FLASK_SECRET_KEY = os.environ.get("FLASK_SECRET_KEY")
STATIC_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")


def create_app():
    app = Flask(FLASK_NAME)
    app.secret_key = FLASK_SECRET_KEY
    app.static_folder = STATIC_PATH
    return app


app = create_app()
