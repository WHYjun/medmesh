import fitbit
from flask import Flask
import requests

# App
from configure import app

# MongoDB
from pymongo import MongoClient
client = MongoClient()
db = client.heartcare

# Endpoints
from endpoints import fitbit_endpoints

app.register_blueprint(fitbit_endpoints.endpoints)


# API


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int("8080"), debug=True)
