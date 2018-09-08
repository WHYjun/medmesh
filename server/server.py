import fitbit
from flask import Flask, jsonify
import logging
import os
import requests

# App
from configure import app

# Log
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
log_formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')
file_handler = logging.FileHandler('logs/server.log')
file_handler.setFormatter(log_formatter)
logger.addHandler(file_handler)

# Modules
from modules import fitbit_module

# MongoDB
from pymongo import MongoClient
client = MongoClient()
db = client.heartcare

# User Data
user_data = {
    'Amy': {
        'age': 58.0,
        'hypertension': 1.0,
        'heart_disease': 1.0,
        'bmi': 38.0,
        'gender_numeric': 2.0,
        'ever_married_numeric': 1.0,
        'work_type_numeric': 1.0,
        'residence_numeric': 2.0,
        'smoking_status_numeric': 1.0
    },
    'Bob': {
        'age': 58.0,
        'hypertension': 1.0,
        'heart_disease': 1.0,
        'bmi': 38.0,
        'gender_numeric': 2.0,
        'ever_married_numeric': 1.0,
        'work_type_numeric': 1.0,
        'residence_numeric': 2.0,
        'smoking_status_numeric': 1.0
    },
    'Charlie': {
        'age': 58.0,
        'hypertension': 1.0,
        'heart_disease': 1.0,
        'bmi': 38.0,
        'gender_numeric': 2.0,
        'ever_married_numeric': 1.0,
        'work_type_numeric': 1.0,
        'residence_numeric': 2.0,
        'smoking_status_numeric': 1.0
    }
}

# API
@app.route('/api/user/<username>', methods=['GET'])
def get_percentage(username):
    user = user_data.get(username)
    if not user:
        logger.error('get_user({}): username not existed'.format(username))
        return None
    if os.environ.get('env') == 'demo':
        # TODO: Get data from real cases
        pass
    else:
        start = '13:00'
        end = '13:01'
        heart_rates = fitbit_module.get_heartrate(start=start, end=end)
    req = user
    req['heart_rate'] = calculate_hr(heart_rates['activities-heart-intraday'])
    # percentage = requests.post('url', json=req)
    return jsonify(req)


def calculate_hr(heart_rates):
    total = 0
    for heart_rate in heart_rates['dataset']:
        total += heart_rate['value']
    avg = total/len(heart_rates['dataset'])
    return round(avg, 2)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int("8080"), debug=True)
