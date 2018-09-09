import fitbit
from flask import Flask, jsonify, render_template, request
import logging
import os
import requests

# App
from configure import app

# add the mlpred folder
import sys
sys.path.insert(0, '../mlPredictor')
import predictionEngine
model = predictionEngine.train_model()

# Log
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
log_formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')
file_handler = logging.FileHandler('logs/server.log')
file_handler.setFormatter(log_formatter)
logger.addHandler(file_handler)

# Modules
from modules import fitbit_module
from modules import betterdoctor

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


@app.route('/')
def register():
    return render_template('register.html')






@app.route('/api/signup', methods=['POST'])
def signup():
    username = request.form['username']
    pw = request.form['password']
    insurance = request.form['insurance']
    age = request.form['']
    db.user.insert({'username': username,
                    'password': pw,
                    'insurance': insurance,
                    'birthday': birthday})
    response = Response()
    response.status_code = 200
    return response.json()


# @app.route('/api/user/<username>', methods=['GET'])
@app.route('/api/user', methods=['POST'])
def get_percentage():
    user = user_data.get('Amy')
    if not user:
        logger.error('get_user({}): username not existed'.format(username))
        return None
    if os.environ.get('env') == 'demo':
        # TODO: Get data from real cases
        pass
    else:
        start = '13:00'
        end = '13:01'
        try:
            heart_rates = fitbit_module.get_heartrate(start=start, end=end)
            avg_hr = calculate_hr(heart_rates['activities-heart-intraday'])
        except Exception as e:
            avg_hr = 80.00
    req = user
    req['heart_rate'] = avg_hr
    stroke_probability = predictionEngine.predict(
        model, req)
    print("stroke_probability >> ", stroke_probability)
    req['stroke_probability'] = stroke_probability
    req['visit'] = getVisitType(req)
    res = {
        "user_id": "2",
        "bot_id": "1",
        "module_id": "3",
        "message": req['visit']
    }
    return jsonify(res)


@app.route('/api/insurance_list', methods=['GET'])
def get_insurance_list():
    location = 'pa-philadelphia'
    insurances = betterdoctor.getInsurances(limit=10)
    insurance_list = []
    for insurance in insurances['data']:
        for plan in insurance['plans']:
            insurance_list.append({
                'name': plan['name'],
                'uid': plan['uid']
            })
    return jsonify(insurance_list)


@app.route('/api/insurance/<insurance_name>', methods=['GET'])
def get_insurance(insurance_name):
    location = 'pa-philadelphia'
    # uid = None
    # insurances = betterdoctor.getInsurances()
    # for insurance in insurances['data']:
    #     if insurance_name in insurance['uid']:
    #         uid = insurance['uid']
    # if not uid:
    #     return None
    uid = 'aetna-aetnabasichmo'
    doctors = betterdoctor.getDoctors(
        location=location, insurance=uid, limit=3)
    return jsonify(doctors)


def calculate_hr(heart_rates):
    total = 0
    for heart_rate in heart_rates['dataset']:
        total += heart_rate['value']
    avg = total/len(heart_rates['dataset'])
    return round(avg, 2)


def getVisitType(req):
    if req['stroke_probability'] < 0.4:
        return "Primary Care"
    elif req['stroke_probability'] >= 0.4 and req['stroke_probability'] < 0.7:
        return "Urgent Care"
    else:
        return "Emergency Room"


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int("8080"), debug=True)
