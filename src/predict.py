"""
Perform inference/prediction as an API endpoint using the saved SVM model that was trained
"""

#Import statements
import pandas as pd #data processing, CSV file I/O (e.g. pd.read_csv)
import os #os file paths
import pickle #Saving and loading models or python objects
from prefect import task, flow #Orchestration Pipeline
from flask import Flask, request, jsonify
import warnings
from logging import FileHandler,WARNING
import sys
import json


#Global Constants
DATA_PATH = r"..\data\MobilePriceClassification" #Path to raw data
MODEL_PATH = r"..\models" #Path to models 

#import modules
import models #models module
import data_preprocessing #data preprocessing module

app = Flask(__name__)

with open(os.path.join(MODEL_PATH,'model.pkl'), 'rb') as f:
    svm_model = pickle.load(f)

def prepare_features(feature_sample):
    #Remove unique identifier column
    feature_sample.drop(["Unnamed: 0"],axis=1,inplace=True)

    feature_sample["old"] = 1
    filter_rows = (feature_sample["wifi"] == 1) & (feature_sample["blue"] == 1) & (feature_sample["dual_sim"] == 1) & (feature_sample["four_g"] == 1) \
    & (feature_sample["touch_screen"] == 1)
    feature_sample.loc[filter_rows,"old"] = 0
    return feature_sample
        
def predict(features):
    preds = svm_model.predict(features)
    return preds[0]


@app.route('/', methods=['GET', 'POST'])
def predict_endpoint():
    phone = request.get_json()
    print(f"This is the phone {phone}, {type(phone)}", file=sys.stderr)
    phone = pd.DataFrame(json.loads(phone),index=[0])

    feature_sample = prepare_features(phone)
    pred = predict(feature_sample)
    print(f"This is the pred {pred}, {type(pred)}", file=sys.stderr)
    result = {
        'phone_cat': str(pred)
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0',port=9696)



