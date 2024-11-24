"""
Perform inference/prediction as an API endpoint using the saved SVM model that was trained
"""

#Import statements
import pandas as pd #data processing, CSV file I/O (e.g. pd.read_csv)
import os #os file paths
import pickle #Saving and loading models or python objects
from flask import Flask, request, jsonify
import sys
import json

#Global Constants
DATA_PATH = r"..\data\MobilePriceClassification" #Path to raw data
MODEL_PATH = r"..\models" #Path to models 


app = Flask(__name__)

with open(os.path.join(MODEL_PATH,'model.pkl'), 'rb') as f:
    svm_model = pickle.load(f)

def prepare_features(feature_sample):
    """ 
    Perform data preprocessing and feature engineering on the 1 sample dataframe

    Args:
        feature_sample: 1 sample dataframe (raw input data)
    """
    #Remove unique identifier column
    feature_sample.drop(["Unnamed: 0"],axis=1,inplace=True)

    feature_sample["old"] = 1
    filter_rows = (feature_sample["wifi"] == 1) & (feature_sample["blue"] == 1) & (feature_sample["dual_sim"] == 1) & (feature_sample["four_g"] == 1) \
    & (feature_sample["touch_screen"] == 1)
    feature_sample.loc[filter_rows,"old"] = 0
    return feature_sample
        
def predict(features):
    """ 
    Perform prediction using the loaded/saved model on the input data

    Args:
        features: 1 sample dataframe with data preprocessing and feature engineering performed on the sample
    """
    preds = svm_model.predict(features)
    return preds[0]


@app.route('/', methods=['GET', 'POST'])
def predict_endpoint():
    """ 
    API Endpoint for predict_api.py script to call this endpoint with a sample of the dataframe
    """    
    #Retrieve the input data
    phone = request.get_json()
    print(f"This is the phone {phone}, {type(phone)}", file=sys.stderr)

    #Convert to dataframe for preprocessing functions later (e.g. prepare functions and predict)
    phone = pd.DataFrame(json.loads(phone),index=[0])

    #Perform preprocessing
    feature_sample = prepare_features(phone)
    pred = predict(feature_sample)
    print(f"This is the pred {pred}, {type(pred)}", file=sys.stderr)
    #Prep data as json
    result = {
        'phone_cat': str(pred)
    }
    #return as a json object
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0',port=9696)



