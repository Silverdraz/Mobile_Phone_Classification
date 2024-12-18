"""
Perform inference/prediction as an API endpoint using the saved SVM model that was trained
"""

#Import statements
import pandas as pd #data processing, CSV file I/O (e.g. pd.read_csv)
import os #os file paths
import pickle #Saving and loading models or python objects
from flask import Flask, request, jsonify #web framework
import sys #system module
import json #json handling
import mlflow #experiment tracking

#Global Constants
MODEL_PATH = r"../models" #Path to models 
RUN_ID = "dcf13e6b81f740ac8d959d27fb18683c" #ID of ML Model saved on MLFlow
mlflow.set_tracking_uri('http://ec2-54-252-156-209.ap-southeast-2.compute.amazonaws.com:5000')
logged_model = f"runs:/{RUN_ID}/best_estimator"

#Load model as a PyFuncModel
#svm_model = mlflow.pyfunc.load_model(logged_model)


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
    preds = svm_model.predict(pd.DataFrame(features))
    return preds[0]


@app.route('/', methods=['GET', 'POST'])
def predict_endpoint():
    """ 
    API Endpoint for predict_api.py script to call this endpoint with a sample of the dataframe
    """    
    #Retrieve the input data
    phone = request.get_json()
    #print(f"This is the phone {phone}, {type(phone)}", file=sys.stderr)

    #Convert to dataframe for preprocessing functions later (e.g. prepare functions and predict)
    phone = pd.DataFrame(json.loads(phone),index=[0])
    #print(f"This is the phone 2 {phone}, {type(phone)}", file=sys.stderr)

    #Perform preprocessing
    feature_sample = prepare_features(phone)
    #print(f"This is the phone 3 {phone}, {type(phone)}", file=sys.stderr)
    #print(f"This is the svm modedl {svm_model}, {type(phone)}", file=sys.stderr)
    pred = predict(feature_sample)
    #print(f"This is the pred {pred}, {type(pred)}", file=sys.stderr)
    #Prep data as json
    result = {
        'phone_cat': str(pred)
    }
    #return as a json object
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0',port=9696)



