"""
Perform inference/prediction as an API endpoint using the saved SVM model that was trained
"""

#Import statements
import pandas as pd #data processing, CSV file I/O (e.g. pd.read_csv)
from flask import Flask, request, jsonify #web framework
import json #json handling
import mlflow #experiment tracking


#Global constant
DATA_PATH = r"..\data\MobilePriceClassification" #Path to raw data


def init(RUN_ID):
    model = load_model(RUN_ID)
    model_service = ModelService(model)
    return model_service

def load_model(RUN_ID):
    #RUN_ID = "dcf13e6b81f740ac8d959d27fb18683c" #ID of ML Model saved on MLFlow
    mlflow.set_tracking_uri('http://ec2-54-252-156-209.ap-southeast-2.compute.amazonaws.com:5000')
    logged_model = f"runs:/{RUN_ID}/best_estimator"

    #Load model as a PyFuncModel
    svm_model = mlflow.pyfunc.load_model(logged_model)
    return svm_model

class ModelService():

    def __init__(self,model):
        self.model = model

    def prepare_features(self,feature_sample):
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
        
    def predict(self,features):
        """ 
        Perform prediction using the loaded/saved model on the input data

        Args:
            features: 1 sample dataframe with data preprocessing and feature engineering performed on the sample
        """
        preds = self.model.predict(pd.DataFrame(features))
        return preds[0]

    def sample_test(self,phone_df):
        """ Retrieve raw test data and samples 1 row/input data and returns it

            Returns:
                feature_sample: raw row/sample of test data in json format
        """    
        #Sample from test df
        feature_sample = phone_df
        feature_sample['px_height'] = feature_sample['px_height'].astype('float64')
        feature_sample['sc_w'] = feature_sample['sc_w'].astype('float64')
        print(type(feature_sample),"this is type of feature sample")
        print(feature_sample.dtypes,"These are the dtypes")
        #Convert to json for input to request.post
        #feature_sample = feature_sample.iloc[0, :].to_json()
        #Json string with square brackets (Extract the first str json)
        feature_sample = feature_sample.to_json(orient='records')[1:-1]
        print(type(feature_sample),"this is type of feature sample")
        print(feature_sample,"These are the dtypes")
        return feature_sample

    def lambda_handler(self,event):
        phone = event["phone"]
        phone_df = pd.json_normalize(phone)
        feature_sample = self.sample_test(phone_df)


        #Convert to dataframe for preprocessing functions later (e.g. prepare functions and predict)
        feature_sample = pd.DataFrame(json.loads(feature_sample),index=[0])

        feature_sample = self.prepare_features(feature_sample)
        print(feature_sample)
        pred = self.predict(feature_sample)
        #Prep data as json
        result = {
            'phone_cat': str(pred)
        }

        print(result)
        return result

