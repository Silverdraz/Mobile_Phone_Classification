"""
Python Script for calling both the lambda function (lambda_function.py) script as well as the lambda function itself (lambda_handler) from the Dockerfile
as "lambda_function.lambda_handler" 
"""
#Import statements
import mlflow #ML experiment tracking

#import modules
import model #model module (Script version of lambda function on AWS console - Data Preprocessing to feature engineering and inference)


#Global Constants
RUN_ID = "dcf13e6b81f740ac8d959d27fb18683c" #ID of ML Model saved on MLFlow
mlflow.set_tracking_uri('http://ec2-54-252-156-209.ap-southeast-2.compute.amazonaws.com:5000') 

#Retrieve the model from the mlflow server on EC2 instance
model_service = model.init(RUN_ID=RUN_ID)

def lambda_handler(event,context):
    """Local version of the Lambda Function that will be invoked when uploaded as an image to ECR and subsequently invoked
    from the lambda function

    Args:
        event: JSON type with 'phone' key as a sample example
        context: required param that is not used
            E.g. of event:
            event = {
                "phone": {
                    "Unnamed: 0": 1.0,
                    "battery_power": 1021,
                    "blue": 1,
                    "clock_speed": 0.5,
                    "dual_sim": 1,
                    "fc": 0,
                    "four_g": 1,
                    "int_memory": 53,
                    "m_dep": 0.7,
                    "mobile_wt": 136,
                    "n_cores": 3,
                    "pc": 6,
                    "px_height": 905,
                    "px_width": 1988,
                    "ram": 2631,
                    "sc_h": 17,
                    "sc_w": 3,
                    "talk_time": 7,
                    "three_g": 1,
                    "touch_screen": 1,
                    "wifi": 0
                }
            }

    Returns: 
        prediction : json with prediction for sample returned
        E.g.:
        {'phone_cat': '2'}
    """
    prediction_json = model_service.lambda_handler(event)
    return prediction_json

#Global Constants
#MODEL_PATH = r"../models" #Path to models 
# logged_model = f"runs:/{RUN_ID}/best_estimator"

# #Load model as a PyFuncModel
# svm_model = mlflow.pyfunc.load_model(logged_model)

# lambda_client = boto3.client('lambda')

# #Global Constants
# MODEL_PATH = r"../models" #Path to models 
# RUN_ID = "dcf13e6b81f740ac8d959d27fb18683c" #ID of ML Model saved on MLFlow
# mlflow.set_tracking_uri('http://ec2-54-252-156-209.ap-southeast-2.compute.amazonaws.com:5000')
# logged_model = f"runs:/{RUN_ID}/best_estimator"

# #Load model as a PyFuncModel
# svm_model = mlflow.pyfunc.load_model(logged_model)

# def prepare_features(feature_sample):
#     """ 
#     Perform data preprocessing and feature engineering on the 1 sample dataframe

#     Args:
#         feature_sample: 1 sample dataframe (raw input data)
#     """
#     #Remove unique identifier column
#     feature_sample.drop(["Unnamed: 0"],axis=1,inplace=True)

#     feature_sample["old"] = 1
#     filter_rows = (feature_sample["wifi"] == 1) & (feature_sample["blue"] == 1) & (feature_sample["dual_sim"] == 1) & (feature_sample["four_g"] == 1) \
#     & (feature_sample["touch_screen"] == 1)
#     feature_sample.loc[filter_rows,"old"] = 0
#     return feature_sample
        
# def predict(features):
#     """ 
#     Perform prediction using the loaded/saved model on the input data

#     Args:
#         features: 1 sample dataframe with data preprocessing and feature engineering performed on the sample
#     """
#     preds = svm_model.predict(pd.DataFrame(features))
#     return preds[0]
#     #return 10

# def sample_test(phone_df):
#     """ Retrieve raw test data and samples 1 row/input data and returns it

#         Returns:
#             feature_sample: raw row/sample of test data in json format
#     """    
#     #Sample from test df
#     feature_sample = phone_df
#     feature_sample['px_height'] = feature_sample['px_height'].astype('float64')
#     feature_sample['sc_w'] = feature_sample['sc_w'].astype('float64')
#     print(type(feature_sample),"this is type of feature sample")
#     print(feature_sample.dtypes,"These are the dtypes")
#     #Convert to json for input to request.post
#     #feature_sample = feature_sample.iloc[0, :].to_json()
#     #Json string with square brackets (Extract the first str json)
#     feature_sample = feature_sample.to_json(orient='records')[1:-1]
#     print(type(feature_sample),"this is type of feature sample")
#     print(feature_sample,"These are the dtypes")
#     return feature_sample


# def lambda_handler(event, context):
#     phone = event["phone"]
#     phone_df = pd.json_normalize(phone)
#     feature_sample = sample_test(phone_df)


#     #Convert to dataframe for preprocessing functions later (e.g. prepare functions and predict)
#     feature_sample = pd.DataFrame(json.loads(feature_sample),index=[0])

#     feature_sample = prepare_features(feature_sample)
#     print(feature_sample)
#     pred = predict(feature_sample)
#     #Prep data as json
#     result = {
#         'phone_cat': str(pred)
#     }

#     print(result)
#     return result
