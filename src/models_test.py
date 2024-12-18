import predict
import pandas as pd
import os
import json
from pandas.testing import assert_frame_equal
import model

DATA_PATH = r"../data/MobilePriceClassification" #Path to raw data

def test_prepare_features():
    model_service = model.ModelService(None)
    #Retrieve test dataframe
    test_data = pd.read_csv(os.path.join(DATA_PATH,f"test.csv"))
    test_sample = test_data.iloc[0, :].to_json()
    phone = pd.DataFrame(json.loads(test_sample),index=[0])
    print(phone)

    actual_features = model_service.prepare_features(phone)
    print(actual_features.to_json())

    phone = {
         "battery_power": 1021.0,
         "blue": 1.0,
         "clock_speed": 0.5,
         "dual_sim": 1.0,
         "fc": 0.0,
         "four_g": 1.0,
         "int_memory": 53.0,
         "m_dep": 0.7,
         "mobile_wt": 136.0,
         "n_cores": 3.0,
         "pc": 6.0,
         "px_height": 905.0,
         "px_width": 1988.0,
         "ram": 2631.0,
         "sc_h": 17.0,
         "sc_w": 3.0,
         "talk_time": 7.0,
         "three_g": 1.0,
         "touch_screen": 1.0,
         "wifi": 0.0,
         "old": 1,
    }
    expected_result = pd.DataFrame(phone,index=[0])
    assert_frame_equal(actual_features,expected_result)


class ModelMock():

    def __init__(self,value):
        self.value = value

    def predict(self,features):
        n = len(features)
        return [self.value] * n


def test_predict():
    model_mock = ModelMock(10)
    model_service = model.ModelService(model_mock)

    phone = {
         "battery_power": 1021.0,
         "blue": 1.0,
         "clock_speed": 0.5,
         "dual_sim": 1.0,
         "fc": 0.0,
         "four_g": 1.0,
         "int_memory": 53.0,
         "m_dep": 0.7,
         "mobile_wt": 136.0,
         "n_cores": 3.0,
         "pc": 6.0,
         "px_height": 905.0,
         "px_width": 1988.0,
         "ram": 2631.0,
         "sc_h": 17.0,
         "sc_w": 3.0,
         "talk_time": 7.0,
         "three_g": 1.0,
         "touch_screen": 1.0,
         "wifi": 0.0,
         "old": 1,
    }
    phone = pd.DataFrame(phone,index=[0])
    actual_prediction = model_service.predict(phone)
    expected_prediction = 10.0
    assert actual_prediction == expected_prediction


def test_lambda_handler():
    model_mock = ModelMock(10)
    model_service = model.ModelService(model_mock)

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
    actual_prediction = model_service.lambda_handler(event=event)
    expected_prediction = {
        'phone_cat': '10'
     }

    assert actual_prediction == expected_prediction

test_prepare_features()
test_predict()
test_lambda_handler()