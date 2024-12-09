from ..src import predict
import pandas as pd
import os
import json
from pandas.testing import assert_frame_equal

DATA_PATH = r"..\data\MobilePriceClassification" #Path to raw data

def test_prepare_features():
    #Retrieve test dataframe
    test_data = pd.read_csv(os.path.join(DATA_PATH,f"test.csv"))
    test_sample = test_data.iloc[0, :].to_json()
    phone = pd.DataFrame(json.loads(test_sample),index=[0])
    print(phone)

    actual_features = predict.prepare_features(phone)
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



test_prepare_features()