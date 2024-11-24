import predict
import pandas as pd
import os
import data_preprocessing
import requests
import json

DATA_PATH = r"..\data\MobilePriceClassification" #Path to raw data

test_data = pd.read_csv(os.path.join(DATA_PATH,f"test.csv"))
feature_sample = test_data.sample(n=1)
feature_sample = feature_sample.iloc[0, :].to_json()
print(feature_sample,"this is the type")

url = "http://localhost:9696/"
response = requests.post(url, json=feature_sample)
print(int(response.json()["phone_cat"]))


