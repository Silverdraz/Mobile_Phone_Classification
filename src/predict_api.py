"""
Send request to API endpoint
"""

import pandas as pd #pandas dataframe
import os #path
import requests #API Endpoint/Server


#Global constant
DATA_PATH = r"..\data\MobilePriceClassification" #Path to raw data
url = "http://localhost:9696/" #url


def main():
    feature_sample = sample_test()
    #send the request to the API endpoint and retrieve the result
    response = requests.post(url, json=feature_sample)
    #retrieve the json response object
    print(int(response.json()["phone_cat"]))

def sample_test():
    """ Retrieve raw test data and samples 1 row/input data and returns it

        Returns:
            feature_sample: raw row/sample of test data in json format
    """    
    #Retrieve test dataframe
    test_data = pd.read_csv(os.path.join(DATA_PATH,f"test.csv"))
    #Sample from test df
    feature_sample = test_data.sample(n=1)
    #Convert to json for input to request.post
    feature_sample = feature_sample.iloc[0, :].to_json()
    return feature_sample

if __name__ == "__main__":
    main()
    