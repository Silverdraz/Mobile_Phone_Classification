"""
Perform inference/prediction on the test dataset and to prepare the submission file. Save the trained model for inference as 
an API endpoint in models directory.
"""

#Import statements
import pandas as pd #data processing, CSV file I/O (e.g. pd.read_csv)
import os #os file paths
import pickle #Saving and loading models or python objects
from prefect import task, flow #Orchestration Pipeline
import mlflow #mlflow

#Global Constants
DATA_PATH = r"..\data\MobilePriceClassification" #Path to raw data
MODEL_PATH = r"..\models" #Path to models 
TRACKING_SERVER_HOST = "ec2-54-252-156-209.ap-southeast-2.compute.amazonaws.com" #Public DNS to AWS EC2 Instance

#import modules
import mlflow_func #mlflow experiment tracking module
import data_preprocessing #data preprocessing module


@flow(log_prints=True)
def main_flow():
    train_data, test_data = train_test_dfs()
 
    #For applying functions on both dataframes
    combined_df = [train_data,test_data]

    #Preserve the id column for submission file
    x_test_column = test_data["Unnamed: 0"].to_frame()
    x_test_column = x_test_column.rename(columns={'Unnamed: 0': 'id'})

    #Remove unique identifier column
    train_data, test_data = data_preprocessing.remove_columns(combined_df)
    
    #Split into training and test (x & y)
    y_train= train_data["price_range"]
    x_train = train_data.drop(["price_range"],axis=1)
    x_test = test_data

    #Feature Engineering
    combine_x = [x_train,x_test]
    x_train, x_test = data_preprocessing.feature_engineering_old(combine_x)

    #Declare the same mlflow experiment for consolidation
    mlflow.set_experiment("mobile_phone_classification")
    svm_clf = mlflow_func.mlflow_final_model(x_train,y_train)
    #save the model for inference
    save_svm_model(svm_clf)

    #predict, combine
    final_prediction_array = svm_clf.predict(x_test)
    final_prediction_series = pd.Series(final_prediction_array)
    print(final_prediction_series.value_counts())
    x_test_column["price_range"] = final_prediction_series
    x_test_column.to_csv(r"..\submission_file.csv")


@task(retries=3, retry_delay_seconds=2)
def train_test_dfs():
    """ Retrieve the raw train and raw test data

        Returns:
            train_data: raw train dataset
            test_data: raw test dataset
    """    
    train_data = pd.read_csv(os.path.join(DATA_PATH,f"train.csv"))
    test_data = pd.read_csv(os.path.join(DATA_PATH,f"test.csv"))
    return train_data, test_data

def save_svm_model(svm_clf):
    """ Save the trained SVM model for inference

        Args:
            svm_clf: trained SVM classifier model
    """    
    # save
    with open(os.path.join(MODEL_PATH,'model.pkl'),'wb') as f:
        pickle.dump(svm_clf,f)


if __name__ == "__main__":

    main_flow()
    from mlflow.tracking import MlflowClient
    client = MlflowClient(f"http://{TRACKING_SERVER_HOST}:5000")

    client.search_experiments()
    print(client.search_experiments())
    run_id = client.search_runs(experiment_ids='1')[0].info.run_id
    print(run_id,"this is the run id")

    #Create prefect orchestration deployment
    #main_flow.serve(name="mobile_phone_1", cron="* * * * *")

 

