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

#import modules
import models #models module
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
    mlflow.set_experiment("mobile_phone_classification")
    with mlflow.start_run():
        #Retrieve the final fitted model with tuned parameters (untrained)
        svm_clf = models.final_svm()

        #train, save model
        svm_clf.fit(x_train,y_train)
        save_svm_model(svm_clf)

        #log to mlflow ui
        mlflow.sklearn.log_model(svm_clf, artifact_path="inference_model")

    #predict, combine
    final_prediction_array = svm_clf.predict(x_test)
    final_prediction_series = pd.Series(final_prediction_array)
    print(final_prediction_series.value_counts())
    x_test_column["price_range"] = final_prediction_series
    x_test_column.to_csv(r"..\submission_file.csv")
    return svm_clf


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

    #main_flow.serve(name="mobile_phone_1", cron="* * * * *")

    # flow.from_source(
    #     source="https://github.com/Silverdraz/Mobile_Phone_Classification.git",
    #     entrypoint=r"src/test.py:main_flow",
    # ).deploy(name="mobile_phone_1",
    #     work_pool_name="mobile_phone_classification",
    #     push=True,
    #     cron="* * * * *")
    # main_flow.deploy(
    #     name="mobile_phone_1",
    #     work_pool_name="mobile_phone_classification",
    #     push=True,
    #     cron="* * * * *",
    # )
    # flow.from_source(
    #         source="https://github.com/Silverdraz/Mobile_Phone_Classification.git",
    #         entrypoint="src\test.py:main_flow",
    # ).deploy(name="mobile_phone_1",
    #     work_pool_name="mobile_phone_classification",
    #     cron="* * * * *")
    
    # flow.from_source(
    #     source="https://github.com/Silverdraz/Mobile_Phone_Classification.git",
    #     entrypoint="src/test.py:main_flow",
    # ).serve(name="my-first-deployment", cron="* * * * *")
 

