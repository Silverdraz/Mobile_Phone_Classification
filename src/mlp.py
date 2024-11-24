"""
File: mlp.py
------------------------------------
Machine Learning Pipeline --- Stages of 
1. Data Preprocessing + Feature Engineering
2. Model Building
3. Model Evaluation
"""

#Import statements
import pandas as pd #data processing, CSV file I/O (e.g. pd.read_csv)
import os #os file paths
import mlflow #ML experiment tracking

#import modules
import models #models module
import data_preprocessing #data preprocessing module
import mlflow_func #mlflow experiment tracking module


#Global Constants
DATA_PATH = r"..\data\MobilePriceClassification" #Path to raw data

def main():
    #Create experiment for mobile phone classification project
    mlflow.set_experiment("mobile_phone_classification")

    #Retrieve the train and test dataframes
    train_data, test_data = train_test_dfs()
    combined_df = [train_data,test_data]

    train_data, test_data = data_preprocessing.remove_columns(combined_df)
    
    #Split into training and test (x & y)
    y_train= train_data["price_range"]
    x_train = train_data.drop(["price_range"],axis=1)
    x_test = test_data

    #Baseline algorithm using XGBoostClassifier
    models.baseline_algo(x_train,y_train)
    x_train_beforefe = x_train.copy(deep=True)
    x_test_beforefe = x_train.copy(deep=True)
    combine_x_beforefe = [x_train_beforefe,x_test_beforefe]

    combine_x = [x_train,x_test]
    x_train, x_test = data_preprocessing.map_missing_values(combine_x)

    #XGBoost Algo with feature engineered old variable + multiple imputation
    print("Performing evaluation using feature engineered old variable + multiple imputation")
    mlflow_func.mlflow_mi_algo(x_train,y_train)

    #Feature Engineering
    combine_x = [x_train,x_test]
    x_train, x_test = data_preprocessing.feature_engineering_old(combine_x)


    #XGBoost Algo with feature engineered old variable
    print("Performing evaluation using baseline XGBoost + FE")
    models.feature_engineered_algo(x_train,y_train)


    #XGBoost Algo with feature engineered old variable + multiple imputation
    print("Performing evaluation using baseline XGBoost + MI + FE")
    models.mi_fe_algo(x_train,y_train)

    #Perform
    preprocessor = data_preprocessing.feature_engineering_spline()
    models.spline_transformed_algo(preprocessor,x_train,y_train)

    #Map 0 to np.nan for dataset without feature engineering
    x_train_beforefe, x_test_beforefe = data_preprocessing.map_missing_values(combine_x_beforefe)

    #Do model comparisons against the baseline (XGBoost) and baseline + iterative improvements (XGBoost + MI + FE)
    print("Model Comparison in progress")
    models.compare_models(x_train,y_train)


    print("Validating SVM without feature engineering dataset")
    models.chosen_svm(x_train_beforefe,y_train)
    print("Validating SVM on feature engineered old mobile phone variable")
    models.chosen_svm(x_train,y_train)

    print("Validating SVM on standardized dataset")
    models.scale_svm(x_train,y_train)

    print("Validating SVM on standardized and spline transformed dataset")
    models.spline_svm(preprocessor,x_train,y_train)

    print("Performing hyperparameter tuning on svm")
    mlflow_func.mlflow_svm_tuning(x_train,y_train)



def train_test_dfs():
    """ Retrieve the raw train and raw test data

        Returns:
            train_data: raw train dataset
            test_data: raw test dataset
    """    
    train_data = pd.read_csv(os.path.join(DATA_PATH,f"train.csv"))
    test_data = pd.read_csv(os.path.join(DATA_PATH,f"test.csv"))
    return train_data, test_data

if __name__ == "__main__":
    main()
    


