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
import numpy as np #array data manipulation
from sklearn.model_selection import KFold #KFold Cross Validation
from sklearn.model_selection import cross_val_score #cross validation score
from sklearn.model_selection import cross_validate #cross validation score
from sklearn.pipeline import Pipeline #for pipeline 
from sklearn.experimental import enable_iterative_imputer #Mandatory to import experimental feature
from sklearn.impute import IterativeImputer #multiple imputation (missing values)
from sklearn.preprocessing import SplineTransformer #spline transformer (non-linearity)
from sklearn.compose import ColumnTransformer #transform specific columns with various pipelines
from sklearn.model_selection import GridSearchCV #grid search cv
from sklearn.preprocessing import StandardScaler #standardization
import mlflow

#Import Models for comparison
from sklearn.linear_model import LogisticRegression #logistic regression
from sklearn import svm #support vector Machine
from sklearn.ensemble import RandomForestClassifier #Random Forest
from sklearn.neighbors import KNeighborsClassifier #KNN
from sklearn.naive_bayes import GaussianNB #Naive bayes
from sklearn.tree import DecisionTreeClassifier #Decision Tree
from xgboost import XGBClassifier #xgboost
from sklearn.ensemble import RandomForestRegressor #RandomForestRegressor

#import modules
import models #models module
import data_preprocessing #data preprocessing module


#Global Constants
DATA_PATH = r"..\data\MobilePriceClassification" #Path to raw data

def main():
    #Set MlFlow tracking sever URI
    print("This is the start of main")
    #mlflow.set_tracking_uri("https://my-tracking-server:5000")
    mlflow.set_experiment("mobile_phone_classification")
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
    with mlflow.start_run(run_name = "grid_search"):

        mlflow.set_tag("Developer","Aaron")

        mlflow.log_param("train_val_data_path","data\MobilePriceClassification\train.csv")

        mlflow.log_param("Model","XGBoost with MI with feature engineered Old variable")

        result = models.mi_algo(x_train,y_train)
        mlflow.log_metric("accuracy",result)

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
    extra_tags = {"Developer": "Aaron"}
    mlflow.sklearn.autolog(extra_tags=extra_tags)
    with mlflow.start_run(run_name = "grid_search"):
        models.svm_tuning(x_train,y_train)
        mlflow.log_artifact
        #grid = models.svm_tuning(x_train,y_train)
        # mlflow.set_tag("Developer","Aaron")

        # # log the best estimator fround by grid search in the outer mlflow run 
        # for k_param in grid.best_params_.keys():
        #     mlflow.log_param(k_param, grid.best_params_[k_param])
        
        # mlflow.log_metric("accuracy",grid.best_score_)
        # mlflow.sklearn.log_model(grid.best_estimator_, 'best_svm_model')
    mlflow.autolog(disable=True)




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
    


