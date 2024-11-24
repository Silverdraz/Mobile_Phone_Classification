"""
Consolidates the mlflow functions here in this python script to enhance modularity of script and decomposition of task by script and functions
"""

#Import statements
import mlflow #ML experiment tracking

#import modules
import models #models module

def mlflow_mi_algo(x_train,y_train):
    """ Log the performance of the model using mlflow. XGBoost is chosen as the model with multiple imputation used instead of using XGBoost
        innate missing value imputation

        Args:
            x_train: features dataframe
            y_train: ground-truth labels
    """
    with mlflow.start_run(run_name = "baseline_feature_engineering_MI"):

        mlflow.set_tag("Developer","Aaron")

        mlflow.log_param("train_val_data_path","data\MobilePriceClassification\train.csv")

        mlflow.log_param("Model","XGBoost with MI with feature engineered Old variable")

        result = models.mi_algo(x_train,y_train)
        mlflow.log_metric("accuracy",result)    

def mlflow_svm_tuning(x_train,y_train):
    """ Log the performance of the model using mlflow. SVM is chosen at the model for prediction/inference. Add-on spline transformation 
        to check if predictive performance is enhanced when non-linearity is considered.

        Standard Scalar is required as spline has high degrees which will non-linearity increase the features distances. SVM uses 
        the features distacnes for classification. Otherwise, results are inaccurate
        Args:
            x_train: features dataframe
            y_train: ground-truth labels   
    """
    extra_tags = {"Developer": "Aaron"}
    mlflow.sklearn.autolog(extra_tags=extra_tags)
    with mlflow.start_run(run_name = "grid_search"):
        models.svm_tuning(x_train,y_train)
    mlflow.autolog(disable=True)

def mlflow_final_model(x_train,y_train):
    """ Log the model as an artifact for loading and inference SVM is chosen at the model for prediction/inference. Using grid search to tune 
        the C parameter, c = 1.1 gives the highest validation accuracy.

        Returns:
            svm_clf: pipeline with MI and parameter that is tuned and trained
    """    
    with mlflow.start_run():
        #Retrieve the final fitted model with tuned parameters (untrained)
        svm_clf = models.final_svm()

        #train, save model
        svm_clf.fit(x_train,y_train)

        #log to mlflow ui
        mlflow.sklearn.log_model(svm_clf, artifact_path="inference_model")
    return svm_clf