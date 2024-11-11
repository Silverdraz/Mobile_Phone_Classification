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
    print("Performing evaluation using baseline XGBoost")
    models.mi_algo(x_train,y_train)

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
    models.svm_tuning(x_train,y_train)



# # Create baseline as a benchmark with conversion of strings to categorical numbers. Features are raw without 
# def baseline_algo(x_train,y_train):
#     """ XGBoost is chosen as the baseline model using raw features (without feature engineering)"""
#     #create a baseline algorithm using cv
#     baseline_clf = XGBClassifier()
#     print("This is the accuracy 1")

#     #Shuffle the dataset to prevent learning of order
#     kfold = KFold(n_splits=5,shuffle=True,random_state=42)
#     print(np.mean(cross_val_score(baseline_clf, x_train, y_train, cv=kfold)))

# #Create new features for the model after baseline evaluation
# def feature_engineering(combine_x):
#     #Feature Engineering
#     #Creation of binary Travel Companion - Family (Follow other features naming convention for Travel Companion - {})
#     for x_df in combine_x:
#         #Creation of binary old variable using domain knowledge of variables wifi, bluetooth, 4G, touchscreen, dual_sim)
#         x_df["old"] = 1
#         filter_rows = (x_df["wifi"] == 1) & (x_df["blue"] == 1) \
#         & (x_df["dual_sim"] == 1) & (x_df["four_g"] == 1) \
#         & (x_df["touch_screen"] == 1)
#         x_df.loc[filter_rows,"old"] = 0
#         x_df["old"].value_counts()
#     x_train, x_test = combine_x[0], combine_x[1]
#     return x_train, x_test

# def feature_engineered_algo(x_train,y_train):
#     """ XGBoost is chosen as the model with iterative improvements especially creating new features (Improved CV Score)"""
#     feature_engineered_clf = XGBClassifier()

#     print("This is the accuracy XGBoost + feature engineered of old feature")

#     #Shuffle the dataset to prevent learning of order
#     kfold = KFold(n_splits=5,shuffle=True,random_state=42)
#     print(np.mean(cross_val_score(feature_engineered_clf, x_train, y_train, cv=kfold)))

# def mi_algo(preprocessor,x_train,y_train):
#     """ XGBoost is chosen as the model with new features and with multiple imputation (Slightly lower CV score than 
#         when tree-base models impute missing values innately)"""
#     Iterative_imputer_estimator = RandomForestRegressor(
#         # We tuned the hyperparameters of the RandomForestRegressor to get a good
#         # enough predictive performance for a restricted execution time.
#         n_estimators=10,
#         max_depth=8,
#         bootstrap=True,
#         max_samples=0.5,
#         n_jobs=2,
#         random_state=42,
#     )

#     #create a baseline algorithm using cv
#     mi_clf = Pipeline(
#         steps=[("preprocessor",preprocessor),('mi', IterativeImputer(estimator=Iterative_imputer_estimator)),("classifier", XGBClassifier())]
#     )
#     print("This is the accuracy 3 in mi alog")
#     #Shuffle the dataset to prevent learning of order
#     kfold = KFold(n_splits=5,shuffle=True,random_state=42)
#     print(np.mean(cross_val_score(mi_clf, x_train, y_train, cv=kfold)))

# # Compare the various models for benchmarking  
# def compare_models(x_train,y_train):
#     """Randomforestclassiifer is has superior performance compared to the rest"""
#     cv_validation_mean = []
#     cv_train_mean = []
#     cv_std = []
#     classifier_names = ['Radial Svm','Logistic Regression','KNN','Decision Tree',
#                         'Naive Bayes','Random Forest','XGBoost']
#     models=[svm.SVC(kernel='rbf'),LogisticRegression(),KNeighborsClassifier(),
#             DecisionTreeClassifier(),GaussianNB(),RandomForestClassifier(),XGBClassifier()]
#     for model in models:
#         Iterative_imputer_estimator =  RandomForestRegressor(
#             # Tune the hyperparameters of the RandomForestRegressor to get a good
#             # enough predictive performance for a restricted execution time.
#             n_estimators=10,
#             max_depth=8,
#             bootstrap=True,
#             max_samples=0.5,
#             n_jobs=2,
#             random_state=0,
#         )
#         pipe = Pipeline(
#                     steps=[('mi', IterativeImputer(estimator=Iterative_imputer_estimator,max_iter=10)),("classifier", model)]
#                 )
#         #Shuffle the dataset to prevent learning of order
#         kfold = KFold(n_splits=5,shuffle=True,random_state=42)        
#         results = cross_validate(pipe,x_train,y_train,cv=kfold,return_train_score=True)
#         print(results,"this is results")
#         cv_train_mean.append(np.mean(results["train_score"]))
#         cv_validation_mean.append(np.mean(results["test_score"]))
#         cv_std.append(np.std(results["test_score"]))
#     results_df = pd.DataFrame({'mean train accuracy': cv_train_mean,'mean validation accuracy':cv_validation_mean,'std accuracy':cv_std},
#                               index=classifier_names)
#     results_df.to_csv("model_results.csv")

# def chosen_algo(x_train,y_train):
#     """ XGBoost is chosen as the model with new features and with multiple imputation (Slightly lower CV score than 
#         when tree-base models impute missing values innately)"""
#     Iterative_imputer_estimator = RandomForestRegressor(
#         # We tuned the hyperparameters of the RandomForestRegressor to get a good
#         # enough predictive performance for a restricted execution time.
#         n_estimators=10,
#         max_depth=8,
#         bootstrap=True,
#         max_samples=0.5,
#         n_jobs=2,
#         random_state=42,
#     )

#     #create a baseline algorithm using cv
#     mi_clf = Pipeline(
#         steps=[('mi', IterativeImputer(estimator=Iterative_imputer_estimator)),("classifier", svm.SVC(kernel='rbf'))]
#     )
#     print("This is the radial svm selected algo")
#     #Shuffle the dataset to prevent learning of order
#     kfold = KFold(n_splits=5,shuffle=True,random_state=42)
#     print(np.mean(cross_val_score(mi_clf, x_train, y_train, cv=kfold)))

# def spline_chosen_algo(preprocessor,x_train,y_train):
#     """ XGBoost is chosen as the model with new features and with multiple imputation (Slightly lower CV score than 
#         when tree-base models impute missing values innately)"""
#     Iterative_imputer_estimator = RandomForestRegressor(
#         # We tuned the hyperparameters of the RandomForestRegressor to get a good
#         # enough predictive performance for a restricted execution time.
#         n_estimators=10,
#         max_depth=8,
#         bootstrap=True,
#         max_samples=0.5,
#         n_jobs=2,
#         random_state=42,
#     )

#     #create a baseline algorithm using cv
#     mi_clf = Pipeline(
#         steps=[('preprocessor',preprocessor),('mi', IterativeImputer(estimator=Iterative_imputer_estimator)),("classifier", svm.SVC(kernel='rbf'))]
#     )
#     print("This is the spline radial svm selected algo")
#     #Shuffle the dataset to prevent learning of order
#     kfold = KFold(n_splits=5,shuffle=True,random_state=42)
#     print(np.mean(cross_val_score(mi_clf, x_train, y_train, cv=kfold)))

# def scale_svm(x_train,y_train):
#     """ XGBoost is chosen as the model with new features and with multiple imputation (Slightly lower CV score than 
#         when tree-base models impute missing values innately)"""
#     Iterative_imputer_estimator = RandomForestRegressor(
#         # We tuned the hyperparameters of the RandomForestRegressor to get a good
#         # enough predictive performance for a restricted execution time.
#         n_estimators=10,
#         max_depth=8,
#         bootstrap=True,
#         max_samples=0.5,
#         n_jobs=2,
#         random_state=42,
#     )

#     #create a baseline algorithm using cv
#     scale_svm = Pipeline(
#         steps=[("scalar",StandardScaler()),('mi', IterativeImputer(estimator=Iterative_imputer_estimator)),("classifier", svm.SVC(kernel='rbf'))]
#     )
#     print("This is the scaled radial svm selected algo")
#     #Shuffle the dataset to prevent learning of order
#     kfold = KFold(n_splits=5,shuffle=True,random_state=42)
#     print(np.mean(cross_val_score(scale_svm, x_train, y_train, cv=kfold)))



# #Tune the model that has the highest accuracy
# def svm_tuning(x_train,y_train):
#     """ Grid-Search Hyperparameter Tuning for random forest using n_estimators"""
#     Iterative_imputer_estimator =  RandomForestRegressor(
#             # Tune the hyperparameters of the RandomForestRegressor to get a good
#             # enough predictive performance for a restricted execution time.
#             n_estimators=10,
#             max_depth=8,
#             bootstrap=True,
#             max_samples=0.5,
#             n_jobs=2,
#             random_state=0,
#         )
#     #Shuffle the dataset to prevent learning of order
#     kfold = KFold(n_splits=5,shuffle=True,random_state=42)
#     pipe = Pipeline(
#                     steps=[('mi', IterativeImputer(estimator=Iterative_imputer_estimator)),
#                            ('svm', svm.SVC(kernel='rbf'))]
#             )
#     parameters = {'svm__C':[0.7,0.8,0.9,0.95,1,1.3]}
#     grid = GridSearchCV(estimator=pipe,param_grid=parameters,cv=kfold)
#     grid.fit(x_train,y_train)
#     print("This is the best score")
#     print(grid.best_score_)
#     print("This is the best params")
#     print(grid.best_params_)
#     print(grid.cv_results_)


#Retrieve the data and split into train and test split
def train_test_dfs():
    # Retrieve the raw train and raw test data
    train_data = pd.read_csv(os.path.join(DATA_PATH,f"train.csv"))
    test_data = pd.read_csv(os.path.join(DATA_PATH,f"test.csv"))
    return train_data, test_data


if __name__ == "__main__":
    main()
    


