"""
Create models, such as baseline with iterative improvements and perform model comparison/evaluation.
Although some functions are essentially the same, they are named differently so that it is clearer and more explicit when comparing
models at the mlp.py module
"""

#Import statements
import pandas as pd #data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np #array data manipulation
from sklearn.model_selection import KFold #KFold Cross Validation
from sklearn.model_selection import cross_val_score #cross validation score
from sklearn.pipeline import Pipeline #for pipeline 
from sklearn.experimental import enable_iterative_imputer #Mandatory to import experimental feature
from sklearn.impute import IterativeImputer #multiple imputation (missing values)
from sklearn.model_selection import cross_validate #cross validation score
from sklearn.preprocessing import StandardScaler #standardization
from sklearn.model_selection import GridSearchCV #grid search cv
import os #os file path

#Import Models for comparison
from sklearn.linear_model import LogisticRegression #logistic regression
from sklearn import svm #support vector Machine
from sklearn.ensemble import RandomForestClassifier #Random Forest
from sklearn.neighbors import KNeighborsClassifier #KNN
from sklearn.naive_bayes import GaussianNB #Naive bayes
from sklearn.tree import DecisionTreeClassifier #Decision Tree
from xgboost import XGBClassifier #xgboost
from sklearn.ensemble import RandomForestRegressor #RandomForestRegressor

#Global Constants
RESULTS_PATH = r"..\results" #Path to raw data

def baseline_algo(x_train,y_train):
    """XGBoost is chosen as the baseline model using raw features (without feature engineering and without MI)

    Args:
        x_train: features dataframe
        y_train: ground-truth labels
    """    
    #create a baseline algorithm using cv
    baseline_clf = XGBClassifier()
    print("This is the accuracy 1 here")

    #Shuffle the dataset to prevent learning of order
    kfold = KFold(n_splits=5,shuffle=True,random_state=42)
    print(np.mean(cross_val_score(baseline_clf, x_train, y_train, cv=kfold)))

def feature_engineered_algo(x_train,y_train):
    """XGBoost is chosen as the baseline model with feature engineered of old variable

    Args:
        x_train: features dataframe
        y_train: ground-truth labels
    """    
    feature_engineered_clf = XGBClassifier()
    #print(x_train.columns,"this is the columns")


    #Shuffle the dataset to prevent learning of order
    kfold = KFold(n_splits=5,shuffle=True,random_state=42)
    print(np.mean(cross_val_score(feature_engineered_clf, x_train, y_train, cv=kfold)))

def spline_transformed_algo(preprocessor,x_train,y_train):
    """XGBoost is chosen as the baseline model with feature engineered of old variable

    Args:
        preprocessor: Spline transformation pipeline compoenent
        x_train: train dataframe
        y_train: ground-truth labels
    """      
    #create a baseline algorithm using cv
    clf_spline = Pipeline(
        steps=[('preprocessor',preprocessor),("clf", XGBClassifier())]
    )

    #Shuffle the dataset to prevent learning of order
    kfold = KFold(n_splits=5,shuffle=True,random_state=42)
    print(np.mean(cross_val_score(clf_spline, x_train, y_train, cv=kfold)))

def mi_algo(x_train,y_train):
    """ XGBoost is chosen as the model with multiple imputation used instead of using XGBoost innate missing value imputation

        Args:
            x_train: features dataframe
            y_train: ground-truth labels
    """
    #Create random forest model to impute the missign values
    Iterative_imputer_estimator = RandomForestRegressor(
        # We tuned the hyperparameters of the RandomForestRegressor to get a good
        # enough predictive performance for a restricted execution time.
        n_estimators=10,
        max_depth=8,
        bootstrap=True,
        max_samples=0.5,
        n_jobs=2,
        random_state=42,
    )

    #create a pipeline for mi
    mi_clf = Pipeline(
        steps=[('mi', IterativeImputer(estimator=Iterative_imputer_estimator)),("classifier", XGBClassifier())]
    )
    #Shuffle the dataset to prevent learning of order
    kfold = KFold(n_splits=5,shuffle=True,random_state=42)
    print(np.mean(cross_val_score(mi_clf, x_train, y_train, cv=kfold)))


def mi_fe_algo(x_train,y_train):
    """ XGBoost is chosen as the model with multiple imputation used with old feature engineered feature
      instead of using XGBoost innate missing value imputation

        Args:
            x_train: features dataframe
            y_train: ground-truth labels
    """
    #Create random forest model to impute the missign values
    Iterative_imputer_estimator = RandomForestRegressor(
        # We tuned the hyperparameters of the RandomForestRegressor to get a good
        # enough predictive performance for a restricted execution time.
        n_estimators=10,
        max_depth=8,
        bootstrap=True,
        max_samples=0.5,
        n_jobs=2,
        random_state=42,
    )

    #Create pipeline with multiple imputation
    mi_fe_clf = Pipeline(
        steps=[('mi', IterativeImputer(estimator=Iterative_imputer_estimator)),("classifier", XGBClassifier())]
    )

    #Shuffle the dataset to prevent learning of order
    kfold = KFold(n_splits=5,shuffle=True,random_state=42)
    print(np.mean(cross_val_score(mi_fe_clf, x_train, y_train, cv=kfold)))



# Compare the various models for benchmarking  
def compare_models(x_train,y_train):
    """ Comparison among the common classification models to check which model is able to best extract the signals
        and provide the highest predictive performance. The results are saved to model_results.csv.

        At method call, the x_train already has old column feature engineered.

        Args:
            x_train: features dataframe
            y_train: ground-truth labels
    """    
    cv_validation_mean = []
    cv_train_mean = []
    cv_std = []
    #For index of dataframe created later to store the results
    classifier_names = ['Radial Svm','Logistic Regression','KNN','Decision Tree',
                        'Naive Bayes','Random Forest','XGBoost']
    models=[svm.SVC(kernel='rbf'),LogisticRegression(),KNeighborsClassifier(),
            DecisionTreeClassifier(),GaussianNB(),RandomForestClassifier(),XGBClassifier()]
    #Iterate through the different models
    for model in models:
        Iterative_imputer_estimator =  RandomForestRegressor(
            # Tune the hyperparameters of the RandomForestRegressor to get a good
            # enough predictive performance for a restricted execution time.
            n_estimators=10,
            max_depth=8,
            bootstrap=True,
            max_samples=0.5,
            n_jobs=2,
            random_state=0,
        )
        pipe = Pipeline(
                    steps=[('mi', IterativeImputer(estimator=Iterative_imputer_estimator,max_iter=10)),("classifier", model)]
                )
        #Shuffle the dataset to prevent learning of order
        kfold = KFold(n_splits=5,shuffle=True,random_state=42)        
        results = cross_validate(pipe,x_train,y_train,cv=kfold,return_train_score=True)
        #Store the results train, test score to check for overfitting
        cv_train_mean.append(np.mean(results["train_score"]))
        cv_validation_mean.append(np.mean(results["test_score"]))
        cv_std.append(np.std(results["test_score"]))
    #Create the dataframe
    results_df = pd.DataFrame({'mean train accuracy': cv_train_mean,'mean validation accuracy':cv_validation_mean,
                               'std accuracy':cv_std},index=classifier_names)
    #Save the results for reference comparison
    results_df.to_csv(os.path.join(RESULTS_PATH,f"model_results.csv"))
    

def chosen_svm(x_train,y_train):
    """ SVM is chosen at the model for prediction/inference as it has the best validation accuracy with much
        lower overfitting than the tree-based models (from the model_results.csv)

        Args:
            x_train: features dataframe
            y_train: ground-truth labels
    """    
    Iterative_imputer_estimator = RandomForestRegressor(
        # We tuned the hyperparameters of the RandomForestRegressor to get a good
        # enough predictive performance for a restricted execution time.
        n_estimators=10,
        max_depth=8,
        bootstrap=True,
        max_samples=0.5,
        n_jobs=2,
        random_state=42,
    )

    #create a pipeline for selected svm
    svm_clf = Pipeline(
        steps=[('mi', IterativeImputer(estimator=Iterative_imputer_estimator)),("classifier", svm.SVC(kernel='rbf'))]
    )

    #Shuffle the dataset to prevent learning of order
    kfold = KFold(n_splits=5,shuffle=True,random_state=42)
    print(np.mean(cross_val_score(svm_clf, x_train, y_train, cv=kfold)))

def scale_svm(x_train,y_train):
    """ SVM is chosen at the model for prediction/inference. Add-on standardisation

        Standard Scalar is required as svm considers distance for it's classification or prediction. 
        Args:
            x_train: features dataframe
            y_train: ground-truth labels
    """    
    Iterative_imputer_estimator = RandomForestRegressor(
        # We tuned the hyperparameters of the RandomForestRegressor to get a good
        # enough predictive performance for a restricted execution time.
        n_estimators=10,
        max_depth=8,
        bootstrap=True,
        max_samples=0.5,
        n_jobs=2,
        random_state=42,
    )

    #create a baseline algorithm using cv
    scale_svm = Pipeline(
        steps=[('mi', IterativeImputer(estimator=Iterative_imputer_estimator)),("scalar",StandardScaler()),("classifier", svm.SVC(kernel='rbf'))]
    )
    #Shuffle the dataset to prevent learning of order
    kfold = KFold(n_splits=5,shuffle=True,random_state=42)
    print(np.mean(cross_val_score(scale_svm, x_train, y_train, cv=kfold)))


def spline_svm(preprocessor,x_train,y_train):
    """ SVM is chosen at the model for prediction/inference. Add-on spline transformation to check if predictive performance is 
        enhanced when non-linearity is considered.

        Standard Scalar is required as spline has high degrees which will non-linearity increase the features distances. SVM uses 
        the features distacnes for classification. Otherwise, results are inaccurate
        Args:
            x_train: features dataframe
            y_train: ground-truth labels
    """    
    Iterative_imputer_estimator = RandomForestRegressor(
        # We tuned the hyperparameters of the RandomForestRegressor to get a good
        # enough predictive performance for a restricted execution time.
        n_estimators=10,
        max_depth=8,
        bootstrap=True,
        max_samples=0.5,
        n_jobs=2,
        random_state=42,
    )

    #create a standard scalar and spline transformation pipeline
    svm_clf = Pipeline(
        steps=[('preprocessor',preprocessor), ('mi', IterativeImputer(estimator=Iterative_imputer_estimator)),
               ("scale",StandardScaler()), ("classifier", svm.SVC(kernel='rbf'))]
    )
    #Shuffle the dataset to prevent learning of order
    kfold = KFold(n_splits=5,shuffle=True,random_state=42)
    print(np.mean(cross_val_score(svm_clf, x_train, y_train, cv=kfold)))


#Tune the model that has the highest accuracy
def svm_tuning(x_train,y_train):
    """ SVM is chosen at the model for prediction/inference. Add-on spline transformation to check if predictive performance is 
        enhanced when non-linearity is considered.

        Standard Scalar is required as spline has high degrees which will non-linearity increase the features distances. SVM uses 
        the features distacnes for classification. Otherwise, results are inaccurate
        Args:
            x_train: features dataframe
            y_train: ground-truth labels
    """   
    Iterative_imputer_estimator =  RandomForestRegressor(
            # Tune the hyperparameters of the RandomForestRegressor to get a good
            # enough predictive performance for a restricted execution time.
            n_estimators=10,
            max_depth=8,
            bootstrap=True,
            max_samples=0.5,
            n_jobs=2,
            random_state=0,
        )
    #Shuffle the dataset to prevent learning of order
    kfold = KFold(n_splits=5,shuffle=True,random_state=42)
    pipe = Pipeline(
                    steps=[('mi', IterativeImputer(estimator=Iterative_imputer_estimator)),
                           ('svm', svm.SVC(kernel='rbf'))]
            )
    parameters = {'svm__C':[0.7,0.8,0.9,0.95,1,1.1,1.2]}
    grid = GridSearchCV(estimator=pipe,param_grid=parameters,cv=kfold)
    grid.fit(x_train,y_train)
    print("This is the best score")
    print(grid.best_score_)
    print("This is the best params")
    print(grid.best_params_)
    print(grid.cv_results_)



def final_svm():
    """ SVM is chosen at the model for prediction/inference. Using grid search to tune the C parameter,
        c = 1.1 gives the highest validation accuracy.

        Returns:
            svm_clf: pipeline with MI and parameter that is tuned
    """    
    Iterative_imputer_estimator = RandomForestRegressor(
        # We tuned the hyperparameters of the RandomForestRegressor to get a good
        # enough predictive performance for a restricted execution time.
        n_estimators=10,
        max_depth=8,
        bootstrap=True,
        max_samples=0.5,
        n_jobs=2,
        random_state=42,
    )

    #create a pipeline for selected svm
    svm_clf = Pipeline(
        steps=[('mi', IterativeImputer(estimator=Iterative_imputer_estimator)),("classifier", svm.SVC(kernel='rbf',C=1.1))]
    )

    return svm_clf

