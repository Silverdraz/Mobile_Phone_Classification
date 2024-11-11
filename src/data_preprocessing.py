"""
Preprocesses the dataframes, such as removing unnecessary identifier(id) columns, perform feature engineering or even conversion of 0s
to np.nan (preprocessing before modelling happens)
"""

#Import statements
import numpy as np #array manipulation
from sklearn.preprocessing import SplineTransformer #spline transformer (non-linearity)
from sklearn.compose import ColumnTransformer #transform specific columns with various pipelines
from sklearn.pipeline import Pipeline #for pipeline 


def remove_columns(combined_df):
    """Remove redundant columns using domain knowledge (unique identifiers are definitely not important in determing price_ranges)

    Args:
        combined_df: List containing both the train_data dataframe and test_data dataframe

    Returns:
        train_data: train dataframe
        test_data: test_dataframe
        E.g. (train_data, test_data) -- combined_df[0], combined_df[1]
    """    
    #Data Preprocessing
    #Proceed to drop columns such as "Unnamed:0" unique identifier/id columns
    for df in combined_df:
        df.drop(["Unnamed: 0"],axis=1,inplace=True)
    return combined_df[0], combined_df[1]



def map_missing_values(combined_x):
    """Map 0 (missing values) to np.nan for sklearn to recognise it as missing values

    Args:
        combined_x: List containing both the x_train dataframe and x_test dataframe

    Returns:
        x_train: x_train dataframe
        x_test: x_test dataframe
        E.g. (x_train, x_test) -- combined_x[0], combined_x[1]
    """      
    #List of columns suspected to have missing values
    cols = ["sc_w","px_height"]
    for x_df in combined_x:
        #Indicate as columns amy be strings or ints
        x_df[cols] = x_df[cols].replace({'0':np.nan, 0:np.nan})  
    return combined_x[0], combined_x[1]


def feature_engineering_old(combine_x):
    """Feature Engineering (Create new features) of old phone model variable

    Args:
        combine_x: List containing both the x_train dataframe and x_test dataframe

    Returns:
        x_train: x_train dataframe
        x_test: x_test dataframe
        E.g. (x_train, x_test) -- combined_x[0], combined_x[1]
    """      
    #Creation of old phone model variable
    for x_df in combine_x:
        #Creation of binary old variable using domain knowledge of variables wifi, bluetooth, 4G, touchscreen, dual_sim)
        x_df["old"] = 1
        filter_rows = (x_df["wifi"] == 1) & (x_df["blue"] == 1) & (x_df["dual_sim"] == 1) & (x_df["four_g"] == 1) \
        & (x_df["touch_screen"] == 1)
        x_df.loc[filter_rows,"old"] = 0
    x_train, x_test = combine_x[0], combine_x[1]
    return x_train, x_test

def feature_engineering_spline():
    """Feature Engineering (Create new features) of splines to model non-linearity

    Returns:
        preprocessor: pipeline component to introduce spline transformation for non-linearity
    """      
    #Create the spline transformation function
    spline_transformer = Pipeline(
        steps=[
            ("spline_transformer", SplineTransformer(degree=2))
        ]
    )    

    #Specify the column to be splined transformed
    spline_columns = ["ram"]
    #Apply the spline transformation only to the ram column. Need to passthrough the rest of the columns else they drop
    preprocessor = ColumnTransformer(
        transformers=[
            ("spline_transform",spline_transformer,spline_columns)
        ],
        remainder='passthrough'
    )
    return preprocessor