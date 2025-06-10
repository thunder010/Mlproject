import sys            # This is used to access system-specific parameters and functions.
from dataclasses import dataclass    # This is a decorator that allows us to define classes that are primarily used to store data.

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer     # This is used to apply different preprocessing steps to different columns in the dataset.
from sklearn.impute import SimpleImputer     # This is used to handle missing values in the dataset.
from sklearn.pipeline import Pipeline         # This is used to create a pipeline of preprocessing steps.
from sklearn.preprocessing import OneHotEncoder,StandardScaler  # This is used to encode categorical variables and scale numerical variables.

from src.exception import CustomException     
from src.logger import logging  
import os           # This is used to handle file paths and directories.

from src.utils import save_object     # This function is used to save the preprocessor object to a file.

@dataclass         
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")   # This is the path where the preprocessor object will be saved.

class DataTransformation:         # This class is responsible for transforming the data by applying various preprocessing steps such as handling missing values, encoding categorical variables, and scaling numerical variables.
    def __init__(self):           # This is the constructor method that initializes the DataTransformation class.
        self.data_transformation_config=DataTransformationConfig()  # This creates an instance of the DataTransformationConfig class, which contains the path for the preprocessor object file.

    def get_data_transformer_object(self):     # This function is responsible for creating a preprocessing pipeline that will be used to transform the data.
        '''
        This function si responsible for data trnasformation
        
        '''
        try:
            numerical_columns = ["writing_score", "reading_score"]  
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())

                ]
            )

            cat_pipeline=Pipeline(

                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
                ]

            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer(           # This is used to apply different preprocessing steps to different columns in the dataset.
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",cat_pipeline,categorical_columns)

                ]


            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):  # This method is responsible for reading the training and test data, applying the preprocessing steps, and saving the transformed data along with the preprocessor object.

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()   # This calls the get_data_transformer_object method to get the preprocessing pipeline.

            target_column_name="math_score"            
            numerical_columns = ["writing_score", "reading_score"]      

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)   
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info("Applying preprocessing object on training dataframe and testing dataframe.")

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)    # This applies the preprocessing steps to the training data.
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)    # This applies the preprocessing steps to the test data.

            train_arr = np.c_[           # This combines the input features and target variable for the training data into a single array to be used for model training as it is required by the model.
                input_feature_train_arr, np.array(target_feature_train_df)   
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]  # This combines the input features and target variable for the test data into a single array to be used for model evaluation.

            logging.info(f"Saved preprocessing object.")

            save_object(                       # This function is used to save the preprocessor object to a file so that it can be reused later without having to retrain the preprocessing steps.

                file_path=self.data_transformation_config.preprocessor_obj_file_path,  # This is the path where the preprocessor object will be saved.
                obj=preprocessing_obj

            )

            return (
                train_arr,       # This returns the transformed training data along with the path to the preprocessor object file.
                test_arr,        # This returns the transformed test data along with the path to the preprocessor object file.
                self.data_transformation_config.preprocessor_obj_file_path,   # This returns the transformed training and test data along with the path to the preprocessor object file.
            )
        except Exception as e:
            raise CustomException(e,sys)