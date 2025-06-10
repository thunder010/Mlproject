import os
import sys

import numpy as np 
import pandas as pd
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):            # This function is used to save the preprocessor object to a file.
    try:
        dir_path = os.path.dirname(file_path)     # Get the directory path from the file path

        os.makedirs(dir_path, exist_ok=True)      # Create the directory if it does not exist

        with open(file_path, "wb") as file_obj:    # Open the file in write-binary mode
            dill.dump(obj, file_obj)               # Use dill to serialize the object and save it to the file, serialize means converting the object into a byte stream that can be saved to a file.
 
    except Exception as e:
        raise CustomException(e, sys)       
    
def evaluate_models(X_train, y_train,X_test,y_test,models,param):    # This function evaluates multiple machine learning models using GridSearchCV for hyperparameter tuning and returns their performance scores.
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)          # Initialize GridSearchCV with the model and parameters, cv=3 means 3-fold cross-validation
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)            # Set the best parameters found by GridSearchCV to the model
            model.fit(X_train,y_train)                 # Fit the model to the training data

            y_train_pred = model.predict(X_train)        # Predict the target variable for the training data

            y_test_pred = model.predict(X_test)         # Predict the target variable for the test data

            train_model_score = r2_score(y_train, y_train_pred)      # Calculate the R-squared score for the training predictions

            test_model_score = r2_score(y_test, y_test_pred)       # Calculate the R-squared score for the test predictions
 
            report[list(models.keys())[i]] = test_model_score       # Store the test score in the report dictionary with the model name as the key

        return report

    except Exception as e:              # This function raises a custom exception if there is an error during the evaluation process, passing the error and the system information.
        raise CustomException(e, sys)
    
def load_object(file_path):                     # This function is used to load a preprocessor object from a file which was saved using the save_object function.
    try:
        with open(file_path, "rb") as file_obj:  # Open the file in read-binary mode
            return dill.load(file_obj)            # Use dill to deserialize the object from the file and return it

    except Exception as e:
        raise CustomException(e, sys)