import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models    # This imports the save_object and evaluate_models functions from the src.utils module.

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")    # This is the path where the trained model will be saved as a pickle file.

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()   # This creates an instance of the ModelTrainerConfig class, which contains the path for the trained model file.


    def initiate_model_trainer(self,train_array,test_array):       # This method is responsible for training the model using the training data and evaluating its performance on the test data.
        try:
            logging.info("Split training and test input data")      
            X_train,y_train,X_test,y_test=(                        # This splits the training and test arrays into features (X) and target variable (y).
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {                                          # This is a dictionary that contains different regression models that will be evaluated.
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            params={                                                 # This is a dictionary that contains hyperparameters for each model that will be used for tuning.
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'splitter':['best','random'],
                    'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    'criterion':['squared_error', 'friedman_mse'],
                    'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,   # This calls the evaluate_models function to evaluate the performance of each model using the training and test data.
                                             models=models,param=params)
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))        # This finds the maximum score from the model report dictionary, which contains the performance scores of each model.

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[            
                list(model_report.values()).index(best_model_score)     # This finds the name of the model that has the best score by looking up the index of the best score in the values of the model report dictionary.
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,   # This saves the best model to the specified file path.
                obj=best_model
            )

            predicted=best_model.predict(X_test)        # This uses the best model to make predictions on the test data.
 
            r2_square = r2_score(y_test, predicted)          # This calculates the R-squared score of the predictions, which is a measure of how well the model explains the variance in the target variable.
            return r2_square
            



            
        except Exception as e:
            raise CustomException(e,sys)