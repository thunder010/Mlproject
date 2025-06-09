import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer
@dataclass                                                   # This is a decorator that allows us to define classes that are primarily used to store data.
class DataIngestionConfig:                              # This class is used to define the configuration for data ingestion.
    train_data_path: str=os.path.join('artifacts',"train.csv")  # This is the path where the training data will be stored.
    test_data_path: str=os.path.join('artifacts',"test.csv")    # This is the path where the test data will be stored.
    raw_data_path: str=os.path.join('artifacts',"data.csv")     # This is the path where the raw data will be stored.

class DataIngestion:                  # This class is responsible for ingesting data from a source (in this case, a CSV file) and splitting it into training and test datasets.
    def __init__(self):             # This is the constructor method that initializes the DataIngestion class.
        self.ingestion_config=DataIngestionConfig()    # This creates an instance of the DataIngestionConfig class, which contains the paths for the train, test, and raw data.

    def initiate_data_ingestion(self):      # This method is responsible for reading the data from the CSV file, splitting it into training and test datasets, and saving them to the specified paths.
        logging.info("Entered the data ingestion method or component")  # This logs the entry into the data ingestion method.
        try:
            df=pd.read_csv('notebook/data/stud.csv')
            logging.info('Read the dataset as dataframe')    # This logs that the dataset has been read into a pandas DataFrame.

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)  # This creates the directory for the train data path if it does not already exist.

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)     # This saves the raw data to the specified path without the index and with headers.

            logging.info("Train test split initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)    # This splits the DataFrame into training and test sets, with 20% of the data going to the test set and a random state for reproducibility.

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)   # This saves the training set to the specified path without the index and with headers.

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)      # This saves the test set to the specified path without the index and with headers.

            logging.info("Inmgestion of the data iss completed")

            return(
                self.ingestion_config.train_data_path,     # This returns the path to the training data.
                self.ingestion_config.test_data_path       # This returns the path to the test data.

            )
        except Exception as e:     
            raise CustomException(e,sys)    # This raises a custom exception if there is an error during the data ingestion process, passing the error and the system information.
        
if __name__=="__main__":               # This block is executed when the script is run directly and not when it is imported as a module.
    obj=DataIngestion()              # This creates an instance of the DataIngestion class.
    train_data,test_data=obj.initiate_data_ingestion()    # This calls the initiate_data_ingestion method to perform the data ingestion process.
