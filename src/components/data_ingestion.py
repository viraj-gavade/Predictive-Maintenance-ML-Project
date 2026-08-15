from src.exception_handler import CustomMachineLearningException
from src.logger import logging
from dataclasses import dataclass
import os 
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

@dataclass
class DataIngestionConfig:
    train_data_path : str = os.path.join('Artifacts/Data','train.csv')
    test_data_path : str = os.path.join('Artifacts/Data','test.csv')
    raw_data_path : str = os.path.join('Artifacts/Data','raw.csv')



class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()


    def initiate_data_ingestion(self):
        try:
            logging.info('***** PHASE 1 - Initiate Data Ingestion Pipeline *****')
            logging.info('Loading the data from the source')
            df = pd.read_csv('data\\ai4i2020.csv')
            logging.info('DataSet Loaded successfully !')

            logging.info('Creating the artifcats folders for paths')
            os.makedirs(os.path.dirname(self.data_ingestion_config.train_data_path),exist_ok=True)
            logging.info('Artifcats Folder created successfully')

            logging.info('Applying the train-test-split to split the data')
            train_df , test_df = train_test_split(df,test_size=0.2,random_state=42)
            logging.info('Train Test split applied successfully')
            logging.info(f'Train Shape : {train_df.shape} , Test Shape : {test_df.shape}')

            logging.info('Saving the raw data to artifacts folder')
            df.to_csv(self.data_ingestion_config.raw_data_path,index=False)
            logging.info('Raw data saved to artifcats folder successfully')

            logging.info('Saving the train data to artifacts folder')
            train_df.to_csv(self.data_ingestion_config.train_data_path,index=False)
            logging.info('Train data saved to artifcats folder successfully')


            logging.info('Saving the test data to artifacts folder')
            test_df.to_csv(self.data_ingestion_config.test_data_path,index=False)
            logging.info('Test data saved to artifcats folder successfully')

            logging.info('*****  PHASE 1 - Data Ingestion Pipeline completed Successfully *****')
            return(
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path,
            )
        except Exception as e:
            logging.info(f'Exception Occured : {e}')
            raise CustomMachineLearningException(e,sys)