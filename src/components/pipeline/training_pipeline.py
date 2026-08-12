from src.logger import logging
from src.exception_handler import CustomMachineLearningException
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
import sys


class TrainingPipeline:
    def initiate_training_pipeline(self):
        try:
            logging.info('***** Intiating the training pipeline *****')
            ingestion_obj = DataIngestion()
            train_path , test_path = ingestion_obj.initiate_data_ingestion()
            print(f'Train Path : {train_path} \n Test Path : {test_path} ')
            
            validation_obj = DataValidation()
            train_df , test_df , _ = validation_obj.initiate_data_validation(train_path,test_path)
            print(f'Train DF : {train_df} \n Test DF : {test_df} ')

        except Exception as e:
            logging.info(f'Exception Occured : {e}')
            raise CustomMachineLearningException(e,sys)


if __name__ == "__main__":
    try:
        train_object = TrainingPipeline()
        train_object.initiate_training_pipeline()
    except Exception as e:
        logging.info(f'Exception Occured : {e}')
        raise CustomMachineLearningException(e,sys)