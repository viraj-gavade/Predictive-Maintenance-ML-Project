from src.exception_handler import CustomMachineLearningException
from src.logger import logging
from dataclasses import dataclass ,field
import os 
import sys
import pandas as pd
from pydantic import BaseModel
from typing import Dict , List
from typing import ClassVar
from imblearn.over_sampling import SMOTE


@dataclass 
class ModelTrainingConfig:
    best_base_model_path : str = os.path.join('Artifacts/Models','best_base_model.pkl')
    best_tunned_model_path : str = os.path.join('Artifacts/Models','best_tunned_model.pkl')


class ModelTrainer:
    def __int__(self):
        self.model_trainer_config = ModelTrainingConfig()


    def handle_class_imbalance(self , X_train_transformed,y_train,X_test_transformed,y_test):
        try:
            logging.info('Handling the class imbalance On train data')
            logging.info(f'Y-train Value Counts Before SMOTE  : {y_train.value_counts()}')
            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train_transformed,y_train)
            logging.info(f'Y-train Value Counts After SMOTE  : {y_train_resampled.value_counts()}')


            logging.info('Handling the class imbalance On test data')
            logging.info(f'Y-test Value Counts Before SMOTE  : {y_test.value_counts()}')
            smote = SMOTE(random_state=42)
            X_test_resampled, y_test_resampled = smote.fit_resample(X_test_transformed,y_test)
            logging.info(f'Y-train Value Counts After SMOTE  : {y_test_resampled.value_counts()}')

            return(
                X_train_resampled,y_train_resampled,X_test_resampled,y_test_resampled
            )

        except Exception as e:
            logging.info(f'Exception Occured : {e}')
            raise CustomMachineLearningException(e,sys)

    def initiate_model_training(self , X_train_transformed,y_train,X_test_transformed,y_test):
        try:
            logging.info('***** PHASE 4 - Initiate Model Training Pipeline *****')
            logging.info('Handling the imbalance dataset')
            X_train_resampled,y_train_resampled,X_test_resampled,y_test_resampled = self.handle_class_imbalance(X_train_transformed,y_train,X_test_transformed,y_test)

        except Exception as e:
            logging.info(f'Exception Occured : {e}')
            raise CustomMachineLearningException(e,sys)
