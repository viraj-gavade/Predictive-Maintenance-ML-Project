from src.exception_handler import CustomMachineLearningException
from src.logger import logging
from dataclasses import dataclass ,field
import os 
import sys
import pandas as pd

@dataclass 
class DataTransformationConfig:
    EXPECTED_COLS = field(default_factory=lambda:[
        'UDI','Product ID','Type','Air temperature [K]','Process temperature [K]','Rotational speed [rpm]','Torque [Nm]','Tool wear [min]',
        'Machine failure','TWF','HDF','PWF','OSF','RNF'
    ])
    UNEXPECTED_COLS = field(default_factory=list)
    MISSING_COLS = field(default_factory=list)
    VALIDATION_REPORT = field(default_factory=dict)
    TARGET_VARIABLE : str = 'Machine failure'
    EXPECTED_DTYPES = field(default_factory=lambda:{
    "UDI": "int64",
    "Product ID": "object",
    "Type": "object",
    "Air temperature [K]": "float64",
    "Process temperature [K]": "float64",
    "Rotational speed [rpm]": "int64",
    "Torque [Nm]": "float64",
    "Tool wear [min]": "int64",
    "Machine failure": "int64",
    "TWF": "int64",
    "HDF": "int64",
    "PWF": "int64",
    "OSF": "int64",
    "RNF": "int64"
})

class DataTranformation:
    def __int__(self):
        self.data_tranformation_config = DataTransformationConfig()

    def check_expected_cols(self , df : pd.DataFrame )->dict:
        try:
            for col in self.data_tranformation_config.EXPECTED_COLS:
                if(col not in df.columns):
                    self.data_tranformation_config.MISSING_COLS.append(col)
                else:
                    logging.info('All columns are present')
            return {'missing_cols':self.data_tranformation_config.MISSING_COLS}
        except Exception as e:
            logging.info(f'Exception Occured : {e}')
            raise CustomMachineLearningException(e,sys)


    def check_unexpected_cols(self , df : pd.DataFrame )->dict:
        try:
            for col in df.columns:
                if(col not in self.data_tranformation_config.EXPECTED_COLS):
                    self.data_tranformation_config.UNEXPECTED_COLS.append(col)
                else:
                    logging.info('No unexpected columns present')
            return {'missing_cols':self.data_tranformation_config.UNEXPECTED_COLS}
        except Exception as e:
            logging.info(f'Exception Occured : {e}')
            raise CustomMachineLearningException(e,sys)

    def target_validation(self , df:pd.DataFrame)->bool:
        try:
            if self.data_tranformation_config.TARGET_VARIABLE not in df.columns:
                logging.info('Target variable not found')
                return False
            else:
                logging.info('Target variable exists')
                return True
        except Exception as e:
            logging.info(f'Exception Occured : {e}')
            raise CustomMachineLearningException(e,sys)


    def data_type_validation(self , df:pd.DataFrame)->dict:
        try:
            data_type_errors = {}
            for column_name , expected_dtype in self.data_tranformation_config.EXPECTED_DTYPES.items():
                if column_name not in df.columns:
                    continue
                actual_dtype = str(df[column_name].dtype)
                if actual_dtype != expected_dtype:
                    data_type_errors[column_name]={
                        'expected_dtype':expected_dtype,
                        'actual_dtype':actual_dtype
                    }

                if(len(data_type_errors)==0):logging.info('All data types are correct')

            return data_type_errors


        except Exception as e:
            logging.info(f'Exception Occured : {e}')
            raise CustomMachineLearningException(e,sys)

    def initiate_data_validation(self, train_path:str , test_path:str)->dict:
        try:
            logging.info('***** PHASE 2 - Initiate Data Validation Pipeline *****')
            logging.info('Loading the train and test data from the path')
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info('Data loaded sucessfully from the train and test path')

            logging.info('Checking the missing columns for the train dataset')
            train_missing_cols = self.check_expected_cols(train_df)

            logging.info('Checking the missing columns for the test dataset')
            test_missing_cols = self.check_expected_cols(test_df)

            if(len(train_missing_cols)and len(test_missing_cols)==0):
                self.data_tranformation_config.VALIDATION_REPORT['Expected_Columns'] = True
            else:
                self.data_tranformation_config.VALIDATION_REPORT['Expected_Columns'] = {
                    'Train_df': train_missing_cols,
                    'Test_df': test_missing_cols,
                }
            logging.info('Expected colums in train and test checked successfully')



            logging.info('Checking the unexpected columns for the train dataset')
            train_unexpected_cols = self.check_unexpected_cols(train_df)

            logging.info('Checking the unexpected columns for the test dataset')
            test_unexpected_cols = self.check_unexpected_cols(test_df)

            if(len(train_unexpected_cols)and len(test_unexpected_cols)==0):
                self.data_tranformation_config.VALIDATION_REPORT['UnExpected_Columns'] = True
            else:
                self.data_tranformation_config.VALIDATION_REPORT['UnExpected_Columns'] = {
                    'Train_df': train_unexpected_cols,
                    'Test_df': test_unexpected_cols,
                }
            logging.info('unexpected colums in train and test checked successfully')


            logging.info('Checking for the target variable')
            train_target_exists = self.target_validation(train_df)
            test_target_exists = self.target_validation(test_df)
            if(train_target_exists and test_target_exists == True):
                self.data_tranformation_config.VALIDATION_REPORT['Target_Validation'] = True
            else:
                self.data_tranformation_config.VALIDATION_REPORT['Target_Validation'] = {
                    'Train Target Exists': train_target_exists,
                    'Test Target Exists': test_target_exists,
                }
            logging.info('Target validation completed successfully')


            logging.info('Checking for the data types validation')
            train_dtypes_validation = self.data_type_validation(train_df)
            test_dtypes_validation = self.data_type_validation(test_df)

            if(len(train_dtypes_validation)and len(test_dtypes_validation)==0):
                self.data_tranformation_config.VALIDATION_REPORT['DataType_Validation'] = True
            else:
                self.data_tranformation_config.VALIDATION_REPORT['DataType_Validation'] = {
                                'Train_df': train_dtypes_validation,
                                'Test_df': test_dtypes_validation,
                            }
            logging.info('Data type validation validation completed successfully')



        except Exception as e:
            logging.info(f'Exception Occured : {e}')
            raise CustomMachineLearningException(e,sys)