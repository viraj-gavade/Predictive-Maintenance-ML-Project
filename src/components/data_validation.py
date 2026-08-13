from src.exception_handler import CustomMachineLearningException
from src.logger import logging
from dataclasses import dataclass ,field
import os 
import sys
import pandas as pd
from pydantic import BaseModel
from typing import Dict , List
from typing import ClassVar


class ValidationReportSchema(BaseModel):
    validation_status: bool

    Expected_Columns: bool | Dict[str, List[str]]
    No_UnExpected_Columns: bool | Dict[str, List[str]]

    Target_Validation: bool | Dict[str, Dict[str, bool]]
    DataType_Validation: bool | Dict[str, Dict[str, str]]
    No_Missing_Values: bool | Dict[str, Dict[str, int]]
    No_Duplicates: bool | Dict[str, int]
    Valid_Categories: bool | Dict[str, bool]
    Range_Validation: bool | Dict[str, bool]



from dataclasses import dataclass, field
from typing import ClassVar, Dict


@dataclass
class DataValidationConfig:
    EXPECTED_COLS: ClassVar[list[str]] = [
        'UDI',
        'Product ID',
        'Type',
        'Air temperature [K]',
        'Process temperature [K]',
        'Rotational speed [rpm]',
        'Torque [Nm]',
        'Tool wear [min]',
        'Machine failure',
        'TWF',
        'HDF',
        'PWF',
        'OSF',
        'RNF'
    ]

    UNEXPECTED_COLS: list[str] = field(default_factory=list)

    MISSING_COLS: list[str] = field(default_factory=list)

    VALIDATION_REPORT: dict = field(default_factory=dict)

    VALIDATION_REPORT_PATH : str = os.path.join('reports','validation_report.json')

    TARGET_VARIABLE: str = 'Machine failure'

    EXPECTED_DTYPES: ClassVar[Dict[str, str]] = {
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
    }

class DataValidation:
    def __init__(self):
        self.data_validation_config = DataValidationConfig()

    def check_expected_cols(self , df : pd.DataFrame )->list:
        try:
            for col in self.data_validation_config.EXPECTED_COLS:
                if(col not in df.columns):
                    self.data_validation_config.MISSING_COLS.append(col)
                
            logging.info('All columns are present')
            return self.data_validation_config.MISSING_COLS
        except Exception as e:
            logging.info(f'Exception Occured : {e}')
            raise CustomMachineLearningException(e,sys)


    def check_unexpected_cols(self , df : pd.DataFrame )->list:
        try:
            for col in df.columns:
                if(col not in self.data_validation_config.EXPECTED_COLS):
                    self.data_validation_config.UNEXPECTED_COLS.append(col)
                
            logging.info('No unexpected columns present')
            return self.data_validation_config.UNEXPECTED_COLS
        except Exception as e:
            logging.info(f'Exception Occured : {e}')
            raise CustomMachineLearningException(e,sys)

    def target_validation(self , df:pd.DataFrame)->bool:
        try:
            if self.data_validation_config.TARGET_VARIABLE not in df.columns:
                logging.info('Target variable not found')
                return False
            else:
                logging.info('Target variable exists')
                return True
        except Exception as e:
            logging.info(f'Exception Occured : {e}')
            raise CustomMachineLearningException(e,sys)

    def check_missing_values(self,df:pd.DataFrame )->dict:
        try:
            missing_vals_cols = {}
            logging.info('Checking for missing values')
            for col in df.columns:
                if(df[col].isnull().sum()> 0 ):
                    missing_vals_cols[col] = df[col].isnull().sum()
                
            logging.info('No missing values found!')
            return missing_vals_cols

        except Exception as e:
            logging.info(f'Exception Occured : {e}')
            raise CustomMachineLearningException(e,sys)
        
    def check_duplicate_records(self,df:pd.DataFrame )->int:
        try:
            duplicates = df.duplicated().sum()
            return duplicates

        except Exception as e:
            logging.info(f'Exception Occured : {e}')
            raise CustomMachineLearningException(e,sys)

    def data_type_validation(self , df:pd.DataFrame)->dict:
        try:
            data_type_errors = {}
            for column_name , expected_dtype in self.data_validation_config.EXPECTED_DTYPES.items():
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


    def category_validation(self , df:pd.DataFrame)->bool:
        try:
            logging.info('Validating the catgoeries')
            valid_types = ['M' ,'L', 'H']
            if(set(df['Type'].unique()).issubset(set(valid_types))):
                logging.info('All Types are valid')
                return True
            else:
                logging.info('Invalid types found!')
                return False
        except Exception as e:
            logging.info(f'Exception Occured : {e}')
            raise CustomMachineLearningException(e,sys)

    def numerical_validtion(self , df:pd.DataFrame)->bool:
        try:
            logging.info('Validating the numerical values')
            range_validation_report = {}
            if(df['Air temperature [K]']< 0).any(): range_validation_report['Air temperature [K]'] = False
            else:range_validation_report['Air temperature [K]'] = True

            if(df['Process temperature [K]']< 0).any(): range_validation_report['Process temperature [K]'] = False
            else:range_validation_report['Process temperature [K]'] = True


            if(df['Rotational speed [rpm]']< 0).any(): range_validation_report['Rotational speed [rpm]'] = False
            else:range_validation_report['Rotational speed [rpm]'] = True

            if(df['Torque [Nm]']< 0).any(): range_validation_report['Torque [Nm]'] = False
            else:range_validation_report['Torque [Nm]'] = True

            return range_validation_report
        
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

            if(len(train_missing_cols)== 0 and len(test_missing_cols)==0):
                self.data_validation_config.VALIDATION_REPORT['Expected_Columns'] = True
            else:
                self.data_validation_config.VALIDATION_REPORT['Expected_Columns'] = {
                    'Train_df': train_missing_cols,
                    'Test_df': test_missing_cols,
                }
            logging.info('Expected colums in train and test checked successfully')



            logging.info('Checking the unexpected columns for the train dataset')
            train_unexpected_cols = self.check_unexpected_cols(train_df)

            logging.info('Checking the unexpected columns for the test dataset')
            test_unexpected_cols = self.check_unexpected_cols(test_df)

            if(len(train_unexpected_cols)== 0 and len(test_unexpected_cols)==0):
                self.data_validation_config.VALIDATION_REPORT['No_UnExpected_Columns'] = True
            else:
                self.data_validation_config.VALIDATION_REPORT['No_UnExpected_Columns'] = {
                    'Train_df': train_unexpected_cols,
                    'Test_df': test_unexpected_cols,
                }
            logging.info('unexpected colums in train and test checked successfully')


            logging.info('Checking for the target variable')
            train_target_exists = self.target_validation(train_df)
            test_target_exists = self.target_validation(test_df)
            if(train_target_exists and test_target_exists == True):
                self.data_validation_config.VALIDATION_REPORT['Target_Validation'] = True
            else:
                self.data_validation_config.VALIDATION_REPORT['Target_Validation'] = {
                    'Train Target Exists': train_target_exists,
                    'Test Target Exists': test_target_exists,
                }
            logging.info('Target validation completed successfully')


            logging.info('Checking for the data types validation')
            train_dtypes_validation = self.data_type_validation(train_df)
            test_dtypes_validation = self.data_type_validation(test_df)

            if(len(train_dtypes_validation)==0 and len(test_dtypes_validation)==0):
                self.data_validation_config.VALIDATION_REPORT['DataType_Validation'] = True
            else:
                self.data_validation_config.VALIDATION_REPORT['DataType_Validation'] = {
                                'Train_df': train_dtypes_validation,
                                'Test_df': test_dtypes_validation,
                            }
            logging.info('Data type validation validation completed successfully')

            logging.info('Checking for the missing values in train and test datasets')
            train_missing_vals = self.check_missing_values(train_df)
            test_missing_vals = self.check_missing_values(test_df)
    
            if(len(train_missing_vals)==0  and  len(test_missing_vals)==0):
                self.data_validation_config.VALIDATION_REPORT['No_Missing_Values'] = True
            else:
                self.data_validation_config.VALIDATION_REPORT['No_Missing_Values'] = {
                                'Train_df': train_missing_vals,
                                'Test_df': test_missing_vals,
                            }
            logging.info('Missing values checked successfully!')

            
            logging.info('Checking for the duplicate records in train and test datasets')
            train_duplicates = self.check_duplicate_records(train_df)
            test_duplicates = self.check_duplicate_records(test_df)
    
            if(train_duplicates==0  and  test_duplicates==0):
                self.data_validation_config.VALIDATION_REPORT['No_Duplicates'] = True
            else:
                self.data_validation_config.VALIDATION_REPORT['No_Duplicates'] = {
                                'Train_df': train_duplicates,
                                'Test_df': test_duplicates,
                            }
            logging.info('Duplicate  values checked successfully!')


            logging.info('Checking the category validation on train and test data')
            is_train_category_valid = self.category_validation(train_df)
            is_test_category_valid = self.category_validation(test_df)
            if(is_train_category_valid==True and is_test_category_valid == True):
                self.data_validation_config.VALIDATION_REPORT['Valid_Categories'] = True
            else:
                self.data_validation_config.VALIDATION_REPORT['Valid_Categories'] = {
                                                'Train_df': is_train_category_valid,
                                                'Test_df': is_test_category_valid,
                                            }
            logging.info('Category validation completed successfully on train and test data')


            logging.info('Checking the numerical validations on train and test data')
            train_range_validation = self.numerical_validtion(train_df)
            test_range_validation = self.numerical_validtion(test_df)

            if (False in train_range_validation and test_range_validation):
                self.data_validation_config.VALIDATION_REPORT['Range_Validation'] = {
                                                'Train_df': train_range_validation,
                                                'Test_df': test_range_validation,
                                            }
            else:
                self.data_validation_config.VALIDATION_REPORT['Range_Validation'] = True

            logging.info('Range validation for numerical features completed successfully')


            if False in  self.data_validation_config.VALIDATION_REPORT:
                self.data_validation_config.VALIDATION_REPORT['VALIDATION_STATUS'] = False
            else:
                self.data_validation_config.VALIDATION_REPORT['VALIDATION_STATUS'] = True

            report = ValidationReportSchema(
                validation_status=self.data_validation_config.VALIDATION_REPORT["VALIDATION_STATUS"],
                Expected_Columns=self.data_validation_config.VALIDATION_REPORT["Expected_Columns"],
                No_UnExpected_Columns=self.data_validation_config.VALIDATION_REPORT["No_UnExpected_Columns"],
                Target_Validation=self.data_validation_config.VALIDATION_REPORT["Target_Validation"],
                DataType_Validation=self.data_validation_config.VALIDATION_REPORT["DataType_Validation"],
                No_Missing_Values=self.data_validation_config.VALIDATION_REPORT["No_Missing_Values"],
                No_Duplicates=self.data_validation_config.VALIDATION_REPORT["No_Duplicates"],
                Valid_Categories=self.data_validation_config.VALIDATION_REPORT["Valid_Categories"],
                Range_Validation=self.data_validation_config.VALIDATION_REPORT["Range_Validation"]
            )

            logging.info("Creating the validation reports directory")

            os.makedirs(
                os.path.dirname(
                    self.data_validation_config.VALIDATION_REPORT_PATH
                ),
                exist_ok=True
            )

            logging.info(
                f"Saving the validation report to the path: "
                f"{self.data_validation_config.VALIDATION_REPORT_PATH}"
            )

            with open(
                self.data_validation_config.VALIDATION_REPORT_PATH,
                "w"
            ) as f:
                f.write(report.model_dump_json(indent=4))

            logging.info("******** PHASE 2 DATA VALIDATION PIPELINE COMPLETED ********")

            return (
                train_df,
                test_df,
                self.data_validation_config.VALIDATION_REPORT,

            )


        except Exception as e:
            logging.info(f'Exception Occured : {e}')
            raise CustomMachineLearningException(e,sys)