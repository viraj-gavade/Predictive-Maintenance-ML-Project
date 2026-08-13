from src.exception_handler import CustomMachineLearningException
from src.logger import logging
from dataclasses import dataclass ,field
import os 
import sys
import pandas as pd
from pydantic import BaseModel
from typing import Dict , List
from typing import ClassVar
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_path : str = os.path.join('Artifcats/encoders','preprocessor.pkl')
    TARGET_VARIABLE : str = 'Machine failure'
    NUMERICAL_FEATURES : list = field(default_factory=list)
    CATEGORICAL_FEATURES : list = field(default_factory=list)
    COLUMSN_TO_DROP : list = field(default_factory=lambda : [ 'UDI','TWF','HDF','PWF','OSF','RNF','Product ID'])


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()


    def initiate_data_transformation(self , train_df : pd.DataFrame , test_df : pd.DataFrame):
        try:
            logging.info('***** PHASE 3 - Initiate Data Transformation Pipeline *****')
            logging.info('Seperating the input features and target Variable')
            X_train = train_df.drop(columns=[self.data_transformation_config.TARGET_VARIABLE],axis=1)
            y_train = train_df[[self.data_transformation_config.TARGET_VARIABLE]]

            X_test = test_df.drop(columns=[self.data_transformation_config.TARGET_VARIABLE],axis=1)
            y_test = test_df[[self.data_transformation_config.TARGET_VARIABLE]]
            logging.info('Input features and target features seperated sucessfully')
            logging.info(f'X_train Shape : {X_train.shape} , Y_train shape : {y_train.shape}')
            logging.info(f'X_test Shape : {X_test.shape} , Y_test shape : {y_test.shape}')

            logging.info('Dropping the unnecessary columns and columns which can cause data leaks')
            logging.info(f'Dropping the following columns : {self.data_transformation_config.COLUMSN_TO_DROP}')
            X_train.drop(columns=[self.data_transformation_config.COLUMSN_TO_DROP],axis=1,inplace=True)
            X_test.drop(columns=[self.data_transformation_config.COLUMSN_TO_DROP],axis=1,inplace=True)
            logging.info(f'Features of X_train and X_test : {X_train.columns.to_list()} , {X_test.columns.tolist()}')
            logging.info('Uncessary columns dropped successfully')


            logging.info('Seperating the numerical and categorical features')
            for col in train_df.columns:
                if(train_df[col].dtype == "O"):self.data_transformation_config.CATEGORICAL_FEATURES.append(col)
                else:self.data_transformation_config.NUMERICAL_FEATURES.append(col)
            logging.info(f'Numerical Features :  {self.data_transformation_config.NUMERICAL_FEATURES}')
            logging.info(f'Categorical Features :  {self.data_transformation_config.CATEGORICAL_FEATURES}')

            logging.info('No missing values and duplicate records in the dataset to handle')

            logging.info('Engineering the new features On Both train and test data')
            logging.info('Create New Feature Named :-> Temperature Difference')
            train_df["Temperature Difference"] = ( train_df["Process temperature [K]"] - train_df["Air temperature [K]"] )
            test_df["Temperature Difference"] = ( test_df["Process temperature [K]"] - test_df["Air temperature [K]"] )

            logging.info('Create New Feature Named :-> Power Proxy')
            train_df["Power Proxy"] = (train_df["Rotational speed [rpm]"]  * train_df["Torque [Nm]"])
            test_df["Power Proxy"] = (test_df["Rotational speed [rpm]"]  * test_df["Torque [Nm]"])

            logging.info('Create New Feature Named :-> Torque Tool Wear')
            train_df["Torque Tool Wear"] = (train_df["Torque [Nm]"] * train_df["Tool wear [min]"])
            test_df["Torque Tool Wear"] = (test_df["Torque [Nm]"] * test_df["Tool wear [min]"])

            logging.info('Create New Feature Named :-> Speed Torque Ratio')
            train_df["Speed Torque Ratio"] = (train_df["Rotational speed [rpm]"]/ (train_df["Torque [Nm]"] + 1e-6))
            test_df["Speed Torque Ratio"] = (test_df["Rotational speed [rpm]"]/ (test_df["Torque [Nm]"] + 1e-6))

            logging.info('New Features created sucessfully')

            logging.info('Applying the simpleImputer and Standard Scalar for numerical Features')
            numerical_pipeline = Pipeline(steps=[ ("imputer", SimpleImputer(strategy="median")),("scaler", StandardScaler()) ])

            logging.info('Applying the SimpleImputer and OneHotEncoder for categorical features')
            categorical_pipeline = Pipeline(
                                            steps=[
                                                ("imputer", SimpleImputer(strategy="most_frequent")),
                                                ("encoder", OneHotEncoder(
                                                    handle_unknown="ignore",
                                                    sparse_output=False
                                                ))
                                            ])


            logging.info('Applying the column transformer for both the numerical and categorical pipelines')
            preprocessor = ColumnTransformer(
                                    transformers=[
                                        (
                                            "numerical_pipeline",
                                            numerical_pipeline,
                                            self.data_transformation_config.NUMERICAL_FEATURES
                                        ),
                                        (
                                            "categorical_pipeline",
                                            categorical_pipeline,
                                            self.data_transformation_config.CATEGORICAL_FEATURES
                                        )
                                    ]
                                )
            logging.info('Fitting the column Transformer')
            X_train_transformed = preprocessor.fit_transform(X_train)
            logging.info('Saving the fitted tranformer')
            save_object(self.data_transformation_config.preprocessor_path,preprocessor)
            X_test_transformed = preprocessor.transform(X_test)
            logging.info(f'X_train_transformed : {X_train_transformed.shape} ,y_train : {y_train.shape} ')
            logging.info(f'X_test_transformed : {X_test_transformed.shape} ,y_test : {y_test.shape} ')

            logging.info('*****  PHASE 3 - Data Transformation Pipeline completed Successfully *****')
            return(X_train_transformed,y_train,X_test_transformed,y_test)


        except Exception as e:
            logging.info(f'Exception Occured : {e}')
            raise CustomMachineLearningException(e,sys)
