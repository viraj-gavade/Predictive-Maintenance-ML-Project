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
import mlflow
from sklearn.metrics import accuracy_score , classification_report , confusion_matrix , f1_score 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.model_selection import RandomizedSearchCV
from mlflow.tracking import MlflowClient
from src.utils import save_object

client = MlflowClient()

models = {
                "K-Nearest Neighbors": KNeighborsClassifier(),
                # "Decision Tree": DecisionTreeClassifier(),
                # "Random Forest": RandomForestClassifier(),
                "AdaBoost": AdaBoostClassifier(),
                # "Support Vector Classifier": SVC(probability=True),
        
                }

@dataclass 
class ModelTrainingConfig:
    best_base_model_path : str = os.path.join('Artifacts/Models','best_base_model.pkl')
    best_tunned_model_path : str = os.path.join('Artifacts/Models','best_tunned_model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainingConfig()

    def register_model_on_prod(self , new_f1):
        try:
            versions = client.search_model_versions(
                "name='CreditRiskPredictorModel'"
            )

            if(len(versions)==0):
                logging.info('No registered model found , Registering version one')
                return True
            
            latest_verison = max(versions,key=lambda x:int(x.version))

            run_id = latest_verison.run_id

            run = client.get_run(run_id)

            prod_f1 = run.data.metrics.get(
                'test_f1', 0 
            )
            
            logging.info(f'PROD F1 SCORE :{prod_f1} || NEW MODEL F1 SCORE : {new_f1} ')

            if new_f1 > prod_f1:
                logging.info("New model is better. Register model.")
                return True
            else:
                logging.info('PROD MODEL IS BETTER')
                return False

        except Exception as e :
            logging.info(f'Error : {e}')
            raise CustomMachineLearningException(e,sys)
    def hyperparamter_tune_base_model(self ,X_train_resampled,y_train_resampled,model_name : str):
        try:
            logging.info(f'Hyper paramtere tunning model : {model_name}')
            params_grid = {}
            model = models[model_name]
            if model == 'K-Nearest Neighbors':
                params_grid =  {
                    "n_neighbors": [3, 5, 7, 9, 11],
                    "weights": ["uniform", "distance"],
                    "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
                    "leaf_size": [20, 30, 40],
                    "p": [1, 2]  # 1 = Manhattan, 2 = Euclidean
                } 
            
            elif model == 'Decision Tree':
                        params_grid =  {
                "criterion": ["gini", "entropy", "log_loss"],
                "max_depth": [None, 5, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": [None, "sqrt", "log2"]
                }
            
            if model == 'Random Forest':
                params_grid ={
                            "n_estimators": [100, 200, 300],
                            "criterion": ["gini", "entropy", "log_loss"],
                            "max_depth": [None, 10, 20, 30],
                            "min_samples_split": [2, 5, 10],
                            "min_samples_leaf": [1, 2, 4],
                            "max_features": ["sqrt", "log2"],
                            "bootstrap": [True, False]
                        }
            else:
                params_grid = {
                            "n_estimators": [50, 100, 200],
                            "learning_rate": [0.01, 0.1, 0.5, 1.0],
                            
                        }

            random_search = RandomizedSearchCV(
                                estimator=model,
                                param_distributions=params_grid,
                                n_iter=5,
                                cv=3,
                                scoring="f1",
                                random_state=42,
                                )

            random_search.fit(X_train_resampled,y_train_resampled)
            tunned_model = random_search.best_estimator_
            tunned_model_score = random_search.best_score_
            tunned_model_best_params = random_search.best_params_
            logging.info(f'Hyper paramter done for model : {model_name} ')
            logging.info(f'Tunned Model Score : {tunned_model_score}')
            logging.info(f'Tunned Model Best Params : {tunned_model_best_params}')
            logging.info('Saving the tunned model')
            save_object(self.model_trainer_config.best_tunned_model_path,tunned_model)
            logging.info('Best model saved successfully')
            return(
                self.model_trainer_config.best_tunned_model_path
            )



        except Exception as e:
            logging.info(f'Exception Occured : {e}')
            raise CustomMachineLearningException(e,sys)


    def evaluate_model(self ,X_train_resampled,y_train_resampled,X_test_resampled,y_test_resampled ,models:dict )->dict:
        try:
            model_report:dict = {}
            logging.info('Training the mutiple models to find the best model')
            for model_name , model in models.items():
                model = model
                logging.info(f'***** Training {model_name}****')
                logging.info(f'X_train : {X_train_resampled.shape} , Y_train : {y_train_resampled.shape}')
                logging.info(f'X_test : {X_test_resampled.shape} , Y_train : {y_test_resampled.shape}')
                logging.info(f'Fitting the {model_name}')
                model.fit(X_train_resampled,y_train_resampled)
                logging.info('Model fitted successfully')

                logging.info('Predicting on the train and test data')
                y_pred_train = model.predict(X_train_resampled)
                y_pred_test = model.predict(X_test_resampled)

                logging.info('Calculating the metrics')
                train_accuracy = accuracy_score(y_train_resampled,y_pred_train)
                test_accuracy = accuracy_score(y_test_resampled,y_pred_test)
                logging.info(f'Train Accurcay : {train_accuracy} \n Test Accuracy : {test_accuracy}')

                train_clf = classification_report(y_train_resampled,y_pred_train)
                test_clf = classification_report(y_test_resampled,y_pred_test)
                logging.info(f'Train Classification report : {train_clf} \n TestClassification report : {test_clf}')


                train_cm = confusion_matrix(y_train_resampled,y_pred_train)
                test_cm = confusion_matrix(y_test_resampled,y_pred_test)
                logging.info(f'Train Confusion Matrix : {train_cm} \n Test Confusion Matrix : {test_cm}')

                train_f1 = f1_score(y_train_resampled,y_pred_train)
                test_f1 = f1_score(y_test_resampled,y_pred_test)
                logging.info(f'Train F1 Score : {train_f1} \n Test F1 Score : {test_f1}')

                logging.info('Tracking Experiment with mlflow')

                with mlflow.start_run(run_name=model_name) as run:
                    logging.info('Logging the params')
                    mlflow.log_params(
                          {
                             'model_name': model_name,
                        'train_sample': len(X_train_resampled),
                        'test_sample': len(X_test_resampled),
                        'n_features' : X_train_resampled.shape[1]
                       }
                    )
                    logging.info('Logging the metrics')
                    mlflow.log_metrics({
                            "train_accuracy": train_accuracy,
                            "test_accuracy": test_accuracy,
                            "train_f1": train_f1,
                            "test_f1": test_f1,
                        })
                    model_info = mlflow.sklearn.log_model(
                        model,
                        name=model_name,
                        serialization_format='pickle'
                    )
                    logging.info('Regstering the best base model on production')
                    if self.register_model_on_prod(test_f1) == True:
                        mlflow.register_model(
                            model_uri=model_info.model_uri,
                            name='PredictiveMaintainaceModel'
                        )
                model_report[model_name] = {
                    'train_accurcy': train_accuracy,
                    'test_accurcy':test_accuracy,
                    'train_clf':train_clf,
                    'test_clf' : test_clf,
                    'train_cm':train_cm,
                    'test_cm':test_cm,
                    'train_f1':train_f1,
                    'test_f1':test_f1

                }
            return model_report
        
        except Exception as e:
            logging.info(f'Exception Occured : {e}')
            raise CustomMachineLearningException(e,sys)



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
            model_report : dict = self.evaluate_model(X_train_resampled , y_train_resampled , X_test_resampled , y_test_resampled ,models=models)
            logging.info('Finding the best base model among the all models')
            best_f1 = 0 
            best_base_model = None
            for model_name in model_report:
                train_f1 = model_report[model_name]['train_f1']
                test_f1 = model_report[model_name]['test_f1']
                if test_f1 > best_f1:
                    best_base_model = models[model_name]
                    best_base_model_name = model_name
                    best_f1 = test_f1
            
            logging.info(f'Best Base Model Name : {best_base_model}  F1 Score : {best_f1}' )
            logging.info(f'Saving the best base model')
            save_object(self.model_trainer_config.best_base_model_path,best_base_model)

            self.hyperparamter_tune_base_model(X_train_resampled,y_train_resampled,best_base_model_name)

        except Exception as e:
            logging.info(f'Exception Occured : {e}')
            raise CustomMachineLearningException(e,sys)
