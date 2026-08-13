import os 
from src.exception_handler import   CustomMachineLearningException
import sys
import dill

def save_object(filepath,obj):
    try:  
        dir_path = os.path.dirname(filepath)

        os.makedirs(dir_path,exist_ok=True)

        with open(filepath,'wb') as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomMachineLearningException(e,sys)
    


def load_object(filepath):
    try:  
        with open(filepath,'rb') as file_obj:
          return  dill.load(file_obj)

    except Exception as e:
        raise CustomMachineLearningException(e,sys)