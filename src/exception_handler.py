import os 
import sys 

def error_message_details(error , error_details:sys):
    _ , _ , error_tb = error_details.exc_info()
    lineno = error_tb.tb_lineno
    error_message = f'Error Occured In Python Script {error_tb.tb_frame.f_code.co_filename} \n At line No : {lineno} \n Error : {str(error)}'
    return error_message


class CustomMachineLearningException(Exception):
    def __init__(self,error_message , error_details:sys):
        super().__init__(error_message)
        self.error_message = error_message_details(error_message,error_details)


    def __str__(self):
        return self.error_message


if __name__ == "__main__":
    try:
        a=1/0
    except Exception as e:
        print('Divide by zero error')
        raise CustomMachineLearningException(e,sys)