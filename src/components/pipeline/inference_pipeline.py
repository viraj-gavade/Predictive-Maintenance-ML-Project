from src.exception_handler import CustomMachineLearningException
from src.logger import logging
import os
import sys
import pandas as pd

from src.utils import load_object


class InferencePipeline:

    def prediction(self, features: pd.DataFrame):

        try:
            logging.info("******** INITIATING THE INFERENCE PIPELINE ********")

            base_dir = os.path.dirname(
                os.path.dirname(
                    os.path.dirname(
                        os.path.dirname(
                            os.path.abspath(__file__)
                        )
                    )
                )
            )

            model_path = os.path.join(
                base_dir,
                "Artifacts",
                "Models",
                "best_base_model.pkl"
            )

            preprocessor_path = os.path.join(
                base_dir,
                "Artifacts",
                "encoders",
                "preprocessor.pkl"
            )

            logging.info("Model and preprocessor paths configured successfully")

            logging.info("Loading the model")
            model = load_object(model_path)
            logging.info("Model loaded successfully")

            logging.info("Loading the preprocessor")
            preprocessor = load_object(preprocessor_path)
            logging.info("Preprocessor loaded successfully")

            logging.info("Engineering features for inference data")

            features["Temperature Difference"] = (
                features["Process temperature [K]"]
                - features["Air temperature [K]"]
            )

            features["Power Proxy"] = (
                features["Rotational speed [rpm]"]
                * features["Torque [Nm]"]
            )

            features["Torque Tool Wear"] = (
                features["Torque [Nm]"]
                * features["Tool wear [min]"]
            )

            features["Speed Torque Ratio"] = (
                features["Rotational speed [rpm]"]
                / (features["Torque [Nm]"] + 1e-6)
            )

            logging.info("Feature engineering completed successfully")

            logging.info("Transforming inference features")
            features_transformed = preprocessor.transform(features)
            logging.info("Features transformed successfully")

            logging.info("Predicting machine failure")
            prediction = model.predict(features_transformed)
            logging.info(f"Prediction completed successfully: {prediction}")

            return prediction

        except Exception as e:
            logging.error(f"Error occurred in inference pipeline: {e}")
            raise CustomMachineLearningException(e, sys)


class CustomData:

    def __init__(
        self,
        Type: str,
        air_temperature: float,
        process_temperature: float,
        rotational_speed: int,
        torque: float,
        tool_wear: int
    ):

        self.Type = Type
        self.air_temperature = air_temperature
        self.process_temperature = process_temperature
        self.rotational_speed = rotational_speed
        self.torque = torque
        self.tool_wear = tool_wear

    def get_data_as_dataframe(self) -> pd.DataFrame:

        try:

            data = {
                "Type": [self.Type],
                "Air temperature [K]": [self.air_temperature],
                "Process temperature [K]": [self.process_temperature],
                "Rotational speed [rpm]": [self.rotational_speed],
                "Torque [Nm]": [self.torque],
                "Tool wear [min]": [self.tool_wear]
            }

            return pd.DataFrame(data)

        except Exception as e:
            logging.error(
                f"Error while creating inference dataframe: {e}"
            )
            raise CustomMachineLearningException(e, sys)