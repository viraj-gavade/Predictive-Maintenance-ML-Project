from fastapi import FastAPI
from pydantic import BaseModel

from src.components.pipeline.inference_pipeline import InferencePipeline, CustomData


app = FastAPI(
    title="Machine Failure Prediction API",
    description="Predict whether a machine is likely to fail.",
    version="1.0.0"
)


class MachineFailureRequest(BaseModel):
    Type: str
    air_temperature: float
    process_temperature: float
    rotational_speed: int
    torque: float
    tool_wear: int


@app.get("/")
def home():
    return {
        "message": "Machine Failure Prediction API is running."
    }


@app.post("/predict")
def predict(request: MachineFailureRequest):

    try:
        custom_data = CustomData(
            Type=request.Type,
            air_temperature=request.air_temperature,
            process_temperature=request.process_temperature,
            rotational_speed=request.rotational_speed,
            torque=request.torque,
            tool_wear=request.tool_wear
        )

        input_df = custom_data.get_data_as_dataframe()

        predictor = InferencePipeline()

        prediction = predictor.prediction(input_df)

        return {
            "prediction": int(prediction[0]),
            "result": (
                "Machine Failure Predicted"
                if prediction[0] == 1
                else "No Machine Failure Predicted"
            )
        }

    except Exception as e:
        return {
            "error": str(e)
        }