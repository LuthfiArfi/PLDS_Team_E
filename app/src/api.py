import preprocessing
import feature_engineering
from predict import main_predict
from fastapi import FastAPI, Form

app = FastAPI()

@app.post("/predict")
def predict_api(data_predict = Form()):
    result = main_predict(data_predict)
    return {
        'result' : result
    }