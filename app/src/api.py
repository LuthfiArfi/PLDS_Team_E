from predict import main_predict
from fastapi import FastAPI
from utils import read_yaml
import joblib
from pydantic import BaseModel


FEATURE_ENGINEERING_CONFIG_PATH = "../config/feature_engineering_config.yaml"
PREDICT_CONFIG_PATH = "../config/predict_config.yaml"

params_prep = read_yaml(FEATURE_ENGINEERING_CONFIG_PATH)
model = joblib.load('../output/model_name_v2.pkl')

app = FastAPI()

class Item(BaseModel):
    LIMIT_BAL: int
    SEX: int
    EDUCATION: int
    MARRIAGE: int
    AGE: int
    BILL_AMT1: int
    BILL_AMT2: int
    BILL_AMT3: int
    BILL_AMT4: int
    BILL_AMT5: int
    BILL_AMT6: int
    PAY_AMT1: int
    PAY_AMT2: int
    PAY_AMT3: int
    PAY_AMT4: int
    PAY_AMT5: int
    PAY_AMT6: int

@app.post("/predict/")
def predict_api(item: Item):
    data_predict = {}

    for i, value in enumerate(item):
        data_predict[value[0]] = [value[1]]
    result = main_predict(data_predict, model=model, params_prep=params_prep)

    return {
        "result": result
    }