import joblib
import pandas as pd
import numpy as np
from preprocessing import preprocessing
from feature_engineering import create_feat

from utils import read_yaml

PREPROCESSING_CONFIG_PATH = "../config/preprocessing_config.yaml"
PREDICT_CONFIG_PATH = "../config/predict_config.yaml"

params_prep = read_yaml(PREPROCESSING_CONFIG_PATH)
params = read_yaml(PREDICT_CONFIG_PATH)

normalizer = joblib.load("../output/normalizer.pkl")
ohe = joblib.load("../output/onehotencoder.pkl")
model = joblib.load('../output/model_name.pkl')
estimator = joblib.load('../output/best_estimator.pkl')

def construct(test_predict, params_prep):    
    df_test = pd.DataFrame(test_predict).astype(int)
    feat = create_feat(df_test)
    df_normalizer = preprocessing(feat, params_prep, state=normalizer)[0]
    df_ohe = ohe.transform(feat[['SEX', 'MARRIAGE']])
    col = ohe.get_feature_names_out()
    df_ohe = pd.DataFrame(df_ohe, columns=col)
    result = pd.merge(df_ohe, df_normalizer, how="left")
    return result

def predict(prediksian):
    to_predict_model = model.predict(prediksian) #sek salah
    if to_predict_model == [0]:
        print("Non Deafult")
    else:
        print("Default")

if __name__ == "__main__":
    n_data = int(input(f"Input data (enter int value): "))
    data_predict = {}
    for i in range(n_data):
        for i in params["x_col"]:
            if i in data_predict:
                data_predict[i].append(input(f"Input {i}: "))
            else:
                data_predict[i] = [input(f"Input {i}: ")]
    to_pred = construct(data_predict, params_prep)
    predict(to_pred)