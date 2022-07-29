import joblib
import pandas as pd
from preprocessing_v2 import create_feat
from feature_engineering_v2 import preprocessing

from utils import read_yaml

FEATURE_ENGINEERING_CONFIG_PATH = "../config/feature_engineering_config.yaml"
PREDICT_CONFIG_PATH = "../config/predict_config.yaml"

params_prep = read_yaml(FEATURE_ENGINEERING_CONFIG_PATH)
params = read_yaml(PREDICT_CONFIG_PATH)

model = joblib.load('../output/model_name_v2.pkl')
estimator = joblib.load('../output/best_estimator_v2.pkl')

def construct(test_predict, params_prep):
    """
    a function to construct an input that about to predict
    
    Args:
    - test_predict (Dict): a dictionary we collect from input by user.
    
    Returns:
    - df (DataFrame): Dataframe has been preprocess and feature engineering and ready to predict.
    """    
    df_test = pd.DataFrame(test_predict).astype(int)
    feat = create_feat(df_test)
    df_preproceed = preprocessing(feat, params_prep, state='transform')
    return df_preproceed

def main_predict(data_predict, model=model, params_prep=params_prep):
    to_pred = construct(data_predict, params_prep)
    prediction = model.predict(to_pred)
    predict_proba = model.predict_proba(to_pred)
    if prediction == [0]:
        return "non-default", predict_proba
    else:
        return "default", predict_proba

if __name__ == "__main__":
    n_data = int(input(f"Input data (enter int value): "))
    data_predict = {}
    for i in range(n_data):
        for i in params["x_col"]:
            if i in data_predict:
                data_predict[i].append(input(f"Input {i}: "))
            else:
                data_predict[i] = [input(f"Input {i}: ")]
    main_predict(data_predict, model=model, params_prep=params_prep)