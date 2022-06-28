import pandas as pd
import joblib
from tqdm import tqdm

from utils import read_yaml

FEATURE_CONFIG_PATH = "../config/feature_config.yaml"

def load_split_data(params):
    """
    Loader for splitted data.
    
    Args:
    - params(dict): featuring engineering params.
    
    Returns:
    - x_train(DataFrame): inputs of train set.
    - x_valid(DataFrame): inputs of valid set.
    - x_test(DataFrame): inputs of test set.
    """

    x_train = joblib.load(params["out_path"]+"x_train.pkl")
    x_valid = joblib.load(params["out_path"]+"x_valid.pkl")
    x_test = joblib.load(params["out_path"]+"x_test.pkl")

    return x_train, x_valid, x_test

def age_bin(df):
    df['AgeBin'] = pd.cut(df['AGE'],[20, 25, 30, 35, 40, 50, 60, 80]).cat.codes
    return df

def compile_value(df):    
    # compile the others value (5,6,0) in education to 1 other (4) value
    col_edit = (df.EDUCATION == 5) | (df.EDUCATION == 6) | (df.EDUCATION == 0)
    df.loc[col_edit, 'EDUCATION'] = 4
    # 0 value in marriage also other, so we put in value 3 also
    df.loc[df.MARRIAGE == 0, 'MARRIAGE'] = 3
    return df

def closeness(df):
    df['Closeness_6'] = (df.LIMIT_BAL - df.BILL_AMT6) / df.LIMIT_BAL
    df['Closeness_5'] = (df.LIMIT_BAL - df.BILL_AMT5) / df.LIMIT_BAL
    df['Closeness_4'] = (df.LIMIT_BAL - df.BILL_AMT4) / df.LIMIT_BAL
    df['Closeness_3'] = (df.LIMIT_BAL - df.BILL_AMT3) / df.LIMIT_BAL
    df['Closeness_2'] = (df.LIMIT_BAL - df.BILL_AMT2) / df.LIMIT_BAL
    df['Closeness_1'] = (df.LIMIT_BAL - df.BILL_AMT1) / df.LIMIT_BAL
    return df

def create_feat(df):
    df = age_bin(df)
    df = compile_value(df)
    df = closeness(df)
    return df

def main_feat(x_train,x_valid,x_test, params):
    x_list = [x_train,x_valid,x_test]

    x_featured = []
    for x in tqdm(x_list):
        temp = create_feat(x)
        x_featured.append(temp)

    name = ['train','valid','test']
    for i,x in tqdm(enumerate(x_featured)):
        joblib.dump(x, f"{params['out_path']}x_{name[i]}_featured.pkl")

if __name__ == "__main__":
    params_feature = read_yaml(FEATURE_CONFIG_PATH)
    x_train, x_valid, x_test = load_split_data(params_feature)
    x_preprocessed_list = main_feat(x_train, x_valid, x_test, params_feature)