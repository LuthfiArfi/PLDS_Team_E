import pandas as pd
import joblib
from tqdm import tqdm
import numpy as np

from utils import read_yaml

PREPROCESSING_CONFIG_PATH = "../config/preprocessing_config.yaml"

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

def dropping(df):
    """
    Dropping unnecessary variables from datasets such as "ID".
    "PAY_0" - "PAY_6" dropped because we will overwrite again with a new function
    
    Args:
    - df (DataFrame): loaded from read_data.
    
    Returns:
    - df (DataFrame): dataframes dropped.
    """
    df = df.drop(['ID', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6'], axis=1)
    return df

def pay_feat(df, bill, pay, feat):
    """
    create pay_feature that we drop from drop function.
    for usability purpose we'd like to take off that feature and recreate here,
    so it's easier when its come to input data by user then.
    
    Args:
    - df (DataFrame): loaded from read_data.
    
    Returns:
    - df (DataFrame): dataframes with payments features like datasets.
    """

    df.loc[bill == pay, feat] = 0
    df.loc[bill > pay, feat] = 1
    df.loc[bill < pay, feat] = -1
    return df

def create_pay_feat(df):

    """
    applying payments features function.
    
    Args:
    - df (DataFrame): loaded from read_data.
    
    Returns:
    - df (DataFrame): dataframes with payments features like datasets.
    """

    pay_feat(df, df['BILL_AMT1'], df['PAY_AMT1'], 'PAY_0')
    pay_feat(df, df['BILL_AMT2'], df['PAY_AMT2'], 'PAY_2')
    pay_feat(df, df['BILL_AMT3'], df['PAY_AMT3'], 'PAY_3')
    pay_feat(df, df['BILL_AMT4'], df['PAY_AMT4'], 'PAY_4')
    pay_feat(df, df['BILL_AMT5'], df['PAY_AMT5'], 'PAY_5')
    pay_feat(df, df['BILL_AMT6'], df['PAY_AMT6'], 'PAY_6')
    return df

def age_bin(df):

    """
    create age bin from AGE feature.
    
    Args:
    - df (DataFrame): loaded from read_data.
    
    Returns:
    - df (DataFrame): dataframes with addtional age bin feature.
    """

    bin_ = np.arange(0,99,10).tolist()
    label_bin = np.arange(0,9).tolist()
    df['AgeBin'] = pd.cut(df['AGE'], bin_, labels=label_bin)
    return df

def compile_value(df):

    """
    compile the others value (5,6,0) in education to 1 other (4) value
    0 value in marriage also other, so we put in value 3 also
    
    Args:
    - df (DataFrame): loaded from read_data.
    
    Returns:
    - df (DataFrame): dataframes edited value in EDUCATION and MARRIAGE.
    """

    col_edit = (df.EDUCATION == 5) | (df.EDUCATION == 6) | (df.EDUCATION == 0)
    df.loc[col_edit, 'EDUCATION'] = 4

    df.loc[df.MARRIAGE == 0, 'MARRIAGE'] = 3
    return df

def closeness(df):

    """
    create closeness feature.
    
    Args:
    - df (DataFrame): loaded from read_data.
    
    Returns:
    - df (DataFrame): closeness feature added.
    """

    df['Closeness_6'] = (df.LIMIT_BAL - df.BILL_AMT6) / df.LIMIT_BAL
    df['Closeness_5'] = (df.LIMIT_BAL - df.BILL_AMT5) / df.LIMIT_BAL
    df['Closeness_4'] = (df.LIMIT_BAL - df.BILL_AMT4) / df.LIMIT_BAL
    df['Closeness_3'] = (df.LIMIT_BAL - df.BILL_AMT3) / df.LIMIT_BAL
    df['Closeness_2'] = (df.LIMIT_BAL - df.BILL_AMT2) / df.LIMIT_BAL
    df['Closeness_1'] = (df.LIMIT_BAL - df.BILL_AMT1) / df.LIMIT_BAL
    return df

def create_feat(df):
    """
    create all preprocesses feature.
    
    Args:
    - df (DataFrame): df.
    
    Returns:
    - df (DataFrame): preprocessed df.
    """
    df = create_pay_feat(df)
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
        joblib.dump(x, f"{params['out_path']}x_{name[i]}_preprocessed_v2.pkl")

if __name__ == "__main__":
    params_feature = read_yaml(PREPROCESSING_CONFIG_PATH)
    x_train, x_valid, x_test = load_split_data(params_feature)
    x_preprocessed_list = main_feat(x_train, x_valid, x_test, params_feature)
