import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from utils import read_yaml

LOAD_SPLIT_CONFIG_PATH = "../config/read_data_config.yaml"

def get_stratify_col(y, stratify_col):
    if stratify_col is None:
        stratification = None
    else:
        stratification = y[stratify_col]
    
    return stratification

def split_in_out_put(df,
                    target_column,
                    set_index = None):

    #rename the target name
    df = df.rename(columns={'default.payment.next.month': 'TARGET'})

    #create input(all of x) and output(all of y)
    y = df[params['y_col']]
    x = df.drop([target_column], axis = 1)
    
    return y, x

def run_split_data(x, y,
                    stratify_col=None,
                    TEST_SIZE=0.2
                    ):
    
    strat_train = get_stratify_col(y, stratify_col)
    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                       stratify = strat_train,
                                       test_size= TEST_SIZE*2,
                                       random_state= 42)
    
    strat_test = get_stratify_col(y_test, stratify_col)
    x_valid, x_test, y_valid, y_test = train_test_split(x_test, y_test,
                                       stratify = strat_test,
                                       test_size= 0.5,
                                       random_state= 42)

    return x_train, y_train, x_valid, y_valid, x_test, y_test

def main_read(params):
    df = pd.read_csv(params['file_loc'])
    y, x = split_in_out_put(df, target_column=params['target'], set_index='ID')
    x_train, y_train,x_valid, y_valid,x_test, y_test = run_split_data(x, y, 
                                                                      params['stratify'],
                                                                      params['test_size'])

    joblib.dump(x_train, params["out_path"]+"x_train.pkl")
    joblib.dump(y_train, params["out_path"]+"y_train.pkl")
    joblib.dump(x_valid, params["out_path"]+"x_valid.pkl")
    joblib.dump(y_valid, params["out_path"]+"y_valid.pkl")
    joblib.dump(x_test, params["out_path"]+"x_test.pkl")
    joblib.dump(y_test, params["out_path"]+"y_test.pkl")

    return x_train, y_train, x_valid, y_valid, x_test, y_test

if __name__ == "__main__":
    params = read_yaml(LOAD_SPLIT_CONFIG_PATH)
    x_train, y_train, x_valid, y_valid, x_test, y_test = main_read(params)