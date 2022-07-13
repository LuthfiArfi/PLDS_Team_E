import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder, Normalizer
from feature_engineering import main_feat as add_feature
from tqdm import tqdm

tqdm.pandas()

from utils import read_yaml

PREPROCESSING_CONFIG_PATH = "../config/preprocessing_config.yaml"

def load_featured_data(params):
    """
    Loader for featured data.
    
    Args:
    - params(dict): featuring params.
    
    Returns:
    - list_of_featured(List): list of featured data.
    """

    name = ['train','valid','test']
    list_of_featured = []
    for i in name:
        path = f"{params['out_path']}x_{i}_featured.pkl"
        temp = joblib.load(path)
        list_of_featured.append(temp)

    return list_of_featured

def one_hot_encoder(params,
                    x_cat,
                    state=None):
    index = x_cat.index
    col = x_cat.columns
    
    if state == None:
        encoder = OneHotEncoder(sparse=False,handle_unknown='ignore').fit(x_cat)
    
        joblib.dump(encoder,
                    params["out_path"]+"onehotencoder.pkl")
    else:
        encoder = joblib.load(params["out_path"]+"onehotencoder.pkl")
    
    encoded = encoder.transform(x_cat)
    feat_names = encoder.get_feature_names_out(col)
    encoded = pd.DataFrame(encoded)
    encoded.index = index
    encoded.columns = feat_names
    return encoded

def normalization(params,
                  x_all,
                  state = None):
    index = x_all.index
    cols = x_all.columns

    if state == None:
        normalizer = Normalizer().fit(x_all)
        
        joblib.dump(normalizer,
                    params["out_path"]+"normalizer.pkl")
    
    else:
        normalizer = joblib.load(params["out_path"]+"normalizer.pkl")
    
    normalized = normalizer.transform(x_all)
    normalized = pd.DataFrame(normalized)
    normalized.index = index
    normalized.columns = cols
    return normalized, normalizer

def preprocessing(house_variables_feat, params, state=None):
    
    house_numerical = house_variables_feat[params['NUM_COLUMN']]
    house_categorical = house_variables_feat[params['CAT_COLUMN']]
    house_label = house_variables_feat[params['LABEL_COLUMN']]

    df_num_normalized = normalization(params, house_numerical, state=None)
    
    df_categorical_encoded = one_hot_encoder(params, house_categorical, state=None)
    
    df_joined = pd.concat([df_categorical_encoded, house_label, df_num_normalized[0]], axis=1)
    
    return df_joined, df_num_normalized[1]

def main_preprocessing(x_featured_list, params):
    x_train_featured, x_valid_featured, x_test_featured = x_featured_list
    x_train_preprocessed, normalizer = preprocessing(x_train_featured, params)
    x_valid_preprocessed = preprocessing(x_valid_featured, params, normalizer)
    x_test_preprocessed = preprocessing(x_test_featured, params, normalizer)
    joblib.dump(x_train_preprocessed, f"{params['out_path']}x_train_preprocessed.pkl")
    joblib.dump(x_valid_preprocessed[0], f"{params['out_path']}x_valid_preprocessed.pkl")
    joblib.dump(x_test_preprocessed[0], f"{params['out_path']}x_test_preprocessed.pkl")

    return x_train_preprocessed, x_valid_preprocessed[0], x_test_preprocessed[0]

if __name__ == "__main__":
    params_prep = read_yaml(PREPROCESSING_CONFIG_PATH)
    x_featured_list = load_featured_data(params_prep)
    x_train_preprocessed, x_valid_preprocessed, x_test_preprocessed = main_preprocessing(x_featured_list, params_prep)