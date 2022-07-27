import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder, Normalizer
from tqdm import tqdm

tqdm.pandas()

from utils import read_yaml

FEATURE_ENGINEERING_CONFIG_PATH = "../config/feature_engineering_config.yaml"

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
        path = f"{params['out_path']}x_{i}_preprocessed_v2.pkl"
        temp = joblib.load(path)
        list_of_featured.append(temp)

    return list_of_featured

def one_hot_encoder(params,
                    x_cat,
                    state=None):
    """
    function to do task of one hot encoding
    
    Args:
    - x_cat(DataFrame): Categorical Input data
    - params(dict): one hot encoding parameters
    - vectorizer(callable): one hot encoding, default to None. 
    If None, then the function will create a new one hot encoder
    """
    index = x_cat.index
    col = x_cat.columns
    
    if state == None:
        encoder = OneHotEncoder(sparse=False,handle_unknown='ignore').fit(x_cat)
    
        joblib.dump(encoder,
                    params["out_path"]+"onehotencoder.pkl")
    elif state == 'transform':
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
    """
    function to normalize numerical data
    
    Args:
    - x_num(DataFrame): numerical Input data
    - params(dict): normalizer parameters
    - normalizer(callable): normalizer, default to None. 
    If None, then the function will create a new normalizer  
    """
    index = x_all.index
    cols = x_all.columns

    if state == None:
        normalizer = Normalizer().fit(x_all)
        joblib.dump(normalizer,
                    params["out_path"]+"normalizer.pkl")
    
    elif state == 'transform':
        normalizer = joblib.load(params["out_path"]+"normalizer.pkl")
    
    normalized = normalizer.transform(x_all)
    normalized = pd.DataFrame(normalized)
    normalized.index = index
    normalized.columns = cols
    return normalized

def preprocessing(house_variables_feat, params, state=None):
    """
    function split data categorical for one hot encoding, and labeling; and data numerical to be normalized
    
    Args:
    - house_variables_feat(DataFrame): Dataframe preprocessed
    - params(dict): parameters
    - state = None

    If None, then the function will create a new normalizer or encoder 
    """
    house_numerical = house_variables_feat[params['NUM_COLUMN']]
    house_categorical = house_variables_feat[params['CAT_COLUMN']]
    house_label = house_variables_feat[params['LABEL_COLUMN']]

    df_num_normalized = normalization(params, house_numerical, state=state)
    
    df_categorical_encoded = one_hot_encoder(params, house_categorical, state=state)
    
    df_joined = pd.concat([df_categorical_encoded, house_label, df_num_normalized], axis=1)
    
    return df_joined

def main_preprocessing(x_featured_list, params):
    x_train_featured, x_valid_featured, x_test_featured = x_featured_list
    x_train_preprocessed = preprocessing(x_train_featured, params, state=None)
    x_valid_preprocessed = preprocessing(x_valid_featured, params, state='transform')
    x_test_preprocessed = preprocessing(x_test_featured, params, state='transform')
    
    joblib.dump(x_train_preprocessed, f"{params['out_path']}x_train_normalized.pkl")
    joblib.dump(x_valid_preprocessed, f"{params['out_path']}x_valid_normalized.pkl")
    joblib.dump(x_test_preprocessed, f"{params['out_path']}x_test_normalized.pkl")

    return x_train_preprocessed, x_valid_preprocessed, x_test_preprocessed

if __name__ == "__main__":
    params = read_yaml(FEATURE_ENGINEERING_CONFIG_PATH)
    x_featured_list = load_featured_data(params)
    x_train_preprocessed, x_valid_preprocessed, x_test_preprocessed = main_preprocessing(x_featured_list, params)