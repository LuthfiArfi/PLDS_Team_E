import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, roc_auc_score, f1_score, make_scorer
tqdm.pandas()
from utils import read_yaml

MODELING_CONFIG_PATH = "../config/modeling_config.yaml"

def load_fed_data():
    """
    Loader for feature engineered data.
    Args:
    - params(dict): modeling params.
    Returns:
    - x_train(DataFrame): inputs of train set.
    - y_train(DataFrame): target of train set.
    - x_valid(DataFrame): inputs of valid set.
    - y_valid(DataFrame): terget of valid set.
    """

    x_train_path = "../output/x_train_preprocessed.pkl"
    y_train_path = "../output/y_train.pkl"
    x_valid_path = "../output/x_valid_preprocessed.pkl"
    y_valid_path = "../output/y_valid.pkl"
    x_train = joblib.load(x_train_path)
    y_train = np.ravel(joblib.load(y_train_path))
    x_valid = joblib.load(x_valid_path)
    y_valid = np.ravel(joblib.load(y_valid_path))
    return x_train, y_train, x_valid, y_valid

def model():
    """
    Function for initiating Random Forest Model
    """
    base_model = MLPClassifier(random_state=1)
    param_dist = {'alpha' : [1],
                  'max_iter': [1000]}

    return base_model, param_dist

def random_search_cv(model, param, scoring, n_iter, x, y, verbosity=0):
    """
    Just a function to run the hyperparameter search
    """
    random_fit = RandomizedSearchCV(estimator = model, 
                                    param_distributions = param, 
                                    scoring = scoring, 
                                    n_iter = n_iter, 
                                    cv = 5, 
                                    random_state = 42, 
                                    verbose = verbosity)
    random_fit.fit(x, y)
    return random_fit

def classif_report(model_obj, x_test, y_test):
    code2rel = {'0': 'Non-default', '1': 'default'}
    
    pred = model_obj.predict(x_test)

    res = classification_report(
        y_test, pred, output_dict=True, zero_division=0)
    res = pd.DataFrame(res).rename(columns=code2rel).T

    return pred, res

def fit(x_train, y_train, model, model_param, scoring='f1', n_iter=1, verbosity=3):
    """
    Fit model
    
    Args:
        - model(callable): sklearn model
        - model_param(dict): sklearn's RandomizedSearchCV params_distribution
    
    Return:
        - model_fitted(callable): model with optimum hyperparams
    """
    model_fitted = random_search_cv(model, model_param, 
                                    scoring, 
                                    n_iter, 
                                    x_train, y_train, 
                                    verbosity)
    print(
        f'Model: {model_fitted.best_estimator_}, {scoring}: {model_fitted.best_score_}')
    
    

    return model_fitted

def validate(x_valid, y_valid, model_fitted):
    pred_model, report_model = classif_report(model_fitted, x_valid, y_valid)
    return report_model, model_fitted

def main_training_model(param_model, x_train, y_train, x_valid, y_valid):
        base_model, param_dist = param_model
        scoring = make_scorer(f1_score,average='macro')
        model_fitted = fit(x_train, y_train, base_model, param_dist, scoring=scoring, verbosity=0)
        report, model_fitted = validate(x_valid, y_valid, model_fitted)
        joblib.dump(model_fitted.best_estimator_, '../output/model_name.pkl')
        joblib.dump(model_fitted.best_params_, '../output/best_estimator.pkl')
        return report, model_fitted

if __name__ == "__main__":
    param_model = model()
    x_train, y_train, x_valid, y_valid = load_fed_data()
    hasil = main_training_model(param_model, x_train, y_train, x_valid, y_valid)