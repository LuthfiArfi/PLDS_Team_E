import pandas as pd
import joblib

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

def main(x):
    df = x.copy()
    df = age_bin(df)
    df = compile_value(df)
    df = closeness(df)
    return df