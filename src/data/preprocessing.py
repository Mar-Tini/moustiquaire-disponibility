import pandas as pd
import os
import yaml

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

    
def drop_taget_nan(df): 
    target = config['features']['target']
    df = df.dropna(subset=[target])

    return df 

def drop_feature(df): 
    drop_cols = config['features']['drop_columns']
    df = df.drop(columns=drop_cols, errors='ignore')

    return df 

def drop_50_nan(df):
    colonnes_50_vide = df.columns[df.isna().mean() >= 0.5]
    df = df.drop(columns=colonnes_50_vide)
    
    return df 


def complte_nan(variables , df): 
    for col in [variables]:
        if col in df.columns:
            df[col].fillna(df[col].mode()[0])

    return df  

def car_code(df):
    cat_cols = df.select_dtypes(exclude=['int64','float64']).columns
    cat_cols = cat_cols.drop(['TN3','lat', 'lon'], errors='ignore')

    for col in cat_cols:
        df[col] = df[col].astype('category').cat.codes

    return df

