import pandas as pd 
import os
import yaml

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)
    
def load_data(type): 

    path = config["data"]["raw"] if type == "raw" else config["data"]["processed_train"] if type == "processed_train" else  config["data"]["processed_test"] 

    df = (
          pd.read_csv(path) 
          if path.endswith('.csv') 
          else pd.read_excel(path) 
          if  path.endswith(('.xlsx', '.xls')) 
          else pd.read_stata(path) 
          if path.endswith('.dta') 
          else None
        )

    return df.copy() 
    

def load_raw_data(path):
    if not path.endswith(".csv"):
        raise ValueError("Invalid format")
    return pd.read_csv(path)

