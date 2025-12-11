import pandas as pd
import streamlit as st
import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.preprocessing import drop_taget_nan, drop_feature, drop_50_nan, car_code
from src.features.build_features import lat_log, feature_engineering



def load_and_clean_data():
    st.sidebar.title("📂 Charger un nouveau dataset")
    uploaded_file = st.sidebar.file_uploader("Choisir un fichier", accept_multiple_files=False)

    df = None
    new_data = False


    if uploaded_file:
        new_data = True
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.dta'):
            df = pd.read_stata(uploaded_file)
        else:
            st.error("Format de fichier non supporté")
            st.stop()


    if df is None:
        df = pd.read_csv('data/processed/test.csv', encoding='utf-8-sig', engine='python')
           # ---------- X et y ----------
        X = df.drop(columns=["TN3"],axis=1)
        y_true = df["TN3"].astype(str).str.upper().str.strip()

        return df, X, y_true



    if "TN3" not in df.columns:
        st.error("Le fichier doit contenir la colonne 'TN3'.")
        st.stop()


    if new_data:

        if all(c in df.columns for c in ['HH7','HH1','hhweight']):

            df = drop_taget_nan(df)
            
            df = lat_log(df)
            
            df = drop_feature(df)

            df = drop_50_nan(df)

            df = car_code(df)

     
          
            rename_dict = {
                'HH7': 'Region',
                'HH1': 'Numero_grappe',
                'hhweight': 'Poids_echantillon',
                'wscore': 'Score_richesse',
                'HH2': 'Numero_menage',
                'schage': 'Age_scolarisation',
                'windex5': 'Quintile_richesse',
                'helevel': 'Instruction_chef',
                'felevel': 'Instruction_pere',
                'HH6': 'Milieu_urbain_rural',
                'melevel': 'Instruction_mere'
            }
            df = df.rename(columns=rename_dict)

            df = feature_engineering(df)

           
            df = df.rename(columns={
                'Instruction_chef':'Niveau_edu_chef',
                'Instruction_mere':'Niveau_edu_mere',
                'Instruction_pere':'Niveau_edu_pere',
                'Age_scolarisation':'Age',
                'Poids_echantillon':'Poids_menage',
                'Score_richesse':'Score_richesse_menage',
                'Quintile_richesse':'Quintile_richesse_menage',
                'presence_parents_x':'Presence_parents',
                'presence_parents_y':'Presence_parents_menage'
            })

    
            cols = [
                'Numero_grappe','Numero_menage','TN3','Milieu_urbain_rural','Region',
                'Niveau_edu_chef','Niveau_edu_mere','Niveau_edu_pere','Age',
                'Poids_menage','Score_richesse_menage','Quintile_richesse_menage',
                'lat','lon','enfants','adultes','Presence_parents','ratio_enfants_adultes',
                'niveau_education_max','taille_menage','Presence_parents_menage',
            ]

            df_clean = df[cols].copy()

        
            df_clean['Niveau_edu_mere'] = df_clean['Niveau_edu_mere'].fillna(df_clean['Niveau_edu_mere'].mode()[0])
            df_clean['Niveau_edu_pere'] = df_clean['Niveau_edu_pere'].fillna(df_clean['Niveau_edu_pere'].mode()[0])

      
            X = df_clean.drop(columns=["TN3"])
            y_true = df_clean["TN3"].astype(str).str.upper().str.strip()
            
            return df_clean, X, y_true
        
        else : 
            X = df.drop(columns=["TN3"])

            y_true = df["TN3"].astype(str).str.upper().str.strip()
            
            return df, X,y_true 
    else: 
        df = pd.read_csv("data/processed/" + new_data)
    
        X = df.drop(columns=["TN3"])
        y_true = df["TN3"].astype(str).str.upper().str.strip()

        return df, X, y_true
    
    