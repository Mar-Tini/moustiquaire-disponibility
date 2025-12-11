import pandas as pd
import numpy as np

def lat_log(df): 

    regions_coords = {
        "DIANA": {"lat": -12.3, "lon": 49.3},                       
        "SAVA": {"lat": -14.8, "lon": 50.3},                        
        "ANALANJIROFO": {"lat": -16.2, "lon": 49.9},                
        "ATSINANANA": {"lat": -18.9, "lon": 48.4},                  
        "ALAOTRA MANGORO": {"lat": -17.5, "lon": 48.5},             
        "BETSIBOKA": {"lat": -16.6, "lon": 47.5},                   
        "BOENY": {"lat": -15.7, "lon": 46.3},                       
        "SOFIA": {"lat": -16.2, "lon": 48.0},                       
        "BONGOLAVA": {"lat": -19.0, "lon": 46.5},                   
        "ITASY": {"lat": -19.2, "lon": 46.1},                        
        "ANALAMANGA": {"lat": -18.9, "lon": 47.5},                   
        "VAKINANKARATRA": {"lat": -19.5, "lon": 46.8},               
        "AMORON'I MANIA": {"lat": -20.0, "lon": 47.1},               
        "HAUTE MATSIATRA": {"lat": -21.2, "lon": 46.9},              
        "VATOVAVY FITOVINANY": {"lat": -19.1, "lon": 48.8},          
        "ATSIMO ATSINANANA": {"lat": -22.8, "lon": 47.8},            
        "IHOROMBE": {"lat": -22.4, "lon": 46.9},                     
        "ANDROY": {"lat": -25.0, "lon": 46.8},                       
        "ANOSY": {"lat": -25.0, "lon": 46.9},                        
        "ATSIMO ANDREFANA": {"lat": -24.5, "lon": 44.0},             
        "MENABE": {"lat": -20.0, "lon": 44.0},                       
        "MELAKY": {"lat": -16.3, "lon": 44.6},                       
    }


    df['HH7'] = df['HH7'].astype('string')
    df['HH7'] = df['HH7'].fillna('UNKNOWN').str.strip().str.upper()

    
  
    df['lat'] = df['HH7'].map(lambda x: regions_coords.get(x, {}).get('lat', np.nan))
    df['lon'] = df['HH7'].map(lambda x: regions_coords.get(x, {}).get('lon', np.nan))

   

    return df 



def feature_engineering(df):

    df['enfants'] = (df['Age_scolarisation'] < 18).astype(int)
    df['adultes'] = (df['Age_scolarisation'] >= 18).astype(int)
    df_ratio = df.groupby('Numero_grappe').agg({'enfants':'sum', 'adultes':'sum'}).reset_index()
    df_ratio['ratio_enfants_adultes'] = df_ratio['enfants'] / df_ratio['adultes']


    df_educ = df.groupby('Numero_grappe').agg({
        'Instruction_chef':'max', 
        'Instruction_mere':'max', 
        'Instruction_pere':'max'
    }).reset_index()

    
    df_educ['niveau_education_max'] = df_educ[['Instruction_chef','Instruction_mere','Instruction_pere']].max(axis=1)

 
    df['presence_parents'] = df['Numero_menage'].apply(lambda x: 1 if x in ['PARENT_CODE1','PARENT_CODE2'] else 0)
    df_parents = df.groupby('Numero_grappe')['presence_parents'].max().reset_index()

   
    df_taille = df.groupby('Numero_grappe').size().reset_index(name='taille_menage')

 
    df = df.merge(df_ratio[['Numero_grappe','ratio_enfants_adultes']], on='Numero_grappe', how='left')
    df = df.merge(df_educ[['Numero_grappe','niveau_education_max']], on='Numero_grappe', how='left')
    df = df.merge(df_parents, on='Numero_grappe', how='left')
    df = df.merge(df_taille, on='Numero_grappe', how='left')

    return df
