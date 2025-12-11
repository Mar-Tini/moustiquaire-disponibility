import streamlit as st
import pandas as pd
from src.data.preprocessing import drop_taget_nan, drop_feature, drop_50_nan, car_code

def set_theme(mode=None):
    if mode == "Sombre":
        bg_color = "#0E1117"
        text_color = "#FFFFFF"
        secondary_bg = "#1F2937"
        plotly_template = "plotly_dark"
        df_header_bg = "#1F2937"
        df_cell_bg = "#111827"
        btn_color = "#1F2937"
        metric_bg = "#1F2937"
        metric_text = "#FFFFFF"
    else:
        bg_color = "#FFFFFF"
        text_color = "#000000"
        secondary_bg = "#F4F4F4"
        plotly_template = "plotly_white"
        df_header_bg = "#E0E0E0"
        df_cell_bg = "#FFFFFF"
        btn_color = "#F0F2F6"
        metric_bg = "#F0F2F6"
        metric_text = "#000000"
    return bg_color, text_color, secondary_bg, plotly_template, df_header_bg, df_cell_bg, btn_color, metric_bg, metric_text

def apply_css(bg_color, text_color, secondary_bg, df_header_bg, df_cell_bg, btn_color, metric_bg, metric_text):
    st.markdown(f"""
    <style>
    body, .stApp {{ background-color: {bg_color} !important; color: {text_color} !important; }}
    section[data-testid="stSidebar"] {{ background-color: {secondary_bg} !important; color: {text_color} !important; }}
    .stButton>button {{ background-color: {btn_color} !important; color: {text_color} !important; font-weight: bold; }}
    .stButton>button:hover {{ filter: brightness(90%); }}
    div[data-testid="metric-container"] {{ background-color: {metric_bg} !important; color: {metric_text} !important; border-radius: 10px; padding: 10px 15px; text-align: center; display: inline-block; }}
    .stDataFrame table {{ color: {text_color} !important; background-color: {df_cell_bg} !important; }}
    .stDataFrame th {{ background-color: {df_header_bg} !important; color: {text_color} !important; }}
    .stDataFrame td {{ background-color: {df_cell_bg} !important; color: {text_color} !important; }}
    input, select, textarea {{ background-color: {df_cell_bg} !important; color: {text_color} !important; border: 1px solid {df_header_bg}; }}
    button[title="Settings"] {{ display: none !important; }}
    #MainMenu {{ visibility: visible; }}
    footer {{ visibility: hidden; }}
    
    
    </style>
    """, unsafe_allow_html=True)

def load_dataset():
    st.sidebar.title("📂 Charger un dataset")
    uploaded_file = st.sidebar.file_uploader("Choisir un fichier", accept_multiple_files=False)
    if uploaded_file:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.dta'):
            df = pd.read_stata(uploaded_file)
        else:
            st.error("Format non supporté")
            st.stop()
    else:
        df = pd.read_csv("../data/processed/train.csv")

    if "TN3" not in df.columns:
        st.error("Le fichier doit contenir la colonne 'TN3'.")
        st.stop()

    
    df = drop_taget_nan(df)
    df = drop_feature(df)
    df = drop_50_nan(df)
    df['melevel'] = df['melevel'].fillna(df['melevel'].mode()[0])
    df['felevel'] = df['felevel'].fillna(df['felevel'].mode()[0])
    df = car_code(df)
    return df



regions_coords = {
        1:"DIANA", 2:"SAVA", 3:"ANALANJIROFO", 4:"ATSINANANA", 5:"ALAOTRA MANGORO",
        6:"BETSIBOKA", 7:"BOENY", 8:"SOFIA", 9:"BONGOLAVA", 10:"ITASY",
        11:"ANALAMANGA", 12:"VAKINANKARATRA", 13:"AMORON'I MANIA", 14:"HAUTE MATSIATRA",
        15:"VATOVAVY FITOVINANY", 16:"ATSIMO ATSINANANA", 17:"IHOROMBE",
        18:"ANDROY", 19:"ANOSY", 20:"ATSIMO ANDREFANA", 21:"MENABE", 22:"MELAKY"
    }

regions_coords_latlon = {
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