import sys 
import os 

sys.path.append(os.path.dirname(__file__)) 

import streamlit as st
from config import get_theme, parse_theme
from data_loader import load_and_clean_data
from model_utils import load_model_and_predict
from dashboard import render_dashboard

st.set_page_config(
    page_title="Dashboard Moustiquaires",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📊"
)


theme = get_theme()
exec(theme) 

df, X, y_true = load_and_clean_data()


model, X_processed, y_prob, optimal_threshold = load_model_and_predict(X, y_true)


bg_color, text_color, secondary_bg, plotly_template, df_header_bg, df_cell_bg, btn_color, metric_bg, metric_text = parse_theme(theme)

render_dashboard(df, X, y_true, model, X_processed, y_prob, optimal_threshold,
                 bg_color, text_color, secondary_bg, plotly_template,
                 df_header_bg, df_cell_bg, btn_color, metric_bg, metric_text)
