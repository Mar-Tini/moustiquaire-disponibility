def get_theme(mode=None):
    if mode == "Sombre":
        return """
bg_color = "#0E1117"
text_color = "#FFFFFF"
secondary_bg = "#1F2937"
plotly_template = "plotly_dark"
df_header_bg = "#1F2937"
df_cell_bg = "#111827"
btn_color = "#1F2937"
metric_bg = "#1F2937"
metric_text = "#FFFFFF"
"""
    else:
        return """
bg_color = "#FFFFFF"
text_color = "#000000"
secondary_bg = "#F4F4F4"
plotly_template = "plotly_white"
df_header_bg = "#E0E0E0"
df_cell_bg = "#FFFFFF"
btn_color = "#F0F2F6"
metric_bg = "#F0F2F6"
metric_text = "#000000"
"""


def apply_css(bg_color, text_color, secondary_bg, plotly_template,
              df_header_bg, df_cell_bg, btn_color, metric_bg, metric_text):
    import streamlit as st
    import matplotlib.pyplot as plt
    import plotly.io as pio

 
    st.markdown(f"""
    <style>
    body, .stApp {{
        background-color: {bg_color} !important;
        color: {text_color} !important;
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background-color: {secondary_bg} !important;
        color: {text_color} !important;
    }}
    section[data-testid="stSidebar"] * {{
        color: {text_color} !important;
    }}

    /* Tabs */
    div[data-baseweb="tab-list"] button {{
        color: {text_color} !important;
    }}
    div[data-baseweb="tab-list"] button[aria-selected="true"] {{
        font-weight: bold;
    }}

    /* Metrics */
    div[data-testid="metric-container"] {{
        background-color: {metric_bg} !important;
        color: {metric_text} !important;
        border-radius: 10px;
        padding: 10px 15px;
        text-align: center;
        display: inline-block;
    }}
    div[data-testid="metric-container"] span {{
        color: {metric_text} !important;
    }}

    /* Boutons */
    .stButton>button {{
        background-color: {btn_color} !important;
        color: {text_color} !important;
        font-weight: bold;
    }}
    .stButton>button:hover {{
        filter: brightness(90%);
    }}

    /* Tables Dataframe */
    .stDataFrame table {{
        color: {text_color} !important;
        background-color: {df_cell_bg} !important;
    }}
    .stDataFrame th {{
        background-color: {df_header_bg} !important;
        color: {text_color} !important;
    }}
    .stDataFrame td {{
        background-color: {df_cell_bg} !important;
        color: {text_color} !important;
    }}
    .stDataFrame td span {{
        color: {text_color} !important;
    }}

    /* Inputs, selects, textareas, selectbox, slider */
    input, select, textarea, .stSelectbox select, .stTextInput input {{
        background-color: {df_cell_bg} !important;
        color: {text_color} !important;
        border: 1px solid {df_header_bg};
    }}
    .stSlider>div>div>div>div>div {{
        color: {text_color} !important;
    }}

    /* Radios, checkboxes, markdown, expander */
    .stRadio label, .stCheckbox label, .stMarkdown, .stText, .stExpanderHeader, .stCaption {{
        color: {text_color} !important;
    }}

    /* Hide settings menu and footer */
    button[title="Settings"] {{ display: none !important; }}
    footer {{ visibility: hidden; }}
    #MainMenu {{ visibility: visible; }}
    </style>
    """, unsafe_allow_html=True)

  
    template_dark = pio.templates[plotly_template].layout.template
    pio.templates["custom_dark"] = pio.templates[plotly_template]
    pio.templates["custom_dark"].layout.font.color = text_color
    pio.templates["custom_dark"].layout.title.font.color = text_color
    pio.templates["custom_dark"].layout.xaxis.color = text_color
    pio.templates["custom_dark"].layout.yaxis.color = text_color
    pio.templates["custom_dark"].layout.legend.font.color = text_color
    pio.templates["custom_dark"].layout.annotations = []
    

    plt.rcParams.update({
        "text.color": text_color,
        "axes.labelcolor": text_color,
        "axes.titlecolor": text_color,
        "xtick.color": text_color,
        "ytick.color": text_color,
        "axes.facecolor": bg_color,
        "figure.facecolor": bg_color,
        "axes.edgecolor": text_color,
        "grid.color": df_header_bg,
        "legend.facecolor": metric_bg,
        "legend.edgecolor": text_color
    })



def parse_theme(theme):
    values = []
    for line in theme.strip().split("\n"):
        _, val = line.split("=")
        val = val.strip().strip('"') 
        values.append(val)
    return values

