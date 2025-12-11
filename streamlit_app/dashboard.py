import streamlit as st
import pandas as pd 
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, auc
from messages import *
import matplotlib.pyplot as plt
from imblearn.pipeline import Pipeline as ImbPipeline
from utils import regions_coords, regions_coords_latlon


# Fonctions utilitaires

def compute_feature_importances(model, X):
    if hasattr(model, "named_steps"):
        estimator = model.named_steps["clf"]
    else:
        estimator = model

    if hasattr(estimator, "feature_importances_"):
       
        return estimator.feature_importances_[: X.shape[1]]
    else:
        return None


# Dashboard principal

def render_dashboard(df, X, y_true, model, X_processed, y_prob, optimal_threshold,
                     bg_color, text_color, secondary_bg, plotly_template,
                     df_header_bg, df_cell_bg, btn_color, metric_bg, metric_text):

    fi = None

    st.sidebar.title("⚙️ Choisir un seuil de décision")
    threshold = st.sidebar.slider("Sélectionnez le seuil", 0.0,1.0,float(optimal_threshold),0.01)


    y_pred_thresh = np.where(y_prob >= threshold, "NON OBSERVEE", "OBSERVEE")
    df["Prédiction"] = y_pred_thresh

    st.title("📊 Disponibilité des moustiquaires")

    tabs = st.tabs([
        "✅ Répartition & Résumé", 
        "\u2003",
        "📈 Courbes Precision-Recall", 
        "\u2003",
        "📊 Importance des facteurs", 
        "\u2003",
        "🔍 Résultats détaillés", 
        "\u2003",
        "🗺️ Carte par Région" 
    ])
    tab1, tab2, tab3, tab4, tab5 = tabs[0], tabs[2], tabs[4], tabs[6], tabs[8]

    
    # Tab 1 : Répartition & Résumé
    
    with tab1:
        y_true_binary = np.where(y_true=="NON OBSERVEE", 1, 0)

        st.subheader("Répartition des classes")
        class_counts = df["TN3"].value_counts().reset_index()
        class_counts.columns = ["Classe","Nombre"]
        fig_dist = px.bar(class_counts, x="Classe", y="Nombre", color="Classe", text="Nombre",
                          color_discrete_sequence=px.colors.qualitative.Set2)
        fig_dist.update_layout(template=plotly_template, bargap=0.6, bargroupgap=0.3,
                               paper_bgcolor=bg_color, plot_bgcolor=bg_color, font_color=text_color)
        st.plotly_chart(fig_dist, use_container_width=True)

        st.subheader("Résumé du modèle")
        accuracy = np.mean(y_true==y_pred_thresh)

     
        non_obs_correct = ((df["Prédiction"]=="NON OBSERVEE") & (y_true=="NON OBSERVEE")).sum()
        non_obs_total = (y_true=="NON OBSERVEE").sum()
        non_obs_precision = non_obs_correct / non_obs_total if non_obs_total>0 else 0

        obs_correct = ((df["Prédiction"]=="OBSERVEE") & (y_true=="OBSERVEE")).sum()
        obs_total = (y_true=="OBSERVEE").sum()
        obs_precision = obs_correct / obs_total if obs_total>0 else 0

        
        precision_curve, recall_curve, _ = precision_recall_curve(y_true_binary,
                                                                 np.where(y_pred_thresh=="NON OBSERVEE",1,0))
        pr_auc = auc(recall_curve, precision_curve)

        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Précision générale", f"{accuracy*100:.2f} %")
        with col2: st.metric("Taux OBSERVEE détectée", f"{obs_precision*100:.2f} %")
        with col3: st.metric("Taux NON OBSERVEE détectée", f"{non_obs_precision*100:.2f} %")
        with col4: st.metric("Qualité détection positives", f"{pr_auc*100:.2f} %")  
       
        report_dict = classification_report(y_true, y_pred_thresh, output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report_dict).transpose().round(2)
        st.subheader("Graphique résumé par classe")
        
        show_full_table = st.checkbox("Afficher le tableau complet du rapport de classification", value=False)
        if show_full_table:
            st.dataframe(report_df, use_container_width=True)
        else:
            report_df_classes = report_df.loc[["NON OBSERVEE","OBSERVEE"], ["precision","recall","f1-score"]].reset_index().rename(columns={'index':'Classe'})
            fig_scores = px.bar(report_df_classes, x="Classe", y=["precision","recall","f1-score"],
                                barmode="group", labels={"value":"Score","variable":"Metric"}, range_y=[0,1],
                                color_discrete_sequence=px.colors.qualitative.Set2)
            fig_scores.update_layout(template=plotly_template, bargap=0.6, bargroupgap=0.3,
                                     paper_bgcolor=bg_color, plot_bgcolor=bg_color, font_color=text_color)
            st.plotly_chart(fig_scores, use_container_width=True)


        cm = confusion_matrix(y_true, y_pred_thresh, labels=["NON OBSERVEE", "OBSERVEE"])
        tn, fp, fn, tp = cm.ravel()
        data_non_obs = pd.DataFrame({"Status": ["Correct", "Erreur"], "Valeur": [tn, fp]})
        data_obs = pd.DataFrame({"Status": ["Correct", "Erreur"], "Valeur": [tp, fn]})

        show_matrix_img = st.checkbox("Afficher en tableau (matrice de confusion)", value=False)
        if show_matrix_img:
            labels = ["NON OBSERVEE", "OBSERVEE"]
            fig = go.Figure(data=go.Heatmap(z=cm, x=labels, y=labels, text=cm, texttemplate="%{text}",
                                            colorscale="Blues", showscale=False,
                                            hovertemplate='Réel: %{y}<br>Prédit: %{x}<br>Valeur: %{z}<extra></extra>'))
            fig.update_traces(textfont_size=24)
            fig.update_layout(xaxis_title="Prédit", yaxis_title="Réel", xaxis=dict(side="top"),
                              template=plotly_template, yaxis=dict(autorange="reversed"),
                              paper_bgcolor=bg_color, plot_bgcolor=bg_color, font_color=text_color,
                              width=700, height=500)
            st.plotly_chart(fig, use_container_width=False)
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("<h3 style='text-align:center;'>NON OBSERVEE</h3>", unsafe_allow_html=True)
                fig1 = px.pie(data_non_obs, names="Status", values="Valeur", color="Status", template=plotly_template)
                fig1.update_traces(textposition='inside', textinfo='percent+label')
                fig1.update_layout(paper_bgcolor=bg_color, plot_bgcolor=bg_color, font_color=text_color)
                st.plotly_chart(fig1, use_container_width=True)
            with col2:
                st.markdown("<h3 style='text-align:center;'>OBSERVEE</h3>", unsafe_allow_html=True)
                fig2 = px.pie(data_obs, names="Status", values="Valeur", color="Status", template=plotly_template)
                fig2.update_traces(textposition='inside', textinfo='percent+label')
                fig2.update_layout(paper_bgcolor=bg_color, plot_bgcolor=bg_color, font_color=text_color)
                st.plotly_chart(fig2, use_container_width=True)

    
    # Tab 2 : PR Curve
    
    with tab2:
        st.subheader("Courbe Précision-Rappel")
        y_true_binary = np.where(y_true=="NON OBSERVEE",1,0)
        y_prob_minor = y_prob if not hasattr(model,"predict_proba") else model.predict_proba(X_processed)[:,0]
        precision, recall, _ = precision_recall_curve(y_true_binary, y_prob_minor)
        pr_auc = auc(recall, precision)
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name=f'PR Curve (AUC={pr_auc:.3f})'))
        fig_pr.update_layout(template="plotly_white", paper_bgcolor="white", plot_bgcolor="white",
                             font_color="black", xaxis_title="Rappel", yaxis_title="Précision")
        st.plotly_chart(fig_pr, use_container_width=True)
        st.markdown(PRECISION_RECALL_DESC)
    
    # Tab 3 : Feature importance
    
    with tab3:
        st.subheader("Facteurs influençant le modèle")
        if isinstance(model, ImbPipeline):
            rf = model.named_steps['clf']
        else:
            rf = model

        features = [c for c in X.columns if c not in ['lat','lon']]
        importances = rf.feature_importances_[:len(features)]
        # Ajuster si tailles différentes
        if len(features) != len(importances):
            min_len = min(len(features), len(importances))
            features = features[:min_len]
            importances = importances[:min_len]

        fi = pd.DataFrame({"Facteur": features, "importance": importances}).sort_values("importance", ascending=False)
        fi["importance"] = fi["importance"].round(3)

        fig_fi = px.bar(fi, x="importance", y="Facteur", color="importance", text="importance",
                        color_continuous_scale=px.colors.sequential.Viridis, orientation='h',
                        category_orders={"Facteur": fi["Facteur"].tolist()}, height=600)
        fig_fi.update_traces(textposition="inside", textfont_color="white")
        fig_fi.update_layout(template=plotly_template, paper_bgcolor=bg_color,
                             plot_bgcolor=bg_color, font_color=text_color)
        st.plotly_chart(fig_fi, use_container_width=True)

    
    # Tab 4 : Résultats détaillés
    
    with tab4:
        st.markdown("### Résultat Prédiction")
        nb_obs = ((df["Prédiction"]=="OBSERVEE") & (df["TN3"]=="OBSERVEE")).sum()
        nb_non_obs = ((df["Prédiction"]=="NON OBSERVEE") & (df["TN3"]=="NON OBSERVEE")).sum()
        errors = (y_true != df["Prédiction"]).sum()

        colA, colB, colC = st.columns(3)
        if "filter_option" not in st.session_state: st.session_state.filter_option = None
        if "error_class_filter" not in st.session_state: st.session_state.error_class_filter = None

        with colA:
            st.markdown(f"<h1 style='color:blue;'>{nb_obs}</h1>", unsafe_allow_html=True)
            if st.button("OBSERVEE"): st.session_state.filter_option = "OBSERVEE"

        with colB:
            st.markdown(f"<h1 style='color:orange;'>{nb_non_obs}</h1>", unsafe_allow_html=True)
            if st.button("NON OBSERVEE"): st.session_state.filter_option = "NON OBSERVEE"

        with colC:
            st.markdown(f"<h1 style='color:red;'>{errors}</h1>", unsafe_allow_html=True)
            if st.button("ERREURS"):
                st.session_state.filter_option = "ERREURS"
                st.session_state.error_class_filter = None

        df_display = df.copy() 
        df_display["proba_NON_OBS"] = y_prob.round(3)
        #st.dataframe(df_display, width="stretch")

        # ------------------- Update --------------------------
        for col in df_display.columns:
            if df_display[col].dtype.name in ['category', 'object']:
                df_display[col] = df_display[col].astype(str)
            else:
                df_display[col] = pd.to_numeric(df_display[col], errors='coerce')

        if st.session_state.filter_option == "OBSERVEE":
            df_display = df_display[(df_display["Prédiction"]=="OBSERVEE") & (df_display["TN3"]=="OBSERVEE")]
        elif st.session_state.filter_option == "NON OBSERVEE":
            df_display = df_display[(df_display["Prédiction"]=="NON OBSERVEE") & (df_display["TN3"]=="NON OBSERVEE")]
        elif st.session_state.filter_option == "ERREURS":
            df_display = df_display[y_true != df_display["Prédiction"]]
            st.markdown("#### 🔹 Filtrer erreurs par classe prédite")
            error_class_options = ["Tous","OBSERVEE","NON OBSERVEE"]
            st.session_state.error_class_filter = st.selectbox("Classe prédite", error_class_options, index=0)
            if st.session_state.error_class_filter != "Tous":
                df_display = df_display[df_display["Prédiction"]==st.session_state.error_class_filter]
            df_display["erreur_type"] = df_display.apply(lambda row: f"Réel={row['TN3']} / Prédit={row['Prédiction']}", axis=1)
            
            def detect_source_error(row, fi):
                fi_sorted = fi.copy()
                fi_sorted["value"] = row[fi_sorted["Facteur"]].values
                fi_sorted["value"] = pd.to_numeric(fi_sorted["value"], errors='coerce').fillna(0)
                
                fi_sorted["contrib"] = abs(fi_sorted["value"]-fi_sorted["value"].mean())*fi_sorted["importance"]
                
                return ", ".join(fi_sorted.sort_values("contrib",ascending=False).head(5)["Facteur"].tolist())
            df_display["erreur_source"] = df_display.apply(lambda row: detect_source_error(row, fi), axis=1)

        st.dataframe(df_display, width="stretch")

        
        if not df_display.empty:
            filtered_index = df_display.index.tolist()
            selected_index_pos = st.selectbox("Choisir une ligne à analyser", options=range(len(filtered_index)),
                                            format_func=lambda x: f"Ligne {filtered_index[x]}")
            selected_row = df_display.loc[filtered_index[selected_index_pos]]

            st.markdown("#### ➤ Probabilité et résultat")
            st.write(f"**Probabilité OBSERVEE :** {selected_row['proba_NON_OBS']}")
            st.write(f"**Prédiction seuil :** {selected_row['Prédiction']}")
            st.write(f"**Valeur réelle TN3 :** {selected_row['TN3']}")

            if st.session_state.filter_option == "ERREURS":
                pass
                # st.markdown("#### ⚠️ Source de l'erreur")
                # st.write(selected_row["erreur_type"])
                #st.markdown("#### 🧩 Variables contributives")
                #st.write(selected_row["erreur_source"])
            
            

            # --- Récupérer feature importances si fi None ---
            if fi is None:
                fi = compute_feature_importances(model, X)

            # S'assurer que fi est un array 1D
            fi_array = np.array(fi).flatten()[:len(X.columns)]

            # DataFrame pour Plotly
            fi_sorted = pd.DataFrame({
                "Facteur": X.columns,
                "importance": pd.to_numeric(fi_array, errors='coerce')
            }).sort_values("importance", ascending=False)
            # Valeurs de la ligne sélectionnée
            row_vals = selected_row[fi_sorted["Facteur"]].astype(float).values
            fi_sorted["value"] = row_vals

            # Normalisation
            fi_sorted["value_norm"] = (
                (fi_sorted["value"] - fi_sorted["value"].min()) /
                (fi_sorted["value"].max() - fi_sorted["value"].min() + 1e-9)
            )


            # Tri final pour l’affichage
            fi_sorted["importance"] = pd.to_numeric(fi_sorted["importance"], errors='coerce').fillna(0)
            fi_sorted = fi_sorted.sort_values("importance", ascending=False)


            fig_local = px.bar(
                fi_sorted, 
                y="Facteur", 
                x="value_norm", 
                text=fi_sorted["value"].round(3),
                color="importance", 
                color_continuous_scale=px.colors.sequential.Viridis,
                height=600 
            )
            fig_local.update_layout(
                template=plotly_template, 
                paper_bgcolor=bg_color, 
                plot_bgcolor=bg_color, 
                font_color=text_color
            )
            st.plotly_chart(fig_local, width="stretch")

            if st.session_state.filter_option == "ERREURS":
                    st.markdown(ERROR_FILTER_DESC)
            st.markdown(VARIABLE_CONTRIBUTION_DESC)

        

    
    # Tab 5 : Carte par région
    
    with tab5:
        st.markdown("### Visualisation par région")

        main_category = st.radio(
            "Sélectionner la catégorie principale",
            options=["OBSERVEE", "NON OBSERVEE", "ERREUR"],
            index=1,
            horizontal=True
        )
        
        extra_category = None
        if main_category == "ERREUR":
            extra_category = st.radio(
                "Afficher les erreurs pour la catégorie",
                options=["OBSERVEE", "NON OBSERVEE"],
                index=0,
                horizontal=True
            )

        df_map = df.copy()
        df_map['Error'] = np.where(df_map['Prédiction'] != df_map['TN3'], 'ERREUR', None)

        if main_category == "ERREUR":
            df_filtered = df_map[(df_map['Error']=='ERREUR') & (df_map['TN3']==extra_category)]
            category_label = extra_category
        else:
            df_filtered = df_map[df_map['Prédiction']==main_category]
            category_label = main_category

        # Remplacer les numéros par le nom de la région
        df_filtered['Region_name'] = df_filtered['Region'].map(regions_coords)

        # Compter par région
        df_counts = df_filtered.groupby('Region_name').size().reset_index(name='Count')
        df_counts['Prédiction'] = category_label
        df_counts['lat'] = df_counts['Region_name'].apply(lambda x: regions_coords_latlon[x]['lat'])
        df_counts['lon'] = df_counts['Region_name'].apply(lambda x: regions_coords_latlon[x]['lon'])

        if df_counts.empty:
            st.warning(f"Aucune donnée pour la sélection {category_label}.")
            df_plot = pd.DataFrame(columns=['Region_name','lat','lon','Count','Prédiction','Percentage','Size','hover_text'])
        else:
            total_count = df_counts['Count'].sum()
            df_counts['Percentage'] = ((df_counts['Count'] / total_count)*100).round(2)
            df_counts['Size'] = df_counts['Count']  # Size = Count

            # Hover personnalisé avec explication des erreurs
            def hover_text(row):
                if main_category == "ERREUR":
                    if row['Prédiction'] == "NON OBSERVEE":
                        return f"<b>Région:</b> {row['Region_name']}<br>" \
                            f"<b>Count:</b> {int(row['Count'])}<br>" \
                            f"<b>Percentage:</b> {row['Percentage']}%<br>" \
                            f"Ce ménage aurait dû avoir une moustiquaire."
                    else:
                        return f"<b>Région:</b> {row['Region_name']}<br>" \
                            f"<b>Count:</b> {int(row['Count'])}<br>" \
                            f"<b>Percentage:</b> {row['Percentage']}%<br>" \
                            f"Ce ménage n'aurait pas dû avoir de moustiquaire."
                else:
                    return f"<b>Région:</b> {row['Region_name']}<br>" \
                        f"<b>Count:</b> {int(row['Count'])}<br>" \
                        f"<b>Percentage:</b> {row['Percentage']}%"

            df_counts['hover_text'] = df_counts.apply(hover_text, axis=1)
            df_plot = df_counts

        fig_map = px.scatter_mapbox(
            df_plot,
            lat="lat",
            lon="lon",
            color="Prédiction",
            size="Size",
            size_max=40,
            hover_name=None,
            hover_data=None,
            custom_data=['hover_text'],
            zoom=5,
            height=800,
            width=1200,
            color_discrete_map={"OBSERVEE":"green","NON OBSERVEE":"red","ERREUR":"orange"}
        )

        fig_map.update_traces(
            marker=dict(opacity=0.7),
            hovertemplate='%{customdata[0]}<extra></extra>'
        )

        if main_category == "ERREUR":
            if extra_category == "NON OBSERVEE":
                title_text = "Ménages mal prédits : Devrait être OBSERVEE"
            else:
                title_text = "Ménages mal prédits : Devrait être NON OBSERVEE"
        else:
            title_text = f"Répartition des ménages : {main_category}"


        fig_map.update_layout(
            mapbox_style="open-street-map",
            paper_bgcolor=bg_color,
            plot_bgcolor=bg_color,
            font=dict(color=text_color, size=18),
            margin=dict(l=0,r=0,t=50,b=0),
            title={'text': title_text, 'x':0.5}
        )

        st.plotly_chart(fig_map, use_container_width=True)
