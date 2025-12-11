# messages.py

DASHBOARD_TITLE = "📊 Disponibilité des moustiquaires"

TAB_NAMES = [
    "Répartition & Résumé",
    "\u2003",
    "Courbes ROC / Precision-Recall",
    "\u2003",
    "Importance des facteurs",
    "\u2003",
    "Résultats détaillés"
]

CLASS_DISTRIBUTION_DESC = "Montre la proportion de moustiquaires observées vs non observées dans le dataset."

MODEL_SUMMARY_DESC = (
    "\n"
    "- Combien de prédictions sont correctes\n"
    "- Taux de détection pour OBSERVEE et NON OBSERVEE\n"
    "- PR-AUC proche de 1, le modèle repère très bien les cas non disponibilité moustiquaire. \n\n"
)

CLASS_SCORE_DESC = "Comparez la précision, le rappel et le F1-score pour chaque classe. Plus ces valeurs sont proches de 1, meilleur est le modèle."

ROC_DESC = (
    "Montre comment le modèle sépare correctement les moustiquaires observées et non-observées à différents seuils.\n"
    "\nPlus la courbe est proche du coin supérieur gauche, meilleur est le modèle."
)

PRECISION_RECALL_DESC = (
    "La courbe Précision-Rappel montre l'équilibre entre :\n"
    "- **Précision** : % des prédictions non observée correctes\n"
    "- **Rappel** : % des moustiquaires non observées correctement détectées\n"
    "\nPlus la courbe est haute et à droite, mieux le modèle détecte correctement les moustiquaires observées tout en limitant les erreurs."
)

FEATURE_IMPORTANCE_DESC = "Barres représentant l’importance de chaque variable pour la prédiction du modèle."

DETAILED_RESULTS_DESC = "Afficher les lignes correctement prédites, non observées, et les erreurs pour analyse détaillée."

ERROR_FILTER_DESC = "#### 🔹 Filtrer erreurs par classe prédite"

VARIABLE_CONTRIBUTION_DESC = "Barres représentant la contribution normalisée de chaque variable pour cette prédiction."
