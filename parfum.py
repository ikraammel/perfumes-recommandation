import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import streamlit as st

# Première commande dans le script Streamlit (doit être en haut)
st.set_page_config(page_title="IF Perfumes - Recommandations", page_icon="💐")

# Charger le dataset 'final_perfume_data.csv' avec pandas
perfumes = pd.read_csv('final_perfume_data.csv', encoding='utf-8', nrows=10)

# Vérification des colonnes requises
required_columns = ['Name', 'Brand', 'Description', 'Notes', 'Image URL']
missing_columns = [col for col in required_columns if col not in perfumes.columns]

if missing_columns:
    st.write(f"Les colonnes suivantes sont manquantes : {', '.join(missing_columns)}")


# Sélectionner les colonnes nécessaires pour l'analyse
perfumes = perfumes[['Name', 'Brand', 'Description', 'Notes', 'Image URL']]

# Supprimer les parfums sans description ou image
perfumes.dropna(subset=['Name', 'Brand', 'Description', 'Notes', 'Image URL'], inplace=True)

# Convertir les descriptions des parfums en vecteurs TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(perfumes['Description'])  # Utiliser 'Description' à la place de 'type'

# Calculer la similarité cosinus entre les vecteurs TF-IDF des descriptions
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Fonction pour obtenir des recommandations de parfums basées sur la similarité des descriptions
def get_recommendations_by_brand(brand, cosine_sim=cosine_sim):
    try:
        # Trouver l'index des parfums de la marque spécifiée
        indices = perfumes[perfumes['Brand'] == brand].index.tolist()
        if not indices:
            raise ValueError("Brand not found.")
    except ValueError as e:
        st.write(str(e))
        return pd.DataFrame(columns=['Name', 'Brand', 'Notes', 'Image URL'])

    # Obtenir les scores de similarité pour tous les parfums de la marque
    sim_scores = []
    for idx in indices:
        sim_scores.extend(list(enumerate(cosine_sim[idx])))

    # Trier les scores de similarité en ordre décroissant
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Sélectionner les indices des 5 parfums les plus similaires
    sim_scores = sim_scores[1:6]
    perfume_indices = [i[0] for i in sim_scores]

    # Retourner les informations des parfums recommandés
    return perfumes[['Name', 'Brand', 'Notes', 'Image URL']].iloc[perfume_indices]


# Titre de l'application Streamlit
st.title("Nos Parfums:")

# CSS personnalisé
st.markdown("""
    <style>
        body {
            background-color: #FFFFFF;
        }
        h1 {
            color: #FFB6C1;
            font-family: 'Arial', sans-serif;
            text-align: center;
            font-size: 36px;
        }
        .stButton>button {
            background-color: #FF69B4;
            color: white;
            border-radius: 5px;
        }
        .stTextInput>div>input, .stTextArea>div>textarea {
            border-color: #FFB6C1;
            background-color: #FFF0F5;
        }
        p {
            color: #D8BFD8;
        }
    </style>
""", unsafe_allow_html=True)

# Champ de saisie pour la marque du parfum
brand_name = st.text_input("Enter the perfume's brand you're looking for :", "Di Ser")

# Afficher les infos et recommandations lorsque l'utilisateur clique sur le bouton
if st.button('Recommend'):
    if brand_name:
        # Vérifier si la marque existe dans le DataFrame
        if brand_name in perfumes['Brand'].values:
            selected_perfume = perfumes[perfumes['Brand'] == brand_name].iloc[0]
            st.subheader(f"Selected Perfume: {selected_perfume['Name']}")
            st.write(f"Brand: {selected_perfume['Brand']}")
            st.write(f"Description: {selected_perfume['Description']}")
            st.write(f"Notes: {selected_perfume['Notes']}")
            st.image(selected_perfume['Image URL'], caption=selected_perfume['Name'])

            # Afficher les recommandations
            st.subheader("Recommended perfumes from the same brand:")
            recommendations = get_recommendations_by_brand(brand_name)
            if not recommendations.empty:
                for idx, row in recommendations.iterrows():
                    st.markdown(f"**{row['Name']}** - brand: {row['Brand']} - notes: {row['Notes']}")
                    st.image(row['Image URL'], caption=row['Name'])
        else:
            st.write("Brand not found. Please check the brand name and try again.")
