"""
Module de nettoyage des données pour le système de recommandation de livres.
Sépare la logique de nettoyage du mécanisme de scoring.
"""

import pandas as pd


def clean_text(text):
    """
    Nettoie et normalise le texte.
    
    Args:
        text: Texte à nettoyer
        
    Returns:
        Texte nettoyé et normalisé (minuscules, espaces supprimés)
    """
    return str(text).strip().lower() if pd.notna(text) else ""


def load_and_clean_dataset(path="Book_Dataset_1.csv"):
    """
    Charge et nettoie le dataset de livres.
    
    Opérations effectuées :
    - Suppression des doublons
    - Filtrage des catégories invalides (vides, 'default', 'add a comment')
    - Suppression des lignes sans titre
    - Nettoyage du texte (titres, descriptions, catégories)
    - Création d'un texte complet pour les embeddings
    
    Args:
        path: Chemin vers le fichier CSV
        
    Returns:
        DataFrame nettoyé avec colonnes supplémentaires :
        - title_clean: Titre nettoyé
        - desc_clean: Description nettoyée
        - genre_clean: Catégorie nettoyée
        - text_full: Texte complet pour embeddings
    """
    print("\n[Nettoyage] Chargement du dataset...")
    
    # Chargement
    df = pd.read_csv(path, sep=',', encoding='latin1')
    original_count = len(df)
    
    # Suppression des doublons
    df = df.drop_duplicates()
    print(f"[OK] Doublons supprimés : {original_count - len(df)}")
    
    # Filtrage des catégories invalides
    df = df[
        (df['Category'].str.strip() != '') &
        (df['Category'].str.lower() != 'default') &
        (df['Category'].str.lower() != 'add a comment')
    ]
    print(f"[OK] Catégories invalides supprimées : {original_count - len(df)}")
    
    # Suppression des lignes sans titre
    df = df.dropna(subset=['Title'])
    
    # Nettoyage du texte pour créer le corpus final
    df['title_clean'] = df['Title'].apply(clean_text)
    df['desc_clean'] = df['Book_Description'].apply(clean_text)
    df['genre_clean'] = df['Category'].apply(clean_text)
    
    # Création du texte complet pour embeddings
    df['text_full'] = (
        df['title_clean'] + ". " +
        df['genre_clean'] + ". " +
        df['desc_clean']
    )
    
    print(f"[OK] {len(df)} livres nettoyés et prêts")
    
    return df


def get_statistics(df):
    """
    Retourne des statistiques sur le dataset nettoyé.
    
    Args:
        df: DataFrame nettoyé
        
    Returns:
        Dictionnaire avec les statistiques
    """
    return {
        'total_books': len(df),
        'unique_categories': df['genre_clean'].nunique(),
        'top_categories': df['genre_clean'].value_counts().head(10).to_dict(),
        'missing_descriptions': df['Book_Description'].isna().sum(),
        'average_description_length': df['desc_clean'].str.len().mean()
    }
