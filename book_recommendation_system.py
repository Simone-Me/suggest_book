# -*- coding: utf-8 -*-
# ============================================================
#  Système de Recommandation Littéraire - SBERT + GenAI
#  Auteur : Héctor - EFREI M1 Data Engineering
#  Fonctionnalités :
#    - Questionnaire utilisateur (EF1)
#    - Embeddings SBERT locaux (EF2)
#    - Similarité cosinus (EF2.3)
#    - Scoring pondéré (EF3)
#    - Recommandations Top 3 (EF3.2)
#    - Augmentation GenAI conditionnelle (EF4)
# ============================================================

import sys
import os
import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# ============================================================
# EF1 : ACQUISITION DE LA DONNÉE - Questionnaire Utilisateur
# ============================================================

def collect_user_preferences():
    """
    EF1.1 : Questionnaire Hybride (numérique + ouvert)
    Collecte les préférences littéraires de l'utilisateur
    """
    print("\n" + "="*80)
    print("     QUESTIONNAIRE DE PREFERENCES LITTERAIRES")
    print("="*80)
    
    preferences = {}
    
    # Questions ouvertes (EF1.1)
    print("\n[QUESTIONS OUVERTES]")
    print("-"*80)
    
    preferences['description'] = input(
        "\n1. Décrivez le type de livre que vous recherchez\n"
        "   (thèmes, intrigue, émotions recherchées) :\n"
        "   > "
    ).strip()
    
    preferences['favorite_books'] = input(
        "\n2. Citez vos livres préférés (séparés par des virgules) :\n"
        "   > "
    ).strip()
    
    preferences['avoid'] = input(
        "\n3. Quels genres ou thèmes souhaitez-vous éviter ?\n"
        "   > "
    ).strip()
    
    # Questions numériques - Échelle de Likert (EF1.1)
    print("\n\n[QUESTIONS NUMERIQUES - Échelle 1-5]")
    print("-"*80)
    print("(1=Pas du tout, 2=Un peu, 3=Modérément, 4=Beaucoup, 5=Énormément)")
    
    def get_likert_score(question):
        while True:
            try:
                score = int(input(f"\n{question}\n   > "))
                if 1 <= score <= 5:
                    return score
                print("   ⚠️ Veuillez entrer un nombre entre 1 et 5")
            except ValueError:
                print("   ⚠️ Veuillez entrer un nombre valide")
    
    preferences['intensity_action'] = get_likert_score(
        "4. Intensité d'action/suspense souhaitée (1-5) ?"
    )
    
    preferences['intensity_romance'] = get_likert_score(
        "5. Intérêt pour les histoires romantiques (1-5) ?"
    )
    
    preferences['intensity_learning'] = get_likert_score(
        "6. Importance de l'aspect éducatif/apprentissage (1-5) ?"
    )
    
    preferences['complexity'] = get_likert_score(
        "7. Niveau de complexité narrative souhaité (1-5) ?"
    )
    
    # Métadonnées
    preferences['timestamp'] = datetime.now().isoformat()
    
    # EF1.2 : Structuration - Sauvegarde JSON
    save_preferences(preferences)
    
    return preferences


def save_preferences(preferences, filepath="user_preferences.json"):
    """
    EF1.2 : Stockage structuré des préponses (JSON)
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(preferences, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Préférences sauvegardées dans {filepath}")


# ============================================================
# EF2 : MOTEUR NLP SÉMANTIQUE (Coût Zéro)
# ============================================================

def load_knowledge_base(path="Book_Dataset_1.csv"):
    """
    EF2.1 : Référentiel de connaissances (livres)
    Charge et nettoie le dataset de livres
    """
    print("\n[EF2.1] Chargement du référentiel de connaissances...")
    
    df = pd.read_csv(path, sep=',', encoding='latin1')
    df = df.drop_duplicates()
    
    # Nettoyage
    df = df[
        (df['Category'].str.strip() != '') &
        (df['Category'].str.lower() != 'default') &
        (df['Category'].str.lower() != 'add a comment')
    ]
    
    df = df.dropna(subset=['Title'])
    
    def clean_text(text):
        return str(text).strip().lower() if pd.notna(text) else ""
    
    df['title_clean'] = df['Title'].apply(clean_text)
    df['desc_clean'] = df['Book_Description'].apply(clean_text)
    df['genre_clean'] = df['Category'].apply(clean_text)
    
    # EF2.1 : Texte complet pour embeddings
    df['text_full'] = (
        df['title_clean'] + ". " +
        df['genre_clean'] + ". " +
        df['desc_clean']
    )
    
    print(f"[OK] {len(df)} livres dans le référentiel")
    
    return df


def load_sbert_and_embeddings(df, embedding_path="embeddings_books.pkl"):
    """
    EF2.2 : Modélisation Sémantique avec SBERT (Open-Source, Local)
    Génère ou charge les embeddings des livres
    """
    print("\n[EF2.2] Modélisation sémantique (SBERT)...")
    
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    if os.path.exists(embedding_path):
        print("[OK] Chargement des embeddings depuis le cache...")
        with open(embedding_path, 'rb') as f:
            embeddings = pickle.load(f)
    else:
        print("[OK] Génération des embeddings (première fois)...")
        embeddings = model.encode(
            df['text_full'].tolist(),
            convert_to_tensor=False,
            show_progress_bar=True,
            batch_size=32
        )
        
        with open(embedding_path, 'wb') as f:
            pickle.dump(embeddings, f)
        print(f"[OK] Embeddings sauvegardés dans {embedding_path}")
    
    embeddings = np.array(embeddings)
    print(f"[OK] Embeddings shape: {embeddings.shape}")
    
    return model, embeddings


# ============================================================
# EF4.1 : AUGMENTATION PRE-PROCESSING (GenAI Conditionnelle)
# ============================================================

def enrich_short_query(text, use_genai=False, api_key=None):
    """
    EF4.1 : Augmentation de l'Entrée
    Enrichit les requêtes trop courtes (<5 mots) avec GenAI
    """
    word_count = len(text.split())
    
    # Condition : seulement si texte court
    if word_count >= 5:
        print(f"[EF4.1] Texte suffisamment long ({word_count} mots) - Pas d'enrichissement")
        return text
    
    print(f"[EF4.1] Texte court ({word_count} mots) - Enrichissement nécessaire")
    
    if not use_genai or not api_key:
        # Fallback sans GenAI
        print("[EF4.1] Mode sans GenAI - Enrichissement basique")
        return f"{text}. Recherche de livre avec ambiance immersive et intrigue captivante."
    
    # Appel GenAI (Google Gemini)
    try:
        import requests
        
        prompt = f"""Enrichis cette description de livre recherché en ajoutant du contexte littéraire :
"{text}"

Génère UNE phrase enrichie (max 30 mots) qui développe les thèmes, l'ambiance et le style narratif recherché."""
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent?key={api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": 150,
                "temperature": 0.8
            }
        }
        
        response = requests.post(url, json=payload, timeout=10)
        data = response.json()
        
        # Vérifier les erreurs API
        if 'error' in data:
            error_msg = data['error'].get('message', 'Erreur inconnue')
            print(f"[Erreur API Gemini] {error_msg}")
            return f"{text}. Recherche de livre avec ambiance immersive et intrigue captivante."
        
        if 'candidates' not in data or not data['candidates']:
            print(f"[Erreur] Réponse API invalide")
            return f"{text}. Recherche de livre avec ambiance immersive et intrigue captivante."
        
        enriched = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        print(f"[EF4.1] Enrichissement GenAI : {enriched}")
        return enriched
        
    except requests.exceptions.RequestException as e:
        print(f"[EF4.1] Erreur Réseau : {e} - Utilisation texte original")
        return text
    except Exception as e:
        print(f"[EF4.1] Erreur GenAI : {e} - Utilisation texte original")
        return text


# ============================================================
# EF2.3 & EF3.1 : CALCUL DE SIMILARITÉ ET SCORING
# ============================================================

def build_query_from_preferences(preferences):
    """
    Construit une requête sémantique enrichie à partir du questionnaire
    """
    parts = []
    
    # Texte principal
    if preferences.get('description'):
        parts.append(preferences['description'])
    
    # Livres préférés
    if preferences.get('favorite_books'):
        parts.append(f"Livres similaires à : {preferences['favorite_books']}")
    
    # Intensités (convertir scores Likert en descripteurs)
    intensities = {
        'intensity_action': ['calme', 'paisible', 'modéré', 'intense', 'très intense'],
        'intensity_romance': ['sans romance', 'romance légère', 'romance présente', 'romance importante', 'histoire d\'amour centrale'],
        'intensity_learning': ['divertissement pur', 'un peu éducatif', 'instructif', 'très éducatif', 'essai pédagogique'],
        'complexity': ['très simple', 'accessible', 'standard', 'complexe', 'très complexe']
    }
    
    for key, descriptors in intensities.items():
        score = preferences.get(key, 3)
        parts.append(descriptors[score - 1])
    
    query = ". ".join(parts)
    return query


def calculate_weighted_similarity(query_emb, book_emb, likert_scores):
    """
    EF3.1 : Formule de Score Pondérée
    Calcul la similarité cosinus pondérée par les préférences Likert
    """
    # EF2.3 : Similarité Cosinus de base
    base_similarity = cosine_similarity(query_emb.reshape(1, -1), book_emb.reshape(1, -1))[0][0]
    
    # EF3.1 : Pondération par intensités
    # Moyenne des scores Likert (normalisée 0-1)
    avg_intensity = np.mean([
        likert_scores.get('intensity_action', 3),
        likert_scores.get('intensity_romance', 3),
        likert_scores.get('intensity_learning', 3),
        likert_scores.get('complexity', 3)
    ]) / 5.0
    
    # Score pondéré : 80% similarité + 20% intensité préférences
    weighted_score = 0.8 * base_similarity + 0.2 * avg_intensity
    
    return weighted_score


# ============================================================
# EF3.2 : SYSTÈME DE RECOMMANDATION TOP 3
# ============================================================

def recommend_books(preferences, df, model, embeddings, top_k=3, use_genai=False, api_key=None):
    """
    EF3.2 : Recommandation des Top 3 livres
    """
    print("\n[EF3] Calcul des recommandations...")
    
    # Construire la requête
    query_text = build_query_from_preferences(preferences)
    
    # EF4.1 : Enrichissement conditionnel
    query_text = enrich_short_query(query_text, use_genai, api_key)
    
    print(f"\n[EF2.3] Requête finale : {query_text[:200]}...")
    
    # Encoder la requête
    query_emb = model.encode(query_text, convert_to_tensor=False)
    
    # Calculer les scores pondérés pour chaque livre
    scores = []
    for i in range(len(df)):
        book_emb = embeddings[i]
        score = calculate_weighted_similarity(query_emb, book_emb, preferences)
        scores.append(score)
    
    scores = np.array(scores)
    
    # EF3.2 : Top 3 recommandations
    top_indices = np.argsort(-scores)[:top_k]
    
    recommendations = []
    for rank, idx in enumerate(top_indices, 1):
        rec = {
            'rank': rank,
            'title': df.iloc[idx]['Title'],
            'genre': df.iloc[idx]['Category'],
            'description': df.iloc[idx]['Book_Description'] if pd.notna(df.iloc[idx]['Book_Description']) else "Description non disponible",
            'similarity_score': float(scores[idx])
        }
        recommendations.append(rec)
    
    return recommendations, query_text


# ============================================================
# EF4.2 & EF4.3 : GÉNÉRATION SYNTHÈSE GENAI
# ============================================================

def generate_personalized_summary(preferences, recommendations, query_text, api_key=None):
    """
    EF4.2 : Plan de Progression / Recommandation détaillée
    EF4.3 : Synthèse Executive
    UN SEUL APPEL API
    """
    if not api_key:
        return "[GenAI désactivé - Aucune clé API fournie]"
    
    print("\n[EF4.2-4.3] Génération de la synthèse personnalisée (GenAI)...")
    
    try:
        import requests
        
        # Préparer le contexte
        top_book = recommendations[0]
        
        prompt = f"""Tu es un conseiller littéraire. Analyse ce profil 

PROFIL : {query_text}

LIVRE : {top_book['title']} ({top_book['genre']})
{top_book['description'][:200]}

TÂCHE (120 mots max) :
1. Pourquoi le premier livre de la liste correspond(mentionne son nom, 3 phrases)
2. 2 aspects clés couverts
3. Suggestion de 2 livres similaires
  
Réponse concise et directe."""
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent?key={api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": 512,
                "temperature": 0.7,
                "topP": 0.9,
                "topK": 40
            }
        }
        
        response = requests.post(url, json=payload, timeout=45)
        data = response.json()
        
        # Vérifier les erreurs API
        if 'error' in data:
            error_msg = data['error'].get('message', 'Erreur inconnue')
            print(f"[Erreur API Gemini] {error_msg}")
            return f"[Erreur GenAI : {error_msg}]"
        
        if 'candidates' not in data or not data['candidates']:
            print(f"[Erreur] Réponse API invalide : {data}")
            return "[Erreur GenAI : Réponse vide ou invalide de l'API]"
        
        # Vérifier la raison de fin
        candidate = data["candidates"][0]
        finish_reason = candidate.get("finishReason", "UNKNOWN")
        
        if finish_reason in ["MAX_TOKENS", "STOP", "SAFETY", "RECITATION"]:
            if finish_reason == "MAX_TOKENS":
                print(f"[Avertissement] Réponse tronquée - Limite de tokens atteinte")
            elif finish_reason != "STOP":
                print(f"[Avertissement] Génération arrêtée : {finish_reason}")
        
        # Concaténer TOUTES les parts (pas seulement la première)
        parts = candidate["content"]["parts"]
        summary = "".join(part.get("text", "") for part in parts if "text" in part).strip()
        
        if not summary:
            print(f"[Erreur] Aucun texte dans la réponse : {data}")
            return "[Erreur GenAI : Réponse vide]"
        
        print(f"[OK] Synthèse générée ({len(summary)} caractères, finishReason: {finish_reason})")
        return summary
        
    except requests.exceptions.RequestException as e:
        print(f"[Erreur Réseau] {e}")
        return f"[Erreur GenAI - Réseau : {e}]"
    except Exception as e:
        return f"[Erreur GenAI : {e}]"


# ============================================================
# AFFICHAGE DES RÉSULTATS
# ============================================================

def display_results(preferences, recommendations, summary):
    """
    Affiche les résultats de manière structurée
    """
    print("\n" + "="*80)
    print("           RESULTATS DE L'ANALYSE SEMANTIQUE")
    print("="*80)
    
    # Synthèse GenAI
    if summary:
        print("\n[SYNTHESE PERSONNALISEE - GenAI]")
        print("-"*80)
        print(summary)
        print()
    
    # Top 3 Recommandations
    print("\n" + "="*80)
    print("[TOP 3 RECOMMANDATIONS]")
    print("="*80)
    
    for rec in recommendations:
        print(f"\n{rec['rank']}. {rec['title']}")
        print(f"   Genre      : {rec['genre']}")
        print(f"   Score      : {rec['similarity_score']:.4f}")
        print(f"   Résumé     : {rec['description'][:200]}...")
        print("-"*80)
    
    # Scores détaillés
    print("\n[SCORES DE COUVERTURE SEMANTIQUE]")
    print("-"*80)
    for rec in recommendations:
        coverage = rec['similarity_score'] * 100
        print(f"{rec['title'][:50]:50} | {coverage:5.2f}%")


# ============================================================
# PIPELINE COMPLET
# ============================================================

def main():
    """
    Pipeline complet du système de recommandation
    """
    print("\n" + "="*80)
    print("  SYSTEME DE RECOMMANDATION LITTERAIRE - SBERT + GenAI")
    print("  Projet EFREI M1 Data Engineering - IA Générative")
    print("="*80)
    
    # Configuration GenAI (optionnel)
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    USE_GENAI = GEMINI_API_KEY is not None
    
    if USE_GENAI:
        print("\n[CONFIG] GenAI activé (Google Gemini)")
    else:
        print("\n[CONFIG] Mode sans GenAI (scoring pur)")
    
    # EF1 : Acquisition des données
    preferences = collect_user_preferences()
    
    # EF2.1 : Chargement du référentiel
    df = load_knowledge_base()
    
    # EF2.2 : Embeddings SBERT
    model, embeddings = load_sbert_and_embeddings(df)
    
    # EF3 : Recommandations
    recommendations, query_text = recommend_books(
        preferences, df, model, embeddings,
        top_k=3,
        use_genai=USE_GENAI,
        api_key=GEMINI_API_KEY
    )
    
    # EF4.2-4.3 : Synthèse GenAI
    summary = generate_personalized_summary(
        preferences, recommendations, query_text,
        api_key=GEMINI_API_KEY
    )
    
    # Affichage
    display_results(preferences, recommendations, summary)
    
    # Sauvegarde résultats
    results = {
        'preferences': preferences,
        'recommendations': recommendations,
        'summary': summary,
        'timestamp': datetime.now().isoformat()
    }
    
    with open('recommendation_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\n[OK] Résultats sauvegardés dans recommendation_results.json")
    print("\n" + "="*80)
    print("[SUCCESS] ANALYSE TERMINEE")
    print("="*80)


# ============================================================
# LANCEMENT
# ============================================================

if __name__ == "__main__":
    main()
