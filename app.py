# -*- coding: utf-8 -*-
"""
Interface Web - Système de Recommandation Littéraire
Flask App - Questionnaire + Résultats
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Variables globales
model = None
embeddings = None
df = None

def init_system():
    """Initialise le système au démarrage"""
    global model, embeddings, df
    
    print("[INIT] Chargement du système...")
    
    # Charger le dataset
    df = pd.read_csv("Book_Dataset_1.csv", sep=',', encoding='latin1')
    df = df.drop_duplicates()
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
    df['text_full'] = (
        df['title_clean'] + ". " +
        df['genre_clean'] + ". " +
        df['desc_clean']
    )
    
    # Charger SBERT
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Charger embeddings
    if os.path.exists("embeddings_books.pkl"):
        with open("embeddings_books.pkl", 'rb') as f:
            embeddings = pickle.load(f)
        embeddings = np.array(embeddings)
    else:
        print("[INIT] Génération des embeddings (première fois)...")
        embeddings = model.encode(
            df['text_full'].tolist(),
            convert_to_tensor=False,
            show_progress_bar=True,
            batch_size=32
        )
        with open("embeddings_books.pkl", 'wb') as f:
            pickle.dump(embeddings, f)
        embeddings = np.array(embeddings)
    
    print(f"[OK] Système initialisé - {len(df)} livres prêts")


def build_query_from_preferences(preferences):
    """Construit la requête sémantique"""
    parts = []
    
    if preferences.get('description'):
        parts.append(preferences['description'])
    
    if preferences.get('favorite_books'):
        parts.append(f"Livres similaires à : {preferences['favorite_books']}")
    
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
    """Calcule le score pondéré"""
    base_similarity = cosine_similarity(query_emb.reshape(1, -1), book_emb.reshape(1, -1))[0][0]
    
    avg_intensity = np.mean([
        likert_scores.get('intensity_action', 3),
        likert_scores.get('intensity_romance', 3),
        likert_scores.get('intensity_learning', 3),
        likert_scores.get('complexity', 3)
    ]) / 5.0
    
    weighted_score = 0.8 * base_similarity + 0.2 * avg_intensity
    return weighted_score


def generate_genai_summary(preferences, recommendations, query_text):
    """Génère la synthèse avec Gemini"""
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        return "[GenAI désactivé - Clé API non configurée. Définissez GEMINI_API_KEY dans les variables d'environnement]"
    
    try:
        import requests
        
        top_book = recommendations[0]
        
        prompt = f"""Tu es un conseiller littéraire. Analyse ce profil et recommande le livre.

PROFIL : {query_text}

LIVRE : {top_book['title']} ({top_book['genre']})
{top_book['description'][:200]}

TÂCHE (120 mots max) :
1. Pourquoi ce livre correspond (3 phrases)
2. 2 aspects clés couverts
3. Suggestion de 2 livres similaires

Réponse concise et directe."""
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent?key={api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": 512,
                "temperature": 0.7,
                "topP": 0.9
            }
        }
        
        response = requests.post(url, json=payload, timeout=30)
        data = response.json()
        
        if 'error' in data:
            return f"[Erreur API : {data['error'].get('message', 'Erreur inconnue')}]"
        
        if 'candidates' in data and data['candidates']:
            candidate = data["candidates"][0]
            parts = candidate["content"]["parts"]
            summary = "".join(part.get("text", "") for part in parts if "text" in part).strip()
            return summary
        
        return "[Erreur : Réponse API invalide]"
        
    except Exception as e:
        return f"[Erreur GenAI : {str(e)}]"


@app.route('/')
def index():
    """Page principale avec le formulaire"""
    return render_template('index.html')


@app.route('/recommend', methods=['POST'])
def recommend():
    """Traite le questionnaire et retourne les recommandations"""
    try:
        # Récupérer les données du formulaire
        preferences = {
            'description': request.form.get('description', ''),
            'favorite_books': request.form.get('favorite_books', ''),
            'avoid': request.form.get('avoid', ''),
            'intensity_action': int(request.form.get('intensity_action', 3)),
            'intensity_romance': int(request.form.get('intensity_romance', 3)),
            'intensity_learning': int(request.form.get('intensity_learning', 3)),
            'complexity': int(request.form.get('complexity', 3)),
            'timestamp': datetime.now().isoformat()
        }
        
        # Construire la requête
        query_text = build_query_from_preferences(preferences)
        
        # Encoder la requête
        query_emb = model.encode(query_text, convert_to_tensor=False)
        
        # Calculer les scores
        scores = []
        for i in range(len(df)):
            book_emb = embeddings[i]
            score = calculate_weighted_similarity(query_emb, book_emb, preferences)
            scores.append(score)
        
        scores = np.array(scores)
        
        # Top 3
        top_indices = np.argsort(-scores)[:3]
        
        recommendations = []
        for rank, idx in enumerate(top_indices, 1):
            rec = {
                'rank': rank,
                'title': df.iloc[idx]['Title'],
                'genre': df.iloc[idx]['Category'],
                'description': df.iloc[idx]['Book_Description'] if pd.notna(df.iloc[idx]['Book_Description']) else "Description non disponible",
                'similarity_score': float(scores[idx]),
                'percentage': float(scores[idx] * 100)
            }
            recommendations.append(rec)
        
        # Générer la synthèse GenAI
        summary = generate_genai_summary(preferences, recommendations, query_text)
        
        return jsonify({
            'success': True,
            'recommendations': recommendations,
            'summary': summary,
            'query_text': query_text
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    init_system()
    app.run(debug=True, host='0.0.0.0', port=5000)
