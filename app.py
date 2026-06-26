# -*- coding: utf-8 -*-
"""
Interface Web - Système de Recommandation Littéraire
Flask App - Questionnaire + Résultats
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import os
import re
import time
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

from data_cleaning import load_and_clean_dataset

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Variables globales
model = None
embeddings = None
df = None
axis_embeddings = None

# Phrases-ancres encodées une fois au démarrage pour estimer, par similarité
# cosinus, à quel point un livre est intense sur chaque axe Likert du
# questionnaire (utilisé pour le graphique radar "profil souhaité vs livre").
AXIS_DESCRIPTORS = {
    'action': "intensité d'action et de suspense très intense",
    'romance': "histoire d'amour centrale et romance passionnée",
    'learning': "essai pédagogique très éducatif et instructif",
    'complexity': "récit littéraire très complexe et sophistiqué",
}

def init_system():
    """Initialise le système au démarrage"""
    global model, embeddings, df, axis_embeddings

    print("[INIT] Chargement du système...")

    # Fusion (Book_Dataset_1 + BooksDatasetClean + Best_Books_Ever) et nettoyage,
    # mis en cache par data_cleaning.py (cleaned_books_cache.pkl)
    df = load_and_clean_dataset()

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

    axis_embeddings = {
        axis: model.encode(text, convert_to_tensor=False)
        for axis, text in AXIS_DESCRIPTORS.items()
    }

    print(f"[OK] Système initialisé - {len(df)} livres prêts")
    return model, embeddings, df


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


def calculate_weighted_similarity(query_emb, book_embeddings, likert_scores):
    """
    Calcule le score pondéré pour tout le corpus en une seule fois (vectorisé).
    Avec ~120k livres après fusion des 3 sources, une boucle Python par livre
    serait beaucoup trop lente pour rester utilisable dans une requête web.
    """
    base_similarity = cosine_similarity(query_emb.reshape(1, -1), book_embeddings)[0]

    avg_intensity = np.mean([
        likert_scores.get('intensity_action', 3),
        likert_scores.get('intensity_romance', 3),
        likert_scores.get('intensity_learning', 3),
        likert_scores.get('complexity', 3)
    ]) / 5.0

    weighted_score = 0.8 * base_similarity + 0.2 * avg_intensity
    return weighted_score


def compute_axis_profile(book_embedding):
    """
    Estime, pour un livre, à quel point il est "intense" sur chacun des 4 axes
    Likert du questionnaire (action, romance, learning, complexity), par
    similarité cosinus avec une phrase-ancre par axe. Sert au graphique radar
    "profil souhaité vs profil du livre" côté interface.

    L'échelle de calibration (-0.1 à 0.4) vient d'une mesure empirique des
    similarités cosinus obtenues sur ce modèle/corpus ; ce n'est pas une
    probabilité, juste une mise à l'échelle 0-100 pour l'affichage.
    """
    profile = {}
    for axis, axis_emb in axis_embeddings.items():
        sim = cosine_similarity(axis_emb.reshape(1, -1), book_embedding.reshape(1, -1))[0][0]
        scaled = np.clip((sim + 0.1) / 0.5, 0, 1) * 100
        profile[axis] = round(float(scaled), 1)
    return profile


def format_value(value, suffix=""):
    """Formate une valeur potentiellement manquante (NaN/None/vide) pour l'affichage."""
    if value is None:
        return "N/A"
    if isinstance(value, float):
        if np.isnan(value):
            return "N/A"
        if value == int(value):
            value = int(value)  # 2008.0 / 374.0 -> "2008" / "374"
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return "N/A"
    return f"{text}{suffix}"


def format_genre(genre_clean):
    """Transforme un genre macro ('fantasy_scifi') en libellé lisible ('Fantasy Scifi')."""
    if not genre_clean or genre_clean in ("unknown", "other"):
        return "Genre non classé"
    return genre_clean.replace("_", " ").title()


def get_already_read_indices(df, favorite_books_text):
    """
    Identifie les livres déjà mentionnés par l'utilisateur dans "livres préférés"
    pour qu'ils ne soient pas re-suggérés dans le Top 3 (un livre déjà lu n'est
    pas une recommandation utile).
    """
    if not favorite_books_text or not favorite_books_text.strip():
        return set()

    favorite_titles = [t.strip().lower() for t in favorite_books_text.split(',') if t.strip()]
    if not favorite_titles:
        return set()

    # Ancré en début de titre (pas une simple sous-chaîne) pour éviter les faux
    # positifs comme "Twilight" qui matcherait aussi "Twilight of the Elites"
    title_norm = df['title'].astype(str).str.strip().str.lower()
    excluded = set()
    for fav in favorite_titles:
        pattern = r'^' + re.escape(fav) + r'(\s*[\(:,\-].*)?$'
        matches = title_norm[title_norm.str.match(pattern, na=False)]
        excluded.update(matches.index.tolist())

    return excluded


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
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": 512,
                "temperature": 0.7,
                "topP": 0.9,
                # Gemini 2.5 consomme sinon le budget de tokens en "réflexion"
                # interne (thoughtsTokenCount) avant d'écrire la réponse visible,
                # ce qui tronquait le résumé à quelques mots (finishReason=MAX_TOKENS).
                "thinkingConfig": {"thinkingBudget": 0}
            }
        }

        response = requests.post(url, json=payload, timeout=30)
        data = response.json()

        if 'error' in data:
            return f"[Erreur API : {data['error'].get('message', 'Erreur inconnue')}]"

        if 'candidates' in data and data['candidates']:
            candidate = data["candidates"][0]
            finish_reason = candidate.get("finishReason", "UNKNOWN")
            parts = candidate["content"]["parts"]
            summary = "".join(part.get("text", "") for part in parts if "text" in part).strip()

            if not summary:
                return f"[Erreur GenAI : réponse vide (finishReason={finish_reason})]"
            if finish_reason == "MAX_TOKENS":
                print(f"[Avertissement] Synthèse tronquée (MAX_TOKENS) - {len(summary)} caractères")

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
    t_start = time.time()
    try:
        print(f"\n{'='*70}")
        print(f"[REQUEST] Nouvelle demande de recommandation - {datetime.now().strftime('%H:%M:%S')}")

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
        print(f"[1/5] Préférences reçues : description={len(preferences['description'])} car., "
              f"favoris=\"{preferences['favorite_books']}\"")

        # Construire la requête
        query_text = build_query_from_preferences(preferences)
        print(f"[2/5] Requête sémantique construite : \"{query_text[:100]}...\"")

        # Encoder la requête
        t0 = time.time()
        query_emb = model.encode(query_text, convert_to_tensor=False)
        print(f"[3/5] Requête encodée par SBERT ({time.time() - t0:.3f}s)")

        # Calculer les scores (vectorisé sur tout le corpus fusionné)
        t0 = time.time()
        scores = calculate_weighted_similarity(query_emb, embeddings, preferences)
        print(f"[4/5] Similarité calculée sur {len(df)} livres ({time.time() - t0:.3f}s)")

        # Exclure les livres déjà lus (mentionnés comme "livres préférés")
        already_read = get_already_read_indices(df, preferences['favorite_books'])
        if already_read:
            print(f"[4/5] {len(already_read)} livre(s) déjà lu(s) exclu(s) du classement : "
                  f"{', '.join(df.loc[list(already_read), 'title'].head(5))}")
            scores = scores.copy()
            scores[list(already_read)] = -np.inf

        # Top 5
        TOP_K = 5
        top_indices = np.argsort(-scores)[:TOP_K]
        top_n = [(idx, scores[idx]) for idx in top_indices]
        print(f"[5/5] Top {TOP_K} : {', '.join(df.iloc[idx]['title'] for idx, _ in top_n)}")

        recommendations = []
        for rank, (idx, score) in enumerate(top_n, 1):
            book = df.iloc[idx]
            recommendations.append({
                'rank': rank,
                'title': book['title'],
                'author': format_value(book['authors_clean']),
                'genre': format_genre(book['genre_clean']),
                'year': format_value(book['year']),
                'series': format_value(book['series']),
                'language': format_value(book['language']),
                'pages': format_value(book['pages']),
                'rating': format_value(book['rating']),
                'score': float(score * 100),
                'description': book['desc_display'],  # Utiliser la version tronquée
                'axis_profile': compute_axis_profile(embeddings[idx]),
            })

        # Profil souhaité par l'utilisateur (mêmes axes, échelle 0-100) pour le radar
        user_profile = {
            'action': preferences['intensity_action'] / 5 * 100,
            'romance': preferences['intensity_romance'] / 5 * 100,
            'learning': preferences['intensity_learning'] / 5 * 100,
            'complexity': preferences['complexity'] / 5 * 100,
        }


        # Générer la synthèse GenAI
        t0 = time.time()
        summary = generate_genai_summary(preferences, recommendations, query_text)
        if summary.startswith('['):
            print(f"[GenAI] Pas de synthèse générée ({time.time() - t0:.2f}s) : {summary}")
        else:
            print(f"[GenAI] Synthèse générée ({len(summary)} car., {time.time() - t0:.2f}s)")

        # Sauvegarder les préférences utilisateur (historique)
        import json
        preferences_history = []
        if os.path.exists('user_preferences.json'):
            try:
                with open('user_preferences.json', 'r', encoding='utf-8') as f:
                    preferences_history = json.load(f)
                    if not isinstance(preferences_history, list):
                        preferences_history = [preferences_history]
            except:
                preferences_history = []
        
        preferences_history.append(preferences)
        
        with open('user_preferences.json', 'w', encoding='utf-8') as f:
            json.dump(preferences_history, f, indent=2, ensure_ascii=False)
        
        # Sauvegarder les résultats complets (historique)
        results_history = []
        if os.path.exists('recommendation_results.json'):
            try:
                with open('recommendation_results.json', 'r', encoding='utf-8') as f:
                    results_history = json.load(f)
                    if not isinstance(results_history, list):
                        results_history = [results_history]
            except:
                results_history = []
        
        current_result = {
            'preferences': preferences,
            'recommendations': recommendations,
            'summary': summary,
            'query_text': query_text,
            'timestamp': datetime.now().isoformat()
        }
        results_history.append(current_result)
        
        with open('recommendation_results.json', 'w', encoding='utf-8') as f:
            json.dump(results_history, f, indent=2, ensure_ascii=False)
        
        print(f"[OK] Fichiers JSON sauvegardés (session #{len(results_history)})")
        print(f"[DONE] Requête traitée en {time.time() - t_start:.2f}s")
        print(f"{'='*70}")

        return jsonify({
            'success': True,
            'recommendations': recommendations,
            'summary': summary,
            'query_text': query_text,
            'user_profile': user_profile
        })

    except Exception as e:
        print(f"[ERREUR] Échec de la requête après {time.time() - t_start:.2f}s : {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    init_system()
    app.run(debug=True, host='0.0.0.0', port=5000)
