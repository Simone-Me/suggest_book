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
title_norm_index = None

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
    global model, embeddings, df, axis_embeddings, title_norm_index

    print("[INIT] Chargement du système...")

    # Fusion (Book_Dataset_1 + BooksDatasetClean + Best_Books_Ever) et nettoyage,
    # mis en cache par data_cleaning.py (cleaned_books_cache.pkl)
    df = load_and_clean_dataset()

    # Charger SBERT
    # Variante multilingue (vs all-MiniLM-L6-v2) : nécessaire car le corpus et
    # les requêtes utilisateur ne sont pas tous en anglais (interface en
    # français, ~5 500 livres confirmés non-anglais + langue inconnue pour
    # 60% du corpus). all-MiniLM-L6-v2 est entraîné quasi exclusivement sur
    # des paires anglaises et ne sait pas bien aligner une requête française
    # avec des descriptions anglaises.
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

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

    # Index pour la vérification anti-hallucination des titres suggérés par
    # le GenAI (cf. evaluate_genai_quality) : correspondance exacte normalisée.
    title_norm_index = set(df['title'].astype(str).str.strip().str.lower())

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

Termine impérativement ta réponse par une dernière ligne au format exact
(utilisée pour vérifier automatiquement tes suggestions, ne pas l'omettre) :
LIVRES_SIMILAIRES: Titre1 | Titre2

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


def extract_suggested_titles(raw_summary):
    """
    Sépare la ligne structurée "LIVRES_SIMILAIRES: Titre1 | Titre2" (demandée
    dans le prompt) du texte affiché à l'utilisateur, pour pouvoir vérifier
    ces titres automatiquement (cf. evaluate_genai_quality).
    """
    match = re.search(r'LIVRES_SIMILAIRES\s*:\s*(.+)', raw_summary, re.IGNORECASE)
    if not match:
        return raw_summary.strip(), []

    cleaned = raw_summary[:match.start()].strip()
    titles = [t.strip() for t in match.group(1).split('|') if t.strip()]
    return cleaned, titles


def evaluate_genai_quality(summary_text, suggested_titles, top_book_embedding):
    """
    Évalue la synthèse GenAI selon 2 axes mesurables (en l'absence de "bonne
    réponse" prédéfinie pour un texte généré) :
    - cohérence sémantique : similarité cosinus entre l'embedding du résumé
      et celui du livre Top 1 qu'il est censé décrire (un résumé qui dérive
      du sujet aurait une similarité faible) ;
    - anti-hallucination : les "livres similaires" suggérés par Gemini
      existent-ils réellement dans le référentiel (~120k titres), par
      correspondance exacte normalisée (minuscules, espaces nettoyés) ?
    """
    if not summary_text or summary_text.startswith('['):
        return None

    summary_embedding = model.encode(summary_text, convert_to_tensor=False)
    coherence = float(
        cosine_similarity(
            summary_embedding.reshape(1, -1),
            top_book_embedding.reshape(1, -1),
        )[0][0]
    )

    verified = [t for t in suggested_titles if t.strip().lower() in title_norm_index]

    return {
        'coherence_score': round(coherence * 100, 1),
        'suggested_titles': suggested_titles,
        'titles_verified_count': len(verified),
        'titles_total_count': len(suggested_titles),
        'hallucination_rate': (
            round((1 - len(verified) / len(suggested_titles)) * 100, 1)
            if suggested_titles else None
        ),
    }


@app.route('/')
def index():
    """Page principale avec le formulaire"""
    return render_template('index.html')


@app.route('/kpi')
def kpi():
    """
    Dashboard KPI : qualité de séparabilité sémantique des genres (embeddings
    SBERT), généré par analysis_improved.py (accuracy/precision/recall/F1 +
    matrice de confusion). Ne mesure pas la pertinence des recommandations
    elles-mêmes (pas de "bonne réponse" prédéfinie pour ça), mais la qualité
    du signal sémantique sous-jacent.
    """
    import json

    if not os.path.exists('kpi_results.json'):
        return render_template('kpi.html', kpi=None)

    with open('kpi_results.json', 'r', encoding='utf-8') as f:
        kpi_data = json.load(f)

    # Normalisation par ligne (= par support réel) pour la heatmap : montre
    # la répartition des prédictions pour un genre réel donné, en %.
    matrix = kpi_data['confusion_matrix']
    matrix_normalized = []
    for row in matrix:
        total = sum(row) or 1
        matrix_normalized.append([round(v / total * 100, 1) for v in row])
    kpi_data['confusion_matrix_normalized'] = matrix_normalized

    return render_template('kpi.html', kpi=kpi_data)


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
        raw_summary = generate_genai_summary(preferences, recommendations, query_text)
        summary, suggested_titles = extract_suggested_titles(raw_summary)

        genai_quality = None
        if summary.startswith('['):
            print(f"[GenAI] Pas de synthèse générée ({time.time() - t0:.2f}s) : {summary}")
        else:
            top_idx = top_n[0][0]
            genai_quality = evaluate_genai_quality(summary, suggested_titles, embeddings[top_idx])
            print(f"[GenAI] Synthèse générée ({len(summary)} car., {time.time() - t0:.2f}s) - "
                  f"cohérence={genai_quality['coherence_score']}%, "
                  f"titres suggérés vérifiés={genai_quality['titles_verified_count']}/{genai_quality['titles_total_count']}")

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
            'genai_quality': genai_quality,
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
            'genai_quality': genai_quality,
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
