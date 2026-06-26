"""
Module de nettoyage et de fusion des données pour le système de recommandation de livres.
Sépare la logique de nettoyage du mécanisme de scoring.

Trois sources brutes sont fusionnées pour maximiser la diversité du référentiel :
- Book_Dataset_1.csv     (catalogue générique, ~1000 livres, pas d'auteur/année)
- BooksDatasetClean.csv  (catalogue éditeur, ~103k livres, taxonomie multi-niveaux)
- Best_Books_Ever.csv    (catalogue Goodreads, ~52k livres, séries/langue/pages/notes)

Schéma commun après fusion : title, authors, description, year, series, language,
pages, rating, genres_raw, source.
"""

import sys
import os
import re
import ast
import pickle

import numpy as np
import pandas as pd
from ftfy import fix_text
from sklearn.feature_extraction.text import TfidfVectorizer

# Fix Windows console encoding (no-op outside a real console, e.g. Jupyter)
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


# Regroupement des centaines de catégories/genres bruts (premier segment ou
# premier élément des trois taxonomies sources) en un petit nombre de genres
# macro exploitables pour le moteur sémantique et le référentiel de livres.
CATEGORY_GROUPS = {
    # Fiction générale
    'fiction': 'fiction', 'novels': 'fiction', 'classics': 'fiction',
    'historical fiction': 'fiction', 'contemporary': 'fiction',
    'literary fiction': 'fiction', 'womens fiction': 'fiction',
    'adult fiction': 'fiction', 'short stories': 'fiction',

    # Fantasy / Science-fiction
    'fantasy': 'fantasy_scifi', 'science fiction': 'fantasy_scifi',
    'paranormal': 'fantasy_scifi', 'urban fantasy': 'fantasy_scifi',
    'science fiction fantasy': 'fantasy_scifi', 'high fantasy': 'fantasy_scifi',
    'epic fantasy': 'fantasy_scifi', 'supernatural': 'fantasy_scifi',
    'dystopia': 'fantasy_scifi',

    # Mystère / Thriller
    'mystery': 'mystery_thriller', 'thriller': 'mystery_thriller',
    'crime': 'mystery_thriller', 'suspense': 'mystery_thriller',
    'true crime': 'mystery_thriller', 'detective': 'mystery_thriller',

    # Romance
    'romance': 'romance', 'historical romance': 'romance',
    'paranormal romance': 'romance', 'chick lit': 'romance',
    'new adult': 'romance', 'm m romance': 'romance',

    # Horreur
    'horror': 'horror', 'vampires': 'horror', 'ghost': 'horror', 'gothic': 'horror',

    # Jeunesse
    'young adult': 'young_adult', 'young adult fiction': 'young_adult',
    'young adult nonfiction': 'young_adult', 'juvenile fiction': 'young_adult',
    'juvenile nonfiction': 'young_adult', 'childrens': 'young_adult',
    "children's fiction": 'young_adult', 'picture books': 'young_adult',
    'middle grade': 'young_adult', 'teen': 'young_adult',

    # Histoire / Biographie
    'history': 'history_biography', 'historical': 'history_biography',
    'biography & autobiography': 'history_biography', 'biography': 'history_biography',
    'memoir': 'history_biography', 'autobiography': 'history_biography',

    # Sciences / Académique
    'science': 'science_academic', 'medical': 'science_academic',
    'technology & engineering': 'science_academic', 'computers': 'science_academic',
    'mathematics': 'science_academic', 'psychology': 'science_academic',
    'social science': 'science_academic', 'political science': 'science_academic',
    'education': 'science_academic', 'language arts & disciplines': 'science_academic',
    'study aids': 'science_academic', 'law': 'science_academic',
    'nature': 'science_academic', 'business & economics': 'science_academic',
    'reference': 'science_academic', 'foreign language study': 'science_academic',

    # Lifestyle / pratique
    'cooking': 'lifestyle', 'health & fitness': 'lifestyle', 'self-help': 'lifestyle',
    'self help': 'lifestyle', 'family & relationships': 'lifestyle',
    'sports & recreation': 'lifestyle', 'crafts & hobbies': 'lifestyle',
    'house & home': 'lifestyle', 'pets': 'lifestyle', 'gardening': 'lifestyle',
    'games': 'lifestyle', 'travel': 'lifestyle', 'antiques & collectibles': 'lifestyle',
    'body': 'lifestyle', 'games & activities': 'lifestyle',

    # Spiritualité / Philosophie
    'religion': 'spirituality_philosophy', 'christian': 'spirituality_philosophy',
    'christian fiction': 'spirituality_philosophy', 'bibles': 'spirituality_philosophy',
    'philosophy': 'spirituality_philosophy', 'spirituality': 'spirituality_philosophy',

    # Arts / Poésie
    'art': 'arts_poetry', 'photography': 'arts_poetry', 'performing arts': 'arts_poetry',
    'music': 'arts_poetry', 'poetry': 'arts_poetry', 'drama': 'arts_poetry',
    'plays': 'arts_poetry', 'literary criticism': 'arts_poetry',
    'literary collections': 'arts_poetry', 'design': 'arts_poetry',
    'architecture': 'arts_poetry',

    # Bande dessinée / manga
    'comics': 'sequential_art', 'graphic novels': 'sequential_art',
    'manga': 'sequential_art', 'comics & graphic novels': 'sequential_art',
    'sequential art': 'sequential_art',

    # Humour
    'humor': 'humor',
}


def clean_text(text):
    """
    Nettoie et normalise le texte avec gestion des encodages.

    Args:
        text: Texte à nettoyer

    Returns:
        Texte nettoyé et normalisé (minuscules, espaces supprimés, encodage corrigé)
    """
    if pd.isna(text):
        return ""

    text = str(text)

    # Réparation automatique des encodages cassés
    text = fix_text(text)

    # Normalisation basique
    text = text.strip().lower()

    # Espaces multiples
    text = re.sub(r"\s+", " ", text)

    return text


def clean_authors(text):
    """
    Nettoie le champ auteur des trois sources :
    - "By Lastname, Firstname" (BooksDatasetClean)
    - "Firstname Lastname (Illustrator)" (Best_Books_Ever, annotations de rôle)
    """
    if pd.isna(text):
        return ""

    text = fix_text(str(text)).strip()

    if text.lower().startswith("by "):
        text = text[3:].strip()

    # Retrait des annotations de rôle entre parenthèses (Illustrator), (Goodreads Author)...
    text = re.sub(r"\([^)]*\)", "", text)
    text = re.sub(r"\s+", " ", text).strip().strip(",").strip()

    return text


def _extract_year(text):
    """Extrait une année à 4 chiffres (1500-2099) d'une chaîne de date libre."""
    if pd.isna(text):
        return np.nan

    match = re.search(r"(1[5-9]\d{2}|20\d{2})", str(text))
    return float(match.group(0)) if match else np.nan


def _extract_pages(text):
    """Extrait le nombre de pages d'une chaîne libre (ex: '374 pages')."""
    if pd.isna(text):
        return np.nan

    match = re.search(r"\d+", str(text))
    return float(match.group(0)) if match else np.nan


def _parse_genres_list(text):
    """
    Convertit le champ 'genres' de Best_Books_Ever (chaîne de liste Python,
    ex: "['Young Adult', 'Fiction']") en chaîne "genre1, genre2, ..." compatible
    avec group_category(), qui s'appuie sur le premier élément.
    """
    if pd.isna(text):
        return np.nan

    text = str(text).strip()
    if text.startswith("["):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list) and parsed:
                return ", ".join(str(p) for p in parsed)
        except (ValueError, SyntaxError):
            pass
        return np.nan

    return text


def group_category(raw_category):
    """
    Regroupe une catégorie/genre brut (premier segment d'une taxonomie multi-niveaux
    ou d'une liste de genres) en un genre macro défini dans CATEGORY_GROUPS.
    Renvoie 'other' si non reconnu, 'unknown' si la catégorie est manquante.
    """
    if pd.isna(raw_category) or str(raw_category).strip() == "":
        return "unknown"

    first_segment = str(raw_category).split(",")[0].strip().lower()
    return CATEGORY_GROUPS.get(first_segment, "other")


def bucket_period(year):
    """
    Transforme une année de publication en période littéraire (décennie).
    """
    if pd.isna(year):
        return "Inconnue"

    year = int(year)
    if year < 1900:
        return "Avant 1900"

    decade = (year // 10) * 10
    return f"{decade}s"


def truncate_description(text, max_words=150):
    """
    Tronque un texte à un nombre maximum de mots (affichage ou embedding).
    """
    if pd.isna(text) or not text:
        return ""

    words = str(text).split()
    if len(words) <= max_words:
        return text

    return " ".join(words[:max_words]) + "..."


def extract_keywords(df, text_col="desc_clean", top_n=5):
    """
    Extrait les mots-clés les plus représentatifs de chaque description via TF-IDF
    calculé sur l'ensemble du corpus (le champ 'Keywords' attendu par le référentiel
    n'existe nativement dans aucune des trois sources).

    Args:
        df: DataFrame contenant la colonne text_col
        text_col: colonne de texte nettoyé à utiliser
        top_n: nombre de mots-clés à conserver par livre

    Returns:
        pd.Series alignée sur l'index de df, une chaîne "mot1, mot2, ..." par ligne
    """
    texts = df[text_col].fillna("")
    non_empty_mask = texts.str.strip() != ""
    keywords = pd.Series([""] * len(df), index=df.index)

    if non_empty_mask.sum() == 0:
        return keywords

    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(texts[non_empty_mask])
    feature_names = vectorizer.get_feature_names_out()

    for pos, idx in enumerate(texts[non_empty_mask].index):
        row = tfidf_matrix[pos]
        if row.nnz == 0:
            continue
        top_indices = row.toarray().flatten().argsort()[-top_n:][::-1]
        top_words = [feature_names[i] for i in top_indices if row[0, i] > 0]
        keywords.loc[idx] = ", ".join(top_words)

    return keywords


def load_book_dataset_1(path="Book_Dataset_1.csv"):
    """Charge Book_Dataset_1.csv et le ramène au schéma commun de fusion."""
    df = pd.read_csv(path, sep=",", encoding="latin1")
    return pd.DataFrame({
        "title": df["Title"],
        "authors": np.nan,
        "description": df["Book_Description"],
        "year": np.nan,
        "series": np.nan,
        "language": np.nan,
        "pages": np.nan,
        "rating": pd.to_numeric(df["Stars"], errors="coerce"),
        "genres_raw": df["Category"],
        "source": "book_dataset_1",
    })


def load_books_dataset_clean(path="BooksDatasetClean.csv"):
    """Charge BooksDatasetClean.csv et le ramène au schéma commun de fusion."""
    df = pd.read_csv(path, encoding="utf-8")
    return pd.DataFrame({
        "title": df["Title"],
        "authors": df["Authors"],
        "description": df["Description"],
        "year": df["Publish Date (Year)"],
        "series": np.nan,
        "language": np.nan,
        "pages": np.nan,
        "rating": np.nan,
        "genres_raw": df["Category"],
        "source": "books_dataset_clean",
    })


def load_best_books_ever(path="Best_Books_Ever.csv"):
    """Charge Best_Books_Ever.csv et le ramène au schéma commun de fusion."""
    df = pd.read_csv(path, encoding="utf-8", low_memory=False)

    year = df["firstPublishDate"].apply(_extract_year)
    year = year.fillna(df["publishDate"].apply(_extract_year))

    return pd.DataFrame({
        "title": df["title"],
        "authors": df["author"],
        "description": df["description"],
        "year": year,
        "series": df["series"],
        "language": df["language"],
        "pages": df["pages"].apply(_extract_pages),
        "rating": pd.to_numeric(df["rating"], errors="coerce"),
        "genres_raw": df["genres"].apply(_parse_genres_list),
        "source": "best_books_ever",
    })


def merge_datasets(
    book_dataset_1_path="Book_Dataset_1.csv",
    books_dataset_clean_path="BooksDatasetClean.csv",
    best_books_ever_path="Best_Books_Ever.csv",
):
    """
    Fusionne les trois sources brutes en un schéma commun :
    title, authors, description, year, series, language, pages, rating,
    genres_raw, source. Aucune colonne secondaire (prix, éditeur, image...)
    n'est conservée.
    """
    frames = [
        load_book_dataset_1(book_dataset_1_path),
        load_books_dataset_clean(books_dataset_clean_path),
        load_best_books_ever(best_books_ever_path),
    ]
    merged = pd.concat(frames, ignore_index=True)

    counts = merged["source"].value_counts()
    print(f"[OK] Fusion : {len(merged)} livres ({', '.join(f'{c} {s}' for s, c in counts.items())})")

    return merged


def deduplicate_books(df):
    """
    Supprime les doublons inter-sources sur la base titre + auteur.

    Un même titre peut légitimement correspondre à des livres différents écrits
    par des auteurs différents (ex: deux romans intitulés "Twilight") : dans ce
    cas les lignes sont conservées distinctes. En revanche, si le titre est
    identique et que l'auteur correspond (ou est inconnu sur l'une des sources),
    on considère qu'il s'agit du même livre référencé par plusieurs sources : la
    description la plus complète est gardée, et les champs manquants (année,
    série, langue, pages, note, genres) sont complétés à partir des autres
    occurrences du même livre.
    """
    df = df.copy()
    df["_title_norm"] = df["title"].fillna("").apply(
        lambda x: re.sub(r"\s+", " ", str(x).strip().lower())
    )
    df["_author_norm"] = df["authors"].apply(
        lambda x: str(x).strip().lower().split(",")[0] if pd.notna(x) and str(x).strip() else None
    )
    df["_desc_len"] = df["description"].fillna("").str.len()

    n_before = len(df)
    rows = []

    for _, group in df.groupby("_title_norm", sort=False):
        if len(group) == 1:
            rows.append(group.iloc[0].to_dict())
            continue

        distinct_authors = set(a for a in group["_author_norm"] if a)
        if len(distinct_authors) > 1:
            # Même titre, auteurs différents -> livres distincts, on ne fusionne pas
            rows.extend(group.to_dict("records"))
            continue

        # Même livre référencé par plusieurs sources : description la plus
        # complète + complément des champs manquants depuis les autres lignes
        best = group.loc[group["_desc_len"].idxmax()].to_dict()
        for col in ["authors", "year", "series", "language", "pages", "rating", "genres_raw"]:
            if pd.isna(best[col]) or best[col] == "":
                candidates = group[col].dropna()
                candidates = candidates[candidates != ""]
                if len(candidates) > 0:
                    best[col] = candidates.iloc[0]
        rows.append(best)

    deduped = pd.DataFrame(rows).drop(columns=["_title_norm", "_author_norm", "_desc_len"])
    print(f"[OK] Doublons fusionnés : {n_before - len(deduped)} livres ({n_before} -> {len(deduped)})")

    return deduped.reset_index(drop=True)


def load_and_clean_dataset(
    book_dataset_1_path="Book_Dataset_1.csv",
    books_dataset_clean_path="BooksDatasetClean.csv",
    best_books_ever_path="Best_Books_Ever.csv",
    embedding_max_words=200,
    cache_path="cleaned_books_cache.pkl",
    force_rebuild=False,
):
    """
    Fusionne les trois sources, nettoie et enrichit le référentiel de livres.

    Opérations effectuées :
    - Fusion des trois datasets (merge_datasets)
    - Dédoublonnage prudent titre+auteur (deduplicate_books)
    - Suppression des lignes sans titre ou sans description (indispensables au NLP)
    - Nettoyage du texte (titres, descriptions, auteurs)
    - Regroupement des catégories/genres bruts en un petit nombre de genres macro
    - Bucketing de la période de publication (décennie)
    - Extraction de mots-clés (TF-IDF) en l'absence de colonne Keywords native
    - Création d'un texte complet (tronqué) pour les embeddings SBERT

    Le résultat est mis en cache (pickle) car la fusion + le TF-IDF sont coûteux ;
    passer force_rebuild=True pour ignorer le cache après une mise à jour des CSV sources.

    Returns:
        DataFrame nettoyé avec colonnes supplémentaires :
        - title_clean, authors_clean, desc_clean, desc_display
        - genre_clean (genre macro), period, keywords
        - text_full: texte complet pour embeddings
    """
    if not force_rebuild and os.path.exists(cache_path):
        print(f"[OK] Chargement du dataset nettoyé depuis le cache ({cache_path})...")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print("\n[Nettoyage] Fusion des datasets...")
    df = merge_datasets(book_dataset_1_path, books_dataset_clean_path, best_books_ever_path)

    df = deduplicate_books(df)

    # Le titre et la description sont indispensables à l'analyse sémantique
    before_dropna = len(df)
    df = df.dropna(subset=["title", "description"])
    print(f"[OK] Lignes sans titre/description supprimées : {before_dropna - len(df)}")

    # Nettoyage du texte pour créer le corpus final
    df["title_clean"] = df["title"].apply(clean_text)
    df["authors_clean"] = df["authors"].apply(clean_authors)
    df["desc_clean"] = df["description"].apply(clean_text)
    df["genre_clean"] = df["genres_raw"].apply(group_category)
    df["period"] = df["year"].apply(bucket_period)

    # Version tronquée pour l'affichage (UI) et pour l'embedding (NLP)
    df["desc_display"] = df["description"].apply(lambda x: truncate_description(x, max_words=150))
    df["desc_embedding"] = df["desc_clean"].apply(lambda x: truncate_description(x, max_words=embedding_max_words))

    # Mots-clés (champ requis par le référentiel, absent des trois sources)
    print("[OK] Extraction des mots-clés (TF-IDF)...")
    df["keywords"] = extract_keywords(df, text_col="desc_clean", top_n=5)

    # Texte complet pour les embeddings SBERT
    df["text_full"] = (
        df["title_clean"] + ". " +
        df["genre_clean"] + ". " +
        df["desc_embedding"]
    )

    df = df.reset_index(drop=True)
    print(f"[OK] {len(df)} livres nettoyés et prêts")

    with open(cache_path, "wb") as f:
        pickle.dump(df, f)
    print(f"[OK] Dataset nettoyé mis en cache ({cache_path})")

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
        'sources': df['source'].value_counts().to_dict(),
        'unique_genres': df['genre_clean'].nunique(),
        'top_genres': df['genre_clean'].value_counts().head(15).to_dict(),
        'missing_authors': (df['authors_clean'].str.strip() == '').sum(),
        'missing_year': df['year'].isna().sum(),
        'average_description_length_words': df['desc_clean'].str.split().apply(len).mean(),
        'period_distribution': df['period'].value_counts().sort_index().to_dict(),
        'language_distribution': df['language'].value_counts().head(10).to_dict(),
    }
