# -*- coding: utf-8 -*-
"""
Évaluation indépendante de la séparabilité sémantique des genres macro.

Ce script ne mesure PAS la qualité du moteur de recommandation (qui restitue
un Top-3 par similarité, sans notion de "bonne/mauvaise réponse"). Il répond à
une question différente mais complémentaire : est-ce que les embeddings SBERT
capturent assez de signal sémantique pour qu'un genre macro (fiction,
romance, ...) soit devinable à partir du titre + de la description seuls ?
C'est un indicateur de la qualité du corpus/embeddings (C5.3), distinct de la
pertinence des recommandations elles-mêmes.
"""
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, precision_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from data_cleaning import load_and_clean_dataset, truncate_description

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

print("=" * 80)
print("BOOK CLASSIFICATION - HYBRID SEMANTIC APPROACH (genres macro, corpus fusionné)")
print("=" * 80)

# =============================================================================
# STEP 1: CHARGEMENT DU CORPUS FUSIONNÉ (déjà nettoyé / regroupé en genres macro)
# =============================================================================
print("\n[STEP 1/6] Chargement du dataset nettoyé...")
model = SentenceTransformer("all-MiniLM-L6-v2")

dataset = load_and_clean_dataset()
dataset = dataset[dataset['genre_clean'] != 'unknown'].reset_index(drop=True)

print(f"[OK] {len(dataset)} livres avec un genre macro connu")

# =============================================================================
# STEP 2: ÉCHANTILLONNAGE STRATIFIÉ (le corpus complet ~120k serait trop long à
# encoder pour un simple benchmark de validation)
# =============================================================================
print("\n[STEP 2/6] Échantillonnage stratifié par genre...")

MAX_PER_GENRE = 200
MIN_BOOKS = 30

sampled_parts = [
    group.sample(min(len(group), MAX_PER_GENRE), random_state=42)
    for _, group in dataset.groupby('genre_clean')
]
sampled = pd.concat(sampled_parts, ignore_index=True)

genre_counts = sampled['genre_clean'].value_counts()
valid_genres = genre_counts[genre_counts >= MIN_BOOKS].index
sampled = sampled[sampled['genre_clean'].isin(valid_genres)].reset_index(drop=True)

print(f"[OK] {len(valid_genres)} genres retenus (>={MIN_BOOKS} livres) : {', '.join(sorted(valid_genres))}")
print(f"[OK] {len(sampled)} livres échantillonnés pour ce benchmark")

# =============================================================================
# STEP 3: EMBEDDINGS SÉMANTIQUES (titre + description SEULEMENT, sans le genre)
# =============================================================================
print("\n[STEP 3/6] Génération des embeddings (sans fuite du label dans le texte)...")

# Important : on n'utilise pas 'text_full' (qui contient déjà le genre macro en
# texte, utilisé par le moteur de recommandation) pour ne pas donner la réponse
# au modèle. On teste ici si le genre est devinable à partir du seul contenu
# narratif (titre + description), pas parce qu'on le lui a soufflé.
sampled['text_for_embedding'] = (
    sampled['title_clean'] + ' ' +
    sampled['desc_clean'].apply(lambda x: truncate_description(x, max_words=100))
)

embeddings = model.encode(
    sampled['text_for_embedding'].tolist(),
    convert_to_tensor=True,
    show_progress_bar=True
)

X_all = embeddings.cpu().numpy()
y_all = sampled['genre_clean']

# Normalize
scaler = StandardScaler()
X_all = scaler.fit_transform(X_all)

print(f"[OK] Embeddings shape: {X_all.shape}")

# =============================================================================
# STEP 4: TRAIN/TEST SPLIT & PROFILS DE GENRE (centroïdes)
# =============================================================================
print("\n[STEP 4/6] Split train/test et création des profils de genre...")

indices = np.arange(len(sampled))
idx_train, idx_test = train_test_split(
    indices,
    test_size=0.25,
    random_state=42,
    stratify=y_all
)

# Create category profiles (centroids)
category_profiles = {}
for category in sampled.iloc[idx_train]['genre_clean'].unique():
    category_mask = sampled.iloc[idx_train]['genre_clean'] == category
    train_indices_for_cat = idx_train[category_mask[idx_train]]
    category_embeddings = X_all[train_indices_for_cat]
    category_profiles[category] = np.mean(category_embeddings, axis=0)

X_train = X_all[idx_train]
y_train = sampled.iloc[idx_train]['genre_clean'].values
y_test = sampled.iloc[idx_test]['genre_clean'].values

print(f"[OK] Train: {len(idx_train)} | Test: {len(idx_test)}")
print(f"[OK] {len(category_profiles)} profils de genre créés")

# =============================================================================
# STEP 5: PRÉDICTION HYBRIDE (CENTROÏDE + K-NN)
# =============================================================================
print("\n[STEP 5/6] Prédiction hybride...")

k = 5  # Number of neighbors
alpha = 0.6  # Weight for centroid
beta = 0.4   # Weight for K-NN

y_pred_hybrid = []

for idx in idx_test:
    query_embedding = X_all[idx].reshape(1, -1)

    # Centroid scores
    centroid_scores = {}
    for category, profile in category_profiles.items():
        similarity = cosine_similarity(query_embedding, profile.reshape(1, -1))[0][0]
        centroid_scores[category] = similarity

    # K-NN scores
    similarities = cosine_similarity(query_embedding, X_train)[0]
    top_k_indices = np.argsort(similarities)[-k:][::-1]
    neighbors_categories = y_train[top_k_indices]

    knn_scores = {}
    for cat in category_profiles.keys():
        knn_scores[cat] = (neighbors_categories == cat).sum() / k

    # Hybrid combination
    hybrid_scores = {}
    for cat in category_profiles.keys():
        hybrid_scores[cat] = alpha * centroid_scores[cat] + beta * knn_scores[cat]

    predicted_category = max(hybrid_scores.items(), key=lambda x: x[1])[0]
    y_pred_hybrid.append(predicted_category)

# =============================================================================
# STEP 6: ÉVALUATION (precision / recall / f1-score / support)
# =============================================================================
print("\n" + "=" * 80)
print("RESULTATS")
print("=" * 80)

accuracy = accuracy_score(y_test, y_pred_hybrid)
precision = precision_score(y_test, y_pred_hybrid, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred_hybrid, average='weighted', zero_division=0)

print(f"\n[STEP 6/6] Méthode hybride (alpha={alpha}, beta={beta}, K={k})")
print(f"   Accuracy:             {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"   Precision (weighted): {precision:.4f}")
print(f"   F1-Score (weighted):  {f1:.4f}")

print("\nRapport détaillé (precision / recall / f1-score / support par genre) :")
report_dict = classification_report(y_test, y_pred_hybrid, zero_division=0, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose().round(3)
print(report_df.to_string())

print("\n" + "=" * 80)
print("[SUCCESS] EVALUATION TERMINEE")
print("=" * 80)
