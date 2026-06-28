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
import json
import pickle
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
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
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

dataset = load_and_clean_dataset()
dataset = dataset[dataset['genre_clean'] != 'unknown'].reset_index(drop=True)

print(f"[OK] {len(dataset)} livres avec un genre macro connu")

# =============================================================================
# STEP 2: ÉCHANTILLONNAGE STRATIFIÉ (le corpus complet ~120k serait trop long à
# encoder pour un simple benchmark de validation)
# =============================================================================
print("\n[STEP 2/6] Échantillonnage stratifié par genre...")

MAX_PER_GENRE = 400
MIN_BOOKS = 100

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
# STEP 5: PRÉDICTION HYBRIDE (CENTROÏDE + K-NN), + variantes centroïde seul et
# K-NN seul (sous-produits du même calcul, sans coût supplémentaire) pour
# pouvoir les comparer entre elles (C4.2).
# =============================================================================
print("\n[STEP 5/6] Prédiction hybride (+ variantes centroïde seul / K-NN seul)...")

k = 5  # Number of neighbors
alpha = 0.6  # Weight for centroid
beta = 0.4   # Weight for K-NN

y_pred_hybrid, y_pred_centroid, y_pred_knn = [], [], []

t0 = time.time()
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

    y_pred_centroid.append(max(centroid_scores.items(), key=lambda x: x[1])[0])
    y_pred_knn.append(max(knn_scores.items(), key=lambda x: x[1])[0])
    y_pred_hybrid.append(max(hybrid_scores.items(), key=lambda x: x[1])[0])

centroid_knn_hybrid_predict_time = time.time() - t0

# =============================================================================
# STEP 6: COMPARAISON DE MODÈLES — accuracy/F1 vs coût de calcul (C4.2/C4.3)
# =============================================================================
print("\n[STEP 6/7] Comparaison de modèles (accuracy/F1 vs coût de calcul)...")


def eval_predictions(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
    }


model_comparison = []

# --- Centroïde seul (alpha=1, beta=0) ---
model_comparison.append({
    "name": "Centroïde seul",
    **eval_predictions(y_test, y_pred_centroid),
    "fit_time_s": 0.0,  # les centroïdes sont déjà calculés à l'étape 4
    "predict_time_s": round(centroid_knn_hybrid_predict_time, 3),
    "model_size_kb": round(len(pickle.dumps(category_profiles)) / 1024, 1),
})

# --- K-NN seul (k=5) ---
model_comparison.append({
    "name": "K-NN seul (k=5)",
    **eval_predictions(y_test, y_pred_knn),
    "fit_time_s": 0.0,  # K-NN ne s'entraîne pas, il stocke tout le train set
    "predict_time_s": round(centroid_knn_hybrid_predict_time, 3),
    "model_size_kb": round(len(pickle.dumps((X_train, y_train))) / 1024, 1),
})

# --- Hybride centroïde + K-NN ---
model_comparison.append({
    "name": f"Hybride centroïde+K-NN (alpha={alpha}/beta={beta})",
    **eval_predictions(y_test, y_pred_hybrid),
    "fit_time_s": 0.0,
    "predict_time_s": round(centroid_knn_hybrid_predict_time, 3),
    "model_size_kb": round(len(pickle.dumps((category_profiles, X_train, y_train))) / 1024, 1),
})

# --- Régression logistique (modèle paramétrique léger, à titre de comparaison) ---
t0 = time.time()
logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_train, y_train)
logreg_fit_time = time.time() - t0

t0 = time.time()
y_pred_logreg = logreg.predict(X_test := X_all[idx_test])
logreg_predict_time = time.time() - t0

model_comparison.append({
    "name": "Régression logistique",
    **eval_predictions(y_test, y_pred_logreg),
    "fit_time_s": round(logreg_fit_time, 3),
    "predict_time_s": round(logreg_predict_time, 3),
    "model_size_kb": round(len(pickle.dumps(logreg)) / 1024, 1),
})

# --- Random Forest (modèle ensembliste plus lourd, pour mesurer le rapport
# coût/bénéfice d'un modèle plus coûteux — angle écoresponsabilité C4.3) ---
t0 = time.time()
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_fit_time = time.time() - t0

t0 = time.time()
y_pred_rf = rf.predict(X_test)
rf_predict_time = time.time() - t0

model_comparison.append({
    "name": "Random Forest (100 arbres)",
    **eval_predictions(y_test, y_pred_rf),
    "fit_time_s": round(rf_fit_time, 3),
    "predict_time_s": round(rf_predict_time, 3),
    "model_size_kb": round(len(pickle.dumps(rf)) / 1024, 1),
})

comparison_df = pd.DataFrame(model_comparison).round(4)
print(comparison_df.to_string(index=False))

# Modèle retenu = meilleur F1 pondéré parmi les 5 testés, sans favoriser une
# méthode a priori : le but de cette comparaison est justement de vérifier si
# la méthode hybride utilisée ailleurs dans le projet est réellement la
# meilleure, pas de confirmer un choix déjà fait (cf. discussion écoresponsabilité
# dans le README — un modèle plus précis qui coûte 100x plus cher à stocker
# n'est pas automatiquement le bon choix).
RETAINED_MODEL = max(model_comparison, key=lambda m: m["f1_weighted"])["name"]
print(f"\n[STEP 6/7] Modèle retenu (meilleur F1 pondéré) : {RETAINED_MODEL}")

# Prédictions correspondant au modèle retenu, pour le rapport détaillé et la
# matrice de confusion (les 2 derniers modèles ne sont gardés que pour la
# comparaison de coût, on n'a pas besoin de garder leurs prédictions au-delà).
y_pred_by_model = {
    "Centroïde seul": y_pred_centroid,
    "K-NN seul (k=5)": y_pred_knn,
    f"Hybride centroïde+K-NN (alpha={alpha}/beta={beta})": y_pred_hybrid,
    "Régression logistique": y_pred_logreg,
    "Random Forest (100 arbres)": y_pred_rf,
}
y_pred_retained = y_pred_by_model[RETAINED_MODEL]

accuracy = next(m["accuracy"] for m in model_comparison if m["name"] == RETAINED_MODEL)
precision = next(m["precision_weighted"] for m in model_comparison if m["name"] == RETAINED_MODEL)
recall = next(m["recall_weighted"] for m in model_comparison if m["name"] == RETAINED_MODEL)
f1 = next(m["f1_weighted"] for m in model_comparison if m["name"] == RETAINED_MODEL)
print(f"   Accuracy:             {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"   Precision (weighted): {precision:.4f}")
print(f"   Recall (weighted):    {recall:.4f}")
print(f"   F1-Score (weighted):  {f1:.4f}")

print("\nRapport détaillé (precision / recall / f1-score / support par genre) :")
report_dict = classification_report(y_test, y_pred_retained, zero_division=0, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose().round(3)
print(report_df.to_string())

# =============================================================================
# STEP 7: MATRICE DE CONFUSION (genre prédit vs genre réel) + EXPORT JSON
# =============================================================================
print("\n[STEP 7/7] Matrice de confusion et export pour le dashboard KPI web...")

genre_labels = sorted(category_profiles.keys())
conf_matrix = confusion_matrix(y_test, y_pred_retained, labels=genre_labels)

kpi_results = {
    "generated_at": datetime.now().isoformat(),
    "method": RETAINED_MODEL,
    "n_books_corpus": int(len(dataset)),
    "n_books_sampled": int(len(sampled)),
    "n_train": int(len(idx_train)),
    "n_test": int(len(idx_test)),
    "n_genres": len(genre_labels),
    "accuracy": float(accuracy),
    "precision_weighted": float(precision),
    "recall_weighted": float(recall),
    "f1_weighted": float(f1),
    "genres": genre_labels,
    "classification_report": report_dict,
    "confusion_matrix": conf_matrix.tolist(),
    "model_comparison": model_comparison,
    "retained_model": RETAINED_MODEL,
}

with open("kpi_results.json", "w", encoding="utf-8") as f:
    json.dump(kpi_results, f, indent=2, ensure_ascii=False)

print("[OK] Résultats sauvegardés dans kpi_results.json (utilisés par la page /kpi de l'app web)")

print("\n" + "=" * 80)
print("[SUCCESS] EVALUATION TERMINEE")
print("=" * 80)
