# 📚 Système de Recommandation Littéraire — SBERT + GenAI

Système de recommandation de livres combinant analyse sémantique locale (SBERT) et IA générative (Google Gemini), construit sur un référentiel fusionné à partir de trois sources publiques pour maximiser la diversité (genres, langues, époques).

Projet EFREI M1 Data Engineering & IA Générative.

> Ce fichier remplace `README_WEB.md` et `GUIDE_COMPLET.md` (désormais supprimés) — toute la documentation technique du projet est centralisée ici. `RAPPORT_PROJET.md` reste un document séparé (livrable de rapport, non un guide d'installation).

## 🌟 Fonctionnalités

- **Questionnaire hybride** : questions ouvertes + échelles de Likert (1-5)
- **Référentiel multi-sources** : 3 datasets fusionnés et dédoublonnés (~120 000 livres)
- **Analyse sémantique** : embeddings SBERT, coût zéro, 100% local
- **Scoring pondéré** : 80% similarité sémantique + 20% préférences Likert, vectorisé pour tenir sur un corpus de cette taille
- **Exclusion des livres déjà lus** : les titres cités dans "livres préférés" ne sont jamais re-suggérés
- **GenAI stratégique** : enrichissement conditionnel des requêtes courtes + synthèse personnalisée (1 seul appel)
- **Interface web** : graphique radar (profil souhaité vs livre), comparatif de scores en barres, couvertures et liens d'achat/emprunt, log temps réel dans le terminal
- **Évaluation indépendante** : benchmark de séparabilité sémantique des genres (precision/recall/f1/support)

## 🏗️ Architecture

```
3 CSV bruts ─→ merge_datasets() ─→ deduplicate_books() ─→ nettoyage + regroupement genres
                                                          ─→ cleaned_books_cache.pkl (cache)
                                                                      ↓
Requête utilisateur ─→ encodage SBERT ─→ similarité cosinus (vectorisée sur tout le corpus)
                                       ─→ exclusion livres déjà lus ─→ Top 3
                                       ─→ synthèse GenAI (1 appel) ─→ affichage (radar, barres, couvertures)
```

## 📦 Installation

1. **Cloner le dépôt** :

```bash
git clone https://github.com/Simone-Me/suggest_book.git
cd suggest_book
```

2. **Installer les dépendances** :

```bash
pip install -r requirements.txt
```

3. **Télécharger les 3 datasets** (non versionnés dans le repo — voir [Dataset](#-dataset)) :

   - `Book_Dataset_1.csv`
   - `BooksDatasetClean.csv`
   - `Best_Books_Ever.csv`

   À télécharger sur Kaggle : ajouter lien moi meme
   - [Best_Books_Ever.csv](https://zenodo.org/records/4265096?preview_file=books_1.Best_Books_Ever.csv)
   - [BooksDatasetClean.csv](https://www.kaggle.com/datasets/elvinrustam/books-dataset/data)
   - [Book_Dataset_1.csv](https://www.kaggle.com/datasets/jalota/books-dataset)

   Placer les 3 fichiers CSV à la racine du projet (même dossier que `app.py`).

4. **Configurer la clé API Gemini** (optionnel, active la synthèse GenAI) :

```bash
# Windows
set GEMINI_API_KEY=votre_cle_api

# Linux/Mac
export GEMINI_API_KEY=votre_cle_api
```

Clé gratuite : https://makersuite.google.com/app/apikey (limite gratuite ~60 requêtes/minute)

## 🚀 Lancement

### Version CLI

```bash
python book_recommendation_system.py
```

### Interface web

```bash
python app.py
```

Puis ouvrir http://localhost:5000 dans le navigateur.

### ⏱️ Le premier lancement est lent — c'est normal

Au tout premier lancement (CLI ou web), deux caches sont construits :

| Cache | Coût | Construit par |
|---|---|---|
| `cleaned_books_cache.pkl` | ~2 min (fusion + dédoublonnage + mots-clés TF-IDF sur ~150k lignes) | `data_cleaning.load_and_clean_dataset()` |
| `embeddings_books.pkl` | ~30-60 min sur CPU (encodage SBERT de ~120k livres) | `app.py` / `book_recommendation_system.py` |

Les deux sont sauvegardés sur disque et réutilisés à chaque lancement suivant — ils ne sont régénérés que si vous les supprimez ou passez `force_rebuild=True`. Ne supprimez pas `embeddings_books.pkl` sans raison, c'est la partie longue.

## 📊 Dataset

Le référentiel n'est **pas un CSV unique** : c'est la fusion de trois sources publiques, pour maximiser la diversité littéraire (genres, langues, époques) au-delà de ce qu'un seul catalogue peut offrir.

| Source | Lignes (brutes) | Apporte |
|---|---|---|
| `Book_Dataset_1.csv` | ~1 000 | Catalogue générique (sans auteur/année) |
| `BooksDatasetClean.csv` | ~103 000 | Catalogue éditeur : titre, auteur, description, catégorie, année |
| `Best_Books_Ever.csv` | ~52 000 | Export Goodreads : série, langue, nombre de pages, note lecteurs |

Après fusion + dédoublonnage + suppression des lignes sans titre/description, le référentiel final contient **~119 700 livres** répartis en **15 genres macro** (réduits depuis 3 106 catégories brutes + des centaines de tags Goodreads en texte libre).

Schéma commun conservé après fusion (volontairement minimal — prix, éditeur, ISBN, etc. sont abandonnés) :

```
title, authors, description, year, series, language, pages, rating, genres_raw, source
```

## 🧹 Comment fonctionne le nettoyage (`data_cleaning.py`)

Toute la logique vit dans `data_cleaning.py`, partagée par `app.py` et `book_recommendation_system.py` (aucune duplication). `load_and_clean_dataset()` exécute, dans l'ordre :

1. **`merge_datasets()`** — lit les 3 CSV bruts et les ramène au schéma commun ci-dessus (ex : le champ `genres` de `Best_Books_Ever`, stocké comme une liste Python en texte, est reparsé ; les dates en texte libre sont converties en année à 4 chiffres via regex).
2. **`deduplicate_books()`** — regroupe les lignes par titre normalisé. Si un même titre correspond à des auteurs *différents*, les lignes restent distinctes (deux livres différents). Si l'auteur correspond (ou est manquant d'un côté, cas de `Book_Dataset_1`), les lignes sont fusionnées en une seule : la **description la plus longue est gardée** (les sources ont rarement la même description mot pour mot pour un même livre), et les champs manquants (année, série, langue, pages, note, genre) sont complétés depuis l'autre source.
3. **Suppression des lignes sans `title` ou `description`** — les deux sont indispensables au moteur sémantique.
4. **Nettoyage du texte** (`clean_text`, `clean_authors`) — réparation d'encodage (`ftfy`), minuscules, normalisation des espaces, suppression du préfixe `"By "` et des annotations de rôle type `"(Illustrator)"` dans les noms d'auteurs.
5. **`group_category()`** — réduit les chaînes de genre/catégorie brutes (taxonomies multi-niveaux *et* listes de genres Goodreads, ~440 valeurs brutes combinées) à ~15 genres macro (`fiction`, `fantasy_scifi`, `mystery_thriller`, `romance`, `young_adult`, `history_biography`, `science_academic`, `lifestyle`, `spirituality_philosophy`, `arts_poetry`, `sequential_art`, `horror`, `humor`, plus `other`/`unknown`).
6. **`bucket_period()`** — transforme l'année de publication en décennie (le référentiel attend un champ "période" qu'aucune des 3 sources ne fournit directement).
7. **`extract_keywords()`** — TF-IDF sur tout le corpus pour générer un champ "Keywords" par livre (également absent des 3 sources, mais attendu par le format du référentiel).
8. **`text_full`** — titre + genre macro + description (tronquée à 200 mots pour éviter que les descriptions très longues ne diluent l'embedding SBERT) — c'est ce texte qui est réellement encodé par SBERT.

Le résultat est mis en cache dans `cleaned_books_cache.pkl` (le TF-IDF seul prend ~1-2 min sur le corpus complet, trop coûteux à refaire à chaque démarrage). Passer `force_rebuild=True` à `load_and_clean_dataset()` après avoir modifié les CSV sources.

## 🧠 Concepts clés

### Embeddings (vecteurs sémantiques)

Un texte est transformé en vecteur de nombres qui capture son **sens**, pas juste ses mots :

```
"the great gatsby. fiction. a story of wealth and love..." → SBERT → [0.24, -0.15, 0.89, ..., 0.33]  (384 nombres)
```

Propriété clé : des textes **similaires** ont des vecteurs **proches** dans cet espace, même sans mot en commun ("livre captivant" ≈ "page-turner").

### Pourquoi SBERT plutôt que Word2Vec/GloVe ?

| Aspect | Word2Vec/GloVe | SBERT |
|---|---|---|
| Niveau | Mots individuels | Phrases/paragraphes entiers |
| Contexte | Fenêtre locale | Contexte global de la phrase |
| Usage ici | Analyse de mots isolés | Matching de descriptions complètes |

SBERT lève aussi l'ambiguïté lexicale ("apple" le fruit vs la marque) en encodant la phrase entière, pas le mot seul.

### Similarité cosinus

Mesure l'angle entre deux vecteurs (1.0 = identique, 0.0 = aucun lien) :

```
cos(θ) = (A · B) / (||A|| × ||B||)
```

Calculée ici en une seule opération **vectorisée** sur tout le corpus (`sklearn.metrics.pairwise.cosine_similarity(query, embeddings)`), pas en boucle Python livre par livre — indispensable à l'échelle de ~120k livres (une boucle Python serait inutilisable en production).

### Pondération du score (80/20)

```
score_final = 0.8 × similarité_cosinus + 0.2 × intensité_likert_moyenne
```

80% sémantique pour que le sens du texte prime, 20% préférences quantifiées pour départager deux livres sémantiquement proches selon l'intensité recherchée (action, romance, apprentissage, complexité).

### Interprétation du score affiché

Avec des embeddings SBERT, une similarité cosinus entre une requête courte et une description de livre dépasse rarement 70-80%, même pour un excellent match (un score proche de 95-100% impliquerait un texte quasi identique mot pour mot). Un score affiché autour de 55-65% est donc un bon signal de correspondance thématique — ce qui compte le plus reste l'écart entre le Top 1 et le reste, signe que le moteur discrimine bien.

## 🤖 Intégration GenAI (Gemini)

Deux appels conditionnels/uniques, jamais plus, pour rester dans le quota gratuit :

1. **Enrichissement conditionnel** (si la requête utilisateur fait moins de 5 mots) — ajoute du contexte littéraire avant l'encodage SBERT. Fallback local si pas de clé API.
2. **Synthèse personnalisée** (1 seul appel, sur le Top 1) — explique pourquoi le livre correspond, ses points forts, et suggère 2 titres proches.

**Piège rencontré et corrigé** : Gemini 2.5 consomme par défaut une partie du budget de tokens en "réflexion" interne (`thinkingConfig`) avant d'écrire la réponse visible — avec un budget de 512 tokens, on a mesuré jusqu'à 487 tokens absorbés par cette réflexion, ne laissant que 20 tokens pour la vraie réponse (synthèse coupée après une phrase). Réglé en forçant `"thinkingConfig": {"thinkingBudget": 0}` dans les 3 appels API du projet.

## 🖥️ Interface web — détails

- **Couvertures** : récupérées côté navigateur via l'API publique Google Books (gratuite, sans clé), par recherche titre+auteur. Si rien n'est trouvé, l'icône 📖 reste affichée.
- **Liens d'achat/emprunt** : recherche Amazon et Fnac par titre+auteur (URLs de recherche standard) ; lien "Bibliothèque de Paris" en best-effort — le format exact de l'URL de recherche du catalogue n'est pas garanti, à vérifier.
- **Radar** : compare le profil souhaité (4 axes Likert) au profil estimé du livre, calculé par similarité cosinus entre l'embedding du livre et une phrase-ancre par axe (ex : "histoire d'amour centrale et romance passionnée" pour l'axe romance). Échelle calibrée empiriquement sur ce modèle/corpus, à but illustratif.
- **Comparatif en barres** : scores des 3 recommandations côte à côte.
- **Animation de chargement** : livre aux pages qui tournent, en CSS pur (pas de PNG/GIF/MP4 — plus léger, sans fichier à charger, couleurs alignées sur le thème).
- **Log terminal** : chaque requête affiche dans la console les étapes (requête construite, encodage SBERT chronométré, similarité calculée, exclusions, Top 3, appel GenAI chronométré, durée totale).

## 🔬 Évaluation indépendante (`analysis_improved.py`)

Ce script ne mesure pas la qualité des recommandations (qui n'ont pas de "bonne réponse" prédéfinie), mais une question complémentaire : **les embeddings SBERT capturent-ils assez de signal sémantique pour deviner le genre macro d'un livre à partir de son titre + sa description seuls ?** C'est un indicateur de qualité du corpus/embeddings (utile pour justifier l'évaluation et l'ajustement des paramètres), pas une mesure directe de pertinence des recommandations.

Important : le texte utilisé pour ce benchmark exclut volontairement le genre macro (contrairement à `text_full`, utilisé par le moteur de recommandation, qui l'inclut) — sinon le modèle "tricherait" en lisant la réponse dans son entrée.

Résultat de référence (échantillon stratifié de 8 400 livres, 600/genre) : accuracy 53.3%, precision pondérée 0.52, f1 pondéré 0.52, avec un rapport complet precision/recall/f1-score/support par genre affiché en sortie. `fiction` et `other` (genres "fourre-tout") sont les moins bien prédits ; `humor`, `lifestyle`, `sequential_art` (genres narrativement distinctifs) les mieux prédits — cohérent avec l'intuition.

## 📁 Structure du projet

```
.
├── book_recommendation_system.py   # Point d'entrée CLI
├── app.py                          # Application web (Flask)
├── data_cleaning.py                # Fusion + dédoublonnage + nettoyage (partagé)
├── analysis_data.ipynb             # EDA : comparaison des datasets, fusion/dédoublonnage, stats du corpus
├── analysis_improved.py            # Benchmark indépendant de séparabilité des genres
├── Book_Dataset_1.csv              # Source brute 1 (à télécharger, voir Dataset)
├── BooksDatasetClean.csv           # Source brute 2 (à télécharger, voir Dataset)
├── Best_Books_Ever.csv             # Source brute 3 (à télécharger, voir Dataset)
├── cleaned_books_cache.pkl         # Cache du dataset fusionné/nettoyé (généré au 1er lancement)
├── embeddings_books.pkl            # Cache des embeddings SBERT (généré au 1er lancement)
├── templates/
│   └── index.html                  # Interface web
├── RAPPORT_PROJET.md               # Rapport de projet (document séparé, non couvert ici)
└── requirements.txt                # Dépendances
```

## 🔍 Technologies clés

- **SentenceTransformers** : embeddings sémantiques locaux (`all-MiniLM-L6-v2`)
- **scikit-learn** : similarité cosinus (vectorisée) + TF-IDF pour les mots-clés
- **ftfy** : réparation d'encodage pendant le nettoyage du texte
- **Flask** : framework web
- **Google Gemini** : enrichissement de texte et synthèse GenAI
- **Pandas/NumPy** : traitement des données

## 📈 Performance

- **Taille du corpus** : ~119 700 livres (fusion de 3 sources, dédoublonné)
- **Genres macro** : 15 (réduits depuis 3 106+ catégories/tags bruts)
- **Modèle d'embedding** : 384 dimensions
- **Similarité** : cosinus, calculée pour tout le corpus en un seul appel vectorisé par requête
- **Pondération** : 80% sémantique + 20% préférences Likert

## 🤝 Contribuer

Les contributions sont bienvenues, n'hésitez pas à proposer une Pull Request.

## 📄 Licence

Projet développé dans le cadre du cours EFREI M1 Data Engineering - IA Générative.

## 👥 Auteurs

- **Maharo** - **Simone** - EFREI M1 Data Engineering

## 🙏 Remerciements

- EFREI Paris pour le cadre pédagogique
- Google Gemini pour les capacités GenAI
- La communauté SentenceTransformers

---

**Note** : projet pédagogique illustrant des techniques de NLP et d'IA générative pour la recommandation personnalisée.
