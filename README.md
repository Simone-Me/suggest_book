# 📚 Système de Recommandation Littéraire — SBERT + GenAI

Système de recommandation de livres combinant analyse sémantique locale (SBERT) et IA générative (Google Gemini), construit sur un référentiel fusionné à partir de trois sources publiques pour maximiser la diversité (genres, langues, époques).

Projet EFREI M1 Data Engineering & IA Générative.

> Ce fichier remplace `README_WEB.md` et `GUIDE_COMPLET.md` (désormais supprimés) — toute la documentation technique du projet est centralisée ici. `RAPPORT_PROJET.md` reste un document séparé (livrable de rapport, non un guide d'installation). `REFERENTIEL_RNCP.md` mappe chaque compétence du Bloc 2 (RNCP40875) au code du projet, pour la préparation du passage devant jury.

## 🕘 Historique des versions

Le projet a été repris et ré-évalué à deux reprises depuis sa première version — chaque section ci-dessous correspond à une itération distincte, pas à un développement isolé.

### v1.0 — Version initiale

- Référentiel unique (~700 livres, une seule source)
- Questionnaire hybride (questions ouvertes + échelles de Likert)
- Embeddings SBERT (`all-MiniLM-L6-v2`) + similarité cosinus
- Scoring pondéré 80% sémantique / 20% préférences Likert
- GenAI : enrichissement conditionnel des requêtes courtes + synthèse personnalisée (1 appel)
- Top 3 recommandations
- Interface web Flask basique (en anglais)
- Sauvegarde de l'historique des préférences/résultats en JSON

### v2.0 — Référentiel multi-source et dashboard enrichi

Refonte majeure du moteur et de l'interface :

- **Données** : fusion de 3 sources (`Book_Dataset_1`, `BooksDatasetClean`, `Best_Books_Ever`) en un référentiel unique de ~120 000 livres (170x plus grand) ; dédoublonnage titre+auteur cross-source ; regroupement de ~3 100 catégories brutes en 15 genres macro ; extraction de mots-clés (TF-IDF) ; bucketing de période
- **Moteur** : scoring vectorisé à l'échelle du nouveau corpus ; exclusion des livres déjà lus ; Top 5 (au lieu de Top 3) ; correctif Gemini (`thinkingConfig` désactivé — les synthèses étaient tronquées)
- **Interface web** : graphique radar (profil souhaité vs livre), comparatif des scores en barres, couvertures (Google Books API), liens d'achat/emprunt, animation de chargement, log temps réel du pipeline dans le terminal
- **Évaluation** : `analysis_improved.py` réécrit sur le référentiel fusionné, sans fuite de label, avec rapport precision/recall/f1/support complet
- **Documentation** : README consolidé en un seul fichier

### v2.1 — Révisions suite aux retours et suggestions

Cette itération traite la v2.0 comme un existant à ré-évaluer plutôt qu'une base figée, en réponse à des retours et suggestions d'amélioration :

- **Identité visuelle / UX** : logo et favicon (`stack-of-books.png`), animation de chargement remplacée par un GIF animé avec attribution ([Flaticon/Freepik](https://www.flaticon.com/free-animated-icons/read)), lien "Bibliothèque de Paris" corrigé avec le vrai format d'URL du catalogue, questionnaire reformulé pour clarifier qu'on décrit un **genre/profil de lecture** et non un titre précis à retrouver
- **Correctifs** : régression CSS découverte et corrigée (logo affiché à 512px au lieu de 48px, animation de chargement et son attribution manquantes — perdues lors d'une récupération de styles depuis l'historique git), résidus nettoyés (placeholder oublié dans le README, fichier de crash debris)
- **Évaluation et KPIs** : nouveau dashboard interactif `/kpi` — cartes accuracy/precision/recall/F1, **comparaison de 5 modèles** de classification (centroïde, K-NN, hybride, régression logistique, Random Forest) avec coût (temps d'entraînement/prédiction, taille mémoire), heatmap de confusion avec **filtre par genre et calcul TP/FP/FN/TN en direct**. Le modèle retenu est désormais sélectionné automatiquement par meilleur F1 plutôt que fixé en dur — ce qui a révélé que le centroïde seul bat l'hybride et les modèles plus lourds, tout en étant ~150x plus léger à stocker (angle écoresponsabilité)
- **GenAI** : évaluation automatique de la qualité de la synthèse Gemini — cohérence sémantique (cosinus résumé ↔ livre recommandé) et vérification anti-hallucination des livres suggérés contre le référentiel (~120k titres), via un format de sortie structuré imposé au prompt
- **Sémantique multilingue** : passage de `all-MiniLM-L6-v2` à `paraphrase-multilingual-MiniLM-L12-v2` pour aligner les requêtes en français avec un corpus partiellement non-anglais (régénération des embeddings en cours sur ~120k livres)
- **Accessibilité** : contraste de texte mis en conformité WCAG AA sur les couleurs principales (vérifié par calcul de ratio de luminance, pas seulement à l'œil) ; attributs ARIA (rôles, labels, descriptions) sur le formulaire, les zones dynamiques et le graphique radar
- **Documentation** : `REFERENTIEL_RNCP.md` — mapping de chaque compétence du Bloc 2 (RNCP40875) au code du projet, avec limites assumées explicitement plutôt que dissimulées

## 🌟 Fonctionnalités

- **Questionnaire hybride** : questions ouvertes + échelles de Likert (1-5)
- **Référentiel multi-sources** : 3 datasets fusionnés et dédoublonnés (~120 000 livres)
- **Analyse sémantique** : embeddings SBERT, coût zéro, 100% local
- **Scoring pondéré** : 80% similarité sémantique + 20% préférences Likert, vectorisé pour tenir sur un corpus de cette taille
- **Exclusion des livres déjà lus** : les titres cités dans "livres préférés" ne sont jamais re-suggérés
- **GenAI stratégique** : enrichissement conditionnel des requêtes courtes + synthèse personnalisée (1 seul appel)
- **Évaluation de la synthèse GenAI** : cohérence sémantique (cosinus résumé ↔ livre) + vérification anti-hallucination des livres suggérés contre le référentiel
- **Interface web** : graphique radar (profil souhaité vs livre), comparatif de scores en barres, couvertures et liens d'achat/emprunt, log temps réel dans le terminal
- **Évaluation indépendante** : benchmark de séparabilité sémantique des genres (precision/recall/f1/support) + matrice de confusion
- **Comparaison de modèles + écoresponsabilité** : 5 méthodes de classification comparées (accuracy/F1 vs temps d'entraînement/prédiction et taille mémoire), modèle retenu choisi automatiquement par meilleur F1, pas fixé en dur
- **Dashboard KPI interactif (`/kpi`)** : accuracy/précision/rappel/F1 pondérés, tableau de comparaison de modèles, heatmap de confusion avec filtre par genre (TP/FP/FN/TN dérivés en direct)
- **Accessibilité** : contraste de texte WCAG AA sur les couleurs principales, ARIA sur les formulaires/graphiques/zones dynamiques

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

   Liens de téléchargement :
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
| `embeddings_books.pkl` | ~60-120 min sur CPU (encodage SBERT multilingue de ~120k livres) | `app.py` / `book_recommendation_system.py` |

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

### Évaluation de la qualité de la synthèse GenAI

Comme pour les recommandations elles-mêmes, un texte généré n'a pas de "bonne réponse" prédéfinie à comparer. Deux métriques mesurables et automatiques sont calculées à chaque requête (`evaluate_genai_quality()` dans `app.py`) :

- **Cohérence sémantique** : similarité cosinus entre l'embedding de la synthèse générée et l'embedding du livre Top 1 qu'elle est censée décrire. Une synthèse qui dérive du sujet (hors-sujet, hallucination généralisée) aurait un score faible.
- **Anti-hallucination des suggestions** : le prompt impose à Gemini de terminer sa réponse par une ligne structurée (`LIVRES_SIMILAIRES: Titre1 | Titre2`), extraite côté serveur (`extract_suggested_titles()`) et vérifiée par correspondance exacte normalisée contre les ~120 000 titres du référentiel. Une suggestion qui n'existe pas dans le corpus est comptée comme une hallucination.

Les deux scores sont affichés sous la synthèse dans l'interface web (cohérence en %, nombre de titres suggérés vérifiés / total) et sauvegardés dans `recommendation_results.json` pour suivi dans le temps. Limite connue : la vérification anti-hallucination est une correspondance exacte (normalisée en minuscules) — un titre suggéré légèrement mal formulé par Gemini sera compté comme non vérifié même s'il existe réellement dans le corpus (faux positif d'hallucination, pas de fuzzy matching pour rester rapide).

## 🖥️ Interface web — détails

- **Couvertures** : récupérées côté navigateur via l'API publique Google Books (gratuite, sans clé), par recherche titre+auteur. Si rien n'est trouvé, l'icône 📖 reste affichée.
- **Liens d'achat/emprunt** : recherche Amazon et Fnac par titre+auteur (URLs de recherche standard) ; lien "Bibliothèque de Paris" en best-effort — le format exact de l'URL de recherche du catalogue n'est pas garanti, à vérifier.
- **Radar** : compare le profil souhaité (4 axes Likert) au profil estimé du livre, calculé par similarité cosinus entre l'embedding du livre et une phrase-ancre par axe (ex : "histoire d'amour centrale et romance passionnée" pour l'axe romance). Échelle calibrée empiriquement sur ce modèle/corpus, à but illustratif.
- **Comparatif en barres** : scores des 3 recommandations côte à côte.
- **Animation de chargement** : GIF de livre animé ([Flaticon, par Freepik](https://www.flaticon.com/free-animated-icons/read)), avec attribution affichée sous l'animation (licence Flaticon).
- **Accessibilité** : contraste de texte vérifié WCAG AA (≥4.5:1) sur les couleurs de texte principales du thème (bleu/violet de marque assombri en `#525fa0` pour les usages texte ; les éléments décoratifs — boutons en dégradé, badges — ne sont pas couverts par cette passe). `role="radiogroup"`/`aria-labelledby` sur les échelles de Likert, `role="alert"`/`role="status"` sur les zones d'erreur/chargement, description textuelle (`aria-label`/`<title>`) sur le graphique radar SVG pour les lecteurs d'écran.
- **Log terminal** : chaque requête affiche dans la console les étapes (requête construite, encodage SBERT chronométré, similarité calculée, exclusions, Top 3, appel GenAI chronométré, durée totale).

## 🔬 Évaluation indépendante (`analysis_improved.py`)

Ce script ne mesure pas la qualité des recommandations (qui n'ont pas de "bonne réponse" prédéfinie), mais une question complémentaire : **les embeddings SBERT capturent-ils assez de signal sémantique pour deviner le genre macro d'un livre à partir de son titre + sa description seuls ?** C'est un indicateur de qualité du corpus/embeddings (utile pour justifier l'évaluation et l'ajustement des paramètres), pas une mesure directe de pertinence des recommandations.

Important : le texte utilisé pour ce benchmark exclut volontairement le genre macro (contrairement à `text_full`, utilisé par le moteur de recommandation, qui l'inclut) — sinon le modèle "tricherait" en lisant la réponse dans son entrée.

### Pourquoi ce KPI plutôt qu'un autre

Le moteur de recommandation lui-même n'a pas de label de vérité terrain (aucune liste de "bonnes réponses" attendues par profil utilisateur), donc une précision/rappel calculée directement sur les recommandations serait soit arbitraire, soit nécessiterait une annotation manuelle hors de portée du projet. **Accuracy / precision / recall / F1 pondérés sur la classification du genre macro (via les embeddings)** est le meilleur proxy disponible parce que :

- le genre macro (`genre_clean`) est la seule métadonnée catégorielle fiable et déjà présente sur tout le corpus (~120k livres) — pas besoin d'annotation supplémentaire ;
- une bonne séparabilité par genre est une **condition nécessaire** à des recommandations sémantiquement cohérentes : si le modèle ne distingue même pas "romance" de "horreur" par les embeddings, la similarité cosinus utilisée pour les recommandations n'est pas fiable non plus ;
- la méthode (split train/test stratifié, aucune fuite du label dans le texte encodé) est reproductible et auditable, contrairement à une évaluation purement qualitative.

### Comparaison de modèles et écoresponsabilité (C4.2/C4.3)

`analysis_improved.py` ne teste pas qu'une seule méthode : 5 approches sont évaluées sur le même split train/test, pour vérifier si un modèle plus coûteux apporte un gain qui le justifie, plutôt que de confirmer un choix déjà fait :

| Modèle | Accuracy | F1 pondéré | Temps entraînement | Temps prédiction | Taille mémoire |
|---|---|---|---|---|---|
| **Centroïde seul** | **55.7%** | **0.546** | 0s | 5.7s (boucle partagée) | **21.8 Ko** |
| K-NN seul (k=5) | 49.0% | 0.463 | 0s | 5.7s (boucle partagée) | 3 154 Ko |
| Hybride centroïde+K-NN (alpha=0.6/beta=0.4) | 53.7% | 0.519 | 0s | 5.7s (boucle partagée) | 3 176 Ko |
| Régression logistique | 44.4% | 0.444 | 0.19s | 0.003s | 43 Ko |
| Random Forest (100 arbres) | 47.7% | 0.464 | 0.54s | 0.05s | 22 176 Ko |

**Résultat surprenant et conservé tel quel** : le centroïde seul (simple moyenne des embeddings par genre, comparée par cosinus) bat à la fois le K-NN, l'hybride centroïde+K-NN, la régression logistique *et* la Random Forest — tout en étant ~150x plus léger à stocker que l'hybride et ~1000x plus léger que la Random Forest. Conclusion écoresponsabilité (C4.3) : ici, le modèle le plus simple est aussi le meilleur ; la complexité supplémentaire (K-NN, ensembles d'arbres) n'achète aucune précision, seulement du coût de stockage/calcul. Le **modèle retenu** (`kpi_results.json["retained_model"]`, affiché sur `/kpi`) est donc sélectionné automatiquement par meilleur F1 pondéré parmi les 5, pas fixé en dur — si un futur ré-entraînement change ce classement, la page `/kpi` et ce README doivent être régénérés/mis à jour en conséquence.

| Métrique (modèle retenu) | Valeur |
|---|---|
| Accuracy | 55.7% |
| Précision pondérée | 0.55 |
| Rappel pondéré | 0.56 |
| F1 pondéré | 0.55 |

`fiction` et `other` (genres "fourre-tout") restent les moins bien prédits ; `humor`, `lifestyle`, `spirituality_philosophy` (genres narrativement distinctifs) les mieux prédits — cohérent avec l'intuition. Le rapport complet (precision/recall/f1-score/support par genre), la matrice de confusion et la comparaison des 5 modèles sont exportés dans `kpi_results.json` et visualisables sur **`/kpi`** dans l'interface web (cartes de synthèse, tableau de comparaison de modèles, tableau par genre, heatmap de confusion avec filtre par genre affichant TP/FP/FN/TN dérivés).

Important : cette comparaison ne remet pas en cause le moteur de recommandation principal (`app.py`), qui n'utilise aucun de ces 5 classifieurs — il reste basé sur la similarité cosinus directe entre l'embedding de la requête et ceux du corpus (cf. [Architecture](#-architecture)). Elle ne concerne que ce benchmark indépendant de séparabilité des genres.

Relancer `python analysis_improved.py` régénère `kpi_results.json` (et donc la page `/kpi`) après toute modification du corpus ou de la méthode.

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
├── kpi_results.json                # Résultats du benchmark (généré par analysis_improved.py, lu par /kpi)
├── templates/
│   ├── index.html                  # Interface web (questionnaire + résultats)
│   └── kpi.html                    # Dashboard KPI (accuracy/precision/recall/F1 + matrice de confusion)
├── RAPPORT_PROJET.md               # Rapport de projet (document séparé, non couvert ici)
├── REFERENTIEL_RNCP.md             # Mapping compétences RNCP40875 Bloc 2 ↔ code du projet
└── requirements.txt                # Dépendances
```

## 🔍 Technologies clés

- **SentenceTransformers** : embeddings sémantiques locaux (`paraphrase-multilingual-MiniLM-L12-v2`, choisi pour la diversité linguistique du corpus et de l'interface en français)
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
