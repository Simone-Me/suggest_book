# Référentiel RNCP40875 — Bloc 2 : mapping compétences ↔ projet

Ce document relie chaque compétence du **Bloc 2 — Piloter et implémenter des solutions d'IA en s'aidant notamment de l'IA générative** (RNCP40875) au code de ce projet, avec preuve (fichier/ligne) et limites connues. Il est destiné à faciliter la préparation du passage devant jury (MSP) — ce n'est pas un livrable de rapport (voir `RAPPORT_PROJET.md` pour ça) ni un guide d'installation (voir `README.md`).

Convention : ✅ = couvert avec preuve concrète, ⚠️ = couvert partiellement (limite explicitée), ❌ = non couvert.

---

## C3.1 — Préparation des données

> Préparer les données en les transformant et en les nettoyant, en utilisant des outils appropriés, afin d'assurer une qualité optimale et universellement accessible pour l'analyse et le reporting.

**Statut : ✅**

| Critère | Preuve |
|---|---|
| Outils de transformation/nettoyage mobilisés efficacement | `data_cleaning.py` : `merge_datasets()` (l.333), `deduplicate_books()` (l.357), `clean_text()`/`clean_authors()` (l.112, 139) — réparation d'encodage (`ftfy`), normalisation, suppression d'annotations parasites |
| Qualité et adéquation aux besoins métiers | `load_and_clean_dataset()` (l.410) : suppression des lignes sans `title`/`description` (champs indispensables au moteur sémantique), complétion des champs manquants entre sources |
| Étapes documentées | README.md section "🧹 Comment fonctionne le nettoyage" (description pas-à-pas des 8 étapes du pipeline) |
| Données prêtes pour l'apprentissage automatique | `text_full` (titre + genre + description tronquée) = entrée directe de l'encodage SBERT ; mise en cache `cleaned_books_cache.pkl` |

**Limite connue** : pas de validation statistique formelle de la qualité post-nettoyage (ex. distribution des valeurs manquantes par champ) au-delà de `get_statistics()` (l.487) — fonction existante mais pas exploitée dans un rapport dédié.

---

## C3.2 — Préparation et visualisation de données (tableaux de bord interactifs inclusifs)

> Construire des tableaux de bord interactifs en collaboration avec les équipes métiers pour communiquer les résultats, extraire des connaissances en temps réel et favoriser la décision, en intégrant l'inclusivité.

**Statut : ✅ (avec limite explicite sur le périmètre accessibilité)**

| Critère | Preuve |
|---|---|
| Visualisations informatives et adaptées | Radar profil souhaité/livre (`templates/index.html`, fonction `renderRadar`), comparatif de scores en barres (`renderScoreBars`), dashboard `/kpi` (cartes KPI, tableau de comparaison de modèles, heatmap de confusion) |
| Graphiques interactifs / temps réel | `/recommend` est asynchrone (fetch + rendu DOM sans rechargement de page) ; le dashboard `/kpi` propose un **filtre par genre avec drill-down** (`templates/kpi.html`, `<select id="genreFilter">` + script associé) qui calcule TP/FP/FN/TN en direct côté client à partir de la matrice de confusion |
| Fonctionnalités avancées (filtres, drill-down) | Filtre par genre sur la heatmap de confusion (surlignage ligne/colonne + panneau TP/FP/FN/TN dynamique), `templates/kpi.html` l.236+ |
| Inclusivité | Contraste de texte vérifié WCAG AA (≥4.5:1) sur les couleurs de texte principales (`#525fa0` remplaçant `#8591c3` pour le texte) ; `role="radiogroup"`/`aria-labelledby` sur les échelles de Likert ; `role="alert"`/`role="status"` sur les zones d'erreur/chargement ; description SVG (`aria-label`/`<title>`) sur le radar pour lecteurs d'écran ; `scope="col"/"row"` et `<caption class="sr-only">` sur les tableaux du dashboard KPI |

**Limite connue** : la passe d'accessibilité couvre le **texte** (la majorité du contenu informatif) mais pas les éléments décoratifs à fort contraste de marque (boutons en dégradé, badges colorés) — ceux-ci restent en `#8591c3`/`#f45f7c` pour préserver l'identité visuelle. Pas de test avec un lecteur d'écran réel (NVDA/VoiceOver), uniquement des attributs ARIA standards.

---

## C3.3 — Analyse exploratoire de données

> Mettre en place des processus d'analyse exploratoire en utilisant des techniques statistiques et des outils adaptés, pour générer des insights exploitables.

**Statut : ✅**

| Critère | Preuve |
|---|---|
| Techniques et outils adaptés | `analysis_data.ipynb` (EDA : comparaison des 3 sources, distribution des genres/langues/années) ; `data_cleaning.get_statistics()` (l.487) |
| Insights pertinents documentés | README.md section "📊 Dataset" (répartition en 15 genres macro depuis 3106+ catégories brutes), section "🔬 Évaluation indépendante" (genres les mieux/moins bien séparés sémantiquement) |
| Étapes alignées sur les objectifs stratégiques | Les insights de l'EDA justifient directement les choix de `group_category()` (regroupement des genres) et la stratégie d'échantillonnage de `analysis_improved.py` |

---

## C4.1 — Stratégie d'intégration de l'IA

> Définir une stratégie d'intégration de l'IA en identifiant les cas d'usage pertinents et leur impact, alignée aux exigences de l'écosystème et de la gouvernance.

**Statut : ✅**

| Critère | Preuve |
|---|---|
| Cas d'usage pertinents et alignés | RAPPORT_PROJET.md section 3.3 "Cadre Théorique" (justification SBERT vs Word2Vec/GloVe), section 3.4 (objectifs pédagogiques) |
| Impact évalué et expliqué | RAPPORT_PROJET.md section 4 (personas, scénarios d'usage, contraintes, métriques de succès) |
| Feuille de route réalisable | README.md "🏗️ Architecture" (schéma du pipeline complet, du CSV brut à l'affichage) |

---

## C4.2 — Implémentation d'algorithmes de machine learning

> Développer des modèles prédictifs de machine learning pour identifier de nouveaux comportements/usages.

**Statut : ✅**

| Critère | Preuve |
|---|---|
| Données prétraitées et adaptées | Embeddings SBERT normalisés (`StandardScaler`), split train/test stratifié (`analysis_improved.py` l.108-134) |
| Algorithmes testés et justifiés | **5 algorithmes comparés** sur le même split : centroïde seul, K-NN seul, hybride centroïde+K-NN, régression logistique, Random Forest (`analysis_improved.py` l.136-258) |
| Code fonctionnel et sans erreur | Testé de bout en bout (`python analysis_improved.py` exécuté avec succès, génère `kpi_results.json`) |
| Résultats répondant aux objectifs métier | Le modèle retenu (meilleur F1 pondéré, sélectionné automatiquement l.269) répond à l'objectif du benchmark : mesurer la séparabilité sémantique des genres comme proxy de qualité des embeddings |

---

## C4.3 — Évaluation comparative des modèles (avec écoresponsabilité)

> Évaluer la performance des modèles en analysant leurs résultats avec des métriques adaptées et leur degré d'écoresponsabilité, en les comparant.

**Statut : ✅**

| Critère | Preuve |
|---|---|
| Plusieurs modèles comparés avec métriques appropriées | Tableau de comparaison (`analysis_improved.py` l.179-261) : accuracy, precision/recall/F1 pondérés, temps d'entraînement, temps de prédiction, taille mémoire (pickle) pour 5 modèles. Visible sur `/kpi` (`templates/kpi.html`, table `.comparison`) |
| Améliorations proposées et expliquées | Découverte documentée dans README.md : le centroïde seul bat l'hybride et les modèles plus lourds tout en étant ~150x plus léger — conclusion explicite que la complexité supplémentaire n'apporte aucun gain ici |
| Modèle final validé et écoresponsable | Sélection automatique par meilleur F1 (`RETAINED_MODEL`, l.269), pas de choix arbitraire ; le modèle retenu est aussi le plus léger en mémoire des 5 testés — alignement direct performance/écoresponsabilité |

**Limite connue** : la mesure d'écoresponsabilité se limite à des proxies mesurables localement (temps CPU, taille mémoire pickle) — pas de mesure d'empreinte carbone réelle (kWh, gCO2eq), qui nécessiterait un outil dédié (ex. CodeCarbon) non intégré ici.

---

## C5.1 — Identification des cas d'usage de l'IA générative

> Identifier les cas d'usage de l'IA générative pertinents pour les processus métiers, en choisissant les modèles appropriés selon les objectifs de gouvernance.

**Statut : ✅**

| Critère | Preuve |
|---|---|
| Cas d'usage pertinents | 2 cas d'usage documentés : enrichissement conditionnel de requêtes courtes, synthèse personnalisée du Top 1 (`app.py` `generate_genai_summary`, l.195 ; README.md "🤖 Intégration GenAI") |
| Modèles adaptés aux objectifs de gouvernance | Gemini 2.5 Flash choisi pour son quota gratuit et sa rapidité ; usage strictement limité (1 seul appel de synthèse par requête) pour rester dans le quota et limiter les coûts/l'empreinte |
| Impacts évalués | RAPPORT_PROJET.md section 8.10 "Gouvernance" (traçabilité des données, RGPD, audit de biais) |

---

## C5.2 — Développement de solutions basées sur l'IA générative

> Développer des solutions basées sur des foundation models/LLM adaptés, prenant en compte l'accessibilité universelle, pour des solutions innovantes et utilisables.

**Statut : ⚠️ (solution fonctionnelle, accessibilité universelle partielle)**

| Critère | Preuve |
|---|---|
| Modèles de base mobilisés et pertinents | Gemini 2.5 Flash via API REST (`app.py` l.195-261), avec gestion explicite du `thinkingConfig` (piège documenté dans README) |
| Solution fonctionnelle, innovante, adaptée au contexte | Synthèse personnalisée intégrée au flux de recommandation, avec format de sortie structuré (`LIVRES_SIMILAIRES: ...`) pour permettre une vérification automatique en aval |
| Respect de l'accessibilité universelle | Partiel : voir C3.2 pour les efforts de contraste/ARIA sur l'ensemble de l'interface (y compris la zone d'affichage de la synthèse GenAI, `#summaryQuality`) |

**Limite connue** : l'accessibilité a été traitée au niveau de l'interface (contraste, ARIA) mais pas au niveau du **contenu généré** lui-même (ex. pas de simplification automatique du langage, pas de version audio). C'est une limite à assumer explicitement devant le jury plutôt qu'à dissimuler.

---

## C5.3 — Évaluation de la qualité des résultats de l'IA générative

> Évaluer la qualité des résultats générés en mettant en place des métriques d'évaluation adaptées, et ajuster les paramètres du modèle pour améliorer les performances.

**Statut : ✅**

| Critère | Preuve |
|---|---|
| Métriques d'évaluation pertinentes | `evaluate_genai_quality()` (`app.py` l.279) : cohérence sémantique (cosinus résumé ↔ embedding du livre Top 1) + taux d'hallucination (vérification des titres suggérés contre le référentiel via `title_norm_index`) |
| Paramètres ajustés pour optimiser les performances | `thinkingConfig: {"thinkingBudget": 0}` (correctif documenté dans README — sans ce réglage, la synthèse était tronquée) ; prompt modifié pour imposer un format structuré (`LIVRES_SIMILAIRES: Titre1 \| Titre2`) afin de rendre la vérification automatique possible |
| Résultats analysés avec clarté | Scores affichés directement dans l'UI sous la synthèse (`templates/index.html`, `#summaryQuality`) et persistés dans `recommendation_results.json` pour suivi dans le temps |

**Limite connue** : la vérification anti-hallucination est une correspondance **exacte normalisée** (pas de fuzzy matching), documentée comme limite assumée dans le README — un titre légèrement reformulé par Gemini sera compté comme non vérifié même s'il existe réellement.

---

## Synthèse des limites assumées (à mentionner spontanément devant jury)

1. **C3.1** : pas de rapport de qualité statistique formel post-nettoyage (la fonction existe, n'est pas exploitée en rapport dédié).
2. **C3.2/C5.2** : accessibilité couverte sur le texte informatif, pas sur les éléments décoratifs de marque ; pas de test avec lecteur d'écran réel.
3. **C4.3** : écoresponsabilité mesurée par proxies (temps CPU, taille mémoire), pas par empreinte carbone réelle.
4. **C5.3** : anti-hallucination par correspondance exacte, pas de fuzzy matching.

Documenter ces limites explicitement (plutôt que les cacher) répond directement à l'esprit du référentiel : la rigueur méthodologique inclut la reconnaissance honnête du périmètre couvert.
