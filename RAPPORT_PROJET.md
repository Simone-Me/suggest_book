2. RÉSUMÉ EXÉCUTIF
But du Projet
Le but du projet est de développer un système de recommandation littéraire intelligent combinant analyse sémantique locale (SBERT) et intelligence artificielle générative (Google Gemini) pour proposer des suggestions de livres personnalisées basées sur les préférences utilisateurs exprimées en langage naturel.

 Principaux Choix Techniques
Architecture Hybride NLP-GenAI : Le système repose sur une approche en deux couches. La première couche utilise Sentence-BERT (SBERT) avec le modèle `all-MiniLM-L6-v2` pour encoder sémantiquement un corpus de plus de 700 livres en vecteurs de 384 dimensions, permettant ainsi une analyse de similarité cosinus totalement gratuite et locale. La seconde couche exploite Google Gemini (2.5-flash-lite) de manière stratégique uniquement pour enrichir les requêtes utilisateurs trop courtes (c’est-à-dire inférieures à 5 mots) en ajoutant du contexte littéraire, et générer une synthèse explicative personnalisée des recommandations.
Collecte de Données Hybride : Un questionnaire combine des questions ouvertes (description libre, livres préférés, thèmes à éviter) avec des échelles de Likert (1-5) évaluant l'intensité d'action, la romance, l'apprentissage et la complexité narrative souhaitée. Cette dualité permet de capturer à la fois la richesse sémantique du langage naturel et des métriques quantitatives pondérables.
Scoring Pondéré Multi-Critères : Le système implémente une formule de scoring sophistiquée combinant 80% de similarité sémantique (cosinus) avec 20% de pondération issue des préférences Likert. Cette approche équilibrée garantit que les recommandations sont à la fois sémantiquement pertinentes et alignées sur les intensités émotionnelles recherchées par l'utilisateur.
Résultats Obtenus
Le système génère trois recommandations de livres ordonnées par score de pertinence, chacune accompagnée de données complètes (titre, auteur, catégorie, prix, note moyenne, description). Les tests qualitatifs démontrent une précision sémantique conforme aux résultats attendus (environ 50%) compte tenu des limites du volume de données d’entrainement du modèle, mais avec des suggestions cohérentes par rapport aux descriptions fournies. L'enrichissement GenAI améliore significativement la qualité des recommandations lorsque l'utilisateur fournit peu d'informations textuelles, transformant des requêtes vagues en contextes littéraires exploitables.

Outil Développé
Interface Web Flask : Application web responsive avec formulaire intelligent permettant la saisie des préférences, affichage des résultats en temps réel avec visuels (prix, notes, descriptions), et génération optionnelle d'un résumé exécutif par IA générative expliquant les raisons des recommandations.
Persistance et Optimisation : Système de cache des embeddings (format pickle) évitant le recalcul à chaque exécution, réduisant le temps de démarrage de ~3 minutes à inférieur à5 secondes. Sauvegarde JSON des préférences utilisateurs pour traçabilité 
Technologies Majeures
- Sentence-BERT (SBERT) : Modèle de transformer pré-entraîné `all-MiniLM-L6-v2` (384D embeddings) pour encodage sémantique local et gratuit
- Similarité Cosinus : Mesure de distance vectorielle implémentée via `sklearn.metrics.pairwise.cosine_similarity` pour matching sémantique
- RAG (Retrieval-Augmented Generation) : Architecture récupérant les top-3 livres les plus similaires avant génération de contenu par IA, limitant les hallucinations
- API Google Gemini : Modèle `gemini-2.5-flash-lite` utilisé pour enrichissement conditionnel (inférieur à150 tokens/requête) et synthèse finale (inférieur à500 tokens)
- Flask : Framework web Python léger pour interface utilisateur responsive
- Pandas/NumPy: Manipulation de données et calculs vectoriels haute performance
- Streamlit (prévu en extension) : Alternative pour dashboards interactifs sans nécessiter HTML/CSS
Contraintes Économiques et Pédagogiques
L'usage de l'API GenAI a été volontairement limité à 2 appels maximum par session utilisateur (enrichissement + synthèse) avec des limites strictes de tokens (150 + 500) pour respecter les quotas gratuits de Google Gemini (1500 requêtes/jour, 1M tokens/mois). Cette approche permet de démontrer les capacités de l'IA générative dans un cadre pédagogique tout en garantissant la soutenabilité économique du projet, 95% du traitement reposant sur des modèles locaux open-source.

3. INTRODUCTION ET CONTEXTE
 3.1 Problématique Adressée
Contexte Métier : Orientation Littéraire
Dans l'écosystème numérique actuel, les consommateurs de livres font face à une surcharge informationnelle massive : Amazon référence plus de 40 millions de titres, les bibliothèques numériques prolifèrent, et chaque année voit la publication de centaines de milliers de nouveaux ouvrages. Ce phénomène d’abondance rend la découverte de livres adapté à ses goûts de plus en plus complexes.

Les systèmes de recommandation traditionnels comme les filtrages collaboratifs type "Les lecteurs de X ont aussi aimé Y" souffrent de biais de popularité (surreprésentation des best-sellers), du problème de démarrage à froid, c’est-à-dire de la difficulté pour une plateforme par exemple, à proposer du contenu, ici des nouveaux livres à un utilisateur sans historique, et d'une compréhension limitée des nuances sémantiques, par exemple, il peut ne pas saisir qu’un utilisateur cherche « une ambiance mélancolique avec un narrateur non fiable », et interpréter la demande comme la recherche d’un simple « genre ».

Cas d'Usage Ciblé
Notre système s'adresse spécifiquement aux lecteurs intentionnels qui savent décrire ce qu'ils recherchent mais peinent à naviguer dans les catalogues. Exemples typiques :
- Lecteur passionné : "Je veux un thriller psychologique dans le style de Gone Girl, mais moins violent et avec plus de profondeur psychologique"
- Étudiant : "Je cherche un essai philosophique accessible (pas trop complexe) sur l'existentialisme moderne"
- Professionnel : "Un livre d'entreprise inspirant avec des études de cas concrètes, pas juste théorique"
Ces demandes requièrent une compréhension sémantique fine que les données classiques (genre, auteur, tags) ne peuvent capturer.

3.2 Contexte Industriel
Évolution du Marché de la Recommandation
Selon l'analyse de MRFR, Market Research Future, un cabinet d’études de marché spécialisé dans les rapports de prévision dans différentes domaines (technologies, IA, santé), la taille du marché des moteurs de recommandation de contenu était estimée à 8,417 milliards USD en 2024. L'industrie des moteurs de recommandation de contenu devrait croître de 10,82 milliards USD en 2025 à 132,76 milliards USD d'ici 2035, affichant un taux de croissance annuel composé de 28,5 pendant la période de prévision 2025 - 2035. Ainsi, les acteurs dominants (Amazon, Goodreads, Spotify/Audiobooks) investissent massivement dans l'IA pour améliorer la personnalisation :
- Amazon Books : Utilise un filtrage collaboratif hybride combinant historiques d'achat et analyse de contenu
- Goodreads : Recommandations sociales basées sur les notes de pairs et les étagères virtuelles partagées
- Apple Books/Kindle : Analyse de patterns de lecture (temps passé par page, passages surlignés)
Cependant, ces systèmes propriétaires restent des boîtes noires inaccessibles au grand public et nécessitent des infrastructures coûteuses.
Démocratisation de l'IA Sémantique
L'arrivée de modèles open-source performants comme Sentence-BERT (2019) et d'APIs GenAI abordables (Google Gemini, OpenAI GPT) rend désormais possible la création de systèmes de recommandation sémantiques avec :
- Des coûts d'infrastructure réduits (modèles légers exécutables sur CPU)
- Une explicabilité accrue (compréhension des raisons des recommandations)
- Une personnalisation immédiate sans historique préalable requis
Notre projet s'inscrit dans cette tendance de démocratisation technologique.

 3.3 Cadre Théorique
 Pourquoi l'Analyse Sémantique ?
L'analyse sémantique consiste à capturer le sens profond,d'un texte au-delà des simples mots-clés. Contrairement aux approches lexicales (comptage de termes, TF-IDF), elle permet de comprendre que :
"livre captivant" ≈ "page-turner" ≈ "impossible à lâcher" (similarité sémantique élevée)
"romance" ≠ "histoire d'amour toxique" (nuances émotionnelles distinctes)
Cette finesse est cruciale en recommandation littéraire où les utilisateurs emploient un vocabulaire riche et subjectif ("atmosphère onirique", "personnages complexes", "rythme haletant").
Pourquoi Sentence-BERT (SBERT) ?
SBERT (Reimers & Gurevych, 2019) est un modèle de transformeur spécialisé dans l'encodage de phrases en vecteurs denses. Ses avantages par rapport à BERT classique :
- Efficacité : Génère des embeddings de phrases en une passe 
-Qualité : Pré-entraîné sur 1 milliard de paires de phrases (NLI, STSbenchmark) garantissant une représentation sémantique robuste
- Portabilité : Modèle `all-MiniLM-L6-v2` (384D) de seulement 80 MB, exécutable sur CPU standard
Pourquoi RAG (Retrieval-Augmented Generation) ?
Le RAG (Lewis et al., 2020) combine :
-Retrieval : Récupération des documents les plus pertinents via recherche vectorielle (ici : top-3 livres par similarité cosinus)
-Augmentation : Injection de ces documents dans le contexte du modèle génératif
-Generation: Synthèse personnalisée par LLM ancré sur des faits réels
Avantages par rapport à la génération pure :
- Réduit les hallucinations (les recommandations proviennent d'un corpus réel)
- Permet la mise à jour des connaissances sans réentraînement du LLM (ajout de livres au CSV)
- Réduit drastiquement les tokens consommés (contexte ciblé vs. Génération from scratch)

Pourquoi l'IA Générative (Limitée) ?
L'usage de Google Gemini dans ce projet est stratégique :
Cas d'usage 1 : Enrichissement de Requêtes Courtes (EF4.1)
- Problème : User input = "thriller" (trop vague pour SBERT)
- Solution : GenAI enrichit en "thriller psychologique avec ambiance sombre, suspense graduellement croissant et révélations imprévisibles"
- Impact : Améliore la précision des embeddings et donc la qualité des matches
Cas d'usage 2 : Synthèse Explicative (EF4.2-4.3)
- Problème : Utilisateur voit 3 livres recommandés mais ne comprend pas pourquoi
- Solution : GenAI génère un paragraphe personnalisé expliquant les liens entre sa requête et les caractéristiques des livres sélectionnés
- Impact : Augmente la confiance utilisateur et l'acceptation des recommandations
Limitations Volontaires :
- Maximum 2 appels API/session (enrichissement optionnel + synthèse optionnelle)
- Caps de tokens strictes : 150 tokens (enrichissement), 500 tokens (synthèse)
- Fallback local : Si API indisponible/quota atteint, système fonctionne en mode 100% local
Justification Pédagogique :
Ce projet vise à démontrer qu'une utilisation minimaliste de l'IA générative, couplée à des techniques NLP classiques robustes, peut produire des résultats performants tout en restant économiquement viable (0€ de coûts réels sous quotas gratuits). Cette approche enseigne la frugalité computationnelle et la conception hybride intelligente plutôt que la dépendance totale à des LLMs coûteux.

 3.4 Objectifs Pédagogiques
Ce projet permet de maîtriser :
- Architecture RAG complète : Implémentation end-to-end d'un pipeline Retrieval-Augmented Generation
- Optimisation coûts-performance : Équilibrage entre traitement local (SBERT) et génération cloud (Gemini)
- Ingénierie des prompts : Conception de prompts GenAI efficaces et contrôlés (limites de tokens, température)
- Évaluation qualitative : Analyse critique des résultats de recommandation et itération
- Déploiement web : Création d'interfaces utilisateur fonctionnelles (Flask)

4. ANALYSE DU BESOIN UTILISATEUR
 4.1 Identification et Segmentation des Utilisateurs Cibles
Le système s’adresse à des utilisateurs recherchant une aide à la sélection de livres, sans expertise préalable approfondie. Deux profils principaux ont été identifiés.
Le premier profil correspond à des lecteurs curieux et autonomes. Ils lisent de manière régulière ou intermittente. Leur objectif est l’enrichissement culturel ou intellectuel. Ils attendent des recommandations contextualisées et pédagogiques. La valeur ajoutée réside dans la mise en relation des ouvrages avec des courants de pensée ou des contextes culturels.
Le second profil regroupe des utilisateurs occasionnels. Ils recherchent un livre dans un objectif utilitaire, souvent pour offrir. Leur connaissance des préférences du lecteur final est limitée. Ils privilégient des suggestions fiables et rapidement exploitables. Ils attendent une justification claire et rassurante du choix proposé.
Ces deux profils partagent un besoin commun. Ils souhaitent réduire la complexité de la recherche. Ainsi, ils attendent des recommandations pertinentes, explicables et immédiatement utilisables.
Profil 1 : L'Explorateur Curieux
Caractéristiques :
- Âge : 18-25 ans (étudiants) ou 50+ ans (retraités avec temps libre)
- Fréquence : Variable (5-40 livres/an)
- Objectif : Élargir horizons culturels, éducation continue
Besoins Spécifiques :
- Recommandations éducatives avec équilibre accessibilité/profondeur
- Liens entre livres et contexte historique/philosophique

Exemple : Thomas, 21 ans, Étudiant en Philosophie
"Je prépare un mémoire sur l'existentialisme. J'ai besoin d'essais accessibles pour compléter les lectures académiques imposées, mais les recherches Google me noient sous des références."

 Profil 2 : L'Acheteur Cadeau
Caractéristiques :
- Achète occasionnellement pour offrir
- Connaissance limitée des goûts précis du destinataire
- Cherche des valeurs sûres avec description évocatrice
Besoins Spécifiques :
- Suggestions rapides basées sur description générique ("elle aime les romans historiques")
- Informations sur prix et disponibilité
- Résumé exécutif rassurant sur la qualité du choix

 4.2 Objectifs Utilisateurs Détaillés
 Objectif 1 : Découverte Personnalisée Immédiate
Besoin : Obtenir des recommandations pertinentes sans historique de lecture préalable.
Différenciation par rapports aux concurrents :
- Amazon/Goodreads nécessitent plusieurs dizaines de notes pour commencer à être pertinents
- Notre système fonctionne en démarrage à froid, grâce à l'analyse sémantique de la première requête
Mesures de Succès :
- Au moins 2/3 recommandations jugées pertinentes par l'utilisateur à la première utilisation
- Temps de réponse total inférieur à 30 secondes (incluant questionnaire interactif)

 Objectif 2 : Expression Nuancée des Préférences
Besoin : Exprimer des préférences complexes et contextuelles au-delà des genres standards.
Capacités attendues :
- Comprendre que "thriller psychologique avec ambiance mélancolique" est différent de « thriller d'action haletant"
- Interpréter les échelles d'intensité (romance légère vs. Histoire d'amour centrale)
- Respecter les exclusions ("pas de fantasy", "éviter les fins tristes")
Mesures de Succès :
- Le système distingue correctement la "complexité narrative" (score 5) et la "simplicité" (score 1-2)
- Recommandations reflètent les nuances textuelles (validé par tests A/B qualitatifs)

 Objectif 3 : Transparence 
Besoin : Comprendre pourquoi ces livres sont recommandés.
Fonctionnalités attendues :
- Affichage du score de pertinence (0-100%)
- Synthèse GenAI expliquant les correspondances ("Ce livre a été sélectionné car il combine [caractéristique utilisateur] avec [caractéristique livre]")
- Mise en évidence des points communs avec livres préférés mentionnés
Mesures de Succès :
- L’utilisateur peut paraphraser la raison d'une recommandation après lecture de la synthèse

 Objectif 4 : Efficacité Temporelle
Besoin : Processus complet en inférieur à 5 minutes (questionnaire + résultats).
Contraintes :
- Questionnaire : 7 questions maximum
- Temps de traitement : inférieur à 10 secondes
- Affichage résultats : Format scannable (données clés visibles d'un coup d'œil)
Mesures de Succès :
- Les utilisateurs complètent le questionnaire sans abandon
- Temps médian du questionnaire et lecture des résultats inférieurs à 5 minutes

 4.3 Scénarios d'Usage Détaillés
 Scénario A : Exploration de Genre Nouveau
Contexte : Thomas (profil1) veut découvrir l'existentialisme via la fiction.
Déroulement :
-Description : "Roman philosophique existentialiste accessible, pas trop déprimant"
-Livres préférés : "1984, Le Meilleur des Mondes" (références connues)
-Scores Likert :
   - Action : 2/5 (introspectif prioritaire)
   - Romance : 2/5
   - Apprentissage : 5/5 (objectif éducatif)
   - Complexité : 3/5 (accessible mais stimulant)
Résultat Attendu :
- Recommandations de fiction philosophique (Camus, Sartre, auteurs contemporains)
- Filtrage automatique des essais académiques denses (grâce au score complexité 3/5)
- Synthèse contextualisant les liens avec les thèmes d'1984
Validation : Thomas comprend les concepts philosophiques après lecture et peut progresser vers des œuvres plus complexes.

 Scénario B : Achat Cadeau Express
Contexte : Paul (profil 2) cherche un cadeau pour sa sœur amateur de romans historiques.

Déroulement :
-Description : "Roman historique féminin, époque victorienne ou similaire, pas trop long"
-Livres préférés : (laisse vide)
-Scores Likert : Tous à 3/5 (valeurs par défaut)
Résultat Attendu :
- Sélection de best-sellers fiables du genre 
- Affichage du prix et des données pour aide à la décision
- Synthèse rassurante : "Ces romans historiques sont plébiscités pour..."
Validation : La sœur apprécie le cadeau et Paul considère le système utile.

 4.5 Contraintes et Limites Identifiées
 Contraintes Techniques
Taille du Corpus : 700 livres (vs. Millions sur Amazon)
   - Impact : Risque de non-couverture de niches très spécifiques
Qualité des données : Descriptions du CSV parfois courtes/génériques
   - Impact : Embeddings moins riches pour certains livres
Latence API GenAI : Appels Gemini peuvent prendre 2-5 secondes
   - Impact : Frustration si temps de réponse total supérieur à15s
 Contraintes Économiques
Quotas API Gratuits : Gemini limite 1500 requêtes/jour
   - Impact : Maximum 750 utilisateurs/jour 
Contraintes Utilisateurs
Capacité d'Expression : Certains utilisateurs peinent à décrire leurs préférences
   - Impact : Requêtes floues ("quelque chose de bien") générant des recommandations médiocres
Biais Culturels : Dataset majoritairement anglophone
   - Impact : Sous-représentation littérature francophone/internationale

 4.6 Critères de Succès Mesurables
Métriques Quantitatives
Temps de réponse : inférieur à 30 secondes
Taux de complétion questionnaire élevé 
Score de similarité élevé
Couverture des besoins : part importante des recommandations jugées pertinente
 Métriques Qualitatives
Cohérence Sémantique : Recommandations alignées sur la description (validé par des évaluateurs humains)
Diversité : offre 3 livres qui ne sont pas quasi-identiques
Explicabilité : Synthèses GenAI jugées utiles et non-redondantes
Utilisabilité : Interface intuitive sans formation requise
 Retours Utilisateurs Cibles
- "Enfin un système qui comprend ce que je veux dire par 'ambiance sombre mais pas déprimante'"
- "Les 3 suggestions sont toutes intéressantes, difficile de choisir !"
- "La synthèse m'a aidé à comprendre pourquoi ces livres correspondent à ma demande"

 CONCLUSION DE L'ANALYSE DES BESOINS
Ce projet répond à un besoin réel et mesurable : l'insuffisance des systèmes de recommandation actuels face à la complexité et la subjectivité des préférences littéraires. En combinant compréhension sémantique profonde (SBERT), personnalisation multi-critères (scoring pondéré) et explicabilité (GenAI synthèse), le système offre une expérience utilisateur de meilleure qualité tout en respectant des contraintes économiques strictes.
Les profils utilisateurs identifiés représentent des segments distincts avec des besoins spécifiques, et les scénarios d'usage validés démontrent la capacité du système à gérer aussi bien des demandes très précises que des explorations ouvertes.
Les critères de succès définis (temps de réponse, taux de complétion, scores de similarité, satisfaction qualitative) fournissent une base solide pour l'évaluation continue et l'amélioration itérative du système dans le cadre d'un déploiement réel ou d'extensions futures.

6. RÉFÉRENTIEL DE DONNÉES
6.1 Corpus et sources de données
Le référentiel de données repose sur un corpus de livres issu de sources publiques.
La source principale est le fichier Book_Dataset_1.csv, agrégé à partir de catalogues accessibles librement.
Le volume initial comprend 1 014 entrées. Après nettoyage et filtrage, plus de 700 livres exploitables sont conservés. Le corpus couvre majoritairement la littérature contemporaine anglophone, avec une minorité de titres francophones.
Les genres dominants sont la fiction, le mystery/thriller, la fiction historique, la romance et la non-fiction.
Les descriptions textuelles constituent la principale source d’information sémantique.
Leur longueur varie entre 50 et 500 mots.
Certaines limites sont identifiées. Le corpus présente un biais en faveur des ouvrages anglophones. Les genres de niche, tels que la poésie ou les romans graphiques, sont sous-représentés. Certaines descriptions sont relativement courtes.

6.2 Modèle conceptuel et schéma de données
Le modèle conceptuel est centré sur l’entité Livre. Cette entité regroupe les attributs nécessaires à la modélisation sémantique.
Les champs exploités sont le titre, la catégorie (genre) et la description textuelle.
Ces champs sont transformés lors du chargement du corpus. Ils sont concaténés dans un champ unique text_full. Cette représentation unifiée permet de capturer le contexte complet du livre. Les transformations appliquées incluent la suppression des doublons, le filtrage des catégories invalides, la normalisation du texte (minuscules, nettoyage)
Le schéma de données est volontairement minimal.
Il est adapté à un traitement NLP centré sur la similarité sémantique.

6.3 Formats de restitution et stockage des données
Plusieurs formats de données sont utilisés, chacun jouant un rôle spécifique dans le pipeline.
Le format CSV constitue le référentiel source. Il permet une lecture simple et une interopérabilité élevée. Il est adapté à un corpus statique de taille modérée.
Le format Pickle est utilisé pour le cache des embeddings. Il stocke une matrice vectorielle de dimension (700, 384). Ce mécanisme réduit fortement les temps de calcul.
Il est limité à l’environnement Python.
Le format JSON est utilisé pour les interactions utilisateur. Il stocke les préférences et les résultats de recommandation. Il facilite l’audit et la traçabilité des sessions.

6.4 Modèles de données et approches de modélisation
La modélisation principale repose sur des embeddings sémantiques. Chaque livre est représenté par un vecteur dense de dimension 384. Les vecteurs sont générés à partir du champ text_full.
La similarité entre la requête utilisateur et les livres est calculée par similarité cosinus. Cette mesure capture la proximité sémantique entre textes.
Un mécanisme de scoring pondéré est appliqué. Il combine la similarité sémantique avec les préférences utilisateur. La pondération favorise la cohérence thématique.
Un pipeline de type Retrieval-Augmented Generation est également utilisé. Le modèle GenAI s’appuie exclusivement sur les résultats du retrieval. Il génère une synthèse explicative sans influencer le classement.

6.5 Qualité des données et gouvernance
Le chargement et le nettoyage du corpus sont assurés par une fonction dédiée :’load_knowledge_base()’. Les transformations sont déterministes et reproductibles.
La complétude des champs textuels est élevée. Les titres et catégories sont systématiquement présents. La majorité des livres dispose d’une description exploitable.
Les limites du corpus : Des extensions futures incluent l’ajout de données multilingues. Un enrichissement externe des descriptions est également envisagé.
Les données utilisées sont publiques. Aucune donnée personnelle n’est traitée.
Le référentiel respecte les principes de transparence et de gouvernance des données.

 7. PIPELINE IA ET ARCHITECTURE
7.1 Objectif et principes généraux
Le projet implémente un système de recommandation de livres fondé sur une architecture hybride NLP–GenAI. L’objectif est de fournir des recommandations personnalisées, pertinentes et explicables.
Le cœur décisionnel repose sur des méthodes locales. Les modèles GenAI sont utilisés de manière contrôlée et non décisionnelle. 
Environ 95 % des traitements sont exécutés localement. Les appels cloud représentent environ 5 % du pipeline.

7.2 Vue d’ensemble du pipeline
Le pipeline couvre l’ensemble de la chaîne de traitement, depuis la collecte des préférences utilisateur jusqu’à la restitution finale. Il est structuré en neuf phases successives et clairement identifiées :
 
Acquisition des données utilisateur
Les préférences utilisateur sont collectées via une interface web Flask. Le questionnaire combine des questions ouvertes et des échelles de type Likert. Trois questions portent sur les goûts littéraires exprimés en langage naturel. Quatre questions évaluent l’intensité de critères narratifs sur une échelle de 1 à 5.
Les réponses sont normalisées et sauvegardées au format JSON. Cette persistance garantit la traçabilité du système.
Construction de la requête sémantique
Les scores Likert sont transformés en descripteurs textuels explicites. Cette transformation permet une intégration homogène avec les réponses textuelles. Les différents éléments sont concaténés pour former une requête sémantique unique. La requête constitue l’entrée principale du pipeline de représentation vectorielle.
Enrichissement conditionnel par GenAI
Un enrichissement par modèle GenAI est déclenché uniquement si la requête est trop courte. Le seuil est fixé à moins de cinq mots. L’objectif est d’améliorer la densité sémantique, sans altérer les préférences exprimées.
Le modèle Gemini est utilisé avec un budget de tokens strictement limité. En cas d’échec ou d’indisponibilité, un mécanisme de repli est appliqué. Le pipeline reste fonctionnel sans enrichissement.
Génération des embeddings SBERT
La requête utilisateur est encodée à l’aide du modèle SBERT all-MiniLM-L6-v2. Le modèle produit des vecteurs de dimension 384. Les embeddings du corpus de livres sont pré-calculés et mis en cache.
Ce choix garantit une exécution rapide et reproductible. Le calcul des embeddings est effectué localement sur CPU ou GPU.
Calcul de la similarité cosinus
La similarité cosinus est calculée entre la requête utilisateur et chaque livre du corpus. Cette mesure évalue la proximité sémantique dans l’espace vectoriel. Les scores obtenus sont continus et normalisés. Le calcul est effectué en temps linéaire par rapport à la taille du corpus.
Agrégation et scoring pondéré
Un score final est calculé pour chaque livre. Il combine la similarité sémantique et les préférences quantitatives de l’utilisateur. La pondération favorise la cohérence thématique à hauteur de 80 %. Les critères d’intensité représentent 20 % du score final. Ce mécanisme permet un compromis entre sens linguistique et préférences explicites.
Sélection des recommandations Top-N
Les livres sont triés par score décroissant. Les N meilleurs résultats sont sélectionnés, avec N = 3. Les métadonnées complètes sont associées à chaque recommandation.Cette étape constitue la sortie algorithmique principale du système.
Génération de la synthèse explicative (RAG)
Un pipeline RAG est utilisé pour générer une synthèse personnalisée. Le contexte injecté inclut le profil utilisateur et les livres recommandés. Le modèle GenAI ne modifie pas le classement établi.
La synthèse fournit une justification textuelle des recommandations. Elle améliore l’interprétabilité et l’expérience utilisateur.
Restitution et persistance
Les résultats sont affichés via une interface web responsive. Les scores et la synthèse sont présentés de manière transparente. L’ensemble de la session est sauvegardé au format JSON. Ces données peuvent être exploitées à des fins d’audit ou d’analyse.

8. IMPLÉMENTATION TECHNIQUE
8.1 Architecture globale de la solution
Le système repose sur une architecture web client-serveur à trois niveaux :
Le backend est développé en Python avec le framework Flask. Il orchestre l'ensemble du pipeline de traitement. Le frontend est constitué de templates HTML avec intégration de JavaScript pour les interactions utilisateur.
Le stockage est assuré par des fichiers locaux. Le référentiel de livres est maintenu au format CSV. Les embeddings pré-calculés sont persistés au format Pickle. Les sessions utilisateur sont sauvegardées en JSON.
Les appels API externes sont limités au modèle GenAI Google Gemini. Ces appels sont conditionnels et non bloquants. Un mécanisme de fallback garantit la continuité de service.
Les composants principaux interagissent selon un flux séquentiel. L'interface web collecte les préférences. Le backend effectue les calculs de similarité. Le module GenAI génère la synthèse explicative. Les résultats sont restitués à l'interface utilisateur.
Cette architecture privilégie la simplicité et la reproductibilité. Elle est adaptée à un contexte pédagogique et à un déploiement local.

8.2 Technologies utilisées et justification
Le langage Python est utilisé pour l'ensemble du projet. Ce choix est motivé par la richesse de l'écosystème NLP et l'accessibilité pédagogique du langage.
Le framework Flask assure le développement web. Flask est léger et permet un démarrage rapide. Il est adapté à une application de démonstration sans complexité excessive.
La librairie sentence-transformers fournit le modèle SBERT. Elle offre une interface simple pour le chargement et l'utilisation de modèles pré-entraînés.
Pandas et NumPy sont utilisés pour la manipulation de données. Pandas facilite le chargement et le nettoyage du corpus CSV. NumPy assure les calculs vectoriels.
Scikit-learn fournit la fonction de similarité cosinus. Cette librairie est standard dans l'écosystème Python scientifique.
La librairie requests gère les appels HTTP vers l'API Gemini. Elle offre un contrôle précis des timeouts et de la gestion d'erreurs.
Le format Pickle assure la persistance des embeddings. Il garantit des temps de chargement rapides et une compatibilité native avec NumPy.
Ces choix technologiques favorisent la maintenabilité et la compatibilité pédagogique. Ils permettent une exécution locale sans infrastructure complexe.

8.3 Choix du modèle SBERT
Le modèle retenu est all-MiniLM-L6-v2. Il s'agit d'une variante légère de la famille Sentence-BERT. Ce modèle génère des embeddings de dimension 384. Il comporte 6 couches transformer et environ 22,7 millions de paramètres. Sa taille compressée est de 80 MB. Plusieurs critères ont motivé ce choix. Le modèle est exécutable sur CPU standard. Il offre un bon compromis entre qualité et performance. Il est pré-entraîné sur plus d'un milliard de paires de phrases.
 Les modèles multi-qa sont spécialisés pour les questions-réponses. Les modèles multilingues sont plus volumineux. Les modèles français spécialisés sont moins robustes pour un corpus mixte anglophone-francophone.
Le modèle est chargé via la librairie sentence-transformers. Le chargement s'effectue au démarrage de l'application. Les embeddings du corpus sont calculés une seule fois puis mis en cache. Cette approche garantit une latence de réponse faible. Elle évite les calculs coûteux à chaque requête.

8.4 Calcul des similarités
La métrique utilisée est la similarité cosinus. Elle mesure l'angle entre deux vecteurs dans l'espace d'embedding. La formule appliquée est la suivante. Pour deux vecteurs u et v, la similarité cosinus est égale au produit scalaire de u et v divisé par le produit des normes de u et v. Le résultat est compris entre moins un et plus un.
L'implémentation repose sur la fonction cosine_similarity de scikit-learn. Cette fonction accepte des matrices et effectue les calculs de manière vectorisée.
La requête utilisateur est encodée en un vecteur de dimension 384. Ce vecteur est comparé à chaque vecteur du corpus. Le résultat est un tableau de 700 scores de similarité. Aucune optimisation avancée n'est appliquée. Le corpus de 700 livres permet un calcul en moins de 100 millisecondes. Pour un corpus plus important, des techniques d'indexation vectorielle seraient nécessaires. 
Les scores de similarité sont ensuite combinés avec les préférences utilisateur. Cette étape constitue le scoring pondéré.

8.5 Système de score et pondérations
Le score final combine deux composantes. La première est la similarité sémantique. La seconde est une mesure d'alignement avec les préférences quantitatives de l'utilisateur. Les préférences quantitatives proviennent des échelles de Likert. Quatre critères sont évalués sur une échelle d’un à cinq. Ces critères sont l'intensité d'action, l'importance de la romance, l'aspect éducatif et la complexité narrative.
Une moyenne normalisée des scores Likert est calculée. Cette moyenne est divisée par cinq pour obtenir une valeur entre zéro et un.vLe score final est calculé selon la formule suivante. Il est égal à 0,8 fois la similarité cosinus plus 0,2 fois la moyenne normalisée des scores Likert. La pondération 80/20 a été déterminée empiriquement. Elle privilégie la cohérence sémantique tout en tenant compte des préférences explicites. Des ratios alternatifs ont été testés. Le ratio 90/10 réduit la diversité. Le ratio 50/50 introduit du bruit. Ce mécanisme permet d'affiner les recommandations. Deux livres sémantiquement proches peuvent être départagés par leur alignement avec les intensités recherchées.

8.6 Logique métier du classement
Le classement des livres repose sur le score final calculé précédemment. Les livres sont triés par ordre décroissant de score. L'algorithme utilisé est un tri complet suivi d'une extraction. La fonction numpy.argsort est appliquée au tableau des scores. Les indices des trois premiers éléments sont extraits.
Les trois livres sélectionnés constituent le Top-3 des recommandations. Leurs métadonnées complètes sont récupérées depuis le data frame Pandas. Ces métadonnées incluent le titre, la catégorie, la description, le prix et la note moyenne.
Aucun filtrage post-ranking n'est appliqué. Les exclusions mentionnées par l'utilisateur ne sont pas traitées automatiquement. Cette fonctionnalité constitue une extension future. Le Top-3 est transmis au module de génération de synthèse. Il constitue également la sortie finale du système.

8.7 Appel API GenAI et stratégie de limitation
Le modèle GenAI utilisé est Google Gemini 2.5-flash-lite. Il est interrogé via une API REST. L'architecture RAG est appliquée. Le contexte injecté dans le prompt inclut le profil utilisateur et les métadonnées du livre le mieux classé. Le modèle génère une synthèse explicative sans influencer le classement.
Deux types d'appels sont effectués. Le premier est un enrichissement conditionnel de la requête utilisateur. Il est déclenché uniquement si la requête contient moins de cinq mots. Le second est la génération de la synthèse finale. Il est systématique sauf si l'API est indisponible.
Un mécanisme de fallback est implémenté. En cas d'échec, un message explicatif est retourné. Le système reste fonctionnel sans génération GenAI. Les recommandations sont affichées avec leurs métadonnées. Cette stratégie garantit une utilisation frugale de l'API. Elle respecte les quotas gratuits de Google Gemini. Elle assure la résilience du système face aux défaillances externes.

8.8 Interface utilisateur Flask
L'interface utilisateur est développée avec Flask et des templates HTML. Elle se compose de deux vues principales.
La première vue affiche le formulaire de collecte des préférences. Ce formulaire comporte sept champs. Trois champs texte permettent l'expression libre. Quatre curseurs implémentent les échelles de Likert.Le formulaire est soumis via une requête HTTP POST. Les données sont envoyées à la route /recommend. Le backend traite la requête et calcule les recommandations.
La seconde vue affiche les résultats. Les trois livres recommandés sont présentés avec leurs métadonnées. Les scores de pertinence sont affichés en pourcentage. La synthèse GenAI est intégrée en haut de page. 
Le flux utilisateur est séquentiel : L'utilisateur remplit le formulaire en soumettant ses préférences, ensuite il attend le calcul des recommandations, et après avoir consulté les résultats, il peut recommencer une nouvelle recherche.
Les composants interactifs sont minimalistes. Les curseurs Likert sont implémentés avec des éléments HTML range. Le bouton de soumission déclenche la requête POST. Cette interface privilégie la simplicité et la rapidité de développement. Elle est adaptée à une démonstration pédagogique.

8.9 Organisation du dépôt Git et structure du projet
Le projet est organisé selon une structure modulaire. 
Les fichiers principaux sont situés à la racine du dépôt. 
Le fichier app.py contient l'application Flask. Il définit les routes et orchestre le pipeline. Il intègre les fonctions de traitement et d'appel GenAI. Le fichier book_recommendation_system.py implémente la version CLI. Il permet une exécution en ligne de commande sans serveur web.
Le dossier templates contient les fichiers HTML. Le fichier index.html définit l'interface utilisateur. Il inclut le formulaire et la zone d'affichage des résultats.
Le fichier Book_Dataset_1.csv stocke le corpus de livres. Il est chargé au démarrage de l'application.
Le fichier embeddings_books.pkl contient les embeddings pré-calculés. Il est généré lors de la première exécution puis réutilisé.
Le fichier requirements.txt liste les dépendances Python. Il permet une installation reproductible de l'environnement.
Un fichier .env peut être créé pour stocker la clé API Gemini. Ce fichier est exclu du contrôle de version via .gitignore. La clé API est chargée via la variable d'environnement GEMINI_API_KEY.
Les fichiers JSON générés en sortie sont user_preferences.json et recommendation_results.json. Ils assurent la traçabilité des sessions.
Les librairies principales sont flask, pandas, numpy, sklearn, sentence_transformers et requests. Aucune arborescence complexe de modules n'est nécessaire.
Le projet utilise une branche principale unique. Aucune stratégie de branches multiples n'est appliquée pour ce prototype.

8.10 Gouvernance et responsabilisation
Plusieurs axes de gouvernance sont pris en compte ou identifiés pour une évolution future.
Qualité et traçabilité des données : Le corpus de livres provient de sources publiques. Les transformations appliquées sont déterministes et documentées. Le nettoyage et le filtrage sont reproductibles. Les embeddings sont versionnés avec le modèle utilisé. Toutes les sessions utilisateur sont sauvegardées au format JSON. Cette traçabilité permet un audit ultérieur. Elle facilite l'analyse des comportements et l'amélioration du système.
Validation humaine des réponses : Les recommandations ne font pas l'objet d'une validation humaine systématique. Des tests qualitatifs ont été effectués durant le développement. Une boucle de feedback utilisateur pourrait être intégrée pour améliorer le système.
Choix responsable des modèles : Le modèle SBERT est open-source et exécutable localement. Il ne nécessite aucun coût financier. Sa taille réduite limite l'empreinte écologique. Le modèle GenAI est utilisé de manière parcimonieuse. Le nombre d'appels est strictement limité. Cette approche minimise les coûts et l'impact environnemental. Les limites des modèles sont documentées. Le corpus est majoritairement anglophone. Les recommandations pour des niches littéraires peuvent être moins pertinentes.
Respect RGPD et vie privée : Le système actuel ne collecte aucune donnée personnelle identifiante. Les préférences littéraires ne sont pas liées à une identité. Les fichiers JSON sont stockés localement. En cas de déploiement en ligne, plusieurs mesures seraient nécessaires. Un consentement explicite devrait être recueilli. Une politique de confidentialité devrait être rédigée. Un droit à l'oubli devrait être implémenté.
Audit de biais : Le corpus présente un biais anglophone. Ce biais est documenté et reconnu. Les recommandations pour la littérature francophone ou internationale sont limitées. Aucun biais discriminatoire n'a été identifié dans les métadonnées. Les catégories littéraires sont neutres. Un audit plus approfondi nécessiterait une analyse quantitative des genres représentés. Il nécessiterait également une évaluation qualitative par des experts littéraires diversifiés.


 

