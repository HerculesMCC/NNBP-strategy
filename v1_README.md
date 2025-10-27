## Projet de groupe — Stratégies de trading en Python

### Objectif
Développer en équipe (3–4 étudiants) une stratégie de trading, de la recherche au code prêt pour la production, dans un dépôt GitHub.

### Livrables
- **Recherche & backtesting (Notebook Jupyter)**
  - Explication théorique de la stratégie
  - Implémentation du backtest dans le même notebook
- **Code prêt pour la production**
  - Un script `main.py` qui génère quotidiennement des signaux ou des sorties de trading
  - Le script doit être capable de récupérer les données et d’enregistrer un fichier de pondérations de portefeuille

### Organisation suggérée du dépôt
- `notebooks/` : recherche, prototypage et backtests
- `src/` : modules Python réutilisables (chargement des données, signaux, allocation, etc.)
- `main.py` : point d’entrée production pour générer les sorties quotidiennes
- `data/` : données (brutes/intermédiaires/outputs) — éviter de versionner les gros fichiers

### Environnement
- Python ≥ 3.10
- Jupyter Notebook / JupyterLab
- Gestion des dépendances recommandée : `venv` ou `conda`

### Mise en route rapide
1. Cloner le dépôt
2. Créer l’environnement et installer les dépendances
   - `python -m venv .venv && .venv\\Scripts\\activate` (Windows PowerShell)
   - `pip install -r requirements.txt` (si fourni)
3. Lancer le notebook pour la recherche : `jupyter lab` (ou `jupyter notebook`)
4. Exécuter la version production : `python main.py`

### Sortie attendue de `main.py`
- Le script doit :
  - Récupérer automatiquement les données nécessaires
  - Calculer les signaux/pondérations
  - Sauvegarder un fichier de pondérations de portefeuille (ex. `data/output/portfolio_weights_YYYY-MM-DD.csv`)

### Évaluation
- Présentation de 20 minutes en fin de projet, suivie de questions des encadrants.
- La présentation doit couvrir :
  - Les motivations derrière la stratégie
  - Les choix de conception et le processus de modélisation
  - Les résultats de backtesting et les métriques de performance
  - Les limites, les enseignements et les pistes d’amélioration

### Stratégies suggérées (au choix)
- **Suivi de tendance par moyennes mobiles (Long Only)**
  - Deux moyennes mobiles (ex. 10 jours vs 90 jours)
  - Achat quand la courte croise au-dessus de la longue, sortie inversement
- **Momentum (Long/Short)**
  - Classer les actifs par performance récente (ex. 30 jours)
  - Long sur les meilleurs, short sur les moins performants
- **Stratégie basée sur la volatilité (GARCH, Long/Short)**
  - Prévoir la volatilité et adapter l’exposition
  - Prise en compte d’effets asymétriques (leverage effect)
- **Prédiction par réseau de neurones (Long Only)**
  - Modéliser la probabilité de rendement positif par actif
  - Prendre position si la probabilité > seuil (ex. 50 %), sans short
- **Retour à la moyenne via pairs trading et exposant de Hurst (Long/Short)**
  - Identifier des paires historiquement corrélées et suivre l’écart
  - Un exposant de Hurst < 0,5 suggère un comportement de retour à la moyenne

### Qualité & bonnes pratiques
- Séparer recherche (notebooks) et code réutilisable (`src/`)
- Typage et tests unitaires sur les composants clés
- Logs et gestion d’erreurs dans `main.py`
- Reproductibilité (graine aléatoire, versions des dépendances)

### Auteurs
Indiquez ici les membres du groupe et leurs contributions principales.