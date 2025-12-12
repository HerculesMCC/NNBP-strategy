#!/usr/bin/env python3
"""
Démonstration de l'analyse d'actions américaines par secteur
"""

import os

def main():
    print("=== DÉMONSTRATION ===")
    print()
    
    print("🎯 OBJECTIF:")
    print("   Analyser 22 actions américaines (S&P 500) avec un modèle LSTM")
    print("   Comparer la performance vs Buy & Hold")
    print("   Stocker les résultats dans une base de données SQLite")
    print()
    
    print("📊 UNIVERS D'INVESTISSEMENT : 22 Actions Américaines")
    print("   - Diversification équilibrée : 2 actions par secteur")
    print("   - 11 secteurs représentés (GICS: Global Industry Classification Standard)")
    print("   - Grandes capitalisations du S&P 500")
    print("   - Liquidité élevée et données complètes")
    print()
    
    # Importer la liste depuis stock_analysis
    import stock_analysis
    stocks = stock_analysis.STOCKS
    
    print("📋 ACTIONS PAR SECTEUR (2 par secteur):")
    sectors = {
        "Technologie": ["AAPL", "MSFT"],
        "Finance": ["JPM", "V"],
        "Santé": ["JNJ", "UNH"],
        "Consommation Discrétionnaire": ["TSLA", "HD"],
        "Consommation Staples": ["WMT", "PG"],
        "Énergie": ["XOM", "CVX"],
        "Industriel": ["BA", "CAT"],
        "Télécommunications": ["T", "VZ"],
        "Matériaux": ["LIN", "APD"],
        "Utilitaires": ["NEE", "DUK"],
        "Immobilier": ["AMT", "PLD"]
    }
    
    for sector, symbols in sectors.items():
        print(f"   {sector}: {', '.join(symbols)}")
    print()
    
    print("🔧 MODÈLE UTILISÉ:")
    print("   - LSTM (Long Short-Term Memory)")
    print("   - 20 jours de données pour prédire le jour suivant")
    print("   - Classification binaire (hausse/baisse)")
    print("   - Architecture : LSTM(50) → Dropout → Dense(25) → Dropout → Sortie")
    print("   - Inspiré des thèses sur la prédiction de cours boursiers")
    print()
    
    print("📈 MÉTRIQUES CALCULÉES:")
    print("   - Précision du modèle")
    print("   - Rendement de la stratégie")
    print("   - Rendement Buy & Hold")
    print("   - Performance relative")
    print()
    
    # Vérifier les fichiers
    required_files = ['stock_analysis.py', 'stock_analyzer.py']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"❌ Fichiers manquants: {', '.join(missing_files)}")
        return
    
    print("🚀 LANCEMENT DE L'ANALYSE...")
    print()
    
    try:
        # Exécuter l'analyse
        import stock_analysis
        stock_analysis.main()
        
        print("\n✅ ANALYSE TERMINÉE !")
        print()
        
        # Analyser les résultats
        print("📊 ANALYSE DES RÉSULTATS...")
        import stock_analyzer
        stock_analyzer.main()
        
        print("\n📁 FICHIERS GÉNÉRÉS:")
        files_to_check = [
            'stock_analysis.db',
            'results.csv'
        ]
        
        for file in files_to_check:
            if os.path.exists(file):
                size = os.path.getsize(file)
                print(f"   ✓ {file} ({size:,} bytes)")
            else:
                print(f"   ✗ {file} (manquant)")
        
    except Exception as e:
        print(f"❌ Erreur: {e}")

if __name__ == "__main__":
    main()
