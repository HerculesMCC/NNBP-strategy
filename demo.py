#!/usr/bin/env python3
"""
Démonstration de l'analyse d'actions américaines par secteur
"""

import os

def main():
    print("=== DÉMONSTRATION ===")
    print()
    
    print("🎯 OBJECTIF:")
    print("   Analyser 100 actions américaines (S&P 500) avec des modèles LSTM par secteur")
    print("   Comparer la performance vs Buy & Hold")
    print("   Stocker les résultats dans une base de données SQLite")
    print()
    
    print("📊 UNIVERS D'INVESTISSEMENT : 100 Actions Américaines")
    print("   - Diversification maximale : ~9-10 actions par secteur")
    print("   - 11 secteurs représentés (GICS: Global Industry Classification Standard)")
    print("   - Grandes capitalisations du S&P 500")
    print("   - Liquidité élevée et données complètes")
    print()
    
    # Importer la liste depuis stock_analysis
    import stock_analysis
    stocks = stock_analysis.STOCKS
    
    # Grouper par secteur
    from collections import defaultdict
    sectors_dict = defaultdict(list)
    for stock in stocks:
        sectors_dict[stock['sector']].append(stock['symbol'])
    
    print("📋 ACTIONS PAR SECTEUR:")
    for sector, symbols in sorted(sectors_dict.items()):
        print(f"   {sector}: {len(symbols)} actions - {', '.join(symbols[:5])}{'...' if len(symbols) > 5 else ''}")
    print()
    
    print("🔧 ARCHITECTURE : DEUX APPROCHES")
    print("   1. MODÈLE GLOBAL:")
    print("      - 1 modèle LSTM entraîné sur TOUTES les actions")
    print("      - Capture les patterns communs à l'ensemble du marché")
    print("      - Maximum de données d'entraînement")
    print()
    print("   2. MODÈLES PAR SECTEUR:")
    print("      - 1 modèle LSTM par secteur (11 modèles)")
    print("      - Entraînement sur toutes les actions du secteur")
    print("      - Capture les patterns spécifiques à chaque secteur")
    print("      - Architecture : LSTM(64) → LSTM(32) → Dense(32) → Sortie")
    print()
    print("   CARACTÉRISTIQUES COMMUNES:")
    print("   - 20 jours de données pour prédire le jour suivant")
    print("   - Classification binaire (hausse/baisse)")
    print("   - ROLLING WINDOW : fenêtre glissante (252 jours train, 63 jours test)")
    print("   - Plus réaliste qu'extending window pour les séries temporelles")
    print("   - PARALLÉLISATION : téléchargements et prédictions en parallèle")
    print("   - Inspiré des thèses sur la prédiction de cours boursiers")
    print()
    
    print("📈 MÉTRIQUES CALCULÉES:")
    print("   - Précision du modèle")
    print("   - Rendement de la stratégie")
    print("   - Rendement Buy & Hold")
    print("   - Performance relative")
    print("   - Sharpe Ratio (performance ajustée au risque)")
    print("   - Sortino Ratio (performance ajustée au risque de baisse)")
    print("   - Maximum Drawdown (perte maximale)")
    print("   - Volatilité annualisée")
    print()
    print("📊 COMPARAISON:")
    print("   - Comparaison des performances ajustées au risque")
    print("   - Identification du meilleur modèle (global vs par secteur)")
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
            'results_sector.csv',
            'results_global.csv',
            'graphique_1_comparaison_performances.png',
            'graphique_2_distribution_sharpe.png',
            'graphique_3_top10_performances.png',
            'graphique_4_performance_vs_risque.png',
            'graphique_5_metriques_risque.png'
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
