#!/usr/bin/env python3
"""
Démonstration de l'analyse d'actions américaines
"""

import os

def main():
    print("=== DÉMONSTRATION ===")
    print()
    
    print("🎯 OBJECTIF:")
    print("   Analyser 5 actions américaines avec un modèle MLP")
    print("   Comparer la performance vs Buy & Hold")
    print("   Stocker les résultats dans une base de données SQLite")
    print()
    
    print("📊 ACTIONS ANALYSÉES:")
    stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    for i, stock in enumerate(stocks, 1):
        print(f"   {i}. {stock}")
    print()
    
    print("🔧 MODÈLE UTILISÉ:")
    print("   - Réseau de neurones (2 couches)")
    print("   - 5 jours de données pour prédire le jour suivant")
    print("   - Classification binaire (hausse/baisse)")
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
