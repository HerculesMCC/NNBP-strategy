#!/usr/bin/env python3
"""
Script de démonstration pour l'analyse de 10 actions américaines
"""

import os
import sys

def main():
    print("=== DÉMONSTRATION: ANALYSE DE 10 ACTIONS AMÉRICAINES ===")
    print()
    
    # Vérifier si les fichiers existent
    required_files = ['multi_stock_analysis.py', 'database_analyzer.py']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"❌ Fichiers manquants: {', '.join(missing_files)}")
        return
    
    print("📊 Étape 1: Analyse des 10 actions américaines")
    print("   - Téléchargement des données (ou génération simulée)")
    print("   - Entraînement des modèles MLP")
    print("   - Calcul des métriques de performance")
    print("   - Sauvegarde en base de données SQLite")
    print()
    
    # Exécuter l'analyse
    try:
        print("🚀 Lancement de l'analyse...")
        import multi_stock_analysis
        multi_stock_analysis.main()
        print("✅ Analyse terminée avec succès!")
        print()
        
        print("📈 Étape 2: Génération du rapport et des graphiques")
        print("   - Analyse des performances par action")
        print("   - Analyse par secteur")
        print("   - Génération des visualisations")
        print()
        
        # Générer le rapport
        import database_analyzer
        database_analyzer.main()
        print("✅ Rapport généré avec succès!")
        print()
        
        print("📁 Fichiers générés:")
        files_to_check = [
            'stock_predictions.db',
            'stock_analysis_results.csv',
            'stock_performance_analysis.png',
            'sector_analysis.png'
        ]
        
        for file in files_to_check:
            if os.path.exists(file):
                size = os.path.getsize(file)
                print(f"   ✓ {file} ({size:,} bytes)")
            else:
                print(f"   ✗ {file} (manquant)")
        
        print()
        print("🎯 Prochaines étapes possibles:")
        print("   1. Examiner la base de données avec un outil SQLite")
        print("   2. Modifier les paramètres dans multi_stock_analysis.py")
        print("   3. Ajouter d'autres actions à analyser")
        print("   4. Implémenter des stratégies plus avancées")
        
    except Exception as e:
        print(f"❌ Erreur lors de l'exécution: {e}")
        print("💡 Assurez-vous que toutes les dépendances sont installées:")
        print("   pip install -r requirements.txt")

if __name__ == "__main__":
    main()
