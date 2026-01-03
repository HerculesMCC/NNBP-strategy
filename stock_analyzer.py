import sqlite3
import pandas as pd
import numpy as np

# Le secteur est maintenant stocké dans la base de données, pas besoin de mapping

def get_latest_analysis(conn):
    """Récupérer uniquement la dernière analyse pour chaque action"""
    # Vérifier si la colonne sector existe
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(stock_results)")
    columns = [col[1] for col in cursor.fetchall()]
    has_sector = 'sector' in columns
    
    if has_sector:
        query = '''
        SELECT symbol, name, sector, accuracy, strategy_return, buy_hold_return, performance, created_at
        FROM stock_results
        WHERE id IN (
            SELECT MAX(id) 
            FROM stock_results 
            GROUP BY symbol
        )
        ORDER BY performance DESC
        '''
    else:
        # Fallback si la colonne sector n'existe pas
        query = '''
        SELECT symbol, name, accuracy, strategy_return, buy_hold_return, performance, created_at
        FROM stock_results
        WHERE id IN (
            SELECT MAX(id) 
            FROM stock_results 
            GROUP BY symbol
        )
        ORDER BY performance DESC
        '''
    
    df = pd.read_sql_query(query, conn)
    
    # Ajouter la colonne sector si elle n'existe pas (avec mapping depuis stock_analysis)
    if 'sector' not in df.columns:
        try:
            import stock_analysis
            sector_mapping = {s['symbol']: s['sector'] for s in stock_analysis.STOCKS}
            df['sector'] = df['symbol'].map(sector_mapping).fillna('N/A')
        except:
            df['sector'] = 'N/A'
    
    return df

def analyze_results():
    """Analyser les résultats de la base de données avec analyses approfondies"""
    conn = sqlite3.connect('stock_analysis.db')
    
    # Récupérer uniquement les dernières analyses (éviter les doublons)
    df = get_latest_analysis(conn)
    
    conn.close()
    
    # Le secteur est déjà dans la base de données
    if 'sector' not in df.columns:
        df['sector'] = 'N/A'
    
    if df.empty:
        print("Aucune donnée trouvée dans la base de données.")
        return
    
    print("="*70)
    print("ANALYSE APPROFONDIE DE LA BASE DE DONNÉES")
    print("="*70)
    print()
    
    # 1. STATISTIQUES GÉNÉRALES
    print("📊 STATISTIQUES GÉNÉRALES")
    print("-" * 70)
    print(f"Nombre d'actions analysées: {len(df)}/100")
    print(f"Nombre d'actions uniques: {df['symbol'].nunique()}")
    print()
    
    print("Moyennes:")
    print(f"  • Précision moyenne: {df['accuracy'].mean():.2%}")
    print(f"  • Performance moyenne: {df['performance'].mean():.2%}")
    print(f"  • Rendement stratégie moyen: {df['strategy_return'].mean():.2%}")
    print(f"  • Rendement Buy & Hold moyen: {df['buy_hold_return'].mean():.2%}")
    print()
    
    print("Médianes:")
    print(f"  • Précision médiane: {df['accuracy'].median():.2%}")
    print(f"  • Performance médiane: {df['performance'].median():.2%}")
    print()
    
    print("Écarts-types:")
    print(f"  • Écart-type précision: {df['accuracy'].std():.2%}")
    print(f"  • Écart-type performance: {df['performance'].std():.2%}")
    print()
    
    # 2. DISTRIBUTION DES PERFORMANCES
    print("📈 DISTRIBUTION DES PERFORMANCES")
    print("-" * 70)
    positive_perf = (df['performance'] > 0).sum()
    negative_perf = (df['performance'] <= 0).sum()
    print(f"Actions avec performance positive: {positive_perf} ({positive_perf/len(df):.1%})")
    print(f"Actions avec performance négative: {negative_perf} ({negative_perf/len(df):.1%})")
    
    # Comparaison stratégie vs Buy & Hold
    strategy_better = (df['strategy_return'] > df['buy_hold_return']).sum()
    print(f"Stratégie meilleure que Buy & Hold: {strategy_better} actions ({strategy_better/len(df):.1%})")
    print()
    
    # 3. TOP/BOTTOM PERFORMANCES
    print("🏆 TOP 5 PAR PERFORMANCE")
    print("-" * 70)
    top5 = df.nlargest(5, 'performance')
    for i, (_, row) in enumerate(top5.iterrows(), 1):
        sector_str = row['sector'] if pd.notna(row['sector']) else 'N/A'
        print(f"{i}. {row['symbol']:6s} ({sector_str:<30s}) | "
              f"Performance: {row['performance']:7.2%} | "
              f"Stratégie: {row['strategy_return']:7.2%} | "
              f"B&H: {row['buy_hold_return']:7.2%} | "
              f"Précision: {row['accuracy']:5.2%}")
    print()
    
    print("📉 5 PIRE PERFORMANCES")
    print("-" * 70)
    bottom5 = df.nsmallest(5, 'performance')
    for i, (_, row) in enumerate(bottom5.iterrows(), 1):
        sector_str = row['sector'] if pd.notna(row['sector']) else 'N/A'
        print(f"{i}. {row['symbol']:6s} ({sector_str:<30s}) | "
              f"Performance: {row['performance']:7.2%} | "
              f"Stratégie: {row['strategy_return']:7.2%} | "
              f"B&H: {row['buy_hold_return']:7.2%} | "
              f"Précision: {row['accuracy']:5.2%}")
    print()
    
    # 4. TOP PRÉCISIONS
    print("🎯 TOP 5 PAR PRÉCISION")
    print("-" * 70)
    top5_acc = df.nlargest(5, 'accuracy')
    for i, (_, row) in enumerate(top5_acc.iterrows(), 1):
        sector_str = row['sector'] if pd.notna(row['sector']) else 'N/A'
        print(f"{i}. {row['symbol']:6s} ({sector_str:<30s}) | "
              f"Précision: {row['accuracy']:5.2%} | "
              f"Performance: {row['performance']:7.2%}")
    print()
    
    # 5. ANALYSE PAR SECTEUR
    if df['sector'].notna().any():
        print("🏭 ANALYSE PAR SECTEUR")
        print("-" * 70)
        sector_stats = df.groupby('sector').agg({
            'performance': ['mean', 'count'],
            'accuracy': 'mean',
            'strategy_return': 'mean',
            'buy_hold_return': 'mean'
        }).round(4)
        
        sector_stats.columns = ['Performance_moy', 'Nb_actions', 'Précision_moy', 'Stratégie_moy', 'B&H_moy']
        sector_stats = sector_stats.sort_values('Performance_moy', ascending=False)
        
        print(f"{'Secteur':<30} {'Nb':<4} {'Perf. moy':<10} {'Préc. moy':<10} {'Strat. moy':<10} {'B&H moy':<10}")
        print("-" * 70)
        for sector, row in sector_stats.iterrows():
            print(f"{sector:<30} {int(row['Nb_actions']):<4} "
                  f"{row['Performance_moy']:>9.2%} {row['Précision_moy']:>9.2%} "
                  f"{row['Stratégie_moy']:>9.2%} {row['B&H_moy']:>9.2%}")
        print()
    
    # 6. RÉSULTATS DES 100 ACTIONS (liste complète par secteur)
    print("📋 RÉSULTATS DES 100 ACTIONS (par secteur)")
    print("-" * 70)
    
    # Importer la liste complète depuis stock_analysis
    try:
        import stock_analysis
        all_expected_stocks = [(s['symbol'], s['sector'], s['name']) for s in stock_analysis.STOCKS]
        df_all = pd.DataFrame(all_expected_stocks, columns=['symbol', 'sector', 'name_expected'])
    except Exception as e:
        # Fallback si import échoue
        print(f"  ⚠ Impossible d'importer la liste complète des actions: {e}")
        if 'sector' in df.columns and 'name' in df.columns:
            df_all = df[['symbol', 'sector', 'name']].copy()
            df_all['name_expected'] = df_all['name']
        else:
            df_all = df[['symbol']].copy()
            df_all['sector'] = 'N/A'
            df_all['name_expected'] = 'N/A'
    
    # Fusionner avec les résultats existants
    df_merged = df_all.merge(df[['symbol', 'accuracy', 'strategy_return', 'buy_hold_return', 'performance', 'name']], 
                             on='symbol', how='left', suffixes=('', '_actual'))
    
    # Utiliser le nom de la base si disponible, sinon celui attendu
    if 'name' in df_merged.columns:
        df_merged['name'] = df_merged['name'].fillna(df_merged['name_expected'])
    else:
        df_merged['name'] = df_merged['name_expected']
    
    df_merged = df_merged.drop(columns=['name_expected'], errors='ignore')
    
    # Afficher par secteur
    current_sector = None
    for _, row in df_merged.iterrows():
        if row['sector'] != current_sector:
            if current_sector is not None:
                print()  # Ligne vide entre secteurs
            print(f"\n🏭 {row['sector']}:")
            print(f"{'Symbole':<10} {'Nom':<35} {'Performance':<12} {'Stratégie':<12} {'B&H':<12} {'Précision':<10}")
            print("-" * 70)
            current_sector = row['sector']
        
        if pd.notna(row['performance']):
            name_str = row['name'] if pd.notna(row['name']) else 'N/A'
            print(f"{row['symbol']:<10} {name_str:<35} "
                  f"{row['performance']:>11.2%} {row['strategy_return']:>11.2%} "
                  f"{row['buy_hold_return']:>11.2%} {row['accuracy']:>9.2%}")
        else:
            print(f"{row['symbol']:<10} {'(Non analysée)':<35} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<10}")
    
    print()
    print(f"✅ Actions analysées: {len(df)}/100")
    print("="*70)

def show_database_structure():
    """Afficher la structure de la base de données"""
    conn = sqlite3.connect('stock_analysis.db')
    cursor = conn.cursor()
    
    print("🗄️ STRUCTURE DE LA BASE DE DONNÉES")
    print("-" * 40)
    
    # Afficher les tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    for table in tables:
        table_name = table[0]
        print(f"\nTable: {table_name}")
        
        # Afficher la structure
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()
        
        for col in columns:
            print(f"  - {col[1]} ({col[2]})")
    
    conn.close()

def main():
    """Fonction principale"""
    print("=== ANALYSEUR APPROFONDI DE RÉSULTATS ===")
    print()
    
    try:
        # Afficher la structure
        show_database_structure()
        print()
        
        # Analyser les résultats avec analyses approfondies
        analyze_results()
        
    except Exception as e:
        print(f"Erreur: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
