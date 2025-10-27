import sqlite3
import pandas as pd

def analyze_results():
    """Analyser les résultats de la base de données"""
    conn = sqlite3.connect('stock_analysis.db')
    
    # Récupérer tous les résultats
    query = '''
    SELECT symbol, name, accuracy, strategy_return, buy_hold_return, performance
    FROM stock_results
    ORDER BY performance DESC
    '''
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if df.empty:
        print("Aucune donnée trouvée dans la base de données.")
        return
    
    print("="*60)
    print("ANALYSE DES RÉSULTATS")
    print("="*60)
    print()
    
    # Statistiques générales
    print("📊 STATISTIQUES GÉNÉRALES")
    print("-" * 30)
    print(f"Nombre d'actions analysées: {len(df)}")
    print(f"Précision moyenne: {df['accuracy'].mean():.2f}")
    print(f"Performance moyenne: {df['performance'].mean():.2f}")
    print()
    
    # Meilleure et pire performance
    best = df.loc[df['performance'].idxmax()]
    worst = df.loc[df['performance'].idxmin()]
    
    print("🏆 MEILLEURE PERFORMANCE")
    print("-" * 30)
    print(f"Action: {best['symbol']} ({best['name']})")
    print(f"Performance: {best['performance']:.2f}")
    print(f"Précision: {best['accuracy']:.2f}")
    print()
    
    print("📉 PIRE PERFORMANCE")
    print("-" * 30)
    print(f"Action: {worst['symbol']} ({worst['name']})")
    print(f"Performance: {worst['performance']:.2f}")
    print(f"Précision: {worst['accuracy']:.2f}")
    print()
    
    # Classement complet
    print("📈 CLASSEMENT COMPLET")
    print("-" * 30)
    for i, (_, row) in enumerate(df.iterrows(), 1):
        print(f"{i}. {row['symbol']}: {row['performance']:.2f} (Précision: {row['accuracy']:.2f})")
    
    print()
    print("="*60)

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
    print("=== ANALYSEUR DE RÉSULTATS ===")
    print()
    
    try:
        # Afficher la structure
        show_database_structure()
        print()
        
        # Analyser les résultats
        analyze_results()
        
    except Exception as e:
        print(f"Erreur: {e}")

if __name__ == "__main__":
    main()
