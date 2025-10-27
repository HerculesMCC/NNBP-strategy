import sqlite3
import pandas as pd

def explore_database():
    """Explorer la base de données avec Python"""
    conn = sqlite3.connect('stock_analysis.db')
    
    print("🗄️ EXPLORATION DE LA BASE DE DONNÉES")
    print("="*50)
    
    # 1. Voir toutes les tables
    print("\n📋 TABLES DISPONIBLES:")
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    for table in tables:
        print(f"  - {table[0]}")
    
    # 2. Voir la structure de la table principale
    print("\n🏗️ STRUCTURE DE LA TABLE 'stock_results':")
    cursor.execute("PRAGMA table_info(stock_results);")
    columns = cursor.fetchall()
    for col in columns:
        print(f"  - {col[1]} ({col[2]})")
    
    # 3. Voir tous les enregistrements
    print("\n📊 DONNÉES COMPLÈTES:")
    df = pd.read_sql_query("SELECT * FROM stock_results", conn)
    print(df.to_string(index=False))
    
    # 4. Requêtes utiles
    print("\n🔍 REQUÊTES UTILES:")
    
    # Meilleure performance
    print("\n🏆 MEILLEURE PERFORMANCE:")
    best = pd.read_sql_query("""
        SELECT symbol, name, performance, accuracy 
        FROM stock_results 
        ORDER BY performance DESC 
        LIMIT 1
    """, conn)
    print(best.to_string(index=False))
    
    # Performance moyenne
    print("\n📈 PERFORMANCE MOYENNE:")
    avg = pd.read_sql_query("""
        SELECT 
            AVG(performance) as performance_moyenne,
            AVG(accuracy) as precision_moyenne,
            COUNT(*) as nombre_actions
        FROM stock_results
    """, conn)
    print(avg.to_string(index=False))
    
    # Classement complet
    print("\n📋 CLASSEMENT COMPLET:")
    ranking = pd.read_sql_query("""
        SELECT symbol, name, performance, accuracy
        FROM stock_results 
        ORDER BY performance DESC
    """, conn)
    print(ranking.to_string(index=False))
    
    conn.close()

def custom_query():
    """Exécuter une requête personnalisée"""
    conn = sqlite3.connect('simple_stock_analysis.db')
    
    print("\n🔍 REQUÊTE PERSONNALISÉE")
    print("="*30)
    
    # Exemple de requête
    query = """
    SELECT symbol, 
           ROUND(performance * 100, 2) as performance_pourcent,
           ROUND(accuracy * 100, 2) as precision_pourcent
    FROM stock_results 
    WHERE performance > 0
    ORDER BY performance DESC
    """
    
    df = pd.read_sql_query(query, conn)
    print("Actions avec performance positive:")
    print(df.to_string(index=False))
    
    conn.close()

if __name__ == "__main__":
    explore_database()
    custom_query()
