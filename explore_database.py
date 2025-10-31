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
    df = pd.read_sql_query(
        """
        SELECT 
            id,
            symbol,
            name,
            ROUND(accuracy * 100, 2) AS accuracy_pct,
            ROUND(strategy_return * 100, 2) AS strategy_return_pct,
            ROUND(buy_hold_return * 100, 2) AS buy_hold_return_pct,
            ROUND(performance * 100, 2) AS performance_pct,
            created_at
        FROM stock_results
        """,
        conn,
    )
    # Afficher avec le symbole %
    for col in [
        "accuracy_pct",
        "strategy_return_pct",
        "buy_hold_return_pct",
        "performance_pct",
    ]:
        df[col] = df[col].map(lambda x: f"{x:.2f}%")
    print(df.to_string(index=False))
    
    # 4. Requêtes utiles
    print("\n🔍 REQUÊTES UTILES:")
    
    # Meilleure performance
    print("\n🏆 MEILLEURE PERFORMANCE:")
    best = pd.read_sql_query(
        """
        SELECT 
            symbol, 
            name, 
            ROUND(performance * 100, 2) AS performance_pct,
            ROUND(accuracy * 100, 2)   AS accuracy_pct
        FROM stock_results 
        ORDER BY performance DESC 
        LIMIT 1
    """,
        conn,
    )
    for col in ["performance_pct", "accuracy_pct"]:
        best[col] = best[col].map(lambda x: f"{x:.2f}%")
    print(best.to_string(index=False))
    
    # Performance moyenne
    print("\n📈 PERFORMANCE MOYENNE:")
    avg = pd.read_sql_query(
        """
        SELECT 
            ROUND(AVG(performance) * 100, 2) AS performance_moyenne_pct,
            ROUND(AVG(accuracy) * 100, 2)    AS precision_moyenne_pct,
            COUNT(*)                         AS nombre_actions
        FROM stock_results
    """,
        conn,
    )
    for col in ["performance_moyenne_pct", "precision_moyenne_pct"]:
        avg[col] = avg[col].map(lambda x: f"{x:.2f}%")
    print(avg.to_string(index=False))
    
    # Classement complet
    print("\n📋 CLASSEMENT COMPLET:")
    ranking = pd.read_sql_query(
        """
        SELECT 
            symbol, 
            name, 
            ROUND(performance * 100, 2) AS performance_pct,
            ROUND(accuracy * 100, 2)   AS accuracy_pct
        FROM stock_results 
        ORDER BY performance DESC
    """,
        conn,
    )
    for col in ["performance_pct", "accuracy_pct"]:
        ranking[col] = ranking[col].map(lambda x: f"{x:.2f}%")
    print(ranking.to_string(index=False))
    
    conn.close()

def custom_query():
    """Exécuter une requête personnalisée"""
    # Utiliser la même base que l'analyse
    conn = sqlite3.connect('stock_analysis.db')
    
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
    for col in ["performance_pourcent", "precision_pourcent"]:
        df[col] = df[col].map(lambda x: f"{x:.2f}%")
    print("Actions avec performance positive:")
    print(df.to_string(index=False))
    
    conn.close()

if __name__ == "__main__":
    explore_database()
    custom_query()
