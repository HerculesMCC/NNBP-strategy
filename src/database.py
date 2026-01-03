"""
Module pour la gestion de la base de données
"""
import sqlite3
import os


def create_database(output_dir='outs'):
    """
    Créer la base de données et vider les tables à chaque run
    
    Args:
        output_dir: Dossier où sauvegarder la base de données
    
    Returns:
        Connexion SQLite
    """
    # Créer le dossier de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)
    
    db_path = os.path.join(output_dir, 'stock_analysis.db')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Table avec tous les résultats (modèles par secteur)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS stock_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            name TEXT NOT NULL,
            sector TEXT NOT NULL,
            model_type TEXT NOT NULL DEFAULT 'sector',
            accuracy REAL NOT NULL,
            strategy_return REAL NOT NULL,
            buy_hold_return REAL NOT NULL,
            performance REAL NOT NULL,
            sharpe_ratio REAL,
            sortino_ratio REAL,
            max_drawdown REAL,
            volatility REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Table pour les résultats du modèle global
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS global_model_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            name TEXT NOT NULL,
            sector TEXT NOT NULL,
            model_type TEXT NOT NULL DEFAULT 'global',
            accuracy REAL NOT NULL,
            strategy_return REAL NOT NULL,
            buy_hold_return REAL NOT NULL,
            performance REAL NOT NULL,
            sharpe_ratio REAL,
            sortino_ratio REAL,
            max_drawdown REAL,
            volatility REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Table pour les métriques agrégées de comparaison
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_comparison (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_type TEXT NOT NULL,
            total_stocks INTEGER NOT NULL,
            avg_accuracy REAL NOT NULL,
            avg_strategy_return REAL NOT NULL,
            avg_buy_hold_return REAL NOT NULL,
            avg_performance REAL NOT NULL,
            avg_sharpe_ratio REAL,
            avg_sortino_ratio REAL,
            avg_max_drawdown REAL,
            avg_volatility REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Migration: ajouter les nouvelles colonnes si elles n'existent pas
    try:
        cursor.execute("PRAGMA table_info(stock_results)")
        columns = [col[1] for col in cursor.fetchall()]
        
        new_columns = {
            'sector': 'TEXT',
            'model_type': 'TEXT DEFAULT "sector"',
            'sharpe_ratio': 'REAL',
            'sortino_ratio': 'REAL',
            'max_drawdown': 'REAL',
            'volatility': 'REAL'
        }
        
        for col_name, col_type in new_columns.items():
            if col_name not in columns:
                cursor.execute(f'ALTER TABLE stock_results ADD COLUMN {col_name} {col_type}')
                print(f"  ✓ Colonne '{col_name}' ajoutée à stock_results")
    except sqlite3.OperationalError as e:
        pass
    
    # VIDER LES TABLES à chaque run
    cursor.execute('DELETE FROM stock_results')
    cursor.execute('DELETE FROM global_model_results')
    cursor.execute('DELETE FROM model_comparison')
    print("  ✓ Tables vidées pour un nouveau run")
    
    conn.commit()
    return conn

