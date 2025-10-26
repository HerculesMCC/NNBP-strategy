import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Import seaborn optionnel
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    print("⚠️  seaborn non disponible - les graphiques utiliseront matplotlib uniquement")

def connect_to_database():
    """Se connecter à la base de données"""
    return sqlite3.connect('stock_predictions.db')

def get_summary_statistics():
    """Obtenir les statistiques de résumé"""
    conn = connect_to_database()
    
    query = '''
    SELECT 
        s.symbol,
        s.name,
        s.sector,
        p.model_accuracy,
        p.strategy_return,
        p.buy_hold_return,
        p.relative_performance,
        p.strategy_volatility,
        p.buy_hold_volatility,
        p.positions_taken,
        p.total_positions,
        p.created_at
    FROM predictions p
    JOIN stocks s ON p.stock_id = s.id
    ORDER BY p.relative_performance DESC
    '''
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def get_sector_analysis():
    """Analyser les performances par secteur"""
    conn = connect_to_database()
    
    query = '''
    SELECT 
        s.sector,
        COUNT(*) as stock_count,
        AVG(p.model_accuracy) as avg_accuracy,
        AVG(p.strategy_return) as avg_strategy_return,
        AVG(p.buy_hold_return) as avg_buy_hold_return,
        AVG(p.relative_performance) as avg_relative_performance,
        AVG(p.strategy_volatility) as avg_strategy_volatility,
        AVG(p.buy_hold_volatility) as avg_buy_hold_volatility
    FROM predictions p
    JOIN stocks s ON p.stock_id = s.id
    GROUP BY s.sector
    ORDER BY avg_relative_performance DESC
    '''
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def get_detailed_metrics(stock_symbol=None):
    """Obtenir les métriques détaillées pour une action spécifique ou toutes"""
    conn = connect_to_database()
    
    if stock_symbol:
        query = '''
        SELECT 
            dm.metric_name,
            dm.metric_value,
            s.symbol,
            p.created_at
        FROM detailed_metrics dm
        JOIN predictions p ON dm.prediction_id = p.id
        JOIN stocks s ON p.stock_id = s.id
        WHERE s.symbol = ?
        ORDER BY dm.metric_name
        '''
        df = pd.read_sql_query(query, conn, params=(stock_symbol,))
    else:
        query = '''
        SELECT 
            dm.metric_name,
            dm.metric_value,
            s.symbol,
            p.created_at
        FROM detailed_metrics dm
        JOIN predictions p ON dm.prediction_id = p.id
        JOIN stocks s ON p.stock_id = s.id
        ORDER BY s.symbol, dm.metric_name
        '''
        df = pd.read_sql_query(query, conn)
    
    conn.close()
    return df

def create_performance_visualization():
    """Créer des visualisations des performances"""
    df = get_summary_statistics()
    
    # Configuration du style
    if SEABORN_AVAILABLE:
        plt.style.use('seaborn-v0_8')
    else:
        plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Analyse des Performances des Actions', fontsize=16, fontweight='bold')
    
    # 1. Performance relative par action
    axes[0, 0].bar(df['symbol'], df['relative_performance'], color='skyblue', alpha=0.7)
    axes[0, 0].set_title('Performance Relative par Action')
    axes[0, 0].set_ylabel('Performance Relative')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # 2. Comparaison Strategy vs Buy&Hold
    x = range(len(df))
    width = 0.35
    axes[0, 1].bar([i - width/2 for i in x], df['strategy_return'], width, label='Stratégie', alpha=0.7)
    axes[0, 1].bar([i + width/2 for i in x], df['buy_hold_return'], width, label='Buy & Hold', alpha=0.7)
    axes[0, 1].set_title('Rendements: Stratégie vs Buy & Hold')
    axes[0, 1].set_ylabel('Rendement')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(df['symbol'], rotation=45)
    axes[0, 1].legend()
    
    # 3. Précision du modèle
    axes[1, 0].bar(df['symbol'], df['model_accuracy'], color='lightgreen', alpha=0.7)
    axes[1, 0].set_title('Précision du Modèle par Action')
    axes[1, 0].set_ylabel('Précision')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].set_ylim(0, 1)
    
    # 4. Volatilité
    axes[1, 1].bar(df['symbol'], df['strategy_volatility'], label='Stratégie', alpha=0.7)
    axes[1, 1].bar(df['symbol'], df['buy_hold_volatility'], label='Buy & Hold', alpha=0.7)
    axes[1, 1].set_title('Volatilité Annualisée')
    axes[1, 1].set_ylabel('Volatilité')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('stock_performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_sector_analysis():
    """Créer une analyse par secteur"""
    df_sector = get_sector_analysis()
    
    if df_sector.empty:
        print("Aucune donnée de secteur disponible")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Analyse par Secteur', fontsize=16, fontweight='bold')
    
    # Performance relative par secteur
    axes[0].bar(df_sector['sector'], df_sector['avg_relative_performance'], 
               color='lightcoral', alpha=0.7)
    axes[0].set_title('Performance Relative Moyenne par Secteur')
    axes[0].set_ylabel('Performance Relative Moyenne')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # Précision par secteur
    axes[1].bar(df_sector['sector'], df_sector['avg_accuracy'], 
               color='lightblue', alpha=0.7)
    axes[1].set_title('Précision Moyenne par Secteur')
    axes[1].set_ylabel('Précision Moyenne')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('sector_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_report():
    """Générer un rapport complet"""
    print("="*80)
    print("RAPPORT D'ANALYSE DES ACTIONS AMÉRICAINES")
    print("="*80)
    print(f"Généré le: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Statistiques générales
    df = get_summary_statistics()
    if df.empty:
        print("Aucune donnée disponible dans la base de données.")
        return
    
    print("1. STATISTIQUES GÉNÉRALES")
    print("-" * 40)
    print(f"Nombre d'actions analysées: {len(df)}")
    print(f"Précision moyenne du modèle: {df['model_accuracy'].mean():.3f}")
    print(f"Rendement stratégie moyen: {df['strategy_return'].mean():.3f} ({df['strategy_return'].mean()*100:.2f}%)")
    print(f"Rendement buy&hold moyen: {df['buy_hold_return'].mean():.3f} ({df['buy_hold_return'].mean()*100:.2f}%)")
    print(f"Performance relative moyenne: {df['relative_performance'].mean():.3f} ({df['relative_performance'].mean()*100:.2f}%)")
    print()
    
    # Top 3 et Bottom 3
    print("2. CLASSEMENT DES PERFORMANCES")
    print("-" * 40)
    top_3 = df.nlargest(3, 'relative_performance')
    bottom_3 = df.nsmallest(3, 'relative_performance')
    
    print("TOP 3:")
    for idx, row in top_3.iterrows():
        print(f"  {row['symbol']}: {row['relative_performance']:.3f} ({row['relative_performance']*100:.2f}%)")
    
    print("\nBOTTOM 3:")
    for idx, row in bottom_3.iterrows():
        print(f"  {row['symbol']}: {row['relative_performance']:.3f} ({row['relative_performance']*100:.2f}%)")
    print()
    
    # Analyse par secteur
    df_sector = get_sector_analysis()
    if not df_sector.empty:
        print("3. ANALYSE PAR SECTEUR")
        print("-" * 40)
        for _, row in df_sector.iterrows():
            print(f"{row['sector']}: {row['stock_count']} actions, "
                  f"Performance moyenne: {row['avg_relative_performance']:.3f}")
        print()
    
    # Métriques de risque
    print("4. MÉTRIQUES DE RISQUE")
    print("-" * 40)
    print(f"Volatilité stratégie moyenne: {df['strategy_volatility'].mean():.3f} ({df['strategy_volatility'].mean()*100:.2f}%)")
    print(f"Volatilité buy&hold moyenne: {df['buy_hold_volatility'].mean():.3f} ({df['buy_hold_volatility'].mean()*100:.2f}%)")
    print(f"Ratio positions prises moyen: {df['positions_taken'].sum() / df['total_positions'].sum():.3f}")
    print()
    
    # Recommandations
    print("5. RECOMMANDATIONS")
    print("-" * 40)
    best_stock = df.loc[df['relative_performance'].idxmax()]
    worst_stock = df.loc[df['relative_performance'].idxmin()]
    
    print(f"Action recommandée: {best_stock['symbol']} (Performance: {best_stock['relative_performance']:.3f})")
    print(f"Action à éviter: {worst_stock['symbol']} (Performance: {worst_stock['relative_performance']:.3f})")
    
    if df['relative_performance'].mean() > 0:
        print("✓ La stratégie MLP montre une performance positive en moyenne")
    else:
        print("✗ La stratégie MLP sous-performe le buy&hold en moyenne")
    
    print("\n" + "="*80)

def main():
    """Fonction principale"""
    try:
        # Générer le rapport
        generate_report()
        
        # Créer les visualisations
        print("\nGénération des graphiques...")
        create_performance_visualization()
        create_sector_analysis()
        
        print("\nAnalyse terminée ! Fichiers générés:")
        print("- stock_performance_analysis.png")
        print("- sector_analysis.png")
        
    except Exception as e:
        print(f"Erreur lors de l'analyse: {e}")

if __name__ == "__main__":
    main()
