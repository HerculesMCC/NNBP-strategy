"""
Module pour la génération de graphiques
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Import seaborn avec gestion d'erreur
try:
    import seaborn as sns
    sns.set_palette("husl")
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("  ⚠ Warning: seaborn non disponible, utilisation de matplotlib uniquement")

# Configuration des graphiques
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        try:
            plt.style.use('ggplot')
        except:
            plt.style.use('default')


def generate_visualizations(df_sector, df_global, sector_metrics, global_metrics, output_dir='outs'):
    """
    Générer 5 graphiques PNG pertinents pour l'analyse
    
    Args:
        df_sector: DataFrame avec les résultats du modèle par secteur
        df_global: DataFrame avec les résultats du modèle global
        sector_metrics: Dictionnaire avec les métriques moyennes du modèle par secteur
        global_metrics: Dictionnaire avec les métriques moyennes du modèle global
        output_dir: Dossier où sauvegarder les graphiques
    """
    # Créer le dossier de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n📊 Génération des graphiques...")
    
    # 1. COMPARAISON DES PERFORMANCES MOYENNES
    fig, ax = plt.subplots(figsize=(12, 6))
    models = ['Modèle Global', 'Modèle par Secteur']
    metrics = {
        'Performance (%)': [global_metrics['avg_performance']*100, sector_metrics['avg_performance']*100],
        'Rendement Stratégie (%)': [global_metrics['avg_strategy_return']*100, sector_metrics['avg_strategy_return']*100],
        'Précision (%)': [global_metrics['avg_accuracy']*100, sector_metrics['avg_accuracy']*100]
    }
    
    x = np.arange(len(models))
    width = 0.25
    multiplier = 0
    
    for metric, values in metrics.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, values, width, label=metric)
        ax.bar_label(rects, fmt='%.2f%%', padding=3)
        multiplier += 1
    
    ax.set_ylabel('Valeur (%)', fontsize=12)
    ax.set_title('Comparaison des Performances Moyennes\nModèle Global vs Modèle par Secteur', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(models)
    ax.legend(loc='upper left', ncol=1)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'graphique_1_comparaison_performances.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Graphique 1: Comparaison des performances moyennes")
    
    # 2. DISTRIBUTION DES SHARPE RATIOS
    fig, ax = plt.subplots(figsize=(10, 6))
    sharpe_global = df_global['sharpe_ratio'].fillna(0) if 'sharpe_ratio' in df_global.columns else pd.Series([0]*len(df_global))
    sharpe_sector = df_sector['sharpe_ratio'].fillna(0) if 'sharpe_ratio' in df_sector.columns else pd.Series([0]*len(df_sector))
    sharpe_data = [sharpe_global, sharpe_sector]
    box = ax.boxplot(sharpe_data, labels=['Modèle Global', 'Modèle par Secteur'], 
                     patch_artist=True, showmeans=True)
    
    colors = ['#3498db', '#e74c3c']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Sharpe Ratio', fontsize=12)
    ax.set_title('Distribution des Sharpe Ratios\nComparaison des Modèles', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'graphique_2_distribution_sharpe.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Graphique 2: Distribution des Sharpe Ratios")
    
    # 3. TOP 10 ACTIONS PAR PERFORMANCE
    fig, ax = plt.subplots(figsize=(12, 7))
    top_10 = df_sector.nlargest(10, 'performance')
    colors_bar = ['#2ecc71' if x > 0 else '#e74c3c' for x in top_10['performance']]
    bars = ax.barh(range(len(top_10)), top_10['performance']*100, color=colors_bar, alpha=0.8)
    ax.set_yticks(range(len(top_10)))
    if 'sector' in top_10.columns:
        labels = [f"{row['symbol']} ({row['sector']})" for _, row in top_10.iterrows()]
    else:
        labels = [f"{row['symbol']}" for _, row in top_10.iterrows()]
    ax.set_yticklabels(labels)
    ax.set_xlabel('Performance Relative (%)', fontsize=12)
    ax.set_title('Top 10 Actions par Performance\nModèle par Secteur', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    ax.grid(axis='x', alpha=0.3)
    
    for i, (idx, row) in enumerate(top_10.iterrows()):
        ax.text(row['performance']*100 + (0.5 if row['performance'] > 0 else -1.5), 
                i, f"{row['performance']*100:.2f}%", 
                va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'graphique_3_top10_performances.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Graphique 3: Top 10 actions par performance")
    
    # 4. SCATTER PLOT: Performance vs Sharpe Ratio
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    sharpe_global = df_global['sharpe_ratio'].fillna(0) if 'sharpe_ratio' in df_global.columns else pd.Series([0]*len(df_global))
    perf_global = df_global['performance']*100
    scatter1 = ax1.scatter(sharpe_global, perf_global, alpha=0.6, s=60, c=perf_global, 
                          cmap='RdYlGn', edgecolors='black', linewidth=0.5)
    ax1.set_xlabel('Sharpe Ratio', fontsize=11)
    ax1.set_ylabel('Performance Relative (%)', fontsize=11)
    ax1.set_title('Modèle Global\nPerformance vs Risque Ajusté', fontsize=12, fontweight='bold')
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax1.axvline(x=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax1.grid(alpha=0.3)
    plt.colorbar(scatter1, ax=ax1, label='Performance (%)')
    
    sharpe_sector = df_sector['sharpe_ratio'].fillna(0) if 'sharpe_ratio' in df_sector.columns else pd.Series([0]*len(df_sector))
    perf_sector = df_sector['performance']*100
    scatter2 = ax2.scatter(sharpe_sector, perf_sector, alpha=0.6, s=60, c=perf_sector, 
                          cmap='RdYlGn', edgecolors='black', linewidth=0.5)
    ax2.set_xlabel('Sharpe Ratio', fontsize=11)
    ax2.set_ylabel('Performance Relative (%)', fontsize=11)
    ax2.set_title('Modèle par Secteur\nPerformance vs Risque Ajusté', fontsize=12, fontweight='bold')
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax2.axvline(x=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax2.grid(alpha=0.3)
    plt.colorbar(scatter2, ax=ax2, label='Performance (%)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'graphique_4_performance_vs_risque.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Graphique 4: Performance vs Risque ajusté (Sharpe Ratio)")
    
    # 5. COMPARAISON DES MÉTRIQUES AJUSTÉES AU RISQUE
    fig, ax = plt.subplots(figsize=(12, 7))
    risk_metrics = {
        'Sharpe Ratio': [global_metrics['avg_sharpe_ratio'], sector_metrics['avg_sharpe_ratio']],
        'Sortino Ratio': [global_metrics['avg_sortino_ratio'], sector_metrics['avg_sortino_ratio']],
        'Max Drawdown (%)': [abs(global_metrics['avg_max_drawdown'])*100, abs(sector_metrics['avg_max_drawdown'])*100],
        'Volatilité (%)': [global_metrics['avg_volatility']*100, sector_metrics['avg_volatility']*100]
    }
    
    x = np.arange(len(risk_metrics))
    width = 0.35
    models_labels = ['Modèle Global', 'Modèle par Secteur']
    
    for i, (metric, values) in enumerate(risk_metrics.items()):
        offset = width * i
        x_pos = np.array([0, 1]) + offset - width * (len(risk_metrics) - 1) / 2
        bars = ax.bar(x_pos, values, width, label=metric, alpha=0.8)
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.2f}' if 'Ratio' in metric else f'{val:.2f}%',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_ylabel('Valeur', fontsize=12)
    ax.set_title('Comparaison des Métriques Ajustées au Risque\nModèle Global vs Modèle par Secteur', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(models_labels)
    ax.legend(loc='upper left', ncol=2)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'graphique_5_metriques_risque.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Graphique 5: Métriques ajustées au risque")
    
    print(f"\n✅ 5 graphiques générés avec succès dans '{output_dir}/'!")
    print("   - graphique_1_comparaison_performances.png")
    print("   - graphique_2_distribution_sharpe.png")
    print("   - graphique_3_top10_performances.png")
    print("   - graphique_4_performance_vs_risque.png")
    print("   - graphique_5_metriques_risque.png")

