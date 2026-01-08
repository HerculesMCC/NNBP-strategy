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
    # CORRECTION: Les métriques étaient inversées
    # - strategy_return = rendement total de la stratégie (produit cumulatif sur la période de test)
    # - performance = surperformance vs Buy & Hold (strategy_return - buy_hold_return)
    metrics = {
        'Rendement Stratégie (%)': [global_metrics['avg_strategy_return']*100, sector_metrics['avg_strategy_return']*100],
        'Performance vs Buy & Hold (%)': [global_metrics['avg_performance']*100, sector_metrics['avg_performance']*100],
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
    fig, ax = plt.subplots(figsize=(14, 8))
    top_10 = df_sector.nlargest(10, 'performance')
    colors_bar = ['#2ecc71' if x > 0 else '#e74c3c' for x in top_10['performance']]
    bars = ax.barh(range(len(top_10)), top_10['performance']*100, color=colors_bar, alpha=0.8)
    ax.set_yticks(range(len(top_10)))
    
    # Labels avec ticker ET secteur bien visibles
    labels = []
    for _, row in top_10.iterrows():
        symbol = row['symbol']
        if 'sector' in row:
            sector = str(row['sector'])
            # Raccourcir le secteur si trop long
            if len(sector) > 18:
                sector = sector[:15] + '...'
            labels.append(f"{symbol} - {sector}")
        else:
            labels.append(f"{symbol}")
    
    ax.set_yticklabels(labels, fontsize=10, fontweight='bold')
    ax.set_xlabel('Performance Relative (%)', fontsize=12, fontweight='bold')
    ax.set_title('Top 10 Actions par Performance\nModèle par Secteur', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    ax.grid(axis='x', alpha=0.3)
    
    # Ajouter les valeurs de performance
    for i, (idx, row) in enumerate(top_10.iterrows()):
        perf_value = row['performance']*100
        # Valeur de performance à côté de la barre
        ax.text(perf_value + (0.8 if perf_value > 0 else -2.5), 
                i, f"{perf_value:.2f}%", 
                va='center', fontsize=9, fontweight='bold', color='black')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'graphique_3_top10_performances.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Graphique 3: Top 10 actions par performance")
    
    # 4. SCATTER PLOT: Performance vs Sharpe Ratio
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Panneau gauche : Modèle Global
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
    
    # Panneau droit : Modèle par Secteur avec annotations
    sharpe_sector = df_sector['sharpe_ratio'].fillna(0) if 'sharpe_ratio' in df_sector.columns else pd.Series([0]*len(df_sector))
    perf_sector = df_sector['performance']*100
    scatter2 = ax2.scatter(sharpe_sector, perf_sector, alpha=0.6, s=60, c=perf_sector, 
                          cmap='RdYlGn', edgecolors='black', linewidth=0.5)
    
    # Ajouter les annotations avec tickers et secteurs pour TOUS les points du modèle par secteur
    if 'symbol' in df_sector.columns and 'sector' in df_sector.columns:
        try:
            # Annoter TOUS les points avec le ticker et le secteur
            for idx in df_sector.index:
                row = df_sector.loc[idx]
                symbol = str(row['symbol'])
                sector = str(row['sector'])
                
                # Récupérer les valeurs correspondantes pour le scatter plot
                pos_in_df = df_sector.index.get_loc(idx)
                if pos_in_df < len(sharpe_sector) and pos_in_df < len(perf_sector):
                    x_pos = float(sharpe_sector.iloc[pos_in_df] if isinstance(sharpe_sector, pd.Series) else sharpe_sector[pos_in_df])
                    y_pos = float(perf_sector.iloc[pos_in_df] if isinstance(perf_sector, pd.Series) else perf_sector[pos_in_df])
                    
                    # Raccourcir le nom du secteur si trop long
                    sector_short = sector[:10] + '...' if len(sector) > 13 else sector
                    
                    # Annotation compacte avec ticker et secteur (petit texte pour éviter la surcharge)
                    ax2.annotate(f'{symbol}\n({sector_short})',
                               xy=(x_pos, y_pos),
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=6, alpha=0.75,
                               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, 
                                        edgecolor='gray', linewidth=0.5),
                               arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', 
                                             alpha=0.4, lw=0.5, color='gray'))
        except Exception as e:
            print(f"    ⚠ Impossible d'ajouter les annotations: {e}")
    
    ax2.set_xlabel('Sharpe Ratio', fontsize=11)
    ax2.set_ylabel('Performance Relative (%)', fontsize=11)
    ax2.set_title('Modèle par Secteur\nPerformance vs Risque Ajusté (avec annotations)', fontsize=12, fontweight='bold')
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax2.axvline(x=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax2.grid(alpha=0.3)
    plt.colorbar(scatter2, ax=ax2, label='Performance (%)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'graphique_4_performance_vs_risque.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Graphique 4: Performance vs Risque ajusté (Sharpe Ratio)")
    
    # 5. COMPARAISON DES MÉTRIQUES AJUSTÉES AU RISQUE
    fig, ax = plt.subplots(figsize=(14, 8))
    risk_metrics = {
        'Sharpe Ratio': [global_metrics['avg_sharpe_ratio'], sector_metrics['avg_sharpe_ratio']],
        'Sortino Ratio': [global_metrics['avg_sortino_ratio'], sector_metrics['avg_sortino_ratio']],
        'Max Drawdown (%)': [abs(global_metrics['avg_max_drawdown'])*100, abs(sector_metrics['avg_max_drawdown'])*100],
        'Volatilité (%)': [global_metrics['avg_volatility']*100, sector_metrics['avg_volatility']*100]
    }
    
    # Positionnement correct des barres groupées
    # Chaque métrique a 2 barres côte à côte (Global et Secteur)
    x = np.arange(len(risk_metrics))
    width = 0.35  # Largeur de chaque barre
    
    models_labels = ['Modèle Global', 'Modèle par Secteur']
    colors = ['#3498db', '#e74c3c']  # Bleu pour Global, Rouge pour Secteur
    
    # Position des barres pour chaque modèle
    x_global = x - width/2  # Barres Global à gauche
    x_sector = x + width/2  # Barres Secteur à droite
    
    # Extraire les valeurs pour chaque modèle
    values_global = [values[0] for values in risk_metrics.values()]
    values_sector = [values[1] for values in risk_metrics.values()]
    
    # Créer les barres groupées
    bars_global = ax.bar(x_global, values_global, width, label=models_labels[0], 
                        color=colors[0], alpha=0.8, edgecolor='black', linewidth=0.5)
    bars_sector = ax.bar(x_sector, values_sector, width, label=models_labels[1], 
                        color=colors[1], alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Ajouter les valeurs sur les barres
    for bars, values in [(bars_global, values_global), (bars_sector, values_sector)]:
        for i, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            metric_name = list(risk_metrics.keys())[i]
            format_str = f'{val:.2f}' if 'Ratio' in metric_name else f'{val:.2f}%'
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                   format_str,
                   ha='center', va='bottom', fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, 
                            edgecolor='gray', linewidth=0.5))
    
    ax.set_ylabel('Valeur', fontsize=12, fontweight='bold')
    ax.set_title('Comparaison des Métriques Ajustées au Risque\nModèle Global vs Modèle par Secteur', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(list(risk_metrics.keys()), fontsize=10)
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
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

