#!/usr/bin/env python3
"""
Point d'entrée principal pour l'analyse de stocks avec modèles LSTM
"""
import os
import pandas as pd
import numpy as np
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.fetch_data import STOCKS, download_all_stocks
from src.data_processing import group_stocks_by_sector
from src.strategy import (
    train_global_model, train_sector_model, 
    process_stock_prediction, save_to_database
)
from src.database import create_database
from src.visualization import generate_visualizations

import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")

# Fixer les seeds pour la reproductibilité
np.random.seed(42)
tf.random.set_seed(42)


def calculate_aggregate_metrics(df, model_name):
    """Calculer les métriques moyennes agrégées"""
    sharpe = df['sharpe_ratio'].mean() if 'sharpe_ratio' in df.columns else 0.0
    sortino = df['sortino_ratio'].mean() if 'sortino_ratio' in df.columns else 0.0
    max_dd = df['max_drawdown'].mean() if 'max_drawdown' in df.columns else 0.0
    vol = df['volatility'].mean() if 'volatility' in df.columns else 0.0
    
    return {
        'model_type': model_name,
        'total_stocks': len(df),
        'avg_accuracy': df['accuracy'].mean(),
        'avg_strategy_return': df['strategy_return'].mean(),
        'avg_buy_hold_return': df['buy_hold_return'].mean(),
        'avg_performance': df['performance'].mean(),
        'avg_sharpe_ratio': sharpe,
        'avg_sortino_ratio': sortino,
        'avg_max_drawdown': max_dd,
        'avg_volatility': vol
    }


def main():
    """Fonction principale"""
    output_dir = 'outs'
    os.makedirs(output_dir, exist_ok=True)
    
    print("=== ANALYSE DE 100 ACTIONS AMÉRICAINES (S&P 500) ===")
    print("Modèle : LSTM par secteur (Long Short-Term Memory)")
    print(f"Univers : {len(STOCKS)} actions réparties sur 11 secteurs")
    print("Architecture : 1 modèle LSTM par secteur + 1 modèle global")
    print("Méthode : ROLLING WINDOW (fenêtre glissante) - 252 jours train, 63 jours test")
    print()
    
    # Créer la base de données
    conn = create_database(output_dir=output_dir)
    
    # Grouper les actions par secteur
    stocks_by_sector = group_stocks_by_sector(STOCKS)
    
    print(f"📊 Répartition par secteur:")
    for sector, stocks in stocks_by_sector.items():
        print(f"   {sector}: {len(stocks)} actions")
    print()
    
    # Télécharger toutes les données en parallèle
    data_cache = download_all_stocks(STOCKS)
    
    all_results = []
    global_results = []
    
    # ============================================
    # ÉTAPE 1: ENTRAÎNER LE MODÈLE GLOBAL
    # ============================================
    print("\n" + "="*70)
    print("MODÈLE GLOBAL - ENTRAÎNEMENT SUR TOUS LES STOCKS")
    print("="*70)
    
    global_model, global_scaler = train_global_model(STOCKS, data_cache=data_cache)
    
    if global_model is not None:
        print(f"\n  🔄 Prédictions parallèles avec le modèle global pour {len(STOCKS)} actions...")
        
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {
                executor.submit(process_stock_prediction, stock_info, global_model, global_scaler, data_cache, 'global'): stock_info 
                for stock_info in STOCKS
            }
            
            for future in as_completed(futures):
                stock_info = futures[future]
                try:
                    results, message = future.result()
                    print(f"  {message}")
                    
                    if results:
                        save_to_database(conn, stock_info, results, model_type='global')
                        global_results.append(results)
                except Exception as e:
                    print(f"  ✗ Erreur pour {stock_info['symbol']}: {e}")
        
        print(f"  ✓ {len(global_results)}/{len(STOCKS)} actions analysées avec le modèle global")
    
    # ============================================
    # ÉTAPE 2: MODÈLES PAR SECTEUR
    # ============================================
    for sector_name, stocks_in_sector in stocks_by_sector.items():
        print(f"\n{'='*70}")
        print(f"SECTEUR: {sector_name}")
        print(f"{'='*70}")
        
        # Entraîner le modèle du secteur
        model, scaler = train_sector_model(sector_name, stocks_in_sector, data_cache=data_cache)
        
        if model is None:
            print(f"  ⚠ Impossible d'entraîner le modèle pour {sector_name}")
            continue
        
        # Prédire toutes les actions du secteur en parallèle
        print(f"\n  🔄 Prédictions parallèles pour {len(stocks_in_sector)} actions...")
        
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {
                executor.submit(process_stock_prediction, stock_info, model, scaler, data_cache, 'sector'): stock_info 
                for stock_info in stocks_in_sector
            }
            
            sector_results = []
            for future in as_completed(futures):
                stock_info = futures[future]
                try:
                    results, message = future.result()
                    print(f"  {message}")
                    
                    if results:
                        save_to_database(conn, stock_info, results, model_type='sector')
                        sector_results.append(results)
                        all_results.append(results)
                except Exception as e:
                    print(f"  ✗ Erreur pour {stock_info['symbol']}: {e}")
            
            print(f"  ✓ {len(sector_results)}/{len(stocks_in_sector)} actions analysées pour {sector_name}")
    
    # ============================================
    # ÉTAPE 3: COMPARAISON DES MODÈLES
    # ============================================
    print("\n" + "="*70)
    print("COMPARAISON DES MODÈLES - PERFORMANCE AJUSTÉE AU RISQUE")
    print("="*70)
    
    df_sector = None
    df_global = None
    sector_metrics = None
    global_metrics = None
    
    if all_results and global_results:
        df_sector = pd.DataFrame(all_results)
        df_global = pd.DataFrame(global_results)
        
        sector_metrics = calculate_aggregate_metrics(df_sector, 'sector')
        global_metrics = calculate_aggregate_metrics(df_global, 'global')
        
        # Sauvegarder les métriques de comparaison
        from src.strategy import db_lock
        with db_lock:
            cursor = conn.cursor()
            for metrics in [sector_metrics, global_metrics]:
                cursor.execute('''
                    INSERT INTO model_comparison (
                        model_type, total_stocks, avg_accuracy, avg_strategy_return,
                        avg_buy_hold_return, avg_performance, avg_sharpe_ratio,
                        avg_sortino_ratio, avg_max_drawdown, avg_volatility
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    metrics['model_type'],
                    metrics['total_stocks'],
                    metrics['avg_accuracy'],
                    metrics['avg_strategy_return'],
                    metrics['avg_buy_hold_return'],
                    metrics['avg_performance'],
                    metrics['avg_sharpe_ratio'],
                    metrics['avg_sortino_ratio'],
                    metrics['avg_max_drawdown'],
                    metrics['avg_volatility']
                ))
            conn.commit()
        
        # Afficher la comparaison
        if sector_metrics and global_metrics:
            print("\n📊 MÉTRIQUES MOYENNES:")
            print("-" * 70)
            print(f"{'Métrique':<25} | {'Modèle Global':<20} | {'Modèle par Secteur':<20}")
            print("-" * 70)
            print(f"{'Nombre d' + 'actions':<25} | {global_metrics['total_stocks']:<20} | {sector_metrics['total_stocks']:<20}")
            print(f"{'Précision moyenne':<25} | {global_metrics['avg_accuracy']:<20.2%} | {sector_metrics['avg_accuracy']:<20.2%}")
            print(f"{'Rendement stratégie':<25} | {global_metrics['avg_strategy_return']:<20.2%} | {sector_metrics['avg_strategy_return']:<20.2%}")
            print(f"{'Rendement Buy & Hold':<25} | {global_metrics['avg_buy_hold_return']:<20.2%} | {sector_metrics['avg_buy_hold_return']:<20.2%}")
            print(f"{'Performance relative':<25} | {global_metrics['avg_performance']:<20.2%} | {sector_metrics['avg_performance']:<20.2%}")
            print()
            print("📈 MÉTRIQUES AJUSTÉES AU RISQUE:")
            print("-" * 70)
            print(f"{'Sharpe Ratio (moyen)':<25} | {global_metrics['avg_sharpe_ratio']:<20.2f} | {sector_metrics['avg_sharpe_ratio']:<20.2f}")
            print(f"{'Sortino Ratio (moyen)':<25} | {global_metrics['avg_sortino_ratio']:<20.2f} | {sector_metrics['avg_sortino_ratio']:<20.2f}")
            print(f"{'Max Drawdown (moyen)':<25} | {global_metrics['avg_max_drawdown']:<20.2%} | {sector_metrics['avg_max_drawdown']:<20.2%}")
            print(f"{'Volatilité (moyenne)':<25} | {global_metrics['avg_volatility']:<20.2%} | {sector_metrics['avg_volatility']:<20.2%}")
            print()
            
            # Déterminer le meilleur modèle
            print("🏆 CONCLUSION:")
            print("-" * 70)
            if global_metrics['avg_sharpe_ratio'] > sector_metrics['avg_sharpe_ratio']:
                print(f"  ✓ Modèle GLOBAL meilleur selon le Sharpe Ratio ({global_metrics['avg_sharpe_ratio']:.2f} vs {sector_metrics['avg_sharpe_ratio']:.2f})")
            else:
                print(f"  ✓ Modèle PAR SECTEUR meilleur selon le Sharpe Ratio ({sector_metrics['avg_sharpe_ratio']:.2f} vs {global_metrics['avg_sharpe_ratio']:.2f})")
            
            if global_metrics['avg_performance'] > sector_metrics['avg_performance']:
                print(f"  ✓ Modèle GLOBAL meilleur selon la Performance ({global_metrics['avg_performance']:.2%} vs {sector_metrics['avg_performance']:.2%})")
            else:
                print(f"  ✓ Modèle PAR SECTEUR meilleur selon la Performance ({sector_metrics['avg_performance']:.2%} vs {global_metrics['avg_performance']:.2%})")
    
    # Résumé final
    print("\n" + "="*70)
    print("RÉSULTATS FINAUX - MODÈLES PAR SECTEUR")
    print("="*70)
    
    if all_results:
        df = pd.DataFrame(all_results)
        
        print(f"\nActions analysées avec succès: {len(all_results)}/{len(STOCKS)}")
        print(f"Précision moyenne: {df['accuracy'].mean():.2%}")
        print(f"Performance moyenne: {df['performance'].mean():.2%}")
        print(f"Rendement stratégie moyen: {df['strategy_return'].mean():.2%}")
        print(f"Rendement Buy & Hold moyen: {df['buy_hold_return'].mean():.2%}")
        print()
        
        print("🏆 TOP 15 PAR PERFORMANCE (Modèles par secteur):")
        print("-" * 70)
        df_sorted = df.sort_values('performance', ascending=False)
        for i, (_, row) in enumerate(df_sorted.head(15).iterrows(), 1):
            sharpe = row.get('sharpe_ratio', 0)
            print(f"{i:2d}. {row['symbol']:6s} | Performance: {row['performance']:7.2%} | "
                  f"Stratégie: {row['strategy_return']:7.2%} | B&H: {row['buy_hold_return']:7.2%} | "
                  f"Précision: {row['accuracy']:5.2%} | Sharpe: {sharpe:5.2f}")
        
        if len(df_sorted) > 15:
            print(f"\n... et {len(df_sorted) - 15} autres actions")
        
        # Sauvegarder en CSV
        df.to_csv(os.path.join(output_dir, 'results_sector.csv'), index=False)
        print(f"\nRésultats modèles par secteur sauvegardés dans '{output_dir}/results_sector.csv'")
    
    if global_results:
        df_global = pd.DataFrame(global_results)
        df_global.to_csv(os.path.join(output_dir, 'results_global.csv'), index=False)
        print(f"Résultats modèle global sauvegardés dans '{output_dir}/results_global.csv'")
    
    # ============================================
    # ÉTAPE 4: GÉNÉRATION DES GRAPHIQUES
    # ============================================
    if all_results and global_results and df_sector is not None and df_global is not None:
        print("\n" + "="*70)
        print("GÉNÉRATION DES GRAPHIQUES")
        print("="*70)
        generate_visualizations(df_sector, df_global, sector_metrics, global_metrics, output_dir=output_dir)
    
    conn.close()
    print(f"\n✅ Analyse terminée ! Fichiers générés dans '{output_dir}/'")
    print(f"   - stock_analysis.db")
    print(f"   - results_sector.csv")
    print(f"   - results_global.csv")
    print(f"   - graphique_*.png (5 graphiques)")


if __name__ == "__main__":
    main()

