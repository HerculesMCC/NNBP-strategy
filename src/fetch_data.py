"""
Module pour le téléchargement des données boursières
"""
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

warnings.filterwarnings("ignore")


# Liste de 100 actions avec leur secteur
STOCKS = [
    # TECHNOLOGIE (15 actions)
    {"symbol": "AAPL", "name": "Apple Inc.", "sector": "Technologie"},
    {"symbol": "MSFT", "name": "Microsoft Corporation", "sector": "Technologie"},
    {"symbol": "GOOGL", "name": "Alphabet Inc.", "sector": "Technologie"},
    {"symbol": "AMZN", "name": "Amazon.com Inc.", "sector": "Technologie"},
    {"symbol": "META", "name": "Meta Platforms Inc.", "sector": "Technologie"},
    {"symbol": "NVDA", "name": "NVIDIA Corporation", "sector": "Technologie"},
    {"symbol": "ORCL", "name": "Oracle Corporation", "sector": "Technologie"},
    {"symbol": "CRM", "name": "Salesforce Inc.", "sector": "Technologie"},
    {"symbol": "INTC", "name": "Intel Corporation", "sector": "Technologie"},
    {"symbol": "AMD", "name": "Advanced Micro Devices Inc.", "sector": "Technologie"},
    {"symbol": "CSCO", "name": "Cisco Systems Inc.", "sector": "Technologie"},
    {"symbol": "ADBE", "name": "Adobe Inc.", "sector": "Technologie"},
    {"symbol": "AVGO", "name": "Broadcom Inc.", "sector": "Technologie"},
    {"symbol": "QCOM", "name": "Qualcomm Inc.", "sector": "Technologie"},
    {"symbol": "TXN", "name": "Texas Instruments Inc.", "sector": "Technologie"},
    
    # FINANCE (12 actions)
    {"symbol": "JPM", "name": "JPMorgan Chase & Co.", "sector": "Finance"},
    {"symbol": "BAC", "name": "Bank of America Corp.", "sector": "Finance"},
    {"symbol": "GS", "name": "Goldman Sachs Group Inc.", "sector": "Finance"},
    {"symbol": "V", "name": "Visa Inc.", "sector": "Finance"},
    {"symbol": "MA", "name": "Mastercard Inc.", "sector": "Finance"},
    {"symbol": "WFC", "name": "Wells Fargo & Company", "sector": "Finance"},
    {"symbol": "C", "name": "Citigroup Inc.", "sector": "Finance"},
    {"symbol": "AXP", "name": "American Express Company", "sector": "Finance"},
    {"symbol": "BLK", "name": "BlackRock Inc.", "sector": "Finance"},
    {"symbol": "SCHW", "name": "Charles Schwab Corporation", "sector": "Finance"},
    {"symbol": "USB", "name": "U.S. Bancorp", "sector": "Finance"},
    {"symbol": "PNC", "name": "PNC Financial Services Group", "sector": "Finance"},
    
    # SANTÉ (12 actions)
    {"symbol": "JNJ", "name": "Johnson & Johnson", "sector": "Santé"},
    {"symbol": "UNH", "name": "UnitedHealth Group Inc.", "sector": "Santé"},
    {"symbol": "PFE", "name": "Pfizer Inc.", "sector": "Santé"},
    {"symbol": "ABBV", "name": "AbbVie Inc.", "sector": "Santé"},
    {"symbol": "TMO", "name": "Thermo Fisher Scientific Inc.", "sector": "Santé"},
    {"symbol": "ABT", "name": "Abbott Laboratories", "sector": "Santé"},
    {"symbol": "LLY", "name": "Eli Lilly and Company", "sector": "Santé"},
    {"symbol": "DHR", "name": "Danaher Corporation", "sector": "Santé"},
    {"symbol": "BMY", "name": "Bristol-Myers Squibb Company", "sector": "Santé"},
    {"symbol": "AMGN", "name": "Amgen Inc.", "sector": "Santé"},
    {"symbol": "GILD", "name": "Gilead Sciences Inc.", "sector": "Santé"},
    {"symbol": "CVS", "name": "CVS Health Corporation", "sector": "Santé"},
    
    # CONSOMMATION DISCRÉTIONNAIRE (10 actions)
    {"symbol": "TSLA", "name": "Tesla Inc.", "sector": "Consommation Discrétionnaire"},
    {"symbol": "NKE", "name": "Nike Inc.", "sector": "Consommation Discrétionnaire"},
    {"symbol": "HD", "name": "Home Depot Inc.", "sector": "Consommation Discrétionnaire"},
    {"symbol": "MCD", "name": "McDonald's Corporation", "sector": "Consommation Discrétionnaire"},
    {"symbol": "SBUX", "name": "Starbucks Corporation", "sector": "Consommation Discrétionnaire"},
    {"symbol": "NFLX", "name": "Netflix Inc.", "sector": "Consommation Discrétionnaire"},
    {"symbol": "LOW", "name": "Lowe's Companies Inc.", "sector": "Consommation Discrétionnaire"},
    {"symbol": "TJX", "name": "TJX Companies Inc.", "sector": "Consommation Discrétionnaire"},
    {"symbol": "BKNG", "name": "Booking Holdings Inc.", "sector": "Consommation Discrétionnaire"},
    {"symbol": "GM", "name": "General Motors Company", "sector": "Consommation Discrétionnaire"},
    
    # CONSOMMATION STAPLES (8 actions)
    {"symbol": "WMT", "name": "Walmart Inc.", "sector": "Consommation Staples"},
    {"symbol": "PG", "name": "Procter & Gamble Co.", "sector": "Consommation Staples"},
    {"symbol": "KO", "name": "The Coca-Cola Company", "sector": "Consommation Staples"},
    {"symbol": "PEP", "name": "PepsiCo Inc.", "sector": "Consommation Staples"},
    {"symbol": "COST", "name": "Costco Wholesale Corporation", "sector": "Consommation Staples"},
    {"symbol": "CL", "name": "Colgate-Palmolive Company", "sector": "Consommation Staples"},
    {"symbol": "MDLZ", "name": "Mondelez International Inc.", "sector": "Consommation Staples"},
    {"symbol": "STZ", "name": "Constellation Brands Inc.", "sector": "Consommation Staples"},
    
    # ÉNERGIE (8 actions)
    {"symbol": "XOM", "name": "Exxon Mobil Corporation", "sector": "Énergie"},
    {"symbol": "CVX", "name": "Chevron Corporation", "sector": "Énergie"},
    {"symbol": "SLB", "name": "Schlumberger Limited", "sector": "Énergie"},
    {"symbol": "COP", "name": "ConocoPhillips", "sector": "Énergie"},
    {"symbol": "EOG", "name": "EOG Resources Inc.", "sector": "Énergie"},
    {"symbol": "MPC", "name": "Marathon Petroleum Corporation", "sector": "Énergie"},
    {"symbol": "PSX", "name": "Phillips 66", "sector": "Énergie"},
    {"symbol": "VLO", "name": "Valero Energy Corporation", "sector": "Énergie"},
    
    # INDUSTRIEL (10 actions)
    {"symbol": "BA", "name": "Boeing Company", "sector": "Industriel"},
    {"symbol": "CAT", "name": "Caterpillar Inc.", "sector": "Industriel"},
    {"symbol": "GE", "name": "General Electric Company", "sector": "Industriel"},
    {"symbol": "HON", "name": "Honeywell International Inc.", "sector": "Industriel"},
    {"symbol": "UPS", "name": "United Parcel Service Inc.", "sector": "Industriel"},
    {"symbol": "RTX", "name": "Raytheon Technologies Corporation", "sector": "Industriel"},
    {"symbol": "LMT", "name": "Lockheed Martin Corporation", "sector": "Industriel"},
    {"symbol": "DE", "name": "Deere & Company", "sector": "Industriel"},
    {"symbol": "EMR", "name": "Emerson Electric Co.", "sector": "Industriel"},
    {"symbol": "ETN", "name": "Eaton Corporation plc", "sector": "Industriel"},
    
    # TÉLÉCOMMUNICATIONS (5 actions)
    {"symbol": "T", "name": "AT&T Inc.", "sector": "Télécommunications"},
    {"symbol": "VZ", "name": "Verizon Communications Inc.", "sector": "Télécommunications"},
    {"symbol": "CMCSA", "name": "Comcast Corporation", "sector": "Télécommunications"},
    {"symbol": "DIS", "name": "Walt Disney Company", "sector": "Télécommunications"},
    {"symbol": "CHTR", "name": "Charter Communications Inc.", "sector": "Télécommunications"},
    
    # MATÉRIAUX (8 actions)
    {"symbol": "LIN", "name": "Linde plc", "sector": "Matériaux"},
    {"symbol": "APD", "name": "Air Products and Chemicals Inc.", "sector": "Matériaux"},
    {"symbol": "ECL", "name": "Ecolab Inc.", "sector": "Matériaux"},
    {"symbol": "SHW", "name": "Sherwin-Williams Company", "sector": "Matériaux"},
    {"symbol": "PPG", "name": "PPG Industries Inc.", "sector": "Matériaux"},
    {"symbol": "FCX", "name": "Freeport-McMoRan Inc.", "sector": "Matériaux"},
    {"symbol": "NEM", "name": "Newmont Corporation", "sector": "Matériaux"},
    {"symbol": "DD", "name": "DuPont de Nemours Inc.", "sector": "Matériaux"},
    
    # UTILITAIRES (6 actions)
    {"symbol": "NEE", "name": "NextEra Energy Inc.", "sector": "Utilitaires"},
    {"symbol": "DUK", "name": "Duke Energy Corporation", "sector": "Utilitaires"},
    {"symbol": "SO", "name": "Southern Company", "sector": "Utilitaires"},
    {"symbol": "AEP", "name": "American Electric Power Company", "sector": "Utilitaires"},
    {"symbol": "SRE", "name": "Sempra Energy", "sector": "Utilitaires"},
    {"symbol": "EXC", "name": "Exelon Corporation", "sector": "Utilitaires"},
    
    # IMMOBILIER (6 actions)
    {"symbol": "AMT", "name": "American Tower Corporation", "sector": "Immobilier"},
    {"symbol": "PLD", "name": "Prologis Inc.", "sector": "Immobilier"},
    {"symbol": "EQIX", "name": "Equinix Inc.", "sector": "Immobilier"},
    {"symbol": "PSA", "name": "Public Storage", "sector": "Immobilier"},
    {"symbol": "WELL", "name": "Welltower Inc.", "sector": "Immobilier"},
    {"symbol": "SPG", "name": "Simon Property Group Inc.", "sector": "Immobilier"}
]


def download_stock_data(symbol, start_date="2020-01-01", end_date="2023-01-01", verbose=True):
    """
    Télécharger les données d'une action
    
    Args:
        symbol: Symbole de l'action
        start_date: Date de début (format YYYY-MM-DD)
        end_date: Date de fin (format YYYY-MM-DD)
        verbose: Afficher les messages
    
    Returns:
        DataFrame avec les données OHLCV ou None en cas d'erreur
    """
    try:
        if verbose:
            print(f"  Téléchargement de {symbol}...")
        data = yf.download(symbol, start=start_date, end=end_date, progress=False)
        
        if data.empty:
            raise ValueError("Données vides")
            
        if verbose:
            print(f"  ✓ {symbol}: {len(data)} jours de données")
        return data
        
    except Exception as e:
        if verbose:
            print(f"  ✗ Erreur pour {symbol}: {e}")
        return None


def download_all_stocks(stocks, max_workers=10, start_date="2020-01-01", end_date="2023-01-01"):
    """
    Télécharger toutes les données en parallèle
    
    Args:
        stocks: Liste des dictionnaires avec les informations des actions
        max_workers: Nombre de threads parallèles
        start_date: Date de début
        end_date: Date de fin
    
    Returns:
        Dictionnaire {symbol: DataFrame} avec toutes les données téléchargées
    """
    data_cache = {}
    
    def download_and_cache(stock_info):
        """Télécharger et mettre en cache les données"""
        symbol = stock_info['symbol']
        try:
            data = download_stock_data(symbol, start_date, end_date, verbose=False)
            if data is not None and not data.empty:
                return symbol, data
            return symbol, None
        except Exception as e:
            return symbol, None
    
    print("\n📥 TÉLÉCHARGEMENT PARALLÈLE DES DONNÉES...")
    print("-" * 70)
    
    # Télécharger toutes les données en parallèle
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_and_cache, stock): stock for stock in stocks}
        completed = 0
        for future in as_completed(futures):
            completed += 1
            symbol, data = future.result()
            if data is not None:
                data_cache[symbol] = data
            if completed % 10 == 0:
                print(f"  Progression: {completed}/{len(stocks)} téléchargements...")
    
    print(f"  ✓ {len(data_cache)}/{len(stocks)} actions téléchargées avec succès\n")
    return data_cache

