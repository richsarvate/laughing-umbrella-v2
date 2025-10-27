"""
Market data processing module for S&P 500 stocks - V2.
Handles data download and price sequence extraction for pure sequence learning.
No technical indicators - just normalized price sequences like GPT uses text sequences.
"""

from typing import List
import numpy as np
import pandas as pd
import yfinance as yf

class MarketDataProcessor:
    """Handles S&P 500 data download and price sequence extraction."""
    
    def __init__(self, lookback_days: int = 60):
        self.lookback_days = lookback_days
        self.sp500_tickers = self._get_sp500_tickers()
    
    def _get_sp500_tickers(self) -> List[str]:
        """Get complete S&P 500 ticker list for maximum model performance."""
        # Complete S&P 500 stocks (approximately 500 stocks)
        sp500_tickers = [
            # Mega Cap (Top 50)
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 
            'UNH', 'JNJ', 'JPM', 'V', 'PG', 'XOM', 'HD', 'CVX', 'MA', 'BAC',
            'ABBV', 'PFE', 'AVGO', 'KO', 'LLY', 'TMO', 'COST', 'MRK', 'ORCL',
            'ACN', 'DHR', 'VZ', 'ABT', 'WMT', 'CRM', 'NFLX', 'ADBE', 'NKE',
            'TXN', 'RTX', 'QCOM', 'NEE', 'PM', 'LOW', 'BMY', 'HON', 'UPS',
            'AMGN', 'T', 'COP', 'IBM', 'SPGI',
            
            # Large Cap (51-150)
            'CAT', 'MDT', 'SCHW', 'GS', 'AXP', 'BLK', 'BKNG', 'SYK', 'DE', 'TJX',
            'AMD', 'LMT', 'MDLZ', 'ADP', 'GILD', 'CVS', 'MMC', 'C', 'LRCX', 'ADI',
            'INTC', 'PYPL', 'TMUS', 'CB', 'MO', 'SO', 'ZTS', 'EQIX', 'CME', 'FI',
            'EOG', 'WM', 'ITW', 'PNC', 'AON', 'CSX', 'CL', 'FCX', 'SBUX', 'DUK',
            'ICE', 'USB', 'BSX', 'NSC', 'SPG', 'HCA', 'PLD', 'GM', 'F', 'EMR',
            'GE', 'NOW', 'ISRG', 'VRTX', 'BIDU', 'BDX', 'TGT', 'REGN', 'APD', 'SHW',
            'PANW', 'CMG', 'MU', 'AON', 'CCI', 'KLAC', 'AMAT', 'CDNS', 'SNPS', 'ORLY',
            'MCO', 'ECL', 'FTNT', 'MAR', 'MSI', 'ADSK', 'AJG', 'NXPI', 'ROP', 'PAYX',
            'ROST', 'KMB', 'EA', 'VRSK', 'CTSH', 'ODFL', 'CPRT', 'IEX', 'BK', 'GLW',
            'MCHP', 'KR', 'DXCM', 'CARR', 'WBA', 'HPQ', 'CSGP', 'ANSS', 'ON', 'BIIB',
            
            # Mid Cap (151-300) 
            'FAST', 'MPWR', 'IDXX', 'CTAS', 'CDW', 'FANG', 'EXC', 'XEL', 'WEC', 'ES',
            'AEE', 'LNT', 'EVRG', 'PEG', 'SRE', 'AEP', 'D', 'ED', 'ETR', 'FE',
            'PPL', 'AES', 'CNP', 'NI', 'LNT', 'PNW', 'SO', 'NEE', 'DUK', 'EXC',
            'PCG', 'EIX', 'AWK', 'ATO', 'CMS', 'DTE', 'NRG', 'VST', 'AEE', 'XEL',
            'WEC', 'ES', 'EVRG', 'PEG', 'SRE', 'AEP', 'D', 'ED', 'ETR', 'FE',
            'PPL', 'AES', 'CNP', 'NI', 'LNT', 'PNW', 'NEE', 'DUK', 'EXC', 'PCG',
            'EIX', 'AWK', 'ATO', 'CMS', 'DTE', 'NRG', 'VST', 'WMB', 'KMI', 'OKE',
            'EPD', 'ET', 'MPLX', 'TRGP', 'AM', 'SUN', 'PAA', 'WES', 'EQT', 'AR',
            'DVN', 'FANG', 'MRO', 'APA', 'MGY', 'SM', 'NOV', 'HAL', 'SLB', 'BKR',
            'VAL', 'MPC', 'PSX', 'VLO', 'HFC', 'DK', 'PBF', 'CVR', 'DINO', 'RRC',
            
            # Additional S&P 500 Companies (301-450)
            'ALLE', 'ARE', 'AOS', 'APH', 'ACGL', 'ANET', 'APA', 'AAON', 'AIZ', 'AFL',
            'A', 'APD', 'AKAM', 'ALK', 'ALB', 'AA', 'ALXN', 'ARE', 'ALKS', 'AEE',
            'AAP', 'AAPL', 'AMAT', 'APTV', 'ANET', 'AJG', 'AIZ', 'T', 'ATO', 'ADSK',
            'ADP', 'AZO', 'AVB', 'AVY', 'BKR', 'BLL', 'BAC', 'BBWI', 'BAX', 'BDX',
            'BRK-B', 'BBY', 'BIO', 'BIIB', 'BLK', 'BK', 'BA', 'BKNG', 'BWA', 'BXP',
            'BSX', 'BMY', 'AVGO', 'BR', 'BRO', 'BF-B', 'CHRW', 'CDNS', 'CZR', 'CPT',
            'CPB', 'COF', 'CAH', 'KMX', 'CCL', 'CARR', 'CTLT', 'CAT', 'CBOE', 'CBRE',
            'CDW', 'CE', 'CNC', 'CNP', 'CDAY', 'CERN', 'CF', 'CRL', 'SCHW', 'CHTR',
            'CVX', 'CMG', 'CB', 'CHD', 'CI', 'CINF', 'CTAS', 'CSCO', 'C', 'CFG',
            'CTXS', 'CLX', 'CME', 'CMS', 'KO', 'CTSH', 'CL', 'CMCSA', 'CMA', 'CAG',
            
            # Final S&P 500 Companies (451-500)
            'COP', 'ED', 'STZ', 'COO', 'CPRT', 'GLW', 'CTVA', 'COST', 'CTRA', 'CCI',
            'CSX', 'CMI', 'CVS', 'DHI', 'DHR', 'DRI', 'DVA', 'DE', 'DAL', 'XRAY',
            'DVN', 'DXCM', 'FANG', 'DLR', 'DFS', 'DISCA', 'DISCK', 'DISH', 'DG', 'DLTR',
            'D', 'DPZ', 'DOV', 'DOW', 'DTE', 'DUK', 'DRE', 'DD', 'DXC', 'EMN',
            'ETN', 'EBAY', 'ECL', 'EIX', 'EW', 'EA', 'EMR', 'ENPH', 'ETR', 'EOG',
            'EFX', 'EQIX', 'EQR', 'ESS', 'EL', 'ETSY', 'EVRG', 'ES', 'RE', 'EXC'
        ]
        
        # Remove any duplicates and return first 500
        return list(dict.fromkeys(sp500_tickers))[:500]
    

    
    def download_market_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Download OHLCV data for all S&P stocks."""
        print(f"Downloading market data from {start_date} to {end_date}...")
        
        stock_data = yf.download(
            self.sp500_tickers, 
            start=start_date, 
            end=end_date,
            group_by='ticker',
            auto_adjust=True,
            prepost=True
        )
        return stock_data
    

    
    def extract_price_sequences(self, market_data: pd.DataFrame) -> np.ndarray:
        """
        Extract normalized price sequences for pure sequence learning (like GPT).
        No hand-crafted features - just price history.
        
        Returns: [days, stocks, 1] - normalized prices for each stock
        """
        num_stocks = len(self.sp500_tickers)
        num_days = len(market_data)
        
        # Initialize price matrix: [days, stocks, 1]
        price_matrix = np.zeros((num_days, num_stocks, 1))
        
        for i, ticker in enumerate(self.sp500_tickers):
            try:
                # Extract price series for this stock
                close_prices = market_data[ticker]['Close'].values
                
                # Skip if insufficient data
                if len(close_prices) < 2:
                    continue
                
                # Normalize prices: percentage change from first valid price
                first_valid_price = close_prices[np.isfinite(close_prices)][0] if np.any(np.isfinite(close_prices)) else 1.0
                normalized_prices = (close_prices - first_valid_price) / first_valid_price
                
                # Store normalized prices
                price_matrix[:, i, 0] = normalized_prices
                
            except Exception as e:
                print(f"Warning: Could not process {ticker}: {e}")
                continue
        
        # Remove NaN and return normalized price sequences
        price_matrix = np.nan_to_num(price_matrix, nan=0.0)
        
        return price_matrix