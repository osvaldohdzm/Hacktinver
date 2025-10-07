"""
Data Provider - Proveedor de datos de mercado
Responsable de obtener datos de precios de diferentes fuentes (yfinance, APIs, etc.)
"""

import logging
import pandas as pd
import yfinance as yf
import requests
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta

from config import ACTINVER_DATA_URL, app_config

logger = logging.getLogger("hacktinver.data_provider")


def _pick_price_series(df: pd.DataFrame) -> pd.Series:
    """
    Extrae una serie de precios 1D prefiriendo 'Adj Close' sobre 'Close'
    
    Args:
        df: DataFrame con datos de precios
    
    Returns:
        Serie de precios 1D
    """
    if df is None or df.empty:
        return pd.Series(dtype=float)
    
    if "Adj Close" in df.columns:
        s = df["Adj Close"]
    else:
        s = df.get("Close", pd.Series(dtype=float))
    
    # Asegurar Serie 1D
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    
    return s


def download_prices_any_listing(
    ticker: str, 
    start: str = None, 
    end: str = None, 
    period: str = None
) -> pd.Series:
    """
    Descarga precios para un símbolo probando con y sin sufijo .MX
    
    Args:
        ticker: Símbolo del activo
        start: Fecha de inicio (YYYY-MM-DD)
        end: Fecha de fin (YYYY-MM-DD)
        period: Período (ej: '6mo', '1y')
    
    Returns:
        Serie de precios o serie vacía si no se encuentra
    """
    t = (ticker or "").strip()
    
    # Preferir US (sin sufijo) primero; solo probar .MX si el usuario lo proporcionó o falló
    if t.endswith(".MX"):
        symbols_to_try = [t, t.replace(".MX", "")]
    else:
        symbols_to_try = [t, f"{t}.MX"]
    
    for sym in symbols_to_try:
        try:
            logger.debug(f"Intentando descargar datos para {sym}")
            
            if start or end:
                df = yf.download(sym, start=start, end=end, interval="1d")
            else:
                df = yf.download(sym, period=period or "6mo")
            
            s = _pick_price_series(df)
            if not s.empty:
                logger.info(f"Datos descargados exitosamente para {sym}")
                return s.rename(ticker)
                
        except Exception as e:
            logger.debug(f"Error descargando {sym}: {e}")
            continue
    
    logger.warning(f"No se pudieron obtener datos para {ticker}")
    return pd.Series(dtype=float, name=ticker)


def download_multiple_tickers(
    tickers: List[str], 
    period: str = "6mo",
    start: str = None,
    end: str = None
) -> Dict[str, pd.Series]:
    """
    Descarga datos para múltiples tickers de forma eficiente
    
    Args:
        tickers: Lista de símbolos
        period: Período de descarga
        start: Fecha de inicio
        end: Fecha de fin
    
    Returns:
        Diccionario con ticker como clave y serie de precios como valor
    """
    results = {}
    
    logger.info(f"Descargando datos para {len(tickers)} tickers")
    
    for ticker in tickers:
        try:
            series = download_prices_any_listing(ticker, start, end, period)
            if not series.empty:
                results[ticker] = series
        except Exception as e:
            logger.error(f"Error descargando {ticker}: {e}")
            continue
    
    logger.info(f"Descarga completada: {len(results)}/{len(tickers)} exitosos")
    return results


def get_actinver_contest_data() -> Optional[pd.DataFrame]:
    """
    Obtiene datos del concurso Actinver desde la URL oficial
    
    Returns:
        DataFrame con datos del concurso o None si hay error
    """
    try:
        logger.info("Descargando datos del concurso Actinver")
        
        response = requests.get(ACTINVER_DATA_URL, timeout=app_config.get_timeout())
        response.raise_for_status()
        
        # Procesar los datos
        lines = response.text.strip().split('\n')
        data = []
        
        for line in lines:
            if not line.strip():
                continue
            
            parts = line.split('|')
            if len(parts) < 5:
                continue
            
            try:
                data.append({
                    'symbol': parts[1],
                    'current_price': float(parts[3]),
                    'previous_price': float(parts[4]),
                    'raw_data': line
                })
            except (ValueError, IndexError):
                continue
        
        df = pd.DataFrame(data)
        logger.info(f"Datos del concurso Actinver obtenidos: {len(df)} símbolos")
        return df
        
    except requests.RequestException as e:
        logger.error(f"Error descargando datos de Actinver: {e}")
        return None
    except Exception as e:
        logger.error(f"Error procesando datos de Actinver: {e}")
        return None


def get_market_data_batch(
    tickers: List[str],
    period: str = "1y",
    include_volume: bool = True
) -> pd.DataFrame:
    """
    Obtiene datos de mercado para múltiples tickers en un solo DataFrame
    
    Args:
        tickers: Lista de símbolos
        period: Período de datos
        include_volume: Si incluir datos de volumen
    
    Returns:
        DataFrame con datos OHLCV para todos los tickers
    """
    try:
        logger.info(f"Descargando datos batch para {len(tickers)} tickers")
        
        # Usar yfinance para descarga batch
        data = yf.download(tickers, period=period, group_by='ticker')
        
        if data.empty:
            logger.warning("No se obtuvieron datos en descarga batch")
            return pd.DataFrame()
        
        # Si es un solo ticker, yfinance no agrupa por ticker
        if len(tickers) == 1:
            data.columns = pd.MultiIndex.from_product([tickers, data.columns])
        
        logger.info("Descarga batch completada exitosamente")
        return data
        
    except Exception as e:
        logger.error(f"Error en descarga batch: {e}")
        return pd.DataFrame()


def validate_ticker_data(data: pd.Series, min_periods: int = 30) -> bool:
    """
    Valida que los datos de un ticker sean suficientes para análisis
    
    Args:
        data: Serie de precios
        min_periods: Número mínimo de períodos requeridos
    
    Returns:
        True si los datos son válidos
    """
    if data is None or data.empty:
        return False
    
    if len(data) < min_periods:
        return False
    
    # Verificar que no todos los valores sean NaN
    if data.isna().all():
        return False
    
    # Verificar que haya variación en los precios
    if data.nunique() <= 1:
        return False
    
    return True


def get_ticker_info(ticker: str) -> Dict[str, Any]:
    """
    Obtiene información básica de un ticker
    
    Args:
        ticker: Símbolo del activo
    
    Returns:
        Diccionario con información del ticker
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        return {
            'symbol': ticker,
            'name': info.get('longName', 'N/A'),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'market_cap': info.get('marketCap', 0),
            'currency': info.get('currency', 'USD'),
            'exchange': info.get('exchange', 'N/A')
        }
        
    except Exception as e:
        logger.error(f"Error obteniendo info de {ticker}: {e}")
        return {'symbol': ticker, 'error': str(e)}


class DataCache:
    """
    Cache simple para datos de mercado
    """
    
    def __init__(self, ttl_minutes: int = 15):
        self.cache = {}
        self.ttl = timedelta(minutes=ttl_minutes)
    
    def get(self, key: str) -> Optional[Any]:
        """Obtiene datos del cache si no han expirado"""
        if not app_config.cache_enabled:
            return None
        
        if key in self.cache:
            data, timestamp = self.cache[key]
            if datetime.now() - timestamp < self.ttl:
                return data
            else:
                del self.cache[key]
        
        return None
    
    def set(self, key: str, data: Any) -> None:
        """Guarda datos en el cache"""
        if app_config.cache_enabled:
            self.cache[key] = (data, datetime.now())
    
    def clear(self) -> None:
        """Limpia el cache"""
        self.cache.clear()


# Instancia global del cache
data_cache = DataCache()