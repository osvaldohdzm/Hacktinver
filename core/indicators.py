"""
Indicators - Calculadora de indicadores técnicos
Responsable de calcular todos los indicadores técnicos utilizados en las estrategias
"""

import logging
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import SMAIndicator, MACD, EMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import VolumeSMAIndicator

logger = logging.getLogger("hacktinver.indicators")


def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """
    Calcula el Índice de Fuerza Relativa (RSI)
    
    Args:
        prices: Serie de precios de cierre
        window: Período de cálculo
    
    Returns:
        Serie con valores RSI
    """
    try:
        rsi_indicator = RSIIndicator(close=prices, window=window)
        return rsi_indicator.rsi()
    except Exception as e:
        logger.error(f"Error calculando RSI: {e}")
        return pd.Series(dtype=float, index=prices.index)


def calculate_macd(
    prices: pd.Series, 
    window_fast: int = 12, 
    window_slow: int = 26, 
    window_sign: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calcula MACD (Moving Average Convergence Divergence)
    
    Args:
        prices: Serie de precios de cierre
        window_fast: Período de EMA rápida
        window_slow: Período de EMA lenta
        window_sign: Período de línea de señal
    
    Returns:
        Tupla con (MACD, Signal, Histogram)
    """
    try:
        macd_indicator = MACD(
            close=prices,
            window_fast=window_fast,
            window_slow=window_slow,
            window_sign=window_sign
        )
        
        macd_line = macd_indicator.macd()
        signal_line = macd_indicator.macd_signal()
        histogram = macd_indicator.macd_diff()
        
        return macd_line, signal_line, histogram
        
    except Exception as e:
        logger.error(f"Error calculando MACD: {e}")
        empty_series = pd.Series(dtype=float, index=prices.index)
        return empty_series, empty_series, empty_series


def calculate_bollinger_bands(
    prices: pd.Series, 
    window: int = 20, 
    window_dev: int = 2
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calcula las Bandas de Bollinger
    
    Args:
        prices: Serie de precios de cierre
        window: Período de la media móvil
        window_dev: Número de desviaciones estándar
    
    Returns:
        Tupla con (Upper Band, Middle Band, Lower Band)
    """
    try:
        bb_indicator = BollingerBands(
            close=prices,
            window=window,
            window_dev=window_dev
        )
        
        upper_band = bb_indicator.bollinger_hband()
        middle_band = bb_indicator.bollinger_mavg()
        lower_band = bb_indicator.bollinger_lband()
        
        return upper_band, middle_band, lower_band
        
    except Exception as e:
        logger.error(f"Error calculando Bandas de Bollinger: {e}")
        empty_series = pd.Series(dtype=float, index=prices.index)
        return empty_series, empty_series, empty_series


def calculate_stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
    smooth_window: int = 3
) -> Tuple[pd.Series, pd.Series]:
    """
    Calcula el Oscilador Estocástico
    
    Args:
        high: Serie de precios máximos
        low: Serie de precios mínimos
        close: Serie de precios de cierre
        window: Período de cálculo
        smooth_window: Período de suavizado
    
    Returns:
        Tupla con (%K, %D)
    """
    try:
        stoch_indicator = StochasticOscillator(
            high=high,
            low=low,
            close=close,
            window=window,
            smooth_window=smooth_window
        )
        
        stoch_k = stoch_indicator.stoch()
        stoch_d = stoch_indicator.stoch_signal()
        
        return stoch_k, stoch_d
        
    except Exception as e:
        logger.error(f"Error calculando Estocástico: {e}")
        empty_series = pd.Series(dtype=float, index=close.index)
        return empty_series, empty_series


def calculate_sma(prices: pd.Series, window: int) -> pd.Series:
    """
    Calcula la Media Móvil Simple (SMA)
    
    Args:
        prices: Serie de precios
        window: Período de la media
    
    Returns:
        Serie con valores SMA
    """
    try:
        sma_indicator = SMAIndicator(close=prices, window=window)
        return sma_indicator.sma_indicator()
    except Exception as e:
        logger.error(f"Error calculando SMA: {e}")
        return pd.Series(dtype=float, index=prices.index)


def calculate_ema(prices: pd.Series, window: int) -> pd.Series:
    """
    Calcula la Media Móvil Exponencial (EMA)
    
    Args:
        prices: Serie de precios
        window: Período de la media
    
    Returns:
        Serie con valores EMA
    """
    try:
        ema_indicator = EMAIndicator(close=prices, window=window)
        return ema_indicator.ema_indicator()
    except Exception as e:
        logger.error(f"Error calculando EMA: {e}")
        return pd.Series(dtype=float, index=prices.index)


def calculate_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14
) -> pd.Series:
    """
    Calcula el Average True Range (ATR)
    
    Args:
        high: Serie de precios máximos
        low: Serie de precios mínimos
        close: Serie de precios de cierre
        window: Período de cálculo
    
    Returns:
        Serie con valores ATR
    """
    try:
        atr_indicator = AverageTrueRange(
            high=high,
            low=low,
            close=close,
            window=window
        )
        return atr_indicator.average_true_range()
    except Exception as e:
        logger.error(f"Error calculando ATR: {e}")
        return pd.Series(dtype=float, index=close.index)


def calculate_true_range(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series
) -> pd.Series:
    """
    Calcula el True Range
    
    Args:
        high: Serie de precios máximos
        low: Serie de precios mínimos
        close: Serie de precios de cierre
    
    Returns:
        Serie con valores True Range
    """
    try:
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return true_range
        
    except Exception as e:
        logger.error(f"Error calculando True Range: {e}")
        return pd.Series(dtype=float, index=close.index)


def calculate_volatility(prices: pd.Series, window: int = 20) -> pd.Series:
    """
    Calcula la volatilidad histórica
    
    Args:
        prices: Serie de precios
        window: Período de cálculo
    
    Returns:
        Serie con volatilidad
    """
    try:
        returns = prices.pct_change()
        volatility = returns.rolling(window=window).std() * np.sqrt(252)  # Anualizada
        return volatility
    except Exception as e:
        logger.error(f"Error calculando volatilidad: {e}")
        return pd.Series(dtype=float, index=prices.index)


def calculate_returns(prices: pd.Series, periods: int = 1) -> pd.Series:
    """
    Calcula los retornos de precios
    
    Args:
        prices: Serie de precios
        periods: Número de períodos para el cálculo
    
    Returns:
        Serie con retornos
    """
    try:
        return prices.pct_change(periods=periods)
    except Exception as e:
        logger.error(f"Error calculando retornos: {e}")
        return pd.Series(dtype=float, index=prices.index)


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252
) -> float:
    """
    Calcula el ratio de Sharpe
    
    Args:
        returns: Serie de retornos
        risk_free_rate: Tasa libre de riesgo anual
        periods_per_year: Períodos por año (252 para días de trading)
    
    Returns:
        Ratio de Sharpe
    """
    try:
        if returns.empty or returns.std() == 0:
            return 0.0
        
        excess_returns = returns.mean() * periods_per_year - risk_free_rate
        volatility = returns.std() * np.sqrt(periods_per_year)
        
        return excess_returns / volatility
        
    except Exception as e:
        logger.error(f"Error calculando Sharpe ratio: {e}")
        return 0.0


def calculate_correlation_matrix(prices_dict: dict) -> pd.DataFrame:
    """
    Calcula la matriz de correlación entre múltiples activos
    
    Args:
        prices_dict: Diccionario con ticker como clave y serie de precios como valor
    
    Returns:
        DataFrame con matriz de correlación
    """
    try:
        # Crear DataFrame con todos los precios
        df = pd.DataFrame(prices_dict)
        
        # Calcular retornos
        returns_df = df.pct_change().dropna()
        
        # Calcular correlación
        correlation_matrix = returns_df.corr()
        
        return correlation_matrix
        
    except Exception as e:
        logger.error(f"Error calculando matriz de correlación: {e}")
        return pd.DataFrame()


def calculate_beta(asset_returns: pd.Series, market_returns: pd.Series) -> float:
    """
    Calcula el beta de un activo respecto al mercado
    
    Args:
        asset_returns: Retornos del activo
        market_returns: Retornos del mercado
    
    Returns:
        Beta del activo
    """
    try:
        # Alinear las series
        aligned_data = pd.DataFrame({
            'asset': asset_returns,
            'market': market_returns
        }).dropna()
        
        if len(aligned_data) < 30:  # Mínimo de observaciones
            return 1.0
        
        covariance = aligned_data['asset'].cov(aligned_data['market'])
        market_variance = aligned_data['market'].var()
        
        if market_variance == 0:
            return 1.0
        
        beta = covariance / market_variance
        return beta
        
    except Exception as e:
        logger.error(f"Error calculando beta: {e}")
        return 1.0


def calculate_drawdown(prices: pd.Series) -> Tuple[pd.Series, float, float]:
    """
    Calcula el drawdown de una serie de precios
    
    Args:
        prices: Serie de precios
    
    Returns:
        Tupla con (serie de drawdown, max drawdown, duración max drawdown)
    """
    try:
        # Calcular el máximo acumulado
        peak = prices.expanding().max()
        
        # Calcular drawdown
        drawdown = (prices - peak) / peak
        
        # Máximo drawdown
        max_drawdown = drawdown.min()
        
        # Duración del máximo drawdown
        max_dd_duration = 0
        current_duration = 0
        
        for dd in drawdown:
            if dd < 0:
                current_duration += 1
                max_dd_duration = max(max_dd_duration, current_duration)
            else:
                current_duration = 0
        
        return drawdown, max_drawdown, max_dd_duration
        
    except Exception as e:
        logger.error(f"Error calculando drawdown: {e}")
        empty_series = pd.Series(dtype=float, index=prices.index)
        return empty_series, 0.0, 0.0


class TechnicalAnalysis:
    """
    Clase para análisis técnico completo de un activo
    """
    
    def __init__(self, ohlcv_data: pd.DataFrame):
        """
        Inicializa el análisis técnico
        
        Args:
            ohlcv_data: DataFrame con columnas Open, High, Low, Close, Volume
        """
        self.data = ohlcv_data.copy()
        self.close = self.data['Close']
        self.high = self.data['High']
        self.low = self.data['Low']
        self.volume = self.data.get('Volume', pd.Series())
        
    def add_all_indicators(self) -> pd.DataFrame:
        """
        Agrega todos los indicadores técnicos al DataFrame
        
        Returns:
            DataFrame con todos los indicadores
        """
        result = self.data.copy()
        
        try:
            # Indicadores de momentum
            result['RSI'] = calculate_rsi(self.close)
            result['Stoch_K'], result['Stoch_D'] = calculate_stochastic(
                self.high, self.low, self.close
            )
            
            # MACD
            result['MACD'], result['MACD_Signal'], result['MACD_Hist'] = calculate_macd(self.close)
            
            # Bandas de Bollinger
            result['BB_Upper'], result['BB_Middle'], result['BB_Lower'] = calculate_bollinger_bands(self.close)
            
            # Medias móviles
            result['SMA_20'] = calculate_sma(self.close, 20)
            result['SMA_50'] = calculate_sma(self.close, 50)
            result['EMA_12'] = calculate_ema(self.close, 12)
            result['EMA_26'] = calculate_ema(self.close, 26)
            
            # Volatilidad
            result['ATR'] = calculate_atr(self.high, self.low, self.close)
            result['Volatility'] = calculate_volatility(self.close)
            
            # Retornos
            result['Returns'] = calculate_returns(self.close)
            
            logger.info("Todos los indicadores técnicos calculados exitosamente")
            
        except Exception as e:
            logger.error(f"Error agregando indicadores: {e}")
        
        return result
    
    def get_current_signals(self) -> dict:
        """
        Obtiene las señales actuales basadas en los indicadores
        
        Returns:
            Diccionario con señales de trading
        """
        signals = {}
        
        try:
            # Calcular indicadores actuales
            rsi_current = calculate_rsi(self.close).iloc[-1]
            macd_line, macd_signal, macd_hist = calculate_macd(self.close)
            bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(self.close)
            
            current_price = self.close.iloc[-1]
            
            # Señales RSI
            if rsi_current > 70:
                signals['RSI'] = 'SOBRECOMPRADO'
            elif rsi_current < 30:
                signals['RSI'] = 'SOBREVENDIDO'
            else:
                signals['RSI'] = 'NEUTRAL'
            
            # Señales MACD
            if macd_hist.iloc[-1] > 0:
                signals['MACD'] = 'ALCISTA'
            else:
                signals['MACD'] = 'BAJISTA'
            
            # Señales Bollinger
            if current_price > bb_upper.iloc[-1]:
                signals['BOLLINGER'] = 'SOBRECOMPRADO'
            elif current_price < bb_lower.iloc[-1]:
                signals['BOLLINGER'] = 'SOBREVENDIDO'
            else:
                signals['BOLLINGER'] = 'NEUTRAL'
            
        except Exception as e:
            logger.error(f"Error calculando señales: {e}")
        
        return signals