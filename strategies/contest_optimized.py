"""
Contest Optimized Strategy - Estrategia optimizada para concurso de 6 semanas
Combina momentum, breakout y gestión de riesgo avanzada para maximizar rendimiento en corto plazo
"""

import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from rich.console import Console
from rich.table import Table
from datetime import datetime, timedelta

from core.data_provider import download_multiple_tickers, validate_ticker_data
from core.indicators import (
    calculate_rsi, calculate_macd, calculate_bollinger_bands,
    calculate_sma, calculate_ema, calculate_atr, calculate_volatility
)
from core.backtesting import run_strategy_comparison_backtest, display_backtest_results
from config import DEFAULT_LEVERAGED_ETFS, DATA_DIR
from ui.display import save_results_to_csv

logger = logging.getLogger("hacktinver.contest_optimized")
console = Console()


def run_contest_optimized_strategy():
    """
    Ejecuta la estrategia optimizada para concurso de 6 semanas
    Combina múltiples enfoques: momentum, breakout, mean reversion y gestión de riesgo
    """
    console.print("[bold blue]🏆 Estrategia Optimizada para Concurso (6 semanas)[/bold blue]")
    console.print("[yellow]Combinando Momentum + Breakout + Gestión de Riesgo Avanzada[/yellow]")
    
    # ETFs apalancados más líquidos y volátiles para el concurso
    contest_etfs = ['SOXL', 'TQQQ', 'SPXL', 'FAS', 'TNA', 'TECL', 'FNGU', 'SOXS', 'SPXS', 'TECS']
    
    # Solicitar configuración del usuario
    try:
        input_tickers = input(f"ETFs para analizar (presiona Enter para usar los recomendados): ")
        if input_tickers.strip():
            tickers = [t.strip().upper() for t in input_tickers.split(',')]
        else:
            tickers = contest_etfs
            console.print("[yellow]📊 Usando ETFs optimizados para concurso...[/yellow]")
        
        capital_total = float(input("Capital total disponible (default: 1000000): ") or "1000000")
        risk_per_trade = float(input("% de riesgo por operación (default: 3.0): ") or "3.0") / 100
        
    except ValueError:
        tickers = contest_etfs
        capital_total = 1000000
        risk_per_trade = 0.03
        console.print("[yellow]⚠️ Usando configuración por defecto[/yellow]")
    
    console.print(f"\n[bold cyan]Configuración del Análisis:[/bold cyan]")
    console.print(f"• ETFs seleccionados: {len(tickers)}")
    console.print(f"• Capital total: ${capital_total:,.0f}")
    console.print(f"• Riesgo por operación: {risk_per_trade*100:.1f}%")
    
    # Descargar datos históricos (últimas 12 semanas para análisis)
    console.print(f"\n[yellow]📊 Descargando datos históricos...[/yellow]")
    tickers_data = download_multiple_tickers(tickers, period="3mo")
    
    if not tickers_data:
        console.print("[bold red]❌ No se pudieron descargar datos[/bold red]")
        return
    
    results = []
    
    for ticker in tickers:
        if ticker not in tickers_data:
            console.print(f"[red]⚠️ No hay datos para {ticker}[/red]")
            continue
        
        try:
            console.print(f"[dim]🔍 Analizando {ticker}...[/dim]")
            
            prices = tickers_data[ticker]
            if not validate_ticker_data(prices, 60):
                console.print(f"[red]⚠️ Datos insuficientes para {ticker}[/red]")
                continue
            
            # Simular datos OHLCV para indicadores
            high_prices = prices * 1.02
            low_prices = prices * 0.98
            volume = pd.Series(np.random.randint(1000000, 10000000, len(prices)), index=prices.index)
            
            # === ANÁLISIS MULTI-ESTRATEGIA ===
            
            # 1. MOMENTUM ANALYSIS (Últimas 4 semanas)
            momentum_4w = calculate_momentum_score(prices, window=20)
            momentum_2w = calculate_momentum_score(prices, window=10)
            momentum_1w = calculate_momentum_score(prices, window=5)
            
            # 2. TECHNICAL INDICATORS
            rsi = calculate_rsi(prices, window=14)
            macd_line, macd_signal, macd_hist = calculate_macd(prices)
            bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(prices, window=20)
            sma_20 = calculate_sma(prices, window=20)
            ema_12 = calculate_ema(prices, window=12)
            atr = calculate_atr(high_prices, low_prices, prices, window=14)
            
            # Valores actuales
            current_price = float(prices.iloc[-1])
            current_rsi = float(rsi.iloc[-1]) if not rsi.empty else 50
            current_macd_hist = float(macd_hist.iloc[-1]) if not macd_hist.empty else 0
            current_bb_upper = float(bb_upper.iloc[-1]) if not bb_upper.empty else current_price * 1.02
            current_bb_lower = float(bb_lower.iloc[-1]) if not bb_lower.empty else current_price * 0.98
            current_sma_20 = float(sma_20.iloc[-1]) if not sma_20.empty else current_price
            current_ema_12 = float(ema_12.iloc[-1]) if not ema_12.empty else current_price
            current_atr = float(atr.iloc[-1]) if not atr.empty else current_price * 0.02
            
            # 3. VOLATILITY BREAKOUT ANALYSIS
            bb_squeeze = calculate_bollinger_squeeze(bb_upper, bb_lower, window=10)
            volatility_breakout = detect_volatility_breakout(prices, bb_upper, bb_lower, volume)
            
            # 4. MEAN REVERSION SIGNALS
            price_position_in_bb = (current_price - current_bb_lower) / (current_bb_upper - current_bb_lower)
            mean_reversion_signal = get_mean_reversion_signal(current_price, current_sma_20, current_rsi)
            
            # === SCORING SYSTEM ===
            
            # Momentum Score (0-40 puntos)
            momentum_score = 0
            if momentum_4w > 5: momentum_score += 15
            elif momentum_4w > 2: momentum_score += 10
            elif momentum_4w > 0: momentum_score += 5
            
            if momentum_2w > 3: momentum_score += 15
            elif momentum_2w > 1: momentum_score += 10
            elif momentum_2w > 0: momentum_score += 5
            
            if momentum_1w > 2: momentum_score += 10
            elif momentum_1w > 0: momentum_score += 5
            
            # Technical Score (0-30 puntos)
            technical_score = 0
            if 30 < current_rsi < 70: technical_score += 10  # RSI en rango saludable
            elif current_rsi < 30: technical_score += 15     # Oversold (oportunidad)
            elif current_rsi > 80: technical_score -= 10     # Overbought (riesgo)
            
            if current_macd_hist > 0: technical_score += 10  # MACD positivo
            if current_price > current_ema_12: technical_score += 10  # Por encima de EMA
            
            # Breakout Score (0-20 puntos)
            breakout_score = 0
            if bb_squeeze: breakout_score += 10  # Bollinger squeeze (preparación)
            if volatility_breakout == 'bullish': breakout_score += 20
            elif volatility_breakout == 'bearish': breakout_score -= 15
            
            # Risk Management Score (0-10 puntos)
            risk_score = 0
            volatility_pct = (current_atr / current_price) * 100
            if volatility_pct < 5: risk_score += 10      # Baja volatilidad
            elif volatility_pct < 8: risk_score += 5     # Volatilidad media
            elif volatility_pct > 12: risk_score -= 5    # Alta volatilidad
            
            # TOTAL SCORE
            total_score = momentum_score + technical_score + breakout_score + risk_score
            
            # === POSITION SIZING ===
            
            # Calcular tamaño de posición basado en ATR y riesgo
            position_risk = capital_total * risk_per_trade
            stop_loss_distance = current_atr * 2  # 2 ATR como stop loss
            position_size = int(position_risk / stop_loss_distance) if stop_loss_distance > 0 else 0
            position_value = position_size * current_price
            portfolio_weight = (position_value / capital_total) * 100 if capital_total > 0 else 0
            
            # === SIGNAL GENERATION ===
            
            signal = "ESPERAR"
            confidence = "BAJA"
            
            if total_score >= 70:
                signal = "COMPRA FUERTE"
                confidence = "ALTA"
            elif total_score >= 50:
                signal = "COMPRA"
                confidence = "MEDIA"
            elif total_score <= -30:
                signal = "VENTA"
                confidence = "MEDIA"
            elif total_score <= -50:
                signal = "VENTA FUERTE"
                confidence = "ALTA"
            
            # Ajustar señal por condiciones especiales
            if current_rsi > 85:
                signal = "VENTA - RSI Extremo"
                confidence = "ALTA"
            elif current_rsi < 15:
                signal = "COMPRA - RSI Extremo"
                confidence = "ALTA"
            
            results.append({
                "Ticker": ticker,
                "Precio Actual": current_price,
                "Momentum 4W (%)": momentum_4w,
                "Momentum 2W (%)": momentum_2w,
                "RSI": current_rsi,
                "MACD Hist": current_macd_hist,
                "BB Position": price_position_in_bb,
                "Volatilidad (%)": volatility_pct,
                "Score Total": total_score,
                "Score Momentum": momentum_score,
                "Score Técnico": technical_score,
                "Score Breakout": breakout_score,
                "Señal": signal,
                "Confianza": confidence,
                "Posición Sugerida": position_size,
                "Valor Posición": position_value,
                "% Portafolio": portfolio_weight,
                "Stop Loss": current_price - stop_loss_distance,
                "Take Profit": current_price + (stop_loss_distance * 2)
            })
            
        except Exception as e:
            console.print(f"[red]❌ Error analizando {ticker}: {e}[/red]")
            logger.error(f"Error en contest strategy {ticker}: {e}")
            continue
    
    if not results:
        console.print("[bold red]❌ No se pudieron analizar ETFs[/bold red]")
        return
    
    # Crear DataFrame y ordenar por score total
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values("Score Total", ascending=False)
    
    # Mostrar tabla de resultados
    display_contest_results(df_results)
    
    # Mostrar recomendaciones específicas
    show_contest_recommendations(df_results, capital_total)
    
    # Ejecutar backtest comparativo
    console.print(f"\n[bold blue]🔄 Ejecutando Backtest Comparativo...[/bold blue]")
    backtest_results = run_strategy_comparison_backtest(
        tickers=tickers[:5],  # Top 5 ETFs para backtest
        period_weeks=6,
        initial_capital=capital_total
    )
    
    if backtest_results:
        display_backtest_results(backtest_results)
    
    # Guardar resultados
    filename = save_results_to_csv(df_results, "contest_optimized_strategy")
    console.print(f"\n[bold yellow]📁 Resultados guardados en: {filename}[/bold yellow]")


def calculate_momentum_score(prices: pd.Series, window: int) -> float:
    """Calcula el score de momentum para un período específico"""
    if len(prices) < window:
        return 0
    
    start_price = prices.iloc[-window]
    end_price = prices.iloc[-1]
    return ((end_price - start_price) / start_price) * 100


def calculate_bollinger_squeeze(bb_upper: pd.Series, bb_lower: pd.Series, window: int = 10) -> bool:
    """Detecta si las Bandas de Bollinger están en squeeze (estrechamiento)"""
    if len(bb_upper) < window or len(bb_lower) < window:
        return False
    
    recent_width = (bb_upper.iloc[-window:] - bb_lower.iloc[-window:]).mean()
    historical_width = (bb_upper - bb_lower).mean()
    
    return recent_width < historical_width * 0.8


def detect_volatility_breakout(prices: pd.Series, bb_upper: pd.Series, bb_lower: pd.Series, volume: pd.Series) -> str:
    """Detecta breakouts de volatilidad"""
    if len(prices) < 5:
        return 'none'
    
    current_price = prices.iloc[-1]
    current_bb_upper = bb_upper.iloc[-1]
    current_bb_lower = bb_lower.iloc[-1]
    
    # Volumen actual vs promedio
    avg_volume = volume.rolling(window=20).mean().iloc[-1]
    current_volume = volume.iloc[-1]
    volume_surge = current_volume > avg_volume * 1.5
    
    # Detectar breakout
    if current_price > current_bb_upper and volume_surge:
        return 'bullish'
    elif current_price < current_bb_lower and volume_surge:
        return 'bearish'
    
    return 'none'


def get_mean_reversion_signal(current_price: float, sma_20: float, rsi: float) -> str:
    """Genera señal de reversión a la media"""
    price_vs_sma = (current_price - sma_20) / sma_20 * 100
    
    if price_vs_sma < -5 and rsi < 30:
        return 'buy_oversold'
    elif price_vs_sma > 5 and rsi > 70:
        return 'sell_overbought'
    
    return 'neutral'


def display_contest_results(df_results: pd.DataFrame):
    """Muestra los resultados de la estrategia de concurso"""
    
    # Tabla principal
    table = Table(title="🏆 Estrategia Optimizada para Concurso - Análisis Multi-Factor")
    table.add_column("Ticker", style="cyan", no_wrap=True)
    table.add_column("Precio", style="magenta")
    table.add_column("Mom 4W", style="green")
    table.add_column("RSI", style="yellow")
    table.add_column("Score", style="bold blue")
    table.add_column("Señal", style="bold")
    table.add_column("Confianza", style="white")
    table.add_column("Posición", style="green")
    table.add_column("% Port.", style="cyan")
    
    for _, row in df_results.head(10).iterrows():  # Top 10
        # Colorear momentum
        mom_4w = row['Momentum 4W (%)']
        if mom_4w > 5:
            mom_str = f"[bold green]+{mom_4w:.1f}%[/bold green]"
        elif mom_4w > 0:
            mom_str = f"[green]+{mom_4w:.1f}%[/green]"
        else:
            mom_str = f"[red]{mom_4w:.1f}%[/red]"
        
        # Colorear RSI
        rsi = row['RSI']
        if rsi > 70:
            rsi_str = f"[red]{rsi:.1f}[/red]"
        elif rsi < 30:
            rsi_str = f"[green]{rsi:.1f}[/green]"
        else:
            rsi_str = f"{rsi:.1f}"
        
        # Colorear score
        score = row['Score Total']
        if score >= 70:
            score_str = f"[bold green]{score}[/bold green]"
        elif score >= 50:
            score_str = f"[green]{score}[/green]"
        elif score <= -30:
            score_str = f"[red]{score}[/red]"
        else:
            score_str = f"{score}"
        
        # Colorear señal
        signal = row['Señal']
        if "COMPRA" in signal:
            signal_str = f"[bold green]{signal}[/bold green]"
        elif "VENTA" in signal:
            signal_str = f"[bold red]{signal}[/bold red]"
        else:
            signal_str = signal
        
        table.add_row(
            row['Ticker'],
            f"${row['Precio Actual']:.2f}",
            mom_str,
            rsi_str,
            score_str,
            signal_str,
            row['Confianza'],
            f"{row['Posición Sugerida']:,}",
            f"{row['% Portafolio']:.1f}%"
        )
    
    console.print(table)


def show_contest_recommendations(df_results: pd.DataFrame, capital_total: float):
    """Muestra recomendaciones específicas para el concurso"""
    
    # Filtrar por señales de compra
    buy_signals = df_results[df_results['Señal'].str.contains('COMPRA', na=False)]
    high_confidence = buy_signals[buy_signals['Confianza'] == 'ALTA']
    
    console.print(f"\n[bold green]🎯 RECOMENDACIONES PARA CONCURSO (6 SEMANAS):[/bold green]")
    
    if not high_confidence.empty:
        console.print(f"\n[bold red]🔥 OPORTUNIDADES DE ALTA CONFIANZA ({len(high_confidence)}):[/bold red]")
        total_allocation = 0
        
        for _, row in high_confidence.iterrows():
            console.print(f"   • {row['Ticker']}: {row['Señal']}")
            console.print(f"     └─ Score: {row['Score Total']}, Momentum 4W: {row['Momentum 4W (%)']:.1f}%, RSI: {row['RSI']:.1f}")
            console.print(f"     └─ Posición: {row['Posición Sugerida']:,} acciones (${row['Valor Posición']:,.0f})")
            console.print(f"     └─ Stop Loss: ${row['Stop Loss']:.2f}, Take Profit: ${row['Take Profit']:.2f}")
            total_allocation += row['% Portafolio']
        
        console.print(f"\n[bold blue]📊 Asignación Total Recomendada: {total_allocation:.1f}% del capital[/bold blue]")
        console.print(f"Capital restante para oportunidades: ${capital_total * (1 - total_allocation/100):,.0f}")
    
    else:
        console.print("   [dim]No hay oportunidades de alta confianza en este momento[/dim]")
    
    # Top 3 por momentum
    top_momentum = df_results.nlargest(3, 'Momentum 4W (%)')
    console.print(f"\n[bold yellow]⚡ TOP 3 MOMENTUM (4 semanas):[/bold yellow]")
    for _, row in top_momentum.iterrows():
        console.print(f"   • {row['Ticker']}: {row['Momentum 4W (%)']:.1f}% (Score: {row['Score Total']})")
    
    # Advertencias de riesgo
    high_risk = df_results[df_results['Volatilidad (%)'] > 10]
    if not high_risk.empty:
        console.print(f"\n[bold red]⚠️ ALTA VOLATILIDAD - GESTIÓN DE RIESGO CRÍTICA:[/bold red]")
        for _, row in high_risk.head(3).iterrows():
            console.print(f"   • {row['Ticker']}: {row['Volatilidad (%)']:.1f}% volatilidad - Stop loss estricto recomendado")
    
    console.print(f"\n[bold blue]💡 CONSEJOS PARA EL CONCURSO:[/bold blue]")
    console.print("• [green]Diversificación[/green]: No más del 20% en un solo ETF")
    console.print("• [yellow]Stop Loss[/yellow]: Usar stops basados en ATR (2x ATR)")
    console.print("• [cyan]Rebalanceo[/cyan]: Revisar posiciones cada 2-3 días")
    console.print("• [red]Gestión de Riesgo[/red]: Máximo 3% de riesgo por operación")
    console.print("• [blue]Momentum[/blue]: Priorizar ETFs con momentum de 4 semanas positivo")