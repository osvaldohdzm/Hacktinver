"""
Volatility Allocation Strategy - Asignación basada en volatilidad
Implementa la estrategia de asignación de capital basada en ATR
"""

import logging
import pandas as pd
import numpy as np
from typing import List, Dict
from rich.console import Console
from rich.table import Table

from core.data_provider import download_multiple_tickers, validate_ticker_data
from core.indicators import calculate_atr, calculate_volatility, calculate_sharpe_ratio
from config import DEFAULT_LEVERAGED_ETFS, DATA_DIR
from ui.display import save_results_to_csv

logger = logging.getLogger("hacktinver.volatility_allocation")
console = Console()


def run_volatility_allocation_strategy():
    """
    Ejecuta la estrategia de asignación de capital basada en volatilidad
    """
    console.print("[bold blue]📊 Asignación de Capital Basada en Volatilidad[/bold blue]")
    console.print("[yellow]Gestión de Riesgo Avanzada con ATR (Average True Range)[/yellow]")
    
    # Solicitar tickers del usuario
    input_tickers = input(f"Ingresa los ETFs separados por comas (presiona Enter para usar: {','.join(DEFAULT_LEVERAGED_ETFS[:5])}): ")
    
    if input_tickers.strip():
        tickers = [ticker.strip().upper() for ticker in input_tickers.split(",")]
    else:
        tickers = DEFAULT_LEVERAGED_ETFS[:9]  # Usar los primeros 9 por defecto
        console.print(f"[yellow]📊 Usando ETFs por defecto...[/yellow]")
    
    # Parámetros de gestión de riesgo
    try:
        capital_total = float(input("Ingresa el capital total disponible (default: 1000000): ") or "1000000")
        riesgo_por_operacion = float(input("Ingresa el % de riesgo por operación (default: 2.0): ") or "2.0") / 100
    except ValueError:
        capital_total = 1000000
        riesgo_por_operacion = 0.02
        console.print("[yellow]⚠️ Usando valores por defecto: Capital=1,000,000, Riesgo=2%[/yellow]")
    
    console.print(f"\n[bold cyan]Analizando {len(tickers)} ETFs para asignación de capital...[/bold cyan]")
    console.print(f"Capital Total: ${capital_total:,.2f}")
    console.print(f"Riesgo por Operación: {riesgo_por_operacion*100:.1f}%")
    
    # Descargar datos para todos los tickers
    tickers_data = download_multiple_tickers(tickers, period="3mo")
    
    if not tickers_data:
        console.print("[bold red]❌ No se pudieron descargar datos para ningún ticker[/bold red]")
        return
    
    results = []
    
    for ticker in tickers:
        if ticker not in tickers_data:
            console.print(f"[red]⚠️ No hay datos para {ticker}[/red]")
            continue
        
        try:
            console.print(f"[dim]Procesando {ticker}...[/dim]")
            
            # Obtener datos OHLC (simulados desde Close)
            close_prices = tickers_data[ticker]
            
            if not validate_ticker_data(close_prices, 20):
                console.print(f"[red]⚠️ Datos insuficientes para {ticker}[/red]")
                continue
            
            # Simular datos OHLC para cálculo de ATR
            # En un caso real, necesitarías descargar datos OHLC completos
            high_prices = close_prices * 1.02  # Aproximación
            low_prices = close_prices * 0.98   # Aproximación
            
            # Calcular ATR
            atr_series = calculate_atr(high_prices, low_prices, close_prices, window=14)
            atr_actual = float(atr_series.iloc[-1]) if not atr_series.empty else 0
            
            if atr_actual == 0:
                console.print(f"[red]⚠️ No se pudo calcular ATR para {ticker}[/red]")
                continue
            
            # Precio actual
            precio_actual = float(close_prices.iloc[-1])
            
            # Calcular volatilidad porcentual
            volatilidad_pct = (atr_actual / precio_actual) * 100
            
            # Calcular tamaño de posición basado en riesgo
            cantidad_acciones = int((capital_total * riesgo_por_operacion) / atr_actual)
            
            # Calcular inversión total para esta posición
            inversion_total = cantidad_acciones * precio_actual
            
            # Calcular porcentaje del portafolio
            porcentaje_portafolio = (inversion_total / capital_total) * 100
            
            # Calcular stop loss sugerido
            stop_loss = precio_actual - atr_actual
            stop_loss_pct = (atr_actual / precio_actual) * 100
            
            # Calcular volatilidad histórica
            volatilidad_historica = calculate_volatility(close_prices, window=20)
            vol_hist_actual = float(volatilidad_historica.iloc[-1]) * 100 if not volatilidad_historica.empty else 0
            
            # Calcular Sharpe ratio
            returns = close_prices.pct_change().dropna()
            sharpe_ratio = calculate_sharpe_ratio(returns.tail(60)) if len(returns) >= 60 else 0
            
            results.append({
                "Ticker": ticker,
                "Precio Actual": precio_actual,
                "ATR (14d)": atr_actual,
                "Volatilidad ATR (%)": volatilidad_pct,
                "Volatilidad Histórica (%)": vol_hist_actual,
                "Cantidad Acciones": cantidad_acciones,
                "Inversión Total": inversion_total,
                "% Portafolio": porcentaje_portafolio,
                "Stop Loss": stop_loss,
                "Stop Loss (%)": stop_loss_pct,
                "Sharpe Ratio": sharpe_ratio
            })
            
        except Exception as e:
            console.print(f"[red]❌ Error procesando {ticker}: {e}[/red]")
            logger.error(f"Error en volatility allocation {ticker}: {e}")
            continue
    
    if not results:
        console.print("[bold red]❌ No se pudieron procesar los ETFs[/bold red]")
        return
    
    # Crear DataFrame y ordenar por volatilidad ATR
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values("Volatilidad ATR (%)", ascending=True)
    
    # Mostrar tabla con Rich
    table = Table(title="📊 Asignación de Capital Basada en Volatilidad (ATR)")
    table.add_column("Ticker", style="cyan", no_wrap=True)
    table.add_column("Precio", style="magenta")
    table.add_column("ATR", style="yellow")
    table.add_column("Vol ATR (%)", style="red")
    table.add_column("Cantidad", style="green")
    table.add_column("Inversión", style="blue")
    table.add_column("% Port.", style="white")
    table.add_column("Stop Loss", style="red")
    table.add_column("Sharpe", style="cyan")
    
    for _, row in df_results.iterrows():
        vol_atr = row['Volatilidad ATR (%)']
        
        # Colorear volatilidad según nivel
        if vol_atr > 8:
            vol_str = f"[bold red]{vol_atr:.2f}%[/bold red]"
        elif vol_atr > 5:
            vol_str = f"[yellow]{vol_atr:.2f}%[/yellow]"
        else:
            vol_str = f"[green]{vol_atr:.2f}%[/green]"
        
        table.add_row(
            row['Ticker'],
            f"${row['Precio Actual']:.2f}",
            f"{row['ATR (14d)']:.2f}",
            vol_str,
            f"{row['Cantidad Acciones']:,}",
            f"${row['Inversión Total']:,.0f}",
            f"{row['% Portafolio']:.1f}%",
            f"${row['Stop Loss']:.2f}",
            f"{row['Sharpe Ratio']:.2f}"
        )
    
    console.print(table)
    
    # Calcular estadísticas del portafolio
    total_inversion = df_results['Inversión Total'].sum()
    capital_restante = capital_total - total_inversion
    num_posiciones = len(df_results)
    vol_promedio = df_results['Volatilidad ATR (%)'].mean()
    
    console.print(f"\n[bold blue]📈 Resumen del Portafolio:[/bold blue]")
    console.print(f"• Total Invertido: ${total_inversion:,.2f} ({total_inversion/capital_total*100:.1f}% del capital)")
    console.print(f"• Capital Restante: ${capital_restante:,.2f}")
    console.print(f"• Número de Posiciones: {num_posiciones}")
    console.print(f"• Volatilidad Promedio ATR: {vol_promedio:.2f}%")
    console.print(f"• Riesgo por Operación: {riesgo_por_operacion*100:.1f}%")
    
    # Recomendaciones por volatilidad
    low_vol = df_results[df_results['Volatilidad ATR (%)'] <= 5]
    medium_vol = df_results[(df_results['Volatilidad ATR (%)'] > 5) & (df_results['Volatilidad ATR (%)'] <= 8)]
    high_vol = df_results[df_results['Volatilidad ATR (%)'] > 8]
    
    console.print(f"\n[bold green]🟢 BAJA VOLATILIDAD ({len(low_vol)} ETFs):[/bold green]")
    if not low_vol.empty:
        for ticker in low_vol['Ticker'].tolist():
            console.print(f"   • {ticker}: Posiciones más grandes, menor riesgo")
    
    console.print(f"\n[bold yellow]🟡 VOLATILIDAD MEDIA ({len(medium_vol)} ETFs):[/bold yellow]")
    if not medium_vol.empty:
        for ticker in medium_vol['Ticker'].tolist():
            console.print(f"   • {ticker}: Posiciones balanceadas")
    
    console.print(f"\n[bold red]🔴 ALTA VOLATILIDAD ({len(high_vol)} ETFs):[/bold red]")
    if not high_vol.empty:
        for ticker in high_vol['Ticker'].tolist():
            console.print(f"   • {ticker}: Posiciones más pequeñas, mayor potencial")
    
    # Guardar resultados
    filename = save_results_to_csv(df_results, "volatility_capital_allocation")
    console.print(f"\n[bold yellow]📁 Resultados guardados en: {filename}[/bold yellow]")
    
    console.print(f"\n[bold blue]💡 Conceptos Clave de Gestión de Riesgo:[/bold blue]")
    console.print("• [green]ATR (Average True Range)[/green]: Mide la volatilidad promedio de 14 días")
    console.print("• [yellow]Riesgo Uniforme[/yellow]: Todas las posiciones tienen el mismo riesgo en pesos")
    console.print("• [cyan]Stop Loss Sugerido[/cyan]: 1 ATR por debajo del precio de entrada")
    console.print("• [red]Posición Inversa[/red]: Mayor volatilidad = menor tamaño de posición")