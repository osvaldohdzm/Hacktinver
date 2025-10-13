#!/usr/bin/env python3
"""
Script de prueba para verificar el fallback de USA a MX
"""

import yfinance as yf
import pandas as pd
import requests
from rich.console import Console

console = Console()

def get_usd_to_mxn_rate():
    """Obtiene el tipo de cambio USD a MXN"""
    try:
        response = requests.get("https://api.exchangerate-api.com/v4/latest/USD", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data['rates'].get('MXN', 20.0)
    except:
        pass
    
    try:
        usd_mxn = yf.download("USDMXN=X", period="1d", progress=False)
        if not usd_mxn.empty:
            return float(usd_mxn['Close'].iloc[-1])
    except:
        pass
    
    console.print("[yellow]⚠️ No se pudo obtener tipo de cambio, usando 20.0 MXN/USD[/yellow]")
    return 20.0

def test_ticker_fallback(ticker):
    """Prueba el fallback para un ticker específico"""
    console.print(f"\n[bold cyan]🧪 Probando ticker: {ticker}[/bold cyan]")
    
    # 1. Intentar versión .MX
    mx_ticker = f"{ticker}.MX"
    console.print(f"[yellow]1️⃣ Intentando {mx_ticker}...[/yellow]")
    
    try:
        mx_data = yf.download(mx_ticker, period="1mo", progress=False)
        
        # Verificar que tenemos datos válidos de forma más simple
        if (not mx_data.empty and 
            len(mx_data) > 5 and 
            'Close' in mx_data.columns):
            
            # Verificar que hay al menos algunos valores no-NaN
            close_series = mx_data['Close']
            valid_prices = close_series.dropna()
            
            if len(valid_prices) > 0:
                precio_actual = float(valid_prices.iloc[-1])
                console.print(f"[green]✅ {mx_ticker} - ÉXITO (datos mexicanos)[/green]")
                console.print(f"   📊 Precio actual: ${precio_actual:.2f} MXN")
                return mx_data, 'MX', None
        
        console.print(f"[yellow]⚠️ {mx_ticker} - Sin datos válidos[/yellow]")
    except Exception as e:
        console.print(f"[red]❌ {mx_ticker} - Error: {str(e)[:50]}[/red]")
    
    # 2. Intentar versión USA
    console.print(f"[yellow]2️⃣ Intentando fallback a {ticker} (USA)...[/yellow]")
    
    try:
        usa_data = yf.download(ticker, period="1mo", progress=False)
        
        # Verificar que tenemos datos válidos USA de forma más simple
        if (not usa_data.empty and 
            len(usa_data) > 5 and 
            'Close' in usa_data.columns):
            
            # Verificar que hay al menos algunos valores no-NaN
            close_series = usa_data['Close']
            valid_prices = close_series.dropna()
            
            if len(valid_prices) > 0:
                # Obtener tipo de cambio
                exchange_rate = get_usd_to_mxn_rate()
                console.print(f"[blue]💱 Tipo de cambio: {exchange_rate:.4f} MXN/USD[/blue]")
                
                # Convertir precios
                price_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close']
                for col in price_columns:
                    if col in usa_data.columns:
                        usa_data[col] = usa_data[col] * exchange_rate
                
                precio_convertido = float(usa_data['Close'].iloc[-1])
                console.print(f"[blue]🔄 {ticker} - ÉXITO (datos USA convertidos a MXN)[/blue]")
                console.print(f"   📊 Precio actual: ${precio_convertido:.2f} MXN (convertido)")
                return usa_data, 'USA', exchange_rate
        
        console.print(f"[red]❌ {ticker} - Sin datos válidos en USA[/red]")
    except Exception as e:
        console.print(f"[red]❌ {ticker} - Error USA: {str(e)[:50]}[/red]")
    
    console.print(f"[red]💀 {ticker} - NO ENCONTRADO en ningún mercado[/red]")
    return None, None, None

def main():
    """Función principal de prueba"""
    console.print("[bold blue]🧪 PRUEBA DE FALLBACK USA → MX[/bold blue]")
    console.print("Probando algunos tickers que sabemos que fallan en .MX pero existen en USA\n")
    
    # Tickers de prueba - algunos que sabemos que fallan en .MX pero existen en USA
    test_tickers = [
        "MSFT",     # Microsoft - debería existir en ambos
        "NVDA",     # NVIDIA - debería existir en ambos
        "GOOGL",    # Google - debería existir en ambos
        "AMZN",     # Amazon - debería existir en ambos
        "FAKE123",  # Ticker falso - no debería existir en ninguno
    ]
    
    results = {}
    
    for ticker in test_tickers:
        data, source, rate = test_ticker_fallback(ticker)
        results[ticker] = {
            'data': data,
            'source': source,
            'exchange_rate': rate,
            'success': data is not None
        }
    
    # Resumen final
    console.print(f"\n[bold cyan]📊 RESUMEN DE PRUEBAS:[/bold cyan]")
    
    mx_count = sum(1 for r in results.values() if r['source'] == 'MX')
    usa_count = sum(1 for r in results.values() if r['source'] == 'USA')
    failed_count = sum(1 for r in results.values() if not r['success'])
    
    console.print(f"[green]🇲🇽 Datos mexicanos: {mx_count}[/green]")
    console.print(f"[blue]🇺🇸 Datos USA convertidos: {usa_count}[/blue]")
    console.print(f"[red]❌ Fallos: {failed_count}[/red]")
    
    if usa_count > 0:
        console.print(f"\n[bold blue]🎯 ¡FALLBACK FUNCIONANDO! {usa_count} símbolos obtenidos desde USA[/bold blue]")
    else:
        console.print(f"\n[bold red]⚠️ Fallback no funcionó - revisar implementación[/bold red]")

if __name__ == "__main__":
    main()