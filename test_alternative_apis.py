#!/usr/bin/env python3
"""
Script de prueba para verificar las fuentes alternativas de datos
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hacktinver import get_stock_data_alternative_apis, get_usd_to_mxn_rate
from rich.console import Console

console = Console()

def test_alternative_apis():
    """Prueba las APIs alternativas de datos"""
    console.print("[bold blue]🧪 PRUEBA DE FUENTES ALTERNATIVAS DE DATOS[/bold blue]")
    
    # Tickers de prueba que probablemente existen en mercados USA
    test_tickers = [
        "AAPL",     # Apple - muy común
        "MSFT",     # Microsoft - muy común
        "GOOGL",    # Google - muy común
        "TSLA",     # Tesla - muy común
        "NVDA",     # NVIDIA - muy común
    ]
    
    console.print(f"[yellow]Probando APIs alternativas con {len(test_tickers)} tickers conocidos...[/yellow]")
    
    # Probar tipo de cambio primero
    console.print(f"\n[cyan]1️⃣ Probando obtención de tipo de cambio...[/cyan]")
    exchange_rate = get_usd_to_mxn_rate()
    console.print(f"[green]✅ Tipo de cambio obtenido: {exchange_rate:.4f} MXN/USD[/green]")
    
    # Probar cada ticker con APIs alternativas
    console.print(f"\n[cyan]2️⃣ Probando APIs alternativas para datos de stocks...[/cyan]")
    
    results = {}
    
    for ticker in test_tickers:
        console.print(f"\n[yellow]🔍 Probando {ticker}...[/yellow]")
        
        try:
            df = get_stock_data_alternative_apis(ticker, period="1mo")
            
            if not df.empty and len(df) > 5:
                latest_price = df['Close'].iloc[-1]
                data_points = len(df)
                date_range = f"{df.index[0].strftime('%Y-%m-%d')} a {df.index[-1].strftime('%Y-%m-%d')}"
                
                results[ticker] = {
                    'success': True,
                    'price': latest_price,
                    'data_points': data_points,
                    'date_range': date_range
                }
                
                console.print(f"[green]✅ {ticker} - Éxito![/green]")
                console.print(f"   📊 Precio más reciente: ${latest_price:.2f} USD")
                console.print(f"   📈 Puntos de datos: {data_points}")
                console.print(f"   📅 Rango de fechas: {date_range}")
                
            else:
                results[ticker] = {'success': False, 'reason': 'Datos insuficientes'}
                console.print(f"[yellow]⚠️ {ticker} - Datos insuficientes[/yellow]")
                
        except Exception as e:
            results[ticker] = {'success': False, 'reason': str(e)[:50]}
            console.print(f"[red]❌ {ticker} - Error: {str(e)[:50]}[/red]")
    
    # Resumen final
    console.print(f"\n[bold cyan]📊 RESUMEN DE PRUEBAS DE APIs ALTERNATIVAS:[/bold cyan]")
    
    successful = sum(1 for r in results.values() if r['success'])
    failed = len(results) - successful
    
    console.print(f"[green]✅ Exitosos: {successful}/{len(test_tickers)}[/green]")
    console.print(f"[red]❌ Fallidos: {failed}/{len(test_tickers)}[/red]")
    
    if successful > 0:
        console.print(f"\n[bold green]🎯 ¡APIs ALTERNATIVAS FUNCIONANDO![/bold green]")
        console.print(f"[green]Las fuentes alternativas pueden proporcionar datos cuando Yahoo Finance falle[/green]")
        
        # Mostrar configuración recomendada
        console.print(f"\n[bold yellow]⚙️ CONFIGURACIÓN RECOMENDADA:[/bold yellow]")
        console.print(f"[yellow]1. Copia .env.example a .env[/yellow]")
        console.print(f"[yellow]2. Obtén claves API gratuitas de:[/yellow]")
        console.print(f"   • Alpha Vantage: https://www.alphavantage.co/support/#api-key")
        console.print(f"   • Polygon.io: https://polygon.io/")
        console.print(f"   • IEX Cloud: https://iexcloud.io/")
        console.print(f"   • Financial Modeling Prep: https://financialmodelingprep.com/")
        console.print(f"[yellow]3. Agrega las claves a tu archivo .env[/yellow]")
        
    else:
        console.print(f"\n[bold red]⚠️ APIs alternativas no funcionaron[/bold red]")
        console.print(f"[red]Esto puede deberse a:[/red]")
        console.print(f"[red]• Falta de claves API en el archivo .env[/red]")
        console.print(f"[red]• Límites de rate en las APIs gratuitas[/red]")
        console.print(f"[red]• Problemas de conectividad[/red]")
    
    return results

if __name__ == "__main__":
    test_alternative_apis()