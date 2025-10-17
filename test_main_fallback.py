#!/usr/bin/env python3
"""
Script de prueba para verificar que el fallback funciona en el programa principal
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hacktinver import download_multiple_mx_tickers
from rich.console import Console

console = Console()

def test_main_fallback():
    """Prueba el fallback en el programa principal"""
    console.print("[bold blue]🧪 PRUEBA DE FALLBACK EN PROGRAMA PRINCIPAL[/bold blue]")
    
    # Tickers de prueba - mezcla de existentes y no existentes en .MX
    test_tickers = [
        "AAPL",     # Debería existir en .MX
        "TSLA",     # Debería existir en .MX
        "ACTI500",  # Probablemente no existe en .MX ni USA
        "MSFT",     # Debería existir en .MX
        "FAKE999",  # No existe en ningún lado
    ]
    
    console.print(f"[yellow]Probando con {len(test_tickers)} tickers: {', '.join(test_tickers)}[/yellow]")
    
    # Usar la función del programa principal
    data = download_multiple_mx_tickers(test_tickers, period="1mo", progress=False)
    
    console.print(f"\n[bold cyan]📊 RESULTADOS:[/bold cyan]")
    
    for ticker in test_tickers:
        if ticker in data:
            df = data[ticker]
            source = getattr(df, 'attrs', {}).get('source', 'Unknown')
            converted = getattr(df, 'attrs', {}).get('converted_from_usd', False)
            exchange_rate = getattr(df, 'attrs', {}).get('exchange_rate', None)
            
            if source == 'MX':
                console.print(f"[green]✅ {ticker} - Datos mexicanos (.MX)[/green]")
            elif source == 'USA' and converted:
                console.print(f"[blue]🔄 {ticker} - Datos USA convertidos a MXN @ {exchange_rate:.4f}[/blue]")
            else:
                console.print(f"[yellow]⚠️ {ticker} - Fuente desconocida[/yellow]")
        else:
            console.print(f"[red]❌ {ticker} - No encontrado[/red]")
    
    # Resumen
    mx_count = sum(1 for df in data.values() if df.attrs.get('source') == 'MX')
    usa_count = sum(1 for df in data.values() if df.attrs.get('converted_from_usd', False))
    
    console.print(f"\n[bold cyan]📈 RESUMEN FINAL:[/bold cyan]")
    console.print(f"[green]🇲🇽 Datos mexicanos: {mx_count}[/green]")
    console.print(f"[blue]🇺🇸 Datos USA convertidos: {usa_count}[/blue]")
    console.print(f"[red]❌ No encontrados: {len(test_tickers) - len(data)}[/red]")
    
    if usa_count > 0:
        console.print(f"\n[bold green]🎯 ¡FALLBACK FUNCIONANDO! {usa_count} símbolos convertidos desde USA[/bold green]")
    else:
        console.print(f"\n[yellow]ℹ️ No se necesitó fallback - todos los símbolos encontrados en .MX o no existen[/yellow]")

if __name__ == "__main__":
    test_main_fallback()