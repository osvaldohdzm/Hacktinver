#!/usr/bin/env python3
"""
Test script para probar la nueva estrategia de acumulación robusta
con múltiples fuentes de datos y manejo de errores mejorado.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hacktinver import (
    estrategia_acumulacion_diaria_estimada,
    _get_daily_data_robust,
    console
)

def test_robust_strategy():
    """Prueba la estrategia robusta con diferentes tickers"""
    
    console.print("[bold blue]🧪 Probando Estrategia de Acumulación Robusta[/bold blue]")
    console.print("[yellow]Esta prueba verificará el funcionamiento con múltiples fuentes de datos[/yellow]\n")
    
    # Lista de tickers para probar (algunos pueden fallar intencionalmente)
    test_tickers = [
        "TECL",     # ETF tecnológico popular
        "SOXL",     # ETF semiconductores
        "QQQ",      # ETF NASDAQ
        "SPY",      # ETF S&P 500
        "TSLA",     # Tesla
        "AAPL",     # Apple
        "NVDA",     # NVIDIA
        "INVALID_TICKER_123"  # Ticker inválido para probar manejo de errores
    ]
    
    successful_tests = 0
    failed_tests = 0
    
    for ticker in test_tickers:
        console.print(f"\n[bold cyan]📊 Probando {ticker}...[/bold cyan]")
        console.print("=" * 50)
        
        try:
            # Probar descarga de datos robusta
            df = _get_daily_data_robust(ticker)
            
            if df is not None and len(df) >= 2:
                console.print(f"[green]✅ Datos obtenidos para {ticker}: {len(df)} días[/green]")
                
                # Probar estrategia de acumulación
                estrategia = estrategia_acumulacion_diaria_estimada(
                    ticker=ticker,
                    capital_total=100000,  # $100,000 MXN
                    num_escalones=3
                )
                
                if estrategia and 'escalones' in estrategia:
                    console.print(f"[green]✅ Estrategia generada exitosamente para {ticker}[/green]")
                    successful_tests += 1
                else:
                    console.print(f"[red]❌ Error generando estrategia para {ticker}[/red]")
                    failed_tests += 1
            else:
                console.print(f"[yellow]⚠️ No se pudieron obtener datos para {ticker}[/yellow]")
                
                # Aún así probar la estrategia (debería usar fallback)
                estrategia = estrategia_acumulacion_diaria_estimada(
                    ticker=ticker,
                    capital_total=100000,
                    num_escalones=3
                )
                
                if estrategia and 'escalones' in estrategia:
                    console.print(f"[blue]🔄 Estrategia con fallback generada para {ticker}[/blue]")
                    successful_tests += 1
                else:
                    console.print(f"[red]❌ Error total para {ticker}[/red]")
                    failed_tests += 1
                    
        except Exception as e:
            console.print(f"[red]❌ Error inesperado con {ticker}: {str(e)[:50]}[/red]")
            failed_tests += 1
    
    # Resumen de resultados
    console.print("\n" + "=" * 60)
    console.print("[bold blue]📋 RESUMEN DE PRUEBAS[/bold blue]")
    console.print(f"[green]✅ Exitosas: {successful_tests}[/green]")
    console.print(f"[red]❌ Fallidas: {failed_tests}[/red]")
    console.print(f"[cyan]📊 Total: {successful_tests + failed_tests}[/cyan]")
    
    if successful_tests > failed_tests:
        console.print("\n[bold green]🎉 ¡Pruebas mayormente exitosas! El sistema robusto funciona correctamente.[/bold green]")
    else:
        console.print("\n[bold yellow]⚠️ Algunas pruebas fallaron. Revisar configuración de APIs alternativas.[/bold yellow]")
    
    return successful_tests, failed_tests


def test_specific_ticker():
    """Permite probar un ticker específico interactivamente"""
    
    console.print("\n[bold blue]🎯 Prueba de Ticker Específico[/bold blue]")
    
    ticker = input("Ingresa el ticker a probar (default: TECL): ").strip() or "TECL"
    capital = input("Capital total en MXN (default: 500000): ").strip() or "500000"
    escalones = input("Número de escalones (default: 3): ").strip() or "3"
    
    try:
        capital_float = float(capital)
        escalones_int = int(escalones)
        
        console.print(f"\n[cyan]🔍 Probando {ticker} con ${capital_float:,.0f} MXN en {escalones_int} escalones...[/cyan]")
        
        estrategia = estrategia_acumulacion_diaria_estimada(
            ticker=ticker,
            capital_total=capital_float,
            num_escalones=escalones_int
        )
        
        if estrategia:
            console.print(f"\n[green]✅ ¡Estrategia generada exitosamente para {ticker}![/green]")
            
            # Mostrar información adicional
            if 'escalones' in estrategia:
                total_acciones = sum(e['acciones'] for e in estrategia['escalones'])
                console.print(f"[blue]📈 Total de acciones si se ejecutan todos los escalones: {total_acciones:,}[/blue]")
        else:
            console.print(f"[red]❌ No se pudo generar estrategia para {ticker}[/red]")
            
    except ValueError as e:
        console.print(f"[red]❌ Error en los valores ingresados: {e}[/red]")
    except Exception as e:
        console.print(f"[red]❌ Error inesperado: {e}[/red]")


if __name__ == "__main__":
    console.print("[bold green]🚀 Iniciando Pruebas de Estrategia Robusta[/bold green]")
    
    # Menú de opciones
    console.print("\n[cyan]Selecciona una opción:[/cyan]")
    console.print("1. Ejecutar pruebas automáticas con múltiples tickers")
    console.print("2. Probar un ticker específico")
    console.print("3. Ejecutar ambas opciones")
    
    opcion = input("\nOpción (1-3, default: 1): ").strip() or "1"
    
    if opcion == "1":
        test_robust_strategy()
    elif opcion == "2":
        test_specific_ticker()
    elif opcion == "3":
        test_robust_strategy()
        test_specific_ticker()
    else:
        console.print("[yellow]Opción no válida, ejecutando pruebas automáticas...[/yellow]")
        test_robust_strategy()
    
    console.print("\n[bold green]✨ Pruebas completadas[/bold green]")