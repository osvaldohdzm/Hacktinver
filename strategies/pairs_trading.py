"""
Pairs Trading Strategy - Estrategia de trading de pares
Implementa el algoritmo avanzado de pairs trading con ETFs apalancados
"""

import logging
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any
from rich.console import Console
from rich.table import Table

from core.data_provider import download_multiple_tickers, validate_ticker_data
from core.indicators import calculate_correlation_matrix
from config import DEFAULT_PAIRS_TRADING, DATA_DIR, DEFAULT_LEVERAGED_ETFS, DEFAULT_NORMAL_ETFS, DEFAULT_FAVORITE_STOCKS
from ui.display import save_results_to_csv
from ui.components import HacktinverUI, InputValidator, DataFormatter

logger = logging.getLogger("hacktinver.pairs_trading")
console = Console()


def get_expanded_pairs_catalog():
    """
    Obtiene el catálogo expandido de pares para análisis
    Incluye múltiples categorías de pares potencialmente cointegrados
    """
    catalog = {
        "predefinidos": DEFAULT_PAIRS_TRADING,
        
        "indices_similares": [
            # ETFs que replican el mismo índice (muy alta probabilidad de cointegración)
            ("SPY", "VOO"),      # S&P 500 - SPDR vs Vanguard
            ("SPY", "IVV"),      # S&P 500 - SPDR vs iShares
            ("VOO", "IVV"),      # S&P 500 - Vanguard vs iShares
            ("QQQ", "QQQM"),     # NASDAQ-100 - Invesco vs Invesco Mini
            ("VTI", "ITOT"),     # Total Stock Market
            ("VEA", "IEFA"),     # Developed Markets
            ("VWO", "IEMG"),     # Emerging Markets
            ("GLD", "IAU"),      # Gold ETFs
            ("SLV", "SIVR"),     # Silver ETFs
            ("TLT", "VGLT"),     # Long-term Treasury
        ],
        
        "sectorial_vs_componente": [
            # Sector ETF vs su componente principal
            ("XLF", "JPM"),      # Financieros vs JP Morgan
            ("XLF", "BAC"),      # Financieros vs Bank of America
            ("XLF", "WFC"),      # Financieros vs Wells Fargo
            ("XLK", "MSFT"),     # Tecnología vs Microsoft
            ("XLK", "AAPL"),     # Tecnología vs Apple
            ("XLK", "GOOGL"),    # Tecnología vs Google
            ("XLE", "XOM"),      # Energía vs Exxon Mobil
            ("XLE", "CVX"),      # Energía vs Chevron
            ("XLV", "JNJ"),      # Salud vs Johnson & Johnson
            ("XLV", "PFE"),      # Salud vs Pfizer
            ("XLI", "GE"),       # Industrial vs General Electric
            ("XLI", "CAT"),      # Industrial vs Caterpillar
            ("XLY", "AMZN"),     # Consumer Discretionary vs Amazon
            ("XLY", "TSLA"),     # Consumer Discretionary vs Tesla
            ("XLP", "PG"),       # Consumer Staples vs Procter & Gamble
            ("XLP", "KO"),       # Consumer Staples vs Coca-Cola
        ],
        
        "competidores": [
            # Empresas competidoras de la misma industria
            ("JPM", "BAC"),      # Bancos grandes
            ("JPM", "WFC"),      # Bancos grandes
            ("BAC", "WFC"),      # Bancos grandes
            ("C", "GS"),         # Bancos de inversión
            ("AMD", "INTC"),     # Semiconductores
            ("AMD", "NVDA"),     # Semiconductores/GPU
            ("MSFT", "GOOGL"),   # Tech giants
            ("MSFT", "AAPL"),    # Tech giants
            ("GOOGL", "META"),   # Internet/Social
            ("KO", "PEP"),       # Bebidas
            ("MCD", "YUM"),      # Fast food
            ("WMT", "TGT"),      # Retail
            ("HD", "LOW"),       # Home improvement
            ("UPS", "FDX"),      # Logistics
            ("BA", "LMT"),       # Aerospace/Defense
            ("XOM", "CVX"),      # Oil majors
            ("JNJ", "PFE"),      # Pharma
            ("DIS", "NFLX"),     # Entertainment
            ("V", "MA"),         # Payment processors
        ],
        
        "apalancados_relacionados": [
            # ETFs apalancados con relaciones potenciales
            ("SOXL", "TECL"),    # Ya incluido pero importante
            ("SPXL", "UPRO"),    # S&P 500 3x diferentes proveedores
            ("TQQQ", "QLD"),     # NASDAQ 3x vs 2x
            ("FAS", "DPST"),     # Financieros 3x diferentes
            ("TNA", "UWM"),      # Small caps 3x vs 2x
            ("LABU", "XBI"),     # Biotech 3x vs 1x
            ("CURE", "XLV"),     # Healthcare 3x vs 1x
            ("DFEN", "XAR"),     # Defense 3x vs 1x
            ("NAIL", "XHB"),     # Homebuilders 3x vs 1x
            ("GUSH", "XLE"),     # Oil & Gas 3x vs 1x
        ],
        
        "inversos_relacionados": [
            # ETFs inversos con sus contrapartes
            ("SPXS", "SPXL"),    # S&P 500 -3x vs +3x
            ("SQQQ", "TQQQ"),    # NASDAQ -3x vs +3x
            ("FAZ", "FAS"),      # Financieros -3x vs +3x
            ("TZA", "TNA"),      # Small caps -3x vs +3x
            ("SOXS", "SOXL"),    # Semiconductores -3x vs +3x
            ("TECS", "TECL"),    # Tecnología -3x vs +3x
            ("SH", "SPY"),       # S&P 500 -1x vs +1x
            ("PSQ", "QQQ"),      # NASDAQ -1x vs +1x
            ("DOG", "DIA"),      # Dow -1x vs +1x
        ]
    }
    
    return catalog


def get_pairs_by_category(category_choice, expanded_pairs):
    """
    Obtiene pares según la categoría seleccionada
    """
    category_map = {
        1: expanded_pairs["predefinidos"],
        2: expanded_pairs["indices_similares"],
        3: expanded_pairs["sectorial_vs_componente"],
        4: expanded_pairs["competidores"],
        5: [],  # Descubrimiento automático - se maneja por separado
        6: []   # Análisis completo - se combina todo
    }
    
    if category_choice == 5:
        # Descubrimiento automático
        return discover_cointegrated_pairs()
    elif category_choice == 6:
        # Análisis completo - combinar todas las categorías
        all_pairs = []
        for key in ["predefinidos", "indices_similares", "sectorial_vs_componente", 
                   "competidores", "apalancados_relacionados", "inversos_relacionados"]:
            all_pairs.extend(expanded_pairs[key])
        return all_pairs
    else:
        return category_map.get(category_choice, expanded_pairs["predefinidos"])


def discover_cointegrated_pairs():
    """
    Descubre automáticamente pares cointegrados usando correlación y análisis estadístico
    """
    HacktinverUI.show_message("Iniciando descubrimiento automático de pares...", "info")
    
    # Combinar todos los tickers disponibles
    all_tickers = list(set(
        DEFAULT_LEVERAGED_ETFS + 
        DEFAULT_NORMAL_ETFS[:20] +  # Limitar para no sobrecargar
        [ticker.replace('.MX', '') for ticker in DEFAULT_FAVORITE_STOCKS[:10]]
    ))
    
    # Descargar datos para análisis de correlación
    console.print(f"[yellow]📊 Descargando datos para {len(all_tickers)} activos...[/yellow]")
    tickers_data = download_multiple_tickers(all_tickers, period="6mo")
    
    if len(tickers_data) < 5:
        HacktinverUI.show_message("Datos insuficientes para descubrimiento automático", "error")
        return []
    
    # Calcular matriz de correlación
    correlation_matrix = calculate_correlation_matrix(tickers_data)
    
    # Encontrar pares con alta correlación (>0.7) pero no perfecta (evitar duplicados)
    discovered_pairs = []
    processed_pairs = set()
    
    for ticker1 in correlation_matrix.index:
        for ticker2 in correlation_matrix.columns:
            if ticker1 != ticker2:
                pair_key = tuple(sorted([ticker1, ticker2]))
                if pair_key not in processed_pairs:
                    correlation = correlation_matrix.loc[ticker1, ticker2]
                    if 0.7 <= abs(correlation) <= 0.98:  # Alta correlación pero no perfecta
                        discovered_pairs.append((ticker1, ticker2))
                        processed_pairs.add(pair_key)
    
    # Limitar a los 20 pares más prometedores
    discovered_pairs = discovered_pairs[:20]
    
    HacktinverUI.show_message(f"Descubiertos {len(discovered_pairs)} pares potenciales", "success")
    return discovered_pairs


def show_selected_pairs(pairs_to_analyze, category_name):
    """
    Muestra los pares seleccionados para análisis
    """
    HacktinverUI.show_section_header(f"Pares Seleccionados - {category_name}", "target")
    
    if len(pairs_to_analyze) <= 15:
        # Mostrar todos los pares si son pocos
        pairs_table = Table(show_header=True, header_style="bold blue")
        pairs_table.add_column("Nº", style="cyan", width=4)
        pairs_table.add_column("Par", style="white", width=15)
        pairs_table.add_column("Tipo", style="dim", width=25)
        
        for i, (etf1, etf2) in enumerate(pairs_to_analyze, 1):
            pair_type = classify_pair_type(etf1, etf2)
            pairs_table.add_row(str(i), f"{etf1} / {etf2}", pair_type)
        
        console.print(pairs_table)
    else:
        # Mostrar resumen si son muchos pares
        console.print(f"[bold cyan]Total de pares a analizar: {len(pairs_to_analyze)}[/bold cyan]")
        console.print(f"[dim]Primeros 5 pares: {', '.join([f'{p[0]}/{p[1]}' for p in pairs_to_analyze[:5]])}...[/dim]")
    
    # Mostrar estadísticas por tipo
    pair_types = {}
    for etf1, etf2 in pairs_to_analyze:
        pair_type = classify_pair_type(etf1, etf2)
        pair_types[pair_type] = pair_types.get(pair_type, 0) + 1
    
    if pair_types:
        console.print(f"\n[bold blue]Distribución por tipo:[/bold blue]")
        for pair_type, count in pair_types.items():
            console.print(f"  • {pair_type}: {count} pares")


def classify_pair_type(etf1, etf2):
    """
    Clasifica el tipo de par para mejor comprensión
    """
    # ETFs apalancados
    leveraged_etfs = set(DEFAULT_LEVERAGED_ETFS)
    if etf1 in leveraged_etfs and etf2 in leveraged_etfs:
        return "ETFs Apalancados"
    elif etf1 in leveraged_etfs or etf2 in leveraged_etfs:
        return "Apalancado vs Normal"
    
    # Índices similares
    sp500_etfs = {"SPY", "VOO", "IVV", "SPLG"}
    nasdaq_etfs = {"QQQ", "QQQM", "ONEQ"}
    gold_etfs = {"GLD", "IAU", "SGOL"}
    
    if {etf1, etf2}.issubset(sp500_etfs):
        return "S&P 500 Similar"
    elif {etf1, etf2}.issubset(nasdaq_etfs):
        return "NASDAQ Similar"
    elif {etf1, etf2}.issubset(gold_etfs):
        return "Gold Similar"
    
    # Sectores
    sector_etfs = {"XLF", "XLK", "XLE", "XLV", "XLI", "XLY", "XLP"}
    if etf1 in sector_etfs or etf2 in sector_etfs:
        return "Sectorial"
    
    # Por defecto
    return "Competidores/Otros"


def run_pairs_trading_strategy():
    """
    Ejecuta la estrategia de pairs trading avanzada con interfaz mejorada
    Incluye descubrimiento automático de nuevos pares cointegrados
    """
    # Mostrar header consistente
    HacktinverUI.show_header(
        "Pairs Trading Avanzado",
        "Estrategia cuantitativa de mercado neutral con validación estadística"
    )
    
    # Verificar dependencias
    try:
        from statsmodels.tsa.stattools import adfuller
        HacktinverUI.show_message("Módulo statsmodels cargado correctamente", "success")
    except ImportError:
        HacktinverUI.show_message(
            "statsmodels no está instalado. Ejecuta: pip install statsmodels", 
            "error",
            "Dependencia Faltante"
        )
        return
    
    # Definir categorías expandidas de pares
    expanded_pairs = get_expanded_pairs_catalog()
    
    # Mostrar opciones de análisis
    HacktinverUI.show_section_header("Opciones de Análisis", "search")
    
    analysis_options = [
        "Pares Predefinidos (ETFs Apalancados)",
        "Índices Similares (Bajo Riesgo)",
        "Sectorial vs Componente Principal",
        "Competidores de la Misma Industria",
        "Descubrimiento Automático de Pares",
        "Análisis Completo (Todas las Categorías)"
    ]
    
    options_table = Table(show_header=True, header_style="bold blue")
    options_table.add_column("Opción", style="cyan", width=8)
    options_table.add_column("Categoría", style="white", width=35)
    options_table.add_column("Descripción", style="dim", width=40)
    
    option_descriptions = [
        "Pares tradicionales de ETFs apalancados",
        "ETFs que replican el mismo índice (SPY/VOO)",
        "Sector ETF vs su componente principal (XLF/JPM)",
        "Empresas competidoras (JPM/BAC, AMD/INTC)",
        "Busca automáticamente pares cointegrados",
        "Analiza todas las categorías disponibles"
    ]
    
    for i, (option, desc) in enumerate(zip(analysis_options, option_descriptions), 1):
        options_table.add_row(str(i), option, desc)
    
    console.print(options_table)
    
    # Selección de categoría de análisis
    category_choice = HacktinverUI.get_user_input(
        "Selecciona el tipo de análisis",
        "choice",
        "1",
        choices=[str(i) for i in range(1, len(analysis_options) + 1)]
    )
    
    # Obtener pares según la selección
    pairs_to_analyze = get_pairs_by_category(int(category_choice), expanded_pairs)
    
    if not pairs_to_analyze:
        HacktinverUI.show_message("No se encontraron pares para analizar", "error")
        return
    
    # Mostrar pares seleccionados
    show_selected_pairs(pairs_to_analyze, analysis_options[int(category_choice) - 1])
    
    # Obtener configuración del usuario con validación
    HacktinverUI.show_section_header("Configuración de Análisis", "gear")
    
    # Permitir selección específica solo si hay pocos pares
    if len(pairs_to_analyze) <= 20:
        selection = HacktinverUI.get_user_input(
            "Selecciona pares específicos (ej: 1,3,5 para múltiples o Enter para todos)",
            "text",
            "",
            validate_func=lambda x: not x or all(
                part.strip().isdigit() and 1 <= int(part.strip()) <= len(pairs_to_analyze)
                for part in x.split(',') if part.strip()
            )
        )
        
        # Procesar selección específica
        if selection and selection.strip():
            try:
                if ',' in selection:
                    selected_indices = [int(x.strip()) - 1 for x in selection.split(',')]
                    selected_pairs = [pairs_to_analyze[idx] for idx in selected_indices 
                                    if 0 <= idx < len(pairs_to_analyze)]
                    if selected_pairs:
                        pairs_to_analyze = selected_pairs
                    else:
                        HacktinverUI.show_message("No se seleccionaron pares válidos, usando todos", "warning")
                else:
                    pair_index = int(selection) - 1
                    if 0 <= pair_index < len(pairs_to_analyze):
                        pairs_to_analyze = [pairs_to_analyze[pair_index]]
                    else:
                        HacktinverUI.show_message("Selección inválida, usando todos los pares", "warning")
            except ValueError:
                HacktinverUI.show_message("Formato inválido, usando todos los pares", "warning")
    else:
        HacktinverUI.show_message(f"Analizando todos los {len(pairs_to_analyze)} pares de la categoría seleccionada", "info")
    
    # Parámetros de gestión de riesgo
    monto_por_pata = HacktinverUI.get_user_input(
        "Monto por cada lado de la operación",
        "float",
        100000,
        validate_func=InputValidator.positive_number
    )
    
    lookback_window = HacktinverUI.get_user_input(
        "Ventana de análisis en días",
        "int", 
        30,
        validate_func=lambda x: 10 <= x <= 90
    )
    
    # Mostrar configuración con tarjetas de resumen
    config_summary = {
        "Monto por Operación": DataFormatter.currency(monto_por_pata),
        "Ventana de Análisis": f"{lookback_window} días",
        "Pares a Analizar": len(pairs_to_analyze),
        "Estrategia": "Mercado Neutral"
    }
    HacktinverUI.show_summary_cards(config_summary, "Configuración del Análisis")
    
    results = []
    cointegrated_pairs = 0
    
    # Mostrar progreso del análisis
    HacktinverUI.show_section_header("Análisis de Cointegración", "search")
    
    for etf1, etf2 in HacktinverUI.show_progress_bar(pairs_to_analyze, "Analizando pares"):
        try:
            console.print(f"\n[dim]🔍 Analizando par: {etf1} / {etf2}...[/dim]")
            
            # Descargar datos para ambos ETFs
            tickers_data = download_multiple_tickers([etf1, etf2], period="1y")
            
            if etf1 not in tickers_data or etf2 not in tickers_data:
                console.print(f"[red]⚠️ No se pudieron obtener datos para {etf1} o {etf2}[/red]")
                continue
            
            close1 = tickers_data[etf1]
            close2 = tickers_data[etf2]
            
            # Validar datos
            if not validate_ticker_data(close1, 60) or not validate_ticker_data(close2, 60):
                console.print(f"[red]⚠️ Datos insuficientes para {etf1}/{etf2}[/red]")
                continue
            
            # Alinear fechas
            common_dates = close1.index.intersection(close2.index)
            close1_aligned = close1.loc[common_dates]
            close2_aligned = close2.loc[common_dates]
            
            # Calcular el ratio
            ratio = close1_aligned / close2_aligned
            ratio_clean = ratio.dropna()
            
            # Test de cointegración
            try:
                adf_result = adfuller(ratio_clean)
                p_value_cointegration = adf_result[1]
                
                if p_value_cointegration >= 0.05:
                    console.print(f"[bold red]❌ Par {etf1}/{etf2} NO es cointegrado (p-valor: {p_value_cointegration:.4f}). Se descarta.[/bold red]")
                    continue
                else:
                    console.print(f"[green]✅ Par {etf1}/{etf2} es cointegrado (p-valor: {p_value_cointegration:.4f})[/green]")
                    cointegrated_pairs += 1
                    
            except Exception as e:
                console.print(f"[yellow]⚠️ Error en test de cointegración para {etf1}/{etf2}: {e}[/yellow]")
                continue
            
            # Calcular estadísticas del ratio
            ratio_sma = ratio_clean.rolling(window=lookback_window).mean()
            ratio_std = ratio_clean.rolling(window=lookback_window).std()
            
            # Valores actuales
            current_ratio = float(ratio_clean.iloc[-1])
            current_sma = float(ratio_sma.iloc[-1])
            current_std = float(ratio_std.iloc[-1])
            
            # Calcular Z-Score
            z_score = (current_ratio - current_sma) / current_std if current_std > 0 else 0
            
            # Precios actuales
            price1_current = float(close1_aligned.iloc[-1])
            price2_current = float(close2_aligned.iloc[-1])
            
            # Calcular correlación
            correlation = close1_aligned.corr(close2_aligned)
            
            # Generar señales
            signal = "ESPERAR"
            confidence = "BAJA"
            
            if abs(z_score) >= 2.0:
                confidence = "ALTA"
                if z_score > 2.0:
                    signal = "VENDER ETF1 / COMPRAR ETF2"
                elif z_score < -2.0:
                    signal = "COMPRAR ETF1 / VENDER ETF2"
            elif abs(z_score) >= 1.5:
                confidence = "MEDIA"
                if z_score > 1.5:
                    signal = "CONSIDERAR VENTA ETF1"
                elif z_score < -1.5:
                    signal = "CONSIDERAR COMPRA ETF1"
            
            # Dimensionamiento de posición
            acciones_etf1 = int(monto_por_pata / price1_current)
            acciones_etf2 = int(monto_por_pata / price2_current)
            
            # Calcular retorno esperado
            expected_return_pct = 0
            if abs(z_score) >= 1.5:
                target_ratio = current_sma
                if z_score > 0:
                    expected_return_pct = ((target_ratio / current_ratio) - 1) * 100
                else:
                    expected_return_pct = ((current_ratio / target_ratio) - 1) * 100
            
            results.append({
                "Par": f"{etf1}/{etf2}",
                "ETF1": etf1,
                "ETF2": etf2,
                "Precio ETF1": price1_current,
                "Precio ETF2": price2_current,
                "Ratio Actual": current_ratio,
                "Z-Score": z_score,
                "Correlación": correlation,
                "P-Valor Cointeg.": p_value_cointegration,
                "Señal": signal,
                "Confianza": confidence,
                "Acciones ETF1": acciones_etf1,
                "Acciones ETF2": acciones_etf2,
                "Retorno Esperado (%)": expected_return_pct
            })
            
        except Exception as e:
            console.print(f"[red]❌ Error analizando {etf1}/{etf2}: {e}[/red]")
            logger.error(f"Error en pairs trading {etf1}/{etf2}: {e}")
            continue
    
    if not results:
        console.print("[bold red]❌ No se encontraron pares cointegrados válidos[/bold red]")
        return
    
    console.print(f"\n[bold green]✅ Pares cointegrados encontrados: {cointegrated_pairs}/{len(pairs_to_analyze)}[/bold green]")
    
    # Crear DataFrame y mostrar resultados
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('Z-Score', key=abs, ascending=False)
    
    # Mostrar tabla
    table = Table(title="🔄 Pairs Trading - Análisis Cuantitativo")
    table.add_column("Par", style="cyan", no_wrap=True)
    table.add_column("Z-Score", style="yellow")
    table.add_column("P-Valor", style="magenta")
    table.add_column("Correlación", style="green")
    table.add_column("Señal", style="bold")
    table.add_column("Confianza", style="blue")
    table.add_column("Ret. Esp.", style="red")
    
    for _, row in df_results.iterrows():
        z_score = row['Z-Score']
        
        # Colorear Z-Score
        if abs(z_score) >= 2.0:
            z_score_str = f"[bold red]{z_score:.2f}[/bold red]"
        elif abs(z_score) >= 1.5:
            z_score_str = f"[bold yellow]{z_score:.2f}[/bold yellow]"
        else:
            z_score_str = f"{z_score:.2f}"
        
        # Colorear señal
        signal = row['Señal']
        if "VENDER" in signal and "COMPRAR" in signal:
            signal_str = f"[bold green]{signal}[/bold green]"
        elif "CONSIDERAR" in signal:
            signal_str = f"[yellow]{signal}[/yellow]"
        else:
            signal_str = signal
        
        table.add_row(
            row['Par'],
            z_score_str,
            f"{row['P-Valor Cointeg.']:.4f}",
            f"{row['Correlación']:.3f}",
            signal_str,
            row['Confianza'],
            f"{row['Retorno Esperado (%)']:.2f}%"
        )
    
    console.print(table)
    
    # Mostrar recomendaciones
    high_confidence = df_results[df_results['Confianza'] == 'ALTA']
    
    console.print(f"\n[bold green]🎯 OPORTUNIDADES DE ALTA CONFIANZA ({len(high_confidence)}):[/bold green]")
    if not high_confidence.empty:
        for _, row in high_confidence.iterrows():
            console.print(f"   • {row['Señal']} - {row['Par']} (Z-Score: {row['Z-Score']:.2f}, Retorno Esp: {row['Retorno Esperado (%)']:.2f}%)")
    else:
        console.print("   [dim]No hay oportunidades de alta confianza en este momento[/dim]")
    
    # Guardar resultados
    filename = save_results_to_csv(df_results, "pairs_trading_advanced")
    console.print(f"\n[bold yellow]📁 Resultados guardados en: {filename}[/bold yellow]")
    
    console.print(f"\n[bold blue]🧠 Conceptos Clave del Pairs Trading:[/bold blue]")
    console.print("• [green]Test de Cointegración[/green]: Solo pares estadísticamente válidos")
    console.print("• [yellow]Z-Score > ±2.0[/yellow]: Señales de alta confianza")
    console.print("• [cyan]Mercado Neutral[/cyan]: Ganancias independientes de la dirección del mercado")
    console.print("• [red]Stop-Loss[/red]: Z-Score > ±3.5 para protección de capital")