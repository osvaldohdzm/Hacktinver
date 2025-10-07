"""
Display - Módulo de visualización y presentación
Maneja la presentación de datos, tablas y gráficos
"""

import logging
import pandas as pd
from datetime import datetime
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from config import DATA_DIR

logger = logging.getLogger("hacktinver.display")
console = Console()


def save_results_to_csv(df: pd.DataFrame, prefix: str) -> str:
    """
    Guarda un DataFrame en un archivo CSV con timestamp
    
    Args:
        df: DataFrame a guardar
        prefix: Prefijo para el nombre del archivo
    
    Returns:
        Ruta del archivo guardado
    """
    try:
        # Crear directorio si no existe
        DATA_DIR.mkdir(exist_ok=True)
        
        # Generar nombre de archivo con timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = DATA_DIR / f"{prefix}_{timestamp}.csv"
        
        # Guardar DataFrame
        df.to_csv(filename, index=False)
        
        logger.info(f"Resultados guardados en: {filename}")
        return str(filename)
        
    except Exception as e:
        logger.error(f"Error guardando resultados: {e}")
        return f"Error: {e}"


def show_investment_tips():
    """
    Muestra consejos de inversión
    """
    console.clear()
    console.print("[bold blue]💡 Consejos de Inversión - Hacktinver[/bold blue]")
    console.print()
    
    tips = [
        "🎯 **Diversificación**: No pongas todos los huevos en una canasta",
        "📊 **Análisis Técnico**: Los indicadores son herramientas, no verdades absolutas",
        "💰 **Gestión de Riesgo**: Nunca arriesgues más del 2-3% por operación",
        "⏰ **Paciencia**: Los mejores traders esperan las mejores oportunidades",
        "📈 **Tendencia**: La tendencia es tu amiga hasta que se rompe",
        "🛡️ **Stop Loss**: Siempre define tu salida antes de entrar",
        "🧠 **Psicología**: Controla tus emociones, no dejes que ellas te controlen",
        "📚 **Educación**: Nunca dejes de aprender y mejorar tus estrategias",
        "💎 **Disciplina**: Sigue tu plan de trading sin excepciones",
        "🔄 **Adaptabilidad**: Los mercados cambian, tus estrategias también deben hacerlo"
    ]
    
    for i, tip in enumerate(tips, 1):
        console.print(f"{i:2d}. {tip}")
        console.print()
    
    console.print("[bold green]🚀 ¡Recuerda: El trading exitoso es un maratón, no una carrera![/bold green]")


def create_performance_table(results: list, title: str = "Resultados de Análisis") -> Table:
    """
    Crea una tabla de rendimiento formateada
    
    Args:
        results: Lista de diccionarios con resultados
        title: Título de la tabla
    
    Returns:
        Tabla Rich formateada
    """
    table = Table(title=title)
    
    if not results:
        table.add_column("Mensaje", style="red")
        table.add_row("No hay datos para mostrar")
        return table
    
    # Agregar columnas basadas en las claves del primer resultado
    first_result = results[0]
    for key in first_result.keys():
        table.add_column(str(key), style="cyan" if key == "Ticker" else None)
    
    # Agregar filas
    for result in results:
        row_data = [str(value) for value in result.values()]
        table.add_row(*row_data)
    
    return table


def show_strategy_summary(strategy_name: str, results: dict):
    """
    Muestra un resumen de estrategia
    
    Args:
        strategy_name: Nombre de la estrategia
        results: Diccionario con resultados
    """
    console.print(f"\n[bold blue]📊 Resumen: {strategy_name}[/bold blue]")
    
    panel_content = ""
    for key, value in results.items():
        panel_content += f"• **{key}**: {value}\n"
    
    panel = Panel(panel_content, title="Resultados", border_style="green")
    console.print(panel)


def format_currency(amount: float) -> str:
    """
    Formatea un monto como moneda
    
    Args:
        amount: Monto a formatear
    
    Returns:
        Monto formateado como string
    """
    return f"${amount:,.2f}"


def format_percentage(value: float) -> str:
    """
    Formatea un valor como porcentaje
    
    Args:
        value: Valor a formatear
    
    Returns:
        Valor formateado como porcentaje
    """
    return f"{value:.2f}%"


def show_loading_message(message: str):
    """
    Muestra un mensaje de carga
    
    Args:
        message: Mensaje a mostrar
    """
    console.print(f"[yellow]⏳ {message}...[/yellow]")


def show_success_message(message: str):
    """
    Muestra un mensaje de éxito
    
    Args:
        message: Mensaje a mostrar
    """
    console.print(f"[bold green]✅ {message}[/bold green]")


def show_error_message(message: str):
    """
    Muestra un mensaje de error
    
    Args:
        message: Mensaje a mostrar
    """
    console.print(f"[bold red]❌ {message}[/bold red]")


def show_warning_message(message: str):
    """
    Muestra un mensaje de advertencia
    
    Args:
        message: Mensaje a mostrar
    """
    console.print(f"[bold yellow]⚠️ {message}[/bold yellow]")