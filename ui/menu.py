"""
Menu - Sistema de menús interactivos mejorado
Maneja la navegación y selección de opciones con interfaz consistente
"""

import logging
import sys
from typing import Callable, List, Tuple
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table
from rich.panel import Panel
from rich.align import Align

from ui.components import HacktinverUI

# Importar funciones de estrategias
from strategies.pairs_trading import run_pairs_trading_strategy
from strategies.volatility_allocation import run_volatility_allocation_strategy
from strategies.contest_optimized import run_contest_optimized_strategy
from strategies.leveraged_etf import run_leveraged_etf_strategy
from strategies.normal_etf import run_normal_etf_strategy
from strategies.fundamental import run_fundamental_analysis
from strategies.preferences import run_preferences_analysis
from strategies.sentiment import run_sentiment_analysis
from strategies.earnings import run_earnings_strategy
from strategies.technical import run_technical_strategy
from strategies.consensus import run_consensus_strategy
from strategies.machine_learning import run_ml_strategy
from strategies.beta_strategies import run_beta1_strategy, run_beta2_strategy
from portfolio.optimization import run_sharpe_optimization, run_markowitz_optimization, run_litterman_optimization
from services.actinver_utilities import run_actinver_utilities
from services.stock_monitor import run_stock_monitor
from ui.display import show_investment_tips

# Importar utilidades del sistema
from config import clear_screen

logger = logging.getLogger("hacktinver.menu")
console = Console()

# Importar getch según el sistema operativo
try:
    from msvcrt import getch  # Windows
except ImportError:
    try:
        from getch import getch  # Unix/Linux
    except ImportError:
        def getch():
            """Fallback para sistemas sin getch"""
            return input("Presiona Enter para continuar...").encode()


def clear_screen():
    """Limpia la pantalla de la consola"""
    import os
    try:
        if os.name == "posix":
            os.system("clear")
        else:
            os.system("cls")
    except Exception:
        pass


def exit_program():
    """Función para salir del programa"""
    console.print("\n[bold blue]👋 ¡Gracias por usar Hacktinver![/bold blue]")
    console.print("[yellow]🚀 ¡Que tengas éxito en tus inversiones![/yellow]")
    sys.exit(0)


def display_main_menu():
    """
    Muestra y maneja el menú principal de la aplicación con interfaz mejorada
    """
    selected_index = 0

    # Definición del menú principal con nombres más concisos
    menu_options: List[Tuple[str, str, Callable]] = [
        ("🏆 Estrategia Optimizada para Concurso", "Momentum + Breakout + Gestión de Riesgo (6 semanas)", run_contest_optimized_strategy),
        ("🔄 Pairs Trading Avanzado", "ETFs Apalancados con Cointegración", run_pairs_trading_strategy),
        ("📊 Asignación por Volatilidad", "Gestión de Riesgo Basada en ATR", run_volatility_allocation_strategy),
        ("⚡ ETFs Apalancados", "Swing Trading con Indicadores Técnicos", run_leveraged_etf_strategy),
        ("📈 ETFs Normales", "Análisis Técnico Tradicional", run_normal_etf_strategy),
        ("🏢 Análisis Fundamental", "Evaluación por Sectores", run_fundamental_analysis),
        ("🎯 Análisis de Preferencias", "Recomendaciones Personalizadas", run_preferences_analysis),
        ("📰 Análisis de Sentimientos", "Basado en Noticias del Mercado", run_sentiment_analysis),
        ("📅 Estrategia de Resultados", "Trading por Publicación de Earnings", run_earnings_strategy),
        ("📊 Análisis Técnico Clásico", "Indicadores Técnicos Tradicionales", run_technical_strategy),
        ("🌐 Consensos Web", "Análisis de Consensos Técnicos", run_consensus_strategy),
        ("🤖 Machine Learning", "Estrategias con Inteligencia Artificial", run_ml_strategy),
        ("🧪 Estrategia Beta 1", "Swing Trading Doble Negativo", run_beta1_strategy),
        ("🧪 Estrategia Beta 2", "Swing Trading Positivo-Negativo-Positivo", run_beta2_strategy),
        ("⚖️ Optimización Sharpe", "Razón de Sharpe Ajustada Corto Plazo", run_sharpe_optimization),
        ("📐 Optimización Markowitz", "Teoría Moderna de Portafolios", run_markowitz_optimization),
        ("🎲 Optimización Litterman", "Modelo Black-Litterman", run_litterman_optimization),
        ("🎮 Utilidades Actinver", "Herramientas del Concurso 2024", run_actinver_utilities),
        ("📺 Monitor de Stocks", "Seguimiento en Tiempo Real", run_stock_monitor),
        ("💡 Consejos de Inversión", "Tips y Mejores Prácticas", show_investment_tips),
        ("🚪 Salir", "Cerrar Aplicación", None),
    ]

    def display_menu(selected_index: int):
        """Muestra el menú con interfaz mejorada"""
        HacktinverUI.show_header(
            "Menú Principal",
            "Selecciona una estrategia de análisis de inversiones"
        )
        
        # Crear tabla del menú
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Opción", style="bold cyan", width=4)
        table.add_column("Estrategia", style="white", width=35)
        table.add_column("Descripción", style="dim", width=40)
        
        for i, (name, description, _) in enumerate(menu_options):
            if name == "🚪 Salir":
                option_num = "q"
                style = "bold red" if i == selected_index else "red"
            else:
                option_num = str(i + 1)
                style = "bold green" if i == selected_index else "white"
            
            # Indicador de selección
            prefix = "→ " if i == selected_index else "  "
            
            table.add_row(
                f"{prefix}{option_num}",
                f"[{style}]{name}[/{style}]",
                f"[dim]{description}[/dim]"
            )
        
        console.print(table)
        
        # Instrucciones de navegación
        nav_panel = Panel(
            "[cyan]↑↓[/cyan] Navegar  •  [green]Enter[/green] Seleccionar  •  [yellow]1-20[/yellow] Acceso directo  •  [blue]:[/blue] Comando  •  [red]q[/red] Salir",
            title="Navegación",
            style="dim"
        )
        console.print(nav_panel)

    # Bucle principal del menú
    ch = ''
    
    while ch != 'q':
        display_menu(selected_index)
        ch = getch()  # Lee un carácter de la entrada

        # Obtener valor ASCII del carácter
        try:
            if isinstance(ch, bytes):
                ch = ch.decode('utf-8')
            ascii_value = ord(ch)
        except (UnicodeDecodeError, TypeError):
            continue
        
        # Procesar las teclas
        if ascii_value == 224:  # Teclas especiales (flechas, etc.) en Windows
            try:
                ch = getch()  # Lee el siguiente carácter para obtener el código de la flecha
                if isinstance(ch, bytes):
                    ch = ch.decode('utf-8')
                ascii_value = ord(ch)
                
                if ascii_value == 72:  # Flecha arriba
                    selected_index = (selected_index - 1) % len(menu_options)
                elif ascii_value == 80:  # Flecha abajo
                    selected_index = (selected_index + 1) % len(menu_options)
            except:
                continue
                
        elif ascii_value == 13:  # Enter
            selected_option = menu_options[selected_index]
            if selected_option[2] is None:  # Opción "Salir"
                exit_program()
            else:
                clear_screen()
                try:
                    logger.info(f"Ejecutando: {selected_option[0]}")
                    selected_option[2]()  # La función está en el índice 2
                except Exception as e:
                    HacktinverUI.show_message(f"Error ejecutando la opción: {e}", "error")
                    logger.error(f"Error en menú: {e}")
                
                HacktinverUI.wait_for_user()
                
        elif ch == 'q' or ch == 'Q':
            exit_program()
            
        elif ch == ':':
            # Modo comando
            try:
                opcion_main_menu = Prompt.ask("[bold green]Ingresa el número de opción o 'quit' para salir[/bold green]")
                
                if opcion_main_menu.lower() == 'quit':
                    exit_program()
                
                try:
                    selected_index = int(opcion_main_menu) - 1  # Ajustar el índice (restar 1)
                    if 0 <= selected_index < len(menu_options) - 1:  # Excluir "Salir"
                        clear_screen()
                        selected_option = menu_options[selected_index]
                        logger.info(f"Ejecutando (comando directo): {selected_option[0]}")
                        selected_option[2]()  # Función está en índice 2
                        HacktinverUI.wait_for_user()
                    else:
                        HacktinverUI.show_message(f"Opción fuera de rango (1-{len(menu_options)-1})", "error")
                        HacktinverUI.wait_for_user()
                        
                except ValueError:
                    HacktinverUI.show_message("Entrada no válida. Introduce un número o 'quit'", "error")
                    HacktinverUI.wait_for_user()
                    
            except KeyboardInterrupt:
                continue
                
        # Navegación directa por número
        elif ch.isdigit():
            try:
                option_num = int(ch)
                if 1 <= option_num <= len(menu_options) - 1:  # Excluir "Salir"
                    selected_index = option_num - 1
                    # Auto-ejecutar después de un breve delay
                    import time
                    time.sleep(0.5)
                    
                    clear_screen()
                    selected_option = menu_options[selected_index]
                    logger.info(f"Ejecutando (número directo): {selected_option[0]}")
                    selected_option[2]()  # Función está en índice 2
                    HacktinverUI.wait_for_user()
            except:
                continue


def show_strategy_submenu(title: str, strategies: List[Tuple[str, Callable]]) -> None:
    """
    Muestra un submenú para estrategias específicas
    
    Args:
        title: Título del submenú
        strategies: Lista de tuplas (nombre, función)
    """
    selected_index = 0
    
    def display_submenu(selected_index: int):
        console.clear()
        console.print(f"[bold blue]{title}[/bold blue]")
        console.print("[yellow]Usa las flechas ↑↓ para navegar, Enter para seleccionar, 'b' para volver[/yellow]")
        console.print()
        
        for i, (strategy_name, _) in enumerate(strategies):
            prefix = "→ " if i == selected_index else "   "
            style = "[bold green]" if i == selected_index else ""
            end_style = "[/bold green]" if i == selected_index else ""
            console.print(f"{prefix}{i + 1}. {style}{strategy_name}{end_style}")
        
        console.print(f"\n   b. [bold yellow]Volver al menú principal[/bold yellow]")
    
    ch = ''
    while ch != 'b':
        display_submenu(selected_index)
        ch = getch()
        
        try:
            if isinstance(ch, bytes):
                ch = ch.decode('utf-8')
            ascii_value = ord(ch)
        except:
            continue
        
        if ascii_value == 224:  # Flechas en Windows
            try:
                ch = getch()
                if isinstance(ch, bytes):
                    ch = ch.decode('utf-8')
                ascii_value = ord(ch)
                
                if ascii_value == 72:  # Flecha arriba
                    selected_index = (selected_index - 1) % len(strategies)
                elif ascii_value == 80:  # Flecha abajo
                    selected_index = (selected_index + 1) % len(strategies)
            except:
                continue
                
        elif ascii_value == 13:  # Enter
            selected_strategy = strategies[selected_index]
            clear_screen()
            try:
                logger.info(f"Ejecutando estrategia: {selected_strategy[0]}")
                selected_strategy[1]()
            except Exception as e:
                console.print(f"[bold red]❌ Error ejecutando la estrategia: {e}[/bold red]")
                logger.error(f"Error en estrategia: {e}")
            
            Prompt.ask("[bold blue]Pulsa Enter para continuar...[/bold blue]")
            
        elif ch == 'b' or ch == 'B':
            break


def show_help_menu():
    """
    Muestra el menú de ayuda
    """
    console.clear()
    console.print("[bold blue]📚 Ayuda - Hacktinver v2.0[/bold blue]")
    console.print()
    
    console.print("[bold green]🎯 Navegación:[/bold green]")
    console.print("• Flechas ↑↓: Navegar entre opciones")
    console.print("• Enter: Seleccionar opción")
    console.print("• Número + Enter: Acceso directo a opción")
    console.print("• ':' + número: Modo comando")
    console.print("• 'q': Salir de la aplicación")
    console.print()
    
    console.print("[bold green]🚀 Estrategias Principales:[/bold green]")
    console.print("• Pairs Trading: Estrategia de mercado neutral")
    console.print("• Volatilidad: Asignación basada en riesgo")
    console.print("• ETFs Apalancados: Análisis de alta volatilidad")
    console.print("• Análisis Fundamental: Evaluación por sectores")
    console.print("• Machine Learning: Estrategias con IA")
    console.print()
    
    console.print("[bold green]📊 Utilidades:[/bold green]")
    console.print("• Monitor de Stocks: Seguimiento en tiempo real")
    console.print("• Optimización de Portafolio: Markowitz, Sharpe")
    console.print("• Utilidades Actinver: Herramientas del concurso")
    console.print()
    
    Prompt.ask("[bold blue]Pulsa Enter para volver al menú principal...[/bold blue]")