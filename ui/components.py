"""
UI Components - Componentes reutilizables de interfaz de usuario
Proporciona elementos consistentes para toda la aplicación
"""

import os
from typing import List, Dict, Any, Optional, Union
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.align import Align
from rich.text import Text
from rich.rule import Rule
from rich.columns import Columns

console = Console()


class HacktinverUI:
    """
    Clase principal para componentes de UI consistentes
    """
    
    # Colores y estilos consistentes
    COLORS = {
        'primary': 'bold blue',
        'secondary': 'cyan',
        'success': 'bold green',
        'warning': 'bold yellow',
        'error': 'bold red',
        'info': 'blue',
        'muted': 'dim',
        'accent': 'magenta'
    }
    
    # Emojis consistentes
    ICONS = {
        'success': '✅',
        'error': '❌',
        'warning': '⚠️',
        'info': 'ℹ️',
        'loading': '⏳',
        'money': '💰',
        'chart': '📊',
        'rocket': '🚀',
        'target': '🎯',
        'fire': '🔥',
        'trophy': '🏆',
        'gear': '⚙️',
        'search': '🔍',
        'save': '💾',
        'up': '📈',
        'down': '📉',
        'neutral': '➡️'
    }
    
    @staticmethod
    def clear_screen():
        """Limpia la pantalla de manera consistente"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    @staticmethod
    def show_header(title: str, subtitle: str = None, version: str = "v2.0"):
        """
        Muestra un header consistente para todas las pantallas
        
        Args:
            title: Título principal
            subtitle: Subtítulo opcional
            version: Versión de la aplicación
        """
        HacktinverUI.clear_screen()
        
        # Header principal
        header_text = Text()
        header_text.append("🚀 ", style="bold blue")
        header_text.append("Hacktinver ", style="bold blue")
        header_text.append(version, style="dim blue")
        header_text.append(" - ", style="dim")
        header_text.append(title, style="bold white")
        
        header_panel = Panel(
            Align.center(header_text),
            style="blue",
            padding=(1, 2)
        )
        console.print(header_panel)
        
        if subtitle:
            console.print(Align.center(f"[yellow]{subtitle}[/yellow]"))
            console.print()
    
    @staticmethod
    def show_section_header(title: str, icon: str = None):
        """
        Muestra un header de sección
        
        Args:
            title: Título de la sección
            icon: Icono opcional
        """
        if icon and icon in HacktinverUI.ICONS:
            title = f"{HacktinverUI.ICONS[icon]} {title}"
        
        console.print(Rule(title, style="blue"))
        console.print()
    
    @staticmethod
    def get_user_input(
        prompt: str,
        input_type: str = "text",
        default: Any = None,
        choices: List[str] = None,
        validate_func: callable = None
    ) -> Any:
        """
        Obtiene input del usuario de manera consistente
        
        Args:
            prompt: Texto del prompt
            input_type: Tipo de input ('text', 'int', 'float', 'bool', 'choice')
            default: Valor por defecto
            choices: Lista de opciones válidas
            validate_func: Función de validación personalizada
        
        Returns:
            Valor ingresado por el usuario
        """
        # Formatear prompt con estilo consistente
        styled_prompt = f"[bold cyan]❓ {prompt}[/bold cyan]"
        
        if default is not None:
            styled_prompt += f" [dim](default: {default})[/dim]"
        
        styled_prompt += ": "
        
        try:
            if input_type == "bool":
                return Confirm.ask(styled_prompt, default=default)
            
            elif input_type == "choice" and choices:
                return Prompt.ask(
                    styled_prompt,
                    choices=choices,
                    default=str(default) if default else None
                )
            
            elif input_type in ["int", "float"]:
                while True:
                    try:
                        response = Prompt.ask(styled_prompt, default=str(default) if default else "")
                        if not response and default is not None:
                            return default
                        
                        value = int(response) if input_type == "int" else float(response)
                        
                        if validate_func and not validate_func(value):
                            console.print(f"[{HacktinverUI.COLORS['error']}]Valor inválido. Intenta de nuevo.[/{HacktinverUI.COLORS['error']}]")
                            continue
                        
                        return value
                    except ValueError:
                        console.print(f"[{HacktinverUI.COLORS['error']}]Por favor ingresa un número válido.[/{HacktinverUI.COLORS['error']}]")
            
            else:  # text
                response = Prompt.ask(styled_prompt, default=str(default) if default else "")
                
                if validate_func and not validate_func(response):
                    console.print(f"[{HacktinverUI.COLORS['error']}]Entrada inválida.[/{HacktinverUI.COLORS['error']}]")
                    return HacktinverUI.get_user_input(prompt, input_type, default, choices, validate_func)
                
                return response
        
        except KeyboardInterrupt:
            console.print(f"\n[{HacktinverUI.COLORS['warning']}]Operación cancelada por el usuario.[/{HacktinverUI.COLORS['warning']}]")
            return None
    
    @staticmethod
    def show_message(message: str, msg_type: str = "info", title: str = None):
        """
        Muestra un mensaje con formato consistente
        
        Args:
            message: Mensaje a mostrar
            msg_type: Tipo de mensaje ('success', 'error', 'warning', 'info')
            title: Título opcional
        """
        icon = HacktinverUI.ICONS.get(msg_type, HacktinverUI.ICONS['info'])
        color = HacktinverUI.COLORS.get(msg_type, HacktinverUI.COLORS['info'])
        
        if title:
            formatted_message = f"[bold]{title}[/bold]\n{message}"
        else:
            formatted_message = message
        
        panel = Panel(
            f"{icon} {formatted_message}",
            style=color,
            padding=(0, 1)
        )
        console.print(panel)
    
    @staticmethod
    def show_loading(message: str = "Procesando..."):
        """
        Muestra un indicador de carga
        
        Args:
            message: Mensaje de carga
        """
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            task = progress.add_task(f"{HacktinverUI.ICONS['loading']} {message}", total=None)
            return progress, task
    
    @staticmethod
    def create_data_table(
        data: List[Dict],
        title: str,
        columns_config: Dict[str, Dict] = None,
        max_rows: int = None
    ) -> Table:
        """
        Crea una tabla de datos con formato consistente
        
        Args:
            data: Lista de diccionarios con los datos
            title: Título de la tabla
            columns_config: Configuración de columnas {nombre: {style, width, etc}}
            max_rows: Número máximo de filas a mostrar
        
        Returns:
            Tabla Rich formateada
        """
        if not data:
            table = Table(title=title)
            table.add_column("Mensaje", style=HacktinverUI.COLORS['warning'])
            table.add_row("No hay datos para mostrar")
            return table
        
        table = Table(title=title, show_header=True, header_style="bold blue")
        
        # Agregar columnas
        first_row = data[0]
        for col_name in first_row.keys():
            col_config = columns_config.get(col_name, {}) if columns_config else {}
            style = col_config.get('style', 'white')
            width = col_config.get('width', None)
            
            table.add_column(
                col_name,
                style=style,
                width=width,
                no_wrap=col_config.get('no_wrap', False)
            )
        
        # Agregar filas
        rows_to_show = data[:max_rows] if max_rows else data
        for row in rows_to_show:
            table.add_row(*[str(value) for value in row.values()])
        
        if max_rows and len(data) > max_rows:
            table.caption = f"Mostrando {max_rows} de {len(data)} resultados"
        
        return table
    
    @staticmethod
    def show_summary_cards(data: Dict[str, Any], title: str = "Resumen"):
        """
        Muestra tarjetas de resumen con métricas clave
        
        Args:
            data: Diccionario con métricas {nombre: valor}
            title: Título del resumen
        """
        console.print(f"\n[{HacktinverUI.COLORS['primary']}]{HacktinverUI.ICONS['chart']} {title}[/{HacktinverUI.COLORS['primary']}]")
        
        cards = []
        for key, value in data.items():
            # Formatear valor según tipo
            if isinstance(value, float):
                if abs(value) >= 1000000:
                    formatted_value = f"${value/1000000:.1f}M"
                elif abs(value) >= 1000:
                    formatted_value = f"${value/1000:.1f}K"
                elif key.lower().endswith(('%', 'pct', 'percent')):
                    formatted_value = f"{value:.2f}%"
                else:
                    formatted_value = f"{value:.2f}"
            else:
                formatted_value = str(value)
            
            card_content = f"[bold white]{key}[/bold white]\n[{HacktinverUI.COLORS['accent']}]{formatted_value}[/{HacktinverUI.COLORS['accent']}]"
            cards.append(Panel(card_content, width=20, padding=(0, 1)))
        
        # Mostrar en columnas
        console.print(Columns(cards, equal=True, expand=True))
        console.print()
    
    @staticmethod
    def show_progress_bar(items: List[Any], description: str = "Procesando"):
        """
        Muestra una barra de progreso para procesar items
        
        Args:
            items: Lista de items a procesar
            description: Descripción del proceso
        
        Returns:
            Generator que yield cada item con progreso
        """
        with Progress(console=console) as progress:
            task = progress.add_task(f"[cyan]{description}...", total=len(items))
            
            for item in items:
                yield item
                progress.advance(task)
    
    @staticmethod
    def confirm_action(message: str, default: bool = False) -> bool:
        """
        Solicita confirmación del usuario
        
        Args:
            message: Mensaje de confirmación
            default: Valor por defecto
        
        Returns:
            True si el usuario confirma
        """
        return Confirm.ask(
            f"[{HacktinverUI.COLORS['warning']}]{HacktinverUI.ICONS['warning']} {message}[/{HacktinverUI.COLORS['warning']}]",
            default=default
        )
    
    @staticmethod
    def wait_for_user(message: str = "Presiona Enter para continuar..."):
        """
        Espera a que el usuario presione Enter
        
        Args:
            message: Mensaje a mostrar
        """
        Prompt.ask(f"[{HacktinverUI.COLORS['info']}]{message}[/{HacktinverUI.COLORS['info']}]", default="")
    
    @staticmethod
    def show_footer():
        """Muestra un footer consistente"""
        console.print()
        console.print(Rule(style="dim"))
        footer_text = Text()
        footer_text.append("Hacktinver v2.0", style="dim blue")
        footer_text.append(" | ", style="dim")
        footer_text.append("Herramienta de Análisis de Inversiones", style="dim")
        footer_text.append(" | ", style="dim")
        footer_text.append("🎯 Concurso Actinver 2024", style="dim yellow")
        
        console.print(Align.center(footer_text))


class InputValidator:
    """
    Validadores comunes para inputs de usuario
    """
    
    @staticmethod
    def positive_number(value: Union[int, float]) -> bool:
        """Valida que el número sea positivo"""
        return value > 0
    
    @staticmethod
    def percentage(value: float) -> bool:
        """Valida que sea un porcentaje válido (0-100)"""
        return 0 <= value <= 100
    
    @staticmethod
    def risk_percentage(value: float) -> bool:
        """Valida porcentaje de riesgo (0.1-10)"""
        return 0.1 <= value <= 10
    
    @staticmethod
    def ticker_list(value: str) -> bool:
        """Valida lista de tickers separados por comas"""
        if not value.strip():
            return True  # Permitir vacío para usar defaults
        
        tickers = [t.strip().upper() for t in value.split(',')]
        return all(len(t) >= 2 and len(t) <= 5 and t.isalpha() for t in tickers)
    
    @staticmethod
    def date_range(value: int) -> bool:
        """Valida rango de días (1-365)"""
        return 1 <= value <= 365


class DataFormatter:
    """
    Formateadores consistentes para diferentes tipos de datos
    """
    
    @staticmethod
    def currency(value: float, decimals: int = 2) -> str:
        """Formatea como moneda"""
        if abs(value) >= 1_000_000:
            return f"${value/1_000_000:.1f}M"
        elif abs(value) >= 1_000:
            return f"${value/1_000:.1f}K"
        else:
            return f"${value:,.{decimals}f}"
    
    @staticmethod
    def percentage(value: float, decimals: int = 2) -> str:
        """Formatea como porcentaje"""
        return f"{value:.{decimals}f}%"
    
    @staticmethod
    def number(value: Union[int, float], decimals: int = 2) -> str:
        """Formatea número con separadores de miles"""
        if isinstance(value, int):
            return f"{value:,}"
        else:
            return f"{value:,.{decimals}f}"
    
    @staticmethod
    def colored_percentage(value: float, decimals: int = 2) -> str:
        """Formatea porcentaje con color según signo"""
        formatted = DataFormatter.percentage(value, decimals)
        if value > 0:
            return f"[green]+{formatted}[/green]"
        elif value < 0:
            return f"[red]{formatted}[/red]"
        else:
            return formatted
    
    @staticmethod
    def colored_currency(value: float, decimals: int = 2) -> str:
        """Formatea moneda con color según signo"""
        formatted = DataFormatter.currency(value, decimals)
        if value > 0:
            return f"[green]+{formatted}[/green]"
        elif value < 0:
            return f"[red]{formatted}[/red]"
        else:
            return formatted
    
    @staticmethod
    def risk_level(value: float) -> str:
        """Formatea nivel de riesgo con color"""
        if value <= 5:
            return f"[green]BAJO ({value:.1f}%)[/green]"
        elif value <= 10:
            return f"[yellow]MEDIO ({value:.1f}%)[/yellow]"
        else:
            return f"[red]ALTO ({value:.1f}%)[/red]"
    
    @staticmethod
    def confidence_level(level: str) -> str:
        """Formatea nivel de confianza con color"""
        colors = {
            'ALTA': 'bold green',
            'MEDIA': 'yellow',
            'BAJA': 'red'
        }
        color = colors.get(level.upper(), 'white')
        return f"[{color}]{level}[/{color}]"