#!/usr/bin/env python3
"""
Hacktinver - Herramienta avanzada de análisis de inversiones
Punto de entrada principal de la aplicación

Autor: Hacktinver Team
Versión: 2.0 (Arquitectura Modular)
"""

import sys
import os

# Agregar el directorio actual al path para imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import load_environment_variables, initialize_logging
from services.notification import initialize_telegram
from ui.menu import display_main_menu


def main():
    """
    Punto de entrada principal de la aplicación Hacktinver.
    
    Inicializa la configuración, servicios externos y muestra el menú principal.
    """
    try:
        # Cargar configuración y variables de entorno
        load_environment_variables()
        
        # Inicializar logging
        logger = initialize_logging()
        logger.info("Iniciando Hacktinver v2.0 - Arquitectura Modular")
        
        # Inicializar servicios externos
        initialize_telegram()
        
        # Mostrar información de bienvenida
        print("🚀 Hacktinver v2.0 - Herramienta Avanzada de Análisis de Inversiones")
        print("📊 Arquitectura Modular | 🎯 Concurso Actinver 2024")
        print("-" * 70)
        
        # Iniciar el bucle del menú principal
        display_main_menu()
        
    except KeyboardInterrupt:
        print("\n\n👋 ¡Hasta luego! Gracias por usar Hacktinver")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error crítico al iniciar la aplicación: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()