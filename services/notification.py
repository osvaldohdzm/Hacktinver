"""
Notification Service - Servicio de notificaciones
Maneja notificaciones por Telegram y otros canales
"""

import logging
from config import get_telegram_config

logger = logging.getLogger("hacktinver.notification")


def initialize_telegram():
    """
    Inicializa el servicio de Telegram
    """
    try:
        telegram_config = get_telegram_config()
        
        if telegram_config["bot_token"]:
            logger.info("Servicio de Telegram inicializado correctamente")
        else:
            logger.warning("Token de Telegram no configurado")
            
    except Exception as e:
        logger.error(f"Error inicializando Telegram: {e}")


def send_telegram_message(message: str):
    """
    Envía un mensaje por Telegram
    
    Args:
        message: Mensaje a enviar
    """
    # Implementación pendiente
    logger.info(f"Mensaje Telegram (pendiente): {message}")