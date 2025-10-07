"""
Configuración central de Hacktinver
Maneja variables de entorno, constantes globales y configuración de la aplicación
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# === CONSTANTES GLOBALES ===

# Colores para terminal
WARNING = "\033[93m"
WHITE = "\033[0m"
OKCYAN = "\033[96m"
OKGREEN = "\033[92m"
ERROR = "\033[91m"

# Configuración de trading
COMISION_PER_OPERATION = 0.0010
IVA_PER_OPERATION = 0.16
COST_PER_OPERATION = COMISION_PER_OPERATION + (COMISION_PER_OPERATION * IVA_PER_OPERATION)

# Directorios del proyecto
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"

# URLs y endpoints
ACTINVER_BASE_URL = "https://www.retoactinver.com"
ACTINVER_DATA_URL = f"{ACTINVER_BASE_URL}/archivos/datosReto.txt"
ACTINVER_LOGIN_URL = f"{ACTINVER_BASE_URL}/minisitio/reto/login.html"

# ETFs y activos por defecto
DEFAULT_LEVERAGED_ETFS = [
    'FAS', 'FAZ', 'PSQ', 'QLD', 'SOXL', 'SOXS', 'SPXL', 'SPXS', 
    'SQQQ', 'TECL', 'TECS', 'TNA', 'TQQQ', 'TZA', 'EDZ'
]

DEFAULT_NORMAL_ETFS = [
    'AAXJ', 'ACWI', 'BIL', 'BOTZ', 'DIA', 'EEM', 'EWZ', 'GDX', 'GLD', 
    'IAU', 'ICLN', 'INDA', 'IVV', 'KWEB', 'LIT', 'MCHI', 'NAFTRAC', 
    'QCLN', 'QQQ', 'SHV', 'SHY', 'SLV', 'SOXX', 'SPLG', 'SPY', 'TAN', 
    'TLT', 'USO', 'VEA', 'VGT', 'VNQ', 'VOO', 'VTI', 'VT', 'VWO', 
    'VYM', 'XLE', 'XLF', 'XLK', 'XLV'
]

DEFAULT_FAVORITE_STOCKS = [
    'TSLA.MX', 'BAC.MX', 'CEMEXCPO.MX', 'PE&OLES.MX', 'ORBIA.MX',
    'AMZN.MX', 'COST.MX', 'AMD.MX', 'FCX.MX', 'AAPL.MX', 'MSFT.MX'
]

# Pares para Pairs Trading
DEFAULT_PAIRS_TRADING = [
    ("SOXL", "TECL"),  # Semiconductores 3x vs Tecnología 3x
    ("SPXL", "TQQQ"),  # S&P 500 3x vs NASDAQ 3x
    ("FAS", "XLF"),    # Financieros 3x vs Financieros 1x
    ("SOXL", "SOXX"),  # Semiconductores 3x vs Semiconductores 1x
    ("TECL", "XLK"),   # Tecnología 3x vs Tecnología 1x
    ("TNA", "IWM"),    # Russell 2000 3x vs Russell 2000 1x
    ("SPXS", "SH"),    # S&P 500 -3x vs S&P 500 -1x (inversos)
]

# === VARIABLES DE ENTORNO ===

def load_environment_variables():
    """
    Carga las variables de entorno desde el archivo .env
    """
    load_dotenv()
    
    # Crear directorios necesarios
    DATA_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)
    
    return True


def get_env_variable(var_name: str, default_value: str = None, required: bool = False) -> str:
    """
    Obtiene una variable de entorno con manejo de errores
    
    Args:
        var_name: Nombre de la variable de entorno
        default_value: Valor por defecto si no existe
        required: Si es True, lanza excepción si no existe
    
    Returns:
        Valor de la variable de entorno
    """
    value = os.getenv(var_name, default_value)
    
    if required and not value:
        raise ValueError(f"Variable de entorno requerida no encontrada: {var_name}")
    
    return value


def get_actinver_users() -> list[dict]:
    """
    Obtiene las credenciales de usuarios de Actinver desde variables de entorno
    
    Returns:
        Lista de diccionarios con credenciales de usuario
    """
    users = []
    
    # Usuario desde variables de entorno
    env_user = get_env_variable("ACTINVER_USER_EMAIL")
    env_pass = get_env_variable("ACTINVER_USER_PASSWORD")
    
    if env_user and env_pass:
        users.append({"usuario": env_user, "password": env_pass})
    
    # Usuarios por defecto (considera mover a .env para mayor seguridad)
    default_users = [
        {"usuario": "natalia.sofia.glz@gmail.com", "password": "Ntlasfa9#19"},
        {"usuario": "osvaldo.hdz.m@outlook.com", "password": "299792458.Light"},
    ]
    
    users.extend(default_users)
    return users


def get_telegram_config() -> dict:
    """
    Obtiene la configuración de Telegram
    
    Returns:
        Diccionario con configuración de Telegram
    """
    return {
        "bot_token": get_env_variable("TELEGRAM_BOT_TOKEN"),
        "chat_ids": get_env_variable("TELEGRAM_CHAT_IDS", "").split(",") if get_env_variable("TELEGRAM_CHAT_IDS") else []
    }


def get_api_keys() -> dict:
    """
    Obtiene las claves de API para servicios externos
    
    Returns:
        Diccionario con claves de API
    """
    return {
        "gemini_api_key": get_env_variable("GEMINI_API_KEY"),
        "alpha_vantage_key": get_env_variable("ALPHA_VANTAGE_API_KEY"),
        "finnhub_key": get_env_variable("FINNHUB_API_KEY")
    }


def initialize_logging() -> logging.Logger:
    """
    Inicializa el sistema de logging
    
    Returns:
        Logger configurado
    """
    # Configuración básica de logging
    log_format = "%(asctime)s %(levelname)s %(name)s - %(message)s"
    
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(LOGS_DIR / "hacktinver.log"),
            logging.StreamHandler()
        ]
    )
    
    # Crear logger específico para Hacktinver
    logger = logging.getLogger("hacktinver")
    
    # Configurar niveles para librerías externas
    logging.getLogger("yfinance").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("selenium").setLevel(logging.WARNING)
    
    return logger


def _redact_sensitive(value: str) -> str:
    """
    Redacta información sensible para logging seguro
    
    Args:
        value: Valor a redactar
    
    Returns:
        Valor redactado
    """
    if not isinstance(value, str):
        return value
    if len(value) > 12:
        return value[:4] + "***" + value[-4:]
    return "***" if value else value


def redact_sensitive_dict(d: dict) -> dict:
    """
    Redacta información sensible de un diccionario
    
    Args:
        d: Diccionario con información potencialmente sensible
    
    Returns:
        Diccionario con información sensible redactada
    """
    try:
        data = dict(d)
    except Exception:
        return {}
    
    keys_to_mask = {
        "TS016e21d6", "tokenApp", "tokenSession", "cxCveUsuario",
        "contacto", "celular", "email", "password"
    }
    
    for k in list(data.keys()):
        if k in keys_to_mask:
            data[k] = _redact_sensitive(str(data.get(k, "")))
    
    return data


# === CONFIGURACIÓN DE APLICACIÓN ===

class AppConfig:
    """
    Clase para manejar la configuración de la aplicación
    """
    
    def __init__(self):
        self.debug_mode = get_env_variable("DEBUG", "False").lower() == "true"
        self.max_concurrent_downloads = int(get_env_variable("MAX_CONCURRENT_DOWNLOADS", "5"))
        self.default_timeout = int(get_env_variable("DEFAULT_TIMEOUT", "30"))
        self.cache_enabled = get_env_variable("CACHE_ENABLED", "True").lower() == "true"
    
    def is_debug(self) -> bool:
        return self.debug_mode
    
    def get_timeout(self) -> int:
        return self.default_timeout


# Instancia global de configuración
app_config = AppConfig()