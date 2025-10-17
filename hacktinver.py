"""

Test xpath and css
$x(".//header/")
SyntaxError: Failed to execute 'evaluate' on 'Document': The string './/header/' is not a valid XPath expression.

$$("header[id=]")
"""

# =================================================================================
# CONFIGURACIÓN GLOBAL DEL SISTEMA
# =================================================================================
# Límite máximo de inversión por ticker individual (configurable)
MAX_ALLOCATION_PER_TICKER = 400000  # $400,000 por ticker máximo

# Capital base según número de tickers
CAPITAL_BASE_HIGH = 800000  # Para más de 3 tickers
CAPITAL_BASE_LOW = 600000   # Para 3 o menos tickers

def show_system_config():
    """
    Muestra la configuración actual del sistema de límites de capital
    """
    console.print("[bold blue]⚙️ CONFIGURACIÓN DEL SISTEMA DE LÍMITES[/bold blue]")
    console.print("=" * 60)
    console.print(f"[cyan]Límite máximo por ticker:[/cyan] ${MAX_ALLOCATION_PER_TICKER:,}")
    console.print(f"[cyan]Capital base (>3 tickers):[/cyan] ${CAPITAL_BASE_HIGH:,}")
    console.print(f"[cyan]Capital base (≤3 tickers):[/cyan] ${CAPITAL_BASE_LOW:,}")
    console.print("=" * 60)
    console.print("[yellow]💡 Esta configuración se aplica automáticamente a todas las funciones de análisis cuantitativo[/yellow]")

def update_system_config(max_per_ticker=None, capital_high=None, capital_low=None):
    """
    Actualiza la configuración del sistema de límites de capital
    
    Args:
        max_per_ticker: Nuevo límite máximo por ticker
        capital_high: Nuevo capital base para >3 tickers
        capital_low: Nuevo capital base para ≤3 tickers
    """
    global MAX_ALLOCATION_PER_TICKER, CAPITAL_BASE_HIGH, CAPITAL_BASE_LOW
    
    if max_per_ticker is not None:
        MAX_ALLOCATION_PER_TICKER = max_per_ticker
    if capital_high is not None:
        CAPITAL_BASE_HIGH = capital_high
    if capital_low is not None:
        CAPITAL_BASE_LOW = capital_low
    
    console.print("[bold green]✅ Configuración actualizada exitosamente[/bold green]")
    show_system_config()

# =================================================================================

# Standard Library Imports
import warnings
import logging
import time
import threading
import sys
import os
import os.path
import math
import csv
import argparse
from decimal import Decimal
from datetime import timedelta, datetime
from functools import lru_cache
from pathlib import Path
from urllib.request import urlopen, Request
from re import sub
import subprocess
from datetime import datetime
# import keyboard
# from pynput import keyboard


# Third-Party Imports
import requests
from prompt_toolkit import prompt
import json
import random
import yfinance as yf
import schedule
import pdfkit
import seaborn as sns
from retrying import retry
# Reduce noisy yfinance logs (HTTP 404 spam). We'll report our own concise status.
try:
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
except Exception:
    pass

import scipy.optimize as sco
import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import html5lib
import emoji
import telegram
import telebot
from yahoo_fin import stock_info as si
from yahoo_earnings_calendar import YahooEarningsCalendar
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt
from rich.table import Table
from rich import print
from pandas_datareader import data as web
from pandas.plotting import scatter_matrix
from pandas import Series, DataFrame
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import style
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import SMAIndicator, MACD
from ta.volatility import BollingerBands
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dotenv import load_dotenv
try:
    from pypfopt import risk_models, expected_returns, EfficientFrontier
except Exception:
    risk_models = expected_returns = EfficientFrontier = None
    logging.getLogger("ractinver").warning(
        "PyPortfolioOpt no disponible. Algunas funciones de optimización estarán deshabilitadas."
    )
# Prueba de importación del ADF; requerido para cointegración/estacionariedad en pairs trading
try:
    from statsmodels.tsa.stattools import adfuller
except Exception:
    adfuller = None  # Será validado en tiempo de ejecución donde se use
    logging.getLogger("ractinver").warning(
        "statsmodels no disponible. Instala con 'pip install statsmodels' para habilitar pruebas ADF."
    )
from selenium import webdriver
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support.wait import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, LinearRegression
from sklearn import preprocessing, svm
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import os
from platform import system

# Importar getch según el sistema operativo
if system() == 'Windows':
    from msvcrt import getch
else:
    try:
        from getch import getch
    except ImportError:
        # Fallback para sistemas sin getch
        def getch():
            import sys, tty, termios
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(sys.stdin.fileno())
                ch = sys.stdin.read(1)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            return ch

# Crear un evento para detener el hilo si es necesario
stop_event = threading.Event()

# Iniciar el scheduler en un hilo separado
scheduler_thread = None
scheduled_tasks = []

load_dotenv()

# Basic logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("ractinver")

console = Console()

warnings.simplefilter(action="ignore", category=FutureWarning)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if TELEGRAM_BOT_TOKEN:
    bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
    tb = telebot.TeleBot(TELEGRAM_BOT_TOKEN)
else:
    bot = None
    tb = None
    logger.warning("TELEGRAM_BOT_TOKEN no configurado. Funciones de Telegram deshabilitadas.")

parser = argparse.ArgumentParser(description="Trading tools actinver")
parser.add_argument("-cmd", help="file JSON name")
parser.add_argument("-tid", help="telegram ids to send messages", type=str)
args = vars(parser.parse_args())
command = args["cmd"]

print(args)

telegram_chat_ids = []
if args["tid"] is not None:
    args["tid"] = [s.strip() for s in args["tid"].split(",")]


telegram_chat_ids = args["tid"]
print(telegram_chat_ids)

pd.options.mode.chained_assignment = None  # default='warn'

WARNING = "\033[93m"
WHITE = "\033[0m"
OKCYAN = "\033[96m"
OKGREEN = "\033[92m"
ERROR = "\033[91m"


comision_per_operation = 0.0010
iva_per_operation = 0.16
cost_per_operation = (comision_per_operation) + (
    comision_per_operation * iva_per_operation
)
dn = os.path.dirname(os.path.realpath(__file__))




def normalize_ticker_to_mx(ticker: str) -> str:
    """
    Normaliza cualquier ticker para usar SIEMPRE la versión mexicana (.MX)
    Maneja casos: SOXL, SOXL.MX, SOXL.mx, soxl.mx -> SOXL.MX
    Evita duplicaciones como .MX.MX
    """
    if not ticker:
        return ""

    # Mapa de alias/normalizaciones para tickers MX y nombres frecuentes del reto
    # Clave sin sufijo; valor con sufijo .MX
    MX_ALIAS_MAP = {
        # Ejemplos de alias corporativos vs. Yahoo
        "ALFA": "ALFAA.MX",
        "BIMBO": "BIMBOA.MX",
        "CEMEX": "CEMEXCPO.MX",
        "GMEXICO": "GMEXICOB.MX",
        "PE&OLES": "PE&OLES.MX",
        "BOLSA": "BOLSAA.MX",
        "LAB": "LABB.MX",
        "KIMBER": "KIMBERA.MX",  # común en BMV
        "LIVEPOL": "LIVEPOLC-1.MX",
        "TLEVISA": "TLEVISACPO.MX",
        "WALMEX": "WALMEX.MX",
        "FEMSA": "FEMSAUBD.MX",
        # Variantes 1 vs sin sufijo
        "OXY1": "OXY.MX",
        "CPE": "CPE.MX",
        # ETFs locales comunes
        "NAFTRAC": "NAFTRAC.MX",
    }

    # Instrumentos de casa Actinver que no existen en Yahoo (omitir para evitar 404)
    ACTINVER_HOUSE_INSTRUMENTS = {
        "ACTI500", "ACTICRE", "ACTICOB", "ACTDUAL", "ACTIREN", "ACTIMED",
        "ACTIGOB", "ACTIG+", "ACTIG+2", "ACTIVAR", "ACTIPLU", "DIGITAL",
        "ESFERA", "ESCALA", "DINAMO", "MAYA", "MAXIMO", "SALUD", "ROBOTIK",
        "OPORT1"
    }

    # Limpiar y convertir a mayúsculas
    raw = ticker.strip()
    t = raw.upper()

    # Si viene ya con .MX exacto, devolverlo tal cual
    if t.endswith(".MX"):
        return t

    # Quitar sufijo .mx en minúscula
    if t.endswith(".mx"):
        t = t[:-3]

    base = t

    # Si ya viene con un sufijo como .B o .C, preferimos alias si existe, si no agregamos .MX
    if "." in base:
        symbol = base.split(".")[0]
    else:
        symbol = base

    # Si es instrumento de casa Actinver, devolver marcador especial para omitir
    if symbol in ACTINVER_HOUSE_INSTRUMENTS:
        return "__SKIP_ACTINVER__"

    # Resolver alias conocidos
    if symbol in MX_ALIAS_MAP:
        return MX_ALIAS_MAP[symbol]

    # Por defecto, agregar .MX
    return f"{symbol}.MX"

def _redact_sensitive(value: str) -> str:
    if not isinstance(value, str):
        return value
    if len(value) > 12:
        return value[:4] + "***" + value[-4:]
    return "***" if value else value


def redact_sensitive_dict(d: dict) -> dict:
    try:
        data = dict(d)
    except Exception:
        return {}
    keys_to_mask = {
        "TS016e21d6",
        "tokenApp",
        "tokenSession",
        "cxCveUsuario",
        "contacto",
        "celular",
        "email",
    }
    for k in list(data.keys()):
        if k in keys_to_mask:
            data[k] = _redact_sensitive(str(data.get(k, "")))
    return data

def generate_gemini_text(prompt: str) -> str:
    """Generate text using Google GenAI SDK. Requires GEMINI_API_KEY env var.
    Returns empty string on error.
    """
    try:
        from google import genai  # type: ignore
    except Exception as import_error:
        logger.error("google-genai SDK not installed: %s", import_error)
        return ""

    api_key = os.getenv("GEMINI_API_KEY")
    try:
        client = genai.Client(api_key=api_key) if api_key else genai.Client()
        resp = client.models.generate_content(
            model="gemini-2.5-flash", contents=prompt
        )
        return getattr(resp, "text", "") or ""
    except Exception as e:
        logger.error("Gemini API call failed: %s", e)
        return ""

def get_actinver_users() -> list[dict]:
    users: list[dict] = []
    env_user = os.getenv("ACTINVER_USER_EMAIL")
    env_pass = os.getenv("ACTINVER_USER_PASSWORD")
    if env_user and env_pass:
        users.append({"usuario": env_user, "password": env_pass})
    # Central default list (consider moving to .env or secure store)
    defaults = [
        {"usuario": "natalia.sofia.glz@gmail.com", "password": "Ntlasfa9#19"},
        {"usuario": "osvaldo.hdz.m@outlook.com", "password": "299792458.Light"},
    ]
    users.extend(defaults)
    return users

def _pick_price_series(df: pd.DataFrame) -> pd.Series:
    """Return a 1D price series preferring 'Adj Close' else 'Close'."""
    if df is None or df.empty:
        return pd.Series(dtype=float)
    if "Adj Close" in df.columns:
        s = df["Adj Close"]
    else:
        s = df.get("Close", pd.Series(dtype=float))
    # ensure 1D Series
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return s



def download_prices_mx_only(ticker: str, start: str = None, end: str = None, period: str = None, progress: bool = False) -> pd.Series:
    """
    Descarga precios EXCLUSIVAMENTE usando tickers mexicanos (.MX)
    Normaliza automáticamente cualquier ticker a su versión .MX
    """
    # Normalizar ticker a versión mexicana
    mx_ticker = normalize_ticker_to_mx(ticker)
    
    try:
        if start or end:
            df = yf.download(mx_ticker, start=start, end=end, interval="1d", progress=progress)
        else:
            df = yf.download(mx_ticker, period=period or "6mo", progress=progress)
        
        s = _pick_price_series(df)
        if not s.empty:
            return s.rename(ticker)  # Renombrar con el ticker original para consistencia
            
    except Exception as e:
        console.print(f"[red]Error descargando {mx_ticker}: {e}[/red]")
    
    return pd.Series(dtype=float, name=ticker)


def get_usd_to_mxn_rate():
    """
    Obtiene el tipo de cambio USD a MXN usando múltiples APIs públicas
    """
    # Lista de APIs de tipo de cambio para probar
    exchange_apis = [
        {
            'name': 'ExchangeRate-API',
            'url': 'https://api.exchangerate-api.com/v4/latest/USD',
            'parser': lambda data: data['rates'].get('MXN', None)
        },
        {
            'name': 'Fixer.io',
            'url': 'https://api.fixer.io/latest?base=USD&symbols=MXN',
            'parser': lambda data: data.get('rates', {}).get('MXN', None)
        },
        {
            'name': 'CurrencyAPI',
            'url': 'https://api.currencyapi.com/v3/latest?apikey=free&currencies=MXN&base_currency=USD',
            'parser': lambda data: data.get('data', {}).get('MXN', {}).get('value', None)
        }
    ]
    
    # Intentar con APIs de tipo de cambio
    for api in exchange_apis:
        try:
            import requests
            response = requests.get(api['url'], timeout=5)
            if response.status_code == 200:
                data = response.json()
                rate = api['parser'](data)
                if rate and isinstance(rate, (int, float)) and 15 <= rate <= 25:  # Validar rango razonable
                    console.print(f"[green]💱 Tipo de cambio obtenido de {api['name']}: {rate:.4f}[/green]")
                    return float(rate)
        except Exception as e:
            console.print(f"[yellow]⚠️ {api['name']} falló: {str(e)[:30]}[/yellow]")
            continue
    
    try:
        # Fallback: usar Yahoo Finance para obtener USDMXN
        console.print("[yellow]💱 Intentando Yahoo Finance para tipo de cambio...[/yellow]")
        usd_mxn = yf.download("USDMXN=X", period="1d", progress=False)
        if not usd_mxn.empty and 'Close' in usd_mxn.columns:
            rate = float(usd_mxn['Close'].iloc[-1])
            if 15 <= rate <= 25:  # Validar rango razonable
                console.print(f"[green]💱 Tipo de cambio de Yahoo Finance: {rate:.4f}[/green]")
                return rate
    except Exception as e:
        console.print(f"[yellow]⚠️ Yahoo Finance tipo de cambio falló: {str(e)[:30]}[/yellow]")
    
    # Fallback final: tipo de cambio aproximado
    console.print("[yellow]⚠️ No se pudo obtener tipo de cambio, usando 18.50 MXN/USD[/yellow]")
    return 18.50


def get_stock_data_alternative_apis(ticker: str, period: str = "3mo") -> pd.DataFrame:
    """
    Intenta obtener datos de stock usando APIs alternativas cuando Yahoo Finance falla
    """
    console.print(f"[cyan]🔍 Buscando {ticker} en fuentes alternativas...[/cyan]")
    
    # 1. Intentar con Alpha Vantage (API gratuita)
    try:
        alpha_vantage_key = os.getenv("ALPHA_VANTAGE_API_KEY", "demo")  # demo key para pruebas
        if alpha_vantage_key and alpha_vantage_key != "demo":
            import requests
            url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&apikey={alpha_vantage_key}&outputsize=compact"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if "Time Series (Daily)" in data:
                    time_series = data["Time Series (Daily)"]
                    
                    # Convertir a DataFrame
                    df_data = []
                    for date_str, values in time_series.items():
                        df_data.append({
                            'Date': pd.to_datetime(date_str),
                            'Open': float(values['1. open']),
                            'High': float(values['2. high']),
                            'Low': float(values['3. low']),
                            'Close': float(values['4. close']),
                            'Volume': int(values['5. volume'])
                        })
                    
                    if df_data:
                        df = pd.DataFrame(df_data)
                        df.set_index('Date', inplace=True)
                        df.sort_index(inplace=True)
                        
                        # Filtrar por período
                        if len(df) > 60:  # Suficientes datos
                            console.print(f"[green]✅ {ticker} encontrado en Alpha Vantage[/green]")
                            return df
    except Exception as e:
        console.print(f"[yellow]⚠️ Alpha Vantage falló para {ticker}: {str(e)[:30]}[/yellow]")
    
    # 2. Intentar con Polygon.io (API gratuita limitada)
    try:
        polygon_key = os.getenv("POLYGON_API_KEY")
        if polygon_key:
            import requests
            from datetime import datetime, timedelta
            
            # Calcular fechas
            end_date = datetime.now()
            if period == "1mo":
                start_date = end_date - timedelta(days=30)
            elif period == "3mo":
                start_date = end_date - timedelta(days=90)
            else:
                start_date = end_date - timedelta(days=90)
            
            url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}?adjusted=true&sort=asc&apikey={polygon_key}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "OK" and "results" in data:
                    results = data["results"]
                    
                    # Convertir a DataFrame
                    df_data = []
                    for item in results:
                        df_data.append({
                            'Date': pd.to_datetime(item['t'], unit='ms'),
                            'Open': float(item['o']),
                            'High': float(item['h']),
                            'Low': float(item['l']),
                            'Close': float(item['c']),
                            'Volume': int(item['v'])
                        })
                    
                    if df_data:
                        df = pd.DataFrame(df_data)
                        df.set_index('Date', inplace=True)
                        df.sort_index(inplace=True)
                        
                        if len(df) > 20:  # Suficientes datos
                            console.print(f"[green]✅ {ticker} encontrado en Polygon.io[/green]")
                            return df
    except Exception as e:
        console.print(f"[yellow]⚠️ Polygon.io falló para {ticker}: {str(e)[:30]}[/yellow]")
    
    # 3. Intentar con IEX Cloud (API gratuita limitada)
    try:
        iex_token = os.getenv("IEX_CLOUD_TOKEN")
        if iex_token:
            import requests
            
            # Determinar rango para IEX
            if period == "1mo":
                range_param = "1m"
            elif period == "3mo":
                range_param = "3m"
            else:
                range_param = "3m"
            
            url = f"https://cloud.iexapis.com/stable/stock/{ticker}/chart/{range_param}?token={iex_token}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if data and isinstance(data, list):
                    # Convertir a DataFrame
                    df_data = []
                    for item in data:
                        if all(k in item for k in ['date', 'open', 'high', 'low', 'close', 'volume']):
                            df_data.append({
                                'Date': pd.to_datetime(item['date']),
                                'Open': float(item['open']) if item['open'] else 0,
                                'High': float(item['high']) if item['high'] else 0,
                                'Low': float(item['low']) if item['low'] else 0,
                                'Close': float(item['close']) if item['close'] else 0,
                                'Volume': int(item['volume']) if item['volume'] else 0
                            })
                    
                    if df_data:
                        df = pd.DataFrame(df_data)
                        df.set_index('Date', inplace=True)
                        df.sort_index(inplace=True)
                        
                        if len(df) > 20:  # Suficientes datos
                            console.print(f"[green]✅ {ticker} encontrado en IEX Cloud[/green]")
                            return df
    except Exception as e:
        console.print(f"[yellow]⚠️ IEX Cloud falló para {ticker}: {str(e)[:30]}[/yellow]")
    
    # 4. Intentar con Financial Modeling Prep (API gratuita limitada)
    try:
        fmp_key = os.getenv("FMP_API_KEY")
        if fmp_key:
            import requests
            
            url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?apikey={fmp_key}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if "historical" in data:
                    historical = data["historical"]
                    
                    # Convertir a DataFrame
                    df_data = []
                    for item in historical[:90]:  # Últimos 90 días
                        df_data.append({
                            'Date': pd.to_datetime(item['date']),
                            'Open': float(item['open']),
                            'High': float(item['high']),
                            'Low': float(item['low']),
                            'Close': float(item['close']),
                            'Volume': int(item['volume'])
                        })
                    
                    if df_data:
                        df = pd.DataFrame(df_data)
                        df.set_index('Date', inplace=True)
                        df.sort_index(inplace=True)
                        
                        if len(df) > 20:  # Suficientes datos
                            console.print(f"[green]✅ {ticker} encontrado en Financial Modeling Prep[/green]")
                            return df
    except Exception as e:
        console.print(f"[yellow]⚠️ Financial Modeling Prep falló para {ticker}: {str(e)[:30]}[/yellow]")
    
    console.print(f"[red]❌ {ticker} no encontrado en ninguna fuente alternativa[/red]")
    return pd.DataFrame()


def download_multiple_mx_tickers(tickers: list, period: str = "3mo", progress: bool = False) -> dict:
    """
    Descarga múltiples tickers con fallback automático:
    1. Intenta primero con versión mexicana (.MX)
    2. Si no encuentra, intenta con versión USA y convierte USD a MXN
    3. Avisa claramente cuando usa datos USA convertidos
    """
    if not tickers:
        return {}
    
    # Normalizar todos los tickers a versión .MX, filtrando los que marcamos para omitir
    normalized = [normalize_ticker_to_mx(t) for t in tickers]
    filtered_pairs = [(orig, mx) for orig, mx in zip(tickers, normalized) if mx != "__SKIP_ACTINVER__"]
    skipped = [orig for orig, mx in zip(tickers, normalized) if mx == "__SKIP_ACTINVER__"]
    if skipped:
        console.print(f"[yellow]⏭️ Omitiendo instrumentos no disponibles en Yahoo: {', '.join(skipped)}[/yellow]")
    if not filtered_pairs:
        return {}
    mx_tickers = [mx for _, mx in filtered_pairs]
    original_order = [orig for orig, _ in filtered_pairs]
    
    data = {}
    failed_mx_tickers = []
    usd_to_mxn_rate = None
    
    try:
        # Intentar descarga batch con tickers .MX
        console.print(f"[cyan]📊 Descargando {len(mx_tickers)} símbolos mexicanos (.MX)...[/cyan]")
        
        # Descargar en lotes de 10 para evitar timeouts
        batch_size = 10
        for i in range(0, len(mx_tickers), batch_size):
            batch_tickers = mx_tickers[i:i+batch_size]
            batch_originals = original_order[i:i+batch_size]
            
            try:
                # Descarga batch
                batch_data = yf.download(batch_tickers, period=period, group_by='ticker', progress=progress)
                
                if not batch_data.empty:
                    for mx_ticker, original_ticker in zip(batch_tickers, batch_originals):
                        try:
                            if len(batch_tickers) == 1:
                                # Solo un ticker en el batch
                                df = batch_data
                            else:
                                # Múltiples tickers - verificar si existe en las columnas
                                if hasattr(batch_data.columns, 'levels') and mx_ticker in batch_data.columns.levels[0]:
                                    df = batch_data[mx_ticker]
                                else:
                                    df = pd.DataFrame()
                            
                            # Verificar que realmente tenemos datos válidos
                            if (not df.empty and len(df) > 5 and 'Close' in df.columns):
                                # Verificar que hay al menos algunos valores no-NaN
                                close_series = df['Close']
                                valid_prices = close_series.dropna()
                                
                                if len(valid_prices) > 0:
                                    data[original_ticker] = df
                                    data[original_ticker].attrs = {'currency': 'MXN', 'source': 'MX'}
                                    console.print(f"[green]✅ {original_ticker} ({mx_ticker})[/green]", end=" ")
                                else:
                                    failed_mx_tickers.append((original_ticker, mx_ticker))
                                    console.print(f"[yellow]⚠️ {original_ticker} (datos vacíos)[/yellow]", end=" ")
                            else:
                                failed_mx_tickers.append((original_ticker, mx_ticker))
                                console.print(f"[yellow]⚠️ {original_ticker} (sin datos)[/yellow]", end=" ")
                                
                        except Exception as e:
                            failed_mx_tickers.append((original_ticker, mx_ticker))
                            console.print(f"[red]❌ {original_ticker} (error: {str(e)[:20]})[/red]", end=" ")
                            
            except Exception as e:
                # Si falla el batch, intentar individual
                console.print(f"\n[yellow]⚠️ Batch falló, intentando individual...[/yellow]")
                for mx_ticker, original_ticker in zip(batch_tickers, batch_originals):
                    try:
                        df = yf.download(mx_ticker, period=period, progress=False)
                        if not df.empty and len(df) > 0:
                            data[original_ticker] = df
                            data[original_ticker].attrs = {'currency': 'MXN', 'source': 'MX'}
                            console.print(f"[green]✅ {original_ticker}[/green]", end=" ")
                        else:
                            failed_mx_tickers.append((original_ticker, mx_ticker))
                            console.print(f"[yellow]⚠️ {original_ticker}[/yellow]", end=" ")
                    except:
                        failed_mx_tickers.append((original_ticker, mx_ticker))
                        console.print(f"[red]❌ {original_ticker}[/red]", end=" ")
        
        # Intentar fallback a versión USA para los que fallaron
        if failed_mx_tickers:
            console.print(f"\n[bold yellow]🔄 FALLBACK: Intentando versión USA para {len(failed_mx_tickers)} símbolos no encontrados en .MX[/bold yellow]")
            
            # Obtener tipo de cambio una sola vez
            usd_to_mxn_rate = get_usd_to_mxn_rate()
            console.print(f"[bold blue]💱 Tipo de cambio USD/MXN: {usd_to_mxn_rate:.4f} pesos por dólar[/bold blue]")
            
            usa_success_count = 0
            alternative_success_count = 0
            
            for original_ticker, mx_ticker in failed_mx_tickers:
                ticker_found = False
                
                # 1. Intentar primero con Yahoo Finance USA
                try:
                    usa_ticker = original_ticker.upper()
                    df = yf.download(usa_ticker, period=period, progress=False)
                    
                    if not df.empty and len(df) > 5 and 'Close' in df.columns:
                        close_series = df['Close']
                        valid_prices = close_series.dropna()
                        
                        if len(valid_prices) > 0:
                            # Convertir precios de USD a MXN
                            price_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close']
                            for col in price_columns:
                                if col in df.columns:
                                    df[col] = df[col] * usd_to_mxn_rate
                            
                            data[original_ticker] = df
                            data[original_ticker].attrs = {
                                'currency': 'MXN', 
                                'source': 'USA_YF', 
                                'exchange_rate': usd_to_mxn_rate,
                                'converted_from_usd': True
                            }
                            usa_success_count += 1
                            ticker_found = True
                            console.print(f"[bold blue]🔄 {original_ticker} (USA→MXN @ {usd_to_mxn_rate:.2f})[/bold blue]", end=" ")
                        
                except Exception as e:
                    pass  # Continuar con fuentes alternativas
                
                # 2. Si Yahoo Finance USA falló, intentar con APIs alternativas
                if not ticker_found:
                    try:
                        alt_df = get_stock_data_alternative_apis(original_ticker, period)
                        
                        if not alt_df.empty and len(alt_df) > 5 and 'Close' in alt_df.columns:
                            close_series = alt_df['Close']
                            valid_prices = close_series.dropna()
                            
                            if len(valid_prices) > 0:
                                # Convertir precios de USD a MXN (asumiendo que las APIs alternativas devuelven USD)
                                price_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close']
                                for col in price_columns:
                                    if col in alt_df.columns:
                                        alt_df[col] = alt_df[col] * usd_to_mxn_rate
                                
                                data[original_ticker] = alt_df
                                data[original_ticker].attrs = {
                                    'currency': 'MXN', 
                                    'source': 'ALTERNATIVE_API', 
                                    'exchange_rate': usd_to_mxn_rate,
                                    'converted_from_usd': True
                                }
                                alternative_success_count += 1
                                ticker_found = True
                                console.print(f"[bold magenta]🔍 {original_ticker} (API Alt→MXN @ {usd_to_mxn_rate:.2f})[/bold magenta]", end=" ")
                    
                    except Exception as e:
                        pass  # Continuar con el siguiente ticker
                
                # 3. Si nada funcionó, reportar como no encontrado
                if not ticker_found:
                    console.print(f"[red]❌ {original_ticker} (no encontrado)[/red]", end=" ")
            
            if usa_success_count > 0:
                console.print(f"\n[bold blue]🎯 ÉXITO Yahoo Finance USA: {usa_success_count} símbolos obtenidos y convertidos a pesos mexicanos[/bold blue]")
            
            if alternative_success_count > 0:
                console.print(f"\n[bold magenta]🎯 ÉXITO APIs Alternativas: {alternative_success_count} símbolos obtenidos y convertidos a pesos mexicanos[/bold magenta]")
        
        console.print(f"\n[bold cyan]📊 RESUMEN FINAL: {len(data)}/{len(tickers)} símbolos descargados exitosamente[/bold cyan]")
        
        # Mostrar resumen detallado de fuentes
        mx_count = sum(1 for df in data.values() if df.attrs.get('source') == 'MX')
        usa_yf_count = sum(1 for df in data.values() if df.attrs.get('source') == 'USA_YF')
        alt_api_count = sum(1 for df in data.values() if df.attrs.get('source') == 'ALTERNATIVE_API')
        
        if mx_count > 0:
            console.print(f"[green]🇲🇽 Datos mexicanos (.MX): {mx_count} símbolos - Precios en pesos mexicanos[/green]")
        if usa_yf_count > 0:
            console.print(f"[blue]🇺🇸 Datos Yahoo Finance USA convertidos: {usa_yf_count} símbolos - Convertidos de USD a MXN @ {usd_to_mxn_rate:.4f}[/blue]")
        if alt_api_count > 0:
            console.print(f"[magenta]🔍 Datos APIs alternativas convertidos: {alt_api_count} símbolos - Convertidos de USD a MXN @ {usd_to_mxn_rate:.4f}[/magenta]")
        
        total_converted = usa_yf_count + alt_api_count
        if total_converted > 0:
            console.print(f"[yellow]⚠️  NOTA: {total_converted} símbolos fueron convertidos automáticamente a pesos mexicanos desde fuentes USA[/yellow]")
        
    except Exception as e:
        console.print(f"[red]Error en descarga múltiple: {e}[/red]")
    
    return data


# Mantener función original para compatibilidad, pero redirigir a versión MX
def download_prices_any_listing(ticker: str, start: str = None, end: str = None, period: str = None) -> pd.Series:
    """
    DEPRECATED: Usa download_prices_mx_only en su lugar
    Mantenido para compatibilidad con código existente
    """
    return download_prices_mx_only(ticker, start, end, period)


def clear_screen():
    try:
        if os.name == "posix":
            os.system("clear")
        else:
            os.system("cls")
    except Exception:
        pass


def configure_chrome_headless_driver_no_profile():
    # Configuración de las opciones del navegador
    options = ChromeOptions()
    # options.add_argument("--headless=old")
    options.add_argument("--allow-running-insecure-content")
    options.add_argument("--disable-breakpad")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-infobars")
    options.add_argument("--disable-web-security")
    options.add_argument("--ignore-certificate-errors-spki-list")
    options.add_argument("--ignore-certificate-errors")
    options.add_argument("--ignore-ssl-errors")
    options.add_argument("--log-level=3")
    options.add_argument("--mute-audio")
    options.add_argument("--no-sandbox")
    options.add_argument("--no-zygote")
    options.add_argument("--allow-insecure-localhost")
    options.add_argument("--ignore-certificate-errors")

    try:
        # Configurar el servicio del WebDriver, descargándolo automáticamente si es necesario
        service = Service(ChromeDriverManager().install())

        # Crear una instancia del WebDriver con las opciones y el servicio especificados
        driver = webdriver.Chrome(service=service, options=options)

        # Navegar a una página inicial, como una página en blanco
        driver.minimize_window()
        driver.get("about:blank")
        print("Controlador de Chrome iniciado exitosamente.")
        return driver

    except Exception as e:
        print(f"Error al iniciar el controlador de Chrome: {e}")
        return None


def configure_chrome_driver_no_profile():
    # Configuración de las opciones del navegador
    options = Options()
    options.headless = (
        False  # Cambiar a True si no deseas abrir la ventana del navegador
    )
    options.add_argument(
        "--ignore-certificate-errors"
    )  # Ignorar errores de certificado SSL
    options.add_argument(
        "--allow-insecure-localhost"
    )  # Permitir conexiones inseguras en localhost
    options.add_argument("--disable-web-security")  # Desactivar la seguridad web
    options.add_argument("start-maximized")  # Iniciar la ventana maximizada
    options.add_argument("--disable-gpu")
    options.add_argument("--mute-audio")
    options.add_argument("--ignore-certificate-errors")
    options.add_argument("--ignore-ssl-errors")
    options.add_argument("--disable-infobars")
    options.add_argument("--ignore-certificate-errors-spki-list")
    options.add_argument("--no-sandbox")
    options.add_argument("--no-zygote")
    options.add_argument("--log-level=3")
    options.add_argument("--allow-running-insecure-content")
    options.add_argument("--disable-web-security")
    options.add_argument("--disable-features=VizDisplayCompositor")
    options.add_argument("--disable-breakpad")

    try:
        # Configurar el servicio del WebDriver, descargándolo automáticamente si es necesario
        service = Service(ChromeDriverManager().install())

        # Crear una instancia del WebDriver con las opciones y el servicio especificados
        driver = webdriver.Chrome(service=service, options=options)

        # Navegar a una página inicial, como una página en blanco
        driver.get("about:blank")

        # Maximizar la ventana (opcional, ya que se establece en las opciones)
        driver.maximize_window()

        print("Controlador de Chrome iniciado exitosamente.")
        return driver

    except Exception as e:
        print(f"Error al iniciar el controlador de Chrome: {e}")
        return None


def configure_firefox_driver_no_profile():
    options = Options()
    # Sin abrirnavegador
    options.headless = False
    options.add_argument("start-maximized")
    driver = webdriver.Firefox(
        options=options, executable_path=os.path.join(dn, "geckodriver.exe")
    )
    driver.get("about:home")
    driver.maximize_window()
    return driver


def configure_firefox_driver_with_profile():
    options = Options()
    # Sin abrirnavegador
    options.headless = False
    options.add_argument("start-maximized")
    profile = webdriver.FirefoxProfile(
        r"C:\Users\osval\AppData\Roaming\Mozilla\Firefox\Profiles\5uyx2cbw.default-release"
    )
    driver = webdriver.Firefox(
        firefox_profile=profile,
        options=options,
        executable_path=os.path.join(dn, "geckodriver.exe"),
    )
    driver.get("about:home")
    driver.maximize_window()
    return driver


def wait_for_window(timeout=2):
    time.sleep(round(timeout / 1000))
    wh_now = driver.window_handles
    wh_then = vars["window_handles"]
    if len(wh_now) > len(wh_then):
        return set(wh_now).difference(set(wh_then)).pop()


def logout_platform_actinver(driver):
    try:
        print(
            "\t" + WARNING + "Conectando con la plataforma del reto actinver..." + WHITE
        )
        driver.get("https://www.retoactinver.com/RetoActinver/#/inicio")
        close_popoup_browser(driver)
        print("\t" + WARNING + "Cerrando sesión en plataforma del reto..." + WHITE)
        WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable(
                (
                    By.CSS_SELECTOR,
                    "mat-list-item.as:nth-child(11) > div:nth-child(1) > div:nth-child(2) > p:nth-child(2)",
                )
            )
        )
        driver.find_element(
            By.CSS_SELECTOR,
            "mat-list-item.as:nth-child(11) > div:nth-child(1) > div:nth-child(2) > p:nth-child(2)",
        ).click()
        print("\t" + WARNING + "Cierre de sesión exitoso! :)" + WHITE)
    except Exception as e:
        print(e)
        exit()


def close_popoup_browser(driver):
    try:
        WebDriverWait(driver, 15).until(
            EC.element_to_be_clickable(
                (
                    By.CSS_SELECTOR,
                    "#mat-dialog-1 > app-modal-explorador:nth-child(1) > mat-dialog-actions:nth-child(2) > div:nth-child(2) > button:nth-child(1)",
                )
            )
        )
        driver.find_element_by_css_selector(
            "#mat-dialog-1 > app-modal-explorador:nth-child(1) > mat-dialog-actions:nth-child(2) > div:nth-child(2) > button:nth-child(1)"
        ).click()
        driver.find_element_by_css_selector(
            "#mat-dialog-1 > app-modal-explorador:nth-child(1) > mat-dialog-actions:nth-child(2) > div:nth-child(2) > button:nth-child(1)"
        ).click()
    except Exception as e:
        print(e)

    try:
        WebDriverWait(driver, 15).until(
            EC.element_to_be_clickable((By.XPATH, '//*[@id="botonCerrar"]'))
        )
        driver.find_element_by_xpath('//*[@id="botonCerrar"]').click()
        driver.find_element_by_xpath('//*[@id="botonCerrar"]').click()
    except Exception as e:
        print(e)


def show_orders():
    try:
        print(
            "\t" + WARNING + "Conectando con la plataforma del reto actinver..." + WHITE
        )
        driver.get("https://www.retoactinver.com/RetoActinver/#/ordenes")

        close_popoup_browser()

        total_value = driver.find_element_by_css_selector(
            "div.text-left:nth-child(1) > p:nth-child(2)"
        ).get_attribute("innerHTML")
        power_value = driver.find_element_by_css_selector(
            "div.text-left:nth-child(2) > p:nth-child(2)"
        ).get_attribute("innerHTML")
        inv_value = driver.find_element_by_css_selector(
            "div.text-left:nth-child(3) > p:nth-child(2)"
        ).get_attribute("innerHTML")
        varia_percent_value = driver.find_element_by_css_selector(
            "div.col-md-12:nth-child(4) > p:nth-child(2)"
        ).get_attribute("innerHTML")
        varia_mount_value = driver.find_element_by_css_selector(
            "div.col-md-12:nth-child(5) > p:nth-child(2)"
        ).get_attribute("innerHTML")
        print("Valuación Total: " + total_value)
        print("Poder de compra: " + power_value)
        print("Inversiones: " + inv_value)
        print("Variación en porcentaje: " + varia_percent_value)
        print("Variación en pesos: " + varia_mount_value)

        current_power_value = float(sub(r"[^\d.]", "", power_value))
        print("Poder de compra actual: " + str(current_power_value))
    except Exception as e:
        print(e)
        exit()


def show_portfolio():
    try:
        print(
            "\t" + WARNING + "Conectando con la plataforma del reto actinver..." + WHITE
        )
        driver.get("https://www.retoactinver.com/RetoActinver/#/inicio")
        close_popoup_browser()

        total_value = driver.find_element_by_css_selector(
            "div.text-left:nth-child(1) > p:nth-child(2)"
        ).get_attribute("innerHTML")
        power_value = driver.find_element_by_css_selector(
            "div.text-left:nth-child(2) > p:nth-child(2)"
        ).get_attribute("innerHTML")
        inv_value = driver.find_element_by_css_selector(
            "div.text-left:nth-child(3) > p:nth-child(2)"
        ).get_attribute("innerHTML")
        varia_percent_value = driver.find_element_by_css_selector(
            "div.col-md-12:nth-child(4) > p:nth-child(2)"
        ).get_attribute("innerHTML")
        varia_mount_value = driver.find_element_by_css_selector(
            "div.col-md-12:nth-child(5) > p:nth-child(2)"
        ).get_attribute("innerHTML")
        current_power_value = float(sub(r"[^\d.]", "", power_value))

        print("Valuación Total: " + total_value)
        print("Poder de compra: " + str("${:,.2f}".format(current_power_value)))
        print("Inversiones: " + inv_value)
        print("Variación en porcentaje: " + varia_percent_value)
        print("Variación en pesos: " + varia_mount_value)

        driver.get("https://www.retoactinver.com/RetoActinver/#/portafolio")
        time.sleep(3)
        rows = driver.find_elements(
            By.XPATH,
            '//*[@id="tpm"]/app-portafolio/div/div[3]/mat-card/app-table/div/gt-column-settings/div/div/generic-table/table/tbody/tr/td/span[2]',
        )

        count = 1
        print("\nPosiciones actuales:")
        for x in rows:
            if count % 8 != 0:
                print(x.get_attribute("innerHTML").ljust(15), end="")
            elif count % 8 == 0:
                print("|")

            count = count + 1

    except Exception as e:
        print(e)


def check_exists_by_css_selector(css_selector):
    try:
        driver.find_element_by_css_selector(css_selector)
    except NoSuchElementException:
        return False
    return True


def buy_stocks():

    try:
        print(
            "\t" + WARNING + "Conectando con la plataforma del reto actinver..." + WHITE
        )
        driver.get("https://www.retoactinver.com/RetoActinver/#/inicio")

        close_popoup_browser()

        power_value = driver.find_element_by_css_selector(
            "div.text-left:nth-child(2) > p:nth-child(2)"
        ).get_attribute("innerHTML")
        current_power_value = float(sub(r"[^\d.]", "", power_value))
    except Exception as e:
        print(e)

    current_stock_buy = input("Escribe el símbolo del stock que quieres comprar >> ")
    if current_stock_buy:
        current_stock_buy = current_stock_buy.upper()
    else:
        Prompt.ask("[bold blue]Pulsa Enter para continuar...[/bold blue]")
        buy_stocks()

    try:
        print(
            "\t" + WARNING + "Conectando con la plataforma del reto actinver..." + WHITE
        )
        driver.get("https://www.retoactinver.com/RetoActinver/#/capitales")

        close_popoup_browser()

        WebDriverWait(driver, 3).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, ".ng-pristine"))
        )
        search_stock_input = driver.find_element_by_css_selector(".ng-pristine")
        search_stock_input.send_keys(current_stock_buy)

        if check_exists_by_css_selector(".gt-no-matching-results"):
            print(
                "\t"
                + ERROR
                + "No hay stock con ese simbolo en el listado del reto! Prueba con otro"
                + WHITE
            )
            pass
        else:

            WebDriverWait(driver, 50).until(
                EC.element_to_be_clickable(
                    (By.XPATH, "//*/generic-table/table/tbody/tr/td[4]/span[2]")
                )
            )
            price = driver.find_element_by_xpath(
                "//*/generic-table/table/tbody/tr/td[4]/span[2]"
            ).get_attribute("innerHTML")
            price = float(sub(r"[^\d.]", "", price))

            WebDriverWait(driver, 50).until(
                EC.element_to_be_clickable(
                    (
                        By.CSS_SELECTOR,
                        "tr.ng-star-inserted:nth-child(1) > td:nth-child(2) > span:nth-child(2)",
                    )
                )
            )
            driver.find_element_by_css_selector(
                "tr.ng-star-inserted:nth-child(1) > td:nth-child(2) > span:nth-child(2)"
            ).click()

            no_titles = round(
                (current_power_value * 0.25) / price
            )  # the percentage of portfolio start 0.25 then 0.30 then 0.5
            print(str(price))
            print(str(current_power_value))
            print(str(no_titles))
            no_titles = str(no_titles)

            selected_stock_name = driver.find_element_by_css_selector(
                ".NombreEmpresa"
            ).get_attribute("innerHTML")

            confirmation = input(
                "El Símbolo seleccionado es: "
                + selected_stock_name
                + ", deseas continuar? (y/n) >> "
            )

            if confirmation == "y":
                WebDriverWait(driver, 3).until(
                    EC.element_to_be_clickable(
                        (
                            By.CSS_SELECTOR,
                            "div.col-6:nth-child(1) > button:nth-child(1)",
                        )
                    )
                )
                buy_button = driver.find_element_by_css_selector(
                    "div.col-6:nth-child(1) > button:nth-child(1)"
                )
                buy_button.click()

                WebDriverWait(driver, 3).until(
                    EC.element_to_be_clickable(
                        (By.CSS_SELECTOR, "#mat-radio-9 > label:nth-child(1)")
                    )
                )
                driver.find_element_by_css_selector(
                    "#mat-radio-9 > label:nth-child(1)"
                ).click()

                WebDriverWait(driver, 6).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, "input.ng-invalid"))
                )
                driver.find_element_by_css_selector("input.ng-invalid").send_keys(
                    no_titles
                )

                confirm_button = driver.find_element_by_css_selector(
                    "div.col-md-6:nth-child(2) > button:nth-child(1)"
                )
                confirm_button.click()

                WebDriverWait(driver, 3).until(
                    EC.presence_of_element_located(
                        (
                            By.CSS_SELECTOR,
                            "table.w-100 > tbody:nth-child(1) > tr:nth-child(1) > td:nth-child(2)",
                        )
                    )
                )
                print("\n---Verifica la orden---")
                print(
                    "Emisora: "
                    + driver.find_element_by_css_selector(
                        "table.w-100 > tbody:nth-child(1) > tr:nth-child(1) > td:nth-child(2)"
                    ).get_attribute("innerHTML")
                )
                print(
                    "Operación: "
                    + driver.find_element_by_css_selector(
                        "table.w-100 > tbody:nth-child(1) > tr:nth-child(2) > td:nth-child(2)"
                    ).get_attribute("innerHTML")
                )
                print(
                    "Tipo Orden: "
                    + driver.find_element_by_css_selector(
                        "table.w-100 > tbody:nth-child(1) > tr:nth-child(3) > td:nth-child(2)"
                    ).get_attribute("innerHTML")
                )
                print(
                    "Títulos: "
                    + driver.find_element_by_css_selector(
                        "table.w-100 > tbody:nth-child(1) > tr:nth-child(4) > td:nth-child(2)"
                    ).get_attribute("innerHTML")
                )
                print(
                    "Precio a mercado: "
                    + driver.find_element_by_css_selector(
                        "table.w-100 > tbody:nth-child(1) > tr:nth-child(5) > td:nth-child(2)"
                    ).get_attribute("innerHTML")
                )
                print(
                    "Precio límite: "
                    + driver.find_element_by_css_selector(
                        "table.w-100 > tbody:nth-child(1) > tr:nth-child(6) > td:nth-child(2)"
                    ).get_attribute("innerHTML")
                )
                print(
                    "Comisión: "
                    + driver.find_element_by_css_selector(
                        "table.w-100 > tbody:nth-child(1) > tr:nth-child(7) > td:nth-child(2)"
                    ).get_attribute("innerHTML")
                )
                print(
                    "IVA: "
                    + driver.find_element_by_css_selector(
                        "table.w-100 > tbody:nth-child(1) > tr:nth-child(8) > td:nth-child(2)"
                    ).get_attribute("innerHTML")
                )
                print(
                    "Importe total: "
                    + driver.find_element_by_css_selector(
                        "table.w-100 > tbody:nth-child(1) > tr:nth-child(9) > td:nth-child(2)"
                    ).get_attribute("innerHTML")
                )
                print(
                    "Vigencía: "
                    + driver.find_element_by_css_selector(
                        "table.w-100 > tbody:nth-child(1) > tr:nth-child(10) > td:nth-child(2)"
                    ).get_attribute("innerHTML")
                )
                print(
                    "Fecha de captura: "
                    + driver.find_element_by_css_selector(
                        "table.w-100 > tbody:nth-child(1) > tr:nth-child(11) > td:nth-child(2)"
                    ).get_attribute("innerHTML")
                )
                print(
                    "Fecha de postura: "
                    + driver.find_element_by_css_selector(
                        "table.w-100 > tbody:nth-child(1) > tr:nth-child(12) > td:nth-child(2)"
                    ).get_attribute("innerHTML")
                )

                operation_confirm = input("Deseas confirmar la operación? (y/n) >> ")
                if operation_confirm == "y":
                    WebDriverWait(driver, 3).until(
                        EC.element_to_be_clickable(
                            (
                                By.CSS_SELECTOR,
                                " div.col-md-6:nth-child(2) > button:nth-child(1)",
                            )
                        )
                    )
                    driver.find_element_by_css_selector(
                        "div.col-md-6:nth-child(2) > button:nth-child(1)"
                    ).click()
                    print("\t" + WARNING + "Operación efectuada" + WHITE)
                elif operation_confirm == "n":
                    pass
                else:
                    Prompt.ask("[bold blue]Pulsa Enter para continuar...[/bold blue]")
                    buy_stocks()

            elif confirmation == "n":
                print("\t" + WARNING + "Orden cancelada" + WHITE)
            else:
                buy_stocks()

    except Exception as e:
        print(e)


def login_platform_investing(driver):
    print("\t" + WARNING + "Iniciando sesión en investing.com..." + WHITE)
    try:
        driver.get("https://mx.investing.com/")
    except Exception as e:
        print("Error al cargar la página:")
        print(e)
        return  # Salir de la función si no se puede cargar la página

    is_logged_flag = False

    try:
        print("\t" + WARNING + "Esperar hasta que el botón sea clicable..." + WHITE)
        WebDriverWait(driver, 3).until(
            EC.element_to_be_clickable(
                (By.CSS_SELECTOR, "#\\:r15\\: > form > div > button")
            )
        )
        driver.find_element(By.CSS_SELECTOR, "#\\:r15\\: > form > div > button").click()
        print("Clic realizado en el botón de notificaciones.")
    except Exception as e:
        print(
            "No se pudo hacer clic en el botón de notificaciones, intentando clic forzado..."
        )
        print(e)

        # Intentar clic forzado solo si el elemento existe
        try:
            button = driver.find_element(
                By.CSS_SELECTOR, "#\\:r15\\: > form > div > button"
            )
            driver.execute_script("arguments[0].click();", button)  # Clic forzado
            print("Clic forzado realizado en el botón de notificaciones.")
        except Exception as e:
            print("Error al intentar hacer clic forzado en el botón de notificaciones:")
            print(e)

    # Ahora intenta cerrar la ventana emergente si está presente
    try:
        close_button = driver.find_element(
            By.XPATH, "/html/body/div[2]/div/div/form/div/button"
        )
        close_button.click()
        print("Clic realizado en el botón para cerrar la ventana emergente.")
    except Exception as e:
        print("No se pudo encontrar el botón para cerrar la ventana emergente:")
        print(e)

    try:
        WebDriverWait(driver, 3).until(
            EC.element_to_be_clickable(
                (By.CSS_SELECTOR, ".allow-notifications-popup-close-button")
            )
        )
        driver.find_element(
            By.CSS_SELECTOR, ".allow-notifications-popup-close-button"
        ).click()
    except Exception as e:
        print("Error al dar lick en notifications")
        print(e)

    try:
        WebDriverWait(driver, 3).until(
            EC.element_to_be_clickable(
                (By.CSS_SELECTOR, ".allow-notifications-popup-close-button")
            )
        )
        driver.find_element(
            By.CSS_SELECTOR, ".allow-notifications-popup-close-button"
        ).click()
    except Exception as e:
        print(e)
    try:
        WebDriverWait(driver, 3).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, ".myAccount"))
        )
        user_tag = driver.find_element_by_css_selector(".myAccount").get_attribute(
            "innerText"
        )
        if user_tag == "Osvaldo":
            is_logged_flag = True
            print("\t" + WARNING + "Sesión ya iniciada con anterioridad" + WHITE)
            return is_logged_flag
    except Exception as e:
        print(e)

    try:
        WebDriverWait(driver, 50).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, ".login"))
        )
        driver.find_element(By.CSS_SELECTOR, ".login").click()
    except Exception as e:
        print(e)
        WebDriverWait(driver, 50).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, ".generalOverlay"))
        )
        driver.find_element(By.CSS_SELECTOR, ".popupCloseIcon").click()
        driver.find_element(By.LINK_TEXT, "Iniciar sesión").click()

    driver.find_element(By.ID, "loginFormUser_email").send_keys(
        "osvaldo.hdz.m@outlook.com"
    )
    driver.find_element(By.ID, "loginForm_password").send_keys("Os23valdo1.")
    try:
        time.sleep(3)
        driver.find_element(By.ID, "loginForm_password").send_keys(Keys.ENTER)
        time.sleep(2)
        driver.find_element(By.ID, "loginForm_password").send_keys(Keys.ENTER)
        time.sleep(1)
        driver.find_element(By.ID, "loginForm_password").send_keys(Keys.ENTER)
    except Exception as e:
        print(e)

    try:
        WebDriverWait(driver, 3).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, ".myAccount"))
        )
        user_tag = driver.find_element_by_css_selector(".myAccount").get_attribute(
            "innerText"
        )
        if user_tag == "Osvaldo":
            is_logged_flag = True
            print("\t" + WARNING + "Sesión iniciada con exito" + WHITE)
            return is_logged_flag
    except Exception as e:
        print(e)

    return is_logged_flag


def login_actinver():
    try:
        print("\t" + WARNING + "Iniciando sesión..." + WHITE)
        # driver.get('https://www.retoactinver.com/RetoActinver/#/login')
        driver.refresh()
        time.sleep(3)
        driver.find_element_by_xpath('//*[@id="botonCerrar"]/mat-icon').click()
        user_input = driver.find_element_by_id("mat-input-0")
        user_input.send_keys(USERNAME)
        password_input = driver.find_element_by_id("mat-input-1")
        password_input.send_keys(PASSWORD)
        login_button = driver.find_element_by_xpath(
            "/html/body/app-root/block-ui/app-login/div/form/button[1]/span"
        )
        login_button.click()
    except:
        reconect_session()


def login_platform_actinver(driver, username, password, email):
    print("Logging with {}".format(username))
    is_logged_web_string = ""
    try:
        print(
            "\t" + WARNING + "Accediendo a la plataforma del reto actinver..." + WHITE
        )
        driver.get("https://www.retoactinver.com/RetoActinver/#/login")
        close_popoup_browser(driver)

        print("\t" + WARNING + "Iniciando sesión en plataforma del reto...")
        WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.ID, "mat-input-0"))
        ).send_keys(username)
        WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.ID, "mat-input-1"))
        ).send_keys(password)
        WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.ID, "mat-input-1"))
        ).send_keys(Keys.RETURN)

        try:

            is_logged_web_string = (
                WebDriverWait(driver, 20)
                .until(
                    EC.element_to_be_clickable(
                        (
                            By.CSS_SELECTOR,
                            "mat-list-item.as:nth-child(1) > div:nth-child(1) > div:nth-child(2) > p:nth-child(2)",
                        )
                    )
                )
                .get_attribute("innerText")
            )
            print(is_logged_web_string)
            if is_logged_web_string == "Dashboard":
                print("\t" + WARNING + "Inicio de sesión exitoso! :)" + WHITE)
                return True
        except Exception as e:
            print("Error iniciando en la plataforma {}".format(e))

        try:
            print(
                "\t"
                + WARNING
                + "Posible sesión iniciada con anterioridad, intentando reestablecer sesión..."
                + WHITE
            )
            WebDriverWait(driver, 20).until(
                EC.element_to_be_clickable(
                    (By.CSS_SELECTOR, "button.btn-stroke-alternativo:nth-child(1)")
                )
            ).click()
            WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "#mat-input-2"))
            ).send_keys(username)
            WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "#mat-input-3"))
            ).send_keys(email)
            WebDriverWait(driver, 15).until(
                EC.element_to_be_clickable(
                    (By.CSS_SELECTOR, "button.btn-block:nth-child(1)")
                )
            )
            WebDriverWait(driver, 15).until(
                EC.element_to_be_clickable(
                    (
                        By.XPATH,
                        "/html/body/div[1]/div[2]/div/mat-dialog-container/app-destroy-session/mat-dialog-actions/div/button",
                    )
                )
            ).click()
            driver.refresh()
            close_popoup_browser(driver)

            user_input = driver.find_element_by_id("mat-input-0")
            user_input.send_keys(username)
            password_input = driver.find_element_by_id("mat-input-1")
            password_input.send_keys(password)
            login_button = driver.find_element_by_xpath(
                "/html/body/app-root/block-ui/app-login/div/div/div[1]/form/button[1]"
            )
            login_button.click()
            print("\t" + WARNING + "Inicio de sesión exitoso! :)" + WHITE)
            try:
                is_logged_web_string = (
                    WebDriverWait(driver, 20)
                    .until(
                        EC.element_to_be_clickable(
                            (
                                By.CSS_SELECTOR,
                                "mat-list-item.as:nth-child(1) > div:nth-child(1) > div:nth-child(2) > p:nth-child(2)",
                            )
                        )
                    )
                    .get_attribute("innerText")
                )
                print(is_logged_web_string)
                if is_logged_web_string == "Dashboard":
                    print("\t" + WARNING + "Inicio de sesión exitoso! :)" + WHITE)
                    return True
            except Exception as e:
                print(e)

        except Exception as e:
            print(e)
            return False

    except Exception as e:
        print(e)
        return False


def retrieve_data_reto_capitales():
    try:
        print("\t" + WARNING + "Accediendo a datos de tabla de capitales..." + WHITE)
        driver.get("https://www.retoactinver.com/RetoActinver/#/capitales")
        driver.refresh()
        time.sleep(3)
        driver.find_element_by_xpath('//*[@id="botonCerrar"]/mat-icon').click()
        time.sleep(6)
        driver.find_element_by_xpath('//*[@id="mat-select-1"]/div/div[1]').click()
        time.sleep(3)
        driver.find_element_by_xpath('//*[@id="mat-option-1"]').click()
        driver.find_element_by_xpath(
            '//*[@id="mat-tab-content-0-0"]/div/div/div/app-table/div/gt-column-settings/div/div/generic-table/table/thead[1]/tr/th[6]/span'
        ).click()
        hoursTable = driver.find_element_by_xpath(
            "/html/body/app-root/block-ui/app-paginas/app-layout-principal/mat-sidenav-container/mat-sidenav-content/div[2]/div/app-capitales/div/div[2]/mat-card/mat-tab-group/div/mat-tab-body[1]/div/div/div/app-table/div/gt-column-settings/div/div/generic-table/table"
        ).get_attribute("outerHTML")
        dfs = pd.read_html(hoursTable)
        df = dfs[0]
        df.drop(
            [
                "Sort:",
                "Unnamed: 1",
                "Información",
                "Precio de Compra",
                "Volumen de Venta",
                "% Variación",
                "Volumen Compra",
                "Precio de Venta",
            ],
            axis=1,
            inplace=True,
        )
        df.rename(columns={"Precio": "Variación"}, inplace=True)
        df["Variación"] = df["Variación"].str.replace("% Variación", "")
        df["Variación"] = df["Variación"].str.replace("%", "")
        df.rename(columns={"Emisora": "Precio"}, inplace=True)
        df["Precio"] = df["Precio"].str.replace("Precio", "")
        df.rename(columns={"Categorias": "Emisora"}, inplace=True)
        df["Emisora"] = df["Emisora"].str.replace("Emisora", "")
        df["Emisora"] = df["Emisora"].str.replace(" *", "")
        df["Emisora"] = df["Emisora"].str.replace("*", "")
        df["Datetime"] = datetime.now().strftime("%x %X")
        print(df.head(5))
        df.to_csv("top_dia.csv", index=False, header=True, encoding="utf-8")
    except:
        login_actinver()
        retrieve_data_reto_capitales()


def retrieve_data_reto_portafolio():
    # while True:
    print("\t" + WARNING + "Accediendo a datos de tabla de portafolio..." + WHITE)
    driver.get("https://www.retoactinver.com/RetoActinver/#/portafolio")
    driver.refresh()
    time.sleep(3)
    driver.find_element_by_xpath('//*[@id="botonCerrar"]').click()
    hoursTable = driver.find_element_by_xpath(
        '//*[@id="tpm"]/app-portafolio/div/div[3]/mat-card/app-table/div/gt-column-settings/div/div/generic-table/table'
    ).get_attribute("outerHTML")
    dfs = pd.read_html(hoursTable)
    df = dfs[0]
    df = df[["Categorias", "Emisora", "Títulos", "Precio Actual", "Variación $"]]
    df.rename(columns={"Variación $": "Variación %"}, inplace=True)
    df["Variación %"] = df["Variación %"].str.replace("% Variación", "")
    df["Variación %"] = df["Variación %"].str.replace("%", "")
    df.rename(columns={"Precio Actual": "Valor actual"}, inplace=True)
    df["Valor actual"] = df["Valor actual"].str.replace("Valor Actual", "")
    df.rename(columns={"Títulos": "Costo de compra"}, inplace=True)
    df["Costo de compra"] = df["Costo de compra"].str.replace("Valor del Costo", "")
    df.rename(columns={"Emisora": "Títulos"}, inplace=True)
    df["Títulos"] = df["Títulos"].str.replace("Títulos", "")
    df.rename(columns={"Categorias": "Emisora"}, inplace=True)
    df["Emisora"] = df["Emisora"].str.replace("Emisora", "")
    df["Emisora"] = df["Emisora"].str.replace(" *", "")
    df["Emisora"] = df["Emisora"].str.replace("*", "")
    df["Datetime"] = datetime.now().strftime("%x %X")
    print(df.head(10))


def percentage_change(col1, col2):
    return ((col2 - col1) / col1) * 100


def containsAny(str, set):
    """Check whether sequence str contains ANY of the items in set."""
    return 1 in [c in str for c in set]


def insert_string_before(stringO, string_to_insert, insert_before_char):
    return stringO.replace(insert_before_char, string_to_insert + insert_before_char, 1)


def day_trading_strategy():

    try:
        login_platform_investing(driver)
        driver.get("https://mx.investing.com/")
        time.sleep(3)
        input()
        WebDriverWait(driver, 50).until(
            EC.element_to_be_clickable((By.XPATH, '//*[@id="portfolioTopBarBtn"]'))
        )
        driver.find_element_by_xpath('//*[@id="portfolioTopBarBtn"]').click()
        WebDriverWait(driver, 50).until(
            EC.element_to_be_clickable(
                (
                    By.CSS_SELECTOR,
                    "#portfolioContainer_25863947 > div:nth-child(1) > a:nth-child(2)",
                )
            )
        )
        driver.find_element(
            By.CSS_SELECTOR,
            "#portfolioContainer_25863947 > div:nth-child(1) > a:nth-child(2)",
        ).click()

        driver.find_element(By.CSS_SELECTOR, "#performance > a:nth-child(1)").click()
        time.sleep(3)
        table = driver.find_element_by_xpath(
            "/html/body/div[5]/section/div[7]/div[7]/div[2]/div[2]/div/table"
        ).get_attribute("outerHTML")
        dfs = pd.read_html(table)
        df1 = dfs[0]
        df1.drop(["Unnamed: 0", "Unnamed: 8"], axis=1, inplace=True, errors="ignore")
        # print(df.columns)

        # df.to_csv('analisis_tecnico_performance.csv',  index=False, header=True, encoding="utf-8")
        driver.find_element(By.CSS_SELECTOR, "#technical > a:nth-child(1)").click()
        time.sleep(3)
        table = driver.find_element_by_xpath(
            "/html/body/div[5]/section/div[7]/div[7]/div[2]/div[2]/div/table"
        ).get_attribute("outerHTML")
        dfs = pd.read_html(table)
        df2 = dfs[0]
        df2.drop(["Unnamed: 0", "Unnamed: 10"], axis=1, inplace=True, errors="ignore")
        # print(df.columns)
        # df.to_csv('analisis_tecnico_algoritmo.csv',index=False, header=True, encoding="utf-8")

        driver.find_element(By.CSS_SELECTOR, "#overview > a:nth-child(1)").click()
        time.sleep(3)
        table = driver.find_element_by_xpath(
            "/html/body/div[5]/section/div[7]/div[7]/div[2]/div[2]/div/div/table"
        ).get_attribute("outerHTML")
        dfs = pd.read_html(table)
        df3 = dfs[0]
        df3.drop(
            ["Unnamed: 0", "Unnamed: 1", "Unnamed: 19", "Unnamed: 20"],
            axis=1,
            inplace=True,
            errors="ignore",
        )
        # print(df.columns)

        # Create dicionary investing simbols portfolio
        WebDriverWait(driver, 50).until(
            EC.element_to_be_clickable(
                (
                    By.XPATH,
                    "/html/body/div[5]/section/div[7]/div[7]/div[2]/div[2]/div/div/table",
                )
            )
        )
        table_elements_simbols = driver.find_elements(
            By.XPATH,
            "/html/body/div[5]/section/div[7]/div[7]/div[2]/div[2]/div/div/table/tbody/tr/td[4]/a",
        )
        dicionary_simbols = {}
        for tag in table_elements_simbols:
            dicionary_simbols[str(tag.get_attribute("innerHTML"))] = str(
                tag.get_attribute("href")
            )

        dfs = [df1, df2, df3]
        dfs = [x.set_index("Nombre") for x in dfs]
        df = pd.concat(dfs, axis=1, keys=range(1, len(dfs) + 1))
        df.columns = df.columns.map("{0[1]}{0[0]}".format)
        df = df.reset_index()
        df["Fecha"] = datetime.now().strftime("%x %X")
        # print(df.columns)

        df["Diario1"] = df["Diario1"].map(lambda x: str(x)[:-1])
        df["Diario1"] = df["Diario1"].astype("float")
        df["Semanal1"] = df["Semanal1"].map(lambda x: str(x)[:-1])
        df["Semanal1"] = df["Semanal1"].astype("float")
        df["Mensual1"] = df["Mensual1"].map(lambda x: str(x)[:-1])
        df["Mensual1"] = df["Mensual1"].astype("float")
        df["Anual1"] = df["Anual1"].map(lambda x: str(x)[:-1])
        df["Anual1"] = df["Anual1"].astype("float")
        df["1 Año1"] = df["1 Año1"].map(lambda x: str(x)[:-1])
        df["1 Año1"] = df["1 Año1"].astype("float")
        df["3 años1"] = df["3 años1"].map(lambda x: str(x).replace("-", "0"))
        df["3 años1"] = df["3 años1"].map(lambda x: str(x).replace("%", ""))
        df["3 años1"] = df["3 años1"].astype("float")
        df["% var.3"] = df["% var.3"].map(lambda x: str(x)[:-1])
        df["% var.3"] = df["% var.3"].astype("float")

        print("\t" + WARNING + "Analizando acciones..." + WHITE)

        print(
            "\t"
            + WARNING
            + "Analizando acciones con ganancias en diferentes periodos..."
            + WHITE
        )
        df = df[(df[["3 años1"]] > 0).all(1)]
        df = df[(df[["1 Año1"]] > 0).all(1)]
        df = df[(df[["Anual1"]] > 0).all(1)]
        df = df[(df[["Mensual1"]] > 0).all(1)]
        df = df[(df[["Semanal1"]] > 0).all(1)]
        print(
            "\t"
            + WARNING
            + "Analizando acciones con ganacia diaria menor al promedio..."
            + WHITE
        )
        daily_mean = df["Diario1"].mean()
        df = df[(df[["Diario1"]] < daily_mean).all(1)]
        print(
            "\t"
            + WARNING
            + "Analizando acciones con indicadores técnicos en consenso de compra en diferentes periodos..."
            + WHITE
        )
        df = df[
            (df[["Mensual2"]] == "Compra").all(1)
            | (df[["Mensual2"]] == "Compra fuerte").all(1)
        ]
        df = df[
            (df[["Semanal2"]] == "Compra").all(1)
            | (df[["Semanal2"]] == "Compra fuerte").all(1)
        ]
        df = df[
            (df[["Diario2"]] == "Compra").all(1)
            | (df[["Diario2"]] == "Compra fuerte").all(1)
        ]
        print(
            "\t"
            + WARNING
            + "Analizando acciones con indicadores técnicos en baja con estimación de alza..."
            + WHITE
        )
        df = df[
            (df[["5 horas2"]] == "Venta").all(1)
            | (df[["5 horas2"]] == "Venta fuerte").all(1)
            | (df[["5 horas2"]] == "Compra fuerte").all(1)
            | (df[["5 horas2"]] == "Compra").all(1)
        ]
        df = df[
            (df[["1 hora2"]] == "Venta").all(1)
            | (df[["1 hora2"]] == "Venta fuerte").all(1)
            | (df[["1 hora2"]] == "Compra fuerte").all(1)
            | (df[["1 hora2"]] == "Compra").all(1)
        ]
        df = df[
            (df[["30 minutos2"]] == "Compra").all(1)
            | (df[["30 minutos2"]] == "Compra fuerte").all(1)
            | (df[["30 minutos2"]] == "Venta").all(1)
            | (df[["30 minutos2"]] == "Venta fuerte").all(1)
        ]
        print(
            "\t"
            + WARNING
            + "Analizando acciones con indicadores técnicos con estimación de alza durante la sesión..."
            + WHITE
        )
        df = df[
            (df[["15 minutos2"]] == "Compra").all(1)
            | (df[["15 minutos2"]] == "Compra fuerte").all(1)
        ]
        df = df[
            (df[["5 minutos2"]] == "Compra").all(1)
            | (df[["5 minutos2"]] == "Compra fuerte").all(1)
        ]

        print("\t" + WARNING + "Calculando precios y ganancias estimadas..." + WHITE)
        for x in df["Símbolo3"]:
            if containsAny(dicionary_simbols[x], ["?"]):
                technical_data_url = insert_string_before(
                    dicionary_simbols[x], "-technical", "?"
                )
            else:
                technical_data_url = dicionary_simbols[x] + "-technical"
            driver.get(technical_data_url)
            WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable(
                    (By.XPATH, "/html/body/div[5]/section/div[8]/ul/li[6]/a")
                )
            )
            driver.find_element(
                By.XPATH, "/html/body/div[5]/section/div[8]/ul/li[6]/a"
            ).click()

            WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable(
                    (By.XPATH, "/html/body/div[5]/section/div[10]/table")
                )
            )
            p1 = float(
                driver.find_element(
                    By.XPATH,
                    "/html/body/div[5]/section/div[10]/table/tbody/tr[2]/td[6]",
                ).get_attribute("innerHTML")
            )
            p2 = float(
                driver.find_element(
                    By.XPATH,
                    "/html/body/div[5]/section/div[10]/table/tbody/tr[2]/td[7]",
                ).get_attribute("innerHTML")
            )
            p3 = float(
                driver.find_element(
                    By.XPATH,
                    "/html/body/div[5]/section/div[10]/table/tbody/tr[2]/td[8]",
                ).get_attribute("innerHTML")
            )
            pe = round((p1 + p2 + p3) / 3, 2)
            df.loc[df["Símbolo3"] == x, "PeEstimado"] = pe

        df["GanEstimada %"] = round(
            percentage_change(df["Último3"], df["PeEstimado"]), 2
        )
        df = df[(df[["GanEstimada %"]] > 0.5).all(1)]
        df["PeVentaCalc"] = round(df["Último3"] * 1.005, 2)

        print("\n\t" + OKGREEN + "Resultado de sugerencias:" + WHITE)

        print(
            df[
                [
                    "Símbolo3",
                    "Nombre",
                    "Último3",
                    "PeEstimado",
                    "GanEstimada %",
                    "PeVentaCalc",
                ]
            ]
        )

    except Exception as e:
        print(e)


def swing_trading_strategy():
    driver = configure_chrome_driver_no_profile()
    try:
        login_platform_investing(driver)
        driver.get("https://mx.investing.com/")
        time.sleep(3)

        WebDriverWait(driver, 50).until(
            EC.element_to_be_clickable((By.XPATH, '//*[@id="portfolioTopBarBtn"]'))
        )
        driver.find_element_by_xpath('//*[@id="portfolioTopBarBtn"]').click()
        WebDriverWait(driver, 50).until(
            EC.element_to_be_clickable(
                (
                    By.CSS_SELECTOR,
                    "#portfolioContainer_25863947 > div:nth-child(1) > a:nth-child(2)",
                )
            )
        )
        driver.find_element(
            By.CSS_SELECTOR,
            "#portfolioContainer_25863947 > div:nth-child(1) > a:nth-child(2)",
        ).click()

        driver.find_element(By.CSS_SELECTOR, "#performance > a:nth-child(1)").click()
        time.sleep(3)
        table = driver.find_element_by_xpath(
            "/html/body/div[5]/section/div[7]/div[7]/div[2]/div[2]/div/table"
        ).get_attribute("outerHTML")
        dfs = pd.read_html(table)
        df1 = dfs[0]
        df1.drop(["Unnamed: 0", "Unnamed: 8"], axis=1, inplace=True, errors="ignore")
        # print(df.columns)

        # df.to_csv('analisis_tecnico_performance.csv',  index=False, header=True, encoding="utf-8")
        driver.find_element(By.CSS_SELECTOR, "#technical > a:nth-child(1)").click()
        time.sleep(3)
        table = driver.find_element_by_xpath(
            "/html/body/div[5]/section/div[7]/div[7]/div[2]/div[2]/div/table"
        ).get_attribute("outerHTML")
        dfs = pd.read_html(table)
        df2 = dfs[0]
        df2.drop(["Unnamed: 0", "Unnamed: 10"], axis=1, inplace=True, errors="ignore")
        # print(df.columns)
        # df.to_csv('analisis_tecnico_algoritmo.csv',index=False, header=True, encoding="utf-8")

        driver.find_element(By.CSS_SELECTOR, "#overview > a:nth-child(1)").click()
        time.sleep(3)
        table = driver.find_element_by_xpath(
            "/html/body/div[5]/section/div[7]/div[7]/div[2]/div[2]/div/div/table"
        ).get_attribute("outerHTML")
        dfs = pd.read_html(table)
        df3 = dfs[0]
        df3.drop(
            ["Unnamed: 0", "Unnamed: 1", "Unnamed: 19", "Unnamed: 20"],
            axis=1,
            inplace=True,
            errors="ignore",
        )
        # print(df.columns)

        # Create dicionary investing simbols portfolio
        WebDriverWait(driver, 50).until(
            EC.element_to_be_clickable(
                (
                    By.XPATH,
                    "/html/body/div[5]/section/div[7]/div[7]/div[2]/div[2]/div/div/table",
                )
            )
        )
        table_elements_simbols = driver.find_elements(
            By.XPATH,
            "/html/body/div[5]/section/div[7]/div[7]/div[2]/div[2]/div/div/table/tbody/tr/td[4]/a",
        )
        dicionary_simbols = {}
        for tag in table_elements_simbols:
            dicionary_simbols[str(tag.get_attribute("innerHTML"))] = str(
                tag.get_attribute("href")
            )

        dfs = [df1, df2, df3]
        dfs = [x.set_index("Nombre") for x in dfs]
        df = pd.concat(dfs, axis=1, keys=range(1, len(dfs) + 1))
        df.columns = df.columns.map("{0[1]}{0[0]}".format)
        df = df.reset_index()
        df["Fecha"] = datetime.now().strftime("%x %X")
        # print(df.columns)

        df["Diario1"] = df["Diario1"].map(lambda x: str(x)[:-1])
        df["Diario1"] = df["Diario1"].astype("float")
        df["Semanal1"] = df["Semanal1"].map(lambda x: str(x)[:-1])
        df["Semanal1"] = df["Semanal1"].astype("float")
        df["Mensual1"] = df["Mensual1"].map(lambda x: str(x)[:-1])
        df["Mensual1"] = df["Mensual1"].astype("float")
        df["Anual1"] = df["Anual1"].map(lambda x: str(x)[:-1])
        df["Anual1"] = df["Anual1"].astype("float")
        df["1 Año1"] = df["1 Año1"].map(lambda x: str(x)[:-1])
        df["1 Año1"] = df["1 Año1"].astype("float")
        df["3 años1"] = df["3 años1"].map(lambda x: str(x).replace("-", "0"))
        df["3 años1"] = df["3 años1"].map(lambda x: str(x).replace("%", ""))
        df["3 años1"] = df["3 años1"].astype("float")
        df["% var.3"] = df["% var.3"].map(lambda x: str(x)[:-1])
        df["% var.3"] = df["% var.3"].astype("float")

        print("\t" + WARNING + "Analizando acciones..." + WHITE)

        print(
            "\t"
            + WARNING
            + "Analizando acciones con ganancias en diferentes periodos..."
            + WHITE
        )
        df = df[(df[["3 años1"]] > 0).all(1)]
        df = df[(df[["1 Año1"]] > 0).all(1)]
        df = df[(df[["Anual1"]] > 0).all(1)]
        df = df[(df[["Mensual1"]] > 0).all(1)]
        df = df[(df[["Semanal1"]] > 0).all(1)]
        print(
            "\t"
            + WARNING
            + "Analizando acciones con ganacia diaria menor al promedio..."
            + WHITE
        )
        df = df[(df[["Diario1"]] < 0).all(1)]
        print(
            "\t"
            + WARNING
            + "Analizando acciones con indicadores técnicos en consenso de compra en diferentes periodos..."
            + WHITE
        )
        df = df[
            (df[["Mensual2"]] == "Compra").all(1)
            | (df[["Mensual2"]] == "Compra fuerte").all(1)
        ]
        df = df[
            (df[["Semanal2"]] == "Compra").all(1)
            | (df[["Semanal2"]] == "Compra fuerte").all(1)
        ]
        df = df[
            (df[["Diario2"]] == "Compra").all(1)
            | (df[["Diario2"]] == "Compra fuerte").all(1)
        ]
        print(
            "\t"
            + WARNING
            + "Analizando acciones con indicadores técnicos en baja con estimación de alza..."
            + WHITE
        )
        df = df[
            (df[["5 horas2"]] == "Venta").all(1)
            | (df[["5 horas2"]] == "Venta fuerte").all(1)
            | (df[["5 horas2"]] == "Compra fuerte").all(1)
            | (df[["5 horas2"]] == "Compra").all(1)
        ]
        df = df[
            (df[["1 hora2"]] == "Venta").all(1)
            | (df[["1 hora2"]] == "Venta fuerte").all(1)
            | (df[["1 hora2"]] == "Compra fuerte").all(1)
            | (df[["1 hora2"]] == "Compra").all(1)
        ]
        df = df[
            (df[["30 minutos2"]] == "Compra").all(1)
            | (df[["30 minutos2"]] == "Compra fuerte").all(1)
            | (df[["30 minutos2"]] == "Venta").all(1)
            | (df[["30 minutos2"]] == "Venta fuerte").all(1)
        ]
        print(
            "\t"
            + WARNING
            + "Analizando acciones con indicadores técnicos con estimación de alza durante la sesión..."
            + WHITE
        )
        df = df[
            (df[["15 minutos2"]] == "Compra").all(1)
            | (df[["15 minutos2"]] == "Compra fuerte").all(1)
            | (df[["15 minutos2"]] == "Neutral").all(1)
            | (df[["15 minutos2"]] == "Venta fuerte").all(1)
            | (df[["15 minutos2"]] == "Venta").all(1)
        ]
        df = df[
            (df[["5 minutos2"]] == "Compra").all(1)
            | (df[["5 minutos2"]] == "Compra fuerte").all(1)
            | (df[["5 minutos2"]] == "Neutral").all(1)
            | (df[["5 minutos2"]] == "Venta fuerte").all(1)
            | (df[["5 minutos2"]] == "Venta").all(1)
        ]

        print("\t" + WARNING + "Calculando precios y ganancias estimadas..." + WHITE)
        for x in df["Símbolo3"]:
            if containsAny(dicionary_simbols[x], ["?"]):
                technical_data_url = insert_string_before(
                    dicionary_simbols[x], "-technical", "?"
                )
            else:
                technical_data_url = dicionary_simbols[x] + "-technical"

            driver.get(technical_data_url)
            WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable(
                    (By.XPATH, "/html/body/div[5]/section/div[8]/ul/li[6]/a")
                )
            )
            driver.find_element(
                By.XPATH, "/html/body/div[5]/section/div[8]/ul/li[6]/a"
            ).click()

            WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable(
                    (By.XPATH, "/html/body/div[5]/section/div[10]/table")
                )
            )
            p1 = float(
                driver.find_element(
                    By.XPATH,
                    "/html/body/div[5]/section/div[10]/table/tbody/tr[2]/td[6]",
                ).get_attribute("innerHTML")
            )
            p2 = float(
                driver.find_element(
                    By.XPATH,
                    "/html/body/div[5]/section/div[10]/table/tbody/tr[2]/td[7]",
                ).get_attribute("innerHTML")
            )
            p3 = float(
                driver.find_element(
                    By.XPATH,
                    "/html/body/div[5]/section/div[10]/table/tbody/tr[2]/td[8]",
                ).get_attribute("innerHTML")
            )
            pe = round((p1 + p2 + p3) / 3, 2)
            df.loc[df["Símbolo3"] == x, "PeEstimado"] = pe

        df["GanEstimada %"] = round(
            percentage_change(df["Último3"], df["PeEstimado"]), 2
        )
        df = df[(df[["GanEstimada %"]] > 0.5).all(1)]
        df["PeVentaCalc"] = round(df["Último3"] * 1.005, 2)

        print("\n\t" + OKGREEN + "Resultado de sugerencias:" + WHITE)

        print(
            df[
                [
                    "Símbolo3",
                    "Nombre",
                    "Último3",
                    "PeEstimado",
                    "GanEstimada %",
                    "PeVentaCalc",
                ]
            ]
        )

    except Exception as e:
        print(e)


def swing_trading_strategy2():

    try:
        print("\t" + WARNING + "Obteniendo datos de acciones..." + WHITE)
        driver.get("https://mx.investing.com/")

        is_logged_flag = ""
        try:
            WebDriverWait(driver, 3).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, ".myAccount"))
            )
            is_logged_flag = driver.find_element_by_css_selector(
                ".myAccount"
            ).get_attribute("innerText")
        except Exception as e:
            print(e)
        if is_logged_flag == "Osvaldo":
            print("\t" + WARNING + "Sesión ya iniciada con anterioridad" + WHITE)
        else:
            try:
                try:
                    try:
                        driver.find_element(By.LINK_TEXT, "Iniciar sesión").click()

                    except Exception as e:
                        print(e)
                        WebDriverWait(driver, 50).until(
                            EC.element_to_be_clickable(
                                (By.CSS_SELECTOR, ".generalOverlay")
                            )
                        )
                        driver.find_element(By.CSS_SELECTOR, ".popupCloseIcon").click()
                        driver.find_element(By.LINK_TEXT, "Iniciar sesión").click()

                except Exception as e:
                    print(e)

                driver.find_element(By.ID, "loginFormUser_email").send_keys(
                    "osvaldo.hdz.m@outlook.com"
                )
                driver.find_element(By.ID, "loginForm_password").send_keys(
                    "Os23valdo1."
                )

                try:
                    time.sleep(3)
                    driver.find_element(By.ID, "loginForm_password").send_keys(
                        Keys.ENTER
                    )
                    time.sleep(2)
                    driver.find_element(By.ID, "loginForm_password").send_keys(
                        Keys.ENTER
                    )
                    time.sleep(1)
                    driver.find_element(By.ID, "loginForm_password").send_keys(
                        Keys.ENTER
                    )
                except Exception as e:
                    print(e)
            except Exception as e:
                print(e)

        print(
            "\t" + WARNING + "Conexión con datos de investing.com establecida" + WHITE
        )
        time.sleep(3)

        WebDriverWait(driver, 50).until(
            EC.element_to_be_clickable((By.XPATH, '//*[@id="portfolioTopBarBtn"]'))
        )
        driver.find_element_by_xpath('//*[@id="portfolioTopBarBtn"]').click()
        WebDriverWait(driver, 50).until(
            EC.element_to_be_clickable(
                (
                    By.CSS_SELECTOR,
                    "#portfolioContainer_25863947 > div:nth-child(1) > a:nth-child(2)",
                )
            )
        )
        driver.find_element(
            By.CSS_SELECTOR,
            "#portfolioContainer_25863947 > div:nth-child(1) > a:nth-child(2)",
        ).click()

        driver.find_element(By.CSS_SELECTOR, "#performance > a:nth-child(1)").click()
        time.sleep(3)
        table = driver.find_element_by_xpath(
            "/html/body/div[5]/section/div[7]/div[7]/div[2]/div[2]/div/table"
        ).get_attribute("outerHTML")
        dfs = pd.read_html(table)
        df1 = dfs[0]
        df1.drop(["Unnamed: 0", "Unnamed: 8"], axis=1, inplace=True, errors="ignore")
        # print(df.columns)

        # df.to_csv('analisis_tecnico_performance.csv',  index=False, header=True, encoding="utf-8")
        driver.find_element(By.CSS_SELECTOR, "#technical > a:nth-child(1)").click()
        time.sleep(3)
        table = driver.find_element_by_xpath(
            "/html/body/div[5]/section/div[7]/div[7]/div[2]/div[2]/div/table"
        ).get_attribute("outerHTML")
        dfs = pd.read_html(table)
        df2 = dfs[0]
        df2.drop(["Unnamed: 0", "Unnamed: 10"], axis=1, inplace=True, errors="ignore")
        # print(df.columns)
        # df.to_csv('analisis_tecnico_algoritmo.csv',index=False, header=True, encoding="utf-8")

        driver.find_element(By.CSS_SELECTOR, "#overview > a:nth-child(1)").click()
        time.sleep(3)
        table = driver.find_element_by_xpath(
            "/html/body/div[5]/section/div[7]/div[7]/div[2]/div[2]/div/div/table"
        ).get_attribute("outerHTML")
        dfs = pd.read_html(table)
        df3 = dfs[0]
        df3.drop(
            ["Unnamed: 0", "Unnamed: 1", "Unnamed: 19", "Unnamed: 20"],
            axis=1,
            inplace=True,
            errors="ignore",
        )
        # print(df.columns)

        # Create dicionary investing simbols portfolio
        WebDriverWait(driver, 50).until(
            EC.element_to_be_clickable(
                (
                    By.XPATH,
                    "/html/body/div[5]/section/div[7]/div[7]/div[2]/div[2]/div/div/table",
                )
            )
        )
        table_elements_simbols = driver.find_elements(
            By.XPATH,
            "/html/body/div[5]/section/div[7]/div[7]/div[2]/div[2]/div/div/table/tbody/tr/td[4]/a",
        )
        dicionary_simbols = {}
        for tag in table_elements_simbols:
            dicionary_simbols[str(tag.get_attribute("innerHTML"))] = str(
                tag.get_attribute("href")
            )

        dfs = [df1, df2, df3]
        dfs = [x.set_index("Nombre") for x in dfs]
        df = pd.concat(dfs, axis=1, keys=range(1, len(dfs) + 1))
        df.columns = df.columns.map("{0[1]}{0[0]}".format)
        df = df.reset_index()
        df["Fecha"] = datetime.now().strftime("%x %X")
        # print(df.columns)

        df["Diario1"] = df["Diario1"].map(lambda x: str(x)[:-1])
        df["Diario1"] = df["Diario1"].astype("float")
        df["Semanal1"] = df["Semanal1"].map(lambda x: str(x)[:-1])
        df["Semanal1"] = df["Semanal1"].astype("float")
        df["Mensual1"] = df["Mensual1"].map(lambda x: str(x)[:-1])
        df["Mensual1"] = df["Mensual1"].astype("float")
        df["Anual1"] = df["Anual1"].map(lambda x: str(x)[:-1])
        df["Anual1"] = df["Anual1"].astype("float")
        df["1 Año1"] = df["1 Año1"].map(lambda x: str(x)[:-1])
        df["1 Año1"] = df["1 Año1"].astype("float")
        df["3 años1"] = df["3 años1"].map(lambda x: str(x).replace("-", "0"))
        df["3 años1"] = df["3 años1"].map(lambda x: str(x).replace("%", ""))
        df["3 años1"] = df["3 años1"].astype("float")
        df["% var.3"] = df["% var.3"].map(lambda x: str(x)[:-1])
        df["% var.3"] = df["% var.3"].astype("float")

        print("\t" + WARNING + "Analizando acciones..." + WHITE)

        print(
            "\t"
            + WARNING
            + "Analizando acciones con ganancias en diferentes periodos..."
            + WHITE
        )
        df = df[(df[["3 años1"]] > 0).all(1)]
        df = df[(df[["1 Año1"]] > 0).all(1)]
        df = df[(df[["Anual1"]] > 0).all(1)]
        df = df[(df[["Mensual1"]] > 0).all(1)]
        df = df[(df[["Semanal1"]] > 0).all(1)]
        print(
            "\t"
            + WARNING
            + "Analizando acciones con ganacia diaria menor al promedio..."
            + WHITE
        )
        df = df[(df[["Diario1"]] > 0).all(1)]

        print(
            "\t"
            + WARNING
            + "Analizando acciones con indicadores técnicos en consenso de compra en diferentes periodos..."
            + WHITE
        )
        df = df[
            (df[["Mensual2"]] == "Compra").all(1)
            | (df[["Mensual2"]] == "Compra fuerte").all(1)
        ]
        df = df[
            (df[["Semanal2"]] == "Compra").all(1)
            | (df[["Semanal2"]] == "Compra fuerte").all(1)
        ]
        df = df[
            (df[["Diario2"]] == "Compra").all(1)
            | (df[["Diario2"]] == "Compra fuerte").all(1)
        ]
        print(
            "\t"
            + WARNING
            + "Analizando acciones con indicadores técnicos en baja con estimación de alza..."
            + WHITE
        )
        df = df[
            (df[["5 horas2"]] == "Venta").all(1)
            | (df[["5 horas2"]] == "Venta fuerte").all(1)
            | (df[["5 horas2"]] == "Compra fuerte").all(1)
            | (df[["5 horas2"]] == "Compra").all(1)
        ]
        # df = df[(df[['1 hora2']] == 'Venta').all(1) | (df[['1 hora2']] == 'Venta fuerte').all(1) | (df[['1 hora2']] == 'Compra fuerte').all(1) | (df[['1 hora2']] == 'Compra').all(1)]
        # df = df[(df[['30 minutos2']] == 'Compra').all(1) | (df[['30 minutos2']] == 'Compra fuerte').all(1) | (df[['30 minutos2']] == 'Venta').all(1) | (df[['30 minutos2']] == 'Venta fuerte').all(1)]
        print(
            "\t"
            + WARNING
            + "Analizando acciones con indicadores técnicos con estimación de alza durante la sesión..."
            + WHITE
        )
        # df = df[(df[['15 minutos2']] == 'Compra').all(1) | (df[['15 minutos2']] == 'Compra fuerte').all(1) | (df[['15 minutos2']] == 'Neutral').all(1) | (df[['15 minutos2']] == 'Venta fuerte').all(1) | (df[['15 minutos2']] == 'Venta').all(1) ]
        # df = df[(df[['5 minutos2']] == 'Compra').all(1) | (df[['5 minutos2']] == 'Compra fuerte').all(1) | (df[['5 minutos2']] == 'Neutral').all(1) | (df[['5 minutos2']] == 'Venta fuerte').all(1) | (df[['5 minutos2']] == 'Venta').all(1)]

        print("\t" + WARNING + "Calculando precios y ganancias estimadas..." + WHITE)
        for x in df["Símbolo3"]:
            if containsAny(dicionary_simbols[x], ["?"]):
                technical_data_url = insert_string_before(
                    dicionary_simbols[x], "-technical", "?"
                )
            else:
                technical_data_url = dicionary_simbols[x] + "-technical"
            driver.get(technical_data_url)
            WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable(
                    (By.XPATH, "/html/body/div[5]/section/div[8]/ul/li[6]/a")
                )
            )
            driver.find_element(
                By.XPATH, "/html/body/div[5]/section/div[8]/ul/li[6]/a"
            ).click()

            WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable(
                    (By.XPATH, "/html/body/div[5]/section/div[10]/table")
                )
            )
            p1 = float(
                driver.find_element(
                    By.XPATH,
                    "/html/body/div[5]/section/div[10]/table/tbody/tr[2]/td[6]",
                ).get_attribute("innerHTML")
            )
            p2 = float(
                driver.find_element(
                    By.XPATH,
                    "/html/body/div[5]/section/div[10]/table/tbody/tr[2]/td[7]",
                ).get_attribute("innerHTML")
            )
            p3 = float(
                driver.find_element(
                    By.XPATH,
                    "/html/body/div[5]/section/div[10]/table/tbody/tr[2]/td[8]",
                ).get_attribute("innerHTML")
            )
            pe = round((p1 + p2 + p3) / 3, 2)
            df.loc[df["Símbolo3"] == x, "PeEstimado"] = pe

        df["GanEstimada %"] = round(
            percentage_change(df["Último3"], df["PeEstimado"]), 2
        )
        df = df[(df[["GanEstimada %"]] > 0.5).all(1)]
        df["PeVentaCalc"] = round(df["Último3"] * 1.005, 2)

        print("\n\t" + OKGREEN + "Resultado de sugerencias:" + WHITE)

        print(
            df[
                [
                    "Símbolo3",
                    "Nombre",
                    "Último3",
                    "PeEstimado",
                    "GanEstimada %",
                    "PeVentaCalc",
                ]
            ]
        )

    except Exception as e:
        print(e)


def predict_machine_daily(stock_simbol):
    mod_ticker = stock_simbol  # use US symbol as-is

    start = datetime.now() - timedelta(days=365)
    end = datetime.now()

    df = web.DataReader(mod_ticker, "yahoo", start, end)

    df = df.replace(0, np.nan).ffill()

    if len(df.index) < 10:
        return "Neutral"
    else:

        dfreg = df.loc[:, ["Adj Close", "Volume"]]
        dfreg["HL_PCT"] = (df["High"] - df["Low"]) / df["Close"] * 100.0
        dfreg["PCT_change"] = (df["Close"] - df["Open"]) / df["Open"] * 100.0

        last_close = float(dfreg["Adj Close"].iloc[-1])

        # Drop missing value
        dfreg.fillna(value=-99999, inplace=True)

        # print(dfreg.shape)
        # We want to separate 1 percent of the data to forecast
        # Number of forecast values in plot
        forecast_out = int(math.ceil(0.01 * len(dfreg)))

        # print("forecast out  : "+ str(forecast_out))

        # Separating the label here, we want to predict the AdjClose
        forecast_col = "Adj Close"
        dfreg["label"] = dfreg[forecast_col].shift(-forecast_out)
        X = np.array(dfreg.drop(["label"], 1))

        # Scale the X so that everyone can have the same distribution for linear regression
        X = preprocessing.scale(X)

        # Finally We want to find Data Series of late X and early X (train) for model generation and evaluation
        X_lately = X[-forecast_out:]
        X = X[:-forecast_out]

        # Separate label and identify it as y
        y = np.array(dfreg["label"])
        y = y[:-forecast_out]

        # print('Dimension of X',X.shape)
        # print('Dimension of y',y.shape)

        # Separation of training and testing of model by cross validation train test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # Linear regression
        clfreg = LinearRegression(n_jobs=-1)
        clfreg.fit(X_train, y_train)

        # Quadratic Regression 2
        clfpoly2 = make_pipeline(PolynomialFeatures(2), Ridge())
        clfpoly2.fit(X_train, y_train)

        # Quadratic Regression 3
        clfpoly3 = make_pipeline(PolynomialFeatures(3), Ridge())
        clfpoly3.fit(X_train, y_train)

        # KNN Regression
        clfknn = KNeighborsRegressor(n_neighbors=2)
        clfknn.fit(X_train, y_train)

        confidencereg = clfreg.score(X_test, y_test)
        confidencepoly2 = clfpoly2.score(X_test, y_test)
        confidencepoly3 = clfpoly3.score(X_test, y_test)
        confidenceknn = clfknn.score(X_test, y_test)

        # print("The linear regression confidence is ",confidencereg)
        # print("The quadratic regression 2 confidence is ",confidencepoly2)
        # print("The quadratic regression 3 confidence is ",confidencepoly3)
        # print("The knn regression confidence is ",confidenceknn)

        # Printing the forecast
        forecast_set = clfreg.predict(X_lately)

        dfreg["Forecast"] = np.nan
        # print(forecast_set, confidencereg, forecast_out)

        last_date = dfreg.iloc[-1].name
        last_unix = last_date
        next_unix = last_unix + timedelta(days=1)

        for i in forecast_set:
            next_date = next_unix
            next_unix += timedelta(days=1)
            dfreg.loc[next_date] = [np.nan for _ in range(len(dfreg.columns) - 1)] + [i]

        # print(dfreg['Forecast'])

        dfreg["Adj Close"].tail(500).plot()
        dfreg["Forecast"].tail(500).plot()

        # plt.legend(loc=4)
        # plt.xlabel('Date')
        # plt.ylabel('Price')

        # Plot de graph
        # plt.show()

        last_forescast = float(dfreg["Forecast"].iloc[-1])

        diference = last_forescast - last_close

        # print(last_close)
        # print(last_forescast)
        # print(diference)
        if diference > 0:
            return "Compra"
        elif diference < 0:
            return "Venta"
        else:
            return "Neutral"


def swing_trading_strategy_machine():

    try:
        print("\t" + WARNING + "Obteniendo datos de acciones..." + WHITE)
        driver.get("https://mx.investing.com/")

        is_logged_flag = ""
        try:
            WebDriverWait(driver, 3).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, ".myAccount"))
            )
            is_logged_flag = driver.find_element_by_css_selector(
                ".myAccount"
            ).get_attribute("innerText")
        except Exception as e:
            print(e)
        if is_logged_flag == "Osvaldo":
            print("\t" + WARNING + "Sesión ya iniciada con anterioridad" + WHITE)
        else:
            try:
                try:
                    try:
                        driver.find_element(By.LINK_TEXT, "Iniciar sesión").click()

                    except Exception as e:
                        print(e)
                        WebDriverWait(driver, 50).until(
                            EC.element_to_be_clickable(
                                (By.CSS_SELECTOR, ".generalOverlay")
                            )
                        )
                        driver.find_element(By.CSS_SELECTOR, ".popupCloseIcon").click()
                        driver.find_element(By.LINK_TEXT, "Iniciar sesión").click()

                except Exception as e:
                    print(e)

                driver.find_element(By.ID, "loginFormUser_email").send_keys(
                    "osvaldo.hdz.m@outlook.com"
                )
                driver.find_element(By.ID, "loginForm_password").send_keys(
                    "Os23valdo1."
                )

                try:
                    time.sleep(3)
                    driver.find_element(By.ID, "loginForm_password").send_keys(
                        Keys.ENTER
                    )
                    time.sleep(2)
                    driver.find_element(By.ID, "loginForm_password").send_keys(
                        Keys.ENTER
                    )
                    time.sleep(1)
                    driver.find_element(By.ID, "loginForm_password").send_keys(
                        Keys.ENTER
                    )
                except Exception as e:
                    print(e)
            except Exception as e:
                print(e)

        print(
            "\t" + WARNING + "Conexión con datos de investing.com establecida" + WHITE
        )
        time.sleep(3)

        WebDriverWait(driver, 50).until(
            EC.element_to_be_clickable((By.XPATH, '//*[@id="portfolioTopBarBtn"]'))
        )
        driver.find_element_by_xpath('//*[@id="portfolioTopBarBtn"]').click()
        WebDriverWait(driver, 50).until(
            EC.element_to_be_clickable(
                (
                    By.CSS_SELECTOR,
                    "#portfolioContainer_25863947 > div:nth-child(1) > a:nth-child(2)",
                )
            )
        )
        driver.find_element(
            By.CSS_SELECTOR,
            "#portfolioContainer_25863947 > div:nth-child(1) > a:nth-child(2)",
        ).click()

        driver.find_element(By.CSS_SELECTOR, "#performance > a:nth-child(1)").click()
        time.sleep(3)
        table = driver.find_element_by_xpath(
            "/html/body/div[5]/section/div[7]/div[7]/div[2]/div[2]/div/table"
        ).get_attribute("outerHTML")
        dfs = pd.read_html(table)
        df1 = dfs[0]
        df1.drop(["Unnamed: 0", "Unnamed: 8"], axis=1, inplace=True, errors="ignore")
        # print(df.columns)

        # df.to_csv('analisis_tecnico_performance.csv',  index=False, header=True, encoding="utf-8")
        driver.find_element(By.CSS_SELECTOR, "#technical > a:nth-child(1)").click()
        time.sleep(3)
        table = driver.find_element_by_xpath(
            "/html/body/div[5]/section/div[7]/div[7]/div[2]/div[2]/div/table"
        ).get_attribute("outerHTML")
        dfs = pd.read_html(table)
        df2 = dfs[0]
        df2.drop(["Unnamed: 0", "Unnamed: 10"], axis=1, inplace=True, errors="ignore")
        # print(df.columns)
        # df.to_csv('analisis_tecnico_algoritmo.csv',index=False, header=True, encoding="utf-8")

        driver.find_element(By.CSS_SELECTOR, "#overview > a:nth-child(1)").click()
        time.sleep(3)
        table = driver.find_element_by_xpath(
            "/html/body/div[5]/section/div[7]/div[7]/div[2]/div[2]/div/div/table"
        ).get_attribute("outerHTML")
        dfs = pd.read_html(table)
        df3 = dfs[0]
        df3.drop(
            ["Unnamed: 0", "Unnamed: 1", "Unnamed: 19", "Unnamed: 20"],
            axis=1,
            inplace=True,
            errors="ignore",
        )
        # print(df.columns)

        # Create dicionary investing simbols portfolio
        WebDriverWait(driver, 50).until(
            EC.element_to_be_clickable(
                (
                    By.XPATH,
                    "/html/body/div[5]/section/div[7]/div[7]/div[2]/div[2]/div/div/table",
                )
            )
        )
        table_elements_simbols = driver.find_elements(
            By.XPATH,
            "/html/body/div[5]/section/div[7]/div[7]/div[2]/div[2]/div/div/table/tbody/tr/td[4]/a",
        )
        dicionary_simbols = {}
        for tag in table_elements_simbols:
            dicionary_simbols[str(tag.get_attribute("innerHTML"))] = str(
                tag.get_attribute("href")
            )

        dfs = [df1, df2, df3]
        dfs = [x.set_index("Nombre") for x in dfs]
        df = pd.concat(dfs, axis=1, keys=range(1, len(dfs) + 1))
        df.columns = df.columns.map("{0[1]}{0[0]}".format)
        df = df.reset_index()
        df["Fecha"] = datetime.now().strftime("%x %X")

        df["Diario1"] = df["Diario1"].map(lambda x: str(x).replace("+-", "+")[:-1])
        df["Diario1"] = df["Diario1"].astype("float")
        df["Semanal1"] = df["Semanal1"].map(lambda x: str(x)[:-1])
        df["Semanal1"] = df["Semanal1"].astype("float")
        df["Mensual1"] = df["Mensual1"].map(lambda x: str(x)[:-1])
        df["Mensual1"] = df["Mensual1"].astype("float")
        df["Anual1"] = df["Anual1"].map(lambda x: str(x)[:-1])
        df["Anual1"] = df["Anual1"].astype("float")
        df["1 Año1"] = df["1 Año1"].map(lambda x: str(x)[:-1])
        df["1 Año1"] = df["1 Año1"].astype("float")
        df["3 años1"] = df["3 años1"].map(lambda x: str(x).replace("-", "0"))
        df["3 años1"] = df["3 años1"].map(lambda x: str(x).replace("%", ""))
        df["3 años1"] = df["3 años1"].astype("float")
        df["% var.3"] = df["% var.3"].map(lambda x: str(x).replace("+-", "+")[:-1])
        df["% var.3"] = df["% var.3"].astype("float")

        print("\t" + WARNING + "Analizando acciones..." + WHITE)

        print(
            "\t"
            + WARNING
            + "Analizando acciones con ganancias en diferentes periodos..."
            + WHITE
        )
        df = df[(df[["3 años1"]] > 0).all(1)]
        df = df[(df[["1 Año1"]] > 0).all(1)]
        df = df[(df[["Anual1"]] > 0).all(1)]
        df = df[(df[["Mensual1"]] > 0).all(1)]
        df = df[(df[["Semanal1"]] > 0).all(1)]
        print(
            "\t"
            + WARNING
            + "Analizando acciones con ganacia diaria menor al promedio..."
            + WHITE
        )
        df = df[(df[["Diario1"]] > 0).all(1)]

        print(
            "\t"
            + WARNING
            + "Analizando acciones con indicadores técnicos en consenso de compra en diferentes periodos..."
            + WHITE
        )
        df = df[
            (df[["Mensual2"]] == "Compra").all(1)
            | (df[["Mensual2"]] == "Compra fuerte").all(1)
        ]
        df = df[
            (df[["Semanal2"]] == "Compra").all(1)
            | (df[["Semanal2"]] == "Compra fuerte").all(1)
        ]
        df = df[
            (df[["Diario2"]] == "Compra").all(1)
            | (df[["Diario2"]] == "Compra fuerte").all(1)
        ]
        print(
            "\t"
            + WARNING
            + "Analizando acciones con indicadores técnicos en baja con estimación de alza..."
            + WHITE
        )
        df = df[
            (df[["5 horas2"]] == "Venta").all(1)
            | (df[["5 horas2"]] == "Venta fuerte").all(1)
            | (df[["5 horas2"]] == "Compra fuerte").all(1)
            | (df[["5 horas2"]] == "Compra").all(1)
        ]
        # df = df[(df[['1 hora2']] == 'Venta').all(1) | (df[['1 hora2']] == 'Venta fuerte').all(1) | (df[['1 hora2']] == 'Compra fuerte').all(1) | (df[['1 hora2']] == 'Compra').all(1)]
        # df = df[(df[['30 minutos2']] == 'Compra').all(1) | (df[['30 minutos2']] == 'Compra fuerte').all(1) | (df[['30 minutos2']] == 'Venta').all(1) | (df[['30 minutos2']] == 'Venta fuerte').all(1)]
        print(
            "\t"
            + WARNING
            + "Analizando acciones con indicadores técnicos con estimación de alza durante la sesión..."
            + WHITE
        )
        # df = df[(df[['15 minutos2']] == 'Compra').all(1) | (df[['15 minutos2']] == 'Compra fuerte').all(1) | (df[['15 minutos2']] == 'Neutral').all(1) | (df[['15 minutos2']] == 'Venta fuerte').all(1) | (df[['15 minutos2']] == 'Venta').all(1) ]
        # df = df[(df[['5 minutos2']] == 'Compra').all(1) | (df[['5 minutos2']] == 'Compra fuerte').all(1) | (df[['5 minutos2']] == 'Neutral').all(1) | (df[['5 minutos2']] == 'Venta fuerte').all(1) | (df[['5 minutos2']] == 'Venta').all(1)]

        print("\t" + WARNING + "Calculando precios y ganancias estimadas..." + WHITE)
        for x in df["Símbolo3"]:
            print("\t" + WARNING + "Analizando {} ...".format(x) + WHITE)

            if containsAny(dicionary_simbols[x], ["?"]):
                technical_data_url = insert_string_before(
                    dicionary_simbols[x], "-technical", "?"
                )
            else:
                technical_data_url = dicionary_simbols[x] + "-technical"
            driver.get(technical_data_url)
            WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable(
                    (By.XPATH, "/html/body/div[5]/section/div[8]/ul/li[6]/a")
                )
            )
            driver.find_element(
                By.XPATH, "/html/body/div[5]/section/div[8]/ul/li[6]/a"
            ).click()

            WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable(
                    (By.XPATH, "/html/body/div[5]/section/div[10]/table")
                )
            )
            p1 = float(
                driver.find_element(
                    By.XPATH,
                    "/html/body/div[5]/section/div[10]/table/tbody/tr[2]/td[6]",
                ).get_attribute("innerHTML")
            )
            p2 = float(
                driver.find_element(
                    By.XPATH,
                    "/html/body/div[5]/section/div[10]/table/tbody/tr[2]/td[7]",
                ).get_attribute("innerHTML")
            )
            p3 = float(
                driver.find_element(
                    By.XPATH,
                    "/html/body/div[5]/section/div[10]/table/tbody/tr[2]/td[8]",
                ).get_attribute("innerHTML")
            )
            pe = round((p1 + p2 + p3) / 3, 2)
            df.loc[df["Símbolo3"] == x, "PeEstimado"] = pe
            df["ML Prediction"] = "NO DISPONIBLE"
            try:
                df.loc[df["Símbolo3"] == x, "ML Prediction"] = predict_machine_daily(x)
            except Exception as e:
                print(e)

        df = df[(df[["ML Prediction"]] == "Compra").all(1)]
        df["GanEstimada %"] = round(
            percentage_change(df["Último3"], df["PeEstimado"]), 2
        )
        df = df[(df[["GanEstimada %"]] > 0.5).all(1)]
        df["PeVentaCalc"] = round(df["Último3"] * 1.005, 2)

        print("\n\t" + OKGREEN + "Resultado de sugerencias:" + WHITE)
        print(
            df[
                [
                    "Símbolo3",
                    "Nombre",
                    "Último3",
                    "PeEstimado",
                    "GanEstimada %",
                    "PeVentaCalc",
                    "ML Prediction",
                ]
            ]
        )

        opcion = input(
            "Deseas ejecutar la optimización de portafolio con estas acciones? (y/n) >> "
        )

        while True:
            if opcion == "y" or opcion == "Y":
                tickers = df["Símbolo3"].astype(str).values.tolist()
                optimal_capital = calculate_optimal_capital(tickers)
                df_allocations = markovitz_portfolio_optimization(tickers, optimal_capital)
                df_result = pd.merge(
                    df_allocations, df, left_on="Ticker", right_on="Símbolo3"
                )
                df_result["Títulos"] = df_result["Allocation $"] / df_result["Último3"]
                df_result["Títulos"] = df_result["Títulos"].astype("int32")
                print("\n\t" + OKGREEN + "Resultado de optimización:" + WHITE)
                print(
                    df_result[
                        [
                            "Símbolo3",
                            "Nombre",
                            "Títulos",
                            "Allocation $",
                            "Último3",
                            "PeEstimado",
                            "GanEstimada %",
                            "PeVentaCalc",
                        ]
                    ]
                )

                break
            elif opcion == "n" or opcion == "N":
                break

    except Exception as e:
        print(e)


def portfolio_annualised_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns * weights) * 252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return std, returns


def random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate):
    results = np.zeros((3, num_portfolios))
    weights_record = []
    for i in range(num_portfolios):
        weights = np.random.random(4)
        weights /= np.sum(weights)
        weights_record.append(weights)
        portfolio_std_dev, portfolio_return = portfolio_annualised_performance(
            weights, mean_returns, cov_matrix
        )
        results[0, i] = portfolio_std_dev
        results[1, i] = portfolio_return
        results[2, i] = (portfolio_return - risk_free_rate) / portfolio_std_dev
    return results, weights_record


def display_simulated_ef_with_random(
    mean_returns, cov_matrix, num_portfolios, risk_free_rate, stocks_data
):
    results, weights = random_portfolios(
        num_portfolios, mean_returns, cov_matrix, risk_free_rate
    )

    max_sharpe_idx = np.argmax(results[2])
    sdp, rp = results[0, max_sharpe_idx], results[1, max_sharpe_idx]
    max_sharpe_allocation = pd.DataFrame(
        weights[max_sharpe_idx], index=stocks_data.columns, columns=["allocation"]
    )
    max_sharpe_allocation.allocation = [
        round(i * 100, 2) for i in max_sharpe_allocation.allocation
    ]
    max_sharpe_allocation = max_sharpe_allocation.T

    min_vol_idx = np.argmin(results[0])
    sdp_min, rp_min = results[0, min_vol_idx], results[1, min_vol_idx]
    min_vol_allocation = pd.DataFrame(
        weights[min_vol_idx], index=stocks_data.columns, columns=["allocation"]
    )
    min_vol_allocation.allocation = [
        round(i * 100, 2) for i in min_vol_allocation.allocation
    ]
    min_vol_allocation = min_vol_allocation.T

    print("------------------------")
    print("Maximum Sharpe Ratio Portfolio Allocation\n")
    print("Annualised Return:", round(rp, 2))
    print("Annualised Volatility:", round(sdp, 2))
    print("\n")
    print(max_sharpe_allocation)
    print("------------------------")
    print("Minimum Volatility Portfolio Allocation\n")
    print("Annualised Return:", round(rp_min, 2))
    print("Annualised Volatility:", round(sdp_min, 2))
    print("\n")
    print(min_vol_allocation)

    plt.figure(figsize=(10, 7))
    plt.scatter(
        results[0, :],
        results[1, :],
        c=results[2, :],
        cmap="YlGnBu",
        marker="o",
        s=10,
        alpha=0.3,
    )
    plt.colorbar()
    plt.scatter(sdp, rp, marker="*", color="r", s=500, label="Maximum Sharpe ratio")
    plt.scatter(
        sdp_min, rp_min, marker="*", color="g", s=500, label="Minimum volatility"
    )
    plt.title("Simulated Portfolio Optimization based on Efficient Frontier")
    plt.xlabel("annualised volatility")
    plt.ylabel("annualised returns")
    plt.legend(labelspacing=0.8)


def neg_sharpe_ratio(
    weights,
    mean_returns,
    cov_matrix,
    risk_free_rate,
):
    p_var, p_ret = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
    return -(p_ret - risk_free_rate) / p_var


def max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
    bound = (0.0, 1.0)
    bounds = tuple(bound for asset in range(num_assets))
    result = sco.minimize(
        neg_sharpe_ratio,
        num_assets
        * [
            1.0 / num_assets,
        ],
        args=args,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    return result


def apply_max_allocation_limit(allocation_data, max_per_ticker=None):
    """
    Aplica el límite máximo configurado por ticker a cualquier asignación de capital
    """
    if max_per_ticker is None:
        max_per_ticker = MAX_ALLOCATION_PER_TICKER
    
    if isinstance(allocation_data, pd.DataFrame):
        if "Allocation $" in allocation_data.columns:
            # Aplicar límite a la columna de asignación en $
            allocation_data["Allocation $"] = allocation_data["Allocation $"].apply(
                lambda x: min(float(str(x).replace(",", "")), max_per_ticker) if pd.notna(x) else 0
            )
        elif "Inversión Total" in allocation_data.columns:
            # Para la función de volatilidad
            allocation_data["Inversión Total"] = allocation_data["Inversión Total"].apply(
                lambda x: min(x, max_per_ticker) if pd.notna(x) else 0
            )
    return allocation_data

def calculate_optimal_capital(tickers, base_capital_high=None, base_capital_low=None, max_per_action=None):
    """
    Calcula el capital óptimo basado en el número de tickers:
    - Más de 3 tickers: CAPITAL_BASE_HIGH
    - 3 o menos tickers: CAPITAL_BASE_LOW
    - Ajusta automáticamente si alguna acción superaría MAX_ALLOCATION_PER_TICKER
    - NUNCA permite que una acción individual supere MAX_ALLOCATION_PER_TICKER
    """
    # Usar variables globales si no se especifican parámetros
    if base_capital_high is None:
        base_capital_high = CAPITAL_BASE_HIGH
    if base_capital_low is None:
        base_capital_low = CAPITAL_BASE_LOW
    if max_per_action is None:
        max_per_action = MAX_ALLOCATION_PER_TICKER
    
    num_tickers = len(tickers) if isinstance(tickers, list) else 1
    
    if num_tickers > 3:
        capital = base_capital_high
    else:
        capital = base_capital_low
    
    # Ajustar si alguna acción individual superaría el máximo
    max_possible_capital = num_tickers * max_per_action
    if capital > max_possible_capital:
        capital = max_possible_capital
    
    # Asegurar que el capital por ticker no exceda el máximo
    capital_per_ticker = capital / num_tickers
    if capital_per_ticker > max_per_action:
        capital = num_tickers * max_per_action
    
    return capital

def portfolio_volatility(weights, mean_returns, cov_matrix):
    return portfolio_annualised_performance(weights, mean_returns, cov_matrix)[0]


def min_variance(mean_returns, cov_matrix):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
    bound = (0.0, 1.0)
    bounds = tuple(bound for asset in range(num_assets))

    result = sco.minimize(
        portfolio_volatility,
        num_assets
        * [
            1.0 / num_assets,
        ],
        args=args,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    return result


def portfolio_return(weights):
    return portfolio_annualised_performance(weights, mean_returns, cov_matrix)[1]


def efficient_return(mean_returns, cov_matrix, target):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)

    constraints = (
        {"type": "eq", "fun": lambda x: portfolio_return(x) - target},
        {"type": "eq", "fun": lambda x: np.sum(x) - 1},
    )
    bounds = tuple((0, 1) for asset in range(num_assets))
    result = sco.minimize(
        portfolio_volatility,
        num_assets
        * [
            1.0 / num_assets,
        ],
        args=args,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    return result


def efficient_frontier(mean_returns, cov_matrix, returns_range):
    efficients = []
    for ret in returns_range:
        efficients.append(efficient_return(mean_returns, cov_matrix, ret))
    return efficients


def display_calculated_ef_with_random(
    mean_returns, cov_matrix, num_portfolios, risk_free_rate, stocks_data
):
    results, _ = random_portfolios(
        num_portfolios, mean_returns, cov_matrix, risk_free_rate
    )

    max_sharpe = max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate)
    sdp, rp = portfolio_annualised_performance(
        max_sharpe["x"], mean_returns, cov_matrix
    )
    max_sharpe_allocation = pd.DataFrame(
        max_sharpe.x, index=stocks_data.columns, columns=["allocation"]
    )
    max_sharpe_allocation.allocation = [
        round(i * 100, 2) for i in max_sharpe_allocation.allocation
    ]
    max_sharpe_allocation = max_sharpe_allocation.T
    max_sharpe_allocation

    min_vol = min_variance(mean_returns, cov_matrix)
    sdp_min, rp_min = portfolio_annualised_performance(
        min_vol["x"], mean_returns, cov_matrix
    )
    min_vol_allocation = pd.DataFrame(
        min_vol.x, index=stocks_data.columns, columns=["allocation"]
    )
    min_vol_allocation.allocation = [
        round(i * 100, 2) for i in min_vol_allocation.allocation
    ]
    min_vol_allocation = min_vol_allocation.T

    print("------------------------")
    print("Maximum Sharpe Ratio Portfolio Allocation\n")
    print("Annualised Return:", round(rp, 2))
    print("Annualised Volatility:", round(sdp, 2))
    print("\n")
    print(max_sharpe_allocation)
    print("------------------------")
    print("Minimum Volatility Portfolio Allocation\n")
    print("Annualised Return:", round(rp_min, 2))
    print("Annualised Volatility:", round(sdp_min, 2))
    print("\n")
    print(min_vol_allocation)

    plt.figure(figsize=(10, 7))
    plt.scatter(
        results[0, :],
        results[1, :],
        c=results[2, :],
        cmap="YlGnBu",
        marker="o",
        s=10,
        alpha=0.3,
    )
    plt.colorbar()
    plt.scatter(sdp, rp, marker="*", color="r", s=500, label="Maximum Sharpe ratio")
    plt.scatter(
        sdp_min, rp_min, marker="*", color="g", s=500, label="Minimum volatility"
    )

    target = np.linspace(rp_min, 0.32, 50)
    efficient_portfolios = efficient_frontier(mean_returns, cov_matrix, target)
    plt.plot(
        [p["fun"] for p in efficient_portfolios],
        target,
        linestyle="-.",
        color="black",
        label="efficient frontier",
    )
    plt.title("Calculated Portfolio Optimization based on Efficient Frontier")
    plt.xlabel("annualised volatility")
    plt.ylabel("annualised returns")
    plt.legend(labelspacing=0.8)


def display_ef_with_selected(mean_returns, cov_matrix, risk_free_rate, stocks_data):
    print("A")
    max_sharpe = max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate)
    sdp, rp = portfolio_annualised_performance(
        max_sharpe["x"], mean_returns, cov_matrix
    )
    max_sharpe_allocation = pd.DataFrame(
        max_sharpe.x, index=stocks_data.columns, columns=["allocation"]
    )
    max_sharpe_allocation.allocation = [
        round(i * 100, 2) for i in max_sharpe_allocation.allocation
    ]
    max_sharpe_allocation = max_sharpe_allocation.T
    print("B")

    min_vol = min_variance(mean_returns, cov_matrix)
    sdp_min, rp_min = portfolio_annualised_performance(
        min_vol["x"], mean_returns, cov_matrix
    )
    min_vol_allocation = pd.DataFrame(
        min_vol.x, index=stocks_data.columns, columns=["allocation"]
    )
    min_vol_allocation.allocation = [
        round(i * 100, 2) for i in min_vol_allocation.allocation
    ]
    min_vol_allocation = min_vol_allocation.T
    print("C")

    an_vol = np.std(returns) * np.sqrt(252)
    an_rt = mean_returns * 252

    print("------------------------")
    print("Maximum Sharpe Ratio Portfolio Allocation\n")
    print("Annualised Return:", round(rp, 2))
    print("Annualised Volatility:", round(sdp, 2))
    print("\n")
    print(max_sharpe_allocation)
    print("------------------------")
    print("Minimum Volatility Portfolio Allocation\n")
    print("Annualised Return:", round(rp_min, 2))
    print("Annualised Volatility:", round(sdp_min, 2))
    print("\n")
    print(min_vol_allocation)
    print("------------------------")
    print("Individual Stock Returns and Volatility\n")
    for i, txt in enumerate(table.columns):
        print(
            txt,
            ":",
            "annuaised return",
            round(an_rt[i], 2),
            ", annualised volatility:",
            round(an_vol[i], 2),
        )
    print("------------------------")

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(an_vol, an_rt, marker="o", s=200)

    for i, txt in enumerate(table.columns):
        ax.annotate(
            txt, (an_vol[i], an_rt[i]), xytext=(10, 0), textcoords="offset points"
        )
    ax.scatter(sdp, rp, marker="*", color="r", s=500, label="Maximum Sharpe ratio")
    ax.scatter(
        sdp_min, rp_min, marker="*", color="g", s=500, label="Minimum volatility"
    )

    target = np.linspace(rp_min, 0.34, 50)
    efficient_portfolios = efficient_frontier(mean_returns, cov_matrix, target)
    ax.plot(
        [p["fun"] for p in efficient_portfolios],
        target,
        linestyle="-.",
        color="black",
        label="efficient frontier",
    )
    ax.set_title("Portfolio Optimization with Individual Stocks")
    ax.set_xlabel("annualised volatility")
    ax.set_ylabel("annualised returns")
    ax.legend(labelspacing=0.8)



# Función para calcular la Razón de Sharpe ajustada para corto plazo
def calculate_sharpe_ratio(returns, risk_free_rate=0.01):
    # Calcular el promedio de retornos y la volatilidad
    mean_return = returns.mean()
    std_dev = returns.std()
    # Si la volatilidad es 0, evitamos dividir por cero
    if std_dev == 0:
        return 0
    sharpe_ratio = (mean_return - risk_free_rate) / std_dev
    return sharpe_ratio

# Función principal para la optimización de portafolio en corto plazo
def portfolio_optimization_sharpe_short_term(tickers, total_mount, initial_date, max_percentaje=45, short_term_days=7):
    stocks_data = pd.DataFrame()
    last_prices = {}
    inicio = initial_date
    fin = datetime.today().strftime("%Y-%m-%d")
    short_term_window = short_term_days  # Ventana de tiempo corta (en días)

    for ticker in tickers:
        try:
            series = download_prices_any_listing(ticker, start=inicio, end=fin)
            if not series.empty:
                stocks_data[ticker] = series
                last_prices[ticker] = float(series.iloc[-1])
            else:
                raise Exception(f"No hay datos disponibles para el ticker {ticker}")
        except Exception as e:
            print(f"Error al obtener datos para el ticker {ticker}: {e}")

    allocation_dataframe = pd.DataFrame()

    try:
        # Obtener los retornos logarítmicos diarios
        returns = stocks_data.pct_change().dropna()

        # Calcular razón de Sharpe ajustada para corto plazo
        short_term_returns = returns.tail(short_term_window)  # Últimos 'n' días
        sharpe_ratios = short_term_returns.apply(calculate_sharpe_ratio)

        # Convertir negativos a 0 y re-normalizar
        sharpe_ratios = sharpe_ratios.clip(lower=0)
        total = sharpe_ratios.sum()
        if total <= 0:
            # fallback uniforme si todo es 0 o NaN
            sharpe_weights = pd.Series(
                [1.0 / len(sharpe_ratios)] * len(sharpe_ratios), index=sharpe_ratios.index
            )
        else:
            sharpe_weights = sharpe_ratios / total

        # Aplicar tope máximo por activo y re-normalizar
        sharpe_weights = sharpe_weights.clip(upper=max_percentaje / 100.0)
        total = sharpe_weights.sum()
        if total > 0:
            sharpe_weights = sharpe_weights / total

        allocation_dataframe = pd.DataFrame({
            "Ticker": sharpe_weights.index,
            "Allocation %": sharpe_weights.values
        })

    except Exception as e:
        print(e)
        print("Algo salió mal, distribuyendo de manera uniforme...")
        allocation_dataframe = pd.DataFrame(stocks_data.columns, columns=["Ticker"])
        tickers_count = len(allocation_dataframe.index)
        if tickers_count > 0:
            allocation_dataframe["Allocation %"] = 1 / tickers_count
        else:
            print("No hay datos válidos disponibles para distribuir.")

    # Convertir porcentaje de asignación a escala 100%
    allocation_dataframe["Allocation %"] = (allocation_dataframe["Allocation %"] * 100).round(2)

    # Calcular la cantidad asignada en $ a cada ticker
    allocation_dataframe["Allocation $"] = (
        allocation_dataframe["Allocation %"] * float(total_mount) / 100
    ).round(2)
    
    # Aplicar límite de $400,000 por ticker
    max_allocation_per_ticker = MAX_ALLOCATION_PER_TICKER
    allocation_dataframe["Allocation $"] = allocation_dataframe["Allocation $"].apply(
        lambda x: min(x, max_allocation_per_ticker)
    )
    
    # Recalcular porcentajes después de aplicar el límite
    total_allocated = allocation_dataframe["Allocation $"].sum()
    if total_allocated > 0:
        allocation_dataframe["Allocation %"] = (
            (allocation_dataframe["Allocation $"] / total_allocated) * 100
        ).round(2)

    # Añadir los precios de cierre recientes
    allocation_dataframe["LastPrice $"] = (
        allocation_dataframe["Ticker"].map(last_prices)
    ).round(2)

    # Calcular el número de títulos a comprar
    allocation_dataframe["TitlesNum"] = (
        allocation_dataframe["Allocation $"] / allocation_dataframe["LastPrice $"]
    ).fillna(0)
    allocation_dataframe.loc[allocation_dataframe["TitlesNum"] < 0, "TitlesNum"] = 0
    allocation_dataframe["TitlesNum"] = allocation_dataframe["TitlesNum"].astype(int)

    # Formatear los montos de dinero con comas
    allocation_dataframe["Allocation $"] = allocation_dataframe.apply(
        lambda x: "{:,}".format(x["Allocation $"]), axis=1
    )

    return allocation_dataframe


def markovitz_portfolio_optimization(tickers, total_mount, initial_date, max_percentaje=45):
    stocks_data = pd.DataFrame()
    inicio = initial_date
    fin = datetime.today().strftime("%Y-%m-%d")
    last_prices = {}

    for ticker in tickers:
        try:
            series = download_prices_any_listing(ticker, start=inicio, end=fin)
            if not series.empty:
                stocks_data[ticker] = series
                last_prices[ticker] = float(series.iloc[-1])
            else:
                raise Exception(f"No data available for the ticker {ticker}")
        except Exception as e:
            print(f"Error al obtener los datos para el ticker {ticker}")
            print(e)

    allocation_dataframe = pd.DataFrame()

    try:
        # Calculate expected returns and sample covariance
        mu = expected_returns.mean_historical_return(stocks_data)
        S = risk_models.sample_cov(stocks_data)

        # Optimize for maximal Sharpe ratio
        ef = EfficientFrontier(mu, S)
        raw_weights = ef.max_sharpe()
        cleaned_weights = ef.clean_weights()

        ef.portfolio_performance(verbose=False)
        allocation_dataframe = pd.DataFrame(
            cleaned_weights.items(), columns=["Ticker", "Allocation %"]
        )
    except Exception as e:
        print(e)
        print("\nSomething went wrong, making uniform distribution...")
        allocation_dataframe = pd.DataFrame(stocks_data.columns, columns=["Ticker"])
        tickers_count = len(allocation_dataframe.index)
        if tickers_count > 0:
            allocation_dataframe["Allocation %"] = 1 / tickers_count
        else:
            print("No valid data available to distribute.")

    allocation_dataframe["Allocation %"] = allocation_dataframe["Allocation %"] * 100
    allocation_dataframe["Allocation %"] = (
        allocation_dataframe["Allocation %"].apply(
            lambda x: max_percentaje if x > max_percentaje else x
        )
    ).round(2)
    
    # Calcular asignación en $ con límite configurado por ticker
    max_allocation_per_ticker = MAX_ALLOCATION_PER_TICKER
    allocation_dataframe["Allocation $"] = (
        allocation_dataframe["Allocation %"] * float(total_mount) / 100
    ).round(2)
    
    # Aplicar límite de $400,000 por ticker
    allocation_dataframe["Allocation $"] = allocation_dataframe["Allocation $"].apply(
        lambda x: min(x, max_allocation_per_ticker)
    )
    
    # Recalcular porcentajes después de aplicar el límite
    total_allocated = allocation_dataframe["Allocation $"].sum()
    if total_allocated > 0:
        allocation_dataframe["Allocation %"] = (
            (allocation_dataframe["Allocation $"] / total_allocated) * 100
        ).round(2)
    allocation_dataframe["LastPrice $"] = (
        allocation_dataframe["Ticker"].map(last_prices)
    ).round(2)
    allocation_dataframe["TitlesNum"] = (
        allocation_dataframe["Allocation $"] / allocation_dataframe["LastPrice $"]
    )

    allocation_dataframe["TitlesNum"] = (
        allocation_dataframe["TitlesNum"].fillna(0).astype(int)
    )
    allocation_dataframe["Allocation $"] = allocation_dataframe.apply(
        lambda x: "{:,}".format(x["Allocation $"]), axis=1
    )

    return allocation_dataframe


def portfolio_optimization(tickers):
    mount = input("Escribe el monto >> ")
    driver.get("https://www.portfoliovisualizer.com/optimize-portfolio")
    driver.find_element(By.CSS_SELECTOR, "#timePeriod_chosen span").click()
    driver.find_element(By.CSS_SELECTOR, ".active-result:nth-child(1)").click()
    driver.find_element(By.CSS_SELECTOR, "#startYear_chosen span").click()
    driver.find_element(By.CSS_SELECTOR, "#startYear_chosen .chosen-results").click()
    driver.find_element(By.CSS_SELECTOR, "#robustOptimization_chosen span").click()
    driver.find_element(
        By.CSS_SELECTOR, "#robustOptimization_chosen .active-result:nth-child(2)"
    ).click()
    index = 1
    for ticker in tickers:
        driver.find_element(By.ID, "symbol" + str(index)).send_keys(ticker)
        index = index + 1
    driver.find_element(By.ID, "submitButton").click()
    table = driver.find_element(
        By.XPATH, "/html/body/div[2]/div[3]/div[1]/div[1]/div[1]/table"
    ).get_attribute("outerHTML")
    dfs = pd.read_html(table)
    df1 = dfs[0].iloc[:-1]
    print("\n")
    df1.reset_index()
    df1["Allocation %"] = df1["Allocation %"].map(lambda x: str(x).replace("%", ""))
    df1["Allocation %"] = df1["Allocation %"].astype("float")
    df1["Allocation $"] = (df1["Allocation %"] / 100) * float(mount)
    return df1


def fetch_google_news(driver, company_name, ticker):
    google_news_url = f"https://www.google.com/search?hl=en-US&q={company_name}&tbm=nws"
    print(google_news_url)

    driver.get(google_news_url)

    # Esperar hasta que los resultados de noticias se carguen
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, '//*[@id="rso"]'))
    )

    headlines = []

    # Obtener todos los elementos de noticias usando el XPath especificado
    news_items = driver.find_elements(
        By.XPATH, '//*[@id="rso"]/div/div/div/div/div/a/div/div[2]/div[2]'
    )

    # Iterar sobre cada elemento de noticia
    for item in news_items:
        try:
            title = item.text.strip()  # Obtener el texto del elemento
            if title:  # Verificar que no esté vacío
                headlines.append(
                    {"title": title}
                )  # Agregar el título a la lista de titulares
        except Exception as e:
            print(f"Error al obtener el título: {e}")

    return headlines


def news_analysis(tickers=["AAPL", "TSLA", "AMZN"]):
    console = Console()
    input_tickers = input(
        "Ingresa las acciones separadas por comas (i.e. OMAB,AAPL,META,MSFT): "
    )

    # Usar los tickers ingresados si no están vacíos
    if input_tickers.strip():
        tickers = [ticker.strip() for ticker in input_tickers.split(",")]
    else:
        console.print(
            f"\n[bold yellow] No se detectó entrada, usando valores sugeridos por defecto...[/bold yellow]"
        )

    sia = SentimentIntensityAnalyzer()
    driver = configure_chrome_headless_driver_no_profile()
    n = 3  # Número de titulares de artículos mostrados por ticker
    summary = {
        ticker: {"Investing": [], "Finviz": [], "Yahoo": []} for ticker in tickers
    }  # Resumen de resultados

    # Análisis de noticias desde Investing.com
    for ticker in tickers:
        stock = yf.Ticker(ticker)  # Obtener la información de la acción
        company_name = (
            stock.info.get("shortName", ticker)
            .split(",")[0]
            .replace(".com", "")
            .replace(".", "")
            .split("(")[0]
            .replace("PE&OLES", "PEÑOLES")
        )
        print(f"\nRecent News Headlines for {company_name}: ")

        try:
            driver.execute_cdp_cmd(
                "Page.addScriptToEvaluateOnNewDocument",
                {"source": "setTimeout(function(){window.stop();}, 3000);"},
            )
            url = "https://mx.investing.com/search/?q=" + company_name + "&tab=news"
            print(url)
            driver.get(url)
            headlines = []  # Lista para almacenar los titulares

            for x in range(1, n + 1):
                retry_count = 0
                while retry_count < 3:  # Reintentos
                    try:
                        headline_element = driver.find_element(
                            By.XPATH,
                            f'//*[@id="fullColumn"]/div/div[4]/div[3]/div/div[{x}]/div/p',
                        )
                        time_element = driver.find_element(
                            By.XPATH,
                            f'//*[@id="fullColumn"]/div/div[4]/div[3]/div/div[{x}]/div/div/time',
                        )
                        headline_text = headline_element.get_attribute("innerHTML")
                        time_text = time_element.get_attribute("innerHTML")

                        print(f"{headline_text} ({time_text})")
                        headlines.append(headline_text)
                        break

                    except (TimeoutException, NoSuchElementException):
                        retry_count += 1
                        time.sleep(0.12)
                else:
                    print(
                        f"No se pudieron obtener los titulares después de varios intentos para {ticker}."
                    )
                    break

            print(f"Titulares recopilados para {ticker}: {headlines}")

            # Análisis de sentimientos
            print(f"\nAnálisis de sentimientos para {company_name}:")
            for headline in headlines:
                sentiment = sia.polarity_scores(headline)
                print(f"{headline}: {sentiment}")
                summary[ticker]["Investing"].append(
                    {"headline": headline, "sentiment": sentiment}
                )

        except Exception as e:
            print(f"Ocurrió un error en Investing.com para {ticker}: {e}")

    # Análisis de noticias desde Finviz
    finwiz_url = "https://finviz.com/quote.ashx?t="

    for ticker in tickers:
        if ticker == "PE&OLES":
            print(f"Saliendo del procesamiento para el ticker {ticker}.")
            continue  # Saltar al siguiente ticker
        try:
            url = finwiz_url + ticker.replace(
                ".MX", ""
            )  # Elimina '.MX' del ticker si está presente
            print(f"Procesando {ticker} en {url}")

            driver.execute_cdp_cmd(
                "Page.addScriptToEvaluateOnNewDocument",
                {"source": "setTimeout(function(){window.stop();}, 3000);"},
            )
            driver.get(url)

            news_table = WebDriverWait(driver, 1).until(
                EC.presence_of_element_located((By.ID, f"news-table"))
            )

            news_rows = news_table.find_elements(By.TAG_NAME, "tr")

            print(f"\nRecent News Headlines from Finviz for {ticker}: ")
            for i, row in enumerate(news_rows):
                a_text = row.find_element(By.TAG_NAME, "a").text
                print(a_text, "(Today)")
                summary[ticker]["Finviz"].append({"headline": a_text})
                if i == n - 1:
                    break

        except Exception as e:
            print(f"Ocurrió un error al procesar el ticker {ticker} en Finviz: {e}")
            print("Pasando al siguiente ticker...\n")
            continue

    # Análisis de noticias desde Yahoo News
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            company_name = stock.info.get("shortName", ticker)
            print(f"\nRecent News Headlines from Yahoo News for {company_name}: ")
            yahoo_url = f"https://news.search.yahoo.com/search?p={company_name}"
            print(yahoo_url)
            driver.get(yahoo_url)

            headlines = []

            for x in range(1, n + 1):
                retry_count = 0
                while retry_count < 3:
                    try:
                        headline_element = WebDriverWait(driver, 1).until(
                            EC.presence_of_element_located(
                                (By.XPATH, f'//*[@id="web"]/ol/li[{x}]/div/ul/li/h4/a')
                            )
                        )
                        headline_text = headline_element.text

                        print(headline_text)
                        headlines.append(headline_text)
                        break

                    except (TimeoutException, NoSuchElementException):
                        retry_count += 1
                        time.sleep(2)
                else:
                    print(
                        f"No se pudieron obtener los titulares después de varios intentos para {ticker}."
                    )
                    break

            print(f"\nAnálisis de sentimientos para Yahoo News {company_name}:")
            for headline in headlines:
                sentiment = sia.polarity_scores(headline)
                print(f"{headline}: {sentiment}")
                summary[ticker]["Yahoo"].append(
                    {"headline": headline, "sentiment": sentiment}
                )

        except Exception as e:
            print(f"Ocurrió un error en Yahoo News para {ticker}: {e}")

    # Análisis de noticias desde Google News
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        company_name = stock.info.get("shortName", ticker)
        print(f"\nRecent News Headlines from Google News for {company_name}: ")
        headlines = fetch_google_news(driver, company_name, ticker)
        if headlines:
            for headline_data in headlines[:n]:
                try:
                    title = headline_data["title"]
                    sentiment = sia.polarity_scores(str(title))
                    print(f"Título: {title} - Sentimiento: {sentiment}")
                    summary[ticker]["Google"].append(
                        {"headline": title, "sentiment": sentiment}
                    )
                except Exception as e:
                    print(
                        f"Ocurrió un error al buscar noticias de Google News para {ticker}: {e}"
                    )
        else:
            print(f"No se encontraron noticias recientes en Google News para {ticker}")

    driver.quit()

    # Crear una tabla con los resultados promedio
    table = Table(title="Resumen de Sentimientos Promedio", title_style="bold blue")
    table.add_column("Ticker", justify="center", style="cyan", no_wrap=True)
    table.add_column("Average Sentiment", justify="right", style="green")

    filtered_tickers = []
    negative = []
    notfound_tickers = []

    print(f"\nResumen de sentimientos en noticias:\n")

    for ticker in tickers:
        if ticker in summary:
            sentiments = []

            for source in summary[ticker]:
                if source in ["Investing", "Yahoo", "Finviz"]:
                    for item in summary[ticker][source]:
                        if "sentiment" in item and "compound" in item["sentiment"]:
                            sentiments.append(item["sentiment"]["compound"])

            if sentiments:
                average_sentiment = sum(sentiments) / len(sentiments)
                print(
                    f"Sentimientos para {ticker}: {sentiments} (Promedio de reputación: {average_sentiment:.2f})"
                )

                if average_sentiment >= -0.5:
                    filtered_tickers.append(ticker)  # Reputación aceptable
                elif average_sentiment < -0.5:
                    negative.append(ticker)  # Reputación negativa
            else:
                notfound_tickers.append(ticker)
        else:
            notfound_tickers.append(ticker)

    # Mostrar los resultados
    print("\nAcciones analizadas:", ", ".join(tickers))
    print("\nAcciones con reputación negativa:\n", ", ".join(negative))
    print("\nAcciones con reputación aceptable:\n", ", ".join(filtered_tickers))
    print("\nAcciones sin noticias encontradas:\n", ", ".join(notfound_tickers))
    print(
        f"\n[bold yellow]Acciones recomendadas a considerar en movimientos:\n {', '.join(filtered_tickers + notfound_tickers)} [/bold yellow]\n"
    )


def analysis_result():
    print("\n\n\t" + OKGREEN + "RESULTADO DE ANALISIS:")

    df = pd.read_csv("analisis_tecnico_algoritmo.csv")
    print(OKGREEN + df)
    df = df.loc[
        (df["30 Minutes"] == "Strong Buy")
        & (df["5 Minutes"] == "Strong Buy")
        & (df["15 Minutes"] == "Strong Buy")
        & (df["Hourly"] == "Strong Buy")
        & (df["Daily"] == "Strong Buy")
        & (df["Weekly"] == "Strong Buy")
        & (df["Monthly"] == "Strong Buy")
    ].copy()
    print(df)
    names = df["Name"].tolist()
    df = df[["Name"]]
    df["Precio estimado de compra $"] = "-"
    df["Precio estimado de venta $"] = "-"
    df["Variacion estimada %"] = "-"
    df["StockName"] = "-"
    df["Datetime"] = datetime.now().strftime("%x %X")

    index_for = 0
    for stock_name in names:
        time.sleep(3)
        driver.get("https://mx.investing.com/search/?q=" + stock_name)
        driver.find_element_by_xpath(
            "/html/body/div[5]/section/div/div[2]/div[2]/div[2]"
        ).click()
        time.sleep(3)
        driver.find_element_by_xpath(
            '/html/body/div[5]/section/div/div[3]/div[3]/div/*/span[contains(text(), "México")]'
        ).click()
        stock_name_found = (
            driver.find_element_by_xpath("/html/body/div[5]/section/div[7]/h2 ")
            .get_attribute("innerText")
            .replace("Panorama ", "")
        )
        print(stock_name_found)

        driver.find_element_by_xpath(
            '//*[@id="pairSublinksLevel1"]/*/a[contains(text(), "Técnico")]'
        ).click()
        time.sleep(3)
        driver.find_element_by_xpath('//*[@id="timePeriodsWidget"]/li[6]').click()
        time.sleep(2)
        table = driver.find_element_by_xpath(
            "/html/body/div[5]/section/div[10]/table"
        ).get_attribute("outerHTML")
        dfsb = pd.read_html(table)
        dfb = dfsb[0]
        print(dfb)
        expected_price_buy = (float(dfb.at[0, "S2"]) + float(dfb.at[0, "S1"])) / 2
        print(format(expected_price_buy, ".2f"))
        expected_price_sell = (float(dfb.at[0, "R2"]) + float(dfb.at[0, "R1"])) / 2
        print(format(expected_price_sell, ".2f"))
        expected_var = ((expected_price_sell / expected_price_buy) - 1) * 100
        print(format(expected_var, ".2f"))

        df.iat[index_for, 1] = format(expected_price_buy, ".2f")
        df.iat[index_for, 2] = format(expected_price_sell, ".2f")
        df.iat[index_for, 3] = format(expected_var, ".2f")
        df.iat[index_for, 4] = stock_name_found
        index_for += 1

    print(OKGREEN + df)
    print("\n" + WHITE)
    f = open("result.html", "w")
    a = df.to_html()
    f.write(a)
    f.close()

    pdfkit.from_file("result.html", "result.pdf")


def retrieve_top_reto():
    print("\t" + WARNING + "Accediendo a pulso del reto..." + WHITE)
    time.sleep(3)
    driver.get("https://www.retoactinver.com/RetoActinver/#/pulso")
    driver.refresh()
    time.sleep(3)
    driver.find_element_by_xpath('//*[@id="botonCerrar"]/mat-icon').click()
    time.sleep(3)
    driver.find_element(By.CSS_SELECTOR, ".col-4:nth-child(3) > .btn-filtros").click()
    time.sleep(3)
    driver.find_element(By.CSS_SELECTOR, ".mat-form-field-infix").click()
    driver.find_element(By.CSS_SELECTOR, "#mat-option-5 > span").click()
    time.sleep(3)
    table = driver.find_element_by_xpath(
        "/html/body/app-root/block-ui/app-paginas/app-layout-principal/mat-sidenav-container/mat-sidenav-content/div[2]/div/pulso-reto/div/div[4]/mat-card/mat-card-content/mat-tab-group/div/mat-tab-body[1]/div/div/tabla-alzas-bajas/div/div/app-table/div/gt-column-settings/div/div/generic-table/table"
    ).get_attribute("outerHTML")
    dfs = pd.read_html(table)
    df = dfs[0]
    df.drop(["% de Variación"], axis=1, inplace=True)
    print(df)
    df.rename(columns={"Precio Actual": "Variación"}, inplace=True)
    df["Variación"] = df["Variación"].str.replace("% de Variación ", "")
    df["Variación"] = df["Variación"].str.replace("%", "")
    df.rename(columns={"Emisora": "Precio Historico"}, inplace=True)
    df["Precio Historico"] = df["Precio Historico"].str.replace("Historico", "")
    df.rename(columns={"Historico": "Precio Actual"}, inplace=True)
    df["Precio Actual"] = df["Precio Actual"].str.replace("Precio Actual", "")
    df.rename(columns={"Sort:": "Emisora"}, inplace=True)
    df["Emisora"] = df["Emisora"].str.replace("Emisora", "")
    df["Emisora"] = df["Emisora"].str.replace(" *", "")
    print(df)
    df.to_csv("top_reto.csv", index=False, header=True, encoding="utf-8")


def write_to_html_file(df, title, filename):
    """
    Write an entire dataframe to an HTML file with nice formatting.
    """

    result = """
<html>
<head>
<style>

@media only screen 
and (min-device-width : 320px)  {
    /* smartphones, iPhone, portrait 480x320 phones */
     h2{
         margin-top: 5%;
         text-align: center;
         font-size: 36px;
    }
     table {
         border-collapse: collapse;
         width:90%;
         margin-left:5%;
         margin-right:5%;
         font-size: 28px;
         font-family: sans-serif;
         min-width: 400px;
         box-shadow: 0 0 20px rgba(0, 0, 0, 0.15);
    }
     table thead tr {
         background-color: #98001F;
         color: #ffffff;
         text-align: left;
    }
     table th, table td {
         padding: 12px 15px;
         text-align: center;
    }
     table tbody tr {
         border-bottom: 1px solid #dddddd;
    }
     table tbody tr:nth-of-type(even) {
         background-color: #f3f3f3;
    }
     table tbody tr:last-of-type {
         border-bottom: 2px solid #98001F;
    }
     table tbody tr.active-row {
         font-weight: bold;
         color: #98001F;
    }
    /*Mediquery*/
}
/* Desktops and laptops ----------- */
 @media only screen and (min-width : 1224px) {
    /* Styles */
     h2{
         margin-top: 5%;
         text-align: center;
         font-size: 32px;
    }
     table {
         border-collapse: collapse;
         width:90%;
         margin-left:5%;
         margin-right:5%;
         font-size: 24px;
         font-family: sans-serif;
         min-width: 20px;
         box-shadow: 0 0 20px rgba(0, 0, 0, 0.15);
    }
     table thead tr {
         background-color: #009879;
         color: #ffffff;
         text-align: left;
    }
     table th, table td {
         padding: 12px 15px;
         text-align: center;
    }
     table tbody tr {
         border-bottom: 1px solid #dddddd;
    }
     table tbody tr:nth-of-type(even) {
         background-color: #f3f3f3;
    }
     table tbody tr:last-of-type {
         border-bottom: 2px solid #009879;
    }
     table tbody tr.active-row {
         font-weight: bold;
         color: #009879;
    }
}


</style>
</head>
<body>
    """
    result += "<h2> %s </h2>\n" % title
    if type(df) == pd.io.formats.style.Styler:
        result += df.render()
    else:
        result += df.to_html(classes="wide", escape=False)
    result += """
</body>
</html>
"""
    with open(filename, "w") as f:
        f.write(result)


def day_trading_alerts(market_closing_hour):
    driver = configure_firefox_driver_with_profile()
    day_trading_stocks = []

    try:
        if not login_platform_investing(driver):
            print("Error starting session!")

        driver.get("https://mx.investing.com/")
        WebDriverWait(driver, 100).until(
            EC.element_to_be_clickable((By.XPATH, '//*[@id="portfolioTopBarBtn"]'))
        ).click()
        WebDriverWait(driver, 50).until(
            EC.element_to_be_clickable(
                (
                    By.CSS_SELECTOR,
                    "#portfolioContainer_25863947 > div:nth-child(1) > a:nth-child(2)",
                )
            )
        )
        driver.find_element(
            By.CSS_SELECTOR,
            "#portfolioContainer_25863947 > div:nth-child(1) > a:nth-child(2)",
        ).click()

        driver.find_element(By.CSS_SELECTOR, "#performance > a:nth-child(1)").click()
        time.sleep(3)
        table = driver.find_element_by_xpath(
            "/html/body/div[5]/section/div[7]/div[7]/div[2]/div[2]/div/table"
        ).get_attribute("outerHTML")
        dfs = pd.read_html(table)
        df1 = dfs[0]
        df1.drop(["Unnamed: 0", "Unnamed: 8"], axis=1, inplace=True, errors="ignore")

        driver.find_element(By.CSS_SELECTOR, "#technical > a:nth-child(1)").click()
        time.sleep(3)
        table = driver.find_element_by_xpath(
            "/html/body/div[5]/section/div[7]/div[7]/div[2]/div[2]/div/table"
        ).get_attribute("outerHTML")
        dfs = pd.read_html(table)
        df2 = dfs[0]
        df2.drop(["Unnamed: 0", "Unnamed: 10"], axis=1, inplace=True, errors="ignore")

        driver.find_element(By.CSS_SELECTOR, "#overview > a:nth-child(1)").click()
        time.sleep(3)
        table = driver.find_element_by_xpath(
            "/html/body/div[5]/section/div[7]/div[7]/div[2]/div[2]/div/div/table"
        ).get_attribute("outerHTML")
        dfs = pd.read_html(table)
        df3 = dfs[0]
        df3.drop(
            ["Unnamed: 0", "Unnamed: 1", "Unnamed: 19", "Unnamed: 20"],
            axis=1,
            inplace=True,
            errors="ignore",
        )

        WebDriverWait(driver, 50).until(
            EC.element_to_be_clickable(
                (
                    By.XPATH,
                    "/html/body/div[5]/section/div[7]/div[7]/div[2]/div[2]/div/div/table",
                )
            )
        )
        table_elements_simbols = driver.find_elements(
            By.XPATH,
            "/html/body/div[5]/section/div[7]/div[7]/div[2]/div[2]/div/div/table/tbody/tr/td[4]/a",
        )
        dicionary_simbols = {}
        for tag in table_elements_simbols:
            dicionary_simbols[str(tag.get_attribute("innerHTML"))] = str(
                tag.get_attribute("href")
            )

        dfs = [df1, df2, df3]
        dfs = [x.set_index("Nombre") for x in dfs]
        df = pd.concat(dfs, axis=1, keys=range(1, len(dfs) + 1))
        df.columns = df.columns.map("{0[1]}{0[0]}".format)
        df = df.reset_index()

        df["Anual1"] = df["Anual1"].map(lambda x: str(x)[:-1])
        df["Anual1"] = df["Anual1"].astype("float")
        df["Mensual1"] = df["Mensual1"].map(lambda x: str(x)[:-1])
        df["Mensual1"] = df["Mensual1"].astype("float")
        df["Semanal1"] = df["Semanal1"].map(lambda x: str(x)[:-1])
        df["Semanal1"] = df["Semanal1"].astype("float")
        df["Diario1"] = df["Diario1"].map(lambda x: str(x).replace("+-", "+")[:-1])
        df["Diario1"] = df["Diario1"].astype("float")
        df["% var.3"] = df["% var.3"].map(lambda x: str(x).replace("+-", "+")[:-1])
        df["% var.3"] = df["% var.3"].astype("float")
        df["3 años1"] = df["3 años1"].map(
            lambda x: str(x).replace("-", "0").replace("%", "")
        )
        df["3 años1"] = df["3 años1"].astype("float")
        df["1 Año1"] = df["1 Año1"].map(lambda x: str(x)[:-1])
        df["1 Año1"] = df["1 Año1"].astype("float")

        dayli_negative_mean = df.loc[df["Diario1"] < 0, "Diario1"].mean()
        weekly_negative_mean = df.loc[df["Semanal1"] < 0, "Semanal1"].mean()
        monthly_negative_mean = df.loc[df["Mensual1"] < 0, "Mensual1"].mean()

        # df = df[(df[['Anual1']] > -11.62).all(1)]
        # df = df[(df[['Mensual1']] > -3.38).all(1)]
        # df = df[(df[['Semanal1']] > -2.59).all(1)]
        # df = df[(df[['Diario1']] > -1.99).all(1)]

        current_hour = datetime.now().hour
        cc = 1
        while current_hour < market_closing_hour:

            for x in df["Símbolo3"]:
                current_hour = datetime.now().hour
                current_time = datetime.now().strftime("%H:%M")

                # Calcular capital óptimo para un solo ticker
                mount = calculate_optimal_capital([x])
                operation_mount = mount * 0.03
                operation_mount_limit = mount * 0.06
                print("Analizando {}".format(x))
                path_5min_chart_image = os.path.join(
                    dn, "img", x + "-5min_chart_image.png"
                )
                path_days_chart_image = os.path.join(
                    dn, "img", x + "-days_chart_image.png"
                )
                path_weekly_chart_image = os.path.join(
                    dn, "img", x + "-weekly_chart_image.png"
                )
                path_30min_chart_image = os.path.join(
                    dn, "img", x + "-30min_chart_image.png"
                )

                try:
                    if containsAny(dicionary_simbols[x], ["?"]):
                        chart_url = insert_string_before(
                            dicionary_simbols[x], "-advanced-chart", "?"
                        )
                    else:
                        chart_url = dicionary_simbols[x] + "-advanced-chart"
                    driver.get(chart_url)

                    driver.execute_script(
                        "arguments[0].scrollIntoView()",
                        driver.find_element(
                            By.XPATH, "/html/body/div[5]/section/div[7]/h2"
                        ),
                    )
                    driver.switch_to.frame(
                        WebDriverWait(driver, 10).until(
                            EC.element_to_be_clickable((By.TAG_NAME, "iframe"))
                        )
                    )

                    # Setting indicators
                    WebDriverWait(driver, 10).until(
                        EC.element_to_be_clickable(
                            (
                                By.XPATH,
                                "/html/body/div/div[1]/div/prochart/div/div[1]/div/div[1]/div/div[2]/div/ul/li[5]/a",
                            )
                        )
                    ).click()

                    WebDriverWait(driver, 15).until(
                        EC.element_to_be_clickable(
                            (
                                By.XPATH,
                                "/html/body/div/div[1]/div/prochart/div/div[1]/div/div[1]/div/div[2]/div/ul/li[5]/div/ul/li[1]/a",
                            )
                        )
                    ).click()
                    WebDriverWait(driver, 10).until(
                        EC.element_to_be_clickable(
                            (
                                By.XPATH,
                                "/html/body/div/div[1]/div/prochart/div/div[1]/div/div[1]/div/div[2]/div/ul/li[5]/a",
                            )
                        )
                    ).click()

                    driver.find_element(
                        By.XPATH,
                        "/html/body/div/div[1]/div/prochart/div/div[1]/div/div[1]/div/div[2]/div/ul/li[5]/div/ul/li[5]/a",
                    ).click()

                    # Select time days
                    WebDriverWait(driver, 10).until(
                        EC.element_to_be_clickable(
                            (
                                By.XPATH,
                                "/html/body/div/div[1]/div/prochart/div/div[1]/div/div[1]/div/div[3]/ul/li[15]/button",
                            )
                        )
                    ).click()

                    time.sleep(4)

                    WebDriverWait(driver, 10).until(
                        EC.element_to_be_clickable(
                            (
                                By.XPATH,
                                "/html/body/div/div[1]/div/prochart/div/div[1]/div/div[4]/div/div[1]",
                            )
                        )
                    ).screenshot(path_days_chart_image)

                    # Select time 5 minutes
                    WebDriverWait(driver, 10).until(
                        EC.element_to_be_clickable(
                            (
                                By.XPATH,
                                "/html/body/div/div[1]/div/prochart/div/div[1]/div/div[1]/div/div[3]/ul/li[20]/button",
                            )
                        )
                    ).click()

                    time.sleep(4)

                    WebDriverWait(driver, 10).until(
                        EC.element_to_be_clickable(
                            (
                                By.XPATH,
                                "/html/body/div/div[1]/div/prochart/div/div[1]/div/div[4]/div/div[1]",
                            )
                        )
                    ).screenshot(path_5min_chart_image)

                    # Select time 30min
                    WebDriverWait(driver, 10).until(
                        EC.element_to_be_clickable(
                            (
                                By.XPATH,
                                "/html/body/div/div[1]/div/prochart/div/div[1]/div/div[1]/div/div[3]/ul/li[17]/button",
                            )
                        )
                    ).click()
                    time.sleep(4)

                    WebDriverWait(driver, 10).until(
                        EC.element_to_be_clickable(
                            (
                                By.XPATH,
                                "/html/body/div/div[1]/div/prochart/div/div[1]/div/div[4]/div/div[1]",
                            )
                        )
                    ).screenshot(path_30min_chart_image)

                    # Select time weekly
                    WebDriverWait(driver, 10).until(
                        EC.element_to_be_clickable(
                            (
                                By.XPATH,
                                "/html/body/div/div[1]/div/prochart/div/div[1]/div/div[1]/div/div[3]/ul/li[21]/button",
                            )
                        )
                    ).click()

                    time.sleep(4)

                    WebDriverWait(driver, 10).until(
                        EC.element_to_be_clickable(
                            (
                                By.XPATH,
                                "/html/body/div/div[1]/div/prochart/div/div[1]/div/div[4]/div/div[1]",
                            )
                        )
                    ).screenshot(path_weekly_chart_image)

                except Exception as e:
                    print(e)
                    print("Error ejecutando {}".format(x))
                    pass

                try:
                    if containsAny(dicionary_simbols[x], ["?"]):
                        technical_data_url = insert_string_before(
                            dicionary_simbols[x], "-technical", "?"
                        )
                    else:
                        technical_data_url = dicionary_simbols[x] + "-technical"

                    driver.get(technical_data_url)

                    # Time dimention in 15 minutes
                    WebDriverWait(driver, 30).until(
                        EC.element_to_be_clickable(
                            (By.XPATH, "/html/body/div[5]/section/div[8]/ul/li[2]/a")
                        )
                    )

                    driver.find_element(
                        By.XPATH, "/html/body/div[5]/section/div[8]/ul/li[2]/a"
                    ).click()
                    WebDriverWait(driver, 30).until(
                        EC.element_to_be_clickable((By.XPATH, '//*[@id="curr_table"]'))
                    )

                    rsi_15min = float(
                        driver.find_element(
                            By.XPATH,
                            "/html/body/div[5]/section/div[10]/div[3]/table/tbody/tr[1]/td[2]",
                        ).get_attribute("innerHTML")
                    )

                    volume = float(
                        driver.find_element(
                            By.XPATH,
                            "/html/body/div[5]/section/div[4]/div[2]/div/ul/li[1]/span[2]/span",
                        )
                        .get_attribute("innerHTML")
                        .replace(",", "")
                        .replace(",", "")
                        .replace(",", "")
                    )

                    current_value = float(
                        driver.find_element(By.CSS_SELECTOR, "#last_last")
                        .get_attribute("innerHTML")
                        .replace(",", "")
                        .replace(",", "")
                        .replace(",", "")
                    )

                    # Time dimention in 5 minutes
                    WebDriverWait(driver, 30).until(
                        EC.element_to_be_clickable(
                            (By.XPATH, "/html/body/div[5]/section/div[8]/ul/li[1]/a")
                        )
                    )
                    driver.find_element(
                        By.XPATH, "/html/body/div[5]/section/div[8]/ul/li[1]/a"
                    ).click()

                    WebDriverWait(driver, 30).until(
                        EC.element_to_be_clickable((By.XPATH, '//*[@id="curr_table"]'))
                    )

                    rsi_5min = float(
                        driver.find_element(
                            By.XPATH,
                            "/html/body/div[5]/section/div[10]/div[3]/table/tbody/tr[1]/td[2]",
                        ).get_attribute("innerHTML")
                    )

                    sma10_5min = float(
                        driver.find_element(
                            By.CSS_SELECTOR,
                            ".movingAvgsTbl > tbody:nth-child(2) > tr:nth-child(2) > td:nth-child(2)",
                        )
                        .get_attribute("innerHTML")
                        .split("<", 1)[0]
                        .replace(",", "")
                        .replace(",", "")
                        .replace(",", "")
                    )
                    sma20_5min = float(
                        driver.find_element(
                            By.CSS_SELECTOR,
                            ".movingAvgsTbl > tbody:nth-child(2) > tr:nth-child(3) > td:nth-child(2)",
                        )
                        .get_attribute("innerHTML")
                        .split("<", 1)[0]
                        .replace(",", "")
                        .replace(",", "")
                        .replace(",", "")
                    )
                    sma50_5min = float(
                        driver.find_element(
                            By.CSS_SELECTOR,
                            ".movingAvgsTbl > tbody:nth-child(2) > tr:nth-child(4) > td:nth-child(2)",
                        )
                        .get_attribute("innerHTML")
                        .split("<", 1)[0]
                        .replace(",", "")
                        .replace(",", "")
                        .replace(",", "")
                    )
                    sma100_5min = float(
                        driver.find_element(
                            By.CSS_SELECTOR,
                            ".movingAvgsTbl > tbody:nth-child(2) > tr:nth-child(5) > td:nth-child(2)",
                        )
                        .get_attribute("innerHTML")
                        .split("<", 1)[0]
                        .replace(",", "")
                        .replace(",", "")
                        .replace(",", "")
                    )
                    sma200_5min = float(
                        driver.find_element(
                            By.CSS_SELECTOR,
                            ".movingAvgsTbl > tbody:nth-child(2) > tr:nth-child(6) > td:nth-child(2)",
                        )
                        .get_attribute("innerHTML")
                        .split("<", 1)[0]
                        .replace(",", "")
                        .replace(",", "")
                        .replace(",", "")
                    )

                    fibonacci_resist1_5min = float(
                        driver.find_element(
                            By.CSS_SELECTOR,
                            "table.closedTbl:nth-child(4) > tbody:nth-child(2) > tr:nth-child(1) > td:nth-child(6)",
                        )
                        .get_attribute("innerHTML")
                        .split("<", 1)[0]
                        .replace(",", "")
                        .replace(",", "")
                        .replace(",", "")
                    )
                    fibonacci_resist2_5min = float(
                        driver.find_element(
                            By.CSS_SELECTOR,
                            "table.closedTbl:nth-child(4) > tbody:nth-child(2) > tr:nth-child(1) > td:nth-child(7)",
                        )
                        .get_attribute("innerHTML")
                        .split("<", 1)[0]
                        .replace(",", "")
                        .replace(",", "")
                        .replace(",", "")
                    )
                    fibonacci_resist3_5min = float(
                        driver.find_element(
                            By.CSS_SELECTOR,
                            "table.closedTbl:nth-child(4) > tbody:nth-child(2) > tr:nth-child(1) > td:nth-child(8)",
                        )
                        .get_attribute("innerHTML")
                        .split("<", 1)[0]
                        .replace(",", "")
                        .replace(",", "")
                        .replace(",", "")
                    )

                    estimated_earn_sma50_5min = round(
                        percentage_change(current_value, sma50_5min), 2
                    )
                    estimated_loss = round(weekly_negative_mean, 2)
                    stop_loss = round(
                        current_value * (1 - (abs(estimated_loss / 100))), 2
                    )

                    risk_earn_coffient = round(
                        estimated_earn_sma50_5min / estimated_loss, 2
                    )

                    estimated_earn_5min_fibonacci_resist = round(
                        percentage_change(current_value, fibonacci_resist1_5min), 2
                    )
                    no_operation_titles = round(operation_mount / current_value)
                    volume_per_hour = volume / 8
                    estimated_volume = volume / 200

                    if no_operation_titles > estimated_volume:
                        no_operation_titles = estimated_volume
                        print(no_operation_titles)
                    if (
                        current_value > operation_mount
                        and current_value < operation_mount_limit
                    ):
                        no_operation_titles = 1
                        print(no_operation_titles)
                    elif (
                        current_value > operation_mount
                        and current_value > operation_mount_limit
                    ):
                        no_operation_titles = 0
                        print(no_operation_titles)

                    # if  estimated_volume <= 10:
                    #    estimated_volume = round(estimated_volume/5)*5
                    # elif estimated_volume <= 100 :
                    #    estimated_volume = round(estimated_volume/10)*10
                    # elif estimated_volume 1000:
                    #    estimated_volume = round(estimated_volume/100)*100
                    # elif estimated_volume > 1000:
                    #    estimated_volume = 1000

                    # if  x not in day_trading_stocks:
                    #    day_trading_stocks.append(x)

                    #
                    if (
                        rsi_5min > 65
                        and current_value > sma10_5min
                        and (
                            df.loc[df["Símbolo3"] == x, "Mensual2"].iloc[0] == "Compra"
                            or df.loc[df["Símbolo3"] == x, "Mensual2"].iloc[0]
                            == "Compra fuerte"
                        )
                    ):

                        message = "\U0001F4CA Oportunidad de compra especulativa de {} \U0001F4CA \n\nCon un valor actual de ${}. El RSI de este símbolo tendencia de sobrecompra que podría ser de interés a las {} hrs.\n\nPara ver más información consulta: {}".format(
                            x,
                            "{:,.2f}".format(current_value),
                            current_time,
                            dicionary_simbols[x],
                        )
                        for chat_id in telegram_chat_ids:
                            with open(path_days_chart_image, "rb") as f1, open(
                                path_5min_chart_image, "rb"
                            ) as f2, open(path_30min_chart_image, "rb") as f3, open(
                                path_weekly_chart_image, "rb"
                            ) as f4:
                                files = bot.send_media_group(
                                    chat_id=chat_id,
                                    media=[
                                        telegram.InputMediaPhoto(f4, caption=message),
                                        telegram.InputMediaPhoto(f2),
                                        telegram.InputMediaPhoto(f3),
                                        telegram.InputMediaPhoto(f1),
                                    ],
                                )

                    # if rsi_15min < 35:
                    if (
                        rsi_15min < 35
                        and rsi_15min > 20
                        and estimated_earn_sma50_5min > 0.3
                        and no_operation_titles > 0
                    ):

                        message = '\U0001F6A8 Oportunidad de compra especulativa de {} \U0001F6A8 \n\nCon un valor actual de ${}. El RSI de este símbolo señala lecturas de sobreventa a las {} hrs, se espera un "pull-back" en corto plazo.\n\nGanancia potencial: {}%\nPrecio objetivo: ${}\nPerdida potencial: {}%\nLímite de perdida sugerido: ${}\nRiesgo/Ganancia: {}\nVolumen de compra/venta: {} títulos\nVolumen de compra sugerido: {} títulos.\n\nPara ver más información consulta: {}'.format(
                            x,
                            "{:,.2f}".format(current_value),
                            current_time,
                            str(estimated_earn_sma50_5min),
                            "{:,.2f}".format(sma50_5min),
                            "{:,.2f}".format(estimated_loss),
                            str(stop_loss),
                            "{:,.2f}".format(risk_earn_coffient),
                            "{:,.0f}".format(volume),
                            "{:,.0f}".format(no_operation_titles),
                            dicionary_simbols[x],
                        )
                        for chat_id in telegram_chat_ids:
                            with open(path_days_chart_image, "rb") as f1, open(
                                path_5min_chart_image, "rb"
                            ) as f2, open(path_30min_chart_image, "rb") as f3, open(
                                path_weekly_chart_image, "rb"
                            ) as f4:
                                files = bot.send_media_group(
                                    chat_id=chat_id,
                                    media=[
                                        telegram.InputMediaPhoto(f4, caption=message),
                                        telegram.InputMediaPhoto(f2),
                                        telegram.InputMediaPhoto(f3),
                                        telegram.InputMediaPhoto(f1),
                                    ],
                                )
                            # bot.send_photo(chat_id, caption=message, photo=open(path_5min_chart_image, 'rb'))
                            # bot.send_photo(chat_id, photo=open(path_days_chart_image, 'rb'))

                    if (
                        estimated_earn_5min_fibonacci_resist > 0.3
                        and no_operation_titles > 0
                        and (
                            df.loc[df["Símbolo3"] == x, "15 minutos2"].iloc[0]
                            == "Compra"
                            or df.loc[df["Símbolo3"] == x, "15 minutos2"].iloc[0]
                            == "Compra fuerte"
                        )
                        and (
                            df.loc[df["Símbolo3"] == x, "5 minutos2"].iloc[0]
                            == "Compra"
                            or df.loc[df["Símbolo3"] == x, "5 minutos2"].iloc[0]
                            == "Compra fuerte"
                        )
                        and (
                            df.loc[df["Símbolo3"] == x, "Semanal2"].iloc[0] == "Compra"
                            or df.loc[df["Símbolo3"] == x, "Semanal2"].iloc[0]
                            == "Compra fuerte"
                        )
                    ):

                        message = "\U0001F50D Oportunidad de compra especulativa de {} \U0001F50D \n\nCon un valor actual de ${}. El símbolo muestra tendencias de compra en el corto plazo a las {} hrs, se espera un avance en corto plazo de {}% en ${}, el símbolo tiene un volumen de compra/venta de {} unidades, aunque se sugiere realizar una compra por menos de {}.\n\nPara ver más información consulta: {}".format(
                            x,
                            "{:,.2f}".format(current_value),
                            current_time,
                            str(estimated_earn_5min_fibonacci_resist),
                            "{:,.2f}".format(fibonacci_resist2_5min),
                            "{:,.0f}".format(volume),
                            "{:,.0f}".format(no_operation_titles),
                            dicionary_simbols[x],
                        )
                        for chat_id in telegram_chat_ids:
                            with open(path_days_chart_image, "rb") as f1, open(
                                path_5min_chart_image, "rb"
                            ) as f2, open(path_30min_chart_image, "rb") as f3, open(
                                path_weekly_chart_image, "rb"
                            ) as f4:
                                files = bot.send_media_group(
                                    chat_id=chat_id,
                                    media=[
                                        telegram.InputMediaPhoto(f4, caption=message),
                                        telegram.InputMediaPhoto(f2),
                                        telegram.InputMediaPhoto(f3),
                                        telegram.InputMediaPhoto(f1),
                                    ],
                                )

                except Exception as e:
                    print(e)
                    print("Error ejecutando {}".format(x))
                    pass
            cc = cc + 1
            time.sleep(900)
        print("Fin del analisis intradia")
        driver.close()
        exit()

    except Exception as e:
        print(e)
        driver.close()
        exit()


def swing_trading_recommendations():
    driver = configure_firefox_driver_with_profile()
    # tb.send_message(chat_id,  "Inicializando análisis de acciones usando estrategía swing trading con machine learning...")
    try:
        if not login_platform_investing(driver):
            print("Error starting session!")

        time.sleep(3)
        driver.get("https://mx.investing.com/")

        WebDriverWait(driver, 50).until(
            EC.element_to_be_clickable((By.XPATH, '//*[@id="portfolioTopBarBtn"]'))
        )
        driver.find_element_by_xpath('//*[@id="portfolioTopBarBtn"]').click()
        WebDriverWait(driver, 50).until(
            EC.element_to_be_clickable(
                (
                    By.CSS_SELECTOR,
                    "#portfolioContainer_25863947 > div:nth-child(1) > a:nth-child(2)",
                )
            )
        )
        driver.find_element(
            By.CSS_SELECTOR,
            "#portfolioContainer_25863947 > div:nth-child(1) > a:nth-child(2)",
        ).click()

        driver.find_element(By.CSS_SELECTOR, "#performance > a:nth-child(1)").click()
        time.sleep(3)
        table = driver.find_element_by_xpath(
            "/html/body/div[5]/section/div[7]/div[7]/div[2]/div[2]/div/table"
        ).get_attribute("outerHTML")
        dfs = pd.read_html(table)
        df1 = dfs[0]
        df1.drop(["Unnamed: 0", "Unnamed: 8"], axis=1, inplace=True, errors="ignore")
        # print(df.columns)

        # df.to_csv('analisis_tecnico_performance.csv',  index=False, header=True, encoding="utf-8")
        driver.find_element(By.CSS_SELECTOR, "#technical > a:nth-child(1)").click()
        time.sleep(3)
        table = driver.find_element_by_xpath(
            "/html/body/div[5]/section/div[7]/div[7]/div[2]/div[2]/div/table"
        ).get_attribute("outerHTML")
        dfs = pd.read_html(table)
        df2 = dfs[0]
        df2.drop(["Unnamed: 0", "Unnamed: 10"], axis=1, inplace=True, errors="ignore")
        # print(df.columns)
        # df.to_csv('analisis_tecnico_algoritmo.csv',index=False, header=True, encoding="utf-8")

        driver.find_element(By.CSS_SELECTOR, "#overview > a:nth-child(1)").click()
        time.sleep(3)
        table = driver.find_element_by_xpath(
            "/html/body/div[5]/section/div[7]/div[7]/div[2]/div[2]/div/div/table"
        ).get_attribute("outerHTML")
        dfs = pd.read_html(table)
        df3 = dfs[0]
        df3.drop(
            ["Unnamed: 0", "Unnamed: 1", "Unnamed: 19", "Unnamed: 20"],
            axis=1,
            inplace=True,
            errors="ignore",
        )
        # print(df.columns)

        # Create dicionary investing simbols portfolio
        WebDriverWait(driver, 50).until(
            EC.element_to_be_clickable(
                (
                    By.XPATH,
                    "/html/body/div[5]/section/div[7]/div[7]/div[2]/div[2]/div/div/table",
                )
            )
        )
        table_elements_simbols = driver.find_elements(
            By.XPATH,
            "/html/body/div[5]/section/div[7]/div[7]/div[2]/div[2]/div/div/table/tbody/tr/td[4]/a",
        )
        dicionary_simbols = {}
        for tag in table_elements_simbols:
            dicionary_simbols[str(tag.get_attribute("innerHTML"))] = str(
                tag.get_attribute("href")
            )

        # tb.send_message(chat_id,  "Analizando acciones...")
        dfs = [df1, df2, df3]
        dfs = [x.set_index("Nombre") for x in dfs]
        df = pd.concat(dfs, axis=1, keys=range(1, len(dfs) + 1))
        df.columns = df.columns.map("{0[1]}{0[0]}".format)
        df = df.reset_index()

        # print("Lista de empresas a analizar: {}".format(df['Nombre']))

        df["Fecha"] = datetime.now().strftime("%x %X")
        df["3 años1"] = df["3 años1"].map(lambda x: str(x).replace("-", "0"))
        df["3 años1"] = df["3 años1"].map(lambda x: str(x).replace("%", ""))

        df["3 años1"] = df["3 años1"].astype("float")
        df["1 Año1"] = df["1 Año1"].map(lambda x: str(x)[:-1])

        df["1 Año1"] = df["1 Año1"].astype("float")
        df["Anual1"] = df["Anual1"].map(lambda x: str(x)[:-1])

        df["Anual1"] = df["Anual1"].astype("float")
        df["Mensual1"] = df["Mensual1"].map(lambda x: str(x)[:-1])

        df["Mensual1"] = df["Mensual1"].astype("float")
        df["Semanal1"] = df["Semanal1"].map(lambda x: str(x)[:-1])

        df["Semanal1"] = df["Semanal1"].astype("float")
        df["Diario1"] = df["Diario1"].map(lambda x: str(x).replace("+-", "+")[:-1])

        df["Diario1"] = df["Diario1"].astype("float")
        df["% var.3"] = df["% var.3"].map(lambda x: str(x).replace("+-", "+")[:-1])

        df["% var.3"] = df["% var.3"].astype("float")

        dayli_negative_mean = df.loc[df["Diario1"] < 0, "Diario1"].mean()
        weekly_negative_mean = df.loc[df["Semanal1"] < 0, "Semanal1"].mean()
        monthly_negative_mean = df.loc[df["Mensual1"] < 0, "Mensual1"].mean()
        annual_negative_mean = df.loc[df["Anual1"] < 0, "Anual1"].mean()
        one_year_negative_mean = df.loc[df["1 Año1"] < 0, "1 Año1"].mean()
        # pd.set_option('display.max_rows', None)
        # print(df['3 años1'] )
        three_years_negative_mean = (
            df.loc[df["3 años1"] < 0, "3 años1"].replace(0, np.NaN).mean()
        )

        df = df[(df[["3 años1"]] > 0).all(1)]  # Mean of negatives
        df = df[(df[["1 Año1"]] > one_year_negative_mean).all(1)]
        df = df[(df[["Anual1"]] > annual_negative_mean).all(1)]
        df = df[(df[["Mensual1"]] > monthly_negative_mean).all(1)]
        df = df[(df[["Semanal1"]] > weekly_negative_mean).all(1)]
        df = df[(df[["Diario1"]] > dayli_negative_mean).all(1)]

        df = df[
            (df[["Mensual2"]] == "Compra fuerte").all(1)
            | (df[["Mensual2"]] == "Compra").all(1)
        ]
        df = df[
            (df[["Semanal2"]] == "Compra fuerte").all(1)
            | (df[["Semanal2"]] == "Compra").all(1)
        ]
        df = df[
            (df[["Diario2"]] == "Compra fuerte").all(1)
            | (df[["Diario2"]] == "Compra").all(1)
        ]

        for x in df["Símbolo3"]:
            # tb.send_message(chat_id,  "Analizando {} ...".format(x)  )
            try:
                if containsAny(dicionary_simbols[x], ["?"]):
                    technical_data_url = insert_string_before(
                        dicionary_simbols[x], "-technical", "?"
                    )
                else:
                    technical_data_url = dicionary_simbols[x] + "-technical"
                driver.get(technical_data_url)
                WebDriverWait(driver, 50).until(
                    EC.element_to_be_clickable(
                        (By.XPATH, "/html/body/div[5]/section/div[8]/ul/li[6]/a")
                    )
                )
                driver.find_element(
                    By.XPATH, "/html/body/div[5]/section/div[8]/ul/li[6]/a"
                ).click()
                WebDriverWait(driver, 15).until(
                    EC.element_to_be_clickable(
                        (By.XPATH, "/html/body/div[5]/section/div[10]/table")
                    )
                )
                p1 = float(
                    driver.find_element(
                        By.XPATH,
                        "/html/body/div[5]/section/div[10]/table/tbody/tr[2]/td[6]",
                    ).get_attribute("innerHTML")
                )
                p2 = float(
                    driver.find_element(
                        By.XPATH,
                        "/html/body/div[5]/section/div[10]/table/tbody/tr[2]/td[7]",
                    ).get_attribute("innerHTML")
                )
                p3 = float(
                    driver.find_element(
                        By.XPATH,
                        "/html/body/div[5]/section/div[10]/table/tbody/tr[2]/td[8]",
                    ).get_attribute("innerHTML")
                )
                pe = round((p1 + p2 + p3) / 3, 2)
                df.loc[df["Símbolo3"] == x, "PeEstimado"] = pe
                try:
                    df.loc[df["Símbolo3"] == x, "ML Prediction"] = (
                        predict_machine_daily(x)
                    )
                except Exception as e:
                    df.loc[df["Símbolo3"] == x, "ML Prediction"] = "NO DISPONIBLE"
                    print(e)
            except Exception as e:
                print(e)
                print("Error ejecutando {}".format(x))
                pass

        df = df[(df[["ML Prediction"]] == "Compra").all(1)]
        df["GanEstimada %"] = round(
            percentage_change(df["Último3"], df["PeEstimado"]), 2
        )
        df = df[(df[["GanEstimada %"]] > 0.5).all(1)]
        df["PeVentaCalc"] = round(df["Último3"] * 1.005, 2)

        dict = {"Símbolo3": "Símbolo", "% var.3": "% Cambio"}
        df.rename(columns=dict, inplace=True)

        print(
            df[
                [
                    "Símbolo",
                    "Nombre",
                    "Último3",
                    "PeEstimado",
                    "GanEstimada %",
                    "PeVentaCalc",
                    "ML Prediction",
                ]
            ]
        )

        tickers = df["Símbolo"].astype(str).values.tolist()
        print(len(tickers))
        if len(tickers) >= 1:
            # tb.send_message(chat_id, "Ejecutando algoritmo de optimización de portafolio...")
            optimal_capital = calculate_optimal_capital(tickers)
            df_allocations = markovitz_portfolio_optimization(tickers, optimal_capital)
            df_result = pd.merge(
                df_allocations, df, left_on="Ticker", right_on="Símbolo"
            )
            df_result["Títulos"] = df_result["Allocation $"] / df_result["Último3"]
            df_result["Títulos"] = df_result["Títulos"].astype("int32")
            df_result = df_result.loc[(df_result[["Títulos"]] != 0).all(axis=1)]
            df_result["Allocation $"] = df_result["Allocation $"].round(decimals=2)
            print(
                df_result[
                    [
                        "Símbolo",
                        "Nombre",
                        "Títulos",
                        "Allocation $",
                        "Último3",
                        "PeEstimado",
                        "GanEstimada %",
                        "PeVentaCalc",
                    ]
                ]
            )
            df_result = df_result.sort_values(["GanEstimada %"], ascending=[False])
            print(
                df_result[
                    [
                        "Símbolo",
                        "Nombre",
                        "Títulos",
                        "Allocation $",
                        "Último3",
                        "PeEstimado",
                        "GanEstimada %",
                        "PeVentaCalc",
                    ]
                ]
            )
            result_df = df_result[["Símbolo", "Nombre", "Títulos", "% Cambio"]]
            path_file_name = os.path.join(dn, "Sugerencias.html")
            write_to_html_file(
                result_df, "Sugerencias para Reto Actinver", path_file_name
            )

            for chat_id in telegram_chat_ids:
                doc = open(path_file_name, "rb")
                tb.send_message(chat_id, "Sugerencias de compra:")
                tb.send_document(chat_id, doc)
        else:
            for chat_id in telegram_chat_ids:
                tb.send_message(
                    chat_id,
                    "Intente en otro momento, no hay símbolos bursátiles que pasen las pruebas del algorítmo en este momento.",
                )

        driver.close()
        exit()

    except Exception as e:
        print(e)
        driver.close()
        exit()


def daily_quizz_solver(username, password, email, index_quizz_answer):
    driver = configure_firefox_driver_no_profile()

    selector = [
        "/html/body/div[1]/div[2]/div/mat-dialog-container/modal-quiz/mat-dialog-content/div/div[2]/div/div[2]/div/div[1]/div/p",
        "/html/body/div[1]/div[2]/div/mat-dialog-container/modal-quiz/mat-dialog-content/div/div[2]/div/div[2]/div/div[2]/div/p",
        "/html/body/div[1]/div[2]/div/mat-dialog-container/modal-quiz/mat-dialog-content/div/div[2]/div/div[2]/div/div[3]/div/p",
    ]
    screenshot = os.path.join(dn, "img", username + "-quizz.png")
    message = "El usuario {} respondió correctamente".format(username)

    try:
        is_logged_flag = login_platform_actinver(driver, username, password, email)
        if is_logged_flag:
            solved = False
            while not solved:
                try:
                    driver.get("https://www.retoactinver.com/RetoActinver/#/inicio")
                    close_popoup_browser(driver)
                    WebDriverWait(driver, 10).until(
                        EC.element_to_be_clickable(
                            (
                                By.XPATH,
                                "/html/body/app-root/block-ui/app-paginas/app-layout-principal/mat-sidenav-container/mat-sidenav-content/div[2]/div/app-inicio/div[6]/div[1]/app-tarjeta-inicio/mat-card/mat-card-footer/div/div[2]/button",
                            )
                        )
                    ).click()
                    WebDriverWait(driver, 10).until(
                        EC.element_to_be_clickable(
                            (By.XPATH, selector[index_quizz_answer])
                        )
                    ).click()
                    WebDriverWait(driver, 10).until(
                        EC.element_to_be_clickable(
                            (
                                By.XPATH,
                                "/html/body/div[1]/div[2]/div/mat-dialog-container/modal-quiz/mat-dialog-content/div/div[2]/div/div[3]/button",
                            )
                        )
                    ).click()
                    driver.save_screenshot(screenshot)
                    bot.send_photo(
                        telegram_chat_ids[0],
                        caption=message,
                        photo=open(screenshot, "rb"),
                    )
                    solved = True
                except Exception as e:
                    print(e)
                    print("Cannot solve quizz at this moment...")
                    print("Browsing to maintain active session...")
                    driver.get("https://www.retoactinver.com/RetoActinver/#/portafolio")
            driver.close()
            exit()
        else:
            print("Error starting session!")
    except Exception as e:
        print(e)
        exit()


def solve_daily_quizz():
    threading.Thread(
        target=daily_quizz_solver,
        args=("osvaldohm9", "Os23valdo1.", "osvaldo.hdz.m@outlook.com", 0),
    ).start()
    threading.Thread(
        target=daily_quizz_solver,
        args=("Gabriela62", "copito55", "hernandezsg62@outlook.com", 1),
    ).start()
    threading.Thread(
        target=daily_quizz_solver,
        args=("kikehedz22", "E93h14M01", "enrique45_v@hotmail.com", 2),
    ).start()


def obtener_datos_acciones(tickers):
    """Obtiene datos fundamentales de las acciones especificadas."""
    data = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            print(info)
            input()
            if "shortName" in info:
                data[ticker] = {
                    "nombre": info.get("shortName", "N/A"),
                    "sector": info.get("sector", "N/A"),
                    "pe_ratio": info.get("trailingPE", np.nan),
                    "dividend_yield": info.get("dividendYield", np.nan),
                    "roe": info.get("returnOnEquity", np.nan),
                    "debt_to_equity": info.get("debtToEquity", np.nan),
                    "market_cap": info.get("marketCap", np.nan),
                }
            else:
                print(f"No se encontró 'shortName' para {ticker}")
        except Exception as e:
            print(f"Error obteniendo datos para {ticker}: {e}")

    # Convertir a DataFrame y manejar posibles errores
    df = pd.DataFrame(data).T
    if df.empty:
        print("No se obtuvieron datos de acciones válidas.")
    return df


def filtrar_acciones_por_sector(data, sector):
    """Filtra las acciones por el sector especificado."""
    return data[data["sector"] == sector]


def obtener_datos_acciones(tickers, acciones_por_sector):
    """Obtiene datos fundamentales de las acciones especificadas y asigna sectores."""
    data = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            if "shortName" in info:
                data[ticker] = {
                    "market_cap": info.get(
                        "marketCap", np.nan
                    ),  # Solo se guarda el marketCap
                    "sector": acciones_por_sector.get(
                        ticker, "Desconocido"
                    ),  # Asigna el sector
                    "beta": info.get("beta", np.nan),
                    "volume": info.get("volume", np.nan),
                    "averageVolume": info.get("averageVolume", np.nan),
                }
            else:
                print(f"No se encontró 'shortName' para {ticker}")
        except Exception as e:
            print(f"Error obteniendo datos para {ticker}: {e}")

    # Convertir a DataFrame y manejar posibles errores
    df = pd.DataFrame(data).T
    if df.empty:
        print("No se obtuvieron datos de acciones válidas.")
    return df


def filtrar_acciones_por_sector(data, sector):
    """Filtra las acciones por el sector especificado."""
    return data[data["sector"] == sector]


def analizar_acciones(
    data, market_cap_limit=800000000, beta_min=0.8, average_volume_min=3500
):

    # Manejo de valores nulos
    data.fillna(0, inplace=True)

    if data.empty:
        print("No hay datos para analizar.")
        return []

    print("Datos de las acciones:")
    print(data)

    # Ajustar los criterios de filtrado
    filtro = (
        (data["market_cap"] > market_cap_limit)  # Capitalización de mercado mínima
        & (data["beta"] >= beta_min)
        & (data["averageVolume"] >= average_volume_min)
    )

    # Obtener tickers sugeridos
    sugerencias = data.index[filtro].tolist()

    # Imprimir resultados filtrados
    if sugerencias:
        print(
            "Acciones sugeridas:", ", ".join(sugerencias)
        )  # Imprimir sugerencias en el formato deseado
    else:
        print("No se encontraron acciones que cumplan con los criterios fundamentales.")

    return sugerencias


def fundamental_analysis():
    clear_screen()
    # Clasificación de tickers por sector
    acciones_por_sector = {
        # Tecnología
        "AAPL.MX": "Tecnología",
        "MSFT.MX": "Tecnología",
        "GOOGL.MX": "Tecnología",
        "AMZN.MX": "Tecnología",
        "NVDA.MX": "Tecnología",
        "AMD.MX": "Tecnología",
        "PYPL.MX": "Fintech",
        "META.MX": "Tecnología",
        "INTC.MX": "Tecnología",
        "CSCO.MX": "Tecnología",
        "QCOM.MX": "Tecnología",
        "ORCL.MX": "Tecnología",
        "AVGO.MX": "Tecnología",
        "CRM.MX": "Tecnología",
        "ADBE.MX": "Tecnología",
        "AMAT.MX": "Tecnología",
        "PLTR.MX": "Tecnología",
        "SQ.MX": "Fintech",
        # Consumo Discrecional
        "WMT.MX": "Consumo",
        "COST.MX": "Consumo",
        "MCD.MX": "Consumo",
        "TSLA.MX": "Consumo",
        "NKE.MX": "Consumo",
        "SBUX.MX": "Consumo",
        "AMXB.MX": "Consumo",
        "BIMBOA.MX": "Consumo",
        "DIS.MX": "Comunicación y Medios",
        "NFLX.MX": "Comunicación y Medios",
        "PINS.MX": "Comunicación y Medios",
        "ETSY.MX": "Consumo",
        "TGT.MX": "Consumo",
        "WALMEX.MX": "Consumo",
        "LIVEPOL.MX": "Consumo",
        "BABA.MX": "Consumo",
        # Salud
        "JNJ.MX": "Salud",
        "PFE.MX": "Salud",
        "GILD.MX": "Salud",
        "MRK.MX": "Salud",
        "LLY.MX": "Salud",
        "BMY.MX": "Salud",
        "UNH.MX": "Salud",
        "MRNA.MX": "Salud",
        "CVS.MX": "Salud",
        # Financiero
        "JPM.MX": "Financiero",
        "WFC.MX": "Financiero",
        "BAC.MX": "Financiero",
        "AXP.MX": "Financiero",
        "C.MX": "Financiero",
        "BRKB.MX": "Financiero",
        "GS.MX": "Financiero",
        "V.MX": "Fintech",
        "MA.MX": "Fintech",
        # Energía
        "XOM.MX": "Energía",
        "VLO.MX": "Energía",
        "CVX.MX": "Energía",
        "FANG.MX": "Energía",
        "DVN.MX": "Energía",
        "APA.MX": "Energía",
        "MRO.MX": "Energía",
        # Industriales
        "BA.MX": "Industriales",
        "GE.MX": "Industriales",
        "CAT.MX": "Industriales",
        "FDX.MX": "Industriales",
        "UPS.MX": "Industriales",
        "DE.MX": "Industriales",
        "RTX.MX": "Industriales",
        "LUV.MX": "Viajes y Transporte",
        # Materiales
        "CEMEXCPO.MX": "Materiales",
        "GAPB.MX": "Materiales",
        "VITROA.MX": "Materiales",
        "PE&OLES.MX": "Materiales",
        "GCC.MX": "Materiales",
        "ORBIA.MX": "Materiales",
        "FCX.MX": "Materiales",
        "CLF.MX": "Materiales",
        # Servicios
        "NFLX.MX": "Comunicación y Medios",
        "DIS.MX": "Comunicación y Medios",
        "RCL.MX": "Viajes y Transporte",
        "OMAB.MX": "Viajes y Transporte",
        "LYV.MX": "Servicios",
        "SPCE.MX": "Viajes y Transporte",
        "VESTA.MX": "Bienes Raíces",
        # Fintech
        "PYPL.MX": "Fintech",
        "SQ.MX": "Fintech",
        "SOFI.MX": "Fintech",
        "NU.MX": "Fintech",
    }
    
    
    # Obtener sectores disponibles
    sectores = sorted(list(set(acciones_por_sector.values())))

    # Mostrar los sectores disponibles usando una tabla de `rich`
    table = Table(title="Sectores Disponibles")
    table.add_column("Índice", justify="center", style="cyan", no_wrap=True)
    table.add_column("Sector", style="magenta")

    for idx, sector in enumerate(sectores, 1):
        table.add_row(str(idx), sector)

    console.print(
            f"\n[bold yellow] Mercado de valores a seleccionar : [/bold yellow] [bold green] Bolsa Mexicana de Valores (BMV) (Único disponible) [/bold green]\n"
        )
    
    console.print(table)

    # Permitir selección de múltiples sectores
    valores_por_defecto = [2, 4, 6, 7]  # Ejemplo de índices predeterminados (3, 5, 7, 8 en base 1)
    seleccion = Prompt.ask(
        "[bold green]Selecciona números de sector separados por comas (i.e. 3,5,7,8): [/bold green]"
    )
    
    # Verificar si la entrada está vacía
    if seleccion.strip() == "":
        console.print(
            f"\n[bold yellow] No se detectó entrada, usando valores sugeridos por defecto...[/bold yellow]"
        )
        indices_seleccionados = [i - 1 for i in valores_por_defecto]
    else:
        # Convertir la entrada en una lista de índices, tolerando caracteres no numéricos (p. ej., ')')
        import re
        numeros = re.findall(r'\d+', seleccion)
        if not numeros:
            console.print(
                f"\n[bold yellow] Entrada no válida, usando valores sugeridos por defecto...[/bold yellow]"
            )
            indices_seleccionados = [i - 1 for i in valores_por_defecto]
        else:
            indices_seleccionados = [int(i) - 1 for i in numeros]
    sectores_seleccionados = [sectores[i] for i in indices_seleccionados]

    # Obtener datos de las acciones
    tickers_seleccionados = [
        ticker
        for sector in sectores_seleccionados
        for ticker in acciones_por_sector.keys()
        if acciones_por_sector[ticker] == sector
    ]
    datos = obtener_datos_acciones(tickers_seleccionados, acciones_por_sector)

    # Filtrar por sector y almacenar resultados
    datos_filtrados = {}
    for sector in sectores_seleccionados:
        datos_filtrados[sector] = filtrar_acciones_por_sector(datos, sector)

    # Analizar acciones y almacenar sugerencias
    sugerencias = []
    for sector, datos_sector in datos_filtrados.items():
        sugerencias += analizar_acciones(datos_sector)

    # Mostrar resultados
    if sugerencias:
        console.print(
            f"\n[bold yellow]Sugerencias de acciones:[/bold yellow] [green]{', '.join(sugerencias)}[/green]"
        )
    else:
        console.print(
            "[bold red]No se encontraron acciones que cumplan con los criterios fundamentales.[/bold red]"
        )

def suggest_stocks_by_preferences():
    console = Console()
    
    # Diccionario de acciones según criterios de preferencia
    stock_preferences = {
        "ecología": ["TSLA", "NIO", "ENPH"],  # Ejemplo de acciones relacionadas con ecología
        "bienestar animal": ["ZOOM", "WOOF"],  # Ejemplo de acciones de empresas de bienestar animal
        "tecnología": ["AAPL", "MSFT", "GOOGL", "NVDA", "AMD", "PYPL", "META", "INTC", "CSCO", "QCOM", "ORCL", "AVGO", "CRM", "ADBE", "AMAT", "PLTR", "SQ"],  # Acciones tecnológicas
        "salud": ["JNJ", "PFE", "GILD", "MRK", "LLY", "BMY", "UNH", "MRNA", "CVS"],  # Acciones en el sector salud
        "financiero": ["JPM", "WFC", "BAC", "AXP", "C", "BRKB", "GS", "V", "MA"],  # Acciones en el sector financiero
        "energía": ["XOM", "VLO", "CVX", "FANG", "DVN", "APA", "MRO"],  # Acciones en el sector energético
        "industriales": ["BA", "GE", "CAT", "FDX", "UPS", "DE", "RTX", "LUV"],  # Acciones industriales
        "materiales": ["CEMEXCPO", "GAPB", "VITROA", "PE&OLES", "GCC", "ORBIA", "FCX", "CLF"],  # Acciones en materiales
        "consumo": ["WMT", "COST", "MCD", "TSLA", "NKE", "SBUX", "AMXB", "BIMBOA", "DIS", "NFLX", "PINS", "ETSY", "TGT", "WALMEX", "LIVEPOL", "BABA"],  # Acciones en consumo
        "productos": {
            "iphone": ["AAPL"],  # Si mencionan iPhone, sugerir Apple
            "laptop": ["MSFT", "AAPL"],  # Mencionar laptops sugiere Microsoft y Apple
            "surface": ["MSFT"],  # Mencionar Surface sugiere Microsoft
            "computadoras": ["NVDA", "DELL"],  # Mencionar computadoras sugiere Nvidia y Dell
            "café": ["SBUX"]
        }
    }

    # Obtener entradas del usuario
    stocks_input = input("Introduce acciones directamente por símbolo (ejemplo: NFLX, MSFT, AAPL): ")
    companies_input = input("Introduce nombres de empresas que te gusten (ejemplo: Microsoft, Apple): ")
    products_input = input("Introduce productos favoritos (ejemplo: iphone, laptop): ")

    # Procesar las entradas
    user_stocks = [stock.strip().upper() for stock in stocks_input.split(",") if stock.strip()]
    user_companies = [company.strip().lower() for company in companies_input.split(",") if company.strip()]
    user_products = [product.strip().lower() for product in products_input.split(",") if product.strip()]

    suggested_stocks = set()  # Usar un set para evitar duplicados

    # Agregar automáticamente todas las acciones introducidas por el usuario
    suggested_stocks.update(user_stocks)

    # Agregar acciones basadas en nombres de empresas introducidos
    for company in user_companies:
        if company == "microsoft":
            suggested_stocks.update(stock_preferences["tecnología"][1:2])  # MSFT
        elif company == "apple":
            suggested_stocks.update(stock_preferences["tecnología"][:1])  # AAPL

    # Agregar acciones basadas en productos introducidos
    for product in user_products:
        if product in stock_preferences["productos"]:
            suggested_stocks.update(stock_preferences["productos"][product])

    # Convertir el conjunto a lista
    
    suggestions = list(suggested_stocks)
    final_suggestions = [stock if stock.endswith(".MX") else f"{stock}.MX" for stock in suggested_stocks]

    # Imprimir las sugerencias
    console.print(
        f"\n[bold yellow]Sugerencias de acciones:[/bold yellow] [green]{', '.join(final_suggestions)}[/green]"
    )


def suggest_technical_soon_results():
    # Lista de tickers por defecto
    default_tickers = ["TSLA", "NVDA", "AMD", "NFLX",  "GOOGL", "ZM", "AMZN", "META"]

    # Solicitar al usuario que ingrese tickers
    input_tickers = input(
        "Ingresa las acciones separadas por comas (i.e. NVDA,AAPL,META,MSFT): "
    )

    # Usar tickers por defecto si el usuario no ingresa nada
    if input_tickers.strip():  # Si se ingresaron tickers, usarlos
        tickers = [ticker.strip() for ticker in input_tickers.split(",")]
    else:
        console.print(
            f"\n[bold yellow] No se detectó entrada, usando valores sugeridos por defecto...[/bold yellow]"
        )
        tickers = default_tickers  # Usar los tickers por defecto

    resultados = []  # Para almacenar los resultados
    acciones_comprar = []  # Para las acciones recomendadas para comprar

    # Obtener la fecha actual
    today = datetime.now().date()  # Asegúrate de obtener solo la fecha

    # Iterar sobre cada ticker
    for ticker in tickers:
        try:
            # Crear una instancia del objeto Ticker
            stock = yf.Ticker(ticker)

            # Obtener las fechas de ganancias
            earnings_date = stock.get_earnings_dates()

            # Verificar si se obtuvo información de fechas de ganancias
            if earnings_date is None or earnings_date.empty:
                print(
                    f"Advertencia: No se encontraron fechas de resultados para {ticker}."
                )
                continue

            # Filtrar las fechas de ganancias futuras
            future_dates = earnings_date.index[earnings_date.index.date > today]

            # Si hay fechas futuras, procesar y ordenar
            if not future_dates.empty:
                # Convertir a una lista de fechas y ordenarlas
                sorted_future_dates = sorted(future_dates)

                # Obtener la próxima fecha de resultados
                next_earning_date = sorted_future_dates[0]  # Tomamos la más cercana

                # Almacenar el resultado
                resultados.append(
                    {"Ticker": ticker, "Earnings Date": next_earning_date.date()}
                )

                # Comprobar si la próxima fecha de ganancias está dentro de los próximos 3 días laborales
                if next_earning_date.date() <= today + timedelta(days=3):
                    acciones_comprar.append(ticker)

        except Exception as e:
            print(f"Error al procesar {ticker}: {e}")

    # Crear un DataFrame a partir de los resultados
    df_resultados = pd.DataFrame(resultados)
        
    # Convertir la columna 'Earnings Date' a tipo datetime
    df_resultados['Earnings Date'] = pd.to_datetime(df_resultados['Earnings Date'])
    
    # Ordenar el DataFrame por 'Earnings Date' en orden ascendente
    df_resultados = df_resultados.sort_values(by='Earnings Date')
    
    # Reiniciar los índices (opcional)
    df_resultados.reset_index(drop=True, inplace=True)

    # Imprimir la tabla de resultados
    print("\nResumen de Fechas de Resultados:\n")
    print(df_resultados if not df_resultados.empty else "No se encontraron resultados.")

    # Mostrar recomendaciones
    console.print(
            f"\n[bold yellow] Acciones recomendadas para compra priotaria:\n {', '.join(acciones_comprar) if acciones_comprar else 'Ninguna'} [/bold yellow]"
        )




def suggest_enhanced_etf_strategy(
    tickers=['AAXJ', 'ACWI', 'BIL', 'BOTZ', 'DIA', 'EEM', 'EWZ', 'GDX', 'GLD', 'IAU', 'ICLN', 'INDA', 
'IVV', 'KWEB', 'LIT', 'MCHI', 'PSQ', 'QCLN', 'QQQ', 'SHV', 'SHY', 'SLV', 'SOXX', 'SPLG', 
'SPY', 'TAN', 'TLT', 'USO', 'VEA', 'VGT', 'VNQ', 'VOO', 'VT', 'VTI', 'VWO', 'VYM', 'XLE', 
'XLF', 'XLK', 'XLV']
):
    """
    Análisis técnico rápido usando BANDAS DE BOLLINGER como indicador principal
    Estrategia de reversión a la media para swing trading con ETFs
    """
    # Solicitar al usuario que ingrese tickers (con soporte de comandos estilo vim)
    input_tickers = input("Ingresa las ETFs separadas por comas (i.e FAS.MX,VNQ.MX,XLE.MX): ")
    cmd = input_tickers.strip()
    if cmd.startswith(":"):
        cmd_lower = cmd[1:].strip().lower()
        if cmd_lower in ("q", "quit"):
            exit_program(); return
        if cmd_lower in ("b", "back"):
            return
        if cmd_lower.isdigit():
            try:
                idx = int(cmd_lower) - 1
                # Volver al menú principal y ejecutar opción
                from rich.prompt import Prompt
                console.print(f"[yellow]Saltando a opción {idx+1} del menú...[/yellow]")
                # Hacer que main maneje este salto mediante excepción controlada
                raise SystemExit(f"__GOTO_MENU_OPTION__:{idx}")
            except Exception:
                return

    if input_tickers.strip():  # Si se ingresaron tickers, usarlos
        tickers = [ticker.strip() for ticker in input_tickers.split(",")]
    else:
        console.print(
            f"\n[bold yellow] No se detectó entrada, usando valores sugeridos por defecto (versión .MX)...[/bold yellow]"
        )

    console.print("[bold blue]📈 Análisis Técnico con Bandas de Bollinger - Estrategia de Reversión a la Media[/bold blue]")

    resultados = []  # Para almacenar los resultados
    etf_comprar = []  # Para las ETFs recomendadas para comprar
    etf_mantener = []  # Para las ETFs recomendadas para esperar
    etf_vender = []  # Para las ETFs recomendadas para vender

    # Iterar sobre cada ticker
    for ticker in tickers:
        try:
            # Descargar datos de los últimos 6 meses usando EXCLUSIVAMENTE ticker mexicano (.MX)
            mx_ticker = normalize_ticker_to_mx(ticker)
            console.print(f"[cyan]Descargando {ticker} como {mx_ticker}...[/cyan]")
            df_ticker = yf.download(mx_ticker, period="6mo", progress=False)
            df_ticker.dropna(inplace=True)

            if df_ticker.empty:
                console.print(f"[yellow]⚠️ No se encontraron datos para {ticker} ({mx_ticker})[/yellow]")
                continue

            # Asegurar Series 1D para indicadores técnicos
            close = df_ticker["Close"]
            high = df_ticker["High"]
            low = df_ticker["Low"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            if isinstance(high, pd.DataFrame):
                high = high.iloc[:, 0]
            if isinstance(low, pd.DataFrame):
                low = low.iloc[:, 0]

            # 🚀 OPTIMIZACIÓN PARA CONCURSO: Parámetros más agresivos para 5 semanas
            bollinger = BollingerBands(close, window=15, window_dev=1.8)  # MÁS SENSIBLE
            bb_upper = bollinger.bollinger_hband()
            bb_middle = bollinger.bollinger_mavg()  # Media móvil (SMA 15)
            bb_lower = bollinger.bollinger_lband()
            
            # Indicadores de apoyo optimizados para concurso
            rsi = RSIIndicator(close, window=9).rsi()  # RSI MÁS RÁPIDO
            sma_50 = SMAIndicator(close, window=20).sma_indicator()  # SMA MÁS CORTO
            
            # 🔥 NUEVO: Filtro de volatilidad (ATR) - ESENCIAL PARA CONCURSO
            from ta.volatility import AverageTrueRange
            atr = AverageTrueRange(high, low, close, window=14).average_true_range()
            atr_actual = float(atr.iloc[-1])
            atr_media = float(atr.mean())
            volatil_alto = atr_actual > atr_media * 1.2  # Solo ETFs con volatilidad alta

            # Obtener datos recientes
            close_hoy = float(close.iloc[-1])
            close_ayer = float(close.iloc[-2])
            bb_upper_actual = float(bb_upper.iloc[-1])
            bb_middle_actual = float(bb_middle.iloc[-1])
            bb_lower_actual = float(bb_lower.iloc[-1])
            rsi_actual = float(rsi.iloc[-1])
            sma_50_actual = float(sma_50.iloc[-1])

            # Calcular posición dentro de las bandas (0 = banda inferior, 1 = banda superior)
            bb_position = (close_hoy - bb_lower_actual) / (bb_upper_actual - bb_lower_actual)
            
            # Calcular variación diaria
            variacion_diaria = (close_hoy - close_ayer) / close_ayer * 100

            # Detectar toques de bandas en los últimos 3 días
            toque_banda_inferior = any(close.tail(3) <= bb_lower.tail(3) * 1.01)  # 1% tolerancia
            toque_banda_superior = any(close.tail(3) >= bb_upper.tail(3) * 0.99)  # 1% tolerancia

            # 🚀 LÓGICA ULTRA-OPTIMIZADA PARA CONCURSO (5 SEMANAS)
            
            # 🔥 SEÑALES DE COMPRA ULTRA-OPTIMIZADAS
            condiciones_compra = (
                # 🔥 COMPRA FUERTE: Toque banda inf + RSI <30
                (close_hoy <= bb_lower_actual * 1.01 and rsi_actual < 30) or
                # 🔥 COMPRA MEDIA: Posición BB <0.2 + rebote + volumen
                (bb_position < 0.20 and close_hoy > close_ayer and rsi_actual < 40) or
                # 🔥 COMPRA DÉBIL: Divergencia RSI + tendencia alcista
                (rsi_actual < 35 and tendencia_alcista and bb_position < 0.4)
            )

            # 🔥 SEÑALES DE VENTA ULTRA-OPTIMIZADAS   
            condiciones_venta = (
                # 🔥 VENTA FUERTE: Toque banda sup + RSI >70
                (close_hoy >= bb_upper_actual * 0.99 and rsi_actual > 70) or
                # 🔥 VENTA MEDIA: Posición BB >0.8 + rechazo
                (bb_position > 0.80 and close_hoy < close_ayer and rsi_actual > 60) or
                # 🔥 VENTA DÉBIL: RSI >65 + tendencia bajista
                (rsi_actual > 65 and tendencia_bajista and bb_position > 0.6)
            )

            # Filtro de tendencia general (SMA 50)
            tendencia_alcista = close_hoy > sma_50_actual
            tendencia_bajista = close_hoy < sma_50_actual

            # 🚀 AJUSTAR SEÑALES CON FILTRO DE VOLATILIDAD (CRÍTICO PARA CONCURSO)
            if condiciones_compra and volatil_alto and not tendencia_bajista:
                accion_recomendacion = "Comprar"
                etf_comprar.append(ticker)
                # 🔥 Confianza ajustada para concurso
                if rsi_actual < 30:
                    confianza = 95
                elif bb_position < 0.20:
                    confianza = 85
                else:
                    confianza = 70
            elif condiciones_venta and volatil_alto and not tendencia_alcista:
                accion_recomendacion = "Vender"
                etf_vender.append(ticker)
                # 🔥 Confianza ajustada para concurso
                if rsi_actual > 70:
                    confianza = 95
                elif bb_position > 0.80:
                    confianza = 85
                else:
                    confianza = 70
            else:
                accion_recomendacion = "Esperar"
                etf_mantener.append(ticker)
                confianza = 50

            # 🚀 GESTIÓN DE RIESGO PARA CONCURSO (MÁS APRETADA)
            if accion_recomendacion == "Comprar":
                stop_loss = bb_lower_actual * 0.995  # MÁS APRETADO (0.5%)
                take_profit = bb_upper_actual * 0.98  # OBJETIVO AMBICIOSO (banda superior)
                ratio_risk_reward = (take_profit - close_hoy) / (close_hoy - stop_loss)
            elif accion_recomendacion == "Vender":
                stop_loss = bb_upper_actual * 1.005
                take_profit = bb_lower_actual * 1.02
                ratio_risk_reward = (close_hoy - take_profit) / (stop_loss - close_hoy)
            else:
                stop_loss = close_hoy * 0.95
                take_profit = close_hoy * 1.05
                ratio_risk_reward = 1.0
            
            # 🔥 SOLO OPERAR SI RATIO RISK/REWARD >= 2.0
            if accion_recomendacion != "Esperar" and ratio_risk_reward < 2.0:
                accion_recomendacion = "Esperar"
                if ticker in etf_comprar:
                    etf_comprar.remove(ticker)
                if ticker in etf_vender:
                    etf_vender.remove(ticker)
                etf_mantener.append(ticker)
                confianza = 40  # Baja confianza por mal ratio

            # Almacenar los resultados
            resultados.append(
                {
                    "Ticker": ticker,
                    "Precio": close_hoy,
                    "BB_Superior": bb_upper_actual,
                    "BB_Media": bb_middle_actual,
                    "BB_Inferior": bb_lower_actual,
                    "Posición_BB": round(bb_position, 3),
                    "RSI": round(rsi_actual, 1),
                    "SMA50": round(sma_50_actual, 2),
                    "Tendencia": "Alcista" if tendencia_alcista else "Bajista",
                    "Variación (%)": round(variacion_diaria, 2),
                    "Acción Recomendada": accion_recomendacion,
                    "Confianza (%)": confianza,
                    "Stop_Loss": round(stop_loss, 2),
                    "Take_Profit": round(take_profit, 2),
                    "Toque_Inf": "Sí" if toque_banda_inferior else "No",
                    "Toque_Sup": "Sí" if toque_banda_superior else "No"
                }
            )

        except Exception as e:
            console.print(f"[red]Error con {ticker}: {e}[/red]")
            continue

    # Crear un DataFrame a partir de los resultados
    df_resultados = pd.DataFrame(resultados)
    os.makedirs('data', exist_ok=True)    
    csv_file_path = f'data/suggest_enhanced_etf_strategy_{datetime.now():%Y%m%d_%H%M%S}.csv'
    df_resultados.to_csv(csv_file_path, index=False)

    # Imprimir la tabla de resultados
    print("\n[bold cyan]Resumen de Indicadores Técnicos:[/bold cyan]\n")
    print(df_resultados)

    # Mostrar recomendaciones
    print(
        f"\n[bold green]ETFs recomendadas para comprar:\n[/bold green] {','.join(etf_comprar)}"
    )
    print(
        f"[bold yellow]ETFs recomendadas para esperar:\n[/bold yellow] {','.join(etf_mantener)}"
    )
    print(
        f"[bold red]ETFs recomendadas para vender:\n[/bold red] {','.join(etf_vender)}"
    )


def pairs_trading_etf_leveraged():
    """
    Algoritmo Pairs Trading Mejorado con ETFs Apalancados
    Estrategia de mercado neutral optimizada para un concurso de corto plazo (4 semanas).
    """
    console.print("\n[bold blue]🔄 Pairs Trading Optimizado para Concurso (Corto Plazo)[/bold blue]")
    console.print("[yellow]Estrategia cuantitativa de mercado neutral con enfoque en velocidad de reversión[/yellow]")

    # =================================================================================
    # --- PARÁMETROS CLAVE DE LA ESTRATEGIA (AJUSTA AQUÍ) ---
    # =================================================================================
    Z_SCORE_ENTRADA = 1.9  # Umbral de Z-Score para considerar una entrada (más agresivo que 2.0)
    MAX_HALF_LIFE = 15     # Máximo de días para que el par revierta (CRUCIAL para concurso)
    PERIODO_DATOS = "6mo"  # Ventana de datos históricos ("1y", "6mo", "3mo")
    # =================================================================================

    # Pares de ETFs apalancados predefinidos con alta correlación histórica
    default_pairs = [
    # --- Pares Originales (Correlación Positiva) ---
    ("SOXL", "TECL"),  # Semiconductores 3x vs. Tecnología 3x
    ("SPXL", "TQQQ"),  # S&P 500 3x vs. NASDAQ 100 3x
    ("FAS", "XLF"),    # Financieros 3x vs. Financieros 1x
    ("SOXL", "SOXX"),  # Semiconductores 3x vs. Semiconductores 1x
    ("TECL", "XLK"),   # Tecnología 3x vs. Tecnología 1x
    ("SPY", "VOO"),    # S&P 500 1x (dos proveedores diferentes)
    ("SPY", "IVV"),    # S&P 500 1x (dos proveedores diferentes)
    ("QQQ", "TQQQ"),   # NASDAQ 100 1x vs. NASDAQ 100 3x
    ("XLK", "VGT"),    # Tecnología 1x (dos proveedores diferentes)
    ("GLD", "IAU"),    # Oro (dos ETFs que replican el precio del oro)
    ("EEM", "VWO"),    # Mercados Emergentes (dos proveedores diferentes)
    ("ACWI", "VT"),    # Mercados Globales (dos proveedores diferentes)
    
    # --- Pares Originales con Correlación Inversa ---
    ("SOXL", "SOXS"),  # Semiconductores 3x (alcista) vs. -3x (bajista)
    ("TECL", "TECS"),  # Tecnología 3x (alcista) vs. -3x (bajista)
    ("SPXL", "SPXS"),  # S&P 500 3x (alcista) vs. -3x (bajista)
    ("TQQQ", "SQQQ"),  # NASDAQ 100 3x (alcista) vs. -3x (bajista)
    ("FAS", "FAZ"),    # Financieros 3x (alcista) vs. -3x (bajista)
    ("TNA", "TZA"),    # Russell 2000 3x (alcista) vs. -3x (bajista)
    ("QQQ", "PSQ"),    # NASDAQ 100 1x (alcista) vs. -1x (bajista)
    
    # --- Nuevos Pares con Correlación Positiva ---
    ("VTI", "VOO"),    # Mercado total vs. S&P 500
    ("IVV", "SPLG"),   # S&P 500 (dos proveedores diferentes)
    ("XLE", "OXY1"),   # ETF de energía vs. Occidental Petroleum
    ("VNQ", "FUNO"),   # ETF de bienes raíces vs. fideicomiso inmobiliario
    ("MCHI", "KWEB"),  # China mercado general vs. China tecnología
    
    # --- Nuevos Pares con Correlación Inversa ---
    ("QLD", "SQQQ"),   # NASDAQ 100 2x (alcista) vs. -3x (bajista)
    ("TQQQ", "PSQ"),   # NASDAQ 100 3x (alcista) vs. -1x (bajista)
]
    
    console.print("\n[bold cyan]Pares de ETFs disponibles para análisis:[/bold cyan]")
    for i, (etf1, etf2) in enumerate(default_pairs, 1):
        console.print(f"   {i}. {etf1} / {etf2}")
    
    # Parámetros de configuración
    try:
        selection = input(f"\nSelecciona el número del par (1-{len(default_pairs)}) o presiona Enter para analizar todos: ")
        if selection.strip():
            pair_index = int(selection) - 1
            pairs_to_analyze = [default_pairs[pair_index]] if 0 <= pair_index < len(default_pairs) else default_pairs
        else:
            pairs_to_analyze = default_pairs
            
        # Calcular monto óptimo basado en número de pares
        optimal_capital = calculate_optimal_capital(pairs_to_analyze)
        monto_por_pata = float(input(f"Monto por cada lado de la operación (default: {optimal_capital}): ") or str(optimal_capital))
        lookback_window = int(input("Ventana de análisis en días (default: 30): ") or "30")
        
    except ValueError:
        console.print("[yellow]⚠️ Usando valores por defecto[/yellow]")
        pairs_to_analyze = default_pairs
        monto_por_pata = calculate_optimal_capital(pairs_to_analyze)
        lookback_window = 30
    
    console.print(f"\n[bold cyan]Configuración del análisis:[/bold cyan]")
    console.print(f"• Monto por operación: ${monto_por_pata:,.0f} por lado")
    console.print(f"• Ventana de análisis: {lookback_window} días")
    console.print(f"• Periodo histórico: {PERIODO_DATOS}")
    console.print(f"[bold red]• Umbral de Entrada (Z-Score): > {Z_SCORE_ENTRADA}[/bold red]")
    console.print(f"[bold green]• Half-Life Máximo: < {MAX_HALF_LIFE} días[/bold green]")
    
    results = []
    cointegrated_pairs = 0
    
    for etf1, etf2 in pairs_to_analyze:
        try:
            console.print(f"\n[dim]🔍 Analizando par: {etf1} / {etf2}...[/dim]")
            
            # Descargar datos históricos (periodo ajustado para concurso) prefiriendo MX; si no hay, usar US y convertir a MXN
            def _download_close_with_fallback(symbol_original: str, period: str) -> pd.Series:
                symbol_mx = normalize_ticker_to_mx(symbol_original)
                if symbol_mx == "__SKIP_ACTINVER__":
                    return pd.Series(dtype=float)
                # Intento 1: versión MX (ya en MXN)
                try:
                    df_mx = yf.download(symbol_mx, period=period, interval="1d", progress=False)
                except Exception:
                    df_mx = pd.DataFrame()
                if not df_mx.empty and "Close" in df_mx and not df_mx["Close"].dropna().empty:
                    s = df_mx["Close"].dropna().copy()
                    try:
                        s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
                    except Exception:
                        s.index = pd.to_datetime(s.index).normalize()
                    return s
                # Intento 2: versión US (USD) -> convertir a MXN con tipo de cambio actual
                try:
                    t = yf.Ticker(symbol_original)
                    h = t.history(period=period, interval="1d", auto_adjust=True, actions=False)
                except Exception:
                    h = pd.DataFrame()
                if not h.empty and "Close" in h and not h["Close"].dropna().empty:
                    s = h["Close"].dropna().copy()
                    try:
                        s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
                    except Exception:
                        s.index = pd.to_datetime(s.index).normalize()
                    try:
                        usd_mxn = float(get_usd_to_mxn_rate())
                        s = s * usd_mxn
                        console.print(f"[yellow]ℹ️ Usando ticker US para {symbol_original} y convirtiendo a MXN (TC={usd_mxn:.4f}).[/yellow]")
                    except Exception:
                        console.print("[yellow]⚠️ No se pudo obtener tipo de cambio. Usando USD sin convertir.[/yellow]")
                    return s
                return pd.Series(dtype=float)

            close1 = _download_close_with_fallback(etf1, PERIODO_DATOS)
            close2 = _download_close_with_fallback(etf2, PERIODO_DATOS)
            
            if close1.empty or close2.empty:
                console.print(f"[red]⚠️ No se pudieron obtener datos para {etf1} o {etf2}[/red]")
                continue
            
            # Alineación de datos (asegurar índices normalizados a fecha)
            try:
                close1.index = pd.to_datetime(close1.index).tz_localize(None).normalize()
                close2.index = pd.to_datetime(close2.index).tz_localize(None).normalize()
            except Exception:
                close1.index = pd.to_datetime(close1.index).normalize()
                close2.index = pd.to_datetime(close2.index).normalize()

            # Reindexar a días hábiles y rellenar hacia adelante para maximizar solapamiento
            start_date = max(close1.index.min(), close2.index.min())
            end_date = min(close1.index.max(), close2.index.max())
            if start_date >= end_date:
                console.print(f"[red]⚠️ Rango de fechas inválido para {etf1}/{etf2}[/red]")
                continue
            bday_idx = pd.date_range(start=start_date, end=end_date, freq='B')
            if len(bday_idx) == 0:
                console.print(f"[red]⚠️ Sin días hábiles para {etf1}/{etf2}[/red]")
                continue
            close1_b = close1.reindex(bday_idx).ffill()
            close2_b = close2.reindex(bday_idx).ffill()

            # Construir DataFrame conjunto con concat (más robusto) y descartar filas sin ambos precios
            try:
                close1_b = close1_b.copy(); close1_b.name = etf1
                close2_b = close2_b.copy(); close2_b.name = etf2
                pair_df = pd.concat([close1_b, close2_b], axis=1).dropna()
                # Asegurar nombres de columnas exactamente como los tickers
                if pair_df.shape[1] == 2:
                    pair_df.columns = [etf1, etf2]
                if pair_df.empty or len(pair_df) < 5:
                    # fallback a intersección estricta
                    common_dates = close1.index.intersection(close2.index)
                    s1 = close1.loc[common_dates].copy(); s1.name = etf1
                    s2 = close2.loc[common_dates].copy(); s2.name = etf2
                    pair_df = pd.concat([s1, s2], axis=1).dropna()
                    if pair_df.shape[1] == 2:
                        pair_df.columns = [etf1, etf2]
                if pair_df.empty:
                    console.print(f"[yellow]⚠️ Sin solapamiento útil para {etf1}/{etf2}[/yellow]")
                    continue
                # Extraer series ya alineadas
                close1_aligned = pair_df.iloc[:, 0]
                close2_aligned = pair_df.iloc[:, 1]
                common_dates = close1_aligned.index
            except Exception as e:
                console.print(f"[red]⚠️ Error alineando {etf1}/{etf2}: {e}")
                continue

            # Si la intersección o el ratio resultan vacíos, reintentar con Ticker.history(auto_adjust=True)
            def _retry_history_us(symbol: str, period: str) -> pd.Series:
                try:
                    t = yf.Ticker(symbol)
                    h = t.history(period=period, interval="1d", auto_adjust=True, actions=False)
                    if not h.empty and "Close" in h:
                        s = h["Close"].dropna().copy()
                        try:
                            s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
                        except Exception:
                            s.index = pd.to_datetime(s.index).normalize()
                        return s
                except Exception:
                    pass
                return pd.Series(dtype=float)

            if len(close1_aligned) == 0:
                us1 = _retry_history_us(etf1, PERIODO_DATOS)
                us2 = _retry_history_us(etf2, PERIODO_DATOS)
                if not us1.empty and not us2.empty:
                    common_dates = us1.index.intersection(us2.index)
                    close1_aligned = us1.loc[common_dates]
                    close2_aligned = us2.loc[common_dates]
            
            if len(close1_aligned) < 40:
                console.print(f"[red]⚠️ Datos insuficientes para {etf1}/{etf2} ({len(close1_aligned)} días)[/red]")
                continue
            
            # Calcular el ratio y Test de Cointegración (ADF)
            ratio = close1_aligned / close2_aligned
            ratio_clean = ratio.dropna()
            # Asegurar estructura 1D para todas las operaciones (algunos retornos pueden ser DataFrame 2D)
            if isinstance(ratio_clean, pd.DataFrame):
                ratio_series = ratio_clean.iloc[:, 0]
            else:
                ratio_series = ratio_clean

            # Validaciones previas: datos suficientes y variabilidad
            ratio_series = ratio_series.dropna()
            if len(ratio_series) < 30:
                console.print(f"[red]⚠️ Datos insuficientes en ratio {etf1}/{etf2} tras limpieza ({len(ratio_series)}). Se omite.[/red]")
                continue
            if ratio_series.nunique() <= 1:
                console.print(f"[red]⚠️ Ratio {etf1}/{etf2} es constante. Se omite.[/red]")
                continue

            if adfuller is None:
                console.print("[red]❌ statsmodels no está instalado: 'pip install statsmodels'. Se omite este par.[/red]")
                continue

            try:
                adf_result = adfuller(ratio_series.values.squeeze())
                p_value_cointegration = adf_result[1]
            except Exception as e:
                console.print(f"[red]❌ Error en ADF para {etf1}/{etf2}: {e}. Se omite.[/red]")
                continue
            
            if p_value_cointegration >= 0.05:
                console.print(f"[bold red]❌ NO es cointegrado (p-valor: {p_value_cointegration:.4f}). Se descarta.[/bold red]")
                continue
            else:
                console.print(f"[green]✅ Es cointegrado (p-valor: {p_value_cointegration:.4f})[/green]")
                cointegrated_pairs += 1
            
            # Optimización de ventana temporal
            windows_to_test = [20, 30, 45, 60]
            best_window = lookback_window
            min_volatility = float('inf')
            
            for window in windows_to_test:
                if len(ratio_series) > window and window > 2:
                    rolling_mean = ratio_series.rolling(window=window).mean()
                    rolling_std = ratio_series.rolling(window=window).std()
                    # Evitar ventanas con desviación estándar nula o NaN al final
                    with np.errstate(divide='ignore', invalid='ignore'):
                        z_series = (ratio_series - rolling_mean) / rolling_std
                    z_volatility = z_series.replace([np.inf, -np.inf], np.nan).dropna().std()
                    if z_volatility < min_volatility:
                        min_volatility = z_volatility
                        best_window = window
            
            # Calcular estadísticas del ratio con la ventana optimizada
            ratio_sma = ratio_series.rolling(window=best_window).mean()
            ratio_std = ratio_series.rolling(window=best_window).std()
            last_std = ratio_std.iloc[-1]
            if pd.isna(last_std) or last_std == 0:
                console.print(f"[yellow]⚠️ STD inválido con ventana {best_window} para {etf1}/{etf2}. Se omite.[/yellow]")
                continue
            z_score = (ratio_series.iloc[-1] - ratio_sma.iloc[-1]) / last_std
            
            # Cálculo de Half-Life de reversión
            ratio_lagged = ratio_series.shift(1).dropna()
            ratio_diff = ratio_series[1:] - ratio_lagged
            try:
                reg = LinearRegression().fit(ratio_lagged.values.reshape(-1, 1), ratio_diff.values)
                beta = float(reg.coef_[0])
                # Cálculo robusto de half-life: usar log1p y proteger casos límite (beta<=-1 o denom>=0)
                # Validar que beta sea finito y mayor que -1 para evitar log1p de valores <= -1
                if not np.isfinite(beta) or beta <= -1.0:
                    half_life = float('inf')
                else:
                    denom = float(np.log1p(beta))
                    if not np.isfinite(denom) or denom >= 0:
                        half_life = float('inf')
                    else:
                        half_life = -np.log(2.0) / denom
            except Exception as e:
                console.print(f"[yellow]⚠️ No se pudo estimar half-life para {etf1}/{etf2}: {e}. Se omite.[/yellow]")
                continue
            
            # Lógica de señales con umbral configurable
            signal, action_detail, confidence, stop_loss_level = "ESPERAR", "", "BAJA", "N/A"
            stop_loss_z_score = Z_SCORE_ENTRADA + 1.5 # Stop loss dinámico
            
            if abs(z_score) >= Z_SCORE_ENTRADA:
                confidence = "ALTA"
                if z_score > Z_SCORE_ENTRADA:
                    signal = "VENDER ETF1 / COMPRAR ETF2"
                    action_detail = f"Vender {etf1}, Comprar {etf2} (Ratio sobrevalorado)"
                    stop_loss_level = f"Z-Score > {stop_loss_z_score:.1f}"
                else:
                    signal = "COMPRAR ETF1 / VENDER ETF2"
                    action_detail = f"Comprar {etf1}, Vender {etf2} (Ratio subvalorado)"
                    stop_loss_level = f"Z-Score < -{stop_loss_z_score:.1f}"

            # Dimensionamiento de posición y calidad del par
            price1_current = close1_aligned.iloc[-1]
            price2_current = close2_aligned.iloc[-1]
            acciones_etf1 = int(monto_por_pata / price1_current)
            acciones_etf2 = int(monto_por_pata / price2_current)
            correlation = close1_aligned.corr(close2_aligned)
            
            pair_quality = "EXCELENTE" if (correlation > 0.8 and half_life < 20 and p_value_cointegration < 0.01) else \
                           "BUENA" if (correlation > 0.7 and half_life < 40 and p_value_cointegration < 0.03) else \
                           "REGULAR"
            
            results.append({
                "Par": f"{etf1}/{etf2}", "Z-Score": z_score, "P-Valor Cointeg.": p_value_cointegration,
                "Half-Life (días)": half_life, "Calidad Par": pair_quality, "Señal": signal, "Confianza": confidence,
                "Acción Detallada": action_detail, "Stop-Loss": stop_loss_level, "Correlación": correlation,
                "Acciones ETF1": acciones_etf1, "Valor ETF1": acciones_etf1 * price1_current,
                "Acciones ETF2": acciones_etf2, "Valor ETF2": acciones_etf2 * price2_current,
            })
            
        except Exception as e:
            console.print(f"[red]❌ Error analizando {etf1}/{etf2}: {e}[/red]")
            continue
    
    if not results:
        console.print("[bold red]❌ No se encontraron pares cointegrados válidos para analizar.[/bold red]")
        return
    
    console.print(f"\n[bold green]✅ Pares cointegrados encontrados: {cointegrated_pairs}/{len(pairs_to_analyze)}[/bold green]")
    
    # Crear DataFrame y ordenar por calidad y Z-Score
    df_results = pd.DataFrame(results)
    quality_order = {"EXCELENTE": 3, "BUENA": 2, "REGULAR": 1}
    df_results['Quality_Score'] = df_results['Calidad Par'].map(quality_order)
    df_results['Z_Score_Abs'] = df_results['Z-Score'].abs()
    df_results = df_results.sort_values(['Quality_Score', 'Z_Score_Abs'], ascending=[False, False])
    
    # --- Tablas de Resultados con Rich ---
    table = Table(title=f"🔄 Análisis de Pares (Z-Score > {Z_SCORE_ENTRADA}, Half-Life < {MAX_HALF_LIFE}d)")
    table.add_column("Par", style="cyan")
    table.add_column("Z-Score", style="yellow")
    table.add_column("P-Valor", style="magenta")
    table.add_column("Half-Life", style="green")
    table.add_column("Calidad", style="blue")
    table.add_column("Señal", style="bold")
    
    for _, row in df_results.iterrows():
        z_score_str = f"[bold red]{row['Z-Score']:.2f}[/bold red]" if abs(row['Z-Score']) >= Z_SCORE_ENTRADA else f"{row['Z-Score']:.2f}"
        half_life_str = f"[bold green]{row['Half-Life (días)']:.1f}d[/bold green]" if row['Half-Life (días)'] <= 10 else f"[yellow]{row['Half-Life (días)']:.1f}d[/yellow]"
        signal_str = f"[bold green]{row['Señal']}[/bold green]" if row['Señal'] != "ESPERAR" else row['Señal']
        
        table.add_row(row['Par'], z_score_str, f"{row['P-Valor Cointeg.']:.4f}", half_life_str, row['Calidad Par'], signal_str)
    
    console.print(table)
    
    # --- Recomendaciones Finales Enfocadas en el Concurso ---
    oportunidades = df_results[
        (df_results['Z_Score_Abs'] >= Z_SCORE_ENTRADA) & 
        (df_results['Half-Life (días)'] <= MAX_HALF_LIFE)
    ]
    
    console.print(f"\n[bold red]🎯 OPORTUNIDADES PRINCIPALES PARA EL CONCURSO ({len(oportunidades)} encontradas):[/bold red]")
    if not oportunidades.empty:
        position_table = Table(title="Posiciones Sugeridas (Mercado Neutral)")
        position_table.add_column("Par", style="cyan")
        position_table.add_column("Acción", style="yellow")
        position_table.add_column("Acciones ETF1", style="green")
        position_table.add_column("Acciones ETF2", style="green")
        position_table.add_column("Stop-Loss", style="red")
        
        for _, row in oportunidades.iterrows():
            console.print(f"  • [bold]{row['Par']}[/bold]: {row['Acción Detallada']}")
            console.print(f"    └─ Z-Score: {row['Z-Score']:.2f}, Half-Life: {row['Half-Life (días)']:.1f}d, Correlación: {row['Correlación']:.3f}")
            position_table.add_row(
                row['Par'],
                "Vender / Comprar" if row['Z-Score'] > 0 else "Comprar / Vender",
                f"{row['Acciones ETF1']:,}",
                f"{row['Acciones ETF2']:,}",
                row['Stop-Loss']
            )
        console.print(position_table)
    else:
        console.print("[yellow]No se encontraron oportunidades que cumplan con todos los criterios de entrada para el concurso.[/yellow]")

    # Guardar resultados
    os.makedirs('data', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file_path = f'data/pairs_trading_concurso_{timestamp}.csv'
    df_results.to_csv(csv_file_path, index=False)
    console.print(f"\n[bold yellow]📁 Análisis completo guardado en: {csv_file_path}[/bold yellow]")
    
    # Mostrar resultados en formato separado por comas para copy-paste
    if not oportunidades.empty:
        console.print(f"\n[bold green]📋 RESULTADOS LISTOS PARA COPY-PASTE:[/bold green]")
        console.print("[bold cyan]ETFs recomendados para COMPRAR (separados por comas):[/bold cyan]")
        
        # Extraer solo los tickers que se recomiendan comprar según la señal
        recommended_tickers = []
        for _, row in oportunidades.iterrows():
            par = row['Par']
            etf1, etf2 = par.split('/')
            z_score = row['Z-Score']
            
            # Si Z-Score > 0, vender ETF1 y comprar ETF2
            # Si Z-Score < 0, comprar ETF1 y vender ETF2
            if z_score > 0:
                recommended_tickers.append(etf2.strip())  # Comprar ETF2
            else:
                recommended_tickers.append(etf1.strip())  # Comprar ETF1
        
        # Crear string separado por comas
        tickers_string = ','.join(recommended_tickers)
        console.print(f"[bold white]{tickers_string}[/bold white]")
        
        # Mostrar también los pares específicos con la acción recomendada
        console.print(f"\n[bold cyan]Pares específicos con acción recomendada:[/bold cyan]")
        pairs_with_action = []
        for _, row in oportunidades.iterrows():
            par = row['Par']
            z_score = row['Z-Score']
            if z_score > 0:
                pairs_with_action.append(f"{par} (Comprar {par.split('/')[1]})")
            else:
                pairs_with_action.append(f"{par} (Comprar {par.split('/')[0]})")
        pairs_string = ','.join(pairs_with_action)
        console.print(f"[bold white]{pairs_string}[/bold white]")
    else:
        console.print(f"\n[bold yellow]No hay pares recomendados para mostrar en formato copy-paste.[/bold yellow]")

def volatility_based_capital_allocation():
    """
    Asignación de Capital Basada en Volatilidad (Gestión de Riesgo Avanzada)
    Ajusta el tamaño de las posiciones según la volatilidad (ATR) de cada activo
    para mantener un riesgo uniforme en todas las operaciones.
    """
    console.print("[bold blue]📊 Asignación de Capital Basada en Volatilidad[/bold blue]")
    console.print("[yellow]Gestión de Riesgo Avanzada con ATR (Average True Range)[/yellow]")
    
    # ETFs apalancados por defecto
    default_tickers = ['SOXL', 'TECL', 'SPXL', 'TQQQ', 'FAS', 'TNA', 'SPXS', 'SOXS', 'TECS']
    
    # Solicitar tickers del usuario
    input_tickers = input(f"Ingresa los ETFs separados por comas (presiona Enter para usar: {','.join(default_tickers)}): ")
    
    if input_tickers.strip():
        tickers = [ticker.strip().upper() for ticker in input_tickers.split(",")]
    else:
        tickers = default_tickers
        console.print(f"[yellow]📊 Usando ETFs por defecto...[/yellow]")
    
    # Parámetros de gestión de riesgo
    try:
        optimal_capital = calculate_optimal_capital(tickers)
        capital_total = float(input(f"Ingresa el capital total disponible (default: {optimal_capital}): ") or str(optimal_capital))
        riesgo_por_operacion = float(input("Ingresa el % de riesgo por operación (default: 2.0): ") or "2.0") / 100
    except ValueError:
        capital_total = calculate_optimal_capital(tickers)
        riesgo_por_operacion = 0.02
        console.print(f"[yellow]⚠️ Usando valores por defecto: Capital={capital_total:,.0f}, Riesgo=2%[/yellow]")
    
    console.print(f"\n[bold cyan]Analizando {len(tickers)} ETFs para asignación de capital...[/bold cyan]")
    console.print(f"Capital Total: ${capital_total:,.2f}")
    console.print(f"Riesgo por Operación: {riesgo_por_operacion*100:.1f}%")
    
    results = []
    
    for ticker in tickers:
        try:
            # Usar ticker mexicano (.MX) exclusivamente
            mx_ticker = normalize_ticker_to_mx(ticker)
            console.print(f"[dim]Procesando {ticker} como {mx_ticker}...[/dim]")
            
            # Descargar datos históricos (3 meses para ATR) usando ticker mexicano
            df = yf.download(mx_ticker, period="3mo", interval="1d", progress=False)
            df.dropna(inplace=True)
            
            if df.empty or len(df) < 20:
                console.print(f"[red]⚠️ Datos insuficientes para {ticker}[/red]")
                continue
            
            # Obtener precios
            high = df["High"]
            low = df["Low"]
            close = df["Close"]
            
            if isinstance(high, pd.DataFrame):
                high = high.iloc[:, 0]
            if isinstance(low, pd.DataFrame):
                low = low.iloc[:, 0]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            
            # Calcular True Range
            prev_close = close.shift(1)
            tr1 = high - low
            tr2 = abs(high - prev_close)
            tr3 = abs(low - prev_close)
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Calcular ATR (Average True Range) de 14 días
            atr_14 = true_range.rolling(window=14).mean()
            atr_actual = float(atr_14.iloc[-1])
            
            # Precio actual
            precio_actual = float(close.iloc[-1])
            
            # Calcular volatilidad porcentual
            volatilidad_pct = (atr_actual / precio_actual) * 100
            
            # Calcular tamaño de posición basado en riesgo
            # Fórmula: (Capital * % Riesgo) / ATR = Cantidad de acciones
            # PERO limitado a máximo configurado por ticker
            max_inversion_por_ticker = MAX_ALLOCATION_PER_TICKER
            inversion_objetivo = min(capital_total * riesgo_por_operacion, max_inversion_por_ticker)
            cantidad_acciones = int(inversion_objetivo / atr_actual)
            
            # Calcular inversión total para esta posición
            inversion_total = cantidad_acciones * precio_actual
            
            # Asegurar que no exceda el límite de $400,000
            if inversion_total > max_inversion_por_ticker:
                cantidad_acciones = int(max_inversion_por_ticker / precio_actual)
                inversion_total = cantidad_acciones * precio_actual
            
            # Calcular porcentaje del portafolio
            porcentaje_portafolio = (inversion_total / capital_total) * 100
            
            # Calcular stop loss sugerido (1 ATR por debajo del precio actual)
            stop_loss = precio_actual - atr_actual
            stop_loss_pct = (atr_actual / precio_actual) * 100
            
            # Calcular rendimientos históricos
            returns = close.pct_change().dropna()
            volatilidad_historica = returns.std() * np.sqrt(252) * 100  # Anualizada
            
            # Calcular Sharpe ratio aproximado (últimos 60 días)
            if len(returns) >= 60:
                recent_returns = returns.tail(60)
                avg_return = recent_returns.mean() * 252  # Anualizado
                sharpe_ratio = avg_return / (recent_returns.std() * np.sqrt(252)) if recent_returns.std() > 0 else 0
            else:
                sharpe_ratio = 0
            
            results.append({
                "Ticker": ticker,
                "Precio Actual": precio_actual,
                "ATR (14d)": atr_actual,
                "Volatilidad ATR (%)": volatilidad_pct,
                "Volatilidad Histórica (%)": volatilidad_historica,
                "Cantidad Acciones": cantidad_acciones,
                "Inversión Total": inversion_total,
                "% Portafolio": porcentaje_portafolio,
                "Stop Loss": stop_loss,
                "Stop Loss (%)": stop_loss_pct,
                "Sharpe Ratio": sharpe_ratio
            })
            
        except Exception as e:
            console.print(f"[red]❌ Error procesando {ticker}: {e}[/red]")
            continue
    
    if not results:
        console.print("[bold red]❌ No se pudieron procesar los ETFs[/bold red]")
        return
    
    # Crear DataFrame y ordenar por volatilidad ATR
    df_results = pd.DataFrame(results)
    
    # Aplicar límite configurado por ticker
    df_results = apply_max_allocation_limit(df_results)
    
    # Recalcular porcentajes después de aplicar el límite
    total_inversion_after_limit = df_results['Inversión Total'].sum()
    if total_inversion_after_limit > 0:
        df_results['% Portafolio'] = (df_results['Inversión Total'] / total_inversion_after_limit) * 100
    
    df_results = df_results.sort_values("Volatilidad ATR (%)", ascending=True)
    
    # Mostrar tabla con Rich
    table = Table(title="📊 Asignación de Capital Basada en Volatilidad (ATR)")
    table.add_column("Ticker", style="cyan", no_wrap=True)
    table.add_column("Precio", style="magenta")
    table.add_column("ATR", style="yellow")
    table.add_column("Vol ATR (%)", style="red")
    table.add_column("Cantidad", style="green")
    table.add_column("Inversión", style="blue")
    table.add_column("% Port.", style="white")
    table.add_column("Stop Loss", style="red")
    table.add_column("Sharpe", style="cyan")
    
    for _, row in df_results.iterrows():
        vol_atr = row['Volatilidad ATR (%)']
        
        # Colorear volatilidad según nivel
        if vol_atr > 8:
            vol_str = f"[bold red]{vol_atr:.2f}%[/bold red]"
        elif vol_atr > 5:
            vol_str = f"[yellow]{vol_atr:.2f}%[/yellow]"
        else:
            vol_str = f"[green]{vol_atr:.2f}%[/green]"
        
        table.add_row(
            row['Ticker'],
            f"${row['Precio Actual']:.2f}",
            f"{row['ATR (14d)']:.2f}",
            vol_str,
            f"{row['Cantidad Acciones']:,}",
            f"${row['Inversión Total']:,.0f}",
            f"{row['% Portafolio']:.1f}%",
            f"${row['Stop Loss']:.2f}",
            f"{row['Sharpe Ratio']:.2f}"
        )
    
    console.print(table)
    
    # Calcular estadísticas del portafolio
    total_inversion = df_results['Inversión Total'].sum()
    capital_restante = capital_total - total_inversion
    num_posiciones = len(df_results)
    vol_promedio = df_results['Volatilidad ATR (%)'].mean()
    
    console.print(f"\n[bold blue]📈 Resumen del Portafolio:[/bold blue]")
    console.print(f"• Total Invertido: ${total_inversion:,.2f} ({total_inversion/capital_total*100:.1f}% del capital)")
    console.print(f"• Capital Restante: ${capital_restante:,.2f}")
    console.print(f"• Número de Posiciones: {num_posiciones}")
    console.print(f"• Volatilidad Promedio ATR: {vol_promedio:.2f}%")
    console.print(f"• Riesgo por Operación: {riesgo_por_operacion*100:.1f}%")
    
    # Recomendaciones por volatilidad
    low_vol = df_results[df_results['Volatilidad ATR (%)'] <= 5]
    medium_vol = df_results[(df_results['Volatilidad ATR (%)'] > 5) & (df_results['Volatilidad ATR (%)'] <= 8)]
    high_vol = df_results[df_results['Volatilidad ATR (%)'] > 8]
    
    console.print(f"\n[bold green]🟢 BAJA VOLATILIDAD ({len(low_vol)} ETFs):[/bold green]")
    if not low_vol.empty:
        for ticker in low_vol['Ticker'].tolist():
            console.print(f"   • {ticker}: Posiciones más grandes, menor riesgo")
    
    console.print(f"\n[bold yellow]🟡 VOLATILIDAD MEDIA ({len(medium_vol)} ETFs):[/bold yellow]")
    if not medium_vol.empty:
        for ticker in medium_vol['Ticker'].tolist():
            console.print(f"   • {ticker}: Posiciones balanceadas")
    
    console.print(f"\n[bold red]🔴 ALTA VOLATILIDAD ({len(high_vol)} ETFs):[/bold red]")
    if not high_vol.empty:
        for ticker in high_vol['Ticker'].tolist():
            console.print(f"   • {ticker}: Posiciones más pequeñas, mayor potencial")
    
    # Guardar resultados
    os.makedirs('data', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file_path = f'data/volatility_capital_allocation_{timestamp}.csv'
    df_results.to_csv(csv_file_path, index=False)
    
    console.print(f"\n[bold blue]💡 Conceptos Clave de Gestión de Riesgo:[/bold blue]")
    console.print("• [green]ATR (Average True Range)[/green]: Mide la volatilidad promedio de 14 días")
    console.print("• [yellow]Riesgo Uniforme[/yellow]: Todas las posiciones tienen el mismo riesgo en pesos")
    console.print("• [cyan]Stop Loss Sugerido[/cyan]: 1 ATR por debajo del precio de entrada")
    console.print("• [red]Posición Inversa[/red]: Mayor volatilidad = menor tamaño de posición")
    
    console.print(f"\n[bold yellow]📁 Resultados guardados en: {csv_file_path}[/bold yellow]")


def suggest_enhanced_etf_strategy_leveraged(
    tickers=['FAS','FAZ', 'QLD', 'SOXL','SOXS', 'SPXL', 'SPXS', 'SQQQ', 'TECL', 'TECS', 'TNA', 'TQQQ']
):
    """
    Algoritmo optimizado para ETFs apalancados:
    Busca ETFs que han tenido tendencia alcista en las últimas 5 semanas
    pero hoy tienen porcentaje negativo (oportunidad de compra en dip)
    """
    console.print("[bold blue]🚀 Análisis Optimizado de ETFs Apalancados[/bold blue]")
    console.print("[yellow]Buscando ETFs con tendencia alcista de 5 semanas pero con caída hoy...[/yellow]")
    
    # Solicitar al usuario que ingrese tickers (con soporte de comandos estilo vim)
    input_tickers = input("Ingresa las ETFs separadas por comas (presiona Enter para usar valores por defecto): ")
    cmd = input_tickers.strip()
    if cmd.startswith(":"):
        cmd_lower = cmd[1:].strip().lower()
        if cmd_lower in ("q", "quit"):
            exit_program(); return
        if cmd_lower in ("b", "back"):
            return
        if cmd_lower.isdigit():
            try:
                idx = int(cmd_lower) - 1
                console.print(f"[yellow]Saltando a opción {idx+1} del menú...[/yellow]")
                raise SystemExit(f"__GOTO_MENU_OPTION__:{idx}")
            except Exception:
                return

    if input_tickers.strip():
        tickers = [ticker.strip().upper() for ticker in input_tickers.split(",")]
    else:
        console.print(f"\n[bold yellow]📊 Usando ETFs apalancados sugeridos por defecto...[/bold yellow]")

    resultados = []
    etf_comprar_oportunidad = []  # ETFs con tendencia alcista pero caída hoy
    etf_comprar_momentum = []     # ETFs con momentum fuerte
    etf_mantener = []
    etf_vender = []

    console.print(f"\n[bold cyan]Analizando {len(tickers)} ETFs apalancados...[/bold cyan]")

    for ticker in tickers:
        try:
            console.print(f"[dim]Procesando {ticker}...[/dim]")
            
            # Descargar datos de los últimos 6 meses para análisis completo usando ticker mexicano
            mx_ticker = normalize_ticker_to_mx(ticker)
            df_ticker = yf.download(mx_ticker, period="6mo", interval="1d", progress=False)
            df_ticker.dropna(inplace=True)

            if df_ticker.empty or len(df_ticker) < 35:  # Necesitamos al menos 5 semanas de datos
                console.print(f"[red]⚠️ Datos insuficientes para {ticker}[/red]")
                continue

            # Asegurar Series 1D
            close = df_ticker["Close"]
            high = df_ticker["High"]
            low = df_ticker["Low"]
            volume = df_ticker["Volume"]
            
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            if isinstance(high, pd.DataFrame):
                high = high.iloc[:, 0]
            if isinstance(low, pd.DataFrame):
                low = low.iloc[:, 0]
            if isinstance(volume, pd.DataFrame):
                volume = volume.iloc[:, 0]

            # === ANÁLISIS DE TENDENCIA DE 5 SEMANAS ===
            # Calcular rendimiento semanal de las últimas 5 semanas
            precio_5_semanas_atras = float(close.iloc[-35])  # 5 semanas = ~35 días
            precio_4_semanas_atras = float(close.iloc[-28])
            precio_3_semanas_atras = float(close.iloc[-21])
            precio_2_semanas_atras = float(close.iloc[-14])
            precio_1_semana_atras = float(close.iloc[-7])
            precio_hoy = float(close.iloc[-1])
            precio_ayer = float(close.iloc[-2])

            # Calcular rendimientos semanales
            rendimiento_5_semanas = ((precio_hoy - precio_5_semanas_atras) / precio_5_semanas_atras) * 100
            rendimiento_semanal_1 = ((precio_4_semanas_atras - precio_5_semanas_atras) / precio_5_semanas_atras) * 100
            rendimiento_semanal_2 = ((precio_3_semanas_atras - precio_4_semanas_atras) / precio_4_semanas_atras) * 100
            rendimiento_semanal_3 = ((precio_2_semanas_atras - precio_3_semanas_atras) / precio_3_semanas_atras) * 100
            rendimiento_semanal_4 = ((precio_1_semana_atras - precio_2_semanas_atras) / precio_2_semanas_atras) * 100
            rendimiento_semanal_5 = ((precio_hoy - precio_1_semana_atras) / precio_1_semana_atras) * 100

            # Contar semanas positivas
            semanas_positivas = sum([
                rendimiento_semanal_1 > 0,
                rendimiento_semanal_2 > 0,
                rendimiento_semanal_3 > 0,
                rendimiento_semanal_4 > 0,
                rendimiento_semanal_5 > 0
            ])

            # Variación diaria (hoy)
            variacion_diaria = ((precio_hoy - precio_ayer) / precio_ayer) * 100

            # === INDICADORES TÉCNICOS OPTIMIZADOS ===
            rsi = RSIIndicator(close, window=14).rsi()
            stochastic = StochasticOscillator(high, low, close, window=14, smooth_window=3)
            macd_indicator = MACD(close, window_slow=26, window_fast=12, window_sign=9)
            bollinger = BollingerBands(close, window=20)
            
            # Medias móviles para tendencia
            sma_20 = close.rolling(window=20).mean()
            sma_50 = close.rolling(window=50).mean()
            ema_12 = close.ewm(span=12).mean()

            # Valores actuales
            rsi_actual = float(rsi.iloc[-1])
            stochastic_actual = float(stochastic.stoch().iloc[-1])
            macd_actual = float(macd_indicator.macd().iloc[-1])
            macd_signal = float(macd_indicator.macd_signal().iloc[-1])
            macd_histogram = float(macd_indicator.macd_diff().iloc[-1])
            bollinger_high = float(bollinger.bollinger_hband().iloc[-1])
            bollinger_low = float(bollinger.bollinger_lband().iloc[-1])
            sma_20_actual = float(sma_20.iloc[-1])
            sma_50_actual = float(sma_50.iloc[-1])
            ema_12_actual = float(ema_12.iloc[-1])

            # Volumen promedio vs actual
            volumen_promedio = float(volume.rolling(window=20).mean().iloc[-1])
            volumen_actual = float(volume.iloc[-1])
            volumen_ratio = volumen_actual / volumen_promedio if volumen_promedio > 0 else 1

            # === CRITERIOS OPTIMIZADOS PARA ETFs APALANCADOS ===
            
            # 1. OPORTUNIDAD DE COMPRA EN DIP (Principal objetivo)
            tendencia_alcista_5_semanas = (
                rendimiento_5_semanas > 3 and  # Ganancia total > 3% en 5 semanas
                semanas_positivas >= 3 and     # Al menos 3 de 5 semanas positivas
                precio_hoy > sma_20_actual     # Precio por encima de SMA 20
            )
            
            caida_hoy_oportunidad = (
                variacion_diaria < -0.5 and    # Caída significativa hoy
                variacion_diaria > -8 and      # Pero no colapso
                rsi_actual < 70 and            # No sobrecomprado
                precio_hoy > bollinger_low     # No en pánico
            )
            
            # 2. MOMENTUM FUERTE (Alternativo)
            momentum_fuerte = (
                variacion_diaria > 2 and       # Subida fuerte hoy
                rsi_actual < 80 and            # Aún no sobrecomprado
                macd_histogram > 0 and         # MACD positivo
                volumen_ratio > 1.2 and        # Volumen alto
                precio_hoy > ema_12_actual     # Por encima de EMA rápida
            )
            
            # 3. CONDICIONES DE VENTA
            condiciones_venta = (
                rsi_actual > 85 or             # Muy sobrecomprado
                (precio_hoy > bollinger_high * 1.02 and variacion_diaria > 5) or  # Muy por encima de Bollinger
                (rendimiento_5_semanas > 25 and rsi_actual > 75)  # Ganancia excesiva
            )

            # === DETERMINAR ACCIÓN RECOMENDADA ===
            if tendencia_alcista_5_semanas and caida_hoy_oportunidad:
                accion_recomendacion = "🎯 COMPRAR - Oportunidad en Dip"
                etf_comprar_oportunidad.append(ticker)
                prioridad = "ALTA"
            elif momentum_fuerte:
                accion_recomendacion = "🚀 COMPRAR - Momentum"
                etf_comprar_momentum.append(ticker)
                prioridad = "MEDIA"
            elif condiciones_venta:
                accion_recomendacion = "💰 VENDER - Tomar Ganancias"
                etf_vender.append(ticker)
                prioridad = "ALTA"
            else:
                accion_recomendacion = "⏳ ESPERAR"
                etf_mantener.append(ticker)
                prioridad = "BAJA"

            # Calcular score de oportunidad
            score_oportunidad = 0
            if tendencia_alcista_5_semanas: score_oportunidad += 40
            if caida_hoy_oportunidad: score_oportunidad += 30
            if rsi_actual < 50: score_oportunidad += 10
            if macd_histogram > 0: score_oportunidad += 10
            if volumen_ratio > 1.1: score_oportunidad += 10

            # Almacenar resultados
            resultados.append({
                "Ticker": ticker,
                "Rendimiento 5 Sem (%)": round(rendimiento_5_semanas, 2),
                "Semanas Positivas": f"{semanas_positivas}/5",
                "Variación Hoy (%)": round(variacion_diaria, 2),
                "RSI": round(rsi_actual, 1),
                "MACD": round(macd_histogram, 4),
                "Precio vs SMA20": "✅" if precio_hoy > sma_20_actual else "❌",
                "Volumen Ratio": round(volumen_ratio, 2),
                "Score Oportunidad": score_oportunidad,
                "Prioridad": prioridad,
                "Acción Recomendada": accion_recomendacion,
            })

        except Exception as e:
            console.print(f"[red]❌ Error con {ticker}: {e}[/red]")
            continue

    # === MOSTRAR RESULTADOS ===
    if not resultados:
        console.print("[bold red]❌ No se pudieron analizar ETFs[/bold red]")
        return

    # Crear DataFrame y ordenar por score de oportunidad
    df_resultados = pd.DataFrame(resultados)
    df_resultados = df_resultados.sort_values("Score Oportunidad", ascending=False)

    # Guardar resultados
    os.makedirs('data', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file_path = f'data/etf_leveraged_optimized_{timestamp}.csv'
    df_resultados.to_csv(csv_file_path, index=False)

    # Mostrar tabla con Rich
    table = Table(title="🚀 Análisis Optimizado de ETFs Apalancados")
    for col in df_resultados.columns:
        table.add_column(col, style="cyan" if col == "Ticker" else None)
    
    for _, row in df_resultados.iterrows():
        table.add_row(*[str(val) for val in row])
    
    console.print(table)

    # === RECOMENDACIONES FINALES ===
    console.print(f"\n[bold green]🎯 OPORTUNIDADES DE COMPRA EN DIP ({len(etf_comprar_oportunidad)}):[/bold green]")
    if etf_comprar_oportunidad:
        for etf in etf_comprar_oportunidad:
            etf_data = df_resultados[df_resultados['Ticker'] == etf].iloc[0]
            console.print(f"   • {etf}: {etf_data['Rendimiento 5 Sem (%)']}% en 5 sem, {etf_data['Variación Hoy (%)']}% hoy (Score: {etf_data['Score Oportunidad']})")
    else:
        console.print("   [dim]No hay oportunidades de dip detectadas[/dim]")

    console.print(f"\n[bold blue]🚀 MOMENTUM FUERTE ({len(etf_comprar_momentum)}):[/bold blue]")
    if etf_comprar_momentum:
        console.print(f"   {', '.join(etf_comprar_momentum)}")
    else:
        console.print("   [dim]No hay ETFs con momentum fuerte[/dim]")

    console.print(f"\n[bold red]💰 CONSIDERAR VENTA ({len(etf_vender)}):[/bold red]")
    if etf_vender:
        console.print(f"   {', '.join(etf_vender)}")
    else:
        console.print("   [dim]No hay ETFs para vender[/dim]")

    console.print(f"\n[bold yellow]📊 Archivo guardado: {csv_file_path}[/bold yellow]")
    
    # Mostrar ETF con mejor score
    if len(df_resultados) > 0:
        mejor_etf = df_resultados.iloc[0]
        console.print(f"\n[bold green]⭐ MEJOR OPORTUNIDAD: {mejor_etf['Ticker']} (Score: {mejor_etf['Score Oportunidad']})[/bold green]")



def suggest_technical_beta1(
    tickers=[
         "XLK", "GDX", "AAPL", "CEMEXCPO.MX", "AXP", "AMZN", "GOOGL", "META",
    "QQQ", "MSFT", "JPM", "BOLSAA.MX", "XLF", "TSLA", "GMEXICOB.MX", "PE&OLES.MX",
    "WFC.MX", "BAC.MX", "SOFI.MX", "GAPB.MX", "ORBIA.MX", "AAPL.MX", "NVDA.MX", 
    "AMD.MX", "INTC.MX", "UBER.MX", "LABB.MX", "BIMBOA.MX", "PINS.MX", "TGT.MX", 
    "LCID.MX", "UPST.MX", "MARA.MX", "RIOT.MX", "BNGO.MX", "ALFAA.MX", "ALSEA.MX", 
    "AMXB.MX", "BBAJIOO.MX", "GM.MX", "NKLA.MX", "ACTINVRB.MX", "TALN.MX", "FSLR.MX", 
    "SPCE.MX", "FUBO.MX", "BYND.MX", "C.MX", "ATOS.MX"
    ]
):
    # Solicitar al usuario que ingrese tickers
    input_tickers = input(
        "Ingresa las acciones separadas por comas (i.e. OMAB.MX,AAPL.MX,META.MX,MSFT.MX): "
    )

    if input_tickers.strip():  # Si se ingresaron tickers, usarlos
        tickers = [ticker.strip() for ticker in input_tickers.split(",")]
    else:
        console.print(
            f"\n[bold yellow] No se detectó entrada, usando valores sugeridos por defecto (versión .MX)...[/bold yellow]"
        )

    resultados = []  # Para almacenar los resultados
    acciones_comprar = []  # Para las acciones recomendadas para comprar
    acciones_mantener = []  # Para las acciones recomendadas para esperar
    acciones_vender = []  # Para las acciones recomendadas para vender

    # Iterar sobre cada ticker
    for ticker in tickers:
        try:
            # Descargar datos de los últimos 6 meses usando EXCLUSIVAMENTE ticker mexicano (.MX)
            mx_ticker = normalize_ticker_to_mx(ticker)
            console.print(f"[dim]Procesando {ticker} como {mx_ticker}...[/dim]")
            df_ticker = yf.download(mx_ticker, period="6mo", progress=False)
            df_ticker.dropna(inplace=True)

            if df_ticker.empty:
                print(f"Advertencia: No se encontraron datos para {ticker}.")
                continue

            # Asegurar Series 1D para indicadores técnicos
            close = df_ticker["Close"]
            high = df_ticker["High"]
            low = df_ticker["Low"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            if isinstance(high, pd.DataFrame):
                high = high.iloc[:, 0]
            if isinstance(low, pd.DataFrame):
                low = low.iloc[:, 0]

            # Calcular indicadores técnicos con Series 1D
            rsi = RSIIndicator(close, window=14).rsi()
            stochastic = StochasticOscillator(
                high,
                low,
                close,
                window=14,
                smooth_window=3,
            )
            macd = MACD(close).macd_diff()
            bollinger = BollingerBands(close)

            # Obtener datos recientes
            rsi_actual = rsi.iloc[-1]
            stochastic_actual = stochastic.stoch().iloc[-1]
            macd_actual = macd.iloc[-1]
            close_hoy = df_ticker["Close"].iloc[-1]
            close_ayer = df_ticker["Close"].iloc[-2]
            close_anteayer = df_ticker["Close"].iloc[-3]
            bollinger_high = bollinger.bollinger_hband().iloc[-1]
            bollinger_low = bollinger.bollinger_lband().iloc[-1]

            # Calcular las variaciones diarias
            variacion_ayer = (close_ayer - close_anteayer) / close_anteayer * 100
            variacion_hoy = (close_hoy - close_ayer) / close_ayer * 100

            # Condiciones de compra: variación ayer positiva, hoy negativa + indicadores técnicos
            condiciones_compra = (
                variacion_ayer < 0.24 and variacion_hoy < 0
                and rsi_actual < 58 and rsi_actual > 45 
                and macd_actual < 3 and macd_actual > -3 
            )

            # Determinar la acción recomendada
            if condiciones_compra:
                accion_recomendacion = "Comprar"
                acciones_comprar.append(ticker)
            else:
                accion_recomendacion = "Esperar"
                acciones_mantener.append(ticker)

            # Almacenar los resultados
            resultados.append(
                {
                    "Ticker": ticker,
                    "RSI": rsi_actual,
                    "Stochastic": stochastic_actual,
                    "MACD": macd_actual,
                    "Bollinger_High": bollinger_high,
                    "Bollinger_Low": bollinger_low,
                    "Variación Ayer (%)": variacion_ayer,
                    "Variación Hoy (%)": variacion_hoy,
                    "Acción Recomendada": accion_recomendacion,
                }
            )

        except Exception as e:
            print(f"Error con {ticker}: {e}")
            continue

    # Crear un DataFrame a partir de los resultados
    df_resultados = pd.DataFrame(resultados)
    
    os.makedirs('data', exist_ok=True)    
    csv_file_path = f'data/suggest_technical_{datetime.now():%Y%m%d_%H%M%S}.csv'
    df_resultados.to_csv(csv_file_path, index=False)

    # Imprimir la tabla de resultados
    print("\nResumen de Indicadores Técnicos:\n")
    print(df_resultados)

    # Mostrar recomendaciones
    print(f"\nAcciones recomendadas para comprar:\n {','.join(acciones_comprar)}")
    print(f"Acciones recomendadas para esperar:\n {','.join(acciones_mantener)}")
    print(f"Acciones recomendadas para vender:\n {','.join(acciones_vender)}")


def suggest_technical_beta2(
    tickers=[
         "XLK", "GDX", "AAPL", "CEMEXCPO.MX", "AXP", "AMZN", "GOOGL", "META",
    "QQQ", "MSFT", "JPM", "BOLSAA.MX", "XLF", "TSLA", "GMEXICOB.MX", "PE&OLES.MX",
    "WFC.MX", "BAC.MX", "SOFI.MX", "GAPB.MX", "ORBIA.MX", "AAPL.MX", "NVDA.MX", 
    "AMD.MX", "INTC.MX", "UBER.MX", "LABB.MX", "BIMBOA.MX", "PINS.MX", "TGT.MX", 
    "LCID.MX", "UPST.MX", "MARA.MX", "RIOT.MX", "BNGO.MX", "ALFAA.MX", "ALSEA.MX", 
    "AMXB.MX", "BBAJIOO.MX", "GM.MX", "NKLA.MX", "ACTINVRB.MX", "TALN.MX", "FSLR.MX", 
    "SPCE.MX", "FUBO.MX", "BYND.MX", "C.MX", "ATOS.MX"
    ]
):
    # Solicitar al usuario que ingrese tickers
    input_tickers = input(
        "Ingresa las acciones separadas por comas (i.e. OMAB,AAPL,META,MSFT): "
    )

    if input_tickers.strip():  # Si se ingresaron tickers, usarlos
        tickers = [ticker.strip() for ticker in input_tickers.split(",")]
    else:
        console.print(
            f"\n[bold yellow] No se detectó entrada, usando valores sugeridos por defecto...[/bold yellow]"
        )

    resultados = []  # Para almacenar los resultados
    acciones_comprar = []  # Para las acciones recomendadas para comprar
    acciones_mantener = []  # Para las acciones recomendadas para esperar
    acciones_vender = []  # Para las acciones recomendadas para vender

    # Iterar sobre cada ticker
    for ticker in tickers:
        try:
            # Descargar datos de los últimos 6 meses
            df_ticker = yf.download(ticker, period="6mo")
            df_ticker.dropna(inplace=True)

            if df_ticker.empty:
                print(f"Advertencia: No se encontraron datos para {ticker}.")
                continue

            # Asegurar Series 1D para indicadores técnicos
            close = df_ticker["Close"]
            high = df_ticker["High"]
            low = df_ticker["Low"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            if isinstance(high, pd.DataFrame):
                high = high.iloc[:, 0]
            if isinstance(low, pd.DataFrame):
                low = low.iloc[:, 0]

            # Calcular indicadores técnicos
            rsi = RSIIndicator(close, window=14).rsi()
            stochastic = StochasticOscillator(
                high,
                low,
                close,
                window=14,
                smooth_window=3,
            )
            macd = MACD(close).macd_diff()
            bollinger = BollingerBands(close)

            # Obtener datos recientes
            rsi_actual = rsi.iloc[-1]
            stochastic_actual = stochastic.stoch().iloc[-1]
            macd_actual = macd.iloc[-1]
            close_hoy = df_ticker["Close"].iloc[-1]
            close_ayer = df_ticker["Close"].iloc[-2]
            close_anteayer = df_ticker["Close"].iloc[-3]
            bollinger_high = bollinger.bollinger_hband().iloc[-1]
            bollinger_low = bollinger.bollinger_lband().iloc[-1]

            # Calcular las variaciones diarias
            variacion_ayer = (close_ayer - close_anteayer) / close_anteayer * 100
            variacion_hoy = (close_hoy - close_ayer) / close_ayer * 100

            # Condiciones de compra: variación ayer positiva, hoy negativa + indicadores técnicos
            condiciones_compra = (
                variacion_ayer < 2 and variacion_hoy > 4
                and rsi_actual < 58 and rsi_actual > 45 
                and macd_actual < 3 and macd_actual > -3 
            )

            # Determinar la acción recomendada
            if condiciones_compra:
                accion_recomendacion = "Comprar"
                acciones_comprar.append(ticker)
            else:
                accion_recomendacion = "Esperar"
                acciones_mantener.append(ticker)

            # Almacenar los resultados
            resultados.append(
                {
                    "Ticker": ticker,
                    "RSI": rsi_actual,
                    "Stochastic": stochastic_actual,
                    "MACD": macd_actual,
                    "Bollinger_High": bollinger_high,
                    "Bollinger_Low": bollinger_low,
                    "Variación Ayer (%)": variacion_ayer,
                    "Variación Hoy (%)": variacion_hoy,
                    "Acción Recomendada": accion_recomendacion,
                }
            )

        except Exception as e:
            print(f"Error con {ticker}: {e}")
            continue

    # Crear un DataFrame a partir de los resultados
    df_resultados = pd.DataFrame(resultados)
    
    os.makedirs('data', exist_ok=True)    
    csv_file_path = f'data/suggest_technical_{datetime.now():%Y%m%d_%H%M%S}.csv'
    df_resultados.to_csv(csv_file_path, index=False)

    # Imprimir la tabla de resultados
    print("\nResumen de Indicadores Técnicos:\n")
    print(df_resultados)

    # Mostrar recomendaciones
    print(f"\nAcciones recomendadas para comprar:\n {','.join(acciones_comprar)}")
    print(f"Acciones recomendadas para esperar:\n {','.join(acciones_mantener)}")
    print(f"Acciones recomendadas para vender:\n {','.join(acciones_vender)}")


def suggest_technical(
    tickers=[
        "XLK", "GDX", "AAPL", "CEMEXCPO.MX", "AXP", "AMZN", "GOOGL", "META",
        "QQQ", "MSFT", "JPM", "BOLSAA.MX", "XLF", "TSLA", "GMEXICOB.MX", "PE&OLES.MX",
    ]
):
    """
    Algoritmo de análisis técnico optimizado para un concurso de corto plazo (3 semanas).
    Estrategia: Reversión a la media con Bandas de Bollinger y RSI.
    Busca condiciones de sobreventa para comprar y sobrecompra para vender.
    """
    input_tickers = input(
        "Ingresa las acciones separadas por comas (o Enter para usar la lista por defecto): "
    )

    if input_tickers.strip():
        tickers = [ticker.strip().upper() for ticker in input_tickers.split(",")]
    else:
        console.print(
            f"\n[bold yellow]No se detectó entrada, usando valores sugeridos por defecto...[/bold yellow]"
        )

    resultados = []
    acciones_comprar = []
    acciones_vender = []
    acciones_mantener = []

    console.print("\n[bold cyan]🚀 Analizando acciones con estrategia optimizada para concurso...[/bold cyan]")

    for ticker in tickers:
        try:
            # --- MEJORA: Descarga de datos más robusta ---
            # Intenta primero con el ticker mexicano (.MX)
            mx_ticker = normalize_ticker_to_mx(ticker)
            df_ticker = yf.download(mx_ticker, period="3mo", progress=False, interval="1d")

            if df_ticker.empty:
                # Si falla, intenta con el ticker de USA
                usa_ticker = ticker.replace('.MX', '')
                df_ticker = yf.download(usa_ticker, period="3mo", progress=False, interval="1d")
                if not df_ticker.empty:
                    console.print(f"[yellow]Usando datos de USA para {ticker}[/yellow]")

            if df_ticker.empty or len(df_ticker) < 25:
                console.print(f"[red]Datos insuficientes para {ticker}. Omitiendo.[/red]")
                continue

            # --- Indicadores con parámetros más sensibles para corto plazo ---
            close = df_ticker["Close"]
            high = df_ticker["High"]
            low = df_ticker["Low"]

            # Bandas de Bollinger (20 periodos, 2 desviaciones estándar) - Estándar y efectivo
            bollinger = BollingerBands(close, window=20, window_dev=2)
            bb_high = bollinger.bollinger_hband()
            bb_low = bollinger.bollinger_lband()
            bb_mid = bollinger.bollinger_mavg() # Media móvil de 20 días

            # RSI (Índice de Fuerza Relativa) - Ventana de 10 para mayor sensibilidad
            rsi = RSIIndicator(close, window=10).rsi()

            # --- Lógica de Decisión Optimizada ---
            precio_actual = float(close.iloc[-1])
            rsi_actual = float(rsi.iloc[-1])
            bb_high_actual = float(bb_high.iloc[-1])
            bb_low_actual = float(bb_low.iloc[-1])
            bb_mid_actual = float(bb_mid.iloc[-1])

            accion_recomendacion = "Esperar"
            confianza = "Baja"

            # --- CONDICIONES DE COMPRA (SOBREVENTA) ---
            # Fuerte: Precio por debajo de la banda inferior Y RSI muy bajo.
            if precio_actual < bb_low_actual and rsi_actual < 30:
                accion_recomendacion = "Comprar"
                confianza = "Alta 🔥"
                acciones_comprar.append(ticker)
            # Moderada: Precio cerca de la banda inferior Y RSI bajo.
            elif precio_actual < (bb_low_actual * 1.015) and rsi_actual < 35:
                accion_recomendacion = "Comprar"
                confianza = "Media"
                acciones_comprar.append(ticker)

            # --- CONDICIONES DE VENTA (SOBRECOMPRA) ---
            # Fuerte: Precio por encima de la banda superior Y RSI muy alto.
            elif precio_actual > bb_high_actual and rsi_actual > 70:
                accion_recomendacion = "Vender"
                confianza = "Alta 🔥"
                acciones_vender.append(ticker)
            # Moderada: Precio cerca de la banda superior Y RSI alto.
            elif precio_actual > (bb_high_actual * 0.985) and rsi_actual > 68:
                accion_recomendacion = "Vender"
                confianza = "Media"
                acciones_vender.append(ticker)
            
            if accion_recomendacion == "Esperar":
                acciones_mantener.append(ticker)

            resultados.append({
                "Ticker": ticker,
                "Precio Actual": f"${precio_actual:,.2f}",
                "RSI(10)": f"{rsi_actual:.2f}",
                "Posición BB": f"{((precio_actual - bb_low_actual) / (bb_high_actual - bb_low_actual) * 100):.1f}%",
                "Acción": accion_recomendacion,
                "Confianza": confianza,
            })

        except Exception as e:
            console.print(f"[red]Error procesando {ticker}: {e}[/red]")
            continue

    if not resultados:
        console.print("[bold red]No se pudieron analizar acciones.[/bold red]")
        return

    # --- Mostrar Resultados ---
    df_resultados = pd.DataFrame(resultados)
    
    # Colorear la tabla para mejor visualización
    def colorear_accion(val):
        if val == "Comprar":
            return "color: #2ECC71" # Verde
        elif val == "Vender":
            return "color: #E74C3C" # Rojo
        return ""

    styled_df = df_resultados.style.applymap(colorear_accion, subset=['Acción'])

    console.print("\n[bold]Resumen de Indicadores Optimizados:[/bold]")
    print(styled_df.to_string()) # Usamos to_string para ver el estilo en la consola

    console.print(f"\n[bold green]✅ Acciones recomendadas para COMPRAR ({len(acciones_comprar)}):[/bold green] {', '.join(acciones_comprar)}")
    console.print(f"\n[bold red]❌ Acciones recomendadas para VENDER ({len(acciones_vender)}):[/bold red] {', '.join(acciones_vender)}")
    console.print(f"\n[bold yellow]⏳ Acciones para ESPERAR Y MONITOREAR ({len(acciones_mantener)}):[/bold yellow] {', '.join(acciones_mantener)}")

def set_optimizar_portafolio():
    input_tickers = str(
        input("Enter tickers separated by commas (i.e. OMAB,AAPL,META,MSFT): ")
    )
    if not input_tickers:  # Si el usuario no ingresa nada
        console.print(
            f"\n[bold yellow] No se detectó entrada, usando valores sugeridos por defecto...[/bold yellow]"
        )
        input_tickers = "OMAB,AAPL,META,MSFT"  # Valor por defecto
    input_initial_date = str(
        input("Enter initial date of historial prices data (i.e. 2018-01-01): ")
    )
    if not input_initial_date:  # Si el usuario no ingresa nada
        input_initial_date = "2018-01-01"  # Valor por defecto
    input_tickers = input_tickers.replace(" ", "")
    # Keep user-provided listing; don't force-remove suffixes
    input_tickers = input_tickers.upper()
    input_tickers_list = input_tickers.split(",")
    tickers = []
    for i in input_tickers_list:
        tickers.append(i)
    
    # Calcular capital óptimo basado en número de tickers
    optimal_capital = calculate_optimal_capital(tickers)
    input_amount = (
        input(f"Enter the amount to invest (default: {optimal_capital}): ")
        .replace(",", "")
        .replace("$", "")
        .strip()
    )
    if not input_amount:  # Si el usuario no ingresa nada
        input_amount = optimal_capital  # Usar capital óptimo
    else:
        input_amount = int(float(input_amount))  # Convertir a número
    
    print("\nTickers list : ", tickers)
    print(f"Capital total: ${input_amount:,}")
    print("\nOptimizing portfolio...")
    try:
        allocation_dataframe = markovitz_portfolio_optimization(
            tickers, input_amount, input_initial_date
        )
        # Aplicar límite de asignación por ticker
        allocation_dataframe = apply_max_allocation_limit(allocation_dataframe)
        print(allocation_dataframe)
    except Exception as e:
        print(e)

def set_optimizar_portafolio2():
    input_tickers = str(
        input("Enter tickers separated by commas (i.e. OMAB,AAPL,META,MSFT): ")
    )
    if not input_tickers:  # Si el usuario no ingresa nada
        console.print(
            f"\n[bold yellow] No se detectó entrada, usando valores sugeridos por defecto...[/bold yellow]"
        )
        input_tickers = "OMAB,AAPL,META,MSFT"  # Valor por defecto
    input_initial_date = str(
        input("Enter initial date of historial prices data (i.e. 2018-01-01): ")
    )
    if not input_initial_date:  # Si el usuario no ingresa nada
        input_initial_date = "2018-01-01"  # Valor por defecto
    input_tickers = input_tickers.replace(" ", "")
    # Keep user-provided listing; don't force-remove suffixes
    input_tickers = input_tickers.upper()
    input_tickers_list = input_tickers.split(",")
    tickers = []
    for i in input_tickers_list:
        tickers.append(i)
    
    # Calcular capital óptimo basado en número de tickers
    optimal_capital = calculate_optimal_capital(tickers)
    input_amount = (
        input(f"Enter the amount to invest (default: {optimal_capital}): ")
        .replace(",", "")
        .replace("$", "")
        .strip()
    )
    if not input_amount:  # Si el usuario no ingresa nada
        input_amount = optimal_capital  # Usar capital óptimo
    else:
        input_amount = int(float(input_amount))  # Convertir a número
    
    print("\nTickers list : ", tickers)
    print(f"Capital total: ${input_amount:,}")
    print("\nOptimizing portfolio by momentum...")
    try:
        allocation_dataframe = portfolio_optimization_sharpe_short_term(
            tickers, input_amount, input_initial_date
        )
        # Aplicar límite de asignación por ticker
        allocation_dataframe = apply_max_allocation_limit(allocation_dataframe)
        print(allocation_dataframe)
    except Exception as e:
        print(e)


def show_portfolio():
    print("Función provisional para Mostrar portafolio")


def buy_stocks():
    print("Función provisional para Comprar acciones")


def show_orders():
    print("Función provisional para Mostrar órdenes")


def establish_session(
    login_data={}
):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.6533.100 Safari/537.36",
    }

    # Start a session
    session = requests.Session()
    print("Starting a new session...")

    try:
        # Get the login page to retrieve necessary tokens (cookies in this case)
        response = session.get(
            "https://www.retoactinver.com/minisitio/reto/login/index.html",
            headers=headers,
        )
        print(f"GET request to login page returned status code: {response.status_code}")

        # Check if 'TS016e21d6' is present in the cookies
        if "TS016e21d6" in response.cookies:
            print("Token 'TS016e21d6' found in cookies.")

            # Extract the token from the cookies
            token = response.cookies.get("TS016e21d6")
            print(f"Extracted token: {token}")

            # Save the session data temporarily
            session_data = {"TS016e21d6": token}
            os.makedirs('data', exist_ok=True)    
            with open("data/SessionInfoTmp01.json", "w") as file:
                json.dump(session_data, file)
                print("Session token saved to 'SessionInfoTmp01.json'.")
                
            print("Trying login user")
            print(json.dumps(login_data, indent=4))

            # Post the login credentials
            print("Sending login POST request with user credentials...")
            login_response = session.post(
                "https://www.retoactinver.com/reto/app/usuarios/login",
                json=login_data,
                headers=headers,
            )
            print(
                f"POST login request returned status code: {login_response.status_code}"
            )

            if login_response.status_code == 200:
                login_response_json = login_response.json()
                print("Login successful. Response received.")

                # Save the login response temporarily
                os.makedirs('data', exist_ok=True)    
                with open("data/SessionInfoTmp02.json", "w") as file:
                    json.dump(login_response_json, file)
                    print("Login response saved to 'SessionInfoTmp02.json'.")
            else:
                print(f"Login failed. Status code: {login_response.status_code}")
                print(f"Response: {login_response.text}")
                return

            # Merge the session data and save to the final file
            with open("data/SessionInfoTmp01.json") as file1, open(
                "data/SessionInfoTmp02.json"
            ) as file2:
                session_info = json.load(file1)
                session_info.update(json.load(file2))
                print("Session info from both files merged.")

            os.makedirs('data', exist_ok=True)
            with open("data/SessionInfo.json", "w") as file:
                json.dump(session_info, file)
                print("Merged session info saved to 'SessionInfo.json'.")
            print("Datos de inicio de sesión (redacted):")
            print(redact_sensitive_dict(session_info))
        else:
            print("Token 'TS016e21d6' not found in cookies.")

    except Exception as e:
        print(f"An error occurred: {e}")


# Recupera la sesión guardada
def recover_session():
    print("Actualización datos de sesión de usuario en SessionInfo.json")
    
    # Cargar información de la sesión desde el archivo JSON
    with open("data/SessionInfo.json") as file:
        session_info = json.load(file)
        
    print("Sessión info actual session_info (redacted):")
    print(redact_sensitive_dict(session_info))

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.5249.62 Safari/537.36",
    }

    session = requests.Session()
    cookies = {
        "TS016e21d6": session_info["TS016e21d6"],
        "tokenapp": session_info["tokenApp"],        
        "tokensesion": session_info["tokenSession"],
    }

    # Reintentar hasta un máximo de 3 veces con un tiempo de espera entre reintentos
    max_retries = 3
    for attempt in range(max_retries):
        try:
            recovery_response = session.post(
                f'https://www.retoactinver.com/reto/app/usuarios/session/recoveryTokenSession?user={session_info["cxCveUsuario"]}&tokenApp={session_info["tokenApp"]}',
                cookies=cookies,
                headers=headers,
                timeout=10  # Limitar el tiempo de espera de la conexión
            )

            # Verificar si la respuesta fue exitosa (código de estado 200)
            recovery_response.raise_for_status()
            
            # Procesar la respuesta
            print(recovery_response.text)
            session_info["tokenSession"] = recovery_response.json()["cxValue"]
            
            print("Sessión info nueva session_info")
            
            # Guardar la nueva información de la sesión en el archivo
            with open("data/SessionInfo.json", "w") as file:
                json.dump(session_info, file)
            
            break  # Salir del ciclo si se logró completar la solicitud con éxito

        except requests.exceptions.RequestException as e:
            print(f"Error al recuperar la sesión (intento {attempt + 1} de {max_retries}): {e}")
            
            if attempt < max_retries - 1:
                print("Reintentando...")
                time.sleep(5)  # Esperar 5 segundos antes de reintentar
            else:
                print("Se alcanzó el número máximo de intentos. No se pudo recuperar la sesión.")


# Cierra la sesión
def close_session():
    print("Cerrando sesión...")
    with open("data/SessionInfo.json") as file:
        session_info = json.load(file)

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.5249.62 Safari/537.36",
    }

    cookies = {
        "TS016e21d6": session_info["TS016e21d6"],
        "tokensesion": session_info["tokenSession"],
    }

    session = requests.Session()
    session.post(
        f'https://www.retoactinver.com/reto/app/usuarios/session/closeSesion?user=osvaldo.hdz.m@outlook.com&tokenSession={session_info["tokenSession"]}&tokenApp={session_info["tokenApp"]}',
        cookies=cookies,
        headers=headers,
    )

def delete_file_if_exists(file_path):
    # Check if the file exists
    if os.path.exists(file_path):
        os.remove(file_path)  # Delete the file
        print(f"The file '{file_path}' has been deleted.")
    else:
        print(f"The file '{file_path}' does not exist do not need delete it.")

def get_weekly_quizz():
    print("Trying get weekly quizz")
    with open("data/SessionInfo.json") as file:
        session_info = json.load(file)

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.5249.62 Safari/537.36",
    }

    cookies = {
        "tokenapp": session_info["tokenApp"],
        "TS016e21d6": session_info["TS016e21d6"],
        "tokensesion": session_info["tokenSession"],
    }

    response = requests.get(
        f'https://www.retoactinver.com/reto/app/quiz/consultaQuizSemanal?cveUsuario={session_info["cxCveUsuario"]}&cx_token_app={session_info["tokenApp"]}&cx_tokenSesionApl={session_info["tokenSession"]}',
        cookies=cookies,
        headers=headers,
    )
    
    # Procesar la respuesta JSON
    data = response.json()
    
    print(data)
    
    # Crear el cuestionario con las respuestas seleccionadas (por ahora, la primera respuesta)
    cuestionario_respuestas = []
    for pregunta in data['collection']['cuestionario']:
        # Elegimos la primera respuesta para cada pregunta como placeholder
        primera_respuesta = pregunta['respustas'][0]['id']
        cuestionario_respuestas.append({"id": pregunta['id'], "respuestaCorrecta": primera_respuesta})
    
    # Construir el prompt para enviar a la IA (Gemini)
    prompt = "Considering a context of stock market analysis of the stock market mainly but also economics. Please provide the correct answer IDs for the following questions, separated by commas. Only return the answer IDs.\n\n"
    for pregunta in data['collection']['cuestionario']:
        prompt += f"Question: {pregunta['pregunta']}\n"
        for respuesta in pregunta['respustas']:
            prompt += f"- Option {respuesta['id']}: {respuesta['respuesta']}\n"
        prompt += "\n"
    
    numero_preguntas = len(data['collection']['cuestionario'])
    
    # Mostrar el prompt generado (para depuración)
    print(prompt)
    
    # Llamar al helper con SDK oficial (clave en env)
    respuestas_ai = generate_gemini_text(prompt).strip()

    # Convertir las respuestas en una lista
    lista_respuestas_ai = respuestas_ai.split(',')
    
    # Verificar si el número de respuestas coincide con el número de preguntas
    if len(lista_respuestas_ai) == numero_preguntas:
        print(f"Las respuestas coinciden con el número de preguntas: {lista_respuestas_ai}")
    else:
        print(f"No coinciden la cantidad de respuestas ({len(lista_respuestas_ai)}) con las preguntas ({numero_preguntas})")
    
    for i, respuesta in enumerate(lista_respuestas_ai):
        cuestionario_respuestas[i]['respuestaCorrecta'] = respuesta
    
    return cuestionario_respuestas


def get_daily_quizz():
    print("Intento obtener el quizz diario...")
    with open("data/SessionInfo.json") as file:
        session_info = json.load(file)

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.5249.62 Safari/537.36",
    }

    cookies = {
        "tokenapp": session_info["tokenApp"],
        "TS016e21d6": session_info["TS016e21d6"],
        "tokensesion": session_info["tokenSession"],
    }

    response = requests.get(
        f'https://www.retoactinver.com/reto/app/quiz/consultaContestoQuizz?cveUsuario={session_info["cxCveUsuario"]}&cx_token_app={session_info["tokenApp"]}&cx_tokenSesionApl={session_info["tokenSession"]}',
        cookies=cookies,
        headers=headers,
    )
    
    quiz_data = response.json()
        
    mensaje = quiz_data['collection'][0]['Pregunta'].get('Mensaje')  
    pregunta = ''  
    if mensaje:
        print("Pregunta:")
        pregunta = quiz_data['collection'][0]['Pregunta']["Pregunta"]
        print(pregunta)
        if "Pregunta contestada" in mensaje:
            print("Pregunta ya contestada previamente, borrando quizz aprevio si existe")
            delete_file_if_exists("SessionQuizData.json")
            pregunta = mensaje
            return pregunta
    else:
        try:
            # Intentamos obtener la pregunta
            pregunta = quiz_data["collection"][0]["Pregunta"]["Pregunta"]["pregunta"]
            respuestas = quiz_data["collection"][0]["Pregunta"]["respuestas"]
            full_question = pregunta + "\n\n" + "\n".join([f"{r['opcion']}. {r['respuesta']}" for r in respuestas])
            print(full_question)

            # Limitar las respuestas a las primeras 3
            quiz_data["collection"][0]["Pregunta"]["respuestas"] = quiz_data["collection"][0]["Pregunta"]["respuestas"][:3]

            # Guardar los datos en un archivo
            with open("data/SessionQuizData.json", "w") as file:
                json.dump(quiz_data, file)
            
            return pregunta        
        except KeyError as e:
            # Manejo de excepción si la clave "Pregunta" no existe
            print(f"Error: {str(e)} - Es posible que la pregunta aún no se haya publicado o haya sido respondida anteriormente.")

# Envía la respuesta del cuestionario
def send_quizz_week_answer(cuestionario_respuestas):
    # Check if the file exists
    if not os.path.exists("data/SessionQuizData.json"):
        return
        
    # Load session information from temporary JSON file
    with open("data/SessionQuizData.json") as tmp_file:
        quiz_data = json.load(tmp_file)
        
    # Load session information from JSON file
    with open("data/SessionInfo.json") as session_file:
        session_info = json.load(session_file)

    # Define cookies for the request
    cookies = {
        "tokenapp": session_info["tokenApp"],
        "TS016e21d6": session_info["TS016e21d6"],
        "tokensesion": session_info["tokenSession"],
        "cxCveUsuario": session_info["cxCveUsuario"],
    }
    
    print(cuestionario_respuestas)
    
    url = 'https://www.retoactinver.com/reto/app/quiz/contestaQuizSemanal'
    
    params = {
    'cveUsuario': session_info["cxCveUsuario"],
    'cx_tokenSesionApl': session_info["tokenSession"],
    'cx_token_app': session_info["tokenApp"]
    }
    
    data = {
    'cuestionario': cuestionario_respuestas
    }
    
    headers = {
    'Content-Type': 'application/json',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.6668.71 Safari/537.36',
    'Accept': 'application/json',
    'Referer': 'https://www.retoactinver.com/minisitio/reto/evaluacion/semanal/',
    'Origin': 'https://www.retoactinver.com',
    'Sec-Fetch-Site': 'same-origin',
    'Sec-Fetch-Mode': 'cors',
    'Sec-Fetch-Dest': 'empty',
    'Connection': 'keep-alive',}

    
    response = requests.put(url, params=params, headers=headers, data=json.dumps(data))
    
    # Print the response status
    print(f"Response Status Code: {response.status_code}")
    print(response.text)
    if response.ok:
        print("Answer submitted successfully!")
    else:
        print("Failed to submit the answer.")


def anwser_daily_quizz_random_method(first_response_id):
    print("Answering by random method")
     # Extract the first response ID from the JSON and select a random ID in the range    
    
    random.seed(time.time_ns())
    random_id = random.randint(first_response_id, first_response_id + 2)
    return random_id

def anwser_daily_quizz_ia_gemini_method(pregunta, answer_options):
    print("Answering by random gemini method ia")
    # Construir el prompt para enviar a la IA (Gemini)
    prompt = (
        "Considering a context of stock and economics. "
        "Given the multiple-choice question below, respond ONLY with the numeric idRespuesta of the correct option.\n\n"
        f"Pregunta: {pregunta}\n"
        f"Opciones: {answer_options}\n"
        "Respuesta:"
    )
    # Mostrar el prompt generado (para depuración)
    print(prompt)
    # Usar helper con SDK oficial (lee GEMINI_API_KEY del entorno)
    text = generate_gemini_text(prompt).strip()
    if not text:
        print("Gemini no disponible, usando heurística de respaldo...")
        # Heurística de respaldo mejorada
        try:
            lowered = str(pregunta).lower()
            keywords = []
            
            # Palabras clave específicas para diferentes tipos de preguntas
            if "cnbv" in lowered or "comisión nacional" in lowered:
                keywords = ["supervisar", "regular", "sistema financiero"]
            elif "volatil" in lowered:
                keywords = ["fluct", "vari", "precios"]
            elif "riesgo" in lowered:
                keywords = ["pérdida", "incertidumbre", "variabilidad"]
            elif "liquidez" in lowered:
                keywords = ["convertir", "dinero", "efectivo"]
            elif "dividendo" in lowered:
                keywords = ["utilidades", "ganancias", "distribución"]
            elif "inflación" in lowered:
                keywords = ["precios", "aumento", "general"]
            elif "tasa" in lowered and "interés" in lowered:
                keywords = ["costo", "dinero", "crédito"]
            
            # Buscar la primera opción que empareje palabras clave
            for opt in (answer_options or []):
                resp_txt = str(opt.get("respuesta", "")).lower()
                if any(k in resp_txt for k in keywords):
                    print(f"Seleccionando opción por palabra clave: {opt.get('opcion')} - {opt.get('respuesta')}")
                    return int(opt.get("idRespuesta"))
            
            # Si no hay coincidencias, seleccionar la opción B (generalmente correcta en exámenes)
            print("No se encontraron palabras clave, seleccionando opción B por defecto")
            for opt in (answer_options or []):
                if opt.get("opcion") == "B":
                    return int(opt.get("idRespuesta"))
                    
            # Último recurso: primera opción
            if answer_options:
                print("Seleccionando primera opción como último recurso")
                return int(answer_options[0].get("idRespuesta"))
                
        except Exception as e:
            print(f"Error en heurística: {e}")
            pass
        
        # Si todo falla, devolver la primera opción disponible
        if answer_options:
            return int(answer_options[0].get("idRespuesta"))
        return None
    
    # Extraer el primer número (idRespuesta)
    try:
        import re
        match = re.search(r"\d+", text)
        result = int(match.group(0)) if match else None
        if result:
            print(f"Gemini seleccionó idRespuesta: {result}")
        return result
    except Exception as e:
        print(f"Error extrayendo respuesta de Gemini: {e}")
        # Fallback a heurística si Gemini falla
        if answer_options:
            for opt in answer_options:
                if opt.get("opcion") == "B":
                    return int(opt.get("idRespuesta"))
            return int(answer_options[0].get("idRespuesta"))
        return None

# Envía la respuesta del cuestionario
def send_quizz_answer(pregunta):
    # Check if the file exists
    if not os.path.exists("data/SessionQuizData.json"):
        print("Pregunta respondida anteriomente por lo que no existe data/SessionQuizData.json")
        return
        
    # Load session information from temporary JSON file
    with open("data/SessionQuizData.json") as tmp_file:
        quiz_data = json.load(tmp_file)
        
    print(f"La pregunta es: {pregunta}")
    answer_options = quiz_data['collection'][0]['Pregunta']['respuestas']
    print(f"Las opciones son: {answer_options}")
    
    first_response_id = quiz_data["collection"][0]["Pregunta"]["respuestas"][0][
        "idRespuesta"
    ]
    
    #answer_id = anwser_daily_quizz_random_method(first_response_id)
    answer_id = anwser_daily_quizz_ia_gemini_method(pregunta, answer_options)

    # Validar que tenemos un answer_id válido
    if answer_id is None:
        print("Error: No se pudo determinar una respuesta válida, usando método aleatorio como fallback")
        answer_id = anwser_daily_quizz_random_method(first_response_id)
    
    print(f"\n\nAnswering with idRespuesta: {answer_id}")

    # Define the request headers
    headers = {
        "Host": "www.retoactinver.com",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.5249.62 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Origin": "https://www.retoactinver.com",
        "Referer": "https://www.retoactinver.com/RetoActinver/",
        "Connection": "close",
        "Sec-Fetch-Site": "same-origin",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Dest": "empty",
        "Accept-Encoding": "gzip, deflate",
        "Accept-Language": "es-419,es;q=0.9",
    }

    # Load session information from JSON file
    with open("data/SessionInfo.json") as session_file:
        session_info = json.load(session_file)

    # Define cookies for the request
    cookies = {
        "tokenapp": session_info["tokenApp"],
        "TS016e21d6": session_info["TS016e21d6"],
        "tokensesion": session_info["tokenSession"],
        "cxCveUsuario": session_info["cxCveUsuario"],
    }

    # Construct the URL with the random idRespuesta
    url = f'https://www.retoactinver.com/reto/app/quiz/contestarQuiz?cveUsuario={session_info["cxCveUsuario"]}&idRespuesta={answer_id}&cx_tokenSesionApl={session_info["tokenSession"]}&cx_token_app={session_info["tokenApp"]}&tokenApp={session_info["tokenApp"]}&tokenSession={session_info["tokenSession"]}'

    # Send the POST request
    response = requests.post(url, headers=headers, cookies=cookies)

    # Print the response status
    print(f"Response Status Code: {response.status_code}")
    print(response.text)
    if response.ok:
        print("Answer submitted successfully!")
    else:
        print("Failed to submit the answer.")

def answer_quiz_weekly_contest_actinver():
    clear_screen()
    
    usuarios = get_actinver_users()
    usuarios = get_actinver_users()
    
    delete_file_if_exists('data/SessionInfo.json')
    delete_file_if_exists('data/SessionInfoTmp01.json')
    delete_file_if_exists('data/SessionInfoTmp02.json')
    delete_file_if_exists('data/SessionInfo.json')
       
    for _ in range(3):
        for login_data in usuarios:
            establish_session(login_data=login_data)
            recover_session()
            cuestionario_respuestas = get_weekly_quizz()
            recover_session()
            send_quizz_week_answer(cuestionario_respuestas)
            close_session()
            time.sleep(3)

    Prompt.ask("[bold blue]Pulsa Enter para continuar...[/bold blue]")
    main()


def answer_quiz_daily_contest_actinver():
    clear_screen()
    
    usuarios = [
        {"usuario": "natalia.sofia.glz@gmail.com", "password": "Ntlasfa9#19"},
        {"usuario": "osvaldo.hdz.m@outlook.com", "password": "299792458.Light"}        
    ]
        
    for login_data in usuarios:
        for _ in range(3):
            establish_session(login_data=login_data)
            recover_session()
            pregunta = get_daily_quizz()
            print("Ya se resondío no neceist aiterar")
            if "Pregunta contestada" in pregunta:
                break            
            send_quizz_answer(pregunta)
            close_session()
            time.sleep(3)

    Prompt.ask("[bold blue]Pulsa Enter para continuar...[/bold blue]")
    main()

def mock_quiz_function():
    print("Función de prueba ejecutada a la hora programada abriendo el Notepad.exe.")
    subprocess.run(["notepad.exe"])

def print_time_remaining(target_time):
    """Calcula y devuelve el tiempo restante hasta la hora objetivo."""
    time_remaining = target_time - datetime.now()
    return str(time_remaining).split('.')[0]  # Formato HH:MM:SS

def schedule_daily_quiz_once(time_str):
    """Programa la función para que se ejecute solo una vez a la hora específica."""
    hour, minute, second = map(int, time_str.split(":"))
    schedule_time = datetime.now().replace(hour=hour, minute=minute, second=second, microsecond=0)

    # Si la hora programada ya pasó para hoy, se programa para mañana
    if schedule_time < datetime.now():
        schedule_time += timedelta(days=1)

    # Calcular el tiempo de espera en segundos hasta la próxima ejecución
    wait_time = (schedule_time - datetime.now()).total_seconds()

    # Programar la tarea
    threading.Thread(target=delayed_daily_quizz_task, args=(wait_time, schedule_time)).start()

    # Añadir la tarea a la lista con indicador de tarea única
    scheduled_tasks.append((None, time_str, schedule_time, "Quiz diario una vez", False))
    print(f"Tarea única programada para ejecutarse a las {time_str}.")

def delayed_daily_quizz_task(wait_time, target_time):
    """Ejecuta la tarea después de un tiempo específico."""
    time.sleep(wait_time)  # Espera el tiempo calculado    
    answer_quiz_daily_contest_actinver() # Existe un funcion mock pos si quiere spurebas 

def schedule_quiz_daily(time_str):
    """Programa la función para que se ejecute a una hora específica cada día."""
    try:
        # Esto verifica que el formato sea válido
        hour, minute, second = map(int, time_str.split(":"))

        # Programar la tarea diariamente
        job = schedule.every().day.at(time_str).do(answer_quiz_daily_contest_actinver)

        # Añadir la tarea a la lista con indicador de tarea diaria
        scheduled_tasks.append((job, time_str, None, "Quiz diario", True))
        print(f"Quiz diario programado para las {time_str}.")
    
    except Exception as e:
        print(f"Error al programar el quiz diario: {str(e)}")
        

def schedule_quiz_weekly_once(time_str):
    """Programa la función para que se ejecute solo una vez a la hora específica."""
    hour, minute, second = map(int, time_str.split(":"))
    schedule_time = datetime.now().replace(hour=hour, minute=minute, second=second, microsecond=0)

    # Si la hora programada ya pasó para hoy, se programa para mañana
    if schedule_time < datetime.now():
        schedule_time += timedelta(days=1)

    # Calcular el tiempo de espera en segundos hasta la próxima ejecución
    wait_time = (schedule_time - datetime.now()).total_seconds()

    # Programar la tarea
    threading.Thread(target=delayed_weekly_quizz_task, args=(wait_time, schedule_time)).start()

    # Añadir la tarea a la lista con indicador de tarea única
    scheduled_tasks.append((None, time_str, schedule_time, "Quiz semanal una vez", False))
    print(f"Tarea única programada para ejecutarse a las {time_str}.")

def delayed_weekly_quizz_task(wait_time, target_time):
    """Ejecuta la tarea después de un tiempo específico."""
    time.sleep(wait_time)  # Espera el tiempo calculado    
    answer_quiz_weekly_contest_actinver() # Existe un funcion mock pos si quiere spurebas 

def schedule_quiz_weekly_daily(time_str):
    """Programa la función para que se ejecute a una hora específica cada día."""
    try:
        # Esto verifica que el formato sea válido
        hour, minute, second = map(int, time_str.split(":"))

        # Programar la tarea diariamente
        job = schedule.every().day.at(time_str).do(answer_quiz_weekly_contest_actinver)

        # Añadir la tarea a la lista con indicador de tarea diaria
        scheduled_tasks.append((job, time_str, None, "Quiz semanal", True))
        print(f"Quiz semanal programado para las {time_str}.")
    
    except Exception as e:
        print(f"Error al programar el quiz semanal: {str(e)}")

def list_tasks_scheduled():
    """Lista todas las tareas programadas, diferenciando entre diarias y únicas."""
    if not scheduled_tasks:
        print("No hay tareas programadas.")
    else:
        print("Tareas programadas:")
        for job, time_str, target_time, name, is_daily in scheduled_tasks:
            schedule_type = "Diaria" if is_daily else "Única"

            if is_daily:
                # Si es una tarea diaria, no tiene un único tiempo objetivo
                print(f" - {name} a las {time_str} ({schedule_type}) (se ejecutará diariamente)")
            else:
                # Calcular el tiempo restante para las tareas únicas
                time_remaining = print_time_remaining(target_time)

                # Verificar si la tarea única ya ha sido ejecutada
                if datetime.now() > target_time:
                    # Si ya fue ejecutada y es única, indicar que ha finalizado
                    print(f" - {name} a las {time_str} ({schedule_type}) (finalizada)")
                else:
                    print(f" - {name} a las {time_str} ({schedule_type}) (Tiempo restante: {time_remaining})")

def run_schedule():
    """Ejecuta el programador en un hilo separado."""
    while not stop_event.is_set():
        schedule.run_pending()
        time.sleep(1)



def start_scheduled_weekly_quiz():
    """Solicita al usuario una hora y programa el concurso según su preferencia."""
    # Obtener la hora actual y sugerir la hora una hora adelante
    current_time = datetime.now()
    default_time = current_time + timedelta(minutes=1)
    default_time_str = default_time.strftime("%H:%M:%S")

    user_input = input(f"Introduce la hora en formato 24 horas (i.e. {default_time_str}): ")

    # Establecer un valor por defecto si el usuario no introduce nada
    if not user_input.strip():
        user_input = default_time_str
        print(f"No se introdujo ninguna hora. Se establecerá el valor por defecto: {default_time_str}")

    try:
        # Preguntar si quiere ejecutar diariamente o solo una vez
        daily_choice = input("¿Deseas ejecutar esto diariamente? (y/n): ").strip().lower()

        if daily_choice == "y":
            # Programa el quiz diario
            # Iniciar el scheduler en un hilo separado
            scheduler_thread = threading.Thread(target=run_schedule)
            scheduler_thread.start()
            schedule_quiz_weekly_daily(user_input)
            print("La tarea del quiz semanal ha sido programada.")
        else:
            # Programa la ejecución solo una vez
            schedule_quiz_weekly_once(user_input)

    except ValueError:
        print("Formato de hora no válido. Por favor, introduce la hora en el formato HH:MM:SS.")

def start_scheduled_quiz():
    """Solicita al usuario una hora y programa el concurso según su preferencia."""
    # Obtener la hora actual y sugerir la hora una hora adelante
    current_time = datetime.now()
    default_time = current_time + timedelta(minutes=1)
    default_time_str = default_time.strftime("%H:%M:%S")

    user_input = input(f"Introduce la hora en formato 24 horas (i.e. {default_time_str}): ")

    # Establecer un valor por defecto si el usuario no introduce nada
    if not user_input.strip():
        user_input = default_time_str
        print(f"No se introdujo ninguna hora. Se establecerá el valor por defecto: {default_time_str}")

    try:
        # Preguntar si quiere ejecutar diariamente o solo una vez
        daily_choice = input("¿Deseas ejecutar esto diariamente? (y/n): ").strip().lower()

        if daily_choice == "y":
            # Programa el quiz diario
            # Iniciar el scheduler en un hilo separado
            scheduler_thread = threading.Thread(target=run_schedule)
            scheduler_thread.start()
            schedule_quiz_daily(user_input)
            print("La tarea del quiz diario ha sido programada.")
        else:
            # Programa la ejecución solo una vez
            schedule_daily_quiz_once(user_input)

    except ValueError:
        print("Formato de hora no válido. Por favor, introduce la hora en el formato HH:MM:SS.")

def stop_scheduled_tasks():
    """Detiene todas las tareas programadas y los hilos en ejecución."""
    global stop_event
    stop_event.set()
    schedule.clear()
    scheduled_tasks.clear()  # Limpiar la lista de tareas programadas
    print("Todas las tareas programadas han sido detenidas.")

def exit_program():
    """Detiene todas las tareas programadas y sale del programa."""
    stop_scheduled_tasks()  # Detiene los hilos y tareas programadas
    console.print("[bold red]Saliendo...[/bold red]")
    exit(0)  # Sale del programa con un código de éxito
    

def test_actinver_credentials(users: list[dict]) -> list[tuple[str, str]]:
    resultados: list[tuple[str, str]] = []
    for creds in users:
        user = creds.get("usuario", "<sin_usuario>")
        try:
            establish_session(login_data=creds)
            recover_session()
            resultados.append((user, "OK"))
        except Exception as e:
            resultados.append((user, f"ERROR: {e}"))
        finally:
            try:
                close_session()
            except Exception:
                pass
    return resultados

def test_session():
    clear_screen()
    console.print("[bold blue]Probando credenciales de sesión...[/bold blue]")
    # Cargar credenciales desde variables de entorno si existen, si no usar lista por defecto
    usuarios = get_actinver_users()

    for u in usuarios:
        console.print(f"- Probando: [cyan]{u['usuario']}[/cyan]")
    resultados = test_actinver_credentials(usuarios)
    console.print("\n[bold]Resultados:[/bold]")
    for user, status in resultados:
        console.print(f"  {user}: {status}")


def option_2():
    print("Función provisional para la opción 2")


def option_3():
    print("Función provisional para la opción 3")


def option_4():
    print("Función provisional para la opción 4")


def option_5():
    print("Función provisional para la opción 5")


def option_6():
    print("Función provisional para la opción 6")


def option_7():
    print("Función provisional para la opción 7")


def option_8():
    print("Función provisional para la opción 8")


def option_9():
    print("Función provisional para la opción 9")


def test_session0():
    print("Función provisional para la opción 10")


def test_session1():
    print("Función provisional para la opción 11")


def utilidades_actinver_2024():
    # Construcción del menú con secciones
    menu_items = [
        { 'type': 'item', 'text': 'Probar Credenciales de sesión en la plataforma del reto', 'action': test_session },
        { 'type': 'item', 'text': 'Obtener pregunta de Quizz diario', 'action': option_2 },
        { 'type': 'item', 'text': 'Resolver Quizz diario', 'action': answer_quiz_daily_contest_actinver },
        { 'type': 'item', 'text': 'Programar respuesta automática de Quizz diario', 'action': start_scheduled_quiz },
        { 'type': 'item', 'text': 'Resolver Quizz semanal', 'action': answer_quiz_weekly_contest_actinver },
        { 'type': 'item', 'text': 'Programar respuesta automática de Quizz semanal', 'action': start_scheduled_weekly_quiz },
        { 'type': 'item', 'text': 'Mostrar sugerencias de compra', 'action': option_5 },
        { 'type': 'item', 'text': 'Mostrar portafolio actual', 'action': option_6 },
        { 'type': 'item', 'text': 'Comprar acciones', 'action': option_7 },
        { 'type': 'item', 'text': 'Mostrar órdenes', 'action': option_8 },
        { 'type': 'item', 'text': 'Monitorear venta', 'action': option_9 },
        { 'type': 'item', 'text': 'Vender todas las posiciones en portafolio (a precio del mercado)', 'action': test_session0 },
        { 'type': 'item', 'text': 'Restaurar sesión en plataforma del reto', 'action': test_session1 },
        { 'type': 'item', 'text': 'Listar tareas programadas', 'action': list_tasks_scheduled },
        { 'type': 'item', 'text': 'Monitor de Stocks (GUI)', 'action': monitor_stocks_gui },
        { 'type': 'item', 'text': 'Reporte Visual', 'action': daily_visual_report },
        { 'type': 'item', 'text': 'Imprimir Consejos', 'action': imprimir_consejos_inversion },
        { 'type': 'item', 'text': 'Regresar', 'action': lambda: None },
    ]

    # Lógica para mapear números a índices de elementos ejecutables
    executable_options = [item for item in menu_items if item['type'] == 'item']
    num_to_index = {}
    current_num = 1
    for i, item in enumerate(menu_items):
        if item['type'] == 'item':
            num_to_index[str(current_num)] = i
            current_num += 1

    selected_index = 0  # Índice inicial para la opción seleccionable

    def display_menu(selected_index):
        console.clear()
        console.print("[bold blue]Utilidades - Reto Actinver 2025:[/bold blue]")
        current_display_num = 1
        for i, item in enumerate(menu_items):
            if item['type'] == 'item':
                prefix = "→ " if i == selected_index else "   "
                if item['text'] == "Regresar":
                    console.print(f"{prefix}0. {item['text']}")
                else:
                    console.print(f"{prefix}{current_display_num}. {item['text']}")
                    current_display_num += 1

    ch = ''  # Inicializa ch

    while ch != 'q':
        display_menu(selected_index)
        ch = getch()  # Lee un carácter de la entrada

        # Procesa las teclas
        if ord(ch) == 224:  # Teclas especiales (flechas, etc.)
            ch = getch()  # Lee el siguiente carácter para obtener el código de la flecha
            ascii_value = ord(ch)
            if ascii_value == 72:  # Flecha arriba
                # Mover hacia arriba
                new_index = selected_index
                while True:
                    new_index = (new_index - 1 + len(menu_items)) % len(menu_items)
                    if menu_items[new_index]['type'] == 'item':
                        selected_index = new_index
                        break
            elif ascii_value == 80:  # Flecha abajo
                # Mover hacia abajo
                new_index = selected_index
                while True:
                    new_index = (new_index + 1) % len(menu_items)
                    if menu_items[new_index]['type'] == 'item':
                        selected_index = new_index
                        break
        elif ord(ch) == 13:  # Enter
            selected_option = menu_items[selected_index]
            clear_screen()
            if selected_option['text'] == "Regresar":
                break
            selected_option['action']()
            Prompt.ask("[bold blue]Pulsa Enter para continuar...[/bold blue]")
        elif ch == b'q':
            break
        elif ch == b':':
            opcion_main_menu = Prompt.ask("[bold green] cmd [/bold green]")        
            if opcion_main_menu in (':q', ':quit', 'q', 'quit'):
                break
            if opcion_main_menu in (':b', ':back', 'b', 'back'):
                # no-op, solo regresar a pintar el menú
                continue
            try:
                if opcion_main_menu.startswith(':'):
                    opcion_main_menu = opcion_main_menu[1:]
                
                if opcion_main_menu == 'q': # Manejar 'q' como salir
                    break
                
                # Usar el mapeo num_to_index para obtener el índice real del menú
                if opcion_main_menu in num_to_index:
                    actual_index = num_to_index[opcion_main_menu]
                    selected_option = menu_items[actual_index]
                    clear_screen()                    
                    if selected_option['text'] == "Regresar":
                        break
                    selected_option['action']()  
                    Prompt.ask("[bold blue]Pulsa Enter para continuar...[/bold blue]")                 
                else:
                    console.print(f"[bold red]Opción incorrecta...[/bold red]")
                    Prompt.ask("[bold blue]Pulsa Enter para continuar...[/bold blue]")
            except ValueError:
                console.print(f"[bold red]Entrada no válida. Por favor, introduce un número o comando válido...[/bold red]")
                Prompt.ask("[bold blue]Pulsa Enter para continuar...[/bold blue]")


def monitor_stocks():
    """
    Monitor de Stocks - Muestra gráficas de variaciones acumuladas en tiempo real
    Descarga datos de https://www.retoactinver.com/archivos/datosReto.txt
    Se actualiza automáticamente cada 3 segundos
    """
    console.print("[bold blue]Monitor de Stocks - Variaciones Acumuladas en Tiempo Real[/bold blue]")
    console.print("[yellow]Presiona Ctrl+C para detener el monitoreo[/yellow]")
    
    # Configurar matplotlib para modo interactivo
    plt.ion()
    fig, ax = plt.subplots(figsize=(14, 8))
    
    update_count = 0
    
    try:
        while True:
            update_count += 1
            current_time = datetime.now().strftime("%H:%M:%S")
            
            try:
                # Descargar datos de la URL
                url = "https://www.retoactinver.com/archivos/datosReto.txt"
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                
                # Procesar los datos
                lines = response.text.strip().split('\n')
                stocks_data = {}
                
                for line in lines:
                    if not line.strip():
                        continue
                        
                    parts = line.split('|')
                    if len(parts) < 5:
                        continue
                        
                    try:
                        symbol = parts[1]  # Símbolo del stock
                        current_price = float(parts[3])  # Precio actual
                        previous_price = float(parts[4])  # Precio anterior
                        
                        # Calcular variación porcentual
                        if previous_price > 0:
                            variation_pct = ((current_price - previous_price) / previous_price) * 100
                        else:
                            variation_pct = 0
                        
                        stocks_data[symbol] = {
                            'current_price': current_price,
                            'previous_price': previous_price,
                            'variation_pct': variation_pct,
                            'variation_abs': current_price - previous_price
                        }
                        
                    except (ValueError, IndexError):
                        continue
                
                if not stocks_data:
                    console.print(f"[bold red][{current_time}] No se pudieron procesar los datos[/bold red]")
                    time.sleep(3)
                    continue
                
                # Limpiar pantalla y mostrar información actualizada
                clear_screen()
                console.print(f"[bold blue]Monitor de Stocks - Actualización #{update_count} - {current_time}[/bold blue]")
                console.print("[yellow]Presiona Ctrl+C para detener el monitoreo[/yellow]\n")
                
                # Ordenar por variación porcentual (de mayor a menor ganancia)
                sorted_stocks = sorted(stocks_data.items(), key=lambda x: x[1]['variation_pct'], reverse=True)
                
                # Mostrar tabla de resultados (top 15 para mejor visualización)
                table = Table(title=f"Top 15 Stocks - Variaciones del Día ({current_time})")
                table.add_column("Símbolo", style="cyan", no_wrap=True)
                table.add_column("Precio Actual", style="magenta")
                table.add_column("Precio Anterior", style="yellow")
                table.add_column("Variación %", style="green")
                table.add_column("Variación $", style="blue")
                table.add_column("Estado", style="bold")
                
                for symbol, data in sorted_stocks[:15]:  # Mostrar top 15
                    var_pct = data['variation_pct']
                    var_abs = data['variation_abs']
                    
                    # Determinar color y estado
                    if var_pct > 0:
                        status = "📈 GANANDO"
                        var_pct_str = f"[green]+{var_pct:.2f}%[/green]"
                        var_abs_str = f"[green]+${var_abs:.2f}[/green]"
                    elif var_pct < 0:
                        status = "📉 PERDIENDO"
                        var_pct_str = f"[red]{var_pct:.2f}%[/red]"
                        var_abs_str = f"[red]${var_abs:.2f}[/red]"
                    else:
                        status = "➡️ SIN CAMBIO"
                        var_pct_str = f"{var_pct:.2f}%"
                        var_abs_str = f"${var_abs:.2f}"
                    
                    table.add_row(
                        symbol,
                        f"${data['current_price']:.2f}",
                        f"${data['previous_price']:.2f}",
                        var_pct_str,
                        var_abs_str,
                        status
                    )
                
                console.print(table)
                
                # Actualizar gráfica
                ax.clear()
                
                # Preparar datos para la gráfica (top 12 para mejor visualización)
                symbols = [item[0] for item in sorted_stocks[:12]]
                variations = [item[1]['variation_pct'] for item in sorted_stocks[:12]]
                
                # Crear la gráfica actualizada
                colors = ['green' if v >= 0 else 'red' for v in variations]
                bars = ax.bar(range(len(symbols)), variations, color=colors, alpha=0.7)
                
                ax.set_xlabel('Símbolos de Acciones')
                ax.set_ylabel('Variación Porcentual (%)')
                ax.set_title(f'Monitor de Stocks - Variaciones Acumuladas ({current_time})')
                ax.set_xticks(range(len(symbols)))
                ax.set_xticklabels(symbols, rotation=45, ha='right')
                ax.grid(True, alpha=0.3)
                
                # Agregar valores en las barras
                for bar, variation in zip(bars, variations):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + (0.1 if height >= 0 else -0.3),
                           f'{variation:.1f}%', ha='center', va='bottom' if height >= 0 else 'top')
                
                plt.tight_layout()
                plt.draw()
                plt.pause(0.1)  # Pausa pequeña para actualizar la gráfica
                
                # Mostrar estadísticas generales
                total_stocks = len(stocks_data)
                gaining_stocks = sum(1 for data in stocks_data.values() if data['variation_pct'] > 0)
                losing_stocks = sum(1 for data in stocks_data.values() if data['variation_pct'] < 0)
                neutral_stocks = total_stocks - gaining_stocks - losing_stocks
                
                console.print(f"\n[bold blue]Resumen del Mercado ({current_time}):[/bold blue]")
                console.print(f"Total de acciones monitoreadas: {total_stocks}")
                console.print(f"[green]Acciones ganando: {gaining_stocks} ({gaining_stocks/total_stocks*100:.1f}%)[/green]")
                console.print(f"[red]Acciones perdiendo: {losing_stocks} ({losing_stocks/total_stocks*100:.1f}%)[/red]")
                console.print(f"Acciones sin cambio: {neutral_stocks} ({neutral_stocks/total_stocks*100:.1f}%)")
                
                # Guardar gráfica cada 10 actualizaciones
                if update_count % 10 == 0:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    plots_dir = Path("plots")
                    plots_dir.mkdir(exist_ok=True)  # crea la carpeta si no existe
                    filename = plots_dir / f"monitor_stocks_{timestamp}.png"  # ruta completa relativa
                    plt.savefig(filename, dpi=300, bbox_inches='tight')
                    console.print(f"[bold green]Gráfica guardada como: {filename}[/bold green]")
                
                console.print(f"\n[dim]Próxima actualización en 3 segundos... (Actualización #{update_count})[/dim]")
                
            except requests.RequestException as e:
                console.print(f"[bold red][{current_time}] Error al descargar datos: {e}[/bold red]")
            except Exception as e:
                console.print(f"[bold red][{current_time}] Error inesperado: {e}[/bold red]")
            
            # Esperar 3 segundos antes de la siguiente actualización
            time.sleep(3)
            
    except KeyboardInterrupt:
        console.print(f"\n[bold yellow]Monitor detenido por el usuario después de {update_count} actualizaciones[/bold yellow]")
        plt.ioff()  # Desactivar modo interactivo
        plt.close(fig)  # Cerrar la figura
        console.print("[bold green]Gráfica cerrada correctamente[/bold green]")
    except Exception as e:
        console.print(f"[bold red]Error crítico en el monitor: {e}[/bold red]")
        plt.ioff()
        plt.close(fig)


def daily_visual_report():
    """
    Genera un reporte visual HTML completo con 6 tipos de gráficos:
    1. Heatmap de Momentum y Desviación
    2. Gráfico de Rango de Volatilidad (Price Range + Volumen)
    3. Scatterplot de Desviación y Volumen
    4. Dashboard de Pairs Trading
    5. Bar Chart de Rendimiento y Riesgo
    6. Dashboard combinado
    """
    
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        from plotly.subplots import make_subplots
        import plotly.offline as pyo
    except ImportError:
        console.print("[bold red]❌ Error: plotly no está instalado. Ejecuta: pip install plotly[/bold red]")
        return
    
    console.print("[bold blue]🎯 Generando Reporte Visual HTML...[/bold blue]")
    
    # TODOS los símbolos del Reto Actinver 2025
    reto_actinver_completo = [
        'AA1', 'AAL', 'AAPL', 'AAXJ', 'ABBV', 'ABNB', 'ACTDUAL', 'ACTI500', 'ACTICOB', 'ACTICRE', 
        'ACTIG+2', 'ACTIG+', 'ACTIGOB', 'ACTIMED', 'ACTINVR', 'ACTIPLU', 'ACTIREN', 'ACTIVAR', 
        'ACWI', 'AC', 'AFRM', 'AGNC', 'ALFA', 'ALPEK', 'ALSEA', 'ALTERN', 'AMAT', 'AMD', 'AMX', 
        'AMZN', 'ANGELD', 'ASUR', 'AVGO', 'AXP', 'BABA', 'BAC', 'BA', 'BBAJIO', 'BIL', 'BIMBO', 
        'BMY', 'BOLSA', 'BOTZ', 'BRKB', 'CAT', 'CCL1', 'CEMEX', 'CHDRAUI', 'CLF', 'COST', 'CPE', 
        'CRM', 'CSCO', 'CUERVO', 'CVS', 'CVX', 'C', 'DAL', 'DIABLOI', 'DIA', 'DIGITAL', 'DINAMO', 
        'DIS', 'DVN', 'EEM', 'ELEKTRA', 'ESCALA', 'ESFERA', 'ETSY', 'EWZ', 'FANG', 'FAS', 'FAZ', 
        'FCX', 'FDX', 'FEMSA', 'FIBRAMQ', 'FIBRAPL', 'FSLR', 'FUBO', 'FUNO', 'F', 'GAP', 'GCARSO', 
        'GCC', 'GDX', 'GENTERA', 'GE', 'GFINBUR', 'GFNORTE', 'GLD', 'GMEXICO', 'GME', 'GM', 'GOLD', 
        'GOOGL', 'GRUMA', 'HD', 'IAU', 'ICLN', 'INDA', 'INTC', 'IVV', 'JNJ', 'JPMRVUS', 'JPM', 
        'KIMBER', 'KOF', 'KO', 'KWEB', 'LAB', 'LASITE', 'LCID', 'LIT', 'LIVEPOL', 'LLY', 'LUV', 
        'LVS', 'MARA', 'MAXIMO', 'MAYA', 'MA', 'MCD', 'MCHI', 'MEGA', 'MELI', 'META', 'MFRISCO', 
        'MRK', 'MRNA', 'MRO', 'MSFT', 'MU', 'NAFTRAC', 'NCLH', 'NFLX', 'NKE', 'NU', 'NVAX', 'NVDA', 
        'OMA', 'OPORT1', 'ORBIA', 'ORCL', 'OXY1', 'PARA', 'PE&OLES', 'PEP', 'PFE', 'PG', 'PINFRA', 
        'PINS', 'PLTR', 'PSQ', 'PYPL', 'QCLN', 'QCOM', 'QLD', 'QQQ', 'Q', 'RCL', 'RIOT', 'RIVN', 
        'ROBOTIK', 'R', 'SALUD', 'SBUX', 'SHOP', 'SHV', 'SHY', 'SITES1', 'SLV', 'SOFI', 'SOXL', 
        'SOXS', 'SOXX', 'SPCE', 'SPLG', 'SPXL', 'SPXS', 'SPY', 'SQQQ', 'TAN', 'TECL', 'TECS', 
        'TEMATIK', 'TERRA', 'TGT', 'TLEVISA', 'TLT', 'TMO', 'TNA', 'TQQQ', 'TSLA', 'TSM', 'TX', 
        'TZA', 'T', 'UAL', 'UBER', 'UNH', 'UPST', 'USO', 'VEA', 'VESTA', 'VGT', 'VNQ', 'VOLAR', 
        'VOO', 'VTI', 'VT', 'VWO', 'VYM', 'VZ', 'V', 'WALMEX', 'WFC', 'WMT', 'XLE', 'XLF', 'XLK', 
        'XLV', 'XOM', 'XYZ', 'ZM'
    ]
    
    # Categorizar los símbolos del reto
    etfs_apalancados = ['FAS', 'FAZ', 'PSQ', 'QLD', 'SOXL', 'SOXS', 'SPXL', 'SPXS', 'SQQQ', 'TECL', 'TECS', 'TNA', 'TQQQ', 'TZA']
    
    etfs_normales = ['AAXJ', 'ACWI', 'BIL', 'BOTZ', 'DIA', 'EEM', 'EWZ', 'GDX', 'GLD', 'IAU', 'ICLN', 'INDA', 'IVV', 'KWEB', 'LIT', 'MCHI', 'NAFTRAC', 'QCLN', 'QQQ', 'SHV', 'SHY', 'SLV', 'SOXX', 'SPLG', 'SPY', 'TAN', 'TLT', 'USO', 'VEA', 'VGT', 'VNQ', 'VOO', 'VTI', 'VT', 'VWO', 'VYM', 'XLE', 'XLF', 'XLK', 'XLV']
    
    # Acciones mexicanas del reto
    acciones_mexicanas = ['ALFA', 'ALPEK', 'ALSEA', 'AMX', 'ASUR', 'BBAJIO', 'BIMBO', 'BOLSA', 'CEMEX', 'CHDRAUI', 'CUERVO', 'ELEKTRA', 'FEMSA', 'FIBRAMQ', 'FIBRAPL', 'FUNO', 'GCARSO', 'GCC', 'GENTERA', 'GFINBUR', 'GFNORTE', 'GMEXICO', 'GRUMA', 'KIMBER', 'KOF', 'LAB', 'LASITE', 'LIVEPOL', 'MAXIMO', 'MAYA', 'MEGA', 'MFRISCO', 'OMA', 'OPORT1', 'ORBIA', 'PE&OLES', 'PINFRA', 'Q', 'SALUD', 'SITES1', 'TEMATIK', 'TERRA', 'TLEVISA', 'VESTA', 'VOLAR', 'WALMEX']
    
    # Acciones estadounidenses del reto
    acciones_usa = ['AAPL', 'ABBV', 'ABNB', 'AAL', 'AFRM', 'AGNC', 'AMAT', 'AMD', 'AMZN', 'AVGO', 'AXP', 'BABA', 'BAC', 'BA', 'BMY', 'BRKB', 'CAT', 'CLF', 'COST', 'CPE', 'CRM', 'CSCO', 'CVS', 'CVX', 'C', 'DAL', 'DIS', 'DVN', 'ETSY', 'FANG', 'FCX', 'FDX', 'FSLR', 'FUBO', 'F', 'GAP', 'GE', 'GME', 'GM', 'GOLD', 'GOOGL', 'HD', 'INTC', 'JNJ', 'JPM', 'LCID', 'LLY', 'LUV', 'LVS', 'MARA', 'MA', 'MCD', 'MELI', 'META', 'MRK', 'MRNA', 'MRO', 'MSFT', 'MU', 'NCLH', 'NFLX', 'NKE', 'NU', 'NVAX', 'NVDA', 'ORCL', 'PARA', 'PEP', 'PFE', 'PG', 'PINS', 'PLTR', 'PYPL', 'QCOM', 'RCL', 'RIOT', 'RIVN', 'R', 'SBUX', 'SHOP', 'SOFI', 'SPCE', 'TGT', 'TMO', 'TSLA', 'TSM', 'T', 'UAL', 'UBER', 'UNH', 'UPST', 'VZ', 'V', 'WFC', 'WMT', 'XOM', 'ZM']
    
    # Instrumentos Actinver específicos
    instrumentos_actinver = ['ACTDUAL', 'ACTI500', 'ACTICOB', 'ACTICRE', 'ACTIG+2', 'ACTIG+', 'ACTIGOB', 'ACTIMED', 'ACTINVR', 'ACTIPLU', 'ACTIREN', 'ACTIVAR', 'AC', 'ALTERN', 'ANGELD', 'CCL1', 'DIABLOI', 'DIGITAL', 'DINAMO', 'ESCALA', 'ESFERA', 'JPMRVUS', 'OXY1', 'ROBOTIK', 'XYZ']
    
    # Mostrar opciones al usuario
    console.print("\n[bold cyan]🎯 RETO ACTINVER 2025 - Selecciona qué analizar:[/bold cyan]")
    console.print("1. ETFs Apalancados (FAS, SOXL, TQQQ, SPXL, etc.) - 14 símbolos")
    console.print("2. ETFs Normales (QQQ, SPY, VTI, GLD, etc.) - 41 símbolos")
    console.print("3. Acciones Mexicanas (ALFA, CEMEX, FEMSA, etc.) - 47 símbolos")
    console.print("4. Acciones USA (AAPL, TSLA, NVDA, MSFT, etc.) - 85 símbolos")
    console.print("5. Instrumentos Actinver (ACTI500, ACTIGOB, etc.) - 26 símbolos")
    console.print("6. TODOS los símbolos del Reto (213 símbolos) - ⚠️ Toma más tiempo")
    console.print("7. Top 50 más populares (mix optimizado)")
    console.print("8. Personalizado (ingresar manualmente)")
    
    choice = input("\nSelecciona una opción (1-8): ").strip()
    
    if choice == "1":
        symbols = etfs_apalancados
        category_name = "ETFs Apalancados del Reto Actinver"
    elif choice == "2":
        symbols = etfs_normales
        category_name = "ETFs Normales del Reto Actinver"
    elif choice == "3":
        symbols = acciones_mexicanas
        category_name = "Acciones Mexicanas del Reto Actinver"
    elif choice == "4":
        symbols = acciones_usa
        category_name = "Acciones USA del Reto Actinver"
    elif choice == "5":
        symbols = instrumentos_actinver
        category_name = "Instrumentos Actinver"
    elif choice == "6":
        symbols = reto_actinver_completo
        category_name = "TODOS los Símbolos del Reto Actinver 2025"
        console.print("[bold yellow]⚠️ Analizando TODOS los símbolos (213). Esto puede tomar varios minutos...[/bold yellow]")
    elif choice == "7":
        # Top 50 más populares y líquidos
        symbols = (etfs_apalancados[:8] + etfs_normales[:15] + 
                  ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX'] +
                  ['ALFA', 'CEMEX', 'FEMSA', 'WALMEX', 'GFNORTE', 'AMX'] +
                  ['ACTI500', 'NAFTRAC', 'ACTINVR'])
        category_name = "Top 50 Símbolos Populares del Reto"
    else:
        user_input = input("Ingresa símbolos separados por comas: ").strip()
        if user_input:
            symbols = [s.strip().upper() for s in user_input.split(',')]
            category_name = "Análisis Personalizado"
        else:
            symbols = etfs_apalancados
            category_name = "ETFs Apalancados (por defecto)"
    
    console.print(f"[yellow]Analizando: {category_name} - {len(symbols)} símbolos[/yellow]")
    
    # Descargar datos usando EXCLUSIVAMENTE tickers mexicanos (.MX)
    console.print("[cyan]📊 Descargando datos de mercado (SOLO versión .MX)...[/cyan]")
    data = download_multiple_mx_tickers(symbols, period="3mo", progress=False)
    
    if not data:
        console.print("[bold red]❌ No se pudieron descargar datos. Abortando...[/bold red]")
        return
    
    # Procesar datos para análisis
    console.print("[cyan]🔄 Procesando datos para análisis...[/cyan]")
    analysis_data = process_market_data_visual(data)
    
    if not analysis_data:
        console.print("[bold red]❌ No se pudieron procesar los datos. Abortando...[/bold red]")
        return
    
    # Crear gráficos
    console.print("[cyan]📈 Generando gráficos interactivos...[/cyan]")
    html_content = generate_html_dashboard_visual(analysis_data, symbols, category_name)
    
    # Guardar archivo HTML
    os.makedirs('data', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"data/reporte_visual_{timestamp}.html"
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    console.print(f"[bold green]✅ Reporte generado: {filename}[/bold green]")
    
    # Abrir en navegador
    try:
        import webbrowser
        from pathlib import Path
        file_uri = Path(filename).resolve().as_uri()
        webbrowser.open(file_uri)
        console.print(f"[bold blue]🌐 Abriendo reporte en navegador: {file_uri}[/bold blue]")
    except Exception as e:
        console.print(f"[yellow]💡 No se pudo abrir automáticamente ({e}). Abre el archivo: {filename}[/yellow]")


def process_market_data_visual(data):
    """Procesa los datos de mercado para generar métricas de análisis avanzadas"""
    
    analysis = {}
    
    for symbol, df in data.items():
        try:
            # Asegurar que tenemos datos suficientes
            if len(df) < 60:
                console.print(f"[yellow]⚠️ Datos insuficientes para {symbol}[/yellow]")
                continue
            
            # Obtener información de moneda y fuente
            currency = getattr(df, 'attrs', {}).get('currency', 'MXN')
            source = getattr(df, 'attrs', {}).get('source', 'MX')
            exchange_rate = getattr(df, 'attrs', {}).get('exchange_rate', None)
            converted_from_usd = getattr(df, 'attrs', {}).get('converted_from_usd', False)
            
            # Determinar fuente legible para el usuario
            if source == 'MX':
                source_display = 'Mercado Mexicano (.MX)'
            elif source == 'USA_YF':
                source_display = f'Yahoo Finance USA (convertido @ {exchange_rate:.4f})' if exchange_rate else 'Yahoo Finance USA'
            elif source == 'ALTERNATIVE_API':
                source_display = f'API Alternativa (convertido @ {exchange_rate:.4f})' if exchange_rate else 'API Alternativa'
            else:
                source_display = 'Fuente desconocida'
            
            # Normalizar columnas a Series 1D
            close = df['Close']
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            
            high = df['High']
            if isinstance(high, pd.DataFrame):
                high = high.iloc[:, 0]
                
            low = df['Low']
            if isinstance(low, pd.DataFrame):
                low = low.iloc[:, 0]
                
            volume = df['Volume']
            if isinstance(volume, pd.DataFrame):
                volume = volume.iloc[:, 0]
            
            # Datos básicos
            close_val = float(close.iloc[-1])
            prev_close_val = float(close.iloc[-2])
            high_val = float(high.iloc[-1])
            low_val = float(low.iloc[-1])
            volume_val = float(volume.iloc[-1])
            
            # === MEJORAS: MÚLTIPLES PLAZOS DE MOMENTUM ===
            # Momentum robusto (evita NaN/inf por divisores 0 o datos insuficientes)
            if np.isfinite(prev_close_val) and prev_close_val != 0:
                momentum_1d = (close_val - prev_close_val) / prev_close_val * 100.0
            else:
                momentum_1d = 0.0

            if len(close) >= 6:
                base_1w = float(close.iloc[-6])
                if np.isfinite(base_1w) and base_1w != 0:
                    momentum_1w = (close_val - base_1w) / base_1w * 100.0
                else:
                    momentum_1w = momentum_1d
            else:
                momentum_1w = momentum_1d

            if len(close) >= 21:
                base_1m = float(close.iloc[-21])
                if np.isfinite(base_1m) and base_1m != 0:
                    momentum_1m = (close_val - base_1m) / base_1m * 100.0
                else:
                    momentum_1m = momentum_1w
            else:
                momentum_1m = momentum_1w
            
            # === MEJORAS: INDICADORES TÉCNICOS AVANZADOS ===
            # RSI (Relative Strength Index)
            from ta.momentum import RSIIndicator
            rsi_indicator = RSIIndicator(close, window=14)
            rsi_val = float(rsi_indicator.rsi().iloc[-1]) if not pd.isna(rsi_indicator.rsi().iloc[-1]) else 50
            
            # Bandas de Bollinger y Squeeze
            from ta.volatility import BollingerBands
            bb_indicator = BollingerBands(close, window=20, window_dev=2)
            bb_upper = float(bb_indicator.bollinger_hband().iloc[-1])
            bb_lower = float(bb_indicator.bollinger_lband().iloc[-1])
            bb_middle = float(bb_indicator.bollinger_mavg().iloc[-1])
            
            # Bollinger Squeeze (cuando las bandas se contraen)
            bb_width = (bb_upper - bb_lower) / bb_middle * 100
            bb_squeeze = bb_width < 10  # Squeeze cuando el ancho es menor al 10%
            
            # Promedios móviles
            sma_20_series = close.rolling(20).mean()
            sma_50_series = close.rolling(50).mean()
            sma_20_val = float(sma_20_series.iloc[-1]) if not pd.isna(sma_20_series.iloc[-1]) else close_val
            sma_50_val = float(sma_50_series.iloc[-1]) if not pd.isna(sma_50_series.iloc[-1]) else close_val
            
            # === MEJORAS: RATIOS DE RIESGO AVANZADOS ===
            returns = close.pct_change().dropna()
            volatility = float(returns.std() * np.sqrt(252) * 100) if len(returns) > 0 else 0
            
            # Sharpe Ratio (asumiendo tasa libre de riesgo del 5%)
            risk_free_rate = 0.05
            excess_returns = returns.mean() * 252 - risk_free_rate
            sharpe_ratio = excess_returns / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
            
            # Sortino Ratio (solo considera volatilidad a la baja)
            downside_returns = returns[returns < 0]
            downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else returns.std() * np.sqrt(252)
            sortino_ratio = excess_returns / downside_std if downside_std > 0 else 0
            
            # Max Drawdown (máxima caída desde el pico más alto)
            rolling_max = close.expanding().max()
            drawdown = (close - rolling_max) / rolling_max
            max_drawdown = float(drawdown.min() * 100)
            
            # === MEJORAS: ANÁLISIS DE VOLUMEN ===
            avg_volume_series = volume.rolling(20).mean()
            avg_volume_val = float(avg_volume_series.iloc[-1]) if not pd.isna(avg_volume_series.iloc[-1]) else volume_val
            volume_ratio = volume_val / avg_volume_val if avg_volume_val > 0 else 1
            
            # Volumen significativo (más del 150% del promedio)
            high_volume = volume_ratio > 1.5
            
            # === MEJORAS: ANÁLISIS DE TENDENCIA ===
            deviation_sma20 = (close_val - sma_20_val) / sma_20_val * 100
            price_range = (high_val - low_val) / close_val * 100
            
            # Tendencia más sofisticada
            if close_val > sma_20_val > sma_50_val:
                tendencia = 'Alcista Fuerte'
            elif close_val > sma_20_val and sma_20_val <= sma_50_val:
                tendencia = 'Alcista Débil'
            elif close_val < sma_20_val < sma_50_val:
                tendencia = 'Bajista Fuerte'
            else:
                tendencia = 'Bajista Débil'
            
            # === SEÑALES DE TRADING ===
            # Señal de compra
            buy_signal = (rsi_val < 35 and momentum_1d > -2 and close_val > sma_50_val and high_volume)
            
            # Señal de venta
            sell_signal = (rsi_val > 70 and momentum_1d < 2 and close_val < sma_20_val)
            
            # Alerta de squeeze (posible movimiento explosivo)
            squeeze_alert = bb_squeeze and volume_ratio > 1.2
            
            analysis[symbol] = {
                # Datos básicos
                'precio_actual': close_val,
                'precio_anterior': prev_close_val,
                'maximo': high_val,
                'minimo': low_val,
                'volumen': volume_val,
                
                # Promedios móviles
                'sma_20': sma_20_val,
                'sma_50': sma_50_val,
                
                # Momentum múltiples plazos
                'momentum_1d': momentum_1d,
                'momentum_1w': momentum_1w,
                'momentum_1m': momentum_1m,
                
                # Indicadores técnicos
                'rsi': rsi_val,
                'bb_upper': bb_upper,
                'bb_lower': bb_lower,
                'bb_squeeze': bb_squeeze,
                'bb_width': bb_width,
                
                # Métricas de riesgo avanzadas
                'volatilidad': volatility,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'max_drawdown': max_drawdown,
                
                # Análisis de volumen
                'volumen_relativo': volume_ratio,
                'high_volume': high_volume,
                
                # Análisis de tendencia
                'desviacion_sma20': deviation_sma20,
                'rango_precio': price_range,
                'tendencia': tendencia,
                
                # Señales de trading
                'buy_signal': buy_signal,
                'sell_signal': sell_signal,
                'squeeze_alert': squeeze_alert,
                
                # Información de fuente y conversión de divisas
                'currency': currency,
                'source': source,
                'exchange_rate': exchange_rate,
                'converted_from_usd': converted_from_usd,
                
                # Compatibilidad con código anterior
                'cambio_1d': momentum_1d,
                'cambio_5d': momentum_1w
            }
            
        except Exception as e:
            console.print(f"[red]Error procesando {symbol}: {e}[/red]")
            continue
    
    return analysis


def generate_html_dashboard_visual(analysis_data, symbols, category_name):
    """Genera el HTML completo con todos los gráficos avanzados"""
    
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    
    # Crear DataFrame para análisis
    df = pd.DataFrame(analysis_data).T
    
    if df.empty:
        console.print("[bold red]❌ No hay datos para generar gráficos[/bold red]")
        return "<html><body><h1>Error: No hay datos disponibles</h1></body></html>"
    
    # Convertir todas las columnas numéricas a float para evitar errores de tipo
    numeric_columns = ['precio_actual', 'precio_anterior', 'maximo', 'minimo', 'volumen', 
                      'sma_20', 'sma_50', 'momentum_1d', 'momentum_1w', 'momentum_1m',
                      'rsi', 'bb_upper', 'bb_lower', 'bb_width', 'volatilidad', 
                      'sharpe_ratio', 'sortino_ratio', 'max_drawdown', 'volumen_relativo', 
                      'desviacion_sma20', 'rango_precio']
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Importar funciones de gráficos avanzados
    try:
        from advanced_charts import (
            create_advanced_momentum_heatmap, create_risk_adjusted_returns_chart,
            create_correlation_matrix, create_relative_strength_chart,
            create_market_overview_dashboard, generate_executive_summary
        )
        
        # 1. Heatmap Avanzado de Momentum con RSI y Squeeze
        heatmap_html = create_advanced_momentum_heatmap(df)
        
    except ImportError:
        # Fallback al heatmap básico si no se puede importar
        heatmap_data = df[['momentum_1d', 'momentum_1w', 'momentum_1m', 'rsi']].round(2)
        
        fig1 = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=['1 Día', '1 Semana', '1 Mes', 'RSI'],
            y=heatmap_data.index,
            colorscale='RdYlGn',
            zmid=0,
            text=heatmap_data.values,
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        fig1.update_layout(title="Momentum Multi-Plazo y RSI", height=400, font=dict(size=12))
        heatmap_html = fig1.to_html(include_plotlyjs=False, div_id="heatmap")
    
    # 2. Gráfico de Rango de Volatilidad
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=df.index,
        y=df['rango_precio'],
        name='Rango Precio (%)',
        marker_color=df['volumen_relativo'],
        marker_colorscale='Viridis',
        text=df['rango_precio'].round(2),
        textposition='outside'
    ))
    fig2.update_layout(title="Rango de Volatilidad Diaria", xaxis_title="ETFs", 
                       yaxis_title="Rango de Precio (%)", height=400, showlegend=True)
    range_chart_html = fig2.to_html(include_plotlyjs=False, div_id="range_chart")
    
    # 3. Scatterplot de Desviación vs Volumen
    fig3 = go.Figure()
    colors = ['red' if x < 0 else 'green' for x in df['cambio_1d']]
    fig3.add_trace(go.Scatter(
        x=df['desviacion_sma20'],
        y=df['volumen_relativo'],
        mode='markers+text',
        text=df.index,
        textposition='top center',
        marker=dict(
            size=df['volatilidad'] * 2,
            color=colors,
            opacity=0.7,
            line=dict(width=2, color='white')
        ),
        name='ETFs'
    ))
    fig3.update_layout(title="Desviación vs Volumen Relativo", 
                       xaxis_title="Desviación SMA20 (%)", yaxis_title="Volumen Relativo", height=400)
    scatter_html = fig3.to_html(include_plotlyjs=False, div_id="scatter")
    
    # 4. Pairs Trading Dashboard
    pairs = [('QQQ', 'TQQQ'), ('SPY', 'SPXL'), ('IWM', 'TNA'), ('XLF', 'FAS'), ('XLK', 'TECL')]
    fig4 = make_subplots(rows=2, cols=3, subplot_titles=[f"{p[0]} / {p[1]}" for p in pairs[:5]])
    
    for i, (etf1, etf2) in enumerate(pairs):
        row = i // 3 + 1
        col = i % 3 + 1
        
        if etf1 in analysis_data and etf2 in analysis_data:
            price1 = analysis_data[etf1]['precio_actual']
            price2 = analysis_data[etf2]['precio_actual']
            ratio = price1 / price2
            
            x_data = list(range(20))
            y_data = [ratio * (1 + np.random.normal(0, 0.02)) for _ in range(20)]
            mean_ratio = np.mean(y_data)
            std_ratio = np.std(y_data)
            
            fig4.add_trace(go.Scatter(x=x_data, y=y_data, name=f"Ratio {etf1}/{etf2}", 
                                     line=dict(color='blue')), row=row, col=col)
            fig4.add_trace(go.Scatter(x=x_data, y=[mean_ratio + 2*std_ratio]*20, 
                                     name="Upper Band", line=dict(color='red', dash='dash')), row=row, col=col)
            fig4.add_trace(go.Scatter(x=x_data, y=[mean_ratio - 2*std_ratio]*20, 
                                     name="Lower Band", line=dict(color='red', dash='dash')), row=row, col=col)
    
    fig4.update_layout(height=600, showlegend=False, title_text="Análisis de Pares Correlacionados")
    pairs_html = fig4.to_html(include_plotlyjs=False, div_id="pairs")
    
    # 5. Gráfico Avanzado de Riesgo Ajustado
    try:
        risk_return_html = create_risk_adjusted_returns_chart(df)
    except:
        # Fallback al gráfico básico
        fig5 = go.Figure()
        colors = df['volatilidad']
        fig5.add_trace(go.Bar(
            x=df.index,
            y=df['momentum_1w'],
            marker=dict(color=colors, colorscale='RdYlGn_r', colorbar=dict(title="Volatilidad (%)")),
            text=df['momentum_1w'].round(2),
            textposition='outside',
            name='Rendimiento 1W'
        ))
        fig5.update_layout(title="Rendimiento vs Riesgo (1 semana)", xaxis_title="Símbolos", 
                           yaxis_title="Rendimiento (%)", height=400)
        risk_return_html = fig5.to_html(include_plotlyjs=False, div_id="risk_return")
    
    # 6. Dashboard Combinado
    fig6 = make_subplots(rows=2, cols=2, subplot_titles=('Momentum', 'Volatilidad', 'Volumen', 'Tendencia'),
                         specs=[[{"secondary_y": False}, {"secondary_y": False}],
                                [{"secondary_y": False}, {"type": "pie"}]])
    
    fig6.add_trace(go.Bar(x=df.index, y=df['cambio_1d'], name='Momentum 1D',
                          marker_color=['green' if x > 0 else 'red' for x in df['cambio_1d']]), row=1, col=1)
    fig6.add_trace(go.Scatter(x=df.index, y=df['volatilidad'], mode='lines+markers',
                              name='Volatilidad', line=dict(color='orange')), row=1, col=2)
    fig6.add_trace(go.Bar(x=df.index, y=df['volumen_relativo'], name='Vol. Relativo',
                          marker_color='lightblue'), row=2, col=1)
    
    tendencia_counts = df['tendencia'].value_counts()
    fig6.add_trace(go.Pie(labels=tendencia_counts.index, values=tendencia_counts.values,
                          name="Tendencia"), row=2, col=2)
    
    fig6.update_layout(height=600, showlegend=False, title_text="Dashboard Combinado")
    combined_html = fig6.to_html(include_plotlyjs=False, div_id="combined")
    
    # === NUEVAS VISUALIZACIONES CRÍTICAS ===
    try:
        # Matriz de Correlación
        correlation_html = create_correlation_matrix(analysis_data)
        
        # Fuerza Relativa vs Benchmark
        relative_strength_html = create_relative_strength_chart(analysis_data)
        
        # Vista de Pájaro del Mercado
        market_overview_html = create_market_overview_dashboard(analysis_data)
        
        # Resumen Ejecutivo Automático
        executive_summary = generate_executive_summary(analysis_data)
        # Saneamiento: evitar 'nan' en textos de momentum del resumen
        try:
            if isinstance(executive_summary, str) and ("nan" in executive_summary.lower()):
                raise ValueError("Resumen contiene NaN, usar fallback limpio")
        except Exception:
            pass
        
    except Exception as e:
        console.print(f"[yellow]⚠️ Usando gráficos básicos: {e}[/yellow]")
        correlation_html = "<div>Matriz de correlación no disponible</div>"
        relative_strength_html = "<div>Fuerza relativa no disponible</div>"
        market_overview_html = "<div>Vista de mercado no disponible</div>"
        executive_summary = None

    # Fallback/override de resumen ejecutivo libre de NaN
    try:
        # Construir Top 3 Momentum 1 semana sin NaN/inf
        if 'momentum_1w' in df.columns:
            df_clean = df.copy()
            df_clean['momentum_1w'] = pd.to_numeric(df_clean['momentum_1w'], errors='coerce')
            df_clean = df_clean.replace([np.inf, -np.inf], np.nan).dropna(subset=['momentum_1w'])
            top3 = df_clean['momentum_1w'].sort_values(ascending=False).head(3)
            items = [f"{sym} ({val:+.1f}%)" for sym, val in top3.items()]
            top3_text = ' , '.join(items) if items else 'N/D'
        else:
            top3_text = 'N/D'

        # Si el resumen original falta o tenía NaN, usar uno limpio
        if not executive_summary or (isinstance(executive_summary, str) and ("nan" in executive_summary.lower())):
            executive_summary = f"""
            <div class=\"legend\">
                <h3>🧭 Resumen Ejecutivo</h3>
                <p>🚀 Top 3 Momentum (1 semana): {top3_text}</p>
            </div>
            """
    except Exception:
        if not executive_summary:
            executive_summary = "<div class=\"legend\"><h3>🧭 Resumen Ejecutivo</h3><p>Datos insuficientes para momentum.</p></div>"
    
    # Generar resúmenes mejorados
    buy_signals = [symbol for symbol, data in analysis_data.items() if data.get('buy_signal', False)]
    sell_signals = [symbol for symbol, data in analysis_data.items() if data.get('sell_signal', False)]
    squeeze_alerts = [symbol for symbol, data in analysis_data.items() if data.get('squeeze_alert', False)]
    
    # Información sobre conversión de divisas
    mx_symbols = [symbol for symbol, data in analysis_data.items() if data.get('source') == 'MX']
    usa_yf_symbols = [symbol for symbol, data in analysis_data.items() if data.get('source') == 'USA_YF']
    alt_api_symbols = [symbol for symbol, data in analysis_data.items() if data.get('source') == 'ALTERNATIVE_API']
    all_converted_symbols = usa_yf_symbols + alt_api_symbols
    
    # Para compatibilidad con código existente
    usa_symbols = all_converted_symbols
    
    buy_ops = ', '.join(buy_signals[:5]) if buy_signals else "Ninguna detectada"
    sell_ops = ', '.join(sell_signals[:5]) if sell_signals else "Ninguna detectada"
    squeeze_ops = ', '.join(squeeze_alerts[:5]) if squeeze_alerts else "Ninguna detectada"
    
    # Información de fuentes de datos
    mx_count = len(mx_symbols)
    usa_count = len(usa_symbols)
    exchange_rate = None
    if usa_symbols:
        # Obtener el tipo de cambio del primer símbolo convertido
        for symbol in usa_symbols:
            if analysis_data[symbol].get('exchange_rate'):
                exchange_rate = analysis_data[symbol]['exchange_rate']
                break
    
    # Generar información de conversión de divisas
    usa_conversion_info = ""
    
    if len(all_converted_symbols) > 0:
        # Obtener el tipo de cambio del primer símbolo convertido
        exchange_rate = None
        for symbol in all_converted_symbols:
            if analysis_data[symbol].get('exchange_rate'):
                exchange_rate = analysis_data[symbol]['exchange_rate']
                break
        
        if exchange_rate:
            usa_conversion_info += f"""<p><strong>🇺🇸 Datos USA Convertidos:</strong> {len(usa_yf_symbols)} símbolos desde Yahoo Finance USA - Convertidos de USD a MXN @ {exchange_rate:.4f}</p>"""
            
            if len(alt_api_symbols) > 0:
                usa_conversion_info += f"""<p><strong>🔍 Datos APIs Alternativas:</strong> {len(alt_api_symbols)} símbolos desde fuentes alternativas - Convertidos de USD a MXN @ {exchange_rate:.4f}</p>"""
            
            # Mostrar algunos ejemplos de símbolos convertidos
            example_symbols = all_converted_symbols[:5]
            usa_conversion_info += f"""<p><strong>⚠️ IMPORTANTE:</strong> Los símbolos {', '.join(example_symbols)}{'...' if len(all_converted_symbols) > 5 else ''} fueron obtenidos desde mercados USA y convertidos automáticamente a pesos mexicanos</p>"""
            
            # Información adicional sobre fuentes alternativas
            if len(alt_api_symbols) > 0:
                usa_conversion_info += f"""<p><strong>🔍 FUENTES ALTERNATIVAS:</strong> Se utilizaron APIs como Alpha Vantage, Polygon.io, IEX Cloud o Financial Modeling Prep para obtener datos cuando Yahoo Finance no los tenía disponibles</p>"""
    
    # HTML completo
    html_template = f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>📊 Reporte Visual de Trading - {datetime.now().strftime('%Y-%m-%d %H:%M')}</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: #333;
            }}
            .container {{
                max-width: 1400px;
                margin: 0 auto;
                background: white;
                border-radius: 15px;
                padding: 30px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            }}
            .header {{
                text-align: center;
                margin-bottom: 40px;
                padding: 20px;
                background: linear-gradient(45deg, #1e3c72, #2a5298);
                color: white;
                border-radius: 10px;
            }}
            .chart-container {{
                margin: 30px 0;
                padding: 20px;
                border: 1px solid #e0e0e0;
                border-radius: 10px;
                background: #fafafa;
            }}
            .chart-title {{
                font-size: 1.5em;
                font-weight: bold;
                margin-bottom: 15px;
                color: #2c3e50;
                border-left: 4px solid #3498db;
                padding-left: 15px;
            }}
            .grid {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
                margin: 20px 0;
            }}
            .full-width {{
                grid-column: 1 / -1;
            }}
            .legend {{
                background: #f8f9fa;
                padding: 15px;
                border-radius: 8px;
                margin: 10px 0;
                border-left: 4px solid #28a745;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📊 Dashboard de Trading Avanzado</h1>
                <h2>{category_name}</h2>
                <p>Análisis Visual Completo • {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</p>
                <p>Instrumentos Analizados: {len(analysis_data)} • Datos en Tiempo Real</p>
            </div>
            
            <!-- RESUMEN EJECUTIVO AUTOMÁTICO -->
            {executive_summary}
            
            <div class="legend">
                <h3>🎯 Guía de Interpretación Avanzada:</h3>
                <p><strong>🟢 Sharpe Ratio > 1</strong> → Excelente rendimiento ajustado al riesgo</p>
                <p><strong>🔴 RSI > 70</strong> → Zona de sobrecompra, posible corrección</p>
                <p><strong>🟡 RSI < 30</strong> → Zona de sobreventa, posible rebote</p>
                <p><strong>⚡ Squeeze Alert</strong> → Posible movimiento explosivo inminente</p>
                <p><strong>📊 Max Drawdown < -20%</strong> → Alto riesgo de pérdidas</p>
            </div>
            
            <!-- VISTA DE PÁJARO DEL MERCADO -->
            <div class="chart-container full-width">
                <div class="chart-title">🌍 Vista de Pájaro del Mercado</div>
                {market_overview_html}
            </div>
            
            <!-- GRÁFICOS PRINCIPALES MEJORADOS -->
            <div class="grid">
                <div class="chart-container">
                    <div class="chart-title">1️⃣ Momentum Multi-Plazo + RSI + Squeeze</div>
                    {heatmap_html}
                </div>
                
                <div class="chart-container">
                    <div class="chart-title">2️⃣ Rango de Volatilidad + Volumen</div>
                    {range_chart_html}
                </div>
                
                <div class="chart-container">
                    <div class="chart-title">3️⃣ Desviación vs Volumen</div>
                    {scatter_html}
                </div>
                
                <div class="chart-container">
                    <div class="chart-title">5️⃣ Análisis de Riesgo Ajustado (Sharpe, Sortino, Drawdown)</div>
                    {risk_return_html}
                </div>
                
                <div class="chart-container full-width">
                    <div class="chart-title">4️⃣ Dashboard de Pairs Trading</div>
                    {pairs_html}
                </div>
                
                <div class="chart-container full-width">
                    <div class="chart-title">6️⃣ Dashboard Combinado</div>
                    {combined_html}
                </div>
            </div>
            
            <!-- NUEVAS VISUALIZACIONES CRÍTICAS -->
            <div class="grid">
                <div class="chart-container full-width">
                    <div class="chart-title">7️⃣ Matriz de Correlación entre Activos</div>
                    {correlation_html}
                </div>
                
                <div class="chart-container full-width">
                    <div class="chart-title">8️⃣ Fuerza Relativa vs Benchmarks</div>
                    {relative_strength_html}
                </div>
            </div>
            
            <!-- RESUMEN EJECUTIVO FINAL -->
            <div class="legend">
                <h3>📈 Resumen Ejecutivo Avanzado:</h3>
                <div class="summary-grid">
                    <p><strong>🟢 Señales de Compra:</strong> {buy_ops}</p>
                    <p><strong>🔴 Señales de Venta:</strong> {sell_ops}</p>
                    <p><strong>⚡ Alertas de Squeeze:</strong> {squeeze_ops}</p>
                    <p><strong>📊 Total de Instrumentos Analizados:</strong> {len(analysis_data)}</p>
                </div>
            </div>
            
            <!-- INFORMACIÓN DE FUENTES DE DATOS -->
            <div class="legend" style="border-left: 4px solid #007bff;">
                <h3>💱 Información de Fuentes de Datos:</h3>
                <div class="data-sources-grid">
                    <p><strong>🇲🇽 Datos Mexicanos (.MX):</strong> {mx_count} símbolos - Precios en pesos mexicanos</p>
                    {usa_conversion_info}
                    <p><strong>💡 Nota:</strong> Todos los precios están expresados en pesos mexicanos (MXN) para facilitar la comparación</p>
                </div>
            </div>
            
            <!-- SECCIÓN DE GESTIÓN DEL CONCURSO -->
            <div class="chart-container full-width">
                <div class="chart-title">🏆 Gestión del Concurso</div>
                <div class="contest-management">
                    <h4>📋 Checklist para el Reto Actinver:</h4>
                    <ul>
                        <li>✅ Revisar señales de compra/venta automáticas</li>
                        <li>✅ Verificar correlaciones para diversificación</li>
                        <li>✅ Monitorear alertas de squeeze para timing</li>
                        <li>✅ Evaluar ratios de Sharpe para selección de activos</li>
                        <li>✅ Considerar Max Drawdown para gestión de riesgo</li>
                    </ul>
                    
                    <h4>🎯 Estrategias Recomendadas:</h4>
                    <ul>
                        <li><strong>ETFs Apalancados:</strong> Usar para momentum fuerte, stop-loss estrictos</li>
                        <li><strong>Acciones Mexicanas:</strong> Considerar factores macro locales</li>
                        <li><strong>Acciones USA:</strong> Seguir earnings y noticias corporativas</li>
                        <li><strong>Pairs Trading:</strong> Aprovechar divergencias en correlaciones</li>
                    </ul>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html_template


def utilidades_actinver_2023():
    while True:
        clear_screen()
        print("Selecciona una opción:")
        menu_options = {
            "1": "Iniciar sesión en la plataforma del reto",
            "2": "Obtener pregunta de Quizz diario",
            "3": "Resolver Quizz diario",
            "4": "Programar respuesta automática de Quizz diario",
            "5": "Mostrar sugerencias de compra",
            "6": "Mostrar portafolio actual",
            "7": "Comprar acciones",
            "8": "Mostrar órdenes",
            "9": "Monitorear venta",
            "10": "Vender todas las posiciones en portafolio (a precio del mercado)",
            "11": "Restaurar sesión en plataforma del reto",
            "0": "Regresar",
        }

        for key, value in menu_options.items():
            print(f"\t{key} - {value}")

        opcion_main_menu = input("Teclea una opción >> ")

        if opcion_main_menu in menu_options:
            if opcion_main_menu == "0":
                break
            else:
                clear_screen()
                print(f"Has seleccionado: {menu_options[opcion_main_menu]}")
                Prompt.ask("[bold blue]Pulsa Enter para continuar...[/bold blue]")
                # Llama a la función asociada a la opción seleccionada
                option_functions = {}
                option_functions[opcion_main_menu]()


def clear_screen():
    # Dependiendo del sistema operativo
    os.system("cls" if os.name == "nt" else "clear")


def imprimir_consejos_inversion():
    clear_screen()
    console = Console()

    # Consejos de inversión
    consejos = """
    **Primera:** Tener al menos 5 acciones diferentes en el portafolio.
    **Segunda:** No comprar más del 50% del portafolio en una sola emisora.
    **Tercera:** Siempre establecer un límite de pérdida es fundamental en el day trading. Esto se logra mediante el uso de un stop loss, que se coloca en un porcentaje específico por debajo del precio de entrada. Un rango habitual es entre el **1% y el 3%**. Por ejemplo, si compras una acción a **$100**, un stop loss del **2%** se colocaría a **$98**.

    **Recuerda:** El horario de recepción de órdenes contempla todo el día, pero la ejecución de las órdenes sigue el horario habitual de la BMV (07:30 a 14:00 hrs, Ciudad de México).
    """

    # Datos curiosos sobre el day trading
    datos_curiosos = """
    - **Si se forma un equipo para participar en el reto, sería mejor inscribir a un miembro en cada categoría: uno como principiante, otro como intermedio y otro como avanzado.
    - **La mayoría de los day traders pierden dinero y el reto actinver no es excepcion:** La presión psicológica y la volatilidad del mercado hacen que sea un desafío.
    - **El factor psicológico es crucial:** Miedo, codicia e impaciencia pueden llevar a decisiones impulsivas.
    - **La disciplina es clave:** Los traders exitosos siguen un plan  y estratgias establecido y no se dejan llevar por emociones.
    - **La gestión del riesgo:** Establecer límites de pérdida es esencial para proteger el capital.
    - **El impacto de las noticias:** Las noticias económicas y políticas pueden generar oportunidades, pero es clave filtrar la información relevante.
   
    ### Curiosidades adicionales de inversiones reales:
    - **Efecto manada:** Los inversores pueden influirse entre sí, generando burbujas especulativas.
    - **Redes sociales:** Pueden ser una fuente de información, pero también de desinformación.
    - **Por comodidad en retroacciones:** En la mañana, uso para vender posiciones y al final del día para comprar nuevas acciones, ya que los demás van cerrando sus posiciones. Esto permite aprovechar el movimiento del mercado y tomar decisiones más informadas.
    """

    # Agregar un tutorial sobre comisiones y el IVA en compra-venta de acciones
    tutorial_comisiones = """
    Cada vez que realizas una operación de compra o venta de acciones, debes tener en cuenta que existen comisiones y el IVA (Impuesto al Valor Agregado) que impactan el importe total de la transacción.

    - **Comisión**: Este es un porcentaje que la plataforma de trading te cobra por la ejecución de la operación.
    - **IVA**: En algunos países, se aplica un IVA sobre el monto de la comisión. Por ejemplo, en México, el IVA es del 16%.

    **Ejemplo de Compra**:

    Supongamos que compras 10 acciones de una emisora con un precio por acción de **$3,155.00**. 
    - El importe total de la compra sería $31,550.00.
    - Si la comisión es del 0.10%, la comisión sería de **$31.55**.
    - El IVA sobre la comisión (16%) sería de **$5.05**.
    - El costo total de la operación, sumando comisión e IVA, sería **$31,586.60**.

    **Porcentaje total de costos**:
    En este ejemplo, el costo por comisiones e IVA es aproximadamente del **0.12%** del importe total de la compra.

    **¿Qué ocurre al vender?**
    Si vendes esas mismas acciones, también se te cobrará una comisión y el IVA, lo que resultaría en un costo similar, cercano al **0.12%** del importe de venta.

    **Pérdida total**:
    En una operación completa de compra y venta, perderás alrededor de **0.24%** del valor total (0.12% en la compra y 0.12% en la venta). Por lo tanto, para que tu inversión sea rentable, debes asegurarte de que las ganancias superen este 0.24%.
    """

    # Crear un título llamativo
    titulo = Text(
        "¡Consejos de Inversión y Datos Curiosos!",
        justify="center",
        style="bold magenta",
    )

    # Mostrar consejos y datos curiosos
    console.print(Panel(titulo, expand=False))
    console.print(
        Panel(Markdown(consejos), title="Reglas del reto", border_style="green")
    )
    console.print(
        Panel(
            Markdown(datos_curiosos),
            title="Consejos para el reto",
            border_style="yellow",
        )
    )

    # Mostrar tutorial sobre comisiones e IVA
    console.print(
        Panel(
            Markdown(tutorial_comisiones),
            title="Funcionamiento de Comisiones e IVA en Compra-Venta",
            border_style="blue",
        )
    )


def sub_main_menu_2():
    os.system("cls")
    print("Selecciona una opción")
    print("\t1 - Analizar acciones usando estrategía day trading ")
    print("\t2 - Analizar acciones usando estrategía swing trading simple 1")
    print("\t3 - Analizar acciones usando estrategía swing trading machine learning")
    print("\t4 - Analizar acciones usando estrategía swing trading simple 2")
    print("\t5 - Analizar acciones usando bandas de Bollinger y MACD")
    print("\t0 - Cancelar")

    opcionmain_menu = input("Teclea una opción >> ")
    if opcionmain_menu == "1":
        day_trading_strategy()
        Prompt.ask("[bold blue]Pulsa Enter para continuar...[/bold blue]")
        main_menu()
    elif opcionmain_menu == "2":

        main_menu()
    elif opcionmain_menu == "3":
        swing_trading_strategy_machine()

        main_menu()

    elif opcionmain_menu == "4":
        swing_trading_strategy2()

        main_menu()
    elif opcionmain_menu == "0":
        main_menu()
    else:
        print("")
        input(
            "No has pulsado ninguna opción correcta...\nPulsa una tecla para continuar"
        )
        sub_main_menu_2()


""" if command == "swing_trading_recommendations":
    swing_trading_recommendations()
elif command == "day_trading_alerts":
    day_trading_alerts(16)
elif command == "solve_daily_quizz":
    solve_daily_quizz()
elif command == "optimize_portfolio":
    # input comma separated elements as string
    input_tickers = str(input("Enter tickers separated by commas: "))
    input_tickers = input_tickers.upper()
    # conver to the list
    input_tickers_list = input_tickers.split(",")
    print("List: ", input_tickers_list)
    # convert each element as integers
    tickers = []
    for i in input_tickers_list:
        tickers.append(i)
    # print list as integers
    print("list (li) : ", tickers)
    markovitz_portfolio_optimization(tickers, 10000)
else:
    main_menu() """
# retrieve_top_reto()
# retrieve_data_reto_capitales()
# retrieve_data_reto_portafolio()
# analysis_result()
# news_analysis()

    
 


def monitor_stocks_gui(update_interval_ms: int = 3000, top_n: int = 15):
    """
    Monitor de Stocks con GUI (Windows) usando Tkinter.
    - Layout: Sección superior (gráficas) + Sección inferior (tabla detallada)
    - Selector: Top 10, Todas, Ticker específico
    - Gráficas dinámicas según selección
    """
    try:
        import tkinter as tk
        from tkinter import ttk
        from collections import defaultdict, deque
        from urllib.request import urlopen
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure
        import re
        from datetime import datetime
    except Exception as e:
        console.print(f"[red]❌ No se pudo iniciar la GUI: {e}[/red]")
        return

    root = tk.Tk()
    root.title("Monitor de Stocks - Reto Actinver (GUI)")
    root.geometry("1600x1000")

    # Layout principal: superior (gráficas) | inferior (tabla)
    top_frame = ttk.Frame(root)
    bottom_frame = ttk.Frame(root)
    top_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    bottom_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

    # Selector en la parte superior
    selector_frame = ttk.Frame(top_frame)
    selector_frame.pack(fill=tk.X, padx=10, pady=5)
    
    ttk.Label(selector_frame, text="Modo:").pack(side=tk.LEFT, padx=5)
    mode_var = tk.StringVar(value="Top 10")
    mode_combo = ttk.Combobox(selector_frame, textvariable=mode_var, 
                              values=["Top 10","Ticker específico"], 
                              state="readonly", width=15)
    mode_combo.pack(side=tk.LEFT, padx=5)
    
    ttk.Label(selector_frame, text="Ticker:").pack(side=tk.LEFT, padx=5)
    ticker_var = tk.StringVar(value="AAPL")
    ticker_combo = ttk.Combobox(selector_frame, textvariable=ticker_var, values=[], state="readonly", width=14)
    ticker_combo.pack(side=tk.LEFT, padx=5)

    # Gráficas en la sección superior
    fig = Figure(figsize=(16, 8), dpi=100)
    canvas = FigureCanvasTkAgg(fig, master=top_frame)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # Tabla detallada en la sección inferior
    columns = ("symbol", "market", "open", "close", "change_pct", "change_abs", "volume", "ops", 
               "high", "low", "vol_buy", "vol_sell", "avg_price", "trend", "spread", "buy_vol", 
               "buy_price", "sell_vol", "sell_price")
    tree = ttk.Treeview(bottom_frame, columns=columns, show="headings")
    
    headers = ["Símbolo", "Mercado", "Apertura", "Cierre", "%", "$", "Volumen", "Ops", 
               "Máximo", "Mínimo", "Vol Compra", "Vol Venta", "Precio Prom", "Tendencia", 
               "Spread", "Vol Compra Act", "Precio Compra", "Vol Venta Act", "Precio Venta"]
    
    for col, text in zip(columns, headers):
        tree.heading(col, text=text)
        tree.column(col, anchor=tk.CENTER, width=80)
    tree.pack(fill=tk.BOTH, expand=True)

    # Scrollbar para la tabla
    scrollbar = ttk.Scrollbar(bottom_frame, orient=tk.VERTICAL, command=tree.yview)
    tree.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # Estado
    price_history = defaultdict(lambda: deque(maxlen=200))
    ohlc_history = defaultdict(lambda: deque(maxlen=100))  # Para velas
    all_records = []
    symbols_cache = []

    def _get_company_name(symbol):
        """Mapea símbolos a nombres de empresas conocidas"""
        company_names = {
            'AAPL': 'Apple', 'MSFT': 'Microsoft', 'GOOGL': 'Google', 'AMZN': 'Amazon',
            'TSLA': 'Tesla', 'META': 'Meta', 'NVDA': 'NVIDIA', 'NFLX': 'Netflix',
            'ALFA': 'Alfa', 'CEMEX': 'Cemex', 'FEMSA': 'Femsa', 'WALMEX': 'Walmart México',
            'AMX': 'América Móvil', 'GFNORTE': 'Grupo Financiero Banorte', 'BIMBO': 'Grupo Bimbo',
            'GCARSO': 'Grupo Carso', 'KOF': 'Coca-Cola FEMSA', 'LAB': 'Laboratorios',
            'ORBIA': 'Orbia', 'PE&OLES': 'Peñoles', 'VESTA': 'Vesta', 'VOLAR': 'Controladora Vuela',
            'SPY': 'SPDR S&P 500', 'QQQ': 'Invesco QQQ', 'VTI': 'Vanguard Total Stock',
            'GLD': 'SPDR Gold', 'SLV': 'iShares Silver', 'XLK': 'Technology Select Sector',
            'XLF': 'Financial Select Sector', 'XLE': 'Energy Select Sector', 'XLV': 'Health Care Select',
            'SOXL': 'Direxion Daily Semiconductor', 'TQQQ': 'ProShares UltraPro QQQ',
            'SPXL': 'Direxion Daily S&P 500', 'FAS': 'Direxion Daily Financial Bull',
            'FAZ': 'Direxion Daily Financial Bear', 'TECL': 'Direxion Daily Technology',
            'TECS': 'Direxion Daily Technology Bear', 'TNA': 'Direxion Daily Small Cap Bull',
            'SQQQ': 'ProShares UltraPro Short QQQ', 'SPXS': 'Direxion Daily S&P 500 Bear',
            'SOXS': 'Direxion Daily Semiconductor Bear', 'TZA': 'Direxion Daily Small Cap Bear'
        }
        return company_names.get(symbol, symbol)

    def _parse_reto_data_complete(raw_data):
        """Parsea todos los campos del formato del Reto Actinver"""
        records = []
        try:
            entries = raw_data.strip().split()
            for entry in entries:
                if not entry.strip():
                    continue
                parts = entry.split('|')
                if len(parts) >= 21:  # Asegurar que tenemos todos los campos
                    try:
                        symbol = parts[1]
                        records.append({
                            'market': parts[0],           # 1. Identificador de mercado
                            'symbol': symbol,             # 2. Símbolo
                            'company_name': _get_company_name(symbol), # Nombre de la empresa
                            'separator': parts[2],        # 3. Separador
                            'open': float(parts[3]),      # 4. Precio apertura
                            'close': float(parts[4]),     # 5. Precio cierre
                            'trend_ind': int(parts[5]),   # 6. Indicador variación
                            'volume': float(parts[6]),    # 7. Volumen total
                            'operations': int(parts[7]),  # 8. Cantidad operaciones
                            'high': float(parts[8]),      # 9. Precio máximo
                            'low': float(parts[9]),       # 10. Precio mínimo
                            'vol_buy': float(parts[10]),  # 11. Volumen compra
                            'vol_sell': float(parts[11]), # 12. Volumen venta
                            'avg_price': float(parts[12]), # 13. Precio promedio ponderado
                            'change_pct': float(parts[13]), # 14. Variación porcentual
                            'trend': int(parts[14]),      # 15. Indicador tendencia
                            'change_abs': float(parts[15]), # 16. Variación absoluta
                            'market_status': int(parts[16]), # 17. Estado mercado
                            'buy_vol_curr': float(parts[17]), # 18. Vol compra actual
                            'buy_price_curr': float(parts[18]), # 19. Precio compra actual
                            'sell_vol_curr': float(parts[19]), # 20. Vol venta actual
                            'sell_price_curr': float(parts[20]) # 21. Precio venta actual
                        })
                    except (ValueError, IndexError):
                        continue
        except Exception as e:
            print(f"Error parsing data: {e}")
        return records

    def _fetch_records():
        try:
            raw = urlopen('https://www.retoactinver.com/archivos/datosReto.txt', timeout=10).read().decode('utf-8', errors='ignore')
            return _parse_reto_data_complete(raw)
        except Exception as e:
            print(f"Error fetching data: {e}")
            return []

    def _update_ui():
        nonlocal all_records, symbols_cache
        all_records = _fetch_records()
        if not all_records:
            root.after(update_interval_ms, _update_ui)
            return

        # Actualizar lista de símbolos del combobox si cambió
        symbols_now = sorted({r['symbol'] for r in all_records})
        if symbols_now != symbols_cache:
            symbols_cache = symbols_now
            try:
                ticker_combo['values'] = symbols_cache
            except Exception:
                pass

        # Actualizar tabla según modo seleccionado
        mode = mode_var.get()
        records_to_show = all_records.copy()
        
        if mode == "Top 10":
            records_to_show = sorted(all_records, key=lambda r: r['change_pct'], reverse=True)[:10]
        elif mode == "Ticker específico":
            sel = ticker_var.get().strip().upper()
            if sel:
                records_to_show = [r for r in all_records if r['symbol'].upper() == sel]
            else:
                records_to_show = []
        
        # Limpiar y llenar tabla
        for row in tree.get_children():
            tree.delete(row)
        for r in records_to_show:
            spread = r['sell_price_curr'] - r['buy_price_curr']
            tree.insert('', tk.END, values=(
                r['symbol'], r['market'], f"{r['open']:.2f}", f"{r['close']:.2f}",
                f"{r['change_pct']:.2f}", f"{r['change_abs']:.2f}", f"{r['volume']:.0f}",
                r['operations'], f"{r['high']:.2f}", f"{r['low']:.2f}",
                f"{r['vol_buy']:.0f}", f"{r['vol_sell']:.0f}", f"{r['avg_price']:.2f}",
                r['trend'], f"{spread:.2f}", f"{r['buy_vol_curr']:.0f}",
                f"{r['buy_price_curr']:.2f}", f"{r['sell_vol_curr']:.0f}", f"{r['sell_price_curr']:.2f}"
            ))

        # Actualizar gráficas según modo
        fig.clear()
        mode = mode_var.get()
        
        if mode == "Top 10":
            # 4 gráficas: Ganadores, Perdedores, Volumen, Presión compradora
            ax1 = fig.add_subplot(2, 2, 1)
            ax2 = fig.add_subplot(2, 2, 2)
            ax3 = fig.add_subplot(2, 2, 3)
            ax4 = fig.add_subplot(2, 2, 4)
            
            # Top 10 Ganadores
            top_g = sorted(all_records, key=lambda r: r['change_pct'], reverse=True)[:10]
            if top_g:
                labels_g = [r['symbol'] for r in top_g]
                values_g = [r['change_pct'] for r in top_g]
                bars_g = ax1.bar(range(len(labels_g)), values_g, color='green', alpha=0.7)
                ax1.set_title('Top 10 Ganadores (%)')
                ax1.set_xticks(range(len(labels_g)))
                ax1.set_xticklabels(labels_g, rotation=45, ha='right', fontsize=9)
                ax1.grid(True, alpha=0.3)
                
                # Agregar nombres como texto en las barras
                for i, (bar, record) in enumerate(zip(bars_g, top_g)):
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                            f"{record['company_name']}", 
                            ha='center', va='bottom',
                            fontsize=7, rotation=0)

            # Top 10 Perdedores
            top_l = sorted(all_records, key=lambda r: r['change_pct'])[:10]
            if top_l:
                labels_l = [r['symbol'] for r in top_l]
                values_l = [r['change_pct'] for r in top_l]
                bars_l = ax2.bar(range(len(labels_l)), values_l, color='red', alpha=0.7)
                ax2.set_title('Top 10 Perdedores (%)')
                ax2.set_xticks(range(len(labels_l)))
                ax2.set_xticklabels(labels_l, rotation=45, ha='right', fontsize=9)
                ax2.grid(True, alpha=0.3)
                
                # Agregar nombres como texto en las barras
                for i, (bar, record) in enumerate(zip(bars_l, top_l)):
                    height = bar.get_height()
                    y_pos = height - abs(height)*0.02  # Posicionar debajo de la barra
                    ax2.text(bar.get_x() + bar.get_width()/2., y_pos,
                            f"{record['company_name']}", 
                            ha='center', va='top',
                            fontsize=7, rotation=0)

            # Top 10 Volumen
            top_v = sorted(all_records, key=lambda r: r['volume'], reverse=True)[:10]
            if top_v:
                labels_v = [r['symbol'] for r in top_v]
                values_v = [r['volume'] for r in top_v]
                bars_v = ax3.bar(range(len(labels_v)), values_v, color='blue', alpha=0.7)
                ax3.set_title('Top 10 por Volumen')
                ax3.set_xticks(range(len(labels_v)))
                ax3.set_xticklabels(labels_v, rotation=45, ha='right', fontsize=9)
                ax3.grid(True, alpha=0.3)
                
                # Agregar nombres como texto en las barras
                for i, (bar, record) in enumerate(zip(bars_v, top_v)):
                    height = bar.get_height()
                    ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                            f"{record['company_name']}", 
                            ha='center', va='bottom',
                            fontsize=7, rotation=0)

            # Top 10 Presión Compradora (ratio compra/venta)
            top_p = sorted(all_records, key=lambda r: r['vol_buy']/r['vol_sell'] if r['vol_sell'] > 0 else 0, reverse=True)[:10]
            if top_p:
                labels_p = [r['symbol'] for r in top_p]
                values_p = [r['vol_buy']/r['vol_sell'] if r['vol_sell'] > 0 else 0 for r in top_p]
                bars_p = ax4.bar(range(len(labels_p)), values_p, color='orange', alpha=0.7)
                ax4.set_title('Top 10 Presión Compradora (Compra/Venta)')
                ax4.set_xticks(range(len(labels_p)))
                ax4.set_xticklabels(labels_p, rotation=45, ha='right', fontsize=9)
                ax4.grid(True, alpha=0.3)
                
                # Agregar nombres como texto en las barras
                for i, (bar, record) in enumerate(zip(bars_p, top_p)):
                    height = bar.get_height()
                    ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                            f"{record['company_name']}", 
                            ha='center', va='bottom',
                            fontsize=7, rotation=0)

        elif mode == "Todas":
            # 2 gráficas: TODAS las variaciones y TODAS por volumen
            ax1 = fig.add_subplot(1, 2, 1)
            ax2 = fig.add_subplot(1, 2, 2)
            
            # TODAS las variaciones (ordenadas por cambio porcentual)
            sorted_by_change = sorted(all_records, key=lambda r: r['change_pct'], reverse=True)
            
            # Usar nombres completos en las etiquetas del eje X
            labels_all = [f"{r['symbol']}\n{r['company_name']}" for r in sorted_by_change]
            values_all = [r['change_pct'] for r in sorted_by_change]
            colors = ['green' if v >= 0 else 'red' for v in values_all]
            
            bars = ax1.bar(range(len(labels_all)), values_all, color=colors, alpha=0.7)
            ax1.set_title('Todas las Variaciones del Día (%)')
            ax1.set_xticks(range(len(labels_all)))
            ax1.set_xticklabels(labels_all, rotation=45, ha='right', fontsize=6)
            ax1.grid(True, alpha=0.3)
            
            # Agregar línea en 0%
            ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)

            # TODAS por volumen (ordenadas por volumen)
            sorted_by_volume = sorted(all_records, key=lambda r: r['volume'], reverse=True)
            labels_vol = [f"{r['symbol']}\n{r['company_name']}" for r in sorted_by_volume]
            values_vol = [r['volume'] for r in sorted_by_volume]
            
            bars_vol = ax2.bar(range(len(labels_vol)), values_vol, color='blue', alpha=0.7)
            ax2.set_title('Todas por Volumen')
            ax2.set_xticks(range(len(labels_vol)))
            ax2.set_xticklabels(labels_vol, rotation=45, ha='right', fontsize=6)
            ax2.grid(True, alpha=0.3)

        elif mode == "Ticker específico":
            # 4 gráficas específicas del ticker
            ax1 = fig.add_subplot(2, 2, 1)
            ax2 = fig.add_subplot(2, 2, 2)
            ax3 = fig.add_subplot(2, 2, 3)
            ax4 = fig.add_subplot(2, 2, 4)
            
            sel = ticker_var.get().strip().upper()
            if sel:
                ticker_data = next((r for r in all_records if r['symbol'].upper() == sel), None)
                if ticker_data:
                    # 1. Sparkline de precio
                    price_history[sel].append(ticker_data['close'])
                    series = list(price_history[sel])
                    if series:
                        ax1.plot(series, color='purple', linewidth=2)
                        min_v = min(series)
                        max_v = max(series)
                        if min_v == max_v:
                            min_v *= 0.99
                            max_v *= 1.01
                        margin = (max_v - min_v) * 0.05
                        ax1.set_ylim(min_v - margin, max_v + margin)
                        ax1.set_title(f'Sparkline Precio - {sel}')
                        ax1.grid(True, alpha=0.3)

                    # 2. Gráfico de velas (simulado con OHLC actual)
                    ohlc_data = [ticker_data['open'], ticker_data['high'], ticker_data['low'], ticker_data['close']]
                    ohlc_history[sel].append(ohlc_data)
                    if len(ohlc_history[sel]) > 1:
                        # Simular velas con datos históricos
                        times = range(len(ohlc_history[sel]))
                        opens = [d[0] for d in ohlc_history[sel]]
                        highs = [d[1] for d in ohlc_history[sel]]
                        lows = [d[2] for d in ohlc_history[sel]]
                        closes = [d[3] for d in ohlc_history[sel]]
                        
                        ax2.plot(times, opens, 'b-', label='Apertura', alpha=0.7)
                        ax2.plot(times, highs, 'g-', label='Máximo', alpha=0.7)
                        ax2.plot(times, lows, 'r-', label='Mínimo', alpha=0.7)
                        ax2.plot(times, closes, 'k-', label='Cierre', linewidth=2)
                        ax2.set_title(f'OHLC - {sel}')
                        ax2.legend()
                        ax2.grid(True, alpha=0.3)

                    # 3. Precio actual vs Precio promedio ponderado
                    current_price = ticker_data['close']
                    avg_price = ticker_data['avg_price']
                    ax3.bar(['Precio Actual', 'Precio Promedio'], [current_price, avg_price], 
                           color=['blue', 'orange'], alpha=0.7)
                    ax3.set_title(f'Precio Actual vs Promedio - {sel}')
                    ax3.grid(True, alpha=0.3)

                    # 4. Spread compra/venta
                    spread = ticker_data['sell_price_curr'] - ticker_data['buy_price_curr']
                    ax4.bar(['Spread'], [spread], color='red', alpha=0.7)
                    ax4.set_title(f'Spread Compra/Venta - {sel}')
                    ax4.grid(True, alpha=0.3)
                else:
                    for ax in [ax1, ax2, ax3, ax4]:
                        ax.text(0.5, 0.5, f"No hay datos para {sel}", 
                               ha='center', va='center', transform=ax.transAxes)

        fig.tight_layout()
        canvas.draw()
        root.after(update_interval_ms, _update_ui)

    # Bindings para refrescar al cambiar modo/ticker
    def _on_mode_change(event=None):
        _update_ui()
    def _on_ticker_change(event=None):
        if mode_var.get() != "Ticker específico":
            mode_var.set("Ticker específico")
        _update_ui()
    mode_combo.bind('<<ComboboxSelected>>', _on_mode_change)
    ticker_combo.bind('<<ComboboxSelected>>', _on_ticker_change)

    # Primer disparo
    root.after(100, _update_ui)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        try:
            root.destroy()
        except Exception:
            pass


def main():
    # Construcción del menú con secciones
    menu_items = [
        { 'type': 'header', 'text': 'Análisis rápido' },
        { 'type': 'item', 'text': 'Análisis técnico rápido: Algoritmo Pairs Trading con ETFs Apalancados', 'action': pairs_trading_etf_leveraged },
        { 'type': 'item', 'text': 'Análisis técnico rápido: Sugerir ETFs apalancados para compra-venta usando estrategia swing trading por indicadores técnicos', 'action': suggest_enhanced_etf_strategy_leveraged },
        { 'type': 'item', 'text': 'Análisis técnico rápido: Sugerir ETFs para compra-venta usando estrategia swing trading por indicadores técnicos', 'action': suggest_enhanced_etf_strategy },

        { 'type': 'header', 'text': 'Análisis fundamental' },
        { 'type': 'item', 'text': 'Análisis fundamental: Sugerir acciones para seguimiento usando análisis por sector', 'action': fundamental_analysis },

        { 'type': 'header', 'text': 'Análisis preferencias' },
        { 'type': 'item', 'text': 'Análisis preferencias: Sugerir acciones para seguimiento según preferencias', 'action': suggest_stocks_by_preferences },

        { 'type': 'header', 'text': 'Análisis sentimientos' },
        { 'type': 'item', 'text': 'Análisis sentimientos: Sugerir consideraciones sobre acciones en seguimiento según noticias actuales', 'action': news_analysis },

        { 'type': 'header', 'text': 'Análisis técnico' },
        { 'type': 'item', 'text': 'Análisis técnico: Sugerir acciones para compra-venta usando estrategia swing trading por publicación próxima de resultados', 'action': suggest_technical_soon_results },
        { 'type': 'item', 'text': 'Análisis técnico: Sugerir acciones para compra-venta usando estrategia swing trading por indicadores técnicos', 'action': suggest_technical },
        { 'type': 'item', 'text': 'Análisis técnico: Sugerir acciones para compra-venta usando estrategia swing trading de consensos técnicos (web scraping)', 'action': swing_trading_strategy },
        { 'type': 'item', 'text': 'Análisis técnico: Sugerir acciones para compra-venta usando estrategia swing trading con machine learning', 'action': swing_trading_strategy_machine },
        { 'type': 'item', 'text': 'Análisis técnico (Beta 1): Hipótesis de acciones para compra-venta usando estrategia swing trading doble negativo', 'action': suggest_technical_beta1 },
        { 'type': 'item', 'text': 'Análisis técnico (Beta 2): Hipótesis de acciones para compra-venta usando estrategia swing trading positivo-negativo-positivo', 'action': suggest_technical_beta2 },

        { 'type': 'header', 'text': 'Análisis cuantitativo' },
        { 'type': 'item', 'text': 'Análisis cuantitativo: Asignación de Capital Basada en Volatilidad (Gestión de Riesgo Avanzada)', 'action': volatility_based_capital_allocation },
        { 'type': 'item', 'text': 'Análisis cuantitativo: Sugerir distribución de portafolio a partir de optimización de la Razón de Sharpe Ajustada para Corto Plazo', 'action': set_optimizar_portafolio2 },
        { 'type': 'item', 'text': 'Análisis cuantitativo: Sugerir distribución de portafolio a partir de optimización Markowitz', 'action': set_optimizar_portafolio },
        { 'type': 'item', 'text': 'Análisis cuantitativo: Sugerir distribución de portafolio a partir de optimización Litterman', 'action': set_optimizar_portafolio },
        { 'type': 'item', 'text': 'Análisis cuantitativo: Estrategia de Acumulación Intradía (Scaling In) - Entrada Escalonada', 'action': lambda: estrategia_acumulacion_intraday_menu() },
        { 'type': 'item', 'text': 'Análisis cuantitativo: Estrategia de Acumulación ROBUSTA con Múltiples Fuentes de Datos', 'action': lambda: estrategia_acumulacion_intraday_menu() },
        { 'type': 'item', 'text': 'Análisis cuantitativo: Caza de Soportes Intradía - Identificación de Niveles Clave', 'action': lambda: caza_soportes_intraday() },
        { 'type': 'item', 'text': 'Análisis cuantitativo: Reversión a la Media con Bandas de Bollinger - ETFs Apalancados', 'action': lambda: reversion_media_bollinger() },
        { 'type': 'item', 'text': 'Análisis cuantitativo: COMPLETO - Todas las Estrategias Integradas', 'action': lambda: analisis_cuantitativo_completo() },

        { 'type': 'header', 'text': 'Utilidades' },
        { 'type': 'item', 'text': 'Utilidades (Reto Actinver 2025)', 'action': utilidades_actinver_2024 },
        { 'type': 'exit', 'text': 'Salir' },
    ]

    # Índices seleccionables (excluye headers y exit)
    selectable_indices = [i for i, it in enumerate(menu_items) if it['type'] == 'item']
    selected_pos = 0  # posición en selectable_indices

    def display_menu():
        console.clear()
        console.print("[bold blue]Menú de Opciones:[/bold blue]")
        number_map = {}
        current_number = 1
        for i, it in enumerate(menu_items):
            if it['type'] == 'header':
                console.print(it['text'])
            elif it['type'] == 'item':
                is_selected = (selectable_indices[selected_pos] == i)
                prefix = "→ " if is_selected else "   "
                console.print(f"{prefix}{current_number}. {it['text']}")
                number_map[current_number] = i
                current_number += 1
            elif it['type'] == 'exit':
                is_selected = False
                console.print(f"   q. {it['text']}")
        return number_map

    ch = ''
    while ch != 'q':
        number_map = display_menu()
        ch = getch()
        ascii_value = ord(ch)

        if ascii_value == 224:
            ch = getch()
            ascii_value = ord(ch)
            if ascii_value == 72:  # up
                selected_pos = (selected_pos - 1) % len(selectable_indices)
            elif ascii_value == 80:  # down
                selected_pos = (selected_pos + 1) % len(selectable_indices)
        elif ascii_value == 13:  # Enter
            idx = selectable_indices[selected_pos]
            action = menu_items[idx]['action']
            clear_screen()
            action()
            Prompt.ask("[bold blue]Pulsa Enter para continuar...[/bold blue]")
        elif ch == b'q':
            exit_program()
        elif ch == b':':
            opcion_main_menu = Prompt.ask("[bold green] cmd [/bold green]")
            if opcion_main_menu in (':q', ':quit', 'q', 'quit'):
                exit_program()
            if opcion_main_menu in (':b', ':back', 'b', 'back'):
                continue
            try:
                if opcion_main_menu.startswith(':'):
                    opcion_main_menu = opcion_main_menu[1:]
                num = int(opcion_main_menu)
                if num in number_map:
                    idx = number_map[num]
                    # Mover selección a ese índice
                    selected_pos = selectable_indices.index(idx)
                    clear_screen()
                    menu_items[idx]['action']()
                    Prompt.ask("[bold blue]Pulsa Enter para continuar...[/bold blue]")
                else:
                    console.print(f"[bold red]Opción incorrecta...[/bold red]")
                    Prompt.ask("[bold blue]Pulsa Enter para continuar...[/bold blue]")
            except ValueError:
                console.print(f"[bold red]Entrada no válida. Por favor, introduce un número o comando válido...[/bold red]")
                Prompt.ask("[bold blue]Pulsa Enter para continuar...[/bold blue]")

# No olvides actualizar la llamada en tu menú principal
def estrategia_acumulacion_intraday_menu():
    ticker = Prompt.ask("Ticker", default="TECL")
    # Calcular capital óptimo basado en un solo ticker
    optimal_capital = calculate_optimal_capital([ticker])
    capital_total_str = Prompt.ask("Capital total", default=str(optimal_capital))
    capital_total = float(capital_total_str) if capital_total_str else optimal_capital
    num_escalones_str = Prompt.ask("Número de escalones (2-4)", default="3")
    num_escalones = int(num_escalones_str) if num_escalones_str else 3
    
    # Llamar a la nueva función
    estrategia_acumulacion_diaria_estimada(ticker, capital_total, num_escalones)


def estrategia_acumulacion_diaria_estimada(ticker: str, capital_total: float = 10000, num_escalones: int = 3, precio_actual_manual: float | None = None):
    """
    Estrategia de Acumulación con Niveles Estimados a partir de Datos Diarios.
    Utiliza Puntos Pivote para calcular los escalones de entrada.
    """
    console.print(f"[bold blue]📊 Estrategia de Acumulación con Soportes Estimados para {ticker}[/bold blue]")

    df = _get_daily_data_robust(ticker)

    if df is None or len(df) < 2:
        console.print("[red]❌ No hay suficientes datos históricos para calcular los niveles.[/red]")
        return {}

    try:
        # Datos del día anterior para los cálculos
        prev_high = float(df['High'].iloc[-2])
        prev_low = float(df['Low'].iloc[-2])
        prev_close = float(df['Close'].iloc[-2])
        
        # Precio de referencia actual (cierre más reciente)
        precio_actual = float(df['Close'].iloc[-1])

        # --- CÁLCULO DE PUNTOS PIVOTE ---
        pivot = (prev_high + prev_low + prev_close) / 3
        s1 = (2 * pivot) - prev_high
        s2 = pivot - (prev_high - prev_low)
        s3 = prev_low - 2 * (prev_high - pivot)
        
        console.print("[cyan]Niveles de Soporte (Pivotes) calculados para hoy:[/cyan]")
        console.print(f"  S1: ${s1:.2f}, S2: ${s2:.2f}, S3: ${s3:.2f}")

        # Usar los pivotes como nuestros niveles de soporte
        soportes_calculados = sorted([s1, s2, s3], reverse=True)
        
        # Filtrar solo los soportes que están por debajo del precio actual
        soportes = [s for s in soportes_calculados if s < precio_actual]

        if not soportes:
            console.print("[yellow]El precio actual ya está por debajo de todos los soportes pivote. Generando niveles teóricos.[/yellow]")
            soportes = [precio_actual * (1 - (i * 0.015)) for i in range(1, num_escalones + 1)]

        # Distribución de capital
        if num_escalones == 2: porcentajes_escalon = [0.6, 0.4]
        elif num_escalones == 3: porcentajes_escalon = [0.4, 0.35, 0.25]
        else: porcentajes_escalon = [0.3, 0.3, 0.25, 0.15]
        
        estrategia = {
            'ticker': ticker, 'precio_actual': precio_actual, 'capital_total': capital_total,
            'num_escalones': num_escalones, 'escalones': []
        }
        
        for i in range(num_escalones):
            if i < len(soportes):
                precio_entrada = soportes[i]
                condicion = f"Entrada en Soporte Estimado S{i+1}"
            else: # Fallback si no hay suficientes soportes
                precio_entrada = soportes[-1] * (1 - 0.01)
                condicion = "Entrada en nivel de seguridad"

            monto_escalon = capital_total * porcentajes_escalon[i]
            acciones = int(monto_escalon / precio_entrada)
            
            estrategia['escalones'].append({
                'escalon': i + 1, 'precio_entrada': round(precio_entrada, 2),
                'monto': round(monto_escalon, 2), 'acciones': acciones,
                'condicion': condicion,
                'stop_loss': round(soportes_calculados[-1] * 0.98, 2) # Stop loss 2% por debajo del último soporte (S3)
            })
        
        # Mostrar la tabla de resultados
        console.print(f"\n[bold green]🎯 Estrategia de Entrada Escalonada (Estimada) para {ticker}[/bold green]")
        console.print(f"💰 Capital Total: ${capital_total:,.2f}")
        console.print(f"📈 Precio de Referencia Actual: ${precio_actual:.2f}")
        
        table = Table(title="Escalones de Entrada Estimados con Puntos Pivote")
        table.add_column("Escalón", style="cyan"); table.add_column("Precio Entrada", style="green")
        table.add_column("Monto", style="yellow"); table.add_column("Acciones", style="blue")
        table.add_column("Condición", style="magenta"); table.add_column("Stop Loss", style="red")
        
        for escalon in estrategia['escalones']:
            table.add_row(
                str(escalon['escalon']), f"${escalon['precio_entrada']:.2f}",
                f"${escalon['monto']:,.2f}", str(escalon['acciones']),
                escalon['condicion'], f"${escalon['stop_loss']:.2f}"
            )
        
        console.print(table)
        
        monto_total_invertido = sum(e['monto'] for e in estrategia['escalones'])
        if monto_total_invertido > 0:
            precio_promedio = sum(e['precio_entrada'] * e['monto'] for e in estrategia['escalones']) / monto_total_invertido
            console.print(f"\n[bold blue]📊 Precio Promedio Ponderado si se ejecutan todos: ${precio_promedio:.2f}[/bold blue]")
        
        return estrategia
        
    except Exception as e:
        console.print(f"[red]❌ Error al calcular la estrategia para {ticker}: {e}[/red]")
        return {}

# No olvides actualizar la llamada en tu menú principal
def estrategia_acumulacion_intraday_menu():
    ticker = Prompt.ask("Ticker", default="TECL")
    # Calcular capital óptimo basado en un solo ticker
    optimal_capital = calculate_optimal_capital([ticker])
    capital_total_str = Prompt.ask("Capital total", default=str(optimal_capital))
    capital_total = float(capital_total_str) if capital_total_str else optimal_capital
    num_escalones_str = Prompt.ask("Número de escalones (2-4)", default="3")
    num_escalones = int(num_escalones_str) if num_escalones_str else 3
    
    # Llamar a la nueva función
    estrategia_acumulacion_diaria_estimada(ticker, capital_total, num_escalones)


def _get_daily_data_robust(ticker: str):
    """
    Descarga datos DIARIOS de forma robusta - VERSIÓN FINAL CORREGIDA.
    Utiliza un 'period' más largo ("6mo") para máxima fiabilidad, replicando
    la lógica de las funciones que sí obtienen datos.
    """
    required_cols = ['Open', 'High', 'Low', 'Close']
    
    def process_df(df, source_name, convert_to_mxn=False):
        if df is None or df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(0)
        
        df.index = pd.to_datetime(df.index)

        if all(col in df.columns for col in required_cols):
            console.print(f"[green]✅ Éxito con datos de {source_name}.[/green]")
            if convert_to_mxn:
                console.print(f"[blue]🔄 Convirtiendo precios a MXN...[/blue]")
                usd_to_mxn_rate = get_usd_to_mxn_rate()
                for col in required_cols + ['Adj Close']:
                    if col in df.columns: df[col] *= usd_to_mxn_rate
            return df
        return None

    # --- Intento 1: yfinance con Ticker Mexicano (.MX) y período robusto ---
    mx_ticker = normalize_ticker_to_mx(ticker)
    if mx_ticker != "__SKIP_ACTINVER__":
        console.print(f"[dim]Intento 1: Descargando datos diarios para {mx_ticker} (yfinance)...[/dim]")
        # --- CORRECCIÓN CLAVE: Usar "6mo" para máxima fiabilidad ---
        df = yf.download(mx_ticker, period="6mo", interval="1d", progress=False)
        processed_df = process_df(df, mx_ticker)
        if processed_df is not None:
            return processed_df

    # --- Intento 2: yfinance con Ticker de USA y período robusto ---
    usa_ticker = ticker.upper().replace('.MX', '')
    console.print(f"[yellow]⚠️ Fallo con yfinance MX. Intentando con {usa_ticker} (yfinance USA)...[/yellow]")
    # --- CORRECCIÓN CLAVE: Usar "6mo" ---
    df = yf.download(usa_ticker, period="6mo", interval="1d", progress=False)
    processed_df = process_df(df, f"{usa_ticker}", convert_to_mxn=True)
    if processed_df is not None:
        return processed_df

    # --- Intento 3: Fallback a APIs Alternativas (como último recurso) ---
    console.print(f"[bold red]❌ yfinance falló por completo. Intentando con APIs alternativas...[/bold red]")
    df_alternative = get_stock_data_alternative_apis(usa_ticker, period="6mo")
    processed_df = process_df(df_alternative, "APIs Alternativas", convert_to_mxn=True)
    if processed_df is not None:
        return processed_df

    console.print(f"[red]❌ No se pudieron descargar datos para {ticker} desde ninguna fuente.[/red]")
    return None

def estrategia_acumulacion_intraday(ticker: str, capital_total: float = 10000, num_escalones: int = 3):
    """
    Estrategia de Acumulación Intradía (Scaling In) - VERSIÓN CORREGIDA FINAL
    
    Soluciona el error 'ambiguous truth value' asegurando que las columnas
    se procesen como Series de datos simples, incluso si yfinance devuelve un formato complejo.
    """
    console.print(f"[bold blue]📊 Estrategia de Acumulación Intradía para {ticker}[/bold blue]")

    df = _get_intraday_data_robust(ticker)

    if df is None:
        return {}

    try:
        # --- INICIO DE LA CORRECCIÓN CLAVE ---
        # Aplanamos las columnas aquí, justo después de recibir el DataFrame.
        # Esto garantiza que el resto de la función trabaje con un formato simple.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(0)
        
        close_series = df['Close']
        lows = df['Low']
        # --- FIN DE LA CORRECCIÓN CLAVE ---

        precio_actual = float(close_series.iloc[-1])
        
        # Lógica para encontrar soportes
        soportes = []
        for i in range(1, min(len(df) - 2, 48)):
            # Ahora la comparación se hace sobre una Serie simple, lo que evita el error.
            if (lows.iloc[i-1] > lows.iloc[i]) and (lows.iloc[i] < lows.iloc[i+1]):
                soportes.append(float(lows.iloc[i]))
        
        soportes_validos = sorted([s for s in set(soportes) if s < precio_actual], reverse=True)
        soportes = soportes_validos[:num_escalones]
        
        # Si no se encuentran suficientes soportes, se crean niveles basados en porcentajes
        if len(soportes) < num_escalones:
            console.print("[yellow]No se encontraron suficientes soportes técnicos. Creando niveles basados en porcentajes.[/yellow]")
            needed = num_escalones - len(soportes)
            last_support = soportes[-1] if soportes else precio_actual
            for i in range(needed):
                soportes.append(last_support * (1 - (i + 1) * 0.01))

        # Distribución de capital
        if num_escalones == 2: porcentajes_escalon = [0.6, 0.4]
        elif num_escalones == 3: porcentajes_escalon = [0.4, 0.35, 0.25]
        else: porcentajes_escalon = [0.3, 0.3, 0.25, 0.15]
        
        estrategia = {
            'ticker': ticker, 'precio_actual': precio_actual, 'capital_total': capital_total,
            'num_escalones': num_escalones, 'escalones': []
        }
        
        for i in range(num_escalones):
            precio_entrada = soportes[i]
            monto_escalon = capital_total * porcentajes_escalon[i]
            acciones = int(monto_escalon / precio_entrada)
            
            estrategia['escalones'].append({
                'escalon': i + 1, 'precio_entrada': round(precio_entrada, 2),
                'monto': round(monto_escalon, 2), 'acciones': acciones,
                'condicion': f"Entrada en soporte a ${precio_entrada:.2f}",
                'stop_loss': round(soportes[-1] * 0.98, 2) # Stop loss unificado para simplicidad
            })
        
        # Mostrar la tabla de resultados
        console.print(f"\n[bold green]🎯 Estrategia de Entrada Escalonada para {ticker}[/bold green]")
        console.print(f"💰 Capital Total: ${capital_total:,.2f}")
        console.print(f"📈 Precio Actual: ${precio_actual:.2f}")
        
        table = Table(title="Escalones de Entrada")
        table.add_column("Escalón", style="cyan"); table.add_column("Precio Entrada", style="green")
        table.add_column("Monto", style="yellow"); table.add_column("Acciones", style="blue")
        table.add_column("Condición", style="magenta"); table.add_column("Stop Loss", style="red")
        
        for escalon in estrategia['escalones']:
            table.add_row(
                str(escalon['escalon']), f"${escalon['precio_entrada']:.2f}",
                f"${escalon['monto']:,.2f}", str(escalon['acciones']),
                escalon['condicion'], f"${escalon['stop_loss']:.2f}"
            )
        
        console.print(table)
        
        monto_total_invertido = sum(e['monto'] for e in estrategia['escalones'])
        if monto_total_invertido > 0:
            precio_promedio = sum(e['precio_entrada'] * e['monto'] for e in estrategia['escalones']) / monto_total_invertido
            console.print(f"\n[bold blue]📊 Precio Promedio Ponderado si se ejecutan todos: ${precio_promedio:.2f}[/bold blue]")
        
        return estrategia
        
    except Exception as e:
        console.print(f"[red]❌ Error al calcular la estrategia para {ticker}: {e}[/red]")
        return {}


def caza_soportes_intraday(tickers: list = None):
    """
    Técnica de "Caza de Soportes" Intradía - VERSIÓN CORREGIDA
    
    Identifica niveles de soporte relevantes y sugiere órdenes de compra.
    Esta versión maneja correctamente la estructura de datos de yfinance.
    """
    if tickers is None:
        tickers = ['SOXS', 'SPXS', 'SOXL', 'SPXL', 'TQQQ', 'SQQQ']
    
    console.print("[bold blue]🎯 Caza de Soportes Intradía[/bold blue]")
    console.print("[yellow]Identificando niveles de soporte clave para órdenes de compra[/yellow]")
    
    resultados = []
    
    for ticker in tickers:
        try:
            # Descargar datos de los últimos 3 días con intervalos de 1 hora
            mx_ticker = normalize_ticker_to_mx(ticker)
            df = yf.download(mx_ticker, period="3d", interval="1h", progress=False)
            
            if df.empty:
                console.print(f"[yellow]⚠️ No hay datos para {ticker}[/yellow]")
                continue

            # --- INICIO DE LA CORRECCIÓN ---
            # Si las columnas son un MultiIndex, aplánalas para evitar errores.
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(0)
            # --- FIN DE LA CORRECCIÓN ---

            precio_actual = float(df['Close'].iloc[-1])
            
            soportes_detectados = []
            
            # Buscar mínimos locales (pivotes) en las últimas 48 horas
            for i in range(2, min(len(df) - 2, 48)):
                idx = len(df) - i - 1
                
                # Lógica de pivote: un mínimo rodeado por valores más altos
                es_pivote = (df['Low'].iloc[idx-1] > df['Low'].iloc[idx] and 
                             df['Low'].iloc[idx] < df['Low'].iloc[idx+1] and
                             df['Low'].iloc[idx-2] > df['Low'].iloc[idx] and 
                             df['Low'].iloc[idx] < df['Low'].iloc[idx+2])

                if es_pivote:
                    soporte = float(df['Low'].iloc[idx])
                    if soporte < precio_actual:
                        soportes_detectados.append({
                            'precio': soporte,
                            'volumen': float(df['Volume'].iloc[idx]),
                            'timestamp': df.index[idx],
                            'distancia_pct': ((precio_actual - soporte) / precio_actual) * 100
                        })
            
            # Ordenar y filtrar soportes
            soportes_detectados.sort(key=lambda x: x['distancia_pct'])
            soportes_top = soportes_detectados[:3]
            
            # Calcular niveles de Fibonacci
            high_24h = float(df['High'].tail(24).max())
            low_24h = float(df['Low'].tail(24).min())
            rango = high_24h - low_24h
            
            fibonacci_levels = {
                '23.6%': high_24h - (rango * 0.236),
                '38.2%': high_24h - (rango * 0.382),
                '50.0%': high_24h - (rango * 0.500),
                '61.8%': high_24h - (rango * 0.618)
            }
            
            fib_soportes = {k: v for k, v in fibonacci_levels.items() if v < precio_actual}
            
            resultados.append({
                'ticker': ticker, 'precio_actual': precio_actual,
                'soportes_tecnicos': soportes_top, 'soportes_fibonacci': fib_soportes,
                'high_24h': high_24h, 'low_24h': low_24h
            })
            
        except Exception as e:
            console.print(f"[red]❌ Error analizando {ticker}: {e}[/red]")
            continue
    
    # El resto de la función para mostrar los resultados continúa sin cambios...
    for resultado in resultados:
        ticker = resultado['ticker']
        precio_actual = resultado['precio_actual']
        
        console.print(f"\n[bold green]🎯 {ticker} - Precio Actual: ${precio_actual:.2f}[/bold green]")
        
        if resultado['soportes_tecnicos']:
            console.print("[cyan]📊 Soportes Técnicos Detectados:[/cyan]")
            for i, soporte in enumerate(resultado['soportes_tecnicos'], 1):
                console.print(f"  {i}. ${soporte['precio']:.2f} (-{soporte['distancia_pct']:.1f}%) Vol: {soporte['volumen']:,.0f}")
        
        if resultado['soportes_fibonacci']:
            console.print("[magenta]🌀 Soportes Fibonacci:[/magenta]")
            for nivel, precio in resultado['soportes_fibonacci'].items():
                distancia = ((precio_actual - precio) / precio_actual) * 100
                console.print(f"  {nivel}: ${precio:.2f} (-{distancia:.1f}%)")
        
        console.print("[bold yellow]💡 Órdenes de Compra Sugeridas:[/bold yellow]")
        
        todos_soportes = []
        for soporte in resultado['soportes_tecnicos']:
            todos_soportes.append({'precio': soporte['precio'], 'tipo': 'Técnico', 'distancia_pct': soporte['distancia_pct']})
        for nivel, precio in resultado['soportes_fibonacci'].items():
            distancia = ((precio_actual - precio) / precio_actual) * 100
            todos_soportes.append({'precio': precio, 'tipo': f'Fibonacci {nivel}', 'distancia_pct': distancia})
        
        todos_soportes.sort(key=lambda x: x['distancia_pct'])
        mejores_soportes = todos_soportes[:3]
        
        if not mejores_soportes:
            console.print("  No se encontraron soportes claros por debajo del precio actual.")
        else:
            for i, soporte in enumerate(mejores_soportes, 1):
                console.print(f"  Orden {i}: Comprar en ${soporte['precio']:.2f} ({soporte['tipo']}, -{soporte['distancia_pct']:.1f}%)")
    
    return resultados



def reversion_media_bollinger(tickers: list = None, periodo_bollinger: int = 20, desviacion: float = 2.0):
    """
    Estrategia de Trading de Reversión a la Media usando Bandas de Bollinger
    
    Algoritmo mejorado que usa Bandas de Bollinger para identificar condiciones
    de sobreventa/sobrecompra en ETFs apalancados.
    
    Args:
        tickers: Lista de tickers a analizar
        periodo_bollinger: Período para las Bandas de Bollinger (default: 20)
        desviacion: Desviación estándar para las bandas (default: 2.0)
    """
    if tickers is None:
        tickers = ['SOXL', 'SOXS', 'SPXL', 'SPXS', 'TQQQ', 'SQQQ', 'TECL', 'TECS']
    
    console.print("[bold blue]📈 Estrategia de Reversión a la Media - Bandas de Bollinger[/bold blue]")
    console.print(f"[yellow]Período: {periodo_bollinger} días, Desviación: {desviacion}σ[/yellow]")
    
    resultados = []
    señales_compra = []
    señales_venta = []
    
    for ticker in tickers:
        try:
            # Descargar datos de los últimos 3 meses
            mx_ticker = normalize_ticker_to_mx(ticker)
            df = yf.download(mx_ticker, period="3mo", progress=False)
            
            if df.empty or len(df) < periodo_bollinger + 5:
                console.print(f"[yellow]⚠️ Datos insuficientes para {ticker}[/yellow]")
                continue
            
            # Calcular Bandas de Bollinger
            close = df['Close']
            bollinger = BollingerBands(close, window=periodo_bollinger, window_dev=desviacion)
            
            # Obtener las bandas
            bb_upper = bollinger.bollinger_hband()
            bb_middle = bollinger.bollinger_mavg()  # Media móvil (banda media)
            bb_lower = bollinger.bollinger_lband()
            
            # Calcular indicadores adicionales
            rsi = RSIIndicator(close, window=14).rsi()
            
            # Datos actuales
            precio_actual = float(close.iloc[-1])
            bb_upper_actual = float(bb_upper.iloc[-1])
            bb_middle_actual = float(bb_middle.iloc[-1])
            bb_lower_actual = float(bb_lower.iloc[-1])
            rsi_actual = float(rsi.iloc[-1])
            
            # Calcular posición dentro de las bandas (0 = banda inferior, 1 = banda superior)
            bb_position = (precio_actual - bb_lower_actual) / (bb_upper_actual - bb_lower_actual)
            
            # Calcular volatilidad (ancho de las bandas)
            bb_width = ((bb_upper_actual - bb_lower_actual) / bb_middle_actual) * 100
            
            # Detectar toques de bandas en los últimos días
            toque_banda_inferior = any(close.tail(3) <= bb_lower.tail(3))
            toque_banda_superior = any(close.tail(3) >= bb_upper.tail(3))
            
            # Calcular variación reciente
            variacion_1d = ((precio_actual - float(close.iloc[-2])) / float(close.iloc[-2])) * 100
            variacion_5d = ((precio_actual - float(close.iloc[-6])) / float(close.iloc[-6])) * 100 if len(close) >= 6 else 0
            
            # Lógica de señales mejorada
            señal = "MANTENER"
            confianza = 0
            razon = ""
            
            # Señales de COMPRA (Reversión desde sobreventa)
            if (precio_actual <= bb_lower_actual * 1.01 or toque_banda_inferior) and rsi_actual < 35:
                señal = "COMPRAR"
                confianza = 85
                razon = "Precio en banda inferior + RSI sobreventa"
                señales_compra.append(ticker)
            elif bb_position < 0.2 and rsi_actual < 40 and variacion_1d < -2:
                señal = "COMPRAR"
                confianza = 75
                razon = "Posición baja en bandas + caída fuerte"
                señales_compra.append(ticker)
            elif precio_actual < bb_middle_actual and rsi_actual < 30:
                señal = "COMPRAR"
                confianza = 70
                razon = "Por debajo de media + RSI muy bajo"
                señales_compra.append(ticker)
            
            # Señales de VENTA (Reversión desde sobrecompra)
            elif (precio_actual >= bb_upper_actual * 0.99 or toque_banda_superior) and rsi_actual > 65:
                señal = "VENDER"
                confianza = 85
                razon = "Precio en banda superior + RSI sobrecompra"
                señales_venta.append(ticker)
            elif bb_position > 0.8 and rsi_actual > 60 and variacion_1d > 2:
                señal = "VENDER"
                confianza = 75
                razon = "Posición alta en bandas + subida fuerte"
                señales_venta.append(ticker)
            elif precio_actual > bb_middle_actual and rsi_actual > 70:
                señal = "VENDER"
                confianza = 70
                razon = "Por encima de media + RSI muy alto"
                señales_venta.append(ticker)
            
            # Calcular niveles de stop-loss y take-profit
            if señal == "COMPRAR":
                stop_loss = min(bb_lower_actual * 0.98, precio_actual * 0.95)
                take_profit = bb_middle_actual
            elif señal == "VENDER":
                stop_loss = max(bb_upper_actual * 1.02, precio_actual * 1.05)
                take_profit = bb_middle_actual
            else:
                stop_loss = precio_actual * 0.95 if bb_position > 0.5 else precio_actual * 1.05
                take_profit = bb_middle_actual
            
            resultado = {
                'ticker': ticker,
                'precio_actual': precio_actual,
                'bb_upper': bb_upper_actual,
                'bb_middle': bb_middle_actual,
                'bb_lower': bb_lower_actual,
                'bb_position': bb_position,
                'bb_width': bb_width,
                'rsi': rsi_actual,
                'variacion_1d': variacion_1d,
                'variacion_5d': variacion_5d,
                'señal': señal,
                'confianza': confianza,
                'razon': razon,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'toque_banda_inferior': toque_banda_inferior,
                'toque_banda_superior': toque_banda_superior
            }
            
            resultados.append(resultado)
            
        except Exception as e:
            console.print(f"[red]❌ Error analizando {ticker}: {e}[/red]")
            continue
    
    # Mostrar resultados en tabla
    if resultados:
        table = Table(title="Análisis de Reversión a la Media - Bandas de Bollinger")
        table.add_column("Ticker", style="cyan")
        table.add_column("Precio", style="white")
        table.add_column("Posición BB", style="yellow")
        table.add_column("RSI", style="blue")
        table.add_column("Señal", style="bold")
        table.add_column("Confianza", style="green")
        table.add_column("Stop Loss", style="red")
        table.add_column("Take Profit", style="green")
        
        for r in resultados:
            # Color de la señal
            if r['señal'] == "COMPRAR":
                señal_color = "[bold green]COMPRAR[/bold green]"
            elif r['señal'] == "VENDER":
                señal_color = "[bold red]VENDER[/bold red]"
            else:
                señal_color = "[yellow]MANTENER[/yellow]"
            
            table.add_row(
                r['ticker'],
                f"${r['precio_actual']:.2f}",
                f"{r['bb_position']:.2f}",
                f"{r['rsi']:.1f}",
                señal_color,
                f"{r['confianza']}%",
                f"${r['stop_loss']:.2f}",
                f"${r['take_profit']:.2f}"
            )
        
        console.print(table)
        
        # Resumen de señales
        console.print(f"\n[bold green]🟢 Señales de COMPRA ({len(señales_compra)}):[/bold green]")
        for ticker in señales_compra:
            r = next(res for res in resultados if res['ticker'] == ticker)
            console.print(f"  • {ticker}: {r['razon']} (Confianza: {r['confianza']}%)")
        
        console.print(f"\n[bold red]🔴 Señales de VENTA ({len(señales_venta)}):[/bold red]")
        for ticker in señales_venta:
            r = next(res for res in resultados if res['ticker'] == ticker)
            console.print(f"  • {ticker}: {r['razon']} (Confianza: {r['confianza']}%)")
        
        # Guardar resultados
        df_resultados = pd.DataFrame(resultados)
        os.makedirs('data', exist_ok=True)
        csv_file = f'data/reversion_media_bollinger_{datetime.now():%Y%m%d_%H%M%S}.csv'
        df_resultados.to_csv(csv_file, index=False)
        console.print(f"\n[blue]💾 Resultados guardados en: {csv_file}[/blue]")
    
    return resultados


def analisis_cuantitativo_completo(tickers: list = None, capital_total: float = None):
    """
    Análisis Cuantitativo Completo que combina todas las estrategias:
    1. Acumulación Intradía (Scaling In)
    2. Caza de Soportes
    3. Reversión a la Media con Bandas de Bollinger
    
    Args:
        tickers: Lista de tickers a analizar
        capital_total: Capital total disponible para distribución (se calcula automáticamente si no se especifica)
    """
    if tickers is None:
        tickers = ['SOXL', 'SOXS', 'SPXL', 'SPXS', 'TQQQ', 'SQQQ', 'TECL', 'TECS', 'FAS', 'FAZ']
    
    # Calcular capital óptimo si no se especifica
    if capital_total is None:
        capital_total = calculate_optimal_capital(tickers)
    
    console.print("[bold blue]🧮 ANÁLISIS CUANTITATIVO COMPLETO[/bold blue]")
    console.print(f"[yellow]Capital Total: ${capital_total:,.2f} | Tickers: {len(tickers)}[/yellow]")
    console.print("=" * 80)
    
    # 1. Análisis de Reversión a la Media (Bandas de Bollinger)
    console.print("\n[bold cyan]1️⃣ ANÁLISIS DE REVERSIÓN A LA MEDIA[/bold cyan]")
    resultados_bollinger = reversion_media_bollinger(tickers)
    
    # Filtrar solo las señales de compra con alta confianza
    oportunidades_compra = [r for r in resultados_bollinger if r['señal'] == 'COMPRAR' and r['confianza'] >= 70]
    
    if not oportunidades_compra:
        console.print("[yellow]⚠️ No se encontraron oportunidades de compra con alta confianza[/yellow]")
        return
    
    console.print(f"\n[bold green]✅ {len(oportunidades_compra)} oportunidades de compra identificadas[/bold green]")
    
    # 2. Análisis de Soportes para las oportunidades
    console.print("\n[bold cyan]2️⃣ ANÁLISIS DE SOPORTES[/bold cyan]")
    tickers_oportunidad = [r['ticker'] for r in oportunidades_compra]
    resultados_soportes = caza_soportes_intraday(tickers_oportunidad)
    
    # 3. Estrategias de Acumulación Escalonada
    console.print("\n[bold cyan]3️⃣ ESTRATEGIAS DE ACUMULACIÓN ESCALONADA[/bold cyan]")
    
    # Distribuir capital entre las oportunidades respetando el límite máximo
    capital_por_ticker = min(capital_total / len(oportunidades_compra), MAX_ALLOCATION_PER_TICKER)
    
    estrategias_completas = []
    
    for oportunidad in oportunidades_compra:
        ticker = oportunidad['ticker']
        console.print(f"\n[bold magenta]📊 ANÁLISIS COMPLETO: {ticker}[/bold magenta]")
        
        # Crear estrategia de acumulación
        estrategia_acum = estrategia_acumulacion_intraday(ticker, capital_por_ticker, 3)
        
        # Combinar con análisis de soportes
        soporte_info = next((s for s in resultados_soportes if s['ticker'] == ticker), None)
        
        # Crear estrategia completa
        estrategia_completa = {
            'ticker': ticker,
            'señal_bollinger': oportunidad,
            'soportes': soporte_info,
            'acumulacion': estrategia_acum,
            'capital_asignado': capital_por_ticker
        }
        
        # Mostrar recomendación final
        console.print(f"\n[bold yellow]💡 RECOMENDACIÓN FINAL PARA {ticker}:[/bold yellow]")
        console.print(f"🎯 Señal: {oportunidad['señal']} (Confianza: {oportunidad['confianza']}%)")
        console.print(f"💰 Capital Asignado: ${capital_por_ticker:,.2f}")
        console.print(f"📈 Precio Actual: ${oportunidad['precio_actual']:.2f}")
        console.print(f"🛡️ Stop Loss: ${oportunidad['stop_loss']:.2f}")
        console.print(f"🎯 Take Profit: ${oportunidad['take_profit']:.2f}")
        console.print(f"📊 Razón: {oportunidad['razon']}")
        
        if estrategia_acum and 'escalones' in estrategia_acum:
            console.print(f"🔄 Escalones de Entrada: {len(estrategia_acum['escalones'])}")
            for escalon in estrategia_acum['escalones']:
                console.print(f"   • Escalón {escalon['escalon']}: ${escalon['precio_entrada']:.2f} "
                            f"(${escalon['monto']:,.2f})")
        
        estrategias_completas.append(estrategia_completa)
    
    # 4. Resumen Final y Reporte
    console.print("\n[bold blue]📋 RESUMEN EJECUTIVO[/bold blue]")
    console.print("=" * 60)
    
    total_oportunidades = len(estrategias_completas)
    capital_utilizado = total_oportunidades * capital_por_ticker
    
    console.print(f"🎯 Total de Oportunidades: {total_oportunidades}")
    console.print(f"💰 Capital Total Utilizado: ${capital_utilizado:,.2f}")
    console.print(f"📊 Capital por Posición: ${capital_por_ticker:,.2f}")
    
    # Crear tabla resumen
    table = Table(title="Resumen de Estrategias Cuantitativas")
    table.add_column("Ticker", style="cyan")
    table.add_column("Precio", style="white")
    table.add_column("Confianza", style="green")
    table.add_column("Capital", style="yellow")
    table.add_column("Stop Loss", style="red")
    table.add_column("Take Profit", style="green")
    table.add_column("Escalones", style="blue")
    
    for estrategia in estrategias_completas:
        ticker = estrategia['ticker']
        bollinger = estrategia['señal_bollinger']
        acum = estrategia['acumulacion']
        
        num_escalones = len(acum.get('escalones', [])) if acum else 0
        
        table.add_row(
            ticker,
            f"${bollinger['precio_actual']:.2f}",
            f"{bollinger['confianza']}%",
            f"${capital_por_ticker:,.0f}",
            f"${bollinger['stop_loss']:.2f}",
            f"${bollinger['take_profit']:.2f}",
            str(num_escalones)
        )
    
    console.print(table)
    
    # Guardar reporte completo
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('data', exist_ok=True)
    
    # Guardar como JSON para análisis posterior
    reporte_json = {
        'timestamp': timestamp,
        'capital_total': capital_total,
        'capital_por_ticker': capital_por_ticker,
        'total_oportunidades': total_oportunidades,
        'estrategias': estrategias_completas
    }
    
    json_file = f'data/analisis_cuantitativo_completo_{timestamp}.json'
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(reporte_json, f, indent=2, default=str, ensure_ascii=False)
    
    console.print(f"\n[blue]💾 Reporte completo guardado en: {json_file}[/blue]")
    
    return estrategias_completas


def safe_input(prompt: str, default: str = "") -> str:
    """
    Función segura para input() que maneja KeyboardInterrupt elegantemente
    
    Args:
        prompt: Mensaje a mostrar al usuario
        default: Valor por defecto si se presiona Ctrl+C
    
    Returns:
        str: Entrada del usuario o valor por defecto
    """
    try:
        return input(prompt)
    except KeyboardInterrupt:
        console.print(f"\n[yellow]⚠️ Operación cancelada por el usuario[/yellow]")
        return default
    except EOFError:
        console.print(f"\n[yellow]⚠️ Entrada finalizada[/yellow]")
        return default


def exit_program():
    """Función para salir del programa de manera elegante"""
    console.print("\n[bold green]👋 ¡Gracias por usar Hacktinver![/bold green]")
    console.print("[cyan]💡 Recuerda: La inversión inteligente requiere análisis constante[/cyan]")
    sys.exit(0)


# ============================================================================
# FUNCIONES ROBUSTAS DE DESCARGA DE DATOS CON MÚLTIPLES FUENTES
# ============================================================================

@retry(stop_max_attempt_number=3, wait_fixed=2000)  # Reintentar 3 veces con 2 segundos de espera
def fetch_yfinance_data(ticker: str, period: str = "1y", interval: str = "1d"):
    """Descarga datos de yfinance con reintentos."""
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df.empty:
            raise ValueError(f"No se encontraron datos para {ticker}")
        return df
    except Exception as e:
        console.print(f"[yellow]⚠️ Error en yfinance para {ticker}: {e}[/yellow]")
        raise


def fetch_alpha_vantage_data(ticker: str, api_key: str):
    """Descarga datos diarios de Alpha Vantage como respaldo."""
    try:
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&outputsize=full&apikey={api_key}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if "Time Series (Daily)" not in data:
            console.print(f"[yellow]⚠️ No se encontraron datos en Alpha Vantage para {ticker}[/yellow]")
            return None
        
        df = pd.DataFrame(data["Time Series (Daily)"]).T
        df = df.rename(columns={
            "1. open": "Open",
            "2. high": "High",
            "3. low": "Low",
            "4. close": "Close",
            "5. volume": "Volume"
        })
        df.index = pd.to_datetime(df.index)
        df = df.astype(float)
        return df[['Open', 'High', 'Low', 'Close', 'Volume']].sort_index()
    except Exception as e:
        console.print(f"[yellow]⚠️ Error en Alpha Vantage para {ticker}: {e}[/yellow]")
        return None


def _get_daily_data_robust(ticker: str):
    """
    Descarga datos DIARIOS de forma robusta con múltiples fuentes y reintentos.
    """
    required_cols = ['Open', 'High', 'Low', 'Close']
    
    def process_df(df, source_name, convert_to_mxn=False):
        if df is None or df.empty:
            console.print(f"[yellow]⚠️ DataFrame vacío de {source_name}[/yellow]")
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(0)
        
        df.index = pd.to_datetime(df.index)
        
        if not all(col in df.columns for col in required_cols):
            console.print(f"[yellow]⚠️ Columnas faltantes en datos de {source_name}: {df.columns}[/yellow]")
            return None
        
        console.print(f"[green]✅ Éxito con datos de {source_name}.[/green]")
        if convert_to_mxn:
            console.print(f"[blue]🔄 Convirtiendo precios a MXN...[/blue]")
            usd_to_mxn_rate = get_usd_to_mxn_rate()
            for col in required_cols:
                df[col] *= usd_to_mxn_rate
        return df

    # Intento 1: yfinance con Ticker Mexicano (.MX)
    mx_ticker = normalize_ticker_to_mx(ticker)
    if mx_ticker != "__SKIP_ACTINVER__":
        console.print(f"[dim]Intento 1: Descargando datos diarios para {mx_ticker} (yfinance)...[/dim]")
        try:
            df = fetch_yfinance_data(mx_ticker, period="1y", interval="1d")
            processed_df = process_df(df, mx_ticker)
            if processed_df is not None and len(processed_df) >= 2:
                return processed_df
        except Exception as e:
            console.print(f"[yellow]⚠️ Fallo con yfinance MX: {e}. Intentando con {ticker} (yfinance USA)...[/yellow]")

    # Intento 2: yfinance con Ticker de USA
    usa_ticker = ticker.upper().replace('.MX', '')
    try:
        df = fetch_yfinance_data(usa_ticker, period="1y", interval="1d")
        processed_df = process_df(df, usa_ticker, convert_to_mxn=True)
        if processed_df is not None and len(processed_df) >= 2:
            return processed_df
    except Exception as e:
        console.print(f"[yellow]⚠️ Fallo con yfinance USA: {e}. Intentando con APIs alternativas...[/yellow]")

    # Intento 3: Alpha Vantage como respaldo
    console.print(f"[dim]Intento 3: Descargando datos para {usa_ticker} (Alpha Vantage)...[/dim]")
    alpha_vantage_api_key = os.getenv("ALPHA_VANTAGE_API_KEY", "YOUR_API_KEY")  # Reemplaza con tu clave API
    df_alternative = fetch_alpha_vantage_data(usa_ticker, alpha_vantage_api_key)
    processed_df = process_df(df_alternative, "Alpha Vantage", convert_to_mxn=True)
    if processed_df is not None and len(processed_df) >= 2:
        return processed_df

    console.print(f"[red]❌ No se pudieron descargar datos para {ticker} desde ninguna fuente.[/red]")
    return None


def estrategia_acumulacion_diaria_estimada(ticker: str, capital_total: float = 10000, num_escalones: int = 3, precio_actual_manual: float | None = None):
    """
    Estrategia de Acumulación con Niveles Estimados a partir de Datos Diarios.
    Utiliza Puntos Pivote para calcular los escalones de entrada.
    """
    console.print(f"[bold blue]📊 Estrategia de Acumulación con Soportes Estimados para {ticker}[/bold blue]")

    df = _get_daily_data_robust(ticker)

    if df is None or len(df) < 2:
        console.print("[red]❌ No hay suficientes datos históricos para calcular los niveles.[/red]")
        # Fallback: Crear niveles teóricos basados en un precio estimado
        precio_actual = precio_actual_manual if (precio_actual_manual is not None and precio_actual_manual > 0) else 100.0
        if precio_actual_manual is None:
            console.print("[yellow]⚠️ Usando precio ficticio ($100) para niveles teóricos.[/yellow]")
        soportes = [precio_actual * (1 - (i * 0.015)) for i in range(1, num_escalones + 1)]
        precio_actual_fallback = precio_actual
    else:
        # Datos del día anterior para los cálculos
        prev_high = float(df['High'].iloc[-2])
        prev_low = float(df['Low'].iloc[-2])
        prev_close = float(df['Close'].iloc[-2])
        precio_actual = float(df['Close'].iloc[-1])
        if precio_actual_manual is not None and precio_actual_manual > 0:
            precio_actual = float(precio_actual_manual)

        # Cálculo de Puntos Pivote
        pivot = (prev_high + prev_low + prev_close) / 3
        s1 = (2 * pivot) - prev_high
        s2 = pivot - (prev_high - prev_low)
        s3 = prev_low - 2 * (prev_high - pivot)
        
        console.print("[cyan]Niveles de Soporte (Pivotes) calculados para hoy:[/cyan]")
        console.print(f"  S1: ${s1:.2f}, S2: ${s2:.2f}, S3: ${s3:.2f}")

        soportes_calculados = sorted([s1, s2, s3], reverse=True)
        soportes = [s for s in soportes_calculados if s < precio_actual]

        if not soportes:
            console.print("[yellow]El precio actual ya está por debajo de todos los soportes pivote. Generando niveles teóricos.[/yellow]")
            soportes = [precio_actual * (1 - (i * 0.015)) for i in range(1, num_escalones + 1)]
        precio_actual_fallback = precio_actual

    # Distribución de capital
    if num_escalones == 2:
        porcentajes_escalon = [0.6, 0.4]
    elif num_escalones == 3:
        porcentajes_escalon = [0.4, 0.35, 0.25]
    else:
        porcentajes_escalon = [0.3, 0.3, 0.25, 0.15]

    estrategia = {
        'ticker': ticker,
        'precio_actual': precio_actual_fallback,
        'capital_total': capital_total,
        'num_escalones': num_escalones,
        'escalones': []
    }

    for i in range(num_escalones):
        if i < len(soportes):
            precio_entrada = soportes[i]
            condicion = f"Entrada en Soporte Estimado S{i+1}"
        else:
            precio_entrada = soportes[-1] * (1 - 0.01) if soportes else precio_actual_fallback * (1 - (i + 1) * 0.01)
            condicion = "Entrada en nivel de seguridad"

        monto_escalon = capital_total * porcentajes_escalon[i]
        acciones = int(monto_escalon / precio_entrada)
        
        estrategia['escalones'].append({
            'escalon': i + 1,
            'precio_entrada': round(precio_entrada, 2),
            'monto': round(monto_escalon, 2),
            'acciones': acciones,
            'condicion': condicion,
            'stop_loss': round(soportes[-1] * 0.98 if soportes else precio_actual_fallback * 0.95, 2)
        })

    # Mostrar la tabla de resultados
    console.print(f"\n[bold green]🎯 Estrategia de Entrada Escalonada (Estimada) para {ticker}[/bold green]")
    console.print(f"💰 Capital Total: ${capital_total:,.2f}")
    console.print(f"📈 Precio de Referencia Actual: ${precio_actual_fallback:.2f}")

    table = Table(title="Escalones de Entrada Estimados con Puntos Pivote")
    table.add_column("Escalón", style="cyan")
    table.add_column("Precio Entrada", style="green")
    table.add_column("Monto", style="yellow")
    table.add_column("Acciones", style="blue")
    table.add_column("Condición", style="magenta")
    table.add_column("Stop Loss", style="red")

    for escalon in estrategia['escalones']:
        table.add_row(
            str(escalon['escalon']),
            f"${escalon['precio_entrada']:.2f}",
            f"${escalon['monto']:,.2f}",
            str(escalon['acciones']),
            escalon['condicion'],
            f"${escalon['stop_loss']:.2f}"
        )

    console.print(table)

    monto_total_invertido = sum(e['monto'] for e in estrategia['escalones'])
    if monto_total_invertido > 0:
        precio_promedio = sum(e['precio_entrada'] * e['monto'] for e in estrategia['escalones']) / monto_total_invertido
        console.print(f"\n[bold blue]📊 Precio Promedio Ponderado si se ejecutan todos: ${precio_promedio:.2f}[/bold blue]")

    return estrategia


def estrategia_acumulacion_intraday_menu():
    """Menú interactivo para la estrategia de acumulación con datos robustos"""
    ticker = input("Ticker (default='TECL'): ") or "TECL"
    # Calcular capital óptimo basado en un solo ticker
    optimal_capital = calculate_optimal_capital([ticker])
    capital_total_str = input(f"Capital total (default='{optimal_capital}'): ") or str(optimal_capital)
    capital_total = float(capital_total_str)
    num_escalones_str = input("Número de escalones (2-4) (default='3'): ") or "3"
    num_escalones = int(num_escalones_str)
    precio_manual_str = input("Precio actual manual (opcional, Enter para omitir): ") or ""
    precio_manual = float(precio_manual_str) if precio_manual_str.strip() else None
    
    # Llamar a la función
    estrategia_acumulacion_diaria_estimada(ticker, capital_total, num_escalones, precio_manual)


if __name__ == "__main__":
    try:
        console.print("[bold blue]🚀 Iniciando Hacktinver - Sistema de Análisis Cuantitativo[/bold blue]")
        console.print("[yellow]💡 Presiona Ctrl+C en cualquier momento para salir elegantemente[/yellow]")
        main()
    except KeyboardInterrupt:
        console.print(f"\n[yellow]⚠️ Programa interrumpido por el usuario[/yellow]")
        exit_program()
    except Exception as e:
        console.print(f"\n[red]❌ Error inesperado: {e}[/red]")
        console.print("[yellow]💡 El programa se cerrará de manera segura[/yellow]")
        sys.exit(1)


def monitor_señales_tiempo_real(tickers: list = None, intervalo_minutos: int = 15):
    """
    Monitor de señales en tiempo real para las estrategias cuantitativas
    
    Args:
        tickers: Lista de tickers a monitorear
        intervalo_minutos: Intervalo de actualización en minutos
    """
    if tickers is None:
        tickers = ['SOXL', 'SOXS', 'SPXL', 'SPXS', 'TQQQ', 'SQQQ', 'TECL', 'TECS']
    
    console.print(f"[bold blue]🔄 Monitor de Señales en Tiempo Real[/bold blue]")
    console.print(f"[yellow]Monitoreando {len(tickers)} tickers cada {intervalo_minutos} minutos[/yellow]")
    console.print("[dim]Presiona Ctrl+C para detener[/dim]")
    
    señales_anteriores = {}
    
    try:
        while True:
            timestamp = datetime.now().strftime("%H:%M:%S")
            console.print(f"\n[bold cyan]🕐 Actualización: {timestamp}[/bold cyan]")
            
            # Ejecutar análisis de reversión a la media
            resultados = reversion_media_bollinger(tickers)
            
            # Detectar cambios en señales
            señales_nuevas = []
            for resultado in resultados:
                ticker = resultado['ticker']
                señal_actual = resultado['señal']
                confianza = resultado['confianza']
                
                # Verificar si hay cambio de señal
                if ticker in señales_anteriores:
                    if señales_anteriores[ticker] != señal_actual and señal_actual != 'MANTENER':
                        señales_nuevas.append({
                            'ticker': ticker,
                            'señal_anterior': señales_anteriores[ticker],
                            'señal_nueva': señal_actual,
                            'confianza': confianza,
                            'precio': resultado['precio_actual'],
                            'razon': resultado['razon']
                        })
                
                señales_anteriores[ticker] = señal_actual
            
            # Mostrar alertas de nuevas señales
            if señales_nuevas:
                console.print("[bold red]🚨 NUEVAS SEÑALES DETECTADAS:[/bold red]")
                for señal in señales_nuevas:
                    color = "green" if señal['señal_nueva'] == 'COMPRAR' else "red"
                    console.print(f"[{color}]• {señal['ticker']}: {señal['señal_anterior']} → {señal['señal_nueva']} "
                                f"(${señal['precio']:.2f}, {señal['confianza']}%)[/{color}]")
                    console.print(f"  Razón: {señal['razon']}")
                
                # Enviar notificación por Telegram si está configurado
                if bot and telegram_chat_ids:
                    mensaje = f"🚨 NUEVAS SEÑALES DETECTADAS:\n\n"
                    for señal in señales_nuevas:
                        emoji = "🟢" if señal['señal_nueva'] == 'COMPRAR' else "🔴"
                        mensaje += f"{emoji} {señal['ticker']}: {señal['señal_nueva']}\n"
                        mensaje += f"   Precio: ${señal['precio']:.2f}\n"
                        mensaje += f"   Confianza: {señal['confianza']}%\n\n"
                    
                    for chat_id in telegram_chat_ids:
                        try:
                            bot.send_message(chat_id=chat_id, text=mensaje)
                        except Exception as e:
                            console.print(f"[yellow]⚠️ Error enviando Telegram: {e}[/yellow]")
            else:
                console.print("[dim]Sin cambios en señales[/dim]")
            
            # Esperar antes de la siguiente actualización
            console.print(f"[dim]Próxima actualización en {intervalo_minutos} minutos...[/dim]")
            time.sleep(intervalo_minutos * 60)
            
    except KeyboardInterrupt:
        console.print("\n[yellow]Monitor detenido por el usuario[/yellow]")


def _parse_reto_line(line: str):
    parts = [p.strip() for p in line.split('|')]
    if len(parts) < 5:
        return None
    try:
        symbol = parts[1]
        price_now = float(parts[3])
        price_prev = float(parts[4])
    except Exception:
        return None
    change_abs = price_now - price_prev
    change_pct = (change_abs / price_prev * 100.0) if price_prev != 0 else 0.0
    return {
        'symbol': symbol,
        'price_now': price_now,
        'price_prev': price_prev,
        'change_abs': change_abs,
        'change_pct': change_pct,
    }


def monitor_reto_actinver(interval_seconds: int = 3, top_n: int = 15):
    url = 'https://www.retoactinver.com/archivos/datosReto.txt'
    console.print("[bold blue]📡 Monitor de Stocks (Reto Actinver) - Tiempo casi real[/bold blue]")
    console.print("[dim]Presiona Ctrl+C para detener el monitoreo[/dim]")
    os.makedirs('data', exist_ok=True)
    history_path = 'data/reto_history.csv'
    # aseguramos archivo
    try:
        with open(history_path, 'a', encoding='utf-8') as _:
            pass
    except Exception:
        pass

    update_count = 0
    try:
        while True:
            update_count += 1
            now_str = datetime.now().strftime('%H:%M:%S')
            try:
                raw = urlopen(url, timeout=10).read().decode('utf-8', errors='ignore')
                lines = [ln for ln in raw.splitlines() if ln.strip()]
                records = []
                for ln in lines:
                    rec = _parse_reto_line(ln)
                    if rec is not None:
                        records.append(rec)
            except Exception as e:
                console.print(f"[red]Error obteniendo datos del reto: {e}[/red]")
                time.sleep(interval_seconds)
                continue

            if not records:
                console.print("[yellow]No se recibieron registros válidos.[/yellow]")
                time.sleep(interval_seconds)
                continue

            total = len(records)
            winners = sum(1 for r in records if r['change_pct'] > 0)
            losers = sum(1 for r in records if r['change_pct'] < 0)
            flats = total - winners - losers

            top = sorted(records, key=lambda r: r['change_pct'], reverse=True)[:top_n]

            table = Table(title=f"Top {top_n} Stocks - Variaciones del Día ({now_str})")
            table.add_column("Símbolo", style="cyan")
            table.add_column("Precio Actual", style="green")
            table.add_column("Precio Anterior", style="yellow")
            table.add_column("Variación %", style="magenta")
            table.add_column("Variación $", style="blue")
            table.add_column("Estado", style="white")
            for r in top:
                estado = "📈 GANANDO" if r['change_pct'] > 0 else ("📉 PERDIENDO" if r['change_pct'] < 0 else "—")
                table.add_row(
                    r['symbol'],
                    f"${r['price_now']:,.2f}",
                    f"${r['price_prev']:,.2f}",
                    f"{r['change_pct']:+.2f}%",
                    f"${r['change_abs']:+.2f}",
                    estado,
                )

            console.clear()
            console.print(f"Monitor de Stocks - Actualización #{update_count} - {now_str}")
            console.print("Presiona Ctrl+C para detener el monitoreo\n")
            console.print(table)
            console.print(
                f"\nResumen del Mercado ({now_str}):\n"
                f"Total de acciones monitoreadas: {total}\n"
                f"Acciones ganando: {winners} ({winners/total*100:.1f}%)\n"
                f"Acciones perdiendo: {losers} ({losers/total*100:.1f}%)\n"
                f"Acciones sin cambio: {flats} ({flats/total*100:.1f}%)\n"
            )

            # persist snapshot
            try:
                ts = datetime.now().isoformat()
                with open(history_path, 'a', encoding='utf-8', newline='') as f:
                    w = csv.writer(f)
                    for r in records:
                        w.writerow([ts, r['symbol'], r['price_now'], r['price_prev'], r['change_abs'], r['change_pct']])
            except Exception:
                pass

            # mini barras
            console.print("\n[bold]Barras (Top 10 por %):[/bold]")
            for r in top[:10]:
                bar_len = int(max(min(abs(r['change_pct'])/0.5, 50), 0))
                bar = ('#' * bar_len)
                sign = '+' if r['change_pct'] > 0 else '-'
                console.print(f"{r['symbol']:<8} {sign}{bar}")

            console.print(f"\nPróxima actualización en {interval_seconds} segundos... (Actualización #{update_count})")
            time.sleep(interval_seconds)

    except KeyboardInterrupt:
        console.print("[yellow]🛑 Monitoreo detenido por el usuario[/yellow]")


def backtest_estrategia_bollinger(ticker: str, capital_inicial: float = 10000, periodo_dias: int = 90):
    """
    Backtest de la estrategia de Bandas de Bollinger
    
    Args:
        ticker: Ticker a analizar
        capital_inicial: Capital inicial para el backtest
        periodo_dias: Período de días para el backtest
    """
    console.print(f"[bold blue]📊 Backtest Estrategia Bollinger - {ticker}[/bold blue]")
    
    try:
        # Descargar datos históricos
        mx_ticker = normalize_ticker_to_mx(ticker)
        df = yf.download(mx_ticker, period=f"{periodo_dias}d", progress=False)
        
        if df.empty or len(df) < 30:
            console.print(f"[red]❌ Datos insuficientes para {ticker}[/red]")
            return
        
        # Calcular indicadores
        close = df['Close']
        bollinger = BollingerBands(close, window=20, window_dev=2.0)
        bb_upper = bollinger.bollinger_hband()
        bb_lower = bollinger.bollinger_lband()
        bb_middle = bollinger.bollinger_mavg()
        rsi = RSIIndicator(close, window=14).rsi()
        
        # Simular trading
        posicion = 0  # 0 = sin posición, 1 = largo, -1 = corto
        capital = capital_inicial
        acciones = 0
        trades = []
        
        for i in range(20, len(df)):  # Empezar después de que se calculen los indicadores
            precio = close.iloc[i]
            bb_up = bb_upper.iloc[i]
            bb_low = bb_lower.iloc[i]
            bb_mid = bb_middle.iloc[i]
            rsi_val = rsi.iloc[i]
            fecha = df.index[i]
            
            # Calcular posición en bandas
            bb_position = (precio - bb_low) / (bb_up - bb_low) if bb_up != bb_low else 0.5
            
            # Señales de compra (reversión desde sobreventa)
            if posicion == 0 and precio <= bb_low * 1.02 and rsi_val < 35:
                # Comprar
                acciones = int(capital * 0.95 / precio)  # Usar 95% del capital
                if acciones > 0:
                    costo_total = acciones * precio
                    capital -= costo_total
                    posicion = 1
                    trades.append({
                        'fecha': fecha,
                        'tipo': 'COMPRA',
                        'precio': precio,
                        'acciones': acciones,
                        'capital_restante': capital,
                        'valor_posicion': acciones * precio
                    })
            
            # Señales de venta (reversión desde sobrecompra o stop loss)
            elif posicion == 1:
                # Condiciones de venta
                vender = False
                razon = ""
                
                if precio >= bb_up * 0.98 and rsi_val > 65:
                    vender = True
                    razon = "Banda superior + RSI alto"
                elif precio >= bb_mid and bb_position > 0.6:
                    vender = True
                    razon = "Objetivo banda media alcanzado"
                elif precio <= trades[-1]['precio'] * 0.95:  # Stop loss 5%
                    vender = True
                    razon = "Stop loss activado"
                
                if vender:
                    # Vender
                    ingresos = acciones * precio
                    capital += ingresos
                    
                    # Calcular P&L del trade
                    precio_compra = trades[-1]['precio']
                    pnl = (precio - precio_compra) / precio_compra * 100
                    
                    trades.append({
                        'fecha': fecha,
                        'tipo': 'VENTA',
                        'precio': precio,
                        'acciones': acciones,
                        'capital_total': capital,
                        'pnl_pct': pnl,
                        'razon': razon
                    })
                    
                    acciones = 0
                    posicion = 0
        
        # Si quedamos con posición abierta, cerrarla al final
        if posicion == 1:
            precio_final = close.iloc[-1]
            ingresos = acciones * precio_final
            capital += ingresos
            
            precio_compra = trades[-1]['precio']
            pnl = (precio_final - precio_compra) / precio_compra * 100
            
            trades.append({
                'fecha': df.index[-1],
                'tipo': 'VENTA_FINAL',
                'precio': precio_final,
                'acciones': acciones,
                'capital_total': capital,
                'pnl_pct': pnl,
                'razon': "Cierre final"
            })
        
        # Calcular estadísticas
        capital_final = capital
        rendimiento_total = ((capital_final - capital_inicial) / capital_inicial) * 100
        
        # Analizar trades
        trades_compra = [t for t in trades if t['tipo'] == 'COMPRA']
        trades_venta = [t for t in trades if t['tipo'] in ['VENTA', 'VENTA_FINAL']]
        
        trades_ganadores = [t for t in trades_venta if t.get('pnl_pct', 0) > 0]
        trades_perdedores = [t for t in trades_venta if t.get('pnl_pct', 0) <= 0]
        
        # Mostrar resultados
        console.print(f"\n[bold green]📈 RESULTADOS DEL BACKTEST[/bold green]")
        console.print(f"Período: {periodo_dias} días")
        console.print(f"Capital Inicial: ${capital_inicial:,.2f}")
        console.print(f"Capital Final: ${capital_final:,.2f}")
        console.print(f"Rendimiento Total: {rendimiento_total:.2f}%")
        console.print(f"Total de Trades: {len(trades_compra)}")
        console.print(f"Trades Ganadores: {len(trades_ganadores)} ({len(trades_ganadores)/len(trades_venta)*100:.1f}%)")
        console.print(f"Trades Perdedores: {len(trades_perdedores)} ({len(trades_perdedores)/len(trades_venta)*100:.1f}%)")
        
        if trades_ganadores:
            pnl_promedio_ganador = sum(t['pnl_pct'] for t in trades_ganadores) / len(trades_ganadores)
            console.print(f"P&L Promedio Ganador: {pnl_promedio_ganador:.2f}%")
        
        if trades_perdedores:
            pnl_promedio_perdedor = sum(t['pnl_pct'] for t in trades_perdedores) / len(trades_perdedores)
            console.print(f"P&L Promedio Perdedor: {pnl_promedio_perdedor:.2f}%")
        
        # Mostrar últimos trades
        console.print(f"\n[bold cyan]📋 ÚLTIMOS 5 TRADES:[/bold cyan]")
        for trade in trades[-10:]:
            if trade['tipo'] in ['VENTA', 'VENTA_FINAL']:
                color = "green" if trade.get('pnl_pct', 0) > 0 else "red"
                console.print(f"[{color}]{trade['fecha'].strftime('%Y-%m-%d')}: "
                            f"{trade['tipo']} ${trade['precio']:.2f} "
                            f"({trade.get('pnl_pct', 0):.2f}%) - {trade.get('razon', '')}[/{color}]")
        
        # Guardar resultados
        df_trades = pd.DataFrame(trades)
        os.makedirs('data', exist_ok=True)
        csv_file = f'data/backtest_bollinger_{ticker}_{datetime.now():%Y%m%d_%H%M%S}.csv'
        df_trades.to_csv(csv_file, index=False)
        console.print(f"\n[blue]💾 Trades guardados en: {csv_file}[/blue]")
        
        return {
            'capital_inicial': capital_inicial,
            'capital_final': capital_final,
            'rendimiento_pct': rendimiento_total,
            'total_trades': len(trades_compra),
            'trades_ganadores': len(trades_ganadores),
            'trades_perdedores': len(trades_perdedores),
            'trades': trades
        }
        
    except Exception as e:
        console.print(f"[red]❌ Error en backtest: {e}[/red]")
        return None


def optimizar_parametros_bollinger(ticker: str, capital_inicial: float = 10000):
    """
    Optimización de parámetros para las Bandas de Bollinger
    Prueba diferentes combinaciones de período y desviación estándar
    
    Args:
        ticker: Ticker a optimizar
        capital_inicial: Capital inicial para las pruebas
    """
    console.print(f"[bold blue]🔧 Optimización de Parámetros Bollinger - {ticker}[/bold blue]")
    
    # Parámetros a probar
    periodos = [15, 20, 25, 30]
    desviaciones = [1.5, 2.0, 2.5, 3.0]
    
    mejores_resultados = []
    
    try:
        # Descargar datos
        mx_ticker = normalize_ticker_to_mx(ticker)
        df = yf.download(mx_ticker, period="6mo", progress=False)
        
        if df.empty or len(df) < 60:
            console.print(f"[red]❌ Datos insuficientes para {ticker}[/red]")
            return
        
        console.print(f"[cyan]Probando {len(periodos) * len(desviaciones)} combinaciones...[/cyan]")
        
        for periodo in periodos:
            for desviacion in desviaciones:
                try:
                    # Calcular indicadores con parámetros específicos
                    close = df['Close']
                    bollinger = BollingerBands(close, window=periodo, window_dev=desviacion)
                    bb_upper = bollinger.bollinger_hband()
                    bb_lower = bollinger.bollinger_lband()
                    bb_middle = bollinger.bollinger_mavg()
                    rsi = RSIIndicator(close, window=14).rsi()
                    
                    # Simular trading simplificado
                    posicion = 0
                    capital = capital_inicial
                    acciones = 0
                    trades_exitosos = 0
                    total_trades = 0
                    
                    for i in range(periodo + 5, len(df)):
                        precio = close.iloc[i]
                        bb_up = bb_upper.iloc[i]
                        bb_low = bb_lower.iloc[i]
                        bb_mid = bb_middle.iloc[i]
                        rsi_val = rsi.iloc[i]
                        
                        # Señal de compra
                        if posicion == 0 and precio <= bb_low * 1.02 and rsi_val < 35:
                            acciones = int(capital * 0.95 / precio)
                            if acciones > 0:
                                capital -= acciones * precio
                                posicion = 1
                                precio_compra = precio
                                total_trades += 1
                        
                        # Señal de venta
                        elif posicion == 1:
                            if (precio >= bb_up * 0.98 and rsi_val > 65) or precio >= bb_mid:
                                capital += acciones * precio
                                if precio > precio_compra:
                                    trades_exitosos += 1
                                acciones = 0
                                posicion = 0
                    
                    # Cerrar posición final si existe
                    if posicion == 1:
                        capital += acciones * close.iloc[-1]
                        if close.iloc[-1] > precio_compra:
                            trades_exitosos += 1
                    
                    # Calcular métricas
                    rendimiento = ((capital - capital_inicial) / capital_inicial) * 100
                    tasa_exito = (trades_exitosos / total_trades * 100) if total_trades > 0 else 0
                    
                    mejores_resultados.append({
                        'periodo': periodo,
                        'desviacion': desviacion,
                        'rendimiento_pct': rendimiento,
                        'total_trades': total_trades,
                        'tasa_exito_pct': tasa_exito,
                        'capital_final': capital,
                        'score': rendimiento * (tasa_exito / 100) if tasa_exito > 0 else rendimiento
                    })
                    
                except Exception as e:
                    console.print(f"[yellow]⚠️ Error con parámetros {periodo}/{desviacion}: {e}[/yellow]")
                    continue
        
        # Ordenar por score (rendimiento ponderado por tasa de éxito)
        mejores_resultados.sort(key=lambda x: x['score'], reverse=True)
        
        # Mostrar mejores resultados
        console.print(f"\n[bold green]🏆 TOP 5 MEJORES COMBINACIONES:[/bold green]")
        
        table = Table(title=f"Optimización de Parámetros - {ticker}")
        table.add_column("Ranking", style="cyan")
        table.add_column("Período", style="blue")
        table.add_column("Desviación", style="blue")
        table.add_column("Rendimiento", style="green")
        table.add_column("Trades", style="yellow")
        table.add_column("Tasa Éxito", style="magenta")
        table.add_column("Score", style="bold green")
        
        for i, resultado in enumerate(mejores_resultados[:5], 1):
            table.add_row(
                str(i),
                str(resultado['periodo']),
                f"{resultado['desviacion']:.1f}",
                f"{resultado['rendimiento_pct']:.2f}%",
                str(resultado['total_trades']),
                f"{resultado['tasa_exito_pct']:.1f}%",
                f"{resultado['score']:.2f}"
            )
        
        console.print(table)
        
        # Recomendación
        mejor = mejores_resultados[0]
        console.print(f"\n[bold yellow]💡 RECOMENDACIÓN ÓPTIMA:[/bold yellow]")
        console.print(f"Período: {mejor['periodo']} días")
        console.print(f"Desviación: {mejor['desviacion']} σ")
        console.print(f"Rendimiento Esperado: {mejor['rendimiento_pct']:.2f}%")
        console.print(f"Tasa de Éxito: {mejor['tasa_exito_pct']:.1f}%")
        
        # Guardar resultados
        df_resultados = pd.DataFrame(mejores_resultados)
        os.makedirs('data', exist_ok=True)
        csv_file = f'data/optimizacion_bollinger_{ticker}_{datetime.now():%Y%m%d_%H%M%S}.csv'
        df_resultados.to_csv(csv_file, index=False)
        console.print(f"\n[blue]💾 Resultados guardados en: {csv_file}[/blue]")
        
        return mejores_resultados
        
    except Exception as e:
        console.print(f"[red]❌ Error en optimización: {e}[/red]")
        return []


def generar_reporte_semanal_cuantitativo():
    """
    Genera un reporte semanal completo de todas las estrategias cuantitativas
    """
    console.print("[bold blue]📊 REPORTE SEMANAL CUANTITATIVO[/bold blue]")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # ETFs principales para análisis
    etfs_principales = ['SOXL', 'SOXS', 'SPXL', 'SPXS', 'TQQQ', 'SQQQ', 'TECL', 'TECS', 'FAS', 'FAZ']
    
    reporte = {
        'fecha_generacion': datetime.now().isoformat(),
        'periodo_analisis': '1 semana',
        'etfs_analizados': etfs_principales,
        'analisis': {}
    }
    
    try:
        # 1. Análisis de Reversión a la Media
        console.print("\n[cyan]1️⃣ Análisis de Reversión a la Media...[/cyan]")
        resultados_bollinger = reversion_media_bollinger(etfs_principales)
        reporte['analisis']['reversion_media'] = resultados_bollinger
        
        # 2. Caza de Soportes
        console.print("\n[cyan]2️⃣ Análisis de Soportes...[/cyan]")
        resultados_soportes = caza_soportes_intraday(etfs_principales)
        reporte['analisis']['soportes'] = resultados_soportes
        
        # 3. Resumen de oportunidades
        oportunidades_compra = [r for r in resultados_bollinger if r['señal'] == 'COMPRAR']
        oportunidades_venta = [r for r in resultados_bollinger if r['señal'] == 'VENDER']
        
        console.print(f"\n[bold green]📈 RESUMEN SEMANAL:[/bold green]")
        console.print(f"ETFs analizados: {len(etfs_principales)}")
        console.print(f"Oportunidades de COMPRA: {len(oportunidades_compra)}")
        console.print(f"Oportunidades de VENTA: {len(oportunidades_venta)}")
        
        # 4. Top oportunidades
        if oportunidades_compra:
            console.print(f"\n[bold green]🟢 TOP OPORTUNIDADES DE COMPRA:[/bold green]")
            top_compras = sorted(oportunidades_compra, key=lambda x: x['confianza'], reverse=True)[:3]
            for i, opp in enumerate(top_compras, 1):
                console.print(f"{i}. {opp['ticker']}: ${opp['precio_actual']:.2f} "
                            f"(Confianza: {opp['confianza']}%)")
                console.print(f"   {opp['razon']}")
        
        if oportunidades_venta:
            console.print(f"\n[bold red]🔴 TOP OPORTUNIDADES DE VENTA:[/bold red]")
            top_ventas = sorted(oportunidades_venta, key=lambda x: x['confianza'], reverse=True)[:3]
            for i, opp in enumerate(top_ventas, 1):
                console.print(f"{i}. {opp['ticker']}: ${opp['precio_actual']:.2f} "
                            f"(Confianza: {opp['confianza']}%)")
                console.print(f"   {opp['razon']}")
        
        # 5. Guardar reporte completo
        os.makedirs('data', exist_ok=True)
        
        # JSON para análisis programático
        json_file = f'data/reporte_semanal_cuantitativo_{timestamp}.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(reporte, f, indent=2, default=str, ensure_ascii=False)
        
        # CSV para análisis en Excel
        if resultados_bollinger:
            df_bollinger = pd.DataFrame(resultados_bollinger)
            csv_file = f'data/reporte_semanal_bollinger_{timestamp}.csv'
            df_bollinger.to_csv(csv_file, index=False)
            console.print(f"[blue]💾 Reporte Bollinger guardado en: {csv_file}[/blue]")
        
        console.print(f"[blue]💾 Reporte completo guardado en: {json_file}[/blue]")
        
        # 6. Enviar por Telegram si está configurado
        if bot and telegram_chat_ids:
            mensaje = f"📊 REPORTE SEMANAL CUANTITATIVO\n\n"
            mensaje += f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
            mensaje += f"📈 ETFs Analizados: {len(etfs_principales)}\n"
            mensaje += f"🟢 Oportunidades Compra: {len(oportunidades_compra)}\n"
            mensaje += f"🔴 Oportunidades Venta: {len(oportunidades_venta)}\n\n"
            
            if oportunidades_compra:
                mensaje += "🟢 TOP COMPRAS:\n"
                for opp in top_compras:
                    mensaje += f"• {opp['ticker']}: ${opp['precio_actual']:.2f} ({opp['confianza']}%)\n"
            
            if oportunidades_venta:
                mensaje += "\n🔴 TOP VENTAS:\n"
                for opp in top_ventas:
                    mensaje += f"• {opp['ticker']}: ${opp['precio_actual']:.2f} ({opp['confianza']}%)\n"
            
            for chat_id in telegram_chat_ids:
                try:
                    bot.send_message(chat_id=chat_id, text=mensaje)
                    console.print(f"[green]✅ Reporte enviado por Telegram[/green]")
                except Exception as e:
                    console.print(f"[yellow]⚠️ Error enviando Telegram: {e}[/yellow]")
        
        return reporte
        
    except Exception as e:
        console.print(f"[red]❌ Error generando reporte: {e}[/red]")
        return None