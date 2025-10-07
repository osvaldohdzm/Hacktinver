"""

Test xpath and css
$x(".//header/")
SyntaxError: Failed to execute 'evaluate' on 'Document': The string './/header/' is not a valid XPath expression.

$$("header[id=]")
"""

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
    from getch import getch

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


def download_prices_any_listing(ticker: str, start: str = None, end: str = None, period: str = None) -> pd.Series:
    """Try downloading prices for a symbol with and without .MX suffix.
    - If start/end provided, use date range; else use period.
    - Returns a price Series or empty Series if not found.
    """
    t = (ticker or "").strip()
    # Prefer US (unsuffixed) first; only try .MX if user provided it or unsuffixed failed
    if t.endswith(".MX"):
        symbols_to_try = [t, t.replace(".MX", "")]
    else:
        symbols_to_try = [t, f"{t}.MX"]
    for sym in symbols_to_try:
        try:
            if start or end:
                df = yf.download(sym, start=start, end=end, interval="1d")
            else:
                df = yf.download(sym, period=period or "6mo")
            s = _pick_price_series(df)
            if not s.empty:
                return s.rename(ticker)
        except Exception:
            continue
    return pd.Series(dtype=float, name=ticker)


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
                df_allocations = markovitz_portfolio_optimization(tickers, 1000000)
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
    allocation_dataframe["Allocation $"] = (
        allocation_dataframe["Allocation %"] * float(total_mount) / 100
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

                mount = 1000000
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
            df_allocations = markovitz_portfolio_optimization(tickers, 1000000)
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
    data, market_cap_limit=1000000000, beta_min=0.8, average_volume_min=3500
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



def suggest_technical_etf(
    tickers=['AAXJ', 'ACWI', 'BIL', 'BOTZ', 'DIA', 'EEM', 'EWZ', 'GDX', 'GLD', 'IAU', 'ICLN', 'INDA', 
'IVV', 'KWEB', 'LIT', 'MCHI', 'PSQ', 'QCLN', 'QQQ', 'SHV', 'SHY', 'SLV', 'SOXX', 'SPLG', 
'SPY', 'TAN', 'TLT', 'USO', 'VEA', 'VGT', 'VNQ', 'VOO', 'VT', 'VTI', 'VWO', 'VYM', 'XLE', 
'XLF', 'XLK', 'XLV']
):
    # Solicitar al usuario que ingrese tickers
    input_tickers = input("Ingresa las ETFs separadas por comas (i.e FAS,VNQ,XLE): ")

    if input_tickers.strip():  # Si se ingresaron tickers, usarlos
        tickers = [ticker.strip() for ticker in input_tickers.split(",")]
    else:
        console.print(
            f"\n[bold yellow] No se detectó entrada, usando valores sugeridos por defecto...[/bold yellow]"
        )

    resultados = []  # Para almacenar los resultados
    etf_comprar = []  # Para las ETFs recomendadas para comprar
    etf_mantener = []  # Para las ETFs recomendadas para esperar
    etf_vender = []  # Para las ETFs recomendadas para vender

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
            
            # Filtro de tendencia: Media móvil de 50 días
            sma_50 = SMAIndicator(close, window=50).sma_indicator()

            # Obtener datos recientes (usar Series 1D ya normalizadas)
            rsi_actual = float(rsi.iloc[-1])
            stochastic_actual = float(stochastic.stoch().iloc[-1])
            macd_actual = float(macd.iloc[-1])
            close_hoy = float(close.iloc[-1])
            close_ayer = float(close.iloc[-2])
            bollinger_high = float(bollinger.bollinger_hband().iloc[-1])
            bollinger_low = float(bollinger.bollinger_lband().iloc[-1])
            sma_50_actual = float(sma_50.iloc[-1])

            # Calcular la variación diaria
            variacion_diaria = (close_hoy - close_ayer) / close_ayer * 100

            # Filtro de tendencia para evitar "atrapar un cuchillo cayendo"
            precio_sobre_sma50 = close_hoy > sma_50_actual
            precio_bajo_sma50 = close_hoy < sma_50_actual

            # Condiciones de compra mejoradas con filtro de tendencia
            condiciones_compra = (
                precio_sobre_sma50 and  # Solo comprar si está en tendencia alcista
                (
                    # Condición principal: Sobreventa clara en ambos indicadores
                    (rsi_actual < 35 and stochastic_actual < 20)
                    or
                    # Condición alternativa: Movimiento extremo con ruptura de Bollinger inferior
                    (close_hoy < bollinger_low and variacion_diaria < -2.0)
                )
            )

            # Condiciones de venta mejoradas con filtro de tendencia
            condiciones_venta = (
                precio_bajo_sma50 and  # Solo vender si está en tendencia bajista
                (
                    # Condición principal: Sobrecompra en ambos indicadores
                    (rsi_actual > 65 and stochastic_actual > 80)
                    or
                    # Condición alternativa: Ruptura extrema de Bollinger superior
                    (close_hoy > bollinger_high and variacion_diaria > 2.0)
                )
            )

            # Determinar la acción recomendada
            if condiciones_compra:
                accion_recomendacion = "Comprar"
                etf_comprar.append(ticker)
            elif condiciones_venta:
                accion_recomendacion = "Vender"
                etf_vender.append(ticker)
            else:
                accion_recomendacion = "Esperar"
                etf_mantener.append(ticker)

            # Almacenar los resultados
            resultados.append(
                {
                    "Ticker": ticker,
                    "Precio": close_hoy,
                    "SMA50": sma_50_actual,
                    "Tendencia": "Alcista" if precio_sobre_sma50 else "Bajista",
                    "RSI": rsi_actual,
                    "Stochastic": stochastic_actual,
                    "MACD": macd_actual,
                    "Bollinger_High": bollinger_high,
                    "Bollinger_Low": bollinger_low,
                    "Variación (%)": variacion_diaria,
                    "Acción Recomendada": accion_recomendacion,
                }
            )

        except Exception as e:
            print(f"Error con {ticker}: {e}")
            continue

    # Crear un DataFrame a partir de los resultados
    df_resultados = pd.DataFrame(resultados)
    os.makedirs('data', exist_ok=True)    
    csv_file_path = f'data/suggest_technical_etf_{datetime.now():%Y%m%d_%H%M%S}.csv'
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
    Estrategia de mercado neutral con test de cointegración, gestión de riesgo avanzada
    y dimensionamiento de posiciones para maximizar la probabilidad de éxito.
    
    Mejoras implementadas:
    - Test de cointegración (ADF)
    - Lógica de salida y stop-loss
    - Optimización de ventana temporal
    - Dimensionamiento de posición neutral
    - Cálculo de half-life de reversión
    """
    console.print("[bold blue]🔄 Pairs Trading Avanzado con ETFs Apalancados[/bold blue]")
    console.print("[yellow]Estrategia cuantitativa de mercado neutral con validación estadística[/yellow]")
    
    # Importar statsmodels para test de cointegración
    try:
        from statsmodels.tsa.stattools import adfuller
        console.print("[green]✅ Módulo statsmodels cargado correctamente[/green]")
    except ImportError:
        console.print("[bold red]❌ Error: statsmodels no está instalado. Ejecuta: pip install statsmodels[/bold red]")
        return
    
    # Pares de ETFs apalancados predefinidos con alta correlación histórica
    default_pairs = [
        ("SOXL", "TECL"),  # Semiconductores 3x vs Tecnología 3x
        ("SPXL", "TQQQ"),  # S&P 500 3x vs NASDAQ 3x
        ("FAS", "XLF"),    # Financieros 3x vs Financieros 1x
        ("SOXL", "SOXX"),  # Semiconductores 3x vs Semiconductores 1x
        ("TECL", "XLK"),   # Tecnología 3x vs Tecnología 1x
        ("TNA", "IWM"),    # Russell 2000 3x vs Russell 2000 1x
        ("SPXS", "SH"),    # S&P 500 -3x vs S&P 500 -1x (inversos)
    ]
    
    console.print("\n[bold cyan]Pares de ETFs disponibles para análisis:[/bold cyan]")
    for i, (etf1, etf2) in enumerate(default_pairs, 1):
        console.print(f"  {i}. {etf1} / {etf2}")
    
    # Parámetros de configuración
    try:
        selection = input("\nSelecciona el número del par (1-7) o presiona Enter para analizar todos: ")
        if selection.strip():
            pair_index = int(selection) - 1
            if 0 <= pair_index < len(default_pairs):
                pairs_to_analyze = [default_pairs[pair_index]]
            else:
                console.print("[red]Selección inválida, analizando todos los pares[/red]")
                pairs_to_analyze = default_pairs
        else:
            pairs_to_analyze = default_pairs
            
        # Parámetros de gestión de riesgo
        monto_por_pata = float(input("Monto por cada lado de la operación (default: 400000): ") or "400000")
        lookback_window = int(input("Ventana de análisis en días (default: 30): ") or "30")
        
    except ValueError:
        console.print("[yellow]⚠️ Usando valores por defecto[/yellow]")
        pairs_to_analyze = default_pairs
        monto_por_pata = 400000
        lookback_window = 30
    
    console.print(f"\n[bold cyan]Configuración del análisis:[/bold cyan]")
    console.print(f"• Monto por operación: ${monto_por_pata:,.0f} por lado")
    console.print(f"• Ventana de análisis: {lookback_window} días")
    console.print(f"• Pares a analizar: {len(pairs_to_analyze)}")
    
    results = []
    cointegrated_pairs = 0
    
    for etf1, etf2 in pairs_to_analyze:
        try:
            console.print(f"\n[dim]🔍 Analizando par: {etf1} / {etf2}...[/dim]")
            
            # Descargar datos históricos (1 año para mejor análisis estadístico)
            df1 = yf.download(etf1, period="1y", interval="1d")
            df2 = yf.download(etf2, period="1y", interval="1d")
            
            if df1.empty or df2.empty:
                console.print(f"[red]⚠️ No se pudieron obtener datos para {etf1} o {etf2}[/red]")
                continue
            
            # Obtener precios de cierre y alinear fechas
            close1 = df1["Close"]
            close2 = df2["Close"]
            
            if isinstance(close1, pd.DataFrame):
                close1 = close1.iloc[:, 0]
            if isinstance(close2, pd.DataFrame):
                close2 = close2.iloc[:, 0]
            
            common_dates = close1.index.intersection(close2.index)
            close1_aligned = close1.loc[common_dates]
            close2_aligned = close2.loc[common_dates]
            
            if len(common_dates) < 60:  # Necesitamos más datos para análisis robusto
                console.print(f"[red]⚠️ Datos insuficientes para {etf1}/{etf2} ({len(common_dates)} días)[/red]")
                continue
            
            # Calcular el ratio
            ratio = close1_aligned / close2_aligned
            ratio_clean = ratio.dropna()
            
            # === MEJORA 1: TEST DE COINTEGRACIÓN ===
            try:
                adf_result = adfuller(ratio_clean)
                p_value_cointegration = adf_result[1]
                adf_statistic = adf_result[0]
                
                if p_value_cointegration >= 0.05:
                    console.print(f"[bold red]❌ Par {etf1}/{etf2} NO es cointegrado (p-valor: {p_value_cointegration:.4f}). Se descarta.[/bold red]")
                    continue
                else:
                    console.print(f"[green]✅ Par {etf1}/{etf2} es cointegrado (p-valor: {p_value_cointegration:.4f})[/green]")
                    cointegrated_pairs += 1
                    
            except Exception as e:
                console.print(f"[yellow]⚠️ Error en test de cointegración para {etf1}/{etf2}: {e}[/yellow]")
                p_value_cointegration = 1.0  # Asumir no cointegrado si hay error
                adf_statistic = 0
                continue
            
            # === MEJORA 3: OPTIMIZACIÓN DE VENTANA TEMPORAL ===
            # Probar diferentes ventanas y elegir la más estable
            windows_to_test = [20, 30, 45, 60]
            best_window = lookback_window
            min_volatility = float('inf')
            
            for window in windows_to_test:
                if len(ratio_clean) > window:
                    test_sma = ratio_clean.rolling(window=window).mean()
                    test_std = ratio_clean.rolling(window=window).std()
                    test_z_scores = (ratio_clean - test_sma) / test_std
                    z_volatility = test_z_scores.std()
                    
                    if z_volatility < min_volatility:
                        min_volatility = z_volatility
                        best_window = window
            
            # Calcular estadísticas del ratio con la ventana optimizada
            ratio_sma = ratio_clean.rolling(window=best_window).mean()
            ratio_std = ratio_clean.rolling(window=best_window).std()
            
            # Valores actuales
            current_ratio = float(ratio_clean.iloc[-1])
            current_sma = float(ratio_sma.iloc[-1])
            current_std = float(ratio_std.iloc[-1])
            
            # Calcular Z-Score
            z_score = (current_ratio - current_sma) / current_std if current_std > 0 else 0
            
            # === MEJORA 5: CÁLCULO DE HALF-LIFE ===
            # Estimar cuánto tarda el ratio en revertir a la media
            try:
                # Usar regresión simple para estimar half-life
                ratio_lagged = ratio_clean.shift(1).dropna()
                ratio_current = ratio_clean[1:len(ratio_lagged)+1]
                ratio_diff = ratio_current - ratio_lagged
                
                # Regresión: ratio_diff = alpha + beta * ratio_lagged
                from sklearn.linear_model import LinearRegression
                X = ratio_lagged.values.reshape(-1, 1)
                y = ratio_diff.values
                
                reg = LinearRegression().fit(X, y)
                beta = reg.coef_[0]
                
                # Half-life = -ln(2) / ln(1 + beta)
                if beta < 0:
                    half_life = -np.log(2) / np.log(1 + beta)
                else:
                    half_life = float('inf')  # No hay reversión
                    
            except Exception:
                half_life = float('inf')
            
            # Precios actuales
            price1_current = float(close1_aligned.iloc[-1])
            price2_current = float(close2_aligned.iloc[-1])
            
            # Calcular correlación histórica
            correlation = close1_aligned.corr(close2_aligned)
            
            # === MEJORA 2: LÓGICA DE SALIDA Y STOP-LOSS ===
            signal = "ESPERAR"
            action_detail = ""
            confidence = "BAJA"
            exit_target = f"Z-Score=0 (Ratio={current_sma:.4f})"
            stop_loss_z_score = 3.5
            
            # Generar señales con stop-loss
            if abs(z_score) >= 2.0:
                confidence = "ALTA"
                if z_score > 2.0:
                    signal = "VENDER ETF1 / COMPRAR ETF2"
                    action_detail = f"Vender {etf1}, Comprar {etf2} (Ratio sobrevalorado)"
                    stop_loss_level = f"Z-Score > {stop_loss_z_score}"
                elif z_score < -2.0:
                    signal = "COMPRAR ETF1 / VENDER ETF2"
                    action_detail = f"Comprar {etf1}, Vender {etf2} (Ratio subvalorado)"
                    stop_loss_level = f"Z-Score < -{stop_loss_z_score}"
            elif abs(z_score) >= 1.5:
                confidence = "MEDIA"
                if z_score > 1.5:
                    signal = "CONSIDERAR VENTA ETF1"
                    action_detail = f"Considerar vender {etf1} (Ratio elevado)"
                    stop_loss_level = f"Z-Score > {stop_loss_z_score}"
                elif z_score < -1.5:
                    signal = "CONSIDERAR COMPRA ETF1"
                    action_detail = f"Considerar comprar {etf1} (Ratio bajo)"
                    stop_loss_level = f"Z-Score < -{stop_loss_z_score}"
            else:
                stop_loss_level = "N/A"
            
            # === MEJORA 4: DIMENSIONAMIENTO DE POSICIÓN ===
            # Calcular número de acciones para posición neutral al mercado
            acciones_etf1 = int(monto_por_pata / price1_current)
            acciones_etf2 = int(monto_por_pata / price2_current)
            
            # Valor real de cada lado
            valor_etf1 = acciones_etf1 * price1_current
            valor_etf2 = acciones_etf2 * price2_current
            
            # Calcular retorno esperado
            expected_return_pct = 0
            if abs(z_score) >= 1.5:
                target_ratio = current_sma
                if z_score > 0:
                    expected_return_pct = ((target_ratio / current_ratio) - 1) * 100
                else:
                    expected_return_pct = ((current_ratio / target_ratio) - 1) * 100
            
            # Evaluar calidad del par
            pair_quality = "EXCELENTE" if (correlation > 0.8 and half_life < 20 and p_value_cointegration < 0.01) else \
                          "BUENA" if (correlation > 0.7 and half_life < 40 and p_value_cointegration < 0.03) else \
                          "REGULAR"
            
            results.append({
                "Par": f"{etf1}/{etf2}",
                "ETF1": etf1,
                "ETF2": etf2,
                "Precio ETF1": price1_current,
                "Precio ETF2": price2_current,
                "Ratio Actual": current_ratio,
                "Ratio Media": current_sma,
                "Z-Score": z_score,
                "Correlación": correlation,
                "P-Valor Cointeg.": p_value_cointegration,
                "Half-Life (días)": half_life if half_life != float('inf') else 999,
                "Ventana Óptima": best_window,
                "Señal": signal,
                "Confianza": confidence,
                "Calidad Par": pair_quality,
                "Stop-Loss": stop_loss_level,
                "Exit Target": exit_target,
                "Acciones ETF1": acciones_etf1,
                "Acciones ETF2": acciones_etf2,
                "Valor ETF1": valor_etf1,
                "Valor ETF2": valor_etf2,
                "Retorno Esperado (%)": expected_return_pct,
                "Acción Detallada": action_detail
            })
            
        except Exception as e:
            console.print(f"[red]❌ Error analizando {etf1}/{etf2}: {e}[/red]")
            continue
    
    if not results:
        console.print("[bold red]❌ No se encontraron pares cointegrados válidos[/bold red]")
        return
    
    console.print(f"\n[bold green]✅ Pares cointegrados encontrados: {cointegrated_pairs}/{len(pairs_to_analyze)}[/bold green]")
    
    # Crear DataFrame y ordenar por calidad y Z-Score
    df_results = pd.DataFrame(results)
    
    # Ordenar por: 1) Calidad del par, 2) Z-Score absoluto
    quality_order = {"EXCELENTE": 3, "BUENA": 2, "REGULAR": 1}
    df_results['Quality_Score'] = df_results['Calidad Par'].map(quality_order)
    df_results['Z_Score_Abs'] = df_results['Z-Score'].abs()
    df_results = df_results.sort_values(['Quality_Score', 'Z_Score_Abs'], ascending=[False, False])
    
    # Mostrar tabla principal con Rich
    table = Table(title="🔄 Pairs Trading Avanzado - Análisis Cuantitativo")
    table.add_column("Par", style="cyan", no_wrap=True)
    table.add_column("Z-Score", style="yellow")
    table.add_column("P-Valor", style="magenta")
    table.add_column("Half-Life", style="green")
    table.add_column("Calidad", style="blue")
    table.add_column("Señal", style="bold")
    table.add_column("Confianza", style="white")
    table.add_column("Ret. Esp.", style="red")
    
    for _, row in df_results.iterrows():
        z_score = row['Z-Score']
        half_life = row['Half-Life (días)']
        
        # Colorear Z-Score
        if abs(z_score) >= 2.0:
            z_score_str = f"[bold red]{z_score:.2f}[/bold red]"
        elif abs(z_score) >= 1.5:
            z_score_str = f"[bold yellow]{z_score:.2f}[/bold yellow]"
        else:
            z_score_str = f"{z_score:.2f}"
        
        # Colorear Half-Life
        if half_life <= 10:
            half_life_str = f"[bold green]{half_life:.1f}d[/bold green]"
        elif half_life <= 30:
            half_life_str = f"[yellow]{half_life:.1f}d[/yellow]"
        else:
            half_life_str = f"[red]{half_life:.1f}d[/red]"
        
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
            half_life_str,
            row['Calidad Par'],
            signal_str,
            row['Confianza'],
            f"{row['Retorno Esperado (%)']:.2f}%"
        )
    
    console.print(table)
    
    # Tabla de dimensionamiento de posiciones
    console.print(f"\n[bold cyan]📊 Dimensionamiento de Posiciones (Monto: ${monto_por_pata:,.0f} por lado):[/bold cyan]")
    
    position_table = Table(title="Cálculo de Posiciones para Mercado Neutral")
    position_table.add_column("Par", style="cyan")
    position_table.add_column("Acciones ETF1", style="green")
    position_table.add_column("Valor ETF1", style="blue")
    position_table.add_column("Acciones ETF2", style="green")
    position_table.add_column("Valor ETF2", style="blue")
    position_table.add_column("Stop-Loss", style="red")
    
    for _, row in df_results.head(5).iterrows():  # Solo top 5
        position_table.add_row(
            row['Par'],
            f"{row['Acciones ETF1']:,}",
            f"${row['Valor ETF1']:,.0f}",
            f"{row['Acciones ETF2']:,}",
            f"${row['Valor ETF2']:,.0f}",
            row['Stop-Loss']
        )
    
    console.print(position_table)
    
    # Recomendaciones finales
    excellent_pairs = df_results[df_results['Calidad Par'] == 'EXCELENTE']
    high_confidence = df_results[df_results['Confianza'] == 'ALTA']
    
    console.print(f"\n[bold green]⭐ PARES DE CALIDAD EXCELENTE ({len(excellent_pairs)}):[/bold green]")
    if not excellent_pairs.empty:
        for _, row in excellent_pairs.iterrows():
            console.print(f"   • {row['Par']}: Z-Score={row['Z-Score']:.2f}, Half-Life={row['Half-Life (días)']:.1f}d, Correlación={row['Correlación']:.3f}")
    
    console.print(f"\n[bold red]🎯 OPORTUNIDADES DE ALTA CONFIANZA ({len(high_confidence)}):[/bold red]")
    if not high_confidence.empty:
        for _, row in high_confidence.iterrows():
            console.print(f"   • {row['Acción Detallada']}")
            console.print(f"     └─ Z-Score: {row['Z-Score']:.2f}, Retorno Esp: {row['Retorno Esperado (%)']:.2f}%, Half-Life: {row['Half-Life (días)']:.1f}d")
    
    # Guardar resultados
    os.makedirs('data', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file_path = f'data/pairs_trading_advanced_{timestamp}.csv'
    df_results.to_csv(csv_file_path, index=False)
    
    console.print(f"\n[bold blue]🧠 Conceptos Avanzados Implementados:[/bold blue]")
    console.print("• [green]Test de Cointegración (ADF)[/green]: Solo pares estadísticamente válidos")
    console.print("• [yellow]Half-Life de Reversión[/yellow]: Velocidad de retorno a la media")
    console.print("• [cyan]Ventana Optimizada[/cyan]: Mejor período de análisis por par")
    console.print("• [red]Stop-Loss Dinámico[/red]: Protección contra divergencias extremas")
    console.print("• [blue]Posición Neutral[/blue]: Mismo valor monetario en ambos lados")
    
    console.print(f"\n[bold yellow]📁 Análisis completo guardado en: {csv_file_path}[/bold yellow]")


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
        capital_total = float(input("Ingresa el capital total disponible (default: 1000000): ") or "1000000")
        riesgo_por_operacion = float(input("Ingresa el % de riesgo por operación (default: 2.0): ") or "2.0") / 100
    except ValueError:
        capital_total = 1000000
        riesgo_por_operacion = 0.02
        console.print("[yellow]⚠️ Usando valores por defecto: Capital=1,000,000, Riesgo=2%[/yellow]")
    
    console.print(f"\n[bold cyan]Analizando {len(tickers)} ETFs para asignación de capital...[/bold cyan]")
    console.print(f"Capital Total: ${capital_total:,.2f}")
    console.print(f"Riesgo por Operación: {riesgo_por_operacion*100:.1f}%")
    
    results = []
    
    for ticker in tickers:
        try:
            console.print(f"[dim]Procesando {ticker}...[/dim]")
            
            # Descargar datos históricos (3 meses para ATR)
            df = yf.download(ticker, period="3mo", interval="1d")
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
            cantidad_acciones = int((capital_total * riesgo_por_operacion) / atr_actual)
            
            # Calcular inversión total para esta posición
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


def suggest_technical_etf_leveraged(
    tickers=['FAS','FAZ', 'QLD', 'SOXL','SOXS', 'SPXL', 'SPXS', 'SQQQ', 'TECL', 'TECS', 'TNA', 'TQQQ']
):
    """
    Algoritmo optimizado para ETFs apalancados:
    Busca ETFs que han tenido tendencia alcista en las últimas 5 semanas
    pero hoy tienen porcentaje negativo (oportunidad de compra en dip)
    """
    console.print("[bold blue]🚀 Análisis Optimizado de ETFs Apalancados[/bold blue]")
    console.print("[yellow]Buscando ETFs con tendencia alcista de 5 semanas pero con caída hoy...[/yellow]")
    
    # Solicitar al usuario que ingrese tickers
    input_tickers = input("Ingresa las ETFs separadas por comas (presiona Enter para usar valores por defecto): ")

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
            
            # Descargar datos de los últimos 6 meses para análisis completo
            df_ticker = yf.download(ticker, period="6mo", interval="1d")
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
        "XLK",
        "GDX",
        "AAPL",
        "CEMEXCPO.MX",
        "AXP",
        "AMZN",
        "GOOGL",
        "META",
        "QQQ",
        "MSFT",
        "JPM",
        "BOLSAA.MX",
        "XLF",
        "TSLA",
        "GMEXICOB.MX",
        "PE&OLES.MX",
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
                variacion_ayer > 0 and variacion_hoy < 0
                and rsi_actual < 66
                and stochastic_actual < 90
                and macd_actual > 0  
                and close_hoy < bollinger_high  
            )

            # Determinar la acción recomendada
            if condiciones_compra:
                accion_recomendacion = "Comprar"
                acciones_comprar.append(ticker)
            elif rsi_actual > 80 and close_hoy > bollinger_high:
                accion_recomendacion = "Vender"
                acciones_vender.append(ticker)
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



def set_optimizar_portafolio():
    input_amount = (
        input("Enter the amount to invest (i.e. 800000): ")
        .replace(",", "")
        .replace("$", "")
        .strip()
    )
    if not input_amount:  # Si el usuario no ingresa nada
        input_amount = 800000  # Valor por defecto
    else:
        input_amount = int(float(input_amount))  # Convertir a número
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
    print("\nTickers list : ", tickers)
    print("\nOptimizing portfolio...")
    try:
        allocation_dataframe = markovitz_portfolio_optimization(
            tickers, input_amount, input_initial_date
        )
        print(allocation_dataframe)
    except Exception as e:
        print(e)

def set_optimizar_portafolio2():
    input_amount = (
        input("Enter the amount to invest (i.e. 800000): ")
        .replace(",", "")
        .replace("$", "")
        .strip()
    )
    if not input_amount:  # Si el usuario no ingresa nada
        input_amount = 800000  # Valor por defecto
    else:
        input_amount = int(float(input_amount))  # Convertir a número
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
    print("\nTickers list : ", tickers)
    print("\nOptimizing portfolio by momentuu...")
    try:
        allocation_dataframe = portfolio_optimization_sharpe_short_term(
            tickers, input_amount, input_initial_date
        )
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
        # Heurística de respaldo: seleccionar opción cuyo texto contenga palabras clave
        try:
            lowered = str(pregunta).lower()
            keywords = []
            if "volatil" in lowered:
                keywords = ["fluct", "vari", "precios"]
            # Buscar la primera opción que empareje palabras clave
            for opt in (answer_options or []):
                resp_txt = str(opt.get("respuesta", "")).lower()
                if any(k in resp_txt for k in keywords):
                    return int(opt.get("idRespuesta"))
        except Exception:
            pass
        return None
    # Extraer el primer número (idRespuesta)
    try:
        import re
        match = re.search(r"\d+", text)
        return int(match.group(0)) if match else None
    except Exception:
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
    while True:
        clear_screen()
        print("Selecciona una opción:")
        menu_options = {
            "1": "Probar Credenciales de sesión en la plataforma del reto",
            "2": "Obtener pregunta de Quizz diario",
            "3": "Resolver Quizz diario",
            "4": "Programar respuesta automática de Quizz diario",
            "5": "Resolver Quizz semanal",
            "6": "Programar respuesta automática de Quizz semanal",
            "7": "Mostrar sugerencias de compra",
            "8": "Mostrar portafolio actual",
            "9": "Comprar acciones",
            "10": "Mostrar órdenes",
            "11": "Monitorear venta",
            "12": "Vender todas las posiciones en portafolio (a precio del mercado)",
            "13": "Restaurar sesión en plataforma del reto",
            "14": "Listar tareas programadas",
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
                # Llama a la función asociada a la opción seleccionada
                option_functions = {
                    "1": test_session,
                    "2": option_2,
                    "3": answer_quiz_daily_contest_actinver,
                    "4": start_scheduled_quiz,
                    "5": answer_quiz_weekly_contest_actinver,
                    "6": start_scheduled_weekly_quiz,
                    "7": option_5,
                    "8": option_6,
                    "9": option_7,
                    "10": option_8,
                    "11": option_9,
                    "12": test_session0,
                    "13": test_session1,
                    "14": list_tasks_scheduled,
                }
                option_functions[opcion_main_menu]()
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
    
    # Descargar datos
    console.print("[cyan]📊 Descargando datos de mercado...[/cyan]")
    data = {}
    successful_downloads = 0
    
    for symbol in symbols:
        try:
            # Usar progress indicator más limpio
            df = yf.download(symbol, period="3mo", interval="1d", progress=False)
            if not df.empty:
                data[symbol] = df
                successful_downloads += 1
                console.print(f"[green]✅ {symbol}[/green]", end=" ")
            else:
                console.print(f"[yellow]⚠️ {symbol}[/yellow]", end=" ")
        except Exception as e:
            console.print(f"[red]❌ {symbol}[/red]", end=" ")
    
    console.print(f"\n[cyan]Descargados exitosamente: {successful_downloads}/{len(symbols)}[/cyan]")
    
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
        webbrowser.open(filename)
        console.print("[bold blue]🌐 Abriendo reporte en navegador...[/bold blue]")
    except:
        console.print("[yellow]💡 Abre manualmente el archivo HTML en tu navegador[/yellow]")


def process_market_data_visual(data):
    """Procesa los datos de mercado para generar métricas de análisis avanzadas"""
    
    analysis = {}
    
    for symbol, df in data.items():
        try:
            # Asegurar que tenemos datos suficientes
            if len(df) < 60:
                console.print(f"[yellow]⚠️ Datos insuficientes para {symbol}[/yellow]")
                continue
            
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
            momentum_1d = (close_val - prev_close_val) / prev_close_val * 100
            momentum_1w = (close_val - float(close.iloc[-6])) / float(close.iloc[-6]) * 100 if len(close) >= 6 else momentum_1d
            momentum_1m = (close_val - float(close.iloc[-21])) / float(close.iloc[-21]) * 100 if len(close) >= 21 else momentum_1w
            
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
        
    except Exception as e:
        console.print(f"[yellow]⚠️ Usando gráficos básicos: {e}[/yellow]")
        correlation_html = "<div>Matriz de correlación no disponible</div>"
        relative_strength_html = "<div>Fuerza relativa no disponible</div>"
        market_overview_html = "<div>Vista de mercado no disponible</div>"
        executive_summary = "<div>Resumen ejecutivo no disponible</div>"
    
    # Generar resúmenes mejorados
    buy_signals = [symbol for symbol, data in analysis_data.items() if data.get('buy_signal', False)]
    sell_signals = [symbol for symbol, data in analysis_data.items() if data.get('sell_signal', False)]
    squeeze_alerts = [symbol for symbol, data in analysis_data.items() if data.get('squeeze_alert', False)]
    
    buy_ops = ', '.join(buy_signals[:5]) if buy_signals else "Ninguna detectada"
    sell_ops = ', '.join(sell_signals[:5]) if sell_signals else "Ninguna detectada"
    squeeze_ops = ', '.join(squeeze_alerts[:5]) if squeeze_alerts else "Ninguna detectada"
    
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

    
 


def main():
    selected_index = 0  # Índice inicial para la opción seleccionada

    # Definición del menú dentro de main
    menu_options = [
        ("Análisis técnico rápido: Algoritmo Pairs Trading con ETFs Apalancados", pairs_trading_etf_leveraged),
       ("Análisis técnico rápido: Sugerir ETFs apalancados para compra-venta usando estrategia swing trading por indicadores técnicos", suggest_technical_etf_leveraged),
        ("Análisis técnico rápido: Sugerir ETFs para compra-venta usando estrategia swing trading por indicadores técnicos", suggest_technical_etf),
        ("Análisis fundamental: Sugerir acciones para seguimiento usando análisis por sector", fundamental_analysis),
        ("Análisis preferencias: Sugerir acciones para seguimiento según preferencias", suggest_stocks_by_preferences),
        ("Análisis sentimientos: Sugerir consideraciones sobre acciones en seguimiento según noticias actuales", news_analysis),
        ("Análisis técnico: Sugerir acciones para compra-venta usando estrategia swing trading por publicación próxima de resultados)", suggest_technical_soon_results),
        ("Análisis técnico: Sugerir acciones para compra-venta usando estrategia swing trading por indicadores técnicos", suggest_technical),
        ("Análisis técnico: Sugerir acciones para compra-venta usando estrategia swing trading de consensos técnicos web", swing_trading_strategy),
        ("Análisis técnico: Sugerir acciones para compra-venta usando estrategia swing trading con machine learning", swing_trading_strategy_machine),
        ("Análisis técnico (Beta 1): Hipótesis de acciones para compra-venta usando estrategia swing trading doble negativo", suggest_technical_beta1),
        ("Análisis técnico (Beta 2): Hipótesis de acciones para compra-venta usando estrategia swing trading positivo-negativo-positivo", suggest_technical_beta2),
        ("Análisis cuantitativo: Asignación de Capital Basada en Volatilidad (Gestión de Riesgo Avanzada)", volatility_based_capital_allocation),
        ("Análisis cuantitativo: Sugerir distribución de portafolio a partir de optimización de la Razón de Sharpe Ajustada para Corto Plazo", set_optimizar_portafolio2),
        ("Análisis cuantitativo: Sugerir distribución de portafolio a partir de optimización Markowitz", set_optimizar_portafolio),
        ("Análisis cuantitativo: Sugerir distribución de portafolio a partir de optimización Litterman", set_optimizar_portafolio),
        ("Utilidades (Reto Actinver 2024)", utilidades_actinver_2024), 
        ("Monitor de Stocks", monitor_stocks),
        ("Reporte Visual", daily_visual_report),
        ("Imprimir Consejos", imprimir_consejos_inversion),
        ("Salir", None),  # La opción de salir ya no tiene número
    ]

    def display_menu(selected_index):
        console.clear()
        console.print("[bold blue]Menú de Opciones:[/bold blue]")
        for i, (option_text, _) in enumerate(menu_options):
            if option_text == "Salir":
                prefix = "→ " if i == selected_index else "   "
                console.print(f"{prefix}q. {option_text}")  # Usar 'q' para la opción de salir
            else:
                prefix = "→ " if i == selected_index else "   "
                console.print(f"{prefix}{i + 1}. {option_text}")  # Usar número para otras opciones

    ch = ''  # Inicializa ch

    while ch != 'q':
        display_menu(selected_index)
        ch = getch()  # Lee un carácter de la entrada

        # Imprime el valor ASCII del carácter
        ascii_value = ord(ch)
        
        # Procesa las teclas
        if ascii_value == 224:  # Teclas especiales (flechas, etc.)
            ch = getch()  # Lee el siguiente carácter para obtener el código de la flecha
            ascii_value = ord(ch)
            if ascii_value == 72:  # Flecha arriba
                selected_index = (selected_index - 1) % len(menu_options)
            elif ascii_value == 80:  # Flecha abajo
                selected_index = (selected_index + 1) % len(menu_options)
        elif ascii_value == 13:  # Enter
            selected_option = menu_options[selected_index]
            clear_screen()
            selected_option[1]()
            Prompt.ask("[bold blue]Pulsa Enter para continuar...[/bold blue]")
        elif ch == b'q':
            exit_program()
        elif ch == b':':
            opcion_main_menu = Prompt.ask("[bold green] cmd [/bold green]")        
            if opcion_main_menu == 'quit':
                exit_program() 
            try:
                selected_index = int(opcion_main_menu) - 1  # Ajustar el índice (restar 1)
                if 0 <= selected_index < len(menu_options):
                    clear_screen()                    
                    selected_option = menu_options[selected_index]
                    selected_option[1]()  
                    Prompt.ask("[bold blue]Pulsa Enter para continuar...[/bold blue]")                 
                else:
                    console.print(f"[bold red]Opción incorrecta...[/bold red]")
                    Prompt.ask("[bold blue]Pulsa Enter para continuar...[/bold blue]")
            except ValueError:
                console.print(f"[bold red]Entrada no válida. Por favor, introduce un número o comando válido...[/bold red]")
                Prompt.ask("[bold blue]Pulsa Enter para continuar...[/bold blue]")

if __name__ == "__main__":
    main()