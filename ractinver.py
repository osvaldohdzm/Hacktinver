"""

Test xpath and css
$x(".//header/")
SyntaxError: Failed to execute 'evaluate' on 'Document': The string './/header/' is not a valid XPath expression.

$$("header[id=]")
"""
import argparse
import csv
import datetime
import emoji
import html5lib
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path
import pandas as pd
import pandas.io.formats.style
import pandas_datareader.data as web
import pdfkit
import pytest
import requests
import schedule
import sys
import telebot # Importamos las librería
import telegram # python-telegram-bot
import threading
import time
import warnings
import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.optimize as sco
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains


from bs4 import BeautifulSoup
from bs4 import BeautifulSoup  #del módulo bs4, necesitamos BeautifulSoup
from datetime import datetime
from datetime import timedelta
from decimal import Decimal
from matplotlib import style
from matplotlib.backends.backend_pdf import PdfPages
from pandas import Series, DataFrame
from pandas.plotting import scatter_matrix
from pathlib import Path
from re import sub
from selenium import webdriver
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support.wait import WebDriverWait
from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from telebot import types
from urllib.request import urlopen, Request
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

warnings.simplefilter(action='ignore', category=FutureWarning)
TOKEN = '1967618042:AAG2sfJp5iTUCqirtW8txriCaVkanum1QNU' # Ponemos nuestro Token generado con el @BotFather
bot = telegram.Bot(token=TOKEN)
tb = telebot.TeleBot(TOKEN) # Combinamos la declaración del Token con la función de la API

parser = argparse.ArgumentParser(description='Trading tools actinver')
parser.add_argument("-cmd", help='file JSON name')
parser.add_argument("-tid", help='telegram ids to send messages', type=str)
args = vars(parser.parse_args())
command = args['cmd']

print(args)

telegram_chat_ids = []
if args["tid"] is not None:
    args["tid"] = [s.strip() for s in args["tid"].split(",")]


telegram_chat_ids = args["tid"]
print(telegram_chat_ids)

pd.options.mode.chained_assignment = None  # default='warn'

WARNING = '\033[93m'
WHITE = '\033[0m'
OKCYAN = '\033[96m'
OKGREEN = '\033[92m'
ERROR = '\033[91m'




comision_per_operation = 0.0010
iva_per_operation = 0.16
cost_per_operation = (comision_per_operation)+(comision_per_operation*iva_per_operation)
dn = os.path.dirname(os.path.realpath(__file__)) 


def configure_firefox_driver_no_profile():
    options = Options()
    # Sin abrirnavegador
    options.headless = False
    options.add_argument('start-maximized')
    driver = webdriver.Firefox(options=options, executable_path=os.path.join(dn,"geckodriver.exe"))
    driver.get('about:home')
    driver.maximize_window()
    return driver

def configure_firefox_driver_with_profile():
    options = Options()
    # Sin abrirnavegador
    options.headless = False
    options.add_argument('start-maximized')
    profile = webdriver.FirefoxProfile(r"C:\Users\osval\AppData\Roaming\Mozilla\Firefox\Profiles\5uyx2cbw.default-release")
    driver = webdriver.Firefox(firefox_profile=profile,options=options, executable_path=os.path.join(dn,"geckodriver.exe"))
    driver.get('about:home')
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
        print('\t' + WARNING + "Conectando con la plataforma del reto actinver..." + WHITE)
        driver.get('https://www.retoactinver.com/RetoActinver/#/inicio')
        close_popoup_browser(driver)
        print('\t' + WARNING + "Cerrando sesión en plataforma del reto..."+ WHITE)
        WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, 'mat-list-item.as:nth-child(11) > div:nth-child(1) > div:nth-child(2) > p:nth-child(2)')))
        driver.find_element(By.CSS_SELECTOR, "mat-list-item.as:nth-child(11) > div:nth-child(1) > div:nth-child(2) > p:nth-child(2)").click()
        print('\t' + WARNING + "Cierre de sesión exitoso! :)" + WHITE)
    except Exception as e: 
        print(e)
        exit()



def close_popoup_browser(driver):
    try:
        WebDriverWait(driver, 15).until(EC.element_to_be_clickable((By.CSS_SELECTOR, '#mat-dialog-1 > app-modal-explorador:nth-child(1) > mat-dialog-actions:nth-child(2) > div:nth-child(2) > button:nth-child(1)')))
        driver.find_element_by_css_selector( '#mat-dialog-1 > app-modal-explorador:nth-child(1) > mat-dialog-actions:nth-child(2) > div:nth-child(2) > button:nth-child(1)').click()
        driver.find_element_by_css_selector( '#mat-dialog-1 > app-modal-explorador:nth-child(1) > mat-dialog-actions:nth-child(2) > div:nth-child(2) > button:nth-child(1)').click()
    except Exception as e: 
        print(e)

    try:
        WebDriverWait(driver, 15).until(EC.element_to_be_clickable((By.XPATH, '//*[@id="botonCerrar"]')))
        driver.find_element_by_xpath( '//*[@id="botonCerrar"]').click()
        driver.find_element_by_xpath( '//*[@id="botonCerrar"]').click()
    except Exception as e: 
        print(e)



        

def show_orders():
    try:
        print('\t' + WARNING + "Conectando con la plataforma del reto actinver..." + WHITE)
        driver.get('https://www.retoactinver.com/RetoActinver/#/ordenes')

        close_popoup_browser()

        total_value = driver.find_element_by_css_selector("div.text-left:nth-child(1) > p:nth-child(2)").get_attribute("innerHTML")
        power_value = driver.find_element_by_css_selector("div.text-left:nth-child(2) > p:nth-child(2)").get_attribute("innerHTML")
        inv_value = driver.find_element_by_css_selector("div.text-left:nth-child(3) > p:nth-child(2)").get_attribute("innerHTML")
        varia_percent_value = driver.find_element_by_css_selector("div.col-md-12:nth-child(4) > p:nth-child(2)").get_attribute("innerHTML")
        varia_mount_value = driver.find_element_by_css_selector("div.col-md-12:nth-child(5) > p:nth-child(2)").get_attribute("innerHTML")
        print('Valuación Total: '+ total_value)
        print('Poder de compra: '+ power_value)
        print('Inversiones: '+ inv_value)
        print('Variación en porcentaje: '+ varia_percent_value)
        print('Variación en pesos: '+ varia_mount_value)

        current_power_value = float(sub(r'[^\d.]', '', power_value))
        print('Poder de compra actual: '+ str(current_power_value))
    except Exception as e: 
        print(e)
        exit()


def show_portfolio():
    try:
        print('\t' + WARNING + "Conectando con la plataforma del reto actinver..." + WHITE)
        driver.get('https://www.retoactinver.com/RetoActinver/#/inicio')
        close_popoup_browser()

        total_value = driver.find_element_by_css_selector("div.text-left:nth-child(1) > p:nth-child(2)").get_attribute("innerHTML")
        power_value = driver.find_element_by_css_selector("div.text-left:nth-child(2) > p:nth-child(2)").get_attribute("innerHTML")
        inv_value = driver.find_element_by_css_selector("div.text-left:nth-child(3) > p:nth-child(2)").get_attribute("innerHTML")
        varia_percent_value = driver.find_element_by_css_selector("div.col-md-12:nth-child(4) > p:nth-child(2)").get_attribute("innerHTML")
        varia_mount_value = driver.find_element_by_css_selector("div.col-md-12:nth-child(5) > p:nth-child(2)").get_attribute("innerHTML")
        current_power_value = float(sub(r'[^\d.]', '', power_value))

        print('Valuación Total: '+ total_value)
        print('Poder de compra: '+ str("${:,.2f}".format(current_power_value)))
        print('Inversiones: '+ inv_value)
        print('Variación en porcentaje: '+ varia_percent_value)
        print('Variación en pesos: '+ varia_mount_value)

       
        driver.get('https://www.retoactinver.com/RetoActinver/#/portafolio')
        time.sleep(3)
        rows = driver.find_elements(By.XPATH, '//*[@id="tpm"]/app-portafolio/div/div[3]/mat-card/app-table/div/gt-column-settings/div/div/generic-table/table/tbody/tr/td/span[2]')

        count = 1
        print("\nPosiciones actuales:")
        for x in rows:
            if count % 8 != 0:
                print(x.get_attribute('innerHTML').ljust(15), end = '')
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
        print('\t' + WARNING + "Conectando con la plataforma del reto actinver..." + WHITE)
        driver.get('https://www.retoactinver.com/RetoActinver/#/inicio')

        close_popoup_browser()

        power_value = driver.find_element_by_css_selector("div.text-left:nth-child(2) > p:nth-child(2)").get_attribute("innerHTML")
        current_power_value = float(sub(r'[^\d.]', '', power_value))
    except Exception as e: 
        print(e)

    current_stock_buy = input("Escribe el símbolo del stock que quieres comprar >> ")
    if current_stock_buy:
        current_stock_buy = current_stock_buy.upper()
    else:
        input("Empty input\nPulsa una tecla para continuar")
        buy_stocks()

    try:
        print('\t' + WARNING + "Conectando con la plataforma del reto actinver..." + WHITE)
        driver.get('https://www.retoactinver.com/RetoActinver/#/capitales')

        close_popoup_browser()

        WebDriverWait(driver, 3).until(EC.element_to_be_clickable((By.CSS_SELECTOR, '.ng-pristine')))
        search_stock_input = driver.find_element_by_css_selector('.ng-pristine')
        search_stock_input.send_keys(current_stock_buy)

        if check_exists_by_css_selector('.gt-no-matching-results'):
            print('\t' + ERROR + "No hay stock con ese simbolo en el listado del reto! Prueba con otro" + WHITE)
            pass
        else:



            WebDriverWait(driver, 50).until(EC.element_to_be_clickable((By.XPATH, '//*/generic-table/table/tbody/tr/td[4]/span[2]')))
            price = driver.find_element_by_xpath('//*/generic-table/table/tbody/tr/td[4]/span[2]').get_attribute('innerHTML')
            price = float(sub(r'[^\d.]', '',price )) 

            
            
            WebDriverWait(driver, 50).until(EC.element_to_be_clickable((By.CSS_SELECTOR, 'tr.ng-star-inserted:nth-child(1) > td:nth-child(2) > span:nth-child(2)')))
            driver.find_element_by_css_selector('tr.ng-star-inserted:nth-child(1) > td:nth-child(2) > span:nth-child(2)').click()
            
            no_titles = round((current_power_value * 0.25 ) / price) # the percentage of portfolio start 0.25 then 0.30 then 0.5
            print(str(price)) 
            print(str(current_power_value))
            print(str(no_titles)) 
            no_titles = str(no_titles)
    
            selected_stock_name = driver.find_element_by_css_selector('.NombreEmpresa').get_attribute('innerHTML')
    
            confirmation = input("El Símbolo seleccionado es: " + selected_stock_name + ", deseas continuar? (y/n) >> ")
    
            if confirmation == 'y':
                WebDriverWait(driver, 3).until(EC.element_to_be_clickable((By.CSS_SELECTOR, 'div.col-6:nth-child(1) > button:nth-child(1)')))
                buy_button = driver.find_element_by_css_selector('div.col-6:nth-child(1) > button:nth-child(1)')
                buy_button.click()
    
                WebDriverWait(driver, 3).until(EC.element_to_be_clickable((By.CSS_SELECTOR, '#mat-radio-9 > label:nth-child(1)')))
                driver.find_element_by_css_selector('#mat-radio-9 > label:nth-child(1)').click()

                WebDriverWait(driver, 6).until(EC.element_to_be_clickable((By.CSS_SELECTOR, 'input.ng-invalid')))
                driver.find_element_by_css_selector('input.ng-invalid').send_keys(no_titles)
                
                confirm_button = driver.find_element_by_css_selector('div.col-md-6:nth-child(2) > button:nth-child(1)')
                confirm_button.click()
    
                WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.CSS_SELECTOR, 'table.w-100 > tbody:nth-child(1) > tr:nth-child(1) > td:nth-child(2)')))
                print("\n---Verifica la orden---")
                print("Emisora: "+ driver.find_element_by_css_selector('table.w-100 > tbody:nth-child(1) > tr:nth-child(1) > td:nth-child(2)').get_attribute('innerHTML'))
                print("Operación: "+ driver.find_element_by_css_selector('table.w-100 > tbody:nth-child(1) > tr:nth-child(2) > td:nth-child(2)').get_attribute('innerHTML'))
                print("Tipo Orden: "+ driver.find_element_by_css_selector('table.w-100 > tbody:nth-child(1) > tr:nth-child(3) > td:nth-child(2)').get_attribute('innerHTML'))
                print("Títulos: "+ driver.find_element_by_css_selector('table.w-100 > tbody:nth-child(1) > tr:nth-child(4) > td:nth-child(2)').get_attribute('innerHTML'))
                print("Precio a mercado: "+ driver.find_element_by_css_selector('table.w-100 > tbody:nth-child(1) > tr:nth-child(5) > td:nth-child(2)').get_attribute('innerHTML'))
                print("Precio límite: "+ driver.find_element_by_css_selector('table.w-100 > tbody:nth-child(1) > tr:nth-child(6) > td:nth-child(2)').get_attribute('innerHTML'))
                print("Comisión: "+ driver.find_element_by_css_selector('table.w-100 > tbody:nth-child(1) > tr:nth-child(7) > td:nth-child(2)').get_attribute('innerHTML'))
                print("IVA: "+ driver.find_element_by_css_selector('table.w-100 > tbody:nth-child(1) > tr:nth-child(8) > td:nth-child(2)').get_attribute('innerHTML'))
                print("Importe total: "+ driver.find_element_by_css_selector('table.w-100 > tbody:nth-child(1) > tr:nth-child(9) > td:nth-child(2)').get_attribute('innerHTML'))
                print("Vigencía: "+ driver.find_element_by_css_selector('table.w-100 > tbody:nth-child(1) > tr:nth-child(10) > td:nth-child(2)').get_attribute('innerHTML'))
                print("Fecha de captura: "+ driver.find_element_by_css_selector('table.w-100 > tbody:nth-child(1) > tr:nth-child(11) > td:nth-child(2)').get_attribute('innerHTML'))
                print("Fecha de postura: "+ driver.find_element_by_css_selector('table.w-100 > tbody:nth-child(1) > tr:nth-child(12) > td:nth-child(2)').get_attribute('innerHTML'))
                
                operation_confirm = input("Deseas confirmar la operación? (y/n) >> ")
                if operation_confirm == 'y':
                    WebDriverWait(driver, 3).until(EC.element_to_be_clickable((By.CSS_SELECTOR, ' div.col-md-6:nth-child(2) > button:nth-child(1)')))
                    driver.find_element_by_css_selector('div.col-md-6:nth-child(2) > button:nth-child(1)').click()
                    print('\t' + WARNING + "Operación efectuada" + WHITE)
                elif operation_confirm == 'n':
                    pass
                else:
                    input("Entrada no valida \nPulsa una tecla para continuar")
                    buy_stocks()
    
            elif confirmation == 'n':
                print('\t' + WARNING + "Orden cancelada" + WHITE)
            else:
                buy_stocks()
    
    except Exception as e: 
        print(e)



def login_platform_investing(driver):
    print('\t' + WARNING + "Iniciando sesión en investing.com..." + WHITE)
    driver.get("https://mx.investing.com/")
    is_logged_flag = False

    try:
       WebDriverWait(driver, 3).until(EC.element_to_be_clickable((By.CSS_SELECTOR, '.allow-notifications-popup-close-button')))
       driver.find_element(By.CSS_SELECTOR, '.allow-notifications-popup-close-button').click()   
    except Exception as e: 
        print(e) 
    try:
       WebDriverWait(driver, 3).until(EC.element_to_be_clickable((By.CSS_SELECTOR, '.myAccount')))
       user_tag = driver.find_element_by_css_selector('.myAccount').get_attribute("innerText")
       if (user_tag=="Osvaldo"):
          is_logged_flag = True
          print('\t' + WARNING + "Sesión ya iniciada con anterioridad" + WHITE)
          return is_logged_flag
    except Exception as e: 
        print(e) 
          
    try:
         WebDriverWait(driver, 50).until(EC.element_to_be_clickable((By.CSS_SELECTOR, ".login")))
         driver.find_element(By.CSS_SELECTOR, ".login").click()                        
    except Exception as e: 
         print(e)
         WebDriverWait(driver, 50).until(EC.element_to_be_clickable((By.CSS_SELECTOR, '.generalOverlay')))
         driver.find_element(By.CSS_SELECTOR, '.popupCloseIcon').click()     
         driver.find_element(By.LINK_TEXT, "Iniciar sesión").click()    
     
    driver.find_element(By.ID, "loginFormUser_email").send_keys("osvaldo.hdz.m@outlook.com")
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
       WebDriverWait(driver, 3).until(EC.element_to_be_clickable((By.CSS_SELECTOR, '.myAccount')))
       user_tag = driver.find_element_by_css_selector('.myAccount').get_attribute("innerText")
       if (user_tag=="Osvaldo"):
          is_logged_flag = True
          print('\t' + WARNING + "Sesión iniciada con exito" + WHITE)
          return is_logged_flag
    except Exception as e: 
        print(e) 

    return is_logged_flag

def login_actinver():
    try:
        print('\t' + WARNING + "Iniciando sesión..." + WHITE)
        # driver.get('https://www.retoactinver.com/RetoActinver/#/login')
        driver.refresh()
        time.sleep(3)
        driver.find_element_by_xpath(
            '//*[@id="botonCerrar"]/mat-icon').click()
        user_input = driver.find_element_by_id('mat-input-0')
        user_input.send_keys(USERNAME)
        password_input = driver.find_element_by_id('mat-input-1')
        password_input.send_keys(PASSWORD)
        login_button = driver.find_element_by_xpath(
            "/html/body/app-root/block-ui/app-login/div/form/button[1]/span")
        login_button.click()
    except:
        reconect_session()





def login_platform_actinver(driver,username,password,email):
    print("Logging with {}".format(username))
    is_logged_web_string = ""
    try:
        print('\t' + WARNING + "Accediendo a la plataforma del reto actinver..." + WHITE)
        driver.get('https://www.retoactinver.com/RetoActinver/#/login')
        close_popoup_browser(driver)

        print('\t' + WARNING + "Iniciando sesión en plataforma del reto...")
        WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.ID, 'mat-input-0'))).send_keys(username)
        WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.ID, 'mat-input-1'))).send_keys(password)
        WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.ID, 'mat-input-1'))).send_keys(Keys.RETURN)
        
        try:
            
            is_logged_web_string = WebDriverWait(driver, 20).until(EC.element_to_be_clickable((By.CSS_SELECTOR, 'mat-list-item.as:nth-child(1) > div:nth-child(1) > div:nth-child(2) > p:nth-child(2)'))).get_attribute("innerText")
            print(is_logged_web_string)
            if (is_logged_web_string=="Dashboard"):
               print('\t' + WARNING + "Inicio de sesión exitoso! :)" + WHITE)
               return True
        except Exception as e: 
            print("Error iniciando en la plataforma {}".format(e))  


        try:
            print('\t' + WARNING + "Posible sesión iniciada con anterioridad, intentando reestablecer sesión..." + WHITE)
            WebDriverWait(driver, 20).until(EC.element_to_be_clickable((By.CSS_SELECTOR, 'button.btn-stroke-alternativo:nth-child(1)'))).click()
            WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, '#mat-input-2'))).send_keys(username)  
            WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, '#mat-input-3'))).send_keys(email)  
            WebDriverWait(driver, 15).until(EC.element_to_be_clickable((By.CSS_SELECTOR, 'button.btn-block:nth-child(1)')))
            WebDriverWait(driver, 15).until(EC.element_to_be_clickable((By.XPATH, '/html/body/div[1]/div[2]/div/mat-dialog-container/app-destroy-session/mat-dialog-actions/div/button'))).click()
            driver.refresh()            
            close_popoup_browser(driver)

            user_input = driver.find_element_by_id('mat-input-0')
            user_input.send_keys(username)
            password_input = driver.find_element_by_id('mat-input-1')
            password_input.send_keys(password)
            login_button = driver.find_element_by_xpath("/html/body/app-root/block-ui/app-login/div/div/div[1]/form/button[1]")
            login_button.click()
            print('\t' + WARNING + "Inicio de sesión exitoso! :)" + WHITE) 
            try:
               is_logged_web_string = WebDriverWait(driver, 20).until(EC.element_to_be_clickable((By.CSS_SELECTOR, 'mat-list-item.as:nth-child(1) > div:nth-child(1) > div:nth-child(2) > p:nth-child(2)'))).get_attribute("innerText")
               print(is_logged_web_string)
               if (is_logged_web_string=="Dashboard"):
                   print('\t' + WARNING + "Inicio de sesión exitoso! :)" + WHITE)
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
        print('\t' + WARNING + "Accediendo a datos de tabla de capitales..." + WHITE)
        driver.get('https://www.retoactinver.com/RetoActinver/#/capitales')
        driver.refresh()
        time.sleep(3)
        driver.find_element_by_xpath(
            '//*[@id="botonCerrar"]/mat-icon').click()
        time.sleep(6)
        driver.find_element_by_xpath(
            '//*[@id="mat-select-1"]/div/div[1]').click()
        time.sleep(3)
        driver.find_element_by_xpath(
            '//*[@id="mat-option-1"]').click()
        driver.find_element_by_xpath(
            '//*[@id="mat-tab-content-0-0"]/div/div/div/app-table/div/gt-column-settings/div/div/generic-table/table/thead[1]/tr/th[6]/span').click()
        hoursTable = driver.find_element_by_xpath(
            '/html/body/app-root/block-ui/app-paginas/app-layout-principal/mat-sidenav-container/mat-sidenav-content/div[2]/div/app-capitales/div/div[2]/mat-card/mat-tab-group/div/mat-tab-body[1]/div/div/div/app-table/div/gt-column-settings/div/div/generic-table/table').get_attribute("outerHTML")
        dfs = pd.read_html(hoursTable)
        df = dfs[0]
        df.drop(['Sort:', 'Unnamed: 1', 'Información', 'Precio de Compra', 'Volumen de Venta',
                 '% Variación', 'Volumen Compra', 'Precio de Venta'], axis=1, inplace=True)
        df.rename(columns={'Precio': 'Variación'}, inplace=True)
        df['Variación'] = df['Variación'].str.replace('% Variación', '')
        df['Variación'] = df['Variación'].str.replace('%', '')
        df.rename(columns={'Emisora': 'Precio'}, inplace=True)
        df['Precio'] = df['Precio'].str.replace('Precio', '')
        df.rename(columns={'Categorias': 'Emisora'}, inplace=True)
        df['Emisora'] = df['Emisora'].str.replace('Emisora', '')
        df['Emisora'] = df['Emisora'].str.replace(' *', '')
        df['Emisora'] = df['Emisora'].str.replace('*', '')
        df['Datetime'] = datetime.now().strftime("%x %X")
        print(df.head(5))
        df.to_csv('top_dia.csv', index=False, header=True, encoding="utf-8")
    except:
        login_actinver()
        retrieve_data_reto_capitales()






def retrieve_data_reto_portafolio():
    # while True:
    print('\t' + WARNING + "Accediendo a datos de tabla de portafolio..." + WHITE)
    driver.get('https://www.retoactinver.com/RetoActinver/#/portafolio')
    driver.refresh()
    time.sleep(3)
    driver.find_element_by_xpath(
        '//*[@id="botonCerrar"]').click()
    hoursTable = driver.find_element_by_xpath(
        '//*[@id="tpm"]/app-portafolio/div/div[3]/mat-card/app-table/div/gt-column-settings/div/div/generic-table/table').get_attribute("outerHTML")
    dfs = pd.read_html(hoursTable)
    df = dfs[0]
    df = df[['Categorias', 'Emisora', 'Títulos',
             'Precio Actual', 'Variación $']]
    df.rename(columns={'Variación $': 'Variación %'}, inplace=True)
    df['Variación %'] = df['Variación %'].str.replace('% Variación', '')
    df['Variación %'] = df['Variación %'].str.replace('%', '')
    df.rename(columns={'Precio Actual': 'Valor actual'}, inplace=True)
    df['Valor actual'] = df['Valor actual'].str.replace('Valor Actual', '')
    df.rename(columns={'Títulos': 'Costo de compra'}, inplace=True)
    df['Costo de compra'] = df['Costo de compra'].str.replace(
        'Valor del Costo', '')
    df.rename(columns={'Emisora': 'Títulos'}, inplace=True)
    df['Títulos'] = df['Títulos'].str.replace('Títulos', '')
    df.rename(columns={'Categorias': 'Emisora'}, inplace=True)
    df['Emisora'] = df['Emisora'].str.replace('Emisora', '')
    df['Emisora'] = df['Emisora'].str.replace(' *', '')
    df['Emisora'] = df['Emisora'].str.replace('*', '')
    df['Datetime'] = datetime.now().strftime("%x %X")
    print(df.head(10))


def percentage_change(col1,col2):
    return ((col2 - col1) / col1) * 100


def containsAny(str, set):
    """ Check whether sequence str contains ANY of the items in set. """
    return 1 in [c in str for c in set]

def insert_string_before(stringO, string_to_insert, insert_before_char):
    return stringO.replace(insert_before_char, string_to_insert + insert_before_char, 1)

def day_trading_strategy():

    try:
        login_platform_investing(driver)
        driver.get("https://mx.investing.com/")
        time.sleep(3)
        input()
        WebDriverWait(driver, 50).until(EC.element_to_be_clickable((By.XPATH, '//*[@id="portfolioTopBarBtn"]')))
        driver.find_element_by_xpath('//*[@id="portfolioTopBarBtn"]').click()
        WebDriverWait(driver, 50).until(EC.element_to_be_clickable((By.CSS_SELECTOR, '#portfolioContainer_25863947 > div:nth-child(1) > a:nth-child(2)')))
        driver.find_element(By.CSS_SELECTOR, '#portfolioContainer_25863947 > div:nth-child(1) > a:nth-child(2)').click()   
     
       
        driver.find_element(By.CSS_SELECTOR, "#performance > a:nth-child(1)").click()
        time.sleep(3)
        table = driver.find_element_by_xpath('/html/body/div[5]/section/div[7]/div[7]/div[2]/div[2]/div/table').get_attribute("outerHTML")
        dfs = pd.read_html(table)
        df1 = dfs[0]
        df1.drop(['Unnamed: 0', 'Unnamed: 8'], axis=1, inplace=True, errors='ignore')
        #print(df.columns)

        #df.to_csv('analisis_tecnico_performance.csv',  index=False, header=True, encoding="utf-8")
        driver.find_element(By.CSS_SELECTOR, "#technical > a:nth-child(1)").click()
        time.sleep(3)
        table = driver.find_element_by_xpath('/html/body/div[5]/section/div[7]/div[7]/div[2]/div[2]/div/table').get_attribute("outerHTML")
        dfs = pd.read_html(table)
        df2 = dfs[0]
        df2.drop(['Unnamed: 0', 'Unnamed: 10'], axis=1, inplace=True, errors='ignore')
        #print(df.columns)
        #df.to_csv('analisis_tecnico_algoritmo.csv',index=False, header=True, encoding="utf-8")


        driver.find_element(By.CSS_SELECTOR, "#overview > a:nth-child(1)").click()
        time.sleep(3)
        table = driver.find_element_by_xpath('/html/body/div[5]/section/div[7]/div[7]/div[2]/div[2]/div/div/table').get_attribute("outerHTML")
        dfs = pd.read_html(table)
        df3 = dfs[0]
        df3.drop(['Unnamed: 0', 'Unnamed: 1', 'Unnamed: 19', 'Unnamed: 20'], axis=1, inplace=True, errors='ignore')
        #print(df.columns)

        # Create dicionary investing simbols portfolio
        WebDriverWait(driver, 50).until(EC.element_to_be_clickable((By.XPATH, '/html/body/div[5]/section/div[7]/div[7]/div[2]/div[2]/div/div/table')))
        table_elements_simbols = driver.find_elements(By.XPATH, "/html/body/div[5]/section/div[7]/div[7]/div[2]/div[2]/div/div/table/tbody/tr/td[4]/a")
        dicionary_simbols = {}
        for tag in table_elements_simbols:
            dicionary_simbols[str(tag.get_attribute('innerHTML'))] = str(tag.get_attribute('href'))
        

        dfs = [df1, df2, df3]
        dfs = [x.set_index('Nombre') for x in dfs]
        df = pd.concat(dfs, axis=1, keys=range(1, len(dfs) + 1))
        df.columns = df.columns.map('{0[1]}{0[0]}'.format)
        df = df.reset_index()
        df['Fecha'] = datetime.now().strftime("%x %X")
        #print(df.columns)

        
        df['Diario1'] = df['Diario1'].map(lambda x: str(x)[:-1])
        df['Diario1'] = df['Diario1'].astype('float') 
        df['Semanal1'] = df['Semanal1'].map(lambda x: str(x)[:-1])
        df['Semanal1'] = df['Semanal1'].astype('float') 
        df['Mensual1'] = df['Mensual1'].map(lambda x: str(x)[:-1])
        df['Mensual1'] = df['Mensual1'].astype('float') 
        df['Anual1'] = df['Anual1'].map(lambda x: str(x)[:-1])
        df['Anual1'] = df['Anual1'].astype('float') 
        df['1 Año1'] = df['1 Año1'].map(lambda x: str(x)[:-1])
        df['1 Año1'] = df['1 Año1'].astype('float') 
        df['3 años1'] = df['3 años1'].map(lambda x: str(x).replace("-","0"))
        df['3 años1'] = df['3 años1'].map(lambda x: str(x).replace("%",""))
        df['3 años1'] = df['3 años1'].astype('float') 
        df['% var.3'] = df['% var.3'].map(lambda x: str(x)[:-1])
        df['% var.3'] = df['% var.3'].astype('float') 

        print('\t' + WARNING + "Analizando acciones..." + WHITE)

        
        print('\t' + WARNING + "Analizando acciones con ganancias en diferentes periodos..." + WHITE)
        df = df[(df[['3 años1']] > 0).all(1)]
        df = df[(df[['1 Año1']] > 0).all(1)]
        df = df[(df[['Anual1']] > 0).all(1)]
        df = df[(df[['Mensual1']] > 0).all(1)]
        df = df[(df[['Semanal1']] > 0).all(1)]
        print('\t' + WARNING + "Analizando acciones con ganacia diaria menor al promedio..." + WHITE)
        daily_mean = df["Diario1"].mean()
        df = df[(df[['Diario1']] < daily_mean).all(1)]        
        print('\t' + WARNING + "Analizando acciones con indicadores técnicos en consenso de compra en diferentes periodos..." + WHITE)
        df = df[(df[['Mensual2']] == 'Compra').all(1) | (df[['Mensual2']] == 'Compra fuerte').all(1) ]
        df = df[(df[['Semanal2']] == 'Compra').all(1) | (df[['Semanal2']] == 'Compra fuerte').all(1) ]
        df = df[(df[['Diario2']] == 'Compra').all(1) | (df[['Diario2']] == 'Compra fuerte').all(1) ]
        print('\t' + WARNING + "Analizando acciones con indicadores técnicos en baja con estimación de alza..." + WHITE)
        df = df[(df[['5 horas2']] == 'Venta').all(1) | (df[['5 horas2']] == 'Venta fuerte').all(1) | (df[['5 horas2']] == 'Compra fuerte').all(1) | (df[['5 horas2']] == 'Compra').all(1)]
        df = df[(df[['1 hora2']] == 'Venta').all(1) | (df[['1 hora2']] == 'Venta fuerte').all(1) | (df[['1 hora2']] == 'Compra fuerte').all(1) | (df[['1 hora2']] == 'Compra').all(1)]
        df = df[(df[['30 minutos2']] == 'Compra').all(1) | (df[['30 minutos2']] == 'Compra fuerte').all(1) | (df[['30 minutos2']] == 'Venta').all(1) | (df[['30 minutos2']] == 'Venta fuerte').all(1)]
        print('\t' + WARNING + "Analizando acciones con indicadores técnicos con estimación de alza durante la sesión..." + WHITE)
        df = df[(df[['15 minutos2']] == 'Compra').all(1) | (df[['15 minutos2']] == 'Compra fuerte').all(1) ]
        df = df[(df[['5 minutos2']] == 'Compra').all(1) | (df[['5 minutos2']] == 'Compra fuerte').all(1)]


        print('\t' + WARNING + "Calculando precios y ganancias estimadas..." + WHITE)
        for x in df['Símbolo3']:
            if containsAny(dicionary_simbols[x],['?']):
                technical_data_url = insert_string_before(dicionary_simbols[x], '-technical', '?')
            else:
                technical_data_url = dicionary_simbols[x] + '-technical'
            driver.get(technical_data_url)
            WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, '/html/body/div[5]/section/div[8]/ul/li[6]/a')))
            driver.find_element(By.XPATH, "/html/body/div[5]/section/div[8]/ul/li[6]/a").click()

            WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, '/html/body/div[5]/section/div[10]/table')))
            p1 = float(driver.find_element(By.XPATH, "/html/body/div[5]/section/div[10]/table/tbody/tr[2]/td[6]").get_attribute('innerHTML'))
            p2 = float(driver.find_element(By.XPATH, "/html/body/div[5]/section/div[10]/table/tbody/tr[2]/td[7]").get_attribute('innerHTML'))
            p3 = float(driver.find_element(By.XPATH, "/html/body/div[5]/section/div[10]/table/tbody/tr[2]/td[8]").get_attribute('innerHTML'))
            pe = round((p1 + p2 + p3)/3,2)
            df.loc[df['Símbolo3'] == x, 'PeEstimado'] = pe
   
       
        df['GanEstimada %'] = round(percentage_change(df['Último3'],df['PeEstimado']),2)
        df = df[(df[['GanEstimada %']] > 0.5).all(1)]   
        df['PeVentaCalc'] = round(df['Último3'] * 1.005,2)
            
        print('\n\t' + OKGREEN + "Resultado de sugerencias:" + WHITE)  
      
        print(df[['Símbolo3','Nombre','Último3','PeEstimado','GanEstimada %','PeVentaCalc']])

   
    except Exception as e: 
        print(e)



def swing_trading_strategy():

    try:
        login_platform_investing(driver)
        driver.get("https://mx.investing.com/")
        time.sleep(3)

        WebDriverWait(driver, 50).until(EC.element_to_be_clickable((By.XPATH, '//*[@id="portfolioTopBarBtn"]')))
        driver.find_element_by_xpath('//*[@id="portfolioTopBarBtn"]').click()
        WebDriverWait(driver, 50).until(EC.element_to_be_clickable((By.CSS_SELECTOR, '#portfolioContainer_25863947 > div:nth-child(1) > a:nth-child(2)')))
        driver.find_element(By.CSS_SELECTOR, '#portfolioContainer_25863947 > div:nth-child(1) > a:nth-child(2)').click()   
     
       
        driver.find_element(By.CSS_SELECTOR, "#performance > a:nth-child(1)").click()
        time.sleep(3)
        table = driver.find_element_by_xpath('/html/body/div[5]/section/div[7]/div[7]/div[2]/div[2]/div/table').get_attribute("outerHTML")
        dfs = pd.read_html(table)
        df1 = dfs[0]
        df1.drop(['Unnamed: 0', 'Unnamed: 8'], axis=1, inplace=True, errors='ignore')
        #print(df.columns)

        #df.to_csv('analisis_tecnico_performance.csv',  index=False, header=True, encoding="utf-8")
        driver.find_element(By.CSS_SELECTOR, "#technical > a:nth-child(1)").click()
        time.sleep(3)
        table = driver.find_element_by_xpath('/html/body/div[5]/section/div[7]/div[7]/div[2]/div[2]/div/table').get_attribute("outerHTML")
        dfs = pd.read_html(table)
        df2 = dfs[0]
        df2.drop(['Unnamed: 0', 'Unnamed: 10'], axis=1, inplace=True, errors='ignore')
        #print(df.columns)
        #df.to_csv('analisis_tecnico_algoritmo.csv',index=False, header=True, encoding="utf-8")


        driver.find_element(By.CSS_SELECTOR, "#overview > a:nth-child(1)").click()
        time.sleep(3)
        table = driver.find_element_by_xpath('/html/body/div[5]/section/div[7]/div[7]/div[2]/div[2]/div/div/table').get_attribute("outerHTML")
        dfs = pd.read_html(table)
        df3 = dfs[0]
        df3.drop(['Unnamed: 0', 'Unnamed: 1', 'Unnamed: 19', 'Unnamed: 20'], axis=1, inplace=True, errors='ignore')
        #print(df.columns)

        # Create dicionary investing simbols portfolio
        WebDriverWait(driver, 50).until(EC.element_to_be_clickable((By.XPATH, '/html/body/div[5]/section/div[7]/div[7]/div[2]/div[2]/div/div/table')))
        table_elements_simbols = driver.find_elements(By.XPATH, "/html/body/div[5]/section/div[7]/div[7]/div[2]/div[2]/div/div/table/tbody/tr/td[4]/a")
        dicionary_simbols = {}
        for tag in table_elements_simbols:
            dicionary_simbols[str(tag.get_attribute('innerHTML'))] = str(tag.get_attribute('href'))
        

        dfs = [df1, df2, df3]
        dfs = [x.set_index('Nombre') for x in dfs]
        df = pd.concat(dfs, axis=1, keys=range(1, len(dfs) + 1))
        df.columns = df.columns.map('{0[1]}{0[0]}'.format)
        df = df.reset_index()
        df['Fecha'] = datetime.now().strftime("%x %X")
        #print(df.columns)

        
        df['Diario1'] = df['Diario1'].map(lambda x: str(x)[:-1])
        df['Diario1'] = df['Diario1'].astype('float') 
        df['Semanal1'] = df['Semanal1'].map(lambda x: str(x)[:-1])
        df['Semanal1'] = df['Semanal1'].astype('float') 
        df['Mensual1'] = df['Mensual1'].map(lambda x: str(x)[:-1])
        df['Mensual1'] = df['Mensual1'].astype('float') 
        df['Anual1'] = df['Anual1'].map(lambda x: str(x)[:-1])
        df['Anual1'] = df['Anual1'].astype('float') 
        df['1 Año1'] = df['1 Año1'].map(lambda x: str(x)[:-1])
        df['1 Año1'] = df['1 Año1'].astype('float') 
        df['3 años1'] = df['3 años1'].map(lambda x: str(x).replace("-","0"))
        df['3 años1'] = df['3 años1'].map(lambda x: str(x).replace("%",""))
        df['3 años1'] = df['3 años1'].astype('float') 
        df['% var.3'] = df['% var.3'].map(lambda x: str(x)[:-1])
        df['% var.3'] = df['% var.3'].astype('float') 

        print('\t' + WARNING + "Analizando acciones..." + WHITE)
        
        print('\t' + WARNING + "Analizando acciones con ganancias en diferentes periodos..." + WHITE)
        df = df[(df[['3 años1']] > 0).all(1)]
        df = df[(df[['1 Año1']] > 0).all(1)]
        df = df[(df[['Anual1']] > 0).all(1)]
        df = df[(df[['Mensual1']] > 0).all(1)]
        df = df[(df[['Semanal1']] > 0).all(1)]
        print('\t' + WARNING + "Analizando acciones con ganacia diaria menor al promedio..." + WHITE)
        df = df[(df[['Diario1']] < 0).all(1)]
        print('\t' + WARNING + "Analizando acciones con indicadores técnicos en consenso de compra en diferentes periodos..." + WHITE)
        df = df[(df[['Mensual2']] == 'Compra').all(1) | (df[['Mensual2']] == 'Compra fuerte').all(1) ]
        df = df[(df[['Semanal2']] == 'Compra').all(1) | (df[['Semanal2']] == 'Compra fuerte').all(1) ]
        df = df[(df[['Diario2']] == 'Compra').all(1) | (df[['Diario2']] == 'Compra fuerte').all(1) ]
        print('\t' + WARNING + "Analizando acciones con indicadores técnicos en baja con estimación de alza..." + WHITE)
        df = df[(df[['5 horas2']] == 'Venta').all(1) | (df[['5 horas2']] == 'Venta fuerte').all(1) | (df[['5 horas2']] == 'Compra fuerte').all(1) | (df[['5 horas2']] == 'Compra').all(1)]
        df = df[(df[['1 hora2']] == 'Venta').all(1) | (df[['1 hora2']] == 'Venta fuerte').all(1) | (df[['1 hora2']] == 'Compra fuerte').all(1) | (df[['1 hora2']] == 'Compra').all(1)]
        df = df[(df[['30 minutos2']] == 'Compra').all(1) | (df[['30 minutos2']] == 'Compra fuerte').all(1) | (df[['30 minutos2']] == 'Venta').all(1) | (df[['30 minutos2']] == 'Venta fuerte').all(1)]
        print('\t' + WARNING + "Analizando acciones con indicadores técnicos con estimación de alza durante la sesión..." + WHITE)
        df = df[(df[['15 minutos2']] == 'Compra').all(1) | (df[['15 minutos2']] == 'Compra fuerte').all(1) | (df[['15 minutos2']] == 'Neutral').all(1) | (df[['15 minutos2']] == 'Venta fuerte').all(1) | (df[['15 minutos2']] == 'Venta').all(1) ]
        df = df[(df[['5 minutos2']] == 'Compra').all(1) | (df[['5 minutos2']] == 'Compra fuerte').all(1) | (df[['5 minutos2']] == 'Neutral').all(1) | (df[['5 minutos2']] == 'Venta fuerte').all(1) | (df[['5 minutos2']] == 'Venta').all(1)]

        print('\t' + WARNING + "Calculando precios y ganancias estimadas..." + WHITE)
        for x in df['Símbolo3']:
            if containsAny(dicionary_simbols[x],['?']):
                technical_data_url = insert_string_before(dicionary_simbols[x], '-technical', '?')
            else:
                technical_data_url = dicionary_simbols[x] + '-technical'

            driver.get(technical_data_url)
            WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, '/html/body/div[5]/section/div[8]/ul/li[6]/a')))
            driver.find_element(By.XPATH, "/html/body/div[5]/section/div[8]/ul/li[6]/a").click()

            WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, '/html/body/div[5]/section/div[10]/table')))
            p1 = float(driver.find_element(By.XPATH, "/html/body/div[5]/section/div[10]/table/tbody/tr[2]/td[6]").get_attribute('innerHTML'))
            p2 = float(driver.find_element(By.XPATH, "/html/body/div[5]/section/div[10]/table/tbody/tr[2]/td[7]").get_attribute('innerHTML'))
            p3 = float(driver.find_element(By.XPATH, "/html/body/div[5]/section/div[10]/table/tbody/tr[2]/td[8]").get_attribute('innerHTML'))
            pe = round((p1 + p2 + p3)/3,2)
            df.loc[df['Símbolo3'] == x, 'PeEstimado'] = pe
   
       
        df['GanEstimada %'] = round(percentage_change(df['Último3'],df['PeEstimado']),2)
        df = df[(df[['GanEstimada %']] > 0.5).all(1)]   
        df['PeVentaCalc'] = round(df['Último3'] * 1.005,2)
            
        print('\n\t' + OKGREEN + "Resultado de sugerencias:" + WHITE)  
      
        print(df[['Símbolo3','Nombre','Último3','PeEstimado','GanEstimada %','PeVentaCalc']])

   
    except Exception as e: 
        print(e)   


def swing_trading_strategy2():

    try:
        print('\t' + WARNING + "Obteniendo datos de acciones..." + WHITE)
        driver.get("https://mx.investing.com/")

        is_logged_flag = ""
        try:
            WebDriverWait(driver, 3).until(EC.element_to_be_clickable((By.CSS_SELECTOR, '.myAccount')))
            is_logged_flag = driver.find_element_by_css_selector('.myAccount').get_attribute("innerText")
        except Exception as e: 
            print(e)      
        if (is_logged_flag=="Osvaldo"):
            print('\t' + WARNING + "Sesión ya iniciada con anterioridad" + WHITE)
        else: 
            try:
                try:
                    try:
                        driver.find_element(By.LINK_TEXT, "Iniciar sesión").click() 
                           
                    except Exception as e: 
                        print(e)
                        WebDriverWait(driver, 50).until(EC.element_to_be_clickable((By.CSS_SELECTOR, '.generalOverlay')))
                        driver.find_element(By.CSS_SELECTOR, '.popupCloseIcon').click()     
                        driver.find_element(By.LINK_TEXT, "Iniciar sesión").click()
        
                except Exception as e: 
                        print(e)
        
                driver.find_element(By.ID, "loginFormUser_email").send_keys("osvaldo.hdz.m@outlook.com")
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
            except Exception as e: 
                        print(e)

        print('\t' + WARNING + "Conexión con datos de investing.com establecida" + WHITE)
        time.sleep(3)

        WebDriverWait(driver, 50).until(EC.element_to_be_clickable((By.XPATH, '//*[@id="portfolioTopBarBtn"]')))
        driver.find_element_by_xpath('//*[@id="portfolioTopBarBtn"]').click()
        WebDriverWait(driver, 50).until(EC.element_to_be_clickable((By.CSS_SELECTOR, '#portfolioContainer_25863947 > div:nth-child(1) > a:nth-child(2)')))
        driver.find_element(By.CSS_SELECTOR, '#portfolioContainer_25863947 > div:nth-child(1) > a:nth-child(2)').click()   
     
       
        driver.find_element(By.CSS_SELECTOR, "#performance > a:nth-child(1)").click()
        time.sleep(3)
        table = driver.find_element_by_xpath('/html/body/div[5]/section/div[7]/div[7]/div[2]/div[2]/div/table').get_attribute("outerHTML")
        dfs = pd.read_html(table)
        df1 = dfs[0]
        df1.drop(['Unnamed: 0', 'Unnamed: 8'], axis=1, inplace=True, errors='ignore')
        #print(df.columns)

        #df.to_csv('analisis_tecnico_performance.csv',  index=False, header=True, encoding="utf-8")
        driver.find_element(By.CSS_SELECTOR, "#technical > a:nth-child(1)").click()
        time.sleep(3)
        table = driver.find_element_by_xpath('/html/body/div[5]/section/div[7]/div[7]/div[2]/div[2]/div/table').get_attribute("outerHTML")
        dfs = pd.read_html(table)
        df2 = dfs[0]
        df2.drop(['Unnamed: 0', 'Unnamed: 10'], axis=1, inplace=True, errors='ignore')
        #print(df.columns)
        #df.to_csv('analisis_tecnico_algoritmo.csv',index=False, header=True, encoding="utf-8")


        driver.find_element(By.CSS_SELECTOR, "#overview > a:nth-child(1)").click()
        time.sleep(3)
        table = driver.find_element_by_xpath('/html/body/div[5]/section/div[7]/div[7]/div[2]/div[2]/div/div/table').get_attribute("outerHTML")
        dfs = pd.read_html(table)
        df3 = dfs[0]
        df3.drop(['Unnamed: 0', 'Unnamed: 1', 'Unnamed: 19', 'Unnamed: 20'], axis=1, inplace=True, errors='ignore')
        #print(df.columns)

        # Create dicionary investing simbols portfolio
        WebDriverWait(driver, 50).until(EC.element_to_be_clickable((By.XPATH, '/html/body/div[5]/section/div[7]/div[7]/div[2]/div[2]/div/div/table')))
        table_elements_simbols = driver.find_elements(By.XPATH, "/html/body/div[5]/section/div[7]/div[7]/div[2]/div[2]/div/div/table/tbody/tr/td[4]/a")
        dicionary_simbols = {}
        for tag in table_elements_simbols:
            dicionary_simbols[str(tag.get_attribute('innerHTML'))] = str(tag.get_attribute('href'))
        

        dfs = [df1, df2, df3]
        dfs = [x.set_index('Nombre') for x in dfs]
        df = pd.concat(dfs, axis=1, keys=range(1, len(dfs) + 1))
        df.columns = df.columns.map('{0[1]}{0[0]}'.format)
        df = df.reset_index()
        df['Fecha'] = datetime.now().strftime("%x %X")
        #print(df.columns)

        
        df['Diario1'] = df['Diario1'].map(lambda x: str(x)[:-1])
        df['Diario1'] = df['Diario1'].astype('float') 
        df['Semanal1'] = df['Semanal1'].map(lambda x: str(x)[:-1])
        df['Semanal1'] = df['Semanal1'].astype('float') 
        df['Mensual1'] = df['Mensual1'].map(lambda x: str(x)[:-1])
        df['Mensual1'] = df['Mensual1'].astype('float') 
        df['Anual1'] = df['Anual1'].map(lambda x: str(x)[:-1])
        df['Anual1'] = df['Anual1'].astype('float') 
        df['1 Año1'] = df['1 Año1'].map(lambda x: str(x)[:-1])
        df['1 Año1'] = df['1 Año1'].astype('float') 
        df['3 años1'] = df['3 años1'].map(lambda x: str(x).replace("-","0"))
        df['3 años1'] = df['3 años1'].map(lambda x: str(x).replace("%",""))
        df['3 años1'] = df['3 años1'].astype('float') 
        df['% var.3'] = df['% var.3'].map(lambda x: str(x)[:-1])
        df['% var.3'] = df['% var.3'].astype('float') 

        print('\t' + WARNING + "Analizando acciones..." + WHITE)
        
        print('\t' + WARNING + "Analizando acciones con ganancias en diferentes periodos..." + WHITE)
        df = df[(df[['3 años1']] > 0).all(1)]
        df = df[(df[['1 Año1']] > 0).all(1)]
        df = df[(df[['Anual1']] > 0).all(1)]
        df = df[(df[['Mensual1']] > 0).all(1)]
        df = df[(df[['Semanal1']] > 0).all(1)]
        print('\t' + WARNING + "Analizando acciones con ganacia diaria menor al promedio..." + WHITE)
        df = df[(df[['Diario1']] > 0).all(1)]

        print('\t' + WARNING + "Analizando acciones con indicadores técnicos en consenso de compra en diferentes periodos..." + WHITE)
        df = df[(df[['Mensual2']] == 'Compra').all(1) | (df[['Mensual2']] == 'Compra fuerte').all(1) ]
        df = df[(df[['Semanal2']] == 'Compra').all(1) | (df[['Semanal2']] == 'Compra fuerte').all(1) ]
        df = df[(df[['Diario2']] == 'Compra').all(1) | (df[['Diario2']] == 'Compra fuerte').all(1) ]
        print('\t' + WARNING + "Analizando acciones con indicadores técnicos en baja con estimación de alza..." + WHITE)
        df = df[(df[['5 horas2']] == 'Venta').all(1) | (df[['5 horas2']] == 'Venta fuerte').all(1) | (df[['5 horas2']] == 'Compra fuerte').all(1) | (df[['5 horas2']] == 'Compra').all(1)]
        #df = df[(df[['1 hora2']] == 'Venta').all(1) | (df[['1 hora2']] == 'Venta fuerte').all(1) | (df[['1 hora2']] == 'Compra fuerte').all(1) | (df[['1 hora2']] == 'Compra').all(1)]
        #df = df[(df[['30 minutos2']] == 'Compra').all(1) | (df[['30 minutos2']] == 'Compra fuerte').all(1) | (df[['30 minutos2']] == 'Venta').all(1) | (df[['30 minutos2']] == 'Venta fuerte').all(1)]
        print('\t' + WARNING + "Analizando acciones con indicadores técnicos con estimación de alza durante la sesión..." + WHITE)
        #df = df[(df[['15 minutos2']] == 'Compra').all(1) | (df[['15 minutos2']] == 'Compra fuerte').all(1) | (df[['15 minutos2']] == 'Neutral').all(1) | (df[['15 minutos2']] == 'Venta fuerte').all(1) | (df[['15 minutos2']] == 'Venta').all(1) ]
        #df = df[(df[['5 minutos2']] == 'Compra').all(1) | (df[['5 minutos2']] == 'Compra fuerte').all(1) | (df[['5 minutos2']] == 'Neutral').all(1) | (df[['5 minutos2']] == 'Venta fuerte').all(1) | (df[['5 minutos2']] == 'Venta').all(1)]

        print('\t' + WARNING + "Calculando precios y ganancias estimadas..." + WHITE)
        for x in df['Símbolo3']:
            if containsAny(dicionary_simbols[x],['?']):
                technical_data_url = insert_string_before(dicionary_simbols[x], '-technical', '?')
            else:
                technical_data_url = dicionary_simbols[x] + '-technical'
            driver.get(technical_data_url)
            WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, '/html/body/div[5]/section/div[8]/ul/li[6]/a')))
            driver.find_element(By.XPATH, "/html/body/div[5]/section/div[8]/ul/li[6]/a").click()

            WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, '/html/body/div[5]/section/div[10]/table')))
            p1 = float(driver.find_element(By.XPATH, "/html/body/div[5]/section/div[10]/table/tbody/tr[2]/td[6]").get_attribute('innerHTML'))
            p2 = float(driver.find_element(By.XPATH, "/html/body/div[5]/section/div[10]/table/tbody/tr[2]/td[7]").get_attribute('innerHTML'))
            p3 = float(driver.find_element(By.XPATH, "/html/body/div[5]/section/div[10]/table/tbody/tr[2]/td[8]").get_attribute('innerHTML'))
            pe = round((p1 + p2 + p3)/3,2)
            df.loc[df['Símbolo3'] == x, 'PeEstimado'] = pe
   
       
        df['GanEstimada %'] = round(percentage_change(df['Último3'],df['PeEstimado']),2)
        df = df[(df[['GanEstimada %']] > 0.5).all(1)]   
        df['PeVentaCalc'] = round(df['Último3'] * 1.005,2)
            
        print('\n\t' + OKGREEN + "Resultado de sugerencias:" + WHITE)  
      
        print(df[['Símbolo3','Nombre','Último3','PeEstimado','GanEstimada %','PeVentaCalc']])

    except Exception as e: 
        print(e)   


def predict_machine_daily(stock_simbol):
    mod_ticker = stock_simbol + '.MX'

    start = datetime.now() - timedelta(days=365)   
    end = datetime.now()
    
    df = web.DataReader(mod_ticker, 'yahoo', start, end)

    df = df.replace(0, np.nan).ffill()

    if len(df.index) < 10:
        return 'Neutral'
    else: 

        dfreg = df.loc[:,['Adj Close','Volume']]
        dfreg['HL_PCT'] = (df['High'] - df['Low']) / df['Close'] * 100.0
        dfreg['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0

        last_close = float(dfreg['Adj Close'].iloc[-1])
    
        # Drop missing value
        dfreg.fillna(value=-99999, inplace=True)
        
        #print(dfreg.shape)
        # We want to separate 1 percent of the data to forecast
        # Number of forecast values in plot
        forecast_out = int(math.ceil(0.01 * len(dfreg)))
    
        #print("forecast out  : "+ str(forecast_out))
        
        # Separating the label here, we want to predict the AdjClose
        forecast_col = 'Adj Close'
        dfreg['label'] = dfreg[forecast_col].shift(-forecast_out)
        X = np.array(dfreg.drop(['label'], 1))
        
        # Scale the X so that everyone can have the same distribution for linear regression
        X = preprocessing.scale(X)
        
        # Finally We want to find Data Series of late X and early X (train) for model generation and evaluation
        X_lately = X[-forecast_out:]
        X = X[:-forecast_out]
        
        # Separate label and identify it as y
        y = np.array(dfreg['label'])
        y = y[:-forecast_out]
        
        #print('Dimension of X',X.shape)
        #print('Dimension of y',y.shape)
    
    
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
        confidencepoly2 = clfpoly2.score(X_test,y_test)
        confidencepoly3 = clfpoly3.score(X_test,y_test)
        confidenceknn = clfknn.score(X_test, y_test)
        
        #print("The linear regression confidence is ",confidencereg)
        #print("The quadratic regression 2 confidence is ",confidencepoly2)
        #print("The quadratic regression 3 confidence is ",confidencepoly3)
        #print("The knn regression confidence is ",confidenceknn)
    
        # Printing the forecast
        forecast_set = clfreg.predict(X_lately)
    
        dfreg['Forecast'] = np.nan
        #print(forecast_set, confidencereg, forecast_out)
    
        last_date = dfreg.iloc[-1].name
        last_unix = last_date
        next_unix = last_unix + timedelta(days=1)
    
        for i in forecast_set:
            next_date = next_unix
            next_unix += timedelta(days=1)
            dfreg.loc[next_date] = [np.nan for _ in range(len(dfreg.columns)-1)]+[i]
    
        #print(dfreg['Forecast'])
    
        dfreg['Adj Close'].tail(500).plot()
        dfreg['Forecast'].tail(500).plot()
        
        #plt.legend(loc=4)
        #plt.xlabel('Date')
        #plt.ylabel('Price')
        
        # Plot de graph 
        #plt.show()
    
        last_forescast = float(dfreg['Forecast'].iloc[-1])
        
        diference = last_forescast-last_close
    
        #print(last_close)
        #print(last_forescast)
        #print(diference)
        if diference > 0:
            return 'Compra'
        elif diference < 0:
            return 'Venta'
        else :
            return 'Neutral'



    



def swing_trading_strategy_machine():


    try:
        print('\t' + WARNING + "Obteniendo datos de acciones..." + WHITE)
        driver.get("https://mx.investing.com/")

        is_logged_flag = ""
        try:
            WebDriverWait(driver, 3).until(EC.element_to_be_clickable((By.CSS_SELECTOR, '.myAccount')))
            is_logged_flag = driver.find_element_by_css_selector('.myAccount').get_attribute("innerText")
        except Exception as e: 
            print(e)      
        if (is_logged_flag=="Osvaldo"):
            print('\t' + WARNING + "Sesión ya iniciada con anterioridad" + WHITE)
        else: 
            try:
                try:
                    try:
                        driver.find_element(By.LINK_TEXT, "Iniciar sesión").click() 
                           
                    except Exception as e: 
                        print(e)
                        WebDriverWait(driver, 50).until(EC.element_to_be_clickable((By.CSS_SELECTOR, '.generalOverlay')))
                        driver.find_element(By.CSS_SELECTOR, '.popupCloseIcon').click()     
                        driver.find_element(By.LINK_TEXT, "Iniciar sesión").click()
        
                except Exception as e: 
                        print(e)
        
                driver.find_element(By.ID, "loginFormUser_email").send_keys("osvaldo.hdz.m@outlook.com")
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
            except Exception as e: 
                        print(e)

        print('\t' + WARNING + "Conexión con datos de investing.com establecida" + WHITE)
        time.sleep(3)

        WebDriverWait(driver, 50).until(EC.element_to_be_clickable((By.XPATH, '//*[@id="portfolioTopBarBtn"]')))
        driver.find_element_by_xpath('//*[@id="portfolioTopBarBtn"]').click()
        WebDriverWait(driver, 50).until(EC.element_to_be_clickable((By.CSS_SELECTOR, '#portfolioContainer_25863947 > div:nth-child(1) > a:nth-child(2)')))
        driver.find_element(By.CSS_SELECTOR, '#portfolioContainer_25863947 > div:nth-child(1) > a:nth-child(2)').click()   
     
       
        driver.find_element(By.CSS_SELECTOR, "#performance > a:nth-child(1)").click()
        time.sleep(3)
        table = driver.find_element_by_xpath('/html/body/div[5]/section/div[7]/div[7]/div[2]/div[2]/div/table').get_attribute("outerHTML")
        dfs = pd.read_html(table)
        df1 = dfs[0]
        df1.drop(['Unnamed: 0', 'Unnamed: 8'], axis=1, inplace=True, errors='ignore')
        #print(df.columns)

        #df.to_csv('analisis_tecnico_performance.csv',  index=False, header=True, encoding="utf-8")
        driver.find_element(By.CSS_SELECTOR, "#technical > a:nth-child(1)").click()
        time.sleep(3)
        table = driver.find_element_by_xpath('/html/body/div[5]/section/div[7]/div[7]/div[2]/div[2]/div/table').get_attribute("outerHTML")
        dfs = pd.read_html(table)
        df2 = dfs[0]
        df2.drop(['Unnamed: 0', 'Unnamed: 10'], axis=1, inplace=True, errors='ignore')
        #print(df.columns)
        #df.to_csv('analisis_tecnico_algoritmo.csv',index=False, header=True, encoding="utf-8")


        driver.find_element(By.CSS_SELECTOR, "#overview > a:nth-child(1)").click()
        time.sleep(3)
        table = driver.find_element_by_xpath('/html/body/div[5]/section/div[7]/div[7]/div[2]/div[2]/div/div/table').get_attribute("outerHTML")
        dfs = pd.read_html(table)
        df3 = dfs[0]
        df3.drop(['Unnamed: 0', 'Unnamed: 1', 'Unnamed: 19', 'Unnamed: 20'], axis=1, inplace=True, errors='ignore')
        #print(df.columns)

        # Create dicionary investing simbols portfolio
        WebDriverWait(driver, 50).until(EC.element_to_be_clickable((By.XPATH, '/html/body/div[5]/section/div[7]/div[7]/div[2]/div[2]/div/div/table')))
        table_elements_simbols = driver.find_elements(By.XPATH, "/html/body/div[5]/section/div[7]/div[7]/div[2]/div[2]/div/div/table/tbody/tr/td[4]/a")
        dicionary_simbols = {}
        for tag in table_elements_simbols:
            dicionary_simbols[str(tag.get_attribute('innerHTML'))] = str(tag.get_attribute('href'))
        

        dfs = [df1, df2, df3]
        dfs = [x.set_index('Nombre') for x in dfs]
        df = pd.concat(dfs, axis=1, keys=range(1, len(dfs) + 1))
        df.columns = df.columns.map('{0[1]}{0[0]}'.format)
        df = df.reset_index()
        df['Fecha'] = datetime.now().strftime("%x %X")

        df['Diario1'] = df['Diario1'].map(lambda x: str(x).replace("+-","+")[:-1])
        df['Diario1'] = df['Diario1'].astype('float') 
        df['Semanal1'] = df['Semanal1'].map(lambda x: str(x)[:-1])
        df['Semanal1'] = df['Semanal1'].astype('float') 
        df['Mensual1'] = df['Mensual1'].map(lambda x: str(x)[:-1])
        df['Mensual1'] = df['Mensual1'].astype('float') 
        df['Anual1'] = df['Anual1'].map(lambda x: str(x)[:-1])
        df['Anual1'] = df['Anual1'].astype('float') 
        df['1 Año1'] = df['1 Año1'].map(lambda x: str(x)[:-1])
        df['1 Año1'] = df['1 Año1'].astype('float') 
        df['3 años1'] = df['3 años1'].map(lambda x: str(x).replace("-","0"))
        df['3 años1'] = df['3 años1'].map(lambda x: str(x).replace("%",""))
        df['3 años1'] = df['3 años1'].astype('float') 
        df['% var.3'] = df['% var.3'].map(lambda x: str(x).replace("+-","+")[:-1])
        df['% var.3'] = df['% var.3'].astype('float') 

        print('\t' + WARNING + "Analizando acciones..." + WHITE)
        
        print('\t' + WARNING + "Analizando acciones con ganancias en diferentes periodos..." + WHITE)
        df = df[(df[['3 años1']] > 0).all(1)]
        df = df[(df[['1 Año1']] > 0).all(1)]
        df = df[(df[['Anual1']] > 0).all(1)]
        df = df[(df[['Mensual1']] > 0).all(1)]
        df = df[(df[['Semanal1']] > 0).all(1)]
        print('\t' + WARNING + "Analizando acciones con ganacia diaria menor al promedio..." + WHITE)
        df = df[(df[['Diario1']] > 0).all(1)]

        print('\t' + WARNING + "Analizando acciones con indicadores técnicos en consenso de compra en diferentes periodos..." + WHITE)
        df = df[(df[['Mensual2']] == 'Compra').all(1) | (df[['Mensual2']] == 'Compra fuerte').all(1) ]
        df = df[(df[['Semanal2']] == 'Compra').all(1) | (df[['Semanal2']] == 'Compra fuerte').all(1) ]
        df = df[(df[['Diario2']] == 'Compra').all(1) | (df[['Diario2']] == 'Compra fuerte').all(1) ]
        print('\t' + WARNING + "Analizando acciones con indicadores técnicos en baja con estimación de alza..." + WHITE)
        df = df[(df[['5 horas2']] == 'Venta').all(1) | (df[['5 horas2']] == 'Venta fuerte').all(1) | (df[['5 horas2']] == 'Compra fuerte').all(1) | (df[['5 horas2']] == 'Compra').all(1)]
        #df = df[(df[['1 hora2']] == 'Venta').all(1) | (df[['1 hora2']] == 'Venta fuerte').all(1) | (df[['1 hora2']] == 'Compra fuerte').all(1) | (df[['1 hora2']] == 'Compra').all(1)]
        #df = df[(df[['30 minutos2']] == 'Compra').all(1) | (df[['30 minutos2']] == 'Compra fuerte').all(1) | (df[['30 minutos2']] == 'Venta').all(1) | (df[['30 minutos2']] == 'Venta fuerte').all(1)]
        print('\t' + WARNING + "Analizando acciones con indicadores técnicos con estimación de alza durante la sesión..." + WHITE)
        #df = df[(df[['15 minutos2']] == 'Compra').all(1) | (df[['15 minutos2']] == 'Compra fuerte').all(1) | (df[['15 minutos2']] == 'Neutral').all(1) | (df[['15 minutos2']] == 'Venta fuerte').all(1) | (df[['15 minutos2']] == 'Venta').all(1) ]
        #df = df[(df[['5 minutos2']] == 'Compra').all(1) | (df[['5 minutos2']] == 'Compra fuerte').all(1) | (df[['5 minutos2']] == 'Neutral').all(1) | (df[['5 minutos2']] == 'Venta fuerte').all(1) | (df[['5 minutos2']] == 'Venta').all(1)]

        print('\t' + WARNING + "Calculando precios y ganancias estimadas..." + WHITE)
        for x in df['Símbolo3']:
            print('\t' + WARNING + "Analizando {} ...".format(x) + WHITE)


            if containsAny(dicionary_simbols[x],['?']):
                technical_data_url = insert_string_before(dicionary_simbols[x], '-technical', '?')
            else:
                technical_data_url = dicionary_simbols[x] + '-technical'
            driver.get(technical_data_url)
            WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, '/html/body/div[5]/section/div[8]/ul/li[6]/a')))
            driver.find_element(By.XPATH, "/html/body/div[5]/section/div[8]/ul/li[6]/a").click()

            WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, '/html/body/div[5]/section/div[10]/table')))
            p1 = float(driver.find_element(By.XPATH, "/html/body/div[5]/section/div[10]/table/tbody/tr[2]/td[6]").get_attribute('innerHTML'))
            p2 = float(driver.find_element(By.XPATH, "/html/body/div[5]/section/div[10]/table/tbody/tr[2]/td[7]").get_attribute('innerHTML'))
            p3 = float(driver.find_element(By.XPATH, "/html/body/div[5]/section/div[10]/table/tbody/tr[2]/td[8]").get_attribute('innerHTML'))
            pe = round((p1 + p2 + p3)/3,2)
            df.loc[df['Símbolo3'] == x, 'PeEstimado'] = pe
            df['ML Prediction'] = 'NO DISPONIBLE'
            try:
                df.loc[df['Símbolo3'] == x, 'ML Prediction'] = predict_machine_daily(x)
            except Exception as e: 
                print(e)

             

   
        df = df[(df[['ML Prediction']] == 'Compra').all(1)]  
        df['GanEstimada %'] = round(percentage_change(df['Último3'],df['PeEstimado']),2)
        df = df[(df[['GanEstimada %']] > 0.5).all(1)]   
        df['PeVentaCalc'] = round(df['Último3'] * 1.005,2)
            
        print('\n\t' + OKGREEN + "Resultado de sugerencias:" + WHITE)        
        print(df[['Símbolo3','Nombre','Último3','PeEstimado','GanEstimada %','PeVentaCalc','ML Prediction']])

        opcion = input("Deseas ejecutar la optimización de portafolio con estas acciones? (y/n) >> ") 
        
        while True:
            if opcion =="y" or opcion =="Y":
                tickers = df['Símbolo3'].astype(str).values.tolist() 
                df_allocations = portfolio_optimization2(tickers,1000000)
                df_result = pd.merge(df_allocations,df,left_on='Ticker',right_on='Símbolo3')
                df_result['Títulos'] = df_result['Allocation $']/df_result['Último3']
                df_result['Títulos'] = df_result['Títulos'].astype('int32') 
                print('\n\t' + OKGREEN + "Resultado de optimización:" + WHITE) 
                print(df_result[['Símbolo3','Nombre','Títulos','Allocation $','Último3','PeEstimado','GanEstimada %','PeVentaCalc']])

                break
            elif opcion =="n" or opcion =="N":
                break 

    except Exception as e: 
        print(e)   



def portfolio_annualised_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns*weights ) *252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return std, returns




def random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate):
    results = np.zeros((3,num_portfolios))
    weights_record = []
    for i in range(num_portfolios):
        weights = np.random.random(4)
        weights /= np.sum(weights)
        weights_record.append(weights)
        portfolio_std_dev, portfolio_return = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
        results[0,i] = portfolio_std_dev
        results[1,i] = portfolio_return
        results[2,i] = (portfolio_return - risk_free_rate) / portfolio_std_dev
    return results, weights_record



def display_simulated_ef_with_random(mean_returns, cov_matrix, num_portfolios, risk_free_rate, stocks_data):
    results, weights = random_portfolios(num_portfolios,mean_returns, cov_matrix, risk_free_rate)
    
    max_sharpe_idx = np.argmax(results[2])
    sdp, rp = results[0,max_sharpe_idx], results[1,max_sharpe_idx]
    max_sharpe_allocation = pd.DataFrame(weights[max_sharpe_idx],index=stocks_data.columns, columns=['allocation'])
    max_sharpe_allocation.allocation = [round(i*100,2)for i in max_sharpe_allocation.allocation]
    max_sharpe_allocation = max_sharpe_allocation.T
    
    min_vol_idx = np.argmin(results[0])
    sdp_min, rp_min = results[0,min_vol_idx], results[1,min_vol_idx]
    min_vol_allocation = pd.DataFrame(weights[min_vol_idx],index=stocks_data.columns,columns=['allocation'])
    min_vol_allocation.allocation = [round(i*100,2)for i in min_vol_allocation.allocation]
    min_vol_allocation = min_vol_allocation.T
    
    print("------------------------")
    print("Maximum Sharpe Ratio Portfolio Allocation\n")
    print("Annualised Return:", round(rp,2))
    print("Annualised Volatility:", round(sdp,2))
    print("\n")
    print(max_sharpe_allocation)
    print("------------------------")
    print("Minimum Volatility Portfolio Allocation\n")
    print("Annualised Return:", round(rp_min,2))
    print("Annualised Volatility:", round(sdp_min,2))
    print("\n")
    print(min_vol_allocation)
    
    plt.figure(figsize=(10, 7))
    plt.scatter(results[0,:],results[1,:],c=results[2,:],cmap='YlGnBu', marker='o', s=10, alpha=0.3)
    plt.colorbar()
    plt.scatter(sdp,rp,marker='*',color='r',s=500, label='Maximum Sharpe ratio')
    plt.scatter(sdp_min,rp_min,marker='*',color='g',s=500, label='Minimum volatility')
    plt.title('Simulated Portfolio Optimization based on Efficient Frontier')
    plt.xlabel('annualised volatility')
    plt.ylabel('annualised returns')
    plt.legend(labelspacing=0.8)




def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate,):
    p_var, p_ret = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
    return -(p_ret - risk_free_rate) / p_var

def max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0,1.0)
    bounds = tuple(bound for asset in range(num_assets))
    result = sco.minimize(neg_sharpe_ratio, num_assets*[1./num_assets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
    return result

def portfolio_volatility(weights, mean_returns, cov_matrix):
    return portfolio_annualised_performance(weights, mean_returns, cov_matrix)[0]

def min_variance(mean_returns, cov_matrix):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0,1.0)
    bounds = tuple(bound for asset in range(num_assets))

    result = sco.minimize(portfolio_volatility, num_assets*[1./num_assets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)

    return result

def efficient_return(mean_returns, cov_matrix, target):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)

    def portfolio_return(weights):
        return portfolio_annualised_performance(weights, mean_returns, cov_matrix)[1]

    constraints = ({'type': 'eq', 'fun': lambda x: portfolio_return(x) - target},
                   {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0,1) for asset in range(num_assets))
    result = sco.minimize(portfolio_volatility, num_assets*[1./num_assets,], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result


def efficient_frontier(mean_returns, cov_matrix, returns_range):
    efficients = []
    for ret in returns_range:
        efficients.append(efficient_return(mean_returns, cov_matrix, ret))
    return efficients

def display_calculated_ef_with_random(mean_returns, cov_matrix, num_portfolios, risk_free_rate, stocks_data):
    results, _ = random_portfolios(num_portfolios,mean_returns, cov_matrix, risk_free_rate)
    
    max_sharpe = max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate)
    sdp, rp = portfolio_annualised_performance(max_sharpe['x'], mean_returns, cov_matrix)
    max_sharpe_allocation = pd.DataFrame(max_sharpe.x,index=stocks_data.columns,columns=['allocation'])
    max_sharpe_allocation.allocation = [round(i*100,2)for i in max_sharpe_allocation.allocation]
    max_sharpe_allocation = max_sharpe_allocation.T
    max_sharpe_allocation

    min_vol = min_variance(mean_returns, cov_matrix)
    sdp_min, rp_min = portfolio_annualised_performance(min_vol['x'], mean_returns, cov_matrix)
    min_vol_allocation = pd.DataFrame(min_vol.x,index=stocks_data.columns,columns=['allocation'])
    min_vol_allocation.allocation = [round(i*100,2)for i in min_vol_allocation.allocation]
    min_vol_allocation = min_vol_allocation.T
    
    print("------------------------")
    print("Maximum Sharpe Ratio Portfolio Allocation\n")
    print("Annualised Return:", round(rp,2))
    print("Annualised Volatility:", round(sdp,2))
    print("\n")
    print(max_sharpe_allocation)
    print("------------------------")
    print("Minimum Volatility Portfolio Allocation\n")
    print("Annualised Return:", round(rp_min,2))
    print("Annualised Volatility:", round(sdp_min,2))
    print("\n")
    print(min_vol_allocation)
    
    plt.figure(figsize=(10, 7))
    plt.scatter(results[0,:],results[1,:],c=results[2,:],cmap='YlGnBu', marker='o', s=10, alpha=0.3)
    plt.colorbar()
    plt.scatter(sdp,rp,marker='*',color='r',s=500, label='Maximum Sharpe ratio')
    plt.scatter(sdp_min,rp_min,marker='*',color='g',s=500, label='Minimum volatility')

    target = np.linspace(rp_min, 0.32, 50)
    efficient_portfolios = efficient_frontier(mean_returns, cov_matrix, target)
    plt.plot([p['fun'] for p in efficient_portfolios], target, linestyle='-.', color='black', label='efficient frontier')
    plt.title('Calculated Portfolio Optimization based on Efficient Frontier')
    plt.xlabel('annualised volatility')
    plt.ylabel('annualised returns')
    plt.legend(labelspacing=0.8)

def display_ef_with_selected(mean_returns, cov_matrix, risk_free_rate,stocks_data):
    print("A")
    max_sharpe = max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate)
    sdp, rp = portfolio_annualised_performance(max_sharpe['x'], mean_returns, cov_matrix)
    max_sharpe_allocation = pd.DataFrame(max_sharpe.x,index=stocks_data.columns,columns=['allocation'])
    max_sharpe_allocation.allocation = [round(i*100,2)for i in max_sharpe_allocation.allocation]
    max_sharpe_allocation = max_sharpe_allocation.T
    print("B")

    min_vol = min_variance(mean_returns, cov_matrix)
    sdp_min, rp_min = portfolio_annualised_performance(min_vol['x'], mean_returns, cov_matrix)
    min_vol_allocation = pd.DataFrame(min_vol.x,index=stocks_data.columns,columns=['allocation'])
    min_vol_allocation.allocation = [round(i*100,2)for i in min_vol_allocation.allocation]
    min_vol_allocation = min_vol_allocation.T
    print("C")
    
    an_vol = np.std(returns) * np.sqrt(252)
    an_rt = mean_returns * 252
    
    print("------------------------")
    print("Maximum Sharpe Ratio Portfolio Allocation\n")
    print("Annualised Return:", round(rp,2))
    print("Annualised Volatility:", round(sdp,2))
    print("\n")
    print(max_sharpe_allocation)
    print("------------------------")
    print("Minimum Volatility Portfolio Allocation\n")
    print("Annualised Return:", round(rp_min,2))
    print("Annualised Volatility:", round(sdp_min,2))
    print("\n")
    print(min_vol_allocation)
    print("------------------------")
    print("Individual Stock Returns and Volatility\n")
    for i, txt in enumerate(table.columns):
        print(txt,":","annuaised return",round(an_rt[i],2),", annualised volatility:",round(an_vol[i],2))
    print("------------------------")
    
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(an_vol,an_rt,marker='o',s=200)

    for i, txt in enumerate(table.columns):
        ax.annotate(txt, (an_vol[i],an_rt[i]), xytext=(10,0), textcoords='offset points')
    ax.scatter(sdp,rp,marker='*',color='r',s=500, label='Maximum Sharpe ratio')
    ax.scatter(sdp_min,rp_min,marker='*',color='g',s=500, label='Minimum volatility')

    target = np.linspace(rp_min, 0.34, 50)
    efficient_portfolios = efficient_frontier(mean_returns, cov_matrix, target)
    ax.plot([p['fun'] for p in efficient_portfolios], target, linestyle='-.', color='black', label='efficient frontier')
    ax.set_title('Portfolio Optimization with Individual Stocks')
    ax.set_xlabel('annualised volatility')
    ax.set_ylabel('annualised returns')
    ax.legend(labelspacing=0.8)




def portolio_optimization2(tickers,total_mount):
    stocks_data = pd.DataFrame()
    inicio = "2015-01-01"
    fin = datetime.today().strftime('%Y-%m-%d')
    last_prices = {}
    for ticker in tickers:
        try:
            stocks_data[ticker] = web.DataReader(ticker+".MX", data_source='yahoo', start=inicio, end=fin)["Adj Close"]
            last_prices[ticker] = stocks_data.iloc[-1, stocks_data.columns.get_loc(ticker)]
        except Exception as e:
            print("Error al obtener los datos ")
            print(e)

    # Calculate expected returns and sample covariance
    mu = expected_returns.mean_historical_return(stocks_data)
    S = risk_models.sample_cov(stocks_data)
    
    # Optimize for maximal Sharpe ratio
    ef = EfficientFrontier(mu, S)
    raw_weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()

    ef.portfolio_performance(verbose=False)

    #print(cleaned_weights)

    allocation_dataframe = pd.DataFrame(cleaned_weights.items(), columns=['Ticker', 'Allocation %'])

#    if len(allocation_dataframe.index) >= 12:
#        mount = total_mount * 0.8
#    elif len(allocation_dataframe.index) >= 6: 
#        mount = total_mount * 0.6
#    elif len(allocation_dataframe.index) >= 3: 
#        mount = total_mount * 0.4
#    else: 
#        mount = total_mount * 0.2 
    mount = total_mount

    if len(allocation_dataframe.index) < 3:
        mount = total_mount * 0.7
        print("\nSetting amount limits due to the number of symbols\n")
    allocation_dataframe['Allocation %'] = allocation_dataframe['Allocation %'] * 100
    allocation_dataframe['Allocation $'] = allocation_dataframe['Allocation %'] * float(mount)/100  
    allocation_dataframe['LastPrice $'] = allocation_dataframe['Ticker'].map(last_prices)  
    allocation_dataframe['TitlesNum'] = allocation_dataframe['Allocation $'] / allocation_dataframe['LastPrice $']
    
    allocation_dataframe['TitlesNum'] = allocation_dataframe['TitlesNum'].fillna(0).astype(int)
    allocation_dataframe['Allocation $'] = allocation_dataframe.apply(lambda x: "{:,}".format(x['Allocation $']), axis=1)
    return allocation_dataframe




def portolio_optimization(tickers):
    mount = input("Escribe el monto >> ") 
    driver.get("https://www.portfoliovisualizer.com/optimize-portfolio")
    driver.find_element(By.CSS_SELECTOR, "#timePeriod_chosen span").click()
    driver.find_element(By.CSS_SELECTOR, ".active-result:nth-child(1)").click()
    driver.find_element(By.CSS_SELECTOR, "#startYear_chosen span").click()
    driver.find_element(By.CSS_SELECTOR, "#startYear_chosen .chosen-results").click()
    driver.find_element(By.CSS_SELECTOR, "#robustOptimization_chosen span").click()
    driver.find_element(By.CSS_SELECTOR, "#robustOptimization_chosen .active-result:nth-child(2)").click()
    index = 1
    for ticker in tickers:
        driver.find_element(By.ID, "symbol"+str(index)).send_keys(ticker)
        index = index + 1
    driver.find_element(By.ID, "submitButton").click()
    table = driver.find_element(By.XPATH,'/html/body/div[2]/div[3]/div[1]/div[1]/div[1]/table').get_attribute("outerHTML")
    dfs = pd.read_html(table)
    df1 = dfs[0].iloc[:-1]
    print("\n")
    df1.reset_index()
    df1['Allocation %'] = df1['Allocation %'].map(lambda x: str(x).replace("%",""))
    df1['Allocation %'] = df1['Allocation %'].astype('float') 
    df1['Allocation $'] = (df1['Allocation %']/100) * float(mount)
    return df1
    

def news_analisys():
    # Parameters
    n = 3  # the # of article headlines displayed per ticker
    tickers = ['AAPL', 'TSLA', 'AMZN']

    for ticker in tickers:
        print('\n')
        print('Recent News Headlines for {}: '.format(ticker))
        driver.get("https://mx.investing.com/search/?q=" + ticker)
        driver.find_element_by_xpath(
                '//*[@id="searchPageResultsTabs"]/*/a[contains(text(), "Noticias")]').click()
        for x in range(1, n):
            print(driver.find_element_by_xpath('//*[@id="fullColumn"]/div/div[4]/div[3]/div/div['+str(x)+']/div/p').get_attribute('innerHTML')+' '+driver.find_element_by_xpath('//*[@id="fullColumn"]/div/div[4]/div[3]/div/div['+str(x)+']/div/div/time').get_attribute('innerHTML'))
   

    # Get Data
    finwiz_url = 'https://finviz.com/quote.ashx?t='
    news_tables = {}

    for ticker in tickers:
        url = finwiz_url + ticker
        req = Request(url=url, headers={'user-agent': 'my-app/0.0.1'})
        resp = urlopen(req)
        html = BeautifulSoup(resp, features="lxml")
        news_table = html.find(id='news-table')
        news_tables[ticker] = news_table

    try:
        for ticker in tickers:
            df = news_tables[ticker]
            df_tr = df.findAll('tr')

            print('\n')
            print('Recent News Headlines for {}: '.format(ticker))

            for i, table_row in enumerate(df_tr):
                a_text = table_row.a.text
                td_text = table_row.td.text
                td_text = td_text.strip()
                print(a_text, '(', td_text, ')')
                if i == n-1:
                    break
    except KeyError:
        pass

    # Iterate through the news
    parsed_news = []
    for file_name, news_table in news_tables.items():
        for x in news_table.findAll('tr'):
            text = x.a.get_text()
            date_scrape = x.td.text.split()

            if len(date_scrape) == 1:
                timer = date_scrape[0]

            else:
                date = date_scrape[0]
                timer = date_scrape[1]

            ticker = file_name.split('_')[0]

            parsed_news.append([ticker, date, timer, text])

    # Sentiment Analysis
    analyzer = SentimentIntensityAnalyzer()

    columns = ['Ticker', 'Date', 'Time', 'Headline']
    news = pd.DataFrame(parsed_news, columns=columns)
    scores = news['Headline'].apply(analyzer.polarity_scores).tolist()

    df_scores = pd.DataFrame(scores)
    news = news.join(df_scores, rsuffix='_right')

    # View Data
    news['Date'] = pd.to_datetime(news.Date).dt.date

    unique_ticker = news['Ticker'].unique().tolist()
    news_dict = {name: news.loc[news['Ticker'] == name]
                 for name in unique_ticker}

    values = []
    for ticker in tickers:
        dataframe = news_dict[ticker]
        dataframe = dataframe.set_index('Ticker')
        dataframe = dataframe.drop(columns=['Headline'])
        print('\n')
        print(dataframe.head())

        mean = round(dataframe['compound'].mean(), 2)
        values.append(mean)

    df = pd.DataFrame(list(zip(tickers, values)), columns=[
                      'Ticker', 'Mean Sentiment'])
    df = df.set_index('Ticker')
    df = df.sort_values('Mean Sentiment', ascending=False)
    print('\n')
    print(df)


def analisys_result():
    print('\n\n\t' + OKGREEN + "RESULTADO DE ANALISIS:")

    df = pd.read_csv('analisis_tecnico_algoritmo.csv')
    print(OKGREEN + df)
    df = df.loc[(df['30 Minutes'] == 'Strong Buy') & (df['5 Minutes'] == 'Strong Buy') & (df['15 Minutes'] == 'Strong Buy') & (
        df['Hourly'] == 'Strong Buy') & (df['Daily'] == 'Strong Buy') & (df['Weekly'] == 'Strong Buy') & (df['Monthly'] == 'Strong Buy')].copy()
    print(df)
    names = df['Name'].tolist()
    df = df[['Name']]
    df['Precio estimado de compra $'] = "-"
    df['Precio estimado de venta $'] = "-"
    df['Variacion estimada %'] = "-"
    df['StockName'] = "-"
    df['Datetime'] = datetime.now().strftime("%x %X")

    index_for = 0
    for stock_name in names:
        time.sleep(3)
        driver.get("https://mx.investing.com/search/?q=" + stock_name)
        driver.find_element_by_xpath(
            '/html/body/div[5]/section/div/div[2]/div[2]/div[2]').click()
        time.sleep(3)
        driver.find_element_by_xpath(
            '/html/body/div[5]/section/div/div[3]/div[3]/div/*/span[contains(text(), "México")]').click()
        stock_name_found = driver.find_element_by_xpath(
            '/html/body/div[5]/section/div[7]/h2 ').get_attribute("innerText").replace('Panorama ', '')
        print(stock_name_found)

        driver.find_element_by_xpath(
            '//*[@id="pairSublinksLevel1"]/*/a[contains(text(), "Técnico")]').click()
        time.sleep(3)
        driver.find_element_by_xpath(
            '//*[@id="timePeriodsWidget"]/li[6]').click()
        time.sleep(2)
        table = driver.find_element_by_xpath(
            '/html/body/div[5]/section/div[10]/table').get_attribute("outerHTML")
        dfsb = pd.read_html(table)
        dfb = dfsb[0]
        print(dfb)
        expected_price_buy = (float(dfb.at[0, 'S2'])+float(dfb.at[0, 'S1']))/2
        print(format(expected_price_buy, '.2f'))
        expected_price_sell = (
            float(dfb.at[0, 'R2'])+float(dfb.at[0, 'R1']))/2
        print(format(expected_price_sell, '.2f'))
        expected_var = ((expected_price_sell/expected_price_buy)-1)*100
        print(format(expected_var, '.2f'))

        df.iat[index_for, 1] = format(expected_price_buy, '.2f')
        df.iat[index_for, 2] = format(expected_price_sell, '.2f')
        df.iat[index_for, 3] = format(expected_var, '.2f')
        df.iat[index_for, 4] = stock_name_found
        index_for += 1

    print(OKGREEN + df)
    print('\n' + WHITE)
    f = open('result.html', 'w')
    a = df.to_html()
    f.write(a)
    f.close()

    pdfkit.from_file('result.html', 'result.pdf')


def retrieve_top_reto():
    print('\t' + WARNING + "Accediendo a pulso del reto..." + WHITE)
    time.sleep(3)
    driver.get('https://www.retoactinver.com/RetoActinver/#/pulso')
    driver.refresh()
    time.sleep(3)
    driver.find_element_by_xpath(
        '//*[@id="botonCerrar"]/mat-icon').click()
    time.sleep(3)
    driver.find_element(
        By.CSS_SELECTOR, ".col-4:nth-child(3) > .btn-filtros").click()
    time.sleep(3)
    driver.find_element(By.CSS_SELECTOR, ".mat-form-field-infix").click()
    driver.find_element(By.CSS_SELECTOR, "#mat-option-5 > span").click()
    time.sleep(3)
    table = driver.find_element_by_xpath(
        '/html/body/app-root/block-ui/app-paginas/app-layout-principal/mat-sidenav-container/mat-sidenav-content/div[2]/div/pulso-reto/div/div[4]/mat-card/mat-card-content/mat-tab-group/div/mat-tab-body[1]/div/div/tabla-alzas-bajas/div/div/app-table/div/gt-column-settings/div/div/generic-table/table').get_attribute("outerHTML")
    dfs = pd.read_html(table)
    df = dfs[0]
    df.drop(['% de Variación'], axis=1, inplace=True)
    print(df)
    df.rename(columns={'Precio Actual': 'Variación'}, inplace=True)
    df['Variación'] = df['Variación'].str.replace('% de Variación ', '')
    df['Variación'] = df['Variación'].str.replace('%', '')
    df.rename(columns={'Emisora': 'Precio Historico'}, inplace=True)
    df['Precio Historico'] = df['Precio Historico'].str.replace(
        'Historico', '')
    df.rename(columns={'Historico': 'Precio Actual'}, inplace=True)
    df['Precio Actual'] = df['Precio Actual'].str.replace('Precio Actual', '')
    df.rename(columns={'Sort:': 'Emisora'}, inplace=True)
    df['Emisora'] = df['Emisora'].str.replace('Emisora', '')
    df['Emisora'] = df['Emisora'].str.replace(' *', '')
    print(df)
    df.to_csv('top_reto.csv', index=False, header=True, encoding="utf-8")


def write_to_html_file(df, title, filename):
    '''
    Write an entire dataframe to an HTML file with nice formatting.
    '''

    result = '''
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
    '''
    result += '<h2> %s </h2>\n' % title
    if type(df) == pd.io.formats.style.Styler:
        result += df.render()
    else:
        result += df.to_html(classes='wide', escape=False)
    result += '''
</body>
</html>
'''
    with open(filename, 'w') as f:
        f.write(result)



def day_trading_alerts(market_closing_hour):   
    driver = configure_firefox_driver_with_profile() 
    day_trading_stocks = []

 
    try:
        if not login_platform_investing(driver):
            print("Error starting session!")

        driver.get("https://mx.investing.com/") 
        WebDriverWait(driver, 100).until(EC.element_to_be_clickable((By.XPATH, '//*[@id="portfolioTopBarBtn"]'))).click()
        WebDriverWait(driver, 50).until(EC.element_to_be_clickable((By.CSS_SELECTOR, '#portfolioContainer_25863947 > div:nth-child(1) > a:nth-child(2)')))
        driver.find_element(By.CSS_SELECTOR, '#portfolioContainer_25863947 > div:nth-child(1) > a:nth-child(2)').click() 


            
        driver.find_element(By.CSS_SELECTOR, "#performance > a:nth-child(1)").click()
        time.sleep(3)
        table = driver.find_element_by_xpath('/html/body/div[5]/section/div[7]/div[7]/div[2]/div[2]/div/table').get_attribute("outerHTML")
        dfs = pd.read_html(table)
        df1 = dfs[0]
        df1.drop(['Unnamed: 0', 'Unnamed: 8'], axis=1, inplace=True, errors='ignore')

        driver.find_element(By.CSS_SELECTOR, "#technical > a:nth-child(1)").click()
        time.sleep(3)
        table = driver.find_element_by_xpath('/html/body/div[5]/section/div[7]/div[7]/div[2]/div[2]/div/table').get_attribute("outerHTML")
        dfs = pd.read_html(table)
        df2 = dfs[0]
        df2.drop(['Unnamed: 0', 'Unnamed: 10'], axis=1, inplace=True, errors='ignore')

        driver.find_element(By.CSS_SELECTOR, "#overview > a:nth-child(1)").click()
        time.sleep(3)
        table = driver.find_element_by_xpath('/html/body/div[5]/section/div[7]/div[7]/div[2]/div[2]/div/div/table').get_attribute("outerHTML")
        dfs = pd.read_html(table)
        df3 = dfs[0]
        df3.drop(['Unnamed: 0', 'Unnamed: 1', 'Unnamed: 19', 'Unnamed: 20'], axis=1, inplace=True, errors='ignore')
        
        WebDriverWait(driver, 50).until(EC.element_to_be_clickable((By.XPATH, '/html/body/div[5]/section/div[7]/div[7]/div[2]/div[2]/div/div/table')))
        table_elements_simbols = driver.find_elements(By.XPATH, "/html/body/div[5]/section/div[7]/div[7]/div[2]/div[2]/div/div/table/tbody/tr/td[4]/a")
        dicionary_simbols = {}
        for tag in table_elements_simbols:
            dicionary_simbols[str(tag.get_attribute('innerHTML'))] = str(tag.get_attribute('href'))        

        dfs = [df1, df2, df3]
        dfs = [x.set_index('Nombre') for x in dfs]
        df = pd.concat(dfs, axis=1, keys=range(1, len(dfs) + 1))
        df.columns = df.columns.map('{0[1]}{0[0]}'.format)
        df = df.reset_index()

        
        
        df['Anual1'] = df['Anual1'].map(lambda x: str(x)[:-1])
        df['Anual1'] = df['Anual1'].astype('float') 
        df['Mensual1'] = df['Mensual1'].map(lambda x: str(x)[:-1])
        df['Mensual1'] = df['Mensual1'].astype('float') 
        df['Semanal1'] = df['Semanal1'].map(lambda x: str(x)[:-1])
        df['Semanal1'] = df['Semanal1'].astype('float') 
        df['Diario1'] = df['Diario1'].map(lambda x: str(x).replace("+-","+")[:-1])
        df['Diario1'] = df['Diario1'].astype('float')
        df['% var.3'] = df['% var.3'].map(lambda x: str(x).replace("+-","+")[:-1])
        df['% var.3'] = df['% var.3'].astype('float') 
        df['3 años1'] = df['3 años1'].map(lambda x: str(x).replace("-","0").replace("%",""))
        df['3 años1'] = df['3 años1'].astype('float') 
        df['1 Año1'] = df['1 Año1'].map(lambda x: str(x)[:-1])
        df['1 Año1'] = df['1 Año1'].astype('float') 
         
        

        dayli_negative_mean = df.loc[df['Diario1']  < 0,'Diario1'].mean()
        weekly_negative_mean = df.loc[df['Semanal1']  < 0,'Semanal1'].mean()
        monthly_negative_mean = df.loc[df['Mensual1']  < 0,'Mensual1'].mean()


        #df = df[(df[['Anual1']] > -11.62).all(1)] 
        #df = df[(df[['Mensual1']] > -3.38).all(1)] 
        #df = df[(df[['Semanal1']] > -2.59).all(1)]         
        #df = df[(df[['Diario1']] > -1.99).all(1)]

        current_hour = datetime.now().hour
        cc = 1
        while (current_hour < market_closing_hour):

            for x in df['Símbolo3']:
                current_hour = datetime.now().hour
                current_time = datetime.now().strftime("%H:%M")

                mount = 1000000
                operation_mount = mount * 0.03
                operation_mount_limit = mount * 0.06
                print("Analizando {}".format(x))
                path_5min_chart_image = os.path.join(dn,"img",x+"-5min_chart_image.png")
                path_days_chart_image = os.path.join(dn,"img",x+"-days_chart_image.png")
                path_weekly_chart_image = os.path.join(dn,"img",x+"-weekly_chart_image.png")
                path_30min_chart_image = os.path.join(dn,"img",x+"-30min_chart_image.png")

                try:
                    if containsAny(dicionary_simbols[x],['?']):
                        chart_url = insert_string_before(dicionary_simbols[x], '-advanced-chart', '?')
                    else:
                        chart_url = dicionary_simbols[x] + '-advanced-chart'
                    driver.get(chart_url)
                    

                    driver.execute_script("arguments[0].scrollIntoView()", driver.find_element(By.XPATH, "/html/body/div[5]/section/div[7]/h2"))
                    driver.switch_to.frame(WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.TAG_NAME, "iframe"))))    


                    # Setting indicators
                    WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, "/html/body/div/div[1]/div/prochart/div/div[1]/div/div[1]/div/div[2]/div/ul/li[5]/a"))).click()
                    
                    WebDriverWait(driver, 15).until(EC.element_to_be_clickable((By.XPATH, "/html/body/div/div[1]/div/prochart/div/div[1]/div/div[1]/div/div[2]/div/ul/li[5]/div/ul/li[1]/a"))).click()  
                    WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, "/html/body/div/div[1]/div/prochart/div/div[1]/div/div[1]/div/div[2]/div/ul/li[5]/a"))).click()
                    
                    driver.find_element(By.XPATH, "/html/body/div/div[1]/div/prochart/div/div[1]/div/div[1]/div/div[2]/div/ul/li[5]/div/ul/li[5]/a").click()


                    # Select time days
                    WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, "/html/body/div/div[1]/div/prochart/div/div[1]/div/div[1]/div/div[3]/ul/li[15]/button"))).click()
                    
                    time.sleep(4)
                    
                    WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, "/html/body/div/div[1]/div/prochart/div/div[1]/div/div[4]/div/div[1]"))).screenshot(path_days_chart_image)  

                    
                    
                       
                    # Select time 5 minutes
                    WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, "/html/body/div/div[1]/div/prochart/div/div[1]/div/div[1]/div/div[3]/ul/li[20]/button"))).click()

                    time.sleep(4)


                    WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, "/html/body/div/div[1]/div/prochart/div/div[1]/div/div[4]/div/div[1]"))).screenshot(path_5min_chart_image) 
                   
                    
                       
                    # Select time 30min
                    WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, "/html/body/div/div[1]/div/prochart/div/div[1]/div/div[1]/div/div[3]/ul/li[17]/button"))).click()
                    time.sleep(4)

                    WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, "/html/body/div/div[1]/div/prochart/div/div[1]/div/div[4]/div/div[1]"))).screenshot(path_30min_chart_image)   

                                   
                       
                    # Select time weekly
                    WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, "/html/body/div/div[1]/div/prochart/div/div[1]/div/div[1]/div/div[3]/ul/li[21]/button"))).click()

                    time.sleep(4)     

                    WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, "/html/body/div/div[1]/div/prochart/div/div[1]/div/div[4]/div/div[1]"))).screenshot(path_weekly_chart_image)                       
                        
                except Exception as e: 
                    print(e)    
                    print("Error ejecutando {}".format(x))
                    pass



                try:
                    if containsAny(dicionary_simbols[x],['?']):
                        technical_data_url = insert_string_before(dicionary_simbols[x], '-technical', '?')
                    else:
                        technical_data_url = dicionary_simbols[x] + '-technical'

                                     
                    driver.get(technical_data_url)
                    
                    # Time dimention in 15 minutes
                    WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.XPATH, '/html/body/div[5]/section/div[8]/ul/li[2]/a')))

                    driver.find_element(By.XPATH, "/html/body/div[5]/section/div[8]/ul/li[2]/a").click()
                    WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.XPATH, '//*[@id="curr_table"]')))

                    rsi_15min = float(driver.find_element(By.XPATH, "/html/body/div[5]/section/div[10]/div[3]/table/tbody/tr[1]/td[2]").get_attribute('innerHTML'))
                    
                    volume = float(driver.find_element(By.XPATH, "/html/body/div[5]/section/div[4]/div[2]/div/ul/li[1]/span[2]/span").get_attribute('innerHTML').replace(',','').replace(',','').replace(',',''))
                    
                    current_value = float(driver.find_element(By.CSS_SELECTOR, "#last_last").get_attribute('innerHTML').replace(',','').replace(',','').replace(',',''))
                    
                    # Time dimention in 5 minutes
                    WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.XPATH, '/html/body/div[5]/section/div[8]/ul/li[1]/a')))
                    driver.find_element(By.XPATH, "/html/body/div[5]/section/div[8]/ul/li[1]/a").click()
                    

                    WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.XPATH, '//*[@id="curr_table"]'))) 

                    rsi_5min = float(driver.find_element(By.XPATH, "/html/body/div[5]/section/div[10]/div[3]/table/tbody/tr[1]/td[2]").get_attribute('innerHTML'))

                    sma10_5min = float(driver.find_element(By.CSS_SELECTOR, ".movingAvgsTbl > tbody:nth-child(2) > tr:nth-child(2) > td:nth-child(2)").get_attribute('innerHTML').split("<", 1)[0].replace(',','').replace(',','').replace(',',''))
                    sma20_5min = float(driver.find_element(By.CSS_SELECTOR, ".movingAvgsTbl > tbody:nth-child(2) > tr:nth-child(3) > td:nth-child(2)").get_attribute('innerHTML').split("<", 1)[0].replace(',','').replace(',','').replace(',',''))
                    sma50_5min =float(driver.find_element(By.CSS_SELECTOR, ".movingAvgsTbl > tbody:nth-child(2) > tr:nth-child(4) > td:nth-child(2)").get_attribute('innerHTML').split("<", 1)[0].replace(',','').replace(',','').replace(',',''))
                    sma100_5min =float(driver.find_element(By.CSS_SELECTOR, ".movingAvgsTbl > tbody:nth-child(2) > tr:nth-child(5) > td:nth-child(2)").get_attribute('innerHTML').split("<", 1)[0].replace(',','').replace(',','').replace(',',''))
                    sma200_5min =float(driver.find_element(By.CSS_SELECTOR, ".movingAvgsTbl > tbody:nth-child(2) > tr:nth-child(6) > td:nth-child(2)").get_attribute('innerHTML').split("<", 1)[0].replace(',','').replace(',','').replace(',',''))


                    fibonacci_resist1_5min =float(driver.find_element(By.CSS_SELECTOR, "table.closedTbl:nth-child(4) > tbody:nth-child(2) > tr:nth-child(1) > td:nth-child(6)").get_attribute('innerHTML').split("<", 1)[0].replace(',','').replace(',','').replace(',',''))
                    fibonacci_resist2_5min =float(driver.find_element(By.CSS_SELECTOR, "table.closedTbl:nth-child(4) > tbody:nth-child(2) > tr:nth-child(1) > td:nth-child(7)").get_attribute('innerHTML').split("<", 1)[0].replace(',','').replace(',','').replace(',',''))
                    fibonacci_resist3_5min =float(driver.find_element(By.CSS_SELECTOR, "table.closedTbl:nth-child(4) > tbody:nth-child(2) > tr:nth-child(1) > td:nth-child(8)").get_attribute('innerHTML').split("<", 1)[0].replace(',','').replace(',','').replace(',',''))
                    
                    estimated_earn_sma50_5min = round(percentage_change(current_value, sma50_5min),2)
                    estimated_loss = round(weekly_negative_mean,2)
                    stop_loss = round(current_value*(1-(abs(estimated_loss/100))),2)
                  
                    risk_earn_coffient = round(estimated_earn_sma50_5min/estimated_loss,2)

                    estimated_earn_5min_fibonacci_resist = round(percentage_change(current_value, fibonacci_resist1_5min),2)
                    no_operation_titles = round(operation_mount/current_value)
                    volume_per_hour = volume/8
                    estimated_volume = volume/200                   

                    if no_operation_titles > estimated_volume: 
                        no_operation_titles = estimated_volume
                        print(no_operation_titles)
                    if current_value > operation_mount  and current_value < operation_mount_limit :
                        no_operation_titles = 1
                        print(no_operation_titles)
                    elif current_value > operation_mount  and current_value > operation_mount_limit :
                        no_operation_titles = 0
                        print(no_operation_titles)


                    
                    #if  estimated_volume <= 10:
                    #    estimated_volume = round(estimated_volume/5)*5
                    #elif estimated_volume <= 100 :
                    #    estimated_volume = round(estimated_volume/10)*10
                    #elif estimated_volume 1000:
                    #    estimated_volume = round(estimated_volume/100)*100
                    #elif estimated_volume > 1000:
                    #    estimated_volume = 1000


 
                    #if  x not in day_trading_stocks:
                    #    day_trading_stocks.append(x)


                    #  
                    if rsi_5min > 65 and current_value > sma10_5min and (df.loc[df['Símbolo3'] == x, 'Mensual2'].iloc[0] == 'Compra' or df.loc[df['Símbolo3'] == x, 'Mensual2'].iloc[0] == 'Compra fuerte'):
                        
                        message = '\U0001F4CA Oportunidad de compra especulativa de {} \U0001F4CA \n\nCon un valor actual de ${}. El RSI de este símbolo tendencia de sobrecompra que podría ser de interés a las {} hrs.\n\nPara ver más información consulta: {}'.format(x,'{:,.2f}'.format(current_value),current_time,dicionary_simbols[x])
                        for chat_id in telegram_chat_ids:
                            with open(path_days_chart_image, 'rb') as f1, open(path_5min_chart_image, 'rb') as f2, open(path_30min_chart_image, 'rb') as f3  , open(path_weekly_chart_image, 'rb') as f4  :
                                files = bot.send_media_group(chat_id=chat_id, media=[telegram.InputMediaPhoto(f4,caption=message), telegram.InputMediaPhoto(f2) , telegram.InputMediaPhoto(f3), telegram.InputMediaPhoto(f1)])
                                
                           
                    #if rsi_15min < 35:
                    if rsi_15min < 35 and rsi_15min > 20 and estimated_earn_sma50_5min > 0.3 and no_operation_titles > 0:
                        
                        message = '\U0001F6A8 Oportunidad de compra especulativa de {} \U0001F6A8 \n\nCon un valor actual de ${}. El RSI de este símbolo señala lecturas de sobreventa a las {} hrs, se espera un "pull-back" en corto plazo.\n\nGanancia potencial: {}%\nPrecio objetivo: ${}\nPerdida potencial: {}%\nLímite de perdida sugerido: ${}\nRiesgo/Ganancia: {}\nVolumen de compra/venta: {} títulos\nVolumen de compra sugerido: {} títulos.\n\nPara ver más información consulta: {}'.format(x,'{:,.2f}'.format(current_value),current_time,str(estimated_earn_sma50_5min),'{:,.2f}'.format(sma50_5min),'{:,.2f}'.format(estimated_loss),str(stop_loss),'{:,.2f}'.format(risk_earn_coffient),'{:,.0f}'.format(volume),'{:,.0f}'.format(no_operation_titles),dicionary_simbols[x])
                        for chat_id in telegram_chat_ids:
                            with open(path_days_chart_image, 'rb') as f1, open(path_5min_chart_image, 'rb') as f2, open(path_30min_chart_image, 'rb') as f3  , open(path_weekly_chart_image, 'rb') as f4  :
                                files = bot.send_media_group(chat_id=chat_id, media=[telegram.InputMediaPhoto(f4,caption=message), telegram.InputMediaPhoto(f2) , telegram.InputMediaPhoto(f3), telegram.InputMediaPhoto(f1)])
                            #bot.send_photo(chat_id, caption=message, photo=open(path_5min_chart_image, 'rb')) 
                            #bot.send_photo(chat_id, photo=open(path_days_chart_image, 'rb'))     

                    if estimated_earn_5min_fibonacci_resist > 0.3 and no_operation_titles > 0  and  (df.loc[df['Símbolo3'] == x, '15 minutos2'].iloc[0] == 'Compra' or df.loc[df['Símbolo3'] == x, '15 minutos2'].iloc[0] == 'Compra fuerte') and (df.loc[df['Símbolo3'] == x, '5 minutos2'].iloc[0] == 'Compra' or df.loc[df['Símbolo3'] == x, '5 minutos2'].iloc[0] == 'Compra fuerte') and (df.loc[df['Símbolo3'] == x, 'Semanal2'].iloc[0] == 'Compra' or df.loc[df['Símbolo3'] == x, 'Semanal2'].iloc[0] == 'Compra fuerte'):

                        message = '\U0001F50D Oportunidad de compra especulativa de {} \U0001F50D \n\nCon un valor actual de ${}. El símbolo muestra tendencias de compra en el corto plazo a las {} hrs, se espera un avance en corto plazo de {}% en ${}, el símbolo tiene un volumen de compra/venta de {} unidades, aunque se sugiere realizar una compra por menos de {}.\n\nPara ver más información consulta: {}'.format(x,'{:,.2f}'.format(current_value),current_time,str(estimated_earn_5min_fibonacci_resist),'{:,.2f}'.format(fibonacci_resist2_5min),'{:,.0f}'.format(volume),'{:,.0f}'.format(no_operation_titles),dicionary_simbols[x])
                        for chat_id in telegram_chat_ids:
                            with open(path_days_chart_image, 'rb') as f1, open(path_5min_chart_image, 'rb') as f2, open(path_30min_chart_image, 'rb') as f3  , open(path_weekly_chart_image, 'rb') as f4  :
                                files = bot.send_media_group(chat_id=chat_id, media=[telegram.InputMediaPhoto(f4,caption=message), telegram.InputMediaPhoto(f2) , telegram.InputMediaPhoto(f3), telegram.InputMediaPhoto(f1)])
                           

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
    #tb.send_message(chat_id,  "Inicializando análisis de acciones usando estrategía swing trading con machine learning...")
    try:
        if not login_platform_investing(driver):
            print("Error starting session!")
            
        time.sleep(3)
        driver.get("https://mx.investing.com/")

        WebDriverWait(driver, 50).until(EC.element_to_be_clickable((By.XPATH, '//*[@id="portfolioTopBarBtn"]')))
        driver.find_element_by_xpath('//*[@id="portfolioTopBarBtn"]').click()
        WebDriverWait(driver, 50).until(EC.element_to_be_clickable((By.CSS_SELECTOR, '#portfolioContainer_25863947 > div:nth-child(1) > a:nth-child(2)')))
        driver.find_element(By.CSS_SELECTOR, '#portfolioContainer_25863947 > div:nth-child(1) > a:nth-child(2)').click()   
     
       
        driver.find_element(By.CSS_SELECTOR, "#performance > a:nth-child(1)").click()
        time.sleep(3)
        table = driver.find_element_by_xpath('/html/body/div[5]/section/div[7]/div[7]/div[2]/div[2]/div/table').get_attribute("outerHTML")
        dfs = pd.read_html(table)
        df1 = dfs[0]
        df1.drop(['Unnamed: 0', 'Unnamed: 8'], axis=1, inplace=True, errors='ignore')
        #print(df.columns)

        #df.to_csv('analisis_tecnico_performance.csv',  index=False, header=True, encoding="utf-8")
        driver.find_element(By.CSS_SELECTOR, "#technical > a:nth-child(1)").click()
        time.sleep(3)
        table = driver.find_element_by_xpath('/html/body/div[5]/section/div[7]/div[7]/div[2]/div[2]/div/table').get_attribute("outerHTML")
        dfs = pd.read_html(table)
        df2 = dfs[0]
        df2.drop(['Unnamed: 0', 'Unnamed: 10'], axis=1, inplace=True, errors='ignore')
        #print(df.columns)
        #df.to_csv('analisis_tecnico_algoritmo.csv',index=False, header=True, encoding="utf-8")


        driver.find_element(By.CSS_SELECTOR, "#overview > a:nth-child(1)").click()
        time.sleep(3)
        table = driver.find_element_by_xpath('/html/body/div[5]/section/div[7]/div[7]/div[2]/div[2]/div/div/table').get_attribute("outerHTML")
        dfs = pd.read_html(table)
        df3 = dfs[0]
        df3.drop(['Unnamed: 0', 'Unnamed: 1', 'Unnamed: 19', 'Unnamed: 20'], axis=1, inplace=True, errors='ignore')
        #print(df.columns)

        # Create dicionary investing simbols portfolio
        WebDriverWait(driver, 50).until(EC.element_to_be_clickable((By.XPATH, '/html/body/div[5]/section/div[7]/div[7]/div[2]/div[2]/div/div/table')))
        table_elements_simbols = driver.find_elements(By.XPATH, "/html/body/div[5]/section/div[7]/div[7]/div[2]/div[2]/div/div/table/tbody/tr/td[4]/a")
        dicionary_simbols = {}
        for tag in table_elements_simbols:
            dicionary_simbols[str(tag.get_attribute('innerHTML'))] = str(tag.get_attribute('href'))
        
        #tb.send_message(chat_id,  "Analizando acciones...")  
        dfs = [df1, df2, df3]
        dfs = [x.set_index('Nombre') for x in dfs]
        df = pd.concat(dfs, axis=1, keys=range(1, len(dfs) + 1))
        df.columns = df.columns.map('{0[1]}{0[0]}'.format)
        df = df.reset_index()

        #print("Lista de empresas a analizar: {}".format(df['Nombre']))

        df['Fecha'] = datetime.now().strftime("%x %X")
        df['3 años1'] = df['3 años1'].map(lambda x: str(x).replace("-","0"))
        df['3 años1'] = df['3 años1'].map(lambda x: str(x).replace("%",""))
   
        df['3 años1'] = df['3 años1'].astype('float') 
        df['1 Año1'] = df['1 Año1'].map(lambda x: str(x)[:-1])
   
        df['1 Año1'] = df['1 Año1'].astype('float') 
        df['Anual1'] = df['Anual1'].map(lambda x: str(x)[:-1])
     
        df['Anual1'] = df['Anual1'].astype('float') 
        df['Mensual1'] = df['Mensual1'].map(lambda x: str(x)[:-1])
      
        df['Mensual1'] = df['Mensual1'].astype('float') 
        df['Semanal1'] = df['Semanal1'].map(lambda x: str(x)[:-1])
        
        df['Semanal1'] = df['Semanal1'].astype('float') 
        df['Diario1'] = df['Diario1'].map(lambda x: str(x).replace("+-","+")[:-1])
        
        df['Diario1'] = df['Diario1'].astype('float')
        df['% var.3'] = df['% var.3'].map(lambda x: str(x).replace("+-","+")[:-1])
      
        df['% var.3'] = df['% var.3'].astype('float') 


        dayli_negative_mean = df.loc[df['Diario1']  < 0,'Diario1'].mean()
        weekly_negative_mean = df.loc[df['Semanal1']  < 0,'Semanal1'].mean()
        monthly_negative_mean = df.loc[df['Mensual1']  < 0,'Mensual1'].mean()
        annual_negative_mean = df.loc[df['Anual1']  < 0,'Anual1'].mean()
        one_year_negative_mean = df.loc[df['1 Año1']  < 0,'1 Año1'].mean()
        #pd.set_option('display.max_rows', None)
        #print(df['3 años1'] )
        three_years_negative_mean = df.loc[df['3 años1']  < 0,'3 años1'].replace(0, np.NaN).mean()



        df = df[(df[['3 años1']] > 0).all(1)] # Mean of negatives
        df = df[(df[['1 Año1']] > one_year_negative_mean).all(1)] 
        df = df[(df[['Anual1']] > annual_negative_mean).all(1)] 
        df = df[(df[['Mensual1']] > monthly_negative_mean).all(1)] 
        df = df[(df[['Semanal1']] > weekly_negative_mean).all(1)]         
        df = df[(df[['Diario1']] > dayli_negative_mean).all(1)]
       
        df = df[(df[['Mensual2']] == 'Compra fuerte').all(1) | (df[['Mensual2']] == 'Compra').all(1) ]
        df = df[(df[['Semanal2']] == 'Compra fuerte').all(1) | (df[['Semanal2']] == 'Compra').all(1) ]
        df = df[(df[['Diario2']] == 'Compra fuerte').all(1) | (df[['Diario2']] == 'Compra').all(1)]

        
       
        for x in df['Símbolo3']:
            #tb.send_message(chat_id,  "Analizando {} ...".format(x)  ) 
            try:
                if containsAny(dicionary_simbols[x],['?']):
                    technical_data_url = insert_string_before(dicionary_simbols[x], '-technical', '?')
                else:
                    technical_data_url = dicionary_simbols[x] + '-technical'
                driver.get(technical_data_url)
                WebDriverWait(driver, 50).until(EC.element_to_be_clickable((By.XPATH, '/html/body/div[5]/section/div[8]/ul/li[6]/a')))
                driver.find_element(By.XPATH, "/html/body/div[5]/section/div[8]/ul/li[6]/a").click()
                WebDriverWait(driver, 15).until(EC.element_to_be_clickable((By.XPATH, '/html/body/div[5]/section/div[10]/table')))
                p1 = float(driver.find_element(By.XPATH, "/html/body/div[5]/section/div[10]/table/tbody/tr[2]/td[6]").get_attribute('innerHTML'))
                p2 = float(driver.find_element(By.XPATH, "/html/body/div[5]/section/div[10]/table/tbody/tr[2]/td[7]").get_attribute('innerHTML'))
                p3 = float(driver.find_element(By.XPATH, "/html/body/div[5]/section/div[10]/table/tbody/tr[2]/td[8]").get_attribute('innerHTML'))
                pe = round((p1 + p2 + p3)/3,2)
                df.loc[df['Símbolo3'] == x, 'PeEstimado'] = pe
                try:
                    df.loc[df['Símbolo3'] == x, 'ML Prediction'] = predict_machine_daily(x)
                except Exception as e: 
                   df.loc[df['Símbolo3'] == x, 'ML Prediction'] = 'NO DISPONIBLE'
                   print(e)  
            except Exception as e: 
                print(e)    
                print("Error ejecutando {}".format(x)  )
                pass    
   
        df = df[(df[['ML Prediction']] == 'Compra').all(1)]  
        df['GanEstimada %'] = round(percentage_change(df['Último3'],df['PeEstimado']),2)
        df = df[(df[['GanEstimada %']] > 0.5).all(1)]   
        df['PeVentaCalc'] = round(df['Último3'] * 1.005,2)        
        
        dict = {'Símbolo3': 'Símbolo','% var.3':'% Cambio'}
        df.rename(columns=dict,inplace=True)  
        
        print(df[['Símbolo','Nombre','Último3','PeEstimado','GanEstimada %','PeVentaCalc','ML Prediction']])
                
        tickers = df['Símbolo'].astype(str).values.tolist() 
        print(len(tickers))
        if len(tickers) >= 1: 
            #tb.send_message(chat_id, "Ejecutando algoritmo de optimización de portafolio...")           
            df_allocations = portolio_optimization2(tickers,1000000)
            df_result = pd.merge(df_allocations,df,left_on='Ticker',right_on='Símbolo')            
            df_result['Títulos'] = df_result['Allocation $']/df_result['Último3']
            df_result['Títulos'] = df_result['Títulos'].astype('int32') 
            df_result = df_result.loc[(df_result[['Títulos']] != 0).all(axis=1)]   
            df_result['Allocation $'] = df_result['Allocation $'].round(decimals = 2)           
            print(df_result[['Símbolo','Nombre','Títulos','Allocation $','Último3','PeEstimado','GanEstimada %','PeVentaCalc']])
            df_result = df_result.sort_values(['GanEstimada %'], ascending=[False])
            print(df_result[['Símbolo','Nombre','Títulos','Allocation $','Último3','PeEstimado','GanEstimada %','PeVentaCalc']])
            result_df = df_result[['Símbolo','Nombre','Títulos','% Cambio']]                       
            path_file_name = os.path.join(dn,'Sugerencias.html')
            write_to_html_file(result_df, 'Sugerencias para Reto Actinver', path_file_name)   
             

            for chat_id in telegram_chat_ids:
                doc = open(path_file_name, 'rb')
                tb.send_message(chat_id, "Sugerencias de compra:")                 
                tb.send_document(chat_id, doc)
        else :
            for chat_id in telegram_chat_ids:
                tb.send_message(chat_id, "Intente en otro momento, no hay símbolos bursátiles que pasen las pruebas del algorítmo en este momento.") 
        
        driver.close()
        exit()

    except Exception as e: 
        print(e)
        driver.close()
        exit()   

def daily_quizz_solver(username, password, email,index_quizz_answer):
    driver = configure_firefox_driver_no_profile()

    selector = ['/html/body/div[1]/div[2]/div/mat-dialog-container/modal-quiz/mat-dialog-content/div/div[2]/div/div[2]/div/div[1]/div/p','/html/body/div[1]/div[2]/div/mat-dialog-container/modal-quiz/mat-dialog-content/div/div[2]/div/div[2]/div/div[2]/div/p','/html/body/div[1]/div[2]/div/mat-dialog-container/modal-quiz/mat-dialog-content/div/div[2]/div/div[2]/div/div[3]/div/p']
    screenshot = os.path.join(dn,"img",username+"-quizz.png")
    message = "El usuario {} respondió correctamente".format(username)
    
    try:
        is_logged_flag = login_platform_actinver(driver,username,password,email)
        if is_logged_flag:                     
           solved = False           
           while not solved:
              try:
                driver.get('https://www.retoactinver.com/RetoActinver/#/inicio')
                close_popoup_browser(driver)         
                WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, '/html/body/app-root/block-ui/app-paginas/app-layout-principal/mat-sidenav-container/mat-sidenav-content/div[2]/div/app-inicio/div[6]/div[1]/app-tarjeta-inicio/mat-card/mat-card-footer/div/div[2]/button'))).click()
                WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, selector[index_quizz_answer]))).click()
                WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, '/html/body/div[1]/div[2]/div/mat-dialog-container/modal-quiz/mat-dialog-content/div/div[2]/div/div[3]/button'))).click()                
                driver.save_screenshot(screenshot)  
                bot.send_photo(telegram_chat_ids[0], caption=message,photo=open(screenshot, 'rb'))   
                solved = True                
              except Exception as e:
                print(e) 
                print("Cannot solve quizz at this moment...")                
                print("Browsing to maintain active session...")
                driver.get('https://www.retoactinver.com/RetoActinver/#/portafolio')
           driver.close()
           exit() 
        else:
            print("Error starting session!")            
    except Exception as e: 
        print(e)
        exit()  

def solve_daily_quizz():
    threading.Thread(target=daily_quizz_solver,args=('osvaldohm9','Os23valdo1.','osvaldo.hdz.m@outlook.com',0)).start()
    threading.Thread(target=daily_quizz_solver,args=('Gabriela62','copito55','hernandezsg62@outlook.com',1)).start()
    threading.Thread(target=daily_quizz_solver,args=('kikehedz22','E93h14M01','enrique45_v@hotmail.com',2)).start()
    
 

def main_menu():
    os.system('cls')
    print ("Selecciona una opción")
    print ("\t1 - Iniciar sesión en la plataforma del reto")
    print ("\t2 - Mostrar sugerencias de compra")
    print ("\t3 - Mostrar portafolio actual")
    print ("\t4 - Comprar acciones")
    print ("\t5 - Mostrar ordenes")
    print ("\t6 - Monitorear venta")
    print ("\t7 - Vender todas las posiciones en portafolio (a precio del mercado)")
    print ("\t8 - Restaurar sesión en plataforma del reto")
    print ("\t9 - Optimizar portafolio")
    print ("\t0 - Salir")

    opcionmain_menu = input("Selecciona una opción >> ") 
    if opcionmain_menu=="1":           
        login_platform_actinver()
        input("\nPulsa una tecla para continuar")
        main_menu()
    elif opcionmain_menu=="2":
        sub_main_menu_2()
    elif opcionmain_menu=="3":        
        show_portfolio()
        input("\nPulsa una tecla para continuar")
        main_menu()
    elif opcionmain_menu=="4":
        buy_stocks()
        input("\nPulsa una tecla para continuar")
        main_menu()
    elif opcionmain_menu=="5":
        show_orders()
        input("\nPulsa una tecla para continuar")
        main_menu()
    elif opcionmain_menu=="6":        
        input("\nPulsa una tecla para continuar")
        main_menu()
    elif opcionmain_menu=="7":        
        input("\nPulsa una tecla para continuar")
        main_menu()
    elif opcionmain_menu=="9":
        input_amount = int(input("Enter the amount to invest (i.e. 800000): "))
        input_tickers = str(input("Enter tickers separated by commas (i.e. OMAB,AAPL,BRKB,MSFT): "))
        input_tickers = input_tickers.upper()
        input_tickers_list = input_tickers.split (",")
        tickers = []
        for i in input_tickers_list:
            tickers.append(i)
        print("\nTickers list : ", tickers)
        print("\nOptimizing portfolio...")
        try:
            allocation_dataframe = portolio_optimization2(tickers,input_amount)
            print(allocation_dataframe)
        except Exception as e: 
            print(e)
        input("\nPulsa una tecla para continuar")        
        main_menu()
    elif opcionmain_menu=="0":
        #logout_platform_actinver()
        #driver.close()
        exit()
    else:
        input("No has pulsado ninguna opción correcta...\nPulsa una tecla para continuar")
        main_menu()

def sub_main_menu_2():
    os.system('cls')
    print ("Selecciona una opción")
    print ("\t1 - Analizar acciones usando estrategía day trading ")
    print ("\t2 - Analizar acciones usando estrategía swing trading simple 1")
    print ("\t3 - Analizar acciones usando estrategía swing trading machine learning")
    print ("\t4 - Analizar acciones usando estrategía swing trading simple 2")
    print ("\t5 - Analizar acciones usando bandas de Bollinger y MACD")
    print ("\t0 - Cancelar")

    opcionmain_menu = input("Selecciona una opción >> ") 
    if opcionmain_menu=="1":           
        day_trading_strategy()
        input("\nPulsa una tecla para continuar")
        main_menu()
    elif opcionmain_menu=="2":
        swing_trading_strategy()
        input("\nPulsa una tecla para continuar")
        main_menu()
    elif opcionmain_menu=="3":
        swing_trading_strategy_machine()
        input("\nPulsa una tecla para continuar")
        main_menu()

    elif opcionmain_menu=="4":
        swing_trading_strategy2()
        input("\nPulsa una tecla para continuar")
        main_menu()
    elif opcionmain_menu=="0":
        main_menu()
    else:
        print ("")
        input("No has pulsado ninguna opción correcta...\nPulsa una tecla para continuar")
        sub_main_menu_2()
 

if command == 'swing_trading_recommendations':
    swing_trading_recommendations()
elif command == 'day_trading_alerts':
    day_trading_alerts(16)
elif command == 'solve_daily_quizz':
    solve_daily_quizz()
elif command == 'optimize_portfolio':
    # input comma separated elements as string 
    input_tickers = str(input("Enter tickers separated by commas: "))
    input_tickers = input_tickers.upper()
    print("Tickers: ", input_tickers)
    # conver to the list
    input_tickers_list = input_tickers.split (",")
    print( "List: ", input_tickers_list)
    # convert each element as integers
    tickers = []
    for i in input_tickers_list:
        tickers.append(i)
    # print list as integers
    print("list (li) : ", tickers)
    portolio_optimization2(tickers,10000)
else :
   main_menu()
#retrieve_top_reto()
#retrieve_data_reto_capitales()
#retrieve_data_reto_portafolio()
#analisys_result()
#news_analisys()


