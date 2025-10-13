#!/usr/bin/env python3
"""
Script para corregir TODAS las llamadas a yf.download para usar exclusivamente tickers mexicanos (.MX)
"""

import re

def fix_yfinance_calls():
    """Corrige todas las llamadas a yf.download en hacktinver.py"""
    
    # Leer el archivo
    with open('hacktinver.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Patrones a corregir
    patterns = [
        # Patrón 1: yf.download(ticker, ...)
        (r'yf\.download\(ticker,', 'yf.download(normalize_ticker_to_mx(ticker),'),
        
        # Patrón 2: yf.download(symbol, ...)
        (r'yf\.download\(symbol,', 'yf.download(normalize_ticker_to_mx(symbol),'),
        
        # Patrón 3: yf.download(etf, ...)
        (r'yf\.download\(etf,', 'yf.download(normalize_ticker_to_mx(etf),'),
        
        # Patrón 4: Agregar progress=False donde no esté
        (r'yf\.download\(([^)]+)\)(?![^,]*progress)', r'yf.download(\1, progress=False)'),
    ]
    
    # Aplicar correcciones
    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content)
    
    # Escribir el archivo corregido
    with open('hacktinver.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Todas las llamadas a yf.download han sido corregidas para usar tickers mexicanos (.MX)")

if __name__ == "__main__":
    fix_yfinance_calls()