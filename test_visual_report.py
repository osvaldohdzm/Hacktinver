#!/usr/bin/env python3
"""
Script de prueba para el reporte visual
"""

import sys
import os
import pandas as pd
import numpy as np
import yfinance as yf

# Simular datos para prueba rápida
def test_process_data():
    """Prueba rápida del procesamiento de datos"""
    print("🧪 Probando procesamiento de datos...")
    
    # Datos de prueba
    test_data = {
        'QQQ': yf.download('QQQ', period='1mo', progress=False)
    }
    
    if test_data['QQQ'].empty:
        print("❌ No se pudieron descargar datos de prueba")
        return False
    
    # Simular el procesamiento
    try:
        df = test_data['QQQ']
        close = df['Close']
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        
        close_val = float(close.iloc[-1])
        prev_close_val = float(close.iloc[-2])
        
        analysis = {
            'precio_actual': close_val,
            'precio_anterior': prev_close_val,
            'cambio_1d': (close_val - prev_close_val) / prev_close_val * 100,
            'volatilidad': float(close.pct_change().std() * np.sqrt(252) * 100),
            'tendencia': 'Alcista' if close_val > prev_close_val else 'Bajista'
        }
        
        print(f"✅ Datos procesados correctamente: {analysis}")
        return True
        
    except Exception as e:
        print(f"❌ Error en procesamiento: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Iniciando pruebas del reporte visual...")
    
    # Prueba básica de procesamiento
    if test_process_data():
        print("✅ Todas las pruebas pasaron!")
    else:
        print("❌ Algunas pruebas fallaron")
        
    print("\n💡 Para usar el reporte completo, ejecuta hacktinver.py y selecciona 'Reporte Visual'")