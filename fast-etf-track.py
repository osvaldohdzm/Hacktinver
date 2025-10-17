import yfinance as yf
import pandas as pd
import datetime

# Solicitar al usuario la lista de ETF separados por comas
etf_list = input("Ingrese la lista de ETF separados por comas: ")
etf_symbols = [symbol.strip() for symbol in etf_list.split(',')]

# Fechas de inicio y fin
start_date = "2020-08-25"
end_date = "2023-08-25"

# Obtener la fecha actual
current_date = datetime.datetime.now()

for etf_symbol in etf_symbols:
    # Obtener datos del ETF de Yahoo Finance
    etf_data = yf.download(etf_symbol, start=start_date, end=end_date)
    
    # Calcular los retornos diarios
    etf_data['Daily_Return'] = etf_data['Adj Close'].pct_change()
    
    # Agregar una columna para el día de la semana
    etf_data['Day_of_Week'] = etf_data.index.dayofweek
    
    # Agregar una columna para el mes
    etf_data['Month'] = etf_data.index.month
    
    # Calcular el número de semana del año usando isocalendar()
    etf_data['Week_of_Year'] = etf_data.index.isocalendar().week
    
    # Calcular promedio de retornos por día de la semana
    avg_returns_by_day = etf_data.groupby('Day_of_Week')['Daily_Return'].mean()
    
    # Calcular promedio de retornos por mes
    avg_returns_by_month = etf_data.groupby('Month')['Daily_Return'].mean()
    
    # Calcular promedio de retornos por número de semana del año
    avg_returns_by_week = etf_data.groupby('Week_of_Year')['Daily_Return'].mean()
    
    # Obtener los nombres de los días de la semana
    day_names = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes"]
    
    print(f"\nAnálisis de retornos para {etf_symbol}:")
    
    # Mostrar promedio de retornos por día de la semana
    print("Promedio de retornos por día de la semana:")
    for day_idx, day_name in enumerate(day_names):
        asterisk = "*" if current_date.weekday() == day_idx else ""
        print(f"{asterisk}{day_name}: {avg_returns_by_day[day_idx]:.4f}")
    
    # Mostrar promedio de retornos por mes
    print("\nPromedio de retornos por mes:")
    for month_idx in avg_returns_by_month.index:
        month_name = datetime.date(1900, month_idx, 1).strftime('%B')
        asterisk = "*" if current_date.month == month_idx else ""
        print(f"{asterisk}{month_name}: {avg_returns_by_month[month_idx]:.4f}")
    
    # Mostrar promedio de retornos por número de semana del año
    print("\nPromedio de retornos por número de semana del año:")
    for week_idx in avg_returns_by_week.index:
        asterisk = "*" if current_date.isocalendar()[1] == week_idx else ""
        print(f"{asterisk}Semana {week_idx}: {avg_returns_by_week[week_idx]:.4f}")
