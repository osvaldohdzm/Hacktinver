"""
Reporte Visual HTML - Dashboard de Trading Avanzado
Genera gráficos interactivos para análisis técnico y decisiones de trading
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

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
    
    print("🎯 Generando Reporte Visual HTML...")
    
    # ETFs por defecto para análisis
    default_etfs = [
        'QQQ', 'SPY', 'IWM', 'VTI', 'GLD', 'TLT', 'XLE', 'XLF', 'XLK', 'XLV',
        'SOXL', 'TQQQ', 'SPXL', 'FAS', 'TNA', 'TECL', 'SOXS', 'SQQQ', 'SPXS'
    ]
    
    # Solicitar ETFs al usuario
    user_input = input(f"Ingresa ETFs separados por comas (Enter para usar por defecto): ").strip()
    if user_input:
        etfs = [etf.strip().upper() for etf in user_input.split(',')]
    else:
        etfs = default_etfs
        print(f"Usando ETFs por defecto: {', '.join(etfs[:10])}...")
    
    # Descargar datos
    print("📊 Descargando datos de mercado...")
    data = {}
    for etf in etfs:
        try:
            df = yf.download(etf, period="3mo", interval="1d")
            if not df.empty:
                data[etf] = df
        except Exception as e:
            print(f"Error descargando {etf}: {e}")
    
    if not data:
        print("❌ No se pudieron descargar datos. Abortando...")
        return
    
    # Procesar datos para análisis
    analysis_data = process_market_data(data)
    
    # Crear gráficos
    html_content = generate_html_dashboard(analysis_data, etfs)
    
    # Guardar archivo HTML
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"data/reporte_visual_{timestamp}.html"
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ Reporte generado: {filename}")
    
    # Abrir en navegador
    try:
        import webbrowser
        webbrowser.open(filename)
        print("🌐 Abriendo reporte en navegador...")
    except:
        print("💡 Abre manualmente el archivo HTML en tu navegador")

def process_market_data(data):
    """Procesa los datos de mercado para generar métricas de análisis"""
    
    analysis = {}
    
    for etf, df in data.items():
        try:
            # Datos básicos
            close = df['Close'].iloc[-1]
            prev_close = df['Close'].iloc[-2]
            high = df['High'].iloc[-1]
            low = df['Low'].iloc[-1]
            volume = df['Volume'].iloc[-1]
            
            # Promedios móviles
            sma_20 = df['Close'].rolling(20).mean().iloc[-1]
            sma_50 = df['Close'].rolling(50).mean().iloc[-1]
            
            # Volatilidad
            returns = df['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) * 100  # Anualizada
            
            # Momentum y desviación
            price_change_1d = (close - prev_close) / prev_close * 100
            price_change_5d = (close - df['Close'].iloc[-6]) / df['Close'].iloc[-6] * 100
            deviation_sma20 = (close - sma_20) / sma_20 * 100
            
            # Volumen relativo
            avg_volume = df['Volume'].rolling(20).mean().iloc[-1]
            volume_ratio = volume / avg_volume if avg_volume > 0 else 1
            
            # Rango de precio
            price_range = (high - low) / close * 100
            
            analysis[etf] = {
                'precio_actual': close,
                'precio_anterior': prev_close,
                'maximo': high,
                'minimo': low,
                'volumen': volume,
                'sma_20': sma_20,
                'sma_50': sma_50,
                'volatilidad': volatility,
                'cambio_1d': price_change_1d,
                'cambio_5d': price_change_5d,
                'desviacion_sma20': deviation_sma20,
                'volumen_relativo': volume_ratio,
                'rango_precio': price_range,
                'tendencia': 'Alcista' if close > sma_20 > sma_50 else 'Bajista'
            }
            
        except Exception as e:
            print(f"Error procesando {etf}: {e}")
            continue
    
    return analysis

def generate_html_dashboard(analysis_data, etfs):
    """Genera el HTML completo con todos los gráficos"""
    
    # Crear DataFrame para análisis
    df = pd.DataFrame(analysis_data).T
    
    # 1. Heatmap de Momentum y Desviación
    heatmap_html = create_momentum_heatmap(df)
    
    # 2. Gráfico de Rango de Volatilidad
    range_chart_html = create_price_range_chart(df)
    
    # 3. Scatterplot de Desviación y Volumen
    scatter_html = create_deviation_volume_scatter(df)
    
    # 4. Dashboard de Pairs Trading
    pairs_html = create_pairs_trading_dashboard(analysis_data)
    
    # 5. Bar Chart de Rendimiento y Riesgo
    risk_return_html = create_risk_return_chart(df)
    
    # 6. Dashboard combinado
    combined_html = create_combined_dashboard(df)
    
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
                <p>Análisis Visual Completo • {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</p>
                <p>ETFs Analizados: {len(analysis_data)} • Datos en Tiempo Real</p>
            </div>
            
            <div class="legend">
                <h3>🎯 Guía de Interpretación Rápida:</h3>
                <p><strong>🟢 Verde intenso + Alto volumen</strong> → Comprar / Tendencia fuerte</p>
                <p><strong>🔴 Rojo intenso + Alto volumen</strong> → Vender o esperar reversión</p>
                <p><strong>⚡ Desviación extrema en pares</strong> → Oportunidad de arbitraje estadístico</p>
            </div>
            
            <div class="grid">
                <div class="chart-container">
                    <div class="chart-title">1️⃣ Heatmap de Momentum y Desviación</div>
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
                    <div class="chart-title">5️⃣ Rendimiento vs Riesgo</div>
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
            
            <div class="legend">
                <h3>📈 Resumen Ejecutivo:</h3>
                <p><strong>Oportunidades de Compra:</strong> {get_buy_opportunities(df)}</p>
                <p><strong>Alertas de Venta:</strong> {get_sell_alerts(df)}</p>
                <p><strong>Pairs Trading:</strong> {get_pairs_opportunities(analysis_data)}</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html_template

def create_momentum_heatmap(df):
    """Crea heatmap de momentum y desviación"""
    
    # Preparar datos para heatmap
    heatmap_data = df[['cambio_1d', 'cambio_5d', 'desviacion_sma20', 'volumen_relativo']].round(2)
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=['Cambio 1D (%)', 'Cambio 5D (%)', 'Desv. SMA20 (%)', 'Vol. Relativo'],
        y=heatmap_data.index,
        colorscale='RdYlGn',
        zmid=0,
        text=heatmap_data.values,
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="Momentum y Desviación por ETF",
        height=400,
        font=dict(size=12)
    )
    
    return fig.to_html(include_plotlyjs=False, div_id="heatmap")

def create_price_range_chart(df):
    """Crea gráfico de rango de precios con volumen"""
    
    fig = go.Figure()
    
    # Barras de rango (High-Low)
    fig.add_trace(go.Bar(
        x=df.index,
        y=df['rango_precio'],
        name='Rango Precio (%)',
        marker_color=df['volumen_relativo'],
        marker_colorscale='Viridis',
        text=df['rango_precio'].round(2),
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Rango de Volatilidad Diaria",
        xaxis_title="ETFs",
        yaxis_title="Rango de Precio (%)",
        height=400,
        showlegend=True
    )
    
    return fig.to_html(include_plotlyjs=False, div_id="range_chart")

def create_deviation_volume_scatter(df):
    """Crea scatterplot de desviación vs volumen"""
    
    fig = go.Figure()
    
    # Scatter plot con colores por momentum
    colors = ['red' if x < 0 else 'green' for x in df['cambio_1d']]
    
    fig.add_trace(go.Scatter(
        x=df['desviacion_sma20'],
        y=df['volumen_relativo'],
        mode='markers+text',
        text=df.index,
        textposition='top center',
        marker=dict(
            size=df['volatilidad'] * 2,  # Tamaño por volatilidad
            color=colors,
            opacity=0.7,
            line=dict(width=2, color='white')
        ),
        name='ETFs'
    ))
    
    fig.update_layout(
        title="Desviación vs Volumen Relativo",
        xaxis_title="Desviación SMA20 (%)",
        yaxis_title="Volumen Relativo",
        height=400
    )
    
    return fig.to_html(include_plotlyjs=False, div_id="scatter")

def create_pairs_trading_dashboard(analysis_data):
    """Crea dashboard de pairs trading"""
    
    # Pares correlacionados comunes
    pairs = [
        ('QQQ', 'TQQQ'), ('SPY', 'SPXL'), ('IWM', 'TNA'),
        ('XLF', 'FAS'), ('XLK', 'TECL'), ('SOXL', 'SOXS')
    ]
    
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[f"{p[0]} / {p[1]}" for p in pairs],
        vertical_spacing=0.1
    )
    
    for i, (etf1, etf2) in enumerate(pairs):
        row = i // 3 + 1
        col = i % 3 + 1
        
        if etf1 in analysis_data and etf2 in analysis_data:
            # Calcular ratio
            price1 = analysis_data[etf1]['precio_actual']
            price2 = analysis_data[etf2]['precio_actual']
            ratio = price1 / price2
            
            # Simular datos históricos del ratio (en producción usar datos reales)
            x_data = list(range(20))
            y_data = [ratio * (1 + np.random.normal(0, 0.02)) for _ in range(20)]
            mean_ratio = np.mean(y_data)
            std_ratio = np.std(y_data)
            
            # Línea del ratio
            fig.add_trace(
                go.Scatter(x=x_data, y=y_data, name=f"Ratio {etf1}/{etf2}", 
                          line=dict(color='blue')),
                row=row, col=col
            )
            
            # Bandas de desviación
            fig.add_trace(
                go.Scatter(x=x_data, y=[mean_ratio + 2*std_ratio]*20, 
                          name="Upper Band", line=dict(color='red', dash='dash')),
                row=row, col=col
            )
            
            fig.add_trace(
                go.Scatter(x=x_data, y=[mean_ratio - 2*std_ratio]*20, 
                          name="Lower Band", line=dict(color='red', dash='dash')),
                row=row, col=col
            )
    
    fig.update_layout(height=600, showlegend=False, title_text="Análisis de Pares Correlacionados")
    
    return fig.to_html(include_plotlyjs=False, div_id="pairs")

def create_risk_return_chart(df):
    """Crea gráfico de rendimiento vs riesgo"""
    
    fig = go.Figure()
    
    # Colores por riesgo
    colors = df['volatilidad']
    
    fig.add_trace(go.Bar(
        x=df.index,
        y=df['cambio_5d'],
        marker=dict(
            color=colors,
            colorscale='RdYlGn_r',
            colorbar=dict(title="Volatilidad (%)")
        ),
        text=df['cambio_5d'].round(2),
        textposition='outside',
        name='Rendimiento 5D'
    ))
    
    fig.update_layout(
        title="Rendimiento vs Riesgo (5 días)",
        xaxis_title="ETFs",
        yaxis_title="Rendimiento (%)",
        height=400
    )
    
    return fig.to_html(include_plotlyjs=False, div_id="risk_return")

def create_combined_dashboard(df):
    """Crea dashboard combinado con múltiples métricas"""
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Momentum', 'Volatilidad', 'Volumen', 'Tendencia'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"type": "pie"}]]
    )
    
    # Momentum
    fig.add_trace(
        go.Bar(x=df.index, y=df['cambio_1d'], name='Momentum 1D',
               marker_color=['green' if x > 0 else 'red' for x in df['cambio_1d']]),
        row=1, col=1
    )
    
    # Volatilidad
    fig.add_trace(
        go.Scatter(x=df.index, y=df['volatilidad'], mode='lines+markers',
                   name='Volatilidad', line=dict(color='orange')),
        row=1, col=2
    )
    
    # Volumen
    fig.add_trace(
        go.Bar(x=df.index, y=df['volumen_relativo'], name='Vol. Relativo',
               marker_color='lightblue'),
        row=2, col=1
    )
    
    # Tendencia (Pie chart)
    tendencia_counts = df['tendencia'].value_counts()
    fig.add_trace(
        go.Pie(labels=tendencia_counts.index, values=tendencia_counts.values,
               name="Tendencia"),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=False, title_text="Dashboard Combinado")
    
    return fig.to_html(include_plotlyjs=False, div_id="combined")

def get_buy_opportunities(df):
    """Identifica oportunidades de compra"""
    opportunities = df[
        (df['cambio_1d'] < -2) & 
        (df['volumen_relativo'] > 1.2) & 
        (df['tendencia'] == 'Alcista')
    ].index.tolist()
    
    return ', '.join(opportunities) if opportunities else "Ninguna detectada"

def get_sell_alerts(df):
    """Identifica alertas de venta"""
    alerts = df[
        (df['cambio_1d'] > 3) & 
        (df['volatilidad'] > 30) & 
        (df['desviacion_sma20'] > 5)
    ].index.tolist()
    
    return ', '.join(alerts) if alerts else "Ninguna detectada"

def get_pairs_opportunities(analysis_data):
    """Identifica oportunidades de pairs trading"""
    # Simplificado - en producción usar análisis estadístico real
    return "Analizar QQQ/TQQQ, SPY/SPXL para divergencias"

if __name__ == "__main__":
    daily_visual_report()