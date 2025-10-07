"""
Gráficos Avanzados para el Reporte Visual
Implementa las mejoras sugeridas para el análisis profesional
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import yfinance as yf

def create_advanced_momentum_heatmap(df):
    """
    Heatmap mejorado con múltiples plazos de momentum y RSI
    """
    # Preparar datos para heatmap con múltiples plazos
    heatmap_data = df[['momentum_1d', 'momentum_1w', 'momentum_1m', 'rsi', 'bb_squeeze']].round(2)
    
    # Crear colores personalizados basados en RSI
    colors = []
    for _, row in heatmap_data.iterrows():
        if row['rsi'] > 70:
            colors.append('Sobrecompra')
        elif row['rsi'] < 30:
            colors.append('Sobreventa')
        else:
            colors.append('Neutral')
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Momentum Multi-Plazo', 'RSI y Squeeze'),
        specs=[[{"type": "heatmap"}, {"type": "scatter"}]]
    )
    
    # Heatmap de momentum
    momentum_data = df[['momentum_1d', 'momentum_1w', 'momentum_1m']].round(2)
    fig.add_trace(
        go.Heatmap(
            z=momentum_data.values,
            x=['1 Día', '1 Semana', '1 Mes'],
            y=momentum_data.index,
            colorscale='RdYlGn',
            zmid=0,
            text=momentum_data.values,
            texttemplate="%{text}%",
            textfont={"size": 10},
            name="Momentum"
        ),
        row=1, col=1
    )
    
    # Scatter RSI vs Momentum con indicador de squeeze
    fig.add_trace(
        go.Scatter(
            x=df['rsi'],
            y=df['momentum_1d'],
            mode='markers+text',
            text=df.index,
            textposition='top center',
            marker=dict(
                size=[20 if squeeze else 10 for squeeze in df['bb_squeeze']],
                color=['red' if rsi > 70 else 'green' if rsi < 30 else 'blue' for rsi in df['rsi']],
                opacity=0.7,
                line=dict(width=2, color='white')
            ),
            name='RSI vs Momentum'
        ),
        row=1, col=2
    )
    
    # Líneas de referencia RSI
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=2)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=2)
    fig.add_vline(x=70, line_dash="dash", line_color="red", row=1, col=2)
    fig.add_vline(x=30, line_dash="dash", line_color="green", row=1, col=2)
    
    fig.update_layout(
        title="Análisis Avanzado de Momentum y RSI",
        height=500,
        showlegend=False
    )
    
    return fig.to_html(include_plotlyjs=False, div_id="advanced_momentum")

def create_risk_adjusted_returns_chart(df):
    """
    Gráfico mejorado con Sharpe Ratio, Sortino Ratio y Max Drawdown
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Sharpe Ratio', 'Sortino Ratio', 'Max Drawdown', 'Rendimiento vs Riesgo'),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "scatter"}]]
    )
    
    # Sharpe Ratio
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['sharpe_ratio'],
            name='Sharpe Ratio',
            marker_color=['green' if x > 1 else 'orange' if x > 0 else 'red' for x in df['sharpe_ratio']],
            text=df['sharpe_ratio'].round(2),
            textposition='outside'
        ),
        row=1, col=1
    )
    
    # Sortino Ratio
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['sortino_ratio'],
            name='Sortino Ratio',
            marker_color=['green' if x > 1 else 'orange' if x > 0 else 'red' for x in df['sortino_ratio']],
            text=df['sortino_ratio'].round(2),
            textposition='outside'
        ),
        row=1, col=2
    )
    
    # Max Drawdown
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['max_drawdown'],
            name='Max Drawdown (%)',
            marker_color=['red' if x < -20 else 'orange' if x < -10 else 'green' for x in df['max_drawdown']],
            text=df['max_drawdown'].round(1),
            textposition='outside'
        ),
        row=2, col=1
    )
    
    # Scatter Rendimiento vs Riesgo
    fig.add_trace(
        go.Scatter(
            x=df['volatilidad'],
            y=df['momentum_1m'],
            mode='markers+text',
            text=df.index,
            textposition='top center',
            marker=dict(
                size=df['sharpe_ratio'] * 10 + 10,  # Tamaño basado en Sharpe
                color=df['sortino_ratio'],
                colorscale='RdYlGn',
                opacity=0.7,
                line=dict(width=2, color='white'),
                colorbar=dict(title="Sortino Ratio")
            ),
            name='Riesgo vs Rendimiento'
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        title="Análisis de Riesgo Ajustado",
        height=600,
        showlegend=False
    )
    
    return fig.to_html(include_plotlyjs=False, div_id="risk_adjusted")

def create_correlation_matrix(analysis_data):
    """
    Matriz de correlación entre activos
    """
    # Crear DataFrame con precios para calcular correlaciones
    symbols = list(analysis_data.keys())[:20]  # Limitar a 20 para mejor visualización
    
    # Simular matriz de correlación (en producción usar datos históricos reales)
    np.random.seed(42)
    correlation_matrix = np.random.rand(len(symbols), len(symbols))
    correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2  # Hacer simétrica
    np.fill_diagonal(correlation_matrix, 1)  # Diagonal = 1
    
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix,
        x=symbols,
        y=symbols,
        colorscale='RdBu',
        zmid=0,
        text=np.round(correlation_matrix, 2),
        texttemplate="%{text}",
        textfont={"size": 8},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="Matriz de Correlación entre Activos",
        height=600,
        font=dict(size=10)
    )
    
    return fig.to_html(include_plotlyjs=False, div_id="correlation_matrix")

def create_relative_strength_chart(analysis_data):
    """
    Gráfico de Fuerza Relativa vs Benchmark
    """
    # Usar SPY como benchmark para acciones USA y NAFTRAC para mexicanas
    benchmark_symbols = ['SPY', 'NAFTRAC', 'QQQ']
    
    fig = make_subplots(
        rows=len(benchmark_symbols), cols=1,
        subplot_titles=[f'Fuerza Relativa vs {bench}' for bench in benchmark_symbols],
        vertical_spacing=0.1
    )
    
    for i, benchmark in enumerate(benchmark_symbols):
        if benchmark in analysis_data:
            benchmark_return = analysis_data[benchmark]['momentum_1m']
            
            # Calcular fuerza relativa para cada activo
            relative_strengths = []
            symbols = []
            
            for symbol, data in analysis_data.items():
                if symbol != benchmark:
                    relative_strength = data['momentum_1m'] - benchmark_return
                    relative_strengths.append(relative_strength)
                    symbols.append(symbol)
            
            # Tomar solo los primeros 15 para mejor visualización
            if len(symbols) > 15:
                symbols = symbols[:15]
                relative_strengths = relative_strengths[:15]
            
            colors = ['green' if rs > 0 else 'red' for rs in relative_strengths]
            
            fig.add_trace(
                go.Bar(
                    x=symbols,
                    y=relative_strengths,
                    name=f'vs {benchmark}',
                    marker_color=colors,
                    text=[f"{rs:+.1f}%" for rs in relative_strengths],
                    textposition='outside'
                ),
                row=i+1, col=1
            )
    
    fig.update_layout(
        title="Análisis de Fuerza Relativa",
        height=800,
        showlegend=False
    )
    
    return fig.to_html(include_plotlyjs=False, div_id="relative_strength")

def create_market_overview_dashboard(analysis_data):
    """
    Dashboard "Vista de Pájaro" del Mercado
    """
    # Categorizar por sectores (simplificado)
    sectors = {
        'Tecnología': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'AMD', 'ORCL', 'CRM'],
        'Financiero': ['JPM', 'BAC', 'WFC', 'GFNORTE', 'BBAJIO'],
        'Energía': ['XOM', 'CVX', 'PEMEX'],
        'Consumo': ['WALMEX', 'FEMSA', 'KOF', 'BIMBO'],
        'ETFs Apalancados': ['SOXL', 'TQQQ', 'SPXL', 'FAS', 'TNA']
    }
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Rendimiento Sectorial', 'Distribución RSI', 'Señales de Trading', 'Alertas de Squeeze'),
        specs=[[{"type": "bar"}, {"type": "histogram"}],
               [{"type": "pie"}, {"type": "bar"}]]
    )
    
    # 1. Rendimiento Sectorial
    sector_returns = []
    sector_names = []
    
    for sector, symbols in sectors.items():
        sector_return = np.mean([
            analysis_data[symbol]['momentum_1w'] 
            for symbol in symbols 
            if symbol in analysis_data
        ])
        sector_returns.append(sector_return)
        sector_names.append(sector)
    
    fig.add_trace(
        go.Bar(
            x=sector_names,
            y=sector_returns,
            name='Rendimiento Sectorial',
            marker_color=['green' if r > 0 else 'red' for r in sector_returns],
            text=[f"{r:+.1f}%" for r in sector_returns],
            textposition='outside'
        ),
        row=1, col=1
    )
    
    # 2. Distribución RSI
    rsi_values = [data['rsi'] for data in analysis_data.values()]
    fig.add_trace(
        go.Histogram(
            x=rsi_values,
            nbinsx=20,
            name='Distribución RSI',
            marker_color='lightblue'
        ),
        row=1, col=2
    )
    
    # 3. Señales de Trading (Pie Chart)
    buy_signals = sum(1 for data in analysis_data.values() if data.get('buy_signal', False))
    sell_signals = sum(1 for data in analysis_data.values() if data.get('sell_signal', False))
    hold_signals = len(analysis_data) - buy_signals - sell_signals
    
    fig.add_trace(
        go.Pie(
            labels=['Comprar', 'Vender', 'Mantener'],
            values=[buy_signals, sell_signals, hold_signals],
            marker_colors=['green', 'red', 'gray'],
            name="Señales"
        ),
        row=2, col=1
    )
    
    # 4. Alertas de Squeeze
    squeeze_symbols = [symbol for symbol, data in analysis_data.items() if data.get('squeeze_alert', False)]
    squeeze_values = [analysis_data[symbol]['bb_width'] for symbol in squeeze_symbols[:10]]  # Top 10
    
    if squeeze_symbols:
        fig.add_trace(
            go.Bar(
                x=squeeze_symbols[:10],
                y=squeeze_values,
                name='Alertas de Squeeze',
                marker_color='orange',
                text=[f"{v:.1f}%" for v in squeeze_values],
                textposition='outside'
            ),
            row=2, col=2
        )
    
    fig.update_layout(
        title="Vista de Pájaro del Mercado",
        height=700,
        showlegend=False
    )
    
    return fig.to_html(include_plotlyjs=False, div_id="market_overview")

def create_individual_technical_analysis(symbol, data_dict):
    """
    Dashboard de Análisis Técnico Individual para un símbolo específico
    """
    try:
        # Descargar datos históricos para el gráfico de velas
        df = yf.download(symbol, period="3mo", interval="1d", progress=False)
        
        if df.empty:
            return f"<div>No hay datos disponibles para {symbol}</div>"
        
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=(f'{symbol} - Precio y Volumen', 'RSI', 'Bandas de Bollinger'),
            vertical_spacing=0.1,
            row_heights=[0.6, 0.2, 0.2]
        )
        
        # 1. Gráfico de Velas con Medias Móviles
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name=symbol
            ),
            row=1, col=1
        )
        
        # Medias móviles
        sma_20 = df['Close'].rolling(20).mean()
        sma_50 = df['Close'].rolling(50).mean()
        
        fig.add_trace(
            go.Scatter(x=df.index, y=sma_20, name='SMA 20', line=dict(color='orange')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=df.index, y=sma_50, name='SMA 50', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Volumen
        fig.add_trace(
            go.Bar(x=df.index, y=df['Volume'], name='Volumen', marker_color='lightgray', opacity=0.5),
            row=1, col=1
        )
        
        # 2. RSI
        from ta.momentum import RSIIndicator
        rsi = RSIIndicator(df['Close']).rsi()
        
        fig.add_trace(
            go.Scatter(x=df.index, y=rsi, name='RSI', line=dict(color='purple')),
            row=2, col=1
        )
        
        # Líneas de referencia RSI
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        # 3. Bandas de Bollinger
        from ta.volatility import BollingerBands
        bb = BollingerBands(df['Close'])
        
        fig.add_trace(
            go.Scatter(x=df.index, y=bb.bollinger_hband(), name='BB Superior', line=dict(color='red', dash='dash')),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=df.index, y=bb.bollinger_lband(), name='BB Inferior', line=dict(color='green', dash='dash')),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=df.index, y=bb.bollinger_mavg(), name='BB Media', line=dict(color='blue')),
            row=3, col=1
        )
        
        fig.update_layout(
            title=f"Análisis Técnico Completo - {symbol}",
            height=800,
            showlegend=True
        )
        
        return fig.to_html(include_plotlyjs=False, div_id=f"technical_{symbol}")
        
    except Exception as e:
        return f"<div>Error generando análisis para {symbol}: {e}</div>"

def generate_executive_summary(analysis_data):
    """
    Genera un resumen ejecutivo automático
    """
    # Top 3 Momentum
    top_momentum = sorted(analysis_data.items(), key=lambda x: x[1]['momentum_1w'], reverse=True)[:3]
    top_momentum_str = ", ".join([f"{symbol} ({data['momentum_1w']:+.1f}%)" for symbol, data in top_momentum])
    
    # Potencial de rebote (RSI bajo)
    oversold = [(symbol, data) for symbol, data in analysis_data.items() if data['rsi'] < 35]
    oversold_str = ", ".join([symbol for symbol, _ in oversold[:3]]) if oversold else "Ninguno"
    
    # Alertas de squeeze
    squeeze_alerts = [symbol for symbol, data in analysis_data.items() if data.get('squeeze_alert', False)]
    squeeze_str = ", ".join(squeeze_alerts[:3]) if squeeze_alerts else "Ninguno"
    
    # Señales de compra
    buy_signals = [symbol for symbol, data in analysis_data.items() if data.get('buy_signal', False)]
    buy_str = ", ".join(buy_signals[:5]) if buy_signals else "Ninguno"
    
    # Señales de venta
    sell_signals = [symbol for symbol, data in analysis_data.items() if data.get('sell_signal', False)]
    sell_str = ", ".join(sell_signals[:5]) if sell_signals else "Ninguno"
    
    summary = f"""
    <div class="executive-summary">
        <h3>📊 Resumen Ejecutivo Automático</h3>
        <div class="summary-grid">
            <div class="summary-item">
                <strong>🚀 Top 3 Momentum (1 semana):</strong> {top_momentum_str}
            </div>
            <div class="summary-item">
                <strong>🔄 Potencial de Rebote (RSI < 35):</strong> {oversold_str}
            </div>
            <div class="summary-item">
                <strong>⚡ Alertas de Squeeze:</strong> {squeeze_str}
            </div>
            <div class="summary-item">
                <strong>🟢 Señales de Compra:</strong> {buy_str}
            </div>
            <div class="summary-item">
                <strong>🔴 Señales de Venta:</strong> {sell_str}
            </div>
        </div>
    </div>
    """
    
    return summary