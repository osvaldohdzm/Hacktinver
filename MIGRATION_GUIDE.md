# 🚀 Guía de Migración - Hacktinver v2.0

## Arquitectura Modular Implementada

Hacktinver ha sido reestructurado siguiendo principios de **Separación de Responsabilidades (SoC)** y **Arquitectura Modular** para mejorar la mantenibilidad, escalabilidad y colaboración.

## 📁 Nueva Estructura del Proyecto

```
hacktinver/
├── main.py                    # 🎯 Punto de entrada principal (NUEVO)
├── config.py                  # ⚙️ Configuración central (NUEVO)
├── hacktinver.py             # 📜 Archivo original (LEGACY)
├── requirements.txt          # 📦 Dependencias actualizadas
├── data/                     # 💾 Datos y resultados
├── logs/                     # 📋 Archivos de log (NUEVO)
│
├── core/                     # 🧠 Lógica central
│   ├── __init__.py
│   ├── data_provider.py      # 📊 Obtención de datos de mercado
│   └── indicators.py         # 📈 Indicadores técnicos
│
├── strategies/               # 🎯 Estrategias de trading
│   ├── __init__.py
│   ├── pairs_trading.py      # ✅ MIGRADO - Pairs Trading avanzado
│   ├── volatility_allocation.py # ✅ MIGRADO - Asignación por volatilidad
│   ├── leveraged_etf.py      # 🚧 En migración
│   ├── normal_etf.py         # 🚧 En migración
│   ├── fundamental.py        # 🚧 En migración
│   ├── preferences.py        # 🚧 En migración
│   ├── sentiment.py          # 🚧 En migración
│   ├── earnings.py           # 🚧 En migración
│   ├── technical.py          # 🚧 En migración
│   ├── consensus.py          # 🚧 En migración
│   ├── machine_learning.py   # 🚧 En migración
│   └── beta_strategies.py    # 🚧 En migración
│
├── portfolio/                # 💼 Gestión de portafolio
│   ├── __init__.py
│   └── optimization.py       # 🚧 Optimización (Markowitz, Sharpe, etc.)
│
├── services/                 # 🔌 Servicios externos
│   ├── __init__.py
│   ├── actinver_utilities.py # 🚧 Utilidades Actinver
│   ├── stock_monitor.py      # 🚧 Monitor de stocks
│   └── notification.py       # 📱 Notificaciones (Telegram)
│
└── ui/                       # 🖥️ Interfaz de usuario
    ├── __init__.py
    ├── menu.py               # ✅ MIGRADO - Sistema de menús
    └── display.py            # ✅ MIGRADO - Visualización
```

## 🎯 Estado de Migración

### ✅ **Completamente Migrado**
- **Arquitectura base**: Estructura modular implementada
- **Sistema de menús**: Navegación interactiva mejorada
- **Pairs Trading**: Algoritmo avanzado con test de cointegración
- **Asignación por Volatilidad**: Gestión de riesgo basada en ATR
- **Configuración**: Sistema centralizado de configuración
- **Data Provider**: Obtención robusta de datos de mercado
- **Indicadores Técnicos**: Biblioteca completa de indicadores

### 🚧 **En Proceso de Migración**
- **ETFs Apalancados**: `suggest_technical_etf_leveraged()`
- **Monitor de Stocks**: Monitoreo en tiempo real
- **Utilidades Actinver**: Automatización del concurso
- **Análisis Fundamental**: Análisis por sectores
- **Machine Learning**: Estrategias con IA
- **Optimización de Portafolio**: Markowitz, Sharpe, Litterman

## 🚀 Cómo Usar la Nueva Versión

### Instalación y Configuración

1. **Instalar dependencias actualizadas**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configurar variables de entorno** (crear archivo `.env`):
   ```env
   # Credenciales Actinver
   ACTINVER_USER_EMAIL=tu_email@ejemplo.com
   ACTINVER_USER_PASSWORD=tu_contraseña
   
   # APIs opcionales
   TELEGRAM_BOT_TOKEN=tu_token_telegram
   GEMINI_API_KEY=tu_api_key_gemini
   
   # Configuración
   DEBUG=False
   MAX_CONCURRENT_DOWNLOADS=5
   DEFAULT_TIMEOUT=30
   ```

3. **Ejecutar la nueva versión**:
   ```bash
   python main.py
   ```

### Funcionalidades Disponibles

#### ✅ **Pairs Trading Avanzado** (Opción 1)
- Test de cointegración estadística (ADF)
- Optimización automática de ventanas
- Gestión de riesgo con stop-loss
- Cálculo de half-life de reversión
- Dimensionamiento de posición neutral

#### ✅ **Asignación por Volatilidad** (Opción 2)
- Cálculo de ATR para gestión de riesgo
- Posiciones inversamente proporcionales a volatilidad
- Stop-loss automático basado en ATR
- Análisis de Sharpe ratio por activo

#### 🚧 **Funciones Legacy** (Opciones 3-19)
- Actualmente muestran mensaje "En migración"
- Funcionalidad completa disponible en `hacktinver.py`
- Migración progresiva en próximas versiones

## 🔄 Transición Gradual

### Usar Ambas Versiones

Durante la transición, puedes usar ambas versiones:

- **Nueva arquitectura**: `python main.py`
- **Versión original**: `python hacktinver.py`

### Beneficios de la Nueva Arquitectura

1. **🧩 Modularidad**: Cada componente tiene una responsabilidad específica
2. **🔧 Mantenibilidad**: Fácil localización y corrección de errores
3. **📈 Escalabilidad**: Agregar nuevas estrategias es simple
4. **🧪 Testabilidad**: Cada módulo se puede probar independientemente
5. **🤝 Colaboración**: Múltiples desarrolladores pueden trabajar sin conflictos
6. **♻️ Reutilización**: Módulos pueden usarse en otros proyectos

## 📋 Plan de Migración Completa

### Fase 1: ✅ **Completada**
- [x] Estructura modular base
- [x] Sistema de configuración
- [x] Menús interactivos
- [x] Pairs Trading avanzado
- [x] Asignación por volatilidad

### Fase 2: 🚧 **En Progreso**
- [ ] Monitor de stocks en tiempo real
- [ ] Utilidades Actinver (quizzes, portafolio)
- [ ] Estrategias de ETFs apalancados
- [ ] Análisis fundamental por sectores

### Fase 3: 📅 **Planificada**
- [ ] Machine Learning strategies
- [ ] Optimización de portafolio completa
- [ ] Web scraping modularizado
- [ ] Sistema de notificaciones avanzado

### Fase 4: 🎯 **Futuro**
- [ ] API REST para integración externa
- [ ] Dashboard web interactivo
- [ ] Backtesting automatizado
- [ ] Alertas en tiempo real

## 🛠️ Para Desarrolladores

### Agregar Nueva Estrategia

1. **Crear archivo en `strategies/`**:
   ```python
   # strategies/mi_nueva_estrategia.py
   from rich.console import Console
   from core.data_provider import download_multiple_tickers
   
   console = Console()
   
   def run_mi_nueva_estrategia():
       console.print("[bold blue]🚀 Mi Nueva Estrategia[/bold blue]")
       # Tu lógica aquí
   ```

2. **Agregar al menú en `ui/menu.py`**:
   ```python
   from strategies.mi_nueva_estrategia import run_mi_nueva_estrategia
   
   menu_options = [
       # ... otras opciones ...
       ("Mi Nueva Estrategia", run_mi_nueva_estrategia),
   ]
   ```

### Estructura de Módulos

Cada módulo debe seguir esta estructura:

```python
"""
Docstring del módulo
Descripción de la funcionalidad
"""

import logging
from typing import List, Dict, Any
from rich.console import Console

# Imports internos
from core.data_provider import download_multiple_tickers
from ui.display import save_results_to_csv

logger = logging.getLogger("hacktinver.nombre_modulo")
console = Console()

def funcion_principal():
    """Función principal del módulo"""
    pass
```

## 🐛 Resolución de Problemas

### Error: Módulo no encontrado
```bash
# Asegúrate de estar en el directorio correcto
cd hacktinver
python main.py
```

### Error: statsmodels no instalado
```bash
pip install statsmodels
```

### Error: Variables de entorno
- Crear archivo `.env` en la raíz del proyecto
- Verificar que las variables estén correctamente definidas

## 📞 Soporte

- **Issues**: Reportar problemas en el repositorio
- **Documentación**: Consultar archivos README.md
- **Logs**: Revisar archivos en `logs/` para debugging
