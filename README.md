# 🚀 Hacktinver v2.0

**Herramienta avanzada de análisis de inversiones con arquitectura modular profesional**

Hacktinver es una plataforma integral de análisis técnico y fundamental diseñada específicamente para optimizar las estrategias de inversión dentro del contexto del concurso anual de trading de Actinver. La versión 2.0 presenta una **arquitectura modular completamente reestructurada** que sigue principios de ingeniería de software profesional.

## 🆕 **Novedades v2.0 - Arquitectura Modular**

- **🏗️ Separación de Responsabilidades**: Cada módulo tiene una función específica
- **🧩 Modularidad**: Fácil mantenimiento y escalabilidad
- **🔧 Configuración Centralizada**: Sistema unificado de configuración
- **📊 Data Provider Robusto**: Obtención optimizada de datos de mercado
- **🎯 Estrategias Independientes**: Cada estrategia en su propio módulo
- **🖥️ UI Mejorada**: Sistema de menús interactivos con Rich
- **📋 Logging Avanzado**: Sistema completo de logs y debugging

## 📊 Características Principales

### 🎯 Análisis Técnico Avanzado
- **ETFs Apalancados Optimizados**: Algoritmo especializado para detectar oportunidades en ETFs con tendencia alcista de 5 semanas pero caída diaria
- **Swing Trading Inteligente**: Estrategias automatizadas basadas en indicadores técnicos múltiples
- **Análisis de Momentum**: Detección de patrones de momentum fuerte y reversión
- **Indicadores Técnicos**: RSI, MACD, Bandas de Bollinger, Medias Móviles, Estocástico

### 📈 Monitor de Stocks en Tiempo Real
- **Actualización cada 3 segundos** de datos del mercado
- **Gráficas interactivas** con variaciones acumuladas
- **Análisis visual** de tendencias con colores dinámicos
- **Estadísticas del mercado** en tiempo real
- **Guardado automático** de gráficas con timestamp

### 🤖 Automatización del Concurso Actinver
- **Gestión automática de sesiones** en la plataforma
- **Respuesta automática** a quizzes diarios y semanales
- **Monitoreo de portafolio** en tiempo real
- **Ejecución automática** de órdenes de compra/venta
- **Programación de tareas** para operaciones recurrentes

### 📊 Análisis Cuantitativo
- **Optimización de Markowitz** para distribución de portafolio
- **Razón de Sharpe Ajustada** para maximizar rendimiento/riesgo
- **Análisis de correlaciones** entre activos
- **Backtesting** de estrategias históricas

### 🔍 Análisis Fundamental y Sentimientos
- **Análisis por sectores** económicos
- **Seguimiento de noticias** y sentimientos del mercado
- **Análisis de resultados** próximos de empresas
- **Recomendaciones personalizadas** según preferencias

## 🛠️ Instalación

### Requisitos Previos
- Python 3.8 o superior
- pip (gestor de paquetes de Python)
- Conexión a internet para datos en tiempo real

### Instalación Rápida

```bash
# Clonar el repositorio
git clone https://github.com/tu-usuario/hacktinver.git
cd hacktinver

# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
# En Windows:
.\venv\Scripts\Activate.ps1
# En Linux/Mac:
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

### Configuración de Variables de Entorno

Crear un archivo `.env` en la raíz del proyecto:

```env
# Credenciales de Actinver
ACTINVER_USER_EMAIL=tu_email@ejemplo.com
ACTINVER_USER_PASSWORD=tu_contraseña

# API Keys (opcional)
TELEGRAM_BOT_TOKEN=tu_token_telegram
GEMINI_API_KEY=tu_api_key_gemini
```

## 🚀 Uso

### Ejecutar la Aplicación

```bash
python hacktinver.py
```

### Menú Principal

La aplicación presenta un menú interactivo con las siguientes opciones:

1. **Análisis Técnico Rápido ETFs Apalancados** - Estrategia optimizada para ETFs con alta volatilidad
2. **Análisis Técnico ETFs Normales** - Análisis para ETFs tradicionales
3. **Análisis Fundamental por Sectores** - Evaluación de sectores económicos
4. **Análisis de Preferencias** - Recomendaciones personalizadas
5. **Análisis de Sentimientos** - Evaluación basada en noticias
6. **Swing Trading por Resultados** - Estrategia basada en publicación de resultados
7. **Swing Trading Técnico** - Análisis técnico tradicional
8. **Consensos Técnicos Web** - Análisis basado en consensos online
9. **Machine Learning Trading** - Estrategias con inteligencia artificial
10. **Estrategias Beta** - Algoritmos experimentales
11. **Optimización Sharpe** - Distribución óptima de portafolio
12. **Optimización Markowitz** - Teoría moderna de portafolios
13. **Optimización Litterman** - Modelo Black-Litterman
14. **Utilidades Actinver** - Herramientas específicas del concurso
15. **🆕 Monitor de Stocks** - Monitoreo en tiempo real con actualización cada 3 segundos

### Navegación

- **Flechas ↑↓**: Navegar entre opciones
- **Enter**: Seleccionar opción
- **:** + número**: Acceso directo a opción
- **q**: Salir de la aplicación

## 📊 ETFs y Activos Soportados

### ETFs Apalancados (2025)
```
FAS, FAZ, PSQ, QLD, SOXL, SOXS, SPXL, SPXS, SQQQ, TECL, TECS, TNA, TQQQ, TZA, EDZ
```

### ETFs Normales (2025)
```
AAXJ, ACWI, BIL, BOTZ, DIA, EEM, EWZ, GDX, GLD, IAU, ICLN, INDA, IVV, KWEB, LIT, 
MCHI, NAFTRAC, QCLN, QQQ, SHV, SHY, SLV, SOXX, SPLG, SPY, TAN, TLT, USO, VEA, 
VGT, VNQ, VOO, VTI, VT, VWO, VYM, XLE, XLF, XLK, XLV
```

### Acciones Favoritas
```
TSLA.MX, BAC.MX, CEMEXCPO.MX, PE&OLES.MX, ORBIA.MX, AMZN.MX, COST.MX, 
AMD.MX, FCX.MX, AAPL.MX, MSFT.MX
```

## 🔧 Funcionalidades Avanzadas

### Monitor de Stocks en Tiempo Real

La nueva funcionalidad de monitoreo incluye:

- **Descarga automática** de datos desde `https://www.retoactinver.com/archivos/datosReto.txt`
- **Procesamiento en tiempo real** de variaciones porcentuales
- **Gráficas dinámicas** que se actualizan cada 3 segundos
- **Tabla interactiva** con colores y emojis indicativos
- **Estadísticas del mercado** (% acciones ganando/perdiendo)
- **Guardado automático** de gráficas cada 10 actualizaciones

### Algoritmo ETFs Apalancados Optimizado

Características especiales:

- **Análisis de 5 semanas**: Evalúa tendencia alcista histórica
- **Detección de dips**: Identifica oportunidades de compra en caídas temporales
- **Sistema de scoring**: Puntuación 0-100 para priorizar oportunidades
- **Múltiples indicadores**: RSI, MACD, Bollinger, SMA, EMA, análisis de volumen
- **Categorización inteligente**: Compra en dip, momentum, venta, espera

## 📁 Estructura del Proyecto

```
hacktinver/
├── hacktinver.py              # Aplicación principal
├── requirements.txt           # Dependencias Python
├── requirements2.txt          # Dependencias adicionales
├── .env                       # Variables de entorno (crear)
├── README.md                  # Este archivo
├── data/                      # Datos y resultados de análisis
├── ActinverMoves/            # Datos históricos de movimientos
├── ActinverContestPositions/ # Capturas de posiciones del concurso
├── TradingViewData/          # Datos de TradingView
├── IndicatorHistory/         # Historial de indicadores
└── Scripts/                  # Scripts auxiliares
```

## 🔐 Seguridad y Privacidad

- **Credenciales encriptadas**: Las contraseñas se manejan de forma segura
- **Tokens de sesión**: Gestión automática de autenticación
- **Datos locales**: Toda la información se almacena localmente
- **Sin tracking**: No se envían datos a terceros

## 📈 Estrategias de Trading Implementadas

### 1. Swing Trading Técnico
- Basado en indicadores técnicos múltiples
- Detección de puntos de entrada y salida
- Gestión de riesgo automatizada

### 2. Momentum Trading
- Identificación de tendencias fuertes
- Análisis de volumen y precio
- Seguimiento de breakouts

### 3. Mean Reversion
- Detección de sobrecompra/sobreventa
- Estrategias de reversión a la media
- Análisis de bandas de Bollinger

### 4. Análisis Cuantitativo
- Optimización matemática de portafolios
- Análisis de correlaciones
- Backtesting estadístico

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Para contribuir:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📞 Soporte

Para soporte técnico o preguntas:

- **Issues**: Abre un issue en GitHub
- **Email**: [tu-email@ejemplo.com]
- **Documentación**: Consulta la wiki del proyecto

## ⚠️ Disclaimer

Esta herramienta es para fines educativos y de investigación. El trading conlleva riesgos financieros. Los usuarios son responsables de sus decisiones de inversión. Los resultados pasados no garantizan rendimientos futuros.

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## 🏆 Concurso Actinver

Hacktinver está específicamente diseñado para el **Concurso Anual de Actinver Casa de Bolsa**, proporcionando herramientas avanzadas para:

- Maximizar rendimientos en el período del concurso
- Automatizar tareas repetitivas
- Analizar oportunidades en tiempo real
- Optimizar la distribución del portafolio
- Gestionar riesgos de manera inteligente

---

**¡Buena suerte en el concurso! 🚀📈**