# Estudio de satifacción de clientes

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)](https://jupyter.org/)
[![Random Forest](https://img.shields.io/badge/Random%20Forest-1.0.0-green)](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)


![GitHub License](https://img.shields.io/github/license/MateoVelasquez/book_catalog)

## Tabla de Contenido
1. [Introducción](#introducción)
2. [Descripción de los Datos](#descripción-de-los-datos)
3. [Imputación de Datos en el Catálogo](#imputación-de-datos-en-el-catálogo)
4. [Preparación de Datos](#preparación-de-datos)
5. [Análisis Descriptivo](#análisis-descriptivo)
6. [Preprocesamiento de Datos](#preprocesamiento-de-datos)
7. [Modelado de Datos](#modelado-de-datos)
8. [Evaluación del Modelo](#evaluación-del-modelo)
9. [Conclusiones y Recomendaciones](#conclusiones-y-recomendaciones)
10. [Contribuciones](#contribuciones)
11. [Contacto](#contacto)

## Introducción
La satisfacción del cliente es un pilar fundamental para la sostenibilidad y el crecimiento de cualquier empresa en el mercado actual. Esta satisfacción se ve influenciada por diversos factores, entre ellos, si el producto o servicio cumple con las expectativas del cliente. Sin embargo, también es crucial considerar cómo reacciona la empresa ante situaciones en las que estas expectativas no se cumplen. La manera en que una empresa gestiona estas experiencias negativas puede tener un impacto significativo en la satisfacción general del cliente. En este proyecto de análisis de satisfacción del cliente, exploraremos tanto la percepción del cliente sobre el producto ofrecido como las respuestas de la empresa frente a posibles inconformidades, con el objetivo de comprender mejor cómo mejorar continuamente la experiencia del cliente y fortalecer la posición competitiva de la empresa en el mercado.

### Descripción de los Datos
Los datos para el entrenamiento se obtuvieron de la página Kaggle [Ecommerce Customer Service Satisfaction Dataset](https://www.kaggle.com/datasets/ddosad/ecommerce-customer-service-satisfaction?select=Customer_support_data.csv). Este conjunto de datos captura las puntuaciones de satisfacción del cliente durante un período de un mes en una plataforma de comercio electrónico llamada Shopzilla (un seudónimo).

El conjunto de datos incluye varias características, tales como:

- Categoría y subcategoría de la interacción
- Comentarios del cliente
- Fecha de respuesta a la encuesta
- Categoría y precio del artículo
- Detalles del agente (nombre, supervisor, gerente)
- Puntuación de satisfacción del cliente (CSAT), entre otros.

| Nombre de la Columna       | Descripción                                           |
|----------------------------|-------------------------------------------------------|
| **Unique id**              | Identificador único para cada registro                |
| **Channel name**           | Nombre del canal de servicio al cliente               |
| **Category**               | Categoría de la interacción                           |
| **Sub-category**           | Subcategoría de la interacción                        |
| **Customer Remarks**       | Comentarios proporcionados por el cliente             |
| **Order id**               | Identificador del pedido asociado con la interacción  |
| **Order date time**        | Fecha y hora del pedido                               |
| **Issue reported at**      | Marca de tiempo cuando se reportó el problema         |
| **Issue responded**        | Marca de tiempo cuando se respondió al problema       |
| **Survey response date**   | Fecha de respuesta de la encuesta del cliente         |
| **Customer city**          | Ciudad del cliente                                    |
| **Product category**       | Categoría del producto                                |
| **Item price**             | Precio del artículo                                   |
| **Connected handling time**| Tiempo tomado para manejar la interacción             |
| **Agent name**             | Nombre del agente de servicio al cliente              |
| **Supervisor**             | Nombre del supervisor                                 |
| **Manager**                | Nombre del gerente                                    |
| **Tenure Bucket**          | Categoría de la antigüedad del agente                 |
| **Agent Shift**            | Horario de turno del agente                           |
| **CSAT Score**             | Puntuación de Satisfacción del Cliente (CSAT)         |

## Estructura de carpetas

```
├── data/
│   ├── processed/
│   └── raw/
├── LICENSE
├── notebooks/
│   ├── 1-EDA_customer_satisfaction.ipynb
│   ├── 2-clean_preprocessing.ipynb
│   └── 3-model.ipynb
├── README copy.md
├── README.md
├── requirements.txt
```

En esta estructura:

- **data/**
  - **raw/**: Contiene los datos crudos en archivos CSV.
  - **processed/**: Contiene los datos procesados para análisis en archivos CSV.

- **notebooks**: Contiene el Jupyter notebook utilizados para la exploración de datos, preprocesamiento, modelado y evaluación de modelos.
- **README.md**: Documentación principal del proyecto que describe la estructura, los datos, los pasos y los resultados del proyecto.
- **requirements.txt**: Archivo que lista las dependencias y versiones de Python necesarias para reproducir el entorno del proyecto.

## Análisis Exploratorio de Datos (EDA)

El análisis exploratorio de datos (EDA) se ha realizado para comprender mejor el conjunto de datos y prepararlo para su posterior análisis. Este análisis incluyó:

- Identificación y manejo de datos nulos.
- Detección y manejo de duplicados en los datos.
- Tratamiento de variables temporales, que consistió en la separación de las columnas temporales en año, mes y día.
- Análisis univariado para comprender la distribución y características de cada variable por separado.
- Análisis bivariado para explorar las relaciones entre pares de variables.
- Cálculo de correlaciones entre variables para identificar posibles relaciones lineales entre ellas.
- Obtención de insights a través de la exploración de patrones, tendencias y anomalías en los datos.

## Preprocesamiento de Datos
Se realizan tareas de codificación y normalización de datos para garantizar la calidad y coherencia de los datos utilizados en el modelado. 

## Modelado de Aprendizaje Automático

Para el modelado de aprendizaje automático, se seleccionó el algoritmo Random Forest debido a su versatilidad y capacidad para manejar conjuntos de datos complejos. Algunas ventajas clave de utilizar Random Forest incluyen:

- **Robustez ante datos ruidosos**: Random Forest es capaz de manejar datos con ruido y valores atípicos de manera efectiva, lo que lo hace adecuado para conjuntos de datos reales.
- **Reducción del sobreajuste**: Su capacidad para entrenar múltiples árboles de decisión y combinar sus resultados ayuda a reducir el sobreajuste y mejorar la generalización del modelo.
- **Manejo automático de variables**: Random Forest puede manejar conjuntos de datos con muchas variables predictoras y seleccionar automáticamente las más importantes para la predicción.

Además, se utilizó el método GridSearch para optimizar los hiperparámetros del modelo y encontrar la combinación óptima de parámetros. Esto nos permitió ajustar el modelo de manera más precisa y mejorar su rendimiento predictivo.

Durante este proceso, también se identificaron las variables más importantes para el modelo, lo que proporcionó valiosas conclusiones sobre el aprendizaje y la predicción en nuestro conjunto de datos.

## Evaluación del Modelo

El modelo fue evaluado utilizando métricas estándar como precisión, recall y F1-score. Además, se empleó la matriz de confusión para evaluar el rendimiento en términos de verdaderos positivos, falsos positivos, verdaderos negativos y falsos negativos. Los resultados de la evaluación proporcionan una visión completa del rendimiento del modelo y su capacidad para generalizar a datos no vistos.

## Conclusiones y recomendaciones


Durante el análisis, identificamos las variables más relevantes que influyen en la satisfacción del cliente, tanto en situaciones donde la satisfacción es baja como en aquellas donde es alta. Algunas de estas variables incluyen:

- **Subcategorías de las interacciones**: Se observa que ciertas subcategorías de interacciones tienen un impacto significativo en la satisfacción del cliente.
- **Horarios de las interacciones**: Los días y horas en que se realizan las interacciones pueden afectar la satisfacción del cliente.
- **Desempeño de supervisores y gerentes**: La participación y desempeño de los supervisores y gerentes también pueden influir en la satisfacción del cliente.

Identificar estas variables nos permite comprender mejor qué aspectos del servicio al cliente están contribuyendo a la satisfacción del cliente y qué áreas necesitan mejoras para aumentar la satisfacción del cliente en general.

Para continuar mejorando el rendimiento del modelo de predicción de ingresos, se recomienda la implementación de diversas estrategias:

1. **Exploración Detallada de Características**: Se propone realizar un análisis exhaustivo de las características utilizadas en el modelo. Esto incluye la evaluación de la relevancia de cada característica, la creación de nuevas características derivadas que puedan capturar patrones más complejos, y la eliminación de aquellas que no contribuyan significativamente a la predicción.

2. **Optimización de Hiperparámetros Avanzada**: Se sugiere continuar con la optimización de hiperparámetros utilizando técnicas avanzadas, como la búsqueda bayesiana de hiperparámetros. Esta metodología busca de manera inteligente la combinación óptima de hiperparámetros para mejorar aún más el rendimiento del modelo y su capacidad predictiva.

3. **Exploración de Otros Modelos**: Se propone explorar la posibilidad de utilizar otros tipos de modelos, como modelos de regresión lineal, redes neuronales o modelos de aumento de gradiente (gradient boosting). La comparación de diferentes modelos puede proporcionar información valiosa sobre cuál se adapta mejor a los datos y tiene un mejor rendimiento predictivo.

Estas estrategias pueden ayudar a aumentar la precisión y la fiabilidad del modelo de predicción de ingresos, lo que resulta en una mejor toma de decisiones y una optimización de recursos en la empresa.

## ¡Contribuciones Bienvenidas!

Las contribuciones son lo que hacen que la comunidad de código abierto sea un lugar increíble para aprender, inspirar y crear. ¡Cualquier contribución que realices es muy apreciada!

Si tienes una sugerencia que pueda mejorar esto, por favor haz un fork del repositorio y crea un pull request. También puedes simplemente abrir un issue con la etiqueta "mejora". ¡No olvides darle una estrella al proyecto! ¡Gracias de nuevo por tu interés y apoyo!

### ¿Cómo Contribuir?

1. Haz un Fork del Proyecto
2. Crea tu Rama de Característica (`git checkout -b feature/1-caracteristica`)
3. Haz tus Cambios (`git commit -m 'Añadir una nueva Característica'`)
4. Haz Push a la Rama (`git push origin feature/Caracteristica`)
5. Abre un Pull Request

## Contacto

Si tienes preguntas, comentarios o sugerencias, no dudes en contactarme. Estoy aquí para ayudar y colaborar contigo en cualquier cosa que necesites. ¡Espero saber de ti pronto! 😊

- José F. Ramos: joseph0001@gmail.com
- Enlace del Proyecto: https://github.com/JRamos84/satifacci-n_cliente_ML

