# Sistema-de-recomendacion-de-libros---modelo-hibrido-CF-CB
Este proyecto implementa un **sistema de recomendación híbrido** que combina **Filtrado Colaborativo (Collaborative Filtering)** y **Filtrado Basado en Contenido (Content-Based)** utilizando datos del conjunto **Book-Crossing Dataset**.



Incluye dos fases principales:
1. **Análisis exploratorio y filtrado K-Core** (`eda_kcore.py`)
2. **Modelado y evaluación del sistema híbrido** (`modelado.py`)

---

##  Descripción General

El objetivo del proyecto es desarrollar un sistema capaz de **recomendar libros personalizados** a los usuarios combinando:
- Preferencias de usuarios similares (CF)
- Similitudes entre los libros a partir de metadatos (CB)

El enfoque híbrido busca aprovechar las fortalezas de ambos métodos para obtener recomendaciones más robustas y precisas.

---

##  Estructura del Proyecto
├── data/
│ ├── Books.csv
│ ├── Users.csv
│ ├── Ratings.csv
│ ├── ratinf.csv
│ ├── usuario_f.csv
│ └── libros_f.csv
├── eda_kcore.py
├── modelado.py
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
└── docs/

## Ejecución
Ejecuta el script eda_kcore.py para limpiar y filtrar los datos originales de Book-Crossing:

````python eda_kcore.py
````


Esto genera los archivos:

-ratinf.csv
-usuario_f.csv
-libros_f.csv
ubicados en la carpeta /data/.

PASO 2: 
Ejecuta el modelo híbrido:

````python modelado.py
````


El script:

-Construye las matrices CF y CB.
-Genera la matriz híbrida R_hibrido.
-Evalúa el rendimiento del sistema con RMSE, MAE, Precision@5 y Recall@5.
-Muestra ejemplos de recomendaciones personalizadas.

## Metricas
-RMSE (Root Mean Squared Error)
-MAE (Mean Absolute Error)
-Precision@5
-Recall@5

El modelo híbrido combina las predicciones de CF y CB según el parámetro:
ALPHA = 0.7
donde:
ALPHA controla el peso del modelo CF (0.7 = 70% CF + 30% CB).

## Metodología del Modelo

-Filtrado Colaborativo (CF)
Se aplica TruncatedSVD sobre la matriz usuario-libro dispersa para obtener embeddings latentes de usuarios y libros.

-Filtrado Basado en Contenido (CB)
Se utiliza TF-IDF sobre los metadatos de los libros (título, autor, editorial, año) para construir representaciones vectoriales y medir similitud coseno.

Modelo Híbrido: se combinan ambos modelos 
Metodología del Modelo

Evaluación y Recomendación
Se calcula RMSE y MAE sobre los ratings conocidos.
Se generan recomendaciones personalizadas (top_k) para usuarios concretos o aleatorios.

Visualizaciones

-Distribución de calificaciones (book-rating)
-Heatmap de selección de umbrales K-core

## Autora 
Verónica Balza

