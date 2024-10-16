# Proyecto MLOps: Sistema de Recomendación de Videojuegos para Usuarios en Steam

¡Bienvenidos al primer proyecto individual de la etapa de labs! En esta ocasión, deberán hacer un trabajo situándose en el rol de un MLOps Engineer.



## Descripción del Problema

Este proyecto se desarrolla como parte de mi rol como Data Scientist en Steam, donde se busca implementar un sistema de recomendación de videojuegos para los usuarios de la plataforma. El objetivo es crear un modelo de Machine Learning (ML) que ofrezca recomendaciones personalizadas basadas en las interacciones y preferencias de los usuarios.

## Objetivos

Limpieza y transformación de datos en formato raw.<br>
Creación de un análisis de sentimiento en las reseñas de los usuarios. <br>
Desarrollo de una API utilizando FastAPI para servir los datos.<br>
Entrenamiento de un modelo de ML para recomendaciones de videojuegos. <br>

## Contexto y Rol a Desarrollar

Como Data Scientist en HENRY obtuvimos la tarea de desarollar un proyecto basandonos en la empresa de videojuegos Steam, mi objetivo es crear un sistema de recomendación de videojuegos que mejore la experiencia de los usuarios. Al comenzar, descubrí que los datos eran de baja calidad: anidados y sin procesos automatizados para la actualización de productos. Esto me llevó a iniciar un trabajo de Data Engineering desde cero para desarrollar un Minimum Viable Product (MVP).

# Tareas clave en el desarollo del mismo

* Transformaciones de datos: Optimicé el rendimiento eliminando columnas innecesarias. <br>
* Feature Engineering: Implementé un análisis de sentimiento en las reseñas, creando una columna sentiment_analysis que clasifica las reseñas como malas (0), neutrales (1) o positivas (2). <br>
* Desarrollo de una API: Construí una API con FastAPI que permite acceder a datos relevantes mediante varios endpoints.
* Análisis Exploratorio de Datos (EDA): Realicé un análisis manual para identificar patrones y relaciones entre variables. <br>
Modelo de Aprendizaje Automático: Entrené un modelo de recomendación con enfoques item-item <br>

## Descripción del Proyecto

Trabajamos con los siguientes datasets en formato json:

- **australian_user_reviews**
- **australian_users_items**
- **output_steam_games**

### Proceso de ETL 

__Proceso de Preparación de Datos__

* Desanidado: Se desanidaron varias columnas que contenían datos anidados (listas o diccionarios) para facilitar las consultas a la API y el análisis posterior.

* Eliminación de Columnas No Utilizadas.

* Tratamiento de Valores Nulos: Se eliminaron las filas que contenían valores nulos en las columnas relevantes.

* Cambio de Tipos de Datos: Se ajustaron los tipos de datos de varias columnas para garantizar una mejor manipulación y análisis.

__Feature Engineering__
Se generó una nueva columna denominada sentiment_analysis en el dataset user_reviews, utilizando un análisis de sentimiento. La clasificación se realizó en tres categorías:

-Malo
-Neutral
-Positivo

__Análisis Exploratorio de Datos (EDA)__

Durante la fase de EDA, se examinaron las características clave de los datasets. Se identificaron tendencias y patrones que serían útiles para alimentar el modelo de recomendación.

### Desarrollo de la API

Usando **FastAPI**, se diseñaron las siguientes consultas:

1.- developer(developer: str): <br>

Descripción: Devuelve la cantidad de ítems y el porcentaje de contenido gratuito por año según la empresa desarrolladora.

2.-userdata(user_id: str):

Descripción: Retorna la cantidad de dinero gastado por el usuario, el porcentaje de recomendación basado en reviews.recommend y la cantidad de ítems.

3.- UserForGenre(genero: str):

Descripción: Devuelve el usuario que acumula más horas jugadas para el género dado, junto con una lista de horas jugadas por año de lanzamiento.

4.- best_developer_year(año: int):

Descripción: Devuelve el top 3 de desarrolladores con juegos más recomendados por usuarios para el año dado (donde reviews.recommend = True y los comentarios son positivos).

5.-developer_reviews_analysis(desarrolladora: str):

Descripción: Devuelve un diccionario con el nombre del desarrollador como llave y una lista con la cantidad total de reseñas de usuarios categorizadas con un análisis de sentimiento como valor positivo o negativo


## Modelo de Aprendizaje Automático

Se desarrolló un **modelo de recomendación item-item** 

Este modelo recomienda juegos basándose en la similitud entre ítems. Para cada juego ingresado, se generan recomendaciones de otros juegos que son similares en función de características específicas.
 
 para realizar este modelo se crearon las siguentes funciones:

 1.- Función para normalizar la columna 'release_date' usando StandardScaler
 2.- Función para fusionar los DataFrames 'df_games' y 'df_genres' en 'df_merged'
 3.- Función para muestrear un subconjunto de datos
 4.- Función para calcular la matriz de similitud del coseno
 5.- Función para recomendar juegos basados en la similitud del coseno
 6.- Simulación del proceso completo


### Despliegue

Se utilizó **Render** para hacer disponible la API en la web, permitiendo el consumo externo del sistema de recomendaciones.


**Dependencias**

Para ejecutar este proyecto, es necesario instalar las siguientes librerías:
- `pandas`
- `sqlalchemy`
- `scikit-learn`
- `pyarrow`
- `fastparquet`
- `fastapi`
- `uvicorn`
- `numpy`
- `vader`
- `ast`
- `re`
- `json`


## Contacto

Desarrollador: [Gabriel Rizo](https://github.com/Rizo12G)  
Correo: rizo.tnt@gmail.com
