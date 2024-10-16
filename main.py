from fastapi import FastAPI
import pandas as pd
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# Instanciamos FastAPI
app = FastAPI()

# Cargamos los DataFrames de ejemplo desde archivos .parquet
df_games = pd.read_parquet("./data_clean_EDA/steam_games.parquet")
df_items = pd.read_parquet("./data_clean_EDA/australian_users_items.parquet")
df_reviews = pd.read_parquet("./data_clean_EDA/user_reviews.parquet")

# sistema de recomendacion
df_genres = pd.read_parquet('./data_clean_EDA/df_dummies.parquet')
df_games = pd.read_parquet('./data_clean_EDA/df_games_ML.parquet')


#copiamos los datasets 
df_reviews_copy = df_reviews
df_items_copy = df_items
df_games_copy = df_games

# Función developer
def developer(desarrollador: str, games_df):
    # convertimos 'release_date' a tipo fecha y filtramos por dev
    games_df['release_date'] = pd.to_datetime(games_df['release_date'], errors='coerce')
    df_dev = games_df[games_df['developer'] == desarrollador]

    # agrupamos por fecha y elemntos y dividimos por contenido gratuito
    items_por_año = df_dev.groupby(df_dev['release_date'].dt.year).size().reset_index(name='Items')
    free_por_año = df_dev[df_dev['price'] == 0].groupby(df_dev['release_date'].dt.year).size().reset_index(name='gratuito')

    # Combinamos los resultaso en uno solo 
    result_df = items_por_año.merge(free_por_año, on='release_date', how='left')
    result_df['gratuito'] = ((result_df['gratuito'].fillna(0) / result_df['Items']) * 100).round(2).astype(str) + '%'

    # Renombrar la columna de release_date para cumplir con el objetivo del ejemplo dato
    result_df.rename(columns={'release_date': 'Año'}, inplace=True)

    return result_df

# Función userdata
def userdata(User_id, df_items, df_games, df_reviews):
    # Filtrmos por usuario y calculamos el dinero gastado 
    user_items = df_items[df_items['user_id'] == User_id]
    dinero_gastado = user_items.merge(df_games[['id', 'price']], left_on='item_id', right_on='id')['price'].sum()
    items_count = user_items['items_count'].sum()

    # Filtrarmos por usuario 
    user_reviews = df_reviews[df_reviews['user_id'] == User_id]
    # Calculamos el porcentaje de recomendación 
    if user_reviews.shape[0] > 0:
        recommendation_percentage = (user_reviews['recommend'].sum() / user_reviews.shape[0]) * 100
    else:
        recommendation_percentage = 0 

    resultado = {
        "Usuario": str(User_id),
        "Dinero gastado": f"{dinero_gastado:.2f} USD",
        "% de recomendación": f"{recommendation_percentage:.2f}%",  
        "Cantidad de items": items_count
    }
    
    result_df = pd.DataFrame(list(resultado.items()), columns=['Descripción', 'Valor'])

    return result_df

# Función UserForGenre
def UserForGenre(genero: str, df_games, df_items):
    # Convertir el campo 'release_date' a un formato legible y extraer el año
    df_games['release_year'] = pd.to_datetime(df_games['release_date'], unit='ms').dt.year
    
    # Filtrar por género asegurando que no sea None
    genero_play = df_games[df_games['genres'].apply(lambda x: x is not None and genero in x)]
    if genero_play.empty:
        return {"Mensaje": f"No hay datos para el género '{genero}'"}

    # Unir con df_items usando la columna id y item_id
    usuario = genero_play.merge(df_items, left_on='id', right_on='item_id')

    # Calcular las horas jugadas por usuario
    usuario_h = usuario.groupby('user_id')['playtime_forever'].sum().reset_index()

    # Verificar si hay datos después del merge
    if usuario_h.empty:
        return {"Mensaje": f"No hay datos para el género '{genero}'"}

    # Encontrar el usuario con más horas jugadas
    usuario_max_horas = usuario_h.loc[usuario_h['playtime_forever'].idxmax()]['user_id']

    # Calcular la acumulación de horas jugadas por año de lanzamiento
    horas_por_año = usuario.groupby('release_year')['playtime_forever'].sum().reset_index()
    horas_por_año.rename(columns={'playtime_forever': 'Horas'}, inplace=True)

    # Convertir a la estructura requerida
    horas_jugadas = horas_por_año.to_dict(orient='records')

    result = {
        f"Usuario con más horas jugadas para Género {genero}": usuario_max_horas,
        "Horas jugadas": horas_jugadas
    }

    return result

# best_developer_year

def best_developer_year(año: int):
    # Filtrar juegos lanzados en el año dado en df_games
    juegos_del_año = df_games[df_games['release_year'] == año]
    
    # Convertir 'item_id' en df_reviews y 'id' en df_games a string para hacer el merge correctamente
    df_reviews['item_id'] = df_reviews['item_id'].astype(str)
    juegos_del_año['id'] = juegos_del_año['id'].astype(str)
    
    # Unir df_reviews con df_games usando item_id y id para obtener los juegos y sus desarrolladores
    reviews_con_juegos = df_reviews.merge(juegos_del_año, left_on='item_id', right_on='id')
    
    # Filtrar solo reseñas recomendadas y con el máximo sentimiento positivo
    df_filtrado = reviews_con_juegos[(reviews_con_juegos['recommend'] == 1) & 
                                     (reviews_con_juegos['sentiment_analysis'] == reviews_con_juegos['sentiment_analysis'].max())]
    
    # Agrupar por desarrollador y contar las recomendaciones
    top_developers = df_filtrado.groupby('developer')['recommend'].sum()
    
    # Ordenar los desarrolladores por número de recomendaciones en orden descendente
    top_developers = top_developers.sort_values(ascending=False)
    
    # Tomar los top 3 desarrolladores
    top_3_developers = top_developers.head(3)
    
    top_developers_dict = {}
    for i, (developer, recomendaciones) in enumerate(top_3_developers.items(), 1):
        puesto = f"Puesto {i}"
        top_developers_dict[puesto] = developer

    return top_developers_dict

# developer_reviews_analysis
@app.get("/developer/{desarrollador}")
def developer_reviews_analysis(desarrollador, df_reviews, df_games):
    # Paso 1: Convertir 'item_id' e 'id' a cadenas (str) para evitar conflictos en el merge
    df_reviews_copy = df_reviews.copy()
    df_games_copy = df_games.copy()
    df_reviews_copy['item_id'] = df_reviews_copy['item_id'].astype(str)
    df_games_copy['id'] = df_games_copy['id'].astype(str)
    merged_data = pd.merge(df_reviews_copy, df_games_copy, left_on='item_id', right_on='id', how='left')

    # reseñas con sentimiento positivo (2) o negativo (0), excluyendo neutros (1)
    filtered_data = merged_data[merged_data['sentiment_analysis'] != 1]

    # Agrupa por 'developer' y 'sentiment_analysis', contando las reseñas
    grouped_data = filtered_data.groupby(['developer', 'sentiment_analysis']).size().unstack(fill_value=0)

    if desarrollador in grouped_data.index:
        # Extraer los conteos de reseñas negativas (0) y positivas (2)
        developer_reviews = grouped_data.loc[desarrollador]
        developer_reviews_list = [
            {"Negativas": developer_reviews.get(0, 0)},  # Cantidad de reseñas negativas
            {"Positivas": developer_reviews.get(2, 0)}   # Cantidad de reseñas positivas
        ]

        return {desarrollador: developer_reviews_list}
    else:
        return f"No se encontró información de: {desarrollador}"

#___________________________________sistema de recoemndacion__________________________________


# Función para normalizar la columna 'release_date' usando StandardScaler
def normalizar_release_date(df, columna):
    scaler = StandardScaler()
    df[columna] = scaler.fit_transform(df[[columna]])
    return df

# Función para fusionar los DataFrames 'df_games' y 'df_genres' en 'df_merged'
def fusionar_dataframes(df1, df2, key, cols_a_incluir):
    df_merged = df1.merge(df2, on=key, how='left')
    df_final = df_merged[['id'] + cols_a_incluir]
    return df_final

# Función para muestrear un subconjunto de datos
def muestrear_datos(df, fraccion, estado_aleatorio):
    return df.sample(frac=fraccion, random_state=estado_aleatorio)

# Función para calcular la matriz de similitud del coseno
def calcular_similitud(df, features):
    matriz_similitud = cosine_similarity(df[features].fillna(0))
    return np.nan_to_num(matriz_similitud)

# Función para recomendar juegos basados en la similitud del coseno
def recomendacion_juego(df_sampled, similarity_matrix, game_id, top_n=5):
    ids_juegos_muestreados = df_sampled['id'].unique()
    
    if game_id not in ids_juegos_muestreados:
        return f"No se encontraron recomendaciones: {game_id} no está en los datos muestreados."
    
    indice_juego = df_sampled.index[df_sampled['id'] == game_id].tolist()
    
    if not indice_juego:
        return f"No se encontraron recomendaciones: {game_id} no está en los datos muestreados."
    
    indice_juego = indice_juego[0]
    
    puntajes_similitud = list(enumerate(similarity_matrix[indice_juego]))
    puntajes_similitud = sorted(puntajes_similitud, key=lambda x: x[1], reverse=True)

    indices_juegos_similares = [i for i, puntaje in puntajes_similitud[1:top_n+1]]
    
    nombres_juegos_similares = df_sampled['app_name'].iloc[indices_juegos_similares].tolist()
    
    mensaje_recomendacion = f"Juegos recomendados basados en el ID del juego {game_id} - {df_sampled['app_name'].iloc[indice_juego]}:"
    
    return [mensaje_recomendacion] + nombres_juegos_similares


@app.get("/proceso_recomendacion/")
def proceso_recomendacion(df_games, df_genres, game_id, top_n=5):
    # Lista de columnas a incluir
    features = ['release_date'] + list(df_genres.columns[1:])
    
    # Normalizar 'release_date' en df_games
    df_games = normalizar_release_date(df_games, 'release_date')

    # Fusionar DataFrames y seleccionar columnas
    df_merged = fusionar_dataframes(df_games, df_genres, key='id', cols_a_incluir=features)

    # Agregar columna 'app_name'
    df_final = df_merged.merge(df_games[['id', 'app_name']], on='id', how='left')

    # Muestrear los datos (20%)
    df_sampled = muestrear_datos(df_final, fraccion=0.2, estado_aleatorio=42)

    # Calcular la matriz de similitud del coseno
    similarity_matrix = calcular_similitud(df_sampled, features)

    # Obtener recomendaciones
    recomendaciones = recomendacion_juego(df_sampled, similarity_matrix, game_id, top_n)
    
    return recomendaciones


#____________________________________________________________________________________________

# Endpoint para developer
@app.get("/developer/{desarrollador}")
def get_developer_info(desarrollador: str):
    result = developer(desarrollador, df_games)
    return result.to_dict(orient='records')

# Endpoint para userdata
@app.get("/userdata/{user_id}")
def get_user_data(user_id: str):
    result = userdata(user_id, df_items, df_games, df_reviews)
    return result.to_dict(orient='records')

# Endpoint para UserForGenre
@app.get("/genre/{genero}")
def get_user_for_genre(genero: str):
    result = UserForGenre(genero, df_games, df_items)
    return result

# Endpoint para best_developer_year
@app.get("/best_developer/{año}")
def get_best_developer(año: int):
    funcion = get_best_developer((año, df_reviews, df_games))
    return funcion


# Endpoint para get_developer_reviews
@app.get("/developer/{desarrollador}")
def get_developer_reviews(desarrollador: str):
    result = developer_reviews_analysis(desarrollador, df_reviews, df_games)

    return result


# sistema de recomendacion 
@app.get("/recomendacion_juego/{item_id}")
def get_recomendacion_juego(item_id: int):
    recomendaciones = recomendacion_juego(item_id)
    return {"juegos_recomendados": recomendaciones}

