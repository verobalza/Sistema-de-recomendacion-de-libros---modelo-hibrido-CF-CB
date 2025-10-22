import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

TOP_K = 5
ALPHA = 0.7 
N_COMPONENTS = 50

#----Cargamos los archivos de data ya filtrados con k_core---


ratings = pd.read_csv('data/ratinf.csv')
usuarios = pd.read_csv('data/usuario_f.csv')
libros = pd.read_csv('data/libros_f.csv')
ratings_train, ratings_test = train_test_split(ratings, test_size=0.2, random_state=42)

#----mapeamos el id de usuario------
usuarios_map = ratings['user-id'].unique()

map_usu={}
for i, user_id in enumerate(usuarios_map):
    map_usu[user_id]=i

#-----hacemos el mapeo inverso y asi lo tenemos para futuras ocaciones----

map_inver_usu = {i: user_id for user_id, i in map_usu.items()}

#-----añadimos a ratings la columna de indice que hicimos-----
ratings['user_idx'] = ratings['user-id'].map(map_usu)

#----preparamos metadatos de libros----
libros['book-title'] = libros['book-title'].fillna('').astype(str)
libros['book-author'] = libros['book-author'].fillna('').astype(str)
libros['year-of-publication'] = libros['year-of-publication'].fillna('').astype(str)
libros['publisher'] = libros['publisher'].fillna('').astype(str)

#-----añadimos a libros la columna meta con todos los datos ---------
libros['meta'] = libros['book-title'] + ' ' + libros['book-author'] + ' ' + libros['publisher'] + ' ' + libros['year-of-publication']


#------seleccionamos los libros unicos en ratings----
libros_map = ratings['isbn'].unique()

#-----guardamos los libros que estan tanto el isbn en libros como en libros_map---
libros_acti = libros[libros['isbn'].isin(libros_map)].copy()
libros_acti = libros_acti.set_index('isbn').reindex(libros_map).reset_index()

#----reconstruimos map_lib y actualizamos ratings para que R_cb y R_cf tengan la misma dimension----
def reconstruir_map_lib(libros_acti, ratings_df):
    isbn_validos = libros_acti['isbn'].unique()
    map_lib = {isbn: i for i, isbn in enumerate(isbn_validos)}
    map_inver_libros = {i: isbn for isbn, i in map_lib.items()}
    ratings_filtrados = ratings_df[ratings_df['isbn'].isin(isbn_validos)].copy()
    ratings_filtrados['lib_idx'] = ratings_filtrados['isbn'].map(map_lib)
    return map_lib, map_inver_libros, ratings_filtrados

map_lib, map_inver_libros, ratings = reconstruir_map_lib(libros_acti, ratings)
libros_map = list(map_lib.keys())

#----matriz dispersa usuario-libro para CF----
R_sparce = csr_matrix(
    (ratings['book-rating'].astype(np.float32),
     (ratings['user_idx'].astype(np.int32), ratings['lib_idx'].astype(np.int32))
    ))

#----filtrado colaborativo CF----
svd = TruncatedSVD(n_components=N_COMPONENTS, random_state=42)
usu_vect = normalize(svd.fit_transform(R_sparce))
lib_vect = normalize(svd.components_.T)
R_cf = np.dot(usu_vect, lib_vect.T)

#------contenido basado en metadatos CB----

tfidf = TfidfVectorizer(max_features=500, stop_words='english')
libros_acti['meta'] = libros_acti['meta'].fillna('')

x_tfidf = normalize(tfidf.fit_transform(libros_acti['meta'].values))

n_usu = len(map_usu)
n_lib = len(map_lib)
R_cb = np.zeros((n_usu, n_lib), dtype=np.float32)

for u_id, u_idx in map_usu.items():
    ratings_usu = ratings[ratings['user_idx'] == u_idx]
    i_idxs = ratings_usu['lib_idx'].values
    rating_val = ratings_usu['book-rating'].values.astype(np.float32)
    perfil = (x_tfidf[i_idxs].multiply(rating_val.reshape(-1, 1))).sum(axis=0)
    perfil = np.array(perfil).ravel()
    norma = np.linalg.norm(perfil)
    if norma > 0:
        perfil /= norma
        si = x_tfidf.dot(perfil)
        R_cb[u_idx, :] = si.ravel()

#----modelo híbrido CF + CB----
R_hibrido = ALPHA * R_cf + (1 - ALPHA) * R_cb

#----evaluación del modelo----
def evaluar_modelo(ratings_df, matriz_pred, map_usu, map_lib):
    rating_verd = []
    rating_predict = []
    for _, row in ratings_df.iterrows():
        uid = map_usu[row['user-id']]
        lid = map_lib[row['isbn']]
        rating_verd.append(row['book-rating'])
        rating_predict.append(matriz_pred[uid, lid])
    rmse = np.sqrt(mean_squared_error(rating_verd, rating_predict))
    mae = mean_absolute_error(rating_verd, rating_predict)
    return rmse, mae

rmse, mae = evaluar_modelo(ratings, R_hibrido, map_usu, map_lib)
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")

#hacemos la funcion para obtener la metrica recall
def precision_recall_at_k(ratings_df, matriz_pred, map_usu, map_lib, k=5):
    precisiones = []
    recalls = []

    for user_id in ratings_df['user-id'].unique():
        if user_id not in map_usu:
            continue
        u_idx = map_usu[user_id]
        scores = matriz_pred[u_idx]
        top_k_idx = np.argsort(scores)[::-1][:k]
        top_k_isbn = [map_inver_libros[i] for i in top_k_idx]

        libros_reales = ratings_df[ratings_df['user-id'] == user_id]['isbn'].unique()
        relevantes = set(libros_reales)
        recomendados = set(top_k_isbn)

        interseccion = relevantes.intersection(recomendados)
        precisiones.append(len(interseccion) / k)
        recalls.append(len(interseccion) / len(relevantes) if len(relevantes) > 0 else 0)

    return np.mean(precisiones), np.mean(recalls)

precision, recall = precision_recall_at_k(ratings_test, R_hibrido, map_usu, map_lib, k=5)
print(f"Precision@5: {precision:.4f}")
print(f"Recall@5: {recall:.4f}")

#creamos el grafico para ver la distribucion de book-rating
import matplotlib.pyplot as plt
ratings['book-rating'].hist(bins=20)
plt.title("Distribución de ratings")
plt.show()

#---hacemos la funcion para hacer la recomendacion a los usuarios segun sus preferencias----

def recomendar_libros_para_usuario(user_id, matriz_pred, map_usu, map_inver_libros, libros_acti, top_k=5):
    if user_id not in map_usu:
        print(f"Usuario {user_id} no encontrado.")
        return
    u_idx = map_usu[user_id]
    scores = matriz_pred[u_idx]
    top_idx = np.argsort(scores)[::-1][:top_k]

    print(f"\n Recomendaciones para el usuario {user_id}:")
    for idx in top_idx:
        isbn = map_inver_libros[idx]
        libro = libros_acti[libros_acti['isbn'] == isbn].iloc[0]
        print(f"→ {libro['book-title']} | {libro['book-author']} ({libro['year-of-publication']})")



#---- Recomendaciones para el VIDEO (usuarios concretos)--
print(f'\n**********Recomendacion usuarios concretos*************\n')
usuarios_concretos = [136382]
#, 185677, 217318, 136382, 153662
for uid in usuarios_concretos:
    recomendar_libros_para_usuario(uid, R_hibrido, map_usu, map_inver_libros, libros_acti, top_k=5)
'''

#---- Recomendaciones para el PDF (usuarios aleatorios--
print(f'\n***********Recomendacion para usuarios aleatorios**********\n')
usuarios_aleatorios = np.random.choice(ratings_test['user-id'].unique(), size=5, replace=False)
for uid in usuarios_aleatorios:
    recomendar_libros_para_usuario(uid, R_hibrido, map_usu, map_inver_libros, libros_acti, top_k=5)

'''
#----vemos los libros que valoro cada usuario y asi vemos la similitud en las recomendaciones---
# dejo un ejmplo de como vi
# usuario 153662
def ver_libros_valorados(user_id, ratings_df, libros_df):
    ratings_usuario = ratings_df[ratings_df['user-id'] == user_id]
    libros_valorados = libros_df[libros_df['isbn'].isin(ratings_usuario['isbn'])]
    print(f"\nLibros valorados por el usuario {user_id}:")
    print(libros_valorados[['book-title', 'book-author', 'year-of-publication']])

ver_libros_valorados(136382, ratings, libros_acti)

