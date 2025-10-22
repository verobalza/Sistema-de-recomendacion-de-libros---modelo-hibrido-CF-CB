import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#cargamos nuestros archivos de data 

usuario = pd.read_csv('data/Users.csv')
libros = pd.read_csv('data/Books.csv')
ratings = pd.read_csv('data/Ratings.csv')

#veremos cuantas filas y columnas tiene cada archivo

#Usuario
print('Usuario filas y columnas: ',usuario.shape)
print('Usuario primeras filas: ', usuario.head(5))
print('Usuario info: ', usuario.info())
print('Usuario valores nulos ', usuario.isnull().sum())


#Libros 
print('Libros filas y columnas: ',libros.shape)
print('Libros primeras filas: ', libros.head(5))
print('Libros info: ', libros.info())
print('Libros valores nulos ', libros.isnull().sum())

#Ratings
print('Ratings filas y columnas: ',ratings.shape)
print('Ratings primeras filas: ', ratings.head(5))
print('Ratings info: ', ratings.info())
print('Ratings valores nulos ', ratings.isnull().sum())

#normalizamos las columnas 
usuario.columns = usuario.columns.str.strip().str.lower()
libros.columns = libros.columns.str.strip().str.lower()
ratings.columns = ratings.columns.str.strip().str.lower()

#Nos aseguramos que ratings (nuestra variable clave) sea numerico
ratings['book-rating']=pd.to_numeric(ratings['book-rating'], errors='coerce')
print(ratings['book-rating'].value_counts().sort_index())




# Vamos a hacer la funcion para filtrarlo con k_core
def filk_core(ranting_expli_df, user_colu='user-id', codigo_libro='isbn', min_user=5, min_libro=5 ):

    #hacemos una copia del df rating_expli y trabajamos en el 
    df= ranting_expli_df.copy()
    iteracion=0

    while True:
        iteracion +=1

        #contamos cuanto aparcen cada usuario y cada libro 
        usuario_conteo= df[user_colu].value_counts() 
        libros_conteo = df[codigo_libro].value_counts()


        #nosquedamos con los user_id y los isbn que cumplen con los min correspondientes de cada uno 
        df_new = df[df[user_colu].isin(usuario_conteo[usuario_conteo>= min_user].index) &df[codigo_libro].isin(libros_conteo[libros_conteo>=min_libro].index)].copy()

        #si df_new y df tienen las mismas filas el bucle se acaba
        if df_new.shape[0] == df.shape[0]:
            break
        
        #en cada iteracion actualizamos el df 
        df = df_new
    print(f' Filtrado k-core completo. Filas finales: {df.shape[0]}')
    return df


#haremos una funcion para probar umbrales y asi poder escoger cual es el mas idoneo
def comparar_umbrales(ratings_explici_df, usu_umb=[5,10,20,30,40,50], libr_umb=[5,10,20,30,40,50]):
    filas=[]
    total= ratings_explici_df

    #recorremos los umbrales a probar
    for u in usu_umb:
        for li in libr_umb:
            filtro_k=filk_core(ratings_explici_df, min_user=u, min_libro=li)
            filas.append(
                {
                    'min_usuario':u, 'min_libro':li,
                    'ratings':total,
                    'filtro_kcore':filtro_k.shape[0], 'filtro_user':filtro_k['user-id'].nunique(), 'filtro_libro':filtro_k['isbn'].nunique()

                }
            )
    return pd.DataFrame(filas)

#separamos valores explicitos de rating para quedarnos con los rating > 0
#Umbrales seleccionados 
MIN_USUARIO = 5
MIN_LIBRO = 10

ratings_expli= ratings[ratings['book-rating']>0 ].copy()


usuario_activo = ratings_expli['user-id'].value_counts()
usuario_activo = usuario_activo[usuario_activo >= MIN_USUARIO].index

libro_activo = ratings_expli['isbn'].value_counts()
libro_activo = libro_activo[libro_activo >= MIN_LIBRO].index

ratings_expli = ratings_expli[
    ratings_expli['user-id'].isin(usuario_activo) &
    ratings_expli['isbn'].isin(libro_activo)
].copy()


tabla_umb= comparar_umbrales(ratings_expli,usu_umb=[5,10,20,30,40,50], libr_umb=[5,10,20,30,40,50])


#Grafico de calor para ver mejor que umbrales escogeremos
pivot_kc = tabla_umb.pivot(index="min_usuario", columns="min_libro", values="filtro_kcore")
tabla_umb.to_csv("tabla_comparativa_kcore.csv", index=False)

plt.figure(figsize=(10,6))
sns.heatmap(pivot_kc, annot=True, fmt="g", cmap="Reds")
plt.title("K-core: Ratings retenidos según umbrales")
plt.xlabel("Umbral mínimo por libro (min_libro)")
plt.ylabel("Umbral mínimo por usuario (min_usuario)")
plt.tight_layout()
plt.show()



#Guardamos en csv los nuevos archivos filtrados ---> ubicados en la carpeta data
rating_f= filk_core(ratings_expli,min_user=MIN_USUARIO, min_libro=MIN_LIBRO)
rating_f.to_csv('ratinf.csv', index=False)

#seleccionamos solo los user-id y isbn que se encuentren en rating_f
usuario_f= usuario[usuario['user-id'].isin(rating_f['user-id'].unique())].copy()
libro_f=libros[libros['isbn'].isin(rating_f['isbn'].unique())].copy()

#csv usuario y libros
usuario_f.to_csv('usuario_f.csv', index=False)
libro_f.to_csv('libros_f.csv', index=False)





