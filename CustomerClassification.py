import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataReader import data
from targetEncoding import numD, cat
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
from tabulate import tabulate

#traemos las variables categoricas con los nuevos valores numericos y actualizamos la data
numData = pd.DataFrame(numD)
for i in cat:
    data[i] = numData[i]
#Calculamos  la media de cada una de las variables agrupadas por cliente
users = data.groupby(['ID_USER']).mean()
users = StandardScaler().fit_transform(users)
users = pd.DataFrame(users)
users.columns = ['genero', 'monto', 'fecha', 'hora', 'dispositivo',
                'establecimiento', 'ciudad', 'tipo_tc', 'linea_tc', 'interes_tc',
                'status_txn', 'is_prime', 'dcto', 'cashback','fraude']
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(users)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Componente  principal 1', fontsize = 15)
ax.set_ylabel('Componente principal 2', fontsize = 15)
ax.set_title(' Visualización de los usuarios con 2 component PCA ', fontsize = 20)
ax.scatter(principalDf['principal component 1'], principalDf['principal component 2'])
#plt.show()
Nc = range(1, 20)
kmeans = [KMeans(n_clusters=i) for i in Nc]
score = [kmeans[i].fit(users).score(users) for i in range(len(kmeans))]
plt.plot(Nc, score)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Curve')
#plt.show()
#Corremos KMeans
kmeans = KMeans(n_clusters=7).fit(users)
#Graficamos los rsultados de KMeans
df_labeled = pd.DataFrame(kmeans.labels_, columns = list(['labels']))
df_labeled['labels'] = df_labeled['labels'].astype('category')
plt.figure(figsize=(10, 8))
df_labeled['labels'].value_counts().plot.bar(color='y')
plt.xlabel("Cluster")
plt.ylabel("Número de clientes")
plt.title("Número clientes por Cluster")
#plt.show()
#Agragamos los resultados las categorias del KMeans a users
users = users.join(df_labeled)
#Graicamos el dendograma
plt.figure(figsize=(20,10))
merg = linkage(users.drop('labels',1), method='ward')
dendrogram(merg, leaf_rotation = 360)
plt.title('Dendrogram')
#plt.show()
#Definimos el clusterin jerarquico
hier_clus = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
cluster = hier_clus.fit_predict(users.drop('labels',1))
#Agregamos las categorias del clustrein jerarqico
users['Agg_label'] = cluster
#Grafica del CH
df_labeled = pd.DataFrame(hier_clus.labels_, columns = list(['labels']))
df_labeled['labels'] = df_labeled['labels'].astype('category')
plt.figure(figsize=(10, 8))
df_labeled['labels'].value_counts().plot.bar(color='y')
plt.xlabel("Cluster")
plt.ylabel("Número de clientes")
plt.title("Número clientes por Cluster")
#plt.show()
#Imprimimos los resultados de las clasificaciones
x = users[['labels','Agg_label']]
x.columns = ['KMeans', 'Jerárquico']
x.index.name = 'ID_USER'
R = tabulate(x, headers='keys', tablefmt='psql')
print(R)