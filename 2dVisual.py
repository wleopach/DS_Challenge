import pandas as pd
import matplotlib.pyplot as plt
from DataReader import data
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from targetEncoding import numD, cat

#traemos las variables categoricas con los nuevos valores numericos
numData = pd.DataFrame(numD)
for i in cat:
    data[i] = numData[i]
features = ['ID_USER','genero', 'monto', 'fecha', 'hora', 'dispositivo',
       'establecimiento', 'ciudad', 'tipo_tc', 'linea_tc', 'interes_tc',
       'status_txn', 'is_prime', 'dcto', 'cashback']
x = data.loc[:, features].values
y = data.loc[:,['fraude']].values
x = StandardScaler().fit_transform(x)

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
finalDf = pd.concat([principalDf, data[['fraude']]], axis = 1)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Componente  principal 1', fontsize = 15)
ax.set_ylabel('Componente principal 2', fontsize = 15)
ax.set_title(' Visualización de la data usando 2 component PCA ', fontsize = 20)
targets = [1, 0]
colors = ['black','y']
for target, color in zip(targets, colors):
    indicesToKeep = finalDf['fraude'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
plt.show()