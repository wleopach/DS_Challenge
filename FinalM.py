from DataReader import data
import pandas as pd
from imblearn.ensemble import  BalancedRandomForestClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from targetEncoding import cat, numD
from sklearn.model_selection import train_test_split
from sklearn.metrics import  confusion_matrix
#traemos las variables categoricas con los nuevos valores numericos y actualizamos la data
numData = pd.DataFrame(numD)
#Definimos un nuevo dataframe para conservar los valores originales en data y los transformados quedan en dataT
dataT = data
NCat = ['ID_USER', 'monto', 'hora', 'linea_tc', 'interes_tc',
        'is_prime', 'dcto', 'cashback']
dataNCat = data[NCat]
dataNCat = StandardScaler().fit_transform(dataNCat)
dataNCat = pd.DataFrame(dataNCat)
dataNCat.columns = NCat
for i in cat:
    dataT[i] = numData[i]
for i in NCat:
    dataT[i] = dataNCat[i]
#Definimos el modelo
ms = BalancedRandomForestClassifier(max_depth=2, random_state=0, n_estimators=10)
#Ahora dividimos la data
X = dataT.drop('fraude', axis = 1)
y = dataT['fraude'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)
steps = [('over', SMOTE()), ('model', ms)]
pipeline = Pipeline(steps=steps)
pipeline.fit(X_train, y_train)
yhat = pipeline.predict(X_test)
print('La matriz de confución es:')
print(confusion_matrix(y_test, yhat))
tn, fp, fn, tp = confusion_matrix(y_test,  yhat).ravel()

print(tn, fp, fn, tp)