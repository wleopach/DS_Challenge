import pandas as pd
import numpy as np
from DataReader import data
from targetEncoding import cat, numD
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,cross_val_score, cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from tabulate import tabulate
from sklearn.metrics import  confusion_matrix, f1_score
from sklearn.feature_selection import SelectFromModel
from imblearn.ensemble import  BalancedRandomForestClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
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
#Ahora dividimos la data
X = dataT.drop('fraude', axis = 1)
y = dataT['fraude'].values


# Modelo Decision Tree
tree_model = DecisionTreeClassifier(max_depth = 2, criterion = 'entropy')

# 2. K-Nearest Neighbors

n = 5

knn = KNeighborsClassifier(n_neighbors = n)


# 3. Logistic Regression

lr = LogisticRegression()


# 4. SVM

svm = SVC()


# 5. Random Forest Tree

rf = RandomForestClassifier(max_depth=2, random_state=0, n_estimators=10)


# 6. Balanced Random Forest
brf = BalancedRandomForestClassifier(max_depth=2, random_state=0, n_estimators=10)
#Listamos todos los modelos y sus  nombres

models = [tree_model, knn, lr, svm, rf, brf]

modNames = ['Decision Tree','KNeighbors','Logistic Regression','Support Vector Machines','Random Forest', 'Balanced RF']
#Creamos un diccionario para guardar los resultados de los modleos
R = dict()
for m in models:
    R[m] = dict()
    R[m]['Nombre'] =modNames[models.index(m)]
    steps = [('over', SMOTE()), ('model', m)]
    pipeline = Pipeline(steps=steps)
    scores = cross_val_score(pipeline, X, y, cv=5, scoring='f1')
    R[m]['cross validate scores'] = np.mean(scores)
# Imprimimos los resultados
R = pd.DataFrame(data=R)
print('Los modelos  considerados son:')
print(tabulate(R, headers='keys', tablefmt='psql'))

#Selección del modelo con menor f1
cvScore = 0
for m in models:
    if R[m]['cross validate scores'] >= cvScore:
        cvScore= R[m]['cross validate scores']
        ms = m
#Imprimimos la informacion del modelo escogido
print('El modelo seleccionado es:')
print(R[ms])

#Dividimos la data y entrenamos el modelo escogido
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)
steps = [('over', SMOTE()), ('model', ms)]
pipeline = Pipeline(steps=steps)
pipeline.fit(X_train, y_train)
yhat = pipeline.predict(X_test)
#Selección de los mejores predictores
imp = pd.DataFrame(ms.feature_importances_).transpose()
imp.columns = ['ID_USER', 'genero', 'monto', 'fecha', 'hora', 'dispositivo',
                'establecimiento', 'ciudad', 'tipo_tc', 'linea_tc', 'interes_tc',
               'status_txn', 'is_prime', 'dcto', 'cashback']
print('En la siguiente tabla está la importancia de cada predictor')
print(tabulate(imp, headers=imp.columns, tablefmt='psql'))
print('La matriz de confución es:')
print(confusion_matrix(y_test, yhat))
tn, fp, fn, tp = confusion_matrix(y_test,  yhat).ravel()

print(tn, fp, fn, tp)