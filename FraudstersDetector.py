import pandas as pd
import numpy as np
from DataReader import data
from targetEncoding import cat, numD
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from tabulate import tabulate
from sklearn.metrics import accuracy_score
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 0)

# Modelo Decision Tree
tree_model = DecisionTreeClassifier(max_depth = 2, criterion = 'entropy')
tree_model.fit(X_train, y_train)
tree_yhat = tree_model.predict(X_test)
# 2. K-Nearest Neighbors

n = 5

knn = KNeighborsClassifier(n_neighbors = n)
knn.fit(X_train, y_train)
knn_yhat = knn.predict(X_test)

# 3. Logistic Regression

lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_yhat = lr.predict(X_test)

# 4. SVM

svm = SVC()
svm.fit(X_train, y_train)
svm_yhat = svm.predict(X_test)

# 5. Random Forest Tree

rf = RandomForestClassifier(max_depth=2, random_state=0)
rf.fit(X_train, y_train)
rf_yhat = rf.predict(X_test)


models = [tree_model, knn, lr, svm, rf]
yEst = [tree_yhat,knn_yhat,lr_yhat, svm_yhat, rf_yhat]
modNames = ['Decision Tree','KNeighbors','Logistic Regression','Support Vector Machines','Random Forest']
R = dict()
for m in models:
    R[m] = dict()
    R[m]['Nombre'] =modNames[models.index(m)]
    R[m]['Accuracy score'] = accuracy_score(y_test, yEst[models.index(m)])
    scores= cross_val_score(m, X, y, cv=5, scoring='accuracy')
    R[m]['cross validate score'] = np.mean(scores)
R=pd.DataFrame(data=R)
print(tabulate(R, headers='keys', tablefmt='psql'))